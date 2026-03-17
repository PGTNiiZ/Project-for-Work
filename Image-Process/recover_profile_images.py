"""
Recover and extend profile-image downloads with aggressive fallback strategies.

This script keeps the old filename convention so it can resume into
Image-Process/downloaded_images without duplicating files.

Notes:
- GPU is intentionally not used here. This workload is network and I/O bound.
- The main recovery boost comes from platform-specific URL recovery and
  username-based profile-image resolution.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import html
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import aiofiles
import aiohttp


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = SCRIPT_DIR.parent / "data-for-project" / "nomalized_profiles.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "downloaded_images"
DEFAULT_REPORT_PATH = SCRIPT_DIR / "download_report.json"
DEFAULT_FAILED_CSV = SCRIPT_DIR / "failed_urls.csv"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
IMAGE_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/gif",
    "image/webp",
    "image/bmp",
    "image/avif",
}


@dataclass(slots=True)
class DownloadItem:
    idx: int
    username: str
    platform: str
    original_url: str
    filename: str
    url_hash: str
    previous_error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover more profile images without re-downloading existing files."
    )
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--failed-csv", type=Path, default=DEFAULT_FAILED_CSV)
    parser.add_argument("--max-concurrent", type=int, default=96)
    parser.add_argument("--external-resolver-concurrent", type=int, default=4)
    parser.add_argument("--timeout-sec", type=int, default=25)
    parser.add_argument("--resolver-timeout-sec", type=int, default=15)
    parser.add_argument("--platform", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--prefer-previous-failures",
        action="store_true",
        help="Prioritize URLs that failed in the previous report.",
    )
    parser.add_argument(
        "--only-previous-failures",
        action="store_true",
        help="Process only URLs listed as failed in the previous report.",
    )
    return parser.parse_args()


def get_url_hash(url: str) -> str:
    return hashlib.md5(url.encode("utf-8")).hexdigest()[:8]


def get_filename(url: str, username: str, idx: int) -> str:
    parsed = urlparse(url)
    ext = Path(parsed.path).suffix.lower()
    if ext not in VALID_EXTENSIONS:
        ext = ".jpg"
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in username)
    url_hash = get_url_hash(url)
    return f"{idx:05d}_{safe_name}_{url_hash}{ext}"


def normalize_username(username: str, platform: str) -> str:
    username = (username or "").strip()
    if platform.lower() in {"twitter", "x"}:
        return username.lstrip("@")
    return username.lstrip("@")


def load_previous_failures(report_path: Path) -> dict[str, str]:
    if not report_path.exists():
        return {}
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    failed = data.get("failed_urls", [])
    result: dict[str, str] = {}
    for row in failed:
        url = (row or {}).get("url")
        error = (row or {}).get("error")
        if url:
            result[str(url)] = str(error or "")
    return result


def load_items(
    csv_path: Path,
    previous_failures: dict[str, str],
    platform_filters: set[str],
    limit: int,
    prefer_previous_failures: bool,
    only_previous_failures: bool,
) -> list[DownloadItem]:
    items: list[DownloadItem] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for idx, row in enumerate(reader):
            url = (row.get("pictureURL") or "").strip()
            if not url.startswith("http"):
                continue
            if only_previous_failures and url not in previous_failures:
                continue
            platform = (row.get("platform") or "").strip().lower()
            if platform_filters and platform not in platform_filters:
                continue
            username = (row.get("userName") or f"user_{idx}").strip()
            items.append(
                DownloadItem(
                    idx=idx,
                    username=username,
                    platform=platform,
                    original_url=url,
                    filename=get_filename(url, username, idx),
                    url_hash=get_url_hash(url),
                    previous_error=previous_failures.get(url),
                )
            )
    if prefer_previous_failures:
        items.sort(key=lambda item: (item.previous_error is None, item.idx))
    if limit > 0:
        items = items[:limit]
    return items


def build_existing_file_set(output_dir: Path) -> tuple[set[str], set[str]]:
    if not output_dir.exists():
        return set(), set()
    filenames: set[str] = set()
    url_hashes: set[str] = set()
    for path in output_dir.iterdir():
        if not path.is_file() or path.stat().st_size <= 0:
            continue
        filenames.add(path.name)
        stem = path.stem
        if "_" in stem:
            url_hashes.add(stem.split("_")[-1])
    return filenames, url_hashes


def dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def build_twitter_candidates(url: str) -> list[str]:
    parsed = urlparse(url)
    path = parsed.path
    stem = Path(path).stem
    suffix = Path(path).suffix or ".jpg"
    replacements = []
    if stem.endswith("_400x400"):
        base = stem[: -len("_400x400")]
        replacements.extend(
            [
                f"{base}_400x400{suffix}",
                f"{base}_bigger{suffix}",
                f"{base}_normal{suffix}",
                f"{base}_200x200{suffix}",
                f"{base}{suffix}",
            ]
        )
    else:
        replacements.extend(
            [
                f"{stem}_400x400{suffix}",
                f"{stem}_bigger{suffix}",
                f"{stem}_normal{suffix}",
                f"{stem}{suffix}",
            ]
        )
    base_dir = str(Path(path).parent).replace("\\", "/")
    candidates = [f"https://pbs.twimg.com{base_dir}/{name}" for name in replacements]
    if url.startswith("http://"):
        candidates.append("https://" + url[len("http://") :])
    candidates.append(url)
    return dedupe_keep_order(candidates)


def build_instagram_candidates(url: str) -> list[str]:
    parsed = urlparse(url)
    path = parsed.path
    candidates = [url]
    if url.startswith("http://"):
        candidates.append("https://" + url[len("http://") :])
    candidates.append(f"https://scontent.cdninstagram.com{path}")
    if "/s150x150/" in path:
        candidates.append(f"https://scontent.cdninstagram.com{path.replace('/s150x150/', '/')}")
        candidates.append(f"https://{parsed.netloc}{path.replace('/s150x150/', '/')}")
    if "/l/t51.2885-19/" in path:
        candidates.append(f"https://scontent.cdninstagram.com{path.replace('/l/t51.2885-19/', '/t51.2885-19/')}")
        candidates.append(f"https://{parsed.netloc}{path.replace('/l/t51.2885-19/', '/t51.2885-19/')}")
    return dedupe_keep_order(candidates)


def build_generic_candidates(url: str) -> list[str]:
    candidates = [url]
    if url.startswith("http://"):
        candidates.append("https://" + url[len("http://") :])
    return dedupe_keep_order(candidates)


async def resolve_instagram_avatar(
    session: aiohttp.ClientSession,
    external_resolver_semaphore: asyncio.Semaphore,
    username: str,
    timeout_sec: int,
) -> list[str]:
    if not username:
        return []
    api_urls = [
        f"https://i.instagram.com/api/v1/users/web_profile_info/?username={username}",
        f"https://www.instagram.com/api/v1/users/web_profile_info/?username={username}",
    ]
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0 Safari/537.36"
        ),
        "X-IG-App-ID": "936619743392459",
        "Accept": "*/*",
        "Referer": f"https://www.instagram.com/{username}/",
    }
    timeout = aiohttp.ClientTimeout(total=timeout_sec)
    for api_url in api_urls:
        try:
            async with external_resolver_semaphore:
                async with session.get(api_url, timeout=timeout, headers=headers, allow_redirects=True) as resp:
                    if resp.status != 200:
                        continue
                    payload = await resp.json(content_type=None)
        except Exception:
            continue
        user = ((payload or {}).get("data") or {}).get("user") or {}
        urls = [
            user.get("profile_pic_url_hd"),
            user.get("profile_pic_url"),
        ]
        return dedupe_keep_order([str(value) for value in urls if value])

    page_url = f"https://www.instagram.com/{username}/"
    try:
        async with external_resolver_semaphore:
            async with session.get(page_url, timeout=timeout, headers=headers, allow_redirects=True) as resp:
                if resp.status == 200:
                    html_text = await resp.text()
                    match = re.search(r'property="og:image" content="([^"]+)"', html_text)
                    if match:
                        return [html.unescape(match.group(1))]
    except Exception:
        pass
    return []


async def resolve_twitter_avatar(
    session: aiohttp.ClientSession,
    external_resolver_semaphore: asyncio.Semaphore,
    username: str,
    timeout_sec: int,
) -> list[str]:
    if not username:
        return []
    api_url = f"https://cdn.syndication.twimg.com/widgets/followbutton/info.json?screen_names={username}"
    timeout = aiohttp.ClientTimeout(total=timeout_sec)
    try:
        async with external_resolver_semaphore:
            async with session.get(api_url, timeout=timeout, allow_redirects=True) as resp:
                if resp.status != 200:
                    payload = []
                else:
                    payload = await resp.json(content_type=None)
    except Exception:
        payload = []
    if not isinstance(payload, list) or not payload:
        payload = []
    first_row = payload[0] if payload else {}
    avatar = (first_row or {}).get("profile_image_url_https") or (first_row or {}).get("profile_image_url")
    candidates: list[str] = []
    if avatar:
        avatar = str(avatar)
        candidates.extend(
            [
                avatar.replace("_normal.", "_400x400."),
                avatar.replace("_normal.", "_bigger."),
                avatar.replace("_normal.", "."),
                avatar,
            ]
        )
    page_urls = [
        f"https://x.com/{username}",
        f"https://twitter.com/{username}",
    ]
    for page_url in page_urls:
        try:
            async with external_resolver_semaphore:
                async with session.get(page_url, timeout=timeout, headers=headers, allow_redirects=True) as resp:
                    if resp.status != 200:
                        continue
                    html_text = await resp.text()
        except Exception:
            continue

        og_matches = re.findall(r'property="og:image" content="([^"]+)"', html_text)
        for match in og_matches:
            image_url = html.unescape(match)
            if "profile_images" in image_url:
                candidates.extend(
                    [
                        image_url.replace("_normal.", "_400x400."),
                        image_url.replace("_normal.", "_bigger."),
                        image_url.replace("_normal.", "."),
                        image_url,
                    ]
                )

        embedded_matches = re.findall(r'"profile_image_url_https":"([^"]+)"', html_text)
        for match in embedded_matches:
            image_url = html.unescape(match).replace("\\u002F", "/").replace("\\/", "/")
            candidates.extend(
                [
                    image_url.replace("_normal.", "_400x400."),
                    image_url.replace("_normal.", "_bigger."),
                    image_url.replace("_normal.", "."),
                    image_url,
                ]
            )
    candidates.extend(
        [
            f"https://unavatar.io/twitter/{username}",
            f"https://unavatar.io/x/{username}",
        ]
    )
    return dedupe_keep_order(candidates)


async def get_resolved_candidates(
    session: aiohttp.ClientSession,
    cache: dict[tuple[str, str], list[str]],
    external_resolver_semaphore: asyncio.Semaphore,
    item: DownloadItem,
    resolver_timeout_sec: int,
) -> list[str]:
    username = normalize_username(item.username, item.platform)
    key = (item.platform, username)
    if key in cache:
        return cache[key]

    resolved: list[str] = []
    if item.platform == "instagram":
        resolved = await resolve_instagram_avatar(session, external_resolver_semaphore, username, resolver_timeout_sec)
    elif item.platform == "twitter":
        resolved = await resolve_twitter_avatar(session, external_resolver_semaphore, username, resolver_timeout_sec)

    cache[key] = resolved
    return resolved


async def fetch_image(
    session: aiohttp.ClientSession,
    url: str,
    timeout_sec: int,
) -> tuple[bool, bytes | None, str | None, int | None]:
    timeout = aiohttp.ClientTimeout(total=timeout_sec)
    try:
        async with session.get(url, timeout=timeout, allow_redirects=True) as resp:
            status = resp.status
            if status != 200:
                return False, None, f"HTTP {status}", status
            content_type = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            payload = await resp.read()
            if not payload:
                return False, None, "Empty response body", status
            if content_type and content_type not in IMAGE_CONTENT_TYPES:
                if not content_type.startswith("image/"):
                    return False, None, f"Non-image content-type: {content_type}", status
            return True, payload, None, status
    except asyncio.TimeoutError:
        return False, None, "Timeout", None
    except aiohttp.ClientError as exc:
        return False, None, f"ClientError: {exc}", None
    except Exception as exc:
        return False, None, f"Unknown: {exc}", None


async def download_one(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    resolver_cache: dict[tuple[str, str], list[str]],
    external_resolver_semaphore: asyncio.Semaphore,
    output_dir: Path,
    item: DownloadItem,
    existing_hashes: set[str],
    timeout_sec: int,
    resolver_timeout_sec: int,
) -> dict[str, Any]:
    filepath = output_dir / item.filename
    if item.url_hash in existing_hashes:
        return {
            "idx": item.idx,
            "username": item.username,
            "platform": item.platform,
            "url": item.original_url,
            "final_url": item.original_url,
            "filename": item.filename,
            "success": True,
            "skipped": True,
            "error": None,
            "previous_error": item.previous_error,
        }
    if filepath.exists() and filepath.stat().st_size > 0:
        return {
            "idx": item.idx,
            "username": item.username,
            "platform": item.platform,
            "url": item.original_url,
            "final_url": item.original_url,
            "filename": item.filename,
            "success": True,
            "skipped": True,
            "error": None,
            "previous_error": item.previous_error,
        }

    if item.platform == "instagram":
        initial_candidates = build_instagram_candidates(item.original_url)
    elif item.platform == "twitter":
        initial_candidates = build_twitter_candidates(item.original_url)
    else:
        initial_candidates = build_generic_candidates(item.original_url)

    resolved_candidates = await get_resolved_candidates(
        session=session,
        cache=resolver_cache,
        external_resolver_semaphore=external_resolver_semaphore,
        item=item,
        resolver_timeout_sec=resolver_timeout_sec,
    )
    candidates = dedupe_keep_order(resolved_candidates + initial_candidates)

    async with semaphore:
        errors: list[str] = []
        for candidate in candidates:
            ok, payload, error, status = await fetch_image(session, candidate, timeout_sec)
            if not ok:
                errors.append(f"{candidate} -> {error}")
                if status in {403, 404, 410}:
                    continue
                continue
            async with aiofiles.open(filepath, "wb") as fh:
                await fh.write(payload or b"")
            existing_hashes.add(item.url_hash)
            return {
                "idx": item.idx,
                "username": item.username,
                "platform": item.platform,
                "url": item.original_url,
                "final_url": candidate,
                "filename": item.filename,
                "success": True,
                "skipped": False,
                "size_bytes": len(payload or b""),
                "error": None,
                "previous_error": item.previous_error,
            }

    return {
        "idx": item.idx,
        "username": item.username,
        "platform": item.platform,
        "url": item.original_url,
        "final_url": None,
        "filename": item.filename,
        "success": False,
        "skipped": False,
        "error": " | ".join(errors[-6:]) if errors else "No candidate URLs available",
        "previous_error": item.previous_error,
    }


async def run_downloads(args: argparse.Namespace, items: list[DownloadItem]) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(args.max_concurrent)
    external_resolver_semaphore = asyncio.Semaphore(args.external_resolver_concurrent)
    connector = aiohttp.TCPConnector(limit=args.max_concurrent, ttl_dns_cache=300, ssl=False)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0 Safari/537.36"
        ),
        "Accept": "*/*",
    }
    resolver_cache: dict[tuple[str, str], list[str]] = {}
    _, existing_hashes = build_existing_file_set(args.output_dir)
    async with aiohttp.ClientSession(connector=connector, headers=headers, trust_env=True) as session:
        tasks = [
            download_one(
                session=session,
                semaphore=semaphore,
                resolver_cache=resolver_cache,
                external_resolver_semaphore=external_resolver_semaphore,
                output_dir=args.output_dir,
                item=item,
                existing_hashes=existing_hashes,
                timeout_sec=args.timeout_sec,
                resolver_timeout_sec=args.resolver_timeout_sec,
            )
            for item in items
        ]

        results: list[dict[str, Any]] = []
        completed = 0
        total = len(tasks)
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            completed += 1
            if completed % 100 == 0 or completed == total:
                success = sum(1 for row in results if row["success"] and not row.get("skipped"))
                skipped = sum(1 for row in results if row.get("skipped"))
                failed = sum(1 for row in results if not row["success"])
                print(
                    f"[{completed}/{total}] new={success:,} skipped={skipped:,} failed={failed:,}"
                )
        return results


def save_outputs(
    results: list[dict[str, Any]],
    elapsed_sec: float,
    args: argparse.Namespace,
    existing_before: int,
) -> None:
    success_new = [row for row in results if row["success"] and not row.get("skipped")]
    success_skipped = [row for row in results if row.get("skipped")]
    failed = [row for row in results if not row["success"]]

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_csv": str(args.csv_path),
        "total_attempted": len(results),
        "success_new": len(success_new),
        "already_present": len(success_skipped),
        "failed": len(failed),
        "success_rate_new_only": f"{(len(success_new) / len(results) * 100):.1f}%" if results else "0.0%",
        "success_rate_including_existing": (
            f"{((len(success_new) + len(success_skipped)) / len(results) * 100):.1f}%"
            if results
            else "0.0%"
        ),
        "elapsed_seconds": round(elapsed_sec, 1),
        "output_directory": str(args.output_dir),
        "files_in_output_before": existing_before,
        "files_in_output_after": existing_before + len(success_new),
        "failed_urls": [
            {
                "idx": row["idx"],
                "username": row["username"],
                "platform": row["platform"],
                "url": row["url"],
                "error": row["error"],
                "previous_error": row.get("previous_error"),
            }
            for row in failed
        ],
        "downloaded": [
            {
                "idx": row["idx"],
                "username": row["username"],
                "platform": row["platform"],
                "url": row["url"],
                "final_url": row["final_url"],
                "filename": row["filename"],
                "size_bytes": row.get("size_bytes"),
                "skipped": bool(row.get("skipped")),
            }
            for row in results
            if row["success"]
        ],
    }

    args.report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with args.failed_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["idx", "username", "platform", "url", "error", "previous_error"],
        )
        writer.writeheader()
        writer.writerows(report["failed_urls"])


def print_summary(results: list[dict[str, Any]], elapsed_sec: float, output_dir: Path) -> None:
    success_new = sum(1 for row in results if row["success"] and not row.get("skipped"))
    success_skipped = sum(1 for row in results if row.get("skipped"))
    failed = sum(1 for row in results if not row["success"])
    print("=" * 72)
    print(f"new downloads    : {success_new:,}")
    print(f"already present  : {success_skipped:,}")
    print(f"failed           : {failed:,}")
    print(f"elapsed seconds  : {elapsed_sec:.1f}")
    print(f"output dir       : {output_dir}")
    print("=" * 72)


async def async_main(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    previous_failures = load_previous_failures(args.report_path)
    platform_filters = {value.strip().lower() for value in args.platform if value.strip()}
    items = load_items(
        csv_path=args.csv_path,
        previous_failures=previous_failures,
        platform_filters=platform_filters,
        limit=args.limit,
        prefer_previous_failures=args.prefer_previous_failures,
        only_previous_failures=args.only_previous_failures,
    )
    existing_filenames, _ = build_existing_file_set(args.output_dir)
    existing_before = len(existing_filenames)

    print(f"csv path         : {args.csv_path}")
    print(f"total url rows   : {len(items):,}")
    print(f"output existing  : {existing_before:,}")
    print(f"platform filter  : {sorted(platform_filters) if platform_filters else 'all'}")
    print(f"max concurrent   : {args.max_concurrent}")

    started = time.time()
    results = await run_downloads(args, items)
    elapsed_sec = time.time() - started

    save_outputs(results, elapsed_sec, args, existing_before)
    print_summary(results, elapsed_sec, args.output_dir)
    return 0


def main() -> int:
    args = parse_args()
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
