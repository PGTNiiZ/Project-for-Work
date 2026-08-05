from __future__ import annotations

import json
import pickle
import textwrap
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


PKG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PKG_DIR.parent
RES_DIR = PKG_DIR / "res"
FIG_DIR = PKG_DIR / "fig"
DOC_DIR = PKG_DIR / "doc"
LOG_DIR = PKG_DIR / "log"

MAIN_RUN = PROJECT_ROOT / "train_data" / "stage7_13_multimodal_suite" / "runs" / "image_context_r075_h20_s42"
PAIR_FEATURES = MAIN_RUN / "artifacts" / "pair_features.parquet"
FEATURE_COLS_PKL = MAIN_RUN / "models" / "feature_cols.pkl"


MODEL_META = {
    "logreg": {
        "why_selected": "Linear baseline ที่ตีความง่าย ใช้ตรวจว่าฟีเจอร์หลักแยกคลาสได้ด้วยเส้นแบ่งตรงหรือไม่",
        "strengths": "เร็วมาก, อธิบาย coefficient ได้, ใช้เป็น baseline ที่เชื่อถือได้",
        "weaknesses": "จับ nonlinear interaction และ threshold effect ได้จำกัด",
    },
    "gb": {
        "why_selected": "Boosted trees เป็นตัวหลักของ pipeline เดิมและเหมาะกับ tabular similarity features",
        "strengths": "จับ nonlinear interaction ได้ดี, ให้ feature importance, ผลมักนิ่งบนข้อมูลตาราง",
        "weaknesses": "เทรนช้ากว่า linear model และต้องคุม depth/learning rate",
    },
    "rf": {
        "why_selected": "Bagged trees เป็น baseline ensemble มาตรฐานสำหรับข้อมูลตาราง",
        "strengths": "ทน noise, ไม่ไวต่อ scaling, จับ interaction ได้ดี",
        "weaknesses": "probability มักไม่คมเท่า boosting และ model ใหญ่",
    },
    "extra_trees": {
        "why_selected": "Randomized tree ensemble ใช้ดูว่าการสุ่ม split เพิ่ม generalization ได้หรือไม่",
        "strengths": "เร็ว, ลด variance ได้ดี, มักแข็งแรงบน feature จำนวนปานกลาง",
        "weaknesses": "อาจ overfit noise บางส่วนและ probability ไม่ smooth",
    },
    "linear_svm": {
        "why_selected": "Large-margin linear model ใช้ดูว่าการแบ่งคลาสแบบเส้นขอบกว้างช่วยกว่าหรือด้อยกว่า logistic regression หรือไม่",
        "strengths": "เร็ว, เหมาะกับ feature space เชิง similarity, เป็น baseline เชิง margin ที่ดี",
        "weaknesses": "ไม่ให้ probability ตรง ๆ, แปลผลยากกว่า logistic regression, จับ nonlinear interaction ไม่ได้",
    },
    "adaboost": {
        "why_selected": "Simple additive ensemble ใช้เป็น reference ของ boosting family ที่เบากว่า",
        "strengths": "ตีความง่ายกว่าบาง ensemble และใช้เป็น baseline ของ boosting",
        "weaknesses": "ไวต่อ noise/outlier และมักด้อยกว่า gradient boosting",
    },
    "mlp": {
        "why_selected": "Neural baseline เพื่อดูว่าการเพิ่มความยืดหยุ่นของโมเดลช่วยกับฟีเจอร์ชุดนี้หรือไม่",
        "strengths": "เรียนรู้ interaction ซับซ้อนได้, เป็นตัวแทนของ neural family",
        "weaknesses": "จูนยาก, แปลผลยาก, ผลลัพธ์ไม่นิ่งเท่า tree ensemble บนข้อมูลตารางขนาดนี้",
    },
}


def ensure_dirs() -> None:
    for path in [RES_DIR, FIG_DIR, DOC_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_dataset() -> tuple[pd.DataFrame, list[str]]:
    feature_df = pd.read_parquet(PAIR_FEATURES)
    with FEATURE_COLS_PKL.open("rb") as fh:
        feature_cols = pickle.load(fh)
    return feature_df, feature_cols


def choose_threshold(labels: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    thresholds = np.arange(0.05, 0.96, 0.01)
    scores = [f1_score(labels, (probs >= thr).astype(int), zero_division=0) for thr in thresholds]
    best_idx = int(np.argmax(scores))
    return float(thresholds[best_idx]), float(scores[best_idx])


def expected_calibration_error(labels: np.ndarray, probs: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        if right == 1.0:
            mask = (probs >= left) & (probs <= right)
        else:
            mask = (probs >= left) & (probs < right)
        if not mask.any():
            continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        total += (mask.mean()) * abs(acc - conf)
    return float(total)


def get_models(seed: int) -> dict[str, tuple[object, bool]]:
    return {
        "logreg": (
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
            True,
        ),
        "gb": (
            GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                random_state=seed,
            ),
            False,
        ),
        "rf": (
            RandomForestClassifier(
                n_estimators=400,
                max_depth=18,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=1,
                random_state=seed,
            ),
            False,
        ),
        "extra_trees": (
            ExtraTreesClassifier(
                n_estimators=500,
                max_depth=20,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=1,
                random_state=seed,
            ),
            False,
        ),
        "linear_svm": (
            LinearSVC(
                C=1.0,
                class_weight="balanced",
                random_state=seed,
            ),
            True,
        ),
        "adaboost": (
            AdaBoostClassifier(
                n_estimators=300,
                learning_rate=0.05,
                random_state=seed,
            ),
            False,
        ),
        "mlp": (
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size=256,
                learning_rate_init=1e-3,
                max_iter=200,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=seed,
            ),
            True,
        ),
    }


def fit_and_score(
    model_name: str,
    model: object,
    use_scaler: bool,
    splits: dict[str, pd.DataFrame],
    feature_cols: list[str],
) -> dict[str, object]:
    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]

    train_x = train_df[feature_cols].fillna(0.0).to_numpy()
    val_x = val_df[feature_cols].fillna(0.0).to_numpy()
    test_x = test_df[feature_cols].fillna(0.0).to_numpy()
    train_y = train_df["label"].to_numpy()
    val_y = val_df["label"].to_numpy()
    test_y = test_df["label"].to_numpy()

    scaler = None
    if use_scaler:
        scaler = StandardScaler()
        train_x = scaler.fit_transform(train_x)
        val_x = scaler.transform(val_x)
        test_x = scaler.transform(test_x)

    started = time.perf_counter()
    model.fit(train_x, train_y)
    fit_seconds = time.perf_counter() - started

    if hasattr(model, "predict_proba"):
        train_raw = model.predict_proba(train_x)[:, 1]
        val_raw = model.predict_proba(val_x)[:, 1]
        test_raw = model.predict_proba(test_x)[:, 1]
    elif hasattr(model, "decision_function"):
        train_raw = model.decision_function(train_x)
        val_raw = model.decision_function(val_x)
        test_raw = model.decision_function(test_x)
    else:
        raise ValueError(f"Model {model_name} does not expose predict_proba or decision_function.")

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(val_raw, val_y)
    train_probs = calibrator.predict(train_raw)
    val_probs = calibrator.predict(val_raw)
    test_probs = calibrator.predict(test_raw)

    threshold, best_val_f1 = choose_threshold(val_y, val_probs)
    test_pred = (test_probs >= threshold).astype(int)
    val_pred = (val_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(test_y, test_pred).ravel()

    return {
        "model": model_name,
        "fit_seconds": float(fit_seconds),
        "feature_count": len(feature_cols),
        "use_scaler": use_scaler,
        "threshold": threshold,
        "val_ap": float(average_precision_score(val_y, val_probs)),
        "val_auc": float(roc_auc_score(val_y, val_probs)),
        "val_f1": float(f1_score(val_y, val_pred, zero_division=0)),
        "val_precision": float(precision_score(val_y, val_pred, zero_division=0)),
        "val_recall": float(recall_score(val_y, val_pred, zero_division=0)),
        "val_ece": expected_calibration_error(val_y, val_probs),
        "best_val_f1": best_val_f1,
        "test_ap": float(average_precision_score(test_y, test_probs)),
        "test_auc": float(roc_auc_score(test_y, test_probs)),
        "test_f1": float(f1_score(test_y, test_pred, zero_division=0)),
        "test_precision": float(precision_score(test_y, test_pred, zero_division=0)),
        "test_recall": float(recall_score(test_y, test_pred, zero_division=0)),
        "test_ece": expected_calibration_error(test_y, test_probs),
        "test_tn": int(tn),
        "test_fp": int(fp),
        "test_fn": int(fn),
        "test_tp": int(tp),
    }


def add_narrative(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["val_composite"] = 0.45 * out["val_ap"] + 0.35 * out["val_f1"] + 0.20 * out["val_auc"]
    out = out.sort_values(["val_composite", "test_ap", "test_f1"], ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    best_score = float(out.loc[0, "val_composite"])

    why, strengths, weaknesses, conclusions = [], [], [], []
    for row in out.itertuples(index=False):
        meta = MODEL_META[row.model]
        why.append(meta["why_selected"])
        strengths.append(meta["strengths"])
        weaknesses.append(meta["weaknesses"])

        if row.rank == 1:
            conclusion = "ตัวหลักที่แนะนำ: ชนะตาม validation composite และให้ test metrics สมดุลที่สุด"
        elif best_score - float(row.val_composite) <= 0.003:
            conclusion = "ตัวใกล้เคียง: คะแนนตามหลังน้อย ใช้เป็น alternative หรือ sensitivity check ได้"
        elif row.model == "logreg":
            conclusion = "baseline เชิงตีความ: ควรคงไว้ในเล่มเพื่อพิสูจน์ว่า nonlinear model ช่วยจริง"
        elif row.model == "mlp":
            conclusion = "neural reference: ใช้พิสูจน์ว่าการเพิ่มความซับซ้อนไม่ได้แปลว่าดีกว่าเสมอ"
        else:
            conclusion = "reference model: มีประโยชน์ในการเทียบ family ของโมเดล แต่ไม่ใช่ตัวหลัก"
        conclusions.append(conclusion)

    out["why_selected"] = why
    out["strengths"] = strengths
    out["weaknesses"] = weaknesses
    out["conclusion"] = conclusions
    return out


def save_plot(df: pd.DataFrame) -> None:
    plot_df = df.copy()
    plot_df["label"] = plot_df["model"] + "  #" + plot_df["rank"].astype(str)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    metric_cols = ["test_ap", "test_auc", "test_f1"]
    x = np.arange(len(plot_df))
    width = 0.24
    for idx, col in enumerate(metric_cols):
        axes[0].bar(x + (idx - 1) * width, plot_df[col], width=width, label=col.replace("test_", "").upper())
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(plot_df["label"], rotation=35, ha="right")
    axes[0].set_ylim(0.7, 1.0)
    axes[0].set_title("Model Family Comparison")
    axes[0].legend()

    axes[1].scatter(plot_df["test_recall"], plot_df["test_precision"], s=80)
    for row in plot_df.itertuples(index=False):
        axes[1].annotate(row.model, (row.test_recall, row.test_precision), xytext=(5, 5), textcoords="offset points")
    axes[1].set_xlabel("Test Recall")
    axes[1].set_ylabel("Test Precision")
    axes[1].set_title("Precision vs Recall")
    axes[1].set_xlim(0.75, 0.95)
    axes[1].set_ylim(0.75, 0.98)
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_family_cmp.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_confusion_grid(df: pd.DataFrame) -> None:
    plot_df = df.sort_values("rank").reset_index(drop=True)
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()

    for ax in axes:
        ax.axis("off")

    for ax, row in zip(axes, plot_df.itertuples(index=False)):
        cm = np.array([[row.test_tn, row.test_fp], [row.test_fn, row.test_tp]], dtype=int)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{row.model}  (F1={row.test_f1:.3f})")
        ax.set_xticks([0, 1], labels=["NO_MATCH", "MATCH"])
        ax.set_yticks([0, 1], labels=["NO_MATCH", "MATCH"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for (i, j), value in np.ndenumerate(cm):
            ax.text(j, i, f"{value:,}", ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis("on")

    fig.suptitle("Confusion Matrices by Model Family", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_family_cm_grid.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_markdown(df: pd.DataFrame) -> None:
    cols = [
        "rank",
        "model",
        "test_ap",
        "test_auc",
        "test_f1",
        "test_precision",
        "test_recall",
        "threshold",
        "why_selected",
        "strengths",
        "weaknesses",
        "conclusion",
    ]
    table_df = df[cols].copy()
    metric_cols = ["test_ap", "test_auc", "test_f1", "test_precision", "test_recall", "threshold"]
    for col in metric_cols:
        table_df[col] = table_df[col].map(lambda value: f"{value:.4f}")

    header = "| " + " | ".join(table_df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(table_df.columns)) + " |"
    body = [
        "| " + " | ".join(str(value).replace("\n", " ") for value in row) + " |"
        for row in table_df.itertuples(index=False, name=None)
    ]

    lines = [
        "# Model Comparison",
        "",
        "ชุดนี้ benchmark บน main multimodal run `image_context_r075_h20_s42` โดยใช้ feature 41 ตัว, split เดิม (`train/val/test`), isotonic calibration บน validation และเลือก threshold ที่ให้ F1 สูงสุดบน validation เช่นเดียวกับ pipeline หลัก",
        "",
        "## Summary",
        "",
        header,
        sep,
        *body,
        "",
        "## Reading Guide",
        "",
        "- `test_ap` ใช้ดูคุณภาพการจัดอันดับภายใต้ class imbalance",
        "- `test_auc` ใช้ดูความสามารถในการแยก positive/negative โดยรวม",
        "- `test_f1` ใช้ดูสมดุล precision/recall หลังเลือก threshold",
        "- model ที่แนะนำควรดูทั้ง ranking, ความต่างของคะแนน, และความสามารถในการอธิบายต่ออาจารย์",
        "",
        "## Recommendation",
        "",
        textwrap.fill(
            "แนะนำให้ใช้ Gradient Boosting หรือ model ที่ชนะใน benchmark นี้เป็นตัวหลักของเล่ม โดยคง Logistic Regression เป็น baseline เชิงตีความ และเก็บอีก 1-2 model ที่คะแนนใกล้เคียงเป็นตัวเปรียบเทียบ เพื่อพิสูจน์ว่าการเลือก final model ไม่ได้มาจากความรู้สึก แต่ผ่านการทดลองบน feature และ split เดียวกันแล้ว",
            width=110,
        ),
        "",
    ]
    (DOC_DIR / "model_compare.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    feature_df, feature_cols = load_dataset()
    splits = {
        name: feature_df[feature_df["split_name"] == name].reset_index(drop=True)
        for name in ["train", "val", "test"]
    }

    rows = []
    for model_name, (model, use_scaler) in get_models(seed=42).items():
        print(f"Running {model_name} ...")
        rows.append(fit_and_score(model_name, model, use_scaler, splits, feature_cols))

    result_df = add_narrative(pd.DataFrame(rows))
    numeric_cols = [col for col in result_df.columns if col not in {"why_selected", "strengths", "weaknesses", "conclusion"}]
    result_df[numeric_cols].to_csv(RES_DIR / "model_family_cmp.csv", index=False, encoding="utf-8-sig")
    result_df.to_csv(RES_DIR / "model_family_table.csv", index=False, encoding="utf-8-sig")
    save_plot(result_df)
    save_confusion_grid(result_df)
    save_markdown(result_df)

    summary = {
        "main_run": str(MAIN_RUN),
        "pair_features": str(PAIR_FEATURES),
        "feature_count": len(feature_cols),
        "rows": int(len(feature_df)),
        "split_counts": {name: int(len(df)) for name, df in splits.items()},
        "best_model": str(result_df.loc[0, "model"]),
        "best_val_composite": float(result_df.loc[0, "val_composite"]),
        "timestamp": pd.Timestamp.now(tz="Asia/Bangkok").isoformat(),
    }
    (LOG_DIR / "model_family_cmp.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(result_df[["rank", "model", "test_ap", "test_auc", "test_f1", "test_precision", "test_recall"]].to_string(index=False))


if __name__ == "__main__":
    main()
