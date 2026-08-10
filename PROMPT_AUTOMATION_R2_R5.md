# Prompt สำหรับ AI agent — รัน Experiment R2/R3/R4/R5 ต่อจาก R1

> วิธีใช้: copy บล็อกข้างล่างทั้งก้อน วางให้ AI coding agent (Claude Code / อื่น ๆ) ที่เปิดอยู่ใน
> `d:\66070260-Year3_Term2\Project1\Code` — R1 เสร็จแล้ว prompt นี้ทำต่อจนจบตาราง R0–R5

---

```text
You are working in an identity-resolution research project (matching social-media
profiles of the same person across platforms). Work autonomously; do not ask
questions unless truly blocked. Read EXPERIMENT_PLAN.md first — it defines the
full R0–R5 experiment matrix. R0 (baseline) and R1 (GA decision) are DONE; your
job is R2, R3, and R4/R5.

## Environment (verified facts — do not re-derive, do not change)
- Python: .venv\Scripts\python.exe (Windows). jellyfish and
  sentence-transformers 5.3.0 are already installed.
- Shared infrastructure: exp_lib.py — ALWAYS import from it. It provides:
  build_cache() -> experiments/scored_pairs_enriched.parquet (2,086,245 pairs:
  profile_id_a/b, score, decision, decision_source, actual, split, name_sim),
  split_constants(), evaluate(), the entity-aware val/test split (hash of
  user_folder, 70/30), and NO_MATCH/REVIEW/MATCH int codes.
- Ground truth: actual comes from user_folder equality (29,247 true pairs
  total). Profiles CSV:
  Project-for-Work/data_for_project/normalized_profiles_with_profile_id.csv
  (key column profile_row_id = the profile_id_a/b in the pairs; text columns
  userName, fullName, bio, location, platform).
- Existing MLP training pipeline (17 features) to copy the architecture from:
  Project-for-Work/train_data/stage10_13_training_pipeline.py and the noleak
  variants in Project-for-Work/train_data/stage10_13_training_noleak*/.
- Feature source for pairs: labeled_pairs.parquet (repo root and
  Project-for-Work/train_data/) — inspect its schema before use.
- R1 reference results: experiments/r1_results.json, best genome
  t_m=0.999 t_r=0.9642 c_promote=0.9875 c_demote=0.5052; GA code in
  exp_r1_ga_redecision.py (reuse decide_code/run_ga/clip, weights
  W_FP=5, W_FN=1, W_REV=0.02, seed 42, pop 60, gens 40).

## Hard rules (violating any of these invalidates the experiment)
1. NEVER create a new train/val/test split. Only the split column from
   exp_lib.build_cache(). Any model training uses val for training/tuning and
   test ONLY for the final report. Pairs with split=='drop' are excluded from
   train and eval but kept in the 'full' projection.
2. Every experiment reports through exp_lib.evaluate() with
   split_constants() — never hand-compute precision/recall.
3. Sanity gate before reporting anything new: reproducing R0 on 'full' must
   give exactly FP=925, REVIEW=86,296, TP=19,624 (already true in
   experiments/r1_results.json; if your pipeline breaks this, stop and fix).
4. Deterministic: seed 42 everywhere (numpy, torch, GA).
5. Do not modify exp_lib.py, exp_r1_ga_redecision.py, or anything under
   Project-for-Work/train_data/stage15_crm_entity_pipeline/artifacts/
   (read-only inputs). New code goes in new files exp_r2_*.py, exp_r4_*.py;
   all outputs go to experiments/.
6. Long jobs (embedding 100k+ profiles, training): run in background, write
   progress to a log file under experiments/, and checkpoint so a rerun
   resumes instead of restarting.

## Task R2 — BERT representation (do this first)
1. Build profile texts: fullName + ' ' + bio + ' ' + location (strip
   empty/'nan'). Encode ALL profiles with
   SentenceTransformer('all-MiniLM-L6-v2'), normalize_embeddings=True,
   batch_size=256, CPU is fine. Cache to
   experiments/profile_embeddings_minilm.npy + a row-id index file.
2. New pair feature bert_cos = dot(emb_a, emb_b) for every pair in the
   scored cache. Save to experiments/pair_bert_cos.parquet.
3. Retrain the existing IdentityMLP with the SAME architecture and
   hyperparameters as the current 17-feature model (find them in
   stage10_13_training_pipeline.py / the noleak variant — copy, do not
   redesign), input widened 17->18 with bert_cos as feature 18. Train on
   val-split pairs only. Recalibrate probabilities the same way the original
   pipeline does (if it uses Platt/isotonic, reuse it).
4. Score all pairs -> probability_r2. Apply the ORIGINAL manual thresholds
   (MATCH >= 0.98, REVIEW >= 0.95, else NO_MATCH) = experiment R2.
   Report on test and full via evaluate().
5. R3 = rerun the GA from exp_r1_ga_redecision.py on probability_r2 (same
   weights/seed/budget, tune on val, report on test) -> experiments/
   r3_results.json.
6. Also report ONE extra diagnostic: among the 2,242 scored-but-rejected
   false negatives (actual==1, old decision NO_MATCH), how many does R2/R3
   recover? That is the number the professor cares about.

## Task R4/R5 — Bloom filter privacy tradeoff (after R2)
1. Implement bigram Bloom encoding: bloom(s, L, k=10) with md5(salt+j+gram),
   and dice(a, b). Encode userName and fullName.
2. Replace the plaintext name-similarity features used by the MLP with
   bloom-dice equivalents (leave non-name features unchanged), retrain the
   same MLP per L in {2000, 1000, 500, 250}; R4 = manual thresholds,
   R5 = GA. Report F1-vs-L as a table and a matplotlib plot ->
   experiments/r4_privacy_tradeoff.png (+ .json).
3. If runtime is prohibitive, it is acceptable to subsample NEGATIVE pairs
   for training only (never for evaluation) — say so explicitly in the report.

## Deliverables (definition of done)
- experiments/r2_results.json, r3_results.json, r4_privacy_tradeoff.json/.png
- experiments/REPORT_R0_R5.md: one page, Thai, containing (a) the completed
  R0–R5 table with FP / FN / REVIEW / precision / recall / F1 on TEST for
  every cell, plus the full-system projection row for the best config;
  (b) the FN-recovery count vs the 2,242 pool; (c) explicit statement that
  3,316 blocking-missed FN are unreachable by every experiment here;
  (d) 3-5 sentence conclusion: which transformation improved eval, by how
  much, and what you recommend deploying.
- Every number in the report must come from a saved JSON produced by
  evaluate(), not typed by hand.
```
