# Prompt Library — Cross-Platform User Identity Linkage (LinkSocial)

**Project focus:** a NEW linkage algorithm that raises accuracy on the *hard* case —
true-positive profile pairs that share **zero lexical/surface similarity** (different names,
bios, text). Pairwise string matching cannot solve this; the library below is redesigned
around that single goal and grounded in the real LinkSocial data.

## How to use
1. **Always prepend the Context Block (§0) to every prompt.** In each prompt `[+CONTEXT]`
   marks where it goes. Without it the model answers with generic CRM theory that assumes
   fields this dataset does not have (timestamps, transactions, journey stages).
2. Run the **Core Novelty Track (§1–§5)** first — it is the heart of the contribution.
3. The **Supporting Pipeline Track (Steps 2–4, 8, 10–12)** fills out the thesis around it.

## Data reality baked into these prompts
- Static profile snapshot: **36,807 profiles**, 3 platforms (Twitter / Google+ / Instagram),
  **15,297 identities** via the `user_folder` ground-truth key.
- Fields that exist: `userName, fullName, bio, location` (35% non-null, some raw GPS),
  `externalUrl` (83% non-null), `pictureURL`, precomputed `bigrams`.
- **No timestamps, activity logs, transactions, or temporal streams.** Anything needing them
  is out of scope for this dataset — flag as future work only.
- The only non-lexical footprints available: **image** (`pictureURL` → CLIP/face embedding),
  **external-URL domain** (83%), **location** (35%, some GPS).
- The new algorithm must lean on: (a) a **learned cross-platform invariant embedding**
  (contrastive metric learning + embedding-space alignment), and (b) **graph/collective
  propagation** to link identities that have no direct pairwise evidence.

---

## §0 — Context Block (prepend to EVERY prompt)

> **PROJECT CONTEXT (prepend to every prompt).** I am a data science student building a
> cross-platform User Identity Linkage (UIL) system on the LinkSocial dataset: 36,807 static
> profiles across Twitter, Google+, and Instagram, grouped into 15,297 ground-truth identities
> via a `user_folder` key. Available fields per profile are ONLY: `userName, fullName, bio,
> location` (35% non-null, some raw GPS), `externalUrl` (83% non-null), `pictureURL`, and
> precomputed `bigrams`. **There are NO timestamps, activity logs, transactions, or
> temporal/behavioral streams** — do not propose any analysis that requires them; if a
> technique needs them, flag it as out-of-scope for this dataset. **The core research
> challenge is linking TRUE-POSITIVE profile pairs that share ZERO lexical/surface similarity**
> (different names, bios, and text), which pairwise string-matching cannot solve. My goal is a
> NEW algorithm that raises linkage accuracy specifically on this hard, zero-overlap stratum.
> Reason step-by-step and ground every recommendation in these exact data constraints.

---

# Core Novelty Track

## §1 — Research Question (Step 1)

> [+CONTEXT] Act as a data science student formulating a rigorous ML research question for the
> hard case of UIL. Reason step-by-step: (1) Frame the problem as "linking identities whose
> surface attributes are non-overlapping" — why do lexical baselines (Jaro-Winkler, TF-IDF on
> names) provably fail here, and what latent signal must replace them given my non-lexical
> fields (image, external-URL domain, location)? (2) Translate this into a measurable
> objective: cross-platform pair classification evaluated *specifically on the
> zero-lexical-similarity subset*, not the easy full set. (3) Define success as lift over a
> lexical baseline on that hard subset, using AUPRC and Recall@k for the minority match class.
> (4) Name confounders: `user_folder` leakage across splits, the 35% location / 83% URL
> missingness, and hard-negative contamination. Output one formal IMRaD research-question
> paragraph, one primary hypothesis ("a learned cross-platform invariant embedding links
> zero-overlap pairs better than lexical + tabular ensembles"), and two sub-hypotheses (one for
> image/URL/geo fusion, one for embedding-space alignment).

## §2 — The New Algorithm (Step 7, core)

> [+CONTEXT] Act as a data science student designing a NEW model whose whole purpose is to link
> profile pairs with zero surface similarity. Reason through the design, layer by layer:
> (1) Justify a **two-tower / siamese multimodal encoder** with a shared projection head, over
> the current pairwise-feature XGBoost — why does metric learning generalize to unseen surface
> forms that a tree on precomputed similarities cannot? (2) Design the **contrastive objective**
> (InfoNCE / triplet with online hard-negative mining) that pulls same-`user_folder`
> cross-platform profiles together and pushes apart the hard negatives I already generate in
> Stage 8. (3) Add a **cross-platform embedding-alignment step** (Procrustes / adversarial
> mapping, à la cross-lingual embedding alignment): learn a mapping so Twitter-space and
> Instagram-space overlap, enabling matches with no shared tokens — explain the training
> procedure. (4) Specify how to keep the strong XGBoost as a meta-learner that consumes the
> learned embedding distance PLUS the residual lexical/geo/URL features, so the new model
> strictly dominates the baseline. Output a layer-by-layer architecture spec, the exact loss,
> and the training loop with hard-negative mining.

## §3 — Non-Lexical Representation (Step 5, core)

> [+CONTEXT] Act as a data science student engineering the non-lexical feature/representation
> set — the only signals that can link zero-overlap pairs in LinkSocial. For EACH of my three
> real footprints, reason through construction and predictive justification: (1) **Image** —
> CLIP and/or face embeddings from `pictureURL`; cosine as a feature AND as a tower input; how
> to handle missing/broken images at inference. (2) **External-URL domain** — registrable-domain
> exact match and domain-set Jaccard; why a rare shared domain is near-deterministic identity
> evidence independent of text. (3) **Location** — parse the embedded GPS vs. free-text branch,
> compute geodesic distance, and location entropy per identity. (4) Explicitly EXCLUDE
> bigrams/name tokens from the hard-subset model to force reliance on latent signal, and prevent
> target leakage (features must be symmetric and computed without peeking at `user_folder`).
> Output a feature/representation spec table: Signal, Construction, Why-it-links-zero-overlap-pairs,
> Missingness handling, Leakage risk.

## §4 — Graph / Collective Propagation (new component)

> [+CONTEXT] Act as a data science student adding a graph/collective entity-resolution layer so
> that identities with NO direct pairwise evidence can still be linked transitively. Reason
> step-by-step: (1) Build a candidate graph where nodes are profiles and edges are non-lexical
> affinities (shared external-domain, high image-cosine, co-location); explain how a pair A–C
> with zero direct similarity gets linked via a common neighbor B. (2) Choose a propagation
> mechanism — label propagation, connected-component clustering with a learned edge threshold,
> or a light GNN (GraphSAGE) over the multimodal node embeddings from §2 — and justify given
> only ~37k nodes. (3) Enforce the cross-platform + one-identity-per-platform constraints during
> clustering. (4) Design the ablation that isolates how many *additional* true zero-overlap
> matches the graph layer recovers beyond the pairwise model. Output a graph-ER design doc and
> the incremental-recall evaluation.

## §5 — Evaluation & Ablation on the Hard Slice (Steps 6 + 9, core)

> [+CONTEXT] Act as a data science student designing the evaluation that proves the new
> algorithm's value on the hard stratum, not just overall. Reason through: (1) Define the
> **"zero-lexical-similarity" test slice** operationally (e.g., Jaro-Winkler(name) < 0.3 AND
> bio SBERT-cosine < 0.3 among true positives) and report all metrics separately on this slice
> vs. the full set — the headline number is hard-slice Recall@k and AUPRC. (2) Mandate a
> **`user_folder`-grouped (component-aware) split** so no identity leaks across train/test, and
> state why random K-fold inflates results here. (3) Design the paired significance test
> (McNemar / bootstrap CI on AUPRC) comparing new model vs. (a) lexical-only baseline and (b)
> the current Stage-8 XGBoost. (4) Design the **ablation ladder**: lexical-only → +image →
> +URL/geo → +contrastive embedding → +alignment → +graph, reporting incremental hard-slice
> recall so each novel component's contribution is isolated. Output an evaluation-and-ablation
> protocol table with the exact slices, metrics, and tests.

---

# Supporting Pipeline Track

## Step 2 — Data Acquisition & Enrichment

### Prompt 2.1 — Enrichment Plan
> [+CONTEXT] Act as a data science student designing a data ENRICHMENT plan for LinkSocial —
> the dataset is already fixed, so the goal is maximizing usable non-lexical signal. Reason
> through: (1) Image acquisition: download from `pictureURL` (a `failed_urls.csv` of ~previous
> failures exists) — what retry, timeout, and dead-link handling policy is needed, and what
> per-platform failure rate makes the image tower unviable? (2) URL resolution: should
> `externalUrl` short-links (bit.ly, goo.gl) be expanded to registrable domains before Jaccard
> matching, and what does an unresolvable Google+ link imply? (3) Geocoding: what free geocoder
> (Nominatim) rate limits apply to ~13k free-text locations, and how do you cache results
> reproducibly? (4) What enrichment is explicitly out-of-scope (no re-scraping live accounts —
> Google+ is dead and handles have churned since collection)? Output an Enrichment Plan table:
> Signal, Method, Expected Coverage Gain, Cost, Risk.

### Prompt 2.2 — Dataset Suitability & Provenance Audit
> [+CONTEXT] Act as a data science student auditing LinkSocial's fitness for the zero-overlap
> research question. Reason through: (1) Provenance: `user_folder` ground truth comes from users
> who publicly cross-linked their accounts — argue why this creates CONFIRMATION BIAS toward
> self-consistent profiles, meaning true zero-overlap pairs are likely UNDER-represented
> relative to the real population, and how this limits external validity. (2) Quantify: what
> fraction of positive pairs actually falls in the zero-lexical slice, and is that subsample
> large enough to train and evaluate on? (3) Freshness: Google+ shut down in 2019 — does
> platform death invalidate conclusions or merely date them? (4) Score the dataset on Relevance,
> Quality, Size (of the hard slice specifically), License, Freshness. Output a Suitability
> Scorecard with a go/mitigate/no-go verdict per dimension.

### Prompt 2.3 — Blocking Strategy that Does Not Kill Hard Cases
> [+CONTEXT] Act as a data science student redesigning the candidate-generation / blocking
> strategy. My current Stage-8 hard negatives block on name-prefix — but reason through this
> trap: (1) Any blocking key built from names or text will, BY CONSTRUCTION, exclude the
> zero-overlap true pairs I care about from the candidate set — prove this and quantify the
> recall ceiling it imposes. (2) Design non-lexical blocking keys instead: shared registrable
> domain, image-embedding ANN neighbors (FAISS top-k), and geohash buckets — estimate the
> candidate-set size each yields on 36,807 profiles. (3) How do you combine multiple blocking
> channels (union) and measure blocking recall on held-out positives, reported separately for
> the zero-overlap slice? (4) What is the fallback when a profile has no image, no URL, and no
> location? Output a Blocking Design doc with per-channel recall/cost estimates.

### Prompt 2.4 — Augmentation for Contrastive Training
> [+CONTEXT] Act as a data science student evaluating data augmentation for the contrastive
> two-tower encoder. Reason through: (1) Standard SMOTE on similarity features is meaningless
> for representation learning — instead, design VIEW-GENERATION augmentations that simulate
> surface divergence: randomly drop/mask the name field, paraphrase or truncate bios, perturb
> image crops — so the encoder is FORCED to rely on latent signal rather than lexical shortcuts.
> (2) Which augmentations are label-safe (never turn person A into person B) and which risk it?
> (3) How do you validate augmentation value — ablate augmented vs. non-augmented training and
> compare hard-slice recall? (4) Should LLM-generated synthetic bios be used, and what
> distribution-shift check gates them? Conclude go/no-go per augmentation with rationale.

### Prompt 2.5 — Benchmark & Transfer Survey
> [+CONTEXT] Act as a data science student surveying external UIL benchmarks to strengthen the
> thesis. Reason through: (1) Which public cross-platform identity datasets beyond LinkSocial
> (e.g., Twitter–Foursquare alignments, entity-matching suites like Magellan/DeepMatcher data)
> could serve as a SECOND evaluation domain to show the algorithm generalizes? (2) Can any be
> used to PRE-TRAIN the contrastive encoder before fine-tuning on LinkSocial, given modality
> mismatch? (3) Compare the top 3 candidates on: multimodality (text+image), presence of a hard
> zero-overlap stratum, license, and size. (4) If none fit, justify single-dataset evaluation
> with grouped CV as academically sufficient. Output a Dataset Comparison Table and final call.

## Step 3 — EDA & Anomaly Detection

### Prompt 3.1 — EDA Plan around Real Footprints
> [+CONTEXT] Act as a data science student writing the EDA plan whose single purpose is
> characterizing the zero-overlap stratum. Before code, reason through: (1) Expected
> distributions: image-embedding cosine (bimodal for positives?), domain-match rate (rare but
> near-deterministic), geodesic distance (log-scale concentration for positives) — state priors
> so plots can confirm or refute. (2) Operationally define the hard slice (e.g.,
> Jaro-Winkler(fullName) < 0.3 AND SBERT-bio-cosine < 0.3 among positives) and measure its size
> FIRST — every later claim depends on it. (3) Which signals are jointly missing (no image + no
> URL + no location) and what fraction of the hard slice is thereby unlink-able by ANY method —
> this is the theoretical recall ceiling. (4) Prioritize anomalies that would silently corrupt
> training. Output: Univariate → Bivariate-by-label → Hard-slice census → Anomaly priority list.

### Prompt 3.2 — Leakage Audit for This Exact Schema
> [+CONTEXT] Act as a data science student auditing this exact schema for label leakage before
> training. Check each suspect and reason through the mechanism: (1) `source_folder` and
> `outputProfileName` — do they encode the same identity key as `user_folder` and thus leak the
> label outright if ever used as features? (2) `bigrams` — derived from names, so any model
> consuming it on the hard slice reintroduces the lexical shortcut we claim to exclude; verify.
> (3) Stage-8 hard negatives were mined by name similarity — does this make "high name
> similarity → negative" a spurious inverted signal the model can exploit, and how do you test
> for it? (4) Do any pairs share identical `pictureURL` strings across platforms (same CDN
> asset) — trivially leaking identity without learning anything? Output a leakage register:
> Feature, Mechanism, Severity, Action (drop/mask/keep-with-guard).

### Prompt 3.3 — Missingness & Coverage Structure
> [+CONTEXT] Act as a data science student analyzing missingness structure instead of class
> distribution (labels here are constructed, not observed). Reason through: (1) Is missingness
> MCAR or structural — e.g., is `location` coverage (35% overall) uniform across
> Twitter/Google+/Instagram, or platform-biased so the model learns platform artifacts?
> (2) Cross-tabulate signal availability per POSITIVE PAIR: what % of pairs have both images,
> both URLs, both locations — the effective training set per modality? (3) Does missingness
> correlate with the label (e.g., linked users maintain fuller profiles), creating a
> "completeness = match" shortcut the model could abuse? (4) Recommend per-modality handling:
> missing-indicator features vs. modality dropout during contrastive training. Output a Coverage
> & Missingness Report with a per-modality effective-N table.

### Prompt 3.4 — Visual EDA for the Paper
> [+CONTEXT] Act as a data science student choosing the 5 figures for the EDA section of the
> thesis. Justify each: (1) Overlaid distributions of image-cosine / domain-Jaccard /
> log-geo-distance, positives vs. negatives, with the hard slice highlighted — the "latent
> signal exists" exhibit. (2) A UMAP of profile embeddings colored by PLATFORM — if platforms
> form separate clusters, that visual IS the motivation for the alignment layer; explain this
> argumentative role. (3) Missingness matrix by platform. (4) Recall-ceiling waterfall: total
> positives → has any non-lexical signal → hard slice with signal. (5) Which figure goes in the
> main body vs. appendix and why? Output a Visual EDA Plan: chart, variables, expected insight,
> section placement.

### Prompt 3.5 — Anomalies that Poison This Model Specifically
> [+CONTEXT] Act as a data science student hunting anomalies that specifically poison identity
> linkage. Reason through: (1) DEFAULT AVATARS — thousands of profiles share the platform's
> default/placeholder image, producing image-cosine ≈ 1.0 for unrelated people; how do you
> detect them (near-duplicate clustering on embeddings, frequency threshold) and neutralize the
> signal? (2) Placeholder bios and locations ("Earth", "everywhere") — same failure in text/geo
> space. (3) Celebrity/brand accounts appearing in many users' data — hub nodes that would wreck
> graph propagation. (4) Which anomalies get removed vs. flagged with an indicator feature,
> given that removal shrinks the already-small hard slice? Output an Anomaly Decision Table:
> Type, Detection, Count, Action, Effect-on-hard-slice.

## Step 4 — Preprocessing & Imbalance

### Prompt 4.1 — Leakage-Safe Pipeline Order
> [+CONTEXT] Act as a data science student ordering the preprocessing pipeline for the two-tower
> + meta-learner system. Reason through the exact sequence: (1) Profile-level steps (clean text
> → parse registrable domain via tldextract → regex-extract GPS then geocode → download/embed
> images → SBERT-embed bios) all happen BEFORE pairing — why does this guarantee feature
> symmetry and prevent pair-order artifacts? (2) Pair-level steps (cosine, Jaccard, geodesic
> distance) must be computed identically for train/test — where exactly do fold boundaries
> constrain fitting (scalers, encoders, calibrators only on train folds grouped by
> `user_folder`)? (3) Where do the anomaly masks from EDA (default avatars, placeholder text)
> enter the sequence? (4) What single script + config reproduces the whole flow? Output a
> numbered pipeline with a leakage checkpoint after each step.

### Prompt 4.2 — Two-Level Imbalance (built 5:1 vs. real 1:10⁴)
> [+CONTEXT] Act as a data science student handling the two-level imbalance problem. Reason
> through: (1) My constructed set is 5:1 negative:positive, but deployment-time candidate
> generation yields ~1:10⁴ — why must EVALUATION be reported at (or extrapolated to) the
> realistic prior, and how does precision degrade mechanically as the prior shifts? (2) For the
> contrastive encoder, class weighting is irrelevant — in-batch negatives with large batch size
> plus mined hard negatives ARE the imbalance strategy; specify batch size math
> (negatives-per-positive as a function of batch). (3) For the XGBoost meta-learner, compare
> `scale_pos_weight` vs. focal loss vs. down-sampling easy negatives near the decision boundary.
> (4) Prove the choice helps: hard-slice recall at fixed FPR, not overall F1. Output a
> recommendation + validation plan.

### Prompt 4.3 — Reproducible Cleaning Protocol
> [+CONTEXT] Act as a data science student writing the cleaning protocol for the Methods section.
> Reason through: (1) Which normalizations are safe (unicode NFC, lowercase, whitespace) vs.
> which DESTROY signal for this task (stripping digits/underscores from usernames deletes
> identity-bearing tokens like "john_doe_1990") — justify each in a transparency table?
> (2) Define "true duplicate" (same platform + same normalized username → one record) vs.
> linkage candidate (cross-platform) — my earlier merges must not collapse the very pairs I'm
> predicting. (3) Version-control the protocol: config-driven rules, seeds, and a data hash per
> pipeline run. (4) What goes verbatim into the paper vs. supplementary? Output a Cleaning
> Protocol Document with a decision table.

### Prompt 4.4 — Modality-by-Modality Plan (no datetime; add image/URL/geo)
> [+CONTEXT] Act as a data science student specifying per-modality preprocessing — note there
> are NO datetime fields, so temporal encoding is out of scope. Cover exactly four modalities:
> (1) IMAGES: CLIP ViT embedding of `pictureURL` downloads; face-detection gate before optional
> face embedding; default-avatar filter; missing → learned null token in the tower, not zeros —
> why? (2) URL: expand shorteners, extract registrable domain, represent as exact-match +
> Jaccard + domain-frequency (rare domain = strong evidence) — why must domain frequency be
> computed on train only? (3) LOCATION: GPS-regex branch vs. geocoded free-text branch, unified
> to lat/long + confidence tier. (4) TEXT (bio): SBERT embedding — used in the full model but
> ABLATED OUT for the hard-slice model. Output the plan as a table: Modality, Steps, Output dim,
> Missing handling, Leakage note.

### Prompt 4.5 — Grouped, Verifiable Split Strategy
> [+CONTEXT] Act as a data science student constructing splits for entity-pair data. Reason
> through: (1) Why must the split be GROUPED BY `user_folder` (component-aware) — demonstrate
> with a concrete example how random pair-level splitting puts the same person in train and test
> and inflates metrics? (2) Implementation: assign folders (not pairs) to folds; then verify
> programmatically that zero folders straddle splits and that no transitive component (via graph
> edges) crosses either. (3) Ratio: with ~15k identities and a small hard slice, what split
> (e.g., 70/15/15 grouped) keeps enough hard-slice positives in test for tight confidence
> intervals — compute the expected count? (4) Congruence: KS-test feature distributions across
> splits; what divergence forces a re-draw? Output a Split Strategy doc + validation checklist
> script description.

## Step 8 — Tuning, Calibration & Optimization

### Prompt 8.1 — HPO for the XGBoost Meta-Learner
> [+CONTEXT] Act as a data science student designing HPO for the XGBoost meta-learner that
> consumes [embedding distance + URL/geo features + graph score]. Reason through: (1) Which
> hyperparameters dominate here — `scale_pos_weight`, `max_depth` (shallow, since inputs are
> already high-level similarities), `min_child_weight`, subsampling — and which are
> second-order? (2) Optuna Bayesian search with a budget of ~100 trials: define the objective as
> hard-slice AUPRC on the inner grouped-CV folds, NOT overall AUPRC — why does tuning on the
> overall metric quietly sacrifice the hard cases? (3) Early-stopping rounds on grouped
> validation. (4) Guard against validation overfitting: nested CV or a final untouched test
> fold. Output a Search Plan: space, budget, objective, guardrails.

### Prompt 8.2 — Training Config for the Contrastive Encoder
> [+CONTEXT] Act as a data science student configuring the contrastive two-tower training.
> Reason through: (1) InfoNCE temperature τ and batch size jointly control hard-negative
> pressure — explain the interaction and give a starting grid (τ ∈ {0.03, 0.07, 0.1}, batch ∈
> {128, 256, 512} subject to GPU memory with frozen backbones). (2) Frozen SBERT/CLIP +
> trainable projection heads: AdamW, LR warmup then cosine decay — why does full fine-tuning
> risk collapse on ~4k usable positive identities? (3) Hard-negative mining schedule: start with
> in-batch, phase in mined name-similar negatives — why does introducing them too early
> destabilize training? (4) Diagnose collapse: monitor embedding-space uniformity/alignment
> metrics, not just loss. Output an Optimization Config Plan.

### Prompt 8.3 — Calibration for the Human-Review Queue
> [+CONTEXT] Act as a data science student calibrating match probabilities that gate auto-merge
> vs. human review. Reason through: (1) Why XGBoost + contrastive scores are miscalibrated out
> of the box, and why the constructed 5:1 prior makes raw probabilities meaningless at the
> deployment prior — include the prior-correction formula. (2) Reliability diagrams + ECE
> computed SEPARATELY on the hard slice — a model calibrated overall but overconfident on
> zero-overlap pairs sends the wrong cases to auto-merge; why is slice-level ECE the metric that
> matters? (3) Platt vs. isotonic on grouped validation: which given ~thousands of validation
> pairs? (4) Show calibration transfers: ECE on the untouched test fold. Output a Calibration
> Audit template with slice-level reporting.

### Prompt 8.4 — Regularization Against Shortcut Learning
> [+CONTEXT] Act as a data science student designing regularization against the specific failure
> mode of this project: the model overfits EASY lexical pairs and ignores latent signal. Reason
> through: (1) Detection: training curves fine, overall validation fine, but hard-slice
> validation recall flat or falling — why is this triad the shortcut-learning signature?
> (2) Structural fixes beyond L1/L2: modality dropout (randomly hide name/bio inputs during
> training), augmentation from Prompt 2.4, and weighting the loss toward hard-slice positives.
> (3) Early stopping criterion = hard-slice validation recall, not overall loss — justify.
> (4) Over-regularization check: does easy-pair performance stay acceptable? Output a
> Regularization Experiment Plan with the monitoring metric per intervention.

### Prompt 8.5 — Stacking Specification
> [+CONTEXT] Act as a data science student specifying the final stacked system: [contrastive
> embedding distance] + [lexical/URL/geo features] + [graph propagation score] → meta-learner.
> Reason through: (1) Error decorrelation: show why the embedding model and the lexical feature
> model fail on DIFFERENT pairs (that's the whole stacking rationale here — lexical fails on
> zero-overlap, embedding fails on sparse-modality profiles). (2) Out-of-fold protocol: OOF
> predictions must respect the `user_folder` grouping — spell out the fold mechanics that keep
> the meta-learner leak-free. (3) Meta-learner choice: constrained logistic/ridge for
> interpretability vs. shallow GBM — decide against the HITL explainability requirement.
> (4) Complexity gate: what hard-slice recall delta justifies shipping the stack over the single
> best model? Output an Ensemble Design Specification.

## Step 10 — Interpretability & Error Analysis

### Prompt 10.1 — XAI Strategy
> [+CONTEXT] Act as a data science student planning explainability for a system whose key
> feature is an OPAQUE learned embedding distance. Reason through: (1) TreeSHAP on the
> meta-learner explains feature-level contributions cheaply — but "embedding_distance
> contributed +0.4" is not an explanation a reviewer can act on; how do you decompose it further
> (nearest-neighbor exemplars, per-modality distance breakdown)? (2) For human reviewers merging
> identities, what's the minimal explanation unit: matched domain, image side-by-side with
> cosine, map of the two locations? (3) Global: SHAP summary proving the model doesn't secretly
> rely on lexical features in the hard slice. (4) Stability: do explanations agree across seeds?
> Output an Explainability Plan split into "for the paper" vs. "for the review UI".

### Prompt 10.2 — Error Analysis on the Hard Slice
> [+CONTEXT] Act as a data science student running structured error analysis focused on the hard
> slice. Reason through: (1) Cluster false negatives by available-signal profile: {no image},
> {default avatar}, {no URL}, {geocode failure}, {all signals present but model still missed} —
> the last cluster is the scientifically interesting one; how large is it? (2) Cluster false
> positives: shared rare domain but genuinely different people (family members? colleagues?),
> near-duplicate stock avatars, dense city co-location. (3) For each cluster: fixable by data
> work, fixable by architecture, or a fundamental ceiling? (4) Convert findings into ranked
> engineering actions for the next iteration. Output an Error Analysis Report: Cluster, Count,
> Root cause, Fix path, Priority.

### Prompt 10.3 — SHAP Plan
> [+CONTEXT] Act as a data science student applying SHAP to the meta-learner. Reason through:
> (1) TreeSHAP is exact and fast for GBM — confirm applicability and why KernelSHAP is
> unnecessary here. (2) Required global artifacts for the thesis: beeswarm overall AND beeswarm
> restricted to hard-slice pairs — the pair of plots that proves the model switches evidence
> source on hard cases (embedding/URL/geo rise, lexical drops to zero). (3) Dependence plots:
> embedding-distance SHAP vs. image-availability interaction — does the model correctly discount
> embedding distance when the image is a default avatar? (4) Local waterfalls for 3 archetypes:
> easy match, zero-overlap true match, near-miss false positive. Output a SHAP Analysis Plan
> keyed to specific thesis figures.

### Prompt 10.4 — Minimal-Evidence Analysis (adapted counterfactual)
> [+CONTEXT] Act as a data science student adapting counterfactual analysis to identity linkage
> (recourse framing doesn't apply — we can't ask users to change their profiles). Reason
> through: (1) Reframe as MINIMAL EVIDENCE analysis: for each predicted match, which single
> signal, if removed, flips the decision — this identifies the load-bearing evidence per match.
> (2) Use it to grade decision robustness: matches resting on ONE signal (a single shared
> domain) route to human review; matches with redundant evidence auto-merge — formalize the
> rule. (3) Implementation: per-feature ablation at inference vs. SHAP-based approximation —
> accuracy/cost tradeoff. (4) Audit that the rule reduces false merges on the test fold. Output
> a Minimal-Evidence Design Spec with the routing rule.

### Prompt 10.5 — Feature-Importance Stability
> [+CONTEXT] Act as a data science student testing whether feature-importance conclusions
> survive experimental variation. Reason through: (1) Retrain across 5 seeds × grouped folds:
> how much do SHAP rankings of the top-5 features move (rank correlation), and is "embedding
> distance dominates the hard slice" stable — that claim is a thesis headline, so it needs error
> bars? (2) Which features show high variance, and does it trace to collinearity (image-cosine
> vs. embedding-distance share signal)? (3) Where do native gain, permutation importance, and
> SHAP disagree, and why (correlated features split credit differently)? (4) Set a variance
> threshold above which a claimed finding is downgraded from "result" to "observation". Output a
> Stability Report template.

## Step 11 — Ethics & Qualitative Analysis

### Prompt 11.1 — Bias Audit on the Real Data Structure
> [+CONTEXT] Act as a data science student auditing performance equity across data-driven
> subgroups (no demographic labels exist, so audit what's observable). Reason through:
> (1) Subgroups: platform pair (Twitter–Instagram vs. Google+–X), profile completeness tier,
> non-Latin-script names (Thai, Arabic, CJK — do SBERT/Jaro-Winkler degrade?), and geocodable
> vs. non-Western locations. (2) Measure per-subgroup hard-slice recall and false-merge rate —
> which disparity is most harmful, given a false merge pollutes someone's unified record?
> (3) Root causes: geocoder coverage bias, CLIP's known demographic performance gaps on faces.
> (4) Uniform threshold vs. per-group operating points — take a position and defend it. Output a
> Subgroup Evaluation Matrix.

### Prompt 11.2 — PDPA/GDPR (the overlooked point: biometrics)
> [+CONTEXT] Act as a data science student assessing PDPA (Thailand) and GDPR compliance for
> this research. Reason through: (1) Inventory: usernames, real names, bios, locations (some raw
> GPS!), and profile photos — and critically, FACE EMBEDDINGS derived from photos are biometric
> data, a special category under PDPA Section 26 and GDPR Article 9; what does that classify my
> image pipeline as, and what lawful basis covers academic research use of a public dataset?
> (2) Purpose limitation: LinkSocial users cross-linked accounts publicly, but did they consent
> to linkage RESEARCH — argue the academic-exemption position honestly, with its limits.
> (3) Minimization: hash usernames in released artifacts, never redistribute images, publish
> embeddings only if non-invertible — verify that claim. (4) Right to erasure: how would you
> remove one identity from trained models? Output a Privacy Compliance Checklist with a risk
> rating per item.

### Prompt 11.3 — Ethics Statement (dual-use is the heart of UIL)
> [+CONTEXT] Act as a data science student writing the formal Ethics Statement for a thesis
> whose core contribution — linking accounts that share NO surface similarity — is precisely a
> de-anonymization capability. Reason through candidly: (1) The better my algorithm works, the
> more it threatens pseudonymous users (activists, whistleblowers, LGBTQ+ users in hostile
> jurisdictions) who deliberately maintain unlinkable profiles — name this tension directly,
> don't soften it. (2) Vulnerable populations harmed by re-identification. (3) Dual-use: a
> commercial CRM tool that is functionally a surveillance/stalking instrument — where's the
> line? (4) Concrete safeguards: no model/image release, aggregate-only results, code license
> forbidding surveillance use, IRB/ethics-board note. Output a 300–400 word Academic Ethics
> Statement that a critical reviewer would accept as genuine, not boilerplate.

### Prompt 11.4 — Representational Harm from Frozen Models
> [+CONTEXT] Act as a data science student assessing representational harms inherited from the
> frozen SBERT and CLIP backbones. Reason through: (1) Documented biases: CLIP's uneven
> face-embedding quality across skin tone and gender, SBERT's Western/English-centric semantics
> — how do these propagate into differential linkage accuracy in MY pipeline specifically?
> (2) Are non-Western, non-English, or non-normative profiles systematically harder to link
> (lower recall) or more falsely merged — which is the worse harm and why? (3) Build a Model
> Card and a Datasheet (Gebru et al.) documenting these limits for anyone reusing the work.
> (4) Since there's no generated-text output here, reframe safeguard (4) as: what disclaimers
> must accompany any per-subgroup metric? Output a Representational Harm Assessment.

### Prompt 11.5 — Human-in-the-Loop Queue Architecture
> [+CONTEXT] Act as a data science student designing the human-in-the-loop review architecture
> for auto-merge decisions. Reason through: (1) Use the CALIBRATED probability (Prompt 8.3) plus
> the minimal-evidence robustness score (Prompt 10.4) to route: high-confidence +
> redundant-evidence → auto-merge; borderline or single-signal → human queue — give the
> threshold logic. (2) Minimal reviewer telemetry: side-by-side images, matched domain
> highlighted, map of both locations, per-modality distance bars — what expedites a correct
> decision fastest? (3) Log overrides as new labeled pairs, tagged as human-verified, fed back
> as high-quality training data with grouping preserved — how to prevent this from re-leaking
> into test folds? (4) Automation-bias mitigation: force review of a random audit sample even
> when confidence is high. Output a HITL Workflow Specification.

## Step 12 — MLOps & Continuous Learning

### Prompt 12.1 — MLOps Architecture (scoped to a thesis)
> [+CONTEXT] Act as a data science student architecting a REALISTIC MLOps setup for a research
> system that must be reproducible for peer review (not hyperscale production). Reason through:
> (1) Two inference profiles: batch embedding + FAISS index build (offline, GPU), vs.
> lightweight pair-scoring (online, CPU) — what containerization actually fits a thesis + a
> demo, and why is Kubernetes likely overkill here? (2) Reproducibility is the real requirement:
> DVC for the LinkSocial artifacts and image cache, MLflow for runs, pinned seeds and env —
> enough that a reviewer reruns and gets identical hard-slice AUPRC. (3) CI checks: schema
> validation, split-leakage test (no `user_folder` across folds), embedding-dim checks.
> (4) Rollback: since a bad model causes false MERGES that corrupt records, why must merges be
> reversible (staged, not destructive)? Output an MLOps spec honestly scoped to the project.

### Prompt 12.2 — Drift Monitoring (ground truth arrives late)
> [+CONTEXT] Act as a data science student designing drift monitoring, acknowledging that in
> deployment new true-match labels arrive slowly (only when humans review). Reason through:
> (1) Input drift on what I CAN observe immediately: PSI/KS on embedding-distance distributions,
> domain-match rate, image-availability rate across incoming batches. (2) Proxy metrics for
> performance when labels lag: auto-merge rate, human-queue size, override rate — how these
> signal silent degradation before ground truth confirms it. (3) Thresholds separating a
> watch-level alert from a retrain trigger. (4) Alert-fatigue control: aggregate windows,
> require persistence before firing. Output a Monitoring Dashboard Specification with
> observable-vs-lagging metric separation.

### Prompt 12.3 — Retraining & Governance
> [+CONTEXT] Act as a data science student writing the retraining policy. Reason through:
> (1) Trigger choice — calendar vs. accumulated human-verified labels (from the HITL queue) vs.
> drift alarm: argue that label-count triggers fit best because new hard-verified pairs are
> exactly the signal that improves the zero-overlap slice. (2) Window: retrain on the full
> accumulated ledger (GBM has no catastrophic forgetting; contrastive encoder benefits from all
> identities) vs. sliding window — decide per component. (3) Gating: shadow-score the new model
> on live candidates, compare hard-slice metrics on a frozen labeled set before promotion —
> canary logic. (4) Registry isolation for governance/audit. Output a Retraining Policy Document.

### Prompt 12.4 — Model Card (Mitchell et al. 2019)
> [+CONTEXT] Act as a data science student producing the Model Card per Mitchell et al. (2019)
> for this identity-linkage system. Reason through each section against MY reality: (1) Intended
> use: academic cross-platform linkage research and consented CRM unification; OUT OF SCOPE and
> dangerous: surveillance, de-anonymizing pseudonymous users, law-enforcement targeting — state
> these as hard prohibitions. (2) Performance disaggregated by platform pair, completeness tier,
> and script/language, with the hard-slice numbers front and center and honest error bars.
> (3) Which choices most shape behavior: the frozen backbones, the grouped split, the
> calibration prior. (4) Ethical considerations and limitations visible to any adopter. Output a
> finalized Model Card.

### Prompt 12.5 — Active Learning Loop
> [+CONTEXT] Act as a data science student designing the active-learning loop that feeds the
> HITL queue back into training. Reason through: (1) Query strategy: uncertainty/margin sampling
> near the calibrated threshold WILL over-sample easy ambiguous cases — instead prioritize
> predicted matches in the zero-overlap slice (high embedding-similarity, zero lexical overlap),
> since those are the informative frontier for MY research goal; justify. (2) Reviewer-fatigue
> design: batch size, pre-filled evidence, keyboard-fast accept/reject. (3) Feedback-loop bias:
> the model's own scores decide what humans see, entrenching its blind spots — inject a
> random-sample quota and periodically label profiles the model is confidently NEGATIVE on;
> explain why. (4) QA gate before labels enter training: dual-review or confidence rule, plus
> the grouping check so fed-back pairs never leak across the test boundary. Output an Active
> Learning Pipeline Description.

---

## Coverage map vs. the original 12-step framework

| Original Step | Covered by | Notes |
|---|---|---|
| 1 — Research Question | §1 | Reframed around the zero-overlap hard slice |
| 2 — Data Acquisition | Prompts 2.1–2.5 | Reframed as enrichment + non-lexical blocking |
| 3 — EDA & Anomaly | Prompts 3.1–3.5 | Hard-slice census, leakage audit, default-avatar anomalies |
| 4 — Preprocessing & Imbalance | Prompts 4.1–4.5 | Multimodal, no datetime; two-level imbalance |
| 5 — Feature Engineering & Representation | §3 | Non-lexical footprints only |
| 6 — Hypothesis & Experimental Design | §5 | Ablation ladder + grouped CV + significance tests |
| 7 — Model Selection & Architecture | §2 + §4 | Two-tower contrastive + alignment; graph propagation |
| 8 — Tuning & Calibration | Prompts 8.1–8.5 | HPO, contrastive config, slice-level calibration |
| 9 — Evaluation Metrics | §5 | Hard-slice AUPRC / Recall@k at realistic prior |
| 10 — Interpretability & Error Analysis | Prompts 10.1–10.5 | XAI, SHAP, minimal-evidence, stability |
| 11 — Ethics | Prompts 11.1–11.5 | Biometrics/PDPA, dual-use de-anonymization |
| 12 — MLOps & Continuous Learning | Prompts 12.1–12.5 | Reproducibility-first, reversible merges |

## Traps these prompts guard against (that generic prompts miss)
- **§5 / Prompt 2.3** — name/text blocking silently excludes the zero-overlap pairs, capping recall before modeling even starts.
- **Prompt 3.2 / 3.5** — `bigrams` and `source_folder` leak the label; default avatars fake image-cosine ≈ 1.0.
- **Prompt 8.4** — shortcut learning: good overall metrics while hard-slice recall stalls.
- **Prompt 11.2 / 11.3** — face embeddings are biometric under PDPA, and the core contribution is a de-anonymization capability; ethics must be genuine, not boilerplate.
