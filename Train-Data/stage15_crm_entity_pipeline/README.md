# Stage 15 CRM Entity Pipeline

This pipeline converts full-candidate identity outputs into CRM-ready tables:

- `match_decisions`
- `review_queue`
- `unified_profiles`
- `profile_mapping`
- `lead_scores`

It reads:

- `stage7_14_full_candidate_pipeline`
- `stage7_13_multimodal_suite/runs/image_context_r075_h20_s42`
- `normalized_profiles_with_profile_id.csv`

The design keeps explicit audit fields such as:

- `decision_source`
- `review_status`
- `mapping_source`
- `merge_source`

Review items include:

- pair snapshots
- top 5 key features
- priority score

Unified profiles are merged from `MATCH_EXACT` and `AUTO_HIGH` edges only.
