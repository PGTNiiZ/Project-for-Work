# EVAL

retrieval:
- all cross-platform pairs = 449,149,239
- exact recall_global = 0.4220
- exact + candidate coverage = 88.67%

classification:
- best model = gb
- test precision = 0.9507
- test recall = 0.9179
- test f1 = 0.9340
- confusion matrix = [[3717, 209], [360, 4027]]

strict report reference:
- model = logreg + sigmoid
- threshold = 0.50
- test precision = 0.9900
- test recall = 0.8169
- test f1 = 0.8951
- confusion matrix = [[461691, 10], [221, 986]]

production:
- final match-only precision = 0.9550
- final match-only recall = 0.6711
- review queue = 86,296
