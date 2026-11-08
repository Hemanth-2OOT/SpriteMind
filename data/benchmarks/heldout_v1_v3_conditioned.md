# Held-out generalization — v3_conditioned

Spec `581bd3fa4c12` (30 triples, 10 category/element pairs), best_of=4, seed=20261105.

| regime/arm | IoU | palette % | colour % | pixels | latency ms | sources |
| --- | --- | --- | --- | --- | --- | --- |
| heldout/neural_only | 0.7707 | 100.0 | 48.52 | 238.2 | - | {"neural": 80} |
| heldout/production | 0.7789 | 100.0 | 48.63 | 232.4 | 668.1 | {"neural": 61, "procedural_guard": 19} |
| in_distribution/neural_only | 0.7514 | 100.0 | 25.82 | 234.7 | - | {"neural": 80} |
| in_distribution/production | 0.7512 | 100.0 | 46.93 | 232.0 | 680.5 | {"neural": 68, "procedural_guard": 12} |
