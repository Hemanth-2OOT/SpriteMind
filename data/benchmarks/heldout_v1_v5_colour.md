# Held-out generalization — v5_colour

Spec `581bd3fa4c12` (30 triples, 10 category/element pairs), best_of=4, seed=20261105.

| regime/arm | IoU | palette % | colour % | pixels | latency ms | sources |
| --- | --- | --- | --- | --- | --- | --- |
| heldout/neural_only | 0.7782 | 100.0 | 47.48 | 239.5 | - | {"neural": 80} |
| heldout/production | 0.7798 | 100.0 | 48.77 | 234.1 | 394.1 | {"neural": 64, "procedural_guard": 16} |
| in_distribution/neural_only | 0.7453 | 100.0 | 46.07 | 250.7 | - | {"neural": 80} |
| in_distribution/production | 0.7595 | 100.0 | 47.73 | 233.9 | 365.9 | {"neural": 68, "procedural_guard": 12} |
