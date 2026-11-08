# Held-out generalization — v4_steer3

Spec `581bd3fa4c12` (30 triples, 10 category/element pairs), best_of=4, seed=20261105.

| regime/arm | IoU | palette % | colour % | pixels | latency ms | sources |
| --- | --- | --- | --- | --- | --- | --- |
| heldout/neural_only | 0.7856 | 100.0 | 52.36 | 238.4 | - | {"neural": 80} |
| heldout/production | 0.7901 | 100.0 | 49.08 | 234.7 | 970.0 | {"neural": 60, "procedural_guard": 20} |
| in_distribution/neural_only | 0.7696 | 100.0 | 55.63 | 241.5 | - | {"neural": 80} |
| in_distribution/production | 0.7698 | 100.0 | 47.29 | 231.9 | 2031.4 | {"neural": 70, "procedural_guard": 10} |
