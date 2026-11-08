# Held-out generalization — svan_v4_palette_ep2

Spec `581bd3fa4c12` (30 triples, 10 category/element pairs), best_of=4, seed=20261105.

| regime/arm | IoU | palette % | colour % | pixels | latency ms | sources |
| --- | --- | --- | --- | --- | --- | --- |
| heldout/neural_only | 0.7904 | 100.0 | 17.43 | 237.7 | - | {"neural": 80} |
| heldout/production | 0.7901 | 100.0 | 49.08 | 234.7 | 365.1 | {"neural": 60, "procedural_guard": 20} |
| in_distribution/neural_only | 0.7767 | 100.0 | 15.36 | 234.8 | - | {"neural": 80} |
| in_distribution/production | 0.7698 | 100.0 | 47.29 | 231.9 | 359.2 | {"neural": 70, "procedural_guard": 10} |
