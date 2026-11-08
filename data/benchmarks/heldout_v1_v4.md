# Held-out generalization — svan_v4_palette

Spec `581bd3fa4c12` (30 triples, 10 category/element pairs), best_of=4, seed=20261105, palette=pico8. Palette conditioning trained: True.

| regime/arm | IoU | palette % | colour % | pixels | latency ms | sources |
| --- | --- | --- | --- | --- | --- | --- |
| heldout/neural_only | 0.784 | 100.0 | 17.14 | 240.2 | - | {"neural": 80} |
| heldout/production | 0.7829 | 100.0 | 49.29 | 237.0 | 365.5 | {"neural": 53, "procedural_guard": 27} |
| in_distribution/neural_only | 0.7623 | 100.0 | 20.0 | 233.1 | - | {"neural": 80} |
| in_distribution/production | 0.7622 | 100.0 | 47.1 | 233.9 | 353.7 | {"neural": 64, "procedural_guard": 16} |
