# Held-out generalization — svan_v3_production

Spec `581bd3fa4c12` (30 triples, 10 category/element pairs), best_of=4, seed=20261105, palette=pico8. Palette conditioning trained: False.

| regime/arm | IoU | palette % | colour % | pixels | latency ms | sources |
| --- | --- | --- | --- | --- | --- | --- |
| heldout/neural_only | 0.7682 | 100.0 | 19.8 | 247.8 | - | {"neural": 80} |
| heldout/production | 0.7789 | 100.0 | 48.63 | 232.4 | 373.1 | {"neural": 61, "procedural_guard": 19} |
| in_distribution/neural_only | 0.7599 | 100.0 | 21.0 | 232.7 | - | {"neural": 80} |
| in_distribution/production | 0.7512 | 100.0 | 46.93 | 232.0 | 364.9 | {"neural": 68, "procedural_guard": 12} |
