# Held-out generalization — v4_conditioned

Spec `581bd3fa4c12` (30 triples, 10 category/element pairs), best_of=4, seed=20261105.

| regime/arm | IoU | palette % | colour % | pixels | latency ms | sources |
| --- | --- | --- | --- | --- | --- | --- |
| heldout/neural_only | 0.7814 | 100.0 | 47.46 | 237.9 | - | {"neural": 80} |
| heldout/production | 0.7901 | 100.0 | 49.08 | 234.7 | 649.1 | {"neural": 60, "procedural_guard": 20} |
| in_distribution/neural_only | 0.7587 | 100.0 | 46.45 | 240.3 | - | {"neural": 80} |
| in_distribution/production | 0.7698 | 100.0 | 47.29 | 231.9 | 644.4 | {"neural": 70, "procedural_guard": 10} |
