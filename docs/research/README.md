# SpriteMind Research & Experimental Architecture Index

**Platform:** SpriteMind — Local AI Pixel Art Generator & Research Platform  
**System Architecture:** SVAN (SpriteMind Visual-Action Network)  
**Primary Execution Profile:** CPU-First Deterministic & Neural Inference with PyTorch 2.5.1 on Intel Core i7-12650HX  
**Repository State:** Stage 5.5 (Data Management, Database Control & Git Preparation)  

---

## Executive Overview

SpriteMind is an original, locally trainable machine learning system designed to understand, construct, inspect, animate, and compose game-ready 2D pixel-art assets. Unlike black-box diffusion models or pretrained cloud APIs, SpriteMind models pixel-art creation as structured drawing actions over a deterministic virtual canvas (PixelScript), evaluates its own renders through a topological and neural critic, and articulates characters through recurrent temporal kinematics and socket alignment networks.

**Consolidated reference:** [`SYSTEM_RESEARCH_OVERVIEW.md`](SYSTEM_RESEARCH_OVERVIEW.md) — one
document covering the whole system as it stands today (architecture, stack, models, data,
evaluation, results, limitations). The stage reports below are the historical record of how each
subsystem was arrived at.

This directory contains the complete technical reports, empirical findings, architectural diagrams, and benchmark evaluations across all stages of development.

---

## Chronological Research Reports

| Stage | Document Title | Primary Milestone | Key Checkpoints | Primary Metrics |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 1** | *Initial Architecture & Canvas Engine* | Deterministic virtual canvas, AST parser, tokenizer, baseline procedural generator | Procedural Baseline (0 params) | 100% valid action execution; 0 external image dependencies |
| **Stage 2** | [`STAGE_2_LEARNING_VALIDATION.md`](STAGE_2_LEARNING_VALIDATION.md) | Supervised action-sequence learning on 10,000-sample balanced dataset | `svan_checkpoint_best.pt` (1.74M params) | 95.25% validation token accuracy; Cross-entropy loss: 0.1239 |
| **Stage 3** | [`STAGE_3_VISUAL_INTELLIGENCE.md`](STAGE_3_VISUAL_INTELLIGENCE.md) | Topological morphological analysis & neural defect critic with closed-loop repair | `visual_critic_best.pt` (368K params) | Defect detection Precision: 100.0%, Recall: 100.0%, F1: 1.000; 80% repair success |
| **Stage 4** | [`STAGE_4_TEMPORAL_INTELLIGENCE.md`](STAGE_4_TEMPORAL_INTELLIGENCE.md) | Recurrent temporal intelligence network for multi-frame character animation across 18 categories | `temporal_svan_best.pt` (426K params) | Validation trajectory MSE: 2.6691; Smoothness: 98.19; 3x lower flicker than procedural baseline |
| **Stage 5** | [`STAGE_5_GAME_READY_ASSET_GENERATION.md`](STAGE_5_GAME_READY_ASSET_GENERATION.md) | Composable game-ready asset catalog, neural socket alignment, 2D texture atlas packing, and Dead Zone integration | `socket_predictor_best.pt` (619K params) | Socket alignment error: 0.55px; 100% attachment accuracy; Reversible atlas extraction |
| **Stage 5.5** | [`STAGE_5_5_DATA_MANAGEMENT_AND_GIT_READINESS.md`](STAGE_5_5_DATA_MANAGEMENT_AND_GIT_READINESS.md) | Auditable SQLite database registry, strict data classification, trash lifecycle, backup/restore, and Git preparation | SQLite `spritemind.db` | 100% protection guarantee; Zero model/dataset loss on asset deletion; 50 passing backend tests |

---

## Detailed Benchmark Register

For an exhaustive, chronological register of every empirical benchmark executed, datasets used, evaluation commands, hardware profiles, raw numerical scores, and limitations, refer to:

👉 **[`docs/research/BENCHMARK_INDEX.md`](BENCHMARK_INDEX.md)**

---

## Model Inventory & Checkpoint Registry

All checkpoints are stored in `models/checkpoints/` and strictly classified as `PROTECTED_MODEL`. They can never be deleted through the asset deletion interface.

| Checkpoint File | Architecture | Parameters | Purpose | Training Provenance |
| :--- | :--- | :--- | :--- | :--- |
| `svan_checkpoint_best.pt` | Action Transformer (Encoder-Decoder) | 1,740,000 | Autoregressive PixelScript action generation | 10,000 synthetic characters (`generalization_10k`) |
| `visual_critic_best.pt` | Residual2D + Heatmap Decoder | 368,044 | Spatial defect localization and severity scoring | Clean & corrupted synthetic samples |
| `temporal_svan_best.pt` | Visual Res2D + GRU Recurrent Core | 426,395 | Kinematic part trajectory prediction ($\Delta x, \Delta y, \Delta \text{rot}$) | 600 multi-frame sequences across 150 unique characters |
| `socket_predictor_best.pt` | Residual2D + Coordinate Regressor | 619,432 | Sub-pixel attachment socket prediction (`hand_r`, `head`, `back`) | 300 character silhouettes (identity split 80/20) |

---

## Data Management & Architecture Documentation

- [`docs/DATA_MANAGEMENT.md`](../DATA_MANAGEMENT.md): Operational guide for data classification, trash lifecycle, and backup management.
- [`docs/STORAGE_ARCHITECTURE.md`](../STORAGE_ARCHITECTURE.md): Technical breakdown of SQLite schemas, file layouts, and immutability gates.
- [`docs/GIT_RELEASE_CHECKLIST.md`](../GIT_RELEASE_CHECKLIST.md): Git repository preparation, ignore rules, and Git LFS strategy.
