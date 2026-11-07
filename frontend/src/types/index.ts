export interface LayerInfo {
  id: number;
  name: string;
  opacity: number;
  visible: boolean;
  locked: boolean;
  pixel_count: number;
}

export interface BoundingBox {
  min_x: number;
  min_y: number;
  max_x: number;
  max_y: number;
  width: number;
  height: number;
}

export interface BodyPartInfo {
  name: string;
  parent: string | null;
  anchor_x: number;
  anchor_y: number;
  pixel_count: number;
  bounding_box: BoundingBox | null;
}

export interface ValidationReport {
  is_valid: boolean;
  errors: string[];
  warnings: string[];
  total_actions: number;
  invalid_action_count: number;
  invalid_action_rate: number;
}

export interface DefectItem {
  id?: string;
  type: string;
  severity: 'minor' | 'moderate' | 'major' | 'critical';
  description: string;
  confidence?: number;
  bbox?: {
    min_x: number;
    min_y: number;
    max_x: number;
    max_y: number;
    width?: number;
    height?: number;
  };
  affected_pixels?: number[][];
  suggested_fix?: string;
}

export interface RepairPlan {
  defect_id: string;
  defect_type: string;
  severity: string;
  description: string;
  pixelscript_patch: string;
  estimated_quality_gain: number;
}

export interface RepairHistoryItem {
  iteration: number;
  script_text: string;
  image_base64: string;
  quality_score: number;
  defects_count: number;
  defects: DefectItem[];
  action_applied: string;
}

export interface RepairResponse {
  initial_quality: number;
  final_quality: number;
  quality_gain: number;
  initial_defects: number;
  final_defects: number;
  resolved_defects: number;
  iterations_completed: number;
  applied_repairs: any[];
  repaired_script: string;
  image_base64: string;
  history: RepairHistoryItem[];
  final_critic: any;
  elapsed_ms: number;
}

export interface VisualCriticReport {
  quality_score: number;
  pixel_fill_count: number;
  detected_defects: DefectItem[];
  has_defects: boolean;
  defects_count?: number;
  repair_plans?: RepairPlan[];
  neural_critic?: any;
  validation_errors?: string[];
  validation_warnings?: string[];
}

export interface RenderPassesData {
  silhouette?: string;
  edge_map?: string;
  segmentation?: string;
}

export interface FrameData {
  frame_index: number;
  width: number;
  height: number;
  image_base64: string;
  script_text: string;
  layers: LayerInfo[];
  parts: Record<string, BodyPartInfo>;
  palette: number[][];
  passes?: RenderPassesData;
  validation: ValidationReport;
  execution_time_ms: number;
  pixel_count: number;
}

export interface GenerationResult {
  engine_used: 'trained_model' | 'untrained_model' | 'procedural_baseline';
  category: string;
  style: string;
  pose: string;
  palette: string;
  dimensions: { width: number; height: number };
  num_frames: number;
  frames: FrameData[];
  critic: VisualCriticReport;
  inference_time_ms: number;
  seed: number;
}

export interface SystemHealth {
  status: string;
  version: string;
  torch_version: string;
  cuda_available: boolean;
  device: string;
  device_memory: Record<string, number>;
  training_active: boolean;
}

export interface ModelsInfo {
  available_checkpoints: string[];
  loaded_model: {
    active_checkpoint: string | null;
    has_trained_model: boolean;
    parameter_count: number;
    device: string;
    vocab_size: number;
  };
  supported_engines: string[];
}

export interface TrainingStatus {
  state: 'idle' | 'training' | 'completed' | 'stopped' | 'error';
  dataset_name: string;
  epoch: number;
  total_epochs: number;
  step: number;
  total_steps: number;
  loss: number;
  val_loss: number;
  token_accuracy: number;
  elapsed_seconds: number;
  learning_rate: number;
  loss_history: { epoch: number; loss: number }[];
  val_loss_history: { epoch: number; val_loss: number }[];
  token_accuracy_history?: { epoch: number; token_accuracy: number }[];
  device?: string;
  error_message?: string;
}

export interface EvaluationResults {
  comparisons: Record<string, {
    engine: string;
    samples_evaluated: number;
    silhouette_iou: number;
    pixel_accuracy: number;
    palette_accuracy: number;
    invalid_action_rate: number;
    defect_f1: number;
    temporal_consistency: number;
    loop_error: number;
  }>;
  metrics: string[];
}

export interface AssetRecord {
  id: string;
  name: string;
  category: string;
  tags: string[];
  created_at: string;
  frames: FrameData[];
  metadata?: Record<string, any>;
}

export interface EngineLatency {
  samples: number;
  p50_ms: number | null;
  p95_ms: number | null;
}

export interface RuntimeMetrics {
  latency: {
    window: number;
    samples: number;
    cache_hits: number;
    p50_ms: number | null;
    p95_ms: number | null;
    mean_ms: number | null;
    engines: Record<string, EngineLatency>;
  };
  cache: { hits: number; misses: number; size: number };
}

export interface ReportCardRow {
  category: string;
  samples: number;
  silhouette_iou: number;
  palette_accuracy: number;
  valid_rate: number;
  mean_score: number;
}

export interface BenchmarkReportCard {
  engine: string;
  categories: ReportCardRow[];
  mean_score: number;
  samples: number;
}

export interface HistoryRun {
  id: number;
  created_at: string;
  prompt: string;
  category: string;
  style: string;
  pose: string | null;
  palette: string;
  engine: string;
  seed: number | null;
  num_frames: number | null;
  best_of: number | null;
  temperature: number | null;
  top_k: number | null;
  latency_ms: number | null;
}

export interface HistoryResponse {
  runs: HistoryRun[];
  total: number;
}

export interface BenchmarkGalleryEntry {
  prompt: string;
  category: string;
  engine?: string;
  seed?: number;
  silhouette_iou?: number;
  palette_accuracy?: number;
  valid?: boolean;
  score?: number;
  image?: string;
}

export interface BenchmarkGallery {
  summary: Record<string, number>;
  gallery: BenchmarkGalleryEntry[];
  total: number;
}

export interface ExportResult {
  format: string;
  /** Data URL for png, spritesheet, gif and onion exports. */
  image_base64?: string;
  metadata?: { width: number; height: number; cols: number; rows: number; frames: number };
  fps?: number;
  frame_index?: number;
  total_frames?: number;
  frames_scripts?: string[];
  width?: number;
  height?: number;
}

export interface ServiceStatus {
  status: string;
  id?: string | number;
  deleted?: boolean;
  trash_id?: string;
}

export interface GenerateParams {
  prompt?: string;
  negative_prompt?: string;
  category?: string;
  style?: string;
  pose?: string;
  palette?: string;
  width?: number;
  height?: number;
  num_frames?: number;
  engine?: string;
  seed?: number;
  max_actions?: number;
  best_of?: number;
  temperature?: number;
  top_k?: number;
}

export interface RefineParams {
  script_text: string;
  suggested_fix?: string;
  width?: number;
  height?: number;
  palette?: string;
}

export interface RenderParams {
  script_text: string;
  width?: number;
  height?: number;
  palette?: string;
  scale?: number;
}

export interface ExportParams {
  frames_scripts: string[];
  width: number;
  height: number;
  palette: string;
  format: 'png' | 'spritesheet' | 'gif' | 'json' | 'onion';
  scale?: number;
  fps?: number;
  cols?: number;
  frame_index?: number;
}

export interface AssetSaveParams {
  name: string;
  category: string;
  tags?: string[];
  frames: FrameData[];
  metadata?: Record<string, unknown>;
}

export interface ActionSpec {
  key: string;
  label: string;
  anim: string;
  frames: number;
  fps: number;
  loop: boolean;
  effects: string[];
}

export interface ActionPlan {
  category: string;
  style: string;
  palette: string;
  archetype: string;
  element: string;
  weapon: string;
  actions: ActionSpec[];
  rationale: string[];
}

export interface ActionPlanResponse {
  status: string;
  plan: ActionPlan;
}

export interface BuiltAction extends Omit<ActionSpec, 'frames'> {
  engine_used: string;
  num_frames: number;
  frames: { image_base64: string; script_text: string; pixel_count: number }[];
  gif_base64?: string;
}

export interface ActionSetResponse {
  plan: ActionPlan;
  actions: BuiltAction[];
}

export interface AnimationMetrics {
  status: string;
  model_architecture: string;
  parameter_count: number;
  supported_animation_categories: string[];
  supported_character_categories: string[];
  training_metrics?: Record<string, unknown>;
  benchmark_results?: Record<string, unknown>;
}

export interface CriticMetrics {
  status: string;
  model_architecture: string;
  parameter_count: number;
  classes: string[];
  evaluation_summary?: Record<string, unknown>;
}

export interface DatasetStats {
  total_samples: number;
  [key: string]: unknown;
}

export interface TrainingHistory {
  loss_history: { epoch: number; loss: number }[];
  val_loss_history: { epoch: number; val_loss: number }[];
  token_accuracy_history?: { epoch: number; token_accuracy: number }[];
  best_val_loss?: number;
  [key: string]: unknown;
}
