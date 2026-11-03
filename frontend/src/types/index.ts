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
