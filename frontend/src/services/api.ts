import {
  SystemHealth,
  ModelsInfo,
  GenerationResult,
  TrainingStatus,
  EvaluationResults,
  AssetRecord,
  FrameData,
  VisualCriticReport,
} from '../types';

const BASE_URL = '';
const API_KEY: string =
  (import.meta.env?.VITE_SPRITEMIND_API_KEY as string | undefined) ||
  (typeof localStorage !== 'undefined' ? localStorage.getItem('spritemind_api_key') || '' : '');

async function apiFetch(input: string, init: RequestInit = {}): Promise<globalThis.Response> {
  if (!API_KEY) return fetch(input, init);
  const headers = new Headers(init.headers || {});
  if (!headers.has('X-API-Key')) headers.set('X-API-Key', API_KEY);
  return fetch(input, { ...init, headers });
}

export async function fetchHealth(): Promise<SystemHealth> {
  const res = await apiFetch(`${BASE_URL}/api/health`);
  return res.json();
}

export async function fetchModels(): Promise<ModelsInfo> {
  const res = await apiFetch(`${BASE_URL}/api/models`);
  return res.json();
}

export async function loadCheckpoint(checkpointName: string): Promise<any> {
  const res = await apiFetch(`${BASE_URL}/api/checkpoints/load`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ checkpoint_name: checkpointName }),
  });
  return res.json();
}

export async function generateSprite(params: {
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
}): Promise<GenerationResult> {
  const res = await apiFetch(`${BASE_URL}/api/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Generation failed');
  }
  return res.json();
}

export async function refineSprite(params: {
  script_text: string;
  suggested_fix?: string;
  width?: number;
  height?: number;
  palette?: string;
}): Promise<{ status: string; frame: FrameData; critic: VisualCriticReport }> {
  const res = await apiFetch(`${BASE_URL}/api/refine`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  return res.json();
}

export async function renderScript(params: {
  script_text: string;
  width?: number;
  height?: number;
  palette?: string;
  scale?: number;
}): Promise<FrameData> {
  const res = await apiFetch(`${BASE_URL}/api/render`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Render failed');
  }
  return res.json();
}

export async function exportAsset(params: {
  frames_scripts: string[];
  width: number;
  height: number;
  palette: string;
  format: 'png' | 'spritesheet' | 'gif' | 'json';
  scale?: number;
  fps?: number;
  cols?: number;
}): Promise<any> {
  const res = await apiFetch(`${BASE_URL}/api/export`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  return res.json();
}

export async function fetchDatasetStats(name = 'default_synth'): Promise<any> {
  const res = await apiFetch(`${BASE_URL}/api/dataset/stats?name=${name}`);
  return res.json();
}

export async function fetchDatasetSamples(name = 'default_synth', limit = 24, offset = 0): Promise<any[]> {
  const res = await apiFetch(`${BASE_URL}/api/dataset/samples?name=${name}&limit=${limit}&offset=${offset}`);
  return res.json();
}

export async function triggerDatasetGeneration(params: {
  name: string;
  total_samples: number;
  defect_rate: number;
  seed: number;
}): Promise<any> {
  const res = await apiFetch(`${BASE_URL}/api/dataset/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  return res.json();
}

export async function startTraining(params: {
  dataset_name: string;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  optimizer: string;
  checkpoint_name: string;
}): Promise<any> {
  const res = await apiFetch(`${BASE_URL}/api/training/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  return res.json();
}

export async function stopTraining(): Promise<any> {
  const res = await apiFetch(`${BASE_URL}/api/training/stop`, { method: 'POST' });
  return res.json();
}

export async function fetchTrainingStatus(): Promise<TrainingStatus> {
  const res = await apiFetch(`${BASE_URL}/api/training/status`);
  return res.json();
}

export async function fetchTrainingHistory(): Promise<any> {
  const res = await apiFetch(`${BASE_URL}/api/training/history`);
  return res.json();
}

export async function fetchBenchmarkGallery(limit = 100): Promise<any> {
  const res = await apiFetch(`${BASE_URL}/api/benchmarks/gallery?limit=${limit}`);
  return res.json();
}

export async function runBenchmark(limit = 20): Promise<any> {
  const res = await apiFetch(`${BASE_URL}/api/benchmarks/run?limit=${limit}`, {
    method: 'POST',
  });
  return res.json();
}

export async function fetchEvaluationResults(): Promise<EvaluationResults> {
  const res = await apiFetch(`${BASE_URL}/api/evaluation/results`);
  return res.json();
}

export async function fetchAssets(query = ''): Promise<AssetRecord[]> {
  const res = await apiFetch(`${BASE_URL}/api/assets${query ? `?query=${encodeURIComponent(query)}` : ''}`);
  return res.json();
}

export async function saveAsset(data: {
  name: string;
  category: string;
  tags?: string[];
  frames: FrameData[];
  metadata?: Record<string, any>;
}): Promise<AssetRecord> {
  const res = await apiFetch(`${BASE_URL}/api/assets`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  return res.json();
}

export async function deleteAsset(id: string): Promise<any> {
  const res = await apiFetch(`${BASE_URL}/api/assets/${id}`, { method: 'DELETE' });
  return res.json();
}

export async function inspectSpriteApi(params: {
  script_text: string;
  category?: string;
  palette_name?: string;
  width?: number;
  height?: number;
}): Promise<any> {
  const res = await apiFetch(`${BASE_URL}/api/critic/inspect`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Inspect failed');
  }
  return res.json();
}

export async function repairSpriteApi(params: {
  script_text: string;
  action_patch?: string;
  auto_refine?: boolean;
  max_iterations?: number;
  category?: string;
  palette_name?: string;
  width?: number;
  height?: number;
}): Promise<any> {
  const res = await apiFetch(`${BASE_URL}/api/critic/repair`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Repair failed');
  }
  return res.json();
}

export async function fetchCriticMetrics(): Promise<any> {
  const res = await apiFetch(`${BASE_URL}/api/critic/metrics`);
  return res.json();
}

export async function generateAnimationApi(params: {
  prompt?: string;
  base_script?: string;
  category?: string;
  style?: string;
  animation_category: string;
  palette?: string;
  width?: number;
  height?: number;
  num_frames?: number;
  fps?: number;
  engine?: string;
  seed?: number;
}): Promise<any> {
  const res = await apiFetch(`${BASE_URL}/api/animation/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Animation generation failed');
  }
  return res.json();
}

export async function regenerateFrameApi(params: {
  base_script: string;
  frame_index: number;
  total_frames: number;
  animation_category: string;
  category?: string;
  palette?: string;
  engine?: string;
  seed?: number;
}): Promise<any> {
  const res = await apiFetch(`${BASE_URL}/api/animation/regenerate_frame`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Frame regeneration failed');
  }
  return res.json();
}

export async function fetchAnimationMetrics(): Promise<any> {
  const res = await apiFetch(`${BASE_URL}/api/animation/metrics`);
  return res.json();
}



export async function planActionSetApi(params: {
  prompt: string;
  category?: string;
  style?: string;
  palette?: string;
  actions?: string[];
}): Promise<any> {
  const res = await apiFetch(`${BASE_URL}/api/action_set/plan`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Action planning failed');
  }
  return res.json();
}

export async function generateActionSetApi(params: {
  prompt: string;
  category?: string;
  style?: string;
  palette?: string;
  width?: number;
  height?: number;
  engine?: string;
  seed?: number;
  actions?: string[];
  max_actions?: number;
  include_gif?: boolean;
}): Promise<any> {
  const res = await apiFetch(`${BASE_URL}/api/action_set`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Action set generation failed');
  }
  return res.json();
}
