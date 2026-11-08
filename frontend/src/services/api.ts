import {
  SystemHealth,
  ModelsInfo,
  GenerationResult,
  TrainingStatus,
  EvaluationResults,
  AssetRecord,
  FrameData,
  VisualCriticReport,
  RuntimeMetrics,
  BenchmarkReportCard,
  BenchmarkGallery,
  HistoryResponse,
  ExportResult,
  ActionSheetResult,
  ExportParams,
  GenerateParams,
  RefineParams,
  RenderParams,
  AssetSaveParams,
  ServiceStatus,
  ActionPlanResponse,
  ActionSetResponse,
  AnimationMetrics,
  CriticMetrics,
  DatasetStats,
  TrainingHistory,
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

type ValidationIssue = {
  loc?: (string | number)[];
  msg?: string;
  type?: string;
};

// FastAPI sends `detail` as a string for HTTPException but as an array of
// issue objects for request validation, so String(detail) used to surface
// "[object Object]" in the UI.
function describeDetail(detail: unknown): string | null {
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    const parts = (detail as ValidationIssue[])
      .map((issue) => {
        const field = (issue?.loc ?? []).filter((part) => part !== 'body').join('.');
        if (!issue?.msg) return field;
        return field ? `${field}: ${issue.msg}` : issue.msg;
      })
      .filter(Boolean);
    return parts.length ? parts.join('; ') : null;
  }
  if (detail && typeof detail === 'object' && 'msg' in (detail as ValidationIssue)) {
    return (detail as ValidationIssue).msg ?? null;
  }
  return null;
}

async function errorMessage(res: globalThis.Response, fallback: string): Promise<string> {
  let detail: unknown;
  try {
    const body = await res.json();
    detail = body?.detail ?? body?.error ?? body?.message;
  } catch {
    detail = null;
  }
  return describeDetail(detail) ?? `${fallback} (HTTP ${res.status})`;
}

export async function fetchHealth(): Promise<SystemHealth> {
  const res = await apiFetch(`${BASE_URL}/api/health`);
  return res.json();
}

export async function fetchModels(): Promise<ModelsInfo> {
  const res = await apiFetch(`${BASE_URL}/api/models`);
  return res.json();
}

export async function fetchRuntimeMetrics(): Promise<RuntimeMetrics> {
  const res = await apiFetch(`${BASE_URL}/api/metrics`);
  return res.json();
}

export async function loadCheckpoint(checkpointName: string): Promise<ServiceStatus> {
  const res = await apiFetch(`${BASE_URL}/api/checkpoints/load`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ checkpoint_name: checkpointName }),
  });
  return res.json();
}

export async function generateSprite(params: GenerateParams): Promise<GenerationResult> {
  const res = await apiFetch(`${BASE_URL}/api/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    throw new Error(await errorMessage(res, 'Generation failed'));
  }
  return res.json();
}

export async function refineSprite(
  params: RefineParams
): Promise<{ status: string; frame: FrameData; critic: VisualCriticReport }> {
  const res = await apiFetch(`${BASE_URL}/api/refine`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  return res.json();
}

export async function renderScript(params: RenderParams): Promise<FrameData> {
  const res = await apiFetch(`${BASE_URL}/api/render`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    throw new Error(await errorMessage(res, 'Render failed'));
  }
  return res.json();
}

export async function exportAsset(params: ExportParams): Promise<ExportResult> {
  const res = await apiFetch(`${BASE_URL}/api/export`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  return res.json();
}

export async function exportActionSheet(params: {
  actions: Record<string, string[]>;
  width?: number;
  height?: number;
  palette?: string;
  scale?: number;
  padding?: number;
}): Promise<ActionSheetResult> {
  const res = await apiFetch(`${BASE_URL}/api/export/sheet`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    throw new Error(await errorMessage(res, 'Sprite sheet export failed'));
  }
  return res.json();
}

export async function fetchDatasetStats(name = 'default_synth'): Promise<DatasetStats> {
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

export async function fetchTrainingHistory(): Promise<TrainingHistory> {
  const res = await apiFetch(`${BASE_URL}/api/training/history`);
  return res.json();
}

export async function fetchHistory(limit = 20): Promise<HistoryResponse> {
  const res = await apiFetch(`${BASE_URL}/api/history?limit=${limit}`);
  return res.json();
}

export async function deleteHistoryRun(id: number): Promise<ServiceStatus> {
  const res = await apiFetch(`${BASE_URL}/api/history/${id}`, { method: 'DELETE' });
  return res.json();
}

export async function clearHistory(): Promise<ServiceStatus> {
  const res = await apiFetch(`${BASE_URL}/api/history`, { method: 'DELETE' });
  return res.json();
}

export async function fetchBenchmarkReportCard(engine = 'trained_model'): Promise<BenchmarkReportCard> {
  const res = await apiFetch(`${BASE_URL}/api/benchmarks/report_card?engine=${engine}`);
  return res.json();
}

export async function fetchBenchmarkGallery(limit = 100): Promise<BenchmarkGallery> {
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

export async function saveAsset(data: AssetSaveParams): Promise<AssetRecord> {
  const res = await apiFetch(`${BASE_URL}/api/assets`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  return res.json();
}

export async function deleteAsset(id: string): Promise<ServiceStatus> {
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
    throw new Error(await errorMessage(res, 'Inspect failed'));
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
    throw new Error(await errorMessage(res, 'Repair failed'));
  }
  return res.json();
}

export async function fetchCriticMetrics(): Promise<CriticMetrics> {
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
    throw new Error(await errorMessage(res, 'Animation generation failed'));
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
    throw new Error(await errorMessage(res, 'Frame regeneration failed'));
  }
  return res.json();
}

export async function fetchAnimationMetrics(): Promise<AnimationMetrics> {
  const res = await apiFetch(`${BASE_URL}/api/animation/metrics`);
  return res.json();
}



export async function planActionSetApi(params: {
  prompt: string;
  category?: string;
  style?: string;
  palette?: string;
  actions?: string[];
  seed?: number;
}): Promise<ActionPlanResponse> {
  const res = await apiFetch(`${BASE_URL}/api/action_set/plan`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    throw new Error(await errorMessage(res, 'Action planning failed'));
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
  facing?: 'left' | 'right';
}): Promise<ActionSetResponse> {
  const res = await apiFetch(`${BASE_URL}/api/action_set`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    throw new Error(await errorMessage(res, 'Action set generation failed'));
  }
  return res.json();
}
