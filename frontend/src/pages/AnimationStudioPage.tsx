import React, { useState, useEffect } from 'react';
import {
  Film,
  Play,
  Pause,
  Repeat,
  Download,
  Plus,
  Trash2,
  Copy,
  Layers,
  Sliders,
  ChevronLeft,
  ChevronRight,
  ZoomIn,
  ZoomOut,
  Sparkles,
  RefreshCw,
  Cpu,
  BarChart2,
  CheckCircle2,
  AlertTriangle,
  Code,
  Save,
} from 'lucide-react';
import { FrameData } from '../types';
import {
  generateAnimationApi,
  regenerateFrameApi,
  exportAsset,
  renderScript,
  fetchAnimationMetrics,
} from '../services/api';

const ANIMATION_CATEGORIES_GROUPED = [
  {
    group: 'Basic Animations',
    items: [
      { id: 'idle', name: 'Idle (Breathing)' },
      { id: 'walk', name: 'Walk Cycle' },
      { id: 'run', name: 'Running Cycle' },
      { id: 'jump', name: 'Jump Arc' },
      { id: 'fall', name: 'Falling' },
      { id: 'death', name: 'Death Collapse' },
    ],
  },
  {
    group: 'Combat Animations',
    items: [
      { id: 'melee_attack', name: 'Melee Attack' },
      { id: 'ranged_attack', name: 'Ranged Attack' },
      { id: 'hit_reaction', name: 'Hit Reaction' },
      { id: 'taking_damage', name: 'Taking Damage' },
      { id: 'blocking', name: 'Shield Blocking' },
      { id: 'special_ability', name: 'Special Ability' },
      { id: 'cast', name: 'Spell Cast' },
      { id: 'dash', name: 'Dash Burst' },
    ],
  },
  {
    group: 'Creature Animations',
    items: [
      { id: 'slime_bouncing', name: 'Slime Bouncing' },
      { id: 'zombie_movement', name: 'Zombie Movement' },
      { id: 'skeleton_walking', name: 'Skeleton Walking' },
      { id: 'robot_movement', name: 'Robot Movement' },
      { id: 'animal_running', name: 'Animal Running' },
      { id: 'flying_creatures', name: 'Flying Flap' },
    ],
  },
];

const CHARACTER_CATEGORIES = [
  'knight',
  'zombie',
  'rogue',
  'robot',
  'slime',
  'skeleton',
  'wizard',
  'animal',
];

export const AnimationStudioPage: React.FC<{ selectedEngine?: string }> = ({
  selectedEngine = 'procedural_baseline',
}) => {
  const [frames, setFrames] = useState<FrameData[]>([]);
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [fps, setFps] = useState(8);
  const [isLooping, setIsLooping] = useState(true);
  const [onionSkin, setOnionSkin] = useState(true);
  const [animCategory, setAnimCategory] = useState('walk');
  const [charCategory, setCharCategory] = useState('knight');
  const [engineMode, setEngineMode] = useState(selectedEngine);
  const [prompt, setPrompt] = useState('');
  const [scale, setScale] = useState(10);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isRegenerating, setIsRegenerating] = useState(false);
  const [engineUsed, setEngineUsed] = useState<string>(selectedEngine);
  const [identityConfidence, setIdentityConfidence] = useState<number | null>(null);
  const [elapsedMs, setElapsedMs] = useState<number>(0);
  const [seed, setSeed] = useState<number>(() => Math.floor(Math.random() * 999999));
  const [seedPinned, setSeedPinned] = useState(false);

  // Inspector & script editing
  const [showInspector, setShowInspector] = useState(false);
  const [editedScript, setEditedScript] = useState('');
  const [showBenchmarkModal, setShowBenchmarkModal] = useState(false);
  const [benchmarkMetrics, setBenchmarkMetrics] = useState<any>(null);

  // Initial load: benchmark data only. Nothing is generated until the user
  // asks for an animation.
  useEffect(() => {
    fetchAnimationMetrics()
      .then(setBenchmarkMetrics)
      .catch(() => {});
  }, []);

  // Follow the navbar engine selector unless this page overrides it.
  useEffect(() => {
    setEngineMode(selectedEngine);
  }, [selectedEngine]);

  // Sync script editor when active frame changes
  useEffect(() => {
    if (frames[currentFrameIndex]?.script_text) {
      setEditedScript(frames[currentFrameIndex].script_text);
    }
  }, [currentFrameIndex, frames]);

  // Playback timer
  useEffect(() => {
    if (!isPlaying || frames.length <= 1) return;
    const interval = 1000 / fps;
    const t = setInterval(() => {
      setCurrentFrameIndex((prev) => {
        if (prev >= frames.length - 1) {
          return isLooping ? 0 : prev;
        }
        return prev + 1;
      });
    }, interval);
    return () => clearInterval(t);
  }, [isPlaying, fps, frames.length, isLooping]);

  const handleGenerate = async () => {
    setIsGenerating(true);
    try {
      const runSeed = seedPinned ? seed : Math.floor(Math.random() * 999999);
      if (!seedPinned) setSeed(runSeed);
      const res = await generateAnimationApi({
        prompt: prompt.trim() || undefined,
        category: charCategory,
        animation_category: animCategory,
        palette: 'pico8',
        width: 32,
        height: 32,
        num_frames: 6,
        fps,
        engine: engineMode,
        seed: runSeed,
      });

      if (res.frames && res.frames.length > 0) {
        setFrames(res.frames);
        setCurrentFrameIndex(0);
        setEngineUsed(res.engine_used || engineMode);
        setIdentityConfidence(
          typeof res.identity_confidence === 'number' ? res.identity_confidence : null
        );
        setElapsedMs(res.elapsed_ms ?? 0);
        setIsLooping(res.is_looping ?? true);
      }
    } catch (e: any) {
      console.error('Animation generation failed:', e);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleRegenerateCurrentFrame = async () => {
    if (!frames.length) return;
    setIsRegenerating(true);
    try {
      const baseScript = frames[0]?.script_text || '';
      const res = await regenerateFrameApi({
        base_script: baseScript,
        frame_index: currentFrameIndex,
        total_frames: frames.length,
        animation_category: animCategory,
        category: charCategory,
        engine: engineMode,
        seed,
      });

      if (res.frame) {
        const updated = [...frames];
        updated[currentFrameIndex] = res.frame;
        setFrames(updated);
        setEngineUsed(res.engine_used || engineUsed);
      }
    } catch (e: any) {
      console.error('Frame regeneration failed:', e);
    } finally {
      setIsRegenerating(false);
    }
  };

  const handleDuplicateFrame = () => {
    if (!frames[currentFrameIndex]) return;
    const dup = { ...frames[currentFrameIndex] };
    const updated = [
      ...frames.slice(0, currentFrameIndex + 1),
      dup,
      ...frames.slice(currentFrameIndex + 1),
    ];
    setFrames(updated);
    setCurrentFrameIndex(currentFrameIndex + 1);
  };

  const handleDeleteFrame = () => {
    if (frames.length <= 2) {
      alert('An animation sequence requires at least 2 frames.');
      return;
    }
    const updated = frames.filter((_, idx) => idx !== currentFrameIndex);
    setFrames(updated);
    setCurrentFrameIndex(Math.min(currentFrameIndex, updated.length - 1));
  };

  const handleUpdateScript = async () => {
    if (!frames[currentFrameIndex]) return;
    try {
      const rendered = await renderScript({
        script_text: editedScript,
        width: 32,
        height: 32,
        palette: 'pico8',
      });
      const updated = [...frames];
      updated[currentFrameIndex] = {
        ...rendered,
        script_text: editedScript,
      };
      setFrames(updated);
    } catch (e: any) {
      alert(`Script update failed: ${e.message}`);
    }
  };

  const handleExport = async (format: 'gif' | 'spritesheet' | 'png') => {
    if (frames.length === 0) return;
    try {
      const res = await exportAsset({
        frames_scripts: frames.map((f) => f.script_text),
        width: 32,
        height: 32,
        palette: 'pico8',
        format,
        scale: 4,
        fps,
      });
      if (res.image_base64) {
        const a = document.createElement('a');
        a.href = res.image_base64;
        a.download = `anim_${charCategory}_${animCategory}.${format === 'gif' ? 'gif' : 'png'}`;
        a.click();
      }
    } catch (e: any) {
      console.error('Export failed:', e);
    }
  };

  const openBenchmarks = async () => {
    setShowBenchmarkModal(true);
    try {
      const metrics = await fetchAnimationMetrics();
      setBenchmarkMetrics(metrics);
    } catch (e) {
      console.error('Failed to load metrics', e);
    }
  };

  const animBench = benchmarkMetrics?.benchmark_results;
  const animSummary = animBench?.summary ?? {};
  const animEngines: Array<[string, string, string]> = [
    ['procedural_baseline', 'Procedural Baseline', 'text-blue-400'],
    ['untrained_model', 'Untrained Baseline', 'text-amber-400'],
    ['trained_model', 'Trained Temporal SVAN', 'text-emerald-400'],
  ];
  const fmt = (v: unknown, digits: number, suffix = '') =>
    typeof v === 'number' ? `${v.toFixed(digits)}${suffix}` : '—';

  const curr = frames[currentFrameIndex];
  const prevFrame =
    currentFrameIndex > 0 ? frames[currentFrameIndex - 1] : frames[frames.length - 1];

  return (
    <div className="flex-1 flex flex-col bg-dark-950 p-6 overflow-y-auto custom-scrollbar select-none">
      <div className="max-w-6xl w-full mx-auto space-y-6">
        {/* Header Bar */}
        <div className="flex flex-wrap items-center justify-between pb-4 border-b border-dark-750 gap-4">
          <div>
            <div className="flex items-center gap-3">
              <h1 className="text-xl font-bold text-white flex items-center gap-2">
                <Film className="w-5 h-5 text-purple-400" />
                <span>Animation Studio</span>
              </h1>
              {/* Attribution Badges */}
              {engineUsed === 'trained_model' && (
                <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-950 border border-emerald-500/50 text-emerald-300 flex items-center gap-1.5 shadow-sm shadow-emerald-950/50">
                  <Cpu className="w-3.5 h-3.5 text-emerald-400" />
                  [Trained Temporal SVAN]
                </span>
              )}
              {engineUsed === 'untrained_model' && (
                <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-amber-950 border border-amber-500/50 text-amber-300 flex items-center gap-1.5 shadow-sm shadow-amber-950/50">
                  <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
                  [Untrained Neural Baseline]
                </span>
              )}
              {engineUsed === 'procedural_baseline' && (
                <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-blue-950 border border-blue-500/50 text-blue-300 flex items-center gap-1.5 shadow-sm shadow-blue-950/50">
                  <Sliders className="w-3.5 h-3.5 text-blue-400" />
                  [Procedural Kinematic Baseline]
                </span>
              )}
            </div>
            <p className="text-xs text-slate-400 mt-1">
              Multi-frame temporal animation, kinematic trajectory modeling, and PixelScript sequencing
            </p>
          </div>

          <div className="flex items-center gap-2 flex-wrap">
            <button
              onClick={openBenchmarks}
              className="px-3 py-1.5 bg-dark-850 hover:bg-dark-800 text-slate-300 rounded-md border border-dark-700 text-xs font-medium flex items-center gap-1.5 transition-colors"
            >
              <BarChart2 className="w-3.5 h-3.5 text-purple-400" />
              <span>Model Benchmarks</span>
            </button>

            <button
              onClick={() => handleExport('spritesheet')}
              className="px-3 py-1.5 bg-purple-600 hover:bg-purple-500 text-white rounded-md text-xs font-medium flex items-center gap-1.5 transition-colors shadow-sm"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Export Spritesheet</span>
            </button>
            <button
              onClick={() => handleExport('gif')}
              className="px-3 py-1.5 bg-dark-850 hover:bg-dark-800 text-purple-300 rounded-md border border-purple-500/30 text-xs font-medium flex items-center gap-1.5 transition-colors"
            >
              <span>GIF</span>
            </button>
            <button
              onClick={() => handleExport('png')}
              className="px-3 py-1.5 bg-dark-850 hover:bg-dark-800 text-slate-300 rounded-md border border-dark-700 text-xs font-medium flex items-center gap-1.5 transition-colors"
            >
              <span>PNG Frame</span>
            </button>
          </div>
        </div>

        {/* Prompt & Config Bar */}
        <div className="bg-dark-900 border border-dark-750 p-4 rounded-xl space-y-3">
          <div className="flex flex-col md:flex-row gap-3">
            <input
              type="text"
              placeholder="Animation prompt (e.g., 'knight swinging sword', 'zombie limping forward')..."
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleGenerate()}
              className="flex-1 bg-dark-950 border border-dark-750 rounded-lg px-3 py-2 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-purple-500"
            />
            <button
              onClick={handleGenerate}
              disabled={isGenerating}
              className="px-4 py-2 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white rounded-lg text-xs font-semibold flex items-center justify-center gap-2 transition-all shadow-md disabled:opacity-50"
            >
              <Sparkles className="w-4 h-4" />
              <span>{isGenerating ? 'Synthesizing...' : 'Generate Animation'}</span>
            </button>
          </div>

          <div className="flex flex-wrap items-center gap-4 text-xs pt-1">
            <div className="flex items-center gap-2">
              <span className="text-slate-400">Character:</span>
              <select
                value={charCategory}
                onChange={(e) => setCharCategory(e.target.value)}
                className="bg-dark-950 border border-dark-750 text-slate-200 text-xs rounded-md px-2.5 py-1 focus:outline-none focus:border-purple-500 capitalize"
              >
                {CHARACTER_CATEGORIES.map((c) => (
                  <option key={c} value={c} className="capitalize">
                    {c}
                  </option>
                ))}
              </select>
            </div>

            <div className="flex items-center gap-2">
              <span className="text-slate-400">Animation Category:</span>
              <select
                value={animCategory}
                onChange={(e) => setAnimCategory(e.target.value)}
                className="bg-dark-950 border border-dark-750 text-slate-200 text-xs rounded-md px-2.5 py-1 focus:outline-none focus:border-purple-500"
              >
                {ANIMATION_CATEGORIES_GROUPED.map((group) => (
                  <optgroup key={group.group} label={group.group} className="text-slate-400 bg-dark-900">
                    {group.items.map((item) => (
                      <option key={item.id} value={item.id} className="text-slate-200 bg-dark-950">
                        {item.name}
                      </option>
                    ))}
                  </optgroup>
                ))}
              </select>
            </div>

            <div className="flex items-center gap-2">
              <span className="text-slate-400">Engine:</span>
              <select
                value={engineMode}
                onChange={(e) => setEngineMode(e.target.value)}
                className="bg-dark-950 border border-dark-750 text-slate-200 text-xs rounded-md px-2.5 py-1 focus:outline-none focus:border-purple-500"
              >
                <option value="trained_model">Trained Temporal SVAN (Recurrent)</option>
                <option value="untrained_model">Untrained Neural Baseline</option>
                <option value="procedural_baseline">Procedural Kinematic Baseline</option>
              </select>
            </div>

            <div className="flex items-center gap-2">
              <span className="text-slate-400">Seed:</span>
              <input
                type="number"
                value={seed}
                onChange={(e) => {
                  setSeed(Number(e.target.value) || 0);
                  setSeedPinned(true);
                }}
                title="Same prompt and seed rebuild the same animation; a new seed makes a new variation"
                className="w-20 bg-dark-950 border border-dark-750 text-slate-200 text-xs rounded-md px-2.5 py-1 font-mono text-right focus:outline-none focus:border-purple-500"
              />
              <button
                onClick={() => {
                  setSeed(Math.floor(Math.random() * 999999));
                  setSeedPinned(false);
                }}
                title="New variation"
                className="px-2 py-1 bg-dark-850 hover:bg-dark-800 text-slate-300 rounded-md border border-dark-700"
              >
                <RefreshCw className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        </div>

        {/* Studio Workspace Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main Stage */}
          <div className="lg:col-span-2 bg-dark-900 rounded-xl border border-dark-750 p-6 flex flex-col items-center justify-center min-h-[440px] relative">
            {/* Viewport Canvas Preview */}
            <div
              className="relative shadow-2xl rounded"
              style={{
                width: 32 * scale,
                height: 32 * scale,
                backgroundImage: `
                  linear-gradient(45deg, #151824 25%, transparent 25%),
                  linear-gradient(-45deg, #151824 25%, transparent 25%),
                  linear-gradient(45deg, transparent 75%, #151824 75%),
                  linear-gradient(-45deg, transparent 75%, #151824 75%)
                `,
                backgroundSize: '16px 16px',
                backgroundPosition: '0 0, 0 8px, 8px -8px, -8px 0px',
              }}
            >
              {onionSkin && prevFrame && (
                <img
                  src={prevFrame.image_base64}
                  alt="Onion Previous"
                  className="absolute inset-0 w-full h-full pointer-events-none opacity-25 filter hue-rotate-180"
                  style={{ imageRendering: 'pixelated' }}
                />
              )}
              {curr && (
                <img
                  src={curr.image_base64}
                  alt="Current Animation Frame"
                  className="w-full h-full object-contain"
                  style={{ imageRendering: 'pixelated' }}
                />
              )}
            </div>

            {/* Stage controls overlay */}
            <div className="mt-6 flex flex-wrap items-center gap-4 bg-dark-950/80 px-4 py-2 rounded-xl border border-dark-750">
              <div className="flex items-center gap-1">
                <button
                  onClick={() => setCurrentFrameIndex((prev) => Math.max(0, prev - 1))}
                  className="p-1.5 text-slate-400 hover:text-white rounded hover:bg-dark-800"
                  title="Previous Frame"
                >
                  <ChevronLeft className="w-4 h-4" />
                </button>
                <button
                  onClick={() => setIsPlaying(!isPlaying)}
                  className="p-2 bg-purple-600 hover:bg-purple-500 text-white rounded-lg transition-colors"
                  title={isPlaying ? 'Pause' : 'Play'}
                >
                  {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 fill-current" />}
                </button>
                <button
                  onClick={() => setCurrentFrameIndex((prev) => (prev + 1) % frames.length)}
                  className="p-1.5 text-slate-400 hover:text-white rounded hover:bg-dark-800"
                  title="Next Frame"
                >
                  <ChevronRight className="w-4 h-4" />
                </button>
              </div>

              <div className="flex items-center gap-2 border-l border-dark-750 pl-4">
                <span className="text-xs text-slate-400 font-mono">Speed: {fps} FPS</span>
                <input
                  type="range"
                  min="1"
                  max="24"
                  value={fps}
                  onChange={(e) => setFps(parseInt(e.target.value))}
                  className="w-20 accent-purple-500 h-1 bg-dark-800 rounded"
                />
              </div>

              <div className="flex items-center gap-3 border-l border-dark-750 pl-4">
                <label className="text-xs text-slate-300 flex items-center gap-1.5 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={onionSkin}
                    onChange={(e) => setOnionSkin(e.target.checked)}
                    className="accent-purple-500 rounded"
                  />
                  <span>Onion Skin</span>
                </label>
                <label className="text-xs text-slate-300 flex items-center gap-1.5 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={isLooping}
                    onChange={(e) => setIsLooping(e.target.checked)}
                    className="accent-purple-500 rounded"
                  />
                  <span>Loop</span>
                </label>
              </div>

              <div className="flex items-center gap-1 border-l border-dark-750 pl-4">
                <button
                  onClick={() => setScale(Math.max(6, scale - 2))}
                  className="p-1 text-slate-400 hover:text-white"
                  title="Zoom Out"
                >
                  <ZoomOut className="w-3.5 h-3.5" />
                </button>
                <span className="text-[11px] font-mono text-slate-400">{scale}x</span>
                <button
                  onClick={() => setScale(Math.min(16, scale + 2))}
                  className="p-1 text-slate-400 hover:text-white"
                  title="Zoom In"
                >
                  <ZoomIn className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          </div>

          {/* Sidebar & Info */}
          <div className="bg-dark-900 rounded-xl border border-dark-750 p-5 space-y-4 flex flex-col justify-between">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold text-slate-200">Sequence Diagnostics</h3>
                <button
                  onClick={() => setShowInspector(!showInspector)}
                  className="text-xs text-purple-400 hover:text-purple-300 flex items-center gap-1"
                >
                  <Code className="w-3.5 h-3.5" />
                  <span>{showInspector ? 'Hide Script' : 'Inspect Script'}</span>
                </button>
              </div>

              <div className="bg-dark-950 p-3 rounded-lg border border-dark-750 space-y-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-slate-400">Total Frames</span>
                  <span className="text-slate-200 font-mono font-bold">{frames.length}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Active Frame</span>
                  <span className="text-purple-400 font-mono font-bold">#{currentFrameIndex + 1}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Identity Confidence</span>
                  <span className="text-emerald-400 font-mono font-bold">
                    {identityConfidence === null ? '—' : `${(identityConfidence * 100).toFixed(1)}%`}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Cycle Duration</span>
                  <span className="text-slate-200 font-mono">
                    {((frames.length / fps) * 1000).toFixed(0)} ms / cycle
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Latency</span>
                  <span className="text-slate-400 font-mono">{elapsedMs.toFixed(1)} ms</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Loop Continuity</span>
                  <span className={`font-mono ${isLooping ? 'text-cyan-400' : 'text-amber-400'}`}>
                    {isLooping ? 'Cyclic Loop' : 'Linear Arc'}
                  </span>
                </div>
              </div>

              {/* Per-Frame Script Inspector & Live Editor */}
              {showInspector ? (
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-xs text-slate-300">
                    <span>PixelScript (Frame #{currentFrameIndex + 1})</span>
                    <button
                      onClick={handleUpdateScript}
                      className="px-2 py-0.5 bg-purple-600 hover:bg-purple-500 text-white rounded text-[11px] flex items-center gap-1"
                    >
                      <Save className="w-3 h-3" />
                      <span>Update</span>
                    </button>
                  </div>
                  <textarea
                    value={editedScript}
                    onChange={(e) => setEditedScript(e.target.value)}
                    rows={8}
                    className="w-full bg-dark-950 border border-dark-750 rounded p-2 text-[11px] font-mono text-purple-200 focus:outline-none focus:border-purple-500 custom-scrollbar"
                  />
                </div>
              ) : (
                /* Frame Actions */
                <div className="space-y-2">
                  <div className="text-xs font-medium text-slate-300">Frame Operations</div>
                  <div className="grid grid-cols-3 gap-2">
                    <button
                      onClick={handleRegenerateCurrentFrame}
                      disabled={isRegenerating}
                      className="p-2 bg-dark-950 hover:bg-dark-850 border border-dark-750 text-slate-300 rounded-lg text-xs flex flex-col items-center gap-1 transition-colors disabled:opacity-50"
                      title="Re-predict this frame using temporal model"
                    >
                      <RefreshCw className={`w-3.5 h-3.5 text-purple-400 ${isRegenerating ? 'animate-spin' : ''}`} />
                      <span>Regen</span>
                    </button>
                    <button
                      onClick={handleDuplicateFrame}
                      className="p-2 bg-dark-950 hover:bg-dark-850 border border-dark-750 text-slate-300 rounded-lg text-xs flex flex-col items-center gap-1 transition-colors"
                      title="Duplicate active frame"
                    >
                      <Copy className="w-3.5 h-3.5 text-cyan-400" />
                      <span>Duplicate</span>
                    </button>
                    <button
                      onClick={handleDeleteFrame}
                      className="p-2 bg-dark-950 hover:bg-dark-850 border border-dark-750 text-slate-300 rounded-lg text-xs flex flex-col items-center gap-1 transition-colors"
                      title="Delete active frame"
                    >
                      <Trash2 className="w-3.5 h-3.5 text-rose-400" />
                      <span>Delete</span>
                    </button>
                  </div>
                </div>
              )}
            </div>

            <div className="text-[11px] text-slate-500 border-t border-dark-750/60 pt-3 flex items-center justify-between">
              <span>Architecture: TemporalSVAN</span>
              <span>
                {typeof benchmarkMetrics?.parameter_count === 'number'
                  ? `${Math.round(benchmarkMetrics.parameter_count / 1000)}K Parameters`
                  : '— Parameters'}
              </span>
            </div>
          </div>
        </div>

        {/* Timeline Track */}
        <div className="bg-dark-900 rounded-xl border border-dark-750 p-4 space-y-3">
          <div className="flex items-center justify-between text-xs">
            <span className="font-semibold text-slate-300 flex items-center gap-2">
              <Layers className="w-4 h-4 text-purple-400" />
              <span>Timeline Track</span>
            </span>
            <span className="text-slate-500 font-mono">
              Frame {currentFrameIndex + 1} of {frames.length}
            </span>
          </div>

          <div className="flex items-center gap-2 overflow-x-auto pb-2 custom-scrollbar">
            {frames.map((f, i) => (
              <div
                key={i}
                onClick={() => setCurrentFrameIndex(i)}
                className={`flex-shrink-0 flex flex-col items-center p-1.5 rounded-lg border cursor-pointer transition-all ${
                  i === currentFrameIndex
                    ? 'bg-purple-950/60 border-purple-500 shadow-md ring-1 ring-purple-500/50'
                    : 'bg-dark-950 border-dark-750 hover:border-dark-700 opacity-80 hover:opacity-100'
                }`}
              >
                <div className="w-14 h-14 bg-dark-900 rounded border border-dark-750/80 flex items-center justify-center p-1">
                  <img
                    src={f.image_base64}
                    alt={`Frame ${i + 1}`}
                    className="w-full h-full object-contain"
                    style={{ imageRendering: 'pixelated' }}
                  />
                </div>
                <span className="text-[10px] font-mono text-slate-300 mt-1">#{i + 1}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Benchmarks Modal */}
        {showBenchmarkModal && (
          <div className="fixed inset-0 z-50 bg-black/70 flex items-center justify-center p-4">
            <div className="bg-dark-900 border border-dark-750 rounded-xl max-w-2xl w-full p-6 space-y-4 shadow-2xl">
              <div className="flex items-center justify-between border-b border-dark-750 pb-3">
                <h3 className="text-base font-bold text-white flex items-center gap-2">
                  <BarChart2 className="w-5 h-5 text-purple-400" />
                  <span>Temporal SVAN Comparative Benchmark Results</span>
                </h3>
                <button
                  onClick={() => setShowBenchmarkModal(false)}
                  className="text-slate-400 hover:text-white text-sm"
                >
                  ✕
                </button>
              </div>

              <div className="space-y-3 text-xs text-slate-300">
                <p>
                  Evaluated across{' '}
                  <span className="text-slate-100 font-semibold">
                    {animBench?.total_sequences_evaluated ?? '—'}
                  </span>{' '}
                  test animation sequences comparing all three execution engines under identical evaluation
                  conditions.
                </p>

                <div className="overflow-x-auto">
                  <table className="w-full text-left border-collapse border border-dark-750 text-xs">
                    <thead>
                      <tr className="bg-dark-950 text-slate-400 border-b border-dark-750">
                        <th className="p-2.5">Engine</th>
                        <th className="p-2.5">Identity IoU</th>
                        <th className="p-2.5">Smoothness</th>
                        <th className="p-2.5">Loop Consistency</th>
                        <th className="p-2.5">Flicker %</th>
                        <th className="p-2.5">Latency</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-dark-750 font-mono">
                      {animEngines.map(([key, label, colour]) => {
                        const row = animSummary[key];
                        return (
                          <tr
                            key={key}
                            className={key === 'trained_model' ? 'bg-purple-950/30' : 'hover:bg-dark-850/50'}
                          >
                            <td className={`p-2.5 font-sans font-semibold ${colour}`}>{label}</td>
                            <td className="p-2.5">{fmt(row?.mean_identity_iou, 4)}</td>
                            <td className="p-2.5">{fmt(row?.mean_smoothness, 2)}</td>
                            <td className="p-2.5">{fmt(row?.mean_loop_consistency, 4)}</td>
                            <td className="p-2.5">{fmt(row?.mean_flicker_rate_pct, 3, '%')}</td>
                            <td className="p-2.5">{fmt(row?.mean_latency_ms, 2, ' ms')}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                <div className="bg-dark-950 p-3 rounded-lg border border-dark-750 text-slate-400 space-y-1">
                  <div className="font-semibold text-slate-200">Read from the benchmark file:</div>
                  {!animBench && <div>No benchmark results loaded.</div>}
                  {animBench && (
                    <>
                      <div>
                        • Trained model vs procedural: identity IoU{' '}
                        {fmt(animSummary.procedural_baseline?.mean_identity_iou, 4)} →{' '}
                        {fmt(animSummary.trained_model?.mean_identity_iou, 4)}, flicker{' '}
                        {fmt(animSummary.procedural_baseline?.mean_flicker_rate_pct, 3, '%')} →{' '}
                        {fmt(animSummary.trained_model?.mean_flicker_rate_pct, 3, '%')}.
                      </div>
                      <div>
                        • Untrained baseline is expected to be static; its high identity and smoothness are artifacts,
                        not articulation.
                      </div>
                      <div>
                        • Measured on {animBench.total_sequences_evaluated ?? '—'} sequences over{' '}
                        {(animBench.categories_tested ?? []).length || '—'} categories.
                      </div>
                    </>
                  )}
                </div>
              </div>

              <div className="flex justify-end pt-2">
                <button
                  onClick={() => setShowBenchmarkModal(false)}
                  className="px-4 py-2 bg-purple-600 hover:bg-purple-500 text-white rounded-lg text-xs font-medium"
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
