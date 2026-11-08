import React, { useState, useEffect } from 'react';
import {
  Package,
  Layers,
  Sparkles,
  Download,
  Play,
  Pause,
  RefreshCw,
  BarChart2,
  Grid,
  Shield,
  Crosshair,
  Cpu,
  CheckCircle2,
  AlertTriangle,
} from 'lucide-react';
import { SPRITE_SIZE } from '../config/sprite';

interface AssetRecord {
  asset_id: string;
  name: string;
  category: string;
  sub_category: string;
  width: number;
  height: number;
  script_text: string;
  sockets: Record<string, [number, number]>;
  hitboxes: Record<string, number[]>;
  description: string;
}

export const AssetStudioPage: React.FC = () => {
  const [catalog, setCatalog] = useState<AssetRecord[]>([]);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [characterId, setCharacterId] = useState<string>('character_player_soldier');
  const [weaponId, setWeaponId] = useState<string>('weapon_pump_shotgun');
  const [helmetId, setHelmetId] = useState<string>('');
  const [backpackId, setBackpackId] = useState<string>('');
  const [fxId, setFxId] = useState<string>('');
  const [engineMode, setEngineMode] = useState<string>('trained_model');

  // Composition results
  const [composedImage, setComposedImage] = useState<string>('');
  const [composedScript, setComposedScript] = useState<string>('');
  const [composedSockets, setComposedSockets] = useState<Record<string, [number, number]>>({});
  const [composedHitboxes, setComposedHitboxes] = useState<Record<string, number[]>>({});
  const [layersUsed, setLayersUsed] = useState<string[]>([]);
  const [pixelCount, setPixelCount] = useState<number>(0);
  const [isComposing, setIsComposing] = useState<boolean>(false);

  // Animated mode
  const [isAnimated, setIsAnimated] = useState<boolean>(false);
  const [animCategory, setAnimCategory] = useState<string>('walk');
  const [animFrames, setAnimFrames] = useState<any[]>([]);
  const [activeFrameIdx, setActiveFrameIdx] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [fps, setFps] = useState<number>(8);

  // Modals
  const [showBenchmarkModal, setShowBenchmarkModal] = useState<boolean>(false);
  const [showAtlasModal, setShowAtlasModal] = useState<boolean>(false);
  const [benchmarkData, setBenchmarkData] = useState<any>(null);

  useEffect(() => {
    loadCatalog();
    handleCompose();
  }, []);

  // Animation timer
  useEffect(() => {
    if (!isPlaying || animFrames.length <= 1) return;
    const interval = 1000 / fps;
    const t = setInterval(() => {
      setActiveFrameIdx((prev) => (prev + 1) % animFrames.length);
    }, interval);
    return () => clearInterval(t);
  }, [isPlaying, fps, animFrames.length]);

  const loadCatalog = async () => {
    try {
      const res = await fetch('/api/asset_catalog');
      const data = await res.json();
      setCatalog(data);
    } catch (e) {
      console.error('Failed to load asset catalog', e);
    }
  };

  const handleCompose = async () => {
    setIsComposing(true);
    try {
      if (isAnimated) {
        const res = await fetch('/api/composition/animate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            character_id: characterId,
            animation_category: animCategory,
            num_frames: 6,
            fps,
            weapon_id: weaponId || undefined,
            helmet_id: helmetId || undefined,
            backpack_id: backpackId || undefined,
            fx_id: fxId || undefined,
            engine: engineMode,
          }),
        });
        const data = await res.json();
        if (data.frames && data.frames.length > 0) {
          setAnimFrames(data.frames);
          setActiveFrameIdx(0);
          setComposedImage(data.frames[0].image_base64);
          setPixelCount(data.frames[0].pixel_count);
        }
      } else {
        const res = await fetch('/api/composition/compose', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            character_id: characterId,
            weapon_id: weaponId || undefined,
            helmet_id: helmetId || undefined,
            backpack_id: backpackId || undefined,
            fx_id: fxId || undefined,
            engine: engineMode,
          }),
        });
        const data = await res.json();
        setComposedImage(data.image_base64);
        setComposedScript(data.script_text);
        setComposedSockets(data.sockets || {});
        setComposedHitboxes(data.hitboxes || {});
        setLayersUsed(data.layers_used || []);
        setPixelCount(data.pixel_count || 0);
      }
    } catch (e) {
      console.error('Composition failed', e);
    } finally {
      setIsComposing(false);
    }
  };

  const handleExportZip = async () => {
    try {
      const res = await fetch('/api/asset_pack/export_zip', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pack_type: 'deadzone' }),
      });
      const data = await res.json();
      if (data.zip_base64) {
        const a = document.createElement('a');
        a.href = data.zip_base64;
        a.download = data.filename || 'deadzone_asset_pack.zip';
        a.click();
      }
    } catch (e) {
      alert(`Export failed: ${e}`);
    }
  };

  const openBenchmarks = async () => {
    setShowBenchmarkModal(true);
    try {
      const res = await fetch('/api/composition/metrics');
      const data = await res.json();
      setBenchmarkData(data);
    } catch (e) {
      console.error(e);
    }
  };

  const compBench = benchmarkData?.benchmark_results;
  const compSummary = compBench?.summary ?? {};
  const compEngines: Array<[string, string, string]> = [
    ['procedural_baseline', 'Procedural Baseline', 'text-blue-400'],
    ['untrained_model', 'Untrained Baseline', 'text-amber-400'],
    ['trained_model', 'Trained SocketPredictor', 'text-emerald-400'],
    ['hybrid', 'Hybrid Model', 'text-purple-300'],
  ];
  const compFmt = (v: unknown, digits: number, suffix = '') =>
    typeof v === 'number' ? `${v.toFixed(digits)}${suffix}` : '—';

  const currentDisplayImage =
    isAnimated && animFrames[activeFrameIdx]
      ? animFrames[activeFrameIdx].image_base64
      : composedImage;

  return (
    <div className="flex-1 flex flex-col bg-dark-950 p-6 overflow-y-auto custom-scrollbar select-none">
      <div className="max-w-6xl w-full mx-auto space-y-6">
        {/* Header Bar */}
        <div className="flex flex-wrap items-center justify-between pb-4 border-b border-dark-750 gap-4">
          <div>
            <div className="flex items-center gap-3">
              <h1 className="text-xl font-bold text-white flex items-center gap-2">
                <Package className="w-5 h-5 text-purple-400" />
                <span>Asset Studio & Composition</span>
              </h1>
              <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-950 border border-emerald-500/50 text-emerald-300 flex items-center gap-1.5 shadow-sm">
                <Cpu className="w-3.5 h-3.5 text-emerald-400" />
                [SocketPredictor: 619K Params]
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-1">
              Multi-layer equipment attachment, kinematic animation sync, and game-ready asset pack export
            </p>
          </div>

          <div className="flex items-center gap-2 flex-wrap">
            <button
              onClick={openBenchmarks}
              className="px-3 py-1.5 bg-dark-850 hover:bg-dark-800 text-slate-300 rounded border border-dark-700 text-xs font-medium flex items-center gap-1.5"
            >
              <BarChart2 className="w-3.5 h-3.5 text-purple-400" />
              <span>Benchmarks</span>
            </button>
            <button
              onClick={handleExportZip}
              className="px-3.5 py-1.5 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white rounded text-xs font-semibold flex items-center gap-1.5 shadow-md"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Export Dead Zone Pack (.zip)</span>
            </button>
          </div>
        </div>

        {/* Studio Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Controls Column (5 cols) */}
          <div className="lg:col-span-5 bg-dark-900 border border-dark-750 rounded-xl p-5 space-y-4">
            <div className="flex items-center justify-between border-b border-dark-750 pb-3">
              <h3 className="text-sm font-semibold text-slate-200">Composition Builder</h3>
              <div className="flex items-center gap-1 bg-dark-950 p-1 rounded-lg border border-dark-750 text-xs">
                <button
                  onClick={() => { setIsAnimated(false); }}
                  className={`px-2.5 py-1 rounded font-medium transition-all ${
                    !isAnimated ? 'bg-purple-600 text-white' : 'text-slate-400 hover:text-white'
                  }`}
                >
                  Static
                </button>
                <button
                  onClick={() => { setIsAnimated(true); }}
                  className={`px-2.5 py-1 rounded font-medium transition-all ${
                    isAnimated ? 'bg-purple-600 text-white' : 'text-slate-400 hover:text-white'
                  }`}
                >
                  Animated
                </button>
              </div>
            </div>

            {/* Form selectors */}
            <div className="space-y-3 text-xs">
              <div>
                <label className="text-slate-400 block mb-1">Base Character:</label>
                <select
                  value={characterId}
                  onChange={(e) => setCharacterId(e.target.value)}
                  className="w-full bg-dark-950 border border-dark-750 text-slate-200 rounded px-2.5 py-1.5 focus:outline-none focus:border-purple-500"
                >
                  <option value="character_player_soldier">Special Forces Operative (Player)</option>
                  <option value="character_zombie_walker">Infected Zombie Walker</option>
                  <option value="character_combat_robot">Autonomous Sentry Android</option>
                </select>
              </div>

              <div>
                <label className="text-slate-400 block mb-1">Held Weapon (hand_r):</label>
                <select
                  value={weaponId}
                  onChange={(e) => setWeaponId(e.target.value)}
                  className="w-full bg-dark-950 border border-dark-750 text-slate-200 rounded px-2.5 py-1.5 focus:outline-none focus:border-purple-500"
                >
                  <option value="">(None - Unarmed)</option>
                  <option value="weapon_pump_shotgun">12-Gauge Pump Action Shotgun</option>
                  <option value="weapon_tactical_pistol">9mm Tactical Pistol</option>
                  <option value="weapon_assault_rifle">Assault Rifle AR-24</option>
                  <option value="weapon_heavy_minigun">Vulcan Rotary Minigun</option>
                  <option value="weapon_combat_knife">Tactical Combat Knife</option>
                </select>
              </div>

              <div>
                <label className="text-slate-400 block mb-1">Headgear / Armor (head):</label>
                <select
                  value={helmetId}
                  onChange={(e) => setHelmetId(e.target.value)}
                  className="w-full bg-dark-950 border border-dark-750 text-slate-200 rounded px-2.5 py-1.5 focus:outline-none focus:border-purple-500"
                >
                  <option value="">(None)</option>
                  <option value="armor_combat_helmet">Kevlar Combat Helmet</option>
                </select>
              </div>

              <div>
                <label className="text-slate-400 block mb-1">Back Equipment (back):</label>
                <select
                  value={backpackId}
                  onChange={(e) => setBackpackId(e.target.value)}
                  className="w-full bg-dark-950 border border-dark-750 text-slate-200 rounded px-2.5 py-1.5 focus:outline-none focus:border-purple-500"
                >
                  <option value="">(None)</option>
                  <option value="equipment_military_backpack">Rucksack Supply Backpack</option>
                </select>
              </div>

              <div>
                <label className="text-slate-400 block mb-1">Active FX (muzzle):</label>
                <select
                  value={fxId}
                  onChange={(e) => setFxId(e.target.value)}
                  className="w-full bg-dark-950 border border-dark-750 text-slate-200 rounded px-2.5 py-1.5 focus:outline-none focus:border-purple-500"
                >
                  <option value="">(None)</option>
                  <option value="fx_muzzle_flash_shotgun">12G Kinetic Muzzle Flash</option>
                  <option value="fx_bullet_impact_sparks">Ricochet Impact Sparks</option>
                  <option value="fx_blood_splatter">Organic Blood Splatter</option>
                </select>
              </div>

              <div>
                <label className="text-slate-400 block mb-1">Socket Engine:</label>
                <select
                  value={engineMode}
                  onChange={(e) => setEngineMode(e.target.value)}
                  className="w-full bg-dark-950 border border-dark-750 text-slate-200 rounded px-2.5 py-1.5 focus:outline-none focus:border-purple-500"
                >
                  <option value="trained_model">Trained Neural Sockets (SocketPredictor)</option>
                  <option value="hybrid">Hybrid (Neural Sockets + Bounding Clamping)</option>
                  <option value="procedural_baseline">Procedural Anatomy Heuristic</option>
                  <option value="untrained_model">Untrained Neural Baseline</option>
                </select>
              </div>

              {isAnimated && (
                <div className="bg-dark-950 p-3 rounded-lg border border-dark-750 space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-slate-300 font-semibold">Animation Cycle:</span>
                    <select
                      value={animCategory}
                      onChange={(e) => setAnimCategory(e.target.value)}
                      className="bg-dark-900 border border-dark-750 text-slate-200 rounded px-2 py-1 text-xs"
                    >
                      <option value="walk">Walk Cycle</option>
                      <option value="idle">Idle Breathe</option>
                      <option value="run">Run Cycle</option>
                      <option value="melee_attack">Melee Attack</option>
                    </select>
                  </div>
                </div>
              )}

              <button
                onClick={handleCompose}
                disabled={isComposing}
                className="w-full py-2.5 bg-purple-600 hover:bg-purple-500 text-white rounded-lg font-semibold flex items-center justify-center gap-2 transition-all shadow-md mt-2 disabled:opacity-50"
              >
                <Sparkles className="w-4 h-4" />
                <span>{isComposing ? 'Synthesizing...' : 'Compose & Align Sockets'}</span>
              </button>
            </div>
          </div>

          {/* Preview & Diagnostic Stage (7 cols) */}
          <div className="lg:col-span-7 bg-dark-900 border border-dark-750 rounded-xl p-5 flex flex-col justify-between space-y-4">
            <div>
              <div className="flex items-center justify-between pb-3 border-b border-dark-750">
                <h3 className="text-sm font-semibold text-slate-200">Composed Visual Stage</h3>
                <span className="text-xs text-purple-400 font-mono">{pixelCount} px</span>
              </div>

              {/* Viewport */}
              <div className="flex flex-col items-center justify-center p-6 bg-dark-950 rounded-xl border border-dark-750 mt-4 min-h-[340px] relative">
                <div
                  className="relative shadow-2xl rounded"
                  style={{
                    width: SPRITE_SIZE * 10,
                    height: SPRITE_SIZE * 10,
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
                  {currentDisplayImage && (
                    <img
                      src={currentDisplayImage}
                      alt="Composed Sprite"
                      className="w-full h-full object-contain"
                      style={{ imageRendering: 'pixelated' }}
                    />
                  )}
                </div>

                {isAnimated && animFrames.length > 0 && (
                  <div className="mt-4 flex items-center gap-3 bg-dark-900 px-4 py-2 rounded-xl border border-dark-750 text-xs">
                    <button
                      onClick={() => setIsPlaying(!isPlaying)}
                      className="px-3 py-1 bg-purple-600 hover:bg-purple-500 text-white rounded font-medium"
                    >
                      {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5 fill-current" />}
                    </button>
                    <span className="font-mono text-slate-300">
                      Frame #{activeFrameIdx + 1} of {animFrames.length}
                    </span>
                    <span className="text-slate-400 border-l border-dark-750 pl-2">
                      Speed: {fps} FPS
                    </span>
                  </div>
                )}
              </div>
            </div>

            {/* Layer Stack & Sockets Metadata */}
            <div className="grid grid-cols-2 gap-3 text-xs pt-2">
              <div className="bg-dark-950 p-3 rounded-lg border border-dark-750 space-y-1">
                <div className="font-semibold text-slate-300 flex items-center gap-1.5 mb-1">
                  <Layers className="w-3.5 h-3.5 text-purple-400" />
                  <span>Layer Ordering Stack</span>
                </div>
                {layersUsed.map((l, i) => (
                  <div key={i} className="text-slate-400 font-mono text-[11px] truncate">
                    #{i + 1} {l}
                  </div>
                ))}
              </div>

              <div className="bg-dark-950 p-3 rounded-lg border border-dark-750 space-y-1">
                <div className="font-semibold text-slate-300 flex items-center gap-1.5 mb-1">
                  <Crosshair className="w-3.5 h-3.5 text-cyan-400" />
                  <span>Aligned Sockets (X, Y)</span>
                </div>
                {Object.entries(composedSockets).map(([sname, coords]) => (
                  <div key={sname} className="flex justify-between text-slate-400 font-mono text-[11px]">
                    <span>{sname}:</span>
                    <span className="text-cyan-300">({coords[0]}, {coords[1]})</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Benchmarks Modal */}
        {showBenchmarkModal && (
          <div className="fixed inset-0 z-50 bg-black/70 flex items-center justify-center p-4">
            <div className="bg-dark-900 border border-dark-750 rounded-xl max-w-2xl w-full p-6 space-y-4 shadow-2xl">
              <div className="flex items-center justify-between border-b border-dark-750 pb-3">
                <h3 className="text-base font-bold text-white flex items-center gap-2">
                  <BarChart2 className="w-5 h-5 text-purple-400" />
                  <span>Stage 5 Composition Comparative Benchmark (100 Cases)</span>
                </h3>
                <button
                  onClick={() => setShowBenchmarkModal(false)}
                  className="text-slate-400 hover:text-white text-sm"
                >
                  ✕
                </button>
              </div>

              <div className="space-y-3 text-xs text-slate-300">
                <div className="overflow-x-auto">
                  <table className="w-full text-left border-collapse border border-dark-750 text-xs font-mono">
                    <thead>
                      <tr className="bg-dark-950 text-slate-400 border-b border-dark-750">
                        <th className="p-2.5 font-sans">Engine</th>
                        <th className="p-2.5 font-sans">Attachment Acc</th>
                        <th className="p-2.5 font-sans">Defect Rate</th>
                        <th className="p-2.5 font-sans">Alpha Mask</th>
                        <th className="p-2.5 font-sans">Latency</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-dark-750">
                      {compEngines.map(([key, label, colour]) => {
                        const row = compSummary[key];
                        const highlight = key === 'trained_model' || key === 'hybrid';
                        return (
                          <tr key={key} className={highlight ? 'bg-purple-950/30' : undefined}>
                            <td className={`p-2.5 font-sans font-semibold ${colour}`}>{label}</td>
                            <td className="p-2.5">{compFmt(row?.mean_attachment_accuracy_pct, 1, '%')}</td>
                            <td className="p-2.5">{compFmt(row?.defect_rate_pct, 1, '%')}</td>
                            <td className="p-2.5">{compFmt(row?.alpha_mask_correctness_pct, 1, '%')}</td>
                            <td className="p-2.5">{compFmt(row?.mean_latency_ms, 2, ' ms')}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                <div className="bg-dark-950 p-3 rounded-lg border border-dark-750 text-slate-400 space-y-1">
                  <div className="font-semibold text-slate-200">Read from the benchmark file:</div>
                  {!compBench && <div>No benchmark results loaded.</div>}
                  {compBench && (
                    <>
                      <div>
                        • {compBench.num_test_cases ?? '—'} composition cases, recorded{' '}
                        {compBench.timestamp ?? 'unknown date'}.
                      </div>
                      <div>
                        • SocketPredictor alignment {compFmt(compSummary.procedural_baseline?.mean_attachment_accuracy_pct, 1, '%')}{' '}
                        → {compFmt(compSummary.trained_model?.mean_attachment_accuracy_pct, 1, '%')}; alpha mask
                        correctness {compFmt(compSummary.trained_model?.alpha_mask_correctness_pct, 1, '%')}.
                      </div>
                      <div>
                        • Defect rate is not zero for any engine in this run — open Composition to inspect live
                        results rather than assuming a clean sheet.
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
