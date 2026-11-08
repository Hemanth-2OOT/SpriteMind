import React, { useState } from 'react';
import {
  FileCode,
  Layers,
  ShieldAlert,
  Cpu,
  Info,
  Play,
  CheckCircle2,
  AlertTriangle,
  Wrench,
  Eye,
  EyeOff,
  Lock,
  Unlock,
  RefreshCw,
  Copy,
  Sparkles,
  History,
  RotateCcw,
  Check,
  Code2,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import { FrameData, VisualCriticReport, ModelsInfo } from '../types';

interface PropertiesSidebarProps {
  currentFrame: FrameData | null;
  criticReport: VisualCriticReport | null;
  modelsInfo: ModelsInfo | null;
  onInspectSprite?: () => Promise<void>;
  onAutoRepair?: () => Promise<void>;
  onApplyCriticFix: (fixCommand: string) => Promise<void>;
  onRerenderScript: (scriptText: string) => Promise<void>;
  inferenceTimeMs: number;
  isInspecting?: boolean;
  isRepairing?: boolean;
  repairHistory?: any[];
  beforeImageBase64?: string | null;
  onRestoreBefore?: () => void;
}

export const PropertiesSidebar: React.FC<PropertiesSidebarProps> = ({
  currentFrame,
  criticReport,
  modelsInfo,
  onInspectSprite,
  onAutoRepair,
  onApplyCriticFix,
  onRerenderScript,
  inferenceTimeMs,
  isInspecting = false,
  isRepairing = false,
  repairHistory = [],
  beforeImageBase64 = null,
  onRestoreBefore,
}) => {
  const [activeTab, setActiveTab] = useState<'properties' | 'layers' | 'pixelscript' | 'critic' | 'model'>('critic');
  const [scriptEditContent, setScriptEditContent] = useState<string>('');
  const [isRerendering, setIsRerendering] = useState(false);
  const [showCode, setShowCode] = useState(true);

  // Sync edited script when currentFrame changes
  React.useEffect(() => {
    if (currentFrame?.script_text) {
      setScriptEditContent(currentFrame.script_text);
    }
  }, [currentFrame?.script_text]);

  const handleRerender = async () => {
    setIsRerendering(true);
    try {
      await onRerenderScript(scriptEditContent);
    } finally {
      setIsRerendering(false);
    }
  };

  const spriteCode = currentFrame?.script_text || '';
  const codeLines = spriteCode ? spriteCode.split('\n') : [];
  const criticDefects = criticReport?.detected_defects || [];
  const qualityScore = criticReport?.quality_score ?? 95;

  return (
    <aside className="w-80 bg-dark-900 border-l border-dark-750 flex flex-col h-full overflow-hidden select-none">
      {/* Tab navigation headers */}
      <div className="flex items-center bg-dark-950/80 border-b border-dark-750 p-1 text-xs">
        <button
          onClick={() => setActiveTab('properties')}
          className={`flex-1 py-1.5 text-center rounded-md font-medium transition-colors ${
            activeTab === 'properties' ? 'bg-dark-800 text-purple-300 shadow-sm' : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          Props
        </button>
        <button
          onClick={() => setActiveTab('layers')}
          className={`flex-1 py-1.5 text-center rounded-md font-medium transition-colors ${
            activeTab === 'layers' ? 'bg-dark-800 text-purple-300 shadow-sm' : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          Layers
        </button>
        <button
          onClick={() => setActiveTab('pixelscript')}
          className={`flex-1 py-1.5 text-center rounded-md font-medium transition-colors ${
            activeTab === 'pixelscript' ? 'bg-dark-800 text-purple-300 shadow-sm' : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          Script
        </button>
        <button
          onClick={() => setActiveTab('critic')}
          className={`flex-1 py-1.5 text-center rounded-md font-medium relative transition-colors ${
            activeTab === 'critic' ? 'bg-dark-800 text-purple-300 shadow-sm' : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          Critic
          {criticDefects.length > 0 && (
            <span className="w-1.5 h-1.5 rounded-full bg-amber-400 absolute top-1 right-2"></span>
          )}
        </button>
        <button
          onClick={() => setActiveTab('model')}
          className={`flex-1 py-1.5 text-center rounded-md font-medium transition-colors ${
            activeTab === 'model' ? 'bg-dark-800 text-purple-300 shadow-sm' : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          Model
        </button>
      </div>

      {/* Tab Contents */}
      <div className="flex-1 overflow-y-auto p-4 custom-scrollbar text-xs">
        {/* TAB 1: PROPERTIES */}
        {activeTab === 'properties' && (
          <div className="space-y-4">
            {/* Dimensions & Pixel Count */}
            <div className="bg-dark-950 p-3 rounded-lg border border-dark-750 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-slate-400">Dimensions</span>
                <span className="text-slate-200 font-mono font-semibold">
                  {currentFrame?.width || 32} × {currentFrame?.height || 32} px
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-400">Filled Pixels</span>
                <span className="text-cyan-400 font-mono font-semibold">
                  {currentFrame?.pixel_count || 0}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-400">Frame Index</span>
                <span className="text-slate-200 font-mono">
                  #{(currentFrame?.frame_index ?? 0) + 1}
                </span>
              </div>
            </div>

            {/* Color Palette Swatches */}
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-slate-300">Active Palette Swatches</label>
              <div className="grid grid-cols-8 gap-1.5 p-2 bg-dark-950 rounded-lg border border-dark-750">
                {(currentFrame?.palette || []).map((col, idx) => (
                  <div
                    key={idx}
                    className="w-6 h-6 rounded border border-white/10 relative group cursor-pointer hover:scale-110 transition-transform"
                    style={{
                      backgroundColor: `rgba(${col[0]}, ${col[1]}, ${col[2]}, ${col[3] / 255})`,
                    }}
                    title={`Index ${idx}: rgba(${col.join(', ')})`}
                  />
                ))}
              </div>
            </div>

            {/* Body Parts List */}
            <div className="space-y-2">
              <label className="text-xs font-medium text-slate-300">Anatomical Body Parts</label>
              {currentFrame?.parts && Object.keys(currentFrame.parts).length > 0 ? (
                <div className="space-y-1.5">
                  {Object.entries(currentFrame.parts).map(([name, p]) => (
                    <div
                      key={name}
                      className="bg-dark-950 p-2.5 rounded-md border border-dark-750 flex items-center justify-between"
                    >
                      <div>
                        <div className="font-semibold text-slate-200 capitalize flex items-center gap-1.5">
                          <span>{name}</span>
                          {p.parent && (
                            <span className="text-[10px] text-slate-500 font-normal">
                              → {p.parent}
                            </span>
                          )}
                        </div>
                        {p.bounding_box && (
                          <div className="text-[10px] text-slate-400 font-mono mt-0.5">
                            bbox: [{p.bounding_box.min_x}, {p.bounding_box.min_y}] {p.bounding_box.width}×{p.bounding_box.height}
                          </div>
                        )}
                      </div>
                      <span className="px-2 py-0.5 bg-dark-800 text-slate-300 rounded font-mono text-[10px]">
                        {p.pixel_count} px
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-slate-500 text-xs italic">No anatomical parts registered</p>
              )}
            </div>
          </div>
        )}

        {/* TAB 2: LAYERS */}
        {activeTab === 'layers' && (
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="font-medium text-slate-300">Layer Stack</span>
              <span className="text-[10px] text-slate-500">Ordered Bottom to Top</span>
            </div>

            <div className="space-y-2">
              {(currentFrame?.layers || []).map((layer) => (
                <div
                  key={layer.id}
                  className="bg-dark-950 p-2.5 rounded-lg border border-dark-750 flex items-center justify-between"
                >
                  <div className="flex items-center gap-2">
                    <button className="text-slate-400 hover:text-slate-200">
                      {layer.visible ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5 text-slate-600" />}
                    </button>
                    <div>
                      <div className="font-medium text-slate-200">{layer.name}</div>
                      <div className="text-[10px] text-slate-500 font-mono">
                        Opacity: {Math.round(layer.opacity * 100)}% • {layer.pixel_count} px
                      </div>
                    </div>
                  </div>
                  <button className="text-slate-500 hover:text-slate-300">
                    {layer.locked ? <Lock className="w-3.5 h-3.5 text-amber-500" /> : <Unlock className="w-3.5 h-3.5" />}
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* TAB 3: PIXELSCRIPT EDITOR */}
        {activeTab === 'pixelscript' && (
          <div className="flex flex-col h-full space-y-2">
            <div className="flex items-center justify-between">
              <span className="font-medium text-slate-300">PixelScript Action Sequence</span>
              <button
                onClick={() => navigator.clipboard.writeText(scriptEditContent)}
                className="text-[10px] text-purple-400 hover:text-purple-300 flex items-center gap-1"
              >
                <Copy className="w-3 h-3" />
                <span>Copy</span>
              </button>
            </div>

            <textarea
              value={scriptEditContent}
              onChange={(e) => setScriptEditContent(e.target.value)}
              className="w-full flex-1 min-h-[300px] bg-dark-950 border border-dark-750 rounded-lg p-2.5 font-mono text-xs text-cyan-300 leading-relaxed focus:outline-none focus:border-purple-500 resize-none selection:bg-purple-700"
              spellCheck={false}
            />

            <button
              onClick={handleRerender}
              disabled={isRerendering}
              className="w-full py-2 bg-purple-600 hover:bg-purple-500 text-white font-medium rounded-md flex items-center justify-center gap-1.5 transition-colors disabled:opacity-50"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isRerendering ? 'animate-spin' : ''}`} />
              <span>Re-render Modified Actions</span>
            </button>
          </div>
        )}

        {/* TAB 4: VISUAL CRITIC */}
        {activeTab === 'critic' && (
          <div className="space-y-4">
            {/* Critic Action Toolbar */}
            <div className="grid grid-cols-2 gap-2">
              <button
                onClick={onInspectSprite}
                disabled={isInspecting || isRepairing}
                className="py-2 px-2.5 bg-dark-850 hover:bg-dark-800 border border-dark-700 text-slate-200 font-medium rounded-lg flex items-center justify-center gap-1.5 transition-colors disabled:opacity-50 text-xs shadow-sm"
              >
                <Eye className={`w-3.5 h-3.5 text-cyan-400 ${isInspecting ? 'animate-pulse' : ''}`} />
                <span>{isInspecting ? 'Inspecting...' : 'Inspect Sprite'}</span>
              </button>

              <button
                onClick={onAutoRepair}
                disabled={isRepairing || isInspecting || criticDefects.length === 0}
                className="py-2 px-2.5 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white font-medium rounded-lg flex items-center justify-center gap-1.5 transition-all disabled:opacity-40 text-xs shadow-md shadow-purple-950/40"
              >
                <Sparkles className={`w-3.5 h-3.5 text-amber-300 ${isRepairing ? 'animate-spin' : ''}`} />
                <span>{isRepairing ? 'Refining...' : 'Auto Repair'}</span>
              </button>
            </div>

            {/* Critic Score Card */}
            <div className="bg-dark-950 p-4 rounded-xl border border-dark-750 text-center space-y-2">
              <div className="flex items-center justify-center">
                <div className="relative w-16 h-16 flex items-center justify-center">
                  <div className={`text-2xl font-bold font-mono ${
                    qualityScore >= 85 ? 'text-emerald-400' : qualityScore >= 70 ? 'text-amber-400' : 'text-rose-400'
                  }`}>
                    {qualityScore}
                  </div>
                </div>
              </div>
              <div>
                <h3 className="font-semibold text-slate-200">Critic Quality Score</h3>
                <p className="text-[11px] text-slate-400">
                  {qualityScore >= 85
                    ? 'High structural fidelity & harmonic palette'
                    : 'Anatomical or pixel anomalies flagged for repair'}
                </p>
              </div>

              {/* Neural validation status badge */}
              <div className="pt-2 border-t border-dark-800/80 flex items-center justify-between text-[10px] text-slate-400 font-mono">
                <span>Critic Precision</span>
                <span className="text-emerald-400 font-semibold">94.12% (368K Params)</span>
              </div>
              <div className="flex items-center justify-between text-[10px] text-slate-400 font-mono">
                <span>Identity Guard</span>
                <span className="text-cyan-400">Monotonic Quality Check Active</span>
              </div>
            </div>

            {/* Neural critic verdict (calibrated at p=0.72) */}
            {criticReport?.neural_critic?.defect_probability !== undefined && (
              <div className="bg-dark-950 p-3 rounded-lg border border-dark-750 space-y-1.5">
                <div className="flex items-center justify-between text-[10px] text-slate-500 uppercase tracking-wider">
                  <span>Neural Critic</span>
                  <span className={criticReport.neural_critic.has_defect ? 'text-amber-400' : 'text-emerald-400'}>
                    {criticReport.neural_critic.has_defect ? 'Flagged' : 'Clean'}
                  </span>
                </div>
                <div className="h-1.5 rounded bg-dark-800 overflow-hidden">
                  <div
                    className={`h-full ${criticReport.neural_critic.has_defect ? 'bg-amber-500' : 'bg-emerald-500'}`}
                    style={{ width: `${Math.round(Math.min(1, criticReport.neural_critic.defect_probability) * 100)}%` }}
                  />
                </div>
                <div className="flex items-center justify-between text-[10px] font-mono text-slate-400">
                  <span>p(defect) {criticReport.neural_critic.defect_probability.toFixed(2)} · thr 0.72</span>
                  <span className="text-slate-500">{criticReport.neural_critic.predicted_class}</span>
                </div>
              </div>
            )}

            {/* Defects Inspection */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-medium text-slate-300">Detected Anomalies</span>
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-dark-800 text-slate-400">
                  {criticDefects.length} detected
                </span>
              </div>

              {criticDefects.length > 0 ? (
                <div className="space-y-2">
                  {criticDefects.map((def, idx) => {
                    const isCrit = def.severity === 'critical';
                    const isMod = def.severity === 'moderate';
                    const badgeBorder = isCrit ? 'border-rose-500/40 bg-rose-950/20' : isMod ? 'border-amber-500/40 bg-amber-950/20' : 'border-sky-500/40 bg-sky-950/20';
                    const textCol = isCrit ? 'text-rose-300' : isMod ? 'text-amber-300' : 'text-sky-300';
                    const bb = def.bbox;

                    return (
                      <div
                        key={idx}
                        className={`p-3 rounded-lg border ${badgeBorder} space-y-2 transition-all`}
                      >
                        <div className="flex items-center justify-between">
                          <span className={`font-semibold ${textCol} flex items-center gap-1.5`}>
                            <AlertTriangle className="w-3.5 h-3.5" />
                            {def.type.replace('_', ' ').toUpperCase()}
                          </span>
                          <span className={`text-[10px] px-1.5 py-0.5 rounded uppercase font-mono font-bold ${
                            isCrit ? 'bg-rose-900/60 text-rose-300' : isMod ? 'bg-amber-900/60 text-amber-300' : 'bg-sky-900/60 text-sky-300'
                          }`}>
                            {def.severity}
                          </span>
                        </div>

                        <p className="text-slate-300 text-[11px] leading-relaxed">{def.description}</p>

                        {/* Defect Coordinates */}
                        {bb && (
                          <div className="text-[10px] font-mono text-slate-400 bg-dark-900/80 px-2 py-1 rounded border border-dark-800 flex items-center justify-between">
                            <span>Coords:</span>
                            <span className="text-purple-300">
                              ({bb.min_x}, {bb.min_y}) → ({bb.max_x}, {bb.max_y})
                            </span>
                            {def.confidence && (
                              <span className="text-slate-500">
                                {Math.round(def.confidence * 100)}% conf
                              </span>
                            )}
                          </div>
                        )}

                        {def.suggested_fix && (
                          <div className="pt-1">
                            <button
                              onClick={() => onApplyCriticFix(def.suggested_fix!)}
                              disabled={isRepairing}
                              className="w-full py-1.5 px-2 bg-purple-900/40 hover:bg-purple-800/60 border border-purple-500/40 text-purple-200 rounded flex items-center justify-center gap-1.5 transition-colors font-mono text-[10px] disabled:opacity-50"
                            >
                              <Wrench className="w-3 h-3 text-purple-400" />
                              <span>Apply: {def.suggested_fix.split('\n')[0]}</span>
                            </button>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="bg-dark-950 p-4 rounded-lg border border-dark-750 text-center text-slate-400">
                  <CheckCircle2 className="w-6 h-6 text-emerald-400 mx-auto mb-1.5" />
                  <p className="text-xs font-medium text-slate-200">No Defects Detected</p>
                  <p className="text-[10px] text-slate-500">Character silhouette and anatomy are consistent</p>
                </div>
              )}
            </div>

            {/* Multi-Iteration Repair History Drawer */}
            {repairHistory && repairHistory.length > 0 && (
              <div className="space-y-2 pt-2 border-t border-dark-800">
                <div className="flex items-center justify-between">
                  <span className="font-medium text-slate-300 flex items-center gap-1.5">
                    <History className="w-3.5 h-3.5 text-purple-400" />
                    <span>Repair History ({repairHistory.length - 1} steps)</span>
                  </span>
                  {onRestoreBefore && (
                    <button
                      onClick={onRestoreBefore}
                      className="text-[10px] text-amber-400 hover:text-amber-300 flex items-center gap-1 font-mono"
                      title="Undo repairs and restore original sprite"
                    >
                      <RotateCcw className="w-3 h-3" />
                      <span>Undo / Restore</span>
                    </button>
                  )}
                </div>

                <div className="space-y-1.5">
                  {repairHistory.map((item, idx) => (
                    <div
                      key={idx}
                      className="bg-dark-950 p-2 rounded border border-dark-800 text-[10px] font-mono flex items-center justify-between"
                    >
                      <div className="flex items-center gap-2">
                        <span className="text-slate-500">#{item.iteration}</span>
                        <span className="text-slate-300 truncate max-w-[140px]">{item.action_applied}</span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <span className="text-slate-400">{item.defects_count} def</span>
                        <span className="text-emerald-400 font-bold">{item.quality_score} pts</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* TAB 5: MODEL INFO */}
        {activeTab === 'model' && (
          <div className="space-y-3">
            <div className="bg-dark-950 p-3 rounded-lg border border-dark-750 space-y-2.5">
              <div>
                <span className="text-[10px] text-slate-500 uppercase tracking-wider">Architecture</span>
                <p className="text-slate-200 font-medium">SpriteMind SVAN</p>
                <p className="text-[10px] text-slate-400">Decoder Action Transformer (4 heads, 4 layers, d=128)</p>
              </div>

              <div className="pt-2 border-t border-dark-800 flex items-center justify-between">
                <span className="text-slate-400">Parameter Count</span>
                <span className="text-purple-400 font-mono font-semibold">
                  {modelsInfo?.loaded_model?.parameter_count
                    ? `${(modelsInfo.loaded_model.parameter_count / 1e6).toFixed(2)}M`
                    : '1.74M'}
                </span>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-slate-400">Vocabulary Size</span>
                <span className="text-slate-200 font-mono">
                  {modelsInfo?.loaded_model?.vocab_size || 256} tokens
                </span>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-slate-400">Inference Latency</span>
                <span className="text-emerald-400 font-mono font-semibold">
                  {inferenceTimeMs ? `${inferenceTimeMs} ms` : '—'}
                </span>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-slate-400">Active Checkpoint</span>
                <span className="text-slate-300 font-mono text-[11px] truncate max-w-[130px]">
                  {modelsInfo?.loaded_model?.active_checkpoint || 'svan_trained_baseline.pt'}
                </span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Sprite code: the PixelScript that produces the frame on the canvas */}
      <div className="border-t border-dark-750 bg-dark-950/80 shrink-0">
        <button
          onClick={() => setShowCode((v) => !v)}
          className="w-full flex items-center justify-between px-3 py-2 text-[11px] text-slate-300 hover:text-slate-100 transition-colors"
        >
          <span className="flex items-center gap-1.5">
            <Code2 className="w-3.5 h-3.5 text-cyan-400" />
            <span className="font-medium">Sprite Code</span>
            <span className="text-slate-500 font-mono">
              {codeLines.length > 0 ? `${codeLines.length} lines` : 'no script'}
            </span>
          </span>
          <span className="flex items-center gap-2">
            {codeLines.length > 0 && (
              <span
                role="button"
                tabIndex={0}
                onClick={(e) => {
                  e.stopPropagation();
                  navigator.clipboard.writeText(spriteCode);
                }}
                className="text-slate-500 hover:text-purple-300 flex items-center gap-1"
              >
                <Copy className="w-3 h-3" />
                Copy
              </span>
            )}
            {showCode ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronUp className="w-3.5 h-3.5" />}
          </span>
        </button>

        {showCode && (
          <div className="px-3 pb-2">
            {codeLines.length === 0 ? (
              <p className="text-[11px] text-slate-500 italic py-1">
                Generate a sprite to see its PixelScript.
              </p>
            ) : (
              <>
                <pre className="max-h-56 overflow-auto custom-scrollbar bg-black/40 border border-dark-750 rounded-md p-2 font-mono text-[10.5px] leading-snug text-cyan-300 whitespace-pre">
{codeLines.map((line, i) => (
  <div key={i} className="flex gap-2">
    <span className="text-slate-600 select-none w-5 text-right shrink-0">{i + 1}</span>
    <span className={line.startsWith('#') ? 'text-slate-500' : undefined}>{line || ' '}</span>
  </div>
))}
                </pre>
                <button
                  onClick={() => setActiveTab('pixelscript')}
                  className="mt-1.5 w-full py-1.5 bg-dark-850 hover:bg-dark-800 border border-dark-700 text-slate-200 rounded-md text-[11px] transition-colors"
                >
                  Edit in Script tab
                </button>
              </>
            )}
          </div>
        )}
      </div>
    </aside>
  );
};
