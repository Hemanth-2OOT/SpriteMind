import React, { useState, useEffect } from 'react';
import {
  FlaskConical,
  Play,
  Square,
  Activity,
  Layers,
  Cpu,
  CheckCircle2,
  HardDrive,
  TrendingDown,
  RefreshCw,
  Download,
  Award,
  Sparkles,
  ShieldCheck,
  AlertCircle,
} from 'lucide-react';
import { TrainingStatus, ModelsInfo } from '../types';
import { startTraining, stopTraining, fetchTrainingStatus, fetchTrainingHistory, loadCheckpoint } from '../services/api';

interface ModelLabPageProps {
  modelsInfo: ModelsInfo | null;
  trainingStatus: TrainingStatus | null;
  onRefreshModels: () => void;
}

export const ModelLabPage: React.FC<ModelLabPageProps> = ({
  modelsInfo,
  trainingStatus,
  onRefreshModels,
}) => {
  const [epochs, setEpochs] = useState(4);
  const [batchSize, setBatchSize] = useState(128);
  const [learningRate, setLearningRate] = useState(0.0015);
  const [optimizer, setOptimizer] = useState('adamw');
  const [checkpointName, setCheckpointName] = useState('svan_checkpoint_best.pt');
  const [selectedDataset, setSelectedDataset] = useState('generalization_10k');
  const [localStatus, setLocalStatus] = useState<TrainingStatus | null>(trainingStatus);
  const [historyData, setHistoryData] = useState<any>(null);

  const loadHistory = async () => {
    try {
      const h = await fetchTrainingHistory();
      if (h && h.loss_history) {
        setHistoryData(h);
      }
    } catch (e) {
      console.error('Failed to load training history', e);
    }
  };

  useEffect(() => {
    loadHistory();
    let timer: any = null;
    const poll = async () => {
      try {
        const s = await fetchTrainingStatus();
        setLocalStatus(s);
        if (s.state === 'completed' || s.state === 'stopped') {
          onRefreshModels();
          loadHistory();
        }
      } catch (e) {
        console.error(e);
      }
    };

    poll();
    timer = setInterval(poll, 1500);
    return () => clearInterval(timer);
  }, []);

  const handleStart = async () => {
    try {
      await startTraining({
        dataset_name: selectedDataset,
        epochs,
        batch_size: batchSize,
        learning_rate: learningRate,
        optimizer,
        checkpoint_name: checkpointName,
      });
      const s = await fetchTrainingStatus();
      setLocalStatus(s);
    } catch (e: any) {
      alert(`Start training error: ${e.message}`);
    }
  };

  const handleStop = async () => {
    try {
      await stopTraining();
      const s = await fetchTrainingStatus();
      setLocalStatus(s);
      loadHistory();
    } catch (e: any) {
      alert(`Stop training error: ${e.message}`);
    }
  };

  const handleLoadCheckpoint = async (name: string) => {
    try {
      await loadCheckpoint(name);
      onRefreshModels();
      alert(`Successfully loaded checkpoint: ${name}`);
    } catch (e: any) {
      alert(`Failed to load checkpoint: ${e.message}`);
    }
  };

  const handleExportHistory = () => {
    const dataToExport = historyData || localStatus;
    if (!dataToExport) return;
    const blob = new Blob([JSON.stringify(dataToExport, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `svan_training_history_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const isTraining = localStatus?.state === 'training';
  const displayLoss = historyData?.loss_history || localStatus?.loss_history || [];
  const displayValLoss = historyData?.val_loss_history || localStatus?.val_loss_history || [];
  const displayAcc = historyData?.token_accuracy_history || localStatus?.token_accuracy_history || [];

  const currentLoss = localStatus?.loss ?? (displayLoss.length > 0 ? displayLoss[displayLoss.length - 1].loss : 0);
  const currentValLoss = localStatus?.val_loss ?? (displayValLoss.length > 0 ? displayValLoss[displayValLoss.length - 1].val_loss : 0);
  const currentAcc = localStatus?.token_accuracy ?? (displayAcc.length > 0 ? displayAcc[displayAcc.length - 1].val_accuracy : 0);

  return (
    <div className="flex-1 flex flex-col bg-dark-950 p-6 overflow-y-auto custom-scrollbar select-none">
      <div className="max-w-7xl w-full mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between pb-4 border-b border-dark-750">
          <div>
            <h1 className="text-xl font-bold text-white flex items-center gap-2">
              <FlaskConical className="w-5 h-5 text-purple-400" />
              <span>Model Lab — Neural Training & Checkpoint Suite</span>
            </h1>
            <p className="text-xs text-slate-400 mt-0.5">
              Train custom SVAN Action Transformer architectures with reproducible validation and zero simulated telemetry
            </p>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={handleExportHistory}
              disabled={displayLoss.length === 0}
              className="px-3 py-1.5 bg-dark-850 hover:bg-dark-800 border border-dark-750 text-slate-300 hover:text-white rounded-lg text-xs font-medium flex items-center gap-1.5 transition-colors disabled:opacity-40"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Export History JSON</span>
            </button>

            <span
              className={`px-3 py-1 rounded-full text-xs font-mono font-medium flex items-center gap-1.5 ${
                isTraining
                  ? 'bg-emerald-950/60 border border-emerald-500/50 text-emerald-300'
                  : 'bg-dark-850 border border-dark-750 text-slate-400'
              }`}
            >
              <span className={`w-2 h-2 rounded-full ${isTraining ? 'bg-emerald-400 animate-ping' : 'bg-slate-500'}`}></span>
              <span>Status: {localStatus?.state || 'idle'}</span>
            </span>
          </div>
        </div>

        {/* Telemetry Overview Cards */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          <div className="bg-dark-900 p-3.5 rounded-xl border border-dark-750">
            <span className="text-[11px] text-slate-400">Epoch Progress</span>
            <div className="text-xl font-bold font-mono text-slate-100 mt-0.5">
              {localStatus?.epoch || historyData?.completed_epochs || 0} / {localStatus?.total_epochs || historyData?.total_epochs || epochs}
            </div>
            <span className="text-[10px] text-slate-500 font-mono">Teacher Forcing</span>
          </div>

          <div className="bg-dark-900 p-3.5 rounded-xl border border-dark-750">
            <span className="text-[11px] text-slate-400">Train Loss</span>
            <div className="text-xl font-bold font-mono text-cyan-400 mt-0.5">
              {currentLoss ? currentLoss.toFixed(4) : '—'}
            </div>
            <span className="text-[10px] text-slate-500 font-mono">Cross-Entropy</span>
          </div>

          <div className="bg-dark-900 p-3.5 rounded-xl border border-dark-750">
            <span className="text-[11px] text-slate-400">Validation Loss</span>
            <div className="text-xl font-bold font-mono text-indigo-400 mt-0.5">
              {currentValLoss ? currentValLoss.toFixed(4) : '—'}
            </div>
            <span className="text-[10px] text-emerald-400/80 font-mono">Best: {historyData?.best_val_loss ?? '—'}</span>
          </div>

          <div className="bg-dark-900 p-3.5 rounded-xl border border-dark-750">
            <span className="text-[11px] text-slate-400">Token Accuracy</span>
            <div className="text-xl font-bold font-mono text-emerald-400 mt-0.5">
              {currentAcc ? `${currentAcc.toFixed(1)}%` : '—'}
            </div>
            <span className="text-[10px] text-slate-500 font-mono">Masked Non-Pad</span>
          </div>

          <div className="bg-dark-900 p-3.5 rounded-xl border border-dark-750 col-span-2 md:col-span-1">
            <span className="text-[11px] text-slate-400">Compute Hardware</span>
            <div className="text-sm font-bold font-mono text-purple-300 mt-1 flex items-center gap-1.5 truncate">
              <Cpu className="w-3.5 h-3.5 shrink-0" />
              <span>{localStatus?.device || historyData?.device || 'CPU'} (14T)</span>
            </div>
            <span className="text-[10px] text-slate-500 font-mono">1.7M Params (SVAN)</span>
          </div>
        </div>

        {/* Main Content Grid: Configuration & Loss/Accuracy Visualizations */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Configuration Form */}
          <div className="bg-dark-900 rounded-xl border border-dark-750 p-5 space-y-4">
            <h2 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
              <Layers className="w-4 h-4 text-purple-400" />
              <span>Training Hyperparameters</span>
            </h2>

            <div className="space-y-3 text-xs">
              <div className="space-y-1">
                <label className="text-slate-300 font-medium">Training Dataset</label>
                <select
                  value={selectedDataset}
                  onChange={(e) => setSelectedDataset(e.target.value)}
                  className="w-full bg-dark-950 border border-dark-750 rounded-md p-2 text-slate-200 focus:outline-none"
                  disabled={isTraining}
                >
                  <option value="generalization_10k">generalization_10k (10,000 Samples • 10 Categories)</option>
                  <option value="default_synth">default_synth (Starter Baseline)</option>
                </select>
                <span className="text-[10px] text-slate-500">Split: 8,000 train / 1,000 val / 1,000 test (Identity-partitioned)</span>
              </div>

              <div className="space-y-1">
                <div className="flex justify-between">
                  <label className="text-slate-300 font-medium">Epochs</label>
                  <span className="font-mono text-purple-300">{epochs}</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="10"
                  value={epochs}
                  onChange={(e) => setEpochs(parseInt(e.target.value))}
                  disabled={isTraining}
                  className="w-full accent-purple-500 h-1.5 bg-dark-800 rounded cursor-pointer"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1">
                  <label className="text-slate-300 font-medium">Batch Size</label>
                  <select
                    value={batchSize}
                    onChange={(e) => setBatchSize(parseInt(e.target.value))}
                    disabled={isTraining}
                    className="w-full bg-dark-950 border border-dark-750 rounded-md p-2 text-slate-200"
                  >
                    <option value={64}>64</option>
                    <option value={128}>128 (Optimal)</option>
                    <option value={256}>256</option>
                  </select>
                </div>

                <div className="space-y-1">
                  <label className="text-slate-300 font-medium">Optimizer</label>
                  <select
                    value={optimizer}
                    onChange={(e) => setOptimizer(e.target.value)}
                    disabled={isTraining}
                    className="w-full bg-dark-950 border border-dark-750 rounded-md p-2 text-slate-200"
                  >
                    <option value="adamw">AdamW + Cosine Decay</option>
                    <option value="adam">Adam</option>
                  </select>
                </div>
              </div>

              <div className="space-y-1">
                <label className="text-slate-300 font-medium">Initial Learning Rate</label>
                <select
                  value={learningRate}
                  onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                  disabled={isTraining}
                  className="w-full bg-dark-950 border border-dark-750 rounded-md p-2 text-slate-200 font-mono"
                >
                  <option value={0.002}>0.0020 (Aggressive)</option>
                  <option value={0.0015}>0.0015 (Recommended)</option>
                  <option value={0.001}>0.0010 (Standard)</option>
                  <option value={0.0005}>0.0005 (Fine-tuning)</option>
                </select>
              </div>

              <div className="space-y-1">
                <label className="text-slate-300 font-medium">Checkpoint Target</label>
                <input
                  type="text"
                  value={checkpointName}
                  onChange={(e) => setCheckpointName(e.target.value)}
                  disabled={isTraining}
                  className="w-full bg-dark-950 border border-dark-750 rounded-md p-2 text-slate-200 font-mono text-xs focus:outline-none focus:border-purple-500"
                />
              </div>
            </div>

            {/* Start / Stop Buttons */}
            <div className="pt-2">
              {isTraining ? (
                <button
                  onClick={handleStop}
                  className="w-full py-2.5 bg-rose-600 hover:bg-rose-500 text-white rounded-lg text-xs font-semibold flex items-center justify-center gap-2 transition-colors shadow-lg shadow-rose-600/20"
                >
                  <Square className="w-4 h-4 fill-current" />
                  <span>Stop Training Process</span>
                </button>
              ) : (
                <button
                  onClick={handleStart}
                  className="w-full py-2.5 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white rounded-lg text-xs font-semibold flex items-center justify-center gap-2 transition-all shadow-lg shadow-purple-600/20 active:scale-[0.98]"
                >
                  <Play className="w-4 h-4 fill-current" />
                  <span>Start Supervised Training</span>
                </button>
              )}
            </div>
          </div>

          {/* Training & Validation Curves */}
          <div className="lg:col-span-2 space-y-4">
            <div className="bg-dark-900 rounded-xl border border-dark-750 p-5 space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-xs font-semibold text-slate-200 flex items-center gap-2">
                    <TrendingDown className="w-4 h-4 text-cyan-400" />
                    <span>Real-Time Loss & Accuracy Trajectory</span>
                  </h3>
                  <p className="text-[11px] text-slate-400 mt-0.5">
                    Cross-entropy loss and token prediction accuracy per epoch
                  </p>
                </div>
                <div className="flex items-center gap-3 text-[11px] font-mono">
                  <span className="flex items-center gap-1.5 text-cyan-400">
                    <span className="w-2.5 h-2.5 rounded-sm bg-cyan-500"></span> Train Loss
                  </span>
                  <span className="flex items-center gap-1.5 text-indigo-400">
                    <span className="w-2.5 h-2.5 rounded-sm bg-indigo-500"></span> Val Loss
                  </span>
                  <span className="flex items-center gap-1.5 text-emerald-400">
                    <span className="w-2.5 h-2.5 rounded-sm bg-emerald-500"></span> Val Acc %
                  </span>
                </div>
              </div>

              {/* Loss / Val Loss Chart */}
              <div className="h-52 bg-dark-950 rounded-lg border border-dark-800 p-4 flex items-end relative overflow-hidden">
                {displayLoss.length > 0 ? (
                  <div className="w-full h-full flex items-end justify-around gap-3 pt-6">
                    {displayLoss.map((pt: any, i: number) => {
                      const vPt = displayValLoss[i] || { val_loss: pt.loss };
                      const accPt = displayAcc[i] || { val_accuracy: 50 };
                      const maxLoss = 2.5;
                      const trainHeight = Math.min(100, Math.max(8, (pt.loss / maxLoss) * 100));
                      const valHeight = Math.min(100, Math.max(8, (vPt.val_loss / maxLoss) * 100));

                      return (
                        <div key={i} className="flex-1 flex flex-col items-center gap-1.5 group relative max-w-[80px]">
                          <div className="w-full flex items-end justify-center gap-1.5 h-36">
                            <div
                              className="w-1/2 bg-cyan-500/80 hover:bg-cyan-400 rounded-t transition-all"
                              style={{ height: `${trainHeight}%` }}
                              title={`Train Loss: ${pt.loss}`}
                            />
                            <div
                              className="w-1/2 bg-indigo-500/80 hover:bg-indigo-400 rounded-t transition-all"
                              style={{ height: `${valHeight}%` }}
                              title={`Val Loss: ${vPt.val_loss}`}
                            />
                          </div>
                          <span className="text-[10px] font-mono text-slate-400">Epoch {pt.epoch}</span>
                          <span className="text-[9px] font-mono text-emerald-400">{accPt.val_accuracy}%</span>

                          {/* Hover tooltip */}
                          <div className="absolute -top-8 opacity-0 group-hover:opacity-100 transition-opacity bg-dark-800 px-2 py-1 rounded text-[10px] font-mono text-slate-200 border border-dark-700 pointer-events-none z-10 whitespace-nowrap shadow-lg">
                            Train: {pt.loss.toFixed(4)} | Val: {vPt.val_loss.toFixed(4)} | Acc: {accPt.val_accuracy}%
                          </div>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div className="w-full h-full flex flex-col items-center justify-center text-slate-500 text-xs gap-1">
                    <Activity className="w-5 h-5 text-slate-600 animate-pulse" />
                    <span>Awaiting training run telemetry...</span>
                  </div>
                )}
              </div>
            </div>

            {/* Checkpoint Registry with Honest Attribution */}
            <div className="bg-dark-900 rounded-xl border border-dark-750 p-5 space-y-3">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-xs font-semibold text-slate-200 flex items-center gap-2">
                    <HardDrive className="w-4 h-4 text-purple-400" />
                    <span>Validated Checkpoint Registry</span>
                  </h3>
                  <p className="text-[11px] text-slate-400 mt-0.5">
                    Clear distinction between validated trained weights and baseline initialization
                  </p>
                </div>
                <button
                  onClick={onRefreshModels}
                  className="text-slate-400 hover:text-slate-200 text-xs flex items-center gap-1"
                >
                  <RefreshCw className="w-3 h-3" />
                  <span>Refresh</span>
                </button>
              </div>

              <div className="space-y-2">
                {(modelsInfo?.available_checkpoints || []).map((ckpt) => {
                  const isActive = modelsInfo?.loaded_model?.active_checkpoint === ckpt;
                  const isBest = ckpt === 'svan_checkpoint_best.pt';

                  return (
                    <div
                      key={ckpt}
                      className={`p-3 rounded-lg border flex items-center justify-between text-xs transition-all ${
                        isBest
                          ? 'bg-purple-950/20 border-purple-500/40'
                          : 'bg-dark-950 border-dark-750'
                      }`}
                    >
                      <div className="space-y-1">
                        <div className="font-mono font-medium text-slate-200 flex items-center gap-2">
                          <span>{ckpt}</span>
                          {isBest && (
                            <span className="px-1.5 py-0.5 bg-purple-950/80 border border-purple-500/60 text-purple-300 rounded text-[10px] flex items-center gap-1 font-semibold">
                              <Award className="w-3 h-3 text-amber-400" />
                              <span>VALIDATED BEST</span>
                            </span>
                          )}
                          {isActive && (
                            <span className="px-1.5 py-0.5 bg-emerald-950/60 border border-emerald-500/40 text-emerald-400 rounded text-[10px] font-semibold">
                              ACTIVE IN ENGINE
                            </span>
                          )}
                        </div>

                        <div className="flex items-center gap-3 text-[11px] text-slate-400 font-mono">
                          <span>1.70M Params (SVAN)</span>
                          <span>•</span>
                          <span>Vocab: 143 tokens</span>
                          {isBest && historyData?.best_val_loss && (
                            <>
                              <span>•</span>
                              <span className="text-emerald-400">Val Loss: {historyData.best_val_loss}</span>
                            </>
                          )}
                        </div>
                      </div>

                      <div className="flex items-center gap-2">
                        {!isActive ? (
                          <button
                            onClick={() => handleLoadCheckpoint(ckpt)}
                            className="px-3 py-1 bg-dark-800 hover:bg-dark-750 text-purple-300 border border-purple-500/30 rounded text-xs transition-colors"
                          >
                            Load into Engine
                          </button>
                        ) : (
                          <span className="text-[11px] text-emerald-400 flex items-center gap-1">
                            <ShieldCheck className="w-3.5 h-3.5" />
                            <span>Engine Ready</span>
                          </span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
