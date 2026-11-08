import React, { useState, useEffect } from 'react';
import {
  Database,
  RefreshCw,
  Sliders,
  CheckCircle2,
  AlertTriangle,
  FileCode,
  Layers,
  Sparkles,
} from 'lucide-react';
import { fetchDatasetStats, fetchDatasetSamples, triggerDatasetGeneration } from '../services/api';

export const DatasetStudioPage: React.FC = () => {
  const [stats, setStats] = useState<any>(null);
  const [samples, setSamples] = useState<any[]>([]);
  const [selectedSample, setSelectedSample] = useState<any>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [sampleCount, setSampleCount] = useState(200);
  const [defectRate, setDefectRate] = useState(0.2);

  const loadData = async () => {
    try {
      const st = await fetchDatasetStats();
      setStats(st);
      const samps = await fetchDatasetSamples('default_synth', 36, 0);
      setSamples(samps);
      if (samps.length > 0) {
        setSelectedSample(samps[0]);
      }
    } catch (e: any) {
      console.error(e);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleGenerateDataset = async () => {
    setIsGenerating(true);
    try {
      await triggerDatasetGeneration({
        name: 'default_synth',
        total_samples: sampleCount,
        defect_rate: defectRate,
        seed: Date.now() % 100000,
      });
      await loadData();
    } catch (e: any) {
      alert(`Generation failed: ${e.message}`);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="flex-1 flex flex-col bg-dark-950 p-6 overflow-y-auto custom-scrollbar select-none">
      <div className="max-w-7xl w-full mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between pb-4 border-b border-dark-750">
          <div>
            <h1 className="text-xl font-bold text-white flex items-center gap-2">
              <Database className="w-5 h-5 text-purple-400" />
              <span>Dataset Studio</span>
            </h1>
            <p className="text-xs text-slate-400 mt-0.5">
              Inspect synthetic pixel-art datasets, defect annotations, and split distributions
            </p>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={loadData}
              className="p-1.5 bg-dark-850 hover:bg-dark-800 text-slate-300 rounded-md border border-dark-750 transition-colors"
              title="Refresh Dataset"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Dataset Statistics Cards */}
        {stats && (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="bg-dark-900 p-4 rounded-xl border border-dark-750 space-y-1">
              <span className="text-xs text-slate-400 font-medium">Total Samples</span>
              <div className="text-2xl font-bold text-white font-mono">{stats.total_samples}</div>
              <p className="text-[11px] text-slate-500">RGBA action sequences (training data)</p>
            </div>

            <div className="bg-dark-900 p-4 rounded-xl border border-dark-750 space-y-1">
              <span className="text-xs text-slate-400 font-medium">Data Splits</span>
              <div className="text-xs font-mono text-slate-300 space-y-0.5 pt-1">
                <div className="flex justify-between">
                  <span className="text-emerald-400">Train:</span>
                  <span>{stats.splits?.train || 0}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-cyan-400">Val:</span>
                  <span>{stats.splits?.val || 0}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-purple-400">Test:</span>
                  <span>{stats.splits?.test || 0}</span>
                </div>
              </div>
            </div>

            <div className="bg-dark-900 p-4 rounded-xl border border-dark-750 space-y-1">
              <span className="text-xs text-slate-400 font-medium">Categories</span>
              <div className="text-xs text-slate-300 flex flex-wrap gap-1 pt-1">
                {Object.keys(stats.category_distribution || {}).map((c) => (
                  <span key={c} className="px-1.5 py-0.5 rounded bg-dark-950 border border-dark-750 text-[10px]">
                    {c}
                  </span>
                ))}
              </div>
            </div>

            <div className="bg-dark-900 p-4 rounded-xl border border-dark-750 space-y-1">
              <span className="text-xs text-slate-400 font-medium">Defect Samples</span>
              <div className="text-2xl font-bold text-amber-400 font-mono">
                {stats.total_samples - (stats.defect_distribution?.clean || 0)}
              </div>
              <p className="text-[11px] text-slate-500">Visual Critic Training Target</p>
            </div>
          </div>
        )}

        {/* Generate Synthetic Samples Control Panel */}
        <div className="bg-dark-900 rounded-xl border border-dark-750 p-5 space-y-4">
          <h2 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
            <Sliders className="w-4 h-4 text-purple-400" />
            <span>Synthetic Dataset Generator</span>
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-end">
            <div className="space-y-1">
              <label className="text-xs text-slate-300">Total Samples: {sampleCount}</label>
              <input
                type="range"
                min="50"
                max="1000"
                step="50"
                value={sampleCount}
                onChange={(e) => setSampleCount(parseInt(e.target.value))}
                className="w-full accent-purple-500 h-1.5 bg-dark-800 rounded cursor-pointer"
              />
            </div>

            <div className="space-y-1">
              <label className="text-xs text-slate-300">
                Defect Ratio: {Math.round(defectRate * 100)}%
              </label>
              <input
                type="range"
                min="0.0"
                max="0.5"
                step="0.05"
                value={defectRate}
                onChange={(e) => setDefectRate(parseFloat(e.target.value))}
                className="w-full accent-purple-500 h-1.5 bg-dark-800 rounded cursor-pointer"
              />
            </div>

            <div>
              <button
                onClick={handleGenerateDataset}
                disabled={isGenerating}
                className="w-full py-2 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white rounded-lg text-xs font-semibold flex items-center justify-center gap-2 transition-all disabled:opacity-50"
              >
                {isGenerating ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    <span>Synthesizing Records...</span>
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4" />
                    <span>Generate New Dataset</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Sample Explorer Gallery & Details Inspector */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Gallery grid */}
          <div className="lg:col-span-2 bg-dark-900 rounded-xl border border-dark-750 p-5 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-slate-200">Dataset Samples Gallery</h3>
              <span className="text-xs text-slate-500">{samples.length} items shown</span>
            </div>

            <div className="grid grid-cols-4 sm:grid-cols-6 gap-2.5 max-h-[460px] overflow-y-auto custom-scrollbar p-1">
              {samples.map((samp) => {
                const isSel = selectedSample?.id === samp.id;
                return (
                  <div
                    key={samp.id}
                    onClick={() => setSelectedSample(samp)}
                    className={`aspect-square rounded-lg p-2 border cursor-pointer relative group transition-all flex flex-col items-center justify-center ${
                      isSel
                        ? 'bg-purple-950/40 border-purple-500 ring-2 ring-purple-500/40'
                        : 'bg-dark-950 border-dark-750 hover:border-dark-600'
                    }`}
                  >
                    <img
                      src={samp.image_base64}
                      alt={samp.id}
                      className="w-12 h-12 object-contain"
                      style={{ imageRendering: 'pixelated' }}
                    />
                    <span className="text-[9px] font-mono text-slate-400 mt-1 capitalize truncate w-full text-center">
                      {samp.category}
                    </span>
                    {samp.has_defect && (
                      <span className="absolute top-1 right-1 w-2 h-2 rounded-full bg-amber-400 ring-2 ring-dark-950"></span>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Selected Record Detail Panel */}
          <div className="bg-dark-900 rounded-xl border border-dark-750 p-5 space-y-4">
            <h3 className="text-sm font-semibold text-slate-200">Sample Inspector</h3>

            {selectedSample ? (
              <div className="space-y-4 text-xs">
                {/* Visual preview */}
                <div className="flex items-center gap-4 bg-dark-950 p-3 rounded-lg border border-dark-750">
                  <div className="w-16 h-16 bg-dark-900 rounded border border-dark-750 p-1 flex items-center justify-center">
                    <img
                      src={selectedSample.image_base64}
                      alt=""
                      className="w-full h-full object-contain"
                      style={{ imageRendering: 'pixelated' }}
                    />
                  </div>
                  <div>
                    <div className="font-bold text-slate-100 font-mono">{selectedSample.id}</div>
                    <div className="text-slate-400 capitalize">{selectedSample.category} • {selectedSample.pose}</div>
                    <span className="inline-block mt-1 px-1.5 py-0.5 rounded bg-dark-800 text-[10px] font-mono text-cyan-300">
                      Split: {selectedSample.split}
                    </span>
                  </div>
                </div>

                {/* Defect tag if any */}
                {selectedSample.has_defect ? (
                  <div className="bg-amber-950/40 border border-amber-500/30 p-2.5 rounded-lg space-y-1">
                    <div className="text-amber-400 font-semibold flex items-center gap-1.5">
                      <AlertTriangle className="w-3.5 h-3.5" />
                      <span>Defect: {selectedSample.defect_info?.defect_type}</span>
                    </div>
                    <p className="text-slate-300 text-[11px]">{selectedSample.defect_info?.description}</p>
                    <p className="text-purple-300 font-mono text-[10px]">Fix: {selectedSample.defect_info?.suggested_fix}</p>
                  </div>
                ) : (
                  <div className="bg-emerald-950/40 border border-emerald-500/30 p-2 rounded-lg text-emerald-300 flex items-center gap-2">
                    <CheckCircle2 className="w-4 h-4" />
                    <span>Clean Ground Truth Record</span>
                  </div>
                )}

                {/* PixelScript */}
                <div className="space-y-1">
                  <span className="font-medium text-slate-300">Ground Truth PixelScript</span>
                  <pre className="bg-dark-950 p-3 rounded-lg border border-dark-750 font-mono text-[11px] text-cyan-300 max-h-48 overflow-y-auto custom-scrollbar">
                    {selectedSample.script_text}
                  </pre>
                </div>
              </div>
            ) : (
              <p className="text-slate-500 text-xs italic">Select a sample to inspect</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
