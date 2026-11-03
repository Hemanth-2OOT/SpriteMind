import React from 'react';
import {
  Sparkles,
  Layers,
  Cpu,
  Activity,
  Film,
  Database,
  FlaskConical,
  BarChart3,
  FolderArchive,
  RefreshCw,
} from 'lucide-react';
import { SystemHealth, ModelsInfo, TrainingStatus } from '../types';

interface NavbarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  health: SystemHealth | null;
  modelsInfo: ModelsInfo | null;
  trainingStatus: TrainingStatus | null;
  selectedEngine: string;
  setSelectedEngine: (eng: string) => void;
  onRefreshHealth: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  health,
  modelsInfo,
  trainingStatus,
  selectedEngine,
  setSelectedEngine,
  onRefreshHealth,
}) => {
  const isTraining = trainingStatus?.state === 'training';

  return (
    <header className="h-14 bg-dark-900 border-b border-dark-750 px-4 flex items-center justify-between select-none z-30">
      {/* Brand logo */}
      <div className="flex items-center gap-6">
        <div className="flex items-center gap-2.5 cursor-pointer" onClick={() => setActiveTab('generate')}>
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-purple-600 via-indigo-600 to-cyan-500 p-0.5 shadow-lg shadow-purple-500/20">
            <div className="w-full h-full bg-dark-950 rounded-[7px] flex items-center justify-center">
              <div className="grid grid-cols-2 gap-0.5">
                <div className="w-1.5 h-1.5 bg-purple-400 rounded-sm"></div>
                <div className="w-1.5 h-1.5 bg-cyan-400 rounded-sm"></div>
                <div className="w-1.5 h-1.5 bg-cyan-400 rounded-sm"></div>
                <div className="w-1.5 h-1.5 bg-purple-400 rounded-sm"></div>
              </div>
            </div>
          </div>
          <div>
            <div className="flex items-center gap-1.5">
              <span className="font-bold text-base tracking-wide bg-gradient-to-r from-purple-300 via-indigo-200 to-cyan-300 bg-clip-text text-transparent">
                SpriteMind
              </span>
              <span className="text-[10px] px-1.5 py-0.2 bg-purple-500/10 text-purple-400 border border-purple-500/20 rounded font-mono">
                v0.1
              </span>
            </div>
            <p className="text-[10px] text-slate-400 -mt-0.5">Local Neural Pixel Engine</p>
          </div>
        </div>

        {/* Navigation tabs */}
        <nav className="flex items-center gap-1 bg-dark-950/60 p-1 rounded-lg border border-dark-800">
          <button
            onClick={() => setActiveTab('generate')}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
              activeTab === 'generate'
                ? 'bg-purple-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200 hover:bg-dark-800'
            }`}
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span>Generate</span>
          </button>

          <button
            onClick={() => setActiveTab('animation')}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
              activeTab === 'animation'
                ? 'bg-purple-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200 hover:bg-dark-800'
            }`}
          >
            <Film className="w-3.5 h-3.5" />
            <span>Animation Studio</span>
          </button>

          <button
            onClick={() => setActiveTab('dataset')}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
              activeTab === 'dataset'
                ? 'bg-purple-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200 hover:bg-dark-800'
            }`}
          >
            <Database className="w-3.5 h-3.5" />
            <span>Dataset Studio</span>
          </button>

          <button
            onClick={() => setActiveTab('model_lab')}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
              activeTab === 'model_lab'
                ? 'bg-purple-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200 hover:bg-dark-800'
            }`}
          >
            <FlaskConical className="w-3.5 h-3.5" />
            <span>Model Lab</span>
            {isTraining && (
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping"></span>
            )}
          </button>

          <button
            onClick={() => setActiveTab('evaluation')}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
              activeTab === 'evaluation'
                ? 'bg-purple-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200 hover:bg-dark-800'
            }`}
          >
            <BarChart3 className="w-3.5 h-3.5" />
            <span>Evaluation Lab</span>
          </button>

          <button
            onClick={() => setActiveTab('assets')}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
              activeTab === 'assets'
                ? 'bg-purple-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200 hover:bg-dark-800'
            }`}
          >
            <FolderArchive className="w-3.5 h-3.5" />
            <span>Asset Library</span>
          </button>

          <button
            onClick={() => setActiveTab('data_manager')}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
              activeTab === 'data_manager'
                ? 'bg-purple-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200 hover:bg-dark-800'
            }`}
          >
            <Database className="w-3.5 h-3.5" />
            <span>Data Manager</span>
          </button>
        </nav>
      </div>

      {/* Model & Device status indicators */}
      <div className="flex items-center gap-3">
        {/* Engine selector */}
        <div className="flex items-center gap-1.5 bg-dark-850 px-2.5 py-1 rounded-md border border-dark-750 text-xs">
          <span className="text-slate-400">Engine:</span>
          <select
            value={selectedEngine}
            onChange={(e) => setSelectedEngine(e.target.value)}
            className="bg-transparent text-slate-200 font-medium focus:outline-none cursor-pointer"
          >
            <option value="procedural_baseline" className="bg-dark-900 text-slate-200">
              Procedural Baseline
            </option>
            <option value="trained_model" className="bg-dark-900 text-slate-200">
              Trained SVAN Model
            </option>
            <option value="untrained_model" className="bg-dark-900 text-slate-200">
              Untrained Neural Model
            </option>
          </select>
        </div>

        {/* Training status */}
        <div
          onClick={() => setActiveTab('model_lab')}
          className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md border text-xs cursor-pointer transition-colors ${
            isTraining
              ? 'bg-emerald-950/40 border-emerald-500/40 text-emerald-300'
              : 'bg-dark-850 border-dark-750 text-slate-400 hover:border-dark-700'
          }`}
        >
          <Activity className={`w-3.5 h-3.5 ${isTraining ? 'text-emerald-400 animate-pulse' : 'text-slate-400'}`} />
          <span>{isTraining ? `Training (Ep ${trainingStatus?.epoch})` : 'Training: Idle'}</span>
        </div>

        {/* Device indicator */}
        <div className="flex items-center gap-1.5 bg-dark-850 px-2.5 py-1 rounded-md border border-dark-750 text-xs">
          <Cpu className="w-3.5 h-3.5 text-cyan-400" />
          <span className="text-slate-300 font-mono">
            {health?.cuda_available ? health.device : (health?.device || 'CPU')}
          </span>
          <button
            onClick={onRefreshHealth}
            title="Refresh System Status"
            className="text-slate-400 hover:text-slate-200 transition-colors ml-1"
          >
            <RefreshCw className="w-3 h-3" />
          </button>
        </div>
      </div>
    </header>
  );
};
