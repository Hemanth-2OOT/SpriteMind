import React, { useState } from 'react';
import {
  Wand2,
  Sparkles,
  Dice5,
  Trash2,
  RefreshCw,
  Zap,
  Sliders,
  ChevronDown,
} from 'lucide-react';
import { GenerationResult } from '../types';

interface GenerationSidebarProps {
  onGenerate: (params: any) => Promise<void>;
  onRefine: () => Promise<void>;
  onClear: () => void;
  isLoading: boolean;
  selectedEngine: string;
}

const PROMPT_PRESETS = [
  { label: 'Zombie', prompt: 'Create a 32x32 green zombie with red eyes, a torn brown shirt, and a front-facing idle pose.' },
  { label: 'Knight', prompt: 'A 32x32 brave knight with steel plate armor, glowing sword, and classic 16-bit aesthetic.' },
  { label: 'Rogue', prompt: 'A 32x32 nimble rogue with dark hood, yellow eyes, leather belt, and dual daggers.' },
  { label: 'Wizard', prompt: 'A 32x32 arcane wizard in deep blue robes holding a mystical wooden staff.' },
  { label: 'Robot', prompt: 'A 32x32 retro cybernetic android with metallic chassis and bright cyan optical sensors.' },
  { label: 'Slime', prompt: 'A 32x32 bouncy acidic lime slime with glossy highlights.' },
];

export const GenerationSidebar: React.FC<GenerationSidebarProps> = ({
  onGenerate,
  onRefine,
  onClear,
  isLoading,
  selectedEngine,
}) => {
  const [prompt, setPrompt] = useState('Create a 32x32 green zombie with red eyes, a torn brown shirt, and a front-facing idle pose.');
  const [negativePrompt, setNegativePrompt] = useState('Blurry pixels, inconsistent outlines, excessive colors, disconnected limbs');
  const [category, setCategory] = useState('zombie');
  const [resolution, setResolution] = useState<number>(32);
  const [style, setStyle] = useState('classic_16bit');
  const [perspective, setPerspective] = useState('front');
  const [pose, setPose] = useState('idle');
  const [palette, setPalette] = useState('pico8');
  const [numFrames, setNumFrames] = useState<number>(1);
  const [seed, setSeed] = useState<number>(42);
  const [maxActions, setMaxActions] = useState<number>(64);
  const [bestOf, setBestOf] = useState<number>(4);
  const [temperature, setTemperature] = useState<number>(0.9);
  const [topK, setTopK] = useState<number>(0);

  const samplingDisabled = selectedEngine === 'procedural_baseline';

  const handleRollDice = () => {
    setSeed(Math.floor(Math.random() * 999999));
  };

  const handleApplyPreset = (presetPrompt: string, presetCat: string) => {
    setPrompt(presetPrompt);
    setCategory(presetCat.toLowerCase());
  };

  const triggerGenerate = () => {
    onGenerate({
      prompt,
      negative_prompt: negativePrompt,
      category,
      style,
      pose,
      palette,
      width: resolution,
      height: resolution,
      num_frames: numFrames,
      engine: selectedEngine,
      seed,
      max_actions: maxActions,
      best_of: bestOf,
      temperature,
      top_k: topK,
    });
  };

  return (
    <aside className="w-80 bg-dark-900 border-r border-dark-750 flex flex-col h-full overflow-y-auto custom-scrollbar select-none">
      <div className="p-4 space-y-4">
        {/* Header */}
        <div className="flex items-center justify-between pb-2 border-b border-dark-750">
          <div className="flex items-center gap-2">
            <Sliders className="w-4 h-4 text-purple-400" />
            <h2 className="text-xs font-semibold text-slate-200 tracking-wider uppercase">
              Generation Studio
            </h2>
          </div>
          <span className="text-[10px] text-purple-400 font-mono bg-purple-950/50 px-2 py-0.5 rounded border border-purple-800/40">
            SVAN 1.0
          </span>
        </div>

        {/* Prompt Input & Presets */}
        <div className="space-y-1.5">
          <div className="flex items-center justify-between">
            <label className="text-xs font-medium text-slate-300">Prompt</label>
            <span className="text-[10px] text-slate-500 font-mono">{prompt.length}/200</span>
          </div>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={3}
            className="w-full bg-dark-950 border border-dark-750 rounded-lg p-2.5 text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:border-purple-500 focus:ring-1 focus:ring-purple-500 transition-all resize-none"
            placeholder="Describe character details, clothing, weapons, colors..."
          />

          {/* Quick presets */}
          <div className="flex flex-wrap gap-1 pt-1">
            {PROMPT_PRESETS.map((p) => (
              <button
                key={p.label}
                onClick={() => handleApplyPreset(p.prompt, p.label)}
                className="text-[10px] px-2 py-0.5 rounded bg-dark-800 text-slate-400 hover:text-purple-300 hover:bg-dark-750 border border-dark-700/60 transition-colors"
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>

        {/* Negative Constraints */}
        <div className="space-y-1">
          <label className="text-xs font-medium text-slate-400">Negative Constraints</label>
          <input
            type="text"
            value={negativePrompt}
            onChange={(e) => setNegativePrompt(e.target.value)}
            className="w-full bg-dark-950 border border-dark-750 rounded-md px-2.5 py-1.5 text-xs text-slate-300 focus:outline-none focus:border-purple-500"
          />
        </div>

        {/* Category & Resolution Grid */}
        <div className="grid grid-cols-2 gap-2.5">
          <div className="space-y-1">
            <label className="text-xs font-medium text-slate-300">Category</label>
            <select
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              className="w-full bg-dark-950 border border-dark-750 rounded-md p-1.5 text-xs text-slate-200 focus:outline-none focus:border-purple-500"
            >
              <option value="knight">Knight</option>
              <option value="zombie">Zombie</option>
              <option value="rogue">Rogue</option>
              <option value="wizard">Wizard</option>
              <option value="robot">Robot</option>
              <option value="slime">Slime</option>
              <option value="skeleton">Skeleton</option>
              <option value="goblin">Goblin</option>
            </select>
          </div>

          <div className="space-y-1">
            <label className="text-xs font-medium text-slate-300">Resolution</label>
            <div className="grid grid-cols-3 gap-1 bg-dark-950 p-1 rounded-md border border-dark-750">
              {[16, 32, 64].map((res) => (
                <button
                  key={res}
                  onClick={() => setResolution(res)}
                  className={`py-1 text-[10px] font-mono rounded transition-colors ${
                    resolution === res ? 'bg-purple-600 text-white font-bold' : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {res}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Style & Palette */}
        <div className="grid grid-cols-2 gap-2.5">
          <div className="space-y-1">
            <label className="text-xs font-medium text-slate-300">Visual Style</label>
            <select
              value={style}
              onChange={(e) => setStyle(e.target.value)}
              className="w-full bg-dark-950 border border-dark-750 rounded-md p-1.5 text-xs text-slate-200 focus:outline-none"
            >
              <option value="classic_16bit">Classic 16-bit</option>
              <option value="retro_8bit">Retro 8-bit</option>
              <option value="flat">Flat Modern</option>
              <option value="dark_fantasy">Dark Fantasy</option>
              <option value="cyberpunk">Cyberpunk</option>
            </select>
          </div>

          <div className="space-y-1">
            <label className="text-xs font-medium text-slate-300">Palette</label>
            <select
              value={palette}
              onChange={(e) => setPalette(e.target.value)}
              className="w-full bg-dark-950 border border-dark-750 rounded-md p-1.5 text-xs text-slate-200 focus:outline-none"
            >
              <option value="pico8">PICO-8 (16)</option>
              <option value="dawnbringer16">DawnBringer (16)</option>
              <option value="gameboy">GameBoy (4)</option>
              <option value="cyberpunk">Cyberpunk Neon</option>
              <option value="endesga32">Endesga 32</option>
            </select>
          </div>
        </div>

        {/* Pose & Animation Type */}
        <div className="grid grid-cols-2 gap-2.5">
          <div className="space-y-1">
            <label className="text-xs font-medium text-slate-300">Pose</label>
            <select
              value={pose}
              onChange={(e) => setPose(e.target.value)}
              className="w-full bg-dark-950 border border-dark-750 rounded-md p-1.5 text-xs text-slate-200 focus:outline-none"
            >
              <option value="idle">Idle (Breathe)</option>
              <option value="walk">Walk Cycle</option>
              <option value="run">Running</option>
              <option value="attack">Attack Slash</option>
              <option value="cast">Spell Cast</option>
              <option value="dash">Dash Burst</option>
              <option value="death">Death / Fall</option>
            </select>
          </div>

          <div className="space-y-1">
            <label className="text-xs font-medium text-slate-300">Frames</label>
            <div className="flex items-center gap-2">
              <input
                type="range"
                min="1"
                max="8"
                value={numFrames}
                onChange={(e) => setNumFrames(parseInt(e.target.value))}
                className="flex-1 accent-purple-500 h-1.5 bg-dark-800 rounded-lg cursor-pointer"
              />
              <span className="text-xs font-mono text-purple-300 w-4 text-center">{numFrames}</span>
            </div>
          </div>
        </div>

        {/* Seed & Max Actions */}
        <div className="space-y-2 pt-1 border-t border-dark-750/60">
          <div className="flex items-center justify-between">
            <label className="text-xs font-medium text-slate-300">Random Seed</label>
            <div className="flex items-center gap-1.5">
              <input
                type="number"
                value={seed}
                onChange={(e) => setSeed(parseInt(e.target.value) || 0)}
                className="w-24 bg-dark-950 border border-dark-750 rounded px-2 py-0.5 text-xs text-slate-200 font-mono text-right"
              />
              <button
                onClick={handleRollDice}
                className="p-1 bg-dark-800 hover:bg-dark-700 text-slate-300 rounded border border-dark-750 transition-colors"
                title="Randomize Seed"
              >
                <Dice5 className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>

          <div className="space-y-1">
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-400">Max Actions / Complexity</span>
              <span className="text-slate-300 font-mono">{maxActions}</span>
            </div>
            <input
              type="range"
              min="16"
              max="128"
              step="8"
              value={maxActions}
              onChange={(e) => setMaxActions(parseInt(e.target.value))}
              className="w-full accent-purple-500 h-1.5 bg-dark-800 rounded cursor-pointer"
            />
          </div>
        </div>

        {/* Sampler Quality */}
        <div className={`space-y-2 pt-1 border-t border-dark-750/60 ${samplingDisabled ? 'opacity-40 pointer-events-none' : ''}`}>
          <div className="flex items-center justify-between">
            <label className="text-xs font-medium text-slate-300">Sampler</label>
            <span className="text-[10px] text-slate-500">
              {samplingDisabled ? 'neural engines only' : 'trade speed for quality'}
            </span>
          </div>

          <div className="space-y-1">
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-400">Best of</span>
              <span className="text-slate-300 font-mono">{bestOf}</span>
            </div>
            <input
              type="range"
              min="1"
              max="8"
              step="1"
              value={bestOf}
              onChange={(e) => setBestOf(parseInt(e.target.value))}
              className="w-full accent-purple-500 h-1.5 bg-dark-800 rounded cursor-pointer"
            />
          </div>

          <div className="space-y-1">
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-400">Temperature</span>
              <span className="text-slate-300 font-mono">{temperature.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="2"
              step="0.05"
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
              className="w-full accent-purple-500 h-1.5 bg-dark-800 rounded cursor-pointer"
            />
          </div>

          <div className="space-y-1">
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-400">Top-k</span>
              <span className="text-slate-300 font-mono">{topK === 0 ? 'off' : topK}</span>
            </div>
            <input
              type="range"
              min="0"
              max="80"
              step="5"
              value={topK}
              onChange={(e) => setTopK(parseInt(e.target.value))}
              className="w-full accent-purple-500 h-1.5 bg-dark-800 rounded cursor-pointer"
            />
          </div>
        </div>

        {/* Action Buttons */}
        <div className="space-y-2 pt-2">
          <button
            onClick={triggerGenerate}
            disabled={isLoading}
            className={`w-full py-2.5 px-4 rounded-lg font-semibold text-xs flex items-center justify-center gap-2 shadow-lg transition-all ${
              isLoading
                ? 'bg-purple-800 text-purple-300 cursor-not-allowed opacity-75'
                : 'bg-gradient-to-r from-purple-600 via-indigo-600 to-purple-600 hover:from-purple-500 hover:to-indigo-500 text-white shadow-purple-600/25 active:scale-[0.98]'
            }`}
          >
            {isLoading ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                <span>Synthesizing Actions...</span>
              </>
            ) : (
              <>
                <Wand2 className="w-4 h-4" />
                <span>Generate Sprite</span>
              </>
            )}
          </button>

          <div className="grid grid-cols-2 gap-2">
            <button
              onClick={onRefine}
              disabled={isLoading}
              className="py-1.5 px-3 bg-dark-850 hover:bg-dark-800 text-slate-300 hover:text-white border border-dark-750 rounded-md text-xs font-medium flex items-center justify-center gap-1.5 transition-colors disabled:opacity-50"
            >
              <Zap className="w-3.5 h-3.5 text-amber-400" />
              <span>Refine / Critic</span>
            </button>
            <button
              onClick={onClear}
              className="py-1.5 px-3 bg-dark-850 hover:bg-dark-800 text-slate-400 hover:text-rose-400 border border-dark-750 rounded-md text-xs font-medium flex items-center justify-center gap-1.5 transition-colors"
            >
              <Trash2 className="w-3.5 h-3.5" />
              <span>Clear</span>
            </button>
          </div>
        </div>
      </div>
    </aside>
  );
};
