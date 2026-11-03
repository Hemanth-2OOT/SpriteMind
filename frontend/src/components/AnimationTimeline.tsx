import React, { useState, useEffect } from 'react';
import {
  Play,
  Pause,
  Repeat,
  Plus,
  Copy,
  Trash2,
  Download,
  SkipBack,
  SkipForward,
  Layers,
  ChevronRight,
} from 'lucide-react';
import { FrameData } from '../types';

interface AnimationTimelineProps {
  frames: FrameData[];
  currentFrameIndex: number;
  onSelectFrame: (index: number) => void;
  onAddFrame: () => void;
  onDuplicateFrame: (index: number) => void;
  onDeleteFrame: (index: number) => void;
  onExport: (format: 'png' | 'spritesheet' | 'gif' | 'json') => void;
  onionSkinEnabled: boolean;
  setOnionSkinEnabled: (enabled: boolean) => void;
}

export const AnimationTimeline: React.FC<AnimationTimelineProps> = ({
  frames,
  currentFrameIndex,
  onSelectFrame,
  onAddFrame,
  onDuplicateFrame,
  onDeleteFrame,
  onExport,
  onionSkinEnabled,
  setOnionSkinEnabled,
}) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLooping, setIsLooping] = useState(true);
  const [fps, setFps] = useState(6);
  const [showExportModal, setShowExportModal] = useState(false);

  // Playback loop
  useEffect(() => {
    if (!isPlaying || frames.length <= 1) return;

    const interval = 1000 / fps;
    const timer = setInterval(() => {
      onSelectFrame((currentFrameIndex + 1) % frames.length);
    }, interval);

    return () => clearInterval(timer);
  }, [isPlaying, fps, currentFrameIndex, frames.length, onSelectFrame]);

  return (
    <div className="h-32 bg-dark-900 border-t border-dark-750 flex flex-col select-none z-20">
      {/* Top timeline controls bar */}
      <div className="h-9 px-4 bg-dark-950/60 border-b border-dark-750 flex items-center justify-between text-xs">
        {/* Playback Controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className={`p-1.5 rounded-md font-medium flex items-center gap-1 transition-colors ${
              isPlaying
                ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                : 'bg-purple-600 hover:bg-purple-500 text-white'
            }`}
            title={isPlaying ? 'Pause' : 'Play'}
          >
            {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5 fill-current" />}
          </button>

          <button
            onClick={() => onSelectFrame(Math.max(0, currentFrameIndex - 1))}
            className="p-1.5 text-slate-400 hover:text-slate-200 hover:bg-dark-800 rounded transition-colors"
            title="Previous Frame"
          >
            <SkipBack className="w-3.5 h-3.5" />
          </button>

          <button
            onClick={() => onSelectFrame((currentFrameIndex + 1) % Math.max(1, frames.length))}
            className="p-1.5 text-slate-400 hover:text-slate-200 hover:bg-dark-800 rounded transition-colors"
            title="Next Frame"
          >
            <SkipForward className="w-3.5 h-3.5" />
          </button>

          <button
            onClick={() => setIsLooping(!isLooping)}
            className={`p-1.5 rounded transition-colors ${
              isLooping ? 'text-purple-400 bg-purple-950/40' : 'text-slate-500 hover:text-slate-300'
            }`}
            title="Toggle Loop"
          >
            <Repeat className="w-3.5 h-3.5" />
          </button>

          {/* FPS Slider */}
          <div className="flex items-center gap-1.5 ml-3 pl-3 border-l border-dark-800">
            <span className="text-[10px] text-slate-400 font-mono">FPS: {fps}</span>
            <input
              type="range"
              min="1"
              max="24"
              value={fps}
              onChange={(e) => setFps(parseInt(e.target.value))}
              className="w-16 accent-purple-500 h-1 bg-dark-800 rounded cursor-pointer"
            />
          </div>

          {/* Onion Skinning Toggle */}
          <button
            onClick={() => setOnionSkinEnabled(!onionSkinEnabled)}
            className={`flex items-center gap-1 px-2 py-1 rounded border text-[11px] ml-2 transition-colors ${
              onionSkinEnabled
                ? 'bg-purple-950/60 border-purple-500/50 text-purple-300'
                : 'bg-dark-850 border-dark-750 text-slate-400 hover:text-slate-200'
            }`}
          >
            <Layers className="w-3 h-3" />
            <span>Onion Skin</span>
          </button>
        </div>

        {/* Frame Actions & Export */}
        <div className="flex items-center gap-2">
          <button
            onClick={onAddFrame}
            className="p-1 text-slate-300 hover:text-white hover:bg-dark-800 rounded border border-dark-750 flex items-center gap-1 px-2 text-[11px] transition-colors"
          >
            <Plus className="w-3 h-3" />
            <span>Add Frame</span>
          </button>

          <button
            onClick={() => onDuplicateFrame(currentFrameIndex)}
            className="p-1 text-slate-300 hover:text-white hover:bg-dark-800 rounded border border-dark-750 flex items-center gap-1 px-2 text-[11px] transition-colors"
            title="Duplicate Frame"
          >
            <Copy className="w-3 h-3" />
            <span>Duplicate</span>
          </button>

          {frames.length > 1 && (
            <button
              onClick={() => onDeleteFrame(currentFrameIndex)}
              className="p-1 text-slate-400 hover:text-rose-400 hover:bg-dark-800 rounded border border-dark-750 transition-colors"
              title="Delete Frame"
            >
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          )}

          {/* Export Dropdown */}
          <div className="relative">
            <button
              onClick={() => setShowExportModal(!showExportModal)}
              className="px-2.5 py-1 bg-dark-800 hover:bg-dark-750 text-purple-300 rounded border border-purple-500/30 flex items-center gap-1.5 text-xs font-medium transition-colors"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Export</span>
            </button>

            {showExportModal && (
              <div className="absolute right-0 bottom-full mb-1 w-44 bg-dark-900 border border-dark-700 rounded-lg shadow-2xl p-1 z-50 text-xs">
                <button
                  onClick={() => { onExport('png'); setShowExportModal(false); }}
                  className="w-full text-left px-2.5 py-1.5 hover:bg-dark-800 text-slate-200 rounded"
                >
                  PNG Sprite (Current)
                </button>
                <button
                  onClick={() => { onExport('spritesheet'); setShowExportModal(false); }}
                  className="w-full text-left px-2.5 py-1.5 hover:bg-dark-800 text-slate-200 rounded"
                >
                  Spritesheet Grid
                </button>
                <button
                  onClick={() => { onExport('gif'); setShowExportModal(false); }}
                  className="w-full text-left px-2.5 py-1.5 hover:bg-dark-800 text-slate-200 rounded"
                >
                  Animated GIF
                </button>
                <button
                  onClick={() => { onExport('json'); setShowExportModal(false); }}
                  className="w-full text-left px-2.5 py-1.5 hover:bg-dark-800 text-slate-200 rounded"
                >
                  JSON PixelScript Bundle
                </button>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Frame Strip Thumbnails */}
      <div className="flex-1 flex items-center gap-2 px-4 overflow-x-auto custom-scrollbar py-2">
        {frames.map((frame, idx) => {
          const isSelected = idx === currentFrameIndex;
          return (
            <div
              key={idx}
              onClick={() => onSelectFrame(idx)}
              className={`flex-shrink-0 w-16 h-16 rounded-md p-1 border cursor-pointer relative group transition-all ${
                isSelected
                  ? 'border-purple-500 bg-purple-950/30 ring-2 ring-purple-500/40'
                  : 'border-dark-750 bg-dark-950/80 hover:border-dark-600'
              }`}
            >
              <div className="w-full h-full flex items-center justify-center overflow-hidden">
                <img
                  src={frame.image_base64}
                  alt={`Frame ${idx + 1}`}
                  className="w-full h-full object-contain"
                  style={{ imageRendering: 'pixelated' }}
                />
              </div>

              {/* Frame number badge */}
              <div className="absolute top-1 left-1 px-1 rounded bg-black/70 text-[9px] font-mono text-slate-300">
                #{idx + 1}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
