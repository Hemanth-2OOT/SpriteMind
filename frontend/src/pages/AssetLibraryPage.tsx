import React, { useState, useEffect } from 'react';
import {
  FolderArchive,
  Search,
  Download,
  Trash2,
  Calendar,
  Layers,
  Sparkles,
  ExternalLink,
} from 'lucide-react';
import { AssetRecord } from '../types';
import { fetchAssets, deleteAsset, exportAsset } from '../services/api';
import { SPRITE_SIZE } from '../config/sprite';

export const AssetLibraryPage: React.FC = () => {
  const [assets, setAssets] = useState<AssetRecord[]>([]);
  const [query, setQuery] = useState('');
  const [selectedAsset, setSelectedAsset] = useState<AssetRecord | null>(null);

  const loadAssets = async () => {
    try {
      const data = await fetchAssets(query);
      setAssets(data);
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    loadAssets();
  }, [query]);

  const handleDelete = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm('Are you sure you want to delete this asset?')) return;
    try {
      await deleteAsset(id);
      loadAssets();
      if (selectedAsset?.id === id) {
        setSelectedAsset(null);
      }
    } catch (e: any) {
      alert(`Delete failed: ${e.message}`);
    }
  };

  const handleExport = async (asset: AssetRecord, format: 'png' | 'spritesheet' | 'gif') => {
    try {
      const res = await exportAsset({
        frames_scripts: asset.frames.map((f) => f.script_text),
        width: SPRITE_SIZE,
        height: SPRITE_SIZE,
        palette: 'pico8',
        format,
        scale: 4,
        fps: 8,
      });

      if (res.image_base64) {
        const a = document.createElement('a');
        a.href = res.image_base64;
        a.download = `${asset.name}_${format}.${format === 'gif' ? 'gif' : 'png'}`;
        a.click();
      }
    } catch (e: any) {
      alert(`Export failed: ${e.message}`);
    }
  };

  return (
    <div className="flex-1 flex flex-col bg-dark-950 p-6 overflow-y-auto custom-scrollbar select-none">
      <div className="max-w-6xl w-full mx-auto space-y-6">
        {/* Header & Search */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-dark-750">
          <div>
            <h1 className="text-xl font-bold text-white flex items-center gap-2">
              <FolderArchive className="w-5 h-5 text-purple-400" />
              <span>Asset Library</span>
            </h1>
            <p className="text-xs text-slate-400 mt-0.5">
              Saved character sprites, animations, and game-ready spritesheets
            </p>
          </div>

          <div className="relative w-full sm:w-64">
            <Search className="w-4 h-4 text-slate-500 absolute left-3 top-2.5" />
            <input
              type="text"
              placeholder="Search assets..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="w-full bg-dark-900 border border-dark-750 rounded-lg pl-9 pr-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-purple-500"
            />
          </div>
        </div>

        {/* Assets Grid */}
        {assets.length > 0 ? (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
            {assets.map((asset) => {
              const thumb = asset.frames?.[0]?.image_base64;
              return (
                <div
                  key={asset.id}
                  onClick={() => setSelectedAsset(asset)}
                  className="bg-dark-900 rounded-xl border border-dark-750 hover:border-purple-500/50 p-3 flex flex-col justify-between cursor-pointer group transition-all"
                >
                  <div className="aspect-square bg-dark-950 rounded-lg p-2 border border-dark-800 flex items-center justify-center relative overflow-hidden mb-2">
                    {thumb ? (
                      <img
                        src={thumb}
                        alt={asset.name}
                        className="w-20 h-20 object-contain group-hover:scale-110 transition-transform"
                        style={{ imageRendering: 'pixelated' }}
                      />
                    ) : (
                      <Sparkles className="w-8 h-8 text-dark-700" />
                    )}

                    <div className="absolute top-1.5 right-1.5 flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                      <button
                        onClick={(e) => handleDelete(asset.id, e)}
                        className="p-1 bg-rose-950/80 hover:bg-rose-900 text-rose-300 rounded border border-rose-800"
                        title="Delete asset"
                      >
                        <Trash2 className="w-3 h-3" />
                      </button>
                    </div>
                  </div>

                  <div>
                    <h3 className="text-xs font-semibold text-slate-200 truncate">{asset.name}</h3>
                    <div className="flex items-center justify-between text-[10px] text-slate-500 mt-0.5">
                      <span className="capitalize">{asset.category}</span>
                      <span>{asset.frames?.length || 1} frames</span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="p-12 text-center bg-dark-900/40 rounded-xl border border-dark-800">
            <FolderArchive className="w-10 h-10 text-dark-700 mx-auto mb-2" />
            <h3 className="text-sm font-medium text-slate-300">No Assets Found</h3>
            <p className="text-xs text-slate-500 mt-1">Export sprites or animations from the Generate page to save them here</p>
          </div>
        )}

        {/* Selected Asset Inspection Modal */}
        {selectedAsset && (
          <div className="fixed inset-0 bg-black/75 backdrop-blur-sm flex items-center justify-center p-4 z-50">
            <div className="bg-dark-900 rounded-xl border border-dark-750 max-w-lg w-full p-6 space-y-4 shadow-2xl">
              <div className="flex items-center justify-between pb-3 border-b border-dark-750">
                <h3 className="text-sm font-bold text-white">{selectedAsset.name}</h3>
                <button
                  onClick={() => setSelectedAsset(null)}
                  className="text-slate-400 hover:text-white text-xs px-2 py-1 rounded bg-dark-800"
                >
                  ✕ Close
                </button>
              </div>

              <div className="flex items-center justify-center p-6 bg-dark-950 rounded-xl border border-dark-750">
                <img
                  src={selectedAsset.frames?.[0]?.image_base64}
                  alt=""
                  className="w-36 h-36 object-contain"
                  style={{ imageRendering: 'pixelated' }}
                />
              </div>

              <div className="grid grid-cols-2 gap-3 text-xs bg-dark-950 p-3 rounded-lg border border-dark-750 font-mono">
                <div>
                  <span className="text-slate-500">Category:</span>
                  <p className="text-slate-200 capitalize">{selectedAsset.category}</p>
                </div>
                <div>
                  <span className="text-slate-500">Created:</span>
                  <p className="text-slate-200">{selectedAsset.created_at}</p>
                </div>
                <div>
                  <span className="text-slate-500">Total Frames:</span>
                  <p className="text-purple-400">{selectedAsset.frames?.length || 1}</p>
                </div>
                <div>
                  <span className="text-slate-500">Dimensions:</span>
                  <p className="text-cyan-400">{SPRITE_SIZE}×{SPRITE_SIZE} px</p>
                </div>
              </div>

              {/* Action buttons */}
              <div className="flex gap-2 pt-2">
                <button
                  onClick={() => handleExport(selectedAsset, 'png')}
                  className="flex-1 py-2 bg-dark-800 hover:bg-dark-750 text-slate-200 rounded text-xs font-medium border border-dark-700 transition-colors"
                >
                  PNG
                </button>
                <button
                  onClick={() => handleExport(selectedAsset, 'spritesheet')}
                  className="flex-1 py-2 bg-purple-600 hover:bg-purple-500 text-white rounded text-xs font-medium transition-colors"
                >
                  Spritesheet
                </button>
                <button
                  onClick={() => handleExport(selectedAsset, 'gif')}
                  className="flex-1 py-2 bg-dark-800 hover:bg-dark-750 text-slate-200 rounded text-xs font-medium border border-dark-700 transition-colors"
                >
                  GIF
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
