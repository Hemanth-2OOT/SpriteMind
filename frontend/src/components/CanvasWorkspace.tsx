import React, { useState, useRef, useEffect } from 'react';
import {
  ZoomIn,
  ZoomOut,
  Maximize2,
  Grid,
  Eye,
  Sliders,
  Download,
  Crosshair,
  ShieldAlert,
  GitCompare,
} from 'lucide-react';
import { FrameData, VisualCriticReport } from '../types';
import { SPRITE_SIZE } from '../config/sprite';

interface CanvasWorkspaceProps {
  currentFrame: FrameData | null;
  selectedViewPass: 'final' | 'grid' | 'segmentation' | 'silhouette' | 'edge_map';
  setSelectedViewPass: (pass: 'final' | 'grid' | 'segmentation' | 'silhouette' | 'edge_map') => void;
  showPixelGrid: boolean;
  setShowPixelGrid: (show: boolean) => void;
  onionSkinFrame?: FrameData | null;
  onionSkinOpacity?: number;
  criticReport?: VisualCriticReport | null;
  showDefectOverlay?: boolean;
  setShowDefectOverlay?: (show: boolean) => void;
  beforeImageBase64?: string | null;
  viewingBefore?: boolean;
  setViewingBefore?: (val: boolean) => void;
}

export const CanvasWorkspace: React.FC<CanvasWorkspaceProps> = ({
  currentFrame,
  selectedViewPass,
  setSelectedViewPass,
  showPixelGrid,
  setShowPixelGrid,
  onionSkinFrame,
  onionSkinOpacity = 0.3,
  criticReport,
  showDefectOverlay = true,
  setShowDefectOverlay,
  beforeImageBase64,
  viewingBefore = false,
  setViewingBefore,
}) => {
  const [zoom, setZoom] = useState<number>(12); // zoom scale factor (e.g. 12x)
  const [hoveredPixel, setHoveredPixel] = useState<{ x: number; y: number; part?: string } | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const canvasWidth = currentFrame?.width || SPRITE_SIZE;
  const canvasHeight = currentFrame?.height || SPRITE_SIZE;

  // Resolve active image based on before/after view or view pass
  let activeImageBase64 = viewingBefore && beforeImageBase64 ? beforeImageBase64 : currentFrame?.image_base64;
  if (!viewingBefore) {
    if (selectedViewPass === 'silhouette' && currentFrame?.passes?.silhouette) {
      activeImageBase64 = currentFrame.passes.silhouette;
    } else if (selectedViewPass === 'edge_map' && currentFrame?.passes?.edge_map) {
      activeImageBase64 = currentFrame.passes.edge_map;
    } else if (selectedViewPass === 'segmentation' && currentFrame?.passes?.segmentation) {
      activeImageBase64 = currentFrame.passes.segmentation;
    }
  }

  const defects = criticReport?.detected_defects || [];

  // Handle pixel hover detection
  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const relX = e.clientX - rect.left;
    const relY = e.clientY - rect.top;

    const px = Math.floor(relX / zoom);
    const py = Math.floor(relY / zoom);

    if (px >= 0 && px < canvasWidth && py >= 0 && py < canvasHeight) {
      // Find body part at pixel
      let foundPart: string | undefined = undefined;
      if (currentFrame?.parts) {
        for (const [pname, pinfo] of Object.entries(currentFrame.parts)) {
          const bb = pinfo.bounding_box;
          if (bb && px >= bb.min_x && px <= bb.max_x && py >= bb.min_y && py <= bb.max_y) {
            foundPart = pname;
            break;
          }
        }
      }
      setHoveredPixel({ x: px, y: py, part: foundPart });
    } else {
      setHoveredPixel(null);
    }
  };

  return (
    <div className="flex-1 flex flex-col bg-dark-950 relative overflow-hidden select-none">
      {/* Top toolbar */}
      <div className="h-11 bg-dark-900/90 border-b border-dark-750 px-4 flex items-center justify-between text-xs z-10 backdrop-blur-sm">
        {/* Pass view toggles */}
        <div className="flex items-center gap-1 bg-dark-950/80 p-0.5 rounded-lg border border-dark-800">
          <button
            onClick={() => setSelectedViewPass('final')}
            className={`px-2.5 py-1 rounded font-medium transition-colors ${
              selectedViewPass === 'final'
                ? 'bg-purple-600 text-white'
                : 'text-slate-400 hover:text-slate-200 hover:bg-dark-800'
            }`}
          >
            Final Artwork
          </button>
          <button
            onClick={() => setSelectedViewPass('segmentation')}
            className={`px-2.5 py-1 rounded font-medium transition-colors ${
              selectedViewPass === 'segmentation'
                ? 'bg-purple-600 text-white'
                : 'text-slate-400 hover:text-slate-200 hover:bg-dark-800'
            }`}
          >
            Body-part Segmentation
          </button>
          <button
            onClick={() => setSelectedViewPass('silhouette')}
            className={`px-2.5 py-1 rounded font-medium transition-colors ${
              selectedViewPass === 'silhouette'
                ? 'bg-purple-600 text-white'
                : 'text-slate-400 hover:text-slate-200 hover:bg-dark-800'
            }`}
          >
            Silhouette
          </button>
          <button
            onClick={() => setSelectedViewPass('edge_map')}
            className={`px-2.5 py-1 rounded font-medium transition-colors ${
              selectedViewPass === 'edge_map'
                ? 'bg-purple-600 text-white'
                : 'text-slate-400 hover:text-slate-200 hover:bg-dark-800'
            }`}
          >
            Edge Map
          </button>
        </div>

        {/* View & Zoom Controls */}
        <div className="flex items-center gap-2">
          {/* Before/After toggle if before image available */}
          {beforeImageBase64 && setViewingBefore && (
            <button
              onClick={() => setViewingBefore(!viewingBefore)}
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded border transition-colors ${
                viewingBefore
                  ? 'bg-amber-950/70 border-amber-500/60 text-amber-300 font-semibold'
                  : 'bg-emerald-950/60 border-emerald-500/60 text-emerald-300 font-semibold'
              }`}
              title="Toggle Before / After Sprite View"
            >
              <GitCompare className="w-3.5 h-3.5" />
              <span>{viewingBefore ? 'Viewing: Before (Corrupted)' : 'Viewing: Repaired'}</span>
            </button>
          )}

          {/* Critic Defect Overlay Toggle */}
          {setShowDefectOverlay && (
            <button
              onClick={() => setShowDefectOverlay(!showDefectOverlay)}
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded border transition-colors ${
                showDefectOverlay
                  ? 'bg-rose-950/60 border-rose-500/60 text-rose-300'
                  : 'bg-dark-850 border-dark-750 text-slate-400 hover:text-slate-200'
              }`}
              title="Toggle Visual Critic Defect Overlay"
            >
              <ShieldAlert className="w-3.5 h-3.5 text-rose-400" />
              <span>Defects ({defects.length})</span>
            </button>
          )}

          {/* Grid toggle */}
          <button
            onClick={() => setShowPixelGrid(!showPixelGrid)}
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded border transition-colors ${
              showPixelGrid
                ? 'bg-purple-950/50 border-purple-500/50 text-purple-300'
                : 'bg-dark-850 border-dark-750 text-slate-400 hover:text-slate-200'
            }`}
            title="Toggle pixel grid"
          >
            <Grid className="w-3.5 h-3.5" />
            <span>Grid</span>
          </button>

          {/* Zoom controls */}
          <div className="flex items-center gap-1 bg-dark-850 px-2 py-0.5 rounded border border-dark-750">
            <button
              onClick={() => setZoom(Math.max(4, zoom - 2))}
              className="p-1 text-slate-400 hover:text-slate-200 transition-colors"
              title="Zoom out"
            >
              <ZoomOut className="w-3.5 h-3.5" />
            </button>
            <span className="text-slate-300 font-mono w-12 text-center">{zoom * 100 / 16}%</span>
            <button
              onClick={() => setZoom(Math.min(24, zoom + 2))}
              className="p-1 text-slate-400 hover:text-slate-200 transition-colors"
              title="Zoom in"
            >
              <ZoomIn className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => setZoom(12)}
              className="p-1 text-slate-400 hover:text-slate-200 transition-colors border-l border-dark-700 ml-1"
              title="Reset Zoom"
            >
              <Maximize2 className="w-3 h-3" />
            </button>
          </div>
        </div>
      </div>

      {/* Main interactive canvas viewport */}
      <div
        ref={containerRef}
        className="flex-1 flex items-center justify-center relative p-8 overflow-auto"
        style={{
          backgroundImage: `
            linear-gradient(45deg, #121520 25%, transparent 25%),
            linear-gradient(-45deg, #121520 25%, transparent 25%),
            linear-gradient(45deg, transparent 75%, #121520 75%),
            linear-gradient(-45deg, transparent 75%, #121520 75%)
          `,
          backgroundSize: '16px 16px',
          backgroundPosition: '0 0, 0 8px, 8px -8px, -8px 0px',
        }}
      >
        {activeImageBase64 ? (
          <div
            className="relative shadow-2xl rounded-sm transition-all"
            style={{
              width: canvasWidth * zoom,
              height: canvasHeight * zoom,
              boxShadow: '0 0 40px rgba(0,0,0,0.8), 0 0 0 1px rgba(255,255,255,0.08)',
            }}
            onMouseMove={handleMouseMove}
            onMouseLeave={() => setHoveredPixel(null)}
          >
            {/* Onion skin frame preview if enabled */}
            {onionSkinFrame && (
              <img
                src={onionSkinFrame.image_base64}
                alt="Onion Skin"
                className="absolute inset-0 w-full h-full pointer-events-none"
                style={{
                  imageRendering: 'pixelated',
                  opacity: onionSkinOpacity,
                  filter: 'sepia(1) hue-rotate(180deg)',
                }}
              />
            )}

            {/* Primary active sprite image */}
            <img
              src={activeImageBase64}
              alt="Pixel Canvas"
              className="w-full h-full block"
              style={{
                imageRendering: 'pixelated',
              }}
            />

            {/* Optional SVG Pixel Grid */}
            {showPixelGrid && zoom >= 6 && (
              <svg
                className="absolute inset-0 w-full h-full pointer-events-none"
                viewBox={`0 0 ${canvasWidth} ${canvasHeight}`}
                style={{ shapeRendering: 'crispEdges' }}
              >
                <defs>
                  <pattern
                    id="pixel-grid-pattern"
                    width="1"
                    height="1"
                    patternUnits="userSpaceOnUse"
                  >
                    <path
                      d="M 1 0 L 0 0 0 1"
                      fill="none"
                      stroke="rgba(255, 255, 255, 0.12)"
                      strokeWidth="0.05"
                    />
                  </pattern>
                </defs>
                <rect width={canvasWidth} height={canvasHeight} fill="url(#pixel-grid-pattern)" />
              </svg>
            )}

            {/* Visual Critic Defect Overlay */}
            {showDefectOverlay && defects.length > 0 && (
              <svg
                className="absolute inset-0 w-full h-full pointer-events-none z-10"
                viewBox={`0 0 ${canvasWidth} ${canvasHeight}`}
              >
                {defects.map((def, i) => {
                  const bb = def.bbox;
                  const isCrit = def.severity === 'critical';
                  const isMod = def.severity === 'moderate';
                  const strokeCol = isCrit ? '#ef4444' : isMod ? '#f59e0b' : '#38bdf8';
                  const fillCol = isCrit ? 'rgba(239, 68, 68, 0.3)' : isMod ? 'rgba(245, 158, 11, 0.25)' : 'rgba(56, 189, 248, 0.2)';

                  return (
                    <g key={def.id || i}>
                      {bb && (
                        <rect
                          x={bb.min_x}
                          y={bb.min_y}
                          width={Math.max(1, bb.max_x - bb.min_x + 1)}
                          height={Math.max(1, bb.max_y - bb.min_y + 1)}
                          fill={fillCol}
                          stroke={strokeCol}
                          strokeWidth="0.2"
                          strokeDasharray={isCrit ? '0.4 0.2' : 'none'}
                        />
                      )}
                      {(def.affected_pixels || []).map(([px, py], pidx) => (
                        <rect
                          key={pidx}
                          x={px}
                          y={py}
                          width="1"
                          height="1"
                          fill="rgba(255, 255, 255, 0.5)"
                          stroke={strokeCol}
                          strokeWidth="0.12"
                        />
                      ))}
                    </g>
                  );
                })}
              </svg>
            )}

            {/* Hovered pixel highlight indicator */}
            {hoveredPixel && (
              <div
                className="absolute border border-cyan-400 bg-cyan-400/20 pointer-events-none z-20"
                style={{
                  left: hoveredPixel.x * zoom,
                  top: hoveredPixel.y * zoom,
                  width: zoom,
                  height: zoom,
                }}
              />
            )}
          </div>
        ) : (
          <div className="text-center p-8 border border-dashed border-dark-750 rounded-xl bg-dark-900/40">
            <Crosshair className="w-10 h-10 text-dark-600 mx-auto mb-2" />
            <p className="text-slate-400 text-sm font-medium">Ready to Generate</p>
            <p className="text-slate-500 text-xs mt-1">Configure prompt in left sidebar and click Generate</p>
          </div>
        )}
      </div>

      {/* Bottom status bar for coordinates and active pixel details */}
      <div className="h-7 bg-dark-900 border-t border-dark-750 px-4 flex items-center justify-between text-[11px] text-slate-400 font-mono">
        <div className="flex items-center gap-4">
          <span>Canvas: {canvasWidth}×{canvasHeight}px</span>
          {hoveredPixel ? (
            <span className="text-cyan-400">
              Pos: ({hoveredPixel.x}, {hoveredPixel.y}) {hoveredPixel.part ? `[Part: ${hoveredPixel.part}]` : ''}
            </span>
          ) : (
            <span className="text-slate-500">Hover over canvas to inspect pixels</span>
          )}
        </div>
        <div className="flex items-center gap-3">
          <span>Pixels: {currentFrame?.pixel_count || 0}</span>
          <span>Zoom: {zoom}x</span>
        </div>
      </div>
    </div>
  );
};
