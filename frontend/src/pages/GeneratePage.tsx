import React, { useState } from 'react';
import { GenerationSidebar } from '../components/GenerationSidebar';
import { CanvasWorkspace } from '../components/CanvasWorkspace';
import { PropertiesSidebar } from '../components/PropertiesSidebar';
import { AnimationTimeline } from '../components/AnimationTimeline';
import { FrameData, VisualCriticReport, ModelsInfo, GenerationResult } from '../types';
import {
  generateSprite,
  refineSprite,
  renderScript,
  exportAsset,
  saveAsset,
  inspectSpriteApi,
  repairSpriteApi,
} from '../services/api';

interface GeneratePageProps {
  modelsInfo: ModelsInfo | null;
  selectedEngine: string;
}

// A rendered frame replaces the caller's copy wholesale, so the script that
// produced it travels with the request rather than being read back from the
// render response.
function frameWithScript(frame: FrameData, script: string): FrameData {
  return { ...frame, script_text: script };
}

export const GeneratePage: React.FC<GeneratePageProps> = ({
  modelsInfo,
  selectedEngine,
}) => {
  const [frames, setFrames] = useState<FrameData[]>([]);
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);
  const [criticReport, setCriticReport] = useState<VisualCriticReport | null>(null);
  const [selectedViewPass, setSelectedViewPass] = useState<'final' | 'grid' | 'segmentation' | 'silhouette' | 'edge_map'>('final');
  const [showPixelGrid, setShowPixelGrid] = useState(true);
  const [showDefectOverlay, setShowDefectOverlay] = useState(true);
  const [onionSkinEnabled, setOnionSkinEnabled] = useState(false);
  const [historyVersion, setHistoryVersion] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [isInspecting, setIsInspecting] = useState(false);
  const [isRepairing, setIsRepairing] = useState(false);
  const [beforeImageBase64, setBeforeImageBase64] = useState<string | null>(null);
  const [beforeScriptText, setBeforeScriptText] = useState<string | null>(null);
  const [viewingBefore, setViewingBefore] = useState(false);
  const [repairHistory, setRepairHistory] = useState<any[]>([]);
  const [inferenceTimeMs, setInferenceTimeMs] = useState(0);
  // Critic inspect/repair must run under the same category and palette the
  // sprite was generated with, otherwise repairs shift the colours.
  const [genContext, setGenContext] = useState<{ category: string; palette: string }>({
    category: 'zombie',
    palette: 'pico8',
  });

  // Initial auto-generation on first mount if empty
  React.useEffect(() => {
    if (frames.length === 0) {
      handleGenerate({
        category: 'zombie',
        style: 'classic_16bit',
        pose: 'idle',
        palette: 'pico8',
        width: 32,
        height: 32,
        num_frames: 4,
        engine: selectedEngine,
        seed: 42,
      });
    }
  }, []);

  const handleGenerate = async (params: any) => {
    setIsLoading(true);
    try {
      const res: GenerationResult = await generateSprite(params);
      setFrames(res.frames || []);
      setCurrentFrameIndex(0);
      setCriticReport(res.critic);
      setInferenceTimeMs(res.inference_time_ms);
      setGenContext({ category: res.category, palette: res.palette });
      setBeforeImageBase64(null);
      setBeforeScriptText(null);
      setViewingBefore(false);
      setRepairHistory([]);
      setHistoryVersion((version) => version + 1);
    } catch (e: any) {
      alert(`Generation Error: ${e.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleInspect = async () => {
    if (!currentFrame) return;
    setIsInspecting(true);
    try {
      const report = await inspectSpriteApi({
        script_text: currentFrame.script_text,
        category: genContext.category,
        palette_name: genContext.palette,
        width: currentFrame.width,
        height: currentFrame.height,
      });
      setCriticReport(report);
      setShowDefectOverlay(true);
    } catch (e: any) {
      alert(`Inspect Error: ${e.message}`);
    } finally {
      setIsInspecting(false);
    }
  };

  const handleAutoRepair = async () => {
    if (!currentFrame) return;
    setIsRepairing(true);
    try {
      if (!beforeImageBase64) {
        setBeforeImageBase64(currentFrame.image_base64);
        setBeforeScriptText(currentFrame.script_text);
      }
      const res = await repairSpriteApi({
        script_text: currentFrame.script_text,
        auto_refine: true,
        max_iterations: 3,
        category: genContext.category,
        palette_name: genContext.palette,
        width: currentFrame.width,
        height: currentFrame.height,
      });

      const rendered = await renderScript({
        script_text: res.repaired_script,
        width: currentFrame.width,
        height: currentFrame.height,
      });

      const updated = [...frames];
      updated[currentFrameIndex] = frameWithScript(rendered, res.repaired_script);
      setFrames(updated);
      setCriticReport(res.final_critic);
      setRepairHistory(res.history || []);
    } catch (e: any) {
      alert(`Auto Repair Error: ${e.message}`);
    } finally {
      setIsRepairing(false);
    }
  };

  const handleRestoreBefore = async () => {
    if (!beforeScriptText || !currentFrame) return;
    try {
      const restored = await renderScript({
        script_text: beforeScriptText,
        width: currentFrame.width,
        height: currentFrame.height,
      });
      const updated = [...frames];
      updated[currentFrameIndex] = frameWithScript(restored, beforeScriptText);
      setFrames(updated);
      setBeforeImageBase64(null);
      setBeforeScriptText(null);
      setViewingBefore(false);
      setRepairHistory([]);
      setHistoryVersion((version) => version + 1);

      const rep = await inspectSpriteApi({
        script_text: beforeScriptText,
        category: genContext.category,
        palette_name: genContext.palette,
        width: currentFrame.width,
        height: currentFrame.height,
      });
      setCriticReport(rep);
    } catch (e: any) {
      alert(`Restore Error: ${e.message}`);
    }
  };

  const handleRefine = async () => {
    if (!currentFrame) return;
    setIsLoading(true);
    try {
      const res = await refineSprite({
        script_text: currentFrame.script_text,
        suggested_fix: criticReport?.detected_defects[0]?.suggested_fix,
        width: currentFrame.width,
        height: currentFrame.height,
      });
      const updated = [...frames];
      updated[currentFrameIndex] = res.frame;
      setFrames(updated);
      setCriticReport(res.critic);
    } catch (e: any) {
      alert(`Refine Error: ${e.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleApplyCriticFix = async (fixCmd: string) => {
    if (!currentFrame) return;
    if (!currentFrame.script_text) {
      alert('This frame has no script attached, so it cannot be patched. Regenerate the sprite first.');
      return;
    }
    setIsLoading(true);
    try {
      if (!beforeImageBase64) {
        setBeforeImageBase64(currentFrame.image_base64);
        setBeforeScriptText(currentFrame.script_text);
      }
      const res = await repairSpriteApi({
        script_text: currentFrame.script_text,
        action_patch: fixCmd,
        category: genContext.category,
        palette_name: genContext.palette,
        width: currentFrame.width,
        height: currentFrame.height,
      });

      const rendered = await renderScript({
        script_text: res.repaired_script,
        width: currentFrame.width,
        height: currentFrame.height,
      });

      const updated = [...frames];
      updated[currentFrameIndex] = frameWithScript(rendered, res.repaired_script);
      setFrames(updated);
      setCriticReport(res.final_critic);
    } catch (e: any) {
      alert(`Fix Error: ${e.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleRerenderScript = async (newScript: string) => {
    if (!currentFrame) return;
    try {
      const rendered = await renderScript({
        script_text: newScript,
        width: currentFrame.width,
        height: currentFrame.height,
      });
      const updated = [...frames];
      updated[currentFrameIndex] = frameWithScript(rendered, newScript);
      setFrames(updated);
    } catch (e: any) {
      alert(`Script Render Error: ${e.message}`);
    }
  };

  const handleAddFrame = () => {
    if (!currentFrame) return;
    const newFrame = { ...currentFrame, frame_index: frames.length };
    setFrames([...frames, newFrame]);
    setCurrentFrameIndex(frames.length);
  };

  const handleDuplicateFrame = (index: number) => {
    const src = frames[index];
    if (!src) return;
    const dup = { ...src, frame_index: frames.length };
    setFrames([...frames, dup]);
    setCurrentFrameIndex(frames.length);
  };

  const handleDeleteFrame = (index: number) => {
    if (frames.length <= 1) return;
    const filtered = frames.filter((_, i) => i !== index);
    setFrames(filtered);
    setCurrentFrameIndex(Math.max(0, currentFrameIndex - 1));
  };

  const handleExport = async (format: 'png' | 'spritesheet' | 'gif' | 'json' | 'onion') => {
    if (frames.length === 0) return;
    try {
      const res = await exportAsset({
        frames_scripts: frames.map((f) => f.script_text),
        width: currentFrame?.width || 32,
        height: currentFrame?.height || 32,
        palette: 'pico8',
        format,
        scale: 4,
        fps: 8,
        frame_index: currentFrameIndex,
      });

      if (format === 'json') {
        const blob = new Blob([JSON.stringify(res, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `spritemind_export.json`;
        a.click();
      } else if (res.image_base64) {
        const a = document.createElement('a');
        a.href = res.image_base64;
        const extension = format === 'gif' ? 'gif' : 'png';
        a.download =
          format === 'onion' ? `spritemind_onion_f${currentFrameIndex}.png` : `spritemind_${format}.${extension}`;
        a.click();
      }

      if (format === 'onion') return;

      // Auto save to asset library
      await saveAsset({
        name: `Sprite_${Date.now().toString().slice(-4)}`,
        category: 'Generated',
        tags: [format, `${frames.length} frames`],
        frames,
      });
    } catch (e: any) {
      alert(`Export Error: ${e.message}`);
    }
  };

  const currentFrame = frames[currentFrameIndex] || null;
  const onionFrame = onionSkinEnabled && currentFrameIndex > 0 ? frames[currentFrameIndex - 1] : null;

  return (
    <div className="flex-1 flex flex-col h-[calc(100vh-3.5rem)] overflow-hidden">
      <div className="flex-1 flex overflow-hidden">
        {/* Left Generation Sidebar */}
        <GenerationSidebar
          onGenerate={handleGenerate}
          onRefine={handleRefine}
          onClear={() => setFrames([])}
          isLoading={isLoading}
          selectedEngine={selectedEngine}
          historyVersion={historyVersion}
        />

        {/* Central Workspace */}
        <CanvasWorkspace
          currentFrame={currentFrame}
          selectedViewPass={selectedViewPass}
          setSelectedViewPass={setSelectedViewPass}
          showPixelGrid={showPixelGrid}
          setShowPixelGrid={setShowPixelGrid}
          onionSkinFrame={onionFrame}
          criticReport={criticReport}
          showDefectOverlay={showDefectOverlay}
          setShowDefectOverlay={setShowDefectOverlay}
          beforeImageBase64={beforeImageBase64}
          viewingBefore={viewingBefore}
          setViewingBefore={setViewingBefore}
        />

        {/* Right Properties & Inspection Sidebar */}
        <PropertiesSidebar
          currentFrame={currentFrame}
          criticReport={criticReport}
          modelsInfo={modelsInfo}
          onInspectSprite={handleInspect}
          onAutoRepair={handleAutoRepair}
          onApplyCriticFix={handleApplyCriticFix}
          onRerenderScript={handleRerenderScript}
          inferenceTimeMs={inferenceTimeMs}
          isInspecting={isInspecting}
          isRepairing={isRepairing}
          repairHistory={repairHistory}
          beforeImageBase64={beforeImageBase64}
          onRestoreBefore={handleRestoreBefore}
        />
      </div>

      {/* Bottom Animation Timeline */}
      <AnimationTimeline
        frames={frames}
        currentFrameIndex={currentFrameIndex}
        onSelectFrame={setCurrentFrameIndex}
        onAddFrame={handleAddFrame}
        onDuplicateFrame={handleDuplicateFrame}
        onDeleteFrame={handleDeleteFrame}
        onExport={handleExport}
        onionSkinEnabled={onionSkinEnabled}
        setOnionSkinEnabled={setOnionSkinEnabled}
      />
    </div>
  );
};
