import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  Wand2,
  Sparkles,
  Play,
  Pause,
  Download,
  Loader2,
  Flame,
  Swords,
  Footprints,
  ShieldCheck,
  Skull,
  Zap,
} from 'lucide-react';
import { generateActionSetApi, planActionSetApi } from '../services/api';

interface ActionSpec {
  key: string;
  label: string;
  anim: string;
  frames: number;
  fps: number;
  loop: boolean;
  effects: string[];
}

interface PlanInfo {
  category: string;
  style: string;
  palette: string;
  archetype: string;
  element: string;
  weapon: string;
  actions: ActionSpec[];
  rationale: string[];
}

interface BuiltAction extends ActionSpec {
  engine_used: string;
  num_frames: number;
  frames: { image_base64: string; script_text: string; pixel_count: number }[];
  gif_base64?: string;
}

const ACTION_ICONS: Record<string, React.ReactNode> = {
  idle: <Sparkles className="w-3.5 h-3.5" />,
  walk: <Footprints className="w-3.5 h-3.5" />,
  run: <Footprints className="w-3.5 h-3.5" />,
  dash: <Zap className="w-3.5 h-3.5" />,
  attack: <Swords className="w-3.5 h-3.5" />,
  attack_heavy: <Swords className="w-3.5 h-3.5" />,
  cast: <Flame className="w-3.5 h-3.5" />,
  cast_burst: <Flame className="w-3.5 h-3.5" />,
  shoot: <Zap className="w-3.5 h-3.5" />,
  block: <ShieldCheck className="w-3.5 h-3.5" />,
  hurt: <Skull className="w-3.5 h-3.5" />,
  death: <Skull className="w-3.5 h-3.5" />,
};

const ActionTile: React.FC<{ action: BuiltAction }> = ({ action }) => {
  const [playing, setPlaying] = useState(true);
  const [frameIdx, setFrameIdx] = useState(0);
  const timer = useRef<number | null>(null);

  useEffect(() => {
    if (timer.current) window.clearInterval(timer.current);
    if (!playing) return;
    const delay = Math.max(40, Math.round(1000 / Math.max(1, action.fps)));
    timer.current = window.setInterval(() => {
      setFrameIdx((i) => (i + 1) % action.frames.length);
    }, delay);
    return () => {
      if (timer.current) window.clearInterval(timer.current);
    };
  }, [playing, action.fps, action.frames.length]);

  const frame = action.frames[Math.min(frameIdx, action.frames.length - 1)];

  const downloadGif = () => {
    if (!action.gif_base64) return;
    const link = document.createElement('a');
    link.href = `data:image/gif;base64,${action.gif_base64}`;
    link.download = `${action.key}.gif`;
    link.click();
  };

  return (
    <div className="bg-dark-900 border border-dark-800 rounded-lg overflow-hidden">
      <div className="flex items-center justify-between px-2.5 py-1.5 border-b border-dark-800">
        <div className="flex items-center gap-1.5 text-xs font-medium text-slate-200">
          {ACTION_ICONS[action.key] || <Sparkles className="w-3.5 h-3.5" />}
          {action.label}
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={() => setPlaying((p) => !p)}
            className="p-1 text-slate-400 hover:text-slate-100"
            title={playing ? 'Pause' : 'Play'}
          >
            {playing ? <Pause className="w-3 h-3" /> : <Play className="w-3 h-3" />}
          </button>
          <button
            onClick={downloadGif}
            disabled={!action.gif_base64}
            className="p-1 text-slate-400 hover:text-slate-100 disabled:opacity-30"
            title="Download GIF"
          >
            <Download className="w-3 h-3" />
          </button>
        </div>
      </div>
      <div className="flex items-center justify-center bg-dark-950 py-2">
        {frame && (
          <img
            src={frame.image_base64}
            alt={action.label}
            className="w-20 h-20"
            style={{ imageRendering: 'pixelated' }}
          />
        )}
      </div>
      <div className="flex items-center justify-between px-2.5 pb-1.5 text-[10px] text-slate-500">
        <span>{action.num_frames} frames</span>
        <span>{action.fps} fps</span>
        <span>{action.effects.length ? action.effects.join('+') : 'no fx'}</span>
      </div>
    </div>
  );
};

export const ActionSetPage: React.FC = () => {
  const [prompt, setPrompt] = useState('fire mage that casts fireballs and dashes');
  const [plan, setPlan] = useState<PlanInfo | null>(null);
  const [selected, setSelected] = useState<string[]>([]);
  const [actions, setActions] = useState<BuiltAction[]>([]);
  const [loading, setLoading] = useState(false);
  const [planning, setPlanning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [facing, setFacing] = useState<'right' | 'left'>('right');

  const refreshPlan = async () => {
    setPlanning(true);
    setError(null);
    try {
      const res = await planActionSetApi({ prompt });
      setPlan(res.plan);
      setSelected(res.plan.actions.map((a: ActionSpec) => a.key));
    } catch (e: any) {
      setError(e.message || 'Planning failed');
    } finally {
      setPlanning(false);
    }
  };

  useEffect(() => {
    refreshPlan();
  }, []);

  const generate = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await generateActionSetApi({ prompt, actions: selected, facing });
      setPlan(res.plan);
      setActions(res.actions);
    } catch (e: any) {
      setError(e.message || 'Generation failed');
    } finally {
      setLoading(false);
    }
  };

  const summary = useMemo(() => {
    if (!plan) return null;
    return [
      ['archetype', plan.archetype],
      ['element', plan.element],
      ['weapon', plan.weapon],
      ['category', plan.category],
      ['style', plan.style],
      ['palette', plan.palette],
    ];
  }, [plan]);

  return (
    <div className="flex-1 flex flex-col overflow-hidden bg-dark-950">
      <header className="px-5 py-3 border-b border-dark-800 flex items-center justify-between">
        <div>
          <h1 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
            <Wand2 className="w-4 h-4 text-purple-400" />
            Action Sets
          </h1>
          <p className="text-[11px] text-slate-500">
            One prompt in, a coherent animation package out: idle, movement, attacks, defense and death.
          </p>
        </div>
        <button
          onClick={generate}
          disabled={loading || selected.length === 0}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-purple-600 hover:bg-purple-500 disabled:opacity-50 text-white rounded-md"
        >
          {loading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Sparkles className="w-3.5 h-3.5" />}
          Generate Set
        </button>
      </header>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        <div className="bg-dark-900 border border-dark-800 rounded-lg p-3 space-y-3">
          <div className="flex gap-2">
            <input
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && refreshPlan()}
              placeholder="e.g. ice paladin with shield who blocks and smashes"
              className="flex-1 bg-dark-950 border border-dark-800 rounded-md px-3 py-1.5 text-xs text-slate-200 outline-none focus:border-purple-500"
            />
            <div className="flex rounded-md overflow-hidden border border-dark-800">
              {(['right', 'left'] as const).map((dir) => (
                <button
                  key={dir}
                  onClick={() => setFacing(dir)}
                  className={`px-2 py-1.5 text-[10px] uppercase tracking-wide ${
                    facing === dir ? 'bg-purple-600 text-white' : 'bg-dark-950 text-slate-400 hover:bg-dark-800'
                  }`}
                  title={`Generate ${dir}-facing sprites`}
                >
                  {dir === 'right' ? 'RT' : 'LT'}
                </button>
              ))}
            </div>
            <button
              onClick={refreshPlan}
              disabled={planning}
              className="px-3 py-1.5 text-xs bg-dark-800 hover:bg-dark-700 rounded-md text-slate-300 flex items-center gap-1.5"
            >
              {planning ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Wand2 className="w-3.5 h-3.5" />}
              Read prompt
            </button>
          </div>

          {summary && (
            <div className="flex flex-wrap gap-1.5">
              {summary.map(([k, v]) => (
                <span
                  key={k}
                  className="text-[10px] px-1.5 py-0.5 rounded bg-dark-950 border border-dark-800 text-slate-400"
                >
                  {k}: <span className="text-slate-200">{v}</span>
                </span>
              ))}
            </div>
          )}

          {plan && (
            <div className="space-y-1.5">
              <div className="text-[10px] uppercase tracking-wide text-slate-500">Animations in this set</div>
              <div className="flex flex-wrap gap-1.5">
                {plan.actions.map((a) => {
                  const on = selected.includes(a.key);
                  return (
                    <button
                      key={a.key}
                      onClick={() =>
                        setSelected((prev) => (on ? prev.filter((k) => k !== a.key) : [...prev, a.key]))
                      }
                      className={`text-[11px] px-2 py-1 rounded border transition-colors ${
                        on
                          ? 'bg-purple-600/20 border-purple-500/60 text-purple-200'
                          : 'bg-dark-950 border-dark-800 text-slate-500 hover:text-slate-300'
                      }`}
                    >
                      {a.label}
                      <span className="ml-1 text-[9px] opacity-60">{a.frames}f</span>
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          {error && <div className="text-[11px] text-red-400">{error}</div>}
        </div>

        {actions.length > 0 && (
          <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-5 gap-2">
            {actions.map((a) => (
              <ActionTile key={a.key} action={a} />
            ))}
          </div>
        )}

        {actions.length === 0 && !loading && (
          <div className="text-center text-[11px] text-slate-500 py-10">
            Read a prompt to see the planned actions, then generate the set.
          </div>
        )}
      </div>
    </div>
  );
};
