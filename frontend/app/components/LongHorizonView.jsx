'use client';

import { useEffect, useState, useCallback } from 'react';
import axios from 'axios';

const STAGE_COLORS = [
  '#3b82f6', '#06b6d4', '#10b981', '#84cc16',
  '#f59e0b', '#f97316', '#ef4444', '#a855f7',
  '#ec4899', '#e11d48',
];

function StageBar({ stage, maxStage = 10, windowAvg, threshold }) {
  const pct = (stage / maxStage) * 100;
  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <span className="panel-label">Stage {stage} / {maxStage}</span>
        <span className="text-xs font-mono font-bold" style={{ color: STAGE_COLORS[stage - 1] ?? '#3b82f6' }}>
          avg {(windowAvg * 100).toFixed(1)}% (threshold {(threshold * 100).toFixed(0)}%)
        </span>
      </div>
      <div className="h-3 rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.06)' }}>
        <div className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, background: `linear-gradient(90deg, #3b82f6, ${STAGE_COLORS[stage - 1] ?? '#3b82f6'})` }} />
      </div>
    </div>
  );
}

function EpisodeTimeline({ stage }) {
  const segments = 10;
  return (
    <div className="flex gap-1">
      {Array.from({ length: segments }, (_, i) => {
        const isReached = i < stage;
        const isCurrent = i === stage - 1;
        return (
          <div key={i} className="flex-1 rounded-md transition-all duration-300"
            style={{
              height: 32,
              background: isReached
                ? (isCurrent ? STAGE_COLORS[i] : `${STAGE_COLORS[i]}55`)
                : 'rgba(255,255,255,0.04)',
              border: isCurrent ? `2px solid ${STAGE_COLORS[i]}` : '1px solid rgba(255,255,255,0.06)',
              boxShadow: isCurrent ? `0 0 16px ${STAGE_COLORS[i]}66` : 'none',
            }}>
            <div className="h-full flex items-center justify-center">
              <span className="text-[10px] font-bold text-white opacity-80">{i + 1}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default function LongHorizonView() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const res = await axios.get('/curriculum/status');
      setData(res.data);
    } catch { /* silent */ }
    setLoading(false);
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 4000);
    return () => clearInterval(id);
  }, [refresh]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <span className="panel-label animate-pulse">Loading curriculum status...</span>
      </div>
    );
  }

  const stage = data?.stage ?? 1;
  const maxSteps = data?.max_steps ?? 100;
  const threshold = data?.threshold ?? 0.65;
  const windowAvg = data?.window_avg ?? 0;
  const episode = data?.episode ?? 0;
  const transitions = data?.transitions ?? [];

  return (
    <div className="flex flex-col gap-6 p-6 overflow-y-auto h-full">
      <div>
        <h2 className="text-lg font-black text-white">Long-Horizon Curriculum Learning</h2>
        <p className="panel-label mt-0.5">10-stage progression · max_steps 100 → 1000</p>
      </div>

      {/* Summary KPIs */}
      <div className="grid grid-cols-4 gap-4">
        {[
          { label: 'Current Stage', value: stage, accent: STAGE_COLORS[stage - 1] ?? '#3b82f6' },
          { label: 'Max Steps', value: maxSteps, accent: '#06b6d4' },
          { label: 'Episodes Run', value: episode, accent: '#10b981' },
          { label: 'Stages Advanced', value: transitions.length, accent: '#a855f7' },
        ].map(({ label, value, accent }) => (
          <div key={label} className="rounded-2xl p-4"
            style={{ background: 'rgba(10,14,28,0.9)', border: '1px solid rgba(255,255,255,0.07)' }}>
            <div className="panel-label mb-1">{label}</div>
            <div className="text-2xl font-black" style={{ color: accent, fontFamily: 'var(--font-mono)' }}>{value}</div>
          </div>
        ))}
      </div>

      {/* Stage progress bar */}
      <div className="rounded-2xl p-4" style={{ background: 'rgba(10,14,28,0.9)', border: '1px solid rgba(255,255,255,0.07)' }}>
        <StageBar stage={stage} windowAvg={windowAvg} threshold={threshold} />
      </div>

      {/* Episode timeline segments */}
      <div className="rounded-2xl p-4" style={{ background: 'rgba(10,14,28,0.9)', border: '1px solid rgba(255,255,255,0.07)' }}>
        <div className="panel-label mb-3">Stage Timeline</div>
        <EpisodeTimeline stage={stage} />
        <div className="flex items-center justify-between mt-2">
          <span className="text-[10px]" style={{ color: 'rgba(148,163,184,0.5)' }}>Stage 1 (100 steps)</span>
          <span className="text-[10px]" style={{ color: 'rgba(148,163,184,0.5)' }}>Stage 10 (1000 steps)</span>
        </div>
      </div>

      {/* Transitions log */}
      <div>
        <div className="panel-label mb-3">Stage Advancement History</div>
        {transitions.length === 0 ? (
          <div className="rounded-xl p-6 text-center" style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)' }}>
            <p className="panel-label">Training not started yet</p>
          </div>
        ) : (
          <div className="space-y-2">
            {[...transitions].reverse().slice(0, 10).map((t, i) => (
              <div key={i} className="flex items-center gap-3 rounded-xl p-3"
                style={{ background: 'rgba(16,185,129,0.05)', border: '1px solid rgba(16,185,129,0.15)' }}>
                <div className="w-8 h-8 rounded-lg flex items-center justify-center flex-none"
                  style={{ background: `${STAGE_COLORS[(t.to_stage ?? 2) - 1]}22` }}>
                  <span className="text-sm">▲</span>
                </div>
                <div className="flex-1">
                  <div className="text-xs font-bold text-white">
                    Stage {t.from_stage} → Stage {t.to_stage}
                  </div>
                  <div className="text-[10px]" style={{ color: 'rgba(148,163,184,0.6)' }}>
                    Episode {t.episode} · avg score {(t.avg_score * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
