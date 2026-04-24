'use client';

import { useEffect, useState, useCallback } from 'react';
import axios from 'axios';

function ExpertGapBar({ gap }) {
  const pct = Math.min(Math.abs(gap) * 100, 100);
  const isClosed = gap <= 0.05;
  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <span className="panel-label">Expert Gap</span>
        <span className="text-xs font-mono font-bold" style={{ color: isClosed ? '#10b981' : '#f59e0b' }}>
          {gap > 0 ? '+' : ''}{(gap * 100).toFixed(1)}%
        </span>
      </div>
      <div className="h-3 rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.06)' }}>
        <div className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, background: isClosed ? '#10b981' : `linear-gradient(90deg, #f59e0b, #ef4444)` }} />
      </div>
      <div className="text-[10px] mt-1" style={{ color: 'rgba(148,163,184,0.5)' }}>
        {isClosed ? 'Agent has closed the gap ✓' : 'Training to close expert gap...'}
      </div>
    </div>
  );
}

function WeaknessCluster({ cluster, index }) {
  const accentColors = ['#ef4444', '#f97316', '#f59e0b', '#84cc16', '#10b981'];
  const accent = accentColors[index % accentColors.length];
  const improving = cluster.is_improving;

  return (
    <div className="rounded-xl p-4"
      style={{ background: 'rgba(10,14,28,0.9)', border: `1px solid ${accent}33` }}>
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-bold text-white">Cluster {index + 1}</span>
        <span className="text-[10px] font-bold px-2 py-0.5 rounded-full"
          style={{
            background: improving ? 'rgba(16,185,129,0.15)' : 'rgba(239,68,68,0.15)',
            color: improving ? '#10b981' : '#ef4444',
          }}>
          {improving ? '↑ Improving' : '↓ Degrading'}
        </span>
      </div>
      <div className="text-2xl font-black mb-1" style={{ color: accent, fontFamily: 'var(--font-mono)' }}>
        {((cluster.avg_score ?? 0) * 100).toFixed(1)}%
      </div>
      <div className="text-[10px]" style={{ color: 'rgba(148,163,184,0.6)' }}>
        {cluster.count ?? 0} scenarios · n_ambs:{cluster.feature_means?.n_ambulances?.toFixed(0) ?? '?'}
        · traffic: {cluster.feature_means?.traffic_intensity?.toFixed(2) ?? '?'}
      </div>
      {/* Mini sparkline */}
      {cluster.improvement_history?.length > 1 && (
        <div className="flex items-end gap-0.5 mt-2 h-6">
          {cluster.improvement_history.slice(-12).map((v, i) => (
            <div key={i} className="flex-1 rounded-sm"
              style={{
                height: `${Math.max(4, (v ?? 0) * 100)}%`,
                background: accent,
                opacity: 0.5 + (i / 12) * 0.5,
              }} />
          ))}
        </div>
      )}
    </div>
  );
}

export default function SelfImprovementView() {
  const [weaknesses, setWeaknesses] = useState(null);
  const [history, setHistory] = useState(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const [wRes, hRes] = await Promise.all([
        axios.get('/selfplay/weaknesses'),
        axios.get('/selfplay/iterations'),
      ]);
      setWeaknesses(wRes.data);
      setHistory(hRes.data);
    } catch { /* silent */ }
    setLoading(false);
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 5000);
    return () => clearInterval(id);
  }, [refresh]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <span className="panel-label animate-pulse">Loading self-improvement data...</span>
      </div>
    );
  }

  const clusters = weaknesses?.clusters ?? [];
  const iteration = weaknesses?.iteration ?? 0;

  // Compute expert gap from cluster history
  const clusterKeys = Object.keys(history ?? {});
  const avgGap = clusterKeys.length > 0
    ? clusterKeys.reduce((sum, k) => sum + (history[k]?.delta ?? 0), 0) / clusterKeys.length
    : 0;

  return (
    <div className="flex flex-col gap-6 p-6 overflow-y-auto h-full">
      <div>
        <h2 className="text-lg font-black text-white">Self-Improvement Loop</h2>
        <p className="panel-label mt-0.5">
          AdversarialScenarioGenerator · KMeans failure clustering · Expert imitation
        </p>
      </div>

      {/* KPIs */}
      <div className="grid grid-cols-3 gap-4">
        {[
          { label: 'Self-Play Iteration', value: iteration, accent: '#6366f1' },
          { label: 'Weakness Clusters', value: clusters.length, accent: '#f59e0b' },
          { label: 'Tracked Dimensions', value: clusterKeys.length, accent: '#10b981' },
        ].map(({ label, value, accent }) => (
          <div key={label} className="rounded-2xl p-4"
            style={{ background: 'rgba(10,14,28,0.9)', border: '1px solid rgba(255,255,255,0.07)' }}>
            <div className="panel-label mb-1">{label}</div>
            <div className="text-2xl font-black" style={{ color: accent, fontFamily: 'var(--font-mono)' }}>{value}</div>
          </div>
        ))}
      </div>

      {/* Expert Gap */}
      <div className="rounded-2xl p-4" style={{ background: 'rgba(10,14,28,0.9)', border: '1px solid rgba(255,255,255,0.07)' }}>
        <ExpertGapBar gap={avgGap} />
      </div>

      {/* Weakness clusters */}
      <div>
        <div className="panel-label mb-3">Top Weakness Clusters</div>
        {clusters.length === 0 ? (
          <div className="rounded-xl p-6 text-center"
            style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)' }}>
            <span className="text-3xl opacity-20">🔍</span>
            <p className="panel-label mt-2">Run train_selfplay.py to generate weakness data</p>
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-4">
            {clusters.slice(0, 6).map((c, i) => (
              <WeaknessCluster key={i} cluster={c} index={i} />
            ))}
          </div>
        )}
      </div>

      {/* Per-cluster improvement history */}
      {clusterKeys.length > 0 && (
        <div>
          <div className="panel-label mb-3">Cluster Improvement Trends</div>
          <div className="space-y-2">
            {clusterKeys.map(k => {
              const d = history[k];
              return (
                <div key={k} className="rounded-xl p-3 flex items-center gap-4"
                  style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)' }}>
                  <div className="w-24 flex-none">
                    <div className="text-[10px] font-bold text-white truncate">{k}</div>
                    <div className="text-[10px]" style={{ color: 'rgba(148,163,184,0.5)' }}>{d?.trend}</div>
                  </div>
                  <div className="flex-1 flex items-end gap-0.5 h-8">
                    {(d?.history ?? []).slice(-16).map((v, i, arr) => (
                      <div key={i} className="flex-1 rounded-sm"
                        style={{
                          height: `${Math.max(4, (v ?? 0) * 100)}%`,
                          background: d?.delta > 0 ? '#10b981' : '#ef4444',
                          opacity: 0.3 + (i / arr.length) * 0.7,
                        }} />
                    ))}
                  </div>
                  <div className="w-16 text-right flex-none">
                    <div className="text-xs font-bold font-mono"
                      style={{ color: d?.delta > 0 ? '#10b981' : '#ef4444' }}>
                      {d?.delta > 0 ? '+' : ''}{((d?.delta ?? 0) * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
