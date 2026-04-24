'use client';

import { useEffect, useState, useCallback } from 'react';
import axios from 'axios';

function AgentCard({ agentId, reward, conflictSignal, epsilon }) {
  const hasConflict = conflictSignal > 0;
  return (
    <div className="rounded-xl p-3 relative overflow-hidden"
      style={{
        background: hasConflict ? 'rgba(239,68,68,0.08)' : 'rgba(255,255,255,0.03)',
        border: `1px solid ${hasConflict ? 'rgba(239,68,68,0.3)' : 'rgba(255,255,255,0.07)'}`,
      }}>
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs font-bold text-white">AMB {agentId + 1}</span>
        {hasConflict && (
          <span className="text-[10px] font-bold px-1.5 py-0.5 rounded-full"
            style={{ background: 'rgba(239,68,68,0.2)', color: '#f87171' }}>
            ⚠ CONFLICT
          </span>
        )}
      </div>
      <div className="text-xl font-black" style={{ color: hasConflict ? '#f87171' : '#10b981', fontFamily: 'var(--font-mono)' }}>
        {reward != null ? (reward > 0 ? '+' : '') + reward.toFixed(1) : '—'}
      </div>
      <div className="text-[10px] mt-1" style={{ color: 'rgba(148,163,184,0.6)' }}>
        ε = {epsilon != null ? epsilon.toFixed(3) : '—'}
      </div>
    </div>
  );
}

export default function MultiAgentView() {
  const [status, setStatus] = useState(null);
  const [conflicts, setConflicts] = useState([]);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const [sRes, cRes] = await Promise.all([
        axios.get('/marl/status'),
        axios.get('/marl/conflicts?last_n=15'),
      ]);
      setStatus(sRes.data);
      setConflicts(Array.isArray(cRes.data) ? cRes.data : []);
    } catch { /* silent */ }
    setLoading(false);
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 3000);
    return () => clearInterval(id);
  }, [refresh]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <span className="panel-label animate-pulse">Loading MARL status...</span>
      </div>
    );
  }

  const agents = status?.agent_metrics ?? {};
  const agentIds = Object.keys(agents).map(Number);

  return (
    <div className="flex flex-col gap-6 p-6 overflow-y-auto h-full">
      {/* Title */}
      <div>
        <h2 className="text-lg font-black text-white">Multi-Agent Fleet Coordination</h2>
        <p className="panel-label mt-0.5">Independent Q-Learning with OversightAgent conflict detection</p>
      </div>

      {/* Summary KPIs */}
      <div className="grid grid-cols-4 gap-4">
        {[
          { label: 'Step', value: status?.step_count ?? 0, accent: '#3b82f6' },
          { label: 'Total Conflicts', value: status?.total_conflicts ?? 0, accent: '#ef4444' },
          { label: 'Conflict Rate', value: `${((status?.conflict_rate ?? 0) * 100).toFixed(1)}%`, accent: '#f59e0b' },
          { label: 'Active Agents', value: agentIds.length, accent: '#10b981' },
        ].map(({ label, value, accent }) => (
          <div key={label} className="rounded-2xl p-4"
            style={{ background: 'rgba(10,14,28,0.9)', border: '1px solid rgba(255,255,255,0.07)' }}>
            <div className="panel-label mb-1">{label}</div>
            <div className="text-2xl font-black" style={{ color: accent, fontFamily: 'var(--font-mono)' }}>{value}</div>
          </div>
        ))}
      </div>

      {/* Agent Cards Grid */}
      {agentIds.length > 0 && (
        <div>
          <div className="panel-label mb-3">Agent Cards</div>
          <div className="grid grid-cols-3 gap-3">
            {agentIds.map(id => (
              <AgentCard
                key={id}
                agentId={id}
                reward={agents[id]?.avg_reward}
                conflictSignal={agents[id]?.conflicts ?? 0}
                epsilon={agents[id]?.epsilon}
              />
            ))}
          </div>
        </div>
      )}

      {/* Conflict Timeline */}
      <div>
        <div className="panel-label mb-3">Recent Conflict Events</div>
        {conflicts.length === 0 ? (
          <div className="rounded-xl p-6 text-center" style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)' }}>
            <span className="text-3xl opacity-20">✓</span>
            <p className="panel-label mt-2">No conflicts detected</p>
          </div>
        ) : (
          <div className="space-y-2">
            {conflicts.map((c, i) => (
              <div key={i} className="flex items-center gap-3 rounded-xl p-3"
                style={{ background: 'rgba(239,68,68,0.05)', border: '1px solid rgba(239,68,68,0.15)' }}>
                <div className="w-8 h-8 rounded-lg flex items-center justify-center flex-none"
                  style={{ background: 'rgba(239,68,68,0.15)' }}>
                  <span className="text-sm">⚡</span>
                </div>
                <div className="flex-1">
                  <div className="text-xs font-bold text-white">
                    AMB {(c.agent_a ?? 0) + 1} ↔ AMB {(c.agent_b ?? 0) + 1}
                  </div>
                  <div className="text-[10px]" style={{ color: 'rgba(148,163,184,0.6)' }}>
                    Step {c.step} · Emergency {c.emergency_id}
                    {c.resolved ? ' · Resolved' : ''}
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
