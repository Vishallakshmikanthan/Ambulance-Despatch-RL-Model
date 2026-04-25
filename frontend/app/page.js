'use client';

import { useState, useEffect, useCallback, useRef } from 'react'
import axios from 'axios'
import CityMap from './components/CityMap'
import AmbulanceTable from './components/AmbulanceTable'
import RewardChart from './components/RewardChart'
import HospitalPanel from './components/HospitalPanel'
import MultiAgentView from './components/MultiAgentView'
import LongHorizonView from './components/LongHorizonView'
import SelfImprovementView from './components/SelfImprovementView'

const TASKS = ['easy', 'medium', 'hard']
const TASK_STEPS = { easy: 30, medium: 60, hard: 100 }
const TASK_COLOR = { easy: '#10b981', medium: '#f59e0b', hard: '#ef4444' }

const VIEWS = [
  { id: 'live',           label: 'LIVE',           icon: '◉' },
  { id: 'multi-agent',    label: 'MULTI-AGENT',    icon: '⬡' },
  { id: 'long-horizon',   label: 'LONG-HORIZON',   icon: '⧖' },
  { id: 'self-improve',   label: 'SELF-IMPROVE',   icon: '↻' },
]

const emptyObs = {
  ambulances: [], emergencies: [], hospitals: [],
  traffic: { global: 1.0 }, done: false, reward: 0.0, step: 0, rubric: null
}

function KpiCard({ label, value, sub, accent = '#3b82f6', icon }) {
  return (
    <div className="relative overflow-hidden rounded-2xl p-4 flex flex-col gap-1"
      style={{ background: 'rgba(10,14,28,0.9)', border: '1px solid rgba(255,255,255,0.07)', boxShadow: `0 0 30px ${accent}18` }}>
      <div className="absolute top-0 right-0 w-20 h-20 rounded-full opacity-10 blur-2xl"
        style={{ background: accent, transform: 'translate(30%, -30%)' }} />
      <div className="flex items-center justify-between mb-1">
        <span className="panel-label">{label}</span>
        <span className="text-lg" style={{ color: accent }}>{icon}</span>
      </div>
      <div className="text-3xl font-black data-value" style={{ color: '#fff', fontFamily: 'var(--font-mono)' }}>{value}</div>
      {sub && <div className="text-xs mt-0.5" style={{ color: 'rgba(148,163,184,0.6)' }}>{sub}</div>}
    </div>
  )
}

function StatusDot({ status }) {
  const map = {
    'CONNECTED':   { color: '#10b981', label: 'Connected' },
    'COMPLETE':    { color: '#3b82f6', label: 'Episode Done' },
    'INITIALIZING...': { color: '#f59e0b', label: 'Initializing' },
    'OFFLINE':     { color: '#6b7280', label: 'Offline' },
  }
  const s = map[status] || { color: '#ef4444', label: status }
  return (
    <div className="flex items-center gap-2">
      <span className="relative flex h-2 w-2">
        <span className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-75" style={{ background: s.color }} />
        <span className="relative inline-flex rounded-full h-2 w-2" style={{ background: s.color }} />
      </span>
      <span className="text-xs font-bold" style={{ color: s.color, fontFamily: 'var(--font-mono)' }}>{s.label}</span>
    </div>
  )
}

export default function Dashboard() {
  const [activeView, setActiveView] = useState('live')
  const [task, setTask] = useState('hard')
  const [obs, setObs] = useState(emptyObs)
  const [rewardHistory, setRewardHistory] = useState([])
  const [episodeScore, setEpisodeScore] = useState(0)
  const [running, setRunning] = useState(false)
  const [autoRun, setAutoRun] = useState(false)
  const [speed, setSpeed] = useState(500)
  const [status, setStatus] = useState('OFFLINE')
  const [metrics, setMetrics] = useState({})
  const autoRef = useRef(false)
  const speedRef = useRef(300)
  const stepRef = useRef(null)

  useEffect(() => { autoRef.current = autoRun }, [autoRun])
  useEffect(() => { speedRef.current = speed }, [speed])

  // Auto-connect and auto-run on first load
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    resetEnv().then(() => {
      // Small delay then start auto-run for instant impressive demo
      setTimeout(() => setAutoRun(true), 1000)
    })
  }, [])

  const resetEnv = useCallback(async () => {
    setStatus('INITIALIZING...')
    setAutoRun(false); autoRef.current = false
    setRewardHistory([]); setEpisodeScore(0); setMetrics({})
    try {
      const resp = await axios.post(`/env/reset`, { task_name: task })
      const data = resp.data
      setObs(data.observation ?? data)
      setStatus('CONNECTED')
    } catch { setStatus('CONNECTION ERROR') }
  }, [task])

  const stepEnv = useCallback(async () => {
    if (running) return
    setRunning(true)
    try {
      const resp = await axios.post(`/env/step`, { action: { is_noop: true } })
      const data = resp.data
      const obsData = data.observation ?? data
      setObs(obsData)
      if (obsData.reward != null) {
        setRewardHistory(prev => [...prev.slice(-79), obsData.reward])
        setEpisodeScore(prev => prev + obsData.reward)
      }
      if (obsData.done) { setAutoRun(false); autoRef.current = false; setStatus('COMPLETE') }
      // Refresh metrics from the dedicated endpoint
      try {
        const mResp = await axios.get('/env/metrics')
        if (mResp.data?.metrics) setMetrics(mResp.data.metrics)
      } catch { /* silent */ }
    } catch { /* silent */ }
    setRunning(false)
  }, [running])

  useEffect(() => { stepRef.current = stepEnv }, [stepEnv])

  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!autoRun) return
    const tick = async () => {
      if (!autoRef.current) return
      await stepRef.current()
      if (autoRef.current) setTimeout(tick, speedRef.current)
    }
    const t = setTimeout(tick, speedRef.current)
    return () => clearTimeout(t)
  }, [autoRun])

  const maxSteps = TASK_STEPS[task]
  const progress = Math.min((obs.step / maxSteps) * 100, 100)
  const activeIncidents = obs.emergencies.length
  const criticalCount = obs.emergencies.filter(e => e.severity === 'CRITICAL').length
  const tm = obs.traffic?.global ?? 1.0

  return (
    <div className="flex flex-col h-screen overflow-hidden" style={{ background: '#04060f' }}>
      {/* ─── HEADER ──────────────────────────────────────────────── */}
      <header className="flex-none flex items-center justify-between px-6 h-[60px]"
        style={{ background: 'rgba(4,6,15,0.95)', borderBottom: '1px solid rgba(255,255,255,0.06)', backdropFilter: 'blur(20px)', zIndex: 50 }}>
        {/* Logo */}
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-xl flex items-center justify-center"
            style={{ background: 'linear-gradient(135deg, #3b82f6, #06b6d4)', boxShadow: '0 0 20px rgba(59,130,246,0.4)' }}>
            <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          </div>
          <div>
            <div className="font-black text-base tracking-tight text-white leading-none">
              Dispatch<span style={{ color: '#3b82f6' }}>Command</span>
            </div>
            <div className="panel-label leading-none mt-0.5">RL Environment Monitor</div>
          </div>
        </div>

        {/* View Switcher */}
        <div className="flex items-center gap-1 p-1 rounded-xl" style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.08)' }}>
          {VIEWS.map(v => (
            <button key={v.id} onClick={() => setActiveView(v.id)}
              className="px-3 py-1.5 rounded-lg text-[10px] font-bold uppercase tracking-widest transition-all duration-200 flex items-center gap-1"
              style={activeView === v.id
                ? { background: '#3b82f6', color: '#fff', boxShadow: '0 0 16px rgba(59,130,246,0.5)' }
                : { color: 'rgba(148,163,184,0.6)' }}>
              <span>{v.icon}</span><span>{v.label}</span>
            </button>
          ))}
        </div>

        {/* Task Selector (only shown in live view) */}
        {activeView === 'live' && (
        <div className="flex items-center gap-1 p-1 rounded-xl" style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.08)' }}>
          {TASKS.map(t => (
            <button key={t} onClick={() => setTask(t)}
              className="px-5 py-1.5 rounded-lg text-xs font-bold uppercase tracking-widest transition-all duration-200"
              style={task === t
                ? { background: TASK_COLOR[t], color: '#fff', boxShadow: `0 0 16px ${TASK_COLOR[t]}66` }
                : { color: 'rgba(148,163,184,0.6)' }}>
              {t}
            </button>
          ))}
        </div>
        )}

        {/* Controls */}
        <div className="flex items-center gap-4">
          <StatusDot status={status} />
          <div className="w-px h-6" style={{ background: 'rgba(255,255,255,0.08)' }} />
          {/* Speed */}
          <div className="flex items-center gap-2">
            <span className="panel-label">Speed</span>
            <input type="range" min={300} max={2000} step={100} value={speed} onChange={e => setSpeed(+e.target.value)}
              className="w-24 accent-blue-500" style={{ cursor: 'pointer' }} />
            <span className="text-xs font-mono" style={{ color: '#64748b', minWidth: 36 }}>{speed}ms</span>
          </div>
          <div className="w-px h-6" style={{ background: 'rgba(255,255,255,0.08)' }} />
          {/* Action Buttons */}
          <div className="flex items-center gap-2">
            <button onClick={resetEnv} title="Reset"
              className="w-8 h-8 rounded-lg flex items-center justify-center transition-all hover:scale-105"
              style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)' }}>
              <svg className="w-4 h-4" style={{ color: '#94a3b8' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
            </button>
            <button onClick={stepEnv} disabled={obs.done || running} title="Step"
              className="w-8 h-8 rounded-lg flex items-center justify-center transition-all hover:scale-105 disabled:opacity-30"
              style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)' }}>
              <svg className="w-4 h-4" style={{ color: '#f59e0b' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 5l7 7-7 7M5 5l7 7-7 7" />
              </svg>
            </button>
            <button onClick={() => setAutoRun(p => !p)} disabled={obs.done}
              className="px-5 py-1.5 rounded-lg text-xs font-bold uppercase tracking-widest transition-all hover:scale-105 disabled:opacity-30"
              style={autoRun
                ? { background: 'rgba(239,68,68,0.15)', color: '#ef4444', border: '1px solid rgba(239,68,68,0.3)', boxShadow: '0 0 12px rgba(239,68,68,0.2)' }
                : { background: 'rgba(16,185,129,0.15)', color: '#10b981', border: '1px solid rgba(16,185,129,0.3)', boxShadow: '0 0 12px rgba(16,185,129,0.2)' }}>
              {autoRun ? '⏹ Stop' : '▶ Launch'}
            </button>
          </div>
        </div>
      </header>

      {/* ─── PROGRESS BAR ───────────────────────────────────────── */}
      <div className="flex-none h-[3px] w-full" style={{ background: 'rgba(255,255,255,0.05)' }}>
        <div className="h-full"
          style={{ width: `${progress}%`, background: `linear-gradient(90deg, #3b82f6, #06b6d4)`, boxShadow: '0 0 12px rgba(59,130,246,0.8)', transition: 'width 0.4s ease' }} />
      </div>

      {/* ─── MAIN LAYOUT ─────────────────────────────────────────── */}
      {activeView === 'multi-agent' && (
        <div className="flex flex-1 overflow-hidden"><MultiAgentView /></div>
      )}
      {activeView === 'long-horizon' && (
        <div className="flex flex-1 overflow-hidden"><LongHorizonView /></div>
      )}
      {activeView === 'self-improve' && (
        <div className="flex flex-1 overflow-hidden"><SelfImprovementView /></div>
      )}
      {activeView === 'live' && (
      <div className="flex flex-1 overflow-hidden">

        {/* ── LEFT SIDEBAR ─────────────────────────────────── */}
        <aside className="w-[280px] flex-none flex flex-col gap-3 p-4 overflow-y-auto"
          style={{ borderRight: '1px solid rgba(255,255,255,0.06)' }}>
          {/* KPI Grid */}
          <div className="grid grid-cols-2 gap-3">
            <KpiCard label="Score" value={episodeScore.toFixed(0)} sub={`Δ ${obs.reward > 0 ? '+' : ''}${obs.reward?.toFixed(1) ?? '0.0'}`} accent="#3b82f6" icon="◈" />
            <KpiCard label="Step" value={obs.step} sub={`of ${maxSteps}`} accent="#06b6d4" icon="⧖" />
            <KpiCard label="Active" value={activeIncidents} sub={criticalCount > 0 ? `${criticalCount} critical` : 'all clear'} accent={criticalCount > 0 ? '#ef4444' : '#10b981'} icon="⚡" />
            <KpiCard label="Traffic" value={tm.toFixed(2)} sub={tm > 1.5 ? 'Heavy' : tm > 1.2 ? 'Moderate' : 'Clear'} accent={tm > 1.5 ? '#ef4444' : tm > 1.2 ? '#f97316' : '#10b981'} icon="↯" />
          </div>

          {/* Fleet Header */}
          <div className="flex items-center justify-between pt-1">
            <span className="panel-label">Fleet Status</span>
            <span className="text-xs font-bold px-2 py-0.5 rounded-full"
              style={{ background: 'rgba(59,130,246,0.1)', color: '#60a5fa', border: '1px solid rgba(59,130,246,0.2)' }}>
              {obs.ambulances.length} units
            </span>
          </div>
          <AmbulanceTable ambulances={obs.ambulances} />

          {/* Hospital Header */}
          <div className="flex items-center justify-between pt-1">
            <span className="panel-label">Hospital Network</span>
            <span className="text-xs font-bold px-2 py-0.5 rounded-full"
              style={{ background: 'rgba(16,185,129,0.1)', color: '#34d399', border: '1px solid rgba(16,185,129,0.2)' }}>
              {obs.hospitals.length} facilities
            </span>
          </div>
          <HospitalPanel hospitals={obs.hospitals} />

          {/* Session Summary */}
          {(metrics.served != null || metrics.missed != null) && (
            <div className="pt-1">
              <div className="flex items-center justify-between mb-2">
                <span className="panel-label">Session Summary</span>
              </div>
              <div className="rounded-xl p-3 space-y-2"
                style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)' }}>
                {[
                  ['Served',    metrics.served  ?? 0, '#10b981'],
                  ['Missed',    metrics.missed  ?? 0, '#ef4444'],
                  ['Critical',  metrics.critical_served ?? 0, '#f97316'],
                  ['Avg RT',    `${(metrics.avg_response_time ?? 0).toFixed(1)}s`, '#3b82f6'],
                  ['Idle %',    `${((metrics.idle_fraction ?? 0) * 100).toFixed(0)}%`, '#8b5cf6'],
                ].map(([label, value, color]) => (
                  <div key={label} className="flex items-center justify-between">
                    <span className="panel-label">{label}</span>
                    <span className="text-xs font-mono font-bold" style={{ color }}>{value}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </aside>

        {/* ── CENTER ───────────────────────────────────────── */}
        <div className="flex flex-1 flex-col overflow-hidden">
          {/* Map */}
          <div className="flex-1 relative min-h-0 p-3">
            <CityMap
              ambulances={obs.ambulances}
              emergencies={obs.emergencies}
              hospitals={obs.hospitals}
              traffic={obs.traffic}
              graphSize={100}
            />
          </div>
          {/* Reward Chart */}
          <div className="flex-none h-[180px] px-3 pb-3">
            <RewardChart history={rewardHistory} rubric={obs.rubric} />
          </div>
        </div>

        {/* ── RIGHT PANEL ──────────────────────────────────── */}
        <aside className="w-[280px] flex-none flex flex-col overflow-hidden"
          style={{ borderLeft: '1px solid rgba(255,255,255,0.06)' }}>
          <div className="flex items-center justify-between px-4 pt-4 pb-3 flex-none">
            <span className="panel-label">Live Incident Feed</span>
            <span
                className="text-xs font-bold px-2 py-0.5 rounded-full"
                style={{ background: activeIncidents > 0 ? 'rgba(239,68,68,0.15)' : 'rgba(16,185,129,0.1)', color: activeIncidents > 0 ? '#f87171' : '#34d399', border: `1px solid ${activeIncidents > 0 ? 'rgba(239,68,68,0.3)' : 'rgba(16,185,129,0.2)'}` }}>
                {activeIncidents} active
              </span>
          </div>

          <div className="flex-1 overflow-y-auto px-4 pb-4 space-y-2">
            
              {obs.emergencies.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-16 gap-3">
                  <div className="text-4xl opacity-20">📡</div>
                  <span className="panel-label text-center">Scanning frequencies...<br />No active incidents</span>
                </div>
              ) : obs.emergencies.map((e) => {
                const sevConfig = {
                  CRITICAL: { color: '#ef4444', bg: 'rgba(239,68,68,0.08)', border: 'rgba(239,68,68,0.25)', glow: 'rgba(239,68,68,0.15)' },
                  HIGH:     { color: '#f97316', bg: 'rgba(249,115,22,0.08)', border: 'rgba(249,115,22,0.25)', glow: 'rgba(249,115,22,0.1)' },
                  NORMAL:   { color: '#10b981', bg: 'rgba(16,185,129,0.06)', border: 'rgba(16,185,129,0.2)',  glow: 'rgba(16,185,129,0.08)' },
                }[e.severity] || { color: '#94a3b8', bg: 'rgba(148,163,184,0.05)', border: 'rgba(148,163,184,0.15)', glow: 'transparent' }
                const urgent = e.time_remaining != null && e.time_remaining < 15

                return (
                  <div key={e.id}
                    className="rounded-xl p-3 relative overflow-hidden"
                    style={{ background: sevConfig.bg, border: `1px solid ${sevConfig.border}`, boxShadow: `0 0 16px ${sevConfig.glow}` }}>
                    {/* Severity bar */}
                    <div className="absolute left-0 top-0 bottom-0 w-[3px] rounded-l-xl" style={{ background: sevConfig.color }} />
                    <div className="pl-2">
                      <div className="flex items-center justify-between mb-1.5">
                        <span className="text-xs font-bold" style={{ color: 'rgba(148,163,184,0.8)', fontFamily: 'var(--font-mono)' }}>INC #{e.id}</span>
                        <span className="text-[10px] font-black px-2 py-0.5 rounded-full" style={{ background: `${sevConfig.color}22`, color: sevConfig.color }}>
                          {e.severity}
                        </span>
                      </div>
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-1.5">
                          <svg className="w-3 h-3" style={{ color: 'rgba(148,163,184,0.4)' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                          </svg>
                          <span className="text-xs" style={{ color: 'rgba(148,163,184,0.6)' }}>Node <span className="font-mono font-bold" style={{ color: '#e2e8f0' }}>{e.node}</span></span>
                        </div>
                        {e.time_remaining != null && (
                          <span className={`text-xs font-mono font-bold ${urgent ? 'animate-pulse' : ''}`}
                            style={{ color: urgent ? '#ef4444' : sevConfig.color }}>
                            {e.time_remaining}s
                          </span>
                        )}
                      </div>
                      {/* Time bar */}
                      {e.time_remaining != null && e.max_time_remaining != null && (
                        <div className="mt-2 h-1 rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.05)' }}>
                          <div className="h-full rounded-full transition-all duration-300"
                            style={{ width: `${(e.time_remaining / e.max_time_remaining) * 100}%`, background: urgent ? '#ef4444' : sevConfig.color }} />
                        </div>
                      )}
                    </div>
                  </div>
                )
              })}
          </div>
        </aside>
      </div>
      )}
    </div>
  )
}

