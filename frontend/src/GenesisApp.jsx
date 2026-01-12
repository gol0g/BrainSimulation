/**
 * Genesis Brain - Free Energy Principle Visualization
 *
 * This UI shows the MECHANICAL quantities that drive behavior.
 * NO emotion labels exist inside the system.
 * Observer interpretations are shown separately and clearly marked as EXTERNAL.
 */

import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Activity } from 'lucide-react';

const API_URL = 'http://127.0.0.1:8002';  // localhost 대신 IP 사용 (Windows DNS 지연 방지)

// World Map Component
const WorldMap = ({ world }) => {
  if (!world) return null;

  const gridSize = 10;
  const cellSize = 22;

  return (
    <div style={{
      display: 'inline-grid',
      gridTemplateColumns: `repeat(${gridSize}, ${cellSize}px)`,
      gap: '1px',
      background: '#222',
      padding: '1px',
      borderRadius: '6px'
    }}>
      {Array.from({ length: gridSize * gridSize }).map((_, idx) => {
        const x = idx % gridSize;
        const y = Math.floor(idx / gridSize);

        const isAgent = world.agent_pos[0] === x && world.agent_pos[1] === y;
        const isFood = world.food_pos[0] === x && world.food_pos[1] === y;
        const isDanger = world.danger_pos[0] === x && world.danger_pos[1] === y;

        let bg = '#0a0a0a';
        let content = '';
        let color = '#333';

        if (isAgent && isFood) {
          bg = '#004400';
          content = '🟢';
        } else if (isAgent && isDanger) {
          bg = '#440000';
          content = '💀';
        } else if (isAgent) {
          bg = '#001a1a';
          content = '◆';
          color = '#00f3ff';
        } else if (isFood) {
          bg = '#002200';
          content = '●';
          color = '#00ff88';
        } else if (isDanger) {
          bg = '#220000';
          content = '▲';
          color = '#ff3e3e';
        }

        return (
          <div
            key={idx}
            style={{
              width: cellSize,
              height: cellSize,
              background: bg,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: isAgent ? '14px' : '10px',
              color
            }}
          >
            {content}
          </div>
        );
      })}
    </div>
  );
};

// Simple line chart for F history
const FHistoryChart = ({ data, height = 80 }) => {
  if (!data || data.length === 0) return null;

  const width = 280;
  const padding = 10;
  const maxF = Math.max(...data, 10);
  const minF = Math.min(...data, 0);
  const range = maxF - minF || 1;

  const points = data.map((f, i) => {
    const x = padding + (i / (data.length - 1 || 1)) * (width - 2 * padding);
    const y = height - padding - ((f - minF) / range) * (height - 2 * padding);
    return `${x},${y}`;
  }).join(' ');

  return (
    <svg width={width} height={height} style={{ background: '#0a0a0a', borderRadius: '6px' }}>
      <polyline
        points={points}
        fill="none"
        stroke="#00f3ff"
        strokeWidth="2"
      />
      <text x={5} y={15} fill="#666" fontSize="10">{maxF.toFixed(1)}</text>
      <text x={5} y={height - 5} fill="#666" fontSize="10">{minF.toFixed(1)}</text>
    </svg>
  );
};

// G decomposition bar
const GBar = ({ action, g, risk, ambiguity, complexity, isSelected, isBest }) => {
  const actionLabels = { 0: '정지', 1: '↑', 2: '↓', 3: '←', 4: '→' };
  const maxG = 15; // For scaling

  return (
    <div style={{
      padding: '8px',
      background: isSelected ? '#001a1a' : '#0a0a0a',
      border: `1px solid ${isSelected ? '#00f3ff' : isBest ? '#00ff88' : '#222'}`,
      borderRadius: '6px',
      marginBottom: '6px'
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
        <span style={{
          color: isSelected ? '#00f3ff' : isBest ? '#00ff88' : '#888',
          fontWeight: isSelected || isBest ? 'bold' : 'normal',
          fontSize: '0.75rem'
        }}>
          {actionLabels[action]} {isSelected && '(선택됨)'} {isBest && !isSelected && '(최적)'}
        </span>
        <span style={{ color: '#fff', fontSize: '0.8rem', fontWeight: 'bold' }}>
          G = {g.toFixed(2)}
        </span>
      </div>

      {/* Risk bar */}
      <div style={{ marginBottom: '4px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.6rem', color: '#ff6b6b' }}>
          <span>위험 (Risk)</span>
          <span>{risk.toFixed(2)}</span>
        </div>
        <div style={{ height: '5px', background: '#111', borderRadius: '3px', overflow: 'hidden' }}>
          <div style={{
            height: '100%',
            width: `${Math.min(100, (risk / maxG) * 100)}%`,
            background: '#ff6b6b',
            transition: 'width 0.2s'
          }} />
        </div>
      </div>

      {/* Ambiguity bar */}
      <div style={{ marginBottom: '4px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.6rem', color: '#bc13fe' }}>
          <span>모호함 (Ambiguity)</span>
          <span>{ambiguity.toFixed(2)}</span>
        </div>
        <div style={{ height: '5px', background: '#111', borderRadius: '3px', overflow: 'hidden' }}>
          <div style={{
            height: '100%',
            width: `${Math.min(100, (ambiguity / maxG) * 100)}%`,
            background: '#bc13fe',
            transition: 'width 0.2s'
          }} />
        </div>
      </div>

      {/* Complexity bar */}
      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.6rem', color: '#00bfff' }}>
          <span>복잡도 (Complexity)</span>
          <span>{(complexity || 0).toFixed(2)}</span>
        </div>
        <div style={{ height: '5px', background: '#111', borderRadius: '3px', overflow: 'hidden' }}>
          <div style={{
            height: '100%',
            width: `${Math.min(100, ((complexity || 0) / maxG) * 100)}%`,
            background: '#00bfff',
            transition: 'width 0.2s'
          }} />
        </div>
      </div>
    </div>
  );
};

// Scenario Panel Component
const ScenarioPanel = ({ currentScenario, onStart, onStop }) => {
  const [scenarios, setScenarios] = useState([]);
  const [selectedScenario, setSelectedScenario] = useState('conflict');
  const [duration, setDuration] = useState(100);
  const [result, setResult] = useState(null);

  // G1 Gate (DRIFT) specific parameters
  const [driftAfter, setDriftAfter] = useState(100);
  const [driftType, setDriftType] = useState('rotate');
  const driftTypes = ['rotate', 'flip_x', 'flip_y', 'reverse'];

  useEffect(() => {
    // Fetch available scenarios
    axios.get(`${API_URL}/scenarios`).then(res => {
      setScenarios(res.data.scenarios || []);
    }).catch(() => {});
  }, []);

  const handleStart = async () => {
    try {
      let url = `${API_URL}/scenario/start/${selectedScenario}?duration=${duration}`;
      // Add DRIFT-specific parameters for G1 Gate
      if (selectedScenario === 'drift') {
        url += `&drift_after=${driftAfter}&drift_type=${driftType}`;
      }
      await axios.post(url);
      setResult(null);
      onStart?.();
    } catch (e) {
      console.error('Failed to start scenario:', e);
    }
  };

  const handleStop = async () => {
    try {
      const res = await axios.post(`${API_URL}/scenario/stop`);
      setResult(res.data);
      onStop?.();
    } catch (e) {
      console.error('Failed to stop scenario:', e);
    }
  };

  return (
    <div style={{
      padding: '15px',
      background: '#0a0a0a',
      borderRadius: '8px',
      border: `1px solid ${currentScenario ? '#ff6b00' : '#222'}`,
      marginBottom: '15px'
    }}>
      <h3 style={{ margin: '0 0 10px 0', color: '#ff6b00', fontSize: '0.8rem' }}>
        테스트 시나리오
      </h3>

      {!currentScenario ? (
        <div>
          <div style={{ marginBottom: '10px' }}>
            <select
              value={selectedScenario}
              onChange={e => setSelectedScenario(e.target.value)}
              style={{
                width: '100%',
                padding: '8px',
                background: '#111',
                border: '1px solid #333',
                color: '#fff',
                borderRadius: '4px',
                fontSize: '0.75rem'
              }}
            >
              {scenarios.map(s => (
                <option key={s.id} value={s.id}>{s.name}</option>
              ))}
            </select>
          </div>
          <div style={{ marginBottom: '10px', fontSize: '0.65rem', color: '#666' }}>
            {scenarios.find(s => s.id === selectedScenario)?.description}
          </div>
          <div style={{ marginBottom: '10px', display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span style={{ fontSize: '0.7rem', color: '#888' }}>스텝:</span>
            <input
              type="number"
              value={duration}
              onChange={e => setDuration(parseInt(e.target.value) || 100)}
              style={{
                width: '80px',
                padding: '5px',
                background: '#111',
                border: '1px solid #333',
                color: '#fff',
                borderRadius: '4px',
                fontSize: '0.75rem'
              }}
            />
          </div>

          {/* G1 Gate (DRIFT) specific parameters */}
          {selectedScenario === 'drift' && (
            <div style={{
              padding: '10px',
              background: '#111',
              borderRadius: '6px',
              marginBottom: '10px',
              border: '1px solid #ff6b00'
            }}>
              <div style={{ fontSize: '0.65rem', color: '#ff6b00', marginBottom: '8px' }}>
                G1 Gate: Drift 파라미터
              </div>
              <div style={{ display: 'flex', gap: '10px', marginBottom: '8px' }}>
                <div style={{ flex: 1 }}>
                  <span style={{ fontSize: '0.65rem', color: '#888' }}>드리프트 시작:</span>
                  <input
                    type="number"
                    value={driftAfter}
                    onChange={e => setDriftAfter(parseInt(e.target.value) || 100)}
                    style={{
                      width: '100%',
                      padding: '5px',
                      background: '#0a0a0a',
                      border: '1px solid #333',
                      color: '#fff',
                      borderRadius: '4px',
                      fontSize: '0.75rem',
                      marginTop: '4px'
                    }}
                  />
                </div>
                <div style={{ flex: 1 }}>
                  <span style={{ fontSize: '0.65rem', color: '#888' }}>드리프트 타입:</span>
                  <select
                    value={driftType}
                    onChange={e => setDriftType(e.target.value)}
                    style={{
                      width: '100%',
                      padding: '5px',
                      background: '#0a0a0a',
                      border: '1px solid #333',
                      color: '#fff',
                      borderRadius: '4px',
                      fontSize: '0.75rem',
                      marginTop: '4px'
                    }}
                  >
                    {driftTypes.map(t => (
                      <option key={t} value={t}>{t}</option>
                    ))}
                  </select>
                </div>
              </div>
              <div style={{ fontSize: '0.6rem', color: '#666' }}>
                {driftAfter} 스텝 후 환경 다이나믹스가 변경됩니다.
              </div>
            </div>
          )}

          <button
            onClick={handleStart}
            style={{
              width: '100%',
              padding: '10px',
              background: '#ff6b00',
              border: 'none',
              color: '#000',
              borderRadius: '4px',
              cursor: 'pointer',
              fontWeight: 'bold',
              fontSize: '0.8rem'
            }}
          >
            시나리오 시작
          </button>
        </div>
      ) : (
        <div>
          <div style={{
            padding: '10px',
            background: '#111',
            borderRadius: '6px',
            marginBottom: '10px'
          }}>
            <div style={{ fontSize: '0.7rem', color: '#888', marginBottom: '5px' }}>
              실행 중: <span style={{ color: '#ff6b00' }}>{currentScenario.type}</span>
            </div>
            <div style={{ fontSize: '0.8rem', color: '#fff' }}>
              진행: {currentScenario.progress} / {currentScenario.duration}
            </div>
            <div style={{
              height: '6px',
              background: '#222',
              borderRadius: '3px',
              marginTop: '8px',
              overflow: 'hidden'
            }}>
              <div style={{
                height: '100%',
                width: `${(currentScenario.progress / currentScenario.duration) * 100}%`,
                background: currentScenario.complete ? '#00ff88' : '#ff6b00',
                transition: 'width 0.2s'
              }} />
            </div>
          </div>
          <button
            onClick={handleStop}
            style={{
              width: '100%',
              padding: '10px',
              background: currentScenario.complete ? '#00ff88' : '#ff3e3e',
              border: 'none',
              color: '#000',
              borderRadius: '4px',
              cursor: 'pointer',
              fontWeight: 'bold',
              fontSize: '0.8rem'
            }}
          >
            {currentScenario.complete ? '결과 보기' : '중단'}
          </button>
        </div>
      )}

      {/* Show result */}
      {result && result.summary && (
        <div style={{
          marginTop: '15px',
          padding: '10px',
          background: '#001a00',
          borderRadius: '6px',
          border: '1px solid #00ff88'
        }}>
          <div style={{ fontSize: '0.75rem', color: '#00ff88', marginBottom: '8px', fontWeight: 'bold' }}>
            결과: {result.scenario}
          </div>
          <div style={{ fontSize: '0.65rem', color: '#aaa', lineHeight: 1.6 }}>
            <div>음식: {result.summary.food_eaten} | 위험: {result.summary.danger_hits}</div>
            <div>평균 F: {result.summary.avg_F} | Risk: {result.summary.avg_risk}</div>
            <div>Ambiguity: {result.summary.avg_ambiguity} | Complexity: {result.summary.avg_complexity}</div>
            <div>진동: {result.summary.oscillation_count} | 회복: {result.summary.recovery_events}</div>
            <div style={{ marginTop: '5px', color: result.analysis?.is_principle_driven ? '#00ff88' : '#ff6b6b' }}>
              원리 기반: {result.analysis?.is_principle_driven ? 'YES' : 'NO'}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Toggle Switch Component
const ToggleSwitch = ({ enabled, onChange, label, color = '#00f3ff' }) => (
  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
    <span style={{ fontSize: '0.7rem', color: '#888' }}>{label}</span>
    <button
      onClick={onChange}
      style={{
        width: '40px',
        height: '20px',
        borderRadius: '10px',
        border: 'none',
        background: enabled ? color : '#333',
        position: 'relative',
        cursor: 'pointer',
        transition: 'background 0.2s'
      }}
    >
      <div style={{
        width: '16px',
        height: '16px',
        borderRadius: '50%',
        background: '#fff',
        position: 'absolute',
        top: '2px',
        left: enabled ? '22px' : '2px',
        transition: 'left 0.2s'
      }} />
    </button>
  </div>
);

// v5.13 Ops Monitor Panel
const OpsMonitorPanel = () => {
  const [expanded, setExpanded] = useState(false);
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [pcZEnabled, setPcZEnabled] = useState(false);

  const fetchDashboard = async () => {
    setLoading(true);
    try {
      const res = await axios.get(`${API_URL}/pc_z/ops/dashboard`);
      setDashboardData(res.data);
    } catch (e) {
      console.error('Failed to fetch ops dashboard:', e);
    } finally {
      setLoading(false);
    }
  };

  const togglePcZ = async () => {
    try {
      if (pcZEnabled) {
        await axios.post(`${API_URL}/pc_z/disable`);
        setPcZEnabled(false);
      } else {
        await axios.post(`${API_URL}/pc_z/enable`);
        setPcZEnabled(true);
      }
    } catch (e) {
      console.error('Failed to toggle PC-Z:', e);
    }
  };

  useEffect(() => {
    if (expanded) {
      fetchDashboard();
      const interval = setInterval(fetchDashboard, 5000); // Refresh every 5s
      return () => clearInterval(interval);
    }
  }, [expanded]);

  const getStageColor = (stage) => {
    switch (stage) {
      case 'HEALTHY': return '#00ff88';
      case 'WARNING': return '#ffaa00';
      case 'UPGRADE_CANDIDATE': return '#ff6b6b';
      case 'UPGRADE_CONFIRMED': return '#ff0000';
      default: return '#888';
    }
  };

  const getStageIcon = (stage) => {
    switch (stage) {
      case 'HEALTHY': return '✓';
      case 'WARNING': return '⚠';
      case 'UPGRADE_CANDIDATE': return '⬆';
      case 'UPGRADE_CONFIRMED': return '🚨';
      default: return '?';
    }
  };

  return (
    <div style={{
      padding: '15px',
      background: '#0a0a0a',
      borderRadius: '8px',
      border: `1px solid ${dashboardData?.v514_trigger?.stage === 'UPGRADE_CONFIRMED' ? '#ff0000' : '#00f3ff'}`,
      marginBottom: '15px'
    }}>
      <div
        onClick={() => setExpanded(!expanded)}
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          cursor: 'pointer'
        }}
      >
        <h3 style={{ margin: 0, color: '#00f3ff', fontSize: '0.8rem' }}>
          📊 v5.13 Ops Monitor
        </h3>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          {dashboardData && (
            <span style={{
              padding: '2px 8px',
              borderRadius: '4px',
              fontSize: '0.6rem',
              background: getStageColor(dashboardData.v514_trigger?.stage) + '22',
              color: getStageColor(dashboardData.v514_trigger?.stage)
            }}>
              {getStageIcon(dashboardData.v514_trigger?.stage)} {dashboardData.v514_trigger?.stage}
            </span>
          )}
          <span style={{ color: '#666' }}>{expanded ? '▼' : '▶'}</span>
        </div>
      </div>

      {expanded && (
        <div style={{ marginTop: '15px' }}>
          {/* PC-Z Toggle */}
          <div style={{ marginBottom: '15px' }}>
            <ToggleSwitch
              enabled={pcZEnabled}
              onChange={togglePcZ}
              label="PC-Z Bridge (데이터 수집)"
              color="#00f3ff"
            />
          </div>

          {loading && !dashboardData ? (
            <div style={{ textAlign: 'center', color: '#666', fontSize: '0.7rem' }}>
              로딩 중...
            </div>
          ) : dashboardData ? (
            <div>
              {/* 4-Panel Dashboard */}
              <div style={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: '8px',
                marginBottom: '12px'
              }}>
                {/* Early Recovery Rate */}
                <div style={{
                  padding: '10px',
                  background: '#111',
                  borderRadius: '6px',
                  textAlign: 'center'
                }}>
                  <div style={{ fontSize: '0.55rem', color: '#666', marginBottom: '4px' }}>
                    Early Recovery Rate
                  </div>
                  <div style={{
                    fontSize: '1.1rem',
                    fontWeight: 'bold',
                    color: dashboardData.dashboard.early_recovery_rate > 0.1 ? '#ff6b6b' : '#00ff88'
                  }}>
                    {(dashboardData.dashboard.early_recovery_rate * 100).toFixed(1)}%
                  </div>
                </div>

                {/* Bad Phase Pattern Rate */}
                <div style={{
                  padding: '10px',
                  background: '#111',
                  borderRadius: '6px',
                  textAlign: 'center'
                }}>
                  <div style={{ fontSize: '0.55rem', color: '#666', marginBottom: '4px' }}>
                    Bad Pattern Rate
                  </div>
                  <div style={{
                    fontSize: '1.1rem',
                    fontWeight: 'bold',
                    color: dashboardData.dashboard.bad_phase_pattern_rate > 0.05 ? '#ffaa00' : '#00ff88'
                  }}>
                    {(dashboardData.dashboard.bad_phase_pattern_rate * 100).toFixed(1)}%
                  </div>
                </div>

                {/* Lag Metric */}
                <div style={{
                  padding: '10px',
                  background: '#111',
                  borderRadius: '6px',
                  textAlign: 'center'
                }}>
                  <div style={{ fontSize: '0.55rem', color: '#666', marginBottom: '4px' }}>
                    Lag Metric
                  </div>
                  <div style={{ fontSize: '1.1rem', fontWeight: 'bold', color: '#00f3ff' }}>
                    {dashboardData.dashboard.lag_metric_mean?.toFixed(1) || '0'} steps
                  </div>
                </div>

                {/* Premature Impact Rate */}
                <div style={{
                  padding: '10px',
                  background: '#111',
                  borderRadius: '6px',
                  textAlign: 'center'
                }}>
                  <div style={{ fontSize: '0.55rem', color: '#666', marginBottom: '4px' }}>
                    Premature Impact
                  </div>
                  <div style={{
                    fontSize: '1.1rem',
                    fontWeight: 'bold',
                    color: dashboardData.dashboard.premature_impact_rate > 0.1 ? '#ff6b6b' : '#00ff88'
                  }}>
                    {(dashboardData.dashboard.premature_impact_rate * 100).toFixed(1)}%
                  </div>
                </div>
              </div>

              {/* Zone-Tagged Impact */}
              {dashboardData.dashboard.zone_tagged_impact && (
                <div style={{
                  padding: '10px',
                  background: '#111',
                  borderRadius: '6px',
                  marginBottom: '12px'
                }}>
                  <div style={{ fontSize: '0.55rem', color: '#666', marginBottom: '6px' }}>
                    Zone-Tagged Impact (cost occurred in)
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-around', fontSize: '0.7rem' }}>
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ color: '#ff6b6b', fontWeight: 'bold' }}>
                        {dashboardData.dashboard.zone_tagged_impact.stable || 0}
                      </div>
                      <div style={{ fontSize: '0.5rem', color: '#888' }}>stable</div>
                    </div>
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ color: '#ffaa00', fontWeight: 'bold' }}>
                        {dashboardData.dashboard.zone_tagged_impact.transition || 0}
                      </div>
                      <div style={{ fontSize: '0.5rem', color: '#888' }}>transition</div>
                    </div>
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ color: '#00ff88', fontWeight: 'bold' }}>
                        {dashboardData.dashboard.zone_tagged_impact.shock || 0}
                      </div>
                      <div style={{ fontSize: '0.5rem', color: '#888' }}>shock</div>
                    </div>
                  </div>
                </div>
              )}

              {/* v5.14 Trigger Checklist */}
              <div style={{
                padding: '10px',
                background: '#111',
                borderRadius: '6px',
                marginBottom: '12px'
              }}>
                <div style={{ fontSize: '0.55rem', color: '#666', marginBottom: '6px' }}>
                  v5.14 Trigger Checklist
                </div>
                <div style={{ fontSize: '0.6rem' }}>
                  {dashboardData.v514_trigger?.checklist && Object.entries(dashboardData.v514_trigger.checklist).map(([key, value]) => (
                    <div key={key} style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      marginBottom: '3px'
                    }}>
                      <span style={{ color: '#888' }}>{key.replace(/_/g, ' ')}</span>
                      <span style={{ color: value ? '#00ff88' : '#666' }}>
                        {value ? '✓' : '✗'}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Status & Reason */}
              <div style={{
                padding: '10px',
                background: getStageColor(dashboardData.v514_trigger?.stage) + '11',
                borderRadius: '6px',
                border: `1px solid ${getStageColor(dashboardData.v514_trigger?.stage)}33`
              }}>
                <div style={{
                  fontSize: '0.7rem',
                  fontWeight: 'bold',
                  color: getStageColor(dashboardData.v514_trigger?.stage),
                  marginBottom: '4px'
                }}>
                  {getStageIcon(dashboardData.v514_trigger?.stage)} {dashboardData.v514_trigger?.stage}
                </div>
                <div style={{ fontSize: '0.55rem', color: '#888' }}>
                  {dashboardData.v514_trigger?.reason}
                </div>
              </div>

              {/* Sample Count */}
              <div style={{
                marginTop: '10px',
                fontSize: '0.5rem',
                color: '#555',
                textAlign: 'right'
              }}>
                Samples: {dashboardData.sample_count || 0} | Last updated: {new Date().toLocaleTimeString()}
              </div>
            </div>
          ) : (
            <div style={{ textAlign: 'center', color: '#666', fontSize: '0.7rem' }}>
              PC-Z를 활성화하고 시뮬레이션을 실행하세요
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// Control Panel Component - v3.6
const ControlPanel = ({ state }) => {
  const [expanded, setExpanded] = useState(false);
  const [temporal, setTemporal] = useState({ enabled: false, horizon: 3, discount: 0.9 });
  const [hierarchy, setHierarchy] = useState({ enabled: false, K: 4 });
  const [think, setThink] = useState({ enabled: false });
  const [prefLearning, setPrefLearning] = useState({ enabled: false });
  const [uncertainty, setUncertainty] = useState({ enabled: false });  // v4.3
  const [memory, setMemory] = useState({ enabled: false });  // v4.0
  const [consolidation, setConsolidation] = useState({ enabled: false });  // v4.1
  const [regret, setRegret] = useState({ enabled: false });  // v4.4
  const [drift, setDrift] = useState({ enabled: false, type: 'rotate' });  // v4.5
  const [internalWeight, setInternalWeight] = useState(0.5);
  const [checkpoints, setCheckpoints] = useState([]);
  const [checkpointName, setCheckpointName] = useState('');
  const [evalResult, setEvalResult] = useState(null);

  // Sync with state from API
  useEffect(() => {
    if (state?.temporal) setTemporal(prev => ({ ...prev, enabled: state.temporal.enabled }));
    if (state?.hierarchy) setHierarchy(prev => ({ ...prev, enabled: state.hierarchy.enabled }));
    if (state?.think) setThink(prev => ({ ...prev, enabled: state.think.enabled }));
    if (state?.preference_learning) setPrefLearning(prev => ({ ...prev, enabled: state.preference_learning.enabled }));
    if (state?.uncertainty) setUncertainty(prev => ({ ...prev, enabled: state.uncertainty.enabled }));  // v4.3
    if (state?.memory) setMemory(prev => ({ ...prev, enabled: state.memory.enabled }));  // v4.0
    if (state?.consolidation) setConsolidation(prev => ({ ...prev, enabled: state.consolidation.enabled }));  // v4.1
    if (state?.regret) setRegret(prev => ({ ...prev, enabled: state.regret.enabled }));  // v4.4
  }, [state]);

  // Fetch checkpoints
  const fetchCheckpoints = async () => {
    try {
      const res = await axios.get(`${API_URL}/checkpoint/list`);
      setCheckpoints(res.data || []);
    } catch (e) {}
  };

  useEffect(() => {
    if (expanded) fetchCheckpoints();
  }, [expanded]);

  // Handlers
  const toggleTemporal = async () => {
    try {
      if (temporal.enabled) {
        await axios.post(`${API_URL}/temporal/disable`);
        setTemporal(prev => ({ ...prev, enabled: false }));
      } else {
        await axios.post(`${API_URL}/temporal/enable`, null, {
          params: { horizon: temporal.horizon, discount: temporal.discount }
        });
        setTemporal(prev => ({ ...prev, enabled: true }));
      }
    } catch (e) { console.error(e); }
  };

  const toggleHierarchy = async () => {
    try {
      if (hierarchy.enabled) {
        await axios.post(`${API_URL}/hierarchy/disable`);
        setHierarchy(prev => ({ ...prev, enabled: false }));
      } else {
        await axios.post(`${API_URL}/hierarchy/enable`, null, { params: { K: hierarchy.K } });
        setHierarchy(prev => ({ ...prev, enabled: true }));
      }
    } catch (e) { console.error(e); }
  };

  const toggleThink = async () => {
    try {
      if (think.enabled) {
        await axios.post(`${API_URL}/think/disable`);
        setThink(prev => ({ ...prev, enabled: false }));
      } else {
        await axios.post(`${API_URL}/think/enable`);
        setThink(prev => ({ ...prev, enabled: true }));
      }
    } catch (e) { console.error(e); }
  };

  const togglePrefLearning = async () => {
    try {
      if (prefLearning.enabled) {
        await axios.post(`${API_URL}/preference/learning/disable`);
        setPrefLearning(prev => ({ ...prev, enabled: false }));
      } else {
        await axios.post(`${API_URL}/preference/learning/enable`);
        setPrefLearning(prev => ({ ...prev, enabled: true }));
      }
    } catch (e) { console.error(e); }
  };

  // v4.3: Uncertainty toggle
  const toggleUncertainty = async () => {
    try {
      if (uncertainty.enabled) {
        await axios.post(`${API_URL}/uncertainty/disable`);
        setUncertainty(prev => ({ ...prev, enabled: false }));
      } else {
        await axios.post(`${API_URL}/uncertainty/enable`);
        setUncertainty(prev => ({ ...prev, enabled: true }));
      }
    } catch (e) { console.error(e); }
  };

  // v4.0: Memory toggle
  const toggleMemory = async () => {
    try {
      if (memory.enabled) {
        await axios.post(`${API_URL}/memory/disable`);
        setMemory(prev => ({ ...prev, enabled: false }));
      } else {
        await axios.post(`${API_URL}/memory/enable`);
        setMemory(prev => ({ ...prev, enabled: true }));
      }
    } catch (e) { console.error(e); }
  };

  // v4.1: Consolidation toggle
  const toggleConsolidation = async () => {
    try {
      if (consolidation.enabled) {
        await axios.post(`${API_URL}/consolidation/disable`);
        setConsolidation(prev => ({ ...prev, enabled: false }));
      } else {
        await axios.post(`${API_URL}/consolidation/enable`);
        setConsolidation(prev => ({ ...prev, enabled: true }));
      }
    } catch (e) { console.error(e); }
  };

  // v4.1: Manual sleep trigger
  const triggerSleep = async () => {
    try {
      const res = await axios.post(`${API_URL}/consolidation/trigger`);
      console.log('Sleep result:', res.data);
    } catch (e) { console.error(e); }
  };

  // v4.4: Regret toggle
  const toggleRegret = async () => {
    try {
      if (regret.enabled) {
        await axios.post(`${API_URL}/regret/disable`);
        setRegret(prev => ({ ...prev, enabled: false }));
      } else {
        await axios.post(`${API_URL}/regret/enable`);
        setRegret(prev => ({ ...prev, enabled: true }));
      }
    } catch (e) { console.error(e); }
  };

  // v4.5: Drift toggle
  const toggleDrift = async () => {
    try {
      if (drift.enabled) {
        await axios.post(`${API_URL}/drift/disable`);
        setDrift(prev => ({ ...prev, enabled: false }));
      } else {
        await axios.post(`${API_URL}/drift/enable`, null, {
          params: { drift_type: drift.type }
        });
        setDrift(prev => ({ ...prev, enabled: true }));
      }
    } catch (e) { console.error(e); }
  };

  // v4.5: Change drift type
  const changeDriftType = async (newType) => {
    setDrift(prev => ({ ...prev, type: newType }));
    if (drift.enabled) {
      try {
        await axios.post(`${API_URL}/drift/enable`, null, {
          params: { drift_type: newType }
        });
      } catch (e) { console.error(e); }
    }
  };

  const updateInternalWeight = async (val) => {
    try {
      await axios.post(`${API_URL}/preference/internal_weight`, null, { params: { weight: val } });
      setInternalWeight(val);
    } catch (e) { console.error(e); }
  };

  const saveCheckpoint = async () => {
    if (!checkpointName) return;
    try {
      await axios.post(`${API_URL}/checkpoint/save`, null, {
        params: { filename: `${checkpointName}.json`, description: `Saved from UI` }
      });
      setCheckpointName('');
      fetchCheckpoints();
    } catch (e) { console.error(e); }
  };

  const loadCheckpoint = async (filename) => {
    try {
      await axios.post(`${API_URL}/checkpoint/load`, null, { params: { filename } });
    } catch (e) { console.error(e); }
  };

  const runEvaluation = async () => {
    try {
      const res = await axios.post(`${API_URL}/evaluate`, null, {
        params: { n_episodes: 5, max_steps: 100 }
      });
      setEvalResult(res.data);
    } catch (e) { console.error(e); }
  };

  return (
    <div style={{
      padding: '15px',
      background: '#0a0a0a',
      borderRadius: '8px',
      border: '1px solid #00f3ff',
      marginBottom: '15px'
    }}>
      <div
        onClick={() => setExpanded(!expanded)}
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          cursor: 'pointer'
        }}
      >
        <h3 style={{ margin: 0, color: '#00f3ff', fontSize: '0.8rem' }}>
          ⚙️ 제어판 (v4.1)
        </h3>
        <span style={{ color: '#666' }}>{expanded ? '▼' : '▶'}</span>
      </div>

      {expanded && (
        <div style={{ marginTop: '15px' }}>
          {/* Quick Stats */}
          {state?.world && (
            <div style={{
              display: 'grid',
              gridTemplateColumns: '1fr 1fr',
              gap: '10px',
              marginBottom: '15px',
              padding: '10px',
              background: '#111',
              borderRadius: '6px'
            }}>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '1.2rem', color: '#00ff88', fontWeight: 'bold' }}>
                  {state.world.total_food || 0}
                </div>
                <div style={{ fontSize: '0.6rem', color: '#666' }}>총 음식</div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '1.2rem', color: '#ff3e3e', fontWeight: 'bold' }}>
                  {state.world.total_deaths || 0}
                </div>
                <div style={{ fontSize: '0.6rem', color: '#666' }}>총 사망</div>
              </div>
            </div>
          )}

          {/* Feature Toggles */}
          <div style={{ marginBottom: '15px' }}>
            <div style={{ fontSize: '0.65rem', color: '#555', marginBottom: '8px' }}>기능 토글</div>
            <ToggleSwitch
              enabled={temporal.enabled}
              onChange={toggleTemporal}
              label="Temporal (Rollout)"
              color="#bc13fe"
            />
            <ToggleSwitch
              enabled={hierarchy.enabled}
              onChange={toggleHierarchy}
              label="Hierarchy (Context)"
              color="#ff6b00"
            />
            <ToggleSwitch
              enabled={think.enabled}
              onChange={toggleThink}
              label="THINK Action"
              color="#00bfff"
            />
            <ToggleSwitch
              enabled={prefLearning.enabled}
              onChange={togglePrefLearning}
              label="Preference Learning"
              color="#00ff88"
            />
            <ToggleSwitch
              enabled={uncertainty.enabled}
              onChange={toggleUncertainty}
              label="Uncertainty (v4.3)"
              color="#ff00ff"
            />
            <ToggleSwitch
              enabled={memory.enabled}
              onChange={toggleMemory}
              label="Memory (v4.0)"
              color="#ffa500"
            />
            <ToggleSwitch
              enabled={consolidation.enabled}
              onChange={toggleConsolidation}
              label="Sleep (v4.1)"
              color="#9370db"
            />
            <ToggleSwitch
              enabled={regret.enabled}
              onChange={toggleRegret}
              label="Regret (v4.4)"
              color="#ff6b6b"
            />
            <ToggleSwitch
              enabled={drift.enabled}
              onChange={toggleDrift}
              label="Drift (v4.5)"
              color="#ffd700"
            />
          </div>

          {/* Drift Control - v4.5 */}
          <div style={{
            marginBottom: '15px',
            padding: '10px',
            background: '#111',
            borderRadius: '6px',
            border: drift.enabled ? '1px solid #ffd70066' : '1px solid #333'
          }}>
            <div style={{ fontSize: '0.65rem', color: '#ffd700', marginBottom: '8px' }}>
              환경 Drift (v4.5)
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
              {['rotate', 'flip_x', 'flip_y', 'reverse', 'probabilistic', 'delayed'].map(type => (
                <button
                  key={type}
                  onClick={() => changeDriftType(type)}
                  style={{
                    padding: '4px 8px',
                    fontSize: '0.5rem',
                    background: drift.type === type ? '#ffd700' : '#222',
                    color: drift.type === type ? '#000' : '#888',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer'
                  }}
                >
                  {type}
                </button>
              ))}
            </div>
            {drift.enabled && (
              <div style={{ marginTop: '8px', fontSize: '0.5rem', color: '#ffd700' }}>
                ACTIVE: 행동 매핑이 변경됨
              </div>
            )}
            {state?.outcome?.action_modified && (
              <div style={{ marginTop: '4px', fontSize: '0.45rem', color: '#ff3e3e' }}>
                이번 스텝: action이 drift로 변경됨
              </div>
            )}
          </div>

          {/* Uncertainty Display - v4.3 */}
          {uncertainty.enabled && state?.uncertainty?.state && (
            <div style={{
              marginBottom: '15px',
              padding: '10px',
              background: '#111',
              borderRadius: '6px',
              border: '1px solid #ff00ff33'
            }}>
              <div style={{ fontSize: '0.65rem', color: '#ff00ff', marginBottom: '8px' }}>
                불확실성 상태
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                <div>
                  <div style={{ fontSize: '0.6rem', color: '#666' }}>Global</div>
                  <div style={{ fontSize: '0.9rem', color: '#fff' }}>
                    {(state.uncertainty.state.global_uncertainty * 100).toFixed(0)}%
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: '0.6rem', color: '#666' }}>Top Factor</div>
                  <div style={{ fontSize: '0.75rem', color: '#ff00ff' }}>
                    {state.uncertainty.state.top_factor}
                  </div>
                </div>
              </div>
              {/* Component bars */}
              <div style={{ marginTop: '8px' }}>
                {['belief', 'action', 'model', 'surprise'].map(key => (
                  <div key={key} style={{ marginBottom: '4px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.55rem', color: '#888' }}>
                      <span>{key}</span>
                      <span>{(state.uncertainty.state.components[key] * 100).toFixed(0)}%</span>
                    </div>
                    <div style={{ height: '3px', background: '#222', borderRadius: '2px', overflow: 'hidden' }}>
                      <div style={{
                        height: '100%',
                        width: `${state.uncertainty.state.components[key] * 100}%`,
                        background: state.uncertainty.state.top_factor === key ? '#ff00ff' : '#666',
                        transition: 'width 0.3s'
                      }} />
                    </div>
                  </div>
                ))}
              </div>
              {/* Memory Gate */}
              {state.uncertainty.modulation && (
                <div style={{ marginTop: '8px', paddingTop: '8px', borderTop: '1px solid #333' }}>
                  <div style={{ fontSize: '0.55rem', color: '#666' }}>Memory Gate</div>
                  <div style={{ height: '4px', background: '#222', borderRadius: '2px', overflow: 'hidden', marginTop: '3px' }}>
                    <div style={{
                      height: '100%',
                      width: `${state.uncertainty.modulation.memory_gate * 100}%`,
                      background: 'linear-gradient(90deg, #00ff88, #ff00ff)',
                      transition: 'width 0.3s'
                    }} />
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Memory Display - v4.0 */}
          {memory.enabled && state?.memory && (
            <div style={{
              marginBottom: '15px',
              padding: '10px',
              background: '#111',
              borderRadius: '6px',
              border: '1px solid #ffa50033'
            }}>
              <div style={{ fontSize: '0.65rem', color: '#ffa500', marginBottom: '8px' }}>
                장기 기억 (LTM)
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                <div>
                  <div style={{ fontSize: '0.6rem', color: '#666' }}>에피소드</div>
                  <div style={{ fontSize: '0.9rem', color: '#fff' }}>
                    {state.memory.stats?.total_episodes || 0}
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: '0.6rem', color: '#666' }}>병합됨</div>
                  <div style={{ fontSize: '0.9rem', color: '#ffa500' }}>
                    {state.memory.stats?.total_merged || 0}
                  </div>
                </div>
              </div>
              {/* Recall Info */}
              {state.memory.last_recall && (
                <div style={{ marginTop: '8px', paddingTop: '8px', borderTop: '1px solid #333' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.55rem', color: '#888', marginBottom: '4px' }}>
                    <span>recall_weight</span>
                    <span>{(state.memory.last_recall.recall_weight || 0).toFixed(2)}</span>
                  </div>
                  <div style={{ height: '3px', background: '#222', borderRadius: '2px', overflow: 'hidden' }}>
                    <div style={{
                      height: '100%',
                      width: `${(state.memory.last_recall.recall_weight || 0) * 100}%`,
                      background: '#ffa500',
                      transition: 'width 0.3s'
                    }} />
                  </div>
                  {state.memory.last_recall.matched_episodes > 0 && (
                    <div style={{ fontSize: '0.55rem', color: '#666', marginTop: '4px' }}>
                      {state.memory.last_recall.matched_episodes}개 유사 기억 리콜
                    </div>
                  )}
                </div>
              )}
              {/* Store Rate */}
              {state.memory.stats?.store_rate && (
                <div style={{ marginTop: '8px', fontSize: '0.55rem', color: '#666' }}>
                  저장률: {state.memory.stats.store_rate}
                </div>
              )}
            </div>
          )}

          {/* Consolidation Display - v4.1 */}
          {consolidation.enabled && state?.consolidation && (
            <div style={{
              marginBottom: '15px',
              padding: '10px',
              background: '#111',
              borderRadius: '6px',
              border: '1px solid #9370db33'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                <div style={{ fontSize: '0.65rem', color: '#9370db' }}>
                  수면/통합 (Sleep)
                </div>
                <button
                  onClick={triggerSleep}
                  style={{
                    padding: '3px 8px',
                    background: '#9370db',
                    border: 'none',
                    color: '#000',
                    borderRadius: '3px',
                    cursor: 'pointer',
                    fontSize: '0.55rem',
                    fontWeight: 'bold'
                  }}
                >
                  Sleep Now
                </button>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                <div>
                  <div style={{ fontSize: '0.6rem', color: '#666' }}>Sleeps</div>
                  <div style={{ fontSize: '0.9rem', color: '#fff' }}>
                    {state.consolidation.stats?.total_sleeps || 0}
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: '0.6rem', color: '#666' }}>Prototypes</div>
                  <div style={{ fontSize: '0.9rem', color: '#9370db' }}>
                    {state.consolidation.stats?.prototype_count || 0}
                  </div>
                </div>
              </div>
              {/* Sleep Trigger Signals */}
              {state.consolidation.trigger_signals && (
                <div style={{ marginTop: '8px', paddingTop: '8px', borderTop: '1px solid #333' }}>
                  <div style={{ fontSize: '0.55rem', color: '#666', marginBottom: '4px' }}>Sleep 트리거</div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '4px', fontSize: '0.5rem' }}>
                    <div style={{ color: state.consolidation.trigger_signals.low_surprise ? '#00ff88' : '#666' }}>
                      Low Surprise {state.consolidation.trigger_signals.low_surprise ? '✓' : '✗'}
                    </div>
                    <div style={{ color: state.consolidation.trigger_signals.high_redundancy ? '#00ff88' : '#666' }}>
                      High Merge {state.consolidation.trigger_signals.high_redundancy ? '✓' : '✗'}
                    </div>
                    <div style={{ color: state.consolidation.trigger_signals.stable_context ? '#00ff88' : '#666' }}>
                      Stable Ctx {state.consolidation.trigger_signals.stable_context ? '✓' : '✗'}
                    </div>
                  </div>
                </div>
              )}
              {/* Transition Std */}
              {state.consolidation.current_transition_std !== undefined && (
                <div style={{ marginTop: '8px', fontSize: '0.55rem', color: '#888' }}>
                  Transition σ: {state.consolidation.current_transition_std.toFixed(3)}
                </div>
              )}
              {/* Last Result */}
              {state.consolidation.last_result && state.consolidation.last_result.episodes_replayed > 0 && (
                <div style={{ marginTop: '8px', paddingTop: '8px', borderTop: '1px solid #333', fontSize: '0.5rem', color: '#666' }}>
                  Last: {state.consolidation.last_result.episodes_replayed} eps,
                  σ {state.consolidation.last_result.transition_std_before.toFixed(3)} → {state.consolidation.last_result.transition_std_after.toFixed(3)}
                </div>
              )}
            </div>
          )}

          {/* Regret Display - v4.4.1 */}
          {regret.enabled && state?.regret?.engine_status && (
            <div style={{
              marginBottom: '15px',
              padding: '10px',
              background: '#111',
              borderRadius: '6px',
              border: '1px solid #ff6b6b33'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                <div style={{ fontSize: '0.65rem', color: '#ff6b6b' }}>
                  Counterfactual + Regret (v4.4)
                </div>
                <div style={{ fontSize: '0.45rem', color: '#666' }}>
                  Optimal: {state.regret.engine_status.modulation?.optimal_basis || 'G_post'}
                </div>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '8px' }}>
                <div>
                  <div style={{ fontSize: '0.55rem', color: '#666' }}>Counterfactuals</div>
                  <div style={{ fontSize: '0.85rem', color: '#fff' }}>
                    {state.regret.engine_status.regret_state?.total_counterfactuals || 0}
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: '0.55rem', color: '#666' }}>Optimal Ratio</div>
                  <div style={{ fontSize: '0.85rem', color: '#ff6b6b' }}>
                    {((state.regret.engine_status.regret_state?.optimality_ratio || 0) * 100).toFixed(0)}%
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: '0.55rem', color: '#666' }}>Regret Z</div>
                  <div style={{
                    fontSize: '0.85rem',
                    color: (state.regret.engine_status.modulation?.regret_z || 0) > 1 ? '#ff3e3e' :
                           (state.regret.engine_status.modulation?.regret_z || 0) < -1 ? '#00ff88' : '#fff'
                  }}>
                    {(state.regret.engine_status.modulation?.regret_z || 0) > 0 ? '+' : ''}
                    {(state.regret.engine_status.modulation?.regret_z || 0).toFixed(1)}σ
                  </div>
                </div>
              </div>
              {/* Regret Values */}
              {state.regret.engine_status.last_result && (
                <div style={{ marginTop: '8px', paddingTop: '8px', borderTop: '1px solid #333' }}>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '8px', fontSize: '0.5rem' }}>
                    <div>
                      <div style={{ color: '#666' }}>Real</div>
                      <div style={{ color: '#ff6b6b' }}>
                        {(state.regret.engine_status.last_result.regret_real || 0).toFixed(3)}
                      </div>
                    </div>
                    <div>
                      <div style={{ color: '#666' }}>Pred</div>
                      <div style={{ color: '#ffaa6b' }}>
                        {(state.regret.engine_status.last_result.regret_pred || 0).toFixed(3)}
                      </div>
                    </div>
                    <div>
                      <div style={{ color: '#666' }}>Normalized</div>
                      <div style={{ color: '#fff' }}>
                        {(state.regret.engine_status.modulation?.normalized_regret || 0).toFixed(2)}
                      </div>
                    </div>
                  </div>
                  <div style={{ marginTop: '4px', fontSize: '0.45rem', color: state.regret.engine_status.last_result.choice_was_optimal ? '#00ff88' : '#ff6b6b' }}>
                    Action {state.regret.engine_status.last_result.chosen_action}
                    {state.regret.engine_status.last_result.choice_was_optimal ? ' (Optimal)' : ' (Suboptimal)'}
                  </div>
                </div>
              )}
              {/* Modulation Effects */}
              {state.regret.engine_status.modulation && (
                <div style={{ marginTop: '8px', paddingTop: '8px', borderTop: '1px solid #333' }}>
                  <div style={{ fontSize: '0.5rem', color: '#666', marginBottom: '4px' }}>연결 효과</div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '4px', fontSize: '0.45rem' }}>
                    <div>
                      <div style={{ color: '#666' }}>Memory Gate</div>
                      <div style={{ color: '#ffa500' }}>
                        +{(state.regret.engine_status.modulation.memory_gate_boost * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div>
                      <div style={{ color: '#666' }}>LR Boost</div>
                      <div style={{ color: '#00ff88' }}>
                        x{state.regret.engine_status.modulation.lr_boost_factor.toFixed(1)}
                      </div>
                    </div>
                    <div>
                      <div style={{ color: '#666' }}>THINK Benefit</div>
                      <div style={{ color: '#00f3ff' }}>
                        +{(state.regret.engine_status.modulation.think_benefit_boost * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>
                  {/* Spike with Cause */}
                  {state.regret.engine_status.modulation.is_spike && (
                    <div style={{
                      marginTop: '6px',
                      padding: '4px 6px',
                      background: '#ff3e3e22',
                      borderRadius: '4px',
                      fontSize: '0.5rem',
                      color: '#ff3e3e',
                      fontWeight: 'bold'
                    }}>
                      SPIKE: {
                        state.regret.engine_status.modulation.spike_cause === 'judgment_error' ? '판단 오류' :
                        state.regret.engine_status.modulation.spike_cause === 'model_mismatch' ? '모델 불일치' :
                        state.regret.engine_status.modulation.spike_cause === 'environment_change' ? '환경 변화' :
                        '원인 불명'
                      }
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Internal Weight Slider */}
          <div style={{ marginBottom: '15px' }}>
            <div style={{ fontSize: '0.65rem', color: '#555', marginBottom: '5px' }}>
              내부 선호 가중치: {internalWeight.toFixed(2)}
            </div>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={internalWeight}
              onChange={(e) => updateInternalWeight(parseFloat(e.target.value))}
              style={{ width: '100%' }}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.55rem', color: '#444' }}>
              <span>외부 (음식/위험)</span>
              <span>내부 (에너지/통증)</span>
            </div>
          </div>

          {/* Checkpoint Section */}
          <div style={{ marginBottom: '15px' }}>
            <div style={{ fontSize: '0.65rem', color: '#555', marginBottom: '8px' }}>체크포인트</div>
            <div style={{ display: 'flex', gap: '5px', marginBottom: '8px' }}>
              <input
                type="text"
                placeholder="이름"
                value={checkpointName}
                onChange={(e) => setCheckpointName(e.target.value)}
                style={{
                  flex: 1,
                  padding: '6px',
                  background: '#111',
                  border: '1px solid #333',
                  color: '#fff',
                  borderRadius: '4px',
                  fontSize: '0.7rem'
                }}
              />
              <button
                onClick={saveCheckpoint}
                style={{
                  padding: '6px 12px',
                  background: '#00f3ff',
                  border: 'none',
                  color: '#000',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '0.7rem',
                  fontWeight: 'bold'
                }}
              >
                저장
              </button>
            </div>
            {checkpoints.length > 0 && (
              <div style={{ maxHeight: '80px', overflowY: 'auto' }}>
                {checkpoints.map((cp, i) => (
                  <div key={i} style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    padding: '5px',
                    background: '#111',
                    borderRadius: '4px',
                    marginBottom: '4px',
                    fontSize: '0.65rem'
                  }}>
                    <span style={{ color: '#888' }}>{cp.filename}</span>
                    <button
                      onClick={() => loadCheckpoint(cp.filename)}
                      style={{
                        padding: '3px 8px',
                        background: '#333',
                        border: 'none',
                        color: '#00f3ff',
                        borderRadius: '3px',
                        cursor: 'pointer',
                        fontSize: '0.6rem'
                      }}
                    >
                      로드
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Evaluation Section */}
          <div>
            <div style={{ fontSize: '0.65rem', color: '#555', marginBottom: '8px' }}>헤드리스 평가</div>
            <button
              onClick={runEvaluation}
              style={{
                width: '100%',
                padding: '8px',
                background: '#333',
                border: '1px solid #00ff88',
                color: '#00ff88',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '0.7rem'
              }}
            >
              5 에피소드 평가 실행
            </button>
            {evalResult && (
              <div style={{
                marginTop: '8px',
                padding: '8px',
                background: '#001a00',
                borderRadius: '4px',
                fontSize: '0.65rem',
                color: '#aaa'
              }}>
                <div>에피소드: {evalResult.n_episodes}</div>
                <div>평균 음식: {evalResult.avg_food_per_episode?.toFixed(1)}</div>
                <div>생존율: {(evalResult.survival_rate * 100).toFixed(0)}%</div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

function GenesisApp() {
  const [state, setState] = useState(null);
  const [status, setStatus] = useState('OFFLINE');
  const [step, setStep] = useState(0);
  const [scenarioStatus, setScenarioStatus] = useState(null);
  const isFetchingRef = useRef(false);

  useEffect(() => {
    const intervalId = setInterval(fetchStep, 150);  // 150ms for smoother performance with all features
    return () => clearInterval(intervalId);
  }, []);

  const fetchStep = async () => {
    if (isFetchingRef.current) return;
    isFetchingRef.current = true;

    try {
      const res = await axios.post(`${API_URL}/step`, {});
      setState(res.data);
      setStep(res.data.world?.step || 0);

      // Update scenario status from step response
      if (res.data.scenario) {
        setScenarioStatus(res.data.scenario.active ? res.data.scenario : null);
      }

      if (status !== 'ONLINE') setStatus('ONLINE');
    } catch (error) {
      if (status !== 'OFFLINE') setStatus('OFFLINE');
    } finally {
      isFetchingRef.current = false;
    }
  };

  const handleReset = async () => {
    try {
      await axios.post(`${API_URL}/reset`);
      setState(null);
      setStep(0);
    } catch (e) {}
  };

  // Find best action (lowest G)
  const findBestAction = () => {
    if (!state?.action?.G) return null;
    let best = null;
    let minG = Infinity;
    Object.entries(state.action.G).forEach(([a, g]) => {
      if (g < minG) {
        minG = g;
        best = parseInt(a);
      }
    });
    return best;
  };

  const bestAction = findBestAction();

  return (
    <div style={{
      maxWidth: '1200px',
      margin: '0 auto',
      padding: '20px',
      fontFamily: 'monospace',
      color: '#fff',
      background: '#000'
    }}>
      {/* Header */}
      <header style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '20px',
        padding: '15px 20px',
        background: '#0a0a0a',
        borderRadius: '8px',
        border: '1px solid #222'
      }}>
        <div>
          <h1 style={{ margin: 0, fontSize: '1.5rem' }}>
            <span style={{ color: '#00f3ff' }}>GENESIS</span> BRAIN
          </h1>
          <p style={{ margin: '5px 0 0 0', color: '#666', fontSize: '0.7rem' }}>
            자유 에너지 원리 - 모든 것은 F 최소화에서 창발한다
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
          <span style={{ color: '#888' }}>Step: {step}</span>
          {state?.world?.phase && (
            <span style={{
              padding: '3px 8px',
              borderRadius: '4px',
              fontSize: '0.7rem',
              background: state.world.phase === 'infant' ? '#ff6b0033' : '#00ff8833',
              color: state.world.phase === 'infant' ? '#ff6b00' : '#00ff88',
            }}>
              {state.world.phase === 'infant' ? '👶 유아기' : '🧑 성인기'}
              {state.world.phase === 'infant' && ` ${Math.round(state.world.phase_progress * 100)}%`}
            </span>
          )}
          <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            <Activity size={14} color={status === 'ONLINE' ? '#0f0' : '#f00'} />
            <span style={{ color: status === 'ONLINE' ? '#0f0' : '#f00', fontSize: '0.8rem' }}>
              {status}
            </span>
          </div>
          <button
            onClick={handleReset}
            style={{
              padding: '8px 16px',
              background: '#222',
              border: '1px solid #444',
              color: '#ff6b00',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            RESET
          </button>
        </div>
      </header>

      {state && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '20px' }}>

          {/* Left Column - World & F */}
          <div>
            {/* World Map */}
            <div style={{
              padding: '15px',
              background: '#0a0a0a',
              borderRadius: '8px',
              border: '1px solid #222',
              marginBottom: '15px'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                <h3 style={{ margin: 0, color: '#888', fontSize: '0.8rem' }}>환경</h3>
                <span style={{ color: state.world.energy < 0.3 ? '#ff3e3e' : '#00ff88', fontSize: '0.8rem' }}>
                  ⚡ 에너지 {(state.world.energy * 100).toFixed(0)}%
                </span>
              </div>

              <div style={{ display: 'flex', justifyContent: 'center' }}>
                <WorldMap world={state.world} />
              </div>

              {/* Legend */}
              <div style={{ display: 'flex', justifyContent: 'center', gap: '15px', marginTop: '10px', fontSize: '0.6rem' }}>
                <span><span style={{ color: '#00f3ff' }}>◆</span> 에이전트</span>
                <span><span style={{ color: '#00ff88' }}>●</span> 음식</span>
                <span><span style={{ color: '#ff3e3e' }}>▲</span> 위험</span>
              </div>

              {/* Outcome */}
              {(state.outcome.ate_food || state.outcome.hit_danger) && (
                <div style={{
                  marginTop: '10px',
                  padding: '8px',
                  background: state.outcome.ate_food ? '#002200' : '#220000',
                  borderRadius: '4px',
                  textAlign: 'center',
                  color: state.outcome.ate_food ? '#00ff88' : '#ff3e3e',
                  fontWeight: 'bold'
                }}>
                  {state.outcome.ate_food && '🍎 음식 획득!'}
                  {state.outcome.hit_danger && '💀 위험 충돌!'}
                </div>
              )}
            </div>

            {/* Free Energy */}
            <div style={{
              padding: '15px',
              background: '#0a0a0a',
              borderRadius: '8px',
              border: '1px solid #222',
              marginBottom: '15px'
            }}>
              <h3 style={{ margin: '0 0 10px 0', color: '#00f3ff', fontSize: '0.8rem' }}>
                자유 에너지 (F)
              </h3>

              <div style={{
                fontSize: '2rem',
                fontWeight: 'bold',
                color: '#00f3ff',
                textAlign: 'center',
                marginBottom: '10px'
              }}>
                {state.free_energy.F.toFixed(3)}
              </div>

              <div style={{ display: 'flex', justifyContent: 'space-around', fontSize: '0.7rem', marginBottom: '15px' }}>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ color: '#666' }}>변화율</div>
                  <div style={{ color: state.free_energy.dF_dt < 0 ? '#00ff88' : state.free_energy.dF_dt > 0 ? '#ff3e3e' : '#888' }}>
                    {state.free_energy.dF_dt > 0 ? '+' : ''}{state.free_energy.dF_dt.toFixed(4)}
                  </div>
                </div>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ color: '#666' }}>예측 오차</div>
                  <div style={{ color: '#ff6b00' }}>{state.free_energy.prediction_error.toFixed(3)}</div>
                </div>
              </div>

              <div>
                <div style={{ fontSize: '0.6rem', color: '#666', marginBottom: '5px' }}>F 히스토리</div>
                <FHistoryChart data={state.free_energy.F_history} />
              </div>
            </div>

            {/* Derived Quantities */}
            <div style={{
              padding: '15px',
              background: '#0a0a0a',
              borderRadius: '8px',
              border: '1px solid #222'
            }}>
              <h3 style={{ margin: '0 0 10px 0', color: '#888', fontSize: '0.8rem' }}>파생 지표</h3>
              <div style={{ fontSize: '0.7rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                  <span style={{ color: '#666' }}>정보 습득률:</span>
                  <span style={{ color: '#bc13fe' }}>{state.derived.information_rate.toFixed(4)}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                  <span style={{ color: '#666' }}>선호 이탈도:</span>
                  <span style={{ color: '#ff6b6b' }}>{state.derived.preference_divergence.toFixed(3)}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: '#666' }}>믿음 변화:</span>
                  <span style={{ color: '#00f3ff' }}>{state.derived.belief_update.toFixed(4)}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Center Column - G Decomposition */}
          <div>
            <div style={{
              padding: '15px',
              background: '#0a0a0a',
              borderRadius: '8px',
              border: '1px solid #222'
            }}>
              <h3 style={{ margin: '0 0 5px 0', color: '#ff6b00', fontSize: '0.8rem' }}>
                기대 자유 에너지 (G)
              </h3>
              <p style={{ margin: '0 0 15px 0', color: '#666', fontSize: '0.6rem' }}>
                G = 위험 + 모호함 + 복잡도 (낮을수록 좋음)
              </p>

              {/* Dominant Factor */}
              <div style={{
                padding: '10px',
                background: '#111',
                borderRadius: '6px',
                marginBottom: '15px',
                textAlign: 'center'
              }}>
                <div style={{ fontSize: '0.6rem', color: '#666', marginBottom: '5px' }}>
                  주도 요인
                </div>
                <div style={{
                  fontSize: '1rem',
                  fontWeight: 'bold',
                  color: state.action.dominant_factor === 'risk_avoidance' ? '#ff6b6b' :
                         state.action.dominant_factor === 'ambiguity_reduction' ? '#bc13fe' :
                         state.action.dominant_factor === 'complexity_avoidance' ? '#00bfff' : '#888'
                }}>
                  {state.action.dominant_factor === 'risk_avoidance' && '위험 회피'}
                  {state.action.dominant_factor === 'ambiguity_reduction' && '모호함 감소'}
                  {state.action.dominant_factor === 'complexity_avoidance' && '복잡도 회피'}
                  {state.action.dominant_factor === 'balanced' && '균형'}
                </div>
                <div style={{ fontSize: '0.55rem', color: '#555', marginTop: '5px' }}>
                  {state.action.dominant_factor === 'risk_avoidance' && '(관찰자: "공포")'}
                  {state.action.dominant_factor === 'ambiguity_reduction' && '(관찰자: "호기심")'}
                  {state.action.dominant_factor === 'complexity_avoidance' && '(관찰자: "습관")'}
                </div>
              </div>

              {/* G bars for each action */}
              {[0, 1, 2, 3, 4].map(a => (
                <GBar
                  key={a}
                  action={a}
                  g={state.action.G[a.toString()] || 0}
                  risk={state.action.risk[a.toString()] || 0}
                  ambiguity={state.action.ambiguity[a.toString()] || 0}
                  complexity={state.action.complexity?.[a.toString()] || 0}
                  isSelected={state.action.selected === a}
                  isBest={bestAction === a}
                />
              ))}
            </div>
          </div>

          {/* Right Column - Why & Interpretation */}
          <div>
            {/* Control Panel - v3.6 */}
            <ControlPanel state={state} />

            {/* Scenario Panel */}
            <ScenarioPanel
              currentScenario={scenarioStatus}
              onStart={() => setScenarioStatus({ active: true })}
              onStop={() => setScenarioStatus(null)}
            />

            {/* v5.13 Ops Monitor Panel */}
            <OpsMonitorPanel />

            {/* Why This Action */}
            <div style={{
              padding: '15px',
              background: '#0a0a0a',
              borderRadius: '8px',
              border: '1px solid #00f3ff',
              marginBottom: '15px'
            }}>
              <h3 style={{ margin: '0 0 10px 0', color: '#00f3ff', fontSize: '0.8rem' }}>
                왜 이 행동을 선택했나?
              </h3>
              <div style={{
                padding: '15px',
                background: '#001a1a',
                borderRadius: '6px',
                fontSize: '0.8rem',
                color: '#fff',
                lineHeight: 1.6
              }}>
                행동 {state.action.selected}: G={state.action.G[state.action.selected.toString()]?.toFixed(2)}
              </div>

              <div style={{ marginTop: '15px' }}>
                <div style={{ fontSize: '0.65rem', color: '#666', marginBottom: '5px' }}>
                  선택된 행동 분해:
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '10px', fontSize: '0.75rem' }}>
                  <div>
                    <span style={{ color: '#ff6b6b' }}>위험: </span>
                    <span>{state.action.selected_risk.toFixed(3)}</span>
                  </div>
                  <div>
                    <span style={{ color: '#bc13fe' }}>모호함: </span>
                    <span>{state.action.selected_ambiguity.toFixed(3)}</span>
                  </div>
                  <div>
                    <span style={{ color: '#00bfff' }}>복잡도: </span>
                    <span>{(state.action.selected_complexity || 0).toFixed(3)}</span>
                  </div>
                </div>
              </div>
            </div>

            {/* F Trend - 고정 위치 */}
            <div style={{
              padding: '15px',
              background: '#0a0a0a',
              borderRadius: '8px',
              border: '1px solid #ff6b00'
            }}>
              <h3 style={{ margin: '0 0 10px 0', color: '#ff6b00', fontSize: '0.8rem' }}>
                F 추세
              </h3>
              <div style={{
                padding: '15px',
                background: '#111',
                borderRadius: '6px',
                textAlign: 'center'
              }}>
                <div style={{
                  fontSize: '1.2rem',
                  fontWeight: 'bold',
                  color: state.interpretation.F_trend === 'decreasing' ? '#00ff88' :
                         state.interpretation.F_trend === 'increasing' ? '#ff3e3e' : '#888'
                }}>
                  {state.interpretation.F_trend === 'decreasing' && '↓ 감소 중'}
                  {state.interpretation.F_trend === 'increasing' && '↑ 증가 중'}
                  {state.interpretation.F_trend === 'stable' && '→ 안정'}
                </div>
                <div style={{ fontSize: '0.65rem', color: '#666', marginTop: '8px' }}>
                  {state.interpretation.F_trend === 'decreasing' && '관찰자 해석: "만족"'}
                  {state.interpretation.F_trend === 'increasing' && '관찰자 해석: "불안"'}
                  {state.interpretation.F_trend === 'stable' && '관찰자 해석: "평온"'}
                </div>
              </div>
            </div>

            {/* Equation Reference */}
            <div style={{
              marginTop: '15px',
              padding: '15px',
              background: '#0a0a0a',
              borderRadius: '8px',
              border: '1px solid #222',
              fontSize: '0.6rem',
              color: '#555'
            }}>
              <div style={{ marginBottom: '5px' }}>F = 예측오차 + 복잡도</div>
              <div style={{ marginBottom: '5px' }}>G = 위험 + 모호함 + 복잡도</div>
              <div style={{ marginBottom: '5px', color: '#ff6b6b' }}>위험 = KL[Q(o|a) || P(o)] 선호 위반</div>
              <div style={{ marginBottom: '5px', color: '#bc13fe' }}>모호함 = 전이 불확실성</div>
              <div style={{ color: '#00bfff' }}>복잡도 = 선호 상태 이탈</div>
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <footer style={{
        marginTop: '20px',
        padding: '10px',
        textAlign: 'center',
        color: '#444',
        fontSize: '0.65rem'
      }}>
        Everything emerges from F minimization. No emotion labels. No hardcoded goals. Just Free Energy.
      </footer>
    </div>
  );
}

export default GenesisApp;
