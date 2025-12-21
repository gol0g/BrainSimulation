
import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import Neuroscope from './components/Neuroscope';
import ControlPanel from './components/ControlPanel';
import { Activity } from 'lucide-react';
import WorldMap from './components/WorldMap';

const API_URL = 'http://localhost:8000';

function App() {
  const [neuronData, setNeuronData] = useState({
    s_up: [], s_down: [], s_left: [], s_right: [],
    h_up: [], h_down: [], h_left: [], h_right: [],
    a_up: [], a_down: [], a_left: [], a_right: [],
    gaba: []
  });
  const [synapses, setSynapses] = useState([]);
  const [worldState, setWorldState] = useState(null);
  const [agencyState, setAgencyState] = useState(null);  // Agency tracking
  const [agencyHistory, setAgencyHistory] = useState([]); // Agency level over time
  const [memoryState, setMemoryState] = useState(null);  // Working Memory tracking
  const [usingMemory, setUsingMemory] = useState(false); // Memory usage indicator
  const [attentionState, setAttentionState] = useState(null);  // Attention tracking
  const [conflictState, setConflictState] = useState(null);  // Value Conflict tracking
  const [selfModelState, setSelfModelState] = useState(null);  // Self-Model: "What am I?"
  const [injectValue, setInjectValue] = useState(0);
  const [neuronParams, setNeuronParams] = useState({ a: 0.02, b: 0.2, c: -65, d: 8 });
  const [noiseLevel, setNoiseLevel] = useState(2.0);
  const [status, setStatus] = useState("OFFLINE");
  const [isBursting, setIsBursting] = useState(false);
  const [rewardFlash, setRewardFlash] = useState(false);
  const [deathFlash, setDeathFlash] = useState(false);
  const [pushFlash, setPushFlash] = useState(false);  // External push indicator
  const [perturbType, setPerturbType] = useState(null);  // 'wall', 'wind', or null
  const [windInfo, setWindInfo] = useState(null);  // Wind state from server

  const injectRef = useRef(injectValue);
  injectRef.current = injectValue;
  const noiseRef = useRef(noiseLevel);
  noiseRef.current = noiseLevel;

  useEffect(() => {
    const intervalId = setInterval(fetchStep, 50);
    return () => clearInterval(intervalId);
  }, []);

  const triggerBurst = () => {
    if (isBursting) return;
    setIsBursting(true);
    const originalValue = injectValue;
    const pulse = (count) => {
      if (count <= 0) {
        setInjectValue(originalValue);
        setIsBursting(false);
        return;
      }
      setInjectValue(20);
      setTimeout(() => {
        setInjectValue(0);
        setTimeout(() => pulse(count - 1), 150);
      }, 150);
    };
    pulse(3);
  };

  const fetchStep = async () => {
    try {
      const res = await axios.post(`${API_URL}/network/step`, {
        currents: { "s_up": injectRef.current },
        noise_level: noiseRef.current
      });

      if (status !== "ONLINE") setStatus("ONLINE");
      const { trajectories, synapses: synData, world } = res.data;

      if (world && world.reward > 0) {
        setRewardFlash(true);
        setTimeout(() => setRewardFlash(false), 300);
      }

      if (world && world.died) {
        setDeathFlash(true);
        setTimeout(() => setDeathFlash(false), 1000);
      }

      setNeuronData(prev => {
        const updated = { ...prev };
        Object.keys(updated).forEach(nid => {
          if (trajectories[nid]) {
            const newPoints = trajectories[nid].map((pt, i) => ({
              t: Date.now() + i, v: pt.v, fired: pt.fired
            }));
            const arr = [...prev[nid], ...newPoints];
            updated[nid] = arr.length > 300 ? arr.slice(-300) : arr;
          }
        });
        return updated;
      });

      if (synData) setSynapses(synData);
      if (world) setWorldState(world);

      // Update agency state
      if (res.data.agency) {
        setAgencyState(res.data.agency);
        setAgencyHistory(prev => {
          const newHistory = [...prev, res.data.agency.agency_level];
          return newHistory.length > 100 ? newHistory.slice(-100) : newHistory;
        });
      }

      // External perturbation flash (wall or wind)
      if (res.data.was_perturbed) {
        setPushFlash(true);
        setPerturbType(res.data.perturb_type);
        setTimeout(() => {
          setPushFlash(false);
          setPerturbType(null);
        }, 500);
      }

      // Update wind info
      if (res.data.world && res.data.world.wind) {
        setWindInfo(res.data.world.wind);
      }

      // Update working memory state
      if (res.data.memory) {
        setMemoryState(res.data.memory);
      }
      if (res.data.using_memory !== undefined) {
        setUsingMemory(res.data.using_memory);
      }

      // Update attention state
      if (res.data.attention) {
        setAttentionState(res.data.attention);
      }

      // Update conflict state
      if (res.data.conflict) {
        setConflictState(res.data.conflict);
      }

      // Update self-model state
      if (res.data.self_model) {
        setSelfModelState(res.data.self_model);
      }
    } catch (error) {
      if (status !== "OFFLINE") setStatus("OFFLINE");
    }
  };

  // External push handler
  const handlePush = async (direction) => {
    try {
      const res = await axios.post(`${API_URL}/agency/push?direction=${direction}`);
      if (res.data.agency) {
        setAgencyState(res.data.agency);
        setAgencyHistory(prev => [...prev, res.data.agency.agency_level].slice(-100));
      }
    } catch (e) { console.error(e); }
  };

  const handleReset = async () => {
    try {
      await axios.post(`${API_URL}/network/reset`);
      setNeuronData(Object.keys(neuronData).reduce((acc, k) => ({ ...acc, [k]: [] }), {}));
      setInjectValue(0);
    } catch (e) { }
  };

  const sensoryNeurons = [
    { id: "s_up", label: "SENSE UP", color: "#00ff88" },
    { id: "s_down", label: "SENSE DOWN", color: "#00ff88" },
    { id: "s_left", label: "SENSE LEFT", color: "#00ff88" },
    { id: "s_right", label: "SENSE RIGHT", color: "#00ff88" },
  ];

  const hiddenNeurons = [
    { id: "h_up", label: "HIDDEN UP", color: "#00f3ff" },
    { id: "h_down", label: "HIDDEN DOWN", color: "#00f3ff" },
    { id: "h_left", label: "HIDDEN LEFT", color: "#00f3ff" },
    { id: "h_right", label: "HIDDEN RIGHT", color: "#00f3ff" },
  ];

  const actionNeurons = [
    { id: "a_up", label: "ACT UP", color: "#bc13fe" },
    { id: "a_down", label: "ACT DOWN", color: "#bc13fe" },
    { id: "a_left", label: "ACT LEFT", color: "#bc13fe" },
    { id: "a_right", label: "ACT RIGHT", color: "#bc13fe" },
  ];

  const gabaNeuron = { id: "gaba", label: "GABA", color: "#ff3e3e" };

  const pushButtonStyle = {
    width: '32px', height: '32px',
    background: '#222', border: '1px solid #444',
    color: '#ff6b00', fontWeight: 'bold', cursor: 'pointer',
    borderRadius: '4px', fontSize: '1rem',
    transition: 'all 0.2s',
  };

  const renderNeuronGraph = (n) => (
    <div key={n.id} style={{ marginBottom: '10px' }}>
      <div style={{ fontSize: '0.6rem', color: n.color, fontWeight: 'bold', marginBottom: '3px' }}>{n.label}</div>
      <Neuroscope dataPoints={neuronData[n.id]} color={n.color} height={60} />
    </div>
  );

  return (
    <div style={{
      maxWidth: '1600px', margin: '0 auto', padding: '20px',
      transition: 'background-color 0.3s ease',
      backgroundColor: deathFlash ? 'rgba(255, 62, 62, 0.15)' :
                       pushFlash ? 'rgba(255, 107, 0, 0.15)' :
                       (rewardFlash ? 'rgba(0, 255, 136, 0.05)' : 'transparent')
    }}>
      {/* Header */}
      <header style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        marginBottom: '20px', borderBottom: '1px solid #333', paddingBottom: '15px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
          <img src="/neuron_no_background.png" alt="Mascot" style={{ width: '60px' }} />
          <h1 style={{ margin: 0, letterSpacing: '2px', fontSize: '1.6rem' }}>
            PROJECT <span style={{ color: '#00f3ff' }}>GENESIS</span>
            <span style={{
              fontSize: '0.7rem',
              color: deathFlash ? '#ff3e3e' : (rewardFlash ? '#00ff88' : '#888'),
              marginLeft: '10px', transition: 'color 0.3s'
            }}>
              PHASE 9 (VALUE CONFLICT)
              {deathFlash && " [!!!] AGENT DIED - RESETTING..."}
              {pushFlash && perturbType === 'wall' && " 🧱 WALL HIT!"}
              {pushFlash && perturbType === 'wind' && " 💨 WIND PUSH!"}
              {windInfo?.active && !pushFlash && ` 🌬️ WIND:${windInfo.direction?.toUpperCase()}`}
              {conflictState?.in_conflict && " ⚖️ CONFLICT!"}
              {attentionState?.mode === 'FOCUSED' && attentionState?.focus && !conflictState?.in_conflict && ` 👁️ ${attentionState.focus.toUpperCase()}`}
              {rewardFlash && " +DOPAMINE SPIKE+"}
            </span>
          </h1>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
          {worldState && (
            <div style={{ display: 'flex', gap: '15px', fontSize: '0.8rem', fontFamily: 'monospace' }}>
              <span style={{
                color: worldState.energy < 20 ? '#ff3e3e' : '#ff6b00',
                fontWeight: worldState.energy < 20 ? 'bold' : 'normal'
              }}>
                ENERGY: {worldState.energy.toFixed(1)}%
              </span>
              <span style={{ color: worldState.reward > 0 ? '#00ff88' : (worldState.reward < 0 ? '#ff3e3e' : '#888') }}>
                LAST REWARD: {worldState.reward.toFixed(1)}
              </span>
            </div>
          )}
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', borderLeft: '1px solid #333', paddingLeft: '15px' }}>
            <Activity size={16} color={status === "ONLINE" ? "#0f0" : "#f00"} />
            <span style={{ color: status === "ONLINE" ? "#0f0" : "#f00", fontFamily: 'monospace', fontSize: '0.8rem' }}>{status}</span>
          </div>
        </div>
      </header>

      {/* Agency Panel */}
      {agencyState && (
        <div className="sci-fi-border" style={{
          padding: '15px', background: '#0a0a0a', marginBottom: '20px',
          display: 'grid', gridTemplateColumns: '200px 1fr 200px', gap: '20px', alignItems: 'center'
        }}>
          {/* Agency Level Display */}
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '0.7rem', color: '#888', marginBottom: '5px' }}>SELF-AGENCY</div>
            <div style={{
              fontSize: '2.5rem',
              fontWeight: 'bold',
              color: agencyState.agency_level > 0.7 ? '#00ff88' :
                     agencyState.agency_level > 0.4 ? '#ff6b00' : '#ff3e3e',
              fontFamily: 'monospace'
            }}>
              {(agencyState.agency_level * 100).toFixed(0)}%
            </div>
            <div style={{
              fontSize: '0.8rem',
              color: agencyState.interpretation === 'SELF_CAUSED' ? '#00ff88' :
                     agencyState.interpretation === 'EXTERNAL_PUSH' ? '#ff3e3e' : '#ff6b00',
              fontWeight: 'bold'
            }}>
              {agencyState.interpretation === 'SELF_CAUSED' ? '✓ 내가 했다' :
               agencyState.interpretation === 'EXTERNAL_PUSH' ? '⚠ 외부 힘' : '? 불확실'}
            </div>
          </div>

          {/* Agency History Graph */}
          <div style={{ height: '60px', position: 'relative', background: '#111', borderRadius: '4px', overflow: 'hidden' }}>
            <svg width="100%" height="100%" preserveAspectRatio="none">
              {/* Threshold lines */}
              <line x1="0" y1="18" x2="100%" y2="18" stroke="#333" strokeDasharray="4,4" />
              <line x1="0" y1="36" x2="100%" y2="36" stroke="#333" strokeDasharray="4,4" />
              {/* Agency line */}
              <polyline
                fill="none"
                stroke="#00f3ff"
                strokeWidth="2"
                points={agencyHistory.map((v, i) =>
                  `${(i / Math.max(agencyHistory.length - 1, 1)) * 100}%,${60 - v * 60}`
                ).join(' ')}
              />
            </svg>
            <div style={{ position: 'absolute', top: '2px', left: '5px', fontSize: '0.6rem', color: '#666' }}>
              AGENCY HISTORY
            </div>
          </div>

          {/* External Push Buttons */}
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '0.7rem', color: '#888', marginBottom: '8px' }}>EXTERNAL PUSH TEST</div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '4px', maxWidth: '120px', margin: '0 auto' }}>
              <div></div>
              <button onClick={() => handlePush('up')} style={pushButtonStyle}>↑</button>
              <div></div>
              <button onClick={() => handlePush('left')} style={pushButtonStyle}>←</button>
              <div style={{ fontSize: '0.6rem', color: '#666', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>PUSH</div>
              <button onClick={() => handlePush('right')} style={pushButtonStyle}>→</button>
              <div></div>
              <button onClick={() => handlePush('down')} style={pushButtonStyle}>↓</button>
              <div></div>
            </div>
          </div>
        </div>
      )}

      {/* Attention Panel */}
      {attentionState && (
        <div className="sci-fi-border" style={{
          padding: '15px', background: '#0a0a0a', marginBottom: '20px',
          display: 'grid', gridTemplateColumns: '150px 1fr 1fr 150px', gap: '15px', alignItems: 'center'
        }}>
          {/* Attention Mode Display */}
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '0.7rem', color: '#888', marginBottom: '5px' }}>ATTENTION</div>
            <div style={{
              fontSize: '1.8rem',
              fontWeight: 'bold',
              color: attentionState.mode === 'FOCUSED' ? '#ff6b00' : '#00f3ff',
              fontFamily: 'monospace'
            }}>
              {attentionState.mode === 'FOCUSED' ? '◉' : '○'}
            </div>
            <div style={{
              fontSize: '0.75rem',
              color: attentionState.mode === 'FOCUSED' ? '#ff6b00' : '#00f3ff',
              fontWeight: 'bold'
            }}>
              {attentionState.mode === 'FOCUSED' ? '집중' : '확산'}
            </div>
            {attentionState.focus && (
              <div style={{
                fontSize: '0.7rem',
                color: '#ffcc00',
                marginTop: '5px'
              }}>
                → {attentionState.focus.toUpperCase()}
              </div>
            )}
          </div>

          {/* Attention Weights - Visual Bars */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '10px' }}>
            {['up', 'down', 'left', 'right'].map(dir => {
              const weight = attentionState.weights?.[dir] || 1.0;
              const isFocused = attentionState.focus === dir;
              const isAmplified = weight > 1.0;
              const isSuppressed = weight < 1.0;
              return (
                <div key={dir} style={{ textAlign: 'center' }}>
                  <div style={{
                    fontSize: '0.6rem',
                    color: isFocused ? '#ffcc00' : (isAmplified ? '#00ff88' : (isSuppressed ? '#ff6b00' : '#666')),
                    fontWeight: isFocused ? 'bold' : 'normal',
                    marginBottom: '4px'
                  }}>
                    {dir.toUpperCase()} {isFocused && '★'}
                  </div>
                  <div style={{
                    height: '30px',
                    background: '#111',
                    borderRadius: '3px',
                    position: 'relative',
                    overflow: 'hidden',
                    border: isFocused ? '1px solid #ffcc00' : '1px solid #222'
                  }}>
                    {/* Baseline (1.0) marker */}
                    <div style={{
                      position: 'absolute',
                      bottom: '40%',
                      left: 0,
                      right: 0,
                      height: '1px',
                      background: '#444'
                    }} />
                    {/* Weight bar */}
                    <div style={{
                      position: 'absolute',
                      bottom: 0,
                      left: 0,
                      right: 0,
                      height: `${Math.min(100, (weight / 2.5) * 100)}%`,
                      background: isFocused ? 'linear-gradient(to top, #ff6b00, #ffcc00)' :
                                 isAmplified ? 'linear-gradient(to top, #006600, #00ff88)' :
                                 isSuppressed ? 'linear-gradient(to top, #662200, #ff6b00)' :
                                 '#555',
                      transition: 'height 0.15s ease-out',
                      borderRadius: '2px'
                    }} />
                  </div>
                  <div style={{
                    fontSize: '0.6rem',
                    color: weight > 1.0 ? '#00ff88' : (weight < 1.0 ? '#ff6b00' : '#666'),
                    marginTop: '2px',
                    fontFamily: 'monospace'
                  }}>
                    {weight.toFixed(2)}x
                  </div>
                </div>
              );
            })}
          </div>

          {/* Salience - What's "interesting" */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '10px' }}>
            {['up', 'down', 'left', 'right'].map(dir => {
              const salience = attentionState.salience?.[dir] || 0;
              const isHigh = salience > 0.3;
              return (
                <div key={dir} style={{ textAlign: 'center' }}>
                  <div style={{
                    fontSize: '0.55rem',
                    color: isHigh ? '#bc13fe' : '#555',
                    marginBottom: '3px'
                  }}>
                    SAL
                  </div>
                  <div style={{
                    height: '20px',
                    background: '#111',
                    borderRadius: '3px',
                    position: 'relative',
                    overflow: 'hidden'
                  }}>
                    <div style={{
                      position: 'absolute',
                      bottom: 0,
                      left: 0,
                      right: 0,
                      height: `${salience * 100}%`,
                      background: isHigh ? 'linear-gradient(to top, #6600aa, #bc13fe)' : '#333',
                      transition: 'height 0.1s ease-out'
                    }} />
                  </div>
                </div>
              );
            })}
          </div>

          {/* Focus Strength & Width */}
          <div style={{ textAlign: 'center', borderLeft: '1px solid #333', paddingLeft: '15px' }}>
            <div style={{ marginBottom: '10px' }}>
              <div style={{ fontSize: '0.6rem', color: '#888' }}>FOCUS STR</div>
              <div style={{
                fontSize: '1.2rem',
                fontWeight: 'bold',
                color: attentionState.strength > 0.5 ? '#ffcc00' : '#666',
                fontFamily: 'monospace'
              }}>
                {(attentionState.strength * 100).toFixed(0)}%
              </div>
            </div>
            <div>
              <div style={{ fontSize: '0.6rem', color: '#888' }}>WIDTH</div>
              <div style={{
                width: '60px',
                height: '8px',
                background: '#111',
                borderRadius: '4px',
                margin: '5px auto',
                position: 'relative',
                overflow: 'hidden'
              }}>
                <div style={{
                  position: 'absolute',
                  left: '50%',
                  transform: 'translateX(-50%)',
                  width: `${attentionState.width * 100}%`,
                  height: '100%',
                  background: attentionState.width < 0.5 ? '#ff6b00' : '#00f3ff',
                  borderRadius: '4px',
                  transition: 'width 0.2s ease-out'
                }} />
              </div>
              <div style={{
                fontSize: '0.6rem',
                color: attentionState.width < 0.5 ? '#ff6b00' : '#00f3ff'
              }}>
                {attentionState.width < 0.3 ? '좁음' : attentionState.width < 0.7 ? '중간' : '넓음'}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Value Conflict Panel */}
      {conflictState && (
        <div className="sci-fi-border" style={{
          padding: '15px', background: '#0a0a0a', marginBottom: '20px',
          display: 'grid', gridTemplateColumns: '120px 1fr 1fr 150px', gap: '20px', alignItems: 'center'
        }}>
          {/* Conflict Level Gauge */}
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '0.7rem', color: '#888', marginBottom: '5px' }}>CONFLICT</div>
            <div style={{
              fontSize: '2rem',
              fontWeight: 'bold',
              color: conflictState.in_conflict ? '#ff6b00' : '#555',
              fontFamily: 'monospace'
            }}>
              {conflictState.in_conflict ? '⚖️' : '○'}
            </div>
            <div style={{
              fontSize: '0.8rem',
              color: conflictState.in_conflict ? '#ff6b00' : '#555',
              fontWeight: 'bold'
            }}>
              {conflictState.in_conflict ? '갈등!' : '없음'}
            </div>
            <div style={{
              fontSize: '0.65rem',
              color: '#888',
              marginTop: '3px'
            }}>
              {(conflictState.conflict * 100).toFixed(0)}%
            </div>
          </div>

          {/* Hesitation & Conflict Bar */}
          <div>
            <div style={{ fontSize: '0.65rem', color: '#888', marginBottom: '5px' }}>HESITATION</div>
            <div style={{
              height: '25px',
              background: '#111',
              borderRadius: '4px',
              position: 'relative',
              overflow: 'hidden',
              border: conflictState.hesitation > 0.3 ? '1px solid #ff6b0066' : '1px solid #222'
            }}>
              <div style={{
                position: 'absolute',
                left: 0,
                top: 0,
                bottom: 0,
                width: `${conflictState.hesitation * 100}%`,
                background: conflictState.hesitation > 0.5 ?
                  'linear-gradient(to right, #ff3e3e, #ff6b00)' :
                  'linear-gradient(to right, #444, #ff6b00)',
                transition: 'width 0.2s ease-out'
              }} />
              <div style={{
                position: 'absolute',
                left: '5px',
                top: '50%',
                transform: 'translateY(-50%)',
                fontSize: '0.6rem',
                color: '#fff',
                textShadow: '1px 1px 2px #000'
              }}>
                {conflictState.hesitation > 0.5 ? '망설이는 중...' : ''}
              </div>
            </div>

            <div style={{
              marginTop: '8px',
              display: 'flex',
              justifyContent: 'space-between',
              fontSize: '0.6rem'
            }}>
              <span style={{ color: '#00ff88' }}>
                즉시 보상 (●)
              </span>
              <span style={{ color: '#ffcc00' }}>
                지연 보상 (★)
              </span>
            </div>
          </div>

          {/* Regret / Satisfaction */}
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '0.65rem', color: '#888', marginBottom: '5px' }}>
              {conflictState.regret > 0 ? 'REGRET' : 'SATISFACTION'}
            </div>
            <div style={{
              height: '50px',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'center'
            }}>
              {conflictState.regret > 0.1 ? (
                <div style={{
                  fontSize: '1.5rem',
                  color: '#ff3e3e'
                }}>
                  😔 {(conflictState.regret * 100).toFixed(0)}%
                </div>
              ) : conflictState.satisfaction > 0.1 ? (
                <div style={{
                  fontSize: '1.5rem',
                  color: '#00ff88'
                }}>
                  😊 {(conflictState.satisfaction * 100).toFixed(0)}%
                </div>
              ) : (
                <div style={{
                  fontSize: '1.2rem',
                  color: '#555'
                }}>
                  😐
                </div>
              )}
            </div>
            {conflictState.last_choice && (
              <div style={{
                fontSize: '0.6rem',
                color: conflictState.last_choice === 'large' ? '#ffcc00' : '#00ff88'
              }}>
                마지막: {conflictState.last_choice === 'large' ? '★ 큰 보상' : '● 작은 보상'}
              </div>
            )}
          </div>

          {/* Statistics */}
          <div style={{
            borderLeft: '1px solid #333',
            paddingLeft: '15px',
            fontSize: '0.65rem',
            color: '#888'
          }}>
            <div style={{ marginBottom: '5px' }}>
              <span style={{ color: '#555' }}>총 갈등:</span>{' '}
              <span style={{ color: '#ff6b00' }}>{conflictState.stats?.total_conflicts || 0}</span>
            </div>
            <div style={{ marginBottom: '5px' }}>
              <span style={{ color: '#00ff88' }}>● 즉시 선택:</span>{' '}
              {conflictState.stats?.chose_small || 0}
            </div>
            <div style={{ marginBottom: '5px' }}>
              <span style={{ color: '#ffcc00' }}>★ 지연 선택:</span>{' '}
              {conflictState.stats?.chose_large || 0}
            </div>
            <div>
              <span style={{ color: '#555' }}>평균 후회:</span>{' '}
              <span style={{ color: '#ff3e3e' }}>
                {((conflictState.stats?.avg_regret || 0) * 100).toFixed(0)}%
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Self-Model Panel - "What kind of being am I?" */}
      {selfModelState && (
        <div className="sci-fi-border" style={{
          padding: '15px', background: '#0a0a0a', marginBottom: '20px',
          display: 'grid', gridTemplateColumns: '120px 1fr 1fr 150px', gap: '20px', alignItems: 'center'
        }}>
          {/* Behavioral State */}
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '0.7rem', color: '#888', marginBottom: '5px' }}>SELF-STATE</div>
            <div style={{
              fontSize: '1.2rem',
              fontWeight: 'bold',
              color: selfModelState.behavioral_label === 'CONFIDENT' ? '#00ff88' :
                     selfModelState.behavioral_label === 'EXPLORING' ? '#00f3ff' :
                     selfModelState.behavioral_label === 'STRUGGLING' ? '#ff6b00' :
                     selfModelState.behavioral_label === 'REACTIVE' ? '#ff3e3e' :
                     selfModelState.behavioral_label === 'FATIGUED' ? '#bc13fe' : '#888',
              fontFamily: 'monospace'
            }}>
              {selfModelState.behavioral_label === 'CONFIDENT' ? '😎' :
               selfModelState.behavioral_label === 'EXPLORING' ? '🔍' :
               selfModelState.behavioral_label === 'STRUGGLING' ? '😰' :
               selfModelState.behavioral_label === 'REACTIVE' ? '⚡' :
               selfModelState.behavioral_label === 'FATIGUED' ? '😴' :
               selfModelState.behavioral_label === 'STABLE' ? '😌' : '🤔'}
            </div>
            <div style={{
              fontSize: '0.7rem',
              color: '#aaa',
              marginTop: '3px'
            }}>
              {selfModelState.behavioral_label}
            </div>
          </div>

          {/* Core Self-State Bars */}
          <div style={{ display: 'grid', gridTemplateRows: 'repeat(3, 1fr)', gap: '8px' }}>
            {/* Confidence */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <div style={{ width: '70px', fontSize: '0.6rem', color: '#00ff88' }}>CONFIDENCE</div>
              <div style={{
                flex: 1, height: '12px', background: '#111', borderRadius: '6px',
                overflow: 'hidden', position: 'relative'
              }}>
                <div style={{
                  position: 'absolute', left: 0, top: 0, bottom: 0,
                  width: `${(selfModelState.state?.confidence || 0) * 100}%`,
                  background: 'linear-gradient(to right, #006644, #00ff88)',
                  transition: 'width 0.2s'
                }} />
              </div>
              <div style={{ width: '35px', fontSize: '0.65rem', color: '#00ff88', textAlign: 'right' }}>
                {((selfModelState.state?.confidence || 0) * 100).toFixed(0)}%
              </div>
            </div>
            {/* Uncertainty */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <div style={{ width: '70px', fontSize: '0.6rem', color: '#ff6b00' }}>UNCERTAINTY</div>
              <div style={{
                flex: 1, height: '12px', background: '#111', borderRadius: '6px',
                overflow: 'hidden', position: 'relative'
              }}>
                <div style={{
                  position: 'absolute', left: 0, top: 0, bottom: 0,
                  width: `${(selfModelState.state?.uncertainty || 0) * 100}%`,
                  background: 'linear-gradient(to right, #663300, #ff6b00)',
                  transition: 'width 0.2s'
                }} />
              </div>
              <div style={{ width: '35px', fontSize: '0.65rem', color: '#ff6b00', textAlign: 'right' }}>
                {((selfModelState.state?.uncertainty || 0) * 100).toFixed(0)}%
              </div>
            </div>
            {/* Effort */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <div style={{ width: '70px', fontSize: '0.6rem', color: '#bc13fe' }}>EFFORT</div>
              <div style={{
                flex: 1, height: '12px', background: '#111', borderRadius: '6px',
                overflow: 'hidden', position: 'relative'
              }}>
                <div style={{
                  position: 'absolute', left: 0, top: 0, bottom: 0,
                  width: `${(selfModelState.state?.effort || 0) * 100}%`,
                  background: 'linear-gradient(to right, #4a0080, #bc13fe)',
                  transition: 'width 0.2s'
                }} />
              </div>
              <div style={{ width: '35px', fontSize: '0.65rem', color: '#bc13fe', textAlign: 'right' }}>
                {((selfModelState.state?.effort || 0) * 100).toFixed(0)}%
              </div>
            </div>
          </div>

          {/* Exploration Need & Stability */}
          <div style={{ display: 'grid', gridTemplateRows: 'repeat(2, 1fr)', gap: '8px' }}>
            {/* Exploration Need */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <div style={{ width: '70px', fontSize: '0.6rem', color: '#00f3ff' }}>EXPLORE</div>
              <div style={{
                flex: 1, height: '12px', background: '#111', borderRadius: '6px',
                overflow: 'hidden', position: 'relative'
              }}>
                <div style={{
                  position: 'absolute', left: 0, top: 0, bottom: 0,
                  width: `${(selfModelState.state?.exploration_need || 0) * 100}%`,
                  background: 'linear-gradient(to right, #005566, #00f3ff)',
                  transition: 'width 0.2s'
                }} />
              </div>
              <div style={{ width: '35px', fontSize: '0.65rem', color: '#00f3ff', textAlign: 'right' }}>
                {((selfModelState.state?.exploration_need || 0) * 100).toFixed(0)}%
              </div>
            </div>
            {/* Stability */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <div style={{ width: '70px', fontSize: '0.6rem', color: '#888' }}>STABILITY</div>
              <div style={{
                flex: 1, height: '12px', background: '#111', borderRadius: '6px',
                overflow: 'hidden', position: 'relative'
              }}>
                <div style={{
                  position: 'absolute', left: 0, top: 0, bottom: 0,
                  width: `${(selfModelState.state?.stability || 0) * 100}%`,
                  background: 'linear-gradient(to right, #444, #888)',
                  transition: 'width 0.2s'
                }} />
              </div>
              <div style={{ width: '35px', fontSize: '0.65rem', color: '#888', textAlign: 'right' }}>
                {((selfModelState.state?.stability || 0) * 100).toFixed(0)}%
              </div>
            </div>
          </div>

          {/* Derived Metrics */}
          <div style={{
            borderLeft: '1px solid #333',
            paddingLeft: '15px',
            fontSize: '0.6rem',
            color: '#666'
          }}>
            <div style={{ marginBottom: '4px' }}>
              <span>Avg Agency:</span>{' '}
              <span style={{ color: '#00ff88' }}>
                {((selfModelState.derived?.avg_agency || 0) * 100).toFixed(0)}%
              </span>
            </div>
            <div style={{ marginBottom: '4px' }}>
              <span>Avg Pred Err:</span>{' '}
              <span style={{ color: '#ff6b00' }}>
                {((selfModelState.derived?.avg_pred_error || 0) * 100).toFixed(0)}%
              </span>
            </div>
            <div style={{ marginBottom: '4px' }}>
              <span>Reward Trend:</span>{' '}
              <span style={{
                color: (selfModelState.derived?.reward_trend || 0) > 0 ? '#00ff88' :
                       (selfModelState.derived?.reward_trend || 0) < 0 ? '#ff3e3e' : '#888'
              }}>
                {(selfModelState.derived?.reward_trend || 0) > 0 ? '↑' :
                 (selfModelState.derived?.reward_trend || 0) < 0 ? '↓' : '→'}
                {((selfModelState.derived?.reward_trend || 0) * 100).toFixed(0)}%
              </span>
            </div>
            <div>
              <span>Focus Duration:</span>{' '}
              <span style={{ color: '#ffcc00' }}>
                {selfModelState.focus_duration || 0}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Working Memory Panel */}
      {memoryState && (
        <div className="sci-fi-border" style={{
          padding: '15px', background: '#0a0a0a', marginBottom: '20px',
          display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr 200px', gap: '15px', alignItems: 'center'
        }}>
          {/* Memory Bars for each direction */}
          {['up', 'down', 'left', 'right'].map(dir => {
            const mem = memoryState[`m_${dir}`];
            if (!mem) return null;
            const activity = mem.activity || 0;
            const isActive = activity > 0.15;
            const isStrongest = memoryState.strongest === dir;
            return (
              <div key={dir} style={{ textAlign: 'center' }}>
                <div style={{
                  fontSize: '0.65rem',
                  color: isStrongest ? '#ffcc00' : (isActive ? '#00f3ff' : '#555'),
                  fontWeight: isStrongest ? 'bold' : 'normal',
                  marginBottom: '5px'
                }}>
                  MEM {dir.toUpperCase()} {isStrongest && '★'}
                </div>
                {/* Activity Bar */}
                <div style={{
                  height: '40px',
                  background: '#111',
                  borderRadius: '4px',
                  position: 'relative',
                  overflow: 'hidden',
                  border: isActive ? '1px solid #00f3ff33' : '1px solid #222'
                }}>
                  <div style={{
                    position: 'absolute',
                    bottom: 0,
                    left: 0,
                    right: 0,
                    height: `${activity * 100}%`,
                    background: isStrongest ? 'linear-gradient(to top, #ff6b00, #ffcc00)' :
                               isActive ? 'linear-gradient(to top, #006688, #00f3ff)' :
                               '#333',
                    transition: 'height 0.1s ease-out',
                    borderRadius: '2px'
                  }} />
                  {/* Threshold line */}
                  <div style={{
                    position: 'absolute',
                    bottom: '15%',
                    left: 0,
                    right: 0,
                    height: '1px',
                    background: '#ff3e3e44',
                    borderStyle: 'dashed'
                  }} />
                </div>
                <div style={{
                  fontSize: '0.7rem',
                  color: isActive ? '#00f3ff' : '#444',
                  marginTop: '3px',
                  fontFamily: 'monospace'
                }}>
                  {(activity * 100).toFixed(0)}%
                </div>
              </div>
            );
          })}

          {/* Memory Status */}
          <div style={{ textAlign: 'center', borderLeft: '1px solid #333', paddingLeft: '15px' }}>
            <div style={{ fontSize: '0.7rem', color: '#888', marginBottom: '8px' }}>WORKING MEMORY</div>
            <div style={{
              fontSize: '1.2rem',
              fontWeight: 'bold',
              color: usingMemory ? '#ffcc00' : (memoryState.using_memory ? '#00f3ff' : '#555'),
              marginBottom: '5px'
            }}>
              {usingMemory ? '🧠 ACTIVE' : (memoryState.strongest ? '💭 STORED' : '○ EMPTY')}
            </div>
            {memoryState.strongest && (
              <div style={{
                fontSize: '0.8rem',
                color: '#ff6b00',
                fontFamily: 'monospace'
              }}>
                기억: {memoryState.strongest.toUpperCase()}
              </div>
            )}
            {usingMemory && (
              <div style={{
                fontSize: '0.65rem',
                color: '#ffcc00',
                marginTop: '5px',
                animation: 'pulse 1s infinite'
              }}>
                ⚡ 기억 사용 중
              </div>
            )}
          </div>
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: '250px 1fr 250px 300px', gap: '20px' }}>
        {/* Sensory Column */}
        <div className="sci-fi-border" style={{ padding: '15px', background: '#0a0a0a' }}>
          <h3 style={{ fontSize: '0.8rem', color: '#888', textTransform: 'uppercase', marginTop: 0 }}>Sensory Input</h3>
          {sensoryNeurons.map(renderNeuronGraph)}
        </div>

        {/* World + Hidden Column */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <div style={{ display: 'flex', justifyContent: 'center' }}>
            <WorldMap world={worldState} />
          </div>
          <div className="sci-fi-border" style={{ padding: '15px', background: '#0a0a0a' }}>
            <h3 style={{ fontSize: '0.8rem', color: '#888', textTransform: 'uppercase', marginTop: 0 }}>Inter-Neurons</h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
              {hiddenNeurons.map(renderNeuronGraph)}
            </div>
          </div>
        </div>

        {/* Action Column */}
        <div className="sci-fi-border" style={{ padding: '15px', background: '#0a0a0a' }}>
          <h3 style={{ fontSize: '0.8rem', color: '#888', textTransform: 'uppercase', marginTop: 0 }}>Action Output</h3>
          {actionNeurons.map(renderNeuronGraph)}
          <div style={{ marginTop: '10px', paddingTop: '10px', borderTop: '1px solid #222' }}>
            {renderNeuronGraph(gabaNeuron)}
          </div>
        </div>

        {/* Controls and Synapses */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
          <ControlPanel
            params={neuronParams} onParamChange={setNeuronParams}
            injectValue={injectValue} onInjectChange={setInjectValue}
            noiseLevel={noiseLevel} onNoiseChange={setNoiseLevel}
            onReset={handleReset} onBurst={triggerBurst} isBursting={isBursting}
          />

          <div className="sci-fi-border" style={{ padding: '15px', background: '#0a0a0a', maxHeight: '400px', overflowY: 'auto' }}>
            <h3 style={{ fontSize: '0.8rem', color: '#ff6b00', textTransform: 'uppercase', marginTop: 0 }}>Synaptic Weights</h3>
            {synapses.map((syn, i) => (
              <div key={i} style={{ marginBottom: '8px', fontSize: '0.7rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', color: '#ccc' }}>
                  <span>{syn.pre}→{syn.post}</span>
                  <span style={{ color: syn.weight > 10 ? '#0f0' : '#888' }}>w: {syn.weight.toFixed(1)}</span>
                </div>
                <div style={{ background: '#1a1a1a', height: '4px', borderRadius: '2px', marginTop: '2px' }}>
                  <div style={{
                    width: `${Math.min(100, (Math.abs(syn.weight) / 30) * 100)}%`, height: '100%',
                    background: syn.weight < 0 ? '#f00' : '#ff6b00', borderRadius: '2px'
                  }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
