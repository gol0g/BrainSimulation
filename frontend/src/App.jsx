
import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import Neuroscope from './components/Neuroscope';
import ControlPanel from './components/ControlPanel';
import { Activity, ChevronDown, ChevronRight } from 'lucide-react';
import WorldMap from './components/WorldMap';

const API_URL = 'http://localhost:8000';

// Collapsible Section Component
const CollapsibleSection = ({ title, icon, children, defaultOpen = false, color = '#888' }) => {
  const [isOpen, setIsOpen] = useState(defaultOpen);
  return (
    <div style={{ marginBottom: '8px' }}>
      <div
        onClick={() => setIsOpen(!isOpen)}
        style={{
          display: 'flex', alignItems: 'center', gap: '8px',
          padding: '8px 12px', background: '#0a0a0a', borderRadius: '6px',
          cursor: 'pointer', border: '1px solid #222',
          transition: 'all 0.2s'
        }}
      >
        {isOpen ? <ChevronDown size={14} color={color} /> : <ChevronRight size={14} color={color} />}
        <span style={{ fontSize: '0.75rem', color, fontWeight: 'bold' }}>{icon} {title}</span>
      </div>
      {isOpen && (
        <div style={{
          padding: '12px', background: '#0a0a0a',
          borderRadius: '0 0 6px 6px', borderTop: 'none',
          border: '1px solid #222', borderTop: 'none', marginTop: '-1px'
        }}>
          {children}
        </div>
      )}
    </div>
  );
};

// Compact Bar Component
const CompactBar = ({ label, value, max = 1, color, showPercent = true, warning = false }) => (
  <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
    <div style={{ width: '55px', fontSize: '0.55rem', color: warning ? '#ff3e3e' : color }}>{label}</div>
    <div style={{ flex: 1, height: '8px', background: '#111', borderRadius: '4px', overflow: 'hidden' }}>
      <div style={{
        height: '100%', width: `${(value / max) * 100}%`,
        background: warning ? '#ff3e3e' : color, transition: 'width 0.2s', borderRadius: '4px'
      }} />
    </div>
    {showPercent && <div style={{ width: '28px', fontSize: '0.55rem', color, textAlign: 'right' }}>{Math.round(value * 100)}%</div>}
  </div>
);

function App() {
  const [neuronData, setNeuronData] = useState({
    s_up: [], s_down: [], s_left: [], s_right: [],
    h_up: [], h_down: [], h_left: [], h_right: [],
    a_up: [], a_down: [], a_left: [], a_right: [],
    gaba: []
  });
  const [synapses, setSynapses] = useState([]);
  const [worldState, setWorldState] = useState(null);
  const [agencyState, setAgencyState] = useState(null);
  const [memoryState, setMemoryState] = useState(null);
  const [usingMemory, setUsingMemory] = useState(false);
  const [attentionState, setAttentionState] = useState(null);
  const [conflictState, setConflictState] = useState(null);
  const [selfModelState, setSelfModelState] = useState(null);
  const [homeostasisState, setHomeostasisState] = useState(null);
  const [emotionState, setEmotionState] = useState(null);
  const [imaginationState, setImaginationState] = useState(null);
  const [actionSource, setActionSource] = useState(null);
  const [developmentState, setDevelopmentState] = useState(null);
  const [ltmState, setLtmState] = useState(null);  // Long-term Memory
  const [injectValue, setInjectValue] = useState(0);
  const [neuronParams, setNeuronParams] = useState({ a: 0.02, b: 0.2, c: -65, d: 8 });
  const [noiseLevel, setNoiseLevel] = useState(2.0);
  const [status, setStatus] = useState("OFFLINE");
  const [isBursting, setIsBursting] = useState(false);
  const [rewardFlash, setRewardFlash] = useState(false);
  const [deathFlash, setDeathFlash] = useState(false);
  const [pushFlash, setPushFlash] = useState(false);
  const [perturbType, setPerturbType] = useState(null);
  const [windInfo, setWindInfo] = useState(null);

  const injectRef = useRef(injectValue);
  injectRef.current = injectValue;
  const noiseRef = useRef(noiseLevel);
  noiseRef.current = noiseLevel;
  const isFetchingRef = useRef(false);

  useEffect(() => {
    const intervalId = setInterval(fetchStep, 50);
    return () => clearInterval(intervalId);
  }, []);

  const triggerBurst = () => {
    if (isBursting) return;
    setIsBursting(true);
    const originalValue = injectValue;
    const pulse = (count) => {
      if (count <= 0) { setInjectValue(originalValue); setIsBursting(false); return; }
      setInjectValue(20);
      setTimeout(() => { setInjectValue(0); setTimeout(() => pulse(count - 1), 150); }, 150);
    };
    pulse(3);
  };

  const fetchStep = async () => {
    if (isFetchingRef.current) return;
    isFetchingRef.current = true;
    try {
      const res = await axios.post(`${API_URL}/network/step`, {
        currents: { "s_up": injectRef.current }, noise_level: noiseRef.current
      });
      if (status !== "ONLINE") setStatus("ONLINE");
      const { trajectories, synapses: synData, world } = res.data;
      if (world && world.reward > 0) { setRewardFlash(true); setTimeout(() => setRewardFlash(false), 300); }
      if (world && world.died) { setDeathFlash(true); setTimeout(() => setDeathFlash(false), 1000); }
      setNeuronData(prev => {
        const updated = { ...prev };
        Object.keys(updated).forEach(nid => {
          if (trajectories[nid]) {
            const newPoints = trajectories[nid].map((pt, i) => ({ t: Date.now() + i, v: pt.v, fired: pt.fired }));
            const arr = [...prev[nid], ...newPoints];
            updated[nid] = arr.length > 300 ? arr.slice(-300) : arr;
          }
        });
        return updated;
      });
      if (synData) setSynapses(synData);
      if (world) setWorldState(world);
      if (res.data.agency) setAgencyState(res.data.agency);
      if (res.data.was_perturbed) {
        setPushFlash(true); setPerturbType(res.data.perturb_type);
        setTimeout(() => { setPushFlash(false); setPerturbType(null); }, 500);
      }
      if (res.data.world?.wind) setWindInfo(res.data.world.wind);
      if (res.data.memory) setMemoryState(res.data.memory);
      if (res.data.using_memory !== undefined) setUsingMemory(res.data.using_memory);
      if (res.data.attention) setAttentionState(res.data.attention);
      if (res.data.conflict) setConflictState(res.data.conflict);
      if (res.data.self_model) setSelfModelState(res.data.self_model);
      if (res.data.homeostasis) setHomeostasisState(res.data.homeostasis);
      if (res.data.emotion) setEmotionState(res.data.emotion);
      if (res.data.imagination) setImaginationState(res.data.imagination);
      if (res.data.action_source) setActionSource(res.data.action_source);
      if (res.data.development) setDevelopmentState(res.data.development);
      if (res.data.long_term_memory) setLtmState(res.data.long_term_memory);
    } catch (error) {
      if (status !== "OFFLINE") setStatus("OFFLINE");
    } finally {
      isFetchingRef.current = false;
    }
  };

  const handleReset = async () => {
    try {
      await axios.post(`${API_URL}/network/reset`);
      setNeuronData(Object.keys(neuronData).reduce((acc, k) => ({ ...acc, [k]: [] }), {}));
      setInjectValue(0);
    } catch (e) { }
  };

  // Get emotion emoji
  const getEmotionEmoji = () => {
    if (!emotionState) return '😐';
    const e = emotionState.dominant;
    return e === 'fear' ? '😨' : e === 'pain' ? '😵' : e === 'anxiety' ? '😰' :
           e === 'satisfaction' ? '😊' : e === 'relief' ? '😌' : e === 'curiosity' ? '🧐' : '😐';
  };

  // Get action source text
  const getActionSourceText = () => {
    if (!actionSource) return '?';
    return actionSource === 'imagine' ? '상상' : actionSource === 'snn+imagine' ? '혼합' :
           actionSource === 'snn' ? '반사' : actionSource === 'explore' ? '탐험' : '랜덤';
  };

  const renderNeuronGraph = (n) => (
    <div key={n.id} style={{ marginBottom: '6px' }}>
      <div style={{ fontSize: '0.5rem', color: n.color, fontWeight: 'bold', marginBottom: '2px' }}>{n.label}</div>
      <Neuroscope dataPoints={neuronData[n.id]} color={n.color} height={40} />
    </div>
  );

  return (
    <div style={{
      maxWidth: '1400px', margin: '0 auto', padding: '10px',
      transition: 'background-color 0.3s ease',
      backgroundColor: deathFlash ? 'rgba(255, 62, 62, 0.15)' :
                       pushFlash ? 'rgba(255, 107, 0, 0.15)' :
                       (rewardFlash ? 'rgba(0, 255, 136, 0.05)' : 'transparent')
    }}>
      {/* Compact Header */}
      <header style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        marginBottom: '10px', padding: '8px 15px', background: '#0a0a0a',
        borderRadius: '8px', border: '1px solid #222'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>
            PROJECT <span style={{ color: '#00f3ff' }}>GENESIS</span>
          </span>
          {developmentState && (
            <span style={{
              padding: '2px 8px', borderRadius: '4px', fontSize: '0.65rem',
              background: developmentState.phase === 'infant' ? '#ff6b0033' : '#00ff8833',
              color: developmentState.phase === 'infant' ? '#ff6b00' : '#00ff88',
            }}>
              {developmentState.phase === 'infant' ? '👶 INFANT' : '🧑 ADULT'}
              {developmentState.phase === 'infant' && ` ${Math.round(developmentState.progress * 100)}%`}
            </span>
          )}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '15px', fontSize: '0.75rem' }}>
          {worldState && (
            <>
              <span style={{ color: worldState.energy < 20 ? '#ff3e3e' : '#ff6b00' }}>
                ⚡ {worldState.energy.toFixed(0)}%
              </span>
              <span style={{ color: worldState.reward > 0 ? '#00ff88' : worldState.reward < 0 ? '#ff3e3e' : '#666' }}>
                R: {worldState.reward > 0 ? '+' : ''}{worldState.reward.toFixed(1)}
              </span>
            </>
          )}
          <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            <Activity size={12} color={status === "ONLINE" ? "#0f0" : "#f00"} />
            <span style={{ color: status === "ONLINE" ? "#0f0" : "#f00", fontSize: '0.7rem' }}>{status}</span>
          </div>
        </div>
      </header>

      {/* Main Content - 3 Column Layout */}
      <div style={{ display: 'grid', gridTemplateColumns: '280px 1fr 280px', gap: '12px' }}>

        {/* Left Column - World & Core Status */}
        <div>
          {/* World Map */}
          <div style={{ marginBottom: '12px' }}>
            <WorldMap world={worldState} />
          </div>

          {/* Agent Mind - Always Visible */}
          <div style={{
            padding: '15px', background: '#0a0a0a', borderRadius: '8px',
            border: '1px solid #222', textAlign: 'center', marginBottom: '12px'
          }}>
            {/* Emotion, Thinking & Remembering */}
            <div style={{ display: 'flex', justifyContent: 'space-around', marginBottom: '12px' }}>
              <div>
                <div style={{ fontSize: '0.6rem', color: '#888', marginBottom: '3px' }}>FEELING</div>
                <div style={{ fontSize: '1.8rem' }}>{getEmotionEmoji()}</div>
                <div style={{ fontSize: '0.6rem', color: '#aaa' }}>{emotionState?.description || '-'}</div>
              </div>
              <div>
                <div style={{ fontSize: '0.6rem', color: '#888', marginBottom: '3px' }}>THINKING</div>
                <div style={{ fontSize: '1.8rem' }}>🤔</div>
                <div style={{ fontSize: '0.6rem', color: '#00f3ff' }}>{imaginationState?.reason || '...'}</div>
              </div>
              <div>
                <div style={{ fontSize: '0.6rem', color: '#888', marginBottom: '3px' }}>REMEMBER</div>
                <div style={{ fontSize: '1.8rem' }}>{ltmState?.has_recall ? '💭' : '○'}</div>
                <div style={{ fontSize: '0.6rem', color: ltmState?.has_recall ? '#ffcc00' : '#555' }}>
                  {ltmState?.recall_reason || '-'}
                </div>
              </div>
            </div>

            {/* Action Decision */}
            {imaginationState && (
              <div style={{
                display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '4px', marginBottom: '10px'
              }}>
                {['up', 'down', 'left', 'right'].map(dir => {
                  const score = imaginationState.scores?.[dir] || 0;
                  const isBest = imaginationState.best_action === dir;
                  return (
                    <div key={dir} style={{
                      padding: '4px', borderRadius: '4px', fontSize: '0.6rem',
                      background: isBest ? '#002200' : '#111',
                      border: isBest ? '1px solid #00ff88' : '1px solid #333',
                      color: isBest ? '#00ff88' : '#666'
                    }}>
                      {dir === 'up' ? '↑' : dir === 'down' ? '↓' : dir === 'left' ? '←' : '→'}
                      {isBest && ' ✓'}
                      <div style={{ fontSize: '0.5rem' }}>{score.toFixed(1)}</div>
                    </div>
                  );
                })}
              </div>
            )}

            {/* Action Source */}
            <div style={{
              display: 'flex', justifyContent: 'center', gap: '10px', fontSize: '0.6rem'
            }}>
              <span style={{ color: '#888' }}>결정:</span>
              <span style={{
                color: actionSource === 'snn' ? '#00ff88' : actionSource === 'imagine' ? '#bc13fe' :
                       actionSource === 'snn+imagine' ? '#00f3ff' : '#ffcc00',
                fontWeight: 'bold'
              }}>
                {getActionSourceText()}
              </span>
              <span style={{ color: '#666' }}>|</span>
              <span style={{ color: '#888' }}>신뢰도:</span>
              <span style={{ color: '#bc13fe' }}>{Math.round((imaginationState?.confidence || 0) * 100)}%</span>
            </div>
          </div>

          {/* Learning Progress - Always Visible */}
          {emotionState?.learned && (
            <div style={{
              padding: '10px', background: '#0a0a0a', borderRadius: '8px',
              border: '1px solid #222', marginBottom: '12px'
            }}>
              <div style={{ fontSize: '0.65rem', color: '#888', marginBottom: '8px' }}>학습된 연관</div>
              <div style={{ display: 'flex', gap: '10px' }}>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: '0.55rem', color: '#ff3e3e', marginBottom: '2px' }}>🔴 위험 ({emotionState.learned.pain_experiences || 0}회)</div>
                  <div style={{ height: '6px', background: '#111', borderRadius: '3px', overflow: 'hidden' }}>
                    <div style={{ height: '100%', width: `${(emotionState.learned.predator_fear || 0) * 100}%`, background: '#ff3e3e' }} />
                  </div>
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: '0.55rem', color: '#00ff88', marginBottom: '2px' }}>🟢 음식 ({emotionState.learned.food_experiences || 0}회)</div>
                  <div style={{ height: '6px', background: '#111', borderRadius: '3px', overflow: 'hidden' }}>
                    <div style={{ height: '100%', width: `${(emotionState.learned.food_seeking || 0) * 100}%`, background: '#00ff88' }} />
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Center Column - Collapsible Details */}
        <div>
          {/* Body: Homeostasis & Drives */}
          <CollapsibleSection title="BODY (항상성)" icon="🫀" color="#ff6b6b">
            {homeostasisState && (
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px' }}>
                <div>
                  <div style={{ fontSize: '0.6rem', color: '#888', marginBottom: '6px' }}>내부 상태</div>
                  <CompactBar label="ENERGY" value={homeostasisState.states?.energy || 0} color="#00ff88" warning={homeostasisState.critical?.starving} />
                  <CompactBar label="HEALTH" value={homeostasisState.states?.health || 0} color="#ff6b6b" warning={homeostasisState.critical?.injured} />
                  <CompactBar label="SAFETY" value={homeostasisState.states?.safety || 0} color="#00f3ff" warning={homeostasisState.critical?.in_danger} />
                  <CompactBar label="FATIGUE" value={homeostasisState.states?.fatigue || 0} color="#bc13fe" warning={homeostasisState.critical?.exhausted} />
                </div>
                <div>
                  <div style={{ fontSize: '0.6rem', color: '#888', marginBottom: '6px' }}>욕구 (Drives)</div>
                  <CompactBar label="HUNGER" value={homeostasisState.drives?.hunger || 0} color="#ff6b00" />
                  <CompactBar label="SAFETY" value={homeostasisState.drives?.safety || 0} color="#ff3e3e" />
                  <CompactBar label="REST" value={homeostasisState.drives?.rest || 0} color="#bc13fe" />
                  {homeostasisState.pain > 0 && (
                    <div style={{ marginTop: '8px', color: '#ff0000', fontSize: '0.65rem', fontWeight: 'bold' }}>
                      🩸 고통! {Math.round(homeostasisState.pain * 100)}%
                    </div>
                  )}
                </div>
              </div>
            )}
          </CollapsibleSection>

          {/* Mind: Emotions */}
          <CollapsibleSection title="MIND (감정)" icon="💭" color="#bc13fe">
            {emotionState && (
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px' }}>
                <div>
                  <div style={{ fontSize: '0.6rem', color: '#888', marginBottom: '6px' }}>부정 감정</div>
                  <CompactBar label="😨 FEAR" value={emotionState.emotions?.fear || 0} color="#ff3e3e" />
                  <CompactBar label="😵 PAIN" value={emotionState.emotions?.pain || 0} color="#ff0000" />
                  <CompactBar label="😰 ANXIETY" value={emotionState.emotions?.anxiety || 0} color="#ff6b00" />
                </div>
                <div>
                  <div style={{ fontSize: '0.6rem', color: '#888', marginBottom: '6px' }}>긍정 감정</div>
                  <CompactBar label="😊 SATISFY" value={emotionState.emotions?.satisfaction || 0} color="#00ff88" />
                  <CompactBar label="😌 RELIEF" value={emotionState.emotions?.relief || 0} color="#00f3ff" />
                  <CompactBar label="🧐 CURIOUS" value={emotionState.emotions?.curiosity || 0} color="#bc13fe" />
                </div>
              </div>
            )}
          </CollapsibleSection>

          {/* Self-Model */}
          <CollapsibleSection title="SELF (자기인식)" icon="🪞" color="#00f3ff">
            {selfModelState && (
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px' }}>
                <div>
                  <CompactBar label="CONF" value={selfModelState.state?.confidence || 0} color="#00ff88" />
                  <CompactBar label="UNCERT" value={selfModelState.state?.uncertainty || 0} color="#ff6b00" />
                  <CompactBar label="EFFORT" value={selfModelState.state?.effort || 0} color="#bc13fe" />
                </div>
                <div>
                  <CompactBar label="EXPLORE" value={selfModelState.state?.exploration_need || 0} color="#00f3ff" />
                  <CompactBar label="STABLE" value={selfModelState.state?.stability || 0} color="#888" />
                  <div style={{ marginTop: '8px', textAlign: 'center' }}>
                    <span style={{
                      fontSize: '0.7rem', padding: '3px 8px', borderRadius: '4px',
                      background: '#111', color: selfModelState.behavioral_label === 'CONFIDENT' ? '#00ff88' :
                                               selfModelState.behavioral_label === 'EXPLORING' ? '#00f3ff' :
                                               selfModelState.behavioral_label === 'STRUGGLING' ? '#ff6b00' : '#888'
                    }}>
                      {selfModelState.behavioral_label}
                    </span>
                  </div>
                </div>
              </div>
            )}
          </CollapsibleSection>

          {/* Agency & Attention */}
          <CollapsibleSection title="AGENCY (주체성)" icon="🎯" color="#ffcc00">
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px' }}>
              {agencyState && (
                <div>
                  <div style={{ fontSize: '0.6rem', color: '#888', marginBottom: '6px' }}>자기 주체감</div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <div style={{
                      fontSize: '1.5rem', fontWeight: 'bold',
                      color: agencyState.agency_level > 0.7 ? '#00ff88' : agencyState.agency_level > 0.4 ? '#ff6b00' : '#ff3e3e'
                    }}>
                      {Math.round(agencyState.agency_level * 100)}%
                    </div>
                    <div style={{
                      fontSize: '0.6rem',
                      color: agencyState.interpretation === 'SELF_CAUSED' ? '#00ff88' : '#ff3e3e'
                    }}>
                      {agencyState.interpretation === 'SELF_CAUSED' ? '✓ 내가 함' : '⚠ 외부 힘'}
                    </div>
                  </div>
                </div>
              )}
              {attentionState && (
                <div>
                  <div style={{ fontSize: '0.6rem', color: '#888', marginBottom: '6px' }}>주의 집중</div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <span style={{
                      fontSize: '1.2rem',
                      color: attentionState.mode === 'FOCUSED' ? '#ff6b00' : '#00f3ff'
                    }}>
                      {attentionState.mode === 'FOCUSED' ? '◉' : '○'}
                    </span>
                    <span style={{ fontSize: '0.65rem', color: '#aaa' }}>
                      {attentionState.focus ? `→ ${attentionState.focus.toUpperCase()}` : '확산'}
                    </span>
                  </div>
                </div>
              )}
            </div>
          </CollapsibleSection>

          {/* Working Memory */}
          <CollapsibleSection title="MEMORY (작업기억)" icon="🧠" color="#00f3ff">
            {memoryState && (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '8px' }}>
                {['up', 'down', 'left', 'right'].map(dir => {
                  const mem = memoryState[`m_${dir}`];
                  const activity = mem?.activity || 0;
                  const isStrongest = memoryState.strongest === dir;
                  return (
                    <div key={dir} style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: '0.55rem', color: isStrongest ? '#ffcc00' : '#666' }}>
                        {dir.toUpperCase()} {isStrongest && '★'}
                      </div>
                      <div style={{
                        height: '30px', background: '#111', borderRadius: '4px',
                        position: 'relative', overflow: 'hidden', marginTop: '3px'
                      }}>
                        <div style={{
                          position: 'absolute', bottom: 0, left: 0, right: 0,
                          height: `${activity * 100}%`,
                          background: isStrongest ? '#ffcc00' : '#00f3ff',
                          transition: 'height 0.1s'
                        }} />
                      </div>
                      <div style={{ fontSize: '0.5rem', color: '#666', marginTop: '2px' }}>
                        {Math.round(activity * 100)}%
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </CollapsibleSection>

          {/* Long-term Memory */}
          <CollapsibleSection title="LTM (장기기억)" icon="📚" color="#ffcc00">
            {ltmState && (
              <div>
                {/* Memory Stats */}
                <div style={{ display: 'flex', justifyContent: 'space-around', marginBottom: '10px' }}>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: '#ffcc00' }}>
                      {ltmState.total_memories || 0}
                    </div>
                    <div style={{ fontSize: '0.5rem', color: '#888' }}>기억</div>
                  </div>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: '#ff3e3e' }}>
                      {ltmState.outcome_distribution?.pain || 0}
                    </div>
                    <div style={{ fontSize: '0.5rem', color: '#888' }}>고통</div>
                  </div>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: '#00ff88' }}>
                      {ltmState.outcome_distribution?.food || 0}
                    </div>
                    <div style={{ fontSize: '0.5rem', color: '#888' }}>음식</div>
                  </div>
                </div>

                {/* Current Recall */}
                {ltmState.has_recall && (
                  <div style={{
                    padding: '8px', background: '#111', borderRadius: '6px',
                    border: '1px solid #ffcc0044', marginBottom: '8px'
                  }}>
                    <div style={{ fontSize: '0.6rem', color: '#ffcc00', marginBottom: '4px' }}>
                      💭 현재 recall: {ltmState.recall_count}개
                    </div>
                    <div style={{ fontSize: '0.7rem', color: '#fff' }}>
                      {ltmState.recall_reason}
                    </div>
                  </div>
                )}

                {/* Memory Influence with Details (피드백 #4: 왜 떠올랐는지 표시) */}
                {ltmState.memory_influence && Object.values(ltmState.memory_influence).some(v => v !== 0) && (
                  <div>
                    <div style={{ fontSize: '0.55rem', color: '#888', marginBottom: '4px' }}>기억 영향 (경험 기반)</div>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '4px' }}>
                      {['up', 'down', 'left', 'right'].map(dir => {
                        const influence = ltmState.memory_influence[dir] || 0;
                        const details = ltmState.recall_details?.[dir];
                        const isPositive = influence > 0;
                        const isNegative = influence < 0;
                        return (
                          <div key={dir} style={{
                            textAlign: 'center', padding: '4px', borderRadius: '4px',
                            background: isNegative ? '#220000' : isPositive ? '#002200' : '#111',
                            border: `1px solid ${isNegative ? '#ff3e3e44' : isPositive ? '#00ff8844' : '#333'}`
                          }}>
                            <div style={{ fontSize: '0.5rem', color: '#888' }}>
                              {dir === 'up' ? '↑' : dir === 'down' ? '↓' : dir === 'left' ? '←' : '→'}
                            </div>
                            <div style={{
                              fontSize: '0.7rem', fontWeight: 'bold',
                              color: isNegative ? '#ff3e3e' : isPositive ? '#00ff88' : '#666'
                            }}>
                              {influence > 0 ? '+' : ''}{influence.toFixed(2)}
                            </div>
                            {/* Show delta breakdown if available */}
                            {details && (
                              <div style={{ fontSize: '0.4rem', color: '#666', marginTop: '2px' }}>
                                {details.delta_pain > 0 && <span style={{ color: '#ff3e3e' }}>⚡{details.delta_pain.toFixed(2)} </span>}
                                {details.delta_energy > 0 && <span style={{ color: '#00ff88' }}>+E{details.delta_energy.toFixed(2)} </span>}
                                {details.delta_energy < 0 && <span style={{ color: '#ff8800' }}>-E{Math.abs(details.delta_energy).toFixed(2)} </span>}
                                <span style={{ color: '#888' }}>({details.memory_count})</span>
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            )}
          </CollapsibleSection>
        </div>

        {/* Right Column - Brain (SNN) */}
        <div>
          <CollapsibleSection title="BRAIN (신경망)" icon="⚡" color="#00ff88" defaultOpen={true}>
            <div style={{ marginBottom: '10px' }}>
              <div style={{ fontSize: '0.6rem', color: '#00ff88', marginBottom: '6px' }}>Sensory</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px' }}>
                {[{ id: "s_up", label: "S↑", color: "#00ff88" },
                  { id: "s_down", label: "S↓", color: "#00ff88" },
                  { id: "s_left", label: "S←", color: "#00ff88" },
                  { id: "s_right", label: "S→", color: "#00ff88" }].map(n => (
                  <div key={n.id}>
                    <div style={{ fontSize: '0.45rem', color: n.color }}>{n.label}</div>
                    <Neuroscope dataPoints={neuronData[n.id]} color={n.color} height={25} />
                  </div>
                ))}
              </div>
            </div>
            <div style={{ marginBottom: '10px' }}>
              <div style={{ fontSize: '0.6rem', color: '#00f3ff', marginBottom: '6px' }}>Hidden</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px' }}>
                {[{ id: "h_up", label: "H↑", color: "#00f3ff" },
                  { id: "h_down", label: "H↓", color: "#00f3ff" },
                  { id: "h_left", label: "H←", color: "#00f3ff" },
                  { id: "h_right", label: "H→", color: "#00f3ff" }].map(n => (
                  <div key={n.id}>
                    <div style={{ fontSize: '0.45rem', color: n.color }}>{n.label}</div>
                    <Neuroscope dataPoints={neuronData[n.id]} color={n.color} height={25} />
                  </div>
                ))}
              </div>
            </div>
            <div>
              <div style={{ fontSize: '0.6rem', color: '#bc13fe', marginBottom: '6px' }}>Action</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px' }}>
                {[{ id: "a_up", label: "A↑", color: "#bc13fe" },
                  { id: "a_down", label: "A↓", color: "#bc13fe" },
                  { id: "a_left", label: "A←", color: "#bc13fe" },
                  { id: "a_right", label: "A→", color: "#bc13fe" }].map(n => (
                  <div key={n.id}>
                    <div style={{ fontSize: '0.45rem', color: n.color }}>{n.label}</div>
                    <Neuroscope dataPoints={neuronData[n.id]} color={n.color} height={25} />
                  </div>
                ))}
              </div>
            </div>
          </CollapsibleSection>

          <CollapsibleSection title="SYNAPSES (가중치)" icon="🔗" color="#ff6b00">
            <div style={{ maxHeight: '200px', overflowY: 'auto' }}>
              {synapses.slice(0, 15).map((syn, i) => (
                <div key={i} style={{ marginBottom: '4px', fontSize: '0.55rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', color: '#888' }}>
                    <span>{syn.pre}→{syn.post}</span>
                    <span style={{ color: syn.weight > 50 ? '#00ff88' : '#666' }}>{syn.weight.toFixed(1)}</span>
                  </div>
                  <div style={{ background: '#111', height: '3px', borderRadius: '2px', marginTop: '1px' }}>
                    <div style={{
                      width: `${Math.min(100, (syn.weight / 100) * 100)}%`, height: '100%',
                      background: syn.weight < 0 ? '#ff3e3e' : '#ff6b00', borderRadius: '2px'
                    }} />
                  </div>
                </div>
              ))}
            </div>
          </CollapsibleSection>

          <CollapsibleSection title="CONTROLS" icon="🎛️" color="#888">
            <div style={{ fontSize: '0.6rem' }}>
              <div style={{ marginBottom: '8px' }}>
                <label style={{ color: '#888' }}>Noise: {noiseLevel.toFixed(1)}</label>
                <input type="range" min="0" max="10" step="0.5" value={noiseLevel}
                  onChange={(e) => setNoiseLevel(parseFloat(e.target.value))}
                  style={{ width: '100%' }}
                />
              </div>
              <div style={{ display: 'flex', gap: '8px' }}>
                <button onClick={handleReset} style={{
                  flex: 1, padding: '6px', background: '#222', border: '1px solid #444',
                  color: '#ff6b00', borderRadius: '4px', cursor: 'pointer', fontSize: '0.6rem'
                }}>
                  RESET
                </button>
                <button onClick={triggerBurst} disabled={isBursting} style={{
                  flex: 1, padding: '6px', background: isBursting ? '#333' : '#222',
                  border: '1px solid #444', color: '#00f3ff', borderRadius: '4px',
                  cursor: isBursting ? 'not-allowed' : 'pointer', fontSize: '0.6rem'
                }}>
                  BURST
                </button>
              </div>
            </div>
          </CollapsibleSection>
        </div>
      </div>

      {/* Status Bar */}
      <div style={{
        marginTop: '10px', padding: '6px 15px', background: '#0a0a0a',
        borderRadius: '6px', border: '1px solid #222',
        display: 'flex', justifyContent: 'center', gap: '20px',
        fontSize: '0.6rem', color: '#666'
      }}>
        {windInfo?.active && <span>🌬️ WIND: {windInfo.direction?.toUpperCase()}</span>}
        {conflictState?.in_conflict && <span style={{ color: '#ff6b00' }}>⚖️ CONFLICT</span>}
        {usingMemory && <span style={{ color: '#00f3ff' }}>🧠 WM ACTIVE</span>}
        {ltmState?.has_recall && <span style={{ color: '#ffcc00' }}>📚 LTM RECALL</span>}
        {homeostasisState?.critical?.starving && <span style={{ color: '#ff3e3e' }}>⚠️ STARVING</span>}
        {homeostasisState?.critical?.in_danger && <span style={{ color: '#ff3e3e' }}>⚠️ DANGER</span>}
        {deathFlash && <span style={{ color: '#ff0000' }}>💀 AGENT DIED</span>}
      </div>
    </div>
  );
}

export default App;
