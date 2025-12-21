
import React from 'react';

const ControlPanel = ({ params, onParamChange, onInjectChange, injectValue, noiseLevel, onNoiseChange, onReset, onBurst, isBursting }) => {
    const presets = {
        'Regular Spiking': { a: 0.02, b: 0.2, c: -65, d: 8 },
        'Fast Spiking': { a: 0.1, b: 0.2, c: -65, d: 2 },
        'Chattering': { a: 0.02, b: 0.2, c: -50, d: 2 },
    };

    const applyPreset = (name) => {
        const p = presets[name];
        onParamChange(p);
    };

    return (
        <div className="sci-fi-border" style={{ padding: '20px', background: '#0a0a0a', color: '#ccc' }}>
            <h3 style={{ marginTop: 0, textTransform: 'uppercase', color: '#bc13fe' }}>Control Nexus</h3>

            <div className="control-group" style={{ opacity: isBursting ? 0.5 : 1 }}>
                <label>Input Current (I): {injectValue.toFixed(1)} pA</label>
                <input
                    type="range" min="0" max="20" step="0.5" value={injectValue}
                    onChange={(e) => onInjectChange(parseFloat(e.target.value))}
                    disabled={isBursting}
                />
            </div>

            <div className="control-group">
                <label style={{ color: '#00f3ff' }}>Noise Intensity: {noiseLevel.toFixed(1)}</label>
                <input
                    type="range" min="0" max="10" step="0.5" value={noiseLevel}
                    onChange={(e) => onNoiseChange(parseFloat(e.target.value))}
                />
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                <div className="control-group">
                    <label>a (Recovery Time): {params.a}</label>
                    <input
                        type="range" min="0.01" max="0.1" step="0.01" value={params.a || 0.02}
                        onChange={(e) => onParamChange({ ...params, a: parseFloat(e.target.value) })}
                    />
                </div>
                <div className="control-group">
                    <label>b (Sensitivity): {params.b}</label>
                    <input
                        type="range" min="0.05" max="0.3" step="0.01" value={params.b || 0.2}
                        onChange={(e) => onParamChange({ ...params, b: parseFloat(e.target.value) })}
                    />
                </div>
                <div className="control-group">
                    <label>c (Reset V): {params.c}</label>
                    <input
                        type="range" min="-80" max="-40" step="1" value={params.c || -65}
                        onChange={(e) => onParamChange({ ...params, c: parseFloat(e.target.value) })}
                    />
                </div>
                <div className="control-group">
                    <label>d (Reset U): {params.d}</label>
                    <input
                        type="range" min="0.1" max="10" step="0.1" value={params.d || 8}
                        onChange={(e) => onParamChange({ ...params, d: parseFloat(e.target.value) })}
                    />
                </div>
            </div>

            <div style={{ marginTop: '20px', display: 'flex', gap: '5px', flexWrap: 'wrap' }}>
                <button
                    className="btn-primary"
                    onClick={onBurst}
                    disabled={isBursting}
                    style={{
                        flex: '1 1 100%',
                        marginBottom: '10px',
                        background: isBursting ? '#333' : 'linear-gradient(45deg, #00f3ff, #bc13fe)',
                        border: 'none',
                        padding: '10px',
                        fontWeight: 'bold',
                        boxShadow: isBursting ? 'none' : '0 0 15px rgba(0,243,255,0.4)'
                    }}
                >
                    {isBursting ? 'BURSTING...' : 'INJECT PULSE PATTERN (BURST)'}
                </button>
                {Object.keys(presets).map(name => (
                    <button key={name} className="btn-primary" onClick={() => applyPreset(name)}>
                        {name}
                    </button>
                ))}
                <button className="btn-danger" onClick={onReset} style={{ marginLeft: 'auto' }}>
                    RESET
                </button>
            </div>
        </div>
    );
};

export default ControlPanel;
