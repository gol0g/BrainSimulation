import React, { useEffect, useRef } from 'react';

const WorldMap = ({ world, lastAction }) => {
    const canvasRef = useRef(null);
    const gridSize = 10;
    const cellSize = 30;

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');

        // Clear
        ctx.fillStyle = '#050505';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw grid lines
        ctx.strokeStyle = '#222';
        ctx.lineWidth = 1;
        for (let i = 0; i <= gridSize; i++) {
            ctx.beginPath();
            ctx.moveTo(i * cellSize, 0);
            ctx.lineTo(i * cellSize, gridSize * cellSize);
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(0, i * cellSize);
            ctx.lineTo(gridSize * cellSize, i * cellSize);
            ctx.stroke();
        }

        if (!world) return;

        // Draw Large Food (if exists) - Gold star, bigger
        if (world.large_food) {
            const [lfx, lfy] = world.large_food;
            ctx.shadowBlur = 20;
            ctx.shadowColor = '#ffcc00';
            ctx.fillStyle = '#ffcc00';
            // Draw as star shape
            const cx = lfx * cellSize + cellSize / 2;
            const cy = lfy * cellSize + cellSize / 2;
            const outerR = cellSize / 2.2;
            const innerR = cellSize / 4.5;
            ctx.beginPath();
            for (let i = 0; i < 5; i++) {
                const outerAngle = (i * 72 - 90) * Math.PI / 180;
                const innerAngle = ((i * 72) + 36 - 90) * Math.PI / 180;
                ctx.lineTo(cx + outerR * Math.cos(outerAngle), cy + outerR * Math.sin(outerAngle));
                ctx.lineTo(cx + innerR * Math.cos(innerAngle), cy + innerR * Math.sin(innerAngle));
            }
            ctx.closePath();
            ctx.fill();
        }

        // Draw Small Food - Green circle
        const [fx, fy] = world.food;
        ctx.shadowBlur = 15;
        ctx.shadowColor = '#00ff88';
        ctx.fillStyle = '#00ff88';
        ctx.beginPath();
        ctx.arc(fx * cellSize + cellSize / 2, fy * cellSize + cellSize / 2, cellSize / 3, 0, Math.PI * 2);
        ctx.fill();

        // Draw Agent
        const [ax, ay] = world.agent;
        ctx.shadowBlur = 10;
        ctx.shadowColor = '#00f3ff';
        ctx.fillStyle = '#00f3ff';
        ctx.fillRect(ax * cellSize + 5, ay * cellSize + 5, cellSize - 10, cellSize - 10);

        ctx.shadowBlur = 0;

        // Highlight Activity
        if (world.reward > 0) {
            ctx.strokeStyle = '#00ff88';
            ctx.lineWidth = 3;
            ctx.strokeRect(ax * cellSize, ay * cellSize, cellSize, cellSize);
        }

    }, [world]);

    return (
        <div style={{ position: 'relative', width: cellSize * gridSize, height: cellSize * gridSize }}>
            <canvas
                ref={canvasRef}
                width={cellSize * gridSize}
                height={cellSize * gridSize}
                className="sci-fi-border"
                style={{ borderRadius: '5px' }}
            />
            <div style={{
                position: 'absolute', top: -30, left: 0, width: '100%',
                display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#888'
            }}>
                <span>Energy: {world?.energy?.toFixed(1) || 100}%</span>
                <span style={{ color: (world?.reward || 0) > 0 ? '#0f0' : '#888' }}>
                    Reward: {world?.reward?.toFixed(1) || 0}
                </span>
            </div>
        </div>
    );
};

export default WorldMap;
