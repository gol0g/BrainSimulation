
import React, { useEffect, useRef, useState } from 'react';
import { Line } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    Filler
} from 'chart.js';

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    Filler
);

const Neuroscope = ({ dataPoints, color = '#00f3ff' }) => {
    // dataPoints is an array of { t: number, v: number }

    const chartData = {
        labels: dataPoints.map((d, i) => i), // Use simple indices instead of timestamps
        datasets: [
            {
                label: 'Membrane Potential (mV)',
                data: dataPoints.map(d => d.v),
                borderColor: color,
                backgroundColor: `${color}22`,
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.4, // Smooth curve
                fill: true,
            },
            {
                label: 'Threshold',
                data: dataPoints.map(() => 30),
                borderColor: 'rgba(255, 0, 0, 0.5)',
                borderWidth: 1,
                pointRadius: 0,
                borderDash: [5, 5],
                fill: false,
            }
        ],
    };

    const options = {
        responsive: true,
        maintainAspectRatio: false,
        animation: {
            duration: 0 // Disable animation for real-time feel
        },
        scales: {
            x: {
                display: false, // Hide time axis for clean look
            },
            y: {
                min: -90,
                max: 50,
                grid: {
                    color: '#1a1a1a',
                },
                ticks: {
                    color: '#555',
                }
            },
        },
        plugins: {
            legend: {
                display: false,
            },
            tooltip: {
                enabled: false,
            }
        },
    };

    return (
        <div className="sci-fi-border" style={{ height: '300px', padding: '10px', background: '#0a0a0a' }}>
            <h3 style={{ margin: '0 0 10px 0', fontSize: '0.9rem', color: '#555', textTransform: 'uppercase' }}>
                Neuroscope <span style={{ color: '#00f3ff' }}>v1.0</span>
            </h3>
            <div style={{ height: 'calc(100% - 30px)' }}>
                <Line data={chartData} options={options} />
            </div>
        </div>
    );
};

export default Neuroscope;
