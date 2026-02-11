"""Genesis Brain Dashboard - Streamlit visualization for brain simulation.

Usage:
    streamlit run dashboard/app.py
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "dashboard"))

from brain_schema import (
    PHASES, PHASE_ORDER, POPULATIONS, CONNECTIONS,
    RATE_KEYS_BY_PHASE, ALL_RATE_KEYS, RATE_KEY_LABELS,
    HEBBIAN_SYNAPSES,
)

st.set_page_config(
    page_title="Genesis Brain Dashboard",
    page_icon="🧠",
    layout="wide",
)


# ─── Data Loading ───

LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")


@st.cache_data(ttl=3)
def list_runs():
    """List available training runs."""
    if not os.path.isdir(LOGS_DIR):
        return []
    runs = []
    for name in sorted(os.listdir(LOGS_DIR), reverse=True):
        config_path = os.path.join(LOGS_DIR, name, "config.json")
        if os.path.isfile(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            runs.append({
                "dir": name,
                "path": os.path.join(LOGS_DIR, name),
                "timestamp": cfg.get("timestamp", "?"),
                "episodes": cfg.get("episodes", "?"),
                "neurons": cfg.get("neurons", "?"),
            })
    return runs


@st.cache_data(ttl=3)
def load_config(run_path):
    with open(os.path.join(run_path, "config.json")) as f:
        return json.load(f)


@st.cache_data(ttl=3)
def load_steps(run_path):
    path = os.path.join(run_path, "steps.jsonl")
    if not os.path.isfile(path):
        return pd.DataFrame()
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows) if rows else pd.DataFrame()


@st.cache_data(ttl=3)
def load_episodes(run_path):
    path = os.path.join(run_path, "episodes.jsonl")
    if not os.path.isfile(path):
        return pd.DataFrame()
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows) if rows else pd.DataFrame()


@st.cache_data(ttl=3)
def load_hebbian(run_path):
    path = os.path.join(run_path, "hebbian.jsonl")
    if not os.path.isfile(path):
        return pd.DataFrame()
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ─── Sidebar ───

st.sidebar.title("Genesis Brain")
st.sidebar.caption("SNN Brain Dashboard")

# Live mode toggle
live_mode = st.sidebar.toggle("Live Mode", value=False, help="Auto-refresh every 3 seconds for real-time monitoring")
if live_mode:
    st_autorefresh(interval=3000, key="live_refresh")
    st.sidebar.success("LIVE - refreshing every 3s")

runs = list_runs()
if not runs:
    st.error("No training runs found. Run training with `--log-data` first.")
    st.code("python forager_brain.py --episodes 20 --render none --log-data")
    st.stop()

run_labels = [f"{r['dir']} ({r['neurons']} neurons, {r['episodes']} ep)" for r in runs]
selected_idx = st.sidebar.selectbox("Select Run", range(len(runs)), format_func=lambda i: run_labels[i])
selected_run = runs[selected_idx]

config = load_config(selected_run["path"])
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Neurons:** {config.get('neurons', '?')}")
st.sidebar.markdown(f"**Episodes:** {config.get('episodes', '?')}")
phases_enabled = config.get("phases_enabled", {})
enabled_count = sum(1 for v in phases_enabled.values() if v)
st.sidebar.markdown(f"**Phases enabled:** {enabled_count}")

# Load data
df_steps = load_steps(selected_run["path"])
df_episodes = load_episodes(selected_run["path"])
df_hebbian = load_hebbian(selected_run["path"])

# Live status
if live_mode:
    if not df_steps.empty:
        cur_ep = int(df_steps["ep"].max())
        cur_step = int(df_steps[df_steps["ep"] == cur_ep]["step"].max())
        n_ep_done = len(df_episodes) if not df_episodes.empty else 0
        st.sidebar.markdown(f"**Episode:** {cur_ep} / Step: {cur_step}")
        st.sidebar.markdown(f"**Completed:** {n_ep_done} episodes")
        st.sidebar.markdown(f"**Data:** {len(df_steps)} steps logged")
    else:
        st.sidebar.info("Waiting for data...")

# ─── Navigation (sidebar radio preserves state across refreshes) ───

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "View",
    ["Brain Architecture", "Brain Activity", "Episode Performance", "Learning Progress"],
    key="nav_page",
)


# ═══════════════════════════════════════════════════
# Page 1: Brain Architecture
# ═══════════════════════════════════════════════════

if page == "Brain Architecture":
    st.header("Brain Architecture (19,240 neurons)")

    # Phase filter
    col1, col2 = st.columns([3, 1])
    with col2:
        selected_phases = st.multiselect(
            "Filter Phases",
            PHASE_ORDER,
            default=PHASE_ORDER,
            format_func=lambda p: f"P{p}: {PHASES[p]['name']}",
        )

    # Build network graph
    node_x, node_y, node_text, node_size, node_color = [], [], [], [], []
    node_ids = {}

    phase_pops = {}
    for pop_id, display, n_neurons, phase, rate_key in POPULATIONS:
        if phase not in selected_phases:
            continue
        phase_pops.setdefault(phase, [])
        phase_pops[phase].append(len(node_x))
        node_ids[pop_id] = len(node_x)

        n_in_phase = sum(1 for _, _, _, p, _ in POPULATIONS if p == phase)
        idx_in_phase = len(phase_pops[phase]) - 1
        x = (idx_in_phase - n_in_phase / 2) * 1.2

        y = PHASES[phase]["y"]
        node_x.append(x)
        node_y.append(y)

        avg_rate = ""
        if not df_steps.empty and rate_key in df_steps.columns:
            avg_rate = f"<br>Avg rate: {df_steps[rate_key].mean():.3f}"

        node_text.append(
            f"<b>{display}</b><br>"
            f"Phase {phase}: {PHASES[phase]['name']}<br>"
            f"Neurons: {n_neurons}"
            f"{avg_rate}"
        )
        node_size.append(max(8, min(30, n_neurons / 20)))
        node_color.append(PHASES[phase]["color"])

    # Edges
    edge_x, edge_y = [], []
    for src, tgt, weight, desc in CONNECTIONS:
        if src in node_ids and tgt in node_ids:
            x0, y0 = node_x[node_ids[src]], node_y[node_ids[src]]
            x1, y1 = node_x[node_ids[tgt]], node_y[node_ids[tgt]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    fig_arch = go.Figure()

    fig_arch.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=0.8, color="rgba(150,150,150,0.3)"),
        hoverinfo="none",
        showlegend=False,
    ))

    fig_arch.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers",
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=1, color="white"),
        ),
        text=node_text,
        hoverinfo="text",
        showlegend=False,
    ))

    for phase_id in selected_phases:
        y = PHASES[phase_id]["y"]
        fig_arch.add_annotation(
            x=max(node_x) + 2 if node_x else 5,
            y=y,
            text=f"<b>P{phase_id}</b>: {PHASES[phase_id]['name']}",
            showarrow=False,
            font=dict(size=10, color=PHASES[phase_id]["color"]),
            xanchor="left",
        )

    fig_arch.update_layout(
        height=900,
        margin=dict(l=20, r=200, t=40, b=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    with col1:
        st.plotly_chart(fig_arch, width="stretch")

    with st.expander("Population Details"):
        pop_data = []
        for pop_id, display, n, phase, rk in POPULATIONS:
            if phase in selected_phases:
                pop_data.append({
                    "Phase": phase,
                    "Region": PHASES[phase]["name"],
                    "Population": display,
                    "Neurons": n,
                    "Rate Key": rk,
                })
        st.dataframe(pd.DataFrame(pop_data), width="stretch", hide_index=True)


# ═══════════════════════════════════════════════════
# Page 2: Brain Activity
# ═══════════════════════════════════════════════════

elif page == "Brain Activity":
    st.header("Brain Activity Heatmap")

    if df_steps.empty:
        st.warning("No step data available.")
    else:
        episodes_available = sorted(df_steps["ep"].unique())
        default_ep = episodes_available[-1] if live_mode else episodes_available[0]
        selected_ep = st.select_slider(
            "Episode",
            options=episodes_available,
            value=default_ep,
        )

        ep_data = df_steps[df_steps["ep"] == selected_ep]

        available_keys = [k for k in ALL_RATE_KEYS if k in ep_data.columns]
        if available_keys:
            heatmap_data = ep_data[available_keys].values.T
            labels = [RATE_KEY_LABELS.get(k, k) for k in available_keys]
            steps = ep_data["step"].values

            fig_heat = go.Figure(data=go.Heatmap(
                z=heatmap_data,
                x=steps,
                y=labels,
                colorscale="YlOrRd",
                zmin=0,
                zmax=0.5,
                colorbar=dict(title="Rate"),
            ))
            fig_heat.update_layout(
                height=max(400, len(available_keys) * 18),
                margin=dict(l=200, r=20, t=40, b=60),
                xaxis_title="Step",
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_heat, width="stretch")

            # Phase-averaged activity
            st.subheader("Phase-averaged Activity")
            phase_avg_data = {}
            for phase_id in PHASE_ORDER:
                keys = [k for k in RATE_KEYS_BY_PHASE.get(phase_id, []) if k in ep_data.columns]
                if keys:
                    phase_avg_data[f"P{phase_id}: {PHASES[phase_id]['name']}"] = ep_data[keys].mean(axis=1).values

            if phase_avg_data:
                fig_phase = go.Figure()
                for name, values in phase_avg_data.items():
                    fig_phase.add_trace(go.Scatter(
                        x=ep_data["step"].values,
                        y=values,
                        mode="lines",
                        name=name,
                        line=dict(width=1.5),
                    ))
                fig_phase.update_layout(
                    height=400,
                    xaxis_title="Step",
                    yaxis_title="Average Firing Rate",
                    legend=dict(font=dict(size=9)),
                    margin=dict(l=60, r=20, t=20, b=60),
                )
                st.plotly_chart(fig_phase, width="stretch")

            # Region time series selector
            st.subheader("Region Time Series")
            selected_regions = st.multiselect(
                "Select regions to plot",
                available_keys,
                default=available_keys[:5],
                format_func=lambda k: RATE_KEY_LABELS.get(k, k),
            )
            if selected_regions:
                fig_ts = go.Figure()
                for key in selected_regions:
                    fig_ts.add_trace(go.Scatter(
                        x=ep_data["step"].values,
                        y=ep_data[key].values,
                        mode="lines",
                        name=RATE_KEY_LABELS.get(key, key),
                    ))
                fig_ts.update_layout(
                    height=350,
                    xaxis_title="Step",
                    yaxis_title="Firing Rate",
                    margin=dict(l=60, r=20, t=20, b=60),
                )
                st.plotly_chart(fig_ts, width="stretch")
        else:
            st.warning("No rate keys found in step data.")


# ═══════════════════════════════════════════════════
# Page 3: Episode Performance
# ═══════════════════════════════════════════════════

elif page == "Episode Performance":
    st.header("Episode Performance")

    if df_episodes.empty:
        st.warning("No episode data available.")
    else:
        total_food = df_episodes["food_eaten"].sum()
        total_steps = df_episodes["steps"].sum()
        timeout_count = (df_episodes["death_cause"] == "timeout").sum()
        n_episodes = len(df_episodes)

        survival_rate = timeout_count / n_episodes * 100
        reward_freq = total_food / total_steps * 100 if total_steps > 0 else 0

        # Pain 교차 검증 지표
        pain_steps_total = df_episodes["pain_steps"].sum() if "pain_steps" in df_episodes.columns else 0
        pain_time_pct = pain_steps_total / total_steps * 100 if total_steps > 0 else 0
        pain_death_count = (df_episodes["death_cause"] == "pain").sum()
        pain_death_pct = pain_death_count / n_episodes * 100
        avg_pain_visits = df_episodes["pain_visits"].mean() if "pain_visits" in df_episodes.columns else 0

        pain_time_ok = pain_time_pct < 15
        pain_death_ok = pain_death_pct < 20
        pain_entry_ok = avg_pain_visits < 10
        pain_pass = pain_time_ok and pain_death_ok and pain_entry_ok

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Survival Rate", f"{survival_rate:.0f}%",
                     delta="PASS" if survival_rate > 40 else "FAIL",
                     delta_color="normal" if survival_rate > 40 else "inverse")
        col2.metric("Reward Freq", f"{reward_freq:.2f}%",
                     delta="PASS" if reward_freq > 2.5 else "FAIL",
                     delta_color="normal" if reward_freq > 2.5 else "inverse")
        col3.metric("Pain Death", f"{pain_death_pct:.0f}%",
                     delta="PASS" if pain_death_ok else "FAIL",
                     delta_color="normal" if pain_death_ok else "inverse")
        col4.metric("Pain Composite", "PASS" if pain_pass else "FAIL",
                     delta=f"time:{pain_time_pct:.0f}% death:{pain_death_pct:.0f}% entry:{avg_pain_visits:.0f}/ep",
                     delta_color="normal" if pain_pass else "inverse")

        # === 모순 탐지 (Contradiction Alerts) ===
        contradictions = []
        if pain_time_ok and not pain_death_ok:
            contradictions.append(
                f"Pain Time {pain_time_pct:.1f}% (low) vs Pain Death {pain_death_pct:.0f}% (high) "
                f"-- Agent enters briefly but repeatedly, accumulating lethal damage")
        if pain_time_ok and avg_pain_visits > 10:
            contradictions.append(
                f"Pain Time {pain_time_pct:.1f}% (low) vs Entries {avg_pain_visits:.0f}/ep (high) "
                f"-- Wall bounce hides repeated entries; brain NOT avoiding")
        avg_bounce_in_pain = df_episodes["wall_bounces_in_pain"].mean() if "wall_bounces_in_pain" in df_episodes.columns else 0
        if avg_bounce_in_pain > avg_pain_visits * 0.5 and avg_pain_visits > 3:
            contradictions.append(
                f"Wall Bounce Exits {avg_bounce_in_pain:.0f} ~ Pain Visits {avg_pain_visits:.0f} "
                f"-- Most 'escapes' are wall bounces, not learned avoidance")
        avg_dist = df_episodes["avg_dist_to_pain"].mean() if "avg_dist_to_pain" in df_episodes.columns else 999
        if avg_dist < 75 and pain_time_ok:
            contradictions.append(
                f"Avg Dist to Pain {avg_dist:.0f}px (close) but Pain Time {pain_time_pct:.1f}% (low) "
                f"-- Agent hugs boundary, doesn't actively avoid")

        if contradictions:
            st.error(f"**CONTRADICTION ALERTS ({len(contradictions)})**")
            for c in contradictions:
                st.warning(c)

        # 정직한 Pain 지표 상세
        if "wall_bounces_in_pain" in df_episodes.columns:
            with st.expander("Pain Honest Metrics (detailed)"):
                pcol1, pcol2, pcol3, pcol4 = st.columns(4)
                pcol1.metric("Avg Pain Entries", f"{avg_pain_visits:.1f}/ep")
                pcol2.metric("Wall Bounce in Pain", f"{avg_bounce_in_pain:.1f}/ep")
                pcol3.metric("Avg Dist to Pain", f"{avg_dist:.1f}px")
                max_dwell = df_episodes["max_pain_dwell"].mean() if "max_pain_dwell" in df_episodes.columns else 0
                pcol4.metric("Max Dwell", f"{max_dwell:.0f} steps")

        col_left, col_right = st.columns(2)

        with col_left:
            fig_steps = px.bar(
                df_episodes, x="ep", y="steps",
                color="death_cause",
                color_discrete_map={"timeout": "#2ecc71", "starve": "#e74c3c", "pain": "#e67e22"},
                title="Survival Time per Episode",
                labels={"ep": "Episode", "steps": "Steps"},
            )
            fig_steps.update_layout(height=300, margin=dict(l=40, r=20, t=40, b=40))
            st.plotly_chart(fig_steps, width="stretch")

            fig_food = px.bar(
                df_episodes, x="ep", y="food_eaten",
                title="Food Eaten per Episode",
                labels={"ep": "Episode", "food_eaten": "Food"},
            )
            fig_food.update_traces(marker_color="#f39c12")
            fig_food.update_layout(height=300, margin=dict(l=40, r=20, t=40, b=40))
            st.plotly_chart(fig_food, width="stretch")

        with col_right:
            death_counts = df_episodes["death_cause"].value_counts()
            fig_death = px.pie(
                values=death_counts.values,
                names=death_counts.index,
                title="Death Causes",
                color=death_counts.index,
                color_discrete_map={"timeout": "#2ecc71", "starve": "#e74c3c", "pain": "#e67e22"},
            )
            fig_death.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_death, width="stretch")

            if "pain_visits" in df_episodes.columns:
                fig_pain = make_subplots(specs=[[{"secondary_y": True}]])
                fig_pain.add_trace(
                    go.Bar(x=df_episodes["ep"], y=df_episodes["pain_visits"],
                           name="Pain Visits", marker_color="#e74c3c", opacity=0.7),
                    secondary_y=False,
                )
                fig_pain.add_trace(
                    go.Scatter(x=df_episodes["ep"], y=df_episodes["pain_steps"],
                               name="Pain Steps", line=dict(color="#e67e22", width=2)),
                    secondary_y=True,
                )
                fig_pain.update_layout(
                    title="Pain Events",
                    height=300, margin=dict(l=40, r=40, t=40, b=40),
                )
                fig_pain.update_xaxes(title_text="Episode")
                fig_pain.update_yaxes(title_text="Visits", secondary_y=False)
                fig_pain.update_yaxes(title_text="Steps", secondary_y=True)
                st.plotly_chart(fig_pain, width="stretch")

        if not df_steps.empty and "energy" in df_steps.columns:
            st.subheader("Energy Over Time")
            ep_select = st.selectbox(
                "Episode", sorted(df_steps["ep"].unique()), key="energy_ep"
            )
            ep_energy = df_steps[df_steps["ep"] == ep_select]
            fig_energy = go.Figure()
            fig_energy.add_trace(go.Scatter(
                x=ep_energy["step"], y=ep_energy["energy"],
                mode="lines", name="Energy", line=dict(color="#3498db"),
            ))
            fig_energy.add_hline(y=30, line_dash="dash", line_color="red", annotation_text="Critical")
            fig_energy.add_hline(y=70, line_dash="dash", line_color="green", annotation_text="Target")
            fig_energy.update_layout(
                height=300, xaxis_title="Step", yaxis_title="Energy",
                margin=dict(l=40, r=20, t=20, b=40),
            )
            st.plotly_chart(fig_energy, width="stretch")


# ═══════════════════════════════════════════════════
# Page 4: Learning Progress
# ═══════════════════════════════════════════════════

elif page == "Learning Progress":
    st.header("Learning Progress")

    if df_hebbian.empty:
        st.warning("No Hebbian learning data available.")
    else:
        st.subheader("Hebbian Weight Evolution")
        synapses = df_hebbian["synapse"].unique()

        fig_hebb = go.Figure()
        for syn in synapses:
            syn_data = df_hebbian[df_hebbian["synapse"] == syn]
            syn_name = HEBBIAN_SYNAPSES.get(syn, {}).get("name", syn)
            fig_hebb.add_trace(go.Scatter(
                x=list(range(len(syn_data))),
                y=syn_data["avg_w"].values,
                mode="lines+markers",
                name=syn_name,
                marker=dict(size=3),
            ))
        fig_hebb.update_layout(
            height=400,
            xaxis_title="Learning Event",
            yaxis_title="Average Weight",
            margin=dict(l=60, r=20, t=20, b=60),
        )
        st.plotly_chart(fig_hebb, width="stretch")

        st.subheader("Per-Synapse Details")
        for syn in synapses:
            syn_data = df_hebbian[df_hebbian["synapse"] == syn]
            syn_info = HEBBIAN_SYNAPSES.get(syn, {})
            syn_name = syn_info.get("name", syn)

            with st.expander(f"{syn_name} (Phase {syn_info.get('phase', '?')})"):
                col1, col2 = st.columns(2)
                food_data = syn_data[syn_data["context"] == "food"]
                pain_data = syn_data[syn_data["context"] == "pain"]

                col1.metric("Food events", len(food_data))
                col2.metric("Pain events", len(pain_data))

                if not syn_data.empty:
                    col1.metric("Start weight", f"{syn_data['avg_w'].iloc[0]:.2f}")
                    col2.metric("Final weight", f"{syn_data['avg_w'].iloc[-1]:.2f}")

    if not df_steps.empty and "dopamine_rate" in df_steps.columns:
        st.subheader("Dopamine Activity")
        ep_avg_da = df_steps.groupby("ep")["dopamine_rate"].mean()
        fig_da = px.bar(
            x=ep_avg_da.index, y=ep_avg_da.values,
            title="Average Dopamine Rate per Episode",
            labels={"x": "Episode", "y": "Dopamine Rate"},
        )
        fig_da.update_traces(marker_color="#f1c40f")
        fig_da.update_layout(height=300, margin=dict(l=40, r=20, t=40, b=40))
        st.plotly_chart(fig_da, width="stretch")

    if not df_steps.empty and "food_eaten" in df_steps.columns:
        st.subheader("Reward Events Timeline")
        ep_select_learn = st.selectbox(
            "Episode", sorted(df_steps["ep"].unique()), key="learn_ep"
        )
        ep_rew = df_steps[df_steps["ep"] == ep_select_learn]
        reward_steps = ep_rew[ep_rew["food_eaten"] == True]

        fig_rew = go.Figure()
        fig_rew.add_trace(go.Scatter(
            x=ep_rew["step"], y=ep_rew.get("energy", pd.Series(dtype=float)),
            mode="lines", name="Energy", line=dict(color="#3498db", width=1),
        ))
        if not reward_steps.empty:
            fig_rew.add_trace(go.Scatter(
                x=reward_steps["step"],
                y=reward_steps.get("energy", pd.Series(dtype=float)),
                mode="markers", name="Food Eaten",
                marker=dict(color="#2ecc71", size=10, symbol="star"),
            ))
        fig_rew.update_layout(
            height=300, xaxis_title="Step", yaxis_title="Energy",
            margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_rew, width="stretch")

    if not df_steps.empty and len(df_steps["ep"].unique()) >= 4:
        st.subheader("Early vs Late Episode Activity")
        eps = sorted(df_steps["ep"].unique())
        n_compare = max(1, len(eps) // 4)
        early_eps = eps[:n_compare]
        late_eps = eps[-n_compare:]

        early_data = df_steps[df_steps["ep"].isin(early_eps)]
        late_data = df_steps[df_steps["ep"].isin(late_eps)]

        comparison_keys = [k for k in ALL_RATE_KEYS[:30] if k in df_steps.columns]
        if comparison_keys:
            early_means = [early_data[k].mean() for k in comparison_keys]
            late_means = [late_data[k].mean() for k in comparison_keys]
            labels = [RATE_KEY_LABELS.get(k, k) for k in comparison_keys]

            fig_compare = go.Figure()
            fig_compare.add_trace(go.Bar(
                name=f"Early (ep {early_eps[0]}-{early_eps[-1]})",
                x=labels, y=early_means,
                marker_color="rgba(52,152,219,0.7)",
            ))
            fig_compare.add_trace(go.Bar(
                name=f"Late (ep {late_eps[0]}-{late_eps[-1]})",
                x=labels, y=late_means,
                marker_color="rgba(231,76,60,0.7)",
            ))
            fig_compare.update_layout(
                barmode="group",
                height=400,
                xaxis=dict(tickangle=-45, tickfont=dict(size=8)),
                yaxis_title="Average Firing Rate",
                margin=dict(l=60, r=20, t=20, b=120),
            )
            st.plotly_chart(fig_compare, width="stretch")
