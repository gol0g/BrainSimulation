"""DataLogger for Genesis Brain training runs.

Logs per-step debug_info, episode summaries, and Hebbian learning events
to JSONL files for visualization in the Streamlit dashboard.
"""

import json
import os
from datetime import datetime
import numpy as np


class DataLogger:
    """Logs training data to JSONL/JSON files for dashboard visualization."""

    def __init__(self, log_dir=None, sample_rate=5, project_root=None):
        """
        Args:
            log_dir: Directory to save logs. Auto-generated if None.
            sample_rate: Log every N steps (default 5).
            project_root: Project root for logs/ directory. Auto-detected if None.
        """
        self.sample_rate = sample_rate

        if project_root is None:
            # Detect project root: go up from this file's directory
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))

        if log_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = os.path.join(project_root, "logs", f"run_{timestamp}")

        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        # Open file handles for streaming writes
        self._steps_file = open(os.path.join(self.log_dir, "steps.jsonl"), "w")
        self._episodes_file = open(os.path.join(self.log_dir, "episodes.jsonl"), "w")
        self._hebbian_file = open(os.path.join(self.log_dir, "hebbian.jsonl"), "w")

        self._step_count = 0
        self._flush_interval = 10  # flush every 10 logged steps for live dashboard

    def log_config(self, brain_config, env_config, episodes, cli_args=None):
        """Save run configuration (called once at start)."""
        config = {
            "timestamp": datetime.now().isoformat(),
            "episodes": episodes,
            "neurons": brain_config.total_neurons,
            "sample_rate": self.sample_rate,
            "phases_enabled": {
                "amygdala": brain_config.amygdala_enabled,
                "hippocampus": brain_config.hippocampus_enabled,
                "basal_ganglia": brain_config.basal_ganglia_enabled,
                "prefrontal": brain_config.prefrontal_enabled,
                "cerebellum": brain_config.cerebellum_enabled,
                "thalamus": brain_config.thalamus_enabled,
                "v1": brain_config.v1_enabled,
                "v2v4": brain_config.v2v4_enabled,
                "it": brain_config.it_enabled,
                "auditory": brain_config.auditory_enabled,
                "multimodal": brain_config.multimodal_enabled,
                "parietal": brain_config.parietal_enabled,
                "premotor": brain_config.premotor_enabled,
                "social_brain": brain_config.social_brain_enabled,
                "mirror": brain_config.mirror_enabled,
                "tom": brain_config.tom_enabled,
                "association_cortex": brain_config.association_cortex_enabled,
                "language": brain_config.language_enabled,
                "wm_expansion": brain_config.wm_expansion_enabled,
                "metacognition": brain_config.metacognition_enabled,
                "self_model": brain_config.self_model_enabled,
            },
            "cli_args": cli_args or {},
        }
        with open(os.path.join(self.log_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2, default=str)

    def log_step(self, episode, step, debug_info, env_info):
        """Log per-step data (sampled every sample_rate steps)."""
        if step % self.sample_rate != 0:
            return

        row = {"ep": int(episode), "step": int(step)}
        # env_info essentials
        row["energy"] = float(env_info.get("energy", 0))
        row["in_pain"] = bool(env_info.get("in_pain", False))
        row["food_eaten"] = bool(env_info.get("food_eaten", False))
        # All debug_info rates (already flat dict of floats)
        for key, value in debug_info.items():
            if isinstance(value, (np.floating, float)):
                row[key] = round(float(value), 4)
            elif isinstance(value, (np.integer, int)):
                row[key] = int(value)
            elif isinstance(value, (np.bool_, bool)):
                row[key] = bool(value)
        self._steps_file.write(json.dumps(row) + "\n")
        self._step_count += 1
        # Flush frequently for live dashboard updates
        if self._step_count % self._flush_interval == 0:
            self._steps_file.flush()
            self._hebbian_file.flush()

    def _to_native(self, obj):
        """Convert numpy types to native Python for JSON serialization."""
        if isinstance(obj, (np.floating, float)):
            return round(float(obj), 4)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return obj

    def log_episode(self, episode, summary):
        """Log episode summary (called once per episode end)."""
        row = {"ep": int(episode)}
        for k, v in summary.items():
            row[k] = self._to_native(v)
        self._episodes_file.write(json.dumps(row, default=str) + "\n")
        self._episodes_file.flush()

    def log_hebbian(self, episode, step, synapse_name, avg_w, context):
        """Log Hebbian learning event."""
        row = {
            "ep": episode,
            "step": step,
            "synapse": synapse_name,
            "avg_w": round(avg_w, 4),
            "context": context,
        }
        self._hebbian_file.write(json.dumps(row) + "\n")

    def close(self):
        """Close all file handles."""
        for f in [self._steps_file, self._episodes_file, self._hebbian_file]:
            if f and not f.closed:
                f.flush()
                f.close()
        print(f"  [LOG] Data saved to: {self.log_dir}")
