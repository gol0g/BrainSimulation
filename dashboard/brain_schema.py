"""Genesis Brain structure definition for dashboard visualization.

Static definitions of all ~80 populations, key connections,
and debug_info key mappings across 20 phases.
"""

# ─── Phase definitions ───
PHASES = {
    "1":   {"name": "Brainstem Reflexes", "color": "#e74c3c", "y": 0},
    "2a":  {"name": "Hypothalamus",       "color": "#e67e22", "y": 1},
    "2b":  {"name": "Amygdala",           "color": "#c0392b", "y": 2},
    "3":   {"name": "Hippocampus",        "color": "#27ae60", "y": 3},
    "4":   {"name": "Basal Ganglia",      "color": "#f1c40f", "y": 4},
    "5":   {"name": "Prefrontal Cortex",  "color": "#9b59b6", "y": 5},
    "6a":  {"name": "Cerebellum",         "color": "#1abc9c", "y": 6},
    "6b":  {"name": "Thalamus",           "color": "#3498db", "y": 7},
    "8":   {"name": "V1 Visual Cortex",   "color": "#2ecc71", "y": 8},
    "9":   {"name": "V2/V4 Visual",       "color": "#16a085", "y": 9},
    "10":  {"name": "IT Cortex",          "color": "#0e6655", "y": 10},
    "11":  {"name": "Auditory Cortex",    "color": "#2980b9", "y": 11},
    "12":  {"name": "STS Multimodal",     "color": "#8e44ad", "y": 12},
    "13":  {"name": "Parietal PPC",       "color": "#d35400", "y": 13},
    "14":  {"name": "Premotor PMC",       "color": "#a93226", "y": 14},
    "15":  {"name": "Social Brain",       "color": "#e91e63", "y": 15},
    "15b": {"name": "Mirror Neurons",     "color": "#ff5722", "y": 16},
    "15c": {"name": "Theory of Mind",     "color": "#d81b60", "y": 17},
    "16":  {"name": "Association Cortex", "color": "#795548", "y": 18},
    "17":  {"name": "Language Circuit",   "color": "#607d8b", "y": 19},
    "18":  {"name": "Working Memory Exp", "color": "#9c27b0", "y": 20},
    "19":  {"name": "Metacognition",      "color": "#00bcd4", "y": 21},
    "20":  {"name": "Self-Model",         "color": "#ff9800", "y": 22},
}

PHASE_ORDER = [
    "1", "2a", "2b", "3", "4", "5", "6a", "6b",
    "8", "9", "10", "11", "12", "13", "14",
    "15", "15b", "15c", "16", "17", "18", "19", "20",
]

# ─── Population definitions ───
# (id, display_name, neuron_count, phase, rate_key)
# rate_key maps to debug_info dict key (None = input signal, not a rate)
POPULATIONS = [
    # Phase 1: Brainstem
    ("food_eye_left",    "Food Eye L",    400, "1",  "food_l"),
    ("food_eye_right",   "Food Eye R",    400, "1",  "food_r"),
    ("wall_eye_left",    "Wall Eye L",    200, "1",  "wall_l"),
    ("wall_eye_right",   "Wall Eye R",    200, "1",  "wall_r"),
    ("motor_left",       "Motor L",       500, "1",  "motor_left_rate"),
    ("motor_right",      "Motor R",       500, "1",  "motor_right_rate"),

    # Phase 2a: Hypothalamus
    ("low_energy",       "Low Energy",    200, "2a", "low_energy_rate"),
    ("high_energy",      "High Energy",   200, "2a", "high_energy_rate"),
    ("hunger_drive",     "Hunger",        500, "2a", "hunger_rate"),
    ("satiety_drive",    "Satiety",       500, "2a", "satiety_rate"),

    # Phase 2b: Amygdala
    ("pain_eye_left",    "Pain Eye L",    200, "2b", "pain_l"),
    ("pain_eye_right",   "Pain Eye R",    200, "2b", "pain_r"),
    ("danger_sensor",    "Danger",        200, "2b", "danger_signal"),
    ("lateral_amygdala", "LA",            500, "2b", "la_rate"),
    ("central_amygdala", "CEA",           300, "2b", "cea_rate"),
    ("fear_response",    "Fear",          200, "2b", "fear_rate"),

    # Phase 3: Hippocampus
    ("place_cells",      "Place Cells",   400, "3",  "place_cell_rate"),
    ("food_memory_left", "Food Mem L",    100, "3",  "food_memory_rate"),
    ("food_memory_right","Food Mem R",    100, "3",  "food_memory_rate"),

    # Phase 4: Basal Ganglia
    ("striatum",         "Striatum",      400, "4",  "striatum_rate"),
    ("direct_pathway",   "Direct (Go)",   200, "4",  "direct_rate"),
    ("indirect_pathway", "Indirect (NoGo)", 200, "4", "indirect_rate"),
    ("dopamine",         "Dopamine",      100, "4",  "dopamine_rate"),

    # Phase 5: Prefrontal Cortex
    ("working_memory",   "Working Mem",   200, "5",  "working_memory_rate"),
    ("goal_food",        "Goal Food",      50, "5",  "goal_food_rate"),
    ("goal_safety",      "Goal Safety",    50, "5",  "goal_safety_rate"),
    ("inhibitory_ctrl",  "Inhibitory",    100, "5",  "inhibitory_rate"),

    # Phase 6a: Cerebellum
    ("granule_cells",    "Granule",       300, "6a", "granule_rate"),
    ("purkinje_cells",   "Purkinje",      100, "6a", "purkinje_rate"),
    ("deep_nuclei",      "Deep Nuclei",   100, "6a", "deep_nuclei_rate"),
    ("error_signal",     "Error",          50, "6a", "error_rate"),

    # Phase 6b: Thalamus
    ("food_relay",       "Food Relay",    100, "6b", "food_relay_rate"),
    ("danger_relay",     "Danger Relay",  100, "6b", "danger_relay_rate"),
    ("trn",              "TRN",           100, "6b", "trn_rate"),
    ("arousal",          "Arousal",        50, "6b", "arousal_rate"),

    # Phase 8: V1 Visual Cortex
    ("v1_food_left",     "V1 Food L",     100, "8",  "v1_food_left_rate"),
    ("v1_food_right",    "V1 Food R",     100, "8",  "v1_food_right_rate"),
    ("v1_danger_left",   "V1 Danger L",   100, "8",  "v1_danger_left_rate"),
    ("v1_danger_right",  "V1 Danger R",   100, "8",  "v1_danger_right_rate"),

    # Phase 9: V2/V4
    ("v2_edge_food",     "V2 Food",       150, "9",  "v2_edge_food_rate"),
    ("v2_edge_danger",   "V2 Danger",     150, "9",  "v2_edge_danger_rate"),
    ("v4_food_object",   "V4 Food",       100, "9",  "v4_food_object_rate"),
    ("v4_danger_object", "V4 Danger",     100, "9",  "v4_danger_object_rate"),
    ("v4_novel_object",  "V4 Novel",      100, "9",  "v4_novel_object_rate"),

    # Phase 10: IT Cortex
    ("it_food_cat",      "IT Food",       200, "10", "it_food_category_rate"),
    ("it_danger_cat",    "IT Danger",     200, "10", "it_danger_category_rate"),
    ("it_neutral_cat",   "IT Neutral",    150, "10", "it_neutral_category_rate"),
    ("it_association",   "IT Assoc",      200, "10", "it_association_rate"),
    ("it_memory_buf",    "IT Memory",     250, "10", "it_memory_buffer_rate"),

    # Phase 11: Auditory Cortex
    ("a1_danger",        "A1 Danger",     150, "11", "a1_danger_rate"),
    ("a1_food",          "A1 Food",       150, "11", "a1_food_rate"),
    ("a2_association",   "A2 Assoc",      200, "11", "a2_association_rate"),

    # Phase 12: STS Multimodal
    ("sts_food",         "STS Food",      200, "12", "sts_food_rate"),
    ("sts_danger",       "STS Danger",    200, "12", "sts_danger_rate"),
    ("sts_congruence",   "STS Congr",     150, "12", "sts_congruence_rate"),
    ("sts_mismatch",     "STS Mismatch",  100, "12", "sts_mismatch_rate"),

    # Phase 13: Parietal PPC
    ("ppc_space_left",   "PPC Space L",   150, "13", "ppc_space_left_rate"),
    ("ppc_space_right",  "PPC Space R",   150, "13", "ppc_space_right_rate"),
    ("ppc_goal_food",    "PPC Goal F",    150, "13", "ppc_goal_food_rate"),
    ("ppc_goal_safety",  "PPC Goal S",    150, "13", "ppc_goal_safety_rate"),
    ("ppc_attention",    "PPC Attention",  200, "13", "ppc_attention_rate"),
    ("ppc_path_buffer",  "PPC Path",      200, "13", "ppc_path_buffer_rate"),

    # Phase 14: Premotor PMC
    ("pmd_left",         "PMd L",         100, "14", "pmd_left_rate"),
    ("pmd_right",        "PMd R",         100, "14", "pmd_right_rate"),
    ("pmv_approach",     "PMv Approach",  100, "14", "pmv_approach_rate"),
    ("pmv_avoid",        "PMv Avoid",     100, "14", "pmv_avoid_rate"),
    ("sma_sequence",     "SMA Seq",       150, "14", "sma_sequence_rate"),
    ("motor_prep",       "Motor Prep",    150, "14", "motor_prep_rate"),

    # Phase 15: Social Brain
    ("sts_social",       "STS Social",    200, "15", "sts_social_rate"),
    ("tpj_self",         "TPJ Self",      100, "15", "tpj_self_rate"),
    ("tpj_other",        "TPJ Other",     100, "15", "tpj_other_rate"),
    ("tpj_compare",      "TPJ Compare",   100, "15", "tpj_compare_rate"),
    ("acc_conflict",     "ACC Conflict",  100, "15", "acc_conflict_rate"),
    ("acc_monitor",      "ACC Monitor",   100, "15", "acc_monitor_rate"),
    ("social_approach",  "Soc Approach",  100, "15", "social_approach_rate"),
    ("social_avoid",     "Soc Avoid",     100, "15", "social_avoid_rate"),

    # Phase 15b: Mirror Neurons
    ("social_obs",       "Social Obs",    200, "15b", "social_obs_rate"),
    ("mirror_food",      "Mirror Food",   150, "15b", "mirror_food_rate"),
    ("vicarious_reward", "Vicar Reward",  100, "15b", "vicarious_reward_rate"),
    ("social_memory",    "Social Mem",    150, "15b", "social_memory_rate"),

    # Phase 15c: Theory of Mind
    ("tom_intention",    "ToM Intent",    100, "15c", "tom_intention_rate"),
    ("tom_belief",       "ToM Belief",     80, "15c", "tom_belief_rate"),
    ("tom_prediction",   "ToM Predict",    80, "15c", "tom_prediction_rate"),
    ("tom_surprise",     "ToM Surprise",   60, "15c", "tom_surprise_rate"),
    ("coop",             "Cooperate",      80, "15c", "coop_rate"),
    ("compete",          "Compete",       100, "15c", "compete_rate"),

    # Phase 16: Association Cortex
    ("assoc_edible",     "Edible",        120, "16", "assoc_edible_rate"),
    ("assoc_threatening","Threatening",    120, "16", "assoc_threatening_rate"),
    ("assoc_animate",    "Animate",       100, "16", "assoc_animate_rate"),
    ("assoc_context",    "Assoc Context", 100, "16", "assoc_context_rate"),
    ("assoc_valence",    "Valence",        80, "16", "assoc_valence_rate"),
    ("assoc_binding",    "Assoc Binding", 100, "16", "assoc_binding_rate"),
    ("assoc_novelty",    "Novelty",        80, "16", "assoc_novelty_rate"),

    # Phase 17: Language Circuit
    ("wernicke_food",    "Wern Food",      80, "17", "wernicke_food_rate"),
    ("wernicke_danger",  "Wern Danger",    80, "17", "wernicke_danger_rate"),
    ("wernicke_social",  "Wern Social",    60, "17", "wernicke_social_rate"),
    ("wernicke_context", "Wern Context",   60, "17", "wernicke_context_rate"),
    ("broca_food",       "Broca Food",     80, "17", "broca_food_rate"),
    ("broca_danger",     "Broca Danger",   80, "17", "broca_danger_rate"),
    ("broca_social",     "Broca Social",   60, "17", "broca_social_rate"),
    ("broca_sequence",   "Broca Seq",      60, "17", "broca_sequence_rate"),
    ("vocal_gate",       "Vocal Gate",     80, "17", "vocal_gate_rate"),
    ("call_mirror",      "Call Mirror",    80, "17", "call_mirror_rate"),
    ("call_binding",     "Call Binding",   80, "17", "call_binding_rate"),

    # Phase 18: Working Memory Expansion
    ("wm_thalamic",      "WM Thalamic",   100, "18", "wm_thalamic_rate"),
    ("wm_update_gate",   "WM Gate",        50, "18", "wm_update_gate_rate"),
    ("temporal_recent",  "Temp Recent",    80, "18", "temporal_recent_rate"),
    ("temporal_prior",   "Temp Prior",     40, "18", "temporal_prior_rate"),
    ("goal_pending",     "Goal Pending",   80, "18", "goal_pending_rate"),
    ("goal_switch",      "Goal Switch",    70, "18", "goal_switch_rate"),
    ("wm_context_bind",  "WM Ctx Bind",   100, "18", "wm_context_binding_rate"),
    ("wm_inhibitory",    "WM Inhibit",    100, "18", "wm_inhibitory_rate"),

    # Phase 19: Metacognition
    ("meta_confidence",  "Confidence",     80, "19", "meta_confidence_rate"),
    ("meta_uncertainty", "Uncertainty",    80, "19", "meta_uncertainty_rate"),
    ("meta_evaluate",    "Evaluate",       80, "19", "meta_evaluate_rate"),
    ("meta_arousal",     "Arousal Mod",    70, "19", "meta_arousal_mod_rate"),
    ("meta_inhibitory",  "Meta Inhibit",   70, "19", "meta_inhibitory_rate"),

    # Phase 20: Self-Model
    ("self_body",        "Body (Insula)",   80, "20", "self_body_rate"),
    ("self_efference",   "Efference",      80, "20", "self_efference_rate"),
    ("self_predict",     "Predict",        70, "20", "self_predict_rate"),
    ("self_agency",      "Agency",         70, "20", "self_agency_rate"),
    ("self_narrative",   "Narrative",      80, "20", "self_narrative_rate"),
    ("self_inhibitory",  "Self Inhibit",   60, "20", "self_inhibitory_sm_rate"),
]

# ─── Rate keys grouped by phase (for heatmap Y-axis grouping) ───
RATE_KEYS_BY_PHASE = {
    "1":   ["motor_left_rate", "motor_right_rate"],
    "2a":  ["low_energy_rate", "high_energy_rate", "hunger_rate", "satiety_rate"],
    "2b":  ["la_rate", "cea_rate", "fear_rate"],
    "3":   ["place_cell_rate", "food_memory_rate"],
    "4":   ["striatum_rate", "direct_rate", "indirect_rate", "dopamine_rate"],
    "5":   ["working_memory_rate", "goal_food_rate", "goal_safety_rate", "inhibitory_rate"],
    "6a":  ["granule_rate", "purkinje_rate", "deep_nuclei_rate", "error_rate"],
    "6b":  ["food_relay_rate", "danger_relay_rate", "trn_rate", "arousal_rate"],
    "8":   ["v1_food_left_rate", "v1_food_right_rate", "v1_danger_left_rate", "v1_danger_right_rate"],
    "9":   ["v2_edge_food_rate", "v2_edge_danger_rate", "v4_food_object_rate", "v4_danger_object_rate", "v4_novel_object_rate"],
    "10":  ["it_food_category_rate", "it_danger_category_rate", "it_neutral_category_rate", "it_association_rate", "it_memory_buffer_rate"],
    "11":  ["a1_danger_rate", "a1_food_rate", "a2_association_rate"],
    "12":  ["sts_food_rate", "sts_danger_rate", "sts_congruence_rate", "sts_mismatch_rate"],
    "13":  ["ppc_space_left_rate", "ppc_space_right_rate", "ppc_goal_food_rate", "ppc_goal_safety_rate", "ppc_attention_rate", "ppc_path_buffer_rate"],
    "14":  ["pmd_left_rate", "pmd_right_rate", "pmv_approach_rate", "pmv_avoid_rate", "sma_sequence_rate", "motor_prep_rate"],
    "15":  ["sts_social_rate", "tpj_self_rate", "tpj_other_rate", "tpj_compare_rate", "acc_conflict_rate", "acc_monitor_rate", "social_approach_rate", "social_avoid_rate"],
    "15b": ["social_obs_rate", "mirror_food_rate", "vicarious_reward_rate", "social_memory_rate"],
    "15c": ["tom_intention_rate", "tom_belief_rate", "tom_prediction_rate", "tom_surprise_rate", "coop_rate", "compete_rate"],
    "16":  ["assoc_edible_rate", "assoc_threatening_rate", "assoc_animate_rate", "assoc_context_rate", "assoc_valence_rate", "assoc_binding_rate", "assoc_novelty_rate"],
    "17":  ["wernicke_food_rate", "wernicke_danger_rate", "wernicke_social_rate", "wernicke_context_rate", "broca_food_rate", "broca_danger_rate", "broca_social_rate", "broca_sequence_rate", "vocal_gate_rate", "call_mirror_rate", "call_binding_rate"],
    "18":  ["wm_thalamic_rate", "wm_update_gate_rate", "temporal_recent_rate", "temporal_prior_rate", "goal_pending_rate", "goal_switch_rate", "wm_context_binding_rate", "wm_inhibitory_rate"],
    "19":  ["meta_confidence_rate", "meta_uncertainty_rate", "meta_evaluate_rate", "meta_arousal_mod_rate", "meta_inhibitory_rate"],
    "20":  ["self_body_rate", "self_efference_rate", "self_predict_rate", "self_agency_rate", "self_narrative_rate", "self_inhibitory_sm_rate"],
}

# Flat ordered list of all rate keys (for heatmap Y-axis)
ALL_RATE_KEYS = []
RATE_KEY_LABELS = {}  # rate_key -> "Phase X: Display Name"
for phase_id in PHASE_ORDER:
    phase_name = PHASES[phase_id]["name"]
    for key in RATE_KEYS_BY_PHASE.get(phase_id, []):
        ALL_RATE_KEYS.append(key)
        # Find matching population for display name
        pop_name = key.replace("_rate", "").replace("_", " ").title()
        for _, display, _, p, rk in POPULATIONS:
            if rk == key and p == phase_id:
                pop_name = display
                break
        RATE_KEY_LABELS[key] = f"P{phase_id}: {pop_name}"

# ─── Key connections (major pathways only, ~50) ───
# (source_id, target_id, weight, description)
CONNECTIONS = [
    # Phase 1: Reflexes
    ("wall_eye_left",    "motor_right",      60.0, "Pain Push L→R"),
    ("wall_eye_left",    "motor_left",      -40.0, "Pain Pull L"),
    ("wall_eye_right",   "motor_left",       60.0, "Pain Push R→L"),
    ("wall_eye_right",   "motor_right",     -40.0, "Pain Pull R"),
    ("food_eye_left",    "motor_left",       40.0, "Food Approach L"),
    ("food_eye_right",   "motor_right",      40.0, "Food Approach R"),

    # Phase 2a: Homeostasis
    ("low_energy",       "hunger_drive",     30.0, "Low E → Hunger"),
    ("high_energy",      "satiety_drive",    25.0, "High E → Satiety"),
    ("hunger_drive",     "satiety_drive",   -20.0, "H↔S WTA"),
    ("hunger_drive",     "food_eye_left",    12.0, "Hunger Amplify"),

    # Phase 2b: Fear
    ("pain_eye_left",    "lateral_amygdala", 50.0, "Pain → LA"),
    ("lateral_amygdala", "central_amygdala", 30.0, "LA → CEA"),
    ("central_amygdala", "fear_response",    25.0, "CEA → Fear"),
    ("hunger_drive",     "central_amygdala",-15.0, "Hunger suppress Fear"),

    # Phase 3: Hippocampus
    ("place_cells",      "food_memory_left",  2.0, "Place → FoodMem (Hebbian)"),
    ("food_memory_left", "motor_left",       12.0, "FoodMem → Motor"),

    # Phase 4: Basal Ganglia
    ("striatum",         "direct_pathway",   20.0, "Striatum → Go"),
    ("striatum",         "indirect_pathway", 15.0, "Striatum → NoGo"),
    ("direct_pathway",   "motor_left",       15.0, "Go → Motor"),
    ("indirect_pathway", "motor_left",       -8.0, "NoGo → Motor"),
    ("dopamine",         "direct_pathway",   25.0, "DA → Go"),
    ("dopamine",         "indirect_pathway",-20.0, "DA → NoGo"),

    # Phase 5: PFC
    ("working_memory",   "goal_food",         8.0, "WM → Goal"),
    ("goal_food",        "motor_left",       18.0, "Goal → Motor"),

    # Phase 6a: Cerebellum
    ("granule_cells",    "purkinje_cells",   10.0, "Granule → Purkinje"),
    ("deep_nuclei",      "motor_left",        5.0, "Cerebellum → Motor"),

    # Phase 6b: Thalamus
    ("food_relay",       "food_eye_left",     8.0, "Thal relay Food"),
    ("trn",              "food_relay",       -8.0, "TRN gate"),

    # Phase 8-10: Visual pathway
    ("food_eye_left",    "v1_food_left",     10.0, "Eye → V1"),
    ("v1_food_left",     "v2_edge_food",      8.0, "V1 → V2"),
    ("v4_food_object",   "it_food_cat",       8.0, "V4 → IT"),

    # Phase 12: Multimodal
    ("v1_food_left",     "sts_food",          8.0, "V1 → STS"),
    ("a1_food",          "sts_food",          8.0, "A1 → STS"),

    # Phase 15: Social
    ("sts_social",       "tpj_other",         8.0, "STS → TPJ"),
    ("acc_conflict",     "inhibitory_ctrl",   8.0, "ACC → PFC Inhibit"),

    # Phase 16: Association
    ("it_food_cat",      "assoc_edible",      6.0, "IT → Edible"),
    ("assoc_edible",     "assoc_threatening", -6.0, "Edible↔Threat WTA"),

    # Phase 17: Language
    ("wernicke_food",    "broca_food",        8.0, "Wernicke → Broca"),
    ("broca_food",       "vocal_gate",        6.0, "Broca → Gate"),

    # Phase 18: WM Expansion
    ("working_memory",   "wm_thalamic",       6.0, "WM → Thalamic"),
    ("wm_thalamic",      "working_memory",    5.0, "Thalamic → WM (loop)"),
    ("wm_update_gate",   "wm_thalamic",      -6.0, "Gate → Thalamic"),

    # Phase 19: Metacognition
    ("meta_confidence",  "goal_food",          1.5, "Confidence → Goal"),
    ("meta_uncertainty", "goal_switch",        2.0, "Uncertainty → Switch"),

    # Phase 20: Self-Model
    ("self_body",        "self_narrative",     3.0, "Body → Narrative"),
    ("self_efference",   "self_agency",        4.0, "Efference → Agency"),
    ("self_agency",      "goal_food",          1.0, "Agency → Goal"),
]

# ─── Hebbian synapse definitions ───
HEBBIAN_SYNAPSES = {
    "hippo": {"name": "Hippocampus", "from": "place_cells", "to": "food_memory_left", "phase": "3"},
    "assoc_binding": {"name": "Assoc Binding", "from": "assoc_edible", "to": "assoc_binding", "phase": "16"},
    "call_binding": {"name": "Call Binding", "from": "wernicke_food", "to": "call_binding", "phase": "17"},
    "wm_context": {"name": "WM Context", "from": "wm_thalamic", "to": "wm_context_bind", "phase": "18"},
    "meta_confidence": {"name": "Meta Confidence", "from": "meta_evaluate", "to": "meta_confidence", "phase": "19"},
    "self_narrative": {"name": "Self Narrative", "from": "self_body", "to": "self_narrative", "phase": "20"},
}
