from typing import Dict, List, Optional
from neuron import IzhikevichNeuron
from synapse import Synapse

class Network:
    """
    Manages neurons and synapses with STDP learning.
    """
    
    def __init__(self):
        self.neurons: Dict[str, IzhikevichNeuron] = {}
        self.synapses: List[Synapse] = []
        self.time_step = 0
        
    def add_neuron(
        self, 
        neuron_id: str, 
        neuron_type: str = "RS"
    ) -> IzhikevichNeuron:
        params = {
            "RS": {"a": 0.02, "b": 0.2, "c": -65, "d": 8},
            "FS": {"a": 0.1, "b": 0.2, "c": -65, "d": 2},
            "CH": {"a": 0.02, "b": 0.2, "c": -50, "d": 2},
        }
        p = params.get(neuron_type, params["RS"])
        
        neuron = IzhikevichNeuron(
            neuron_id=neuron_id,
            a=p["a"], b=p["b"], c=p["c"], d=p["d"]
        )
        self.neurons[neuron_id] = neuron
        return neuron
    
    def connect(
        self,
        pre_id: str,
        post_id: str,
        weight: float = 5.0,
        delay: int = 3,
        enable_stdp: bool = True,
        is_inhibitory: bool = False
    ) -> Synapse:
        if pre_id not in self.neurons or post_id not in self.neurons:
            raise ValueError(f"Neuron not found: {pre_id} or {post_id}")

        syn = Synapse(
            pre_neuron_id=pre_id,
            post_neuron_id=post_id,
            weight=weight,
            delay=delay,
            enable_stdp=enable_stdp,
            is_inhibitory=is_inhibitory
        )
        self.synapses.append(syn)
        return syn
    
    def step(
        self, 
        external_currents: Optional[Dict[str, float]] = None,
        dt: float = 0.5,
        noise_level: float = 0.0
    ) -> Dict:
        if external_currents is None:
            external_currents = {}
            
        import random  # Ensure random is available

        # 1. Collect synaptic currents
        synaptic_currents: Dict[str, float] = {nid: 0.0 for nid in self.neurons}
        
        for syn in self.synapses:
            current = syn.step()
            synaptic_currents[syn.post_neuron_id] += current
            
        # 2. Step each neuron
        results = {}
        fired_neurons = []
        
        for nid, neuron in self.neurons.items():
            ext_current = external_currents.get(nid, 0.0)
            syn_current = synaptic_currents.get(nid, 0.0)
            total_current = ext_current + syn_current
            
            # Add stochastic noise
            if noise_level > 0:
                total_current += random.uniform(-noise_level, noise_level)
            
            state = neuron.step(i_inject=total_current, dt=dt)
            results[nid] = state
            
            if state["fired"]:
                fired_neurons.append(nid)
                
        # 3. Process pre-synaptic spikes (propagate to synapses)
        for syn in self.synapses:
            if syn.pre_neuron_id in fired_neurons:
                syn.receive_spike(self.time_step)
                
        # 4. Process post-synaptic spikes (for STDP)
        for syn in self.synapses:
            if syn.post_neuron_id in fired_neurons:
                syn.notify_post_spike(self.time_step)
                
        self.time_step += 1
        
        return {
            "time": self.time_step,
            "neurons": results,
            "synapses": [s.to_dict() for s in self.synapses]
        }
    
    def apply_reward(self, reward: float):
        """Broadcast global reward signal to all synapses for 3rd factor learning."""
        for syn in self.synapses:
            syn.apply_reward(reward, self.time_step)
            
    def reset_neurons(self):
        """Reset only neuron states and synaptic queues, preserve learned weights."""
        for neuron in self.neurons.values():
            neuron.v = -65.0
            neuron.u = neuron.b * neuron.v
            neuron.spike_history = []

        for syn in self.synapses:
            syn.spike_queue.clear()
            syn.last_pre_spike_time = None
            syn.last_post_spike_time = None
            syn.eligibility = 0.0
            syn.synaptic_current = 0.0
            # Reset activity tracking to prevent incorrect homeostasis decay
            syn.current_time = 0
            syn.last_activity_time = 0

        self.time_step = 0

    def reset(self):
        """Full reset including weights (for complete restart)."""
        for neuron in self.neurons.values():
            neuron.v = -65.0
            neuron.u = neuron.b * neuron.v
            neuron.spike_history = []

        for syn in self.synapses:
            syn.spike_queue.clear()
            syn.last_pre_spike_time = None
            syn.last_post_spike_time = None
            syn.weight = getattr(syn, 'initial_weight', 25.0)
            syn.weight_history = [syn.weight]
            syn.stdp_events = []
            syn.eligibility = 0.0

        self.time_step = 0
        
    def get_state(self) -> Dict:
        return {
            "time": self.time_step,
            "neurons": {
                nid: {"v": n.v, "u": n.u}
                for nid, n in self.neurons.items()
            },
            "synapses": [s.to_dict() for s in self.synapses]
        }
