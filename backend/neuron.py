import random

class IzhikevichNeuron:
    def __init__(self, neuron_id, a=0.02, b=0.2, c=-65, d=8):
        """
        Initialize the Izhikevich neuron model.
        
        Parameters:
        - a: Time scale of the recovery variable u. Smaller values => slower recovery.
        - b: Sensitivity of u to the subthreshold fluctuations of v.
        - c: After-spike reset value of v.
        - d: After-spike reset of u.
        
        Standard behaviors:
        - Regular Spiking (RS): a=0.02, b=0.2, c=-65, d=8
        - Fast Spiking (FS):    a=0.1,  b=0.2, c=-65, d=2
        """
        self.neuron_id = neuron_id
        
        # Parameters
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        
        # State variables
        self.v = -65.0  # Membrane potential (mV)
        self.u = self.b * self.v  # Recovery variable
        
        # History for visualization (keep last N points if needed, 
        # but primarily we stream state)
        self.spike_history = [] 

    def step(self, i_inject=0.0, dt=1.0):
        """
        Step the simulation by dt milliseconds.
        
        Differential Equations:
        dv/dt = 0.04*v^2 + 5*v + 140 - u + I
        du/dt = a*(b*v - u)
        
        Using simple Euler integration.
        """
        # Calculate derivatives
        dv = (0.04 * self.v**2 + 5 * self.v + 140 - self.u + i_inject)
        du = (self.a * (self.b * self.v - self.u))
        
        # Update state
        self.v += dv * dt
        self.u += du * dt
        
        # Check for spike
        fired = False
        if self.v >= 30.0:
            fired = True
            self.spike_history.append(True)
            # Reset
            self.v = self.c
            self.u += self.d
        else:
            self.spike_history.append(False)
            
        return {
            "id": self.neuron_id,
            "v": self.v,
            "u": self.u,
            "fired": fired
        }

    def set_parameters(self, a=None, b=None, c=None, d=None):
        if a is not None: self.a = a
        if b is not None: self.b = b
        if c is not None: self.c = c
        if d is not None: self.d = d
