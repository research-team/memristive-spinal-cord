NEURON {
  POINT_PROCESS AMPA_S
  RANGE g
  NONSPECIFIC_CURRENT i
  GLOBAL Cdur, Alpha, Beta, Erev, Rinf, Rtau
}
UNITS {
  (nA)   = (nanoamp)
  (mV)   = (millivolt)
  (umho) = (micromho)
}
PARAMETER {
  Cdur = 1.0 (ms) : transmitter duration (rising phase) 
  Alpha = 1.1 (/ms) : forward (binding) rate
  Beta = 0.19 (/ms) : backward (dissociation) rate
  Erev =0 (mV)  : equilibrium potential
}

ASSIGNED {
  v (mV) : postsynaptic voltage 
  i (nA) 
  g (umho) : conductance
  Rtau (ms) : time constant of channel binding
  Rinf : fraction of open channels if xmtr is present "forever" 
  synon : sum of weights of all synapses in the "onset" state
}
STATE { Ron Roff } : initialized to 0 by default
: Ron and Roff are the total conductances of all synapses
: that are in the "onset" (transmitter pulse ON)
: and "offset" (transmitter pulse OFF) states, respectively
INITIAL {
  synon = 0
  Rtau = 1 / (Alpha + Beta) Rinf = Alpha / (Alpha + Beta)
}
BREAKPOINT {
  SOLVE release METHOD cnexp
  g = (Ron + Roff)*1(umho)
  i = g*(v - Erev)
}
DERIVATIVE release {
  Ron' = (synon*Rinf - Ron)/Rtau
  Roff' = -Beta*Roff
}
NET_RECEIVE(weight, on, r0, t0 (ms)) {
  : flag is an implicit argument of NET_RECEIVE, normally 0
  if (flag == 0) {
: a spike arrived, start onset state if not already on 
  if (!on) {
      : this synapse joins the set of synapses in onset state
      synon = synon + weight
      r0 = r0*exp(-Beta*(t - t0)) : r0 at start of onset state
      Ron = Ron + r0
      Roff = Roff - r0
      t0 = t
      on = 1
: come again in Cdur with flag = 1 
      net_send(Cdur, 1)
    } else {
      : already in onset state, so move offset time
      net_move(t + Cdur)
    }
  }
  if (flag == 1) {
    : "turn off transmitter"
    : i.e. this synapse enters the offset state
    synon = synon - weight
    : r0 at start of offset state
    r0 = weight*Rinf + (r0 - weight*Rinf)*exp(-(t - t0)/Rtau)
    Ron = Ron - r0
    Roff = Roff + r0
    t0 = t
    on = 0
} 
}