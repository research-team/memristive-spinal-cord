
NEURON {
  POINT_PROCESS glutamate_syn
  NONSPECIFIC_CURRENT inmda,iampa
  RANGE e ,gampamax,gnmdamax,local_v,inmda,iampa,local_ca
  RANGE e ,local_v,inmda,iampa,local_ca
  RANGE decayampa,decaynmda,dampa,dnmda
  RANGE gnmda,gampa,xloc,yloc,tag1,tag2
  GLOBAL icaconst
  USEION ca READ cai WRITE ica
}

UNITS {
  (nA)  = (nanoamp)
  (mV)  = (millivolt)
  (nS)  = (nanomho)
  (mM)    = (milli/liter)
    F = 96480 (coul)
    R       = 8.314 (volt-coul/degC)
  PI = (pi) (1)
  (mA) = (milliamp)
  (um) = (micron)
}

PARAMETER {
  gnmdamax=1  (nS)
  gampamax=1  (nS)
  icaconst =0.1:1e-6
  e= 0.0  (mV)
  tau1=50 (ms)  : NMDA inactivation
  tau2=2  (ms)  : NMDA activation
  tau3=2  (ms)  : AMPA inactivation
  tau4=0.1  (ms)  : AMPA activation
  tau_ampa=2  (ms)  
  n=0.25  (/mM) 
  gama=0.08   (/mV) 
  dt    (ms)
  v   (mV)
  decayampa=.5
  decaynmda=.5
  taudampa=200  (ms):tau decay
  taudnmda=200  (ms):tau decay
  
  xloc=0
  yloc=0
  tag1=0
  tag2=0
}

ASSIGNED { 
  inmda   (nA)  
  iampa   (nA)  
  gnmda   (nS)
  gampa   (nS)
  ica     (nA)
  cai
  factor1   : NMDA normalization factor
  factor2   : AMPA normalization factor

}
STATE {
  A     (nS)
  B     (nS)
  C     (nS)
  D     (nS)
  :gampa  (nS)
  dampa
  dnmda
}


INITIAL {
  LOCAL tp1, tp2
      gnmda=0 
      gampa=0 
  A=0
  B=0
  C=0
  D=0
  dampa=1
  dnmda=1
  ica=0
  
  tp1 = (tau2*tau1)/(tau1 - tau2) * log(tau1/tau2)
  factor1 = -exp(-tp1/tau2) + exp(-tp1/tau1)
  factor1 = 1/factor1
  
  tp2 = (tau4*tau3)/(tau3 - tau4) * log(tau3/tau4)
  factor2 = -exp(-tp2/tau4) + exp(-tp2/tau3)
  factor2 = 1/factor2
}    

BREAKPOINT {  
    
  LOCAL count
  SOLVE state METHOD cnexp
  gnmda=(A-B)/(1+n*exp(-gama*v) )
  gampa=(C-D)
  inmda =(1e-3)*gnmda*(v-e)
  iampa= (1e-3)*gampa*(v- e)
  ica=inmda*0.1/(PI*diam)*icaconst
  inmda=inmda*.9

}
NET_RECEIVE(weight_ampa, weight_nmda) {
 
  INITIAL {
    gampamax = weight_ampa
    gnmdamax = weight_nmda
  }
  gampamax = weight_ampa
  gnmdamax = weight_nmda
  
  state_discontinuity( A, A+ factor1*gnmdamax*(dnmda))
  state_discontinuity( B, B+ factor1*gnmdamax*(dnmda))
  state_discontinuity( C, C+ factor2*gampamax*(dampa))
  state_discontinuity( D, D+ factor2*gampamax*(dampa))
  :state_discontinuity( gampa, gampa+ gampamax*dampa)
  state_discontinuity( dampa, dampa* decayampa)
  state_discontinuity( dnmda, dnmda* decaynmda)
}
DERIVATIVE state {
  A'=-A/tau1
  B'=-B/tau2
  C'=-C/tau3
  D'=-D/tau4
  :gampa'=-gampa/tau_ampa
  dampa'=(1-dampa)/taudampa
  dnmda'=(1-dnmda)/taudnmda
}
