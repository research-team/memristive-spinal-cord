NEURON {
	SUFFIX rybak_model
	NONSPECIFIC_CURRENT ik
	NONSPECIFIC_CURRENT inap
	NONSPECIFIC_CURRENT il
	RANGE  gnap, gl, ena, ek, el, gk
}


UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	(nS) = (nanosiemens)
}

PARAMETER {
	:SOMA PARAMETERS
	gnap = 3.5	(nS)
	gl	= 1.6 (nS)
	gk = 4.5  (nS)
	ena     = 55.0  (mV)
	ek      = -80.0 (mV)
	el	= -64.0 (mV)
	tau_hmax = 600 (ms)
	dt              (ms)
	v               (mV)
}

STATE {
	 mk mnap hnap
}

ASSIGNED {
	inap 	(mA)
	il 	(mA)
	ik 	(mA)
	mk_inf
	mnap_inf
	hnap_inf
	tau_hnap
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	inap = gnap * mnap*hnap*(v - ena)
	ik   = gk *mk*mk*mk*mk*(v - ek)   
	il   = gl * (v - el)
}

DERIVATIVE states {  
	 : exact Hodgkin-Huxley equations
        evaluate_fct(v)
    :mk' = mk_inf
	:mnap' = mnap_inf
	hnap' = (hnap_inf - hnap) / tau_hnap
}

UNITSOFF

INITIAL {
	evaluate_fct(v)
	mk = mk_inf
	mnap = mnap_inf
	hnap = hnap_inf
}

PROCEDURE evaluate_fct(v(mV)) { LOCAL a,b,v2
	  
	 
	: SODIUM
	mnap_inf =  1/(1 + exp(-(v + 47.1)/3.1))
	:h
	tau_hnap = tau_hmax / (cosh((v + 51)/8))
	hnap_inf = 1/(1 + exp((v + 51)/4))

	
	: POTASSIUM 
	mk_inf = 1/(1+exp(-(v+44.5)/5))

}


UNITSON