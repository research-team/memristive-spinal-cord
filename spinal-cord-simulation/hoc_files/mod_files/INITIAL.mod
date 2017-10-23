:INITIAL SEGMENT


: Marco Capogrosso
:
:
: This model has been adapted and is described in detail in:
:
: McIntyre CC and Grill WM. Extracellular Stimulation of Central Neurons:
: Influence of Stimulus Waveform and Frequency on Neuronal Output
: Journal of Neurophysiology 88:1592-1604, 2002.

TITLE Motor Neuron initial segment
INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
	SUFFIX initial
	NONSPECIFIC_CURRENT ina
	NONSPECIFIC_CURRENT ikrect
	NONSPECIFIC_CURRENT inap
	NONSPECIFIC_CURRENT il
	RANGE  gnabar, gl, ena, ek, el, gkrect, gnap
	RANGE p_inf, m_inf, h_inf, n_inf
	RANGE tau_p, tau_m, tau_h, tau_n
}


UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {
	:SOMA PARAMETERS
	gnabar	= 0.5	(mho/cm2)
	gl	= 0.01 (mho/cm2)
	gkrect = 0.1  (mho/cm2)
	gnap =0.01 (mho/gm2)
	ena     = 50.0  (mV)
	ek      = -80.0 (mV)
	el	= -70.0 (mV)
	dt              (ms)
	v               (mV)
	amA = 0.4
	amB = 60
	amC = 5
	bmA = 0.4
	bmB = 40
	bmC = 5
	ampA = 0.0353
	ampB = 28.4
	ampC = 5
	bmpA = 0.000883
	bmpB = 32.7
	bmpC = 5
}

STATE {
	 p m h n 
}

ASSIGNED {
	ina	 (mA/cm2)
	il      (mA/cm2)
	ikrect    (mA/cm2)
	inap  (mA/cm2)
	m_inf
	h_inf
	n_inf
	p_inf
	tau_m
	tau_h
	tau_p
	tau_n
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	ina = gnabar * m*m*m*h*(v - ena)
	ikrect   = gkrect *n*n*n*n*(v - ek)   
	inap = gnap *p*p*p*(v-ena)
	il   = gl * (v - el)
}

DERIVATIVE states {  
	 : exact Hodgkin-Huxley equations
        evaluate_fct(v)
	m' = (m_inf - m) / tau_m
	h' = (h_inf - h) / tau_h
	p' = (p_inf - p) / tau_p
	n' = (n_inf - n) / tau_n
}

UNITSOFF

INITIAL {
	evaluate_fct(v)
	m = m_inf
	h = h_inf
	p = p_inf
	n = n_inf
}

PROCEDURE evaluate_fct(v(mV)) { LOCAL a,b,v2
	  
	 
	:FAST SODIUM
	:m
	a = alpham(v)
	b = betam(v)
	tau_m = 1 / (a + b)
	m_inf = a / (a + b)
	:h
	tau_h = 30 / (Exp((v+60)/15) + Exp(-(v+60)/16))
	h_inf = 1 / (1 + Exp((v+65)/7))


	:Persistent Sodium
	a = vtrap1(v)
	b = vtrap2(v)
	tau_p = 1 / (a + b)
	p_inf = a / (a + b)

	
	:DELAYED RECTIFIER POTASSIUM 
	tau_n = 5 / (Exp((v+50)/40) + Exp(-(v+50)/50))
	n_inf = 1 / (1 + Exp(-(v+38)/15))


}


FUNCTION alpham(x) {
	if (fabs((x+amB)/amC) < 1e-6) {
		alpham = amA*amC
	}else{
		alpham = (amA*(x+amB)) / (1 - Exp(-(x+amB)/amC))
	}
}



FUNCTION betam(x) {
	if (fabs((x+bmB)/bmC) < 1e-6) {
		betam = -bmA*bmC
	}else{
		betam = (bmA*(-(x+bmB))) / (1 - Exp((x+bmB)/bmC))
	}
}


FUNCTION vtrap1(x) {
	if (fabs((x+ampB)/ampC) < 1e-6) {
		vtrap1 = ampA*ampC
	}else{
		vtrap1 = (ampA*(x+ampB)) / (1 - Exp(-(x+ampB)/ampC))
	}
}

FUNCTION vtrap2(x) {
	if (fabs((x+bmpB)/bmpC) < 1e-6) {
		vtrap2 = -bmpA*bmpC
	}else{
		vtrap2 = (bmpA*(-(x+bmpB))) / (1 - Exp((x+bmpB)/bmpC))
	}
}

FUNCTION Exp(x) {
	if (x < -100) {
		Exp = 0
	}else{
		Exp = exp(x)
	}
}

UNITSON
