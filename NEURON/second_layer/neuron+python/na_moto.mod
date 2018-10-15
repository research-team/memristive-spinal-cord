TITLE Na
: Na current for motoneuron from Safronov and Vogel (1995).
: M.Migliore Dec. 2001

NEURON {
	SUFFIX namot
	USEION na READ ena WRITE ina
	RANGE  gbar
	GLOBAL minf, mtau, hinf, htau
}

PARAMETER {
	gbar = 0.02   	(mho/cm2)	
								
	celsius
	ena		(mV)            : must be explicitly def. in hoc
	v 		(mV)
	a0m=0.7
	vhalfm=-38.9
	zetam=1.5
	gmm=0.2
	mmin=0.02

	a0h=0.06
	vhalfh=-81.6
	zetah=3
	gmh=0.6
	hmin=0.8
	
	q10=2.3
}


UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	(pS) = (picosiemens)
	(um) = (micron)
} 

ASSIGNED {
	ina 		(mA/cm2)
	minf 		mtau (ms)	 	
	hinf 		htau (ms)	 	
}
 

STATE { m h}

BREAKPOINT {
        SOLVE states METHOD cnexp
	ina = gbar*m*h* (v - ena)
} 

INITIAL {
	trates(v)
	m=minf  
	h=hinf  
}

DERIVATIVE states {   
        trates(v)      
        m' = (minf-m)/mtau
        h' = (hinf-h)/htau
}

PROCEDURE trates(v) {  
	LOCAL qt
        qt=q10^((celsius-22)/10)
        minf = 1/(1 + exp(-(v+38.9)/6.1))
	mtau = betm(v)/(qt*a0m*(1+alpm(v)))
	if (mtau<mmin/qt) {mtau=mmin/qt}

        hinf = 1/(1 + exp((v+81.6)/10.2))
	htau = beth(v)/(qt*a0h*(1+alph(v)))
	if (htau<hmin/qt) {htau=hmin/qt}
}

FUNCTION alpm(v(mV)) {
  alpm = exp(1.e-3*zetam*(v-vhalfm)*9.648e4/(8.315*(273.16+celsius))) 
}

FUNCTION betm(v(mV)) {
  betm = exp(1.e-3*zetam*gmm*(v-vhalfm)*9.648e4/(8.315*(273.16+celsius))) 
}

FUNCTION alph(v(mV)) {
  alph = exp(1.e-3*zetah*(v-vhalfh)*9.648e4/(8.315*(273.16+celsius))) 
}

FUNCTION beth(v(mV)) {
  beth = exp(1.e-3*zetah*gmh*(v-vhalfh)*9.648e4/(8.315*(273.16+celsius))) 
}
