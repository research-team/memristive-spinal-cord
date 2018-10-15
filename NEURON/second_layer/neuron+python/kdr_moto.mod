TITLE KDR
: K-DR current for motoneuron from Safronov and Vogel (1995).
: M.Migliore Dec. 2001

NEURON {
	SUFFIX kdrmot
	USEION k READ ek WRITE ik
	RANGE  gbar
	GLOBAL minf, mtau
}

PARAMETER {
	gbar = 0.02   	(mho/cm2)	
								
	celsius
	ek		(mV)            : must be explicitly def. in hoc
	v 		(mV)
	a0m=0.008
	vhalfm=-48
	zetam=3
	gmm=0.45
	mmin=0.1

	q10=3
}


UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	(pS) = (picosiemens)
	(um) = (micron)
} 

ASSIGNED {
	ik 		(mA/cm2)
	minf 		mtau (ms)	 	
}
 

STATE { m }

BREAKPOINT {
        SOLVE states METHOD cnexp
	ik = gbar*m*(v - ek)
} 

INITIAL {
	trates(v)
	m=minf  
}

DERIVATIVE states {   
        trates(v)      
        m' = (minf-m)/mtau
}

PROCEDURE trates(v) {  
	LOCAL qt
        qt=q10^((celsius-22)/10)
        minf = 1/(1 + exp(-(v+43.8)/8.5))
	mtau = betm(v)/(qt*a0m*(1+alpm(v)))
	if (mtau<mmin/qt) {mtau=mmin/qt}
}

FUNCTION alpm(v(mV)) {
  alpm = exp(1.e-3*zetam*(v-vhalfm)*9.648e4/(8.315*(273.16+celsius))) 
}

FUNCTION betm(v(mV)) {
  betm = exp(1.e-3*zetam*gmm*(v-vhalfm)*9.648e4/(8.315*(273.16+celsius))) 
}