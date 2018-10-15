TITLE KA
: K-A current for motoneuron from Safronov and Vogel (1995).
: M.Migliore Dec. 2001

NEURON {
	SUFFIX kamot
	USEION k READ ek WRITE ik
	RANGE  gbar
	GLOBAL minf, mtau, hinf, htau
}

PARAMETER {
	gbar = 0.02   	(mho/cm2)	
								
	celsius
	ek		(mV)            : must be explicitly def. in hoc
	v 		(mV)
	a0m=0.085
	vhalfm=-75
	zetam=2
	gmm=0.7
	mmin=0.1

	a0h=0.008
	vhalfh=-85.8
	zetah=3
	gmh=0.75
	hmin=0.1
	
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
	hinf 		htau (ms)	 	
}
 

STATE { m h}

BREAKPOINT {
        SOLVE states METHOD cnexp
	ik = gbar*m*h* (v - ek)
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
        minf = 1/(1 + exp(-(v+36.2)/9.1))
	mtau = betm(v)/(qt*a0m*(1+alpm(v)))
	if (mtau<mmin/qt) {mtau=mmin/qt}

        hinf = 1/(1 + exp((v+85.8)/13.2))
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
