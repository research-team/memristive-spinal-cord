TITLE Intracellular calcium dynamics

NEURON {
	SUFFIX Ca_conc
	USEION ca READ ica WRITE cai
	RANGE cai, cca
}

UNITS	{
	(mV) 		= (millivolt)
	(mA) 		= (milliamp)
	FARADAY 	= (faraday) (coulombs)
	(molar) 	= (1/liter)
	(mM) 		= (millimolar)
}

PARAMETER {
	f = 0.004
	kCa = 8			(/ms)
	alpha = 1    	(mol/C/cm2)
}

ASSIGNED {
	cai			(mM)
	ica			(mA/cm2)
}

STATE {
	cca		(mM)
}

BREAKPOINT {
	SOLVE state METHOD cnexp
}

INITIAL {
	cca = 0.0001
}

DERIVATIVE state {
	cca' = f * (- alpha * ica - kCa * cca)
	cai = cca
}
