TITLE Modified Hill-Mashima muscle model

UNITS {}

NEURON {
    SUFFIX fHill
    RANGE a0, b0, c0, d0
	RANGE p0, g1, g2, g3
	RANGE Kse, A, Fc, F
	RANGE xm_init, xm, xce_init, xce
	USEION mg READ mgi VALENCE 2
	USEION cl READ cli VALENCE -1
}

PARAMETER {
	a0 = 2.35		:[N]
	b0 = 24.35		:[mm*s-1]
	c0 = -7.4		:[N]
	d0 = 30.3		:[mm*s-1]
	p0 = 23			:[N]
	g1 = -8			:[mm]
	g2 = 21.4		:[mm]
	xm_init = -8	:[mm]
	xce_init = -8	:[mm]
	Kse = 0.4		:[mm-1]
}

STATE {
	A
	xce
	xm
}

ASSIGNED {
	F
	Fc
	mgi
	cli
}

BREAKPOINT { LOCAL d_xm, d_xce, d_se
	A = mgi
	xm = cli

	SOLVE state_hill METHOD cnexp

	F = p0*Kse*xse(xm, xce)
}

DERIVATIVE state_hill {
	Fc = p0*g(xm)*A
	xce' = dxdt (F, Fc)
}

FUNCTION xse (x, y) { LOCAL d_xm, d_xce, d_se
	d_xm = xm - xm_init
	d_xce = xce - xce_init
	d_se = d_xm - d_xce

	if (d_se <= 0) {xse = 0}
	else {xse = d_se}
}

FUNCTION g (x) {
	g = exp(-((x-g1)/g2)^2)
}

FUNCTION dxdt (x, xc) {LOCAL gain_length
	if (x <= xc) {
		dxdt = (10^-3)*(-b0*(xc-x))/(x+a0*xc/p0)
	} else {
		gain_length = (-d0*(xc-x))/(2*xc-x+c0*xc/p0)
		if (gain_length <= 0) {dxdt = (10^-3)*1e5}
		else {dxdt = (10^-3)*gain_length}
	}
}

INITIAL {
	A = 0
	xm = xm_init
	xce = xce_init
	F=1e-5
}
