NEURON {
	SUFFIX cac1
	RANGE gmax, Ev
	NONSPECIFIC_CURRENT i}

UNITS{
	(pA) = (picoamp)
	(molar) = (1/liter)
	(uM) = (micromolar)
	(mV) = (millivolt)
	(pS) = (picosiemens)

  FARADAY = (faraday)  (kilocoulombs)
	R = (k-mole) (joule/degC)
}

PARAMETER {

	a0 = 0.34 (/s)
	b0 = 0.22 (/s)
	l0 = 0.13 (/s)
	u0 = 0.6 (/s)
  celsius = 37	(degC)

  az = -0.5
	bz = 0.2
	lz = -0.3
	uz = 0.2

	gmax = 0.1 (mho/cm2)	: conductivity
	Ev = -60 (mV)
}

ASSIGNED {
	v (mV)	: voltage
	i (mA/cm2)	: current
	g  (mho/cm2)	: conductance
  a1 (/s)
  b1 (/s)
  l1 (/s)
  u1 (/s)
}

STATE {
	Cs
	C1
	C2
	O1
	O2
  C0
}

INITIAL {
	Cs=1
}

BREAKPOINT {
	SOLVE kstates METHOD sparse
	g = gmax*(O1+O2)
	i = g * (v - Ev)
}

KINETIC kstates{

  a1 = update_state(v, a0, az)
	b1 = update_state(v, b0, bz)
  l1 = update_state(v, l0, lz)
  u1 = update_state(v, u0, uz)

	~ Cs <-> O1 (2*a1, b1)
	~ O1 <-> O2 (a1, 2*b1)
	~ C0 <-> C1 (2*a1, b1)
	~ C1 <-> C2 (a1, 2*b1)
	~ O2 <-> C2 (u1, l1)
	~ O1 <-> C1 (u1, l1)
  ~ Cs <-> C0 (u1, l1)


	CONSERVE 	Cs+C0+C1+C2+O1+O2=1
}

FUNCTION update_state(v(mV), state0, z0){
	update_state = state0*exp(z0*FARADAY*v/R/(273.15 + celsius))
}
