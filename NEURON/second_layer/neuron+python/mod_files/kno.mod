TITLE K_No.mod   nodal slow potassium channel

COMMENT
This is the original Hodgkin-Huxley treatment for the set of potassium channel found
in the squid giant axon membrane.
Some parameters have been changed to correspond to McIntyre and Grill (2002) "Extracellular
stimulation of central neurons"
Author: Balbi
ENDCOMMENT

UNITS {
        (mA) = (milliamp)
        (mV) = (millivolt)
		(S) = (siemens)
}

NEURON {
        SUFFIX K_No
        USEION k READ ek WRITE ik
        RANGE gkmax, gk
        RANGE sinf, stau
}

PARAMETER {
        gkmax = 0.08 (S/cm2)   <0,1e9>
}

STATE {
        s
}

ASSIGNED {
        v (mV)
        ek (mV)

		gk (S/cm2)
        ik (mA/cm2)
        sinf
		stau (ms)
}

BREAKPOINT {
        SOLVE states METHOD cnexp
        gk = gkmax*s
		ik = gk*(v - ek)
}

INITIAL {
	rates(v)
	s = sinf
}

DERIVATIVE states {
        rates(v)
        s' = (sinf-s)/stau
}

PROCEDURE rates(v(mV)) {
		:Call once from HOC to initialize inf at resting v.
        LOCAL  alpha, beta, sum

UNITSOFF
                :"s" potassium activation system
        alpha = 0.3/(1+exp((v+43)/-5))
        beta =  0.03/(1+exp((v+80)/-1))
        sum = alpha + beta
		stau = 1/sum
        sinf = alpha/sum
}

UNITSON
