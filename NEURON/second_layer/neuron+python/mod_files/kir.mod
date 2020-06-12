TITLE inward rectifier potassium (Kir) channel

COMMENT

Mod File by A. Hanuschkin <AH, 2011> for:
Yim MY, Hanuschkin A, Wolfart J (2015) Hippocampus 25:297-308.
http://onlinelibrary.wiley.com/doi/10.1002/hipo.22373/abstract

Channel description and parameters from:
Stegen M, Kirchheim F, Hanuschkin A, Staszewski O, Veh R, and Wolfart J. Cerebral Cortex, 22:9, 2087-2101, 2012.

Mod File history:
- tau(V), linf(V) fitted to experimental values of human dentate gyrus granual cells
- ModelDB file adapted from
  Wolf JA, Moyer JT, Lazarewicz MT, Contreras D, Benoit-Marand M, O'Donnell P, Finkel LH (2005) J Neurosci 25:9080-95
  https://senselab.med.yale.edu/ModelDB/ShowModel.cshtml?model=112834&file=/nacb_msp/kir.mod
- file modified to uses nomoclature of
  Li X, Ascoli GA (2006) J of Comput Neurosci 21(2):191-209
  Li X, Ascoli GA (2008) Neural Comput 20:1717-31

A. Hanuschkin(c) 2011,2012

ENDCOMMENT


UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
        (S)  = (siemens)
}

PARAMETER {
	v 		(mV)
	gkbar  = 1.44e-05	(S/cm2) 	: to be fitted

	: Boltzman steady state curve
        vhalfl = -98.92  (mV)    		: fitted to patch data, Stegen et al. 2012
        kl = 10.89       (mV)    		: Stegen et al. 2012

	: tau_infty
        vhalft=67.0828	 (mV)    		: fitted #100 \muM sens curr 350a,  Stegen et al. 2012
        at=0.00610779	 (/ms)   		: Stegen et al. 2012
	bt=0.0817741	 (/ms)	 		: Note: typo in Stegen et al. 2012

	: Temperature dependence
        celsius         (degC)  		: unused if q10 == 1.
        q10 = 1.                              	: temperature scaling
}



NEURON {
	SUFFIX kir
	USEION k READ ek WRITE ik
        RANGE  ik, gkbar, vhalfl, kl, vhalft, at, bt, q10
        GLOBAL linf,taul
}


STATE {
        l
}

ASSIGNED {
        ik                              (mA/cm2)
        gk                              (S/cm2)
        ek                              (mV)
        linf
        taul
}


INITIAL {
	rate(v)
	l=linf
}


BREAKPOINT {
	SOLVE states METHOD cnexp	: solve differential equations in states with method 'cnexp'
	gk = gkbar*l			: use state l to calulate gk
        ik = gk * ( v - ek )		: calculate ik
}



DERIVATIVE states {
        rate(v)
        l' =  (linf - l)/taul		: differential equation
}

PROCEDURE rate(v (mV)) { :callable from hoc
        LOCAL qt
	qt=q10^((celsius-33)/10)
        linf = 1/(1 + exp((v-vhalfl)/kl))			: l_steadystate
 	taul = 1/(qt *(at*exp(-v/vhalft) + bt*exp(v/vhalft) ))
}
