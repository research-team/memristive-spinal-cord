COMMENT
//****************************//
// Created by Alon Polsky 	//
//    apmega@yahoo.com		//
//		2010			//
//****************************//
based on Sun et al 2006

Modified by Robert Egger 2013 to include
normalization of synaptic conductances
ENDCOMMENT
TITLE GABAA synapse activated by the network
NEURON {
	POINT_PROCESS GABAA_S
	NONSPECIFIC_CURRENT i
    RANGE xloc,yloc,tag1,tag2
	RANGE gmax,local_v,i
	RANGE taudgaba,dgaba,ggaba
	RANGE R,D
	RANGE risetime,decaytime ,e ,decaygaba
}
PARAMETER {
	gmax=.5	(nS)
	e= -60.0	(mV)
	risetime=1	(ms)	:2
	decaytime=20(ms)	:40

	v		(mV)
	taudgaba=200	(ms)
	decaygaba=0.8
	xloc=0
	yloc=0
	tag1=0
	tag2=0
}
ASSIGNED {
	i		(nA)  
 	local_v	(mV):local voltage
	ggaba
    factor     : conductance normalization factor
}

STATE {
	dgaba
	R
	D
}

INITIAL {
	LOCAL tp
      dgaba=1 
	R=0
	D=0
	ggaba=0
	
    tp = (risetime*decaytime)/(decaytime - risetime) * log(decaytime/risetime)
    factor = -exp(-tp/risetime) + exp(-tp/decaytime)
    factor = 1/factor
}
BREAKPOINT {
	SOLVE state METHOD cnexp
	ggaba=D-R
	i=(1e-3)*ggaba*(v-e)
}
NET_RECEIVE(weight) {
    state_discontinuity( R, R+ factor*weight*(dgaba))
    state_discontinuity( D, D+ factor*weight*(dgaba))
    state_discontinuity( dgaba, dgaba* decaygaba)
}
DERIVATIVE state {
	R'=-R/risetime
	D'=-D/decaytime
	dgaba'=(1-dgaba)/taudgaba

}