TITLE slow subs diffusion
NEURON{
	POINT_PROCESS diff_slow
	RANGE subs,h,c0cleft
	RANGE tx1}

UNITS{
		(molar)=(1/liter)
		(uM)=(micromolar)
		(um)=(micron)
		(nA)=(nanoamp)
}
CONSTANT {
	PI=3.1415927
}
PARAMETER { 
	c0cleft = 1 (uM):initial quantity subs
	h(um)
	tx1(ms)

 }
ASSIGNED{
   subs (uM)
}
INITIAL {
	:tx1=10
	subs=0
}
BREAKPOINT
{
	at_time(tx1)
	if (t<=tx1){
		subs=0
}
if(t>tx1) {
UNITSOFF
	subs = (t-tx1)*0.00025
    if(subs>c0cleft){subs=c0cleft}
}
}
NET_RECEIVE (weight)
{
tx1=t 
}




