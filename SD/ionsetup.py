import math


def InitIons():
    xF=96480.0e0 #Faraday constant (C/mol)
    KA=KA0
	KN=KN0
	KJ=KI0
	NaA=NaA0
	NaN=NaN0
	NaI=NaI0
	CaA=CaA0
	CaN=CaN0
	CaI=CaI0
	ClA=ClA0
	ClN=ClN0
	ClI=ClI0
	VmA=VmA0
	VmN=VmN0
	GlutA=glut0
	GlutB=glut0
	NMDAy=0.0e0



	SetLeaks(NaI0,NaA0,NaN0, KI0,KN0,KA0, ClI0,ClA0,ClN0, CaA0,CaI0,CaN0, VmA0,VmN0)
'''-------
 Set up NMDA channel with appropriate conversion factors
 ------
 The general calculation comes out in mmol/litre
 First convert to C/litre
 '''

	NMDAKfac=xF*0.001e0	# include conversion mmol to mol
	NMDANafac=xF*0.001e0
	NMDACafac=4e0*xF*0.001e0	# include double charge on Ca2+

# Now multiply by some nominal permeabilities...

	PK=0.001e0	# cm/s plus conversion from litre to cm3
	PNa=0.001e0
	PCa=0.006e0	# larger for Ca2+ than for K+ and Na+
# ... to get A/cm2

	NMDAKfac=NMDAKfac*PK
	NMDANafac=NMDANafac*PNa
	NMDACafac=NMDACafac*PCa

# Convert to pA/um2

	NMDAKfac=NMDAKfac*1e4
	NMDANafac=NMDANafac*1e4
	NMDACafac=NMDACafac*1e4

# And include dimensionless conductivity

	NMDAKfac=NMDAKfac*gKNaCaN
	NMDANafac=NMDANafac*gKNaCaN
	NMDACafac=NMDACafac*gKNaCaN


def Setleaks(NaIX,NaAX,NaNX, KIX,KNX,KAX, ClIX,ClAX,ClNX, CaAX,CaIX,CaNX, VmAZ,VmNX):
	# Nernst potentials
	EKA=RTF*log(KIX/KAX)
	ENaA=RTF*log(NaIX/NaAX)
	ECaA=0.5e0*RTF*log(CaIX/CaAX)
	EClA=-RTF*log(ClIX/ClAX)
	EKN=RTF*log(KIX/KNX)
	ENaN=RTF*log(NaIX/NaNX)
	ECaN=0.5e0*RTF*log(CaIX/CaNX)
	EClN=-RTF*log(ClIX/ClNX)
	
	# K delayed rectifier

	dV=VmAZ-20e0
	alpha=0.0047e0*(dV+12e0)/(1-math.exp(-(dV+12e0)/12e0))
	beta=math.exp(-(dV+147e0)/30e0)
	hinf=1e0/(1e0+math.exp(VmAZ+25e0)/4e0)
	minf=alpha/(alpha+beta)
	IKDRA=gKDRA*minf*minf*hinf*(VmAZ-EKA)

	dV=VmNX-20e0
	alpha=0.0047e0*(dV+12e0)/(1e0-math.exp(-(dV+12e0)/12e0))
	beta=math.exp(-(dV+147e0)/30e0)
	hinf=1e0/(1e0+math.exp(VmNX+25e0)/4e0)
	minf=alpha/(alpha+beta)
	IKDRN=gKDRN*minf*minf*hinf*(VmNX-EKN)

	# K BK channel (Ca dependent)

	minf=250e0*CaNX*math.exp(VmNX/24)
	minf=minf/(minf+0.1e0*math.exp(-VmNX/24e0))
	IBKN=gBKN*minf*(VmNX-EKN)

	minf=250e0*CaAX*math.exp(VmAZ/24)
	minf=minf/(minf+0.1e0*math.exp(-VmAZ/24e0))
	IBKA=gBKA*minf*(VmAZ-EKA)

	# K inward rectifier (astrocyte only)

	#minf=1D0/(2D0+exp((1.62D0/RTF)*(VmAZ-EKA)))
	#IKirA=gKirA*minf*(KIX/(KIX+13D0))*(VmAZ-EKA)
	IKirA=0e0

	# K A-channel (transient outward)

	#minf=1D0/(1D0+exp(-(VmNX+42D0)/13D0))
	#hinf=1D0/(1D0+exp((VmNX+110D0)/18D0))
	#IKAN=gKAN*minf*hinf*(VmNX-EKN)
	IKAN=0e0

	#minf=1D0/(1D0+exp(-(VmAZ+42D0)/13D0))
	#hinf=1D0/(1D0+exp((VmAZ+110D0)/18D0))
	#IKAA=gKAA*minf*hinf*(VmAZ-EKA)
	IKAA=0e0

	# K M-channel (non-inactivating muscarinic)

	minf=1e0/(1e0+math.exp(-(VmNX+35e0)/10e0))
	IKMN=gKMN*minf*(VmNX-EKN)

	minf=1e0/(1e0+math.exp(-(VmAZ+35e0)/10e0))
	IKMA=gKMA*minf*(VmAZ-EKA)

	# K SK-channel (voltage-independent Ca-activated)

	alpha=1.25e8*CaNX**2
	beta=2.5e0
	minf=alpha/(alpha+beta)
	ISKN=gSKN*minf*minf*(VmNX-EKN)

	alpha=1.25e8*CaAX**2
	beta=2.5e0
	minf=alpha/(alpha+beta)
	ISKA=gSKA*minf*minf*(VmAZ-EKA)

	# K IK-channel (K2 Ca-activated)

	alpha=25e0
	beta=0.075e0*math.exp(-(VmNX+5e0)/10e0)
	minf=alpha/(alpha+beta)
	IIKN=gIKN*minf*(VmNX-EKN)/(1e0+0.0002e0/CaNX)**2

	alpha=25e0
	beta=0.075e0*math.exp(-(VmAZ+5e0)/10e0)
	minf=alpha/(alpha+beta)
	IIKA=gIKA*minf*(VmAZ-EKA)/(1e0+0.0002e0/CaAX)**2

	# Na transient (fast)

	alpha=35e0*math.exp((VmNX+5e0)/10e0)
	beta=7e0*math.exp(-(VmNX+65e0)/20e0)
	minf=alpha/(alpha+beta)
	alpha=0.225e0/(1e0+math.exp((VmNX+80e0)/10e0))
	beta=7.5e0*math.exp((VmNX-3e0)/18e0)
	hinf=alpha/(alpha+beta)
	INaFN=gNaFN*minf**3*hinf*(VmNX-ENaN)

	alpha=35e0*math.exp((VmAZ+5e0)/10e0)
	beta=7e0*math.exp(-(VmAZ+65e0)/20e0)
	minf=alpha/(alpha+beta)
	alpha=0.225e0/(1+math.exp((VmAZ+80e0)/10e0))
	beta=7.5e0*math.exp((VmAZ-3e0)/18e0)
	hinf=alpha/(alpha+beta)
	INaFA=gNaFA*minf**3*hinf*(VmAZ-ENaA)

	# Na persistent current

	alpha=200e0/(1e0+math.exp(-(VmAZ-18e0)/16e0))
	beta=25e0/(1e0+math.exp((VmAZ+58e0)/8e0))
	minf=alpha/(alpha+beta)
	INaPA=gNaPA*minf**3*(VmAZ-ENaA)

	alpha=200e0/(1+math.exp(-(VmNX-18e0)/16e0))
	beta=25e0/(1+math.exp((VmNX+58e0)/8e0))
	minf=alpha/(alpha+beta)
	INaPN=gNaPN*minf**3*(VmNX-ENaN)

	# Ca high voltage activated (P-type)

	alpha=8.5e0/(1e0+math.exp(-(VmAZ-8e0)/12.5e0))
	beta=35e0/(1e0+math.exp((VmAZ+74e0)/14.5e0))
	minf=alpha/(alpha+beta)
	alpha=0.0015e0/(1e0+math.exp((VmAZ+29e0)/8e0))
	beta=0.0055e0/(1e0+math.exp(-(VmAZ+23e0)/8e0))
	hinf=alpha/(alpha+beta)
	ICaHVAA=gCaHVAA*minf*hinf*(VmAZ-ECaA)

	alpha=8.5e0/(1+math.exp(-(VmNX-8e0)/12.5e0))
	beta=35e0/(1+math.exp((VmNX+74e0)/14.5e0))
	minf=alpha/(alpha+beta)
	alpha=0.0015e0/(1e0+math.exp((VmNX+29e0)/8e0))
	beta=0.0055e0/(1e0+math.exp(-(VmNX+23e0)/8e0))
	hinf=alpha/(alpha+beta)
	ICaHVAN=gCaHVAN*minf*hinf*(VmNX-ECaN)

	# Ca low voltage activated (T-type)

	#alpha=2.6D0/(1D0+exp(-(VmNX+21D0)/8D0))
	#beta=0.018D0/(1D0+exp((VmNX+40D0)/4D0))
	#minf=alpha/(alpha+beta)
	#alpha=0.0025D0/(1D0+exp((VmNX+40D0)/8D0))
	#beta=0.19D0/(1D0+exp(-(VmNX+50D0)/10D0))
	#hinf=alpha/(alpha+beta)
	#ICaLVAN=gCaLVAN*minf*hinf*(VmNX-ECaN)
	ICaLVAN=0

	#alpha=2.6D0/(1D0+exp(-(VmAZ+21D0)/8D0))
	#beta=0.018D0/(1D0+exp((VmAZ+40D0)/4D0))
	#minf=alpha/(alpha+beta)
	#alpha=0.0025D0/(1D0+exp((VmAZ+40D0)/8D0))
	#beta=0.19D0/(1D0+exp(-(VmAZt+50D0)/10D0))
	#hinf=alpha/(alpha+beta)
	#ICaLVAA=gCaLVAA*minf*hinf*(VmAZ-ECaA)
	ICaLVAA=0

	# NMDA receptor is closed in initial state

	# Cl pump

	IClpumpA=-rClA*ClAX/(ClAX+25e0)
	IClpumpN=-rClN*ClNX/(ClNX+25e0)

	# Ca pump

	ICapumpA=rCaA*CaAX/(CaAX+0.0002e0)
	ICapumpN=rCaN*CaNX/(CaNX+0.0002e0)

	# K/Na exchange

	phi=(VmAZ+176.5e0)/RTF
	phi=0.052e0*sinh(phi)/(0.026e0*math.exp(phi)+22.5e0*math.exp(-phi))
	IKpumpA=-rNaKA*(KIX/(KIX+3.7e0))**2*(NaAX/(NaAX+0.6e0))**3*phi
	INapumpA=-1.5e0*IKpumpA

	phi=(VmNX+176.5e0)/RTF
	phi=0.052e0*sinh(phi)/(0.026e0*math.exp(phi)+22.5e0*math.exp(-phi))
	IKpumpN=-rNaKN*(KIX/(KIX+3.7e0))**2*(NaNX/(NaNX+0.6e0))**3*phi
	INapumpN=-1.5e0*IKpumpN

	# Na/Ca exchanger

	phi=VmNX/RTF
	numerator=(NaNX**3*CaIX*math.exp(0.35e0*phi)-NaIX**3*CaNX*2.5e0*math.exp(-0.65e0*phi))
	denominator=(87.5e0**3+NaIX**3)*(1.38e0+CaIX)*(1e0+0.1e0*math.exp(-0.65e0*phi))
	ICaantN=-rNaCaN*numerator/denominator
	INaantN=-1.5e0*ICaantN

	phi=VmAZ/RTF
	numerator=(NaAX**3*CaIX*math.exp(0.35e0*phi)-NaIX**3*CaAX*2.5e0*math.exp(-0.65e0*phi))
	denominator=(87.5e0**3+NaIX**3)*(1.38e0+CaIX)*(1e0+0.1e0*math.exp(-0.65e0*phi))
	ICaantA=-rNaCaA*numerator/denominator
	INaantA=-1.5e0*ICaantA

	# K/Cl transporter

	IKexchN=gKClN*RTF*log((KNX/KIX)*(ClNX/ClIX))
	IClexchN=-IKexchN

	IKexchA=gKClA*RTF*log((KAX/KIX)*(ClAX/ClIX))
	IClexchA=-IKexchA

	# Na/K/Cl transporter

	IKastroA=-gNaKClA*RTF*log((NaIX/NaAX)*(KIX/KAX)*(ClIX/ClAX)**2)
	INaastroA=IKastroA
	IClastroA=-2e0*IKastroA

	# Total currents excluding leaks - these to be in pA/um2

	IKTotalN=IKDRN+IBKN+IKAN+IKMN+ISKN+IIKN+IKpumpN+IKexchN
	INaTotalN=INaFN+INaPN+INapumpN+INaantN
	ICaTotalN=ICaHVAN+ICaLVAN+ICapumpN+ICaantN
	IClTotalN=IClpumpN+IClexchN
	IKTotalA=IKDRA+IBKA+IKirA+IKAA+IKMA+ISKA+IIKA+IKpumpA+IKexchA+IKastroA
	INaTotalA=INaFA+INaPA+INapumpA+INaantA+INaastroA
	ICaTotalA=ICaHVAA+ICaLVAA+ICapumpA+ICaantA
	IClTotalA=IClpumpA+IClexchA+IClastroA

	# ==== Set leaks to create initial balance ====

	gKLeakN=-IKTotalN/(VmNX-EKN)
	gNaLeakN=-INaTotalN/(VmNX-ENaN)
	gCaLeakN=-ICaTotalN/(VmNX-ECaN)
	gClLeakN=-IClTotalN/(VmNX-EClN)

	gKLeakA=-IKTotalA/(VmAZ-EKA)
	gNaLeakA=-INaTotalA/(VmAZ-ENaA)
	gCaLeakA=-ICaTotalA/(VmAZ-ECaA)
	gClLeakA=-IClTotalA/(VmAZ-EClA)

	'''
	if (gKLeakN<0) write(*,*)'Bad N-K leak'
	if (gNaLeakN<0) write(*,*)'Bad N-Na leak'
	if (gCaLeakN<0) write(*,*)'Bad N-Ca leak'
	if (gClLeakN<0) write(*,*)'Bad N-Cl leak'
	if (gKLeakA<0) write(*,*)'Bad A-K leak'
	if (gNaLeakA<0) write(*,*)'Bad A-Na leak'
	if (gCaLeakA<0) write(*,*)'Bad A-Ca leak'
	if (gClLeakA<0) write(*,*)'Bad A-Cl leak'

	if ((gKLeakN<0).OR.(gNaLeakN<0).OR.(gCaLeakN<0).OR.(gClLeakN<0)) STOP
	if ((gKLeakA<0).OR.(gNaLeakA<0).OR.(gCaLeakA<0).OR.(gClLeakA<0)) STOP
	'''
