subroutine InitIons
use global
use ionglob
implicit none
double precision PK,PNa,PCa
double precision xF
xF=96480.0D0	! Faraday constant (C/mol)
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
NMDAy=0.0D0
if (startup.eq.'VMEM') then
	call InitVmem
endif
call SetLeaks(NaI0,NaA0,NaN0, KI0,KN0,KA0, ClI0,ClA0,ClN0, CaA0,CaI0,CaN0,	&
	VmA0,VmN0)
!-------
! Set up NMDA channel with appropriate conversion factors
!------
! The general calculation comes out in mmol/litre
! First convert to C/litre

NMDAKfac=xF*0.001D0	! include conversion mmol to mol
NMDANafac=xF*0.001D0
NMDACafac=4D0*xF*0.001D0	! include double charge on Ca2+

! Now multiply by some nominal permeabilities...

PK=0.001D0	! cm/s plus conversion from litre to cm3
PNa=0.001D0
PCa=0.006D0	! larger for Ca2+ than for K+ and Na+

! ... to get A/cm2

NMDAKfac=NMDAKfac*PK
NMDANafac=NMDANafac*PNa
NMDACafac=NMDACafac*PCa

! Convert to pA/um2

NMDAKfac=NMDAKfac*1d4
NMDANafac=NMDANafac*1d4
NMDACafac=NMDACafac*1d4

! And include dimensionless conductivity

NMDAKfac=NMDAKfac*gKNaCaN
NMDANafac=NMDANafac*gKNaCaN
NMDACafac=NMDACafac*gKNaCaN

end subroutine InitIons

subroutine InitVmem
use ionglob
use global
implicit none
VmN(startcellX)=startVmem
end subroutine InitVmem

subroutine SetLeaks(NaIX,NaAX,NaNX, KIX,KNX,KAX, ClIX,ClAX,ClNX, CaAX,CaIX,CaNX,	&
	VmAZ,VmNX)
use ionglob
implicit none

! inputs

double precision :: NaIX,NaAX,NaNX, KIX,KNX,KAX, ClIX,ClAX,ClNX, CaAX,CaIX,CaNX
double precision :: VmAZ,VmNX

! local

double precision :: phi,dV,hinf,minf,alpha,beta
double precision :: numerator,denominator
double precision :: EKA,ENaA,ECaA,EClA, EKN,ENaN,ECaN,EClN

double precision :: IKDRA,IKDRN, IBKA,IBKN, IKirA, IKAN,IKAA, IKMN,IKMA
double precision :: ISKA, ISKN, IIKA,IIKN
double precision :: INaFA,INaFN, INaPA,INaPN
double precision :: ICaLVAA,ICaLVAN,ICaHVAA,ICaHVAN
double precision :: IClpumpA,iClpumpN,ICapumpA,ICapumpN
double precision :: IKpumpA,INapumpA,IKpumpN,INapumpN
double precision :: ICaantA,INaantA,ICaantN,INaantN
double precision :: IKexchN,IClexchN,IKexchA,IClexchA
double precision :: IKastroA,INaastroA,IClastroA
double precision :: INaTotalA,IKTotalA,ICaTotalA,IClTotalA
double precision :: INaTotalN,IKTotalN,ICaTotalN,IClTotalN


! Nernst potentials

EKA=RTF*log(KIX/KAX)
ENaA=RTF*log(NaIX/NaAX)
ECaA=0.5D0*RTF*log(CaIX/CaAX);
EClA=-RTF*log(ClIX/ClAX);
EKN=RTF*log(KIX/KNX)
ENaN=RTF*log(NaIX/NaNX)
ECaN=0.5D0*RTF*log(CaIX/CaNX);
EClN=-RTF*log(ClIX/ClNX);

! K delayed rectifier

dV=VmAZ-20D0
alpha=0.0047D0*(dV+12D0)/(1D0-exp(-(dV+12D0)/12D0))
beta=exp(-(dV+147D0)/30D0)
hinf=1D0/(1D0+exp(VmAZ+25D0)/4D0)
minf=alpha/(alpha+beta)
IKDRA=gKDRA*minf*minf*hinf*(VmAZ-EKA)

dV=VmNX-20D0
alpha=0.0047D0*(dV+12D0)/(1D0-exp(-(dV+12D0)/12D0))
beta=exp(-(dV+147D0)/30D0)
hinf=1D0/(1D0+exp(VmNX+25D0)/4D0)
minf=alpha/(alpha+beta)
IKDRN=gKDRN*minf*minf*hinf*(VmNX-EKN)

! K BK channel (Ca dependent)

minf=250D0*CaNX*exp(VmNX/24)
minf=minf/(minf+0.1D0*exp(-VmNX/24D0))
IBKN=gBKN*minf*(VmNX-EKN)

minf=250D0*CaAX*exp(VmAZ/24)
minf=minf/(minf+0.1D0*exp(-VmAZ/24D0))
IBKA=gBKA*minf*(VmAZ-EKA)

! K inward rectifier (astrocyte only)

!minf=1D0/(2D0+exp((1.62D0/RTF)*(VmAZ-EKA)))
!IKirA=gKirA*minf*(KIX/(KIX+13D0))*(VmAZ-EKA)
IKirA=0D0

! K A-channel (transient outward)

!minf=1D0/(1D0+exp(-(VmNX+42D0)/13D0))
!hinf=1D0/(1D0+exp((VmNX+110D0)/18D0))
!IKAN=gKAN*minf*hinf*(VmNX-EKN)
IKAN=0D0

!minf=1D0/(1D0+exp(-(VmAZ+42D0)/13D0))
!hinf=1D0/(1D0+exp((VmAZ+110D0)/18D0))
!IKAA=gKAA*minf*hinf*(VmAZ-EKA)
IKAA=0D0

! K M-channel (non-inactivating muscarinic)

minf=1D0/(1D0+exp(-(VmNX+35D0)/10D0))
IKMN=gKMN*minf*(VmNX-EKN)

minf=1D0/(1D0+exp(-(VmAZ+35D0)/10D0))
IKMA=gKMA*minf*(VmAZ-EKA)

! K SK-channel (voltage-independent Ca-activated)

alpha=1.25D8*CaNX**2
beta=2.5D0
minf=alpha/(alpha+beta)
ISKN=gSKN*minf*minf*(VmNX-EKN)

alpha=1.25D8*CaAX**2
beta=2.5D0
minf=alpha/(alpha+beta)
ISKA=gSKA*minf*minf*(VmAZ-EKA)

! K IK-channel (K2 Ca-activated)

alpha=25D0
beta=0.075D0*exp(-(VmNX+5D0)/10D0)
minf=alpha/(alpha+beta)
IIKN=gIKN*minf*(VmNX-EKN)/(1D0+0.0002D0/CaNX)**2

alpha=25D0
beta=0.075D0*exp(-(VmAZ+5D0)/10D0)
minf=alpha/(alpha+beta)
IIKA=gIKA*minf*(VmAZ-EKA)/(1D0+0.0002D0/CaAX)**2

! Na transient (fast)

alpha=35D0*exp((VmNX+5D0)/10D0)
beta=7D0*exp(-(VmNX+65D0)/20D0)
minf=alpha/(alpha+beta)
alpha=0.225D0/(1D0+exp((VmNX+80D0)/10D0))
beta=7.5D0*exp((VmNX-3D0)/18D0)
hinf=alpha/(alpha+beta)
INaFN=gNaFN*minf**3*hinf*(VmNX-ENaN)

alpha=35D0*exp((VmAZ+5D0)/10D0)
beta=7D0*exp(-(VmAZ+65D0)/20D0)
minf=alpha/(alpha+beta)
alpha=0.225D0/(1+exp((VmAZ+80D0)/10D0))
beta=7.5D0*exp((VmAZ-3D0)/18D0)
hinf=alpha/(alpha+beta)
INaFA=gNaFA*minf**3*hinf*(VmAZ-ENaA)

! Na persistent current

alpha=200D0/(1D0+exp(-(VmAZ-18D0)/16D0))
beta=25D0/(1D0+exp((VmAZ+58D0)/8D0))
minf=alpha/(alpha+beta)
INaPA=gNaPA*minf**3*(VmAZ-ENaA)

alpha=200D0/(1+exp(-(VmNX-18D0)/16D0))
beta=25D0/(1+exp((VmNX+58D0)/8D0))
minf=alpha/(alpha+beta)
INaPN=gNaPN*minf**3*(VmNX-ENaN)

! Ca high voltage activated (P-type)

alpha=8.5D0/(1D0+exp(-(VmAZ-8D0)/12.5D0))
beta=35D0/(1D0+exp((VmAZ+74D0)/14.5D0))
minf=alpha/(alpha+beta)
alpha=0.0015D0/(1D0+exp((VmAZ+29D0)/8D0))
beta=0.0055D0/(1D0+exp(-(VmAZ+23D0)/8D0))
hinf=alpha/(alpha+beta)
ICaHVAA=gCaHVAA*minf*hinf*(VmAZ-ECaA)

alpha=8.5D0/(1+exp(-(VmNX-8D0)/12.5D0))
beta=35D0/(1+exp((VmNX+74D0)/14.5D0))
minf=alpha/(alpha+beta)
alpha=0.0015D0/(1D0+exp((VmNX+29D0)/8D0))
beta=0.0055D0/(1D0+exp(-(VmNX+23D0)/8D0))
hinf=alpha/(alpha+beta)
ICaHVAN=gCaHVAN*minf*hinf*(VmNX-ECaN)

! Ca low voltage activated (T-type)

!alpha=2.6D0/(1D0+exp(-(VmNX+21D0)/8D0))
!beta=0.018D0/(1D0+exp((VmNX+40D0)/4D0))
!minf=alpha/(alpha+beta)
!alpha=0.0025D0/(1D0+exp((VmNX+40D0)/8D0))
!beta=0.19D0/(1D0+exp(-(VmNX+50D0)/10D0))
!hinf=alpha/(alpha+beta)
!ICaLVAN=gCaLVAN*minf*hinf*(VmNX-ECaN)
ICaLVAN=0

!alpha=2.6D0/(1D0+exp(-(VmAZ+21D0)/8D0))
!beta=0.018D0/(1D0+exp((VmAZ+40D0)/4D0))
!minf=alpha/(alpha+beta)
!alpha=0.0025D0/(1D0+exp((VmAZ+40D0)/8D0))
!beta=0.19D0/(1D0+exp(-(VmAZt+50D0)/10D0))
!hinf=alpha/(alpha+beta)
!ICaLVAA=gCaLVAA*minf*hinf*(VmAZ-ECaA)
ICaLVAA=0

! NMDA receptor is closed in initial state

! Cl pump

IClpumpA=-rClA*ClAX/(ClAX+25D0)
IClpumpN=-rClN*ClNX/(ClNX+25D0)

! Ca pump

ICapumpA=rCaA*CaAX/(CaAX+0.0002D0)
ICapumpN=rCaN*CaNX/(CaNX+0.0002D0)

! K/Na exchange

phi=(VmAZ+176.5D0)/RTF
phi=0.052D0*sinh(phi)/(0.026D0*exp(phi)+22.5D0*exp(-phi))
IKpumpA=-rNaKA*(KIX/(KIX+3.7D0))**2*(NaAX/(NaAX+0.6D0))**3*phi
INapumpA=-1.5D0*IKpumpA

phi=(VmNX+176.5D0)/RTF
phi=0.052D0*sinh(phi)/(0.026D0*exp(phi)+22.5D0*exp(-phi))
IKpumpN=-rNaKN*(KIX/(KIX+3.7D0))**2*(NaNX/(NaNX+0.6D0))**3*phi
INapumpN=-1.5D0*IKpumpN

! Na/Ca exchanger

phi=VmNX/RTF
numerator=(NaNX**3*CaIX*exp(0.35D0*phi)-NaIX**3*CaNX*2.5D0*exp(-0.65D0*phi))
denominator=(87.5D0**3+NaIX**3)*(1.38D0+CaIX)*(1D0+0.1D0*exp(-0.65D0*phi))
ICaantN=-rNaCaN*numerator/denominator
INaantN=-1.5D0*ICaantN

phi=VmAZ/RTF
numerator=(NaAX**3*CaIX*exp(0.35D0*phi)-NaIX**3*CaAX*2.5D0*exp(-0.65D0*phi))
denominator=(87.5D0**3+NaIX**3)*(1.38D0+CaIX)*(1D0+0.1D0*exp(-0.65D0*phi))
ICaantA=-rNaCaA*numerator/denominator
INaantA=-1.5D0*ICaantA

! K/Cl transporter

IKexchN=gKClN*RTF*log((KNX/KIX)*(ClNX/ClIX))
IClexchN=-IKexchN

IKexchA=gKClA*RTF*log((KAX/KIX)*(ClAX/ClIX))
IClexchA=-IKexchA

! Na/K/Cl transporter

IKastroA=-gNaKClA*RTF*log((NaIX/NaAX)*(KIX/KAX)*(ClIX/ClAX)**2)
INaastroA=IKastroA
IClastroA=-2D0*IKastroA

! Total currents excluding leaks - these to be in pA/um2

IKTotalN=IKDRN+IBKN+IKAN+IKMN+ISKN+IIKN+IKpumpN+IKexchN
INaTotalN=INaFN+INaPN+INapumpN+INaantN
ICaTotalN=ICaHVAN+ICaLVAN+ICapumpN+ICaantN
IClTotalN=IClpumpN+IClexchN
IKTotalA=IKDRA+IBKA+IKirA+IKAA+IKMA+ISKA+IIKA+IKpumpA+IKexchA+IKastroA
INaTotalA=INaFA+INaPA+INapumpA+INaantA+INaastroA
ICaTotalA=ICaHVAA+ICaLVAA+ICapumpA+ICaantA
IClTotalA=IClpumpA+IClexchA+IClastroA

! ==== Set leaks to create initial balance ====

gKLeakN=-IKTotalN/(VmNX-EKN)
gNaLeakN=-INaTotalN/(VmNX-ENaN)
gCaLeakN=-ICaTotalN/(VmNX-ECaN)
gClLeakN=-IClTotalN/(VmNX-EClN)

gKLeakA=-IKTotalA/(VmAZ-EKA)
gNaLeakA=-INaTotalA/(VmAZ-ENaA)
gCaLeakA=-ICaTotalA/(VmAZ-ECaA)
gClLeakA=-IClTotalA/(VmAZ-EClA)

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
end subroutine SetLeaks
