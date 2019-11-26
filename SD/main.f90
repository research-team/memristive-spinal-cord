!**********************
! ASTROCYTE DIFFUSION *
!**********************
program  Astrocyte

use global

implicit none

! local variables

integer :: it,iii	! time counter
double precision :: nextmovie,savetimestep,savetwostep
integer :: imovie
!
! read the data
!

call SetDefaults
open(10,file='astro.dat',status='OLD')

run_loop: do
	call NextBatch
	if (irunno.lt.0) exit	! exit after several runs
!
! setup
!

	call InitRAN3(0)
	call FlagCells
	call SetFlux
	call InitATP(.TRUE.)
	call InitCa
	call InitIons
	if (lvaryKR.GT.0) then
		call InitvKR
	endif
	call OpenOutput
! the initial state used to be saved here, incorrectly
!
! preliminary loop to bring the IP3 to a sort of equilibrium
!
	if (lusedelta) then
		interactloop: do iii=1,npreiter
			ip3loop: do it=1,npresteps
				call IP3all(1)
				call IP3all(2)
!!t = (it + (iii-1)*(npresteps+npreca))* 2.0D0 * timestep
!!if (mod(it,10).eq.0) call SaveResults
			enddo ip3loop
			savetimestep=timestep
			savetwostep=twostep
			timestep=timestep*100.0D0
			twostep=twostep*100.0D0
			caloop: do it=1,npreca
				call Caall
!!t = (it+npresteps + (iii-1)*(npresteps+npreca)) * 2.0D0 * savetimestep
!!if (mod(it,10).eq.0) call SaveResults
			enddo caloop
			timestep=savetimestep
			twostep=savetwostep
		enddo interactloop
		call ReInitCa
	endif

!!call CloseOutput
!!call EndBatch
!!stop

!
! save initial state
!

t=0.0D0
call SaveResults
if (movietime.GT.0.0D0) then
	imovie=1
	call SaveFinal
	nextmovie=movietime
	call ResetOutput(imovie)
else
	nextmovie=2.0D0*endtime
endif

!
! loop over time
!
	tloop: do it=1,ntsteps
		call stepall(1)
		t = (it * 2.0D0 - 1.0D0) * timestep
!		call BoundaryConds
		if (startperiod.GT.0.0D0.and.t.le.startperiod) call ExtendStep
		call StepAll(2)

		t = it * 2.0D0 * timestep
!		call BoundaryConds

		call StepIons

		if (startperiod.GT.0.0D0.and.t.le.startperiod) call ExtendStep
		if (mod(it,10).eq.0) call SaveResults
		if (t.GT.nextmovie) THEN
			call SaveFinal
			nextmovie=nextmovie+movietime
			imovie=imovie+1
			call ResetOutput(imovie)
		endif
		if (it.GE.ntsteps) Call SaveFinal
	enddo tloop

!
! clean up
!
	call CloseOutput
	call EndBatch
	if (irunno.eq.0) exit	! exit after a single unnumbered run
end do run_loop
close(10)
stop
end program Astrocyte
