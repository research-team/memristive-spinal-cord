import numpy as np
import pylab
import random
import matplotlib.pyplot as plt
import os
import time
from mpi4py import MPI
from scipy import interpolate

class NeuralNetwork:

	from neuron import h

	def __init__(self):

		global starting_time_total
		global iteration_count
		global iteration_start
		starting_time_total = time.time()
		iteration_start = starting_time_total
		iteration_count = 0

		self.EES = 0
		self.EES_freqeqncy = 0
		self.weightStimMn = 20
		self.weightStimAff = 10

		self.comm = MPI.COMM_WORLD
		self.sizeComm = self.comm.Get_size()
		self.rank = self.comm.Get_rank()

		self.percFibersIa_GM = 0
		self.percFibersII_GM = 0
		self.percMn_GM = 0
		self.percFibersIa_TA = 0
		self.percFibersII_TA = 0
		self.percMn_TA = 0

		#Initializing
		self.h.xopen("NeuralNetwork.hoc")
		self.h('{t=pc.set_maxstep(0.5)}')
		self.h.stdinit()
		self.h.t=0

	def __del__(self):

		self.h('{pc.runworker()}')
		self.h('{pc.done()}')
		self.h('t2 = startsw() // timer')
		self.h('print "model setup time ", t1-t0, " run time ", t2-t1, " total ", t2-t0')
		self.h.quit()

	# Compute indexes in the different hosts
	def computeInd(self,nCells):
		StimInd_0 = nCells//self.h.Nhost
		if(self.h.PcID< nCells%self.h.Nhost):
			StimInd_0+=1
		return StimInd_0

	#EES stimulation Ia fibers
	def set_IA_stim(self,percFlex,percExt, w):
		i=0
		"""
		Activating Flexors
		"""
		nFlex = int(percFlex*self.h.stimIafEES_Flex.count())
		nFlex_dec = percFlex*self.h.stimIafEES_Flex.count()-nFlex
		nFlex_dec = self.comm.gather(nFlex_dec,root=0)
		nFlex_extra = []
		if self.rank == 0:
			nFlex_extra = round(sum(nFlex_dec))
		nFlex_extra = self.comm.bcast(nFlex_extra,root=0)
		nFlex_extra = self.computeInd(nFlex_extra)
		nFlex += int(nFlex_extra)

		# Randomize excited Iaf
		nIAfxhost = np.zeros(self.h.stimIafEES_Flex.count())
		for i in range(nFlex):
			nIAfxhost[i]=1
		random.shuffle(nIAfxhost)
		indFlex = np.nonzero(nIAfxhost)

		for i in indFlex[0]:
			self.h.stimIafEES_Flex.object(i).weight[0]=w


		"""
		Activating Extensors
		"""
		nExt = int(percExt*self.h.stimIafEES_Ext.count())
		nExt_dec = percExt*self.h.stimIafEES_Ext.count()-nExt
		nExt_dec = self.comm.gather(nExt_dec,root=0)
		nExt_extra = []
		if self.rank == 0:
			nExt_extra = round(sum(nExt_dec))
		nExt_extra = self.comm.bcast(nExt_extra,root=0)
		nExt_extra = self.computeInd(nExt_extra)
		nExt += int(nExt_extra)

		# Randomize excited Iaf
		nIAfxhost = np.zeros(self.h.stimIafEES_Ext.count())
		for i in range(nExt):
			nIAfxhost[i]=1
		random.shuffle(nIAfxhost)
		indExt = np.nonzero(nIAfxhost)

		for i in indExt[0]:
			self.h.stimIafEES_Ext.object(i).weight[0]=w

	#EES stimulation II fibers
	def set_II_stim(self,percFlex,percExt, w):
		i=0
		"""
		Activating Flexors
		"""
		nFlex = int(percFlex*self.h.stimIIfEES_Flex.count())
		nFlex_dec = percFlex*self.h.stimIIfEES_Flex.count()-nFlex
		nFlex_dec = self.comm.gather(nFlex_dec,root=0)
		nFlex_extra = []
		if self.rank == 0:
			nFlex_extra = round(sum(nFlex_dec))
		nFlex_extra = self.comm.bcast(nFlex_extra,root=0)
		nFlex_extra = self.computeInd(nFlex_extra)
		nFlex += int(nFlex_extra)
		for i in range(nFlex):
			self.h.stimIIfEES_Flex.object(i).weight[0]=w

		"""
		Activating Extensors
		"""
		nExt = int(percExt*self.h.stimIIfEES_Ext.count())
		nExt_dec = percExt*self.h.stimIIfEES_Ext.count()-nExt
		nExt_dec = self.comm.gather(nExt_dec,root=0)
		nExt_extra = []
		if self.rank == 0:
			nExt_extra = round(sum(nExt_dec))
		nExt_extra = self.comm.bcast(nExt_extra,root=0)
		nExt_extra = self.computeInd(nExt_extra)
		nExt += int(nExt_extra)

		for i in range(nExt):
			self.h.stimIIfEES_Ext.object(i).weight[0]=w

	#EES stimulation Mns
	def set_Mn_stim(self,percFlex,percExt, w):

		i=0
		"""
		Activating Flexors
		"""
		nFlex = int(percFlex*self.h.stimMn_Flex.count())
		nFlex_dec = percFlex*self.h.stimMn_Flex.count()-nFlex
		nFlex_dec = self.comm.gather(nFlex_dec,root=0)
		nFlex_extra = []
		if self.rank == 0:
			nFlex_extra = round(sum(nFlex_dec))
		nFlex_extra = self.comm.bcast(nFlex_extra,root=0)
		nFlex_extra = self.computeInd(nFlex_extra)
		nFlex += int(nFlex_extra)

		for i in range(nFlex):
			self.h.stimMn_Flex.object(i).weight[0]=w

		"""
		Activating Extensors
		"""
		nExt = int(percExt*self.h.stimMn_Ext.count())
		nExt_dec = percExt*self.h.stimMn_Ext.count()-nExt
		nExt_dec = self.comm.gather(nExt_dec,root=0)
		nExt_extra = []
		if self.rank == 0:
			nExt_extra = round(sum(nExt_dec))
		nExt_extra = self.comm.bcast(nExt_extra,root=0)
		nExt_extra = self.computeInd(nExt_extra)
		nExt += int(nExt_extra)
		for i in range(nExt):
			self.h.stimMn_Ext.object(i).weight[0]=w

	#Set the EES frequency (freq in Hz)
	def set_EES_freq(self,freq):
		if (self.h.pc.gid_exists(self.h.nCell*2)):
			EES = self.h.pc.gid2cell(self.h.nCell*2)
			EES.interval=1000/freq

	# Set Iaf natural firing rate
	def set_IA_natural_firing(self,FiringRateFl,FiringRateExt,w):
		i=0
		"""
		Flexors
		"""
		#Setting to 0 the weight of all populations
		for i in range(int(self.h.stimIafNat_Flex_50.count())):
			self.h.stimIafNat_Flex_50.object(i).weight[0]=0
		for i in range(int(self.h.stimIafNat_Flex_40.count())):
			self.h.stimIafNat_Flex_40.object(i).weight[0]=0
		for i in range(int(self.h.stimIafNat_Flex_30.count())):
			self.h.stimIafNat_Flex_30.object(i).weight[0]=0
		for i in range(int(self.h.stimIafNat_Flex_20.count())):
			self.h.stimIafNat_Flex_20.object(i).weight[0]=0

		if np.mean(FiringRateFl)>=50:
			#set the weight to the selected stim population
			for i in range(int(self.h.stimIafNat_Flex_50.count())):
				self.h.stimIafNat_Flex_50.object(i).weight[0]=w
			#set the firing rate
			for i in range(int(self.h.nIAf)):
				if (self.h.pc.gid_exists(self.h.Ind_IaFibStimFlex+i)):
					Iaf = self.h.pc.gid2cell(self.h.Ind_IaFibStimFlex+i)
					Iaf.interval=1000/FiringRateFl[i]
		elif (np.mean(FiringRateFl)<50 and np.mean(FiringRateFl)>35):
			#set the weight to the selected stim population
			for i in range(int(self.h.stimIafNat_Flex_40.count())):
				self.h.stimIafNat_Flex_40.object(i).weight[0]=w
		elif (np.mean(FiringRateFl)<=35 and np.mean(FiringRateFl)>25):
			#set the weight to the selected stim population
			for i in range(int(self.h.stimIafNat_Flex_30.count())):
				self.h.stimIafNat_Flex_30.object(i).weight[0]=w
		elif (np.mean(FiringRateFl)<=25 and np.mean(FiringRateFl)>15):
			#set the weight to the selected stim population
			for i in range(int(self.h.stimIafNat_Flex_20.count())):
				self.h.stimIafNat_Flex_20.object(i).weight[0]=w

		"""
		Extensor
		"""
		#Setting to 0 the weight of all populations
		for i in range(int(self.h.stimIafNat_Ext_50.count())):
			self.h.stimIafNat_Ext_50.object(i).weight[0]=0
		for i in range(int(self.h.stimIafNat_Ext_40.count())):
			self.h.stimIafNat_Ext_40.object(i).weight[0]=0
		for i in range(int(self.h.stimIafNat_Ext_30.count())):
			self.h.stimIafNat_Ext_30.object(i).weight[0]=0
		for i in range(int(self.h.stimIafNat_Ext_20.count())):
			self.h.stimIafNat_Ext_20.object(i).weight[0]=0

		if np.mean(FiringRateExt)>=50:
			#set the weight to the selected stim population
			for i in range(int(self.h.stimIafNat_Ext_50.count())):
				self.h.stimIafNat_Ext_50.object(i).weight[0]=w
			#set the firing rate
			for i in range(int(self.h.nIAf)):
				if (self.h.pc.gid_exists(self.h.Ind_IaFibStimExt+i)):
					Iaf = self.h.pc.gid2cell(self.h.Ind_IaFibStimExt+i)
					Iaf.interval=1000/FiringRateExt[i]
		elif (np.mean(FiringRateExt)<50 and np.mean(FiringRateExt)>35):
			#set the weight to the selected stim population
			for i in range(int(self.h.stimIafNat_Ext_40.count())):
				self.h.stimIafNat_Ext_40.object(i).weight[0]=w
		elif (np.mean(FiringRateExt)<=35 and np.mean(FiringRateExt)>25):
			#set the weight to the selected stim population
			for i in range(int(self.h.stimIafNat_Ext_30.count())):
				self.h.stimIafNat_Ext_30.object(i).weight[0]=w
		elif (np.mean(FiringRateExt)<=25 and np.mean(FiringRateExt)>15):
			#set the weight to the selected stim population
			for i in range(int(self.h.stimIafNat_Ext_20.count())):
				self.h.stimIafNat_Ext_20.object(i).weight[0]=w

	# Set IIf natural firing rate
	def set_II_natural_firing(self,FiringRateFl,FiringRateExt,w):
		i=0
		"""
		Flexors
		"""
		#Setting to 0 the weight of all populations
		for i in range(int(self.h.stimIIfNat_Flex_50.count())):
			self.h.stimIIfNat_Flex_50.object(i).weight[0]=0
		for i in range(int(self.h.stimIIfNat_Flex_40.count())):
			self.h.stimIIfNat_Flex_40.object(i).weight[0]=0
		for i in range(int(self.h.stimIIfNat_Flex_30.count())):
			self.h.stimIIfNat_Flex_30.object(i).weight[0]=0
		for i in range(int(self.h.stimIIfNat_Flex_20.count())):
			self.h.stimIIfNat_Flex_20.object(i).weight[0]=0

		if np.mean(FiringRateFl)>=50:
			#set the weight to the selected stim population
			for i in range(int(self.h.stimIIfNat_Flex_50.count())):
				self.h.stimIIfNat_Flex_50.object(i).weight[0]=w
			#set the firing rate
			for i in range(int(self.h.nIIf)):
				if (self.h.pc.gid_exists(self.h.Ind_IIFibStimFlex+i)):
					IIf = self.h.pc.gid2cell(self.h.Ind_IIFibStimFlex+i)
					IIf.interval=1000/FiringRateFl[i]
		elif (np.mean(FiringRateFl)<50 and np.mean(FiringRateFl)>35):
			#set the weight to the selected stim population
			for i in range(int(self.h.stimIIfNat_Flex_40.count())):
				self.h.stimIIfNat_Flex_40.object(i).weight[0]=w
		elif (np.mean(FiringRateFl)<=35 and np.mean(FiringRateFl)>25):
			#set the weight to the selected stim population
			for i in range(int(self.h.stimIIfNat_Flex_30.count())):
				self.h.stimIIfNat_Flex_30.object(i).weight[0]=w
		elif (np.mean(FiringRateFl)<=25 and np.mean(FiringRateFl)>15):
			#set the weight to the selected stim population
			for i in range(int(self.h.stimIIfNat_Flex_20.count())):
				self.h.stimIIfNat_Flex_20.object(i).weight[0]=w
		"""
		Extensor
		"""
		#Setting to 0 the weight of all populations
		for i in range(int(self.h.stimIIfNat_Ext_50.count())):
			self.h.stimIIfNat_Ext_50.object(i).weight[0]=0
		for i in range(int(self.h.stimIIfNat_Ext_40.count())):
			self.h.stimIIfNat_Ext_40.object(i).weight[0]=0
		for i in range(int(self.h.stimIIfNat_Ext_30.count())):
			self.h.stimIIfNat_Ext_30.object(i).weight[0]=0
		for i in range(int(self.h.stimIIfNat_Ext_20.count())):
			self.h.stimIIfNat_Ext_20.object(i).weight[0]=0

		if np.mean(FiringRateExt)>=50:
			#set the weight to the selected stim population
			for i in range(int(self.h.stimIIfNat_Ext_50.count())):
				self.h.stimIIfNat_Ext_50.object(i).weight[0]=w
			#set the firing rate
			for i in range(int(self.h.nIIf)):
				if (self.h.pc.gid_exists(self.h.Ind_IIFibStimExt+i)):
					IIf = self.h.pc.gid2cell(self.h.Ind_IIFibStimExt+i)
					IIf.interval=1000/FiringRateExt[i]
		elif (np.mean(FiringRateExt)<50 and np.mean(FiringRateExt)>35):
			#set the weight to the selected stim population
			for i in range(int(self.h.stimIIfNat_Ext_40.count())):
				self.h.stimIIfNat_Ext_40.object(i).weight[0]=w
		elif (np.mean(FiringRateExt)<=35 and np.mean(FiringRateExt)>25):
			#set the weight to the selected stim population
			for i in range(int(self.h.stimIIfNat_Ext_30.count())):
				self.h.stimIIfNat_Ext_30.object(i).weight[0]=w
		elif (np.mean(FiringRateExt)<=25 and np.mean(FiringRateExt)>15):
			#set the weight to the selected stim population
			for i in range(int(self.h.stimIIfNat_Ext_20.count())):
				self.h.stimIIfNat_Ext_20.object(i).weight[0]=w

	# Run simulation
	def runSimulation(self,frequency=40,name="",amplitude="optimal"):

		#Initializing
		self.h('{t=pc.set_maxstep(0.5)}')
		self.h.finitialize(0)
		self.h.stdinit()
		self.h.t=0

		dt=5 						# simulation step
		changeParam = 20    		# change parameters every 20 ms
		self.h.tstop= 7500			# 7.5 s, simulation max time

		#loading the FIRING RATES of IA and II fibers, updates at 50 Hz
		#matrix[nIAf][nTIME]
		Ia_MG=np.loadtxt('../afferentsData/fr_Ia_GM'+str(name)+'.txt')
		Ia_TA=np.loadtxt('../afferentsData/fr_Ia_TA'+str(name)+'.txt')
		II_MG=np.loadtxt('../afferentsData/fr_II_GM'+str(name)+'.txt')
		II_TA=np.loadtxt('../afferentsData/fr_II_TA'+str(name)+'.txt')

		nTIME=Ia_TA[1].size

		if amplitude == "optimal":
			optimalRecGm=np.loadtxt('../Recruitment_data/OptimalAmpRecrGM_IaIIMnCur.txt')
			suboptimalRecTa=np.loadtxt('../Recruitment_data/SuboptimalAmpRecrTa_IaIIMnCur.txt')
			self.percFibersIa_GM= optimalRecGm[0]
			self.percFibersII_GM= optimalRecGm[1]
			self.percMn_GM = optimalRecGm[2]
			self.percFibersIa_TA= suboptimalRecTa[0]
			self.percFibersII_TA= suboptimalRecTa[1]
			self.percMn_TA = suboptimalRecTa[2]
		elif amplitude > 0 and amplitude <600:
			availableCurrents = np.linspace(0,600,20)
 			temp = abs(availableCurrents-amplitude)
 			indx = temp.argmin()

			recIa_MG=np.loadtxt('../Recruitment_data/GM_full_S1_wire1')
			recII_MG=np.loadtxt('../Recruitment_data/GM_full_ii_S1_wire1')
			recMn_MG=np.loadtxt('../Recruitment_data/MGM_full_S1_wire1')
			recIa_TA=np.loadtxt('../Recruitment_data/TA_full_S1_wire1')
			recII_TA=np.loadtxt('../Recruitment_data/TA_full_ii_S1_wire1')
			recMn_TA=np.loadtxt('../Recruitment_data/MTA_full_S1_wire1')
			self.percFibersIa_GM= recIa_MG[indx]/max(recIa_MG)
			self.percFibersII_GM= recII_MG[indx]/max(recII_MG)
			self.percMn_GM = recMn_MG[indx]/max(recMn_MG)
			self.percFibersIa_TA= recIa_TA[indx]/max(recIa_TA)
			self.percFibersII_TA= recII_TA[indx]/max(recII_TA)
			self.percMn_TA = recMn_TA[indx]/max(recMn_TA)


		self.set_IA_stim(1,1,0)
		self.set_II_stim(1,1,0)
		self.set_Mn_stim(1,1,0)
		if frequency>0:
			self.EES=1
			self.EES_freqeqncy = frequency
			self.set_IA_stim(self.percFibersIa_TA,self.percFibersIa_GM, self.weightStimAff)
			self.set_II_stim(self.percFibersII_TA,self.percFibersII_GM, self.weightStimAff)
			self.set_Mn_stim(self.percMn_TA, self.percMn_GM, self.weightStimMn)

			# Setting EES frequency
			if self.h.PcID ==0:
				print "EES set at "+str(self.EES_freqeqncy)+" Hz"
				self.set_EES_freq(self.EES_freqeqncy)
				print "\t{:.2f}".format(self.percFibersIa_GM*100)+"% of GM Ia fibers, "+"{:.2f}".format(self.percFibersII_GM*100)+"% of GM II fibers and "+"{:.2f}".format(self.percMn_GM*100)+"% of GM Mn receive EES "
				print "\t{:.2f}".format(self.percFibersIa_TA*100)+"% of TA Ia fibers, "+"{:.2f}".format(self.percFibersII_TA*100)+"% of TA II fibers and "+"{:.2f}".format(self.percMn_TA*100)+"% of TA Mn receive EES "

		"""
		MAIN RUNNING LOOP
		"""
		t_old = self.h.t
		j=0
		while (self.h.t<self.h.tstop and j< nTIME):

			if (self.h.t>=t_old+changeParam-0.01):
				self.set_IA_natural_firing(Ia_TA[:,j],Ia_MG[:,j],self.weightStimAff)
				self.set_II_natural_firing(II_TA[:,j],II_MG[:,j],self.weightStimAff)
				j+=1
				t_old = self.h.t
				if(self.h.PcID==0):
					print "\t{:.1f}".format(self.h.t*100/self.h.tstop)+"%"

			self.h.pc.psolve(self.h.t+dt)


		"""
		EXTRACT EMG
		"""
		if self.rank==0:
			print "\nExctracting cells firings..."
		AP_Flex = self.apListToMatrix(self.h.nMN,self.h.AP_MN_Flex)
		AP_Ext = self.apListToMatrix(self.h.nMN,self.h.AP_MN_Ext)
		if self.rank==0:
			firings_Flex = self.extract_firings(AP_Flex, self.h.tstop/1000)
			firings_Ext = self.extract_firings(AP_Ext, self.h.tstop/1000)

			print "\nComputing EMG signals..."
			EMG_Flex = self.synth_EMG(firings_Flex, self.h.tstop/1000)
			EMG_Ext = self.synth_EMG(firings_Ext, self.h.tstop/1000)

			# plotting
			f1, ax_1 = plt.subplots(2, figsize=(20, 9), dpi=80, facecolor='w', edgecolor='k', sharex=True, sharey=True)
			ax_1[0].plot(EMG_Flex,'-',label='Flexor EMG')
			ax_1[1].plot(EMG_Ext,'-',label='Extensor EMG')

			ax_1[0].legend(loc='upper left')
			ax_1[1].legend(loc='upper left')
			ax_1[0].set_title("Estimated EMG - EES frequency: "+str(frequency)+ "Hz - EES amplitude: "+str(amplitude))

			if not os.path.exists('../Results/'):
				os.makedirs('../Results/')
			f1.savefig("../Results/DynamicSimulation_EMG_EES_fr_"+str(int(frequency))+"Hz_amp_"+str(amplitude)+".pdf")
			np.savetxt("../Results/DynamicSimulation_EMG_Flex_EES_fr_"+str(int(frequency))+"Hz_amp_"+str(amplitude)+'.txt',EMG_Flex,delimiter='')
			np.savetxt("../Results/DynamicSimulation_EMG_Ext_EES_fr_"+str(int(frequency))+"Hz_amp_"+str(amplitude)+'.txt',EMG_Ext,delimiter='')

			plt.show()

	# ghater and transform the vector of vector of AP into a matrix in process 0
	def apListToMatrix(self,cellType,apList):

		nCellxHost=self.computeInd(cellType)
		tot_nCellxHost=[]
		self.comm.gather(nCellxHost,tot_nCellxHost,root=0)
		tot_nCellxHost = self.comm.bcast(tot_nCellxHost,root=0)
		hostsWithMoreCells = [i for i, x in enumerate(tot_nCellxHost) if x==max(tot_nCellxHost)]

		nApXhost = [apList[z].size() for z in range(int(nCellxHost))]
		maxNapXhost = max(nApXhost)
		maxNapXhost = self.comm.gather(maxNapXhost,root=0)

		maxNap = None
		if self.rank == 0:
			maxNap = max(maxNapXhost)
		maxNap = self.comm.bcast(maxNap,root=0)

		apXhost = -1*np.ones([nCellxHost,maxNap])
		for z in range(int(nCellxHost)):
			for k in range(int(apList[z].size())):
				apXhost[z,k]=apList[z].x[k]

		if self.sizeComm<=1:
			return apXhost

		ap = self.comm.gather(apXhost, root=0)
		AP = None
		if self.rank==0:
			if self.sizeComm>1:
				AP = np.concatenate([ap[0],ap[1]])
				for i in range(2,self.sizeComm):
					AP = np.concatenate([AP,ap[i]])
		return AP

	# extract firings from a matrix/vector of AP event
	def extract_firings(self, AP, nSec): # AP in ms
		sampling_rate = 5000.
		dt = 1000./sampling_rate


		firings = np.zeros([AP.shape[0],int(sampling_rate*nSec)])
		# check wheter we have more AP for each cell or not
		if len(AP.shape)==2:
			for i in range(AP.shape[0]):
				for ii in range(AP.shape[1]):
					for j in range(int(sampling_rate*nSec)):
						if AP[i,ii]>=j*dt and AP[i,ii]<(j+1)*dt:
							firings[i,j]=1
		elif len(AP.shape)==1:
			for i in range(AP.shape[0]):
				for j in range(int(sampling_rate*nSec)):
					if AP[i]>=j*dt and AP[i]<(j+1)*dt:
						firings[i,j]=1

		return firings

	# sythetise the EMG from a firings matrix
	def synth_EMG(self, firings, nSec): # AP in ms

		sampling_rate = 5000.
		dt = 1000./sampling_rate
		delay = int(2/dt)

		# MUAP duration between 5-10ms (Day et al 2001) -> 7.5 +-1
		menaLenMUAP = int(7.5/dt)
		stdLenMUAP = int(1/dt)
		nS = [int(menaLenMUAP+random.gauss(0,stdLenMUAP)) for i in range(firings.shape[0])]
		Amp = [abs(1+random.gauss(0,0.2)) for i in range(firings.shape[0])]

		EMG = np.zeros(sampling_rate*nSec+ max(nS)+delay);

		# create MUAP shape
		for i in range(firings.shape[0]):
			n40perc = int(nS[i]*0.4)
			n60perc = nS[i]-n40perc
			amplitudeMod = (1-(np.linspace(0,1,nS[i])**2)) * np.concatenate((np.ones(n40perc),1/np.linspace(1,3,n60perc)))
			logBase = 1.05
			freqMod = np.log(np.linspace(1,logBase**(4*np.pi),nS[i]))/np.log(logBase)
			EMG_unit = Amp[i]*amplitudeMod*np.sin(freqMod);
			for j in range(int(sampling_rate*nSec)):
				if firings[i,j]==1:
					EMG[j+delay:j+delay+nS[i]]=EMG[j+delay:j+delay+nS[i]]+EMG_unit

		return EMG[:sampling_rate*nSec]

	# srecruitment curve
	def computeRecruitCurve(self,network="extensor"):

		if network=="extensor":
			#loading the number of IA and II fibers activated at a given current from the FEM model results
			Ia_nAct=np.loadtxt('../Recruitment_data/GM_full_S1_wire1')
			II_nAct=np.loadtxt('../Recruitment_data/GM_full_ii_S1_wire1')
			Mn_nAct=np.loadtxt('../Recruitment_data/MGM_full_S1_wire1')
			startCurr = 5


			if(self.h.PcID==0):
				print "Extensor h-reflex computation"
		else:
			#loading the number of IA and II fibers activated at a given current from the FEM model results
			Ia_nAct=np.loadtxt('../Recruitment_data/TA_full_S1_wire1')
			II_nAct=np.loadtxt('../Recruitment_data/TA_full_ii_S1_wire1')
			Mn_nAct=np.loadtxt('../Recruitment_data/MTA_full_S1_wire1')
			startCurr = 7

			network = "flexor"
			if(self.h.PcID==0):
				print "Flexor h-reflex computation"

		MnInd = int(self.computeInd(self.h.nMN))
		IafInd = int(self.computeInd(self.h.nIAf))
		IIfInd = int(self.computeInd(self.h.nIIf))
		IaiInd = int(self.computeInd(self.h.nIAint))
		self.set_EES_freq(8)

		ampResponse_Early = np.zeros(Ia_nAct.size)
		ampResponse_MediumLate = np.zeros(Ia_nAct.size)

		currAmp = range(startCurr,Ia_nAct.size)
		low_curr = 9
		high_curr = 13

		for jj in currAmp:

			# set EES to 0 on both Ia e II f
			self.set_IA_stim(1,1, 0)
			self.set_II_stim(1,1, 0)
			self.set_Mn_stim(1,1, 0)

			percFibersIa= Ia_nAct[jj]/max(Ia_nAct) # / n cells in FEM model
			percFibersII= II_nAct[jj]/max(II_nAct)
			percMn = Mn_nAct[jj]/max(Mn_nAct)

			if network=="extensor":
				self.set_IA_stim(0,percFibersIa, self.weightStimAff)
				self.set_II_stim(0,percFibersII, self.weightStimAff)
				self.set_Mn_stim(0,percMn, self.weightStimMn)
			else:
				self.set_IA_stim(percFibersIa, 0, self.weightStimAff)
				self.set_II_stim(percFibersII, 0, self.weightStimAff)
				self.set_Mn_stim(percMn, 0, self.weightStimMn)

			if(self.h.PcID==0):
				print "\nComputing the response on the "+network+"s MNs due to:"
				print "{:.2f}".format(percFibersIa*100) + "%  of the population of Ia fibers"
				print "{:.2f}".format(percFibersII*100) + "%  of the population of II fibers"
				print "{:.2f}".format(percMn*100) + "%  of the population of Mn cells\n"

			self.h.finitialize(0)
			self.h.stdinit()
			self.h.t = 0

			dt=0.025 						# simulation step
			self.h.tstop = 160	 			# Length of simulation
			self.h.pc.psolve(self.h.t+120) 	# remove initialization effects

			AP_MN_init = np.zeros(MnInd)
			AP_Iaf_init = np.zeros(IafInd)
			AP_IIf_init = np.zeros(IIfInd)

			if network=="extensor":
				for i in range(int(MnInd)):
					AP_MN_init[i] = self.h.AP_MN_Ext[i].size()
				for i in range(int(IafInd)):
					AP_Iaf_init[i] = self.h.AP_IA_Ext[i].size()
				for i in range(int(IIfInd)):
					AP_IIf_init[i] = self.h.AP_II_Ext[i].size()
			else :
				for i in range(int(MnInd)):
					AP_MN_init[i] = self.h.AP_MN_Flex[i].size()
				for i in range(int(IafInd)):
					AP_Iaf_init[i] = self.h.AP_IA_Flex[i].size()
				for i in range(int(IIfInd)):
					AP_IIf_init[i] = self.h.AP_II_Flex[i].size()

			AP_MN =  np.zeros((int(40/dt),MnInd))
			AP_Iaf = np.zeros((int(40/dt),IafInd))
			AP_IIf = np.zeros((int(40/dt),IIfInd))

			if(self.h.PcID==0):
				print "Initialization completed...\n "
				time = np.zeros(int(40/dt))

			count =0
			while (self.h.t<self.h.tstop and count < int(40/dt)):

				self.h.pc.psolve(self.h.t+dt)
				if network=="extensor":
					for i in range(int(MnInd)):
						AP_MN[count,i] = self.h.AP_MN_Ext[i].size() - AP_MN_init[i]
						AP_MN_init[i] = self.h.AP_MN_Ext[i].size()
					for i in range(int(IafInd)):
						AP_Iaf[count,i] = self.h.AP_IA_Ext[i].size() - AP_Iaf_init[i]
						AP_Iaf_init[i] = self.h.AP_IA_Ext[i].size()
					for i in range(int(IIfInd)):
						AP_IIf[count,i] = self.h.AP_II_Ext[i].size() - AP_IIf_init[i]
						AP_IIf_init[i] = self.h.AP_II_Ext[i].size()
				else :
					for i in range(int(MnInd)):
						AP_MN[count,i] = self.h.AP_MN_Flex[i].size() - AP_MN_init[i]
						AP_MN_init[i] = self.h.AP_MN_Flex[i].size()
					for i in range(int(IafInd)):
						AP_Iaf[count,i] = self.h.AP_IA_Flex[i].size() - AP_Iaf_init[i]
						AP_Iaf_init[i] = self.h.AP_IA_Flex[i].size()
					for i in range(int(IIfInd)):
						AP_IIf[count,i] = self.h.AP_II_Flex[i].size() - AP_IIf_init[i]
						AP_IIf_init[i] = self.h.AP_II_Flex[i].size()

				if(self.h.PcID==0):
					time[count] = self.h.t

				count+=1

			AP_MN = self.comm.gather(AP_MN, root=0)
			AP_Iaf = self.comm.gather(AP_Iaf, root=0)
			AP_IIf = self.comm.gather(AP_IIf, root=0)

			if(self.h.PcID==0):

				if self.sizeComm>1:
					# gathering the pot in one array
					Global_AP_MN = np.concatenate((AP_MN[0],AP_MN[1]),axis=1)
					for i in range(2,self.sizeComm):
						Global_AP_MN = np.concatenate((Global_AP_MN ,AP_MN[i]),axis=1)

					Global_AP_Iaf = np.concatenate((AP_Iaf[0],AP_Iaf[1]),axis=1)
					for i in range(2,self.sizeComm):
						Global_AP_Iaf = np.concatenate((Global_AP_Iaf ,AP_Iaf[i]),axis=1)

					Global_AP_IIf = np.concatenate((AP_IIf[0],AP_IIf[1]),axis=1)
					for i in range(2,self.sizeComm):
						Global_AP_IIf = np.concatenate((Global_AP_IIf ,AP_IIf[i]),axis=1)
				else:
					Global_AP_MN = AP_MN[0]
					Global_AP_Iaf= AP_Iaf[0]
					Global_AP_IIf= AP_IIf[0]

				IndexEES_Mn = np.concatenate((np.nonzero(Global_AP_MN[:,0]),np.nonzero(Global_AP_MN[:,1])), axis=1)
				for i in range(2,int(self.h.nMN)):
					IndexEES_Mn = np.concatenate((IndexEES_Mn,np.nonzero(Global_AP_MN[:,i])), axis=1)
				IndexEES_Mn=np.extract(IndexEES_Mn>1/dt,IndexEES_Mn)


				IndexEES_Iaf = np.concatenate((np.nonzero(Global_AP_Iaf[:,0]),np.nonzero(Global_AP_Iaf[:,1])), axis=1)
				for i in range(2,int(self.h.nIAf)):
					IndexEES_Iaf = np.concatenate((IndexEES_Iaf,np.nonzero(Global_AP_Iaf[:,i])), axis=1)
				IndexEES_Iaf=np.extract(IndexEES_Iaf>1/dt,IndexEES_Iaf)

				if IndexEES_Mn.size >0:
					EESpulseTime = np.mean(IndexEES_Iaf)*dt
					IndexMn_Early=np.extract(IndexEES_Mn<(EESpulseTime+3.5)/dt,IndexEES_Mn) # first EES pulse at 10 ms and second at 135 (15th ms recorded)
					IndexMn_MediumLate=np.extract(IndexEES_Mn>=(EESpulseTime+3.5)/dt,IndexEES_Mn)

					ampResponse_Early[jj] = IndexMn_Early.shape[0]/self.h.nMN
					ampResponse_MediumLate[jj] = IndexMn_MediumLate.shape[0]/self.h.nMN

					# to create the EMG responses
					if jj==low_curr:
						EESpulseTime_low = EESpulseTime
						IndexMn_Early_low = IndexMn_Early
						IndexMn_MediumLate_low = IndexMn_MediumLate

					elif jj==high_curr:
						EESpulseTime_high = EESpulseTime
						IndexMn_Early_high = IndexMn_Early
						IndexMn_MediumLate_high = IndexMn_MediumLate

				print "\nThe amplitude of the early response is: " + "{:.2f}".format(ampResponse_Early[jj]*100) + "% "
				print "The amplitude of the medium-late response is: " + "{:.2f}".format(ampResponse_MediumLate[jj]*100) + "% \n"

				iteration_count += 1
				print "\nIteration number: #" + iteration_count
				print "Iteration time: " + time.strftime("%H:%M:%S", time.gmtime(time.time() - iteration_start))
				print "Total simulation time: " + time.strftime("%H:%M:%S", time.gmtime(time.time() - starting_time_total))
				iteration_start = time.time()


		if(self.h.PcID==0) :

			# find current threshold
			current = np.linspace(0,600,Ia_nAct.size)
			tck = interpolate.splrep(current, ampResponse_MediumLate, s=0)
			current_precise = np.linspace(0,600,current.size*5)
			ampResp_ML_precise = interpolate.splev(current_precise,tck,der=0)
			for i in range(ampResp_ML_precise.size):
				if ampResp_ML_precise[i]<0:ampResp_ML_precise[i]=0

			temp = abs(ampResp_ML_precise[:ampResp_ML_precise.argmax()]-0.1)
			threshold = current_precise[temp.argmin()]
			curr_thresholds = current/threshold
			low_curr_thr = curr_thresholds[low_curr]
			high_curr_thr = curr_thresholds[high_curr]

			firings = self.extract_firings(IndexMn_MediumLate_low*dt, 0.04)
			EMG_ML_low = self.synth_EMG(firings, 0.04)

			firings = self.extract_firings(IndexMn_MediumLate_high*dt, 0.04)
			EMG_ML_high = self.synth_EMG(firings, 0.04)

			firings = self.extract_firings(IndexMn_Early_low*dt, 0.04)
			EMG_E_low = self.synth_EMG(firings, 0.04)

			firings = self.extract_firings(IndexMn_Early_high*dt, 0.04)
			EMG_E_high = self.synth_EMG(firings, 0.04)


			f1, ax_1 = plt.subplots(2, figsize=(20, 9), dpi=80, facecolor='w', edgecolor='k', sharex=True, sharey=True)
			ax_1[0].plot(EMG_E_low+EMG_ML_low,'-',label='EMG response - '+"{:.1f}".format(low_curr_thr)+'x motor thr')
			ax_1[0].plot([EESpulseTime_low*5,EESpulseTime_low*5],[0,50],'k',label = 'Stimulation pulse',linewidth=3.0)
			ax_1[0].axvspan(EESpulseTime_low*5, (EESpulseTime_low+3.5)*5, color='r', alpha=0.25, lw=0,label='Early response')
			ax_1[0].axvspan((EESpulseTime_low+3.5)*5,200, color='g', alpha=0.25, lw=0,label='Medium-late response')


			ax_1[1].plot(EMG_E_high+EMG_ML_high,'-',label='EMG response - '+"{:.1f}".format(high_curr_thr)+'x motor thr')
			ax_1[1].plot([EESpulseTime_high*5,EESpulseTime_high*5],[0,50],'k',label = 'Stimulation pulse',linewidth=3.0)
			ax_1[1].axvspan(EESpulseTime_high*5, (EESpulseTime_high+3.5)*5, color='r', alpha=0.25, lw=0,label='Early response')
			ax_1[1].axvspan((EESpulseTime_high+3.5)*5,200, color='g', alpha=0.25, lw=0,label='Medium-late response')

			ax_1[0].legend(loc='upper left')
			ax_1[1].legend(loc='upper left')
			ax_1[0].set_title("EMG responses - " + network + " network")

			f2 = plt.figure()
			ax_2 = f2.add_subplot(1,1,1)
			ax_2.plot(curr_thresholds,ampResponse_MediumLate,label='Medium-late response')
			ax_2.plot(curr_thresholds,ampResponse_Early,label='Early response')
			ax_2.legend(loc='upper left')
			ax_2.set_title("Recruitment curve - " + network + " network")
			ax_2.set_xlabel("Stimulation amp (x motor thr)")
			ax_2.set_ylabel("Response amplitude")
			ax_2.grid(True)

			if not os.path.exists('../Results/'):
				os.makedirs('../Results/')

			f1.savefig("../Results/EMG_responses"+network+".pdf")
			f2.savefig("../Results/Recruitment_curve"+network+".pdf")

			np.savetxt('../Results/current.txt',current,delimiter='')
			np.savetxt('../Results/ampResponse_ER_'+network+'.txt',ampResponse_Early,delimiter='')
			np.savetxt('../Results/ampResponse_MLR_'+network+'.txt',ampResponse_MediumLate,delimiter='')

			plt.show()
