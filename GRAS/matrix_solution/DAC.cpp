#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <dlfcn.h>
#include <pthread.h>
#include <iostream>
#include <math.h>

using namespace std;

#define INITGUID

#include "include/stubs.h"
#include "include/ioctl.h"
#include "include/ifc_ldev.h"
#include <errno.h>

typedef IDaqLDevice *(*CREATEFUNCPTR)(ULONG Slot);

CREATEFUNCPTR CreateInstance;

USHORT *data1;
ULONG *sync1;


void *thread_func(void *arg) {
	printf("HI!\n");
}


void errorchk(bool condition, string text, char* err_text) {
	cout << text << "... ";
	if (condition) {
		cout << "ERROR (" << err_text << ")" << endl;
		cout << "FAILED !" << endl;
		exit(0);
	}
	else
		cout << "OK" << endl;
}


int main() {
	PLATA_DESCR_U2 pd;
	SLOT_PAR sl;
	ADC_PAR adcPar;
	DAC_PAR dacPar;
	ULONG size = 512000;
	IOCTL_BUFFER ibuf;
	HANDLE hVxd;
	void *handle;
	HRESULT hr;
	IDaqLDevice *pI;

	char *error;
	pthread_t thread1;

	handle = dlopen("/home/alex/Programs/drivers/lcomp/liblcomp.so", RTLD_LAZY);
	errorchk(!handle, "open dll", dlerror());

	CreateInstance = (CREATEFUNCPTR) dlsym(handle, "CreateInstance");
	error = dlerror();
	errorchk(error != NULL, "create instance", error);


	LUnknown *pIUnknown = CreateInstance(0);
	errorchk(pIUnknown == NULL, "pIUnknown create instance", error);

	cout << "Get IDaqLDevice interface" << endl;

	hr = pIUnknown->QueryInterface(IID_ILDEV, (void **) &pI);
	errorchk(hr != S_OK, "get IDaqLDevice", error);

	pIUnknown->Release();
	cout << "free IUnknown" << endl;

	// открываем устройство
	hVxd = pI->OpenLDevice();
	cout << "OpenLDevice Handle " << hVxd << endl;


	cout << endl << "Slot parameters" << endl;
	// считали параметры слота - интересует тип платы
	pI->GetSlotParam(&sl);
	pI->LoadBios("e154"); // загружаем биос
	pI->PlataTest();

	pI->ReadPlataDescr(&pd);  // обязательно прочитали флеш, он нужен для расчетов внутри библиотеки

	cout << "Base        : " << hex << sl.Base << endl;
	cout << "BaseL       : " << sl.BaseL << endl;
	cout << "Mem         : " << sl.Mem << endl;
	cout << "MemL        : " << sl.MemL << endl;
	cout << "Type        : " << sl.BoardType << endl;
	cout << "DSPType     : " << sl.DSPType << endl;
	cout << "Irq         : " << sl.Irq << endl;
	cout << "SerNum      : " << pd.t7.SerNum << endl;
	cout << "BrdName     : " << pd.t7.BrdName << endl;
	cout << "Rev         : " << pd.t7.Rev << endl;
	cout << "DspType     : " << pd.t7.DspType << endl;
	cout << "IsDacPresent: " << pd.t7.IsDacPresent << endl;
	cout << "Quartz      : " << dec << pd.t7.Quartz << endl;
	cout << "Alloc size  : " << size << endl;

//	adcPar.t1.s_Type = L_ADC_PARAM;
//	adcPar.t1.AutoInit = 1;
//	adcPar.t1.dRate = 100.0;
//	adcPar.t1.dKadr = 0;
//	adcPar.t1.dScale = 0;
//	adcPar.t1.SynchroType = 0;
//	adcPar.t1.SynchroSensitivity = 0;
//	adcPar.t1.SynchroMode = 0;
//	adcPar.t1.AdChannel = 0;
//	adcPar.t1.AdPorog = 0;
//	adcPar.t1.NCh = 4;
//	adcPar.t1.Chn[0] = 0x0;
//	adcPar.t1.Chn[1] = 0x1;
//	adcPar.t1.Chn[2] = 0x2;
//	adcPar.t1.Chn[3] = 0x3;
//	adcPar.t1.FIFO = 4096;
//	adcPar.t1.IrqStep = 4096;
//	adcPar.t1.Pages = 32;
//	adcPar.t1.IrqEna = 1;
//	adcPar.t1.AdcEna = 1;


	dacPar.t1.s_Type = L_DAC_PARAM;
	dacPar.t1.AutoInit=1;
	dacPar.t1.dRate=100.0;       // for e140m dac - very limited set of freq value
	dacPar.t1.FIFO=2048; // 512
	dacPar.t1.IrqStep=2048; // 512
	dacPar.t1.Pages=4;
	dacPar.t1.IrqEna=1;
	dacPar.t1.DacEna=1;
	dacPar.t1.DacNumber=0;

	cout << "FillDAQparameters" << endl;
	pI->FillDAQparameters(&dacPar.t1);

	cout << "RequestBufferStream" << endl;
	pI->RequestBufferStream(&size, L_STREAM_DAC);

	cout << "SetParametersStream" << endl;
	pI->SetParametersStream(&dacPar.t1, &size, (void **)&data1, (void **)&sync1, L_STREAM_DAC);

	ULONG Ver = sync1[0xFF4 >> 2];
	cout << endl << "Current Firmware Version 0x" << hex << Ver << dec << endl;

	cout << "FOR LOOP" << endl;

	for(int i=0;i<5;i++) {
		cout << i << endl;

		data1[i] = (USHORT) (512 * sin((2.0 * (3.1415 * i) / 1024.0))); // for all
	}

	// тест цифровых линий
	cout << "START TEST" << endl;

	ASYNC_PAR pp1;
	pp1.s_Type = L_ASYNC_TTL_CFG;
	pp1.Mode = 1;
	pI->IoAsync(&pp1);

	pp1.s_Type = L_ASYNC_TTL_OUT;
	pp1.Data[0] = 0xA525;
	pI->IoAsync(&pp1);

	pp1.s_Type = L_ASYNC_TTL_INP;
	pp1.Data[0] = 1;
	pI->IoAsync(&pp1);

	cout << "TEST FINISHED" << endl;


	printf("\n ttl input %X ",pp1.Data[0]);

//	pI->FillDAQparameters(&adcPar.t1);
//	pI->SetParametersStream(&adcPar.t1, &size, (void **) &data1, (void **) &sync1, L_STREAM_ADC);
//	cout << "Word size   : " << size << endl;
//	cout << "Pages:      : " << adcPar.t1.Pages << endl;
//	cout << "IrqStep:    : " << adcPar.t1.IrqStep << endl;
//	cout << "FIFO:       : " << adcPar.t1.FIFO << endl;
//	cout << "Rate:       : " << adcPar.t1.dRate << endl;



	pI->EnableCorrection();

	cout << "init startL device" << endl;

	pI->InitStartLDevice();

	cout << "Create a thread" << endl;

	pthread_create(&thread1, NULL, thread_func, pI);

	cout << "Start device" << endl;

	pI->StartLDevice();

	printf("shared word %x\n", sync1[0]);

	pthread_join(thread1, NULL);

	cout << "Thread is finished" << endl;

	pI->StopLDevice();
	// Завершение работы
	pI->CloseLDevice();

	if (handle)
		dlclose(handle);


	cout << "Closed handle" << endl;
	cout << "SUCCESS" << endl;

	return 0;
}
