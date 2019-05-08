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

#include <termios.h>

static struct termios stored_settings, new_settings;
static int peek_character = -1;

static int ctrlc = 0;

void reset_keypress(void) {
	ctrlc = 1;
	tcsetattr(0, TCSANOW, &stored_settings);
	return;
}


void set_keypress(void) {
	tcgetattr(0, &stored_settings);

	new_settings = stored_settings;

	/* Disable canonical mode, and set buffer size to 1 byte */
	new_settings.c_lflag &= (~ICANON);
	new_settings.c_lflag &= ~ECHO;
	new_settings.c_lflag &= ~ISIG;
	new_settings.c_cc[VTIME] = 0;
	new_settings.c_cc[VMIN] = 1;

	atexit(reset_keypress);
	tcsetattr(0, TCSANOW, &new_settings);
	return;
}


int kbhit() {
	unsigned char ch;
	int nread;
	if (peek_character != -1)
		return 1;
	new_settings.c_cc[VMIN] = 0;
	tcsetattr(0, TCSANOW, &new_settings);
	nread = read(0, &ch, 1);
	new_settings.c_cc[VMIN] = 1;
	tcsetattr(0, TCSANOW, &new_settings);
	if (nread == 1) {
		peek_character = ch;
		return 1;
	}
	return 0;
}

int readch() {
	char ch;
	if (peek_character != -1) {
		ch = peek_character;
		peek_character = -1;
		return ch;
	}
	read(0, &ch, 1);
	return ch;
}


#define INITGUID

#include "include/stubs.h"
#include "include/ioctl.h"
#include "include/ifc_ldev.h"
#include <errno.h>

typedef IDaqLDevice *(*CREATEFUNCPTR)(ULONG Slot);

CREATEFUNCPTR CreateInstance;

unsigned short *data1;
unsigned int *sync1;

int IrqStep = 1024;
int pages = 256;
int multi = 64;
unsigned short complete;


void *thread_func(void *arg) {
	int halfbuffer;
	int fl2, fl1;
	unsigned short *tmp, *tmp1;
	int i;

	FILE *fd;

	fd = fopen("test.dat", "wb");

	halfbuffer = IrqStep * pages / 2;
	fl1 = fl2 = (*sync1 <= halfbuffer) ? 0 : 1;

	for (i = 0; i < multi; i++) {
		while (fl2 == fl1) {
			fl2 = (*sync1 <= halfbuffer) ? 0 : 1;
			if (ctrlc) break;
			usleep(10);
		}
		if (ctrlc) break;
		tmp1 = data1 + (halfbuffer * fl1);
		fwrite(tmp1, 1, halfbuffer * sizeof(short), fd);
		fl1 = (*sync1 <= halfbuffer) ? 0 : 1;
	}

	fclose(fd);
	complete = 1;
}

void errorchk(bool condition, string text, char* err_text="failed") {
	cout << text << " ... ";
	if (condition) {
		cout << "ERROR (" << err_text << ")" << endl;
		cout << "FAILED !" << endl;
		reset_keypress();
		exit(0);
	}
	else
		cout << "OK" << endl;
}

void errorchk(bool condition, HRESULT result, string text) {
	cout << hex << text << " ... ";
	if (condition) {
		cout << "ERROR" << endl;
		cout << "FAILED !" << endl;
		reset_keypress();
		exit(0);
	}
	else
		cout << hex << result << " OK" << endl;
}

int main(int argc, char **argv) {
	ULONG size = 131072;
	SLOT_PAR sl;
	ADC_PAR adcPar;
	DAC_PAR dacPar;
	IDaqLDevice *pI;
	PLATA_DESCR_U2 pd;
	IOCTL_BUFFER ibuf;
	HANDLE handle;
	char *error;
	void *dll_handle;
	ULONG iresult;
	pthread_t thread1;
	HRESULT hresult;
	LUnknown *pIUnknown;

	set_keypress();

	dll_handle = dlopen("/home/alex/Programs/drivers/lcomp/liblcomp.so", RTLD_LAZY);
	errorchk(!dll_handle, "open DLL", dlerror());

	CreateInstance = (CREATEFUNCPTR) dlsym(dll_handle, "CreateInstance");
	errorchk(dlerror() != NULL, "create instance");

	pIUnknown = CreateInstance(0);
	errorchk(pIUnknown == NULL, "call create instance");

	hresult = pIUnknown->QueryInterface(IID_ILDEV, (void **) &pI);
	errorchk(hresult != S_OK, hresult, "query interface");

	pIUnknown->Release();
	errorchk(false, "free IUnknown");

	handle = pI->OpenLDevice();
	errorchk(false, handle, "open device");

	pI->GetSlotParam(&sl);
	errorchk(false, "get slot parameters");

	iresult = pI->LoadBios("e154");
	errorchk(iresult == 0, "load BIOS");

	iresult = pI->PlataTest();
	errorchk(iresult != 0, "plata test");

	pI->ReadPlataDescr(&pd); // fill up properties
	errorchk(false, "read plata description");

	cout << endl << "Press any key" << dec << endl;

	readch();

	pI->RequestBufferStream(&size);
	errorchk(false, "request buffer stream");

	cout << "alloc size " << size << endl;

	// fill parameters
	adcPar.t1.s_Type = L_ADC_PARAM;
	adcPar.t1.AutoInit = 1;
	adcPar.t1.dRate = 100.0;
	adcPar.t1.dKadr = 0;
	adcPar.t1.dScale = 0;
	adcPar.t1.SynchroType = 0;
	adcPar.t1.SynchroSensitivity = 0;
	adcPar.t1.SynchroMode = 0;
	adcPar.t1.AdChannel = 0;
	adcPar.t1.AdPorog = 0;
	adcPar.t1.NCh = 4;
	adcPar.t1.Chn[0] = 0x0;
	adcPar.t1.Chn[1] = 0x1;
	adcPar.t1.Chn[2] = 0x2;
	adcPar.t1.Chn[3] = 0x3;
	adcPar.t1.FIFO = 4096;
	adcPar.t1.IrqStep = 4096;
	adcPar.t1.Pages = 32;
	adcPar.t1.IrqEna = 1;
	adcPar.t1.AdcEna = 1;

	pI->FillDAQparameters(&adcPar.t1);
	errorchk(false, "fill DAQ parameters");

	pI->SetParametersStream(&adcPar.t1, &size, (void **) &data1, (void **) &sync1, L_STREAM_ADC);
	errorchk(false, "set ADC parameters stream");

	// show slot parameters
	cout << "Base              : " << hex << sl.Base << endl;
	cout << "BaseL             : " << sl.BaseL << endl;
	cout << "Mem               : " << sl.Mem << endl;
	cout << "MemL              : " << sl.MemL << endl;
	cout << "Type              : " << sl.BoardType << endl;
	cout << "DSPType           : " << sl.DSPType << endl;
	cout << "Irq               : " << sl.Irq << endl;
	// show properties
	cout << "SerNum            : " << pd.t7.SerNum << endl;
	cout << "BrdName           : " << pd.t7.BrdName << endl;
	cout << "Rev               : " << pd.t7.Rev << endl;
	cout << "DspType           : " << pd.t7.DspType << endl;
	cout << "IsDacPresent      : " << pd.t7.IsDacPresent << endl;
	cout << "Quartz            : " << dec << pd.t7.Quartz << endl;
	// show ADC parameters
	cout << "Buffer size (word): " << size << endl;
	cout << "Pages             : " << adcPar.t1.Pages << endl;
	cout << "IrqStep           : " << adcPar.t1.IrqStep << endl;
	cout << "FIFO              : " << adcPar.t1.FIFO << endl;
	cout << "Rate              : " << adcPar.t1.dRate << endl;

	IrqStep = adcPar.t1.IrqStep;
	pages = adcPar.t1.Pages;

	ULONG Ver = sync1[0xFF4 >> 2];
	cout << endl << "Current Firmware Version 0x" << hex << Ver << dec << endl;

	cout << endl << "Press any key" << dec << endl;

	readch();

	complete = 0;

	pI->EnableCorrection();
	errorchk(false, "enable correction");

	pI->InitStartLDevice();
	errorchk(false, "init start device");

	pI->StartLDevice();
	errorchk(false, "START device");

	// write to DAC
	ASYNC_PAR pp;
	pp.s_Type = L_ASYNC_DAC_OUT;
	pp.Mode = 0; // 0 channel DAC; if 1 then 1st channel
	pp.Data[0] = 0x001F;
	// the code for DAC
	pI->IoAsync(&pp);

	// read from ADC
	pp.s_Type = L_ASYNC_ADC_INP;
	pp.Chn[0] = 0x00; // channel number
	pI->IoAsync(&pp);
	cout << (short)pp.Data[0] << endl; // code ADC placed into the Data[0]

	errorchk(false, "my TEST");

	cout << endl << "Press any key" << dec << endl;
	readch();

	pI->StopLDevice();
	errorchk(false, "STOP device");

	pI->CloseLDevice();
	errorchk(false, "close device");

	reset_keypress();

	if (dll_handle)
		dlclose(dll_handle);

	cout << endl << "SUCCESS" << endl;

	return 0;
}
