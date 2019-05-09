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

void reset_keypress(void) {
	tcsetattr(0, TCSANOW, &stored_settings);
	return;
}

void set_keypress(void) {
	tcgetattr(0, &stored_settings);

	new_settings = stored_settings;

	// disable canonical mode, and set buffer size to 1 byte
	new_settings.c_lflag &= (~ICANON);
	new_settings.c_lflag &= ~ECHO;
	new_settings.c_lflag &= ~ISIG;
	new_settings.c_cc[VTIME] = 0;
	new_settings.c_cc[VMIN] = 1;

	atexit(reset_keypress);
	tcsetattr(0, TCSANOW, &new_settings);
	return;
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

int main() {
	ULONG mem_buffer_size = 131072;
	SLOT_PAR slot_param;
	ASYNC_PAR async_par;
	ADC_PAR adc_par;
	DAC_PAR dacPar;
	IDaqLDevice *device;
	PLATA_DESCR_U2 plata_descr;
	HANDLE handle;
	PVOID dll_handle;
	ULONG iresult;
	HRESULT hresult;
	LUnknown *pIUnknown;

	set_keypress();

	// load the dynamic shared object (shared library). RTLD_LAZY - perform lazy binding
	dll_handle = dlopen("/home/alex/Programs/drivers/lcomp/liblcomp.so", RTLD_LAZY);
	errorchk(!dll_handle, "open DLL", dlerror());

	// return the address where that symbol is loaded into memory
	CreateInstance = (CREATEFUNCPTR) dlsym(dll_handle, "CreateInstance");
	errorchk(dlerror() != NULL, "create instance");

	// create an object which related with a specific virtual slot (default 0)
	pIUnknown = CreateInstance(0);
	errorchk(pIUnknown == NULL, "call create instance");

	// get a pointer to the interface
	hresult = pIUnknown->QueryInterface(IID_ILDEV, (void **) &device);
	errorchk(hresult != S_OK, hresult, "query interface");

	// close an interface
	pIUnknown->Release();
	errorchk(false, "free IUnknown");

	// open an appropriate link of the board driver
	handle = device->OpenLDevice();
	errorchk(false, handle, "open device");

	// get an information of the specific virtual slot
	device->GetSlotParam(&slot_param);
	errorchk(false, "get slot parameters");

	// load a BIOS to the board
	iresult = device->LoadBios("e154");
	errorchk(iresult == 0, "load BIOS");

	// Test for board availability (always success)
	iresult = device->PlataTest();
	errorchk(iresult != 0, "plata test");

	// read an user Flash
	device->ReadPlataDescr(&plata_descr);
	errorchk(false, "read the board description");

	// allocate memory for a big ring buffer
	device->RequestBufferStream(&mem_buffer_size);
	errorchk(false, "request buffer stream");

	cout << "allocated size " << mem_buffer_size << endl;

	// fill ADC parameters
	adc_par.t1.s_Type = L_ADC_PARAM;  // тип структуры
	adc_par.t1.AutoInit = 1;          // флаг указывающий на тип сбора данных 0 - однократный 1 -циклически
	adc_par.t1.dRate = 100.0;         // частота опроса каналов в кадре (кГц)
	adc_par.t1.dKadr = 0;             // интервал между кадрами (мс), фактически определяет скоростьсбора данных;
	adc_par.t1.dScale = 0;            // масштаб работы таймера для 1250 или делителя для 1221
	adc_par.t1.AdChannel = 0;         // номер канала, выбранный для аналоговой синхронизации
	adc_par.t1.AdPorog = 0;           // пороговое значение для аналоговой синхронизации в коде АЦП
	adc_par.t1.NCh = 4;               // количество опрашиваемых в кадре каналов (для E154 макс. 16)
	adc_par.t1.Chn[0] = 0x0;          // массив с номерами каналов и усилением на них,
	adc_par.t1.Chn[1] = 0x1;          // описывает порядок опроса каналов
	adc_par.t1.Chn[2] = 0x2;
	adc_par.t1.Chn[3] = 0x3;

	adc_par.t1.FIFO = 4096;           // размер половины аппаратного буфера FIFO на плате
	adc_par.t1.IrqStep = 4096;        // шаг генерации прерываний
	adc_par.t1.Pages = 32;            // размер кольцевого буфера в шагах прерываний
	adc_par.t1.IrqEna = 1;            // разрешение генерации прерывания от платы (1/0);
	adc_par.t1.AdcEna = 1;            // разрешение работы AЦП (1/0)

	// 0 - нет синхронизации
	// 1 - цифровая синхронизация старта, остальные параметры синхронизации не используются
	// 2 - покадровая синхронизация, остальные параметры синхронизации не используются
	// 3 - аналоговая синхронизация старта по выбранному каналу АЦП
	adc_par.t1.SynchroType = 0;

	// 0 - аналоговая синхронизация по уровню
	// 1 - аналоговая синхронизация по переходу
	adc_par.t1.SynchroSensitivity = 0;

	// 0 - по уровню «выше» или переходу «снизу-вверх»
	// 1 - по уровню «ниже» или переходу «сверху-вниз»
	adc_par.t1.SynchroMode = 0;

	// fill inner parameter's structure of data collection values ​​from the structure ADC_PAR, DAC_PAR
	device->FillDAQparameters(&adc_par.t1);
	errorchk(false, "fill DAQ parameters");

	// setup the ADC/DAC board setting based on specific i/o parameters
	device->SetParametersStream(&adc_par.t1, &mem_buffer_size, (void **) &data1, (void **) &sync1, L_STREAM_ADC);
	errorchk(false, "set ADC parameters stream");

	// show slot parameters
	cout << "Base              : " << hex << slot_param.Base << endl;
	cout << "BaseL             : " << slot_param.BaseL << endl;
	cout << "Mem               : " << slot_param.Mem << endl;
	cout << "MemL              : " << slot_param.MemL << endl;
	cout << "Type              : " << slot_param.BoardType << endl;
	cout << "DSPType           : " << slot_param.DSPType << endl;
	cout << "Irq               : " << slot_param.Irq << endl;

	// show properties
	cout << "SerNum            : " << plata_descr.t7.SerNum << endl;
	cout << "BrdName           : " << plata_descr.t7.BrdName << endl;
	cout << "Rev               : " << plata_descr.t7.Rev << endl;
	cout << "DspType           : " << plata_descr.t7.DspType << endl;
	cout << "IsDacPresent      : " << plata_descr.t7.IsDacPresent << endl;
	cout << "Quartz            : " << dec << plata_descr.t7.Quartz << endl;

	// show ADC parameters
	cout << "Buffer size (word): " << mem_buffer_size << endl;
	cout << "Pages             : " << adc_par.t1.Pages << endl;
	cout << "IrqStep           : " << adc_par.t1.IrqStep << endl;
	cout << "FIFO              : " << adc_par.t1.FIFO << endl;
	cout << "Rate              : " << adc_par.t1.dRate << endl;

	// get an firmware version
	ULONG version = sync1[0xFF4 >> 2];
	cout << endl << "current firmware version 0x" << hex << version << dec << endl;

	cout << endl << "Press any key to start" << dec << endl;
	readch();

	// turn on an correction mode. Itself loads coefficients to the board
	device->EnableCorrection();
	errorchk(false, "enable correction mode");

	// init inner variables of the driver before starting collect a data
	device->InitStartLDevice();
	errorchk(false, "init start device");

	// start collect data from the board into the big ring buffer
	device->StartLDevice();
	errorchk(false, "START device");

	// write data to the DAC
	async_par.s_Type = L_ASYNC_DAC_OUT;
	async_par.Mode = 0; // номер ЦАП (0/1). задает различные режимы при конфигурации.
	async_par.Data[0] = 0x001F; // данные для ЦАП (массив для данных)
	device->IoAsync(&async_par);  // asyncronous i/o operations

	// read data from the ADC
	async_par.s_Type = L_ASYNC_ADC_INP;
	async_par.Chn[0] = 0x00;  // logic number of the channel
	device->IoAsync(&async_par);
	cout << (short)async_par.Data[0] << endl;  // ADC data

	errorchk(false, "TEST");

	// stop collecting data from the board
	device->StopLDevice();
	errorchk(false, "STOP device");

	// finishing work with the board
	device->CloseLDevice();
	errorchk(false, "close device");

	reset_keypress();

	// close an dll handle (decrements the reference count on the dynamic library handle)
	if (dll_handle)
		dlclose(dll_handle);

	// you are awesome
	cout << endl << "SUCCESS" << endl;

	return 0;
}
