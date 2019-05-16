#ifndef __VXDAPI_IOCTL
#define __VXDAPI_IOCTL 1
// Board Type macro definitions

//#define LABVIEW_FW

#define NONE  0 // no board in slot
#define L1250 1 // L1250 board
#define N1250 2 // N1250 board (may be not work)
#define L1251 3 // L1251 board
#define L1221 4 // L1221 board
#define PCIA  5 // PCI rev A board
#define PCIB  6 // PCI rev B board
#define L264  8 // L264 ISA board
#define L305  9 // L305 ISA board
#define L1450C 10
#define L1450 11
#define L032 12
#define HI8 13
#define PCIC 14

#define LYNX2  15
#define TIGER2 16
#define TIGER3 17
#define LION   18

#define L791     19

#define E440     30
#define E140     31
#define E2010    32
#define E270     33
#define CAN_USB  34
#define AK9      35
#define LTR010   36
#define LTR021   37
#define E154     38
#define E2010B   39
#define LTR031   40
#define LTR030   41



// ERROR CODES
#define L_SUCCESS 0
#define L_NOTSUPPORTED 1
#define L_ERROR 2
#define L_ERROR_NOBOARD 3
#define L_ERROR_INUSE 4


// define s_Type for FillDAQparameters
#define L_ADC_PARAM 1
#define L_DAC_PARAM 2


#define L_ASYNC_ADC_CFG 3
#define L_ASYNC_TTL_CFG 4
#define L_ASYNC_DAC_CFG 5

#define L_ASYNC_ADC_INP 6
#define L_ASYNC_TTL_INP 7

#define L_ASYNC_TTL_OUT 8
#define L_ASYNC_DAC_OUT 9

#define L_STREAM_ADC 1
#define L_STREAM_DAC 2
#define L_STREAM_TTLIN 3
#define L_STREAM_TTLOUT 4

#define L_EVENT_ADC_BUF 1
#define L_EVENT_DAC_BUF 2

#define L_EVENT_ADC_OVF 3
#define L_EVENT_ADC_FIFO 4
#define L_EVENT_DAC_USER 5
#define L_EVENT_DAC_UNF 6
#define L_EVENT_PWR_OVR 7

#pragma pack(1)

// internal
typedef struct _PORT_PARAM_
{
   ULONG port;
   ULONG datatype;
} PORT_PAR, *PPORT_PAR;

// exported
typedef struct __SLOT_PARAM
{
   ULONG Base;
   ULONG BaseL;
   ULONG Base1;
   ULONG BaseL1;
   ULONG Mem;
   ULONG MemL;
   ULONG Mem1;
   ULONG MemL1;
   ULONG Irq;
   ULONG BoardType;
   ULONG DSPType;
   ULONG Dma;
   ULONG DmaDac;
   ULONG      DTA_REG;
   ULONG      IDMA_REG;
   ULONG      CMD_REG;
   ULONG      IRQ_RST;
   ULONG      DTA_ARRAY;
   ULONG      RDY_REG;
   ULONG      CFG_REG;
} SLOT_PAR, *PSLOT_PAR;


typedef struct _DAQ_PARAM_
{
   ULONG s_Type;
   ULONG FIFO;
   ULONG IrqStep;
   ULONG Pages;
} DAQ_PAR, *PDAQ_PAR;


// descr async i/o routines for adc,dac & ttl
typedef struct _ASYNC_PARAM_
#ifndef LABVIEW_FW
 : public DAQ_PAR
#endif
{
   double dRate;
   ULONG Rate;
   ULONG NCh;
   ULONG Chn[128];
   ULONG Data[128];
   ULONG Mode;
} ASYNC_PAR, *PASYNC_PAR;

// same as above but for wlcomp.dll W_/W prefix...C style 
typedef struct W_ASYNC_PARAM_
{
   ULONG s_Type;
   ULONG FIFO;
   ULONG IrqStep;
   ULONG Pages;

   double dRate;
   ULONG Rate;
   ULONG NCh;
   ULONG Chn[128];
   ULONG Data[128];
   ULONG Mode;
} WASYNC_PAR, *PWASYNC_PAR;


typedef struct _DAC_PARAM_U_0
#ifndef LABVIEW_FW
 : public DAQ_PAR
#endif
{
   ULONG AutoInit;

   double dRate;
   ULONG Rate;

   ULONG IrqEna;
   ULONG DacEna;
   ULONG DacNumber;
} DAC_PAR_0, *PDAC_PAR_0;

typedef struct _DAC_PARAM_U_1
#ifndef LABVIEW_FW
 : public DAQ_PAR
#endif
{
   ULONG AutoInit;

   double dRate;
   ULONG Rate;

   ULONG IrqEna;
   ULONG DacEna;
   ULONG Reserved1;
} DAC_PAR_1, *PDAC_PAR_1;

typedef union _DAC_PARAM_U_
{
   DAC_PAR_0 t1;
   DAC_PAR_1 t2;
} DAC_PAR, *PDAC_PAR;

typedef struct W_DAC_PARAM_U_0
{
   ULONG s_Type;
   ULONG FIFO;
   ULONG IrqStep;
   ULONG Pages;

   ULONG AutoInit;

   double dRate;
   ULONG Rate;

   ULONG IrqEna;
   ULONG DacEna;
   ULONG DacNumber;
} WDAC_PAR_0, *PWDAC_PAR_0;

typedef struct W_DAC_PARAM_U_1
{
   ULONG s_Type;
   ULONG FIFO;
   ULONG IrqStep;
   ULONG Pages;

   ULONG AutoInit;

   double dRate;
   ULONG Rate;

   ULONG IrqEna;
   ULONG DacEna;
   ULONG Reserved1;
} WDAC_PAR_1, *PWDAC_PAR_1;



/*
typedef struct _ADC_PARAM_U_ : public DAQ_PAR
{
   USHORT AutoInit;

   double dRate;
   double dKadr;
   double dScale;
   USHORT Rate;
   USHORT Kadr;
   USHORT Scale;
   USHORT FPDelay;

   USHORT SynchroType;
   USHORT SynchroSensitivity;
   USHORT SynchroMode;
   USHORT AdChannel;
   USHORT AdPorog;
   USHORT NCh;
   USHORT Chn[128];
//   USHORT FIFO;
//   USHORT IrqStep;
//   USHORT Pages;
   USHORT IrqEna;
   USHORT AdcEna;
} ADC_PAR, *PADC_PAR;
*/

typedef struct _ADC_PARAM_U_0
#ifndef LABVIEW_FW
 : public DAQ_PAR
#endif
{
   ULONG AutoInit;

   double dRate;
   double dKadr;
   double dScale;
   ULONG Rate;
   ULONG Kadr;
   ULONG Scale;
   ULONG FPDelay;

   ULONG SynchroType;
   ULONG SynchroSensitivity;
   ULONG SynchroMode;
   ULONG AdChannel;
   ULONG AdPorog;

   ULONG NCh;
   ULONG Chn[128];
   ULONG IrqEna;
   ULONG AdcEna;
} ADC_PAR_0, *PADC_PAR_0;



typedef struct _ADC_PARAM_U_1
#ifndef LABVIEW_FW
 : public DAQ_PAR
#endif
{
   ULONG AutoInit;

   double dRate;
   double dKadr;
   USHORT Reserved1;
   USHORT DigRate;
   ULONG DM_Ena;    // data marker ena/dis

   ULONG Rate;
   ULONG Kadr;
   ULONG StartCnt;    // задержка сбора при старте в количестве кадров
   ULONG StopCnt;     // остановка сбора после количества кадров

   ULONG SynchroType;   // in e20-10 start type
   ULONG SynchroMode;    // advanced synchro mode + chan number
   ULONG AdPorog;         // порог синхронизации
   ULONG SynchroSrc;    // in e20-10 clock source
   ULONG AdcIMask;  // cange from Reserved4 to AdcIMask for e20-10 adc input config

   ULONG NCh;
   ULONG Chn[128];
   ULONG IrqEna;
   ULONG AdcEna;
} ADC_PAR_1, *PADC_PAR_1;


typedef union _ADC_PARAM_U_
{
   ADC_PAR_0 t1;
   ADC_PAR_1 t2;
} ADC_PAR, *PADC_PAR;

typedef struct W_ADC_PARAM_U_0
{
   ULONG s_Type;
   ULONG FIFO;
   ULONG IrqStep;
   ULONG Pages;

   ULONG AutoInit;

   double dRate;
   double dKadr;
   double dScale;
   ULONG Rate;
   ULONG Kadr;
   ULONG Scale;
   ULONG FPDelay;

   ULONG SynchroType;
   ULONG SynchroSensitivity;
   ULONG SynchroMode;
   ULONG AdChannel;
   ULONG AdPorog;
   ULONG NCh;
   ULONG Chn[128];
   ULONG IrqEna;
   ULONG AdcEna;
} WADC_PAR_0, *PWADC_PAR_0;



typedef struct W_ADC_PARAM_U_1
{
   ULONG s_Type;
   ULONG FIFO;
   ULONG IrqStep;
   ULONG Pages;

   ULONG AutoInit;

   double dRate;
   double dKadr;
   USHORT Reserved1;
   USHORT DigRate;
   ULONG DM_Ena;    // data marker ena/dis

   ULONG Rate;
   ULONG Kadr;
   ULONG StartCnt;    // задержка сбора при старте в количестве кадров
   ULONG StopCnt;     // остановка сбора после количества кадров

   ULONG SynchroType;
   ULONG SynchroMode;    // advanced synchro mode + chan number
   ULONG AdPorog;         
   ULONG SynchroSrc;
   ULONG AdcIMask;

   ULONG NCh;
   ULONG Chn[128];
   ULONG IrqEna;
   ULONG AdcEna;
} WADC_PAR_1, *PWADC_PAR_1;


typedef struct __USHORT_IMAGE
{
   USHORT data[512];
} USHORT_IMAGE, *PUSHORT_IMAGE;


typedef union W_DAQ_PARAM_
{
   WDAC_PAR_0 t1;
   WDAC_PAR_1 t2;
   WADC_PAR_0 t3;
   WADC_PAR_1 t4;
   USHORT_IMAGE wi;
} WDAQ_PAR, *PWDAQ_PAR;


//exported
typedef struct __PLATA_DESCR
{
   char SerNum[9];
   char BrdName[5];
   char Rev;
   char DspType[5];
   unsigned int Quartz;
   USHORT IsDacPresent;
   USHORT Reserv1[7];
   USHORT KoefADC[8];
   USHORT KoefDAC[4];
   USHORT Custom[32];
} PLATA_DESCR, *PPLATA_DESCR;

//exported
typedef struct __PLATA_DESCR_1450
{
   char SerNum[9];
   char BrdName[7];
   char Rev;
   char DspType[5];
   char IsDacPresent;
   char IsExtMemPresent;
   unsigned int Quartz;
   USHORT Reserv1[6];
   USHORT KoefADC[8];
   USHORT KoefDAC[4];
   USHORT Custom[32];
} PLATA_DESCR_1450, *PPLATA_DESCR_1450;


typedef struct __PLATA_DESCR_L791
{
   USHORT CRC16;
   char SerNum[16];            
   char BrdName[16];          
   char Rev;                  
   char DspType[5];          
   unsigned int Quartz;               
   USHORT IsDacPresent;       
   float KoefADC[16];          
   float KoefDAC[4];         
   USHORT Custom;
} PLATA_DESCR_L791, *PPLATA_DESCR_L791;

typedef struct __PLATA_DESCR_E440
{
   char SerNum[9];
   char BrdName[7];
   char Rev;
   char DspType[5];
   char IsDacPresent;
   unsigned int Quartz;
   char Reserv2[13];
   USHORT KoefADC[8];
   USHORT KoefDAC[4];
   USHORT Custom[32];
} PLATA_DESCR_E440, *PPLATA_DESCR_E440;

typedef struct __PLATA_DESCR_E140
{
   char SerNum[9];
   char BrdName[11];
   char Rev;
   char DspType[11];
   char IsDacPresent;
   unsigned int Quartz;
   char Reserv2[3];
   float KoefADC[8]; // 4 offs 4 scale
   float KoefDAC[4];  // 2 off 2 scale
   USHORT Custom[20];
} PLATA_DESCR_E140, *PPLATA_DESCR_E140;

typedef struct __PACKED_PLATA_DESCR_E140
{
   UCHAR SerNum1; //0-9
   char SerNum2; // L,C
   ULONG SerNum3; // serial long
   char Name[10];
   char Rev;
   char DspType[10];
   ULONG Quartz;
   UCHAR CRC1; // from 0 to 30 (31)
   /////////////
   UCHAR IsDacPresent;
   float AdcOffs[4];
   float AdcScale[4];
   float DacOffs[2];
   float DacScale[2];
   UCHAR Reserv[46];
   UCHAR CRC2; // from 32 to end (95)
} PACKED_PLATA_DESCR_E140, *PPACKED_PLATA_DESCR_E140;

typedef struct __PLATA_DESCR_E154
{
   char SerNum[9];
   char BrdName[11];
   char Rev;
   char DspType[11];
   char IsDacPresent;
   unsigned int Quartz;
   char Reserv2[3];
   float KoefADC[8]; // 4 offs 4 scale
   float KoefDAC[4];  // 2 off 2 scale
   USHORT Custom[20];
} PLATA_DESCR_E154, *PPLATA_DESCR_E154;

typedef struct __PACKED_PLATA_DESCR_E154
{
   UCHAR SerNum1; //0-9
   char SerNum2; // L,C
   ULONG SerNum3; // serial long
   char Name[10];
   char Rev;
   char DspType[10];
   ULONG Quartz;
   UCHAR CRC1; // from 0 to 30 (31)
   /////////////
   UCHAR IsDacPresent;
   float AdcOffs[4];
   float AdcScale[4];
   float DacOffs[2];
   float DacScale[2];
   UCHAR Reserv[46];
   UCHAR CRC2; // from 32 to end (95)
} PACKED_PLATA_DESCR_E154, *PPACKED_PLATA_DESCR_E154;

typedef struct __WORD_IMAGE
{
   USHORT data[64];
} WORD_IMAGE, *PWORD_IMAGE;

typedef struct __BYTE_IMAGE
{
   UCHAR data[128];
} BYTE_IMAGE, *PBYTE_IMAGE;

typedef union __PLATA_DESCR_U
{
   PLATA_DESCR t1;
   PLATA_DESCR_1450 t2;
   PLATA_DESCR_L791 t3;
   PLATA_DESCR_E440 t4;
   PLATA_DESCR_E140 t5;
   PACKED_PLATA_DESCR_E140 pt5;
   
   WORD_IMAGE wi;
   BYTE_IMAGE bi;
} PLATA_DESCR_U, *PPLATA_DESCR_U;

// введены тк у платы 2010 флеш 256 байт и никак его не втиснуть в 128
// соответсвенно объедененный образ увеличен до 256
// size - 256 byte
typedef struct __PLATA_DESCR_E2010
{
    char BrdName[16];
    char SerNum[16];
    char DspType[16];
    ULONG Quartz;
    char Rev;
    char IsDacPresent;
    float KoefADC[24]; // 12 offs 12 scale
    float KoefDAC[4];  // 2 off 2 scale
    USHORT Custom[44];
    USHORT CRC;
} PLATA_DESCR_E2010, *PPLATA_DESCR_E2010;

typedef struct __WORD_IMAGE_256
{
   USHORT data[128];
} WORD_IMAGE_256, *PWORD_IMAGE_256;

typedef struct __BYTE_IMAGE_256
{
   UCHAR data[256];
} BYTE_IMAGE_256, *PBYTE_IMAGE_256;

typedef union __PLATA_DESCR_U2
{
   PLATA_DESCR t1;
   PLATA_DESCR_1450 t2;
   PLATA_DESCR_L791 t3;
   PLATA_DESCR_E440 t4;
   PLATA_DESCR_E140 t5;
   PACKED_PLATA_DESCR_E140 pt5;
   PLATA_DESCR_E2010 t6;
   PLATA_DESCR_E154 t7;
   PACKED_PLATA_DESCR_E154 pt7;

   WORD_IMAGE wi;
   BYTE_IMAGE bi;
   WORD_IMAGE_256 wi256;
   BYTE_IMAGE_256 bi256;    
} PLATA_DESCR_U2, *PPLATA_DESCR_U2;


//  used internaly in driver for e140 //////////////////
#define MAKE_E140CHAN(w) ((w&0x3)<<2)|((w&0xC)>>2)|(w&0xF0)

typedef struct _ADC_PARAM_E140_PACK                   //
{                                                     //
   UCHAR  Chn[128];                                   //
                                                      //
   USHORT Rate;                                       //
   UCHAR NCh;                                         //
   UCHAR Kadr;                                        //
                                                      //
   UCHAR SynchroType;                                 //
   UCHAR AdChannel;                                   //
   USHORT AdPorog;                                    //
} ADC_PAR_E140_PACK, *PADC_PAR_E140_PACK;             //
                                                      //
typedef struct __BYTE_IMAGE_E140                      //
{                                                     //
   UCHAR data[136];                                   //
} BYTE_IMAGE_E140, *PBYTE_IMAGE_E140;                 //
                                                      //
typedef union __ADC_PARAM_E140_PACK                   //
{                                                     //
   ADC_PAR_E140_PACK t1;                              //
   BYTE_IMAGE_E140 bi;                                     //
} ADC_PAR_E140_PACK_U, *PADC_PAR_E140_PACK_U;         //
                                                      //
////////////////////////////////////////////////////////


//  used internaly in driver for e154 //////////////////
typedef struct _ADC_PARAM_E154_PACK                   //
{                                                     //
   UCHAR  Chn[16];                                    //
                                                      //
   USHORT Rate;                                       //
   UCHAR NCh;                                         //
   UCHAR Kadr;                                        //
                                                      //
   UCHAR  SynchroType;                                //
   UCHAR  AdChannel;                                  //
   USHORT AdPorog;                                    //
   UCHAR  Scale;                                      //
   UCHAR  Kadr1;                                      //
} ADC_PAR_E154_PACK, *PADC_PAR_E154_PACK;             //
                                                      //
typedef struct __BYTE_IMAGE_E154                      //
{                                                     //
   UCHAR data[26];                                   //
} BYTE_IMAGE_E154, *PBYTE_IMAGE_E154;                 //
                                                      //
typedef union __ADC_PARAM_E154_PACK                   //
{                                                     //
   ADC_PAR_E154_PACK t1;                              //
   BYTE_IMAGE_E154 bi;                                     //
} ADC_PAR_E154_PACK_U, *PADC_PAR_E154_PACK_U;         //
                                                      //
////////////////////////////////////////////////////////


//  used internaly in driver for e2010 ///////////////////
typedef struct _ADC_PARAM_E2010_PACK                    //
{                                                       //
   UCHAR    SyncMode;                                   //
   UCHAR    Rate;                                       //
   USHORT   Kadr;                                       //
   USHORT   ChanMode;                                   //
   UCHAR    NCh;                                        //
   UCHAR    Chn[256];                                   //
} ADC_PAR_E2010_PACK, *PADC_PAR_E2010_PACK;             //
                                                        //
typedef struct __BYTE_IMAGE_E2010                       //
{                                                       //
   UCHAR data[256+7];                                   //
} BYTE_IMAGE_E2010, *PBYTE_IMAGE_E2010;                 //
                                                        //
typedef union __ADC_PARAM_E2010_PACK                    //
{                                                       //
   ADC_PAR_E2010_PACK t1;                               //
   BYTE_IMAGE_E2010 bi;                                 //
} ADC_PAR_E2010_PACK_U, *PADC_PAR_E2010_PACK_U;         //
                                                        //
typedef struct _ADC_PAR_EXTRA_E2010_PACK                //
{                                                       //
   ULONG  StartCnt;                                     //
   ULONG  StopCnt;                                      //
   USHORT SynchroMode;                                  //
   USHORT AdPorog;                                      //
   UCHAR  DM_Ena;                                       //
} ADC_PAR_EXTRA_E2010_PACK, *PADC_PAR_EXTRA_E2010_PACK; //
                                                        //
typedef struct __BYTE_IMAGE_E2010_13                    //
{                                                       //
   UCHAR data[13];                                      //
} BYTE_IMAGE_E2010_13, *PBYTE_IMAGE_E2010_13;           //
                                                        //
typedef union __ADC_PAR_EXTRA_E2010_PACK                //
{                                                       //
   ADC_PAR_EXTRA_E2010_PACK t1;                         //
   BYTE_IMAGE_E2010_13 bi;                              //
} ADC_PAR_EXTRA_E2010_PACK_U, *PADC_PAR_EXTRA_E2010_PACK_U; //
                                                        //
////////////////////////////////////////////////////////

// ioctl struct for ioctl access...
typedef struct __IOCTL_BUFFER
{
   int inSize; // size in bytes
   int outSize; // size in bytes 
   unsigned char inBuffer[4096];
   unsigned char outBuffer[4096];   
} IOCTL_BUFFER, *PIOCTL_BUFFER;

#pragma pack()



#ifdef LCOMP_LINUX

#define DIOC_SETUP                _IOWR(0x97,1,IOCTL_BUFFER)

/*
#define DIOC_SETEVENT \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 2, METHOD_IN_DIRECT, FILE_ANY_ACCESS)
*/

#define DIOC_START                _IOWR(0x97,3,IOCTL_BUFFER)


#define DIOC_STOP                 _IOWR(0x97,4,IOCTL_BUFFER)


#define DIOC_OUTP                 _IOWR(0x97,5,IOCTL_BUFFER)

#define DIOC_INP                  _IOWR(0x97,6,IOCTL_BUFFER)

#define DIOC_OUTM                 _IOWR(0x97,7,IOCTL_BUFFER)

#define DIOC_INM                  _IOWR(0x97,8,IOCTL_BUFFER)

#define DIOC_SETBUFFER            _IOWR(0x97,9,IOCTL_BUFFER)

/*

#define DIOC_ADD_BOARDS \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 10, METHOD_OUT_DIRECT, FILE_ANY_ACCESS)

#define DIOC_CLEAR_BOARDS \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 11, METHOD_OUT_DIRECT, FILE_ANY_ACCESS)
*/    

#define DIOC_INIT_SYNC            _IOWR(0x97, 12,IOCTL_BUFFER)

/*
//
//#define DIOC_SETBUFFER_DAC \
//      CTL_CODE(FILE_DEVICE_UNKNOWN, 13, METHOD_OUT_DIRECT, FILE_ANY_ACCESS)
//

#define DIOC_SETEVENT_DAC \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 14, METHOD_OUT_DIRECT, FILE_ANY_ACCESS)

#define DIOC_SEND_COMMAND \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 15, METHOD_OUT_DIRECT, FILE_ANY_ACCESS)
*/

#define DIOC_SEND_COMMAND        _IOWR(0x97,15,IOCTL_BUFFER)

#define DIOC_COMMAND_PLX         _IOWR(0x97,16,IOCTL_BUFFER)


/*
#define DIOC_PUT_DATA_A \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 17, METHOD_IN_DIRECT, FILE_ANY_ACCESS)

#define DIOC_GET_DATA_A \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 18, METHOD_OUT_DIRECT, FILE_ANY_ACCESS)
*/

#define DIOC_PUT_DM_A           _IOWR(0x97,19,IOCTL_BUFFER)

#define DIOC_GET_DM_A           _IOWR(0x97,20,IOCTL_BUFFER)

#define DIOC_PUT_PM_A           _IOWR(0x97,21,IOCTL_BUFFER)

#define DIOC_GET_PM_A           _IOWR(0x97,22,IOCTL_BUFFER)

#define DIOC_GET_PARAMS         _IOWR(0x97, 23, IOCTL_BUFFER)

#define DIOC_SET_DSP_TYPE       _IOWR(0x97, 24, IOCTL_BUFFER)

#define DIOC_SETBUFFER_1        _IOWR(0x97, 25, IOCTL_BUFFER)


#define DIOC_SETUP_DAC          _IOWR(0x97, 26, IOCTL_BUFFER)

#define DIOC_READ_FLASH_WORD    _IOWR(0x97, 27, IOCTL_BUFFER)

#define DIOC_WRITE_FLASH_WORD   _IOWR(0x97, 28, IOCTL_BUFFER)

#define DIOC_ENABLE_FLASH_WRITE _IOWR(0x97, 29, IOCTL_BUFFER)

/*
#define DIOC_SETEVENT_1 \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 30, METHOD_IN_DIRECT, FILE_ANY_ACCESS)

#define DIOC_SETEVENT_2 \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 31, METHOD_IN_DIRECT, FILE_ANY_ACCESS)

#define DIOC_SETEVENT_3 \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 32, METHOD_IN_DIRECT, FILE_ANY_ACCESS)

#define DIOC_SETEVENT_4 \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 33, METHOD_IN_DIRECT, FILE_ANY_ACCESS)

#define DIOC_SETEVENT_5 \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 34, METHOD_IN_DIRECT, FILE_ANY_ACCESS)
*/

#define DIOC_ADCSAMPLE              _IOWR(0x97, 35, IOCTL_BUFFER)

#define DIOC_LOAD_BIOS              _IOWR(0x97, 36, IOCTL_BUFFER)

#define DIOC_TTL_IN                 _IOWR(0x97, 37, IOCTL_BUFFER)
#define DIOC_TTL_OUT                _IOWR(0x97, 38, IOCTL_BUFFER)
#define DIOC_TTL_CFG                _IOWR(0x97, 39, IOCTL_BUFFER)
#define DIOC_DAC_OUT                _IOWR(0x97, 40, IOCTL_BUFFER)


#define DIOC_RESET_PLX              _IOWR(0x97, 41, IOCTL_BUFFER)

#define DIOC_WAIT_COMPLETE          _IOWR(0x97, 42, IOCTL_BUFFER)
#define DIOC_WAIT_COMPLETE_DAC      _IOWR(0x97, 43, IOCTL_BUFFER)

#define DIOC_SEND_BIOS              _IOWR(0x97, 44, IOCTL_BUFFER)

#define DIOC_WAIT_COMPLETE_ADC_OVF   _IOWR(0x97, 45, IOCTL_BUFFER)
#define DIOC_WAIT_COMPLETE_ADC_BUF   _IOWR(0x97, 46, IOCTL_BUFFER)
#define DIOC_WAIT_COMPLETE_DAC_UNF   _IOWR(0x97, 47, IOCTL_BUFFER)
#define DIOC_WAIT_COMPLETE_PWR       _IOWR(0x97, 48, IOCTL_BUFFER)
#define DIOC_ENABLE_CORRECTION       _IOWR(0x97, 50, IOCTL_BUFFER) 

#else
#define DIOC_SETUP \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 1, METHOD_OUT_DIRECT, FILE_ANY_ACCESS)

#define DIOC_SETEVENT \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 2, METHOD_IN_DIRECT, FILE_ANY_ACCESS)
        
#define DIOC_START \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 3, METHOD_OUT_DIRECT, FILE_ANY_ACCESS)

#define DIOC_STOP \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 4, METHOD_OUT_DIRECT, FILE_ANY_ACCESS)

#define DIOC_OUTP \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 5, METHOD_IN_DIRECT, FILE_ANY_ACCESS)

#define DIOC_INP \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 6, METHOD_OUT_DIRECT, FILE_ANY_ACCESS)

#define DIOC_OUTM \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 7, METHOD_IN_DIRECT, FILE_ANY_ACCESS)

#define DIOC_INM \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 8, METHOD_OUT_DIRECT, FILE_ANY_ACCESS)

#define DIOC_SETBUFFER \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 9, METHOD_OUT_DIRECT, FILE_ANY_ACCESS)

#define DIOC_ADD_BOARDS \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 10, METHOD_OUT_DIRECT, FILE_ANY_ACCESS)

#define DIOC_CLEAR_BOARDS \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 11, METHOD_OUT_DIRECT, FILE_ANY_ACCESS)

#define DIOC_INIT_SYNC \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 12, METHOD_OUT_DIRECT, FILE_ANY_ACCESS)
/*
#define DIOC_SETBUFFER_DAC \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 13, METHOD_OUT_DIRECT, FILE_ANY_ACCESS)
*/

#define DIOC_SETEVENT_DAC \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 14, METHOD_OUT_DIRECT, FILE_ANY_ACCESS)

#define DIOC_SEND_COMMAND \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 15, METHOD_OUT_DIRECT, FILE_ANY_ACCESS)

#define DIOC_COMMAND_PLX \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 16, METHOD_OUT_DIRECT, FILE_ANY_ACCESS)

#define DIOC_PUT_DATA_A \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 17, METHOD_IN_DIRECT, FILE_ANY_ACCESS)

#define DIOC_GET_DATA_A \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 18, METHOD_OUT_DIRECT, FILE_ANY_ACCESS)

#define DIOC_PUT_DM_A \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 19, METHOD_IN_DIRECT, FILE_ANY_ACCESS)

#define DIOC_GET_DM_A \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 20, METHOD_OUT_DIRECT, FILE_ANY_ACCESS)

#define DIOC_PUT_PM_A \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 21, METHOD_IN_DIRECT, FILE_ANY_ACCESS)

#define DIOC_GET_PM_A \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 22, METHOD_OUT_DIRECT, FILE_ANY_ACCESS)

#define DIOC_GET_PARAMS \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 23, METHOD_OUT_DIRECT, FILE_ANY_ACCESS)

#define DIOC_SET_DSP_TYPE \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 24, METHOD_OUT_DIRECT, FILE_ANY_ACCESS)

#define DIOC_SETBUFFER_1 \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 25, METHOD_OUT_DIRECT, FILE_ANY_ACCESS)

#define DIOC_SETUP_DAC \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 26, METHOD_OUT_DIRECT, FILE_ANY_ACCESS)

#define DIOC_READ_FLASH_WORD \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 27, METHOD_OUT_DIRECT, FILE_ANY_ACCESS)

#define DIOC_WRITE_FLASH_WORD \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 28, METHOD_IN_DIRECT, FILE_ANY_ACCESS)

#define DIOC_ENABLE_FLASH_WRITE \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 29, METHOD_IN_DIRECT, FILE_ANY_ACCESS)

#define DIOC_SETEVENT_1 \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 30, METHOD_IN_DIRECT, FILE_ANY_ACCESS)

#define DIOC_SETEVENT_2 \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 31, METHOD_IN_DIRECT, FILE_ANY_ACCESS)

#define DIOC_SETEVENT_3 \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 32, METHOD_IN_DIRECT, FILE_ANY_ACCESS)

#define DIOC_SETEVENT_4 \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 33, METHOD_IN_DIRECT, FILE_ANY_ACCESS)

#define DIOC_SETEVENT_5 \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 34, METHOD_IN_DIRECT, FILE_ANY_ACCESS)

#define DIOC_ADCSAMPLE \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 35, METHOD_OUT_DIRECT, FILE_ANY_ACCESS)

#define DIOC_LOAD_BIOS \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 36, METHOD_IN_DIRECT, FILE_ANY_ACCESS)

#define DIOC_TTL_IN \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 37, METHOD_OUT_DIRECT, FILE_ANY_ACCESS)

#define DIOC_TTL_OUT \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 38, METHOD_IN_DIRECT, FILE_ANY_ACCESS)

#define DIOC_TTL_CFG \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 39, METHOD_IN_DIRECT, FILE_ANY_ACCESS)

#define DIOC_DAC_OUT \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 40, METHOD_IN_DIRECT, FILE_ANY_ACCESS)

#define DIOC_RESET_PLX \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 41, METHOD_IN_DIRECT, FILE_ANY_ACCESS)

#define DIOC_SEND_BIOS \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 44, METHOD_IN_DIRECT, FILE_ANY_ACCESS)

#define DIOC_ENABLE_CORRECTION \
      CTL_CODE(FILE_DEVICE_UNKNOWN, 50, METHOD_OUT_DIRECT, FILE_ANY_ACCESS)


// api from ldevusb
/*
#define DIOC_START \
   CTL_CODE (FILE_DEVICE_UNKNOWN,0x101,METHOD_OUT_DIRECT,FILE_ANY_ACCESS)

#define DIOC_STOP \
   CTL_CODE (FILE_DEVICE_UNKNOWN,0x102,METHOD_OUT_DIRECT,FILE_ANY_ACCESS)

#define DIOC_LOAD_BIOS_USB \
   CTL_CODE (FILE_DEVICE_UNKNOWN,0x103,METHOD_OUT_DIRECT,FILE_ANY_ACCESS)

#define DIOC_SEND_COMMAND \
   CTL_CODE (FILE_DEVICE_UNKNOWN,0x104,METHOD_OUT_DIRECT,FILE_ANY_ACCESS)
*/

#define DIOC_RESET_PIPE1 \
   CTL_CODE (FILE_DEVICE_UNKNOWN,0x105,METHOD_OUT_DIRECT,FILE_ANY_ACCESS)

#define DIOC_RESET_PIPE3 \
   CTL_CODE (FILE_DEVICE_UNKNOWN,0x106,METHOD_OUT_DIRECT,FILE_ANY_ACCESS)

#define DIOC_ABORT_PIPE1 \
   CTL_CODE (FILE_DEVICE_UNKNOWN,0x107,METHOD_OUT_DIRECT,FILE_ANY_ACCESS)

#define DIOC_ABORT_PIPE3 \
   CTL_CODE (FILE_DEVICE_UNKNOWN,0x108,METHOD_OUT_DIRECT,FILE_ANY_ACCESS)


#endif

#endif
// следующий 51
