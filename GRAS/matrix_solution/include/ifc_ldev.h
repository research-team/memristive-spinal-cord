#ifndef _IFC_LDEV_H
#define _IFC_LDEV_H

#define IFC(type) virtual type __stdcall

#define FDF(type) type __stdcall

#ifndef __LUNKNOWN__
#define __LUNKNOWN__
struct LUnknown
{
   IFC(HRESULT)   QueryInterface(const IID& iid, void** ppv) = 0;
   IFC(ULONG)     AddRef() = 0;
   IFC(ULONG)     Release() = 0;
};
#endif

struct IDaqLDevice:LUnknown
{
   IFC(ULONG)  inbyte ( ULONG offset, PUCHAR data, ULONG len=1, ULONG key=0) = 0;
   IFC(ULONG)  inword ( ULONG offset, PUSHORT data, ULONG len=2, ULONG key=0) = 0;
   IFC(ULONG)  indword( ULONG offset, PULONG data, ULONG len=4, ULONG key=0) = 0;

   IFC(ULONG)  outbyte ( ULONG offset, PUCHAR data, ULONG len=1, ULONG key=0) = 0;
   IFC(ULONG)  outword ( ULONG offset, PUSHORT data, ULONG len=2, ULONG key=0) = 0;
   IFC(ULONG)  outdword( ULONG offset, PULONG data, ULONG len=4, ULONG key=0) = 0;

   // Working with MEM ports
   IFC(ULONG)  inmbyte ( ULONG offset, PUCHAR data, ULONG len=1, ULONG key=0) = 0;
   IFC(ULONG)  inmword ( ULONG offset, PUSHORT data, ULONG len=2, ULONG key=0) = 0;
   IFC(ULONG)  inmdword( ULONG offset, PULONG data, ULONG len=4, ULONG key=0) = 0;

   IFC(ULONG)  outmbyte ( ULONG offset, PUCHAR data, ULONG len=1, ULONG key=0) = 0;
   IFC(ULONG)  outmword ( ULONG offset, PUSHORT data, ULONG len=2, ULONG key=0) = 0;
   IFC(ULONG)  outmdword( ULONG offset, PULONG data, ULONG len=4, ULONG key=0) = 0;

   IFC(ULONG)  GetWord_DM(USHORT Addr, PUSHORT Data) = 0;
   IFC(ULONG)  PutWord_DM(USHORT Addr, USHORT Data) = 0;
   IFC(ULONG)  PutWord_PM(USHORT Addr, ULONG Data) = 0;
   IFC(ULONG)  GetWord_PM(USHORT Addr, PULONG Data) = 0;

   IFC(ULONG)  GetArray_DM(USHORT Addr, ULONG Count, PUSHORT Data) = 0;
   IFC(ULONG)  PutArray_DM(USHORT Addr, ULONG Count, PUSHORT Data) = 0;
   IFC(ULONG)  PutArray_PM(USHORT Addr, ULONG Count, PULONG Data) = 0;
   IFC(ULONG)  GetArray_PM(USHORT Addr, ULONG Count, PULONG Data) = 0;

   IFC(ULONG)  SendCommand(USHORT Cmd) = 0;

   IFC(ULONG)  PlataTest() = 0;

   IFC(ULONG)  GetSlotParam(PSLOT_PAR slPar) = 0;

   IFC(HANDLE) OpenLDevice() = 0;
   IFC(ULONG)  CloseLDevice() = 0;

///
   IFC(ULONG)  SetParametersStream(PDAQ_PAR sp, ULONG *UsedSize, void** Data, void** Sync, ULONG StreamId = L_STREAM_ADC) = 0;
   IFC(ULONG)  RequestBufferStream(ULONG *Size, ULONG StreamId = L_STREAM_ADC) = 0; //in words
   IFC(ULONG)  FillDAQparameters(PDAQ_PAR sp) = 0;  
///

   IFC(ULONG)  InitStartLDevice() = 0;
   IFC(ULONG)  StartLDevice() = 0;
   IFC(ULONG)  StopLDevice() = 0;

   IFC(ULONG)  LoadBios(char *FileName) = 0;
/*
   IFC(ULONG)  InputADC(USHORT Chan, PUSHORT Data) = 0;
   IFC(ULONG)  InputTTL(PULONG Data, ULONG Mode) = 0;
   IFC(ULONG)  OutputTTL(ULONG Data, ULONG Mode) = 0;
   IFC(ULONG)  ConfigTTL(ULONG Data) = 0;
   IFC(ULONG)  OutputDAC(short Data, ULONG Mode) = 0;
   IFC(ULONG)  ConfigDAC(ULONG Mode, ULONG Number) = 0;
*/

   IFC(ULONG)  IoAsync(PDAQ_PAR sp) =0;  // collect all async io operations

   IFC(ULONG)  ReadPlataDescr(LPVOID pd) = 0;
   IFC(ULONG)  WritePlataDescr(LPVOID pd, USHORT Ena) = 0;
   IFC(ULONG)  ReadFlashWord(USHORT FlashAddress, PUSHORT Data) = 0;
   IFC(ULONG)  WriteFlashWord(USHORT FlashAddress, USHORT Data) = 0;
   IFC(ULONG)  EnableFlashWrite(USHORT Flag) = 0;

   IFC(ULONG)  EnableCorrection(USHORT Ena=1) = 0;

   IFC(ULONG)  GetParameter(ULONG name, PULONG param) = 0;
   IFC(ULONG)  SetParameter(ULONG name, PULONG param) = 0;

   IFC(ULONG)  SetLDeviceEvent(HANDLE hEvent,ULONG EventId = L_STREAM_ADC) = 0;
};

#ifdef LCOMP_LINUX
DEFINE_GUID(IID_IUnknown, 0x00000000, 0x0000,0x0000, 0xC0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46);
#endif

DEFINE_GUID(IID_ILDEV, 0x32bb8320, 0xb41b,0x11cf, 0xa6, 0xbb, 0x00, 0x80, 0xc7, 0xb2, 0xd6, 0x82);

#endif
