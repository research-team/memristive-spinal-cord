class DaqE154: public LDaqBoard
{
public:
   // Base functions
   IFC(ULONG) GetWord_DM(USHORT Addr, PUSHORT Data);
   IFC(ULONG) PutWord_DM(USHORT Addr, USHORT Data);
   IFC(ULONG) PutWord_PM(USHORT Addr, ULONG Data); // Р±Р°Р№С‚РѕРІС‹Рµ РѕРїСЂРµР°С†РёРё
   IFC(ULONG) GetWord_PM(USHORT Addr, PULONG Data); // Р±Р°Р№С‚РѕРІС‹Рµ РѕРїРµСЂР°С†РёРё

   IFC(ULONG) GetArray_DM(USHORT Addr, ULONG Count, PUSHORT Data);
   IFC(ULONG) PutArray_DM(USHORT Addr, ULONG Count, PUSHORT Data);
   //IFC(ULONG) PutArray_PM(USHORT Addr, DWORD Count, PULONG Data);
   //IFC(ULONG) GetArray_PM(USHORT Addr, DWORD Count, PULONG Data);

   IFC(ULONG) SendCommand(USHORT cmd);
      
   // Service functions
   IFC(ULONG) PlataTest();


   //IFC(ULONG) EnableCorrection(USHORT Ena);
      
//   IFC(ULONG)  LoadBios(char *FileName);


//   IFC(ULONG)  ReadFlashWord(USHORT Addr, PUSHORT Data);
//   IFC(ULONG)  WriteFlashWord(USHORT FlashAddress, USHORT Data);
   IFC(ULONG)  ReadPlataDescr(LPVOID pd);
   IFC(ULONG)  WritePlataDescr(LPVOID pd, USHORT Ena);
   IFC(ULONG)  EnableFlashWrite(USHORT Flag);   

public:
   DaqE154(ULONG Slot) :LDaqBoard(Slot) {}
   ULONG  FillADCparameters(PDAQ_PAR sp);
//   ULONG  FillDACparameters(PDAQ_PAR sp);

   ULONG InputTTL(PDAQ_PAR sp);  //2 in 1 all
   ULONG OutputTTL(PDAQ_PAR sp);  // in each set channel
   ULONG ConfigTTL(PDAQ_PAR sp); // 1221 and 1450 780C e400

   ULONG OutputDAC(PDAQ_PAR sp); //2 in 1

   ULONG InputADC(PDAQ_PAR sp);

   inline UCHAR CRC8CALC(UCHAR *Buffer, USHORT Size)
   {
      UCHAR crc = 0x55;
      for(USHORT i=0;i<Size;i++) crc = crc + Buffer[i];
      return crc;
   }

   ULONG PackModuleDescriptor(PPLATA_DESCR_U2 ppd);
   ULONG UnPackModuleDescriptor(PPLATA_DESCR_U2 ppd);

protected:
};
