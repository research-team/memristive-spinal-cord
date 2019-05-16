class LDaqBoard: public IDaqLDevice
{
public:
      virtual HRESULT __stdcall QueryInterface(const IID& iid, void** ppv);
      virtual ULONG __stdcall AddRef();
      virtual ULONG __stdcall Release();

      IFC(ULONG) inbyte( ULONG offset, PUCHAR data, ULONG len=1, ULONG key=0);
      IFC(ULONG) inword ( ULONG offset, PUSHORT data, ULONG len=2, ULONG key=0);
      IFC(ULONG) indword( ULONG offset, PULONG data, ULONG len=4, ULONG key=0);

      IFC(ULONG) outbyte ( ULONG offset, PUCHAR data, ULONG len=1, ULONG key=0);
      IFC(ULONG) outword ( ULONG offset, PUSHORT data, ULONG len=2, ULONG key=0);
      IFC(ULONG) outdword( ULONG offset, PULONG data, ULONG len=4, ULONG key=0);

      IFC(ULONG) inmbyte ( ULONG offset, PUCHAR data, ULONG len=1, ULONG key=0);
      IFC(ULONG) inmword ( ULONG offset, PUSHORT data, ULONG len=2, ULONG key=0);
      IFC(ULONG) inmdword( ULONG offset, PULONG data, ULONG len=4, ULONG key=0);

      IFC(ULONG) outmbyte ( ULONG offset, PUCHAR  data, ULONG len=1, ULONG key=0);
      IFC(ULONG) outmword ( ULONG offset, PUSHORT  data, ULONG len=2, ULONG key=0);
      IFC(ULONG) outmdword( ULONG offset, PULONG data, ULONG len=4, ULONG key=0);

      // Base functions
      IFC(ULONG) GetWord_DM(USHORT Addr, PUSHORT Data) { return L_NOTSUPPORTED;}
      IFC(ULONG) PutWord_DM(USHORT Addr, USHORT Data) { return L_NOTSUPPORTED;}
      IFC(ULONG) PutWord_PM(USHORT Addr, ULONG Data) { return L_NOTSUPPORTED;}
      IFC(ULONG) GetWord_PM(USHORT Addr, PULONG Data) { return L_NOTSUPPORTED;}

      IFC(ULONG) GetArray_DM(USHORT Addr, ULONG Count, PUSHORT Data) { return L_NOTSUPPORTED;}
      IFC(ULONG) PutArray_DM(USHORT Addr, ULONG Count, PUSHORT Data) { return L_NOTSUPPORTED;}
      IFC(ULONG) PutArray_PM(USHORT Addr, ULONG Count, PULONG Data) { return L_NOTSUPPORTED;}
      IFC(ULONG) GetArray_PM(USHORT Addr, ULONG Count, PULONG Data) { return L_NOTSUPPORTED;}

      IFC(ULONG) SendCommand(USHORT cmd) { return L_NOTSUPPORTED;}

      // Service functions
      IFC(ULONG) PlataTest() { return L_NOTSUPPORTED;}

      IFC(ULONG) GetSlotParam(PSLOT_PAR slPar);// { return 0;}

      // Common functions
      IFC(HANDLE) OpenLDevice();
      IFC(ULONG)  CloseLDevice();

      IFC(ULONG)  SetParametersStream(PDAQ_PAR sp, ULONG *UsedSize, void** Data, void** Sync, ULONG StreamId = L_STREAM_ADC);
      IFC(ULONG)  RequestBufferStream(ULONG *Size, ULONG StreamId = L_STREAM_ADC); //in words
      IFC(ULONG)  FillDAQparameters(PDAQ_PAR sp);// {return L_NOTSUPPORTED; }

// two step must be
      IFC(ULONG)  InitStartLDevice();
      IFC(ULONG)  StartLDevice();
      IFC(ULONG)  StopLDevice();

      IFC(ULONG)  LoadBios(char *FileName) { return L_NOTSUPPORTED;}

      IFC(ULONG)  IoAsync(PDAQ_PAR sp);  // collect all async io operations
/*
      IFC(ULONG) InputTTL(PULONG Data, ULONG Mode) {return L_NOTSUPPORTED;} //2 in 1 all
      IFC(ULONG) OutputTTL(ULONG Data, ULONG Mode) {return L_NOTSUPPORTED;}  // in each set channel
      IFC(ULONG) ConfigTTL(ULONG Data) {return L_NOTSUPPORTED;} // 1221 and 1450

      IFC(ULONG) ConfigDAC(ULONG Mode, ULONG Number) {return L_NOTSUPPORTED;}
      IFC(ULONG) OutputDAC(short Data, ULONG Mode) {return L_NOTSUPPORTED;} //2 in 1

      IFC(ULONG) InputADC(USHORT Chan, PUSHORT Data) {return L_NOTSUPPORTED;}
*/
      IFC(ULONG) ReadPlataDescr(LPVOID pd) {return L_NOTSUPPORTED;}
      IFC(ULONG) WritePlataDescr(LPVOID pd, USHORT Ena=0) {return L_NOTSUPPORTED;} // ena - enables owerwrite 32 first word
      IFC(ULONG) ReadFlashWord(USHORT Addr, PUSHORT Data) {return L_NOTSUPPORTED;} // and byte
      IFC(ULONG) WriteFlashWord(USHORT Addr, USHORT Data) {return L_NOTSUPPORTED;}
      IFC(ULONG) EnableFlashWrite(USHORT Flag) {return L_NOTSUPPORTED;}

      IFC(ULONG) EnableCorrection(USHORT Ena) {return L_NOTSUPPORTED;}

      IFC(ULONG) GetParameter(ULONG name, PULONG param) {return L_NOTSUPPORTED;}
      IFC(ULONG) SetParameter(ULONG name, PULONG param) {return L_NOTSUPPORTED;}

      IFC(ULONG)  SetLDeviceEvent(HANDLE hEvent,ULONG EventId = L_STREAM_ADC)/* {return L_NOTSUPPORTED; }  */;


public:
   LDaqBoard(ULONG Slot)
   {
   	m_cRef.counter =0;
   	m_Slot=Slot;
   	hVxd=INVALID_HANDLE_VALUE;
   	hEvent=0;
   	DataBuffer = 0;
   	DataSize = 0;

      map_inSize =0;
      map_inBuffer=NULL;
      map_outSize =0;
      map_outBuffer=NULL;
	}

   ~LDaqBoard() {}

public:
   virtual ULONG  FillADCparameters(PDAQ_PAR sp) {return L_NOTSUPPORTED;}
   virtual ULONG  FillDACparameters(PDAQ_PAR sp) {return L_NOTSUPPORTED;}

   virtual ULONG InputTTL(PDAQ_PAR sp) {return L_NOTSUPPORTED;} //2 in 1 all
   virtual ULONG OutputTTL(PDAQ_PAR sp) {return L_NOTSUPPORTED;}  // in each set channel
   virtual ULONG ConfigTTL(PDAQ_PAR sp) {return L_NOTSUPPORTED;} // 1221 and 1450

   virtual ULONG ConfigDAC(PDAQ_PAR sp) {return L_NOTSUPPORTED;}
   virtual ULONG OutputDAC(PDAQ_PAR sp) {return L_NOTSUPPORTED;} //2 in 1

   virtual ULONG InputADC(PDAQ_PAR sp) {return L_NOTSUPPORTED;}
   virtual ULONG ConfigADC(PDAQ_PAR sp) {return L_NOTSUPPORTED;}

// class specific function to extend base
public:
   virtual HANDLE csOpenLDevice() { return hVxd; }
   virtual ULONG  csCloseLDevice(ULONG status) { return status; }
   virtual ULONG  csRequestBufferStream(ULONG *Size, ULONG StreamId, ULONG status) { return status; } //in words
   virtual ULONG  csSetParametersStream(PDAQ_PAR sp, PULONG UsedSize, void** Data, void** Sync, ULONG StreamId, ULONG status) { return status; }

// service function
public:
	virtual void CopyDAQtoWDAQ(PDAQ_PAR dp, LPVOID ss, int sp_type);

private:
	atomic_t m_cRef;
//   	long     m_cRef;
   ULONG       m_Slot;

protected:
   //  this is DEV_ALL
   HANDLE      hVxd;

   HANDLE      hEvent; // for ovelapped DIOC_START under NT
   OVERLAPPED  ov;

   SLOT_PAR sl;

   ADC_PAR adc_par;  // to fill with FillDAQparam
   DAC_PAR dac_par;

   // add for C-style driver in Linux
   WDAQ_PAR wadc_par;
   WDAQ_PAR wdac_par;

   PLATA_DESCR_U2 pdu;

   ULONG *DataBuffer; // pointer for data buffer for busmaster boards in windows
   ULONG DataSize;    // size of buffer

	// size and pointers for data buffers in Linux
   int map_inSize;
   void *map_inBuffer;
   int map_outSize;
   void *map_outBuffer;
   int map_regSize;
   void *map_regBuffer;
};

