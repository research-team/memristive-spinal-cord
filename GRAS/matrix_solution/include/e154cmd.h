#ifndef _E154CMD_H
#define _E154CMD_H
// В­В®В¬ТђР°В  В¤В®Р±РІРіР‡В­Р»Рµ Р‡В®В«РјВ§В®СћВ РІТђВ«РјР±Р„РЃРµ В§В Р‡Р°В®Р±В®Сћ В¤В«Рї USB (vendor request)
#define V_RESET_DSP_E154       0
#define V_PUT_ARRAY_E154       1
#define V_GET_ARRAY_E154       2
#define V_START_ADC_E154       3
#define V_STOP_ADC_E154        4
#define V_START_ADC_ONCE_E154  5
//#define V_GO_SLEEP_E440        6
//#define V_WAKEUP_E440          7
#define V_GET_MODULE_NAME_E154 11



#define L_ADC_PARS_BASE_E154        0x0060
#define L_ADC_ONCE_FLAG_E154        (L_ADC_PARS_BASE_E154 + 136)
#define L_FLASH_ENABLED_E154        (L_ADC_PARS_BASE_E154 + 137)
#define L_TTL_OUT_E154              0x0400
#define L_TTL_IN_E154               0x0400
#define L_ENABLE_TTL_OUT_E154       0x0402
#define L_ADC_SAMPLE_E154           0x0410
#define L_ADC_CHANNEL_SELECT_E154   0x0412
#define L_ADC_START_E154            0x0413
#define L_DAC_SAMPLE_E154           0x0420
#define L_SUSPEND_MODE_E154         0x0430
#define L_DATA_FLASH_BASE_E154      0x0800
#define L_CODE_FLASH_BASE_E154      0x1000
#define L_BIOS_VERSION_E154         0x1080
#define L_DESCRIPTOR_BASE_E154      0x2780
#define L_RAM_E154                  0x8000

#endif
