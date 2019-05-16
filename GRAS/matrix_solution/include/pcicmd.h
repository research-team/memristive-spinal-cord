#ifndef _PCICMD_H
#define _PCICMD_H

// Working with PCI PLX boards


// Internal variables
#define  L_CONTROL_TABLE_PLX                 0x8A00

#define  L_SCALE_PLX                         0x8D00
#define  L_ZERO_PLX                          0x8D04

#define  L_CONTROL_TABLE_LENGHT_PLX          0x8D08

#define  L_BOARD_REVISION_PLX                0x8D3F
#define  L_READY_PLX                         0x8D40
#define  L_TMODE1_PLX                        0x8D41
#define  L_TMODE2_PLX                        0x8D42
#define  L_DAC_IRQ_SOURCE_PLX                0x8D43
#define  L_DAC_ENABLE_IRQ_VALUE_PLX          0x8D44
#define  L_DAC_IRQ_FIFO_ADDRESS_PLX          0x8D45
#define  L_DAC_IRQ_STEP_PLX                  0x8D46
#define  L_ENABLE_TTL_OUT_PLX                0x8D47
#define  L_DSP_TYPE_PLX                      0x8D48
#define  L_COMMAND_PLX                       0x8D49
#define  L_FIRST_SAMPLE_DELAY_PLX            0x8D4A
#define  L_TTL_OUT_PLX                       0x8D4C
#define  L_TTL_IN_PLX                        0x8D4D
#define  L_DAC_FIFO_PTR_PLX                  0x8D4F
#define  L_FIFO_PTR_PLX                      0x8D50
#define  L_TEST_LOAD_PLX                     0x8D52
#define  L_ADC_RATE_PLX                      0x8D53
#define  L_INTER_KADR_DELAY_PLX              0x8D54
#define  L_DAC_RATE_PLX                      0x8D55
#define  L_DAC_VALUE_PLX                     0x8D56
#define  L_ENABLE_IRQ_PLX                    0x8D57
#define  L_IRQ_STEP_PLX                      0x8D58
#define  L_IRQ_FIFO_ADDRESS_PLX              0x8D5A
#define  L_ENABLE_IRQ_VALUE_PLX              0x8D5B
#define  L_ADC_SAMPLE_PLX                    0x8D5C
#define  L_ADC_CHANNEL_PLX                   0x8D5D
#define  L_DAC_SCLK_DIV_PLX                  0x8D5E
#define  L_CORRECTION_ENABLE_PLX             0x8D60

#define  L_ADC_ENABLE_PLX                    0x8D62
#define  L_ADC_FIFO_BASE_ADDRESS_PLX         0x8D63
#define  L_ADC_FIFO_BASE_ADDRESS_INDEX_PLX   0x8D64
#define  L_ADC_FIFO_LENGTH_PLX               0x8D65
#define  L_ADC_NEW_FIFO_LENGTH_PLX           0x8D66

#define  L_DAC_ENABLE_STREAM_PLX             0x8D67
#define  L_DAC_FIFO_BASE_ADDRESS_PLX         0x8D68
#define  L_DAC_FIFO_LENGTH_PLX               0x8D69
#define  L_DAC_NEW_FIFO_LENGTH_PLX           0x8D6A
#define  L_DAC_ENABLE_IRQ_PLX						0x8D6B

#define  L_SYNCHRO_TYPE_PLX                  0x8D70
#define  L_SYNCHRO_AD_CHANNEL_PLX            0x8D73
#define  L_SYNCHRO_AD_POROG_PLX              0x8D74
#define  L_SYNCHRO_AD_MODE_PLX               0x8D75
#define  L_SYNCHRO_AD_SENSITIVITY_PLX        0x8D76
#define  L_DAC                               0x8F00


// command defines
#define cmTEST_PLX                  0
#define cmLOAD_CONTROL_TABLE_PLX    1
#define cmADC_ENABLE_PLX            2
#define cmADC_FIFO_CONFIG_PLX       3
#define cmSET_ADC_KADR_PLX          4
#define cmENABLE_DAC_STREAM_PLX     5
#define cmDAC_FIFO_CONFIG_PLX       6
#define cmSET_DAC_RATE_PLX          7
#define cmADC_SAMPLE_PLX            8
#define cmTTL_IN_PLX                9
#define cmTTL_OUT_PLX               10
#define cmSYNCHRO_CONFIG_PLX        11
#define cmENABLE_IRQ_PLX            12
#define cmIRQ_TEST_PLX              13
#define cmSET_DSP_TYPE_PLX          14
#define cmENABLE_TTL_OUT_PLX        15

#endif
