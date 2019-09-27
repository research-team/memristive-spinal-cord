#include <unistd.h>
#include "l502api.h"
#include "e502api.h"
#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TCP_CONNECTION_TOUT 5000
#define OUT_SIGNAL_SIZE 2000
#define OUT_BLOCK_SIZE 256
#define SEND_TOUT 500


static uint32_t get_all_devrec(t_x502_devrec **pdevrec_list, uint32_t *ip_addr_list, unsigned ip_cnt) {
	/* Функция находит все подключенные модули по интерфейсам PCI-Express и USB и
	 * сохраняет записи о этих устройствах в выделенный массив.
	 * Также создаются записи по переданным IP-адресам модулей и добавляются в конец массива.
	 * Указатель на выделенный массив, который должен быть потом очищен, сохраняется
	 * в pdevrec_list, а количество действительных элементов (память которых должна
	 * быть в дальнейшем освобождена с помощью X502_FreeDevRecordList()) возвращается
	 * как результат функции */
    int32_t fnd_devcnt = 0;
    uint32_t pci_devcnt = 0;
    uint32_t usb_devcnt = 0;

    t_x502_devrec *rec_list = NULL;

    // получаем количество подключенных устройств по интерфейсам PCI и USB
    L502_GetDevRecordsList(NULL, 0, 0, &pci_devcnt);
    E502_UsbGetDevRecordsList(NULL, 0, 0, &usb_devcnt);

    if ((pci_devcnt + usb_devcnt + ip_cnt) != 0) {
        // выделяем память для массива для сохранения найденного количества записей
        rec_list = (t_x502_devrec*) malloc((pci_devcnt + usb_devcnt + ip_cnt) * sizeof(t_x502_devrec));

        if (rec_list != NULL) {
            unsigned i;
            // получаем записи о модулях L502, но не больше pci_devcnt
            if (pci_devcnt!=0) {
                int32_t res = L502_GetDevRecordsList(&rec_list[fnd_devcnt], pci_devcnt, 0, NULL);
                if (res >= 0) {
                    fnd_devcnt += res;
                }
            }
            // добавляем записи о модулях E502, подключенных по USB, в конец массива
            if (usb_devcnt!=0) {
                int32_t res = E502_UsbGetDevRecordsList(&rec_list[fnd_devcnt], usb_devcnt, 0, NULL);
                if (res >= 0) {
                    fnd_devcnt += res;
                }
            }

            // создаем записи для переданного массива ip-адресов
            for (i=0; i < ip_cnt; i++) {
                if (E502_MakeDevRecordByIpAddr(&rec_list[fnd_devcnt], ip_addr_list[i],0, TCP_CONNECTION_TOUT) == X502_ERR_OK) {
                    fnd_devcnt++;
                }
            }
        }
    }

    if (fnd_devcnt != 0) {
        // если создана хотя бы одна запись, то сохраняем указатель на выделенный массив
        *pdevrec_list = rec_list;
    } else {
        *pdevrec_list = NULL;
        free(rec_list);
    }

    return fnd_devcnt;
}


static t_x502_hnd dev_open(int argc, char** argv) {
    t_x502_hnd hnd = NULL;
    uint32_t fnd_devcnt;
    t_x502_devrec *devrec_list = NULL;
    uint32_t *ip_addr_list = NULL;
    uint32_t ip_cnt = 0;

    // получаем список модулей для выбора
    fnd_devcnt = get_all_devrec(&devrec_list, ip_addr_list, ip_cnt);

	if (devrec_list[0].iface != X502_IFACE_ETH)
        printf("Сер. номер: %s\n", devrec_list[0].serial);
    else
        printf("Адрес: %s\n", devrec_list[0].location);
	
	// создаем описатель
    hnd = X502_Create();
    if (hnd==NULL) {
        fprintf(stderr, "Ошибка создания описателя модуля!");
    } else {
        // устанавливаем связь с модулем по записи
        int32_t err = X502_OpenByDevRecord(hnd, &devrec_list[0]);
        if (err != X502_ERR_OK) {
            fprintf(stderr, "Ошибка установления связи с модулем: %s!", X502_GetErrorString(err));
            X502_Free(hnd);
            hnd = NULL;
        }
    }
    // освобождение ресурсов действительных записей из списка
    X502_FreeDevRecordList(devrec_list, fnd_devcnt);
    return hnd;
}

int main(int argc, char **argv) {
    int32_t err = 0;
    uint32_t ver;
    t_x502_hnd hnd = NULL;

    // получаем версию библиотеки
    ver = X502_GetLibraryVersion();
    printf("Верисия библиотеки: %d.%d.%d\n", (ver >> 24)&0xFF, (ver>>16)&0xFF, (ver>>8)&0xFF);

    // получение списка устройств и выбор, с каким будем работать
    hnd = dev_open(argc, argv);

    // если успешно выбрали модуль и установили с ним связь - продолжаем работу
    if (hnd != NULL) {
        // получаем информацию
        t_x502_info info;
        err = X502_GetDevInfo(hnd, &info);
        if (err != X502_ERR_OK) {
            fprintf(stderr, "Ошибка получения серийного информации о модуле: %s!", X502_GetErrorString(err));
        } else {
            // выводим полученную информацию
            printf("Серийный номер          : %s\n", info.serial);
            printf("Наличие ЦАП             : %s\n", info.devflags & X502_DEVFLAGS_DAC_PRESENT ? "Да" : "Нет");
            printf("Наличие BlackFin        : %s\n", info.devflags & X502_DEVFLAGS_BF_PRESENT ? "Да" : "Нет");
            printf("Наличие гальваноразвязки: %s\n", info.devflags & X502_DEVFLAGS_GAL_PRESENT ? "Да" : "Нет");
            printf("Индустриальное исп.     : %s\n", info.devflags & X502_DEVFLAGS_INDUSTRIAL ? "Да" : "Нет");
            printf("Наличие интерф. PCI/PCIe: %s\n", info.devflags & X502_DEVFLAGS_IFACE_SUPPORT_PCI ? "Да" : "Нет");
            printf("Наличие интерф. USB     : %s\n", info.devflags & X502_DEVFLAGS_IFACE_SUPPORT_USB ? "Да" : "Нет");
            printf("Наличие интерф. Ethernet: %s\n", info.devflags & X502_DEVFLAGS_IFACE_SUPPORT_ETH ? "Да" : "Нет");
            printf("Версия ПЛИС             : %d.%d\n", (info.fpga_ver >> 8) & 0xFF, info.fpga_ver & 0xFF);
            printf("Версия PLDA             : %d\n", info.plda_ver);
            if (info.mcu_firmware_ver != 0) {
                printf("Версия прошивки ARM     : %d.%d.%d.%d\n",
                       (info.mcu_firmware_ver >> 24) & 0xFF,
                       (info.mcu_firmware_ver >> 16) & 0xFF,
                       (info.mcu_firmware_ver >>  8) & 0xFF,
                       info.mcu_firmware_ver & 0xFF);
            }
        }

        // MAX 30 000
        for (int i = 0; i < 30000; i+=100) {
        	printf("Value %d \n", i);
			err = X502_AsyncOutDac(hnd, X502_DAC_CH1, i, X502_DAC_FLAGS_CALIBR);

	        if (err != X502_ERR_OK) {
	            fprintf(stderr, "Ошибка (%d): %s!", err, X502_GetErrorString(err));
	            exit(100);
	        }
	        usleep(50000);
    	}
    	// reset V to 0
        err = X502_AsyncOutDac(hnd, X502_DAC_CH1, 0, X502_DAC_FLAGS_VOLT);

        // закрываем связь с модулем
        X502_Close(hnd);
        // освобождаем описатель
        X502_Free(hnd);
    }
    return err;
}
