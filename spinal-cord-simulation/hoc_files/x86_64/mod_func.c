#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;

extern void _AXNODE_reg(void);
extern void _INITIAL_reg(void);
extern void _MOTONEURON_5HT_reg(void);
extern void _MOTONEURON_reg(void);

void modl_reg(){
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");

    fprintf(stderr," ./mod_files//AXNODE.mod");
    fprintf(stderr," ./mod_files//INITIAL.mod");
    fprintf(stderr," ./mod_files//MOTONEURON_5HT.mod");
    fprintf(stderr," ./mod_files//MOTONEURON.mod");
    fprintf(stderr, "\n");
  }
  _AXNODE_reg();
  _INITIAL_reg();
  _MOTONEURON_5HT_reg();
  _MOTONEURON_reg();
}
