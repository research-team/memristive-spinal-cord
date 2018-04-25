#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;

extern void _ampa_reg(void);
extern void _gaba_reg(void);
extern void _pregen_reg(void);
extern void _stdp_reg(void);
extern void _stdwa_soft_reg(void);

void modl_reg(){
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");

    fprintf(stderr," ampa.mod");
    fprintf(stderr," gaba.mod");
    fprintf(stderr," pregen.mod");
    fprintf(stderr," stdp.mod");
    fprintf(stderr," stdwa_soft.mod");
    fprintf(stderr, "\n");
  }
  _ampa_reg();
  _gaba_reg();
  _pregen_reg();
  _stdp_reg();
  _stdwa_soft_reg();
}