#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;

extern void _5ht_reg(void);
extern void _AXNODE_reg(void);
extern void _FC_reg(void);
extern void _MOTONEURON_reg(void);
extern void _MOTONEURON_5HT_reg(void);
extern void _diffusion_reg(void);
extern void _pregen_reg(void);
extern void _slow5HT_reg(void);
extern void _stdwa_soft_reg(void);

void modl_reg(){
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");

    fprintf(stderr," 5ht.mod");
    fprintf(stderr," AXNODE.mod");
    fprintf(stderr," FC.mod");
    fprintf(stderr," MOTONEURON.mod");
    fprintf(stderr," MOTONEURON_5HT.mod");
    fprintf(stderr," diffusion.mod");
    fprintf(stderr," pregen.mod");
    fprintf(stderr," slow5HT.mod");
    fprintf(stderr," stdwa_soft.mod");
    fprintf(stderr, "\n");
  }
  _5ht_reg();
  _AXNODE_reg();
  _FC_reg();
  _MOTONEURON_reg();
  _MOTONEURON_5HT_reg();
  _diffusion_reg();
  _pregen_reg();
  _slow5HT_reg();
  _stdwa_soft_reg();
}
