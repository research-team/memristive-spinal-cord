library("ks")
test <- function() {
  
  chem_bio_times <- read.table(file = "/gpfs/GLOBAL_JOB_REPO_KPFU/openlab/GRAS/multi_gpu_test/dat/3/bio_times.txt", sep = ";")
  chem_bio_ampls <- read.table(file = "/gpfs/GLOBAL_JOB_REPO_KPFU/openlab/GRAS/multi_gpu_test/dat/3/bio_ampls.txt", sep = ";")
  chem_test_times <- read.table(file = "/gpfs/GLOBAL_JOB_REPO_KPFU/openlab/GRAS/multi_gpu_test/dat/3/test_times.txt", sep = ";")
  chem_test_ampls <- read.table(file = "/gpfs/GLOBAL_JOB_REPO_KPFU/openlab/GRAS/multi_gpu_test/dat/3/test_ampls.txt", sep = ";")
  
  for (i in 1:ncol(chem_bio_times)) {
    x1 <- chem_bio_times[i]
    x1 <- x1[x1 >= 0]
    y1 <- chem_bio_ampls[i]
    y1 <- y1[y1 >= 0]
    
    for (j in 1:ncol(chem_test_times)) {
      x2 <- chem_test_times[j]
      x2 <- x2[x2 >= 0]
      y2 <- chem_test_ampls[j]
      y2 <- y2[y2 >= 0]
      #
      mat1 <- matrix(c(x1, y1), nrow=length(x1))
      mat2 <- matrix(c(x2, y2), nrow=length(x2))
      #
      res_time <- kde.test(x1=x1, x2=x2)$pvalue
      res_ampl <- kde.test(x1=y1, x2=y2)$pvalue
      # res_2d <- kde.test(x1=mat1, x2=mat2)$pvalue
      res_2d <- 0
      write.table(res_time, "/gpfs/GLOBAL_JOB_REPO_KPFU/openlab/GRAS/multi_gpu_test/dat/3/time.txt", append = TRUE, quote=FALSE, sep=" ", row.names=FALSE, col.names=FALSE)
      write.table(res_ampl, "/gpfs/GLOBAL_JOB_REPO_KPFU/openlab/GRAS/multi_gpu_test/dat/3/ampl.txt", append = TRUE, quote=FALSE, sep=" ", row.names=FALSE, col.names=FALSE)
      write.table(res_2d, "/gpfs/GLOBAL_JOB_REPO_KPFU/openlab/GRAS/multi_gpu_test/dat/3/2d.txt", append = TRUE, quote=FALSE, sep=" ", row.names=FALSE, col.names=FALSE)
    }
  }
}

test()
