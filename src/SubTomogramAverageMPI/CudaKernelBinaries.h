// Cuda kernels are compiled into binary headers and imported into necessary 
// source files. The kernels necessary for SubTomogramAverageMPI are collected
// here into one include file.

#ifndef STA_CUDAKERNELBINARYS_H
#define STA_CUDAKERNELBINARYS_H

#include "cuda_kernels/basicKernels.cu.h"
#include "cuda_kernels/kernel.cu.h"

#endif