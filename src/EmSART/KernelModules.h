//
// Created by uermel on 9/29/23.
//

#ifndef ARTIATOMI_KERNELMODULES_H
#define ARTIATOMI_KERNELMODULES_H

#include "EmSartDefault.h"
#include "CudaContext.h"
#include "CudaKernelBinarys.h"

class KernelModules
{
private:
    bool compilerOutput;
    bool infoOutput;

public:
    KernelModules(Cuda::CudaContext* aCuCtx);
    CUmodule modFP;
    CUmodule modSlicer;
    CUmodule modVolTravLen;
    CUmodule modComp;
    CUmodule modWBP;
    CUmodule modBP;
    CUmodule modCTF;
    CUmodule modCTS;
    CUmodule modFPLUT;
    CUmodule modBPLUT;
    CUmodule modSplines;
};


#endif //ARTIATOMI_KERNELMODULES_H
