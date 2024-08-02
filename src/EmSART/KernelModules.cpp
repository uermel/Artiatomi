//
// Created by uermel on 9/29/23.
//

#include "KernelModules.h"

KernelModules::KernelModules(Cuda::CudaContext* aCuCtx)
        :compilerOutput(false),
         infoOutput(false)
{
    modFP = aCuCtx->LoadModulePTX(ForwardProjectionRayMarcher_TL, 0, infoOutput, compilerOutput);
    modSlicer = aCuCtx->LoadModulePTX(ForwardProjectionSlicer, 0, infoOutput, compilerOutput);
    modVolTravLen = modSlicer;
    modComp = aCuCtx->LoadModulePTX(Compare, 0, infoOutput, compilerOutput);
    modWBP = aCuCtx->LoadModulePTX(wbpWeighting, 0, infoOutput, compilerOutput);
    modBP = aCuCtx->LoadModulePTX(BackProjectionSquareOS, 0, infoOutput, compilerOutput);
    modCTF = aCuCtx->LoadModulePTX(ctf, 0, infoOutput, compilerOutput);
    modCTS = aCuCtx->LoadModulePTX(CopyToSquare, 0, infoOutput, compilerOutput);
    modFPLUT = aCuCtx->LoadModulePTX(ForwardProjectionLUT, 0, infoOutput, compilerOutput);
    modBPLUT = aCuCtx->LoadModulePTX(BackProjectionLUT, 0, infoOutput, compilerOutput);
    modSplines = aCuCtx->LoadModulePTX(splines, 0, infoOutput, compilerOutput);
}