//
// Created by uermel on 9/30/21.
//

#ifndef ARTIATOMI_VOLTRAVLENGTHKERNEL_H
#define ARTIATOMI_VOLTRAVLENGTHKERNEL_H

#include "EmSartDefault.h"
#include <CudaArrays.h>
#include <CudaContext.h>
#include <CudaTextures.h>
#include <CudaSurfaces.h>
#include <CudaKernel.h>
#include <CudaDeviceProperties.h>
#include "Projection.h"
#include "Volume.h"

class VolTravLengthKernel : public Cuda::CudaKernel
{
public:
    VolTravLengthKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
    VolTravLengthKernel(CUmodule aModule);

    float operator()(int x, int y, Cuda::CudaPitchedDeviceVariable& distMap);
    float operator()(int x, int y, Cuda::CudaPitchedDeviceVariable& distMap, int2 roiMin, int2 roiMax);
};

void SetConstantValues(VolTravLengthKernel& kernel, Volume<float>& vol, Projection& proj, int index, int subVol, Matrix<float>& m, Matrix<float>& mInv);
void SetConstantValues(VolTravLengthKernel& kernel, Volume<unsigned short>& vol, Projection& proj, int index, int subVol, Matrix<float>& m, Matrix<float>& mInv);

#endif //ARTIATOMI_VOLTRAVLENGTHKERNEL_H
