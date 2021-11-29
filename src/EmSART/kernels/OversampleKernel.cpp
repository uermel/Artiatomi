//
// Created by uermel on 10/8/21.
//

#include "OversampleKernel.h"

using namespace Cuda;

OversampleKernel::OversampleKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : Cuda::CudaKernel("oversample", aModule, aGridDim, aBlockDim, 0)
{

}

OversampleKernel::OversampleKernel(CUmodule aModule)
        : Cuda::CudaKernel("oversample", aModule, make_dim3(1,1,1), make_dim3(1, 1, 1), 0)
{

}

float OversampleKernel::operator()(int x,
                                   int y,
                                   int maxOverSample,
                                   Cuda::CudaTextureObject2D& projection,
                                   Cuda::CudaPitchedDeviceVariable& outprojection)
{
    float maxOverSampleInv = 1.f/maxOverSample;
    float maxOverSampleInvH = 0.5f * 1.f/maxOverSample;
    CUtexObject inTex = projection.GetTexObject();
    CUdeviceptr outPtr = outprojection.GetDevicePtr();
    size_t outStride = outprojection.GetPitch();

    void** arglist = (void**)new void*[8];

    printf("\nx: %i\n", x);
    printf("\ny: %i\n", y);
    //printf("\nx: %zu\n", outprojection.GetWidth());
    //printf("\ny: %zu\n", outprojection.GetHeight());
    printf("\ny: %zu\n", outStride);
    printf("\nmaxOversample: %i\n", maxOverSample);
    printf("\nmaxOverSampleInvH: %f\n", maxOverSampleInvH);
    printf("\nmaxOversampleInv: %f\n", maxOverSampleInv);
    //printf("\nmaxOverSampleInvH: %f\n", maxOverSampleInvH);

    arglist[0] = &x;
    arglist[1] = &y;
    arglist[2] = &maxOverSample;
    arglist[3] = &maxOverSampleInv;
    arglist[4] = &maxOverSampleInvH;
    arglist[5] = &inTex;
    arglist[6] = &outPtr;
    arglist[7] = &outStride;

    float ms;

    CUevent eventStart;
    CUevent eventEnd;
    CUstream stream = 0;
    cudaSafeCall(cuEventCreate(&eventStart, CU_EVENT_BLOCKING_SYNC));
    cudaSafeCall(cuEventCreate(&eventEnd, CU_EVENT_BLOCKING_SYNC));

    cudaSafeCall(cuEventRecord(eventStart, stream));
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 10000000);
    cudaSafeCall(cuLaunchKernel(mFunction, mGridDim.x, mGridDim.y, mGridDim.z, mBlockDim.x, mBlockDim.y, mBlockDim.z, mSharedMemSize, NULL, arglist, NULL));

    cudaSafeCall(cuCtxSynchronize());

    cudaSafeCall(cuEventRecord(eventEnd, stream));
    cudaSafeCall(cuEventSynchronize(eventEnd));
    cudaSafeCall(cuEventElapsedTime(&ms, eventStart, eventEnd));

    cudaSafeCall(cuEventDestroy(eventStart));
    cudaSafeCall(cuEventDestroy(eventEnd));

    delete[] arglist;
    return ms;
}
