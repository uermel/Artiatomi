//
// Created by uermel on 10/10/23.
//

#include "FreqSampleKernel.h"

using namespace Cuda;

FreqSampleKernel::FreqSampleKernel(CUmodule aModule)
        : CudaKernel("sampleFreqs", aModule,
                     make_dim3(1, 1, 1),
                     make_dim3(32, 1, 1), 0)
{

}

float FreqSampleKernel::operator()(Cuda::CudaDeviceVariable& dst,
                                   Cuda::CudaTextureObject1D& src,
                                   uint dstLength,
                                   uint dstNyq,
                                   uint srcLength,
                                   float voxelSizeDst,
                                   float voxelSizeSrc,
                                   SF_EXTRAP_MODE mode,
                                   float extrapVal)
{
    CUdeviceptr dst_dptr = dst.GetDevicePtr();
    CUtexObject src_obj = src.GetTexObject();

    float fstepDst = (0.5f/voxelSizeDst) / (float)dstNyq;
    float fstepSrc = (0.5f/voxelSizeSrc) / (float)srcLength;
    float scaleFac = fstepDst / fstepSrc;

//    printf("\nfstepDst %.10f\n", fstepDst);
//    printf("\nfstepSrc %.10f\n", fstepSrc);
//    printf("\nscaleFac %f\n", scaleFac);
//    printf("\ndstNyq %i\n", dstNyq);

    void** arglist = (void**)new void*[7];

    arglist[0] = &dst_dptr;
    arglist[1] = &src_obj;
    arglist[2] = &dstLength;
    arglist[3] = &srcLength;
    arglist[4] = &scaleFac;
    arglist[5] = &extrapVal;
    arglist[6] = &mode;

    float ms;

    CUevent eventStart;
    CUevent eventEnd;
    CUstream stream = 0;
    cudaSafeCall(cuEventCreate(&eventStart, CU_EVENT_BLOCKING_SYNC));
    cudaSafeCall(cuEventCreate(&eventEnd, CU_EVENT_BLOCKING_SYNC));

    cudaSafeCall(cuEventRecord(eventStart, stream));
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