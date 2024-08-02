//
// Created by uermel on 2/7/22.
//

#include "RadialSumKernel.h"

using namespace Cuda;

RadialSumAbsKernel::RadialSumAbsKernel(CUmodule aModule)
        : CudaKernel("radialSumAbs", aModule,
                     make_dim3(1, 1, 1),
                     make_dim3(16, 16, 1), 0)
{

}

float RadialSumAbsKernel::operator()(Cuda::CudaDeviceVariable& image,
                                     Cuda::CudaDeviceVariable& sum,
                                     Cuda::CudaDeviceVariable& multiplicity,
                                     uint2 fftDim,
                                     float2 imDim,
                                     float2 asymCorrFac,
                                     float scaleFactor)
{
    CUdeviceptr image_dptr = image.GetDevicePtr();
    CUdeviceptr sum_dptr = sum.GetDevicePtr();
    CUdeviceptr multiplicity_dptr = multiplicity.GetDevicePtr();

    void** arglist = (void**)new void*[7];

    arglist[0] = &image_dptr;
    arglist[1] = &sum_dptr;
    arglist[2] = &multiplicity_dptr;
    arglist[3] = &fftDim;
    arglist[4] = &imDim;
    arglist[5] = &asymCorrFac;
    arglist[6] = &scaleFactor;

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

RadialSum3DKernel::RadialSum3DKernel(CUmodule aModule)
        : CudaKernel("radialSum3D", aModule,
                     make_dim3(1, 1, 1),
                     make_dim3(4, 4, 4), 0)
{

}

float RadialSum3DKernel::operator()(Cuda::CudaDeviceVariable& volume,
                                    Cuda::CudaDeviceVariable& sum,
                                    Cuda::CudaDeviceVariable& multiplicity,
                                    uint3 fftDim,
                                    uint3 volDim,
                                    float3 asymCorrFac,
                                    float scaleFactor,
                                    int maxShell)
{
    CUdeviceptr volume_dptr = volume.GetDevicePtr();
    CUdeviceptr sum_dptr = sum.GetDevicePtr();
    CUdeviceptr multiplicity_dptr = multiplicity.GetDevicePtr();

    void** arglist = (void**)new void*[8];

    arglist[0] = &volume_dptr;
    arglist[1] = &sum_dptr;
    arglist[2] = &multiplicity_dptr;
    arglist[3] = &fftDim;
    arglist[4] = &volDim;
    arglist[5] = &asymCorrFac;
    arglist[6] = &scaleFactor;
    arglist[7] = &maxShell;

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

RadialSumAbs3DKernel::RadialSumAbs3DKernel(CUmodule aModule)
        : CudaKernel("radialSumAbs3D", aModule,
                     make_dim3(1, 1, 1),
                     make_dim3(4, 4, 4), 0)
{

}

float RadialSumAbs3DKernel::operator()(Cuda::CudaDeviceVariable& volume,
                                     Cuda::CudaDeviceVariable& sum,
                                     Cuda::CudaDeviceVariable& multiplicity,
                                     uint3 fftDim,
                                     uint3 volDim,
                                     float3 asymCorrFac,
                                     float scaleFactor,
                                     int maxShell)
{
    CUdeviceptr volume_dptr = volume.GetDevicePtr();
    CUdeviceptr sum_dptr = sum.GetDevicePtr();
    CUdeviceptr multiplicity_dptr = multiplicity.GetDevicePtr();

    void** arglist = (void**)new void*[8];

    arglist[0] = &volume_dptr;
    arglist[1] = &sum_dptr;
    arglist[2] = &multiplicity_dptr;
    arglist[3] = &fftDim;
    arglist[4] = &volDim;
    arglist[5] = &asymCorrFac;
    arglist[6] = &scaleFactor;
    arglist[7] = &maxShell;

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


FSC3DKernel::FSC3DKernel(CUmodule aModule)
        : CudaKernel("fsc3D", aModule,
                     make_dim3(1, 1, 1),
                     make_dim3(4, 4, 4), 0)
{

}

float FSC3DKernel::operator()(Cuda::CudaDeviceVariable& image_1,
                              Cuda::CudaDeviceVariable& image_2,
                              Cuda::CudaDeviceVariable& amp1_sum,
                              Cuda::CudaDeviceVariable& amp2_sum,
                              Cuda::CudaDeviceVariable& ampdiff_sum,
                              Cuda::CudaDeviceVariable& multiplicity,
                              uint3 fftDim,
                              uint3 volDim,
                              float3 asymCorrFac,
                              float scaleFactor,
                              int maxShell)
{
    CUdeviceptr image_1_dptr = image_1.GetDevicePtr();
    CUdeviceptr image_2_dptr = image_2.GetDevicePtr();
    CUdeviceptr amp1_sum_dptr = amp1_sum.GetDevicePtr();
    CUdeviceptr amp2_sum_dptr = amp2_sum.GetDevicePtr();
    CUdeviceptr ampdiff_sum_dptr = ampdiff_sum.GetDevicePtr();
    CUdeviceptr multiplicity_dptr = multiplicity.GetDevicePtr();

    void** arglist = (void**)new void*[11];

    arglist[0] = &image_1_dptr;
    arglist[1] = &image_2_dptr;
    arglist[2] = &amp1_sum_dptr;
    arglist[3] = &amp2_sum_dptr;
    arglist[4] = &ampdiff_sum_dptr;
    arglist[5] = &multiplicity_dptr;
    arglist[6] = &fftDim;
    arglist[7] = &volDim;
    arglist[8] = &asymCorrFac;
    arglist[9] = &scaleFactor;
    arglist[10] = &maxShell;

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

FSCNorm1DKernel::FSCNorm1DKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : Cuda::CudaKernel("fscNorm", aModule, aGridDim, aBlockDim, 0)
{

}

FSCNorm1DKernel::FSCNorm1DKernel(CUmodule aModule)
        : Cuda::CudaKernel("fscNorm", aModule, make_dim3(1,1,1), make_dim3(32, 1, 1), 0)
{

}

float FSCNorm1DKernel::operator()(CudaDeviceVariable& amp1_sum,
                                  CudaDeviceVariable& amp2_sum,
                                  CudaDeviceVariable& ampdiff_sum,
                                  CudaDeviceVariable& multiplicity,
                                  uint length,
                                  float threshold)
{
    CUdeviceptr amp1_sum_dptr = amp1_sum.GetDevicePtr();
    CUdeviceptr amp2_sum_dptr = amp2_sum.GetDevicePtr();
    CUdeviceptr ampdiff_sum_dptr = ampdiff_sum.GetDevicePtr();
    CUdeviceptr multiplicity_dptr = multiplicity.GetDevicePtr();

    void** arglist = (void**)new void*[6];

    arglist[0] = &amp1_sum_dptr;
    arglist[1] = &amp2_sum_dptr;
    arglist[2] = &ampdiff_sum_dptr;
    arglist[3] = &multiplicity_dptr;
    arglist[4] = &length;
    arglist[5] = &threshold;

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



