//
// Created by uermel on 10/5/21.
//

#include "CubicResampleKernel.h"

using namespace Cuda;

CubicResampleKernel::CubicResampleKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : Cuda::CudaKernel("sample", aModule, aGridDim, aBlockDim, 0)
{

}

CubicResampleKernel::CubicResampleKernel(CUmodule aModule)
        : Cuda::CudaKernel("sample", aModule, make_dim3(1,1,1), make_dim3(32, 8, 1), 0)
{

}

float CubicResampleKernel::operator()(CudaTextureObject3D& involume,
                                      CudaSurfaceObject3D& outvolume,
                                      Volume<float>* vol)
{
    CUtexObject texVol = involume.GetTexObject();
    CUsurfObject outvol = outvolume.GetSurfObject();
    float3 volDim = vol->GetDimension();

    void** arglist = (void**)new void*[3];

    arglist[0] = &texVol;
    arglist[1] = &outvol;
    arglist[2] = &volDim;

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