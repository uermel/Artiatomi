//
// Created by uermel on 10/5/21.
//

#include "CubicResampleKernel.h"

using namespace Cuda;

CubicResampleKernel2D::CubicResampleKernel2D(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : Cuda::CudaKernel("sample2D", aModule, aGridDim, aBlockDim, 0)
{

}

CubicResampleKernel2D::CubicResampleKernel2D(CUmodule aModule)
        : Cuda::CudaKernel("sample2D", aModule, make_dim3(1,1,1), make_dim3(32, 8, 1), 0)
{

}

float CubicResampleKernel2D::operator()(CudaTextureObject2D& inimage,
                                        CudaPitchedDeviceVariable& outimage)
{
    CUtexObject texObj = inimage.GetTexObject();
    CUdeviceptr out_ptr = outimage.GetDevicePtr();
    size_t pitch = outimage.GetPitch();
    dim3 imdim = make_dim3(outimage.GetWidth(), outimage.GetHeight(), 0);
    size_t offset = 0;

    void** arglist = (void**)new void*[5];

    arglist[0] = &texObj;
    arglist[1] = &out_ptr;
    arglist[2] = &imdim;
    arglist[3] = &pitch;
    arglist[4] = &offset;

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

float CubicResampleKernel2D::operator()(CudaTextureObject2D& inimage,
                                        CudaDeviceVariable& outimage,
                                        int width, int height, int z)
{
    CUtexObject texObj = inimage.GetTexObject();
    CUdeviceptr out_ptr = outimage.GetDevicePtr();
    dim3 imdim = make_dim3(width, height, 0);
    size_t pitch = width * sizeof(float);
    // ptr offset
    size_t offset = width * height * z * sizeof(float);

    void** arglist = (void**)new void*[5];

    arglist[0] = &texObj;
    arglist[1] = &out_ptr;
    arglist[2] = &imdim;
    arglist[3] = &pitch;
    arglist[4] = &offset;

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

CubicResampleKernel3D::CubicResampleKernel3D(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : Cuda::CudaKernel("sample3D", aModule, aGridDim, aBlockDim, 0)
{

}

CubicResampleKernel3D::CubicResampleKernel3D(CUmodule aModule)
        : Cuda::CudaKernel("sample3D", aModule, make_dim3(1,1,1), make_dim3(32, 8, 1), 0)
{

}

float CubicResampleKernel3D::operator()(CudaTextureObject3D& involume,
                                        CudaSurfaceObject3D& outvolume,
                                        Volume<float>* vol,
                                        bool addToExisting)
{
    CUtexObject texVol = involume.GetTexObject();
    CUsurfObject outvol = outvolume.GetSurfObject();
    float3 volDim = vol->GetDimension();

    void** arglist = (void**)new void*[4];

    arglist[0] = &texVol;
    arglist[1] = &outvol;
    arglist[2] = &volDim;
    arglist[3] = &addToExisting;

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

Add3DKernel::Add3DKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : Cuda::CudaKernel("add3D", aModule, aGridDim, aBlockDim, 0)
{

}

Add3DKernel::Add3DKernel(CUmodule aModule)
        : Cuda::CudaKernel("add3D", aModule, make_dim3(1,1,1), make_dim3(8, 8, 4), 0)
{

}

float Add3DKernel::operator()(CudaSurfaceObject3D& involume,
                              CudaSurfaceObject3D& outvolume,
                              uint3 volDim,
                              float scaleFactor)
{
    CUsurfObject inVol = involume.GetSurfObject();
    CUsurfObject outVol = outvolume.GetSurfObject();

    void** arglist = (void**)new void*[4];

    arglist[0] = &inVol;
    arglist[1] = &outVol;
    arglist[2] = &volDim;
    arglist[3] = &scaleFactor;

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

Set3DKernel::Set3DKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : Cuda::CudaKernel("set3D", aModule, aGridDim, aBlockDim, 0)
{

}

Set3DKernel::Set3DKernel(CUmodule aModule)
        : Cuda::CudaKernel("set3D", aModule, make_dim3(1,1,1), make_dim3(8, 8, 4), 0)
{

}

float Set3DKernel::operator()(CudaSurfaceObject3D& inoutvolume,
                              float value,
                              uint3 voldim)
{
    CUsurfObject inoutVol = inoutvolume.GetSurfObject();

    void** arglist = (void**)new void*[3];

    arglist[0] = &inoutVol;
    arglist[1] = &value;
    arglist[2] = &voldim;

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


Add3DMaskedKernel::Add3DMaskedKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : Cuda::CudaKernel("add3Dmasked", aModule, aGridDim, aBlockDim, 0)
{

}

Add3DMaskedKernel::Add3DMaskedKernel(CUmodule aModule)
        : Cuda::CudaKernel("add3Dmasked", aModule, make_dim3(1,1,1), make_dim3(8, 8, 4), 0)
{

}

float Add3DMaskedKernel::operator()(CudaSurfaceObject3D& involume,
                                    CudaSurfaceObject3D& outvolume,
                                    CudaSurfaceObject3D& maskvolume,
                                    uint3 volDim,
                                    float scaleFactor)
{
    CUsurfObject inVol = involume.GetSurfObject();
    CUsurfObject outVol = outvolume.GetSurfObject();
    CUsurfObject maskVol = maskvolume.GetSurfObject();

    void** arglist = (void**)new void*[5];

    arglist[0] = &inVol;
    arglist[1] = &outVol;
    arglist[2] = &maskVol;
    arglist[3] = &volDim;
    arglist[4] = &scaleFactor;

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


Mask3DKernel::Mask3DKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : Cuda::CudaKernel("mask3D", aModule, aGridDim, aBlockDim, 0)
{

}

Mask3DKernel::Mask3DKernel(CUmodule aModule)
        : Cuda::CudaKernel("mask3D", aModule, make_dim3(1,1,1), make_dim3(8, 8, 4), 0)
{

}

float Mask3DKernel::operator()(CudaSurfaceObject3D& involume,
                               CudaSurfaceObject3D& outvolume,
                               CudaSurfaceObject3D& maskvolume,
                               uint3 volDim)
{
    CUsurfObject inVol = involume.GetSurfObject();
    CUsurfObject outVol = outvolume.GetSurfObject();
    CUsurfObject maskVol = maskvolume.GetSurfObject();

    void** arglist = (void**)new void*[4];

    arglist[0] = &inVol;
    arglist[1] = &outVol;
    arglist[2] = &maskVol;
    arglist[3] = &volDim;

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


Div3DMaskKernel::Div3DMaskKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : Cuda::CudaKernel("div3Dmask", aModule, aGridDim, aBlockDim, 0)
{

}

Div3DMaskKernel::Div3DMaskKernel(CUmodule aModule)
        : Cuda::CudaKernel("div3Dmask", aModule, make_dim3(1,1,1), make_dim3(8, 8, 4), 0)
{

}

float Div3DMaskKernel::operator()(CudaSurfaceObject3D& involume,
                                  CudaSurfaceObject3D& outvolume,
                                  CudaSurfaceObject3D& maskvolume,
                                  float divVal,
                                  uint3 volDim)
{
    CUsurfObject inVol = involume.GetSurfObject();
    CUsurfObject outVol = outvolume.GetSurfObject();
    CUsurfObject maskVol = maskvolume.GetSurfObject();

    void** arglist = (void**)new void*[5];

    arglist[0] = &inVol;
    arglist[1] = &outVol;
    arglist[2] = &maskVol;
    arglist[3] = &divVal;
    arglist[4] = &volDim;

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


Mask3DTransformKernel::Mask3DTransformKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : Cuda::CudaKernel("mask3Dtrans", aModule, aGridDim, aBlockDim, 0)
{

}

Mask3DTransformKernel::Mask3DTransformKernel(CUmodule aModule)
        : Cuda::CudaKernel("mask3Dtrans", aModule, make_dim3(1,1,1), make_dim3(8, 8, 4), 0)
{

}

float Mask3DTransformKernel::operator()(CudaSurfaceObject3D& volume,
                                        CudaTextureObject3D& maskTex,
                                        float4x4 transformMatrix,
                                        uint3 volDim,
                                        uint3 offset)
{
    CUsurfObject vol_obj = volume.GetSurfObject();
    CUtexObject mask_tex = maskTex.GetTexObject();

    void** arglist = (void**)new void*[5];

    arglist[0] = &vol_obj;
    arglist[1] = &mask_tex;
    arglist[2] = &transformMatrix;
    arglist[3] = &volDim;
    arglist[4] = &offset;

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

Add3DTransformKernel::Add3DTransformKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : Cuda::CudaKernel("add3Dtrans", aModule, aGridDim, aBlockDim, 0)
{

}

Add3DTransformKernel::Add3DTransformKernel(CUmodule aModule)
        : Cuda::CudaKernel("add3Dtrans", aModule, make_dim3(1,1,1), make_dim3(8, 8, 4), 0)
{

}

float Add3DTransformKernel::operator()(CudaSurfaceObject3D& volume,
                                       CudaTextureObject3D& maskTex,
                                       float4x4 transformMatrix,
                                       uint3 volDim,
                                       uint3 offset)
{
    CUsurfObject vol_obj = volume.GetSurfObject();
    CUtexObject mask_tex = maskTex.GetTexObject();

    void** arglist = (void**)new void*[5];

    arglist[0] = &vol_obj;
    arglist[1] = &mask_tex;
    arglist[2] = &transformMatrix;
    arglist[3] = &volDim;
    arglist[4] = &offset;

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


Norm3DOverlapKernel::Norm3DOverlapKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : Cuda::CudaKernel("norm3DOverlap", aModule, aGridDim, aBlockDim, 0)
{

}

Norm3DOverlapKernel::Norm3DOverlapKernel(CUmodule aModule)
        : Cuda::CudaKernel("norm3DOverlap", aModule, make_dim3(1,1,1), make_dim3(8, 8, 4), 0)
{

}

float Norm3DOverlapKernel::operator()(CudaSurfaceObject3D& volume,
                                      uint3 volDim)
{
    CUsurfObject vol_obj = volume.GetSurfObject();

    void** arglist = (void**)new void*[2];

    arglist[0] = &vol_obj;
    arglist[1] = &volDim;

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


Multiplicity3DKernel::Multiplicity3DKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : Cuda::CudaKernel("multiplicity", aModule, aGridDim, aBlockDim, 0)
{

}

Multiplicity3DKernel::Multiplicity3DKernel(CUmodule aModule)
        : Cuda::CudaKernel("multiplicity", aModule, make_dim3(1,1,1), make_dim3(4, 4, 4), 0)
{

}

float Multiplicity3DKernel::operator()(CudaDeviceVariable& outvolume,
                                       uint3 volDim,
                                       float3x3 Msys)
{
    CUsurfObject outvol = outvolume.GetDevicePtr();

    void** arglist = (void**)new void*[3];

    arglist[0] = &outvol;
    arglist[1] = &volDim;
    arglist[2] = &Msys;

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

MultNorm1DKernel::MultNorm1DKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : Cuda::CudaKernel("multNorm1D", aModule, aGridDim, aBlockDim, 0)
{

}

MultNorm1DKernel::MultNorm1DKernel(CUmodule aModule)
        : Cuda::CudaKernel("multNorm1D", aModule, make_dim3(1,1,1), make_dim3(32, 1, 1), 0)
{

}

float MultNorm1DKernel::operator()(CudaDeviceVariable& line,
                                   CudaDeviceVariable& lineMult,
                                   uint length,
                                   float threshold)
{
    CUdeviceptr line_ptr = line.GetDevicePtr();
    CUdeviceptr line_mult_ptr = lineMult.GetDevicePtr();

    void** arglist = (void**)new void*[4];

    arglist[0] = &line_ptr;
    arglist[1] = &line_mult_ptr;
    arglist[2] = &length;
    arglist[3] = &threshold;

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

MultNorm1DcompKernel::MultNorm1DcompKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : Cuda::CudaKernel("multNorm1Dcomp", aModule, aGridDim, aBlockDim, 0)
{

}

MultNorm1DcompKernel::MultNorm1DcompKernel(CUmodule aModule)
        : Cuda::CudaKernel("multNorm1Dcomp", aModule, make_dim3(1,1,1), make_dim3(32, 1, 1), 0)
{

}

float MultNorm1DcompKernel::operator()(CudaDeviceVariable& lineFFT,
                                       CudaDeviceVariable& lineMult,
                                       uint length,
                                       float threshold)
{
    CUdeviceptr linefft_ptr = lineFFT.GetDevicePtr();
    CUdeviceptr line_mult_ptr = lineMult.GetDevicePtr();

    void** arglist = (void**)new void*[4];

    arglist[0] = &linefft_ptr;
    arglist[1] = &line_mult_ptr;
    arglist[2] = &length;
    arglist[3] = &threshold;

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

MultNorm3DKernel::MultNorm3DKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : Cuda::CudaKernel("multNorm3D", aModule, aGridDim, aBlockDim, 0)
{

}

MultNorm3DKernel::MultNorm3DKernel(CUmodule aModule)
        : Cuda::CudaKernel("multNorm3D", aModule, make_dim3(1,1,1), make_dim3(4, 4, 4), 0)
{

}

float MultNorm3DKernel::operator()(CudaDeviceVariable& vol,
                                   CudaDeviceVariable& volMult,
                                   uint3 volDim,
                                   float threshold)
{
    CUdeviceptr vol_ptr = vol.GetDevicePtr();
    CUdeviceptr vol_mult_ptr = volMult.GetDevicePtr();

    void** arglist = (void**)new void*[4];

    arglist[0] = &vol_ptr;
    arglist[1] = &vol_mult_ptr;
    arglist[2] = &volDim;
    arglist[3] = &threshold;

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

MultNorm3DcompKernel::MultNorm3DcompKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : Cuda::CudaKernel("multNorm3Dcomp", aModule, aGridDim, aBlockDim, 0)
{

}

MultNorm3DcompKernel::MultNorm3DcompKernel(CUmodule aModule)
        : Cuda::CudaKernel("multNorm3Dcomp", aModule, make_dim3(1,1,1), make_dim3(4, 4, 4), 0)
{

}

float MultNorm3DcompKernel::operator()(CudaDeviceVariable& volFFT,
                                       CudaDeviceVariable& volMult,
                                       uint3 volDim,
                                       float threshold)
{
    CUdeviceptr vol_fft = volFFT.GetDevicePtr();
    CUdeviceptr vol_mult = volMult.GetDevicePtr();

    void** arglist = (void**)new void*[4];

    arglist[0] = &vol_fft;
    arglist[1] = &vol_mult;
    arglist[2] = &volDim;
    arglist[3] = &threshold;

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