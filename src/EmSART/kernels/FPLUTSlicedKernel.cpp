//
// Created by uermel on 11/19/21.
//

#include "FPLUTSlicedKernel.h"

using namespace Cuda;

FPLUTSlicedKernel::FPLUTSlicedKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("forwardProjectionLUTNoDivSliced", aModule, aGridDim, aBlockDim, 0)
{

}

FPLUTSlicedKernel::FPLUTSlicedKernel(CUmodule aModule)
        : CudaKernel("forwardProjectionLUTNoDivSliced", aModule, make_dim3(1, 1, 1), make_dim3(4, 4, 4), 0)
{

}

void FPLUTSlicedKernel::MatrixVector3Mul(float4x4 M, float3& v, float2& erg)
{
    erg.x = M.m[0].x * v.x + M.m[0].y * v.y + M.m[0].z * v.z + 1.f * M.m[0].w;
    erg.y = M.m[1].x * v.x + M.m[1].y * v.y + M.m[1].z * v.z + 1.f * M.m[1].w;
}

int2 FPLUTSlicedKernel::computeSupport(Volume<float> &vol,
                                       Projection &proj,
                                       int index,
                                       int3 voxelBlockDim,
                                       int supportsize,
                                       int maxOversample)
{
    int2 rangeMax;
    float2 center2D;
    float2 borderMinV;
    float2 borderMaxV;
    float2 borderMin = make_float2(proj.GetWidth(), proj.GetHeight());
    float2 borderMax = make_float2(0.f, 0.f);;
    float3 center3D;

    float3 voxelsize = vol.GetVoxelSize();

    float matrix[16];
    proj.GetDetectorMatrix(index, matrix, 1);

    float supporthalf = supportsize * 0.5f;

    for (int ix = 0; ix < voxelBlockDim.x; ix++){
        for (int iy = 0; ix < voxelBlockDim.y; ix++){
            for (int iz = 0; ix < voxelBlockDim.z; ix++){
                center3D = make_float3((float)ix * voxelsize.x + voxelsize.x * 0.5f,
                                       (float)iy * voxelsize.y + voxelsize.y * 0.5f,
                                       (float)iz * voxelsize.z + voxelsize.z * 0.5f);

                MatrixVector3Mul(*(float4x4*)matrix, center3D, center2D);

                borderMinV.x = floorf(center2D.x - supporthalf);
                borderMinV.y = floorf(center2D.y - supporthalf);
                borderMaxV.x = ceilf(center2D.x + supporthalf);
                borderMaxV.y = ceilf(center2D.y + supporthalf);

                borderMin.x = fminf(borderMin.x, borderMinV.x);
                borderMin.y = fminf(borderMin.y, borderMinV.y);
                borderMax.x = fmaxf(borderMax.x, borderMaxV.x);
                borderMax.y = fmaxf(borderMax.y, borderMaxV.y);
            }
        }
    }

    printf("support: %f %f\n", borderMin.x, borderMin.y);
    printf("support: %f %f\n", borderMax.x, borderMax.y);
    printf("diff: %f %f %f %f\n", borderMax.x - borderMin.x, borderMax.y - borderMin.y, ceilf(borderMax.x - borderMin.x), ceilf(borderMax.y - borderMin.y));

    // +1 for safety
    rangeMax.x = (int)roundf(borderMax.x - borderMin.x + 1) * maxOversample;
    rangeMax.y = (int)roundf(borderMax.y - borderMin.y + 1) * maxOversample;

    return rangeMax;
}

float FPLUTSlicedKernel::operator()(int proj_x,
                                    int proj_y,
                                    float lambda,
                                    float maxOverSample,
                                    CudaDeviceVariable& projection,
                                    CudaTextureObject2D& LUT,
                                    CudaSurfaceObject3D& volume,
                                    float tmin,
                                    float tmax,
                                    Volume<float>* vol)
{
    CUdeviceptr projPtr = projection.GetDevicePtr();
    CUtexObject LUTObj = LUT.GetTexObject();
    CUsurfObject volObj = volume.GetSurfObject();
    float maxOverSampleInv = 1.f/maxOverSample;
    float maxOverSampleInvH = 0.5f * 1.f/maxOverSample;
    //float supporthalf = support * 0.5f;
    int OS = (int) maxOverSample;
    //size_t stride = projection.GetPitch();

    int gridX = (int)ceilf(vol->GetDimension().x/2);
    int gridY = (int)ceilf(vol->GetDimension().y/2);
    int gridZ = (int)ceilf(vol->GetDimension().z/2);

    //SetGridDimensions(make_dim3(1, 1, 1));
    //SetBlockDimensions(make_dim3(1, 1, 1));
    SetGridDimensions(make_dim3(gridX, gridY, gridZ));
    SetBlockDimensions(make_dim3(2, 2, 2));
    //SetDynamicSharedMemory(sizeof(float) * 24 * 2 * 2 * 2);

    printf("Voxel Block Kernel\n");
    printf("\nmaxOversampleInv: %f\n", maxOverSampleInv);
    printf("\nmaxOverSampleInvH: %f\n", maxOverSampleInvH);
    //printf("\nsupporthalf: %f\n", supporthalf);
    //printf("\nLUTstepinv: %f\n", LUTstepinv);
    //printf("\nLUTcenter: %f\n", LUTcenter);

    printf("\nGridX: %d\n", gridX);
    printf("\nGridY: %d\n", gridY);
    printf("\nGridZ: %d\n", gridZ);
    //printf("\nBlockY: %d\n", blockY);

    void** arglist = (void**)new void*[11];

    arglist[0] = &proj_x;
    arglist[1] = &proj_y;
    arglist[2] = &lambda;
    arglist[3] = &maxOverSampleInv;
    arglist[4] = &maxOverSampleInvH;
    arglist[5] = &projPtr;
    arglist[6] = &LUTObj;
    arglist[7] = &volObj;
    arglist[8] = &tmin;
    arglist[9] = &tmax;
    arglist[10] = &OS;

    float ms;

    CUevent eventStart;
    CUevent eventEnd;
    CUstream stream = 0;
    cudaSafeCall(cuEventCreate(&eventStart, CU_EVENT_BLOCKING_SYNC));
    cudaSafeCall(cuEventCreate(&eventEnd, CU_EVENT_BLOCKING_SYNC));

    cudaSafeCall(cuEventRecord(eventStart, stream));
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 100000000);
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

void SetConstantValues(FPLUTSlicedKernel& kernel,
                       Volume<float>& vol,
                       Projection& proj,
                       int index,
                       int subVol,
                       Matrix<float>& m,
                       Matrix<float>& mInv,
                       float LUTstepinv,
                       float LUTcenter,
                       int support,
                       int maxOverSample,
                       int sliceNumber,
                       float entry,
                       float sliceThickness)
{
    //printf("==============================================================\n");
    //printf("START Back Projection Constants FLOAT START\n");
    //printf("==============================================================\n");
    //Set constant values
    int2 supportTest = kernel.computeSupport(vol, proj, index, make_int3(1, 1, 1), support, (int)maxOverSample);
    printf("support: %i %i\n", supportTest.x, supportTest.y);
    kernel.SetConstantValue("c_blockSupportSize", &supportTest);

    kernel.SetConstantValue("c_LUTstepinv", &LUTstepinv);
    kernel.SetConstantValue("c_LUTcenter", &LUTcenter);
    kernel.SetConstantValue("c_voxelSupportSize", &support);

    float sh = (float)support * 0.5f;
    kernel.SetConstantValue("c_voxelSupportHalf", &sh);
    //kernel.SetConstantValue("c_oversampleFactor", &maxOverSample);

    printf("c_entry: %f\n", entry);
    printf("c_sliceThickness: %f\n", sliceThickness);
    printf("c_sliceNumber: %i\n", sliceNumber);

    kernel.SetConstantValue("c_entry", &entry);
    kernel.SetConstantValue("c_sliceThickness", &sliceThickness);
    kernel.SetConstantValue("c_sliceNumber", &sliceNumber);

    float3 temp = proj.GetNormalVector(index);
//    temp.x = -temp.x;
//    temp.y = -temp.y;
//    temp.z = -temp.z;
    kernel.SetConstantValue("c_projNorm", &temp);
    //printf("c_projNorm: %f, %f, %f\n", temp.x, temp.y, temp.z);

    temp = proj.GetPosition(index);
    kernel.SetConstantValue("c_detektor", &temp);
    //printf("c_detektor: %f, %f, %f\n", temp.x, temp.y, temp.z);

    temp = proj.GetPixelUPitch(index);
    kernel.SetConstantValue("c_uPitch", &temp);
    //printf("c_uPitch: %f, %f, %f\n", temp.x, temp.y, temp.z);

    temp = proj.GetPixelVPitch(index);
    kernel.SetConstantValue("c_vPitch", &temp);
    //printf("c_vPitch: %f, %f, %f\n", temp.x, temp.y, temp.z);

    temp = vol.GetSubVolumeBBoxRcp(subVol);
    kernel.SetConstantValue("c_volumeBBoxRcp", &temp);
    //printf("c_volumeBBoxRcp: %f, %f, %f\n", temp.x, temp.y, temp.z);

    temp = vol.GetVolumeBBoxMin();
    kernel.SetConstantValue("c_bBoxMinComplete", &temp);
    //printf("c_bBoxMinComplete: %f, %f, %f\n", temp.x, temp.y, temp.z);

    temp = vol.GetVolumeBBoxMax();
    kernel.SetConstantValue("c_bBoxMaxComplete", &temp);
    //printf("c_bBoxMaxComplete: %f, %f, %f\n", temp.x, temp.y, temp.z);

    temp = vol.GetDimension();
    kernel.SetConstantValue("c_volumeDimComplete", &temp);
    //printf("c_volumeDimComplete: %f, %f, %f\n", temp.x, temp.y, temp.z);

    temp = vol.GetSubVolumeDimension(subVol);
    kernel.SetConstantValue("c_volumeDim", &temp);
    //printf("c_volumeDim: %f, %f, %f\n", temp.x, temp.y, temp.z);

    temp = vol.GetVoxelSize();
    kernel.SetConstantValue("c_voxelSize", &temp);
    //printf("c_voxelSize: %f, %f, %f\n", temp.x, temp.y, temp.z);

    float t = vol.GetSubVolumeZShift(subVol);
    kernel.SetConstantValue("c_zShiftForPartialVolume", &t);
    //printf("c_zShiftForPartialVolume: %f\n", t);

    int volquat = (int)vol.GetDimension().x / 4;
    kernel.SetConstantValue("c_volumeDim_x_quarter", &volquat);
    //printf("c_volumeDim_x_quarter: %i\n", volquat);

    temp = vol.GetSubVolumeBBoxMin(subVol);
    kernel.SetConstantValue("c_bBoxMin", &temp);
    //printf("c_bBoxMin: %f, %f, %f\n", temp.x, temp.y, temp.z);

    float matrix[16];
    proj.GetDetectorMatrix(index, matrix, 1);
    kernel.SetConstantValue("c_DetectorMatrix", matrix);
    //printf("c_DetectorMatrix:\n %f, %f, %f, %f,\n %f, %f, %f, %f,\n %f, %f, %f, %f,\n %f, %f, %f, %f,\n",
    //       matrix[0],matrix[1],matrix[2],matrix[3],
    //       matrix[4],matrix[5],matrix[6],matrix[7],
    //       matrix[8],matrix[9],matrix[10],matrix[11],
    //       matrix[12],matrix[13],matrix[14],matrix[15]);

    kernel.SetConstantValue("c_magAniso", m.GetData());
    //printf("c_projNorm: %f, %f, %f\n", temp.x, temp.y, temp.z);
    kernel.SetConstantValue("c_magAnisoInv", mInv.GetData());
    //printf("c_projNorm: %f, %f, %f\n", temp.x, temp.y, temp.z);

    //printf("==============================================================\n");
    //printf("END Back Projection Constants FLOAT END\n");
    //printf("==============================================================\n");
}