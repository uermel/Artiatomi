//
// Created by uermel on 10/1/21.
//

#include "BPLUTBWCGKernel.h"

using namespace Cuda;

BPLUTBWCGKernel::BPLUTBWCGKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("backProjectionLUTBlockwiseCG", aModule, aGridDim, aBlockDim, 0)
{

}

BPLUTBWCGKernel::BPLUTBWCGKernel(CUmodule aModule)
        : CudaKernel("backProjectionLUTBlockwiseCG", aModule, make_dim3(1, 1, 1), make_dim3(4, 4, 4), 0)
{

}

float BPLUTBWCGKernel::operator()(int proj_x,
                                int proj_y,
                                float lambda,
                                float maxOverSample,
        //Cuda::CudaSurfaceObject2D& projection,
                                Cuda::CudaPitchedDeviceVariable& projection,
                                Cuda::CudaTextureObject2D& LUT,
                                Cuda::CudaSurfaceObject3D& volume,
                                float tmin,
                                float tmax,
                                float support,
                                float LUTstepinv,
                                float LUTcenter,
                                Volume<float>* vol)
{
    CUdeviceptr projPtr = projection.GetDevicePtr();
    CUtexObject LUTObj = LUT.GetTexObject();
    CUsurfObject volObj = volume.GetSurfObject();
    float maxOverSampleInv = 1.f/maxOverSample;
    float maxOverSampleInvH = 0.5f * 1.f/maxOverSample;
    float supporthalf = support * 0.5f;
    int OS = (int) maxOverSample;
    size_t stride = projection.GetPitch();

    // Block, Grid and Memory
    int gridX = vol->GetSubVolumeSizeInVoxels(0);
    int esize = (int)((ceil(support/2) * 2 + 1)*maxOverSample);

    int blockY = esize;
    int blockZ = esize;

    if(blockY % 8 > 0)
        blockY = (blockY/8 + 1) * 8;//blockY + (blockY % 8);

    if (blockZ % 2 > 0)
        blockZ = blockZ + 1;


    SetGridDimensions(make_dim3(1, 1, 1));
    SetBlockDimensions(make_dim3(1, blockY, blockZ));
    SetDynamicSharedMemory(sizeof(float) * blockY * blockZ);

    printf("\nmaxOversampleInv: %f\n", maxOverSampleInv);
    printf("\nmaxOverSampleInvH: %f\n", maxOverSampleInvH);
    printf("\nsupporthalf: %f\n", supporthalf);
    printf("\nLUTstepinv: %f\n", LUTstepinv);
    printf("\nLUTcenter: %f\n", LUTcenter);

    printf("\nGridX: %d\n", gridX);
    printf("\nBlockY: %d\n", blockY);

    void** arglist = (void**)new void*[16];

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
    arglist[10] = &supporthalf;
    arglist[11] = &LUTstepinv;
    arglist[12] = &LUTcenter;
    arglist[13] = &OS;
    arglist[14] = &stride;
    arglist[15] = &blockY;

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

void SetConstantValues(BPLUTBWCGKernel& kernel, Volume<float>& vol, Projection& proj, int index, int subVol, Matrix<float>& m, Matrix<float>& mInv)
{
    //printf("==============================================================\n");
    //printf("START Back Projection Constants FLOAT START\n");
    //printf("==============================================================\n");
    //Set constant values
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