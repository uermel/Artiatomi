//
// Created by uermel on 11/19/21.
//

#include "FPDistOrthoKernel.h"

using namespace Cuda;

FPDistOrthoKernel::FPDistOrthoKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("fpOrthoDist", aModule, aGridDim, aBlockDim, 0)
{

}

FPDistOrthoKernel::FPDistOrthoKernel(CUmodule aModule)
        : CudaKernel("fpOrthoDist", aModule, make_dim3(1, 1, 1), make_dim3(4, 4, 4), 0)
{

}

float FPDistOrthoKernel::operator()(uint2 projDim,
                                    uint3 volDim,
                                    ctfImageConstants imageConstants,
                                    float4x4 systemMatrix,
                                    CudaDeviceVariable& distanceMap)
{
    CUdeviceptr projPtr = distanceMap.GetDevicePtr();
    CUsurfObject volObj = 0;

    float lambda = 0;
    float ctfCenter = 0;

    int gridX = (int)ceilf((float)volDim.x/2);
    int gridY = (int)ceilf((float)volDim.y/2);
    int gridZ = (int)ceilf((float)volDim.z/2);


    SetGridDimensions(make_dim3(gridX, gridY, gridZ));
    SetBlockDimensions(make_dim3(2, 2, 2));

//    printf("Dist Kernel\n");
//    printf("\nGridX: %d\n", gridX);
//    printf("\nGridY: %d\n", gridY);
//    printf("\nGridZ: %d\n", gridZ);

    void** arglist = (void**)new void*[6];

    arglist[0] = &projDim;
    arglist[1] = &volDim;
    arglist[2] = &imageConstants;
    arglist[3] = &systemMatrix;
    arglist[4] = &projPtr;
    arglist[5] = &volObj;

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

void SetConstantValues(FPDistOrthoKernel& kernel,
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
    int2 supportTest = make_int2(0, 0);//kernel.computeSupport(vol, proj, index, make_int3(1, 1, 1), 4, (int)maxOverSample);
    printf("support: %i %i\n", supportTest.x, supportTest.y);
    kernel.SetConstantValue("c_blockSupportSize", &supportTest);

    kernel.SetConstantValue("c_LUTstepinv", &LUTstepinv);
    kernel.SetConstantValue("c_LUTcenter", &LUTcenter);
    int msupp = 4;
    kernel.SetConstantValue("c_voxelSupportSize", &msupp);

    //float sh = (float)support * 0.5f;
    float sh = (float)msupp * 0.5f;
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
