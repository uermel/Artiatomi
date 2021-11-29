//  Copyright (c) 2018, Michael Kunz and Frangakis Lab, BMLS,
//  Goethe University, Frankfurt am Main.
//  All rights reserved.
//  http://kunzmi.github.io/Artiatomi
//
//  This file is part of the Artiatomi package.
//
//  Artiatomi is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  Artiatomi is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with Artiatomi. If not, see <http://www.gnu.org/licenses/>.
//
////////////////////////////////////////////////////////////////////////


/**********************************************
*
* CUDA SART FRAMEWORK
* 2009,2010 Michael Kunz, Lukas Marsalek
*
*
* BackProjectionSquareOS.cu
* CVR back projection kernel with squared
* oversampling pattern
*
**********************************************/
#ifndef BACKPROJECTIONSQUAREOS_CU
#define BACKPROJECTIONSQUAREOS_CU

#define _SIZE_T_DEFINED
#ifndef __CUDACC__
#define __CUDACC__
#endif
#ifndef __cplusplus
#define __cplusplus
#endif

#include <cuda.h>
#include <stdio.h>
#include "cutil.h"
#include "cutil_math.h"
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>
#include "float.h"
#include <builtin_types.h>
#include <vector_functions.h>
#include <device_functions.h>
#include "cubic_interpolation/cubicTex2D.cu"

#include "Constants.h"
#include "DeviceVariables.cuh"
#include <cuda_fp16.h>
#include <cooperative_groups.h>


//#include <curand_kernel.h>

//#define CONST_LENGTH_MODE
#define PRECISE_LENGTH_MODE

#define SM20 1
#if __CUDA_ARCH__ >= 200
//#warning compiling for SM20
#else
#if __CUDA_ARCH__ >= 130
//#warning compiling for SM13
#endif
#endif

using namespace cooperative_groups;

// transform vector by matrix
__device__
void MatrixVector3Mul(float4x4 M, float3* v)
{
    float3 erg;
    erg.x = M.m[0].x * v->x + M.m[0].y * v->y + M.m[0].z * v->z + 1.f * M.m[0].w;
    erg.y = M.m[1].x * v->x + M.m[1].y * v->y + M.m[1].z * v->z + 1.f * M.m[1].w;
    erg.z = M.m[2].x * v->x + M.m[2].y * v->y + M.m[2].z * v->z + 1.f * M.m[2].w;
    *v = erg;
}

// transform vector by matrix
__device__
void MatrixVector3Mul(float4x4 M, float3& v, float2& erg)
{
    erg.x = M.m[0].x * v.x + M.m[0].y * v.y + M.m[0].z * v.z + 1.f * M.m[0].w;
    erg.y = M.m[1].x * v.x + M.m[1].y * v.y + M.m[1].z * v.z + 1.f * M.m[1].w;
}

extern "C"
__global__
void backProjectionLUT(int proj_x,
                       int proj_y,
                       float lambda,
                       float maxOverSampleInv,
                       float maxOverSampleInvH,
                       float* projection,
                       CUtexObject LUT,
                       CUsurfObject volume,
                       float tmin,
                       float tmax,
                       float supporthalf,
                       float LUTstepinv,
                       float LUTcenter,
                       int maxOverSample, size_t stride) {

    float2 pixel;
    float2 borderMin;
    float2 borderMax;
    float3 corner3D;
    float3 center3D;

    // Vol coords
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Drop threads outside volume
    if (x >= c_volumeDim.x || y >= c_volumeDim.y || z >= c_volumeDim.z) return;

    // Voxel corner,
    corner3D = make_float3(c_bBoxMin.x + x * c_voxelSize.x, c_bBoxMin.y + y * c_voxelSize.y, c_bBoxMin.z + z * c_voxelSize.z);
    // Voxel center
    center3D = make_float3(corner3D.x + c_voxelSize.x * 0.5f, corner3D.y + c_voxelSize.y * 0.5f, corner3D.z + c_voxelSize.z * 0.5f);

    // Distance to projection plane (for CTF slices)
    float t;
    t = (c_projNorm.x * corner3D.x + c_projNorm.y * corner3D.y + c_projNorm.z * corner3D.z);
    t += (-c_projNorm.x * c_detektor.x - c_projNorm.y * c_detektor.y - c_projNorm.z * c_detektor.z);
    t = abs(t);

    // Drop threads outside CTF slice
    if (!(t >= tmin && t < tmax)) return;

    // Project Center
    // TODO: Account for mag anisotropy
    MatrixVector3Mul(c_DetectorMatrix, center3D, pixel);

    // Maximum extent of the support clamped by projection size
    borderMin.x = floor(pixel.x - supporthalf);
    borderMin.y = floor(pixel.y - supporthalf);
    borderMax.x = ceil(pixel.x + supporthalf);
    borderMax.y = ceil(pixel.y + supporthalf);

    borderMin.x = fminf(fmaxf(borderMin.x, 0), proj_x);
    borderMin.y = fminf(fmaxf(borderMin.y, 0), proj_y);
    borderMax.x = fminf(fmaxf(borderMax.x, 0), proj_x);
    borderMax.y = fminf(fmaxf(borderMax.y, 0), proj_y);

    // Known voxel value
    float accumulator = 0.f;
    surf3Dread(&accumulator, volume, x * 4, y, z);

    // Range to scan
    int rangeMaxX = (borderMax.x - borderMin.x) * maxOverSample;
    int rangeMaxY = (borderMax.y - borderMin.y) * maxOverSample;

//    # if __CUDA_ARCH__>=200
//            //printf("%d %d %d %f %f %f %f %d %d %f %f %f %f %f %f\n", x, y, z, borderMin.x, borderMin.y, borderMax.x, borderMax.y, rangeMaxX, rangeMaxY, pixel.x, pixel.y, center.x, center.y, center.z, accumulator);
//    #endif

    // Loop over potentially covered pixels
    for (int osy = 0; osy < rangeMaxY; osy++) {
        // Projected Voxel center in projection space
        float pixel_y = borderMin.y + osy * maxOverSampleInv;
        float disty = (pixel_y - pixel.y) * LUTstepinv + LUTcenter;

        // Projected Voxel center in oversampled projection space
        int idy = maxOverSample * borderMin.y + osy;

        for (int osx = 0; osx < rangeMaxX; osx++) {
            // Projected Voxel center in projection space
            float pixel_x = borderMin.x + osx * maxOverSampleInv;
            float distx = (pixel_x - pixel.x) * LUTstepinv + LUTcenter;

            // Projected Voxel center in oversampled projection space
            int idx = maxOverSample * borderMin.x + osx;

            // Weight from LUT
            float weight = tex2D<float>(LUT, distx, disty);

            // Value from image
            float val = *(((float*)((char*)projection + stride * idy)) + idx);

            // Accumulate
            accumulator += val * weight;

//# if __CUDA_ARCH__>=200
//            //printf("%f %f %i %i %f %f %f %f %f\n", pixel_x, pixel_y, idx, idy, distx, disty, weight, val, accumulator);
//#endif
        }
    }

    surf3Dwrite(accumulator * lambda, volume, x * 4, y, z);
}

extern __shared__ float sBuffer[];

__forceinline__ __device__ unsigned lane_id()
{
    unsigned ret;
    asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}

extern "C"
__global__
void backProjectionLUTBlockwiseDiff(int proj_x,
                       int proj_y,
                       float lambda,
                       float maxOverSampleInv,
                       float maxOverSampleInvH,
                       CUsurfObject projection,
                       CUtexObject LUT,
                       CUsurfObject volume,
                       float tmin,
                       float tmax,
                       float supporthalf,
                       float LUTstepinv,
                       float LUTcenter,
                       int maxOverSample,
                       size_t stride) {

    float2 center2D;
    float2 borderMin;
    float2 borderMax;
    float3 c_source;
    float3 corner;
    float3 center3D;

    // Vol coords
    const unsigned int x = blockIdx.x % (uint)c_volumeDim.x;
    const unsigned int y = (blockIdx.x - x)/(uint)c_volumeDim.x % (uint)c_volumeDim.y;
    const unsigned int z = ((blockIdx.x - x)/(uint)c_volumeDim.x-y)/(uint)c_volumeDim.y;
//    const unsigned int x = 31;//blockIdx.x % (uint)c_volumeDim.x;
//    const unsigned int y = 31;//(blockIdx.x - x)/(uint)c_volumeDim.x % (uint)c_volumeDim.y;
//    const unsigned int z = 31;//((blockIdx.x - x)/(uint)c_volumeDim.x-y)/(uint)c_volumeDim.y;

    // Full volume size
    //if (x >= c_volumeDim.x || y >= c_volumeDim.y || z >= c_volumeDim.z) return;

    // Voxel corner, center
    corner = make_float3(c_bBoxMin.x + x * c_voxelSize.x, c_bBoxMin.y + y * c_voxelSize.y, c_bBoxMin.z + z * c_voxelSize.z);
    center3D = make_float3(corner.x + c_voxelSize.x * 0.5f, corner.y + c_voxelSize.y * 0.5f, corner.z + c_voxelSize.z * 0.5f);

    // Distance to projection plane (for CTF slices)
    float t;
    t = (c_projNorm.x * corner.x + c_projNorm.y * corner.y + c_projNorm.z * corner.z);
    t += (-c_projNorm.x * c_detektor.x - c_projNorm.y * c_detektor.y - c_projNorm.z * c_detektor.z);
    t = abs(t);

    // Return if outside current CTF slice
    //if (!(t >= tmin && t < tmax)) return;

    // Project
    // TODO: Account for mag anisotropy
    MatrixVector3Mul(c_DetectorMatrix, center3D, center2D);

    // Image subscript coords (extent of support)
    float esize = blockDim.y;

    // Idx in oversampled image
    int idx = threadIdx.y - (int)ceilf(esize/2) + (int)roundf(center2D.x * maxOverSample);
    int idy = threadIdx.z - (int)ceilf(esize/2) + (int)roundf(center2D.y * maxOverSample);

    // Idx in projection
    float pixel_x = (float)idx/(float)maxOverSample;
    float pixel_y = (float)idy/(float)maxOverSample;




    //int pixel_x = threadIdx.y + (center2D.x * maxOverSample) - (supporthalf * maxOverSample);
    //int pixel_y = threadIdx.z + (center2D.y * maxOverSample) - (supporthalf * maxOverSample);

    // Shared buffer linear coords
    int linIdx = threadIdx.y + blockDim.y * threadIdx.z;

    // Distance from projected voxel
    float distx = (pixel_x - center2D.x) * LUTstepinv + LUTcenter;
    float disty = (pixel_y - center2D.y) * LUTstepinv + LUTcenter;

    // Clamp to full image size
    float val = 0.f;
    if (pixel_x < proj_x || pixel_y < proj_y || pixel_x >= 0 || pixel_y >= 0){
        // To shared mem
        sBuffer[linIdx] = *(((float*)((char*)projection + stride * idy)) + idx);
        //val = *(((float*)((char*)projection + stride * idy)) + idx) * weight;
    }
    else {
        sBuffer[linIdx] = 0.f;
        //val = 0.f;
    }

    // Weight from LUT
    sBuffer[linIdx] = sBuffer[linIdx] * tex2D<float>(LUT, distx, disty);

//# if __CUDA_ARCH__>=200
//    //printf("%u %u %u %d %d %f %f %d %f %f %f %f %f %f %u\n", x, y, z, idx, idy, pixel_x, pixel_y, linIdx, center2D.x, center2D.y, center3D.x, center3D.y, center3D.z, sBuffer[linIdx], lane_id());
//#endif

    __syncthreads();
    // Block-parallel reduction
    int blocksize = blockDim.y * blockDim.z;
    for (int size = blocksize/2; size>0; size/=2) {
        if (linIdx<size)
            sBuffer[linIdx] += sBuffer[linIdx+size];
        __syncthreads();
    }

    if (linIdx == 0) {
        float old = 0.f;
        surf3Dread(&old, volume, x * 4, y, z);
        surf3Dwrite(sBuffer[0] + old, volume, x * 4, y, z);
    }
}


extern "C"
__global__
void backProjectionLUTBlockwiseCG(int proj_x,
                                int proj_y,
                                float lambda,
                                float maxOverSampleInv,
                                float maxOverSampleInvH,
                                CUsurfObject projection,
                                CUtexObject LUT,
                                CUsurfObject volume,
                                float tmin,
                                float tmax,
                                float supporthalf,
                                float LUTstepinv,
                                float LUTcenter,
                                int maxOverSample,
                                size_t stride,
                                int esize) {

    float2 center2D;
    float2 borderMin;
    float2 borderMax;
    float3 c_source;
    float3 corner;
    float3 center3D;

    thread_block wholeBlock = this_thread_block();
    thread_block_tile<8> tile8 = tiled_partition<8>(wholeBlock);
    const uint pix_y = tile8.thread_rank();
    const uint pix_x = threadIdx.z;

    // Vol coords
    const unsigned int x = blockIdx.x % (uint)c_volumeDim.x;
    const unsigned int y = (blockIdx.x - x)/(uint)c_volumeDim.x % (uint)c_volumeDim.y;
    const unsigned int z = ((blockIdx.x - x)/(uint)c_volumeDim.x-y)/(uint)c_volumeDim.y;
//    const unsigned int x = 32;//blockIdx.x % (uint)c_volumeDim.x;
//    const unsigned int y = 32;//(blockIdx.x - x)/(uint)c_volumeDim.x % (uint)c_volumeDim.y;
//    const unsigned int z = 32;//((blockIdx.x - x)/(uint)c_volumeDim.x-y)/(uint)c_volumeDim.y;

//    # if __CUDA_ARCH__>=200
//            printf("%u %u %u\n", x, y, z);
//    #endif

    // Full volume size
    //if (x >= c_volumeDim.x || y >= c_volumeDim.y || z >= c_volumeDim.z) return;

    // Voxel corner, center
    corner = make_float3(c_bBoxMin.x + x * c_voxelSize.x, c_bBoxMin.y + y * c_voxelSize.y, c_bBoxMin.z + z * c_voxelSize.z);
    center3D = make_float3(corner.x + c_voxelSize.x * 0.5f, corner.y + c_voxelSize.y * 0.5f, corner.z + c_voxelSize.z * 0.5f);

    // Distance to projection plane (for CTF slices)
    float t;
    t = (c_projNorm.x * corner.x + c_projNorm.y * corner.y + c_projNorm.z * corner.z);
    t += (-c_projNorm.x * c_detektor.x - c_projNorm.y * c_detektor.y - c_projNorm.z * c_detektor.z);
    t = abs(t);

    // Return if outside current CTF slice
    //if (!(t >= tmin && t < tmax)) return;

    // Project
    // TODO: Account for mag anisotropy
    MatrixVector3Mul(c_DetectorMatrix, center3D, center2D);

    // Image subscript coords (extent of support)
    //float esize = blockDim.y;

    // Idx in oversampled image
    int idx = pix_x - (int)ceilf(esize/2) + (int)roundf(center2D.x * maxOverSample);
    int idy = pix_y - (int)ceilf(esize/2) + (int)roundf(center2D.y * maxOverSample);

    // Idx in projection
    float pixel_x = (float)idx/(float)maxOverSample;
    float pixel_y = (float)idy/(float)maxOverSample;

    // Idx in texture coords (+0.5)
    float texel_x = pixel_x + maxOverSampleInvH;
    float texel_y = pixel_y + maxOverSampleInvH;


    //int pixel_x = threadIdx.y + (center2D.x * maxOverSample) - (supporthalf * maxOverSample);
    //int pixel_y = threadIdx.z + (center2D.y * maxOverSample) - (supporthalf * maxOverSample);

    // Shared buffer linear coords
    int linIdx = pix_x + esize * pix_y;

    // Distance from projected voxel
    float distx = (pixel_x - center2D.x) * LUTstepinv + LUTcenter;
    float disty = (pixel_y - center2D.y) * LUTstepinv + LUTcenter;

    // Weight from LUT
    float weight = tex2D<float>(LUT, distx, disty);

    // Full image size
    float val = 0.f;
    if (pixel_x < proj_x || pixel_y < proj_y || pixel_x >= 0 || pixel_y >= 0){
        // To shared mem
        //sBuffer[linIdx] = *(((float*)((char*)projection + stride * idy)) + idx) * weight;
        val = *(((float*)((char*)projection + stride * idy)) + idx) * weight;
    }
    else {
        //sBuffer[linIdx] = 0.f;
        val = 0.f;
    }

    tile8.sync();
    //__syncthreads();

# if __CUDA_ARCH__>=200
    //printf("%u %u %u %d %d %f %f %d %f %f %f %f %f %f %f %u %u\n", x, y, z, idx, idy, pixel_x, pixel_y, linIdx, center2D.x, center2D.y, center3D.x, center3D.y, center3D.z, weight, sBuffer[linIdx], pix_x, pix_y);
#endif

    // Group-parallel reduction
    for (int size = tile8.size()/2; size>0; size/=2) {
        val += tile8.shfl_down(val, size);
    }

    if (tile8.thread_rank() == 0)
        atomicAdd(sBuffer, val);

    wholeBlock.sync();

    if (linIdx == 0) {
//        printf("%f", sBuffer[0]);

//        int blocksize = blockDim.y * blockDim.z;
//        float sum = 0;
//        for (int i = 0; i < blocksize; i++)
//        {
//            sum += sBuffer[i];
//        }

        float old = 0.f;
        surf3Dread(&old, volume, x * 4, y, z);
        surf3Dwrite(sBuffer[0] + old, volume, x * 4, y, z);
        //printf("%f", sum);
    }

    //__syncthreads();
}

extern __shared__ float sharedPtr[];

extern "C"
__global__
void backProjectionLUTVoxelBlock(int proj_x,
                       int proj_y,
                       float lambda,
                       float maxOverSampleInv,
                       float maxOverSampleInvH,
                       float* projection,
                       CUtexObject LUT,
                       CUsurfObject volume,
                       float tmin,
                       float tmax,
                       float supporthalf,
                       float LUTstepinv,
                       float LUTcenter,
                       int maxOverSample,
                       size_t stride) {

    float2 pixel;
    float2 borderMinV;
    float2 borderMaxV;
    float2 borderMin = make_float2(proj_x, proj_y);
    float2 borderMax = make_float2(0.f, 0.f);

    // Vol coords
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

//    const unsigned int x = 16;//blockIdx.x * blockDim.x + threadIdx.x;
//    const unsigned int y = 16;//blockIdx.y * blockDim.y + threadIdx.y;
//    const unsigned int z = 16;//blockIdx.z * blockDim.z + threadIdx.z;


    float ts[8];
    float accumulators[8];
    float2 suppCenters2D[8];
    float2 rangeMax[8];
    float3 centers3D[8];

//    int threadOffset = (threadIdx.x + 2 * threadIdx.y + 4 * threadIdx.z) * 72;
//    float* base = sharedPtr + threadOffset;
//    float* ts = base;
//    float* accumulators = base + 8;
//    float2* suppCenters2D = (float2*)(base + 16);
//    float2* rangeMax = (float2*)(base + 32);
//    float3* centers3D = (float3*)(base + 48);

    for (int ix = 0; ix < 2; ix++) {
        for (int iy = 0; iy < 2; iy++) {
            for (int iz = 0; iz < 2; iz++) {
                // Linear voxel index
                int voxelIdx = ix + iy * 2 + iz * 4;

                // Voxel corners
                centers3D[voxelIdx].x = c_bBoxMin.x + x * 2 * c_voxelSize.x + ix * c_voxelSize.x;
                centers3D[voxelIdx].y = c_bBoxMin.y + y * 2 * c_voxelSize.y + iy * c_voxelSize.y;
                centers3D[voxelIdx].z = c_bBoxMin.z + z * 2 * c_voxelSize.z + iz * c_voxelSize.z;

                // CTF-slice
                ts[voxelIdx] = (c_projNorm.x * centers3D[voxelIdx].x + c_projNorm.y * centers3D[voxelIdx].y +
                                c_projNorm.z * centers3D[voxelIdx].z);
                ts[voxelIdx] += (-c_projNorm.x * c_detektor.x - c_projNorm.y * c_detektor.y -
                                 c_projNorm.z * c_detektor.z);
                ts[voxelIdx] = abs(ts[voxelIdx]);

                // Voxel centers
                centers3D[voxelIdx] = centers3D[voxelIdx] + (c_voxelSize * 0.5);

                // Center of support on projection
                // TODO: Account for mag anisotropy
                MatrixVector3Mul(c_DetectorMatrix, centers3D[voxelIdx], suppCenters2D[voxelIdx]);

                // Maximum extent of the support of this voxel
                borderMinV.x = floor(suppCenters2D[voxelIdx].x - supporthalf);
                borderMinV.y = floor(suppCenters2D[voxelIdx].y - supporthalf);
                borderMaxV.x = ceil(suppCenters2D[voxelIdx].x + supporthalf);
                borderMaxV.y = ceil(suppCenters2D[voxelIdx].y + supporthalf);

                // Update full min/max box for voxel block
                borderMin.x = fminf(borderMin.x, borderMinV.x);
                borderMin.y = fminf(borderMin.y, borderMinV.y);
                borderMax.x = fmaxf(borderMax.x, borderMaxV.x);
                borderMax.y = fmaxf(borderMax.y, borderMaxV.y);

                // Known voxel value
                surf3Dread(&accumulators[voxelIdx], volume, (x * 2 + ix) * 4, y * 2 + iy, z * 2 + iz);
            }
        }
    }

    // Clamp to total projection size
    borderMin.x = fminf(fmaxf(borderMin.x, 0), proj_x - 1);
    borderMin.y = fminf(fmaxf(borderMin.y, 0), proj_y - 1);
    borderMax.x = fminf(fmaxf(borderMax.x, 0), proj_x - 1);
    borderMax.y = fminf(fmaxf(borderMax.y, 0), proj_y - 1);

    // Final pixel range
    int rangeMaxX = (borderMax.x - borderMin.x) * maxOverSample;
    int rangeMaxY = (borderMax.y - borderMin.y) * maxOverSample;

    //printf("%i %i %f %f %f %f\n", rangeMaxX, rangeMaxY, borderMin.x, borderMax.x, borderMax.y);

    // Loop over potentially covered pixels
    for (int osy = 0; osy < rangeMaxY; osy++) {
        // Non-oversampled pixel coordinate
        float projpixel_y = borderMin.y + osy * maxOverSampleInv;
        // Oversampled pixel coordinate
        int idy = maxOverSample * borderMin.y + osy;

        for (int osx = 0; osx < rangeMaxX; osx++) {
            // Non-oversampled pixel coordinate
            float projpixel_x = borderMin.x + osx * maxOverSampleInv;
            // Oversampled pixel coordinate
            int idx = maxOverSample * borderMin.x + osx;

            float val = *(((float*)((char*)projection + stride * idy)) + idx);
            //float val = 0.f;

            // Loop over voxels in voxel block
            for (int voxelIdx = 0; voxelIdx < 8; voxelIdx++){
                // Distance for this voxel
                float distx = (projpixel_x - suppCenters2D[voxelIdx].x) * LUTstepinv + LUTcenter;
                float disty = (projpixel_y - suppCenters2D[voxelIdx].y) * LUTstepinv + LUTcenter;

                //printf("%f %f %f %f\n", distx, disty, (projpixel_x - suppCenters2D[voxelIdx].x), (projpixel_y - suppCenters2D[voxelIdx].y));

                // Weight from LUT for this voxel
                float weight = tex2D<float>(LUT, distx, disty);

                // Accumulate
                accumulators[voxelIdx] += val * weight;

//# if __CUDA_ARCH__>=200
//                //printf("%f %f %i %i %f %f %f %f %f\n", projpixel_x, projpixel_y, idx, idy, distx, disty, weight, val, accumulators[voxelIdx]);
//#endif
            }
        }
    }

    for (int ix = 0; ix < 2; ix++) {
        for (int iy = 0; iy < 2; iy++) {
            for (int iz = 0; iz < 2; iz++) {
                int voxelIdx = ix + iy * 2 + iz * 4;
                surf3Dwrite(accumulators[voxelIdx] * lambda, volume, (x * 2 + ix) * 4, y * 2 + iy, z * 2 + iz);
            }
        }
    }
}


extern "C"
__global__
void backProjectionLUTVoxelBlockNoDiv(int proj_x,
                                 int proj_y,
                                 float lambda,
                                 float maxOverSampleInv,
                                 float maxOverSampleInvH,
                                 float* projection,
                                 CUtexObject LUT,
                                 CUsurfObject volume,
                                 float tmin,
                                 float tmax,
                                 //float supporthalf,
                                 //float LUTstepinv,
                                 //float LUTcenter,
                                 int maxOverSample,
                                 size_t stride) {

    //float2 borderMinV;
    //float2 borderMaxV;
    float2 borderMin = make_float2(proj_x, proj_y);
    float2 borderMax = make_float2(0.f, 0.f);

    // Vol coords
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

//    const unsigned int x = 16;//blockIdx.x * blockDim.x + threadIdx.x;
//    const unsigned int y = 16;//blockIdx.y * blockDim.y + threadIdx.y;
//    const unsigned int z = 16;//blockIdx.z * blockDim.z + threadIdx.z;


    float ts[8];
    float accumulators[8];
    float2 suppCenters2D[8];
    //float2 rangeMax[8];
    //float3 centers3D[8];
    float2 center2D;
    float3 center3D;

    //int threadOffset = (threadIdx.x + 4 * threadIdx.y + 16 * threadIdx.z) * 24;
    //float* base = sharedPtr + threadOffset;
    //float* ts = base;
    //float2* suppCenters2D = (float2*)(base + 8);
//    float* ts = base;
//    float* accumulators = base + 8;
//    float2* suppCenters2D = (float2*)(base + 16);
//    float2* rangeMax = (float2*)(base + 32);
//    float3* centers3D = (float3*)(base + 48);

    for (int ix = 0; ix < 2; ix++) {
        for (int iy = 0; iy < 2; iy++) {
            for (int iz = 0; iz < 2; iz++) {
                // Linear voxel index
                int voxelIdx = ix + iy * 2 + iz * 4;

                // Voxel corners
                center3D.x = c_bBoxMin.x + x * 2 * c_voxelSize.x + ix * c_voxelSize.x;
                center3D.y = c_bBoxMin.y + y * 2 * c_voxelSize.y + iy * c_voxelSize.y;
                center3D.z = c_bBoxMin.z + z * 2 * c_voxelSize.z + iz * c_voxelSize.z;

                // CTF-slice
                ts[voxelIdx] = (c_projNorm.x * center3D.x + c_projNorm.y * center3D.y +
                                c_projNorm.z * center3D.z);
                ts[voxelIdx] += (-c_projNorm.x * c_detektor.x - c_projNorm.y * c_detektor.y -
                                 c_projNorm.z * c_detektor.z);
                ts[voxelIdx] = abs(ts[voxelIdx]);// + DIST;
                //ts[voxelIdx] = fminf(fmaxf(floorf((ts[voxelIdx] - c_entry)/c_sliceThickness), 0), c_sliceNumber);

                //printf("%f\n", ts[voxelIdx]);

                // Voxel centers
                center3D = center3D + (c_voxelSize * 0.5);

                // Center of support on projection //suppCenters2D[voxelIdx]
                // TODO: Account for mag anisotropy
                MatrixVector3Mul(c_DetectorMatrix, center3D, center2D);

                // Center shifted for mag anisotropy
                MatrixVector3Mul(c_magAnisoInv, center2D.x, center2D.y, suppCenters2D[voxelIdx].x, suppCenters2D[voxelIdx].y);

                // Lowest corner of the support of this voxel
                //borderMinV.x = floorf(suppCenters2D[voxelIdx].x - supporthalf);
                //borderMinV.y = floorf(suppCenters2D[voxelIdx].y - supporthalf);
                //borderMaxV.x = ceil(suppCenters2D[voxelIdx].x + supporthalf);
                //borderMaxV.y = ceil(suppCenters2D[voxelIdx].y + supporthalf);

                // Lowest corner of the support of this voxel block
                borderMin.x = fminf(borderMin.x, floorf(suppCenters2D[voxelIdx].x - c_voxelSupportHalf));
                borderMin.y = fminf(borderMin.y, floorf(suppCenters2D[voxelIdx].y - c_voxelSupportHalf));
                //borderMax.x = fmaxf(borderMax.x, borderMaxV.x);
                //borderMax.y = fmaxf(borderMax.y, borderMaxV.y);

                // Known voxel value
                surf3Dread(&accumulators[voxelIdx], volume, (x * 2 + ix) * 4, y * 2 + iy, z * 2 + iz);
            }
        }
    }

    // Clamp to total projection size
    //borderMin.x = fminf(fmaxf(borderMin.x, 0), proj_x - 1);
    //borderMin.y = fminf(fmaxf(borderMin.y, 0), proj_y - 1);
    //borderMax.x = fminf(fmaxf(borderMax.x, 0), proj_x - 1);
    //borderMax.y = fminf(fmaxf(borderMax.y, 0), proj_y - 1);

    // Final pixel range
    //int rangeMaxX = (borderMax.x - borderMin.x) * maxOverSample;
    //int rangeMaxY = (borderMax.y - borderMin.y) * maxOverSample;

    //printf("%i %i %f %f %f %f\n", rangeMaxX, rangeMaxY, borderMin.x, borderMax.x, borderMax.y);

    // Loop over potentially covered pixels
    for (int osy = 0; osy < c_blockSupportSize.y; osy++) {
        // Non-oversampled pixel coordinate
        float projpixel_y = borderMin.y + osy * maxOverSampleInv;

        // Skip if outside projection
        if (projpixel_y > proj_y - 1 || projpixel_y < 0)
            continue;

        // Oversampled pixel coordinate
        int idy = maxOverSample * borderMin.y + osy;

        for (int osx = 0; osx < c_blockSupportSize.x; osx++) {
            // Non-oversampled pixel coordinate
            float projpixel_x = borderMin.x + osx * maxOverSampleInv;

            // Skip if outside projection
            if (projpixel_x > proj_x - 1 || projpixel_x < 0)
                continue;

            // Oversampled pixel coordinate
            int idx = maxOverSample * borderMin.x + osx;

            float val = *(((float*)((char*)projection + stride * idy)) + idx);
            //float val = 0.f;

            // Loop over voxels in voxel block
            for (int voxelIdx = 0; voxelIdx < 8; voxelIdx++){
                // Distance for this voxel
                float2 suppCenter2D = suppCenters2D[voxelIdx];
                float distx = (projpixel_x - suppCenter2D.x) * c_LUTstepinv + c_LUTcenter;
                float disty = (projpixel_y - suppCenter2D.y) * c_LUTstepinv + c_LUTcenter;

                //printf("%f %f %f %f\n", distx, disty, (projpixel_x - suppCenters2D[voxelIdx].x), (projpixel_y - suppCenters2D[voxelIdx].y));

                // Weight from LUT for this voxel
                float weight = tex2D<float>(LUT, distx, disty);

                // Accumulate
                accumulators[voxelIdx] += val * weight;

//# if __CUDA_ARCH__>=200
//                //printf("%f %f %i %i %f %f %f %f %f\n", projpixel_x, projpixel_y, idx, idy, distx, disty, weight, val, accumulators[voxelIdx]);
//#endif
            }
        }
    }

    for (int ix = 0; ix < 2; ix++) {
        for (int iy = 0; iy < 2; iy++) {
            for (int iz = 0; iz < 2; iz++) {
                int voxelIdx = ix + iy * 2 + iz * 4;
                surf3Dwrite(accumulators[voxelIdx] * lambda, volume, (x * 2 + ix) * 4, y * 2 + iy, z * 2 + iz);
            }
        }
    }
}

extern "C"
__global__
void backProjectionLUTVoxelBlockNoDivSlice(int proj_x,
                                           int proj_y,
                                           float lambda,
                                           float maxOverSampleInv,
                                           float maxOverSampleInvH,
                                           float* projection,
                                           CUtexObject LUT,
                                           CUsurfObject volume,
                                           float tmin,
                                           float tmax,
                                           int maxOverSample) {

    //float2 borderMinV;
    //float2 borderMaxV;
    float2 borderMin = make_float2(proj_x, proj_y);
    float2 borderMax = make_float2(0.f, 0.f);

    // Vol coords
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

//    const unsigned int x = 16;//blockIdx.x * blockDim.x + threadIdx.x;
//    const unsigned int y = 16;//blockIdx.y * blockDim.y + threadIdx.y;
//    const unsigned int z = 16;//blockIdx.z * blockDim.z + threadIdx.z;


    int ts[8];
    float accumulators[8];
    float2 suppCenters2D[8];
    //float2 rangeMax[8];
    //float3 centers3D[8];
    float2 center2D;
    float3 center3D;

    //int threadOffset = (threadIdx.x + 4 * threadIdx.y + 16 * threadIdx.z) * 24;
    //float* base = sharedPtr + threadOffset;
    //float* ts = base;
    //float2* suppCenters2D = (float2*)(base + 8);
//    float* ts = base;
//    float* accumulators = base + 8;
//    float2* suppCenters2D = (float2*)(base + 16);
//    float2* rangeMax = (float2*)(base + 32);
//    float3* centers3D = (float3*)(base + 48);

    for (int ix = 0; ix < 2; ix++) {
        for (int iy = 0; iy < 2; iy++) {
            for (int iz = 0; iz < 2; iz++) {
                // Linear voxel index
                int voxelIdx = ix + iy * 2 + iz * 4;

                // Voxel corners
                center3D.x = c_bBoxMin.x + x * 2 * c_voxelSize.x + ix * c_voxelSize.x;
                center3D.y = c_bBoxMin.y + y * 2 * c_voxelSize.y + iy * c_voxelSize.y;
                center3D.z = c_bBoxMin.z + z * 2 * c_voxelSize.z + iz * c_voxelSize.z;

                // CTF-slice
                float t;
                t = (c_projNorm.x * center3D.x + c_projNorm.y * center3D.y +
                                c_projNorm.z * center3D.z);
                t += (-c_projNorm.x * c_detektor.x - c_projNorm.y * c_detektor.y -
                                 c_projNorm.z * c_detektor.z);
                t = abs(t) + DIST;
                //printf("%f\n", t);
                ts[voxelIdx] = (int)fminf(fmaxf(floorf((t - c_entry)/c_sliceThickness), 0), c_sliceNumber);

                //if (t > 0)
                //    printf("%f %i %f %f %i %f\n", t, ts[voxelIdx], c_entry, c_sliceThickness, c_sliceNumber, fminf(fmaxf(floorf((t - c_entry)/c_sliceThickness), 0), c_sliceNumber));

                //if (!(ts[voxelIdx] > 2 && ts[voxelIdx] < 4))
                //    return;

                // Voxel centers
                center3D = center3D + (c_voxelSize * 0.5);

                // Center of support on projection //suppCenters2D[voxelIdx]
                // TODO: Account for mag anisotropy
                MatrixVector3Mul(c_DetectorMatrix, center3D, center2D);

                // Center shifted for mag anisotropy
                MatrixVector3Mul(c_magAnisoInv, center2D.x, center2D.y, suppCenters2D[voxelIdx].x, suppCenters2D[voxelIdx].y);

                // Lowest corner of the support of this voxel
                //borderMinV.x = floorf(suppCenters2D[voxelIdx].x - supporthalf);
                //borderMinV.y = floorf(suppCenters2D[voxelIdx].y - supporthalf);
                //borderMaxV.x = ceil(suppCenters2D[voxelIdx].x + supporthalf);
                //borderMaxV.y = ceil(suppCenters2D[voxelIdx].y + supporthalf);

                // Lowest corner of the support of this voxel block
                borderMin.x = fminf(borderMin.x, floorf(suppCenters2D[voxelIdx].x - c_voxelSupportHalf));
                borderMin.y = fminf(borderMin.y, floorf(suppCenters2D[voxelIdx].y - c_voxelSupportHalf));
                //borderMax.x = fmaxf(borderMax.x, borderMaxV.x);
                //borderMax.y = fmaxf(borderMax.y, borderMaxV.y);

                // Known voxel value
                surf3Dread(&accumulators[voxelIdx], volume, (x * 2 + ix) * 4, y * 2 + iy, z * 2 + iz);
            }
        }
    }

    // Clamp to total projection size
    //borderMin.x = fminf(fmaxf(borderMin.x, 0), proj_x - 1);
    //borderMin.y = fminf(fmaxf(borderMin.y, 0), proj_y - 1);
    //borderMax.x = fminf(fmaxf(borderMax.x, 0), proj_x - 1);
    //borderMax.y = fminf(fmaxf(borderMax.y, 0), proj_y - 1);

    // Final pixel range
    //int rangeMaxX = (borderMax.x - borderMin.x) * maxOverSample;
    //int rangeMaxY = (borderMax.y - borderMin.y) * maxOverSample;

    //printf("%i %i %f %f %f %f\n", rangeMaxX, rangeMaxY, borderMin.x, borderMax.x, borderMax.y);

    // Loop over potentially covered pixels
    for (int osy = 0; osy < c_blockSupportSize.y; osy++) {
        // Non-oversampled pixel coordinate
        float projpixel_y = borderMin.y + osy * maxOverSampleInv;

        // Skip if outside projection
        if (projpixel_y >= proj_y || projpixel_y < 0)
            continue;

        // Oversampled pixel coordinate
        int idy = maxOverSample * borderMin.y + osy;

        for (int osx = 0; osx < c_blockSupportSize.x; osx++) {
            // Non-oversampled pixel coordinate
            float projpixel_x = borderMin.x + osx * maxOverSampleInv;

            // Skip if outside projection
            if (projpixel_x >= proj_x || projpixel_x < 0)
                continue;

            // Oversampled pixel coordinate
            int idx = maxOverSample * borderMin.x + osx;

            //float val = *(((float*)((char*)projection + stride * idy)) + idx);
            //float val = 0.f;

            // Loop over voxels in voxel block
            for (int voxelIdx = 0; voxelIdx < 8; voxelIdx++){
                // Value
                float val = projection[ts[voxelIdx] * proj_y * proj_x + idy * proj_x + idx];

                // Distance for this voxel
                float2 suppCenter2D = suppCenters2D[voxelIdx];
                float distx = (projpixel_x - suppCenter2D.x) * c_LUTstepinv + c_LUTcenter;
                float disty = (projpixel_y - suppCenter2D.y) * c_LUTstepinv + c_LUTcenter;

                //printf("%f %f %f %f\n", distx, disty, (projpixel_x - suppCenters2D[voxelIdx].x), (projpixel_y - suppCenters2D[voxelIdx].y));

                // Weight from LUT for this voxel
                float weight = tex2D<float>(LUT, distx, disty);

                // Accumulate
                accumulators[voxelIdx] += val * weight;

//# if __CUDA_ARCH__>=200
//                //printf("%f %f %i %i %f %f %f %f %f\n", projpixel_x, projpixel_y, idx, idy, distx, disty, weight, val, accumulators[voxelIdx]);
//#endif
            }
        }
    }

    for (int ix = 0; ix < 2; ix++) {
        for (int iy = 0; iy < 2; iy++) {
            for (int iz = 0; iz < 2; iz++) {
                int voxelIdx = ix + iy * 2 + iz * 4;
                surf3Dwrite(accumulators[voxelIdx] * lambda, volume, (x * 2 + ix) * 4, y * 2 + iy, z * 2 + iz);
            }
        }
    }
}

extern "C"
__global__
void backProjectionLUTsliced(int proj_x,
                       int proj_y,
                       float lambda,
                       float maxOverSampleInv,
                       float maxOverSampleInvH,
                       float* projection,
                       CUtexObject LUT,
                       CUsurfObject volume,
                       float tmin,
                       float tmax,
                       int maxOverSample) {


    float2 pixel;
    float2 borderMin = make_float2(proj_x, proj_y);

    // Vol coords
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Drop threads outside volume
    if (x >= c_volumeDim.x || y >= c_volumeDim.y || z >= c_volumeDim.z) return;

    float2 center2D;
    float2 suppCenter2D;
    float3 center3D;

    // Voxel corner,
    center3D = make_float3(c_bBoxMin.x + x * c_voxelSize.x, c_bBoxMin.y + y * c_voxelSize.y, c_bBoxMin.z + z * c_voxelSize.z);

    // Distance to projection plane (for CTF slices)
    float t;
    int ts;
    t = (c_projNorm.x * center3D.x + c_projNorm.y * center3D.y + c_projNorm.z * center3D.z);
    t += (-c_projNorm.x * c_detektor.x - c_projNorm.y * c_detektor.y - c_projNorm.z * c_detektor.z);
    t = abs(t) + DIST;

    // Slice index
    ts = (int)fminf(fmaxf(floorf((t - c_entry)/c_sliceThickness), 0), c_sliceNumber);

    // Voxel center
    //center3D = make_float3(center3D.x + c_voxelSize.x * 0.5f, center3D.y + c_voxelSize.y * 0.5f, center3D.z + c_voxelSize.z * 0.5f);

    // Project Center
    MatrixVector3Mul(c_DetectorMatrix, center3D, center2D);
    MatrixVector3Mul(c_magAnisoInv, center2D.x, center2D.y, suppCenter2D.x, suppCenter2D.y);

    // Lowest corner of the support of this voxel block
    borderMin.x = fminf(borderMin.x, floorf(suppCenter2D.x - c_voxelSupportHalf));
    borderMin.y = fminf(borderMin.y, floorf(suppCenter2D.y - c_voxelSupportHalf));

    // Known voxel value
    float accumulator = 0.f;
    surf3Dread(&accumulator, volume, x * 4, y, z);

    // Loop over potentially covered pixels
    for (int osy = 0; osy < c_blockSupportSize.y; osy++) {
        // Non-oversampled pixel coordinate
        float projpixel_y = borderMin.y + osy * maxOverSampleInv;

        // Skip if outside projection
        if (projpixel_y >= proj_y || projpixel_y < 0)
            continue;

        // Oversampled pixel coordinate
        int idy = maxOverSample * borderMin.y + osy;

        for (int osx = 0; osx < c_blockSupportSize.x; osx++) {
            // Non-oversampled pixel coordinate
            float projpixel_x = borderMin.x + osx * maxOverSampleInv;

            // Skip if outside projection
            if (projpixel_x >= proj_x || projpixel_x < 0)
                continue;

            // Oversampled pixel coordinate
            int idx = maxOverSample * borderMin.x + osx;

            // Distance for this voxel
            float distx = (projpixel_x - suppCenter2D.x) * c_LUTstepinv + c_LUTcenter;
            float disty = (projpixel_y - suppCenter2D.y) * c_LUTstepinv + c_LUTcenter;

            // Weight from LUT
            float weight = tex2D<float>(LUT, distx+0.5f, disty+0.5f);

            //printf("%f %f %f %f\n", tex2D<float>(LUT, c_LUTcenter, c_LUTcenter), c_LUTcenter, tex2D<float>(LUT, 500, 500), tex2D<float>(LUT, 500.5, 500.5));

            // Value from image
            float val = projection[ts * proj_y * proj_x + idy * proj_x + idx];//*(((float*)((char*)projection + stride * idy)) + idx);
            //float val = 0.f;

            // Accumulate
            accumulator += val * weight;
        }
    }

    surf3Dwrite(accumulator * lambda, volume, x * 4, y, z);
}


extern "C"
__global__
void oversample(int proj_x,
                int proj_y,
                int maxOverSample,
                float maxOverSampleInv,
                float maxOverSampleInvH,
                CUtexObject inprojection,
                float* outprojection,
                size_t out_stride)
{
    // integer pixel coordinates
    const unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x >= proj_x || y >= proj_y) return;
    if (x < 0 || y < 0) return;

    float val = 0.f;
    for (float osy = 0; osy < maxOverSample; osy++) {
        for (float osx = 0; osx < maxOverSample; osx++) {
            unsigned int idx = maxOverSample * x + osx;
            unsigned int idy = maxOverSample * y + osy;

            float pixel_x = x + maxOverSampleInvH + osx * maxOverSampleInv;
            float pixel_y = y + maxOverSampleInvH + osy * maxOverSampleInv;

# if __CUDA_ARCH__>=200
            //printf("%f %f\n", pixel_x, pixel_y);
#endif

            *(((float*)((char*)outprojection + out_stride * idy)) + idx) = cubicTex2D<float>(inprojection, pixel_x, pixel_y);
            //val = cubicTex2D<float>(inprojection, pixel_x, pixel_y);
            //surf2Dwrite(val, outprojection, idx * 4, idy);
        }
    }
}

#endif //BACKPROJECTIONSQUAREOS_CU
