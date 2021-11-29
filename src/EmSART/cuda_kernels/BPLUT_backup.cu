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

extern volatile __shared__ unsigned char sBuffer[];


extern "C"
__global__
void backProjectionLUT(int proj_x,
                       int proj_y,
                       float lambda,
                       float maxOverSampleInv,
                       float maxOverSampleInvH,
                       CUtexObject projection,
                       CUtexObject LUT,
                       CUsurfObject volume,
                       float tmin,
                       float tmax,
                       float supporthalf,
                       float LUTstepinv,
                       float LUTcenter) {

    float2 pixel;
    float2 borderMin;
    float2 borderMax;
    float3 c_source;
    float3 corner;
    float3 center;

    // Vol coords
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Full volume size
    if (x >= c_volumeDim.x || y >= c_volumeDim.y || z >= c_volumeDim.z) return;

    // Voxel corner, center
    corner = make_float3(c_bBoxMin.x + x * c_voxelSize.x, c_bBoxMin.y + y * c_voxelSize.y, c_bBoxMin.z + z * c_voxelSize.z);
    center = make_float3(corner.x + c_voxelSize.x * 0.5f, corner.y + c_voxelSize.y * 0.5f, corner.z + c_voxelSize.z * 0.5f);

    // Distance to projection plane (for CTF slices)
    float t;
    t = (c_projNorm.x * corner.x + c_projNorm.y * corner.y + c_projNorm.z * corner.z);
    t += (-c_projNorm.x * c_detektor.x - c_projNorm.y * c_detektor.y - c_projNorm.z * c_detektor.z);
    t = abs(t);

    // Return if outside current CTF slice
    if (!(t >= tmin && t < tmax)) return;

    // Project
    // TODO: Account for mag anisotropy
    MatrixVector3Mul(c_DetectorMatrix, center, pixel);

    // Maximum extent of the support clamped by projection size
    borderMin.x = floor(pixel.x - supporthalf);
    borderMin.y = floor(pixel.y - supporthalf);
    borderMax.x = ceil(pixel.x + supporthalf);
    borderMax.y = ceil(pixel.y + supporthalf);

    borderMin.x = fminf(fmaxf(borderMin.x, 0), proj_x);
    borderMin.y = fminf(fmaxf(borderMin.y, 0), proj_y);
    borderMax.x = fminf(fmaxf(borderMax.x, 0), proj_x);
    borderMax.y = fminf(fmaxf(borderMax.y, 0), proj_y);

    // Known value
    float accumulator = 0.f;
    surf3Dread(&accumulator, volume, x * 4, y, z);

    // Loop over potentially covered pixels
    for (float pixel_y = borderMin.y + maxOverSampleInvH; pixel_y < borderMax.y; pixel_y += maxOverSampleInv) {
        // Distance to projected voxel center y
        float disty = (pixel_y - pixel.y) * LUTstepinv + LUTcenter;

        for (float pixel_x = borderMin.x + maxOverSampleInvH; pixel_x < borderMax.x; pixel_x += maxOverSampleInv) {
            // Distance to projected voxel center x
            float distx = (pixel_x - pixel.x) * LUTstepinv + LUTcenter;

            // Weight from LUT
            float weight = tex2D<float>(LUT, distx, disty);

            // Value from image
            float val = cubicTex2D<float>(projection, pixel_x, pixel_y) * maxOverSampleInv * maxOverSampleInv;

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

    for (float osy = 0; osy < maxOverSample; osy++) {
        for (float osx = 0; x < maxOverSample; osx++) {
            int idx = maxOverSample * x + osx;
            int idy = maxOverSample * y + osy;

            float pixel_x = x + maxOverSampleInvH + osx * maxOverSampleInv;
            float pixel_y = y + maxOverSampleInvH + osy * maxOverSampleInv;

            *(((float*)((char*)outprojection + out_stride * idy)) + idx) = cubicTex2D<float>(inprojection, pixel_x, pixel_y);
        }
    }
}

#endif //BACKPROJECTIONSQUAREOS_CU
