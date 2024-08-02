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
* ForwardProjectionAPriori.cu
* DDA forward projection with trilinear
* interpolation
*
**********************************************/

#ifndef FORWARDPROJECTIONRAYMARCHERNN_CU
#define FORWARDPROJECTIONRAYMARCHERNN_CU


#include <cuda.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>
#include <builtin_types.h>
#include <vector_functions.h>
#include "Constants.h"
#include "DeviceVariables.cuh"
#include "float.h"
#include "cutil_math.h"
#include "cubic_interpolation/cubicTex2D.cu"
#include "cubic_interpolation/cubicTex3D.cu"
#include "common_types.h"

//texture< ushort, 3, cudaReadModeNormalizedFloat > t_dataset;
//texture< float, 3, cudaReadModeElementType > t_dataset;

// transform vector by matrix
//__device__
//void MatrixVector3Mul(float4x4 M, float3* v)
//{
//    float3 erg;
//    erg.x = M.m[0].x * v->x + M.m[0].y * v->y + M.m[0].z * v->z + 1.f * M.m[0].w;
//    erg.y = M.m[1].x * v->x + M.m[1].y * v->y + M.m[1].z * v->z + 1.f * M.m[1].w;
//    erg.z = M.m[2].x * v->x + M.m[2].y * v->y + M.m[2].z * v->z + 1.f * M.m[2].w;
//    *v = erg;
//}

// transform vector by matrix
__device__
void MatrixVector3Mul(float4x4 M, float3& v, float2& erg)
{
    erg.x = M.m[0].x * v.x + M.m[0].y * v.y + M.m[0].z * v.z + 1.f * M.m[0].w;
    erg.y = M.m[1].x * v.x + M.m[1].y * v.y + M.m[1].z * v.z + 1.f * M.m[1].w;
}


typedef unsigned long long int ulli;

extern "C"
__global__
void forwardProjectionLUT(int proj_x,
                          int proj_y,
                          size_t stride,
                          float* projection,
                          float* distanceMap,
                          CUtexObject LUT,
                          CUsurfObject volume,
                          float tminDefocus,
                          float tmaxDefocus,
                          int2 roiMin,
                          int2 roiMax,
                          float supporthalf,
                          float LUTstepinv,
                          float LUTcenter)
{
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
    if (!(t >= tminDefocus && t < tmaxDefocus)) return;

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

    unsigned int xstart = static_cast<unsigned int>(borderMin.x);
    unsigned int xend = static_cast<unsigned int>(borderMax.x);
    unsigned int ystart = static_cast<unsigned int>(borderMin.y);
    unsigned int yend = static_cast<unsigned int>(borderMax.y);

    // Known value
    float coefficient = 0.f;
    surf3Dread(&coefficient, volume, x * 4, y, z);

    // Loop over potentially covered pixels
    unsigned int xi, yi;
    for (yi = ystart; yi < yend; yi++) {
        // Distance to projected voxel center y
        float pixel_y = (float)yi + 0.5f;
        float disty = ((float) pixel_y - pixel.y) * LUTstepinv + LUTcenter;

        for (xi = xstart; xi < xend; xi++) {
            // Distance to projected voxel center x
            float pixel_x = (float)xi + 0.5f;
            float distx = ((float) pixel_x - pixel.x) * LUTstepinv + LUTcenter;

            // Weight from LUT
            float weight = tex2D<float>(LUT, distx, disty);

            // Accumulate
            atomicAdd((((float *) ((char *) projection + stride * yi)) + xi), coefficient * weight);
            atomicAdd((((float *) ((char *) distanceMap + stride * yi)) + xi), weight);
        }
    }
}


extern "C"
__global__
void forwardProjectionLUTNoDivSliced(int proj_x,
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
    ts = (int)fminf(fmaxf(floorf((t - c_entry)/c_sliceThickness), 0), c_sliceNumber);

    // Voxel center
    //center3D = make_float3(center3D.x + c_voxelSize.x * 0.5f, center3D.y + c_voxelSize.y * 0.5f, center3D.z + c_voxelSize.z * 0.5f);

    // Project Center
    MatrixVector3Mul(c_DetectorMatrix, center3D, center2D);
    MatrixVector3Mul(c_magAnisoInv, center2D.x, center2D.y, suppCenter2D.x, suppCenter2D.y);

//    if (center3D.x == 0 && center3D.y == 0 && center3D.z == 0)
//    {
//        printf("center2D %f %f", suppCenter2D.x, suppCenter2D.y);
//    }

    // Lowest corner of the support of this voxel block
    borderMin.x = fminf(borderMin.x, floorf(suppCenter2D.x - c_voxelSupportHalf));
    borderMin.y = fminf(borderMin.y, floorf(suppCenter2D.y - c_voxelSupportHalf));

    // Known voxel value
    float coefficient = 0.f;
    surf3Dread(&coefficient, volume, x * 4, y, z);

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
            float weight = tex2D<float>(LUT, distx, disty);

            // Value from image
            //float val = projection[ts * proj_y * proj_x + idy * proj_x + idx];

            // Accumulate
            //accumulator += val * weight;
            atomicAdd(projection + (ts * proj_y * proj_x + idy * proj_x + idx), coefficient * weight);
            //atomicAdd((((float *) ((char *) distanceMap + stride * yi)) + xi), weight);

        }
    }
    //surf3Dwrite(accumulator * lambda, volume, x * 4, y, z);
}

extern "C"
__global__
void distanceImageLUTNoDiv(int proj_x,
                           int proj_y,
                           float lambda,
                           float maxOverSampleInv,
                           float maxOverSampleInvH,
                           float* distanceMap,
                           CUtexObject LUT,
                           CUsurfObject volume,
                           float tmin,
                           float tmax,
                           int maxOverSample,
                           int stride) {

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
    float coefficient = 0.f;
    surf3Dread(&coefficient, volume, x * 4, y, z);

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

            // Value from image
            //float val = projection[ts * proj_y * proj_x + idy * proj_x + idx];

            // Accumulate
            //accumulator += val * weight;
            //atomicAdd(projection + (ts * proj_y * proj_x + idy * proj_x + idx), coefficient * weight);
            atomicAdd((((float *) ((char *) distanceMap + stride * idy)) + idx), weight);

        }
    }
    //surf3Dwrite(accumulator * lambda, volume, x * 4, y, z);
}

extern "C"
__global__
void forwardProjectionOrthoNoDivSliced(int proj_x,
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
    ts = (int)fminf(fmaxf(floorf((t - c_entry)/c_sliceThickness), 0), c_sliceNumber);

    // Voxel center
    //center3D = make_float3(center3D.x + c_voxelSize.x * 0.5f, center3D.y + c_voxelSize.y * 0.5f, center3D.z + c_voxelSize.z * 0.5f);

    // Project Center
    MatrixVector3Mul(c_DetectorMatrix, center3D, center2D);
    MatrixVector3Mul(c_magAnisoInv, center2D.x, center2D.y, suppCenter2D.x, suppCenter2D.y);

//    if (center3D.x == 0 && center3D.y == 0 && center3D.z == 0)
//    {
//        printf("center2D %f %f", suppCenter2D.x, suppCenter2D.y);
//    }

    // Lowest corner of the support of this voxel block
    borderMin.x = fminf(borderMin.x, floorf(suppCenter2D.x - c_voxelSupportHalf));
    borderMin.y = fminf(borderMin.y, floorf(suppCenter2D.y - c_voxelSupportHalf));

    // Known voxel value
    float coefficient = 0.f;
    surf3Dread(&coefficient, volume, x * 4, y, z);

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
            //float distx = (projpixel_x - suppCenter2D.x) * c_LUTstepinv + c_LUTcenter;
            //float disty = (projpixel_y - suppCenter2D.y) * c_LUTstepinv + c_LUTcenter;
            float distx = (projpixel_x - suppCenter2D.x);
            float disty = (projpixel_y - suppCenter2D.y);

            // Weight from LUT
            float weight = bspline(distx) * bspline(disty);
            //float weight = tex2D<float>(LUT, distx, disty);

            // Value from image
            //float val = projection[ts * proj_y * proj_x + idy * proj_x + idx];

            // Accumulate
            //accumulator += val * weight;
            atomicAdd(projection + (ts * proj_y * proj_x + idy * proj_x + idx), coefficient * weight);
            //atomicAdd((((float *) ((char *) distanceMap + stride * yi)) + xi), weight);

        }
    }
    //surf3Dwrite(accumulator * lambda, volume, x * 4, y, z);
}

extern "C"
__global__
void distanceMapOrthoNoDiv(int proj_x,
                           int proj_y,
                           float maxOverSampleInv,
                           float maxOverSampleInvH,
                           float* distanceMap,
                           size_t stride,
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
    ts = (int)fminf(fmaxf(floorf((t - c_entry)/c_sliceThickness), 0), c_sliceNumber);

    // Project Center
    MatrixVector3Mul(c_DetectorMatrix, center3D, center2D);
    MatrixVector3Mul(c_magAnisoInv, center2D.x, center2D.y, suppCenter2D.x, suppCenter2D.y);

    // Lowest corner of the support of this voxel block
    borderMin.x = fminf(borderMin.x, floorf(suppCenter2D.x - c_voxelSupportHalf));
    borderMin.y = fminf(borderMin.y, floorf(suppCenter2D.y - c_voxelSupportHalf));

    // Known voxel value
//    float coefficient = 0.f;
//    surf3Dread(&coefficient, volume, x * 4, y, z);

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
            float distx = (projpixel_x - suppCenter2D.x);
            float disty = (projpixel_y - suppCenter2D.y);

            // Weight from LUT
            float weight = bspline(distx) * bspline(disty);

            // Accumulate
            //atomicAdd(projection + (ts * proj_y * proj_x + idy * proj_x + idx), coefficient * weight);
            atomicAdd((((float *) ((char *) distanceMap + stride * idy)) + idx), weight);// * c_voxelSize.x * c_voxelSize.x);
        }
    }
}

// Bspline computation without divergence using int casting trick.
inline __device__ float bspline_nodiv(float t)
{
    float res[3];
    t = fabsf(t);
    int idx = max(__float2int_rd(t), 2);
    const float a = 2.0f - t;

    res[0] = 2.0f/3.0f - 0.5f*t*t*a;
    res[1] = a*a*a / 6.0f;
    res[2] = 0.f;

    return res[idx];
}

template<int support, bool onlyWeight, bool forceSingleSlice, bool accountForOverlap>
__device__
void forwardProjectionOrthoSimpleSliced(uint2 projDim,
                                        uint3 volDim,
                                        const ctfImageConstants imageConstants,
                                        const float4x4 systemMatrix,
                                        float* projection,
                                        CUsurfObject volume,
                                        int2 minmaxSlice,
                                        CUtexObject overlapVolume,
                                        float4x4 childToParent)
{
    // Vol coords
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Drop threads outside volume
    if (x >= volDim.x || y >= volDim.y || z >= volDim.z) return;

    // Project Voxel
    float3 center2D;
    float3 center3D = make_float3((float)x, (float)y, (float)z);
    MatrixVector3Mul(systemMatrix, center3D, center2D);

    // Distance to projection plane (for CTF slices)
    int ts = 0;
    if(!forceSingleSlice) {
        float distCenter = imageConstants.ctfCenter - center2D.z - imageConstants.entryPoint;
        ts = min(max((int) floorf(distCenter / imageConstants.sliceThickness), 0), imageConstants.sliceNumber);

        // If outside the current batch, return
        if (ts < minmaxSlice.x || ts > minmaxSlice.y) return;

        // If inside, adjust the slice index by the offset
        ts = ts - minmaxSlice.x;
    }

    // Lowest corner of the support of this voxel, clamp between 0 and projection size
    int2 borderMin;
    borderMin.x = max(min((int)projDim.x, (int)floorf(center2D.x - (float)support * 0.5f)), 0);
    borderMin.y = max(min((int)projDim.y, (int)floorf(center2D.y - (float)support * 0.5f)), 0);

    // Completely outside projection, no need to enter the loop.
    if (borderMin.x == projDim.x && borderMin.x == projDim.y) return;

    // Highest corner of the support of this voxel, clamp between -1 and projection size
    int2 borderMax;
    borderMax.x = min(max(-1, (int)ceilf(center2D.x + (float)support * 0.5f)), (int)projDim.x);
    borderMax.y = min(max(-1, (int)ceilf(center2D.y + (float)support * 0.5f)), (int)projDim.y);

    // Completely outside projection, no need to enter the loop.
    if (borderMax.x == -1 && borderMax.y == -1) return;

    // Known voxel value
    float coefficient = 0.f;
    // Reading the volume value is only necessary when we compute the FP,
    // not when computing distance only
    if(!onlyWeight) {
        surf3Dread(&coefficient, volume, x * 4, y, z);
    }

    // Read from the 3D overlap volume and downweight the coefficient if necessary

    if (accountForOverlap){
        // Pixel coords of this volume to parent volume
        float3 childCoords = make_float3((float)x, (float)y, float(z));
        MatrixVector3Mul(childToParent, &childCoords);

        // Interpolate overlap
        float overlap = 1.f;
        overlap = tex3D<float>(overlapVolume,
                               childCoords.x + 0.5f,
                               childCoords.y + 0.5f,
                               childCoords.z + 0.5f);
        coefficient *= overlap;
    }



    for (int dy = 0; dy < support+2; dy++)
    {
        // Pixel coordinate
        int projpixel_y = borderMin.y + dy;

        // Skip if outside projection, we only check max border,
        // because borderMin.y was already clamped to 0 before.
        if (projpixel_y >= projDim.y)
            continue;

        // Distance y for this voxel
        float disty = ((float)projpixel_y) - center2D.y;
        float by = bspline(disty);

        for (int dx = 0; dx < support+2; dx++)
        {
            // Pixel coordinate
            int projpixel_x = borderMin.x + dx;

            // Skip if outside projection, we only check max border,
            // because borderMin.x was already clamped to 0 before.
            if (projpixel_x >= projDim.x)
                continue;

            // Distance x for this voxel
            float distx = ((float)projpixel_x) - center2D.x;
            //float bx = bspline(distx);
            float bx = bspline(distx);

            // Weight exact
            float weight = bx * by;

            // Project
            // This is not ideal, but the amount of serialization should be relatively low
            // due to the offset of one voxel to another.
            if (onlyWeight) {
                atomicAdd(projection + (ts * projDim.y * projDim.x + projpixel_y * projDim.x + projpixel_x),
                          weight);
            } else {
                atomicAdd(projection + (ts * projDim.y * projDim.x + projpixel_y * projDim.x + projpixel_x),
                          coefficient * weight);
            }
        }
    }
}

// Distance
extern "C"
__global__ void fpOrthoDist(uint2 projDim,
                            uint3 volDim,
                            const ctfImageConstants imageConstants,
                            const float4x4 systemMatrix,
                            float* projection,
                            CUsurfObject volume)
{
    forwardProjectionOrthoSimpleSliced<4, true, true, false>(projDim,
                                                             volDim,
                                                             imageConstants,
                                                             systemMatrix,
                                                             projection,
                                                             volume,
                                                             make_int2(0, 0),
                                                             0,
                                                             {});
}

// Forward projection
extern "C"
__global__ void fpOrthoProject(uint2 projDim,
                               uint3 volDim,
                               const ctfImageConstants imageConstants,
                               const float4x4 systemMatrix,
                               float* projection,
                               CUsurfObject volume,
                               int2 minmaxSlice)
{
    forwardProjectionOrthoSimpleSliced<4, false, false, false>(projDim,
                                                               volDim,
                                                               imageConstants,
                                                               systemMatrix,
                                                               projection,
                                                               volume,
                                                               minmaxSlice,
                                                               0,
                                                               {});
}

// Forward projection no slices
extern "C"
__global__ void fpOrthoProjectSS(uint2 projDim,
                                 uint3 volDim,
                                 const ctfImageConstants imageConstants,
                                 const float4x4 systemMatrix,
                                 float* projection,
                                 CUsurfObject volume)
{
    forwardProjectionOrthoSimpleSliced<4, false, true, false>(projDim,
                                                              volDim,
                                                              imageConstants,
                                                              systemMatrix,
                                                              projection,
                                                              volume,
                                                              make_int2(0, 0),
                                                              0,
                                                              {});
}

// Forward projection with overlap map
extern "C"
__global__ void fpOrthoProjectOV(uint2 projDim,
                                 uint3 volDim,
                                 const ctfImageConstants imageConstants,
                                 const float4x4 systemMatrix,
                                 float* projection,
                                 CUsurfObject volume,
                                 int2 minmaxSlice,
                                 CUtexObject overlapVolume,
                                 float4x4 childToParent)
{
    forwardProjectionOrthoSimpleSliced<4, false, false, true>(projDim,
                                                              volDim,
                                                              imageConstants,
                                                              systemMatrix,
                                                              projection,
                                                              volume,
                                                              minmaxSlice,
                                                              overlapVolume,
                                                              childToParent);
}


extern "C"
__global__
void sample2D(CUtexObject inimage, float* outimage, dim3 imDim, size_t stride, size_t offset)
{
    // Vol coords
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Full volume size
    if (x >= imDim.x || y >= imDim.y) return;

    // Interpolate
    //float3 coord = make_float3((float)x+0.5f, (float)y+0.5f);
    float val = cubicTex2DSimple<float>(inimage, ((float)x)+0.5f, ((float)y)+0.5f);
    //val = cubicTex2D<float>(inimage, ((float)x)+0.5f, ((float)y)+0.5f);

    // Write
    //surf2Dwrite(val, outvol, x * 4, y, z);
    *(((float*)((char*)outimage + offset + stride * y)) + x) = val;
}

extern "C"
__global__
void sample3D(CUtexObject involume, CUsurfObject outvol, float3 volDim, bool add)
{
    // Vol coords
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Full volume size
    if (x >= volDim.x || y >= volDim.y || z >= volDim.z) return;

    // Interpolate
    float3 coord = make_float3((float)x+0.5f, (float)y+0.5f, (float)z+0.5f);
    float val = cubicTex3DSimple<float>(involume, coord);
    //float val = cubicTex3D<float>(involume, coord);

    // Write
    float prev = 0.f;
    if (add) {
        surf3Dread(&prev, outvol, x * 4, y, z);
    }

    surf3Dwrite(val+prev, outvol, x * 4, y, z);
}

template<bool useMask>
__device__
void add3Dmask(CUsurfObject invol,
               CUsurfObject outvol,
               CUsurfObject mask,
               uint3 volDim,
               float scaleFactor)
{
    // Vol coords
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Full volume size
    if (x >= volDim.x || y >= volDim.y || z >= volDim.z) return;

    // Read
    float in = 0.f;
    float out = 0.f;
    surf3Dread(&in, invol, x * 4, y, z);
    surf3Dread(&out, outvol, x * 4, y, z);

    float maskVal = 1;
    if (useMask){
        surf3Dread(&maskVal, mask, x * 4, y, z);
    }

    // Write
    surf3Dwrite((in * scaleFactor * maskVal) + out, outvol, x * 4, y, z);
}

extern "C"
__global__
void add3D(CUsurfObject invol,
           CUsurfObject outvol,
           uint3 volDim,
           float scaleFactor)
{
    add3Dmask<false>(invol,
                     outvol,
                     0,
                     volDim,
                     scaleFactor);
}

extern "C"
__global__
void add3Dmasked(CUsurfObject invol,
                 CUsurfObject outvol,
                 CUsurfObject mask,
                 uint3 volDim,
                 float scaleFactor)
{
    add3Dmask<true>(invol,
                    outvol,
                    mask,
                    volDim,
                    scaleFactor);
}


extern "C"
__global__
void mask3D(CUsurfObject invol,
            CUsurfObject outvol,
            CUsurfObject mask,
            uint3 volDim)
{
    // Vol coords
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Full volume size
    if (x >= volDim.x || y >= volDim.y || z >= volDim.z) return;

    // Read
    float in = 0.f;
    float maskVal = 1.f;
    surf3Dread(&in, invol, x * 4, y, z);
    surf3Dread(&maskVal, mask, x * 4, y, z);

    // Write
    surf3Dwrite((in * maskVal), outvol, x * 4, y, z);
}

extern "C"
__global__
void mask3Dtrans(CUsurfObject vol,
                 CUtexObject maskTex,
                 float4x4 transform,
                 uint3 volDim,
                 uint3 offset)
{
    // Vol coords
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x + offset.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y + offset.y;
    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z + offset.z;

    // Full volume size
    if (x >= volDim.x || y >= volDim.y || z >= volDim.z) return;

    // Read
    float in = 0.f;
    surf3Dread(&in, vol, x * 4, y, z);

    // Tomo frame to particle frame
    float3 maskCoords = make_float3((float)x, (float)y, float(z));
    MatrixVector3Mul(transform, &maskCoords);

    auto maskVal = tex3D<float>(maskTex, maskCoords.x + 0.5f, maskCoords.y + 0.5f, maskCoords.z + 0.5f);

    // Write
    surf3Dwrite((in * maskVal), vol, x * 4, y, z);
}

extern "C"
__global__
void add3Dtrans(CUsurfObject vol,
                CUtexObject objTex,
                float4x4 transform,
                uint3 volDim,
                uint3 offset)
{
    // Vol coords
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x + offset.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y + offset.y;
    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z + offset.z;

    // Full volume size
    if (x >= volDim.x || y >= volDim.y || z >= volDim.z) return;

    // Read
    float in = 0.f;
    surf3Dread(&in, vol, x * 4, y, z);

    // Tomo frame to particle frame
    float3 maskCoords = make_float3((float)x, (float)y, float(z));
    MatrixVector3Mul(transform, &maskCoords);

    auto objVal = tex3D<float>(objTex, maskCoords.x + 0.5f, maskCoords.y + 0.5f, maskCoords.z + 0.5f);
    objVal = (objVal > 0) ? 1 : 0;

    // Write
    surf3Dwrite(in + objVal, vol, x * 4, y, z);
}

extern "C"
__global__
void div3Dmask(CUsurfObject volin,
               CUsurfObject volout,
               CUsurfObject mask,
               float divVal,
               uint3 volDim)
{
    // Vol coords
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Full volume size
    if (x >= volDim.x || y >= volDim.y || z >= volDim.z) return;

    // Read
    float inval = 0.f;
    surf3Dread(&inval, volin, x * 4, y, z);
    float maskval = 1.f;
    surf3Dread(&maskval, mask, x * 4, y, z);
    maskval = 1.f - maskval;
    maskval = (maskval > 0) ? maskval * divVal : 1.f;

    // Write
    surf3Dwrite(inval/maskval, volout, x * 4, y, z);
}

extern "C"
__global__
void norm3DOverlap(CUsurfObject vol,
                 uint3 volDim)
{
    // Vol coords
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Full volume size
    if (x >= volDim.x || y >= volDim.y || z >= volDim.z) return;

    // Read
    float in = 0.f;
    surf3Dread(&in, vol, x * 4, y, z);

    // If Overlap > 1, invert, otherwise overlap = 1
    float overlap = (in > 1.f) ? 1.f/in : 1;

    // Write
    surf3Dwrite(overlap, vol, x * 4, y, z);
}

extern "C"
__global__
void set3D(CUsurfObject inoutvol,
           float value,
           uint3 volDim)
{
    // Vol coords
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Full volume size
    if (x >= volDim.x || y >= volDim.y || z >= volDim.z) return;

    // Write
    surf3Dwrite(value, inoutvol, x * 4, y, z);
}

#endif
