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
#include "cubic_interpolation/cubicTex3D.cu"

//texture< ushort, 3, cudaReadModeNormalizedFloat > t_dataset;
//texture< float, 3, cudaReadModeElementType > t_dataset;

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
void sample(CUtexObject involume, CUsurfObject outvol, float3 volDim)
{
    // Vol coords
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Full volume size
    if (x >= volDim.x || y >= volDim.y || z >= volDim.z) return;

    // Interpolate
    float3 coord = make_float3((float)x+0.5f, (float)y+0.5f, (float)z+0.5f);
    float val = cubicTex3D<float>(involume, coord);

    // Write
    surf3Dwrite(val, outvol, x * 4, y, z);
}

#endif
