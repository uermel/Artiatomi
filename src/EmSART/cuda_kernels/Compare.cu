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
* Compare.cu
* Compare kernel. Compares real and virtual
* projection and include volume traversal
* length per pixel.
*
**********************************************/
#ifndef COMPARE_CU
#define COMPARE_CU


#include <cuda.h>
#include "cutil.h"
#include "cutil_math.h"
#include "float.h"

#include "DeviceVariables.cuh"
//#include "float.h"


extern "C"
__global__
void dimBorders(int proj_x, int proj_y, size_t stride, float* image, float4 cutLength, float4 dimLength)
{
	// integer pixel coordinates
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= proj_x || y >= proj_y)
		return;


	float pixel = *(((float*)((char*)image + stride * y)) + x);
	float distXA = 1.0f;
	float distXB = 1.0f;
	float distYA = 1.0f;
	float distYB = 1.0f;


	//dim border
	if (y < cutLength.z + dimLength.z)
	{
		float w = (y - cutLength.z) / dimLength.z;
		if (w < 0) w = 0;
		distYB = 1.0f - expf(-(w * w * 9.0f));
	}
	else
	{
		distYB = 1.0f;
	}

	if (y > proj_y - dimLength.w - cutLength.w - 1)
	{

        // incorrect:
        //float w = ((proj_y - y - 1) - cutLength.w - (proj_y - 1)) / dimLength.w;

        //correct:
        float w = (proj_y - 1 - y - cutLength.w) / dimLength.w;

		if (w < 0) w = 0.0f;
		distYA = 1.0f - expf(-(w * w * 9.0f));
	}
	else
	{
		distYA = 1.0f;
	}

	if (x < cutLength.y + dimLength.y)
	{
		float w = (x - cutLength.y) / dimLength.y;
		if (w < 0) w = 0;
		distXB = 1.0f - expf(-(w * w * 9.0f));
	}
	else
	{
		distXB = 1.0f;
	}

	if (x > proj_x - dimLength.x - cutLength.x - 1)
	{

        // incorrect:
        //float w = ((proj_x - x - 1) - cutLength.x - (proj_x - 1)) / dimLength.x;

        // correct:
        float w = (proj_x - 1 - x - cutLength.x)/dimLength.x;

		if (w < 0) w = 0.0f;
		distXA = 1.0f - expf(-(w * w * 9.0f));
	}
	else
	{
		distXA = 1.0f;
	}


	*(((float*)((char*)image + stride * y)) + x) = distXA * distXB * distYA * distYB * pixel;
}

extern "C"
__global__
void subtract_error(int proj_x, int proj_y, size_t stride, float* real_raw, const float* error, const float* vol_distance_map)
{
    // integer pixel coordinates
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= proj_x || y >= proj_y)
        return;

    unsigned int i = (y * stride / sizeof(float)) + x;

    // Is pixel covered by volume?
    float val = vol_distance_map[i];

    // If yes, subtract error, if no set 0
    if (val >= 1.0f)
    {
        real_raw[i] = real_raw[i] - error[i];
    }
    else
    {
        real_raw[i] = 0.;
    }
}


__device__ inline float distance(float ax, float ay, float bx, float by)
{
	return sqrt((ax - bx) * (ax - bx) + (ay - by) * (ay - by));
}


__device__ float GetDistance(int2 v, int2 w, int x, int y)
{
	auto l2 = (float)((v.x - w.x) * (v.x - w.x) + (v.y - w.y) * (v.y - w.y));

	if (l2 == 0)
		return distance((float)v.x, (float)v.y, (float)x, (float)y);
	
	float x1, y1;
	x1 = (float)(x - v.x);
	y1 = (float)(y - v.y);
	
	float x2, y2;
	x2 = (float)(w.x - v.x);
	y2 = (float)(w.y - v.y);

	float dot = x1 * x2 + y1 * y2;
	
	float t = dot / l2;

	if (t < 0.0f) return distance((float)x, (float)y, (float)v.x, (float)v.y);
	else if (t > 1.0f) return distance((float)x, (float)y, (float)w.x, (float)w.y);
	else 
	{
		float x3, y3;
		x3 = (float)v.x + t * (float)(w.x - v.x);
		y3 = (float)v.y + t * (float)(w.y - v.y);
		return distance((float)x, (float)y, x3, y3);
	}
}

__device__ float GetDistance(float2 v, float2 w, float x, float y)
{
    float l2 = (v.x - w.x) * (v.x - w.x) + (v.y - w.y) * (v.y - w.y);

    if (l2 == 0)
        return distance(v.x, v.y, x, y);

    float x1, y1;
    x1 = x - v.x;
    y1 = y - v.y;

    float x2, y2;
    x2 = w.x - v.x;
    y2 = w.y - v.y;

    float dot = x1 * x2 + y1 * y2;

    float t = dot / l2;

    if (t < 0.0f) return distance(x, y, v.x, v.y);
    else if (t > 1.0f) return distance(x, y, w.x, w.y);
    else
    {
        float x3, y3;
        x3 = v.x + t * (w.x - v.x);
        y3 = v.y + t * (w.y - v.y);
        return distance(x, y, x3, y3);
    }
}


//extern "C"
//__global__
//void compare(uint2 projDim,
//             size_t stride,
//             const float* real_raw,
//             float* virtual_raw,
//             const float* vol_distance_map,
//             float realLength,
//             float4 cutLength,
//             float4 dimLength,
//             int2 p1, int2 p2, int2 p3, int2 p4)
//{
//    // integer pixel coordinates
//    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
//    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
//
//    if (x >= projDim.x || y >= projDim.y)
//        return;
//
//    float distX = 0.0f;
//    float distY = 0.0f;
//
//    if (       (p2.x - p1.x)*((int)y - p1.y) - (p2.y - p1.y)*((int)x - p1.x) < 0
//            && (p3.x - p4.x)*((int)y - p4.y) - (p3.y - p4.y)*((int)x - p4.x) < 0
//            && (p1.x - p3.x)*((int)y - p3.y) - (p1.y - p3.y)*((int)x - p3.x) < 0
//            && (p4.x - p2.x)*((int)y - p2.y) - (p4.y - p2.y)*((int)x - p2.x) < 0)
//    {
//        distX = 1;
//        distY = 1;
//
//        float minDistX = 3.f * (float)projDim.x;
//        float minDistY = 3.f * (float)projDim.y;
//        minDistX = fminf(minDistX, GetDistance(p1, p2, (int)x, (int)y));
//        minDistX = fminf(minDistX, GetDistance(p4, p3, (int)x, (int)y));
//        minDistX = fminf(minDistX, (float)x);
//        minDistX = fminf(minDistX, (float)projDim.x - (float)x - 1.f);
//
//        minDistY = fminf(minDistY, GetDistance(p3, p1, (int)x, (int)y));
//        minDistY = fminf(minDistY, GetDistance(p2, p4, (int)x, (int)y));
//        minDistY = fminf(minDistY, (float)y);
//        minDistY = fminf(minDistY, (float)projDim.x - (float)y - 1.f);
//
//
//        if (minDistX < cutLength.x + dimLength.x)
//        {
//            float w = (minDistX - cutLength.x) / dimLength.x;
//            if (w < 0) w = 0;
//            distX = 1.0f - expf(-(w * w * 9.0f));
//        }
//
//        if (minDistY < cutLength.y + dimLength.y)
//        {
//            float w = (minDistY - cutLength.y) / dimLength.y;
//            if (w < 0) w = 0;
//            distY = 1.0f - expf(-(w * w * 9.0f));
//        }
//    }
//
//
//    // save error difference in virtual projection
//    float distance = *(((float*)((char*)vol_distance_map + stride * y)) + x);
//    float real = (*(((float*)((char*)real_raw + stride * y)) + x)) * distX * distY;
//    float fwd_proj = (*(((float*)((char*)virtual_raw + stride * y)) + x));
//    float error;
//
//    float distXA = 1.0f;
//    float distXB = 1.0f;
//    float distYA = 1.0f;
//    float distYB = 1.0f;
//
//    if (distance >= 0)
//    {
//        //error = ((*(((float*)((char*)real_raw + stride * y)) + x)) - ((*(((float*)((char*)virtual_raw + stride * y)) + x) / projValScale) / val * realLength )) / realLength * projValScale;
//
//        error = (real - (fwd_proj / distance * realLength )) / realLength;
//
//        //error = (real - fwd_proj) / (realLength);
//        //error = ((*(((float*)((char*)real_raw + stride * y)) + x)) - ((*(((float*)((char*)virtual_raw + stride * y)) + x)))) / 1.f;
//    }
//    else
//    {
//        error = 0;
//    }
//
////	//dim border
////	if (y < cutLength.z + dimLength.z)
////	{
////		float w = (y - cutLength.z) / dimLength.z;
////		if (w<0) w = 0;
////		distYB = 1.0f - expf(-(w * w * 9.0f));
////	}
////	else
////    {
////        distYB = 1.0f;
////    }
////
////	if (y > proj_y - dimLength.w-cutLength.w - 1)
////	{
////
////        // incorrect:
////        //float w = ((proj_y - y - 1) - cutLength.w - (proj_y - 1)) / dimLength.w;
////
////        //correct:
////        float w = (proj_y - 1 - y - cutLength.w) / dimLength.w;
////
////		if (w<0) w = 0.0f;
////		distYA = 1.0f - expf(-(w * w * 9.0f));
////	}
////    else
////    {
////        distYA = 1.0f;
////    }
////
////	if (x < cutLength.y + dimLength.y)
////	{
////		float w = (x - cutLength.y) / dimLength.y;
////		if (w<0) w = 0;
////		distXB = 1.0f - expf(-(w * w * 9.0f));
////	}
////	else
////    {
////        distXB = 1.0f;
////    }
////
////	if (x > proj_x - dimLength.x-cutLength.x - 1)
////	{
////
////        // incorrect:
////        //float w = ((proj_x - x - 1) - cutLength.x - (proj_x - 1)) / dimLength.x;
////
////        // correct:
////        float w = (proj_x - 1 - x - cutLength.x)/dimLength.x;
////
////		if (w<0) w = 0.0f;
////		distXA = 1.0f - expf(-(w * w * 9.0f));
////	}
////    else
////    {
////        distXA = 1.0f;
////    }
//
//    *(((float*)((char*)virtual_raw + stride * y)) + x) = error;//distXA * distXB * distYA * distYB * error;
//}
//
//
//extern "C"
//__global__
//void cropBorder(int proj_x, int proj_y, size_t stride, float* image, float2 cutLength, float2 dimLength, int2 p1, int2 p2, int2 p3, int2 p4)
//{
//	// integer pixel coordinates
//	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
//    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
//
//	if (x >= proj_x || y >= proj_y)
//		return;
//
//    float distX = 0.0f;
//    float distY = 0.0f;
//
//
//	if ( (p2.x - p1.x)*((int)y - p1.y) - (p2.y - p1.y)*((int)x - p1.x) < 0
//	  && (p3.x - p4.x)*((int)y - p4.y) - (p3.y - p4.y)*((int)x - p4.x) < 0
//	  && (p1.x - p3.x)*((int)y - p3.y) - (p1.y - p3.y)*((int)x - p3.x) < 0
//	  && (p4.x - p2.x)*((int)y - p2.y) - (p4.y - p2.y)*((int)x - p2.x) < 0)
//	{
//		distX = 1;
//		distY = 1;
//
//		float minDistX = 3.f * (float)proj_x;
//		float minDistY = 3.f * (float)proj_y;
//		minDistX = fminf(minDistX, GetDistance(p1, p2, (int)x, (int)y));
//		minDistX = fminf(minDistX, GetDistance(p4, p3, (int)x, (int)y));
//		minDistX = fminf(minDistX, (float)x);
//		minDistX = fminf(minDistX, (float)proj_x - (float)x - 1.f);
//
//		minDistY = fminf(minDistY, GetDistance(p3, p1, (int)x, (int)y));
//		minDistY = fminf(minDistY, GetDistance(p2, p4, (int)x, (int)y));
//		minDistY = fminf(minDistY, (float)y);
//		minDistY = fminf(minDistY, (float)proj_y - (float)y - 1.f);
//
//
//		if (minDistX < cutLength.x + dimLength.x)
//		{
//			float w = (minDistX - cutLength.x) / dimLength.x;
//			if (w < 0) w = 0;
//			distX = 1.0f - expf(-(w * w * 9.0f));
//		}
//
//		if (minDistY < cutLength.y + dimLength.y)
//		{
//			float w = (minDistY - cutLength.y) / dimLength.y;
//			if (w < 0) w = 0;
//			distY = 1.0f - expf(-(w * w * 9.0f));
//		}
//	}
//
//
//	*(((float*)((char*)image + stride * y)) + x) *= distX * distY;
//}
//
//extern "C"
//__global__
//void cropBorderInv(int proj_x,
//                   int proj_y,
//                   size_t stride,
//                   float* image,
//                   float2 cutLength,
//                   float2 dimLength,
//                   float2 p1, float2 p2, float2 p3, float2 p4)
//{
//    // integer pixel coordinates
//    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
//    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
//
//    if (x >= proj_x || y >= proj_y)
//        return;
//
//    //printf("\n Crop Points DIST: pA %f %f pB %f %f pC %f %f pD %f %f\n", p1.x, p1.y, p2.x, p2.y, p3.x, p3.y, p4.x, p4.y);
//
//    float distX = 0.0f;
//    float distY = 0.0f;
//
//    auto xx = (float)x;
//    auto yy = (float)y;
//
//    auto max_x = (float) proj_x;
//    auto max_y = (float) proj_y;
//
//    if (    ((p2.x - p1.x)*(yy - p1.y) - (p2.y - p1.y)*(xx - p1.x)) <= 0
//         && ((p3.x - p4.x)*(yy - p4.y) - (p3.y - p4.y)*(xx - p4.x)) <= 0
//         && ((p1.x - p3.x)*(yy - p3.y) - (p1.y - p3.y)*(xx - p3.x)) <= 0
//         && ((p4.x - p2.x)*(yy - p2.y) - (p4.y - p2.y)*(xx - p2.x)) <= 0)
//    {
//        distX = 1;
//        distY = 1;
//
//        float minDistX = 3.f * (float)proj_x;
//        float minDistY = 3.f * (float)proj_y;
//        minDistX = fminf(minDistX, GetDistance(p1, p2, xx, yy));
//        minDistX = fminf(minDistX, GetDistance(p4, p3, xx, yy));
//        minDistX = fminf(minDistX, xx);
//        minDistX = fminf(minDistX, max_x - xx - 1.f);
//
//        minDistY = fminf(minDistY, GetDistance(p3, p1, xx, yy));
//        minDistY = fminf(minDistY, GetDistance(p2, p4, xx, yy));
//        minDistY = fminf(minDistY, yy);
//        minDistY = fminf(minDistY, max_y - yy - 1.f);
//
//
//        if (minDistX < cutLength.x + dimLength.x)
//        {
//            float w = (minDistX - cutLength.x) / dimLength.x;
//            if (w < 0) w = 0;
//            distX = 1.0f - expf(-(w * w * 9.0f));
//        }
//
//        if (minDistY < cutLength.y + dimLength.y)
//        {
//            float w = (minDistY - cutLength.y) / dimLength.y;
//            if (w < 0) w = 0;
//            distY = 1.0f - expf(-(w * w * 9.0f));
//        }
//    }
//
//    //printf("cropdist = %f\n", 1 - (distX * distY));
//
//    *(((float*)((char*)image + stride * y)) + x) = 1 - (distX * distY);
//}
//
//extern "C"
//__global__
//void cropBorderSlices(int proj_x,
//                      int proj_y,
//                      int sliceNumber,
//                      float* image,
//                      float2 cutLength,
//                      float2 dimLength,
//                      int2 p1,
//                      int2 p2,
//                      int2 p3,
//                      int2 p4)
//{
//    // integer pixel coordinates
//    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
//    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
//    const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;
//
//    if (x >= proj_x || y >= proj_y || z >= sliceNumber)
//        return;
//
//    float distX = 0.0f;
//    float distY = 0.0f;
//
//
//    if (    (p2.x - p1.x)*((int)y - p1.y) - (p2.y - p1.y)*((int)x - p1.x) < 0
//         && (p3.x - p4.x)*((int)y - p4.y) - (p3.y - p4.y)*((int)x - p4.x) < 0
//         && (p1.x - p3.x)*((int)y - p3.y) - (p1.y - p3.y)*((int)x - p3.x) < 0
//         && (p4.x - p2.x)*((int)y - p2.y) - (p4.y - p2.y)*((int)x - p2.x) < 0)
//    {
//        distX = 1;
//        distY = 1;
//
//        float minDistX = 3.f * (float)proj_x;
//        float minDistY = 3.f * (float)proj_y;
//        minDistX = fminf(minDistX, GetDistance(p1, p2, (int)x, (int)y));
//        minDistX = fminf(minDistX, GetDistance(p4, p3, (int)x, (int)y));
//        minDistX = fminf(minDistX, (float)x);
//        minDistX = fminf(minDistX, (float)proj_x - (float)x - 1.f);
//
//        minDistY = fminf(minDistY, GetDistance(p3, p1, (int)x, (int)y));
//        minDistY = fminf(minDistY, GetDistance(p2, p4, (int)x, (int)y));
//        minDistY = fminf(minDistY, (float)y);
//        minDistY = fminf(minDistY, (float)proj_y - (float)y - 1.f);
//
//
//        if (minDistX < cutLength.x + dimLength.x)
//        {
//            float w = (minDistX - cutLength.x) / dimLength.x;
//            if (w < 0) w = 0;
//            distX = 1.0f - expf(-(w * w * 9.0f));
//        }
//
//        if (minDistY < cutLength.y + dimLength.y)
//        {
//            float w = (minDistY - cutLength.y) / dimLength.y;
//            if (w < 0) w = 0;
//            distY = 1.0f - expf(-(w * w * 9.0f));
//        }
//    }
//
//
//    //*(((float*)((char*)image + stride * y)) + x) *= distX * distY;
//    image[z * proj_x * proj_y + y * proj_x + x] *= distX * distY;
//}
//
//extern "C"
//__global__
//void cropBorderSlicesInv(int proj_x,
//                         int proj_y,
//                         int sliceNumber,
//                         float* image,
//                         float2 cutLength,
//                         float2 dimLength,
//                         int2 p1,
//                         int2 p2,
//                         int2 p3,
//                         int2 p4)
//{
//    // integer pixel coordinates
//    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
//    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
//    const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;
//
//    if (x >= proj_x || y >= proj_y || z >= sliceNumber)
//        return;
//
//    float distX = 0.0f;
//    float distY = 0.0f;
//
//    if (       (p2.x - p1.x)*((int)y - p1.y) - (p2.y - p1.y)*((int)x - p1.x) < 0
//            && (p3.x - p4.x)*((int)y - p4.y) - (p3.y - p4.y)*((int)x - p4.x) < 0
//            && (p1.x - p3.x)*((int)y - p3.y) - (p1.y - p3.y)*((int)x - p3.x) < 0
//            && (p4.x - p2.x)*((int)y - p2.y) - (p4.y - p2.y)*((int)x - p2.x) < 0)
//    {
//        distX = 1;
//        distY = 1;
//
//        float minDistX = 3.f * (float)proj_x;
//        float minDistY = 3.f * (float)proj_y;
//        minDistX = fminf(minDistX, GetDistance(p1, p2, (int)x, (int)y));
//        minDistX = fminf(minDistX, GetDistance(p4, p3, (int)x, (int)y));
//        minDistX = fminf(minDistX, (float)x);
//        minDistX = fminf(minDistX, (float)proj_x - (float)x - 1.f);
//
//        minDistY = fminf(minDistY, GetDistance(p3, p1, (int)x, (int)y));
//        minDistY = fminf(minDistY, GetDistance(p2, p4, (int)x, (int)y));
//        minDistY = fminf(minDistY, (float)y);
//        minDistY = fminf(minDistY, (float)proj_y - (float)y - 1.f);
//
//
//        if (minDistX < cutLength.x + dimLength.x)
//        {
//            float w = (minDistX - cutLength.x) / dimLength.x;
//            if (w < 0) w = 0;
//            distX = 1.0f - expf(-(w * w * 9.0f));
//        }
//
//        if (minDistY < cutLength.y + dimLength.y)
//        {
//            float w = (minDistY - cutLength.y) / dimLength.y;
//            if (w < 0) w = 0;
//            distY = 1.0f - expf(-(w * w * 9.0f));
//        }
//    }
//
//    image[z * proj_x * proj_y + y * proj_x + x] *= 1 - (distX * distY);
//}

#define M_PI_F       3.14159265358979323846f

__device__ __constant__ float2 c_polygon[16];
__device__ __constant__ float c_polynorm[16];

template<bool inverse, bool slices, bool compare>
__device__
void cropComparePoly(uint2 projDim,
                     uint sliceNumber,
                     size_t stride,
                     float* image,
                     float cutLength,
                     float dimLength,
                     uint cornerCount,
                     const float* real_image,
                     const float* distance_image,
                     float real_distance,
                     float voxelSizeParent) {

    // integer pixel coordinates
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int z = 0;

    if (slices){
        z = blockIdx.z * blockDim.z + threadIdx.z;
    }

    // Bounds
    if (slices){
        if (x >= projDim.x || y >= projDim.y || z >= sliceNumber)
            return;
    } else {
        if (x >= projDim.x || y >= projDim.y)
            return;
    }

    // Sides
    int pos = 0;
    int neg = 0;

    // Point
    float2 p = make_float2((float)x, (float)y);

    // Is inside
    bool isInside = true;
    bool notCut = true;

    // Taper
    float taper = 1;

    // For each edge
    for (int i = 0; i < cornerCount; i++){

        // Point is corner
        if (p.x == c_polygon[i].x && p.y == c_polygon[i].y){
            isInside = true;
            taper *= min(max(0 - cutLength, 0.f), dimLength);
            notCut &= cutLength == 0;
            break;
        }

        // Corner 1
        float2 p1 = c_polygon[i];

        // Corner 2
        int i2 = (i+1) % (int)cornerCount;//(i == cornerCount-1) ? 0 : i+1;//(i+1) % (int)cornerCount;
        float2 p2 = c_polygon[i2];

        // Signed distance to edge
        float d = ((p.x - p1.x) * (p2.y - p1.y) - (p.y - p1.y) * (p2.x - p1.x))/c_polynorm[i];

        // Clamp absolute distance to cutLength and dimLength, and multiply taper
        float dist = min(max(abs(d) - cutLength, 0.f), dimLength);
        if (dimLength > 0) taper *= 1.f - (0.5f + 0.5f * __cosf(M_PI_F * dist / dimLength));
        //if (dimLength > 0) taper *= 1.f - expf(-(dist/dimLength * dist/dimLength * 9.0f));

        // Check if it is cut?
        notCut &= abs(d) >= cutLength;

        // Keep track of sign changes
        if (d > 0) pos++;
        if (d < 0) neg++;

        // If sign changed, is not inside
        if (pos > 0 && neg > 0) {
            isInside = false;
            notCut = false;
            taper = 0;
            break;
        }
    }

    float weight = (isInside && notCut) ? taper : 0.f;

    // Weight
    if (inverse) {
        weight = 1.f - weight;
    }

    // Save
    if (!compare) {
        if (slices) {
            image[z * projDim.x * projDim.y + y * projDim.x + x] *= weight;
        } else {
            *(((float *) ((char *) image + stride * y)) + x) *= weight;
        }
    } else {
        float distance = *(((float*)((char*)distance_image + stride * y)) + x) * voxelSizeParent * voxelSizeParent;
        float real = (*(((float*)((char*)real_image + stride * y)) + x)) * weight;
        float fwd_proj = (*(((float*)((char*)image + stride * y)) + x));
        float error;

        if (distance >= 1)
        {
            error = (real - fwd_proj) / (real_distance);
            //error = (real - (fwd_proj / distance * real_distance )) / real_distance;
        }
        else
        {
            error = 0;
        }

        *(((float*)((char*)image + stride * y)) + x) = error * weight;
    }
}

extern "C"
__global__
void cropBorder(uint2 projDim,
                size_t stride,
                float* image,
                float cutLength,
                float dimLength,
                uint cornerCount)
{
    cropComparePoly<false, false, false>(projDim,
                                         0,
                                         stride,
                                         image,
                                         cutLength,
                                         dimLength,
                                         cornerCount,
                                         nullptr,
                                         nullptr,
                                         0, 0);
}

extern "C"
__global__
void cropBorderInv(uint2 projDim,
                   size_t stride,
                   float* image,
                   float cutLength,
                   float dimLength,
                   uint cornerCount)
{
    cropComparePoly<true, false, false>(projDim,
                                        0,
                                        stride,
                                        image,
                                        cutLength,
                                        dimLength,
                                        cornerCount,
                                        nullptr,
                                        nullptr,
                                        0, 0);
}

extern "C"
__global__
void cropBorderSlices(uint2 projDim,
                      uint sliceNumber,
                      float* image,
                      float cutLength,
                      float dimLength,
                      uint cornerCount)
{
    cropComparePoly<false, true, false>(projDim,
                                        sliceNumber,
                                        0,
                                        image,
                                        cutLength,
                                        dimLength,
                                        cornerCount,
                                        nullptr,
                                        nullptr,
                                        0, 0);
}

extern "C"
__global__
void cropBorderSlicesInv(uint2 projDim,
                         uint sliceNumber,
                         float* image,
                         float cutLength,
                         float dimLength,
                         uint cornerCount)
{
    cropComparePoly<true, true, false>(projDim,
                                       sliceNumber,
                                       0,
                                       image,
                                       cutLength,
                                       dimLength,
                                       cornerCount,
                                       nullptr,
                                       nullptr,
                                       0, 0);
}

extern "C"
__global__
void compare(uint2 projDim,
             size_t stride,
             float* image,
             float cutLength,
             float dimLength,
             uint cornerCount,
             float* real_image,
             float* distance_image,
             float real_distance,
             float voxelSizeParent)
{
    cropComparePoly<false, false, true>(projDim,
                                        0,
                                        stride,
                                        image,
                                        cutLength,
                                        dimLength,
                                        cornerCount,
                                        real_image,
                                        distance_image,
                                        real_distance,
                                        voxelSizeParent);
}

extern "C"
__global__
void compareSpecial(uint2 projDim,
                     size_t stride,
                     float* fwdParent,
                     float* fwdChild,
                     float cutLength,
                     float dimLength,
                     uint cornerCount,
                     const float* real_image,
                     const float* distance_parent,
                     const float* distance_child,
                     float maxDistParent,
                     float maxDistChild,
                     float voxelSizeParent,
                     float voxelSizeChild) {

    // integer pixel coordinates
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= projDim.x || y >= projDim.y)
        return;

    // Sides
    int pos = 0;
    int neg = 0;

    // Point
    float2 p = make_float2((float)x, (float)y);

    // Is inside
    bool isInside = true;
    bool notCut = true;

    // Taper
    float taper = 1;

    // For each edge
    for (int i = 0; i < cornerCount; i++){

        // Point is corner
        if (p.x == c_polygon[i].x && p.y == c_polygon[i].y){
            isInside = true;
            taper *= min(max(0 - cutLength, 0.f), dimLength);
            notCut &= cutLength == 0;
            break;
        }

        // Corner 1
        float2 p1 = c_polygon[i];

        // Corner 2
        int i2 = (i+1) % (int)cornerCount;//(i == cornerCount-1) ? 0 : i+1;//(i+1) % (int)cornerCount;
        float2 p2 = c_polygon[i2];

        // Signed distance to edge
        float d = ((p.x - p1.x) * (p2.y - p1.y) - (p.y - p1.y) * (p2.x - p1.x))/c_polynorm[i];

        // Clamp absolute distance to cutLength and dimLength, and multiply taper
        float dist = min(max(abs(d) - cutLength, 0.f), dimLength);
        if (dimLength > 0) taper *= 1.f - (0.5f + 0.5f * __cosf(M_PI_F * dist / dimLength));
        //if (dimLength > 0) taper *= 1.f - expf(-(dist/dimLength * dist/dimLength * 9.0f));

        // Check if it is cut?
        notCut &= abs(d) >= cutLength;

        // Keep track of sign changes
        if (d > 0) pos++;
        if (d < 0) neg++;

        // If sign changed, is not inside
        if (pos > 0 && neg > 0) {
            isInside = false;
            notCut = false;
            taper = 0;
            break;
        }
    }

    float weight = (isInside && notCut) ? taper : 0.f;


    // Save
    float distance_p = *(((float*)((char*)distance_parent + stride * y)) + x) * voxelSizeParent * voxelSizeParent;
    float distance_c = *(((float*)((char*)distance_child + stride * y)) + x) * voxelSizeChild * voxelSizeChild;
    float real = (*(((float*)((char*)real_image + stride * y)) + x)) * weight;
    float proj_parent = (*(((float*)((char*)fwdParent + stride * y)) + x));
    float proj_child = (*(((float*)((char*)fwdChild + stride * y)) + x));
    float error;
    float error_child;
    //float error_parent;

    if (distance_p >= 1.f)
    {
        error = (real - proj_parent - proj_child);// / full_distance;
        //error = (real - (fwd_proj / distance * real_distance )) / real_distance;
    }
    else
    {
        error = 0.f;
        error_child = 0.f;
    }

    *(((float*)((char*)fwdParent + stride * y)) + x) = error / maxDistParent * weight;
    *(((float*)((char*)fwdChild + stride * y)) + x) = error / maxDistChild * weight;
}


#endif
