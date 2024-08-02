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


#ifndef EMSARTDEFAULT_H
#define EMSARTDEFAULT_H

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#ifdef WIN32
typedef long long long64;
typedef unsigned long long ulong64;
typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned char uchar;
#else
typedef long long long64;
typedef unsigned long long ulong64;
typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned char uchar;
#endif


#ifdef WIN32
#include <string>
#include <math.h>
#include <float.h>
#else
#include <cstring>
#include <cfloat>
#include <cmath>
#endif

#include <cstdio>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <climits>
#include <sstream>
//#include <vector_types.h>
//#include <vector_functions.h>
//#include <cutil_math_.h>
#include "cuda_kernels/common_types.h"
#include "Matrix.h"

#define RAD2DEG(angrad) (angrad/M_PI * 180)
#define DEG2RAD(angdeg) (angdeg/180 * M_PI)

// Dim3 without CUDA
inline dim3 make_dim3(uint a, uint b, uint c)
{
	dim3 ret;
	ret.x = a;
	ret.y = b;
	ret.z = c;
	return ret;
}

inline dim3 make_dim3(uint3 val)
{
	dim3 ret;
	ret.x = val.x;
	ret.y = val.y;
	ret.z = val.z;
	return ret;
}

// Matrix to GPU Matrix
template<class T>
float3 MatrixTo3x1(Matrix<T> &Min){
    float3 Mout;
    Mout.x = (float)Min(0, 0);
    Mout.y = (float)Min(1, 0);
    Mout.z = (float)Min(2, 0);
    return Mout;
}
template float3 MatrixTo3x1(Matrix<float> &Min);
template float3 MatrixTo3x1(Matrix<double> &Min);

template<class T>
float3x3 MatrixTo3x3(Matrix<T> &Min){
    float3x3 Mout;
    Mout.m[0].x = (float)Min(0, 0);
    Mout.m[0].y = (float)Min(0, 1);
    Mout.m[0].z = (float)Min(0, 2);
    Mout.m[1].x = (float)Min(1, 0);
    Mout.m[1].y = (float)Min(1, 1);
    Mout.m[1].z = (float)Min(1, 2);
    Mout.m[2].x = (float)Min(2, 0);
    Mout.m[2].y = (float)Min(2, 1);
    Mout.m[2].z = (float)Min(2, 2);
    return Mout;
}
template float3x3 MatrixTo3x3(Matrix<float> &Min);
template float3x3 MatrixTo3x3(Matrix<double> &Min);

template<class T>
float4x4 MatrixTo4x4(Matrix<T> &Min){
    float4x4 Mout;
    Mout.m[0].x = Min(0, 0);
    Mout.m[0].y = Min(0, 1);
    Mout.m[0].z = Min(0, 2);
    Mout.m[0].w = Min(0, 3);
    Mout.m[1].x = Min(1, 0);
    Mout.m[1].y = Min(1, 1);
    Mout.m[1].z = Min(1, 2);
    Mout.m[1].w = Min(1, 3);
    Mout.m[2].x = Min(2, 0);
    Mout.m[2].y = Min(2, 1);
    Mout.m[2].z = Min(2, 2);
    Mout.m[2].w = Min(2, 3);
    Mout.m[3].x = Min(3, 0);
    Mout.m[3].y = Min(3, 1);
    Mout.m[3].z = Min(3, 2);
    Mout.m[3].w = Min(3, 3);
    return Mout;
}
template float4x4 MatrixTo4x4(Matrix<float> &Min);
template float4x4 MatrixTo4x4(Matrix<double> &Min);

//// Matrix Vector Multiply
//inline void MatrixVector3Mul(float3x3& M, float xIn, float yIn, float& xOut, float& yOut)
//{
//    xOut = M.m[0].x * xIn + M.m[0].y * yIn + M.m[0].z * 1.f;
//    yOut = M.m[1].x * xIn + M.m[1].y * yIn + M.m[1].z * 1.f;
//}
//
//inline void MatrixVector3Mul(float3x3 M, float3* v)
//{
//    float3 erg;
//    erg.x = M.m[0].x * v->x + M.m[0].y * v->y + M.m[0].z * v->z;
//    erg.y = M.m[1].x * v->x + M.m[1].y * v->y + M.m[1].z * v->z;
//    erg.z = M.m[2].x * v->x + M.m[2].y * v->y + M.m[2].z * v->z;
//    *v = erg;
//}
//
//inline void MatrixVector3Mul(float4x4 M, float3* v)
//{
//    float3 erg;
//    erg.x = M.m[0].x * v->x + M.m[0].y * v->y + M.m[0].z * v->z + 1.f * M.m[0].w;
//    erg.y = M.m[1].x * v->x + M.m[1].y * v->y + M.m[1].z * v->z + 1.f * M.m[1].w;
//    erg.z = M.m[2].x * v->x + M.m[2].y * v->y + M.m[2].z * v->z + 1.f * M.m[2].w;
//    *v = erg;
//}

//inline float3 make_float3(uint3 val)
//{
//	float3 ret;
//	ret.x = val.x;
//	ret.y = val.y;
//	ret.z = val.z;
//	return ret;
//}
//
//inline uint3 make_uint3(uint a, uint b, uint c)
//{
//	uint3 ret;
//	ret.x = a;
//	ret.y = b;
//	ret.z = c;
//	return ret;
//}
//
//inline uint3 make_uint3(dim3 val)
//{
//	uint3 ret;
//	ret.x = val.x;
//	ret.y = val.y;
//	ret.z = val.z;
//	return ret;
//}
//
//inline float3 make_uint3(dim3 val)
//{
//	uint3 ret;
//	ret.x = val.x;
//	ret.y = val.y;
//	ret.z = val.z;
//	return ret;
//}





#endif
