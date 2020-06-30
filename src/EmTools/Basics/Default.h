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


#ifndef DEFAULT_H
#define DEFAULT_H

#include <stdio.h>
#include <string>
#include <vector>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <limits.h>

//switches to enable/disable technology-specific implementations
//#define NO_CUDA
#ifndef NO_CUDA
#define USE_CUDA
#endif
#ifndef NO_OPENCL
#define USE_OPENCL
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS //NVIDIA has limited OpenCL 2 support
#endif
#define USE_SSE

#define CUDA_THREAD_ID 123 //some unique number...
#define OPENCL_THREAD_ID 1234

#ifdef _MSC_VER
#define _USE_WINDOWS_COMPILER_SETTINGS
#endif

#ifdef __GNUC__
#define _USE_LINUX_COMPILER_SETTINGS
#endif
#if defined (__APPLE__) || defined(MACOSX)
#define _USE_APPLE_COMPILER_SETTINGS
#endif


#ifdef _USE_WINDOWS_COMPILER_SETTINGS
typedef long long long64;
typedef unsigned long long ulong64;
typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned char uchar;
#ifdef USE_OPENCL
static const char* CL_GL_SHARING_EXT = "cl_khr_gl_sharing";
#endif
#else
#ifdef _USE_LINUX_COMPILER_SETTINGS
typedef long long long64;
typedef unsigned long long ulong64;
typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned char uchar;
#include <cstring>
#include <cfloat>

#ifdef USE_OPENCL
static const char* CL_GL_SHARING_EXT = "cl_khr_gl_sharing";
#endif //USE_OPENCL
#endif //_USE_LINUX_COMPILER_SETTINGS
#endif //ELSE
#ifdef _USE_APPLE_COMPILER_SETTINGS
typedef long long long64;
typedef unsigned long long ulong64;
typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned char uchar;
#ifdef USE_OPENCL
static const char* CL_GL_SHARING_EXT = "cl_APPLE_gl_sharing";
#endif
#endif


enum DataType_enum
{
	DT_UNKNOWN,
	DT_UCHAR,
	DT_USHORT,
	DT_UINT,
	DT_ULONG,
	DT_FLOAT,
	DT_DOUBLE,
	DT_CHAR,
	DT_SHORT,
	DT_INT,
	DT_LONG,
	DT_UCHAR2,
	DT_FLOAT2,
	DT_SHORT2,
	DT_USHORT2,
	DT_DOUBLE2,
	DT_FLOAT3,
	DT_FLOAT4,
	DT_UCHAR3,
	DT_UCHAR4,
	DT_SHORT3,
	DT_SHORT4,
	DT_USHORT3,
	DT_USHORT4,
	DT_RGB
};

inline size_t GetDataTypeSize(DataType_enum aDataType)
{
	switch (aDataType)
	{
	case DT_UNKNOWN:
		return 1;
	case DT_UCHAR:
		return 1;
	case DT_USHORT:
		return 2;
	case DT_UINT:
		return 4;
	case DT_ULONG:
		return 8;
	case DT_FLOAT:
		return 4;
	case DT_DOUBLE:
		return 8;
	case DT_CHAR:
		return 1;
	case DT_SHORT:
		return 2;
	case DT_INT:
		return 4;
	case DT_LONG:
		return 8;
	case DT_FLOAT2:
		return 8;
	case DT_SHORT2:
		return 4;
	case DT_DOUBLE2:
		return 16;
	case DT_FLOAT3:
		return 12;
	case DT_FLOAT4:
		return 16;
	case DT_UCHAR2:
		return 2;
	case DT_USHORT2:
		return 2;
	case DT_UCHAR3:
		return 3;
	case DT_UCHAR4:
		return 4;
	case DT_SHORT3:
		return 6;
	case DT_USHORT3:
		return 6;
	case DT_SHORT4:
		return 8;
	case DT_USHORT4:
		return 8;
	case DT_RGB:
		return 3;
	default:
		return 1;
	}
}

enum ImageType_enum
{
	IT_TILTSERIES,
	IT_SINGLEFRAME,
	IT_MOVIESTACK,
	IT_VOLUME,
	IT_NONE
};

#endif
