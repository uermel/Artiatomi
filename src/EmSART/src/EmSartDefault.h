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
#include <string.h>
#include <float.h>
#include <math.h>
#endif
#include <stdio.h>
#include <vector>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <limits.h>
#include <sstream>
#include <vector_types.h>
#include <vector_functions.h>
#include <cutil_math_.h>

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
