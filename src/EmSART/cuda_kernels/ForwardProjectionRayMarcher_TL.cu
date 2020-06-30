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
#include "cutil.h"
#include "cutil_math.h"
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>
#include "float.h"
#include <builtin_types.h>
#include <vector_functions.h>

#include "Constants.h"
#include "DeviceVariables.cuh"

#include "float.h"


extern "C"
__global__
void march(int proj_x, int proj_y, size_t stride, float* projection, float* volume_traversal_length, CUtexObject tex, int2 roiMin, int2 roiMax)
{
	float t_in;
	float t_out;
	float4 f;											//helper variable
	float3 g;											//helper variable
	float3 c_source;

	// integer pixel coordinates
	const unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x >= roiMax.x || y >= roiMax.y) return;
	if (x < roiMin.x || y < roiMin.y) return;

	c_source = c_detektor;

	float temp = 0.0f;
	g.z = 0;
	g.x = 0;

	for (float  osx = 0.25f; osx < 0.9f; osx+=0.5f)
	{
		for (float osy = 0.25f; osy < 0.9f; osy+=0.5f)
		{
			float xAniso;
			float yAniso;

			MatrixVector3Mul(c_magAniso, (float)x + osx, (float)y + osy, xAniso, yAniso);
			c_source = c_detektor;
			c_source = c_source + (xAniso) * c_uPitch;
			c_source = c_source + (yAniso) * c_vPitch;

			float3 tEntry;
			tEntry.x = (c_bBoxMin.x - c_source.x) / (c_projNorm.x);
			tEntry.y = (c_bBoxMin.y - c_source.y) / (c_projNorm.y);
			tEntry.z = (c_bBoxMin.z - c_source.z) / (c_projNorm.z);

			float3 tExit;
			tExit.x = (c_bBoxMax.x - c_source.x) / (c_projNorm.x);
			tExit.y = (c_bBoxMax.y - c_source.y) / (c_projNorm.y);
			tExit.z = (c_bBoxMax.z - c_source.z) / (c_projNorm.z);


			float3 tmin = fminf(tEntry, tExit);
			float3 tmax = fmaxf(tEntry, tExit);

			t_in  = fmaxf(fmaxf(tmin.x, tmin.y), tmin.z);
			t_out = fminf(fminf(tmax.x, tmax.y), tmax.z);

			////////////////////////////////////////////////////////////////

			//default grey value
			g.y = 0.f;

			// if the ray hits the dataset (partial Volume)
			if( (t_out - t_in) > 0.0f)
			{
				g.x++;
				g.z += (t_out - t_in);
				// calculate entry point
				f.x = c_source.x;
				f.y = c_source.y;
				f.z = c_source.z;

				f.w = t_in;

				f.x += (f.w * c_projNorm.x);
				f.y += (f.w * c_projNorm.y);
				f.z += (f.w * c_projNorm.z);

				while (t_in <= t_out)
				{
					f.x = (f.x - c_bBoxMin.x) * c_volumeBBoxRcp.x * c_volumeDim.x;
					f.y = (f.y - c_bBoxMin.y) * c_volumeBBoxRcp.y * c_volumeDim.y;
					f.z = (f.z - c_bBoxMin.z) * c_volumeBBoxRcp.z * c_volumeDim.z - c_zShiftForPartialVolume;
			
					float test = tex3D<float>(tex, f.x, f.y, f.z);
					
					temp += test * c_voxelSize.x * 0.15f;


					t_in += c_voxelSize.x * 0.15f;

					f.x = c_source.x;
					f.y = c_source.y;
					f.z = c_source.z;
					f.w = t_in ;				

					f.x += f.w * c_projNorm.x;
					f.y += f.w * c_projNorm.y;
					f.z += f.w * c_projNorm.z;
				}

			}
		}
	}

	*(((float*)((char*)projection + stride * y)) + x) += temp * 0.25f; //  With Oversampling use * 0.25f
	*(((float*)((char*)volume_traversal_length + stride * y)) + x) = fmaxf(0,g.z/g.x);
}

#endif
