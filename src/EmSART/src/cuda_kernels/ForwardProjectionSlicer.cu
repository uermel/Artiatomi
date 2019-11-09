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

//texture< ushort, 3, cudaReadModeNormalizedFloat > t_dataset;
//texture< float, 3, cudaReadModeElementType > t_dataset;

typedef unsigned long long int ulli;

extern "C"
__global__
void slicer(int proj_x, int proj_y, size_t stride, float* projection, float tminDefocus, float tmaxDefocus, CUtexObject tex, int2 roiMin, int2 roiMax)
{
	float t_in;
	float t_out;
	float4 f;      //helper variable
	float3 g;											//helper variable
	float3 c_source;
	//float val = 0;

	// integer pixel coordinates
	const unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	//if (x >= proj_x || y >= proj_y) return;
	if (x >= roiMax.x || y >= roiMax.y) return;
	if (x < roiMin.x || y < roiMin.y) return;


	c_source = c_detektor;

	//c_source = c_source + ((float)x + 0.5f) * c_uPitch;
	//c_source = c_source + ((float)y + 0.5f) * c_vPitch;
	float temp = 0.0f;
	g.z = 0;
	g.x = 0;

	//No oversampling now (to enable OS use osx = osy = 0.25f)
	for (float  osx = 0.25f; osx < 0.8f; osx+=0.5f)
	{
		for (float osy = 0.25f; osy < 0.8f; osy+=0.5f)
		{
			float xAniso;
			float yAniso;

			MatrixVector3Mul(c_magAniso, (float)x + osx, (float)y + osy, xAniso, yAniso);
			c_source = c_detektor;
			c_source = c_source + (xAniso) * c_uPitch;
			c_source = c_source + (yAniso) * c_vPitch;

			//////////// BOX INTERSECTION (partial Volume) /////////////////
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
			//if( t_out > t_in && t_in < t && t < t_out) //
			if( (t_out - t_in) > 0.0f)
			{
				t_in = fmaxf(t_in, tminDefocus);
				t_out = fminf(t_out, tmaxDefocus);

				g.x++;
				g.z += (t_out - t_in);
				// calculate entry point
				f.x = c_source.x;
				f.y = c_source.y;
				f.z = c_source.z;
				
				f.w = t_in;
				
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

				/*f.x += (t * c_projNorm.x);
				f.y += (t * c_projNorm.y);
				f.z += (t * c_projNorm.z);

				f.x = (f.x - c_bBoxMin.x) * c_volumeBBoxRcp.x * c_volumeDim.x;
				f.y = (f.y - c_bBoxMin.y) * c_volumeBBoxRcp.y * c_volumeDim.y;
				f.z = (f.z - c_bBoxMin.z) * c_volumeBBoxRcp.z * c_volumeDim.z - c_zShiftForPartialVolume;
				*///if (x == 2048 && y == 2048)
					//val = f.z;
				//val += tex3D(t_dataset, f.x, f.y, f.z);
				//float distX = 1.0f;
				//float distY = 1.0f;
				//float distZ = 1.0f;
				////dim border
				//float filterWidth = 150.0f;
				//if (f.y < filterWidth)
				//{
				//	float w = f.y / filterWidth;
				//	if (w<0) w = 0;
				//	distY = 1.0f - expf(-(w * w * 9.0f));
				//}
				//else if (f.y > c_volumeDimComplete.y - filterWidth)
				//{
				//	float w = (c_volumeDimComplete.y-f.y-1.0f) / filterWidth;
				//	if (w<0) w = 0;
				//	distY = 1.0f - expf(-(w * w * 9.0f));
				//}

				//if (f.x < filterWidth)
				//{
				//	float w = f.x / filterWidth;
				//	if (w<0) w = 0;
				//	distX = 1.0f - expf(-(w * w * 9.0f));
				//}
				//else if (f.x > c_volumeDimComplete.x - filterWidth)
				//{
				//	float w = (c_volumeDimComplete.x-f.x-1.0f) / filterWidth;
				//	if (w<0) w = 0;
				//	distX = 1.0f - expf(-(w * w * 9.0f));
				//}

				//if (f.z < 50.0f)
				//{
				//	float w = f.z / 50.0f;
				//	if (w<0) w = 0;
				//	distZ = 1.0f - expf(-(w * w * 9.0f));
				//}
				//else if (f.z > c_volumeDimComplete.z - 50.0f)
				//{
				//	float w = (c_volumeDimComplete.z-f.z-1.0f) / 50.0f;
				//	if (w<0) w = 0;
				//	distZ = 1.0f - expf(-(w * w * 9.0f));
				//}
				//val = val * distX * distY * distZ;
				//val = val * (expf(-(distX * distX + distY * distY + distZ * distZ)));
				
			}
		}
	}

	unsigned int i = (y * stride / sizeof(float)) + x;
	projection[i] += temp * 0.25f; // With Oversampling use * 0.25f
}

extern "C"
__global__
void volTraversalLength(int proj_x, int proj_y, size_t stride, float* volume_traversal_length, int2 roiMin, int2 roiMax)
{
	float t_in;
	float t_out;
	float3 c_source;
	float val = 0;

	//volume_traversal_length[0] = c_detektor.x;
	//volume_traversal_length[1] = c_detektor.y;
	//volume_traversal_length[2] = c_detektor.z;
	// integer pixel coordinates
	const unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int y = (blockIdx.y * blockDim.y) + threadIdx.y;


	//if (x >= proj_x || y >= proj_y) return;
	if (x >= roiMax.x || y >= roiMax.y) return;
	if (x < roiMin.x || y < roiMin.y) return;

	float xAniso;
	float yAniso;

	MatrixVector3Mul(c_magAniso, (float)x + 0.5f, (float)y + 0.5f, xAniso, yAniso);
	c_source = c_detektor;

	c_source = c_source + (xAniso) * c_uPitch;
	c_source = c_source + (yAniso) * c_vPitch;


	//No oversampling now (to enable OS use osx = osy = 0.25f)
	for (float  osx = 0.5f; osx < 0.8f; osx+=0.5f)
	{
		for (float osy = 0.5f; osy < 0.8f; osy+=0.5f)
		{
			//////////// BOX INTERSECTION (partial Volume) /////////////////
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
			
			// if the ray hits the dataset (partial Volume)
			if( t_out > t_in)
			{
				val = t_out - t_in;				
			}
		}
	}

	unsigned int i = (y * stride / sizeof(float)) + x;
	volume_traversal_length[i] += val; // With Oversampling use * 0.25f
}
#endif
