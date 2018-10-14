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



#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

extern "C"
__global__
void fourierFilter(float2* img, size_t stride, int pixelcount, float lp, float hp, float lps, float hps)
{
	//compute x,y indices 
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= pixelcount / 2 + 1) return;
	if (y >= pixelcount) return;

	float mx = (float)x;
	float my = (float)y;
	if (my > pixelcount * 0.5f)
		my = (pixelcount - my) * -1.0f;

	float dist = sqrtf(mx * mx + my * my);
	float fil = 0;

	lp = lp - lps;
	hp = hp + hps;
	//Low pass
	if (lp > 0)
	{
		if (dist <= lp) fil = 1;
	}
	else
	{
		if (dist <= pixelcount / 2 - 1) fil = 1;
	}
	//Gauss
	if (lps > 0)
	{
		float fil2;
		if (dist < lp) fil2 = 1;
		else fil2 = 0;

		fil2 = (-fil + 1.0f) * (float)expf(-((dist - lp) * (dist - lp) / (2 * lps * lps)));
		if (fil2 > 0.001f)
			fil = fil2;
	}

	if (lps > 0 && lp == 0 && hp == 0 && hps == 0)
		fil = (float)expf(-((dist - lp) * (dist - lp) / (2 * lps * lps)));

	if (hp > 0)
	{
		float fil2 = 0;
		if (dist >= hp) fil2 = 1;

		fil *= fil2;

		if (hps > 0)
		{
			float fil3 = 0;
			if (dist < hp) fil3 = 1;
			fil3 = (-fil2 + 1) * (float)expf(-((dist - hp) * (dist - hp) / (2 * hps * hps)));
			if (fil3 > 0.001f)
				fil = fil3;
		}
	}

	float2* row = (float2*)((char*)img + stride * y);
	float2 erg = row[x];
	erg.x *= fil;
	erg.y *= fil;
	row[x] = erg;
}

extern "C"
__global__
void conjMul(float2* complxA, float2* complxB, size_t stride, int pixelcount)
{
	//compute x,y,z indiced 
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= pixelcount / 2 + 1) return;
	if (y >= pixelcount) return;

	float2* rowA = (float2*)((char*)complxA + stride * y);
	float2* rowB = (float2*)((char*)complxB + stride * y);
	float2 a = rowA[x];
	float2 b = rowB[x];
	float2 erg;
	//conj. complex of a: -a.y
	erg.x = a.x * b.x + a.y * b.y;
	erg.y = a.x * b.y - a.y * b.x;
	
	rowA[x] = erg;

}



extern "C"
__global__
void maxShift(float* img, size_t stride, int pixelcount, int maxShift)
{
	//compute x,y,z indiced 
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= pixelcount) return;
	if (y >= pixelcount) return;

	float dist = 0;
	float mx = x;
	float my = y;

	if (mx > pixelcount / 2)
		mx = pixelcount - mx;

	if (my > pixelcount / 2)
		my = pixelcount - my;

	dist = sqrtf(mx * mx + my * my);

	if (dist > maxShift)
	{
		float* row = (float*)((char*)img + stride * y);
		row[x] = 0;
	}
}

extern "C"
__global__
void SumRow(float* img, size_t stride, int width, int height, float* sum)
{
	//compute x,y,z indiced 
	//unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.x * blockDim.x + threadIdx.x;

	//if (x >= pixelcount) return;
	if (y >= height) return;

	float val = 0;

	float* row = (float*)((char*)img + stride * y);

	for (int x = 0; x < width; x++)
	{
		val +=row[x];
	}

	sum[y] = val;
}



extern "C"
__global__
void CreateMask(unsigned char* mask, size_t stride, int width, int height, float* sum)
{
	//compute x,y,z indiced 
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width) return;
	if (y >= height) return;

	unsigned char maskVal = 255;

	if (sum[y] == 0)
	{
		maskVal = 0;
	}

	unsigned char* row = (unsigned char*)((char*)mask + stride * y);

	row[x] = maskVal;
}