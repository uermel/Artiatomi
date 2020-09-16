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


#ifndef WBPWEIGHTING_CU
#define WBPWEIGHTING_CU

//Includes for IntelliSense 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <texture_fetch_functions.h>
#include <surface_functions.h>


#include <stdio.h>
#include "cufft.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


// transform vector by matrix

enum FilterMethod
{
	FM_RAMP,
	FM_EXACT,
	FM_CONTRAST2,
	FM_CONTRAST10,
	FM_CONTRAST30
};

__device__ float sinc(float x)
{
	float res = 1;
	if (x != 0)
	{
		res = sinf(M_PI * x) / (M_PI * x);
	}
	return res;
}

extern "C"
__global__ 
void wbpWeighting(cuComplex* img, size_t stride, unsigned int pixelcount, float psiAngle, FilterMethod fm, int proj_index, int projectionCount, float thickness, const float* __restrict__ tiltAngles)
{
	//compute x,y,z indiced
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;	
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (x >= pixelcount/2 + 1) return;
	if (y >= pixelcount) return;

	float xpos = x;
	float ypos = y;
	if (ypos > pixelcount * 0.5f)
		ypos = (pixelcount - ypos) * -1.0f;

	float temp = xpos;
	float sinus =  __sinf(psiAngle);
	float cosin =  __cosf(psiAngle);

	xpos = cosin * xpos - sinus * ypos;
	ypos = sinus * temp + cosin * ypos;

	float length = ypos / (pixelcount / 2.0f);
	float weight = 1;
	switch (fm)
	{
	case FM_RAMP:
		weight = fminf(abs(length), 1.0f);
		break;
	case FM_EXACT:
		{
			//psiAngle += M_PI * 0.5f;
			float x_st = -ypos * cos(tiltAngles[proj_index])*sin(psiAngle) - xpos *cos(psiAngle);
			float y_st =  ypos * cos(tiltAngles[proj_index])*cos(psiAngle) - xpos *sin(psiAngle);
			float z_st =  ypos * sin(tiltAngles[proj_index]);

            //float x_st = -xpos * cos(tiltAngles[proj_index])*sin(psiAngle) - ypos *cos(psiAngle);
            //float y_st =  xpos * cos(tiltAngles[proj_index])*cos(psiAngle) - ypos *sin(psiAngle);
            //float z_st =  xpos * sin(tiltAngles[proj_index]);

			float w = 0;

			for (int tilt = 0; tilt < projectionCount; tilt++)
			{
				if (tilt != proj_index && tiltAngles[tilt] != -999.0f)
				{
					// Berechnung der geometrischen Distanz zu der Ebene
					float d_tmp = x_st*sin(tiltAngles[tilt])*sin(psiAngle) - y_st*sin(tiltAngles[tilt])*cos(psiAngle) + z_st*cos(tiltAngles[tilt]);

					float d2 = abs(sin(tiltAngles[tilt])) * thickness + cos(tiltAngles[tilt]);

					if (abs(d_tmp) > d2)
						d_tmp = d2;

					w += sinc(d_tmp / d2);
				}
			}
			// Normalize, such that the center is 1 / number of projections, and
			// the boundary is one!!
			w += 1.0f;
			w = 1.0f / w;

			//Added normalization(zero frequencies set to zero)
			if (ypos == 0)
			{
				w = 0;
			}
			weight = w;
		}
		break;
	case FM_CONTRAST2:
		{//1.000528623371163   0.006455924123082   0.005311341463650   0.001511856638478 1024
		 //1.000654227857550   0.006008581017124   0.004159659493151   0.000975903396538 1856
			const float p1 = 1.000654227857550f;
			const float p2 = 0.006008581017124f;
			const float p3 = 0.004159659493151f;
			const float p4 = 0.000975903396538f;
			if (length == 0)
			{
				weight = 0;
			}
			else
			{
				float logfl = logf(abs(length));
				weight = p1 + p2 * logfl + p3 * logfl * logfl + p4 * logfl * logfl * logfl;
			}
			weight = fmaxf(0, fminf(weight, 1));
		}
		break;
	case FM_CONTRAST10:		
		{//1.001771328635575   0.019634409648661   0.014871972759515   0.004962873817517 1024
		 //1.003784816598589   0.029016377161629   0.019582940715148   0.004559409669984 1856
			const float p1 = 1.003784816598589f;
			const float p2 = 0.029016377161629f;
			const float p3 = 0.019582940715148f;
			const float p4 = 0.004559409669984f;
			if (length == 0)
			{
				weight = 0;
			}
			else
			{
				float logfl = logf(abs(length));
				weight = p1 + p2 * logfl + p3 * logfl * logfl + p4 * logfl * logfl * logfl;
			}
			weight = fmaxf(0, fminf(weight, 1));
		}
		break;
	case FM_CONTRAST30:		
		{//0.998187224092783   0.019542575617926   0.010359773048706   0.006975890938967 1024
		 //0.999884616010943   0.000307646262566   0.004742915272196   0.004806551368900 1856
			const float p1 = 0.999884616010943f;
			const float p2 = 0.000307646262566f;
			const float p3 = 0.004742915272196f;
			const float p4 = 0.004806551368900f;
			if (length == 0)
			{
				weight = 0;
			}
			else
			{
				float logfl = logf(abs(length));
				weight = p1 + p2 * logfl + p3 * logfl * logfl + p4 * logfl * logfl * logfl;
			}
			weight = fmaxf(0, fminf(weight, 1));
		}
		break;
	}
	
	cuComplex res = *(((cuComplex*)((char*)img + stride * y)) + x);
	res.x *= weight;
	res.y *= weight;
	/*res.x = weight;
	res.y = 0;*/

	*(((cuComplex*)((char*)img + stride * y)) + x) = res;
	
}

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

	float2 erg = *(((float2*)((char*)img + stride * y)) + x);
	erg.x *= fil;
	erg.y *= fil;
	*(((float2*)((char*)img + stride * y)) + x) = erg;
}

extern "C"
__global__
void doseWeighting(float2* img, size_t stride, int pixelcount, float dose, float pixelsize)
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

	dist = dist / (pixelcount / 2 / pixelsize);
	fil = expf(-dose * dist);

	float2 erg = *(((float2*)((char*)img + stride * y)) + x);
	erg.x *= fil;
	erg.y *= fil;
	*(((float2*)((char*)img + stride * y)) + x) = erg;
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

	float2 a = *(((float2*)((char*)complxA + stride * y)) + x);
	float2 b = *(((float2*)((char*)complxB + stride * y)) + x);
	float2 erg;
	//conj. complex of a: -a.y
	erg.x = a.x * b.x + a.y * b.y;
	erg.y = a.x * b.y - a.y * b.x;
	*(((float2*)((char*)complxA + stride * y)) + x) = erg;

}

extern "C"
__global__
void conjMulPC(float2* complxA, float2* complxB, size_t stride, int pixelcount)
{
    //compute x,y,z indiced
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= pixelcount / 2 + 1) return;
    if (y >= pixelcount) return;

    float2 a = *(((float2*)((char*)complxA + stride * y)) + x);
    float2 b = *(((float2*)((char*)complxB + stride * y)) + x);
    float2 erg;

    //conj. complex of a: -a.y
    erg.x = a.x * b.x + a.y * b.y;
    erg.y = a.x * b.y - a.y * b.x;

    float amplitude = sqrtf(erg.x * erg.x + erg.y * erg.y);

    if (amplitude != 0)
    {
        erg.x /= amplitude;
        erg.y /= amplitude;
    }
    else
    {
        erg.x = erg.y = 0;
    }

    *(((float2*)((char*)complxA + stride * y)) + x) = erg;

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
		*(((float*)((char*)img + stride * y)) + x) = 0;
	}
}

extern "C"
__global__
void maxShiftWeighted(float* img, size_t stride, int pixelcount, int maxShift)
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
		*(((float*)((char*)img + stride * y)) + x) = 0;
	}
	else
	{
		*(((float*)((char*)img + stride * y)) + x) /= dist+0.0001f;
	}
}

extern "C"
__global__
void findPeak(float* img, size_t stride, char* maskInv, size_t strideMask, int pixelcount, float maxThreshold)
{
	//compute x,y,z indiced 
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= pixelcount) return;
	if (y >= pixelcount) return;

	
	int xm = x - 1;
	int ym = y - 1;
	int xp = x + 1;
	int yp = y + 1;

	//note: the cc map is fft-shifted!
	//wrap negative indices
	if (xm < 0)
	{
		xm = pixelcount - 1;
	}
	if (ym < 0)
	{
		ym = pixelcount - 1;
	}
	if (xp >= pixelcount)
	{
		xp = 0;
	}
	if (yp >= pixelcount)
	{
		yp = 0;
	}
	
	float* rowImgP = (float*)((char*)img + stride * yp);
	float* rowImg = (float*)((char*)img + stride * y);
	float* rowImgM = (float*)((char*)img + stride * ym);
	unsigned char* rowMask = (unsigned char*)((char*)maskInv + strideMask * y);

	float val = rowImg[x];
	if (rowImg[xm] < val && rowImg[xp] < val &&
		rowImgM[x] < val && rowImgP[x] < val &&
		rowImgM[xm] <= val && rowImgP[xm] < val &&
		rowImgM[xp] <= val && rowImgP[xp] < val &&
		val >= maxThreshold
		)
	{
		rowMask[x] = 0;
	}
	else
	{
		rowMask[x] = 1;
	}
}


texture<float, 3, cudaReadModeElementType> texVol;
extern "C"
__global__ void rot3d(int size, float3 rotMat0, float3 rotMat1, float3 rotMat2, float* outVol)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

	float center = size / 2;

	float3 vox = make_float3(x - center, y - center, z - center);
	float3 rotVox;
	rotVox.x = center + rotMat0.x * vox.x + rotMat1.x * vox.y + rotMat2.x * vox.z;
	rotVox.y = center + rotMat0.y * vox.x + rotMat1.y * vox.y + rotMat2.y * vox.z;
	rotVox.z = center + rotMat0.z * vox.x + rotMat1.z * vox.y + rotMat2.z * vox.z;

	outVol[z * size * size + y * size + x] = tex3D(texVol, rotVox.x + 0.5f, rotVox.y + 0.5f, rotVox.z + 0.5f);
}

extern "C"
__global__ void sphericalMask3D(int size, float radius, float* outVol)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

    float center = size / 2;

    float3 vox = make_float3(x - center, y - center, z - center);

    float rad = sqrtf(vox.x*vox.x + vox.y*vox.y + vox.z*vox.z);
    if (rad > radius) {
        outVol[z * size * size + y * size + x] = 1;
    }
    else {
        outVol[z * size * size + y * size + x] = 0;
    }
}
#endif


/// Apply mask to a volume split on multiple GPUs
// volume -- CUDA surface object bound to CUDA arrays of the volume on different nodes
// mask -- CUDA device pointer to mask volume (needs to be replicated on each node)
// tempStore -- CUDA device pointer for storing the modified region to revert the volume later
// volmin -- The min global coordinate on this MPI node
// volmax -- The max global coodinate on this MPI node
// dimMask -- Dimensions of the mask volume
// radiusMask -- radius of the mask applied
// centerInVol -- The center of the mask in global volume coordinates (Matlab convention, i.e. 1-based, so -1)
extern "C"
__global__
void applyMask(cudaSurfaceObject_t volume, const float* mask, float* tempStore, int3 volmin, int3 volmax, int3 dimMask, int3 radiusMask, int3 centerInVol)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Return if outside mask
    if ((x >= dimMask.x) || (y >= dimMask.y) || (z >= dimMask.z)) return;

    // Compute mask position in complete volume (-1 because Matlab --> C++)
    int3 maskGrid;
    maskGrid.x = x - radiusMask.x + centerInVol.x - 1;
    maskGrid.y = y - radiusMask.y + centerInVol.y - 1;
    maskGrid.z = z - radiusMask.z + centerInVol.z - 1;

    // Return if outside current subvolume
    bool mincond = (maskGrid.x < volmin.x) || (maskGrid.y < volmin.y) || (maskGrid.z < volmin.z);
    bool maxcond = (maskGrid.x >= volmax.x) || (maskGrid.y >= volmax.y) || (maskGrid.z >= volmax.z);
    if (mincond || maxcond) return;

    // Adjust coordinates to current subvolume
    int3 volGrid;
    volGrid.x = maskGrid.x - volmin.x;
    volGrid.y = maskGrid.y - volmin.y;
    volGrid.z = maskGrid.z - volmin.z;

    // Do the deed.
    size_t mask_idx = z * dimMask.x * dimMask.y + y * dimMask.x + x;

    float data;
    surf3Dread<float>(&data, volume, volGrid.x * sizeof(float), volGrid.y, volGrid.z, cudaBoundaryModeTrap);
    tempStore[mask_idx] = data;
    data *= mask[mask_idx];
    surf3Dwrite<float>(data, volume, volGrid.x * sizeof(float), volGrid.y, volGrid.z, cudaBoundaryModeTrap);
}


extern "C"
__global__
void restoreVolume(cudaSurfaceObject_t volume, const float* tempStore, int3 volmin, int3 volmax, int3 dimMask, int3 radiusMask, int3 centerInVol)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Return if outside mask
    if ((x >= dimMask.x) || (y >= dimMask.y) || (z >= dimMask.z)) return;

    // Compute mask position in complete volume (-1 because Matlab --> C++)
    int3 maskGrid;
    maskGrid.x = x - radiusMask.x + centerInVol.x - 1;
    maskGrid.y = y - radiusMask.y + centerInVol.y - 1;
    maskGrid.z = z - radiusMask.z + centerInVol.z - 1;

    // Return if outside current MPI-subvolume
    bool mincond = (maskGrid.x < volmin.x) || (maskGrid.y < volmin.y) || (maskGrid.z < volmin.z);
    bool maxcond = (maskGrid.x >= volmax.x) || (maskGrid.y >= volmax.y) || (maskGrid.z >= volmax.z);
    if (mincond || maxcond) return;

    // Adjust coordinates to current MPI-subvolume
    int3 volGrid;
    volGrid.x = maskGrid.x - volmin.x;
    volGrid.y = maskGrid.y - volmin.y;
    volGrid.z = maskGrid.z - volmin.z;

    // Do the deed.
    size_t mask_idx = z * dimMask.x * dimMask.y + y * dimMask.x + x;

    float data = tempStore[mask_idx];
    surf3Dwrite<float>(data, volume, volGrid.x * sizeof(float), volGrid.y, volGrid.z, cudaBoundaryModeTrap);
}