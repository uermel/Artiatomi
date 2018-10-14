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


#ifndef CTF_CU
#define CTF_CU


//Includes for IntelliSense 
#define _SIZE_T_DEFINED
#ifndef __CUDACC__
#define __CUDACC__
#endif
#ifndef __cplusplus
#define __cplusplus
#endif
#include <cuda.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>
#include "float.h"
#include "math.h"
#include "cufft.h"
#include <builtin_types.h>
#include <vector_functions.h>

#define M_PI       3.14159265358979323846f
//#define _voltage (300.0f)
#define h ((float)6.63E-34) //Planck's quantum
#define c ((float)3.00E+08) //Light speed
#define Cs (c_cs * 0.001f)
#define Cc (c_cs * 0.001f)
				
#define PhaseShift (0)
#define EnergySpread (0.7f) //eV
#define E0 (511) //keV
#define RelativisticCorrectionFactor ((1 + c_voltage / (E0 * 1000))/(1 + ((c_voltage*1000) / (2 * E0 * 1000))))
#define H ((Cc * EnergySpread * RelativisticCorrectionFactor) / (c_voltage * 1000))

#define a1 (1.494f) //Scat.Profile Carbon Amplitude 1
#define a2 (0.937f) //Scat.Profile Carbon Amplitude 2
#define b1 (23.22f * (float)1E-20) //Scat.Profile Carbon Halfwidth 1
#define b2 (3.79f * (float)1E-20)  //Scat.Profile Carbon Halfwidth 2

#define lambda ((h * c) / sqrtf(((2 * E0 * c_voltage * 1000.0f * 1000.0f) + (c_voltage * c_voltage * 1000.0f * 1000.0f)) * 1.602E-19 * 1.602E-19))

__device__ __constant__ float c_cs;
__device__ __constant__ float c_voltage;
__device__ __constant__ float c_openingAngle;
__device__ __constant__ float c_ampContrast;
__device__ __constant__ float c_phaseContrast;
__device__ __constant__ float c_pixelsize;
__device__ __constant__ float c_pixelcount;
__device__ __constant__ float c_maxFreq;
__device__ __constant__ float c_freqStepSize;
//__device__ __constant__ float c_lambda;
__device__ __constant__ float c_applyScatteringProfile;
__device__ __constant__ float c_applyEnvelopeFunction;

// transform vector by matrix

extern "C"
__global__ 
void ctf(cuComplex* ctf, size_t stride, float defocusMin, float defocusMax, float angle, bool applyForFP, bool phaseFlipOnly, float WienerFilterNoiseLevel, float4 betaFac)
{
	//compute x,y,z indiced
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;	
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (x >= c_pixelcount/2 + 1) return;
	if (y >= c_pixelcount) return;

			
	//float length = sqrtf((x-c_pixelcount/2) * (x-c_pixelcount/2) + (y-c_pixelcount/2) * (y-c_pixelcount/2));
	float xpos = x;
	float ypos = y;//-c_pixelcount/2;
	if (ypos > c_pixelcount * 0.5f)
		ypos = (c_pixelcount - ypos) * -1.0f;
	
	float alpha;
	if (xpos == 0)
		alpha = (M_PI * 0.5f);
	else
		alpha = (atan2(ypos , xpos));
		
	float beta = ((alpha - angle));
	
	float def0 = defocusMin;
	float def1 = defocusMax;

	float defocus = def0 + (1 - cos(2*beta)) * (def1 - def0);

	float length = sqrtf(xpos * xpos + ypos * ypos);

	//float angle = 76.0f / M_PI * 180.0f;
	//float stretchX = 1.066f;
	/*float angle = 80.0f / M_PI * 180.0f;
	float stretchX = 1.026f;

	float temp = cos(angle) * xpos - sin(angle) * ypos;
	ypos = sin(angle) * xpos + cos(angle) * ypos;
	xpos = temp;
	xpos *= stretchX;*/
	
	//float w = 1.0f;
	//if (length > c_pixelcount * 0.5f)
	//{
	//	w = (length - c_pixelcount * 0.1f) / (c_pixelcount - c_pixelcount * 0.1f) ;
	//	w = expf(-(w * w * 9 * 9));
	//}
    length *= c_freqStepSize;

    float o = expf(-14.238829f * (c_openingAngle * c_openingAngle * ((Cs * lambda * lambda * length * length * length - defocus * length) * (Cs * lambda * lambda * length * length * length - defocus * length))));
    float p = expf(-((0.943359f * lambda * length * length * H) * (0.943359f * lambda * length * length * H)));
    float q = (a1 * expf(-b1 * (length * length)) + a2 * expf(-b2 * (length * length))) / 2.431f;

    float m = -PhaseShift + (M_PI / 2.0f) * (Cs * lambda * lambda * lambda * length * length * length * length - 2 * defocus * lambda * length * length);
    float n = c_phaseContrast * sinf(m) + c_ampContrast * cosf(m);

    //float r = (o * p * ((1 - c_applyScatteringProfile) + (q * c_applyScatteringProfile))) * c_applyEnvelopeFunction + (1 - c_applyEnvelopeFunction);

    cuComplex res = ctf[y * stride / sizeof(cuComplex) + x];
//res.x = 1;res.y=0;
    //float limit1 = 4084-(sinf(5.5f * 3.14159265f / 180.0f) * x);
    //float limit2 = 4095-(sinf(1.5f * 3.14159265f / 180.0f) * x);
	
    if (applyForFP && sqrtf(xpos * xpos + ypos * ypos) > betaFac.x && !phaseFlipOnly)// && length < 317382812)
    {
		//double faq = coeefs[0] * Math.Exp(-(Math.Abs(coeefs[1])) * freq - coeefs[2] * freq * freq - coeefs[3] * freq * freq * freq);
		//const float coeff0 = 1.0f;
		length = length / 100000000.0f;
		/*const float coeff1 = 0.0000000000000001f;
		const float coeff2 = -0.00652468f;
		const float coeff3 = 0.003948657f;*/
		/*const float coeff1 = 0;
		const float coeff2 = 0.008f;
		const float coeff3 = 0;*/
		float coeff1 = betaFac.y;
		float coeff2 = betaFac.z;
		float coeff3 = betaFac.w;
		float expfun = expf((-coeff1 * length - coeff2 * length * length - coeff3 * length * length * length));
		expfun = max(expfun, 0.01f);
		float val = n * expfun;
		//val = fmaxf(val, 0.005f);
		if (abs(val) < 0.0001f && val >=0 ) val = 0.0001f;
		if (abs(val) < 0.0001f && val < 0 ) val = -0.0001f;
		
		
		res.x = res.x * -val;
		res.y = res.y * -val;
		//res.x *= val;
		//res.y *= val;
    }
  //   if (n > 0)//)// && y <= limit2 && length < 317382812)&& y < 4040 && (y <= limit1 && y > 1)
  //  {
		//res.x *= -1.0f;
		//res.y *= -1.0f;
  //  }
    /*if (n > 0 && !absolut)//)// && y <= limit2 && length < 317382812)&& y < 4040 && (y <= limit1 && y > 1)
    {
		res.x *= -1.0f;
		res.y *= -1.0f;
    }*/
	//else
	//{
	//	/*res.x = 0;
	//	res.y = 0;*/
	//}
	//res.x = 1;
    if (!applyForFP && sqrtf(xpos * xpos + ypos * ypos) > betaFac.x && !phaseFlipOnly)// && length < 317382812)
    {
		//double faq = coeefs[0] * Math.Exp(-(Math.Abs(coeefs[1])) * freq - coeefs[2] * freq * freq - coeefs[3] * freq * freq * freq);
		//const float coeff0 = 1.0f;
		length = length / 100000000.0f;
		float coeff1 = betaFac.y;
		float coeff2 = betaFac.z;
		float coeff3 = betaFac.w;
		float expfun = expf((-coeff1 * length - coeff2 * length * length - coeff3 * length * length * length));
		expfun = max(expfun, WienerFilterNoiseLevel);
		float val = n * expfun;
		//val = fmaxf(val, 0.005f);
		/*if (abs(val) < 0.00001f && val >=0 ) val = 0.00001f;
		if (abs(val) < 0.00001f && val < 0 ) val = -0.00001f;*/
		
		
		res.x = res.x * -val / (val * val + WienerFilterNoiseLevel);
		res.y = res.y * -val / (val * val + WienerFilterNoiseLevel);
		//res.x = expfun;
		//res.y = expfun;
    }
    
	if (phaseFlipOnly)
	{
		if (n >= 0)
		{
			res.x = -res.x;
			res.y = -res.y;
		}
	}
    
    //else
    //{
	//if (abs(r*n) > 0.5f)
	//{
	    //res.x /= abs(r*n);
	    //res.y /= abs(r*n);
        //}
    //}
    //res.x = 1;
    //if (absolut)
    //    res.y = r * n * w;
    //else
    //{
        //res.y = r * n;
        //if (r * n < 0) //res.y = 1.0f;
        //else 
	//res.y *= -1.0f;	
    //res.x = 0; res.y = 0;
    ctf[y * stride / sizeof(cuComplex) + x] = res;
    //}
//res.x = w;
}

#endif
