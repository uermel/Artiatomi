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
//#include "cutil_math.h"
#include "cufft.h"
#include <builtin_types.h>
//#include <vector_functions.h>
#include "common_types.h"

#define M_PI       3.14159265358979323846f
//#define _voltage (300.0f)
#define h ((float)6.63E-34) //Planck's quantum
#define c ((float)3.00E+08) //Light speed
#define Cs_ (c_cs * 0.001f)
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

#define iu make_cuComplex(0, 1) // Imaginary unit
#define lambda_ ((h * c) / sqrtf(((2 * E0 * c_voltage * 1000.0f * 1000.0f) + (c_voltage * c_voltage * 1000.0f * 1000.0f)) * 1.602E-19 * 1.602E-19))

#define ji make_float2(0.f, 1.f)

__device__ __constant__ float c_cs;
__device__ __constant__ float c_voltage;
__device__ __constant__ float c_openingAngle;
__device__ __constant__ float c_ampContrast;
__device__ __constant__ float c_phaseContrast;
__device__ __constant__ float c_pixelsize;
__device__ __constant__ float2 c_pixelcount;
__device__ __constant__ float c_maxFreq;
__device__ __constant__ float2 c_freqStepSize;
//__device__ __constant__ float c_lambda;
__device__ __constant__ float c_applyScatteringProfile;
__device__ __constant__ float c_applyEnvelopeFunction;


extern "C"
__global__ 
void ctf(cuComplex* ctf, size_t stride, float defocusMin, float defocusMax, float angle, bool applyForFP, bool phaseFlipOnly, float WienerFilterNoiseLevel, float4 betaFac)
{
	//compute x,y indiced
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;	
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (x >= c_pixelcount.x/2 + 1) return;
	if (y >= c_pixelcount.y) return;

			
	float xpos = (float)x;
	float ypos = (float)y;
	if (ypos > c_pixelcount.y * 0.5f)
		ypos = (c_pixelcount.y - ypos) * -1.0f;

    // Astigmatic Defocus
	float alpha;
	if (xpos == 0)
		alpha = (M_PI * 0.5f);
	else
		alpha = (atan2(ypos , xpos));
	float beta = ((alpha - angle));
	float def0 = defocusMin;
	float def1 = defocusMax;
	float defocus = def0 + (1 - cos(2*beta)) * (def1 - def0) * 0.5f;

    xpos = xpos * c_freqStepSize.x;
    ypos = ypos * c_freqStepSize.y;
	float length = sqrtf(xpos * xpos + ypos * ypos);

    //length *= c_freqStepSize;

    float m = -PhaseShift + (M_PI / 2.0f) * (Cs_ * lambda_ * lambda_ * lambda_ * length * length * length * length - 2 * defocus * lambda_ * length * length);
    float n = c_phaseContrast * sinf(m) + c_ampContrast * cosf(m);
	
	cuComplex res = *(((cuComplex*)((char*)ctf + stride * y)) + x);
	
    if (applyForFP && sqrtf(xpos * xpos + ypos * ypos) > betaFac.x && !phaseFlipOnly)// && length < 317382812)
    {
		length = length / 100000000.0f;
		float coeff1 = betaFac.y;
		float coeff2 = betaFac.z;
		float coeff3 = betaFac.w;
		float expfun = expf((-coeff1 * length - coeff2 * length * length - coeff3 * length * length * length));
		expfun = max(expfun, 0.01f);
		float val = n * expfun;
		if (abs(val) < 0.0001f && val >=0 ) val = 0.0001f;
		if (abs(val) < 0.0001f && val < 0 ) val = -0.0001f;
		
		
		res.x = res.x * -val;
		res.y = res.y * -val;
    }

    if (!applyForFP && sqrtf(xpos * xpos + ypos * ypos) > betaFac.x && !phaseFlipOnly)// && length < 317382812)
    {
		length = length / 100000000.0f;
		float coeff1 = betaFac.y;
		float coeff2 = betaFac.z;
		float coeff3 = betaFac.w;
		float expfun = expf((-coeff1 * length - coeff2 * length * length - coeff3 * length * length * length));
		expfun = max(expfun, WienerFilterNoiseLevel);
		float val = n * expfun;
		
		res.x = res.x * -val / (val * val + WienerFilterNoiseLevel);
		res.y = res.y * -val / (val * val + WienerFilterNoiseLevel);
    }
    
	if (phaseFlipOnly)
	{
		if (n >= 0)
		{
			res.x = -res.x;
			res.y = -res.y;
		}
	}
    
	*(((cuComplex*)((char*)ctf + stride * y)) + x) = res;
}

extern "C"
__global__
void ctfSliced(cuComplex* ctf,
               int ctf_x,
               int ctf_y,
               int sliceNumber,
               //int maxsize,
               float defocusMin,
               float defocusMax,
               float* offsets,
               float angle,
               bool applyForFP,
               bool phaseFlipOnly,
               float WienerFilterNoiseLevel,
               float4 betaFac)
{
    //compute x,y indiced
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

//    if (x >= c_pixelcount/2 + 1) return;
//    if (y >= c_pixelcount) return;
//    if (z >= sliceNumber) return;
    if (x >= ctf_x) return;
    if (y >= ctf_y) return;
    if (z >= sliceNumber) return;


    float xpos = (float)x;
    float ypos = (float)y;
    if (ypos > c_pixelcount.y * 0.5f)
        ypos = (c_pixelcount.y - ypos) * -1.0f;

    float alpha;
    if (xpos == 0)
        alpha = (M_PI * 0.5f);
    else
        alpha = (atan2(ypos , xpos));

    float beta = ((alpha - angle));

    //printf("offset: %f\n", offsets[z]);

    float def0 = defocusMin + offsets[z];
    float def1 = defocusMax + offsets[z];

    float defocus = def0 + (1 - cos(2*beta)) * (def1 - def0) * 0.5f;

    xpos = xpos * c_freqStepSize.x;
    ypos = ypos * c_freqStepSize.y;
    float length = sqrtf(xpos * xpos + ypos * ypos);

    //length *= c_freqStepSize;

    float o = expf(-14.238829f * (c_openingAngle * c_openingAngle * ((Cs_ * lambda_ * lambda_ * length * length * length - defocus * length) * (Cs_ * lambda_ * lambda_ * length * length * length - defocus * length))));
    float p = expf(-((0.943359f * lambda_ * length * length * H) * (0.943359f * lambda_ * length * length * H)));
    float q = (a1 * expf(-b1 * (length * length)) + a2 * expf(-b2 * (length * length))) / 2.431f;

    float m = -PhaseShift + (M_PI / 2.0f) * (Cs_ * lambda_ * lambda_ * lambda_ * length * length * length * length - 2 * defocus * lambda_ * length * length);
    float n = c_phaseContrast * sinf(m) + c_ampContrast * cosf(m);

    cuComplex res = ctf[z * ctf_y * ctf_x + y * ctf_x + x];//*(((cuComplex*)((char*)ctf + stride * y)) + x);

    if (applyForFP && sqrtf(xpos * xpos + ypos * ypos) > betaFac.x && !phaseFlipOnly)// && length < 317382812)
    {
        length = length / 100000000.0f;
        float coeff1 = betaFac.y;
        float coeff2 = betaFac.z;
        float coeff3 = betaFac.w;
        float expfun = expf((-coeff1 * length - coeff2 * length * length - coeff3 * length * length * length));
        expfun = max(expfun, 0.01f);
        float val = n * expfun;
        if (abs(val) < 0.0001f && val >=0 ) val = 0.0001f;
        if (abs(val) < 0.0001f && val < 0 ) val = -0.0001f;


        res.x = res.x * -val;
        res.y = res.y * -val;
    }

    if (!applyForFP && sqrtf(xpos * xpos + ypos * ypos) > betaFac.x && !phaseFlipOnly)// && length < 317382812)
    {
        length = length / 100000000.0f;
        float coeff1 = betaFac.y;
        float coeff2 = betaFac.z;
        float coeff3 = betaFac.w;
        float expfun = expf((-coeff1 * length - coeff2 * length * length - coeff3 * length * length * length));
        expfun = max(expfun, WienerFilterNoiseLevel);
        float val = n * expfun;

        res.x = res.x * -val / (val * val + WienerFilterNoiseLevel);
        res.y = res.y * -val / (val * val + WienerFilterNoiseLevel);
        //res.x = val;
        //res.y = val;
    }

    if (phaseFlipOnly)
    {
        if (n >= 0)
        {
            res.x = -res.x;
            res.y = -res.y;
        }
    }

    //*(((cuComplex*)((char*)ctf + stride * y)) + x) = res;
    ctf[z * ctf_y * ctf_x + y * ctf_x + x] = res;
}


__device__ __forceinline__ cuComplex cmulf(cuComplex a, cuComplex b)
{
    cuComplex res;
    res.x = a.x * b.x - a.y * b.y;
    res.y = a.x * b.y + a.y * b.x;
    return res;
}

__device__ void atomicAddComplex(cuComplex* a, cuComplex b)
{
    auto  *x = (float*)a;
    float *y = x+1;
    //cuComplex res;

    atomicAdd(x, b.x);
    atomicAdd(y, b.y);
}

template<bool computeProjNoCTF>
__device__
void postFilterSum_base(const cuComplex* image,    // 2D image stack
                        const cuComplex* filter,   // 2D image
                        cuComplex* res,            // 2D image
                        uint2 fftDim,
                        const ctfConstants c_glb,
                        const ctfImageConstants c_img,
                        const float* offsets,
                        int2 minmaxSlice,
                        cuComplex* res2)
{
    //compute x,y indices
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z + minmaxSlice.x;
    const unsigned int z_batch = blockIdx.z * blockDim.z + threadIdx.z;

    // Bounds
    if (x >= fftDim.x) return;
    if (y >= fftDim.y) return;
    if (z >= c_img.sliceNumber) return;
    if (z_batch > minmaxSlice.y) return;

    // FFT shift
    auto xpos = (float)x;
    auto ypos = (float)y;
    if (ypos > (c_img.pixelcount.y) * 0.5f)
        ypos = (c_img.pixelcount.y - ypos) * -1.0f;

    // Correction for non-square image
    xpos = xpos * c_img.asymCorrFac.x;
    ypos = ypos * c_img.asymCorrFac.y;

    // Compute defocus
    float alpha;

    if (xpos == 0)
        alpha = (M_PI * 0.5f);
    else
        alpha = (atan2(ypos , xpos));

    float beta = ((alpha - c_img.astigAngle));
    float def0 = c_img.defocusMin + offsets[z];
    float def1 = c_img.defocusMax + offsets[z];
    float defocus = def0 + (1 - cos(2*beta)) * (def1 - def0) * 0.5f;

    // CTF
    xpos = xpos * c_img.freqStepSize.x;
    ypos = ypos * c_img.freqStepSize.y;
    float length = sqrtf(xpos * xpos + ypos * ypos);

    float m = -c_img.phaseShift + (M_PI / 2.0f) * (c_glb.cs_m * c_glb.lambda * c_glb.lambda * c_glb.lambda * length * length * length * length - 2 * defocus * c_glb.lambda * length * length);
    float n = -c_glb.phaseContrast * sinf(m) + c_glb.ampContrast * cosf(m);

    // Data
    cuComplex signal = image[z_batch * fftDim.y * fftDim.x + y * fftDim.x + x];
    cuComplex val_filter = filter[y * fftDim.x + x];

    // B-Factor
    length = length / 100000000.0f;
    float coeff1 = c_img.B;
    float coeff2 = c_img.Bsqr;
    float coeff3 = c_img.Bcub;
    float expfun = expf((-coeff1 * length - coeff2 * length * length - coeff3 * length * length * length));
    //expfun = max(expfun, WienerFilterNoiseLevel);
    n = n * expfun;

    // The CTF
    cuComplex val = make_cuComplex(n, 0);

    // Convolution with the spline filter
    signal = cmulf(signal, val_filter);

    if (computeProjNoCTF) {
        atomicAddComplex(res2 + y * fftDim.x + x, signal / (c_img.pixelcount.x * c_img.pixelcount.y));
    }

    // Convolution with the CTF
    signal = cmulf(signal, val);

    // Normalize for IFFT
    signal = signal / (c_img.pixelcount.x * c_img.pixelcount.y);
    
    atomicAddComplex(res + y * fftDim.x + x, signal);
}

extern "C"
__global__
void postFilterSum(const cuComplex* image,    // 2D image stack
                   const cuComplex* filter,   // 2D image
                   cuComplex* res,            // 2D image
                   uint2 fftDim,
                   const ctfConstants c_glb,
                   const ctfImageConstants c_img,
                   const float* offsets,
                   int2 minmaxSlice)
{
    postFilterSum_base<false>(image,
                              filter,
                              res,
                              fftDim,
                              c_glb,
                              c_img,
                              offsets,
                              minmaxSlice,
                              0);
}

extern "C"
__global__
void postFilterSumCTFfree(const cuComplex* image,    // 2D image stack
                          const cuComplex* filter,   // 2D image
                          cuComplex* res,            // 2D image
                          uint2 fftDim,
                          const ctfConstants c_glb,
                          const ctfImageConstants c_img,
                          const float* offsets,
                          int2 minmaxSlice,
                          cuComplex* res2)
{
    postFilterSum_base<true>(image,
                             filter,
                             res,
                             fftDim,
                             c_glb,
                             c_img,
                             offsets,
                             minmaxSlice,
                             res2);
}


__device__ __forceinline__ cuComplex cexpf (cuComplex z)
{
    cuComplex res;
    float t = expf(z.x);
    sincosf(z.y, &res.y, &res.x);
    res.x *= t;
    res.y *= t;
    return res;
}


template<bool applyWienerFilter, bool applySNR, bool relion_weighting>
__device__ void preFilterSpread(const cuComplex* image,    // 2D image
                                const cuComplex* filter,   // 2D image
                                cuComplex* res,            // Stack of 2D
                                CUtexObject snr,           // 1D texture
                                CUtexObject np,           // 1D texture
                                uint2 fftDim,
                                ctfConstants c_glb,
                                ctfImageConstants c_img,
                                const float* offsets,
                                int2 minmaxSlice,
                                float deconvStrength)
{
    //compute x,y indices
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z + minmaxSlice.x;
    const unsigned int z_batch = blockIdx.z * blockDim.z + threadIdx.z;

    // Bounds
    if (x >= fftDim.x) return;
    if (y >= fftDim.y) return;
    if (z >= c_img.sliceNumber) return;
    if (z_batch > minmaxSlice.y) return;

    // FFT shift
    auto xpos = (float)x;
    auto ypos = (float)y;
    if (ypos > (c_img.pixelcount.y) * 0.5f)
        ypos = (c_img.pixelcount.y - ypos) * -1.0f;

    // Correction for non-square image
    xpos = xpos * c_img.asymCorrFac.x;
    ypos = ypos * c_img.asymCorrFac.y;

    // For SNR interpolation
    float r = sqrt(xpos * xpos + ypos * ypos);
    float shell = floor(r);

    // Compute defocus
    float alpha;
    if (xpos == 0)
        alpha = (M_PI * 0.5f);
    else
        alpha = (atan2(ypos , xpos));

    float beta = ((alpha - c_img.astigAngle));
    float def0 = c_img.defocusMin + offsets[z];
    float def1 = c_img.defocusMax + offsets[z];
    float defocus = def0 + (1 - cos(2*beta)) * (def1 - def0) * 0.5f;

    // CTF
    xpos = xpos * c_img.freqStepSize.x;
    ypos = ypos * c_img.freqStepSize.y;
    float length = sqrtf(xpos * xpos + ypos * ypos);

    float m = -c_img.phaseShift + (M_PI / 2.0f) * (c_glb.cs_m * c_glb.lambda * c_glb.lambda * c_glb.lambda * length * length * length * length - 2 * defocus * c_glb.lambda * length * length);
    float n = -c_glb.phaseContrast * sinf(m) + c_glb.ampContrast * cosf(m);

    // Data
    cuComplex signal = image[y * fftDim.x + x];
    cuComplex val_filter = filter[y * fftDim.x + x];

    // B-Factor
    length = length / 100000000.0f;
    float coeff1 = c_img.B;
    float coeff2 = c_img.Bsqr;
    float coeff3 = c_img.Bcub;
    float expfun = expf((-coeff1 * length - coeff2 * length * length - coeff3 * length * length * length));
    //n = n * expfun;

    // Wiener Filter
    if (applyWienerFilter) {
        if (applySNR) {
            auto var_pow = tex1D<float>(np, shell + 0.5f);
            auto sig_pow = tex1D<float>(snr, shell + 0.5f);

            if (relion_weighting){
                var_pow = 0.5f * var_pow * var_pow;
                sig_pow = 0.5f * sig_pow * sig_pow;
                n = (expfun * n/var_pow) / (((n * n)/var_pow) + 1 / (sig_pow * deconvStrength));
            } else {
                float stor = sig_pow / var_pow;
                n = (expfun * n) / (((n * n)) + 1 / (stor * deconvStrength));
            }
        } else {
            n = (expfun * n) / (n * n + c_glb.WienerFilterNoiseLevel);
        }
    }

    // The Wiener Filter
    cuComplex val = make_cuComplex(n, 0);

    // Convolution with Wiener Filter and spline filter
    signal = cmulf(signal, val_filter);
    signal = cmulf(signal, val);

    // Normalize for IFFT
    signal = signal / (c_img.pixelcount.x * c_img.pixelcount.y);

    res[z_batch * fftDim.y * fftDim.x + y * fftDim.x + x] = signal;
}

// No Wiener Filter and no SNR
extern "C"
__global__
void preFilterSpread(const cuComplex* image,    // 2D image
                     const cuComplex* filter,   // 2D image
                     cuComplex* res,            // Stack of 2D
                     uint2 fftDim,
                     ctfConstants c_glb,
                     ctfImageConstants c_img,
                     const float* offsets,
                     int2 minmaxSlice)
{
    preFilterSpread<false, false, false>(image,    // 2D image
                                         filter,   // 2D image
                                         res,      // Stack of 2D
                                         0,      // 1D texture
                                         0,
                                         fftDim,
                                         c_glb,
                                         c_img,
                                         offsets,
                                         minmaxSlice,
                                         0);
}

// Wiener Filter with ad-hoc constant
extern "C"
__global__
void preFilterSpreadAdHoc(const cuComplex* image,    // 2D image
                          const cuComplex* filter,   // 2D image
                          cuComplex* res,            // Stack of 2D
                          uint2 fftDim,
                          ctfConstants c_glb,
                          ctfImageConstants c_img,
                          const float* offsets,
                          int2 minmaxSlice)
{
    preFilterSpread<true, false, false>(image,    // 2D image
                                        filter,   // 2D image
                                        res,      // Stack of 2D
                                        0,      // 1D texture
                                        0,
                                        fftDim,
                                        c_glb,
                                        c_img,
                                        offsets,
                                        minmaxSlice,
                                        0);
}

// Wiener Filter with SNR-based constant
extern "C"
__global__
void preFilterSpreadSNR(const cuComplex* image,    // 2D image
                        const cuComplex* filter,   // 2D image
                        cuComplex* res,            // Stack of 2D
                        CUtexObject snr,           // 1D texture
                        CUtexObject np,           // 1D texture
                        uint2 fftDim,
                        ctfConstants c_glb,
                        ctfImageConstants c_img,
                        const float* offsets,
                        int2 minmaxSlice,
                        float deconvStrength)
{
    preFilterSpread<true, true, true>(image,    // 2D image
                                       filter,   // 2D image
                                       res,      // Stack of 2D
                                       snr,      // 1D texture
                                       np,
                                       fftDim,
                                       c_glb,
                                       c_img,
                                       offsets,
                                       minmaxSlice,
                                       deconvStrength);
}

extern "C"
__global__
void preFilterSpreadSNRrelion(const cuComplex* image,    // 2D image
                              const cuComplex* filter,   // 2D image
                              cuComplex* res,            // Stack of 2D
                              CUtexObject snr,           // 1D texture
                              CUtexObject np,           // 1D texture
                              uint2 fftDim,
                              ctfConstants c_glb,
                              ctfImageConstants c_img,
                              const float* offsets,
                              int2 minmaxSlice,
                              float deconvStrength)
{
    preFilterSpread<true, true, true>(image,    // 2D image
                                      filter,   // 2D image
                                      res,      // Stack of 2D
                                      snr,      // 1D texture
                                      np,
                                      fftDim,
                                      c_glb,
                                      c_img,
                                      offsets,
                                      minmaxSlice,
                                      deconvStrength);
}


extern "C"
__global__
void postFilter(cuComplex* image,
                cuComplex* filter,
                cuComplex* res,
                uint2 fftDim,
                float normFactor)
{
    //compute x,y indices
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= fftDim.x) return;
    if (y >= fftDim.y) return;

    cuComplex signal = image[y * fftDim.x + x];
    cuComplex val_filter = filter[y * fftDim.x + x];
    signal = cmulf(signal, val_filter) / normFactor;

    res[y * fftDim.x + x] = signal;
}

extern "C"
__global__
void radialSumAbs(cuComplex* image,
                  float* sum,
                  float* multiplicity,
                  uint2 fftDim,
                  float2 imDim,
                  float2 asymCorrFac,
                  float scaleFactor)
{
    //compute x,y indices
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= fftDim.x) return;
    if (y >= fftDim.y) return;

    // FFT-shifted coords
    auto xx = (float)x;
    auto yy = (float)y;
    if (yy > (imDim.y) * 0.5f)
        yy = (imDim.y - yy) * -1.0f;

    xx = xx * asymCorrFac.x;
    yy = yy * asymCorrFac.y;

    // Shell
    float r = sqrt(xx*xx + yy*yy);
    int shell = (int) r;

    // Sum of absolute value
    float2 component = image[x + y * fftDim.x] * scaleFactor;
    float var = sqrt(component.x*component.x + component.y*component.y);

    atomicAdd(sum + shell, var);
    atomicAdd(multiplicity + shell, 1);
}

extern "C"
__global__
void radialSum3D(const float* image,
                 float* sum,
                 float* multiplicity,
                 uint3 fftDim,
                 uint3 volDim,
                 float3 asymCorrFac,
                 float scaleFactor,
                 int maxShell)
{
    //compute x,y indices
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= fftDim.x) return;
    if (y >= fftDim.y) return;
    if (z >= fftDim.z) return;

    // FFT shift
    int j = ((int)y + (int)volDim.y / 2) % (int)volDim.y;
    int k = ((int)z + (int)volDim.z / 2) % (int)volDim.z;

    // Center
    float xx = (float) x;
    float yy = (float) j - (float)volDim.y/2;
    float zz = (float) k - (float)volDim.z/2;

    xx = xx * asymCorrFac.x;
    yy = yy * asymCorrFac.y;
    zz = zz * asymCorrFac.z;

    // Shell
    float r = sqrt(xx*xx + yy*yy + zz*zz);
    int shell = (int) r;
    if (shell >= maxShell) return;

    // Value
    float component = image[x + y * fftDim.x + z * fftDim.x * fftDim.y] * scaleFactor;

    atomicAdd(sum + shell, component);
    atomicAdd(multiplicity + shell, 1);
}


extern "C"
__global__
void radialSumAbs3D(const cuComplex* image,
                    float* sum,
                    float* multiplicity,
                    uint3 fftDim,
                    uint3 volDim,
                    float3 asymCorrFac,
                    float scaleFactor,
                    int maxShell)
{
    //compute x,y indices
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= fftDim.x) return;
    if (y >= fftDim.y) return;
    if (z >= fftDim.z) return;

    // FFT shift
    int j = ((int)y + (int)volDim.y / 2) % (int)volDim.y;
    int k = ((int)z + (int)volDim.z / 2) % (int)volDim.z;

    // Center
    float xx = (float) x;
    float yy = (float) j - (float)volDim.y/2;
    float zz = (float) k - (float)volDim.z/2;

    xx = xx * asymCorrFac.x;
    yy = yy * asymCorrFac.y;
    zz = zz * asymCorrFac.z;

    // Shell
    float r = sqrt(xx*xx + yy*yy + zz*zz);
    int shell = (int) r;
    if (shell >= maxShell) return;

    // Sum of absolute value
    float2 component = image[x + y * fftDim.x + z * fftDim.x * fftDim.y] * scaleFactor;
    float var = sqrt(component.x*component.x + component.y*component.y);

    atomicAdd(sum + shell, var);
    atomicAdd(multiplicity + shell, 1);
}


template<bool computePhaseRes>
__device__
void fsc3Dbase(const cuComplex* image_1,
               const cuComplex* image_2,
               float* amp1s,
               float* amp2s,
               float* ampds,
               float* phares1,
               float* phares2,
               float* multiplicity,
               uint3 fftDim,
               uint3 volDim,
               float3 asymCorrFac,
               float scaleFactor,
               int maxShell)
{
    //compute x,y indices
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= fftDim.x) return;
    if (y >= fftDim.y) return;
    if (z >= fftDim.z) return;

    // FFT shift
    int j = ((int)y + (int)volDim.y / 2) % (int)volDim.y;
    int k = ((int)z + (int)volDim.z / 2) % (int)volDim.z;

    // Center
    float xx = (float) x;
    float yy = (float) j - (float)volDim.y/2;
    float zz = (float) k - (float)volDim.z/2;

    xx = xx * asymCorrFac.x;
    yy = yy * asymCorrFac.y;
    zz = zz * asymCorrFac.z;

    // Shell
    float r = sqrt(xx*xx + yy*yy + zz*zz);
    int shell = (int) r;
    if (shell >= maxShell) return;

    // FSC
    float2 val1 = image_1[x + y * fftDim.x + z * fftDim.x * fftDim.y];
    float2 val2 = image_2[x + y * fftDim.x + z * fftDim.x * fftDim.y];
    float2 diff = val1 - val2;

    float amp1 = cuCrealf(cuCmulf(val1, cuConjf(val1)));
    float amp2 = cuCrealf(cuCmulf(val2, cuConjf(val2)));
    float ampd = cuCrealf(cuCmulf(diff, cuConjf(diff)));

    // Sum all of them up
    atomicAdd(amp1s + shell, amp1);
    atomicAdd(amp2s + shell, amp2);
    atomicAdd(ampds + shell, ampd);
    atomicAdd(multiplicity + shell, 1);

    if (computePhaseRes) {
        float phares_v = sqrtf(amp1) + sqrtf(amp1);

        float arg = 2 * sqrt(amp1 * amp2);
        if (arg > 0) {
            arg = (amp1 + amp2 - ampd) / arg;

            if (arg > 1) arg = 1;
            if (arg < -1) arg = -1;
        }

        float delta = acos(arg); // Phase shift

        atomicAdd(phares1 + shell, phares_v);
        atomicAdd(phares2 + shell, phares_v*delta*delta);
    }
}

extern "C"
__global__
void fsc3D(const cuComplex* image_1,
           const cuComplex* image_2,
           float* amp1s,
           float* amp2s,
           float* ampds,
           float* multiplicity,
           uint3 fftDim,
           uint3 volDim,
           float3 asymCorrFac,
           float scaleFactor,
           int maxShell)
{
    fsc3Dbase<false>(image_1,
                     image_2,
                     amp1s,
                     amp2s,
                     ampds,
                     0,
                     0,
                     multiplicity,
                     fftDim,
                     volDim,
                     asymCorrFac,
                     scaleFactor,
                     maxShell);
}

extern "C"
__global__
void fscNorm(float* amp1s,
             float* amp2s,
             float* ampds,
             const float* mult,
             uint length,
             float threshold)
{
    // Line coords
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    // Drop threads outside volume
    if (x >= length) return;

    // Thresholded multiplicity
    float multval = mult[x];
    float multmask = (multval == 0) ? 0 : 1;
    float multiplicity = max(threshold, multval);

    // FSC
    float amp1 = amp1s[x];
    float amp2 = amp2s[x];
    float ampd = ampds[x];
    float ccc = (amp1+amp2-ampd)/sqrt(amp1*amp2)/2;
    float mean = sqrtf(abs((amp1+amp2-ampd) / (multval*2)));
    float rmsd = sqrt(ampd/(2*(amp1+amp2)-ampd));

    amp1s[x] = ccc * multmask;
    amp2s[x] = mean * multmask;
    ampds[x] = rmsd * multmask;
}

//__device__ __forceinline__ cuComplex cmulf(cuComplex a, cuComplex b)
//{
//    cuComplex res;
//    res.x = a.x * b.x - a.y * b.y;
//    res.y = a.x * b.y + a.y * b.x;
//    return res;
//}

//extern "C"
//__global__
//void postfilterSliced(cuComplex* signal,
//                      int signal_x,
//                      int signal_y,
//                      int sliceNumber)
//{
//    //compute x,y indiced
//    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
//    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
//    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;
//
////    if (x >= c_pixelcount/2 + 1) return;
////    if (y >= c_pixelcount) return;
////    if (z >= sliceNumber) return;
//    if (x >= signal_x) return;
//    if (y >= signal_y) return;
//    if (z >= sliceNumber) return;
//
//
//    float xpos = (float)x;
//    float ypos = (float)y;
//    if (ypos > c_pixelcount * 0.5f)
//        ypos = (c_pixelcount - ypos) * -1.0f;
//
//    float xfreq = xpos/signal_x;
//    float yfreq = ypos/signal_y;
//
//    // Dual -> BSpline
//    float2 zn = ji * 2 * M_PI * xfreq;
//    float2 zn2 = cmulf(zn, zn);
//    float2 zn3 = cmulf(zn2, zn);
//    float2 d2bx = 5040 / ()
//
//    float2 bsplx = 6.f / (cexpf() + 4 + cexpf(ji * 2 * M_PI * xfreq));
//    float2 bsply = 6.f / (cexpf(ji * 2 * M_PI * yfreq) + 4 + cexpf(ji * 2 * M_PI * yfreq));
//
//    ctf[z * ctf_y * ctf_x + y * ctf_x + x] = res;
//
//    float alpha;
//    if (xpos == 0)
//        alpha = (M_PI * 0.5f);
//    else
//        alpha = (atan2(ypos , xpos));
//
//    float beta = ((alpha - angle));
//
//    //printf("offset: %f\n", offsets[z]);
//
//    float def0 = defocusMin + offsets[z];
//    float def1 = defocusMax + offsets[z];
//
//    float defocus = def0 + (1 - cos(2*beta)) * (def1 - def0) * 0.5f;
//
//    float length = sqrtf(xpos * xpos + ypos * ypos);
//
//    length *= c_freqStepSize;
//
//    float o = expf(-14.238829f * (c_openingAngle * c_openingAngle * ((Cs_ * lambda_ * lambda_ * length * length * length - defocus * length) * (Cs_ * lambda_ * lambda_ * length * length * length - defocus * length))));
//    float p = expf(-((0.943359f * lambda_ * length * length * H) * (0.943359f * lambda_ * length * length * H)));
//    float q = (a1 * expf(-b1 * (length * length)) + a2 * expf(-b2 * (length * length))) / 2.431f;
//
//    float m = -PhaseShift + (M_PI / 2.0f) * (Cs_ * lambda_ * lambda_ * lambda_ * length * length * length * length - 2 * defocus * lambda_ * length * length);
//    float n = c_phaseContrast * sinf(m) + c_ampContrast * cosf(m);
//
//    cuComplex res = ctf[z * ctf_y * ctf_x + y * ctf_x + x];//*(((cuComplex*)((char*)ctf + stride * y)) + x);
//
//    if (applyForFP && sqrtf(xpos * xpos + ypos * ypos) > betaFac.x && !phaseFlipOnly)// && length < 317382812)
//    {
//        length = length / 100000000.0f;
//        float coeff1 = betaFac.y;
//        float coeff2 = betaFac.z;
//        float coeff3 = betaFac.w;
//        float expfun = expf((-coeff1 * length - coeff2 * length * length - coeff3 * length * length * length));
//        expfun = max(expfun, 0.01f);
//        float val = n * expfun;
//        if (abs(val) < 0.0001f && val >=0 ) val = 0.0001f;
//        if (abs(val) < 0.0001f && val < 0 ) val = -0.0001f;
//
//
//        res.x = res.x * -val;
//        res.y = res.y * -val;
//    }
//
//    if (!applyForFP && sqrtf(xpos * xpos + ypos * ypos) > betaFac.x && !phaseFlipOnly)// && length < 317382812)
//    {
//        length = length / 100000000.0f;
//        float coeff1 = betaFac.y;
//        float coeff2 = betaFac.z;
//        float coeff3 = betaFac.w;
//        float expfun = expf((-coeff1 * length - coeff2 * length * length - coeff3 * length * length * length));
//        expfun = max(expfun, WienerFilterNoiseLevel);
//        float val = n * expfun;
//
//        res.x = res.x * -val / (val * val + WienerFilterNoiseLevel);
//        res.y = res.y * -val / (val * val + WienerFilterNoiseLevel);
//        //res.x = val;
//        //res.y = val;
//    }
//
//    if (phaseFlipOnly)
//    {
//        if (n >= 0)
//        {
//            res.x = -res.x;
//            res.y = -res.y;
//        }
//    }
//
//    //*(((cuComplex*)((char*)ctf + stride * y)) + x) = res;
//    ctf[z * ctf_y * ctf_x + y * ctf_x + x] = res;
//}

#endif
