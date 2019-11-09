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


#ifndef CUDACROSSCORRELATOR_H
#define CUDACROSSCORRELATOR_H

#include <cufft.h>
#include <nppdefs.h> //Need to import nppdefs.h before nppi.h, because of wrong extern "C" statement inside nppi.h
#include <nppi.h>

// EmTools
#include "CudaHelpers/CudaContext.h"
#include "CudaHelpers/NPPImages.h"

// Self
#include "Kernels.h"

class CudaCrossCorrelator
{
private:
	int size;
	NPPImage_8uC1 mask;
	NPPImage_32fC1 maskF;

	NPPImage_32fC1 mc1;
	NPPImage_32fC1 mcn;
	NPPImage_32fC1 mca;
	NPPImage_32fC1 mca2;

	Cuda::CudaDeviceVariable* buffer;
	Cuda::CudaDeviceVariable rowSum;

	Cuda::CudaDeviceVariable fftRef;
	Cuda::CudaDeviceVariable fftIm;
	Cuda::CudaDeviceVariable fftIm2;
	Cuda::CudaDeviceVariable fftMask;
	
	FourierFilterKernel four_filter;
	ConjMulKernel conj;
	MaxShiftKernel maxShift;
	SumRowKernel sumRow;
	CreateMaskKernel createMask;

	cufftHandle plan;
	cufftHandle planInv;

	float* rowSum1H;
	float* rowSum2H;
	float* rowSumH;
	
	
public:
	CudaCrossCorrelator(int aSize, CUmodule module);
	~CudaCrossCorrelator();

	float2 GetShift(NPPImage_32fC1& reference, NPPImage_32fC1& img, int maxShift, float lp, float hp, float lps, float hps);



};

#endif //CUDACROSSCORRELATOR_H
