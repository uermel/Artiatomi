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


#ifndef CORRELATIONKERNEL_H
#define CORRELATIONKERNEL_H

#include "../Basics/Default.h"

#include <CudaContext.h>
#include <CudaKernel.h>
#include <cutil_math_.h>


class FftshiftRealKernel : public Cuda::CudaKernel
{
public:
	FftshiftRealKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
	FftshiftRealKernel(CUmodule aModule);

	float operator()(int size, Cuda::CudaDeviceVariable& inVol, Cuda::CudaDeviceVariable& outVol);
};

class EnergynormKernel : public Cuda::CudaKernel
{
public:
	EnergynormKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
	EnergynormKernel(CUmodule aModule);

	float operator()(int size, Cuda::CudaDeviceVariable& particle, Cuda::CudaDeviceVariable& partSqr, Cuda::CudaDeviceVariable& cccMap, Cuda::CudaDeviceVariable& energyRef, Cuda::CudaDeviceVariable& nVox);
};

class BinarizeKernel : public Cuda::CudaKernel
{
public:
	BinarizeKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
	BinarizeKernel(CUmodule aModule);

	float operator()(int length, Cuda::CudaDeviceVariable& inVol, Cuda::CudaDeviceVariable& outVol);
};

class ConvKernel : public Cuda::CudaKernel
{
public:
	ConvKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
	ConvKernel(CUmodule aModule);

	float operator()(int length, Cuda::CudaDeviceVariable& inVol, Cuda::CudaDeviceVariable& outVol);
};

class CorrelKernel : public Cuda::CudaKernel
{
public:
	CorrelKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
	CorrelKernel(CUmodule aModule);

	float operator()(int length, Cuda::CudaDeviceVariable& inVol, Cuda::CudaDeviceVariable& outVol);
};

class PhaseCorrelKernel : public Cuda::CudaKernel
{
public:
	PhaseCorrelKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
	PhaseCorrelKernel(CUmodule aModule);

	float operator()(int length, Cuda::CudaDeviceVariable& inVol, Cuda::CudaDeviceVariable& outVol);
};

class BandpassFFTShiftKernel : public Cuda::CudaKernel
{
public:
	BandpassFFTShiftKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
	BandpassFFTShiftKernel(CUmodule aModule);

	float operator()(int size, Cuda::CudaDeviceVariable& vol, float rDown, float rUp, float smooth);
};

class MulRealCplxFFTShiftKernel : public Cuda::CudaKernel
{
public:
	MulRealCplxFFTShiftKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
	MulRealCplxFFTShiftKernel(CUmodule aModule);

	float operator()(int size, Cuda::CudaDeviceVariable& realVol, Cuda::CudaDeviceVariable& cplxVol);
};

class WedgeNormKernel : public Cuda::CudaKernel
{
public:
	WedgeNormKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
	WedgeNormKernel(CUmodule aModule);

	float operator()(int size, Cuda::CudaDeviceVariable& wedge, Cuda::CudaDeviceVariable& part, float maxVal);
};

#endif