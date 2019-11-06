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


#ifndef KERNELS_H
#define KERNELS_H

// EmTools
#include "CudaHelpers/CudaKernel.h"
#include "CudaHelpers/CudaVariables.h"
#include "CudaHelpers/NPPImages.h"

class FourierFilterKernel : public Cuda::CudaKernel
{
public:
	FourierFilterKernel(CUmodule aModule, dim3 aGriDim, dim3 aBlockDim);
	FourierFilterKernel(CUmodule aModule);
	//float2* img, size_t stride, int pixelcount, float lp, float hp, float lps, float hps)
	float operator()(Cuda::CudaDeviceVariable & img, size_t stride, int pixelcount, float lp, float hp, float lps, float hps);
	float operator()(Cuda::CudaPitchedDeviceVariable & img, int pixelcount, float lp, float hp, float lps, float hps);
	float operator()(NPPImage_32fcC1 & img, float lp, float hp, float lps, float hps);
};


class ConjMulKernel : public Cuda::CudaKernel
{
public:
	ConjMulKernel(CUmodule aModule, dim3 aGriDim, dim3 aBlockDim);
	ConjMulKernel(CUmodule aModule);
	//conjMul(float2* complxA, float2* complxB, size_t stride, int pixelcount)
	float operator()(Cuda::CudaDeviceVariable & complxA, Cuda::CudaDeviceVariable & complxB, size_t stride, int pixelcount);
	float operator()(Cuda::CudaPitchedDeviceVariable & complxA, Cuda::CudaPitchedDeviceVariable & complxB, int pixelcount);
	float operator()(NPPImage_32fcC1 & complxA, NPPImage_32fcC1 & complxB);
};


class MaxShiftKernel : public Cuda::CudaKernel
{
public:
	MaxShiftKernel(CUmodule aModule, dim3 aGriDim, dim3 aBlockDim);
	MaxShiftKernel(CUmodule aModule);

	float operator()(Cuda::CudaDeviceVariable & img, size_t stride, int pixelcount, int maxShift);
	float operator()(Cuda::CudaPitchedDeviceVariable & img, int pixelcount, int maxShift);
	float operator()(NPPImage_32fC1 & img, int maxShift);
};


class SumRowKernel : public Cuda::CudaKernel
{
public:
	SumRowKernel(CUmodule aModule, dim3 aGriDim, dim3 aBlockDim);
	SumRowKernel(CUmodule aModule);
	//SumRow(float* img, size_t stride, int pixelcount, float* sum)
	float operator()(Cuda::CudaDeviceVariable & img, size_t stride, int width, int height, Cuda::CudaDeviceVariable & sum);
	float operator()(Cuda::CudaPitchedDeviceVariable & img, int width, int height, Cuda::CudaDeviceVariable & sum);
	float operator()(NPPImage_32fC1 & img, Cuda::CudaDeviceVariable & sum);
};


class CreateMaskKernel : public Cuda::CudaKernel
{
public:
	CreateMaskKernel(CUmodule aModule, dim3 aGriDim, dim3 aBlockDim);
	CreateMaskKernel(CUmodule aModule);
	//CreateMaskKernel(unsigned char* mask, size_t stride, int pixelcount, float* sum)
	float operator()(Cuda::CudaDeviceVariable & mask, size_t stride, int width, int height, Cuda::CudaDeviceVariable & sum);
	float operator()(Cuda::CudaPitchedDeviceVariable & mask, int width, int height, Cuda::CudaDeviceVariable & sum);
	float operator()(NPPImage_8uC1 & mask, Cuda::CudaDeviceVariable & sum);
};

#endif