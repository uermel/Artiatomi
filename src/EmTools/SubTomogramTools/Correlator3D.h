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


#ifndef CORRELATOR3D_H
#define CORRELATOR3D_H

#include "../Basics/Default.h"
#include "CorrelationKernel.h"

#include <CudaContext.h>
#include <CudaKernel.h>
#include <CudaVariables.h>

#include <cufft.h>
#include <npps.h>

class Correlator3D
{
private:
	Cuda::CudaContext* ctx;
	int size;
	size_t totalSize;

	FftshiftRealKernel* kernelFftshiftReal;
	EnergynormKernel* kernelEnergynorm;
	ConvKernel* kernelConv;
	CorrelKernel* kernelCorrel;
	PhaseCorrelKernel* kernelPhaseCorrel;
	BandpassFFTShiftKernel* kernelBandpassFFTShift;
	MulRealCplxFFTShiftKernel* kernelMulRealCplxFFTShift;
	BinarizeKernel* kernelBinarize;
	WedgeNormKernel* kernelWedgeNorm;

	Cuda::CudaDeviceVariable* scratchMemory;
	Cuda::CudaDeviceVariable sumValue;
	Cuda::CudaDeviceVariable sumSqrValue;
	Cuda::CudaDeviceVariable nVox;
	Cuda::CudaDeviceVariable& filter;

	Cuda::CudaDeviceVariable particle;
	Cuda::CudaDeviceVariable particleFFT;
	Cuda::CudaDeviceVariable particleSqrFFT;
	Cuda::CudaDeviceVariable maskFFT;
	Cuda::CudaDeviceVariable ref;
	Cuda::CudaDeviceVariable refFFT;

	float rDown;
	float rUp;
	float smooth;
	bool useFilterVolume;
	float sum_h;
	float sumSqr_h;
	float nVox_h;
	cufftHandle planR2C;
	cufftHandle planC2R;

public:
	Correlator3D(Cuda::CudaContext* aCtx, int aSize, Cuda::CudaDeviceVariable& aFilter, float aRDown, float aRUp, float aSmooth, bool aUseFilterVolume);
	
	~Correlator3D();

	void FourierFilter(Cuda::CudaDeviceVariable& particle);

	void PrepareParticle(Cuda::CudaDeviceVariable& particle, Cuda::CudaDeviceVariable& wedge);

	void PrepareMask(Cuda::CudaDeviceVariable& mask, bool binarize);

	void GetCC(Cuda::CudaDeviceVariable& mask, Cuda::CudaDeviceVariable& aRef, Cuda::CudaDeviceVariable& wedge, Cuda::CudaDeviceVariable& ccVolOut);

	float GettCCFast(Cuda::CudaDeviceVariable& mask, Cuda::CudaDeviceVariable& particle, Cuda::CudaDeviceVariable& ref, Cuda::CudaDeviceVariable& wedge);
	void PhaseCorrelate(Cuda::CudaDeviceVariable& particle, Cuda::CudaDeviceVariable& mask, Cuda::CudaDeviceVariable& aRef, Cuda::CudaDeviceVariable& wedge, Cuda::CudaDeviceVariable& ccVolOut);

	void MultiplyWedge(Cuda::CudaDeviceVariable& particle, Cuda::CudaDeviceVariable& wedge);

	void NormalizeWedge(Cuda::CudaDeviceVariable& particleSum, Cuda::CudaDeviceVariable& wedgeSum);
};

#endif