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


#include "Correlator3D.h"
#include "Kernels/CorrelationKernel.cu.h"
#include "Kernels/SubDeviceKernel.cu.h"
#include <CudaException.h>
#include <vector>
#include <EmFile.h>

using namespace Cuda;


void Correlator3D::streamCallback(CUstream stream, CUresult error, void* data)
{
	userData* ud = (userData*)data;
	*(ud->dstPointer) = *(ud->srcPointer);
	delete ud;
}

Correlator3D::Correlator3D(CudaContext* aCtx, int aSize, CudaDeviceVariable* aFilter, float aRDown, float aRUp, float aSmooth, bool aUseFilterVolume, CUstream aStream) :
	ctx(aCtx),
	size(aSize),
	totalSize((size_t)aSize * (size_t)aSize * (size_t)aSize),
	particle((size_t)aSize * (size_t)aSize * (size_t)aSize * sizeof(float)),
	particleFFT(((size_t)aSize / 2 + 1)* (size_t)aSize* (size_t)aSize * sizeof(float) * 2),
	particleSqrFFT(((size_t)aSize / 2 + 1)* (size_t)aSize* (size_t)aSize * sizeof(float) * 2),
	maskFFT(((size_t)aSize / 2 + 1)* (size_t)aSize* (size_t)aSize * sizeof(float) * 2),
	ref((size_t)aSize * (size_t)aSize * (size_t)aSize * sizeof(float)),
	refFFT(((size_t)aSize / 2 + 1) * (size_t)aSize * (size_t)aSize * sizeof(float) * 2),
	scratchMemory(),
	filter(*aFilter, false),
	sumValue(sizeof(float)),
	sumSqrValue(sizeof(float)),
	sumSqrValuePart(sizeof(float)),
	sumSqrValueRef(sizeof(float)),
	nVox(sizeof(float)),
	sum_h(0),
	sumSqr_h(0),
	nVox_h(0),
	rDown(aRDown),
	rUp(aRUp),
	smooth(aSmooth),
	useFilterVolume(aUseFilterVolume),
	stream(aStream),
	pl_CCFastResult(sizeof(float), 0)
{
	int sizeNpps;
	int sizeNpps2;
	nppSafeCall(nppsSumGetBufferSize_32f(aSize * aSize * aSize, &sizeNpps));
	nppSafeCall(nppsMaxGetBufferSize_32f(aSize * aSize * aSize, &sizeNpps2));
	sizeNpps = std::max(sizeNpps, sizeNpps2);

	cufftSafeCall(cufftCreate(&planR2C));
	cufftSafeCall(cufftCreate(&planC2R));
	cufftSafeCall(cufftSetAutoAllocation(planR2C, 0));
	cufftSafeCall(cufftSetAutoAllocation(planC2R, 0));
	size_t workSizeR2C;
	size_t workSizeC2R;
	cufftSafeCall(cufftMakePlan3d(planR2C, aSize, aSize, aSize, CUFFT_R2C, &workSizeR2C));
	cufftSafeCall(cufftMakePlan3d(planC2R, aSize, aSize, aSize, CUFFT_C2R, &workSizeC2R));

	size_t maxSize = std::max(std::max(workSizeC2R, workSizeR2C), (size_t)sizeNpps);
	scratchMemory.Alloc(maxSize);

	cufftSetWorkArea(planR2C, (void*)scratchMemory.GetDevicePtr());
	cufftSetWorkArea(planC2R, (void*)scratchMemory.GetDevicePtr());

	CUmodule cuMod = aCtx->LoadModulePTX(SubTomogramCorrelationKernel, 0, false, false);
	CUmodule cuMod2 = aCtx->LoadModulePTX(SubTomogramSubDeviceKernel, 0, false, false);

	kernelFftshiftReal = new FftshiftRealKernel(cuMod);
	kernelEnergynorm = new EnergynormKernel(cuMod);
	kernelConv = new ConvKernel(cuMod);
	kernelCorrel = new CorrelKernel(cuMod);
	kernelPhaseCorrel = new PhaseCorrelKernel(cuMod);
	kernelBandpassFFTShift = new BandpassFFTShiftKernel(cuMod);
	kernelMulRealCplxFFTShift = new MulRealCplxFFTShiftKernel(cuMod);
	kernelBinarize = new BinarizeKernel(cuMod);
	kernelWedgeNorm = new WedgeNormKernel(cuMod);
	kernelSubDiv = new SubDivDeviceKernel(cuMod2);
	kernelNormalize = new NormalizeKernel(cuMod);

	cufftSafeCall(cufftSetStream(planR2C, stream));
	cufftSafeCall(cufftSetStream(planC2R, stream));

	nppSafeCall(nppSetStream(stream));
	nppSafeCall(nppGetStreamContext(&streamCtx));

	cudaSafeCall(cuEventCreate(&ev, CU_EVENT_DISABLE_TIMING));

}

Correlator3D::~Correlator3D()
{
	cudaSafeCall(cuEventDestroy(ev));

	//destroy FFT plans
	cufftSafeCall(cufftDestroy(planC2R));
	cufftSafeCall(cufftDestroy(planR2C));

	//free kernels
	delete kernelFftshiftReal;
	delete kernelEnergynorm;
	delete kernelConv;
	delete kernelCorrel;
	delete kernelPhaseCorrel;
	delete kernelBandpassFFTShift;
	delete kernelMulRealCplxFFTShift;
	delete kernelBinarize;
	delete kernelWedgeNorm;
}

void Correlator3D::FourierFilter(Cuda::CudaDeviceVariable& aParticle)
{
	cufftSafeCall(cufftExecR2C(planR2C, (float*)aParticle.GetDevicePtr(), (cufftComplex*)particleFFT.GetDevicePtr()));

	if (useFilterVolume)
	{
		(*kernelMulRealCplxFFTShift)(stream, size, filter, particleFFT);
	}
	else
	{
		(*kernelBandpassFFTShift)(stream, size, particleFFT, rDown, rUp, smooth);
	}

	cufftSafeCall(cufftExecC2R(planC2R, (cufftComplex*)particleFFT.GetDevicePtr(), (float*)aParticle.GetDevicePtr()));
	nppSafeCall(nppsDivC_32f_I_Ctx((float)totalSize, (float*)aParticle.GetDevicePtr(), (int)totalSize, streamCtx));
}

void Correlator3D::PrepareParticle(CudaDeviceVariable& aParticle, CudaDeviceVariable& wedge)
{
	cufftSafeCall(cufftExecR2C(planR2C, (float*)aParticle.GetDevicePtr(), (cufftComplex*)particleFFT.GetDevicePtr()));
	(*kernelMulRealCplxFFTShift)(stream, size, wedge, particleFFT);

	if (useFilterVolume)
	{
		(*kernelMulRealCplxFFTShift)(stream, size, filter, particleFFT);
	}
	else
	{
		(*kernelBandpassFFTShift)(stream, size, particleFFT, rDown, rUp, smooth);
	}

	cufftSafeCall(cufftExecC2R(planC2R, (cufftComplex*)particleFFT.GetDevicePtr(), (float*)particle.GetDevicePtr()));
	nppSafeCall(nppsDivC_32f_I_Ctx((float)totalSize, (float*)particle.GetDevicePtr(), (int)totalSize, streamCtx));

	nppSafeCall(nppsSum_32f_Ctx((float*)particle.GetDevicePtr(), (int)totalSize, (float*)sumValue.GetDevicePtr(), (Npp8u*)scratchMemory.GetDevicePtr(), streamCtx));

	(*kernelSubDiv)(stream, sumValue, particle, totalSize, totalSize);

	//sumValue.CopyDeviceToHost(&sum_h, sizeof(float));
	
	//nppSafeCall(nppsSubC_32f_I(sum_h / totalSize, (float*)particle.GetDevicePtr(), (int)totalSize));

	cufftSafeCall(cufftExecR2C(planR2C, (float*)particle.GetDevicePtr(), (cufftComplex*)particleFFT.GetDevicePtr()));
	
	nppSafeCall(nppsSqr_32f_I_Ctx((float*)particle.GetDevicePtr(), (int)totalSize, streamCtx));

	cufftSafeCall(cufftExecR2C(planR2C, (float*)particle.GetDevicePtr(), (cufftComplex*)particleSqrFFT.GetDevicePtr()));

	(*kernelMulRealCplxFFTShift)(stream, size, wedge, particleFFT);

	if (useFilterVolume)
	{
		(*kernelMulRealCplxFFTShift)(stream, size, filter, particleFFT);
	}
	else
	{
		(*kernelBandpassFFTShift)(stream, size, particleFFT, rDown, rUp, smooth);
	}
}

void Correlator3D::PrepareMask(Cuda::CudaDeviceVariable& mask, bool binarize)
{
	if (binarize)
	{
		(*kernelBinarize)(stream, size*size*size, mask, mask);
	}
	nppSafeCall(nppsSum_32f_Ctx((float*)mask.GetDevicePtr(), (int)totalSize, (float*)nVox.GetDevicePtr(), (Npp8u*)scratchMemory.GetDevicePtr(), streamCtx));
	
	nVox.CopyDeviceToHostAsync(stream, pl_CCFastResult.GetHostPtr(), sizeof(float));
	cudaSafeCall(cuStreamSynchronize(stream));
	nVox_h = *((float*)pl_CCFastResult.GetHostPtr());

	cufftSafeCall(cufftExecR2C(planR2C, (float*)mask.GetDevicePtr(), (cufftComplex*)maskFFT.GetDevicePtr()));
}

void Correlator3D::GetCC(CudaDeviceVariable& mask, CudaDeviceVariable& aRef, CudaDeviceVariable& wedge, CudaDeviceVariable& ccVolOut)
{
	cufftSafeCall(cufftExecR2C(planR2C, (float*)aRef.GetDevicePtr(), (cufftComplex*)refFFT.GetDevicePtr()));
	(*kernelMulRealCplxFFTShift)(stream, size, wedge, refFFT);

	if (useFilterVolume)
	{
		(*kernelMulRealCplxFFTShift)(stream, size, filter, refFFT);
	}
	else
	{
		(*kernelBandpassFFTShift)(stream, size, refFFT, rDown, rUp, smooth);
	}

	cufftSafeCall(cufftExecC2R(planC2R, (cufftComplex*)refFFT.GetDevicePtr(), (float*)ref.GetDevicePtr()));
	nppSafeCall(nppsDivC_32f_I_Ctx((float)totalSize, (float*)ref.GetDevicePtr(), (int)totalSize, streamCtx));

	nppSafeCall(nppsMul_32f_I_Ctx((float*)mask.GetDevicePtr(), (float*)ref.GetDevicePtr(), (int)totalSize, streamCtx));

	nppSafeCall(nppsSum_32f_Ctx((float*)ref.GetDevicePtr(), (int)totalSize, (float*)sumValue.GetDevicePtr(), (Npp8u*)scratchMemory.GetDevicePtr(), streamCtx));

	(*kernelSubDiv)(stream, sumValue, ref, totalSize, nVox_h);
	//sumValue.CopyDeviceToHost(&sum_h, sizeof(float));

	//nppSafeCall(nppsSubC_32f_I(sum_h / nVox_h, (float*)ref.GetDevicePtr(), (int)totalSize));

	nppSafeCall(nppsMul_32f_I_Ctx((float*)mask.GetDevicePtr(), (float*)ref.GetDevicePtr(), (int)totalSize, streamCtx));

	nppSafeCall(nppsSqr_32f_Ctx((float*)ref.GetDevicePtr(), (float*)ccVolOut.GetDevicePtr(), (int)totalSize, streamCtx));
	
	nppSafeCall(nppsSum_32f_Ctx((float*)ccVolOut.GetDevicePtr(), (int)totalSize, (float*)sumSqrValue.GetDevicePtr(), (Npp8u*)scratchMemory.GetDevicePtr(), streamCtx));

//	float sumSqr;
//	sumSqrValue.CopyDeviceToHost(&sumSqr);

	cufftSafeCall(cufftExecR2C(planR2C, (float*)ref.GetDevicePtr(), (cufftComplex*)refFFT.GetDevicePtr()));

	(*kernelCorrel)(stream, (size / 2 + 1) * size * size, particleFFT, refFFT);

	(*kernelConv)(stream, (size / 2 + 1) * size * size, maskFFT, particleFFT);

	(*kernelConv)(stream, (size / 2 + 1) * size * size, maskFFT, particleSqrFFT);
	
	cufftSafeCall(cufftExecC2R(planC2R, (cufftComplex*)refFFT.GetDevicePtr(), (float*)ref.GetDevicePtr()));
	nppSafeCall(nppsDivC_32f_I_Ctx((float)totalSize, (float*)ref.GetDevicePtr(), (int)totalSize, streamCtx));

	cufftSafeCall(cufftExecC2R(planC2R, (cufftComplex*)particleFFT.GetDevicePtr(), (float*)particle.GetDevicePtr()));
	nppSafeCall(nppsDivC_32f_I_Ctx((float)totalSize, (float*)particle.GetDevicePtr(), (int)totalSize, streamCtx));

	cufftSafeCall(cufftExecC2R(planC2R, (cufftComplex*)particleSqrFFT.GetDevicePtr(), (float*)ccVolOut.GetDevicePtr()));
	nppSafeCall(nppsDivC_32f_I_Ctx((float)totalSize, (float*)ccVolOut.GetDevicePtr(), (int)totalSize, streamCtx));

	(*kernelEnergynorm)(stream, size, particle, ccVolOut, ref, sumSqrValue, nVox);

	(*kernelFftshiftReal)(stream, size, ref, ccVolOut);
}

void Correlator3D::GettCCFast(CudaDeviceVariable& aMask, CudaDeviceVariable& aParticle, CudaDeviceVariable& aRef, CudaDeviceVariable& aWedge, float* result, bool normalizeAmplitudes)
{

	/*x = x - mean(mean(mean(x)));
	y = y - mean(mean(mean(y)));


	nx = sum(sum(sum(x. ^ 2)));
	ny = sum(sum(sum(y. ^ 2)));
	u = sum(sum(sum(x.*y)));
	u = u / sqrt(nx * ny);*/


	cufftSafeCall(cufftExecR2C(planR2C, (float*)aParticle.GetDevicePtr(), (cufftComplex*)particleFFT.GetDevicePtr()));
	cufftSafeCall(cufftExecR2C(planR2C, (float*)aRef.GetDevicePtr(), (cufftComplex*)refFFT.GetDevicePtr()));

	if (normalizeAmplitudes)
	{
		(*kernelNormalize)(stream, size * size * (size / 2 + 1), particleFFT);
		(*kernelNormalize)(stream, size * size * (size / 2 + 1), refFFT);
	}

	(*kernelMulRealCplxFFTShift)(stream, size, aWedge, particleFFT);
	(*kernelMulRealCplxFFTShift)(stream, size, aWedge, refFFT);

	if (useFilterVolume)
	{
		(*kernelMulRealCplxFFTShift)(stream, size, filter, particleFFT);
		(*kernelMulRealCplxFFTShift)(stream, size, filter, refFFT);
	}
	else
	{
		(*kernelBandpassFFTShift)(stream, size, particleFFT, rDown, rUp, smooth);
		(*kernelBandpassFFTShift)(stream, size, refFFT, rDown, rUp, smooth);
	}

	cufftSafeCall(cufftExecC2R(planC2R, (cufftComplex*)particleFFT.GetDevicePtr(), (float*)particle.GetDevicePtr()));
	nppSafeCall(nppsDivC_32f_I_Ctx((float)totalSize, (float*)particle.GetDevicePtr(), (int)totalSize, streamCtx));

	cufftSafeCall(cufftExecC2R(planC2R, (cufftComplex*)refFFT.GetDevicePtr(), (float*)ref.GetDevicePtr()));
	nppSafeCall(nppsDivC_32f_I_Ctx((float)totalSize, (float*)ref.GetDevicePtr(), (int)totalSize, streamCtx));


	nppSafeCall(nppsMul_32f_I_Ctx((float*)aMask.GetDevicePtr(), (float*)particle.GetDevicePtr(), (int)totalSize, streamCtx));
	nppSafeCall(nppsMul_32f_I_Ctx((float*)aMask.GetDevicePtr(), (float*)ref.GetDevicePtr(), (int)totalSize, streamCtx));

	nppSafeCall(nppsSum_32f_Ctx((float*)particle.GetDevicePtr(), (int)totalSize, (float*)sumValue.GetDevicePtr(), (Npp8u*)scratchMemory.GetDevicePtr(), streamCtx));
	(*kernelSubDiv)(stream, sumValue, particle, totalSize, nVox_h);

	//float sumPart= 1;
	//sumValue.CopyDeviceToHost(&sumPart, sizeof(float));
	nppSafeCall(nppsSum_32f_Ctx((float*)ref.GetDevicePtr(), (int)totalSize, (float*)sumValue.GetDevicePtr(), (Npp8u*)scratchMemory.GetDevicePtr(), streamCtx));
	(*kernelSubDiv)(stream, sumValue, ref, totalSize, nVox_h);
	//float sumRef;
	//sumValue.CopyDeviceToHost(&sumRef, sizeof(float));

	/*float* t1 = new float[64 * 64 * 64];
	float* t2 = new float[64 * 64 * 64];
	aWedge.CopyDeviceToHost(t1);
	emwrite("j:\\TestData\\Ribosom\\t11.em", t1, 64, 64, 64);*/
	//emwrite("j:\\TestData\\Ribosom\\t21.em", particleBlock2[p2 - pBlock2], 64, 64, 64);



	//nppSafeCall(nppsSubC_32f_I(sumPart / nVox_h, (float*)particle.GetDevicePtr(), (int)totalSize));
	//nppSafeCall(nppsSubC_32f_I(sumRef / nVox_h, (float*)ref.GetDevicePtr(), (int)totalSize));

	nppSafeCall(nppsMul_32f_I_Ctx((float*)aMask.GetDevicePtr(), (float*)particle.GetDevicePtr(), (int)totalSize, streamCtx));
	nppSafeCall(nppsMul_32f_I_Ctx((float*)aMask.GetDevicePtr(), (float*)ref.GetDevicePtr(), (int)totalSize, streamCtx));

	nppSafeCall(nppsSqr_32f_Ctx((float*)particle.GetDevicePtr(), (float*)aWedge.GetDevicePtr(), (int)totalSize, streamCtx));
	nppSafeCall(nppsSqr_32f_Ctx((float*)ref.GetDevicePtr(), (float*)aRef.GetDevicePtr(), (int)totalSize, streamCtx));
	

	nppSafeCall(nppsSum_32f_Ctx((float*)aWedge.GetDevicePtr(), (int)totalSize, (float*)sumSqrValuePart.GetDevicePtr(), (Npp8u*)scratchMemory.GetDevicePtr(), streamCtx));
	//float sumSqrPart;
	//sumSqrValue.CopyDeviceToHost(&sumSqrPart);
	nppSafeCall(nppsSum_32f_Ctx((float*)aRef.GetDevicePtr(), (int)totalSize, (float*)sumSqrValueRef.GetDevicePtr(), (Npp8u*)scratchMemory.GetDevicePtr(), streamCtx));
	//float sumSqrRef;
	//sumSqrValue.CopyDeviceToHost(&sumSqrRef);

	nppSafeCall(nppsMul_32f_I_Ctx((float*)ref.GetDevicePtr(), (float*)particle.GetDevicePtr(), (int)totalSize, streamCtx));

	nppSafeCall(nppsSum_32f_Ctx((float*)particle.GetDevicePtr(), (int)totalSize, (float*)sumSqrValue.GetDevicePtr(), (Npp8u*)scratchMemory.GetDevicePtr(), streamCtx));

	//float sumXY;
	//sumSqrValue.CopyDeviceToHost(&sumXY);

	//sumXY / sqrtf(sumSqrPart * sumSqrRef);
	nppSafeCall(nppsMul_32f_I_Ctx((float*)sumSqrValueRef.GetDevicePtr(), (float*)sumSqrValuePart.GetDevicePtr(), 1, streamCtx));
	nppSafeCall(nppsSqrt_32f_I_Ctx((float*)sumSqrValuePart.GetDevicePtr(), 1, streamCtx));

	nppSafeCall(nppsDiv_32f_I_Ctx((float*)sumSqrValuePart.GetDevicePtr(), (float*)sumSqrValue.GetDevicePtr(), 1, streamCtx));

	sumSqrValue.CopyDeviceToHostAsync(stream, pl_CCFastResult.GetHostPtr(), sizeof(float));

	//cudaSafeCall(cuEventRecord(ev, stream));
	//cudaSafeCall(cuEventSynchronize(ev));

	cudaSafeCall(cuStreamSynchronize(stream));
	*result = *((float*)pl_CCFastResult.GetHostPtr());
	//cudaSafeCall(cuStreamSynchronize(stream));
	/*userData* ud = new userData();
	ud->dstPointer = result;
	ud->srcPointer = (float*)pl_CCFastResult.GetHostPtr();

	cudaSafeCall(cuStreamAddCallback(stream, streamCallback, ud, 0));
	*/ 
}

void Correlator3D::PhaseCorrelate(Cuda::CudaDeviceVariable& aParticle, Cuda::CudaDeviceVariable& mask, Cuda::CudaDeviceVariable& aRef, Cuda::CudaDeviceVariable& wedge, Cuda::CudaDeviceVariable& ccVolOut)
{
	cufftSafeCall(cufftExecR2C(planR2C, (float*)aParticle.GetDevicePtr(), (cufftComplex*)particleFFT.GetDevicePtr()));
	cufftSafeCall(cufftExecR2C(planR2C, (float*)aRef.GetDevicePtr(), (cufftComplex*)refFFT.GetDevicePtr()));

	(*kernelPhaseCorrel)(stream, (size / 2 + 1) * size * size, particleFFT, refFFT);
	(*kernelPhaseCorrel)(stream, (size / 2 + 1) * size * size, particleFFT, particleFFT);

	(*kernelMulRealCplxFFTShift)(stream, size, wedge, refFFT);
	(*kernelMulRealCplxFFTShift)(stream, size, wedge, particleFFT);

	if (useFilterVolume)
	{
		(*kernelMulRealCplxFFTShift)(stream, size, filter, refFFT);
		(*kernelMulRealCplxFFTShift)(stream, size, filter, particleFFT);
	}
	else
	{
		(*kernelBandpassFFTShift)(stream, size, refFFT, rDown, rUp, smooth);
		(*kernelBandpassFFTShift)(stream, size, particleFFT, rDown, rUp, smooth);
	}

	cufftSafeCall(cufftExecC2R(planC2R, (cufftComplex*)refFFT.GetDevicePtr(), (float*)ref.GetDevicePtr()));
	nppSafeCall(nppsDivC_32f_I_Ctx((float)totalSize, (float*)ref.GetDevicePtr(), (int)totalSize, streamCtx));

	cufftSafeCall(cufftExecC2R(planC2R, (cufftComplex*)particleFFT.GetDevicePtr(), (float*)particle.GetDevicePtr()));
	nppSafeCall(nppsDivC_32f_I_Ctx((float)totalSize, (float*)particle.GetDevicePtr(), (int)totalSize, streamCtx));

	nppSafeCall(nppsDiv_32f_I_Ctx((float*)particle.GetDevicePtr(), (float*)ref.GetDevicePtr(), (int)totalSize, streamCtx));

	(*kernelFftshiftReal)(stream, size, ref, ccVolOut);
}



void Correlator3D::MultiplyWedge(Cuda::CudaDeviceVariable& aParticle, Cuda::CudaDeviceVariable& wedge)
{
	cufftSafeCall(cufftExecR2C(planR2C, (float*)aParticle.GetDevicePtr(), (cufftComplex*)particleFFT.GetDevicePtr()));
	(*kernelMulRealCplxFFTShift)(stream, size, wedge, particleFFT);
	cufftSafeCall(cufftExecC2R(planC2R, (cufftComplex*)particleFFT.GetDevicePtr(), (float*)aParticle.GetDevicePtr()));
	nppSafeCall(nppsDivC_32f_I_Ctx((float)totalSize, (float*)aParticle.GetDevicePtr(), (int)totalSize, streamCtx));

}

void Correlator3D::NormalizeWedge(Cuda::CudaDeviceVariable& particleSum, Cuda::CudaDeviceVariable& wedgeSum)
{
	nppSafeCall(nppsMax_32f_Ctx((float*)wedgeSum.GetDevicePtr(), (int)totalSize, (float*)sumValue.GetDevicePtr(), (Npp8u*)scratchMemory.GetDevicePtr(), streamCtx));
	//float maxVal = 0;
	//sumValue.CopyDeviceToHost(&maxVal);

	cufftSafeCall(cufftExecR2C(planR2C, (float*)particleSum.GetDevicePtr(), (cufftComplex*)particleFFT.GetDevicePtr()));

	(*kernelWedgeNorm)(stream, size, wedgeSum, particleFFT, sumValue);
	
	cufftSafeCall(cufftExecC2R(planC2R, (cufftComplex*)particleFFT.GetDevicePtr(), (float*)particleSum.GetDevicePtr()));
	nppSafeCall(nppsDivC_32f_I_Ctx((float)totalSize, (float*)particleSum.GetDevicePtr(), (int)totalSize, streamCtx));

}

