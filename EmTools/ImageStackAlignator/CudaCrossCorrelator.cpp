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


#include "CudaCrossCorrelator.h"
#include <algorithm>

#include "../FileIO/EmFile.h"
using namespace Cuda;

CudaCrossCorrelator::CudaCrossCorrelator(int aSize, CUmodule module) :
	size(aSize),
	rowSum(sizeof(float) * aSize),
	mask(aSize, aSize),
	maskF(aSize, aSize),
	mc1(aSize, aSize),
	mcn(aSize, aSize),
	mca(aSize, aSize),
	mca2(aSize, aSize),
	fftRef((aSize / 2 + 1) * aSize * sizeof(cufftComplex)),
	fftIm((aSize / 2 + 1) * aSize * sizeof(cufftComplex)),
	fftIm2((aSize / 2 + 1) * aSize * sizeof(cufftComplex)),
	fftMask((aSize / 2 + 1) * aSize * sizeof(cufftComplex)),
	four_filter(module),
	conj(module),
	maxShift(module),
	sumRow(module),
	createMask(module)
{
	four_filter.SetComputeSize(aSize, aSize);
	conj.SetComputeSize(aSize, aSize);
	maxShift.SetComputeSize(aSize, aSize);
	sumRow.SetComputeSize(aSize, 1);
	createMask.SetComputeSize(aSize, aSize);

	int sizeBuffer = 0;
	NppiSize roi;
	roi.width = aSize;
	roi.height = aSize;

	int size1;
	int size2;
	nppSafeCall(nppiMeanGetBufferHostSize_32f_C1MR(roi, &size1));
	nppSafeCall(nppiSumGetBufferHostSize_32f_C1R(roi, &size2));
	sizeBuffer = std::max(size1, size2);
	nppiMaxIndxGetBufferHostSize_32f_C1R(roi, &size1);
	sizeBuffer = std::max(size1, sizeBuffer);

	buffer = new CudaDeviceVariable(sizeBuffer);
	rowSum1H = new float[aSize];
	rowSum2H = new float[aSize];
	rowSumH = new float[aSize];

	int n[] = { aSize, aSize };
	int inembed[] = { aSize, mc1.GetPitch() / sizeof(float) };
	int onembed[] = { aSize, (aSize / 2 + 1) };

	cufftPlanMany(&plan, 2, n, inembed, 1, 1, onembed, 1, 1, cufftType::CUFFT_R2C, 1);
	cufftPlanMany(&planInv, 2, n, onembed, 1, 1, inembed, 1, 1, cufftType::CUFFT_C2R, 1);

}

CudaCrossCorrelator::~CudaCrossCorrelator()
{
	delete buffer;
	delete[] rowSum1H;
	delete[] rowSum2H;
	delete[] rowSumH;
}

float2 CudaCrossCorrelator::GetShift(NPPImage_32fC1 & reference, NPPImage_32fC1 & img, int aMaxShift, float lp, float hp, float lps, float hps)
{
	float2 res;

	//Create mask for K2 camera stripes (missing data in frames)
	sumRow(reference, rowSum);
	rowSum.CopyDeviceToHost(rowSum1H);

	sumRow(img, rowSum);
	rowSum.CopyDeviceToHost(rowSum2H);

	for (size_t i = 0; i < size; i++)
	{
		rowSumH[i] = 0;
		rowSum1H[i] = rowSum1H[i] * rowSum2H[i];
	}

	//expand mask by maxShift allowed
	for (size_t i = aMaxShift; i < size- aMaxShift; i++)
	{
		if (rowSum1H[i] == 0)
		{
			rowSumH[i] = 0;
		}
		else
		{
			rowSumH[i] = 1;
			for (size_t j = i - aMaxShift; j < i + aMaxShift; j++)
			{
				if (rowSum1H[j] == 0)
				{
					rowSumH[i] = 0;
					break;
				}
			}
		}
	}

	rowSum.CopyHostToDevice(rowSumH);
	createMask(mask, rowSum);

	nppSafeCall(nppiSet_32f_C1R(0, maskF.GetPtrRoi(), maskF.GetPitch(), maskF.GetSizeRoi()));
	nppSafeCall(nppiSet_32f_C1MR(1, maskF.GetPtrRoi(), maskF.GetPitch(), maskF.GetSizeRoi(), mask.GetPtrRoi(), mask.GetPitch()));

	double sumMask;
	nppSafeCall(nppiSum_32f_C1R(maskF.GetPtrRoi(), maskF.GetPitch(), maskF.GetSizeRoi(), (uchar*)buffer->GetDevicePtr(), (double*)rowSum.GetDevicePtr()));
	rowSum.CopyDeviceToHost(&sumMask, sizeof(double));

	//Remove dead pixels:





	//Fourier filter the input images
	cufftSafeCall(cufftExecR2C(plan, reference.GetPtrRoi(), (cufftComplex*)fftRef.GetDevicePtr()));
	cufftSafeCall(cufftExecR2C(plan, img.GetPtrRoi(), (cufftComplex*)fftIm.GetDevicePtr()));
	four_filter(fftRef, (size / 2 + 1) * sizeof(cufftComplex), size, lp, hp, lps, hps);
	four_filter(fftIm, (size / 2 + 1) * sizeof(cufftComplex), size, lp, hp, lps, hps);
	cufftSafeCall(cufftExecC2R(planInv, (cufftComplex*)fftRef.GetDevicePtr(), reference.GetPtrRoi()));
	cufftSafeCall(cufftExecC2R(planInv, (cufftComplex*)fftIm.GetDevicePtr(), img.GetPtrRoi()));


	nppSafeCall(nppiMean_32f_C1MR(reference.GetPtrRoi(), reference.GetPitch(), mask.GetPtrRoi(), mask.GetPitch(), mask.GetSizeRoi(), (uchar*)buffer->GetDevicePtr(), (double*)rowSum.GetDevicePtr()));
	double meanImRef;
	rowSum.CopyDeviceToHost(&meanImRef, sizeof(double));
	nppSafeCall(nppiSubC_32f_C1R(reference.GetPtrRoi(), reference.GetPitch(), (float)meanImRef, mca.GetPtrRoi(), mca.GetPitch(), mca.GetSizeRoi()));
	nppSafeCall(nppiMul_32f_C1IR(maskF.GetPtrRoi(), maskF.GetPitch(), mca.GetPtrRoi(), mca.GetPitch(), mca.GetSizeRoi()));

	nppSafeCall(nppiSqr_32f_C1R(mca.GetPtrRoi(), mca.GetPitch(), mca2.GetPtrRoi(), mca2.GetPitch(), mca2.GetSizeRoi()));

	double norm_patch1;
	nppSafeCall(nppiSum_32f_C1R(mca2.GetPtrRoi(), mca2.GetPitch(), mca2.GetSizeRoi(), (uchar*)buffer->GetDevicePtr(), (double*)rowSum.GetDevicePtr()));
	rowSum.CopyDeviceToHost(&norm_patch1, sizeof(double));

	norm_patch1 = sqrt(norm_patch1);

	nppSafeCall(nppiSqr_32f_C1R(img.GetPtrRoi(), img.GetPitch(), mca2.GetPtrRoi(), mca2.GetPitch(), mca2.GetSizeRoi()));
	cufftSafeCall(cufftExecR2C(plan, mca.GetPtrRoi(), (cufftComplex*)fftRef.GetDevicePtr()));
	cufftSafeCall(cufftExecR2C(plan, img.GetPtrRoi(), (cufftComplex*)fftIm.GetDevicePtr()));
	cufftSafeCall(cufftExecR2C(plan, mca2.GetPtrRoi(), (cufftComplex*)fftIm2.GetDevicePtr()));
	cufftSafeCall(cufftExecR2C(plan, maskF.GetPtrRoi(), (cufftComplex*)fftMask.GetDevicePtr()));

	conj(fftRef, fftIm, (size / 2 + 1) * sizeof(cufftComplex), size);
	cufftSafeCall(cufftExecC2R(planInv, (cufftComplex*)fftRef.GetDevicePtr(), mc1.GetPtrRoi()));

	fftRef.CopyDeviceToDevice(fftMask);
	conj(fftRef, fftIm, (size / 2 + 1) * sizeof(cufftComplex), size);
	conj(fftMask, fftIm2, (size / 2 + 1) * sizeof(cufftComplex), size);

	cufftSafeCall(cufftExecC2R(planInv, (cufftComplex*)fftRef.GetDevicePtr(), mca.GetPtrRoi()));
	cufftSafeCall(cufftExecC2R(planInv, (cufftComplex*)fftMask.GetDevicePtr(), mcn.GetPtrRoi()));

	//normalize from FFT
	nppSafeCall(nppiDivC_32f_C1IR(size*size, mc1.GetPtrRoi(), mc1.GetPitch(), mc1.GetSizeRoi()));
	nppSafeCall(nppiDivC_32f_C1IR(size*size, mca.GetPtrRoi(), mca.GetPitch(), mca.GetSizeRoi()));
	nppSafeCall(nppiDivC_32f_C1IR(size*size, mcn.GetPtrRoi(), mcn.GetPitch(), mcn.GetSizeRoi()));
	
	nppSafeCall(nppiSqr_32f_C1R(mca.GetPtrRoi(), mca.GetPitch(), mca2.GetPtrRoi(), mca2.GetPitch(), mca2.GetSizeRoi()));

	nppSafeCall(nppiDivC_32f_C1IR((float)sumMask, mca2.GetPtrRoi(), mca2.GetPitch(), mca2.GetSizeRoi()));
	nppSafeCall(nppiSub_32f_C1IR(mca2.GetPtrRoi(), mca2.GetPitch(), mcn.GetPtrRoi(), mcn.GetPitch(), mcn.GetSizeRoi()));
	nppSafeCall(nppiSqrt_32f_C1IR(mcn.GetPtrRoi(), mcn.GetPitch(), mcn.GetSizeRoi()));
	nppSafeCall(nppiMulC_32f_C1IR(norm_patch1, mcn.GetPtrRoi(), mcn.GetPitch(), mcn.GetSizeRoi()));

	nppSafeCall(nppiDiv_32f_C1IR(mcn.GetPtrRoi(), mcn.GetPitch(), mc1.GetPtrRoi(), mc1.GetPitch(), mc1.GetSizeRoi()));

	maxShift(mc1, aMaxShift);

	nppSafeCall(nppiMaxIndx_32f_C1R(mc1.GetPtrRoi(), mc1.GetPitch(), mc1.GetSizeRoi(), (Npp8u*)buffer->GetDevicePtr(), (float*)rowSum.GetDevicePtr(), ((int*)rowSum.GetDevicePtr()) + 1, ((int*)rowSum.GetDevicePtr()) + 2));

	float maxVal;
	int xy[3];
	rowSum.CopyDeviceToHost(&maxVal, sizeof(float));
	rowSum.CopyDeviceToHost(xy, sizeof(int) * 3);

	//Get shift:
	res.x = xy[1];
	res.y = xy[2];

	if (res.x > size / 2)
	{
		res.x -= size;
	}

	if (res.y > size / 2)
	{
		res.y -= size;
	}

	return res;
}
