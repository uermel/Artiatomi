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


#ifndef BASEALIGNER_H
#define BASEALIGNER_H

// System
#include <fstream>
#include <npp.h>
#include <cufft.h>

// EmTools
#include "FileIO/MovieStack.h"
#include "CudaHelpers/CudaContext.h"
#include "CudaHelpers/CudaVariables.h"
#include "CudaHelpers/CudaKernel.h"
#include "CudaHelpers/NPPImages.h"

// Self
#include "AlignmentOptions.h"
#include "Kernels.h"

class BaseAligner
{
private:
	int width;
	int height;
	int maxFrameCount;
	int patchSize;
	DataType_enum datatype;
	ofstream logFile;
	cuComplex** ffts;
	NppiPoint* shifts;
	FourierFilterKernel* kernelFourierFilter;
	MaxShiftKernel* kernelMaxShift;
	ConjMulKernel* kernelConjMul;
	Cuda::CudaContext* ctx;
	NPPImage_32fC1* imgOrig;
	NPPImage_32fC1* imgPatch;
	Cuda::CudaDeviceVariable* imgFFT_A;
	Cuda::CudaDeviceVariable* imgFFT_B;

public:
	BaseAligner(int aWidth, int aHeight, int aMaxFrameCount, DataType_enum aDataType, string aLogFile, int aPatchSize);
	void LoadFile(string aFilename, int lp, int lps, int hp, int hps);
	virtual void ComputeShifts(int maxShift) = 0;
	void CorrectShifts(Interpolation_enum interpolation);
	void SaveFinalImage(string aOutFile, bool append);
};

#endif