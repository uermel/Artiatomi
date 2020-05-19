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


#ifndef AVGPROCESS_H
#define AVGPROCESS_H

#include "basics/default.h"
#include <cuda.h>
#include <cufft.h>
#include "io/EMFile.h"
#include "cuda/CudaVariables.h"
#include "cuda/CudaKernel.h"
#include "cuda/CudaArrays.h"
#include "cuda/CudaTextures.h"
#include "cuda/CudaContext.h"

#include "BasicKernel.h"
#include "CudaReducer.h"
#include "CudaRot.h"
#include <map>

using namespace std;


#define cufftSafeCall(err) __cufftSafeCall (err, __FILE__, __LINE__)

struct maxVals_t
{
	float ccVal;
	int index;
	float rphi;
	float rpsi;
	float rthe;

	void getXYZ(int size, int& x, int& y, int&z);
	static int getIndex(int size, int x, int y, int z);
	size_t ref;
};


class AvgProcess
{
private:
	bool binarizeMask;
	bool rotateMaskCC;
	bool useFilterVolume;
	float phi, psi, theta;
	float shiftX, shiftY, shiftZ;
	float maxCC;
	float phi_angiter;
	float phi_angincr;
	float angiter;
	float angincr;

	size_t sizeVol, sizeTot;

	float* mask;
	float* ref;
	float* ccMask;
	float* sum_h;
	int*   index;
	float2* sumCplx;

	CUstream stream;
	CudaContext* ctx;

	CudaRot rot;
	CudaRot rotMask;
	CudaRot rotMaskCC;
	CudaReducer reduce;
	CudaSub sub;
	CudaMakeCplxWithSub makecplx;
	CudaBinarize binarize;
	CudaMul mul;
	CudaFFT fft;
	CudaMax max;

	CudaDeviceVariable d_ffttemp;
	CudaDeviceVariable d_particle, d_particleCplx, d_particleSqrCplx, d_particleCplx_orig, d_particleSqrCplx_orig, d_filter;
	CudaDeviceVariable d_wedge;
	CudaDeviceVariable d_reference, d_reference_orig, d_referenceCplx;
	CudaDeviceVariable d_mask, d_mask_orig, d_maskCplx;
	CudaDeviceVariable d_ccMask, d_ccMask_Orig;
	CudaDeviceVariable d_buffer;
	CudaDeviceVariable d_index;
	CudaDeviceVariable nVox;
	CudaDeviceVariable sum;
	CudaDeviceVariable sumSqr;
	CudaDeviceVariable maxVals;

	cufftHandle ffthandle;

public:
	AvgProcess(size_t _sizeVol, CUstream _stream, CudaContext* _ctx, float* _mask, float* _ref, float* _ccMask, float aPhiAngIter, float aPhiAngInc, float aAngIter, float aAngIncr, bool aBinarizeMask, bool aRotateMaskCC, bool aUseFilterVolume, bool linearInterpolation);
	~AvgProcess();

	maxVals_t execute(float* _data, float* wedge, float* filter, float oldphi, float oldpsi, float oldtheta, float rDown, float rUp, float smooth, float3 oldShift, bool couplePhiToPsi, bool computeCCValOnly, int oldIndex);
	maxVals_t executePhaseCorrelation(float* _data, float* wedge, float* filter, float oldphi, float oldpsi, float oldtheta, float rDown, float rUp, float smooth, float3 oldShift, bool couplePhiToPsi, bool computeCCValOnly, int oldIndex);

	//maxVals_t executeMaxAll(float* _data, float oldphi, float oldpsi, float oldtheta, float rDown, float rUp, float smooth, vector<float>& allCCs);
};

inline void __cufftSafeCall(cufftResult_t err, const char *file, const int line)
{
	if( CUFFT_SUCCESS != err)
	{
		std::string errMsg;
		switch(err)
		{
		case CUFFT_INVALID_PLAN:
			errMsg = "Invalid plan";
			break;
		case CUFFT_ALLOC_FAILED:
			errMsg = "CUFFT_ALLOC_FAILED";
			break;
		case CUFFT_INVALID_TYPE:
			errMsg = "CUFFT_INVALID_TYPE";
			break;
		case CUFFT_INVALID_VALUE:
			errMsg = "CUFFT_INVALID_VALUE";
			break;
		case CUFFT_INTERNAL_ERROR:
			errMsg = "CUFFT_INTERNAL_ERROR";
			break;
		case CUFFT_EXEC_FAILED:
			errMsg = "CUFFT_EXEC_FAILED";
			break;
		case CUFFT_SETUP_FAILED:
			errMsg = "CUFFT_SETUP_FAILED";
			break;
		case CUFFT_INVALID_SIZE:
			errMsg = "CUFFT_INVALID_SIZE";
			break;
		case CUFFT_UNALIGNED_DATA:
			errMsg = "CUFFT_UNALIGNED_DATA";
			break;
		case CUFFT_INCOMPLETE_PARAMETER_LIST:
			errMsg = "CUFFT_INCOMPLETE_PARAMETER_LIST";
			break;
		case CUFFT_INVALID_DEVICE:
			errMsg = "CUFFT_INVALID_DEVICE";
			break;
		case CUFFT_PARSE_ERROR:
			errMsg = "CUFFT_PARSE_ERROR";
			break;
		case CUFFT_NO_WORKSPACE:
			errMsg = "CUFFT_NO_WORKSPACE";
			break;
		}

		CudaException ex(file, line, errMsg, (CUresult)err);
		throw ex;
	} //if CUDA_SUCCESS
}

#endif //AVGPROCESS_H
