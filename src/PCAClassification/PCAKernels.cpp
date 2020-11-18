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

#include "PCAKernels.h"
using namespace Cuda;

ComputeEigenImagesKernel::ComputeEigenImagesKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
	: CudaKernel("computeEigenImages", aModule, aGridDim, aBlockDim, 0)
{

}

ComputeEigenImagesKernel::ComputeEigenImagesKernel(CUmodule aModule)
	: CudaKernel("computeEigenImages", aModule, make_dim3(1, 1, 1), make_dim3(32*16, 1, 1), 0)
{

}

float ComputeEigenImagesKernel::operator()(int numberOfVoxels, int numberOfEigenImages, int particle, int numberOfParticles, CudaDeviceVariable& ccMatrix,
	CudaDeviceVariable& volIn, CudaDeviceVariable& eigenImages)
{
	CudaKernel::SetComputeSize(numberOfVoxels, numberOfEigenImages, 1);
	CUdeviceptr ccMatrix_dptr = ccMatrix.GetDevicePtr();
	CUdeviceptr volIn_dptr = volIn.GetDevicePtr();
	CUdeviceptr eigenImages_dptr = eigenImages.GetDevicePtr();

	void* arglist[7];

	arglist[0] = &numberOfVoxels;
	arglist[1] = &numberOfEigenImages;
	arglist[2] = &particle;
	arglist[3] = &numberOfParticles;
	arglist[4] = &ccMatrix_dptr;
	arglist[5] = &volIn_dptr;
	arglist[6] = &eigenImages_dptr;

	float ms;

	CUevent eventStart;
	CUevent eventEnd;
	CUstream stream = 0;
	cudaSafeCall(cuEventCreate(&eventStart, CU_EVENT_BLOCKING_SYNC));
	cudaSafeCall(cuEventCreate(&eventEnd, CU_EVENT_BLOCKING_SYNC));

	cudaSafeCall(cuEventRecord(eventStart, stream));
	cudaSafeCall(cuLaunchKernel(mFunction, mGridDim.x, mGridDim.y, mGridDim.z, mBlockDim.x, mBlockDim.y, mBlockDim.z, mSharedMemSize, NULL, arglist, NULL));

	cudaSafeCall(cuCtxSynchronize());

	cudaSafeCall(cuEventRecord(eventEnd, stream));
	cudaSafeCall(cuEventSynchronize(eventEnd));
	cudaSafeCall(cuEventElapsedTime(&ms, eventStart, eventEnd));

	cudaSafeCall(cuEventDestroy(eventStart));
	cudaSafeCall(cuEventDestroy(eventEnd));

	return ms;
}