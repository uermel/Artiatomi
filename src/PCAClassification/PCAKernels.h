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


#ifndef PCAKERNEL_H
#define PCAKERNEL_H

#include <CudaContext.h>
#include <CudaKernel.h>
#include <cutil_math_.h>


class ComputeEigenImagesKernel : public Cuda::CudaKernel
{
public:
	ComputeEigenImagesKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
	ComputeEigenImagesKernel(CUmodule aModule);

	float operator()(int numberOfVoxels, int numberOfEigenImages, int particle, int numberOfParticles, Cuda::CudaDeviceVariable& ccMatrix, Cuda::CudaDeviceVariable& volIn, Cuda::CudaDeviceVariable& eigenImages);
};

#endif