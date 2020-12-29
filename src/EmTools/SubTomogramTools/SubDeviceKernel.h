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


#ifndef SUBDEVICEKERNEL_H
#define SUBDEVICEKERNEL_H

#include "../Basics/Default.h"

#include <CudaContext.h>
#include <CudaTextures.h>
#include <CudaKernel.h>
#include <cutil_math_.h>


class SubDeviceKernel : public Cuda::CudaKernel
{
public:
	SubDeviceKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
	SubDeviceKernel(CUmodule aModule);

	float operator()(Cuda::CudaDeviceVariable& value, Cuda::CudaDeviceVariable& dataInOut, int size);
	void operator()(CUstream stream, Cuda::CudaDeviceVariable& value, Cuda::CudaDeviceVariable& dataInOut, int size);
};

class SubDivDeviceKernel : public Cuda::CudaKernel
{
public:
	SubDivDeviceKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
	SubDivDeviceKernel(CUmodule aModule);

	float operator()(Cuda::CudaDeviceVariable& value, Cuda::CudaDeviceVariable& dataInOut, int size, float div);
	void operator()(CUstream stream, Cuda::CudaDeviceVariable& value, Cuda::CudaDeviceVariable& dataInOut, int size, float div);
};
#endif