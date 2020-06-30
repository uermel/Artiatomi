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


#include <stdio.h>
#include <tchar.h>
#include "../Basics/Default.h"
#include "../CudaHelpers/CudaHelpers.h"
#include <iostream>

void Alloc()
{
	//cuMemAlloc(&ptr, 10000);
}

void PrintFreeMem()
{
	
}


int _tmain(int argc, _TCHAR* argv[])
{
#ifdef USE_CUDA
	Cuda::CudaContext* ctx = Cuda::CudaThreadBoundContext::Get(0);
	Cuda::CudaThreadBoundContext::Cleanup();
	ctx = Cuda::CudaThreadBoundContext::Get(0);

	
	
	CUdeviceptr ptr;
	size_t freeMem;
	size_t totSize;

	auto ret0 = RunInCudaThread(cuMemGetInfo, &freeMem, &totSize);
	ret0.get();
	std::cout << "Total size: " << totSize << "; Free size: " << freeMem << std::endl;

	auto ret1 = RunInCudaThread(cuMemAlloc, &ptr, 100000);
	ret1.get();

	auto ret2 = RunInCudaThread(cuMemGetInfo, &freeMem, &totSize);
	ret2.get();
	std::cout << "Total size: " << totSize << "; Free size: " << freeMem << std::endl;
#endif
}