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


#include "CudaContext.h"

Cuda::CudaContext::CudaContext(int deviceID, CUctx_flags ctxFlags) : 
	mDeviceID(deviceID), mCtxFlags(ctxFlags), mHcuContext(0), mHcuDevice(0)
{
	cudaSafeCall(cuDeviceGet(&mHcuDevice, deviceID));
	cudaSafeCall(cuCtxCreate(&mHcuContext, mCtxFlags, mHcuDevice));
}

Cuda::CudaContext::~CudaContext()
{
	if (mHcuContext)
	{
		cudaSafeCall(cuCtxDetach(mHcuContext));
	}
}
	
Cuda::CudaContext* Cuda::CudaContext::CreateInstance(int aDeviceID, CUctx_flags ctxFlags)
{
	int deviceCount;
	// test whether the driver has already been initialized
	if(cuDeviceGetCount(&deviceCount) == CUDA_ERROR_NOT_INITIALIZED)
	{
		cudaSafeCall(cuInit(0));
	}

	cudaSafeCall(cuDeviceGetCount(&deviceCount));        
	if (deviceCount == 0)
	{
		CudaException ex("Cuda initialization error: There is no device supporting CUDA");
		throw ex;
		return NULL;
	} 
	return new CudaContext(aDeviceID, ctxFlags);
}

void Cuda::CudaContext::DestroyInstance(CudaContext* aCtx)
{
	delete aCtx;
}    

void Cuda::CudaContext::DestroyContext(CudaContext* aCtx)
{
	cudaSafeCall(cuCtxDestroy(aCtx->mHcuContext));
	aCtx->mHcuContext = 0;
}   

void Cuda::CudaContext::PushContext()
{
	cudaSafeCall(cuCtxPushCurrent(mHcuContext));
}

void Cuda::CudaContext::PopContext()
{
	cudaSafeCall(cuCtxPopCurrent(NULL));
}

void Cuda::CudaContext::SetCurrent()
{
	cudaSafeCall(cuCtxSetCurrent(mHcuContext));
}

void Cuda::CudaContext::Synchronize()
{
	cudaSafeCall(cuCtxSynchronize());
}

CUmodule Cuda::CudaContext::LoadModule(const char* aModulePath)
{   
	CUmodule hcuModule;
	cudaSafeCall(cuModuleLoad(&hcuModule, aModulePath));
	return hcuModule;
}

Cuda::CudaKernel* Cuda::CudaContext::LoadKernel(std::string aModulePath, std::string aKernelName, dim3 aGridDim, dim3 aBlockDim, uint aDynamicSharedMemory)
{   
	CUmodule hcuModule = LoadModulePTX(aModulePath.c_str(), 0, NULL, NULL);

	Cuda::CudaKernel* kernel = new Cuda::CudaKernel(aKernelName, hcuModule, /*this, */aGridDim, aBlockDim, aDynamicSharedMemory);

	return kernel;
}

Cuda::CudaKernel* Cuda::CudaContext::LoadKernel(std::string aModulePath, std::string aKernelName, uint aGridDimX, uint aGridDimY, uint aGridDimZ, uint aBlockDimX, uint aBlockDimY, uint aDynamicSharedMemory)
{   
	CUmodule hcuModule = LoadModulePTX(aModulePath.c_str(), 0, NULL, NULL);

	dim3 aGridDim;
	aGridDim.x = aGridDimX;
	aGridDim.y = aGridDimY;
	aGridDim.z = aGridDimZ;
	dim3 aBlockDim;
	aBlockDim.x = aBlockDimX;
	aBlockDim.y = aBlockDimY;

	Cuda::CudaKernel* kernel = new Cuda::CudaKernel(aKernelName, hcuModule, /*this, */aGridDim, aBlockDim, aDynamicSharedMemory);

	return kernel;
}

CUmodule Cuda::CudaContext::LoadModulePTX(const char* aModulePath, uint aOptionCount, CUjit_option* aOptions, void** aOptionValues)
{   
	
	std::ifstream file(aModulePath, ios::in|ios::binary|ios::ate);
	if (!file.good())
	{
		std::string filename;
		filename = "File not found: ";
		filename += aModulePath;
		CudaException ex(__FILE__, __LINE__, filename, CUDA_ERROR_FILE_NOT_FOUND);
	}
	ifstream::pos_type size;
	size = file.tellg();
	char* memblock = new char [(size_t)size+1];
	file.seekg (0, ios::beg);
	file.read (memblock, size);
	file.close();
	memblock[size] = 0;
	//cout << endl << endl << "Filesize is: " << size << endl;
	
	CUmodule hcuModule = LoadModulePTX(aOptionCount, aOptions, aOptionValues, memblock);
	
	//CUmodule hcuModule;
	//cudaSafeCall(cuModuleLoadData(&hcuModule, memblock));
	
	//return hcuModule;
	delete[] memblock;
	return hcuModule;
}

CUmodule Cuda::CudaContext::LoadModulePTX(uint aOptionCount, CUjit_option* aOptions, void** aOptionValues, const void* aModuleImage)
{   
	CUmodule hcuModule;
	cudaSafeCall(cuModuleLoadDataEx(&hcuModule, aModuleImage, aOptionCount, aOptions, aOptionValues));
	
	return hcuModule;
}

CUmodule Cuda::CudaContext::LoadModulePTX(const void* aModuleImage, uint aMaxRegCount, bool showInfoBuffer, bool showErrorBuffer)
{   	
	uint jitOptionCount = 0;
	CUjit_option ptxOptions[6];
	void* jitValues[6];
	int indexSet = 0;
	char infoBuffer[1025];
	char compilerBuffer[1025];
	uint compilerBufferSize = 1024;
	uint infoBufferSize = 1024;
	infoBuffer[1024] = 0;
	compilerBuffer[1024] = 0;
	int indexInfoBufferSize = 0;
	int indexCompilerBufferSize = 0;

	if (showInfoBuffer)
	{
		jitOptionCount += 3;
		ptxOptions[indexSet] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
		jitValues[indexSet] = (void*)infoBufferSize;
		indexInfoBufferSize = indexSet;
		indexSet++;
		ptxOptions[indexSet] = CU_JIT_INFO_LOG_BUFFER;
		jitValues[indexSet] = infoBuffer;
		indexSet++;
		ptxOptions[indexSet] = CU_JIT_LOG_VERBOSE;
		jitValues[indexSet] = (void*)1;
		indexSet++;
	}

	if (showErrorBuffer)
	{
		jitOptionCount += 2;
		ptxOptions[indexSet] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
		jitValues[indexSet] = (void*)compilerBufferSize;
		indexCompilerBufferSize = indexSet;
		indexSet++;
		ptxOptions[indexSet] = CU_JIT_ERROR_LOG_BUFFER;
		jitValues[indexSet] = compilerBuffer;
		indexSet++;
	}
	
	if (aMaxRegCount > 0)
	{
		jitOptionCount += 1;
		ptxOptions[indexSet] = CU_JIT_MAX_REGISTERS;
		jitValues[indexSet] = (void*)aMaxRegCount;
		indexSet++;
	}
	
	CUmodule hcuModule;
	cudaSafeCall(cuModuleLoadDataEx(&hcuModule, aModuleImage, jitOptionCount, ptxOptions, jitValues));

	if (showInfoBuffer)
	{
		if (jitValues[indexInfoBufferSize])
			printf("Cuda JIT Info: \n%s\n", infoBuffer);
	}

	if (showErrorBuffer)
	{
		if (jitValues[indexCompilerBufferSize])
			printf("Cuda JIT Error: \n%s\n", compilerBuffer);
	}
	
	return hcuModule;
}

void Cuda::CudaContext::UnloadModule(CUmodule& aModule)
{
	if (aModule)
	{
		cudaSafeCall(cuModuleUnload(aModule));
		aModule = 0;
	}
}		

//CUarray Cuda::CudaContext::CreateArray1D(unsigned int aNumElements, CUarray_format aFormat, unsigned int aNumChannels)
//{
//	CUarray hCuArray;
//	CUDA_ARRAY_DESCRIPTOR props;
//	props.Width = aNumElements;
//	props.Height = 0;
//	props.Format = aFormat;
//	props.NumChannels = aNumChannels;
//	cudaSafeCall(cuArrayCreate(&hCuArray, &props));
//	return hCuArray;
//}
//
//CUarray Cuda::CudaContext::CreateArray2D(unsigned int aWidth, unsigned int aHeight, CUarray_format aFormat, unsigned int aNumChannels)
//{
//	CUarray hCuArray;
//	CUDA_ARRAY_DESCRIPTOR props;
//	props.Width = aWidth;
//	props.Height = aHeight;
//	props.Format = aFormat;
//	props.NumChannels = aNumChannels;
//	cudaSafeCall(cuArrayCreate(&hCuArray, &props));
//	return hCuArray;
//}
//
//CUarray Cuda::CudaContext::CreateArray3D(unsigned int aWidth, unsigned int aHeight, unsigned int aDepth, CUarray_format aFormat, unsigned int aNumChannels, int aFlags)
//{
//	CUarray hCuArray;
//	CUDA_ARRAY3D_DESCRIPTOR props;
//	props.Width = aWidth;
//	props.Height = aHeight;
//	props.Depth = aDepth;
//	props.Format = aFormat;
//	props.NumChannels = aNumChannels;
//	props.Flags = aFlags;
//	cudaSafeCall(cuArray3DCreate(&hCuArray, &props));
//	return hCuArray;
//}
//
//CUdeviceptr Cuda::CudaContext::AllocateMemory(size_t aSizeInBytes)
//{
//	CUdeviceptr dBuffer;
//	cudaSafeCall(cuMemAlloc(&dBuffer, aSizeInBytes));
//	return dBuffer;
//}
		
void Cuda::CudaContext::ClearMemory(CUdeviceptr aPtr, unsigned int aValue, size_t aSizeInBytes)
{
	cudaSafeCall(cuMemsetD32(aPtr, aValue, aSizeInBytes / sizeof(unsigned int)));
}

/*void Cuda::CudaContext::FreeMemory(CUdeviceptr dBuffer)
{
	cudaSafeCall(cuMemFree(dBuffer));
}	*/	

//void Cuda::CudaContext::CopyToDevice(CUdeviceptr aDest, const void* aSource, unsigned int aSizeInBytes)
//{
//	cudaSafeCall(cuMemcpyHtoD(aDest, aSource, aSizeInBytes));
//}
//
//void Cuda::CudaContext::CopyToHost(void* aDest, CUdeviceptr aSource, unsigned int aSizeInBytes)
//{
//	cudaSafeCall(cuMemcpyDtoH(aDest, aSource, aSizeInBytes));
//}

Cuda::CudaDeviceProperties* Cuda::CudaContext::GetDeviceProperties()
{
	Cuda::CudaDeviceProperties* props = new Cuda::CudaDeviceProperties(mHcuDevice, mDeviceID);
	return props;
}

//void Cuda::CudaContext::SetTextureProperties(CUtexref aHcuTexRef, const TextureProperties& aTexProps)
//{
//	cudaSafeCall(cuTexRefSetAddressMode(aHcuTexRef, 0, aTexProps.addressMode[0]));
//	cudaSafeCall(cuTexRefSetAddressMode(aHcuTexRef, 1, aTexProps.addressMode[1]));
//	cudaSafeCall(cuTexRefSetFilterMode(aHcuTexRef, aTexProps.filterMode));
//	cudaSafeCall(cuTexRefSetFlags(aHcuTexRef, aTexProps.otherFlags));
//	cudaSafeCall(cuTexRefSetFormat(aHcuTexRef, aTexProps.format, aTexProps.numChannels));
//}      
//
//void Cuda::CudaContext::SetTexture3DProperties(CUtexref aHcuTexRef, const Texture3DProperties& aTexProps)
//{
//	cudaSafeCall(cuTexRefSetAddressMode(aHcuTexRef, 0, aTexProps.addressMode[0]));
//	cudaSafeCall(cuTexRefSetAddressMode(aHcuTexRef, 1, aTexProps.addressMode[1]));
//	cudaSafeCall(cuTexRefSetAddressMode(aHcuTexRef, 2, aTexProps.addressMode[2]));
//	cudaSafeCall(cuTexRefSetFilterMode(aHcuTexRef, aTexProps.filterMode));
//	cudaSafeCall(cuTexRefSetFlags(aHcuTexRef, aTexProps.otherFlags));
//	cudaSafeCall(cuTexRefSetFormat(aHcuTexRef, aTexProps.format, aTexProps.numChannels));
//}

size_t Cuda::CudaContext::GetFreeMemorySize()
{
	size_t sizeTotal, sizeFree;
	cudaSafeCall(cuMemGetInfo(&sizeFree, &sizeTotal));
	return sizeFree;
}

size_t Cuda::CudaContext::GetMemorySize()
{
	size_t sizeTotal, sizeFree;
	cudaSafeCall(cuMemGetInfo(&sizeFree, &sizeTotal));
	return sizeTotal;
}