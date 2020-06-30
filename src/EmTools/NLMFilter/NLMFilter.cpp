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


#define USE_CUDA

#include "../Basics/Default.h"
#include "../FileIO/Volume.h"
#include "../Threading/SpecificBackgroundThread.h"

#include <cuda.h>
#include "../CudaHelpers/CudaContext.h"
#include "../CudaHelpers/CudaArrays.h"
#include "../CudaHelpers/CudaKernel.h"
#include "../CudaHelpers/CudaVariables.h"

#ifdef _DEBUG
#define PTX "C:\\Users\\Michael Kunz\\Source\\Repos\\EmTools\\x64\\Debug\\kernelNLM.ptx"
#else
#include "kernelsNLM.h"
//#define PTX "kernelNLM.ptx"
#endif

using namespace Cuda;

class ComputeDistanceForShiftKernel : public CudaKernel
{
private:
	dim3 _size;
public: 
	ComputeDistanceForShiftKernel(CUmodule module, dim3 size)
		: CudaKernel("ComputeDistanceForShift", module, dim3((size.x + 7) / 8, (size.y + 7) / 8, (size.z + 7) / 8), dim3(8, 8, 8), 0),
		_size(size)
	{
	
	}

	void operator()(dim3 size, CudaDeviceVariable& imgIn, CudaDeviceVariable& dist, int shiftX, int shiftY, int shiftZ)
	{
		_size = size;
		CUdeviceptr p_imgIn = imgIn.GetDevicePtr();
		CUdeviceptr p_dist = dist.GetDevicePtr();
		int stride = _size.x * sizeof(float);

		void* arglist[10];
		arglist[0] = &p_imgIn;
		arglist[1] = &stride;
		arglist[2] = &p_dist;
		arglist[3] = &stride;
		arglist[4] = &_size.x;
		arglist[5] = &_size.y;
		arglist[6] = &_size.z;
		arglist[7] = &shiftX;
		arglist[8] = &shiftY;
		arglist[9] = &shiftZ;

		cudaSafeCall(cuLaunchKernel(mFunction, mGridDim.x, mGridDim.y, mGridDim.z, mBlockDim.x, mBlockDim.y, mBlockDim.z, mSharedMemSize, NULL, arglist, NULL));
		cudaSafeCall(cuCtxSynchronize());
	}
};

class AddWeightedPixelKernel : public CudaKernel
{
private:
	dim3 _size;
public:
	AddWeightedPixelKernel(CUmodule module, dim3 size)
		: CudaKernel("AddWeightedPixel", module, dim3((size.x + 7) / 8, (size.y + 7) / 8, (size.z + 7) / 8), dim3(8, 8, 8), 0),
		_size(size)
	{

	}

	void operator()(dim3 size, CudaDeviceVariable& imgIn, CudaDeviceVariable& imgOut, CudaDeviceVariable& weight, CudaDeviceVariable& weightMax, CudaDeviceVariable& weightSum, int shiftX, int shiftY, int shiftZ, float sigma, float filterParam, float patchSize)
	{
		_size = size;
		CUdeviceptr p_imgIn = imgIn.GetDevicePtr();
		CUdeviceptr p_imgOut = imgOut.GetDevicePtr();
		CUdeviceptr p_weight = weight.GetDevicePtr();
		CUdeviceptr p_weightMax = weightMax.GetDevicePtr();
		CUdeviceptr p_weightSum = weightSum.GetDevicePtr();
		int stride = _size.x * sizeof(float);

		void* arglist[17];
		arglist[0] = &p_imgIn;
		arglist[1] = &stride;
		arglist[2] = &p_imgOut;
		arglist[3] = &stride;
		arglist[4] = &p_weight;
		arglist[5] = &stride;
		arglist[6] = &p_weightMax;
		arglist[7] = &p_weightSum;
		arglist[8] = &_size.x;
		arglist[9] = &_size.y;
		arglist[10] = &_size.z;
		arglist[11] = &shiftX;
		arglist[12] = &shiftY;
		arglist[13] = &shiftZ;
		arglist[14] = &sigma;
		arglist[15] = &filterParam;
		arglist[16] = &patchSize;

		cudaSafeCall(cuLaunchKernel(mFunction, mGridDim.x, mGridDim.y, mGridDim.z, mBlockDim.x, mBlockDim.y, mBlockDim.z, mSharedMemSize, NULL, arglist, NULL));
		cudaSafeCall(cuCtxSynchronize());
	}
};


class AddWeightedPixelFinalKernel : public CudaKernel
{
private:
	dim3 _size;
public:
	AddWeightedPixelFinalKernel(CUmodule module, dim3 size)
		: CudaKernel("AddWeightedPixelFinal", module, dim3((size.x + 7) / 8, (size.y + 7) / 8, (size.z + 7) / 8), dim3(8, 8, 8), 0),
		_size(size)
	{

	}

	void operator()(dim3 size, CudaDeviceVariable& imgIn, CudaDeviceVariable& imgOut, CudaDeviceVariable& weightMax, CudaDeviceVariable& weightSum)
	{
		_size = size;
		CUdeviceptr p_imgIn = imgIn.GetDevicePtr();
		CUdeviceptr p_imgOut = imgOut.GetDevicePtr();
		CUdeviceptr p_weightMax = weightMax.GetDevicePtr();
		CUdeviceptr p_weightSum = weightSum.GetDevicePtr();
		int stride = _size.x * sizeof(float);

		void* arglist[17];
		arglist[0] = &p_imgIn;
		arglist[1] = &stride;
		arglist[2] = &p_imgOut;
		arglist[3] = &stride;
		arglist[4] = &p_weightMax;
		arglist[5] = &stride;
		arglist[6] = &p_weightSum;
		arglist[7] = &_size.x;
		arglist[8] = &_size.y;
		arglist[9] = &_size.z;

		cudaSafeCall(cuLaunchKernel(mFunction, mGridDim.x, mGridDim.y, mGridDim.z, mBlockDim.x, mBlockDim.y, mBlockDim.z, mSharedMemSize, NULL, arglist, NULL));
		cudaSafeCall(cuCtxSynchronize());
	}
};

class ComputeFinalPixelKernel : public CudaKernel
{
private:
	dim3 _size;
public:
	ComputeFinalPixelKernel(CUmodule module, dim3 size)
		: CudaKernel("ComputeFinalPixel", module, dim3((size.x + 7) / 8, (size.y + 7) / 8, (size.z + 7) / 8), dim3(8, 8, 8), 0),
		_size(size)
	{

	}

	void operator()(dim3 size, CudaDeviceVariable& imgIn, CudaDeviceVariable& imgOut, CudaDeviceVariable& totalWeight)
	{
		_size = size;
		CUdeviceptr p_imgIn = imgIn.GetDevicePtr();
		CUdeviceptr p_imgOut = imgOut.GetDevicePtr();
		CUdeviceptr p_totalWeight = totalWeight.GetDevicePtr();
		int stride = _size.x * sizeof(float);

		void* arglist[9];
		arglist[0] = &p_imgIn;
		arglist[1] = &stride;
		arglist[2] = &p_imgOut;
		arglist[3] = &stride;
		arglist[4] = &p_totalWeight;
		arglist[5] = &stride;
		arglist[6] = &_size.x;
		arglist[7] = &_size.y;
		arglist[8] = &_size.z;

		cudaSafeCall(cuLaunchKernel(mFunction, mGridDim.x, mGridDim.y, mGridDim.z, mBlockDim.x, mBlockDim.y, mBlockDim.z, mSharedMemSize, NULL, arglist, NULL));
		cudaSafeCall(cuCtxSynchronize());
	}
};

class ConvolveXKernel : public CudaKernel
{
private:
	dim3 _size;
	int _filterRadius;
public:
	ConvolveXKernel(CUmodule module, dim3 size, int filterRadius)
		: CudaKernel("convolveX", module, dim3((size.x + 511) / 512, (size.y), (size.z)), dim3(512, 1, 1), (512 + 2 * filterRadius) * sizeof(float)),
		_size(size), _filterRadius(filterRadius)
	{

	}

	void operator()(dim3 size, CudaDeviceVariable& imgIn, CudaDeviceVariable& imgOut)
	{
		_size = size;
		CUdeviceptr p_imgIn = imgIn.GetDevicePtr();
		CUdeviceptr p_imgOut = imgOut.GetDevicePtr();

		int stride = _size.x * sizeof(float);

		void* arglist[7];
		arglist[0] = &p_imgIn;
		arglist[1] = &p_imgOut;
		arglist[2] = &_size.x;
		arglist[3] = &_size.y;
		arglist[4] = &_size.z;
		arglist[5] = &stride;
		arglist[6] = &_filterRadius;

		cudaSafeCall(cuLaunchKernel(mFunction, mGridDim.x, mGridDim.y, mGridDim.z, mBlockDim.x, mBlockDim.y, mBlockDim.z, mSharedMemSize, NULL, arglist, NULL));
		cudaSafeCall(cuCtxSynchronize());
	}
};

class ConvolveYKernel : public CudaKernel
{
private:
	dim3 _size;
	int _filterRadius;
public:
	ConvolveYKernel(CUmodule module, dim3 size, int filterRadius)
		: CudaKernel("convolveY", module, dim3((size.x), (size.y + 511) / 512, (size.z)), dim3(1, 512, 1), (512 + 2 * filterRadius) * sizeof(float)),
		_size(size), _filterRadius(filterRadius)
	{

	}

	void operator()(dim3 size, CudaDeviceVariable& imgIn, CudaDeviceVariable& imgOut)
	{
		_size = size;
		CUdeviceptr p_imgIn = imgIn.GetDevicePtr();
		CUdeviceptr p_imgOut = imgOut.GetDevicePtr();

		int stride = _size.x * sizeof(float);

		void* arglist[7];
		arglist[0] = &p_imgIn;
		arglist[1] = &p_imgOut;
		arglist[2] = &_size.x;
		arglist[3] = &_size.y;
		arglist[4] = &_size.z;
		arglist[5] = &stride;
		arglist[6] = &_filterRadius;

		cudaSafeCall(cuLaunchKernel(mFunction, mGridDim.x, mGridDim.y, mGridDim.z, mBlockDim.x, mBlockDim.y, mBlockDim.z, mSharedMemSize, NULL, arglist, NULL));
		cudaSafeCall(cuCtxSynchronize());
	}
};

class ConvolveZKernel : public CudaKernel
{
private:
	dim3 _size;
	int _filterRadius;
public:
	ConvolveZKernel(CUmodule module, dim3 size, int filterRadius)
		: CudaKernel("convolveZ", module, dim3((size.x + 511) / 512, (size.y), (size.z)), dim3(512, 1, 1), (512 + 2 * filterRadius) * sizeof(float)),
		_size(size), _filterRadius(filterRadius)
	{

	}

	void operator()(dim3 size, CudaDeviceVariable& imgIn, CudaDeviceVariable& imgOut)
	{
		_size = size;
		CUdeviceptr p_imgIn = imgIn.GetDevicePtr();
		CUdeviceptr p_imgOut = imgOut.GetDevicePtr();

		int stride = _size.x * sizeof(float);

		void* arglist[7];
		arglist[0] = &p_imgIn;
		arglist[1] = &p_imgOut;
		arglist[2] = &_size.x;
		arglist[3] = &_size.y;
		arglist[4] = &_size.z;
		arglist[5] = &stride;
		arglist[6] = &_filterRadius;

		cudaSafeCall(cuLaunchKernel(mFunction, mGridDim.x, mGridDim.y, mGridDim.z, mBlockDim.x, mBlockDim.y, mBlockDim.z, mSharedMemSize, NULL, arglist, NULL));
		cudaSafeCall(cuCtxSynchronize());
	}
};


struct threadData
{
	CudaContext* ctx;
	CudaDeviceVariable* d_dataIn;
	CudaDeviceVariable* d_dataOut;
	CudaDeviceVariable* d_dataWeight;
	CudaDeviceVariable* d_dataSummedWeight;
	CudaDeviceVariable* d_dataWeightMax;
	CudaDeviceVariable* d_dataDistance;
	ComputeDistanceForShiftKernel* kernelComputeDistanceForShift;
	AddWeightedPixelFinalKernel* kernelAddWeightedPixelFinal;
	AddWeightedPixelKernel* kernelAddWeightedPixel;
	ComputeFinalPixelKernel* kernelComputeFinalPixelKernel;	
	ConvolveXKernel* kernelConvolveX;
	ConvolveYKernel* kernelConvolveY;
	ConvolveZKernel* kernelConvolveZ;
	int totalWidth, totalHeight, totalDepth;
	int partialWidth, partialHeight, partialDepth;
	float* partialVolume;
	int id;
};

int InitCudaOnDevice(int deviceID, int partialWidth, int partialHeight, int partialDepth, int totalWidth, int totalHeight, int totalDepth, int filterRadius, threadData* data)
{
	data->id = deviceID;
	data->ctx = CudaContext::CreateInstance(deviceID);

	data->partialVolume = new float[(size_t)partialWidth * (size_t)partialHeight * (size_t)partialDepth];
	data->partialWidth = partialWidth;
	data->partialHeight = partialHeight;
	data->partialDepth = partialDepth;
	data->totalWidth = totalWidth;
	data->totalHeight = totalHeight;
	data->totalDepth = totalDepth;

	size_t size = (size_t)partialWidth * (size_t)partialHeight * (size_t)partialDepth * 4;
	data->d_dataIn = new CudaDeviceVariable(size);
	data->d_dataOut = new CudaDeviceVariable(size);
	data->d_dataWeight = new CudaDeviceVariable(size);
	data->d_dataSummedWeight = new CudaDeviceVariable(size);
	data->d_dataWeightMax = new CudaDeviceVariable(size);
	data->d_dataDistance = new CudaDeviceVariable(size);

#ifdef _DEBUG
	CUmodule mod = data->ctx->LoadModulePTX(PTX, 0, NULL, NULL);
#else
	CUmodule mod = data->ctx->LoadModulePTX(kernelsNLM, 0, NULL, NULL);
#endif

	dim3 s(partialWidth, partialHeight, partialDepth);

	data->kernelComputeDistanceForShift = new ComputeDistanceForShiftKernel(mod, s);
	data->kernelAddWeightedPixel = new AddWeightedPixelKernel(mod, s);
	data->kernelAddWeightedPixelFinal = new AddWeightedPixelFinalKernel(mod, s);
	data->kernelComputeFinalPixelKernel = new ComputeFinalPixelKernel(mod, s);
	data->kernelConvolveX = new ConvolveXKernel(mod, s, filterRadius);
	data->kernelConvolveY = new ConvolveYKernel(mod, s, filterRadius);
	data->kernelConvolveZ = new ConvolveZKernel(mod, s, filterRadius);

	return 0;
}

int RunFilter(float* dataIn, float* dataOut, int startY, int dataWidth, int dataHeight, int dataDepth, int roiY, int roiWidth, int roiHeight, int roiDepth, int searchRadius, int patchRadius, float sigma, float filterParam, threadData* data)
{
	printf("Starting on %d!\n", data->id);
	memset(data->partialVolume, 0, (size_t)data->partialWidth * (size_t)data->partialHeight * (size_t)data->partialDepth * sizeof(float));
	
	for (size_t z = 0; z < dataDepth; z++)
	{
		for (size_t y = 0; y < dataHeight; y++)
		{
			for (size_t x = 0; x < dataWidth; x++)
			{
				data->partialVolume[z * dataWidth * dataHeight + y * dataWidth + x] = 
					dataIn[(z) * data->totalWidth * data->totalHeight + (y + startY) * data->totalWidth + x];
			}
		}
	}

	dim3 s(dataWidth, dataHeight, dataDepth);
	data->kernelComputeDistanceForShift->SetGridDimensions(dim3((dataWidth + 7) / 8, (dataHeight + 7) / 8, (dataDepth + 7) / 8));
	data->kernelAddWeightedPixel->SetGridDimensions(dim3((dataWidth + 7) / 8, (dataHeight + 7) / 8, (dataDepth + 7) / 8));
	data->kernelComputeFinalPixelKernel->SetGridDimensions(dim3((dataWidth + 7) / 8, (dataHeight + 7) / 8, (dataDepth + 7) / 8));
	data->kernelConvolveX->SetGridDimensions(dim3((dataWidth + 511) / 512, (dataHeight), (dataDepth)));
	data->kernelConvolveY->SetGridDimensions(dim3((dataWidth), (dataHeight + 511) / 512, (dataDepth)));
	data->kernelConvolveZ->SetGridDimensions(dim3((dataDepth + 511) / 512, (dataHeight), (dataWidth)));

	data->d_dataOut->Memset(0);
	data->d_dataWeight->Memset(0);
	data->d_dataSummedWeight->Memset(0);
	data->d_dataWeightMax->Memset(0);
	data->d_dataIn->CopyHostToDevice(data->partialVolume);
	

	for (int z = -searchRadius; z <= searchRadius; z++)
	{
		for (int y = -searchRadius; y <= searchRadius; y++)
		{
			for (int x = -searchRadius; x <= searchRadius; x++)
			{
				if (!(x == 0 && y == 0 && z == 0))
				{
					(*data->kernelComputeDistanceForShift)(s, *data->d_dataIn, *data->d_dataDistance, x, y, z);
					//save to to disk!
					/*data->d_dataDistance->CopyDeviceToHost(data->partialVolume);
					EmFile::InitHeader("C:\\Users\\Michael Kunz\\Desktop\\testOut.em", dataWidth, dataHeight, dataDepth, 0, DT_FLOAT);
					EmFile::WriteRawData("C:\\Users\\Michael Kunz\\Desktop\\testOut.em", data->partialVolume, (size_t)dataWidth * (size_t)dataHeight * (size_t)dataDepth * 4);*/

					(*data->kernelConvolveX)(s, *data->d_dataDistance, *data->d_dataWeight);
					//save to to disk!
					/*data->d_dataWeight->CopyDeviceToHost(data->partialVolume);
					EmFile::InitHeader("C:\\Users\\Michael Kunz\\Desktop\\testOutX.em", dataWidth, dataHeight, dataDepth, 0, DT_FLOAT);
					EmFile::WriteRawData("C:\\Users\\Michael Kunz\\Desktop\\testOutX.em", data->partialVolume, (size_t)dataWidth * (size_t)dataHeight * (size_t)dataDepth * 4);*/
					(*data->kernelConvolveY)(s, *data->d_dataWeight, *data->d_dataDistance);
					//save to to disk!
					/*data->d_dataDistance->CopyDeviceToHost(data->partialVolume);
					EmFile::InitHeader("C:\\Users\\Michael Kunz\\Desktop\\testOutY.em", dataWidth, dataHeight, dataDepth, 0, DT_FLOAT);
					EmFile::WriteRawData("C:\\Users\\Michael Kunz\\Desktop\\testOutY.em", data->partialVolume, (size_t)dataWidth * (size_t)dataHeight * (size_t)dataDepth * 4);*/
					(*data->kernelConvolveZ)(s, *data->d_dataDistance, *data->d_dataWeight);
					//save to to disk!
					/*data->d_dataWeight->CopyDeviceToHost(data->partialVolume);
					EmFile::InitHeader("C:\\Users\\Michael Kunz\\Desktop\\testOutZ.em", dataWidth, dataHeight, dataDepth, 0, DT_FLOAT);
					EmFile::WriteRawData("C:\\Users\\Michael Kunz\\Desktop\\testOutZ.em", data->partialVolume, (size_t)dataWidth * (size_t)dataHeight * (size_t)dataDepth * 4);*/

					(*data->kernelAddWeightedPixel)(s, *data->d_dataIn, *data->d_dataOut, *data->d_dataWeight, *data->d_dataWeightMax, *data->d_dataSummedWeight, x, y, z, sigma, filterParam, (float)(2 * patchRadius + 1));
					//save to to disk!
					/*data->d_dataOut->CopyDeviceToHost(data->partialVolume);
					EmFile::InitHeader("C:\\Users\\Michael Kunz\\Desktop\\testOut2.em", dataWidth, dataHeight, dataDepth, 0, DT_FLOAT);
					EmFile::WriteRawData("C:\\Users\\Michael Kunz\\Desktop\\testOut2.em", data->partialVolume, (size_t)dataWidth * (size_t)dataHeight * (size_t)dataDepth * 4);*/

				}
			}
		}
		if (data->id == 0)
		{
			printf("done for Z = %d\n", z);
		}
	}

	(*data->kernelAddWeightedPixelFinal)(s, *data->d_dataIn, *data->d_dataOut, *data->d_dataWeightMax, *data->d_dataSummedWeight);
	//data->d_dataSummedWeight->CopyDeviceToHost(data->partialVolume);
	////save to to disk!
	//EmFile::InitHeader("C:\\Users\\Michael Kunz\\Desktop\\testOutSum.em", dataWidth, dataHeight, dataDepth, 0, DT_FLOAT);
	//EmFile::WriteRawData("C:\\Users\\Michael Kunz\\Desktop\\testOutSum.em", data->partialVolume, (size_t)dataWidth * (size_t)dataHeight * (size_t)dataDepth * 4);
	//data->d_dataWeightMax->CopyDeviceToHost(data->partialVolume);
	////save to to disk!
	//EmFile::InitHeader("C:\\Users\\Michael Kunz\\Desktop\\testOutMax.em", dataWidth, dataHeight, dataDepth, 0, DT_FLOAT);
	//EmFile::WriteRawData("C:\\Users\\Michael Kunz\\Desktop\\testOutMax.em", data->partialVolume, (size_t)dataWidth * (size_t)dataHeight * (size_t)dataDepth * 4);
	//data->d_dataOut->CopyDeviceToHost(data->partialVolume);
	////save to to disk!
	//EmFile::InitHeader("C:\\Users\\Michael Kunz\\Desktop\\testOut.em", dataWidth, dataHeight, dataDepth, 0, DT_FLOAT);
	//EmFile::WriteRawData("C:\\Users\\Michael Kunz\\Desktop\\testOut.em", data->partialVolume, (size_t)dataWidth * (size_t)dataHeight * (size_t)dataDepth * 4);
	(*data->kernelComputeFinalPixelKernel)(s, *data->d_dataIn, *data->d_dataOut, *data->d_dataSummedWeight);

	data->d_dataOut->CopyDeviceToHost(data->partialVolume);

	for (size_t z = 0; z < roiDepth; z++)
	{
		for (size_t y = 0; y < roiHeight; y++)
		{
			for (size_t x = 0; x < roiWidth; x++)
			{
				dataOut[(z)* data->totalWidth * data->totalHeight + (y + roiY) * data->totalWidth + x] =
					data->partialVolume[z * dataWidth * dataHeight + (y - startY + roiY) * dataWidth + x];
			}
		}
	}
	printf("Finished on %d!\n", data->id);
	return 0;
}

int ShutDown(threadData* data)
{
	printf("thread %d finished!\n", data->id);
	delete data->d_dataIn;
	delete data->d_dataOut;
	delete data->d_dataWeight;
	delete data->d_dataSummedWeight;
	delete data->d_dataWeightMax;
	delete data->d_dataDistance;

	CudaContext::DestroyContext(data->ctx);
	return 0;
}

int main(int argc, char* argv[])
{
	if (argc < 8)
	{
		printf("Usage: NLMFilter inputfile outputfile gpuCount sigma filterParam searchRadius patchRadius\n");
		return 0;
	}

	char* inputFile = argv[1];
	char* outputFile = argv[2];
	int gpuCount = atoi(argv[3]);
	float sigma = atof(argv[4]);
	float filterParam = atof(argv[5]);
	int searchRadius = atoi(argv[6]);
	int patchRadius = atoi(argv[7]);
		
	ThreadPool* threads = new ThreadPool(gpuCount);

	CudaContext* ctx = CudaContext::CreateInstance(0);
	size_t memSize = ctx->GetFreeMemorySize();
	CudaContext::DestroyContext(ctx);
	/*size_t memSize = (20 * 6)*1024*1024;*/

	Volume dataIn(inputFile);
	printf("File loaded. Dimensions: %d; %d, %d\n", dataIn.GetWidth(), dataIn.GetHeight(), dataIn.GetDepth());
	if (dataIn.GetFileDataType() != DT_FLOAT)
	{
		printf("Only float data type is supported!\n");
		return -1;
	}

	float* dIn = (float*)dataIn.GetData();
	size_t totalSize = (size_t)dataIn.GetWidth() * (size_t)dataIn.GetHeight() * (size_t)dataIn.GetDepth();
	float minVal = FLT_MAX;
	float maxVal = -FLT_MAX;
	for (size_t i = 0; i < totalSize; i++)
	{
		float v = dIn[i];
		minVal = min(minVal, v);
		maxVal = max(maxVal, v);
	}
	printf("Data min: %f max: %f\n", minVal, maxVal);
	maxVal -= minVal;
	maxVal = 1.0f / maxVal;
	for (size_t i = 0; i < totalSize; i++)
	{
		dIn[i] = (dIn[i] - minVal) * maxVal;
	}
	printf("Data normalized to 0..1\n");

	float* dataOut = new float[(size_t)dataIn.GetWidth() * (size_t)dataIn.GetHeight() * (size_t)dataIn.GetDepth()];
	memset(dataOut, 0, sizeof(float) * (size_t)dataIn.GetWidth() * (size_t)dataIn.GetHeight() * (size_t)dataIn.GetDepth());

	//We need distance, weight, weightMax, summedWeight, input and output = 6 times the amount of data
	size_t memPerElement = memSize / 6 -100 * 1024 * 1024; //100 MB safety...

	//We split along y: find largest possible y that is smaller than the maximum memory availble.
	int computeX = dataIn.GetWidth();
	int computeY = 2 * searchRadius + 1;
	int computeZ = dataIn.GetDepth();

	for (int ySearch = computeY; ySearch <= dataIn.GetHeight(); ySearch++)
	{
		if ((size_t)computeX * (size_t)computeZ * (size_t)ySearch * 4 > memPerElement)
		{
			computeY = ySearch - 1;
			break;
		}
		computeY = ySearch;
	}

	if (computeY == 2 * searchRadius)
	{
		printf("Not enough memory on device!");
		return 0;
	}

	int batchCount = 1;
	int effectiveHeight = computeY - searchRadius * 2 - patchRadius * 2;
	int totalEffectiveHeight = computeY;

	if (computeY < dataIn.GetHeight())
	{
		batchCount = 0;
		while (totalEffectiveHeight < dataIn.GetHeight())
		{
			batchCount++;
			totalEffectiveHeight = batchCount * effectiveHeight + 2 * searchRadius + patchRadius * 2;
		}
	}

	if (batchCount < gpuCount)
	{
		batchCount = gpuCount; //use always all GPUs!
		effectiveHeight = (dataIn.GetHeight() + 2 * searchRadius + patchRadius * 2) / batchCount;
		totalEffectiveHeight = batchCount * effectiveHeight + 2 * searchRadius + patchRadius * 2;
	}
	computeY = effectiveHeight + 2 * searchRadius + patchRadius * 2;


	threadData* datas = new threadData[gpuCount];

	for (int thread = 0; thread < gpuCount; thread++)
	{
		auto ret = __runInThread(thread, InitCudaOnDevice, thread, computeX, computeY, computeZ, dataIn.GetWidth(), dataIn.GetHeight(), dataIn.GetDepth(), patchRadius, &datas[thread]);
		ret.wait();
	}

	printf("BatchCount: %d\n", batchCount);

	for (size_t batch = 0; batch < batchCount; batch+=gpuCount)
	{
		for (size_t gpu = 0; gpu < gpuCount; gpu++)
		{
			int startX = 0; //start point of volumes
			int startZ = 0;
			int startY = 0; 
			int dataWidth = computeX; //size of volume to copy, including overlap but limited in the last part
			int dataDepth = computeZ;
			int dataHeight = effectiveHeight + 2 * searchRadius + patchRadius * 2;
			
			for (size_t rY = 0; rY < batch + gpu; rY++)
			{
				startY += effectiveHeight;
			}
			int roiY = startY + searchRadius + patchRadius;
			int roiSizeY = effectiveHeight;
			if (batch + gpu == 0)
			{
				roiY = 0;
				roiSizeY = dataHeight - searchRadius - patchRadius;
			}

			if (startY + dataHeight > dataIn.GetHeight())
			{
				dataHeight = dataIn.GetHeight() - startY;
				roiSizeY = dataHeight - searchRadius - patchRadius;
			}

			if (batchCount == 1)
			{
				roiY = 0;
				roiSizeY = dataHeight;
			}
			printf("Y Start: %d; Y Height: %d, roiY: %d, roiHeight: %d\n", startY, dataHeight, roiY, roiSizeY);
													//float* dataIn,    dataOut, startX, startY, startZ, dataWidth, dataHeight, dataDepth, roiX  , roiY,  roiZ,  roiWidth,  roiHeight,roiDepth,  searchRadius, patchRadius, sigma, filterParam, threadData& data
			
			auto ret2 = __runInThread(gpu, RunFilter, ((float*)dataIn.GetData()), dataOut, startY, dataWidth, dataHeight, dataDepth, roiY, dataWidth, roiSizeY, dataDepth, searchRadius, patchRadius, sigma, filterParam, &datas[gpu]);
			
		}
	}

	for (int gpu = gpuCount-1; gpu >= 0; gpu--)
	{
		auto res = __runInThread(gpu, ShutDown, &datas[gpu]);
		res.wait();
	}

	//save to to disk!
	EmFile::InitHeader(outputFile, dataIn.GetWidth(), dataIn.GetHeight(), dataIn.GetDepth(), 0, DT_FLOAT);
	EmFile::WriteRawData(outputFile, dataOut, (size_t)dataIn.GetWidth() * (size_t)dataIn.GetHeight() * (size_t)dataIn.GetDepth() * 4);
	return 0;
}
