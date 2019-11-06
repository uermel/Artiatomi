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

// System
#include <algorithm>
#include <fstream>

// EmTools
#include "FileIO/TIFFFile.h"
#include "Minimization/levmar.h"
#include "FileIO/FileIO.h"
#include "FileIO/MovieStack.h"
#include "CudaHelpers/CudaContext.h"
//#include "../MKLog/MKLog.h"

// Self
//#include "kernel.h"
#include "Kernels.h"
#include "CudaCrossCorrelator.h"
#include "AlignmentOptions.h"


using namespace std;
using namespace Cuda;

#define LOG(a, ...) (MKLog::Get()->Log(LL_INFO, a, __VA_ARGS__))

static void computeError(float *p, float *hx, int m, int n, void *adata)
{	
	// Pointers now 64-bits long for 64-bit systems
	ulong64 s = (ulong64)adata;
	float2* p2 = (float2*)p;
	float2** mat = new float2*[s];
	for (ulong64 i = 0; i < s; i++)
	{
		mat[i] = new float2[s];
	}

	for (ulong64 i = 0; i < s; i++)
	{
		for (ulong64 j = i + 1; j < s; j++)
		{
			mat[i][j] = make_float2(0, 0);

			for (ulong64 k = i; k < j; k++)
			{
				mat[i][j].x += p2[k].x;
				mat[i][j].y += p2[k].y;
			}
		}
	}
	ulong64 c = 0;
	for (ulong64 i = 0; i < s; i++)
	{
		for (ulong64 j = i + 1; j < s; j++)
		{
			hx[c] = mat[i][j].x;
			c++;
			hx[c] = mat[i][j].y;
			c++;
		}
	}

	for (ulong64 i = 0; i < s; i++)
	{
		delete[] mat[i];
	}
	delete[] mat;
}


int main(int argc, char* argv[])
{
	int s = sizeof(bool);



	AlignmentOptions options(argc, argv);

	//Get maximum frame count (may varry in a tilt series)
	int maxFrameCount = 0;
	int width = 0, height = 0, size = 0;
	DataType_enum datatype;

	if (options.FileList.size() < 1)
	{
		//no command line arguments...
		cout << "Usage:" << endl;
		cout << "ImageStackAlignator filename [options]" << endl;
		cout << "TODO: output options!" << endl;
		return -1;
	}
	
	for (size_t i = 0; i < options.FileList.size(); i++)
	{
		string file = options.FileList[i];

		FileType_enum ft;
		int w, h, count;
		DataType_enum dt;
		if (!MovieStack::CanReadFile(options.Path + file, ft, w, h, count, dt))
		{
			cout << "Can't read the following file:" << endl;
			cout << file << endl;
			cout << "Make sure it is a movie stack file." << endl;
			return -1;
		}

		maxFrameCount = max(maxFrameCount, count);
		if ((width != 0 && w != width) || (height != 0 && h != height))
		{
			cout << "Image dimensions are not the same in the entire series!" << endl;
			return -1;
		}
		width = w;
		height = h;
		size = std::min(width, height);
		datatype = dt;
	}

	CudaContext* ctx = CudaContext::CreateInstance(options.DeviceID);

	CUmodule mod = ctx->LoadModule("kernel.ptx");
	CudaCrossCorrelator cc(size, mod);
	CreateMaskKernel createMask(mod);
	createMask.SetComputeSize(width, height);
	SumRowKernel sumRow(mod);
	sumRow.SetComputeSize(height);


	NPPImageBase* img1b;
	NPPImageBase* img2b;

	switch (datatype)
	{
	case DT_UCHAR:
		img1b = new NPPImage_8uC1(width, height);
		img2b = new NPPImage_8uC1(width, height);
		break;
	case DT_USHORT:
		img1b = new NPPImage_16uC1(width, height);
		img2b = new NPPImage_16uC1(width, height);
		break;
	case DT_FLOAT:
		img1b = new NPPImage_32fC1(width, height);
		img2b = new NPPImage_32fC1(width, height);
		break;
	case DT_SHORT:
		img1b = new NPPImage_16sC1(width, height);
		img2b = new NPPImage_16sC1(width, height);
		break;
	default:
		cout << "ERROR: Unsupported data type!" << endl;
		exit(-1);
		break;
	}

	/*NPPImage_8uC1 img18u(width, height);
	NPPImage_8uC1 img28u(width, height);
	NPPImage_16uC1 img116u(width, height);
	NPPImage_16uC1 img216u(width, height);
	NPPImage_32fC1 img132f(width, height);
	NPPImage_32fC1 img232f(width, height);*/
	NPPImage_16uC1 imgus(width, height);

	NPPImage_32fC1 imgf(width, height);
	NPPImage_32fC1 shifted(width, height);
	NPPImage_32fC1 imgSum(width, height);
	NPPImage_32fC1 maskf(width, height);
	NPPImage_32fC1 maskSum(width, height);

	NPPImage_8uC1 mask(width, height);

	NPPImage_32fC1 img1f(size, size);
	NPPImage_32fC1 img2f(size, size);

	CudaDeviceVariable lines(height * sizeof(float));
	int bufferSize;
	int bufferSize2;
	nppiMeanStdDevGetBufferHostSize_32f_C1R(imgf.GetSizeRoi(), &bufferSize);
	nppiSumGetBufferHostSize_8u_C1R(mask.GetSizeRoi(), &bufferSize2);
	CudaDeviceVariable buffer(std::max(bufferSize, bufferSize2));
	CudaDeviceVariable mean_d(sizeof(double));
	CudaDeviceVariable std_d(sizeof(double));

	int left = width / 2 - size / 2;
	int top = height / 2 - size / 2;
	img1b->SetRoi(left, top, size, size);
	img2b->SetRoi(left, top, size, size);

	char** data = new char*[maxFrameCount];
	for (size_t i = 0; i < maxFrameCount; i++)
	{
		data[i] = new char[width * height * GetDataTypeSize(datatype)];
	}


	DataType_enum outputType = DT_USHORT;
	void* output;

	if (datatype == DT_FLOAT)
	{
		outputType = DT_FLOAT;
		output = new float[width * height];
	}
	else
	{
		output = new ushort[width * height];
	}

	MRCFile::InitHeaders(options.Output, width, height, options.PixelSize/10.0f, outputType, true);
	//MRCFile::InitHeaders("F:\\testfileMaske.mrc", width, height, options.PixelSize/10.0f, DataType_enum::DT_FLOAT, true);

	ofstream log(options.Output + ".log");

	cout << "Image dimensions: " << width << " x " << height << endl;
	log << "Image dimensions: " << width << " x " << height << endl;


	for (size_t filenr = 0; filenr < options.FileList.size(); filenr++)
	{
		string file = options.FileList[filenr];

		cout << "Reading file: " << file << endl;
		log << "File: " << file << endl;

		MovieStack ms(options.Path + file);
		uint z = ms.GetImageCount();
	
		float2** distances = new float2*[z];
		for (size_t i = 0; i < z; i++)
		{
			distances[i] = new float2[z];
		}
		for (int i = 0; i < z; i++)
		{
			for (int j = 0; j < z; j++)
			{
				distances[i][j] = make_float2(0, 0);
			}
		}

		for (size_t frame = 0; frame < z; frame++)
		{
			//Remove bad pixels:
			img1b->CopyToDevice(ms.GetData(frame));
			switch (datatype)
			{
			case DT_UCHAR:
				nppSafeCall(nppiConvert_8u32f_C1R((Npp8u*)img1b->GetDevicePointerRoi(), img1b->GetPitch(), imgf.GetPtrRoi(), imgf.GetPitch(), imgf.GetSizeRoi()));
				break;
			case DT_USHORT:
				nppSafeCall(nppiConvert_16u32f_C1R((Npp16u*)img1b->GetDevicePointerRoi(), img1b->GetPitch(), imgf.GetPtrRoi(), imgf.GetPitch(), imgf.GetSizeRoi()));
				break;
			case DT_FLOAT:
				nppSafeCall(nppiCopy_32f_C1R((Npp32f*)img1b->GetDevicePointerRoi(), img1b->GetPitch(), imgf.GetPtrRoi(), imgf.GetPitch(), imgf.GetSizeRoi()));
				break;
			case DT_SHORT:
				nppSafeCall(nppiConvert_16s32f_C1R((Npp16s*)img1b->GetDevicePointerRoi(), img1b->GetPitch(), imgf.GetPtrRoi(), imgf.GetPitch(), imgf.GetSizeRoi()));
				break;
			}
			nppSafeCall(nppiCompareC_32f_C1R(imgf.GetPtrRoi(), imgf.GetPitch(), options.DeadPixelThreshold, mask.GetPtrRoi(), mask.GetPitch(), mask.GetSizeRoi(), NppCmpOp::NPP_CMP_GREATER_EQ));

			nppSafeCall(nppiSum_8u_C1R(mask.GetPtrRoi(), mask.GetPitch(), mask.GetSizeRoi(), (Npp8u*)buffer.GetDevicePtr(), (double*)mean_d.GetDevicePtr()));
			double sum;
			mean_d.CopyDeviceToHost(&sum);


			nppSafeCall(nppiMean_StdDev_32f_C1R(imgf.GetPtrRoi(), imgf.GetPitch(), imgf.GetSizeRoi(), (Npp8u*)buffer.GetDevicePtr(), (double*)mean_d.GetDevicePtr(), (double*)std_d.GetDevicePtr()));

			double mean, std;
			mean_d.CopyDeviceToHost(&mean);
			std_d.CopyDeviceToHost(&std);

			cout << "Found " << sum / 255 << " dead pixels. Mean: " << mean << endl;
			log << "Found " << sum / 255 << " dead pixels. Mean: " << mean << endl;

			if (mean + 3 * std > options.DeadPixelThreshold)
			{
				cout << "WARNING: Threshold might be too low! Mean + 3 * STD = " << mean + 3 * std << ". Threshold = " << options.DeadPixelThreshold << endl;
				log  << "WARNING: Threshold might be too low! Mean + 3 * STD = " << mean + 3 * std << ". Threshold = " << options.DeadPixelThreshold << endl;
			}

			shifted.ResetRoi();
			imgf.ResetRoi();

			nppSafeCall(nppiSet_32f_C1MR((float)mean, imgf.GetPtrRoi(), imgf.GetPitch(), imgf.GetSizeRoi(), mask.GetPtrRoi(), mask.GetPitch()));

			NppiPoint point;
			point.x = 0;
			point.y = 0;
			nppSafeCall(nppiFilterGaussBorder_32f_C1R(imgf.GetPtrRoi(), imgf.GetPitch(), imgf.GetSizeRoi(), point, shifted.GetPtrRoi(), shifted.GetPitch(), shifted.GetSizeRoi(), 
				NppiMaskSize::NPP_MASK_SIZE_5_X_5, NppiBorderType::NPP_BORDER_REPLICATE));
			nppSafeCall(nppiCopy_32f_C1MR(shifted.GetPtrRoi(), shifted.GetPitch(), imgf.GetPtrRoi(), imgf.GetPitch(), imgf.GetSizeRoi(), mask.GetPtrRoi(), mask.GetPitch()));
			
			switch (datatype)
			{
			case DT_UCHAR:
				nppSafeCall(nppiConvert_32f8u_C1R(imgf.GetPtrRoi(), imgf.GetPitch(), (Npp8u*)img1b->GetDevicePointerRoi(), img1b->GetPitch(), imgf.GetSizeRoi(), NppRoundMode::NPP_RND_NEAR));
				break;
			case DT_USHORT:
				nppSafeCall(nppiConvert_32f16u_C1R(imgf.GetPtrRoi(), imgf.GetPitch(), (Npp16u*)img1b->GetDevicePointerRoi(), img1b->GetPitch(), imgf.GetSizeRoi(), NppRoundMode::NPP_RND_NEAR));
				break;
			case DT_FLOAT:
				nppSafeCall(nppiCopy_32f_C1R(imgf.GetPtrRoi(), imgf.GetPitch(), (Npp32f*)img1b->GetDevicePointerRoi(), img1b->GetPitch(), imgf.GetSizeRoi()));
				break;
			case DT_SHORT:
				nppSafeCall(nppiConvert_32f16s_C1R(imgf.GetPtrRoi(), imgf.GetPitch(), (Npp16s*)img1b->GetDevicePointerRoi(), img1b->GetPitch(), imgf.GetSizeRoi(), NppRoundMode::NPP_RND_NEAR));
				break;
			}

			img1b->CopyToHost(data[frame]);
		}

		for (int a = 0; a < z; a++)
		{
			img1b->CopyToDevice(data[a]);
			for (int b = a+1; b < z; b++)
			{
				img2b->CopyToDevice(data[b]);

				img1b->SetRoi(left, top, size, size);
				img2b->SetRoi(left, top, size, size);
				switch (datatype)
				{
				case DT_UCHAR:
					nppSafeCall(nppiConvert_8u32f_C1R((Npp8u*)img1b->GetDevicePointerRoi(), img1b->GetPitch(), img1f.GetPtrRoi(), img1f.GetPitch(), img1f.GetSizeRoi()));
					nppSafeCall(nppiConvert_8u32f_C1R((Npp8u*)img2b->GetDevicePointerRoi(), img2b->GetPitch(), img2f.GetPtrRoi(), img2f.GetPitch(), img2f.GetSizeRoi()));
					break;
				case DT_USHORT:
					nppSafeCall(nppiConvert_16u32f_C1R((Npp16u*)img1b->GetDevicePointerRoi(), img1b->GetPitch(), img1f.GetPtrRoi(), img1f.GetPitch(), img1f.GetSizeRoi()));
					nppSafeCall(nppiConvert_16u32f_C1R((Npp16u*)img2b->GetDevicePointerRoi(), img2b->GetPitch(), img2f.GetPtrRoi(), img2f.GetPitch(), img2f.GetSizeRoi()));
					break;
				case DT_FLOAT:
					nppSafeCall(nppiCopy_32f_C1R((Npp32f*)img1b->GetDevicePointerRoi(), img1b->GetPitch(), img1f.GetPtrRoi(), img1f.GetPitch(), img1f.GetSizeRoi()));
					nppSafeCall(nppiCopy_32f_C1R((Npp32f*)img2b->GetDevicePointerRoi(), img2b->GetPitch(), img2f.GetPtrRoi(), img2f.GetPitch(), img2f.GetSizeRoi()));
					break;
				case DT_SHORT:
					nppSafeCall(nppiConvert_16s32f_C1R((Npp16s*)img1b->GetDevicePointerRoi(), img1b->GetPitch(), img1f.GetPtrRoi(), img1f.GetPitch(), img1f.GetSizeRoi()));
					nppSafeCall(nppiConvert_16s32f_C1R((Npp16s*)img2b->GetDevicePointerRoi(), img2b->GetPitch(), img2f.GetPtrRoi(), img2f.GetPitch(), img2f.GetSizeRoi()));
					break;
				}
				//nppSafeCall(nppiConvert_8u32f_C1R(img1b.GetPtrRoi(), img1b.GetPitch(), img1f.GetPtrRoi(), img1f.GetPitch(), img1f.GetSizeRoi()));
				//nppSafeCall(nppiConvert_8u32f_C1R(img2b.GetPtrRoi(), img2b.GetPitch(), img2f.GetPtrRoi(), img2f.GetPitch(), img2f.GetSizeRoi()));
				img1b->ResetRoi();
				img2b->ResetRoi();

				distances[a][b] = make_float2(0, 0);
				if (!options.AssumeZeroShift)
				{
					distances[a][b] = cc.GetShift(img1f, img2f, options.MaxShift, options.LP, options.HP, options.LPS, options.HPS);
				}
				cout << a << " : " << b << " = " << (int)distances[a][b].x << "; " << (int)distances[a][b].y << endl;
				log << a << " : " << b << " = " << (int)distances[a][b].x << "; " << (int)distances[a][b].y << endl;
			}
			cout << endl;
			log << endl;
		}

		//Minimize shift matrix

		float info[10];
		float opt[5];
		int res;
		float* start;
		{
			int c = 0;
			int nUnkowns = (z - 1) * z / 2;
			start = new float[2 * (z - 1)];
			
			for (int i = 0; i < z - 1; i++)
			{
				start[c] = distances[i][i + 1].x - 0.001f;
				c++;
				start[c] = distances[i][i + 1].y - 0.001f;
				c++;
			}
			float* measurements = new float[nUnkowns * 2];// (float*)distances;
			c = 0;
			for (int i = 0; i < z; i++)
			{
				for (int j = i + 1; j < z; j++)
				{
					measurements[c] = distances[i][j].x;
					c++;
					measurements[c] = distances[i][j].y;
					c++;
				}
			}
			opt[0] = 1E-03f;
			opt[1] = opt[2] = opt[3] = 1E-17f;
			opt[4] = 1E-06f;

			res = LEVMAR_DIF(&computeError, start, measurements, (z - 1) * 2, nUnkowns * 2, 1000, opt, info, NULL, NULL, (void*)(ulong64)z);
			
			cout << "finished: Reason   = " << info[6] << ", Iterations: " << info[5] << endl;
			cout << "finished: FunEvals = " << info[7] << ", e: " << info[1] << ", e_init: " << info[0] << endl;
			log << "finished: Reason   = " << info[6] << ", Iterations: " << info[5] << endl;
			log << "finished: FunEvals = " << info[7] << ", e: " << info[1] << ", e_init: " << info[0] << endl;
		}
		cout << endl;
		log << endl;

		float2* totalShifts = new float2[z];
		int minIndex = -1;
		//Reduce total shift:
		{
			float2* shifts = (float2*)start;

			float2* shiftLengthMin = new float2[z-1];

			for (int i = 0; i < z - 1; i++)
			{
				float2 a = make_float2(0, 0);
				for (int j = 0; j < i; j++)
				{
					a.x -= shifts[j].x;
					a.y -= shifts[j].y;
				}

				float2 b = make_float2(0, 0);
				for (int j = i; j < z - 1; j++)
				{
					b.x += shifts[j].x;
					b.y += shifts[j].y;
				}

				shiftLengthMin[i].x = a.x + b.x;
				shiftLengthMin[i].y = a.y + b.y;
			}

			minIndex = -1;
			float minShift = 1000000000.0f;
			for (int i = 0; i < z - 1; i++)
			{
				float2 a = shiftLengthMin[i];
				float d = (float)sqrt(a.x * a.x + a.y * a.y);
				if (d < minShift)
				{
					minShift = d;
					minIndex = i;
				}
			}
			cout << "Min shift-index: " << minIndex << endl;
			log << "Min shift-index: " << minIndex << endl;


			for (int i = 0; i < z; i++)
			{
				float2 a = make_float2(0, 0);
				for (int j = i; j < minIndex; j++)
				{
					a.x -= shifts[j].x;
					a.y -= shifts[j].y;
				}

				for (int j = minIndex; j < i; j++)
				{
					a.x += shifts[j].x;
					a.y += shifts[j].y;
				}
				totalShifts[i] = a;
				cout << "Index: " << i << " = " << a.x << "; " << a.y << endl;
				log << "Index: " << i << " = " << a.x << "; " << a.y << endl;
			}

			delete[] shiftLengthMin;
		}



		//Cross validate the shifts:
		{

			for (size_t iteration = 0; iteration < 10; iteration++)
			{
				cout << endl << "Iteration: " << iteration + 1 << endl;
				log  << endl << "Iteration: " << iteration + 1 << endl;
				float diff = 0;

				if (options.GroupStack <= 1)
				{
					cout << "Not Grouping " << options.GroupStack << "images!" << endl;
					for (size_t i = 0; i < z; i++)
					{

						//apply Shifts and sum up:
						{
							nppSafeCall(nppiSet_32f_C1R(0, imgSum.GetPtrRoi(), imgSum.GetPitch(), imgSum.GetSizeRoi()));
							nppSafeCall(nppiSet_32f_C1R(0, maskSum.GetPtrRoi(), maskSum.GetPitch(), maskSum.GetSizeRoi()));
							nppSafeCall(nppiSet_8u_C1R(0, mask.GetPtrRoi(), mask.GetPitch(), mask.GetSizeRoi()));

							for (size_t j = 0; j < z; j++)
							{
								if (i != j)
								{
									double coeff[2][3] = { { 1, 0, std::round(-totalShifts[j].x) },{ 0, 1, std::round(-totalShifts[j].y) } };

									img1b->CopyToDevice(data[j]);
									switch (datatype)
									{
									case DT_UCHAR:
										nppSafeCall(nppiConvert_8u32f_C1R((Npp8u*)img1b->GetDevicePointerRoi(), img1b->GetPitch(), imgf.GetPtrRoi(), imgf.GetPitch(), imgf.GetSizeRoi()));
										break;
									case DT_USHORT:
										nppSafeCall(nppiConvert_16u32f_C1R((Npp16u*)img1b->GetDevicePointerRoi(), img1b->GetPitch(), imgf.GetPtrRoi(), imgf.GetPitch(), imgf.GetSizeRoi()));
										break;
									case DT_FLOAT:
										nppSafeCall(nppiCopy_32f_C1R((Npp32f*)img1b->GetDevicePointerRoi(), img1b->GetPitch(), imgf.GetPtrRoi(), imgf.GetPitch(), imgf.GetSizeRoi()));
										break;
									case DT_SHORT:
										nppSafeCall(nppiConvert_16s32f_C1R((Npp16s*)img1b->GetDevicePointerRoi(), img1b->GetPitch(), imgf.GetPtrRoi(), imgf.GetPitch(), imgf.GetSizeRoi()));
										break;
									}
									//nppSafeCall(nppiConvert_8u32f_C1R(img1b.GetPtrRoi(), img1b.GetPitch(), imgf.GetPtrRoi(), imgf.GetPitch(), imgf.GetSizeRoi()));
									sumRow(imgf, lines);
									createMask(mask, lines);
									nppSafeCall(nppiSet_32f_C1R(0, maskf.GetPtrRoi(), maskf.GetPitch(), maskf.GetSizeRoi()));
									nppSafeCall(nppiSet_32f_C1MR(1, maskf.GetPtrRoi(), maskf.GetPitch(), maskf.GetSizeRoi(), mask.GetPtrRoi(), mask.GetPitch()));
									NppiRect roi;
									roi.width = imgf.GetWidth();
									roi.height = imgf.GetHeight();
									roi.x = 0;
									roi.y = 0;

									nppSafeCall(nppiSet_32f_C1R(0, shifted.GetPtrRoi(), shifted.GetPitch(), shifted.GetSizeRoi()));
									nppSafeCall(nppiWarpAffine_32f_C1R(imgf.GetPtrRoi(), imgf.GetSizeRoi(), imgf.GetPitch(), roi, shifted.GetPtrRoi(), shifted.GetPitch(), roi, coeff, NPPI_INTER_NN));

									nppSafeCall(nppiAdd_32f_C1IR(shifted.GetPtrRoi(), shifted.GetPitch(), imgSum.GetPtrRoi(), imgSum.GetPitch(), imgSum.GetSizeRoi()));

									nppSafeCall(nppiSet_32f_C1R(0, shifted.GetPtrRoi(), shifted.GetPitch(), shifted.GetSizeRoi()));
									nppSafeCall(nppiWarpAffine_32f_C1R(maskf.GetPtrRoi(), maskf.GetSizeRoi(), maskf.GetPitch(), roi, shifted.GetPtrRoi(), shifted.GetPitch(), roi, coeff, NPPI_INTER_NN));

									nppSafeCall(nppiAdd_32f_C1IR(shifted.GetPtrRoi(), shifted.GetPitch(), maskSum.GetPtrRoi(), maskSum.GetPitch(), maskSum.GetSizeRoi()));
								}
							}

							nppSafeCall(nppiMulC_32f_C1IR(z - 1, imgSum.GetPtrRoi(), imgSum.GetPitch(), imgSum.GetSizeRoi()));
							nppSafeCall(nppiDiv_32f_C1IR(maskSum.GetPtrRoi(), maskSum.GetPitch(), imgSum.GetPtrRoi(), imgSum.GetPitch(), imgSum.GetSizeRoi()));

							nppSafeCall(nppiCompareC_32f_C1R(maskSum.GetPtrRoi(), maskSum.GetPitch(), 1, mask.GetPtrRoi(), mask.GetPitch(), mask.GetSizeRoi(), NppCmpOp::NPP_CMP_LESS));
							nppSafeCall(nppiSet_32f_C1MR(0, imgSum.GetPtrRoi(), imgSum.GetPitch(), imgSum.GetSizeRoi(), mask.GetPtrRoi(), mask.GetPitch()));
						}




						img1b->CopyToDevice(data[i]);

						img1b->SetRoi(left, top, size, size);
						imgSum.SetRoi(left, top, size, size);
						switch (datatype)
						{
						case DT_UCHAR:
							nppSafeCall(nppiConvert_8u32f_C1R((Npp8u*)img1b->GetDevicePointerRoi(), img1b->GetPitch(), img1f.GetPtrRoi(), img1f.GetPitch(), img1f.GetSizeRoi()));
							break;
						case DT_USHORT:
							nppSafeCall(nppiConvert_16u32f_C1R((Npp16u*)img1b->GetDevicePointerRoi(), img1b->GetPitch(), img1f.GetPtrRoi(), img1f.GetPitch(), img1f.GetSizeRoi()));
							break;
						case DT_FLOAT:
							nppSafeCall(nppiCopy_32f_C1R((Npp32f*)img1b->GetDevicePointerRoi(), img1b->GetPitch(), img1f.GetPtrRoi(), img1f.GetPitch(), img1f.GetSizeRoi()));
							break;
						case DT_SHORT:
							nppSafeCall(nppiConvert_16s32f_C1R((Npp16s*)img1b->GetDevicePointerRoi(), img1b->GetPitch(), img1f.GetPtrRoi(), img1f.GetPitch(), img1f.GetSizeRoi()));
							break;
						}
						//nppSafeCall(nppiConvert_8u32f_C1R(img1b.GetPtrRoi(), img1b.GetPitch(), img1f.GetPtrRoi(), img1f.GetPitch(), img1f.GetSizeRoi()));
						nppSafeCall(nppiCopy_32f_C1R(imgSum.GetPtrRoi(), imgSum.GetPitch(), img2f.GetPtrRoi(), img2f.GetPitch(), img2f.GetSizeRoi()));
						img1b->ResetRoi();
						imgSum.ResetRoi();

						float2 shift = cc.GetShift(img1f, img2f, options.MaxShift, options.LP, options.HP, options.LPS, options.HPS);

						if (shift.x != 0)
							shift.x *= -1;
						if (shift.y != 0)
							shift.y *= -1;

						cout << "Actual shift: " << std::round(totalShifts[i].x) << "; " << std::round(totalShifts[i].y) << endl;
						cout << "Cross check : " << shift.x << "; " << shift.y << endl << endl;
						log << "Actual shift: " << std::round(totalShifts[i].x) << "; " << std::round(totalShifts[i].y) << endl;
						log << "Cross check : " << shift.x << "; " << shift.y << endl << endl;

						diff += fabsf(shift.x - std::round(totalShifts[i].x)) + fabsf(shift.y - std::round(totalShifts[i].y));

						totalShifts[i].x = shift.x;
						totalShifts[i].y = shift.y;
					}
				}
				else
				{
					cout << "Grouping " << options.GroupStack << "images!" << endl;
					//Group stack:
					for (size_t i = 0; i < z; i += options.GroupStack)
					{

						//apply Shifts and sum up:
						{
							nppSafeCall(nppiSet_32f_C1R(0, imgSum.GetPtrRoi(), imgSum.GetPitch(), imgSum.GetSizeRoi()));
							nppSafeCall(nppiSet_32f_C1R(0, maskSum.GetPtrRoi(), maskSum.GetPitch(), maskSum.GetSizeRoi()));
							nppSafeCall(nppiSet_8u_C1R(0, mask.GetPtrRoi(), mask.GetPitch(), mask.GetSizeRoi()));

							for (size_t j = 0; j < z; j += options.GroupStack)
							{
								if (i != j)
								{
									double coeff[2][3] = { { 1, 0, std::round(-totalShifts[j].x) },{ 0, 1, std::round(-totalShifts[j].y) } };

									img1b->CopyToDevice(data[j]);
									switch (datatype)
									{
									case DT_UCHAR:
										nppSafeCall(nppiConvert_8u32f_C1R((Npp8u*)img1b->GetDevicePointerRoi(), img1b->GetPitch(), imgf.GetPtrRoi(), imgf.GetPitch(), imgf.GetSizeRoi()));
										break;
									case DT_USHORT:
										nppSafeCall(nppiConvert_16u32f_C1R((Npp16u*)img1b->GetDevicePointerRoi(), img1b->GetPitch(), imgf.GetPtrRoi(), imgf.GetPitch(), imgf.GetSizeRoi()));
										break;
									case DT_FLOAT:
										nppSafeCall(nppiCopy_32f_C1R((Npp32f*)img1b->GetDevicePointerRoi(), img1b->GetPitch(), imgf.GetPtrRoi(), imgf.GetPitch(), imgf.GetSizeRoi()));
										break;
									case DT_SHORT:
										nppSafeCall(nppiConvert_16s32f_C1R((Npp16s*)img1b->GetDevicePointerRoi(), img1b->GetPitch(), imgf.GetPtrRoi(), imgf.GetPitch(), imgf.GetSizeRoi()));
										break;
									}
									//nppSafeCall(nppiConvert_8u32f_C1R(img1b.GetPtrRoi(), img1b.GetPitch(), imgf.GetPtrRoi(), imgf.GetPitch(), imgf.GetSizeRoi()));
									sumRow(imgf, lines);
									createMask(mask, lines);
									nppSafeCall(nppiSet_32f_C1R(0, maskf.GetPtrRoi(), maskf.GetPitch(), maskf.GetSizeRoi()));
									nppSafeCall(nppiSet_32f_C1MR(1, maskf.GetPtrRoi(), maskf.GetPitch(), maskf.GetSizeRoi(), mask.GetPtrRoi(), mask.GetPitch()));
									NppiRect roi;
									roi.width = imgf.GetWidth();
									roi.height = imgf.GetHeight();
									roi.x = 0;
									roi.y = 0;

									nppSafeCall(nppiSet_32f_C1R(0, shifted.GetPtrRoi(), shifted.GetPitch(), shifted.GetSizeRoi()));
									nppSafeCall(nppiWarpAffine_32f_C1R(imgf.GetPtrRoi(), imgf.GetSizeRoi(), imgf.GetPitch(), roi, shifted.GetPtrRoi(), shifted.GetPitch(), roi, coeff, NPPI_INTER_NN));

									nppSafeCall(nppiAdd_32f_C1IR(shifted.GetPtrRoi(), shifted.GetPitch(), imgSum.GetPtrRoi(), imgSum.GetPitch(), imgSum.GetSizeRoi()));

									nppSafeCall(nppiSet_32f_C1R(0, shifted.GetPtrRoi(), shifted.GetPitch(), shifted.GetSizeRoi()));
									nppSafeCall(nppiWarpAffine_32f_C1R(maskf.GetPtrRoi(), maskf.GetSizeRoi(), maskf.GetPitch(), roi, shifted.GetPtrRoi(), shifted.GetPitch(), roi, coeff, NPPI_INTER_NN));

									nppSafeCall(nppiAdd_32f_C1IR(shifted.GetPtrRoi(), shifted.GetPitch(), maskSum.GetPtrRoi(), maskSum.GetPitch(), maskSum.GetSizeRoi()));
								}
							}

							nppSafeCall(nppiMulC_32f_C1IR(z - 1, imgSum.GetPtrRoi(), imgSum.GetPitch(), imgSum.GetSizeRoi()));
							nppSafeCall(nppiDiv_32f_C1IR(maskSum.GetPtrRoi(), maskSum.GetPitch(), imgSum.GetPtrRoi(), imgSum.GetPitch(), imgSum.GetSizeRoi()));

							nppSafeCall(nppiCompareC_32f_C1R(maskSum.GetPtrRoi(), maskSum.GetPitch(), 1, mask.GetPtrRoi(), mask.GetPitch(), mask.GetSizeRoi(), NppCmpOp::NPP_CMP_LESS));
							nppSafeCall(nppiSet_32f_C1MR(0, imgSum.GetPtrRoi(), imgSum.GetPitch(), imgSum.GetSizeRoi(), mask.GetPtrRoi(), mask.GetPitch()));
						}

						imgSum.SetRoi(left, top, size, size);
						nppSafeCall(nppiCopy_32f_C1R((Npp32f*)imgSum.GetPtrRoi(), imgSum.GetPitch(), img2f.GetPtrRoi(), img2f.GetPitch(), img2f.GetSizeRoi()));
						img1b->ResetRoi();
						imgSum.ResetRoi();
						nppSafeCall(nppiSet_32f_C1R(0, (Npp32f*)imgSum.GetPtr(), imgSum.GetPitch(), imgSum.GetSize()));

						for (size_t k = 0; k < options.GroupStack; k++)
						{
							if (i + k >= z)
								break;

							img1b->ResetRoi();
							img1b->CopyToDevice(data[i+k]);

							imgSum.SetRoi(left, top, size, size);
							img1b->SetRoi(left, top, size, size);
							//imgSum.SetRoi(left, top, size, size);
							switch (datatype)
							{
							case DT_UCHAR:
								nppSafeCall(nppiConvert_8u32f_C1R((Npp8u*)img1b->GetDevicePointerRoi(), img1b->GetPitch(), imgSum.GetPtrRoi(), imgSum.GetPitch(), imgSum.GetSizeRoi()));
								break;
							case DT_USHORT:
								nppSafeCall(nppiConvert_16u32f_C1R((Npp16u*)img1b->GetDevicePointerRoi(), img1b->GetPitch(), imgSum.GetPtrRoi(), imgSum.GetPitch(), imgSum.GetSizeRoi()));
								break;
							case DT_FLOAT:
								nppSafeCall(nppiCopy_32f_C1R((Npp32f*)img1b->GetDevicePointerRoi(), img1b->GetPitch(), imgSum.GetPtrRoi(), imgSum.GetPitch(), imgSum.GetSizeRoi()));
								break;
							case DT_SHORT:
								nppSafeCall(nppiConvert_16s32f_C1R((Npp16s*)img1b->GetDevicePointerRoi(), img1b->GetPitch(), imgSum.GetPtrRoi(), imgSum.GetPitch(), imgSum.GetSizeRoi()));
								break;
							}
							nppSafeCall(nppiAdd_32f_C1IR((Npp32f*)imgSum.GetPtrRoi(), imgSum.GetPitch(), (Npp32f*)img1f.GetPtrRoi(), img1f.GetPitch(), img1f.GetSizeRoi()));
							//nppSafeCall(nppiConvert_8u32f_C1R(img1b.GetPtrRoi(), img1b.GetPitch(), img1f.GetPtrRoi(), img1f.GetPitch(), img1f.GetSizeRoi()));
							//nppiCopy_32f_C1R(imgSum.GetPtrRoi(), imgSum.GetPitch(), img2f.GetPtrRoi(), img2f.GetPitch(), img2f.GetSizeRoi()));
							img1b->ResetRoi();
							imgSum.ResetRoi();
						}

						float2 shift = cc.GetShift(img1f, img2f, options.MaxShift, options.LP, options.HP, options.LPS, options.HPS);

						if (shift.x != 0)
							shift.x *= -1;
						if (shift.y != 0)
							shift.y *= -1;

						cout << "Actual shift: " << std::round(totalShifts[i].x) << "; " << std::round(totalShifts[i].y) << endl;
						cout << "Cross check : " << shift.x << "; " << shift.y << endl << endl;
						log << "Actual shift: " << std::round(totalShifts[i].x) << "; " << std::round(totalShifts[i].y) << endl;
						log << "Cross check : " << shift.x << "; " << shift.y << endl << endl;

						diff += fabsf(shift.x - std::round(totalShifts[i].x)) + fabsf(shift.y - std::round(totalShifts[i].y));
						for (size_t k = 0; k < options.GroupStack; k++)
						{
							if (i + k >= z)
								break;
							totalShifts[i+k].x = shift.x;
							totalShifts[i+k].y = shift.y;
						}
					}
				}
				if (diff == 0)
				{
					cout << "No more optimization: skip further iterations..." << endl;
					log  << "No more optimization: skip further iterations..." << endl;
					break;
				}
				else
				{
					cout << "Summed shift differences: " << diff << endl;
					log << "Summed shift differences: " << diff << endl;
				}
			}
		}

		cout << "Finally applied shifts: " << endl;
		log << "Finally applied shifts: " << endl;

		//Again: subtract shift from min shift index from all shifts:
		float2 minShift = totalShifts[minIndex];
		for (size_t i = 0; i < z; i++)
		{
			totalShifts[i].x -= minShift.x;
			totalShifts[i].y -= minShift.y;


			cout << "Frame " << i << ": " << std::round(totalShifts[i].x) << "; " << std::round(totalShifts[i].y) << endl;
			log << "Frame " << i << ": " << std::round(totalShifts[i].x) << "; " << std::round(totalShifts[i].y) << endl;
		}



		cout << endl << "--------------------------------------------------------------------------------" << endl;
		log << endl << "--------------------------------------------------------------------------------" << endl;

		//apply Shifts and sum up:
		{
			nppSafeCall(nppiSet_32f_C1R(0, imgSum.GetPtrRoi(), imgSum.GetPitch(), imgSum.GetSizeRoi()));
			nppSafeCall(nppiSet_32f_C1R(0, maskSum.GetPtrRoi(), maskSum.GetPitch(), maskSum.GetSizeRoi()));
			nppSafeCall(nppiSet_8u_C1R(0, mask.GetPtrRoi(), mask.GetPitch(), mask.GetSizeRoi()));

			for (size_t i = 0; i < z; i++)
			{
				double coeff[2][3] = { { 1, 0, std::round(-totalShifts[i].x) },{ 0, 1, std::round(-totalShifts[i].y) } };
				
				img1b->CopyToDevice(data[i]);
				switch (datatype)
				{
				case DT_UCHAR:
					nppSafeCall(nppiConvert_8u32f_C1R((Npp8u*)img1b->GetDevicePointerRoi(), img1b->GetPitch(), imgf.GetPtrRoi(), imgf.GetPitch(), imgf.GetSizeRoi()));
					break;
				case DT_USHORT:
					nppSafeCall(nppiConvert_16u32f_C1R((Npp16u*)img1b->GetDevicePointerRoi(), img1b->GetPitch(), imgf.GetPtrRoi(), imgf.GetPitch(), imgf.GetSizeRoi()));
					break;
				case DT_FLOAT:
					nppSafeCall(nppiCopy_32f_C1R((Npp32f*)img1b->GetDevicePointerRoi(), img1b->GetPitch(), imgf.GetPtrRoi(), imgf.GetPitch(), imgf.GetSizeRoi()));
					break;
				case DT_SHORT:
					nppSafeCall(nppiConvert_16s32f_C1R((Npp16s*)img1b->GetDevicePointerRoi(), img1b->GetPitch(), imgf.GetPtrRoi(), imgf.GetPitch(), imgf.GetSizeRoi()));
					break;
				}
				//nppSafeCall(nppiConvert_8u32f_C1R(img1b.GetPtrRoi(), img1b.GetPitch(), imgf.GetPtrRoi(), imgf.GetPitch(), imgf.GetSizeRoi()));
				sumRow(imgf, lines);
				createMask(mask, lines);
				nppSafeCall(nppiSet_32f_C1R(0, maskf.GetPtrRoi(), maskf.GetPitch(), maskf.GetSizeRoi()));
				nppSafeCall(nppiSet_32f_C1MR(1, maskf.GetPtrRoi(), maskf.GetPitch(), maskf.GetSizeRoi(), mask.GetPtrRoi(), mask.GetPitch()));
				NppiRect roi;
				roi.width = imgf.GetWidth();
				roi.height = imgf.GetHeight();
				roi.x = 0;
				roi.y = 0;

				nppSafeCall(nppiSet_32f_C1R(0, shifted.GetPtrRoi(), shifted.GetPitch(), shifted.GetSizeRoi()));
				nppSafeCall(nppiWarpAffine_32f_C1R(imgf.GetPtrRoi(), imgf.GetSizeRoi(), imgf.GetPitch(), roi, shifted.GetPtrRoi(), shifted.GetPitch(), roi, coeff, NPPI_INTER_NN));

				nppSafeCall(nppiAdd_32f_C1IR(shifted.GetPtrRoi(), shifted.GetPitch(), imgSum.GetPtrRoi(), imgSum.GetPitch(), imgSum.GetSizeRoi()));

				nppSafeCall(nppiSet_32f_C1R(0, shifted.GetPtrRoi(), shifted.GetPitch(), shifted.GetSizeRoi()));
				nppSafeCall(nppiWarpAffine_32f_C1R(maskf.GetPtrRoi(), maskf.GetSizeRoi(), maskf.GetPitch(), roi, shifted.GetPtrRoi(), shifted.GetPitch(), roi, coeff, NPPI_INTER_NN));

				nppSafeCall(nppiAdd_32f_C1IR(shifted.GetPtrRoi(), shifted.GetPitch(), maskSum.GetPtrRoi(), maskSum.GetPitch(), maskSum.GetSizeRoi()));
			}

			nppSafeCall(nppiMulC_32f_C1IR(10*z, imgSum.GetPtrRoi(), imgSum.GetPitch(), imgSum.GetSizeRoi()));
			nppSafeCall(nppiDiv_32f_C1IR(maskSum.GetPtrRoi(), maskSum.GetPitch(), imgSum.GetPtrRoi(), imgSum.GetPitch(), imgSum.GetSizeRoi()));
			
			nppSafeCall(nppiCompareC_32f_C1R(maskSum.GetPtrRoi(), maskSum.GetPitch(), 1, mask.GetPtrRoi(), mask.GetPitch(), mask.GetSizeRoi(), NppCmpOp::NPP_CMP_LESS));
			nppSafeCall(nppiSet_32f_C1MR(0, imgSum.GetPtrRoi(), imgSum.GetPitch(), imgSum.GetSizeRoi(), mask.GetPtrRoi(), mask.GetPitch()));
		}

		////Cross validate the shifts:
		//{

		//	for (size_t i = 0; i < z; i++)
		//	{
		//		img1b.CopyToDevice(data[i]);
		//		
		//		img1b.SetRoi(left, top, size, size);
		//		imgSum.SetRoi(left, top, size, size);
		//		nppSafeCall(nppiConvert_8u32f_C1R(img1b.GetPtrRoi(), img1b.GetPitch(), img1f.GetPtrRoi(), img1f.GetPitch(), img1f.GetSizeRoi()));
		//		nppSafeCall(nppiCopy_32f_C1R(imgSum.GetPtrRoi(), imgSum.GetPitch(), img2f.GetPtrRoi(), img2f.GetPitch(), img2f.GetSizeRoi()));
		//		img1b.ResetRoi();
		//		imgSum.ResetRoi();

		//		float2 shift = cc.GetShift(img1f, img2f, options.MaxShift, options.LP, options.HP, options.LPS, options.HPS);

		//		cout << "Actual shift: " << std::round(totalShifts[i].x) << "; " << std::round(totalShifts[i].y) << endl;
		//		cout << "Cross check : " << -shift.x << "; " << -shift.y << endl;
		//	}
		//}


		if (outputType == DataType_enum::DT_FLOAT)
		{
			//Save to file:
			imgSum.CopyToHost(output);
			MRCFile::AddPlaneToMRCFile(options.Output, DataType_enum::DT_FLOAT, output, options.TiltAngles[filenr]);
		}
		else
		{ 
			//convert to unisgned short
			//Save to file:
			nppSafeCall(nppiConvert_32f16u_C1R(imgSum.GetPtrRoi(), imgSum.GetPitch(), imgus.GetPtrRoi(), imgus.GetPitch(), imgSum.GetSizeRoi(), NppRoundMode::NPP_RND_NEAR));
			imgus.CopyToHost(output);

			MRCFile::AddPlaneToMRCFile(options.Output, DataType_enum::DT_USHORT, output, options.TiltAngles[filenr]);
			//maskSum.CopyToHost(output);
			//MRCFile::AddPlaneToMRCFile("F:\\testfileMaske.mrc", DataType_enum::DT_FLOAT, output, options.TiltAngles[filenr]);
		}
		for (size_t i = 0; i < z; i++)
		{
			delete[] distances[i];
		}
		delete[] distances;
		delete[] totalShifts;
	}

	// Free output
	FileReader::DeleteData(output, outputType);

	CudaContext::DestroyInstance(ctx);
	
}