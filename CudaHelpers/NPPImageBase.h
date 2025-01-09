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


#ifndef NPPIMAGEBASE_H
#define NPPIMAGEBASE_H

#include <cuda.h>
#include <npp.h>
#include "CudaVariables.h"

class NPPImageBase
{
protected:
	/// <summary>
	/// Base pointer to image data.
	/// </summary>
	CUdeviceptr _devPtr;
	/// <summary>
	/// Base pointer moved to actual ROI.
	/// </summary>
	CUdeviceptr _devPtrRoi;
	/// <summary>
	/// Size of the entire image.
	/// </summary>
	NppiSize _sizeOriginal;
	/// <summary>
	/// Size of the actual ROI.
	/// </summary>
	NppiSize _sizeRoi;
	/// <summary>
	/// First pixel in the ROI.
	/// </summary>
	NppiPoint _pointRoi;
	/// <summary>
	/// Width of one image line + alignment bytes.
	/// </summary>
	int _pitch;
	/// <summary>
	/// Number of color channels in image.
	/// </summary>
	int _channels;
	/// <summary>
	/// Type size in bytes of one pixel in one channel.
	/// </summary>
	int _typeSize;
	/// <summary>
	/// 
	/// </summary>
	bool _isOwner;

public:
	virtual ~NPPImageBase();
	NPPImageBase(CUdeviceptr devPtr, int aWidth, int aHeight, int aTypeSize, int aPitch, int aChannels, bool aIsOwner);

	/// <summary>
	/// Copy from Host to device memory
	/// </summary>
	/// <param name="hostSrc">Source</param>
	/// <param name="stride">Size of one image line in bytes with padding</param>
	void CopyToDevice(void* hostSrc, size_t stride);

	/// <summary>
	/// Copy from device to device memory
	/// </summary>
	/// <param name="deviceSrc">Source</param>
	void CopyToDevice(Cuda::CudaPitchedDeviceVariable& deviceSrc);

	/// <summary>
	/// Copy from device to device memory
	/// </summary>
	/// <param name="deviceSrc">Source</param>
	void CopyToDevice(NPPImageBase& deviceSrc);

	/// <summary>
	/// Copy from device to device memory
	/// </summary>
	/// <param name="deviceSrc">Source</param>
	void CopyToDevice(Cuda::CudaDeviceVariable& deviceSrc);

	/// <summary>
	/// Copy from device to device memory
	/// </summary>
	/// <param name="deviceSrc">Source</param>
	void CopyToDevice(CUdeviceptr deviceSrc);

	/// <summary>
	/// Copy from device to device memory
	/// </summary>
	/// <param name="deviceSrc">Source</param>
	/// <param name="pitch">Pitch of deviceSrc</param>
	void CopyToDevice(CUdeviceptr deviceSrc, size_t pitch);

	/// <summary>
	/// Copy data from device to host memory
	/// </summary>
	/// <param name="hostDest">void* to destination in host memory</param>
	/// <param name="stride">Size of one image line in bytes with padding</param>
	///// <param name="width">Width in bytes</param>
	///// <param name="height">Height in elements</param>
	void CopyToHost(void* hostDest, size_t stride);

	/// <summary>
	/// Copy from Host to device memory
	/// </summary>
	/// <param name="hostSrc">Source</param>
	void CopyToDevice(void* hostSrc);

	/// <summary>
	/// Copy data from device to host memory
	/// </summary>
	/// <param name="hostDest">void* to destination in host memory</param>
	void CopyToHost(void* hostDest);


	///// <summary>
	///// Copy from Host to device memory
	///// </summary>
	///// <param name="hostSrc">Source</param>
	///// <param name="stride">Size of one image line in bytes with padding</param>
	///// <param name="roi">ROI of source image</param>
	/////// <param name="height">Height in elements</param>
	//void CopyToDeviceRoi(void* hostSrc, size_t stride, NppiRect roi);

	///// <summary>
	///// Copy from device to device memory
	///// </summary>
	///// <param name="deviceSrc">Source</param>
	///// <param name="roi">ROI of source image</param>
	//void CopyToDeviceRoi(Cuda::CudaPitchedDeviceVariable& deviceSrc, NppiRect roi);

	///// <summary>
	///// Copy from device to device memory
	///// </summary>
	///// <param name="deviceSrc">Source</param>
	//void CopyToDeviceRoi(NPPImageBase& deviceSrc);

	///// <summary>
	///// Copy from device to device memory
	///// </summary>
	///// <param name="deviceSrc">Source</param>
	///// <param name="roi">ROI of source image</param>
	//void CopyToDeviceRoi(Cuda::CudaDeviceVariable& deviceSrc, NppiRect roi);

	///// <summary>
	///// Copy from device to device memory
	///// </summary>
	///// <param name="deviceSrc">Source</param>
	///// <param name="roi">ROI of source image</param>
	//void CopyToDeviceRoi(CUdeviceptr deviceSrc, NppiRect roi);

	///// <summary>
	///// Copy from device to device memory
	///// </summary>
	///// <param name="deviceSrc">Source</param>
	///// <param name="pitch">Pitch of deviceSrc</param>
	///// <param name="roi">ROI of source image</param>
	//void CopyToDeviceRoi(CUdeviceptr deviceSrc, size_t pitch, NppiRect roi);

	///// <summary>
	///// Copy data from device to host memory
	///// </summary>
	///// <param name="hostDest">void* to destination in host memory</param>
	///// <param name="stride">Size of one image line in bytes with padding</param>
	///// <param name="roi">ROI of destination image</param>
	/////// <param name="width">Width in bytes</param>
	/////// <param name="height">Height in elements</param>
	//void CopyToHostRoi(void* hostDest, size_t stride, NppiRect roi);

	///// <summary>
	///// Copy from Host to device memory
	///// </summary>
	///// <param name="hostSrc">Source</param>
	///// <param name="roi">ROI of source image</param>
	//void CopyToDeviceRoi(void* hostSrc, NppiRect roi);

	///// <summary>
	///// Copy data from device to host memory
	///// </summary>
	///// <param name="hostDest">void* to destination in host memory</param>
	///// <param name="roi">ROI of destination image</param>
	//void CopyToHostRoi(void* hostDest, NppiRect roi);


		/// <summary>
		/// Size of the entire image.
		/// </summary>
	NppiSize GetSize();

		/// <summary>
		/// Size of the actual ROI.
		/// </summary>
	NppiSize GetSizeRoi();

		/// <summary>
		/// First pixel in the ROI.
		/// </summary>
	NppiPoint GetPointRoi();

		/// <summary>
		/// Device pointer to image data.
		/// </summary>
	CUdeviceptr GetDevicePointer();

		/// <summary>
		/// Device pointer to first pixel in ROI.
		/// </summary>
	CUdeviceptr GetDevicePointerRoi();

		/// <summary>
		/// Width in pixels
		/// </summary>
	int GetWidth();

		/// <summary>
		/// Width in bytes
		/// </summary>
	int GetWidthInBytes();

		/// <summary>
		/// Height in pixels
		/// </summary>
	int GetHeight();

		/// <summary>
		/// Width in pixels
		/// </summary>
	int GetWidthRoi();

		/// <summary>
		/// Width in bytes
		/// </summary>
	int GetWidthRoiInBytes();

		/// <summary>
		/// Height in pixels
		/// </summary>
	int GetHeightRoi();

		/// <summary>
		/// Pitch in bytes
		/// </summary>
	int GetPitch();


	/// <summary>
	/// Total size in bytes (Pitch * Height)
	/// </summary>
	int GetTotalSizeInBytes();

	/// <summary>
	/// Color channels
	/// </summary>
	int GetChannels();

	/// <summary>
	/// Defines the ROI on which all following operations take place
	/// </summary>
	/// <param name="roi"></param>
	void SetRoi(NppiRect roi);

	/// <summary>
	/// Defines the ROI on which all following operations take place
	/// </summary>
	/// <param name="x"></param>
	/// <param name="y"></param>
	/// <param name="width"></param>
	/// <param name="height"></param>
	void SetRoi(int x, int y, int width, int height);

	/// <summary>
	/// Resets the ROI to the full image
	/// </summary>
	void ResetRoi();

};

#endif
