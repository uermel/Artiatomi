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


#ifndef CUDAVARRAYS_H
#define CUDAVARRAYS_H

#include "CudaDefault.h"

#ifdef USE_CUDA
#include "CudaException.h"
#include "CudaVariables.h"

namespace Cuda
{
	//!  A wrapper class for a CUDA Array.
	/*!
	  CudaArray1D manages all the functionality of a 1D CUDA Array.
	  \author Michael Kunz
	  \date   January 2010
	  \version 1.0
	*/
	//A wrapper class for a CUDA Array.
	class CudaArray1D
	{
	private:
		CUarray mCUarray;
		CUDA_ARRAY_DESCRIPTOR mDescriptor;

	public:
		//! CudaArray1D constructor
		/*!
			Creates a new 1D CUDA Array size \p aSizeInElements and \p aNumChannels channels.
			\param aFormat Array format
			\param aSizeInElements Size of the array
			\param aNumChannels Number of array channels. Must be 1, 2 or 4.
		*/
		CudaArray1D(CUarray_format aFormat, size_t aSizeInElements, uint aNumChannels);
		CudaArray1D(const CudaArray1D&) = delete;

		//! CudaArray1D destructor
		~CudaArray1D();

		//! Copy data from device memory to this array
		/*!
			Copies the data given by \p aSource in device memory to this array.
			\param aSource Data source in device memory
			\param aOffsetInBytes An Offset in the destination array.
		*/
		void CopyFromDeviceToArray(CudaDeviceVariable& aSource, size_t aOffsetInBytes = 0);

		//! Copy data from this array to device memory
		/*!
			Copies from this array to device memory given by \p aDest.
			\param aDest Data destination in device memory
			\param aOffsetInBytes An Offset in the source array.
		*/
		void CopyFromArrayToDevice(CudaDeviceVariable& aDest, size_t aOffsetInBytes = 0);


		//! Copy data from host memory to this array
		/*!
			Copies the data given by \p aSource in host memory to this array.
			\param aSource Data source in host memory
			\param aOffsetInBytes An Offset in the destination array.
		*/
		void CopyFromHostToArray(void* aSource, size_t aOffsetInBytes = 0);

		//! Copy data from this array to host memory
		/*!
			Copies from this array to host memory given by \p aDest.
			\param aDest Data destination in host memory
			\param aOffsetInBytes An Offset in the source array.
		*/
		void CopyFromArrayToHost(void* aDest, size_t aOffsetInBytes = 0);


		//! Copy data from host memory to this array
		/*!
			Copies the data given by \p aSource in host memory to this array.
			\param aSource Data source in host memory
			\param aOffsetInBytes An Offset in the destination array.
		*/
		void CopyFromHostToArrayAsync(CUstream stream, void* aSource, size_t aOffsetInBytes = 0);

		//! Copy data from this array to host memory
		/*!
			Copies from this array to host memory given by \p aDest.
			\param aDest Data destination in host memory
			\param aOffsetInBytes An Offset in the source array.
		*/
		void CopyFromArrayToHostAsync(CUstream stream, void* aDest, size_t aOffsetInBytes = 0);


		//! Get the CUDA_ARRAY_DESCRIPTOR of this array
		/*!
			Get method for the wrapped CUDA_ARRAY_DESCRIPTOR.
			\return Wrapped CUDA_ARRAY_DESCRIPTOR.
		*/
		CUDA_ARRAY_DESCRIPTOR GetArrayDescriptor();

		//! Get the CUarray of this array
		/*!
			Get method for the wrapped CUDA_ARRAY_DESCRIPTOR.
			\return Wrapped CUarray.
		*/
		CUarray GetCUarray();
	};

	//!  A wrapper class for a CUDA Array.
	/*!
	  CudaArray2D manages all the functionality of a 2D CUDA Array.
	  \author Michael Kunz
	  \date   January 2010
	  \version 1.0
	*/
	class CudaArray2D
	{
	private:
		CUarray mCUarray;
		CUDA_ARRAY_DESCRIPTOR mDescriptor;

	public:
		//! CudaArray2D constructor
		/*!
			Creates a new 2D CUDA Array of size \p aWidthInElements x \p aHeightInElements and \p aNumChannels channels.
			\param aFormat Array format
			\param aWidthInElements Width of the array
			\param aHeightInElements Height of the array
			\param aNumChannels Number of array channels. Must be 1, 2 or 4.
		*/
		CudaArray2D(CUarray_format aFormat, size_t aWidthInElements, size_t aHeightInElements, uint aNumChannels);
		CudaArray2D(const CudaArray2D&) = delete;

		//! CudaArray2D destructor
		~CudaArray2D();

		//! Copy data from device memory to this array
		/*!
			Copies the data given by \p aSource in device memory to this array.
			\param aSource Data source in device memory
		*/
		void CopyFromDeviceToArray(CudaPitchedDeviceVariable& aSource);

		//! Copy data from this array to device memory
		/*!
			Copies from this array to device memory given by \p aDest.
			\param aDest Data destination in device memory
		*/
		void CopyFromArrayToDevice(CudaPitchedDeviceVariable& aDest);


		//! Copy data from host memory to this array
		/*!
			Copies the data given by \p aSource in host memory to this array.
			\param aSource Data source in host memory
		*/
		void CopyFromHostToArray(void* aSource);

		//! Copy data from this array to host memory
		/*!
			Copies from this array to host memory given by \p aDest.
			\param aDest Data destination in host memory
		*/
		void CopyFromArrayToHost(void* aDest);


		//! Copy data from device memory to this array
		/*!
			Copies the data given by \p aSource in device memory to this array.
			\param aSource Data source in device memory
		*/
		void CopyFromDeviceToArrayAsync(CUstream stream, CudaPitchedDeviceVariable& aSource);

		//! Copy data from this array to device memory
		/*!
			Copies from this array to device memory given by \p aDest.
			\param aDest Data destination in device memory
		*/
		void CopyFromArrayToDeviceAsync(CUstream stream, CudaPitchedDeviceVariable& aDest);


		//! Copy data from host memory to this array
		/*!
			Copies the data given by \p aSource in host memory to this array.
			\param aSource Data source in host memory
		*/
		void CopyFromHostToArrayAsync(CUstream stream, void* aSource);

		//! Copy data from this array to host memory
		/*!
			Copies from this array to host memory given by \p aDest.
			\param aDest Data destination in host memory
		*/
		void CopyFromArrayToHostAsync(CUstream stream, void* aDest);


		//! Get the CUDA_ARRAY_DESCRIPTOR of this array
		/*!
			Get method for the wrapped CUDA_ARRAY_DESCRIPTOR.
			\return Wrapped CUDA_ARRAY_DESCRIPTOR.
		*/
		CUDA_ARRAY_DESCRIPTOR GetArrayDescriptor();

		//! Get the CUarray of this array
		/*!
			Get method for the wrapped CUDA_ARRAY_DESCRIPTOR.
			\return Wrapped CUarray.
		*/
		CUarray GetCUarray();
	};

	//!  A wrapper class for a CUDA Array.
	/*!
	  CudaArray3D manages all the functionality of a 3D CUDA Array.
	  \author Michael Kunz
	  \date   January 2010
	  \version 1.0
	*/
	class CudaArray3D
	{
	private:
		CUarray mCUarray;
		CUDA_ARRAY3D_DESCRIPTOR mDescriptor;

	public:
		//! CudaArray3D constructor
		/*!
			Creates a new 3D CUDA Array of size \p aWidthInElements x \p aHeightInElements x \p aDepthInElements and \p aNumChannels channels.
			\param aFormat Array format
			\param aWidthInElements Width of the array
			\param aHeightInElements Height of the array
			\param aDepthInElements Depth of the array
			\param aNumChannels Number of array channels. Must be 1, 2 or 4.
			\param aFlags Array creation flags.
		*/
		CudaArray3D(CUarray_format aFormat, size_t aWidthInElements, size_t aHeightInElements, size_t aDepthInElements, uint aNumChannels, uint aFlags = 0);
		CudaArray3D(const CudaArray3D&) = delete;

		//! CudaArray3D destructor
		~CudaArray3D();


		//! Copy data from device memory to this array
		/*!
			Copies the data given by \p aSource in device memory to this array.
			\param aSource Data source in device memory
		*/
		void CopyFromDeviceToArray(CudaDeviceVariable& aSource);

		//! Copy data from this array to device memory
		/*!
			Copies from this array to device memory given by \p aDest.
			\param aDest Data destination in device memory
		*/
		void CopyFromArrayToDevice(CudaDeviceVariable& aDest);

		//! Copy data from device memory to this array
		/*!
			Copies the data given by \p aSource in device memory to this array.
			\param aSource Data source in device memory
		*/
		void CopyFromDeviceToArray(CudaPitchedDeviceVariable& aSource);

		//! Copy data from this array to device memory
		/*!
			Copies from this array to device memory given by \p aDest.
			\param aDest Data destination in device memory
		*/
		void CopyFromArrayToDevice(CudaPitchedDeviceVariable& aDest);


		//! Copy data from host memory to this array
		/*!
			Copies the data given by \p aSource in host memory to this array.
			\param aSource Data source in host memory
		*/
		void CopyFromHostToArray(void* aSource);

		//! Copy data from this array to host memory
		/*!
			Copies from this array to host memory given by \p aDest.
			\param aDest Data destination in host memory
		*/
		void CopyFromArrayToHost(void* aDest);


		//! Copy data from device memory to this array
		/*!
			Copies the data given by \p aSource in device memory to this array.
			\param aSource Data source in device memory
		*/
		void CopyFromDeviceToArrayAsync(CUstream stream, CudaDeviceVariable& aSource);

		//! Copy data from this array to device memory
		/*!
			Copies from this array to device memory given by \p aDest.
			\param aDest Data destination in device memory
		*/
		void CopyFromArrayToDeviceAsync(CUstream stream, CudaDeviceVariable& aDest);

		//! Copy data from device memory to this array
		/*!
			Copies the data given by \p aSource in device memory to this array.
			\param aSource Data source in device memory
		*/
		void CopyFromDeviceToArrayAsync(CUstream stream, CudaPitchedDeviceVariable& aSource);

		//! Copy data from this array to device memory
		/*!
			Copies from this array to device memory given by \p aDest.
			\param aDest Data destination in device memory
		*/
		void CopyFromArrayToDeviceAsync(CUstream stream, CudaPitchedDeviceVariable& aDest);


		//! Copy data from host memory to this array
		/*!
			Copies the data given by \p aSource in host memory to this array.
			\param aSource Data source in host memory
		*/
		void CopyFromHostToArrayAsync(CUstream stream, void* aSource);

		//! Copy data from this array to host memory
		/*!
			Copies from this array to host memory given by \p aDest.
			\param aDest Data destination in host memory
		*/
		void CopyFromArrayToHostAsync(CUstream stream, void* aDest);


		//! Get the CUDA_ARRAY_DESCRIPTOR of this array
		/*!
			Get method for the wrapped CUDA_ARRAY_DESCRIPTOR.
			\return Wrapped CUDA_ARRAY_DESCRIPTOR.
		*/
		CUDA_ARRAY3D_DESCRIPTOR GetArrayDescriptor();

		//! Get the CUarray of this array
		/*!
			Get method for the wrapped CUDA_ARRAY_DESCRIPTOR.
			\return Wrapped CUarray.
		*/
		CUarray GetCUarray();
	};

	//! Get the channel size in bytes for a given CUarray_format.
	unsigned int GetChannelSize(const CUarray_format aFormat);
}
#endif //USE_CUDA
#endif //CUDAVARRAYS_H
