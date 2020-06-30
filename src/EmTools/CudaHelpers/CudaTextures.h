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


#ifndef CUDATEXTURES_H
#define CUDATEXTURES_H

#include "CudaDefault.h"
#ifdef USE_CUDA
#include "CudaException.h"
#include "CudaContext.h"
#include "CudaKernel.h"
#include "CudaVariables.h"
#include "CudaArrays.h"

namespace Cuda
{
	//! A wrapper class for a linear 1D CUDA Texture. 
	/*!
	  \author Michael Kunz
	  \date   January 2010
	  \version 1.0
	*/
	//A wrapper class for a linear 1D CUDA Texture. 
	class CudaTextureLinear1D
	{
	private:
		CUtexref mTexref; //Wrapped CUDA Texture Reference
		CUfilter_mode mFilterMode; //Filter mode (CU_TR_FILTER_MODE_POINT or CU_TR_FILTER_MODE_LINEAR)
		uint mTexRefSetFlag; 
		CUaddress_mode mAddressMode; //Texture addess mode (CU_TR_ADDRESS_MODE_WRAP or CU_TR_ADDRESS_MODE_CLAMP or CU_TR_ADDRESS_MODE_MIRROR or CU_TR_ADDRESS_MODE_BORDER)
		CUarray_format mArrayFormat; //Array format (CU_AD_FORMAT_FLOAT, CU_AD_FORMAT_SIGNED_INT8, etc)
		size_t mSizeInElements; //Size of one texture element
		uint mChannelSize; //Texture channel size
		size_t mSizeInBytes; //Data size in bytes
		uint mNumChannels; //Number of texture channels  (1, 2 or 4)
		string mName; //Texture name as defined in the *.cu source file
		CudaKernel* mKernel; //CudaKernel using the texture
		CudaDeviceVariable* mDevVar; //Device Variable where the texture data is stored
		bool mCleanUp; //Indicates if the device variable was created by the object itself

	public:	
		//! CudaTextureLinear1D constructor
		/*!
			Creates a new linear 1D Texture and allocates the needed memory in device memory
			\param aKernel CudaKernel using the texture
			\param aTexName Texture name as defined in the *.cu source file
			\param aAddressMode Texture addess mode (CU_TR_ADDRESS_MODE_WRAP or CU_TR_ADDRESS_MODE_CLAMP or CU_TR_ADDRESS_MODE_MIRROR or CU_TR_ADDRESS_MODE_BORDER)
			\param aTexRefSetFlag TexRefSetFlag
			\param aFormat Array format (CU_AD_FORMAT_FLOAT, CU_AD_FORMAT_SIGNED_INT8, etc)
			\param aSizeInElements Texture size in elements
			\param aNumChannels Number of texture channels  (must be 1, 2 or 4)
		*/
		//CudaTextureLinear1D constructor (allocates new device memory)
		CudaTextureLinear1D(CudaKernel* aKernel, string aTexName, CUaddress_mode aAddressMode,
			uint aTexRefSetFlag, CUarray_format aFormat, size_t aSizeInElements, uint aNumChannels);
		
		//! CudaTextureLinear1D constructor
		/*!
			Creates a new linear 1D Texture and using \p aDevVar as data storage
			\param aKernel CudaKernel using the texture
			\param aTexName Texture name as defined in the *.cu source file
			\param aAddressMode Texture addess mode (CU_TR_ADDRESS_MODE_WRAP or CU_TR_ADDRESS_MODE_CLAMP or CU_TR_ADDRESS_MODE_MIRROR or CU_TR_ADDRESS_MODE_BORDER)
			\param aTexRefSetFlag TexRefSetFlag
			\param aFormat Array format (CU_AD_FORMAT_FLOAT, CU_AD_FORMAT_SIGNED_INT8, etc)
			\param aDevVar Device variable with the texture data
			\param aSizeInElements Texture size in elements
			\param aNumChannels Number of texture channels  (must be 1, 2 or 4)
		*/
		//CudaTextureLinear1D constructor
		CudaTextureLinear1D(CudaKernel* aKernel, string aTexName, CUaddress_mode aAddressMode, uint aTexRefSetFlag, 
			CUarray_format aFormat, CudaDeviceVariable* aDevVar, size_t aSizeInElements, uint aNumChannels);
		
		//! CudaTextureLinear1D destructor
		//CudaTextureLinear1D destructor
		~CudaTextureLinear1D();
				
		CUtexref GetTextureReference();
		CUfilter_mode GetFilterMode();
		uint GetTexRefSetFlags();
		CUaddress_mode GetAddressMode();
		CUarray_format GetArrayFormat();
		size_t GetSizeInElements();
		uint GetChannelSize();
		size_t GetSizeInBytes();
		uint GetNumChannels();
		string GetName();
		CudaKernel* GetCudaKernel();
		CudaDeviceVariable* GetDeviceVariable();
	};
	
	//! A wrapper class for a array 1D CUDA Texture. 
	/*!
	  \author Michael Kunz
	  \date   January 2010
	  \version 1.0
	*/
	//A wrapper class for a array 1D CUDA Texture. 
	class CudaTextureArray1D
	{
	private:
		CUtexref mTexref; //Wrapped CUDA Texture Reference
		CUfilter_mode mFilterMode; //Filter mode (CU_TR_FILTER_MODE_POINT or CU_TR_FILTER_MODE_LINEAR)
		uint mTexRefSetFlag;
		CUaddress_mode mAddressMode; //Texture addess mode (CU_TR_ADDRESS_MODE_WRAP or CU_TR_ADDRESS_MODE_CLAMP or CU_TR_ADDRESS_MODE_MIRROR or CU_TR_ADDRESS_MODE_BORDER)
		CUarray_format mArrayFormat; //Array format (CU_AD_FORMAT_FLOAT, CU_AD_FORMAT_SIGNED_INT8, etc)
		size_t mSizeInElements; //Size of one texture element
		uint mChannelSize; //Texture channel size
		size_t mSizeInBytes; //Data size in bytes
		uint mNumChannels; //Number of texture channels  (1, 2 or 4)
		string mName; //Texture name as defined in the *.cu source file
		CudaKernel* mKernel; //CudaKernel using the texture
		CudaArray1D* mArray; //CUDA Array where the texture data is stored
		bool mCleanUp; //Indicates if the cuda array was created by the object itself

	public:
		//! CudaTextureArray1D constructor
		/*!
			Creates a new array 1D Texture and a new 1D CUDA Array
			\param aKernel CudaKernel using the texture
			\param aTexName Texture name as defined in the *.cu source file
			\param aFilterMode Filter mode (CU_TR_FILTER_MODE_POINT or CU_TR_FILTER_MODE_LINEAR)
			\param aAddressMode Texture addess mode (CU_TR_ADDRESS_MODE_WRAP or CU_TR_ADDRESS_MODE_CLAMP or CU_TR_ADDRESS_MODE_MIRROR or CU_TR_ADDRESS_MODE_BORDER)
			\param aTexRefSetFlag TexRefSetFlag
			\param aFormat Array format (CU_AD_FORMAT_FLOAT, CU_AD_FORMAT_SIGNED_INT8, etc)
			\param aSizeInElements Texture size in elements
			\param aNumChannels Number of texture channels  (must be 1, 2 or 4)
		*/
		//CudaTextureArray1D constructor (Creates new array)
		CudaTextureArray1D(CudaKernel* aKernel, string aTexName, CUaddress_mode aAddressMode, CUfilter_mode aFilterMode,
			uint aTexRefSetFlag, CUarray_format aFormat, size_t aSizeInElements, uint aNumChannels);
		
		//! CudaTextureArray1D constructor
		/*!
			Creates a new array 1D Texture based on \p aArray.
			\param aKernel CudaKernel using the texture
			\param aTexName Texture name as defined in the *.cu source file
			\param aFilterMode Filter mode (CU_TR_FILTER_MODE_POINT or CU_TR_FILTER_MODE_LINEAR)
			\param aAddressMode Texture addess mode (CU_TR_ADDRESS_MODE_WRAP or CU_TR_ADDRESS_MODE_CLAMP or CU_TR_ADDRESS_MODE_MIRROR or CU_TR_ADDRESS_MODE_BORDER)
			\param aTexRefSetFlag TexRefSetFlag
			\param aArray CUDA Array with the texture data
		*/
		//CudaTextureArray1D constructor
		CudaTextureArray1D(CudaKernel* aKernel, string aTexName, CUaddress_mode aAddressMode, CUfilter_mode aFilterMode,
			uint aTexRefSetFlag, CudaArray1D* aArray);
		
		//! CudaTextureArray1D destructor
		//CudaTextureArray1D destructor
		~CudaTextureArray1D();
				
		CUtexref GetTextureReference();
		CUfilter_mode GetFilterMode();
		uint GetTexRefSetFlags();
		CUaddress_mode GetAddressMode();
		CUarray_format GetArrayFormat();
		size_t GetSizeInElements();
		uint GetChannelSize();
		size_t GetSizeInBytes();
		uint GetNumChannels();
		string GetName();
		CudaKernel* GetCudaKernel();
		CudaArray1D* GetArray();
	};
	
	//! A wrapper class for a array 2D CUDA Texture. 
	/*!
	  \author Michael Kunz
	  \date   January 2010
	  \version 1.0
	*/
	//A wrapper class for a array 2D CUDA Texture. 
	class CudaTextureArray2D
	{
	private:
		CUtexref mTexref; //Wrapped CUDA Texture Reference
		CUfilter_mode mFilterMode; //Filter mode (CU_TR_FILTER_MODE_POINT or CU_TR_FILTER_MODE_LINEAR)
		uint mTexRefSetFlag;
		CUaddress_mode mAddressMode0; //Texture addess mode (CU_TR_ADDRESS_MODE_WRAP or CU_TR_ADDRESS_MODE_CLAMP or CU_TR_ADDRESS_MODE_MIRROR or CU_TR_ADDRESS_MODE_BORDER)
		CUaddress_mode mAddressMode1; //Texture addess mode (CU_TR_ADDRESS_MODE_WRAP or CU_TR_ADDRESS_MODE_CLAMP or CU_TR_ADDRESS_MODE_MIRROR or CU_TR_ADDRESS_MODE_BORDER)
		CUarray_format mArrayFormat; //Array format (CU_AD_FORMAT_FLOAT, CU_AD_FORMAT_SIGNED_INT8, etc)
		size_t mHeight; //Height of the texture in elements
		size_t mWidth; //Width of the texture in elements
		uint mChannelSize; //Texture channel size
		size_t mSizeInBytes; //Data size in bytes
		uint mNumChannels; //Number of texture channels  (1, 2 or 4)
		string mName; //Texture name as defined in the *.cu source file
		CudaKernel* mKernel; //CudaKernel using the texture
		CudaArray2D* mArray; //CUDA Array where the texture data is stored
		bool mCleanUp; //Indicates if the cuda array was created by the object itself

	public:
		//! CudaTextureArray2D constructor
		/*!
			Creates a new array 2D Texture and a new 2D CUDA Array
			\param aKernel CudaKernel using the texture
			\param aTexName Texture name as defined in the *.cu source file
			\param aFilterMode Filter mode (CU_TR_FILTER_MODE_POINT or CU_TR_FILTER_MODE_LINEAR)
			\param aAddressMode0 Texture addess mode in x direction (CU_TR_ADDRESS_MODE_WRAP or CU_TR_ADDRESS_MODE_CLAMP or CU_TR_ADDRESS_MODE_MIRROR or CU_TR_ADDRESS_MODE_BORDER)
			\param aAddressMode1 Texture addess mode in y direction (CU_TR_ADDRESS_MODE_WRAP or CU_TR_ADDRESS_MODE_CLAMP or CU_TR_ADDRESS_MODE_MIRROR or CU_TR_ADDRESS_MODE_BORDER)
			\param aTexRefSetFlag TexRefSetFlag
			\param aFormat Array format (CU_AD_FORMAT_FLOAT, CU_AD_FORMAT_SIGNED_INT8, etc)
			\param aHeight Texture height in elements
			\param aWidth Texture width in elements
			\param aNumChannels Number of texture channels  (must be 1, 2 or 4)
		*/
		//CudaTextureArray2D constructor (Creates new array)
		CudaTextureArray2D(CudaKernel* aKernel, string aTexName, CUaddress_mode aAddressMode0, CUaddress_mode aAddressMode1, 
			CUfilter_mode aFilterMode, uint aTexRefSetFlag, CUarray_format aFormat, size_t aWidth, size_t aHeight, uint aNumChannels);
		
		//! CudaTextureArray2D constructor
		/*!
			Creates a new array 2D Texture based on \p aArray.
			\param aKernel CudaKernel using the texture
			\param aTexName Texture name as defined in the *.cu source file
			\param aFilterMode Filter mode (CU_TR_FILTER_MODE_POINT or CU_TR_FILTER_MODE_LINEAR)
			\param aAddressMode0 Texture addess mode in x direction (CU_TR_ADDRESS_MODE_WRAP or CU_TR_ADDRESS_MODE_CLAMP or CU_TR_ADDRESS_MODE_MIRROR or CU_TR_ADDRESS_MODE_BORDER)
			\param aAddressMode1 Texture addess mode in y direction (CU_TR_ADDRESS_MODE_WRAP or CU_TR_ADDRESS_MODE_CLAMP or CU_TR_ADDRESS_MODE_MIRROR or CU_TR_ADDRESS_MODE_BORDER)
			\param aTexRefSetFlag TexRefSetFlag
			\param aArray CUDA Array with the texture data
		*/
		//CudaTextureArray2D constructor
		CudaTextureArray2D(CudaKernel* aKernel, string aTexName, CUaddress_mode aAddressMode0, CUaddress_mode aAddressMode1, 
			CUfilter_mode aFilterMode, uint aTexRefSetFlag, CudaArray2D* aArray);
		
		//! CudaTextureArray2D destructor
		//CudaTextureArray2D destructor
		~CudaTextureArray2D();		

		CUtexref GetTextureReference();
		CUfilter_mode GetFilterMode();
		uint GetTexRefSetFlags();
		CUaddress_mode* GetAddressModes();
		CUarray_format GetArrayFormat();
		size_t GetHeight();
		size_t GetWidth();
		uint GetChannelSize();
		size_t GetSizeInBytes();
		uint GetNumChannels();
		string GetName();
		CudaKernel* GetCudaKernel();
		CudaArray2D* GetArray();
	};
	
	//! A wrapper class for a linear 2D CUDA Texture. 
	/*!
	  \author Michael Kunz
	  \date   January 2010
	  \version 1.0
	*/
	//A wrapper class for a linear 2D CUDA Texture. 
	class CudaTextureLinearPitched2D
	{
	private:
		CUtexref mTexref; //Wrapped CUDA Texture Reference
		CUfilter_mode mFilterMode; //Filter mode (CU_TR_FILTER_MODE_POINT or CU_TR_FILTER_MODE_LINEAR)
		uint mTexRefSetFlag;
		CUaddress_mode mAddressMode0; //Texture addess mode (CU_TR_ADDRESS_MODE_WRAP or CU_TR_ADDRESS_MODE_CLAMP or CU_TR_ADDRESS_MODE_MIRROR or CU_TR_ADDRESS_MODE_BORDER)
		CUaddress_mode mAddressMode1; //Texture addess mode (CU_TR_ADDRESS_MODE_WRAP or CU_TR_ADDRESS_MODE_CLAMP or CU_TR_ADDRESS_MODE_MIRROR or CU_TR_ADDRESS_MODE_BORDER)
		CUarray_format mArrayFormat; //Array format (CU_AD_FORMAT_FLOAT, CU_AD_FORMAT_SIGNED_INT8, etc)
		size_t mHeight; //Height of the texture in elements
		size_t mWidth; //Width of the texture in elements
		uint mChannelSize; //Texture channel size
		size_t mSizeInBytes; //Data size in bytes
		uint mNumChannels; //Number of texture channels  (1, 2 or 4)
		string mName; //Texture name as defined in the *.cu source file
		CudaKernel* mKernel; //CudaKernel using the texture
		CudaPitchedDeviceVariable* mDevVar; //Device Variable where the texture data is stored
		bool mCleanUp; //Indicates if the pitched device variabale was created by the object itself

	public:
		//! CudaTextureLinearPitched2D constructor
		/*!
			Creates a new linear 2D Texture and a new 2D CUDA pitched device variable
			\param aKernel CudaKernel using the texture
			\param aTexName Texture name as defined in the *.cu source file
			\param aFilterMode Filter mode (CU_TR_FILTER_MODE_POINT or CU_TR_FILTER_MODE_LINEAR)
			\param aAddressMode0 Texture addess mode in x direction (CU_TR_ADDRESS_MODE_WRAP or CU_TR_ADDRESS_MODE_CLAMP or CU_TR_ADDRESS_MODE_MIRROR or CU_TR_ADDRESS_MODE_BORDER)
			\param aAddressMode1 Texture addess mode in y direction (CU_TR_ADDRESS_MODE_WRAP or CU_TR_ADDRESS_MODE_CLAMP or CU_TR_ADDRESS_MODE_MIRROR or CU_TR_ADDRESS_MODE_BORDER)
			\param aTexRefSetFlag TexRefSetFlag
			\param aFormat Array format (CU_AD_FORMAT_FLOAT, CU_AD_FORMAT_SIGNED_INT8, etc)
			\param aHeight Texture height in elements
			\param aWidth Texture width in elements
			\param aNumChannels Number of texture channels  (must be 1, 2 or 4)
		*/
		//CudaTextureLinearPitched2D constructor (Creates new array)
		CudaTextureLinearPitched2D(CudaKernel* aKernel, string aTexName, CUaddress_mode aAddressMode0, CUaddress_mode aAddressMode1, 
			CUfilter_mode aFilterMode, uint aTexRefSetFlag, CUarray_format aFormat, size_t aWidth, size_t aHeight, uint aNumChannels);
		
		//! CudaTextureLinearPitched2D constructor
		/*!
			Creates a new linear 2D Texture based on pitched \p aDevVar.
			\param aKernel CudaKernel using the texture
			\param aTexName Texture name as defined in the *.cu source file
			\param aFilterMode Filter mode (CU_TR_FILTER_MODE_POINT or CU_TR_FILTER_MODE_LINEAR)
			\param aAddressMode0 Texture addess mode in x direction (CU_TR_ADDRESS_MODE_WRAP or CU_TR_ADDRESS_MODE_CLAMP or CU_TR_ADDRESS_MODE_MIRROR or CU_TR_ADDRESS_MODE_BORDER)
			\param aAddressMode1 Texture addess mode in y direction (CU_TR_ADDRESS_MODE_WRAP or CU_TR_ADDRESS_MODE_CLAMP or CU_TR_ADDRESS_MODE_MIRROR or CU_TR_ADDRESS_MODE_BORDER)
			\param aTexRefSetFlag TexRefSetFlag
			\param aFormat Array format (CU_AD_FORMAT_FLOAT, CU_AD_FORMAT_SIGNED_INT8, etc)
			\param aDevVar Pitched device variable where the texture data is stored
			\param aNumChannels Number of texture channels  (must be 1, 2 or 4)
		*/
		//CudaTextureLinearPitched2D constructor
		CudaTextureLinearPitched2D(CudaKernel* aKernel, string aTexName, CUaddress_mode aAddressMode0, CUaddress_mode aAddressMode1, 
			CUfilter_mode aFilterMode, uint aTexRefSetFlag, CudaPitchedDeviceVariable* aDevVar, CUarray_format aFormat, uint aNumChannels);
		
		static void Bind(CudaKernel* aKernel, string aTexName, CUaddress_mode aAddressMode0, CUaddress_mode aAddressMode1,
			CUfilter_mode aFilterMode, uint aTexRefSetFlag, CudaPitchedDeviceVariable* aDevVar, CUarray_format aFormat, uint aNumChannels);

		//! CudaTextureLinearPitched2D destructor
		//CudaTextureLinearPitched2D destructor
		~CudaTextureLinearPitched2D();
				
		CUtexref GetTextureReference();
		CUfilter_mode GetFilterMode();
		uint GetTexRefSetFlags();
		CUaddress_mode* GetAddressModes();
		CUarray_format GetArrayFormat();
		size_t GetHeight();
		size_t GetWidth();
		uint GetChannelSize();
		size_t GetSizeInBytes();
		uint GetNumChannels();
		string GetName();
		CudaKernel* GetCudaKernel();
		CudaPitchedDeviceVariable* GetDeviceVariable();
	};
	
	//! A wrapper class for a array 3D CUDA Texture. 
	/*!
	  \author Michael Kunz
	  \date   January 2010
	  \version 1.0
	*/
	//A wrapper class for a array 3D CUDA Texture. 
	class CudaTextureArray3D
	{
	private:
		CUtexref mTexref; //Wrapped CUDA Texture Reference
		CUfilter_mode mFilterMode; //Filter mode (CU_TR_FILTER_MODE_POINT or CU_TR_FILTER_MODE_LINEAR)
		uint mTexRefSetFlag;
		CUaddress_mode mAddressMode0; //Texture addess mode (CU_TR_ADDRESS_MODE_WRAP or CU_TR_ADDRESS_MODE_CLAMP or CU_TR_ADDRESS_MODE_MIRROR or CU_TR_ADDRESS_MODE_BORDER)
		CUaddress_mode mAddressMode1; //Texture addess mode (CU_TR_ADDRESS_MODE_WRAP or CU_TR_ADDRESS_MODE_CLAMP or CU_TR_ADDRESS_MODE_MIRROR or CU_TR_ADDRESS_MODE_BORDER)
		CUaddress_mode mAddressMode2; //Texture addess mode (CU_TR_ADDRESS_MODE_WRAP or CU_TR_ADDRESS_MODE_CLAMP or CU_TR_ADDRESS_MODE_MIRROR or CU_TR_ADDRESS_MODE_BORDER)
		CUarray_format mArrayFormat; //Array format (CU_AD_FORMAT_FLOAT, CU_AD_FORMAT_SIGNED_INT8, etc)
		size_t mHeight; //Height of the texture in elements
		size_t mWidth; //Width of the texture in elements
		size_t mDepth; //Depth of the texture in elements
		uint mChannelSize; //Texture channel size
		size_t mSizeInBytes; //Data size in bytes
		uint mNumChannels; //Number of texture channels  (1, 2 or 4)
		string mName; //Texture name as defined in the *.cu source file
		CudaKernel* mKernel; //CudaKernel using the texture
		CudaArray3D* mArray; //CUDA Array where the texture data is stored
		bool mCleanUp; //Indicates if the cuda array was created by the object itself

	public:
		//! CudaTextureArray3D constructor
		/*!
			Creates a new array 3D Texture and a new 3D CUDA Array
			\param aKernel CudaKernel using the texture
			\param aTexName Texture name as defined in the *.cu source file
			\param aFilterMode Filter mode (CU_TR_FILTER_MODE_POINT or CU_TR_FILTER_MODE_LINEAR)
			\param aAddressMode0 Texture addess mode in x direction (CU_TR_ADDRESS_MODE_WRAP or CU_TR_ADDRESS_MODE_CLAMP or CU_TR_ADDRESS_MODE_MIRROR or CU_TR_ADDRESS_MODE_BORDER)
			\param aAddressMode1 Texture addess mode in y direction (CU_TR_ADDRESS_MODE_WRAP or CU_TR_ADDRESS_MODE_CLAMP or CU_TR_ADDRESS_MODE_MIRROR or CU_TR_ADDRESS_MODE_BORDER)
			\param aAddressMode2 Texture addess mode in z direction (CU_TR_ADDRESS_MODE_WRAP or CU_TR_ADDRESS_MODE_CLAMP or CU_TR_ADDRESS_MODE_MIRROR or CU_TR_ADDRESS_MODE_BORDER)
			\param aTexRefSetFlag TexRefSetFlag
			\param aFormat Array format (CU_AD_FORMAT_FLOAT, CU_AD_FORMAT_SIGNED_INT8, etc)
			\param aHeight Texture height in elements
			\param aWidth Texture width in elements
			\param aDepth Texture width in elements
			\param aNumChannels Number of texture channels  (must be 1, 2 or 4)
		*/
		//CudaTextureArray2D constructor (Creates new array)
		CudaTextureArray3D(CudaKernel* aKernel, string aTexName, CUaddress_mode aAddressMode0, CUaddress_mode aAddressMode1, 
			CUaddress_mode aAddressMode2, CUfilter_mode aFilterMode, uint aTexRefSetFlag, CUarray_format aFormat, 
			size_t aWidth, size_t aHeight, size_t aDepth, uint aNumChannels);
		
		//! CudaTextureArray3D constructor
		/*!
			Creates a new array 3D Texture based on \p aArray.
			\param aKernel CudaKernel using the texture
			\param aTexName Texture name as defined in the *.cu source file
			\param aFilterMode Filter mode (CU_TR_FILTER_MODE_POINT or CU_TR_FILTER_MODE_LINEAR)
			\param aAddressMode0 Texture addess mode in x direction (CU_TR_ADDRESS_MODE_WRAP or CU_TR_ADDRESS_MODE_CLAMP or CU_TR_ADDRESS_MODE_MIRROR or CU_TR_ADDRESS_MODE_BORDER)
			\param aAddressMode1 Texture addess mode in y direction (CU_TR_ADDRESS_MODE_WRAP or CU_TR_ADDRESS_MODE_CLAMP or CU_TR_ADDRESS_MODE_MIRROR or CU_TR_ADDRESS_MODE_BORDER)
			\param aAddressMode2 Texture addess mode in z direction (CU_TR_ADDRESS_MODE_WRAP or CU_TR_ADDRESS_MODE_CLAMP or CU_TR_ADDRESS_MODE_MIRROR or CU_TR_ADDRESS_MODE_BORDER)
			\param aTexRefSetFlag TexRefSetFlag
			\param aArray CUDA Array with the texture data
		*/
		//CudaTextureArray3D constructor
		CudaTextureArray3D(CudaKernel* aKernel, string aTexName, CUaddress_mode aAddressMode0, CUaddress_mode aAddressMode1, 
			CUaddress_mode aAddressMode2, CUfilter_mode aFilterMode, uint aTexRefSetFlag, CudaArray3D* aArray);
		
		//! CudaTextureArray3D destructor
		//CudaTextureArray3D destructor
		~CudaTextureArray3D();
				
		CUtexref GetTextureReference();
		CUfilter_mode GetFilterMode();
		uint GetTexRefSetFlags();
		CUaddress_mode* GetAddressModes();
		CUarray_format GetArrayFormat();
		size_t GetHeight();
		size_t GetWidth();
		size_t GetDepth();
		uint GetChannelSize();
		size_t GetSizeInBytes();
		uint GetNumChannels();
		string GetName();
		CudaKernel* GetCudaKernel();
		CudaArray3D* GetArray();
		void BindToTexRef();
	};

	
	
	class CudaTextureObject3D
	{
	private:
		CUtexObject mTexObj;
		CUDA_RESOURCE_DESC mResDesc;
		CUDA_TEXTURE_DESC mTexDesc;
		CUDA_RESOURCE_VIEW_DESC mResViewDesc;

		CudaArray3D* mArray; //CUDA Array where the texture data is stored
		bool mCleanUp; //Indicates if the cuda array was created by the object itself

	public:
		CudaTextureObject3D(CUaddress_mode aAddressMode0, CUaddress_mode aAddressMode1, 
			CUaddress_mode aAddressMode2, CUfilter_mode aFilterMode, uint aTexRefSetFlag, CudaArray3D* aArray);
		
		~CudaTextureObject3D();
			
		CudaArray3D* GetArray();

		CUtexObject GetTexObject();
	};

	class CudaTextureObject2D
	{
	private:
		CUtexObject mTexObj;
		CUDA_RESOURCE_DESC mResDesc;
		CUDA_TEXTURE_DESC mTexDesc;
		CUDA_RESOURCE_VIEW_DESC mResViewDesc;

		CudaPitchedDeviceVariable* mData; //CUDA Array where the texture data is stored
		bool mCleanUp; //Indicates if the cuda array was created by the object itself

	public:
		CudaTextureObject2D(CUaddress_mode aAddressMode0, CUaddress_mode aAddressMode1,
			CUfilter_mode aFilterMode, uint aTexRefSetFlag, CudaPitchedDeviceVariable* aArray, CUarray_format aDataFormat, uint aNumChannels);
		CudaTextureObject2D();

		~CudaTextureObject2D();

		void Bind(CUaddress_mode aAddressMode0, CUaddress_mode aAddressMode1,
			CUfilter_mode aFilterMode, uint aTexRefSetFlag, CudaPitchedDeviceVariable* aArray, CUarray_format aDataFormat, uint aNumChannels);

		CudaPitchedDeviceVariable* GetData();

		CUtexObject GetTexObject();
	};



//	class CudaSurfaceObject3D
//	{
//	private:
//		CUsurfObject mSurfObj;
//		CUDA_RESOURCE_DESC mResDesc;
//
//		CudaArray3D* mArray; //CUDA Array where the texture data is stored
//		bool mCleanUp; //Indicates if the cuda array was created by the object itself
//
//	public:
//		CudaSurfaceObject3D(CudaArray3D* aArray);
//
//		~CudaSurfaceObject3D();
//
//		CudaArray3D* GetArray();
//
//		CUsurfObject GetSurfObject();
//	};
}
#endif //USE_CUDA
#endif //CUDATEXTURES_H