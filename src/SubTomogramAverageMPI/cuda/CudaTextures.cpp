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


#include "CudaTextures.h"

namespace Cuda
{
	CudaTextureLinear1D::CudaTextureLinear1D(CudaKernel* aKernel, string aTexName, CUaddress_mode aAddressMode,
		uint aTexRefSetFlag, CUarray_format aFormat, size_t aSizeInElements, uint aNumChannels)
		: mKernel(aKernel),
		mName(aTexName),
		mAddressMode(aAddressMode),
		mTexRefSetFlag(aTexRefSetFlag),
		mSizeInElements(aSizeInElements),
		mNumChannels(aNumChannels),
		mArrayFormat(aFormat),
		mFilterMode(CU_TR_FILTER_MODE_POINT),
		mCleanUp(true)
	{
		cudaSafeCall(cuModuleGetTexRef(&mTexref, mKernel->GetCUmodule(), mName.c_str()));
		cudaSafeCall(cuTexRefSetAddressMode(mTexref, 0, mAddressMode));
		cudaSafeCall(cuTexRefSetFilterMode(mTexref, mFilterMode));
		cudaSafeCall(cuTexRefSetFlags(mTexref, mTexRefSetFlag));
		cudaSafeCall(cuTexRefSetFormat(mTexref, mArrayFormat, mNumChannels));

		mChannelSize = Cuda::GetChannelSize(mArrayFormat);
		mSizeInBytes = mSizeInElements * mChannelSize * mNumChannels;

		mDevVar = new CudaDeviceVariable(mSizeInBytes);

		cudaSafeCall(cuTexRefSetAddress(NULL, mTexref, mDevVar->GetDevicePtr(), mSizeInBytes));
		cudaSafeCall(cuParamSetTexRef(mKernel->GetCUfunction(), CU_PARAM_TR_DEFAULT, mTexref));
	}

	CudaTextureLinear1D::CudaTextureLinear1D(CudaKernel* aKernel, string aTexName, CUaddress_mode aAddressMode,
		uint aTexRefSetFlag, CUarray_format aFormat, CudaDeviceVariable* aDevVar, size_t aSizeInElements, uint aNumChannels)
		: mKernel(aKernel),
		mName(aTexName),
		mAddressMode(aAddressMode),
		mTexRefSetFlag(aTexRefSetFlag),
		mSizeInElements(aSizeInElements),
		mNumChannels(aNumChannels),
		mArrayFormat(aFormat),
		mFilterMode(CU_TR_FILTER_MODE_POINT),
		mCleanUp(false)
	{
		cudaSafeCall(cuModuleGetTexRef(&mTexref, mKernel->GetCUmodule(), mName.c_str()));
		cudaSafeCall(cuTexRefSetAddressMode(mTexref, 0, mAddressMode));
		cudaSafeCall(cuTexRefSetFilterMode(mTexref, mFilterMode));
		cudaSafeCall(cuTexRefSetFlags(mTexref, mTexRefSetFlag));
		cudaSafeCall(cuTexRefSetFormat(mTexref, mArrayFormat, mNumChannels));

		mChannelSize = Cuda::GetChannelSize(mArrayFormat);
		mSizeInBytes = mSizeInElements * mChannelSize * mNumChannels;

		mDevVar = aDevVar;

		cudaSafeCall(cuTexRefSetAddress(NULL, mTexref, mDevVar->GetDevicePtr(), mSizeInBytes));
		cudaSafeCall(cuParamSetTexRef(mKernel->GetCUfunction(), CU_PARAM_TR_DEFAULT, mTexref));
	}

	CudaTextureLinear1D::~CudaTextureLinear1D()
	{
		if (mCleanUp && mDevVar)
		{
			delete mDevVar;
			mDevVar = NULL;
		}
	}


	CUtexref CudaTextureLinear1D::GetTextureReference()
	{
		return mTexref;
	}

	CUfilter_mode CudaTextureLinear1D::GetFilterMode()
	{
		return mFilterMode;
	}

	uint CudaTextureLinear1D::GetTexRefSetFlags()
	{
		return mTexRefSetFlag;
	}

	CUaddress_mode CudaTextureLinear1D::GetAddressMode()
	{
		return mAddressMode;
	}

	CUarray_format CudaTextureLinear1D::GetArrayFormat()
	{
		return mArrayFormat;
	}

	size_t CudaTextureLinear1D::GetSizeInElements()
	{
		return mSizeInElements;
	}

	uint CudaTextureLinear1D::GetChannelSize()
	{
		return mChannelSize;
	}

	size_t CudaTextureLinear1D::GetSizeInBytes()
	{
		return mSizeInBytes;
	}

	uint CudaTextureLinear1D::GetNumChannels()
	{
		return mNumChannels;
	}

	string CudaTextureLinear1D::GetName()
	{
		return mName;
	}

	CudaKernel* CudaTextureLinear1D::GetCudaKernel()
	{
		return mKernel;
	}

	CudaDeviceVariable* CudaTextureLinear1D::GetDeviceVariable()
	{
		return mDevVar;
	}




	CudaTextureArray1D::CudaTextureArray1D(CudaKernel* aKernel, string aTexName, CUaddress_mode aAddressMode, CUfilter_mode aFilterMode,
		uint aTexRefSetFlag, CUarray_format aFormat, size_t aSizeInElements, uint aNumChannels)
		: mKernel(aKernel),
		mName(aTexName),
		mAddressMode(aAddressMode),
		mTexRefSetFlag(aTexRefSetFlag),
		mSizeInElements(aSizeInElements),
		mNumChannels(aNumChannels),
		mArrayFormat(aFormat),
		mFilterMode(aFilterMode),
		mCleanUp(true)
	{
		cudaSafeCall(cuModuleGetTexRef(&mTexref, mKernel->GetCUmodule(), mName.c_str()));
		cudaSafeCall(cuTexRefSetAddressMode(mTexref, 0, mAddressMode));
		cudaSafeCall(cuTexRefSetFilterMode(mTexref, mFilterMode));
		cudaSafeCall(cuTexRefSetFlags(mTexref, mTexRefSetFlag));
		cudaSafeCall(cuTexRefSetFormat(mTexref, mArrayFormat, mNumChannels));

		mChannelSize = Cuda::GetChannelSize(mArrayFormat);
		mSizeInBytes = mSizeInElements * mChannelSize * mNumChannels;

		mArray = new CudaArray1D(mArrayFormat, mSizeInElements, mNumChannels);

		cudaSafeCall(cuTexRefSetArray(mTexref, mArray->GetCUarray(), mTexRefSetFlag));
		cudaSafeCall(cuParamSetTexRef(mKernel->GetCUfunction(), CU_PARAM_TR_DEFAULT, mTexref));

	}


	CudaTextureArray1D::CudaTextureArray1D(CudaKernel* aKernel, string aTexName, CUaddress_mode aAddressMode, CUfilter_mode aFilterMode,
		uint aTexRefSetFlag, CudaArray1D* aArray)
		: mKernel(aKernel),
		mName(aTexName),
		mAddressMode(aAddressMode),
		mTexRefSetFlag(aTexRefSetFlag),
		mSizeInElements(aArray->GetArrayDescriptor().Width),
		mNumChannels(aArray->GetArrayDescriptor().NumChannels),
		mArrayFormat(aArray->GetArrayDescriptor().Format),
		mFilterMode(aFilterMode),
		mCleanUp(false)
	{
		cudaSafeCall(cuModuleGetTexRef(&mTexref, mKernel->GetCUmodule(), mName.c_str()));
		cudaSafeCall(cuTexRefSetAddressMode(mTexref, 0, mAddressMode));
		cudaSafeCall(cuTexRefSetFilterMode(mTexref, mFilterMode));
		cudaSafeCall(cuTexRefSetFlags(mTexref, mTexRefSetFlag));
		cudaSafeCall(cuTexRefSetFormat(mTexref, mArrayFormat, mNumChannels));

		mChannelSize = Cuda::GetChannelSize(mArrayFormat);
		mSizeInBytes = mSizeInElements * mChannelSize * mNumChannels;

		mArray = aArray;

		cudaSafeCall(cuTexRefSetArray(mTexref, mArray->GetCUarray(), mTexRefSetFlag));
		cudaSafeCall(cuParamSetTexRef(mKernel->GetCUfunction(), CU_PARAM_TR_DEFAULT, mTexref));

	}

	CudaTextureArray1D::~CudaTextureArray1D()
	{
		if (mCleanUp && mArray)
		{
			delete mArray;
			mArray = NULL;
		}
	}


	CUtexref CudaTextureArray1D::GetTextureReference()
	{
		return mTexref;
	}

	CUfilter_mode CudaTextureArray1D::GetFilterMode()
	{
		return mFilterMode;
	}

	uint CudaTextureArray1D::GetTexRefSetFlags()
	{
		return mTexRefSetFlag;
	}

	CUaddress_mode CudaTextureArray1D::GetAddressMode()
	{
		return mAddressMode;
	}

	CUarray_format CudaTextureArray1D::GetArrayFormat()
	{
		return mArrayFormat;
	}

	size_t CudaTextureArray1D::GetSizeInElements()
	{
		return mSizeInElements;
	}

	uint CudaTextureArray1D::GetChannelSize()
	{
		return mChannelSize;
	}

	size_t CudaTextureArray1D::GetSizeInBytes()
	{
		return mSizeInBytes;
	}

	uint CudaTextureArray1D::GetNumChannels()
	{
		return mNumChannels;
	}

	string CudaTextureArray1D::GetName()
	{
		return mName;
	}

	CudaKernel* CudaTextureArray1D::GetCudaKernel()
	{
		return mKernel;
	}

	CudaArray1D* CudaTextureArray1D::GetArray()
	{
		return mArray;
	}



	CudaTextureArray2D::CudaTextureArray2D(CudaKernel* aKernel, string aTexName, CUaddress_mode aAddressMode0, CUaddress_mode aAddressMode1,
		CUfilter_mode aFilterMode, uint aTexRefSetFlag, CUarray_format aFormat, size_t aWidth, size_t aHeight, uint aNumChannels)
		: mKernel(aKernel),
		mName(aTexName),
		mAddressMode0(aAddressMode0),
		mAddressMode1(aAddressMode1),
		mTexRefSetFlag(aTexRefSetFlag),
		mHeight(aHeight),
		mWidth(aWidth),
		mNumChannels(aNumChannels),
		mArrayFormat(aFormat),
		mFilterMode(aFilterMode),
		mCleanUp(true)
	{
		cudaSafeCall(cuModuleGetTexRef(&mTexref, mKernel->GetCUmodule(), mName.c_str()));
		cudaSafeCall(cuTexRefSetAddressMode(mTexref, 0, mAddressMode0));
		cudaSafeCall(cuTexRefSetAddressMode(mTexref, 1, mAddressMode1));
		cudaSafeCall(cuTexRefSetFilterMode(mTexref, mFilterMode));
		cudaSafeCall(cuTexRefSetFlags(mTexref, mTexRefSetFlag));
		cudaSafeCall(cuTexRefSetFormat(mTexref, mArrayFormat, mNumChannels));

		mChannelSize = Cuda::GetChannelSize(mArrayFormat);
		mSizeInBytes = mHeight * mWidth * mChannelSize * mNumChannels;

		mArray = new CudaArray2D(mArrayFormat, mWidth, mHeight, mNumChannels);

		cudaSafeCall(cuTexRefSetArray(mTexref, mArray->GetCUarray(), mTexRefSetFlag));
		cudaSafeCall(cuParamSetTexRef(mKernel->GetCUfunction(), CU_PARAM_TR_DEFAULT, mTexref));
	}


	CudaTextureArray2D::CudaTextureArray2D(CudaKernel* aKernel, string aTexName, CUaddress_mode aAddressMode0, CUaddress_mode aAddressMode1,
		CUfilter_mode aFilterMode, uint aTexRefSetFlag, CudaArray2D* aArray)
		: mKernel(aKernel),
		mName(aTexName),
		mAddressMode0(aAddressMode0),
		mAddressMode1(aAddressMode1),
		mTexRefSetFlag(aTexRefSetFlag),
		mHeight(aArray->GetArrayDescriptor().Height),
		mWidth(aArray->GetArrayDescriptor().Width),
		mNumChannels(aArray->GetArrayDescriptor().NumChannels),
		mArrayFormat(aArray->GetArrayDescriptor().Format),
		mFilterMode(aFilterMode),
		mCleanUp(false)
	{
		cudaSafeCall(cuModuleGetTexRef(&mTexref, mKernel->GetCUmodule(), mName.c_str()));
		cudaSafeCall(cuTexRefSetAddressMode(mTexref, 0, mAddressMode0));
		cudaSafeCall(cuTexRefSetAddressMode(mTexref, 1, mAddressMode1));
		cudaSafeCall(cuTexRefSetFilterMode(mTexref, mFilterMode));
		cudaSafeCall(cuTexRefSetFlags(mTexref, mTexRefSetFlag));
		cudaSafeCall(cuTexRefSetFormat(mTexref, mArrayFormat, mNumChannels));

		mChannelSize = Cuda::GetChannelSize(mArrayFormat);
		mSizeInBytes = mHeight * mWidth * mChannelSize * mNumChannels;

		mArray = aArray;

		cudaSafeCall(cuTexRefSetArray(mTexref, mArray->GetCUarray(), mTexRefSetFlag));
		cudaSafeCall(cuParamSetTexRef(mKernel->GetCUfunction(), CU_PARAM_TR_DEFAULT, mTexref));
	}

	CudaTextureArray2D::~CudaTextureArray2D()
	{
		if (mCleanUp && mArray)
		{
			delete mArray;
			mArray = NULL;
		}
	}


	CUtexref CudaTextureArray2D::GetTextureReference()
	{
		return mTexref;
	}

	CUfilter_mode CudaTextureArray2D::GetFilterMode()
	{
		return mFilterMode;
	}

	uint CudaTextureArray2D::GetTexRefSetFlags()
	{
		return mTexRefSetFlag;
	}

	CUaddress_mode* CudaTextureArray2D::GetAddressModes()
	{
		CUaddress_mode* retVal = new CUaddress_mode[2];
		retVal[0] = mAddressMode0;
		retVal[1] = mAddressMode1;
		return retVal;
	}

	CUarray_format CudaTextureArray2D::GetArrayFormat()
	{
		return mArrayFormat;
	}

	size_t CudaTextureArray2D::GetHeight()
	{
		return mHeight;
	}

	size_t CudaTextureArray2D::GetWidth()
	{
		return mWidth;
	}

	uint CudaTextureArray2D::GetChannelSize()
	{
		return mChannelSize;
	}

	size_t CudaTextureArray2D::GetSizeInBytes()
	{
		return mSizeInBytes;
	}

	uint CudaTextureArray2D::GetNumChannels()
	{
		return mNumChannels;
	}

	string CudaTextureArray2D::GetName()
	{
		return mName;
	}

	CudaKernel* CudaTextureArray2D::GetCudaKernel()
	{
		return mKernel;
	}

	CudaArray2D* CudaTextureArray2D::GetArray()
	{
		return mArray;
	}



	CudaTextureLinearPitched2D::CudaTextureLinearPitched2D(CudaKernel* aKernel, string aTexName, CUaddress_mode aAddressMode0, CUaddress_mode aAddressMode1,
		CUfilter_mode aFilterMode, uint aTexRefSetFlag, CUarray_format aFormat, size_t aWidth, size_t aHeight, uint aNumChannels)
		: mKernel(aKernel),
		mName(aTexName),
		mAddressMode0(aAddressMode0),
		mAddressMode1(aAddressMode1),
		mTexRefSetFlag(aTexRefSetFlag),
		mHeight(aHeight),
		mWidth(aWidth),
		mNumChannels(aNumChannels),
		mArrayFormat(aFormat),
		mFilterMode(aFilterMode),
		mCleanUp(true)
	{
		cudaSafeCall(cuModuleGetTexRef(&mTexref, mKernel->GetCUmodule(), mName.c_str()));
		cudaSafeCall(cuTexRefSetAddressMode(mTexref, 0, mAddressMode0));
		cudaSafeCall(cuTexRefSetAddressMode(mTexref, 1, mAddressMode1));
		cudaSafeCall(cuTexRefSetFilterMode(mTexref, mFilterMode));
		cudaSafeCall(cuTexRefSetFlags(mTexref, mTexRefSetFlag));
		cudaSafeCall(cuTexRefSetFormat(mTexref, mArrayFormat, mNumChannels));

		mChannelSize = Cuda::GetChannelSize(mArrayFormat);
		mSizeInBytes = mHeight * mWidth * mChannelSize * mNumChannels;

		mDevVar = new CudaPitchedDeviceVariable(mWidth * mChannelSize * mNumChannels, mHeight, mChannelSize * mNumChannels);

		CUDA_ARRAY_DESCRIPTOR arraydesc;
		memset(&arraydesc, 0, sizeof(arraydesc));
		arraydesc.Format = mArrayFormat;
		arraydesc.Height = mHeight;
		arraydesc.NumChannels = mNumChannels;
		arraydesc.Width = mWidth;

		cudaSafeCall(cuTexRefSetAddress2D(mTexref, &arraydesc, mDevVar->GetDevicePtr(), mDevVar->GetPitch()));
		cudaSafeCall(cuParamSetTexRef(mKernel->GetCUfunction(), CU_PARAM_TR_DEFAULT, mTexref));
	}


	CudaTextureLinearPitched2D::CudaTextureLinearPitched2D(CudaKernel* aKernel, string aTexName, CUaddress_mode aAddressMode0, CUaddress_mode aAddressMode1,
		CUfilter_mode aFilterMode, uint aTexRefSetFlag, CudaPitchedDeviceVariable* aDevVar, CUarray_format aFormat, uint aNumChannels)
		: mKernel(aKernel),
		mName(aTexName),
		mAddressMode0(aAddressMode0),
		mAddressMode1(aAddressMode1),
		mTexRefSetFlag(aTexRefSetFlag),
		mHeight(aDevVar->GetHeight()),
		mWidth(aDevVar->GetWidth()),
		mNumChannels(aNumChannels),
		mArrayFormat(aFormat),
		mFilterMode(aFilterMode),
		mCleanUp(false)
	{
		cudaSafeCall(cuModuleGetTexRef(&mTexref, mKernel->GetCUmodule(), mName.c_str()));
		cudaSafeCall(cuTexRefSetAddressMode(mTexref, 0, mAddressMode0));
		cudaSafeCall(cuTexRefSetAddressMode(mTexref, 1, mAddressMode1));
		cudaSafeCall(cuTexRefSetFilterMode(mTexref, mFilterMode));
		cudaSafeCall(cuTexRefSetFlags(mTexref, mTexRefSetFlag));
		cudaSafeCall(cuTexRefSetFormat(mTexref, mArrayFormat, mNumChannels));

		mChannelSize = Cuda::GetChannelSize(mArrayFormat);
		mSizeInBytes = mHeight * mWidth * mChannelSize * mNumChannels;

		mDevVar = aDevVar;

		CUDA_ARRAY_DESCRIPTOR arraydesc;
		memset(&arraydesc, 0, sizeof(arraydesc));
		arraydesc.Format = mArrayFormat;
		arraydesc.Height = mHeight;
		arraydesc.NumChannels = mNumChannels;
		arraydesc.Width = mWidth;

		cudaSafeCall(cuTexRefSetAddress2D(mTexref, &arraydesc, mDevVar->GetDevicePtr(), mDevVar->GetPitch()));
		cudaSafeCall(cuParamSetTexRef(mKernel->GetCUfunction(), CU_PARAM_TR_DEFAULT, mTexref));
	}

	CudaTextureLinearPitched2D::~CudaTextureLinearPitched2D()
	{
		if (mCleanUp && mDevVar)
		{
			delete mDevVar;
			mDevVar = NULL;
		}
	}


	CUtexref CudaTextureLinearPitched2D::GetTextureReference()
	{
		return mTexref;
	}

	CUfilter_mode CudaTextureLinearPitched2D::GetFilterMode()
	{
		return mFilterMode;
	}

	uint CudaTextureLinearPitched2D::GetTexRefSetFlags()
	{
		return mTexRefSetFlag;
	}

	CUaddress_mode* CudaTextureLinearPitched2D::GetAddressModes()
	{
		CUaddress_mode* retVal = new CUaddress_mode[2];
		retVal[0] = mAddressMode0;
		retVal[1] = mAddressMode1;
		return retVal;
	}

	CUarray_format CudaTextureLinearPitched2D::GetArrayFormat()
	{
		return mArrayFormat;
	}

	size_t CudaTextureLinearPitched2D::GetHeight()
	{
		return mHeight;
	}

	size_t CudaTextureLinearPitched2D::GetWidth()
	{
		return mWidth;
	}

	uint CudaTextureLinearPitched2D::GetChannelSize()
	{
		return mChannelSize;
	}

	size_t CudaTextureLinearPitched2D::GetSizeInBytes()
	{
		return mSizeInBytes;
	}

	uint CudaTextureLinearPitched2D::GetNumChannels()
	{
		return mNumChannels;
	}

	string CudaTextureLinearPitched2D::GetName()
	{
		return mName;
	}

	CudaKernel* CudaTextureLinearPitched2D::GetCudaKernel()
	{
		return mKernel;
	}

	CudaPitchedDeviceVariable* CudaTextureLinearPitched2D::GetDeviceVariable()
	{
		return mDevVar;
	}





	CudaTextureArray3D::CudaTextureArray3D(CudaKernel* aKernel, string aTexName, CUaddress_mode aAddressMode0, CUaddress_mode aAddressMode1,
		CUaddress_mode aAddressMode2, CUfilter_mode aFilterMode, uint aTexRefSetFlag, CUarray_format aFormat,
		size_t aWidth, size_t aHeight, size_t aDepth, uint aNumChannels)
		: mKernel(aKernel),
		mName(aTexName),
		mAddressMode0(aAddressMode0),
		mAddressMode1(aAddressMode1),
		mAddressMode2(aAddressMode2),
		mTexRefSetFlag(aTexRefSetFlag),
		mHeight(aHeight),
		mWidth(aWidth),
		mDepth(aDepth),
		mNumChannels(aNumChannels),
		mArrayFormat(aFormat),
		mFilterMode(aFilterMode),
		mCleanUp(true)
	{
		cudaSafeCall(cuModuleGetTexRef(&mTexref, mKernel->GetCUmodule(), mName.c_str()));
		cudaSafeCall(cuTexRefSetAddressMode(mTexref, 0, mAddressMode0));
		cudaSafeCall(cuTexRefSetAddressMode(mTexref, 1, mAddressMode1));
		cudaSafeCall(cuTexRefSetAddressMode(mTexref, 2, mAddressMode2));
		cudaSafeCall(cuTexRefSetFilterMode(mTexref, mFilterMode));
		cudaSafeCall(cuTexRefSetFlags(mTexref, mTexRefSetFlag));
		cudaSafeCall(cuTexRefSetFormat(mTexref, mArrayFormat, mNumChannels));

		mChannelSize = Cuda::GetChannelSize(mArrayFormat);
		mSizeInBytes = mDepth * mHeight * mWidth * mChannelSize * mNumChannels;

		mArray = new CudaArray3D(mArrayFormat, mWidth, mHeight, mDepth, mNumChannels);

		cudaSafeCall(cuTexRefSetArray(mTexref, mArray->GetCUarray(), mTexRefSetFlag));
		cudaSafeCall(cuParamSetTexRef(mKernel->GetCUfunction(), CU_PARAM_TR_DEFAULT, mTexref));
	}


	CudaTextureArray3D::CudaTextureArray3D(CudaKernel* aKernel, string aTexName, CUaddress_mode aAddressMode0, CUaddress_mode aAddressMode1,
		CUaddress_mode aAddressMode2, CUfilter_mode aFilterMode, uint aTexRefSetFlag, CudaArray3D* aArray)
		: mKernel(aKernel),
		mName(aTexName),
		mAddressMode0(aAddressMode0),
		mAddressMode1(aAddressMode1),
		mAddressMode2(aAddressMode2),
		mTexRefSetFlag(aTexRefSetFlag),
		mHeight(aArray->GetArrayDescriptor().Height),
		mWidth(aArray->GetArrayDescriptor().Width),
		mDepth(aArray->GetArrayDescriptor().Depth),
		mNumChannels(aArray->GetArrayDescriptor().NumChannels),
		mArrayFormat(aArray->GetArrayDescriptor().Format),
		mFilterMode(aFilterMode),
		mCleanUp(false)
	{
		cudaSafeCall(cuModuleGetTexRef(&mTexref, mKernel->GetCUmodule(), mName.c_str()));
		cudaSafeCall(cuTexRefSetAddressMode(mTexref, 0, mAddressMode0));
		cudaSafeCall(cuTexRefSetAddressMode(mTexref, 1, mAddressMode1));
		cudaSafeCall(cuTexRefSetAddressMode(mTexref, 2, mAddressMode2));
		cudaSafeCall(cuTexRefSetFilterMode(mTexref, mFilterMode));
		cudaSafeCall(cuTexRefSetFlags(mTexref, mTexRefSetFlag));
		cudaSafeCall(cuTexRefSetFormat(mTexref, mArrayFormat, mNumChannels));

		mChannelSize = Cuda::GetChannelSize(mArrayFormat);
		mSizeInBytes = mDepth * mHeight * mWidth * mChannelSize * mNumChannels;

		mArray = aArray;

		cudaSafeCall(cuTexRefSetArray(mTexref, mArray->GetCUarray(), CU_TRSA_OVERRIDE_FORMAT));
		cudaSafeCall(cuParamSetTexRef(mKernel->GetCUfunction(), CU_PARAM_TR_DEFAULT, mTexref));
	}

	CudaTextureArray3D::~CudaTextureArray3D()
	{
		if (mCleanUp && mArray)
		{
			delete mArray;
			mArray = NULL;
		}
	}


	CUtexref CudaTextureArray3D::GetTextureReference()
	{
		return mTexref;
	}

	CUfilter_mode CudaTextureArray3D::GetFilterMode()
	{
		return mFilterMode;
	}

	uint CudaTextureArray3D::GetTexRefSetFlags()
	{
		return mTexRefSetFlag;
	}

	CUaddress_mode* CudaTextureArray3D::GetAddressModes()
	{
		CUaddress_mode* retVal = new CUaddress_mode[3];
		retVal[0] = mAddressMode0;
		retVal[1] = mAddressMode1;
		retVal[2] = mAddressMode2;
		return retVal;
	}

	CUarray_format CudaTextureArray3D::GetArrayFormat()
	{
		return mArrayFormat;
	}

	size_t CudaTextureArray3D::GetHeight()
	{
		return mHeight;
	}

	size_t CudaTextureArray3D::GetWidth()
	{
		return mWidth;
	}

	size_t CudaTextureArray3D::GetDepth()
	{
		return mDepth;
	}

	uint CudaTextureArray3D::GetChannelSize()
	{
		return mChannelSize;
	}

	size_t CudaTextureArray3D::GetSizeInBytes()
	{
		return mSizeInBytes;
	}

	uint CudaTextureArray3D::GetNumChannels()
	{
		return mNumChannels;
	}

	string CudaTextureArray3D::GetName()
	{
		return mName;
	}

	CudaKernel* CudaTextureArray3D::GetCudaKernel()
	{
		return mKernel;
	}

	CudaArray3D* CudaTextureArray3D::GetArray()
	{
		return mArray;
	}
}
