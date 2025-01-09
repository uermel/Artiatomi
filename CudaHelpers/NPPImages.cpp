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


#include "NPPImages.h"

using namespace Cuda;

NPPImage_32fC1::NPPImage_32fC1(int aWidth, int aHeight) :
	NPPImageBase(0, aWidth, aHeight, sizeof(float), 0, 1, true)
{
	_devPtr = (CUdeviceptr)nppiMalloc_32f_C1(aWidth, aHeight, &_pitch);
	if (_devPtr == 0)
	{
		nppSafeCall(NPP_INVALID_DEVICE_POINTER_ERROR);
	}
	_devPtrRoi = _devPtr;
}

Npp32f * NPPImage_32fC1::GetPtr()
{
	return (Npp32f*)_devPtr;
}

Npp32f * NPPImage_32fC1::GetPtrRoi()
{
	return (Npp32f*)_devPtrRoi;
}

NPPImage_32fcC1::NPPImage_32fcC1(int aWidth, int aHeight) : 
	NPPImageBase(0, aWidth, aHeight, sizeof(cuComplex), 0, 1, true)
{
	_devPtr = (CUdeviceptr)nppiMalloc_32fc_C1(aWidth, aHeight, &_pitch);
	if (_devPtr == 0)
	{
		nppSafeCall(NPP_INVALID_DEVICE_POINTER_ERROR);
	}
	_devPtrRoi = _devPtr;
}

Npp32fc * NPPImage_32fcC1::GetPtr()
{
	return (Npp32fc*)_devPtr;
}

Npp32fc * NPPImage_32fcC1::GetPtrRoi()
{
	return (Npp32fc*)_devPtrRoi;
}

NPPImage_8uC1::NPPImage_8uC1(int aWidth, int aHeight) : 
	NPPImageBase(0, aWidth, aHeight, sizeof(unsigned char), 0, 1, true)
{
	_devPtr = (CUdeviceptr)nppiMalloc_8u_C1(aWidth, aHeight, &_pitch);
	if (_devPtr == 0)
	{
		nppSafeCall(NPP_INVALID_DEVICE_POINTER_ERROR);
	}
	_devPtrRoi = _devPtr;
}

Npp8u * NPPImage_8uC1::GetPtr()
{
	return (Npp8u*) _devPtr;
}

Npp8u * NPPImage_8uC1::GetPtrRoi()
{
	return (Npp8u*) _devPtrRoi;
}

NPPImage_16uC1::NPPImage_16uC1(int aWidth, int aHeight) :
	NPPImageBase(0, aWidth, aHeight, sizeof(unsigned short), 0, 1, true)
{
	_devPtr = (CUdeviceptr)nppiMalloc_16u_C1(aWidth, aHeight, &_pitch);
	if (_devPtr == 0)
	{
		nppSafeCall(NPP_INVALID_DEVICE_POINTER_ERROR);
	}
	_devPtrRoi = _devPtr;
}

Npp16u * NPPImage_16uC1::GetPtr()
{
	return (Npp16u*)_devPtr;
}

Npp16u * NPPImage_16uC1::GetPtrRoi()
{
	return (Npp16u*)_devPtrRoi;
}

NPPImage_16sC1::NPPImage_16sC1(int aWidth, int aHeight) :
	NPPImageBase(0, aWidth, aHeight, sizeof(unsigned short), 0, 1, true)
{
	_devPtr = (CUdeviceptr)nppiMalloc_16s_C1(aWidth, aHeight, &_pitch);
	if (_devPtr == 0)
	{
		nppSafeCall(NPP_INVALID_DEVICE_POINTER_ERROR);
	}
	_devPtrRoi = _devPtr;
}

Npp16s * NPPImage_16sC1::GetPtr()
{
	return (Npp16s*)_devPtr; 
}

Npp16s * NPPImage_16sC1::GetPtrRoi()
{
	return (Npp16s*)_devPtrRoi; 
}
