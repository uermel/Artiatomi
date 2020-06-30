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


// libFileIO.cpp : Definiert die exportierten Funktionen für die DLL-Anwendung.
//

#include "stdafx.h"
#include "libFileIO.h"
#include <cstring>


// Dies ist das Beispiel einer exportierten Funktion.
LIBFILEIO_API int CanReadAsSingleFrame(char* aFilename, int* aWidth, int* aHeight, DataType_enum* aDataType)
{
	FileType_enum ft;
	if (SingleFrame::CanReadFile(aFilename, ft, *aWidth, *aHeight, *aDataType))
	{
		return 1;
	}
	else
		return 0;
}

LIBFILEIO_API int GetImageData(char* aFilename, void* aData, int* needsFlipOnAxis, int* isPlanar)
{
	SingleFrame sf(aFilename);

	void* data = sf.GetData();

	size_t size = GetDataTypeSize(sf.GetFileDataType()) * sf.GetWidth() * sf.GetHeight();

	memcpy(aData, data, size);

	*needsFlipOnAxis = sf.NeedsFlipOnYAxis() ? 1 : 0;
	*isPlanar = sf.GetIsPlanar() ? 1 : 0;
	return 1;
}
	
LIBFILEIO_API int CanReadAsMovieStack(char* aFilename, int* aWidth, int* aHeight, int* aFrameCount, DataType_enum* aDataType)
{
	FileType_enum ft;
	if (MovieStack::CanReadFile(aFilename, ft, *aWidth, *aHeight, *aFrameCount, *aDataType))
	{
		return 1;
	}
	else
		return 0;
}

LIBFILEIO_API int OpenMovieStack(char* aFilename, void** aHandle, int* needsFlipOnAxis, float* aPixelsize)
{
	MovieStack* ms = new MovieStack(aFilename);

	*needsFlipOnAxis = ms->NeedsFlipOnYAxis() ? 1 : 0;
	*aPixelsize = ms->GetPixelSize();
	*aHandle = ms;
	return 1;
}

LIBFILEIO_API int GetData(void * aHandle, int idx, void** aData)
{
	MovieStack* ms = (MovieStack*)aHandle;
	*aData = ms->GetData(idx);
	return 1;
}

LIBFILEIO_API int CloseMovieStack(void * aHandle)
{
	MovieStack* ms = (MovieStack*)aHandle;
	delete ms;
	return 1;
}

LIBFILEIO_API int CanWriteAsTIFF(int aWidth, int aHeight, DataType_enum aDataType)
{
	int res = 0;
	if (TIFFFile::CanWriteAsTIFF(aWidth, aHeight, aDataType))
		res = 1;
	return res;
}

LIBFILEIO_API int WriteTIFF(char* aFileName, int aDimX, int aDimY, float aPixelSize, DataType_enum aDatatype, void * aData)
{
	int res = 0;
	if (TIFFFile::WriteTIFF(aFileName, aDimX, aDimY, aPixelSize, aDatatype, aData))
		res = 1;
	return res;
}

LIBFILEIO_API int CanWriteAsEM(int aWidth, int aHeight, int aDepth, DataType_enum aDataType)
{
	int res = 0;
	if (EmFile::CanWriteAsEM(aWidth, aHeight, aDepth, aDataType))
		res = 1;
	return res;
}

LIBFILEIO_API int WriteEM(char* aFileName, int aDimX, int aDimY, int aDimZ, float aPixelSize, DataType_enum aDatatype, void * aData)
{
	if (!EmFile::InitHeader(aFileName, aDimX, aDimY, aDimZ, aPixelSize, aDatatype))
		return 0;

	if (!EmFile::WriteRawData(aFileName, aData, (size_t)aDimX * (size_t)aDimY * (size_t)aDimZ * GetDataTypeSize(aDatatype)))
		return 0;
	return 1;
}
