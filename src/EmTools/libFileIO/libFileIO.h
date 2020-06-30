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


#include "../Basics/Default.h"

#ifdef _USE_WINDOWS_COMPILER_SETTINGS
#ifdef LIBFILEIO_EXPORTS
#define LIBFILEIO_API __declspec(dllexport)
#else
#define LIBFILEIO_API __declspec(dllimport)
#endif
#else
#ifdef LIBFILEIO_EXPORTS
#define LIBFILEIO_API __attribute__((visibility("default")))
#else
#define LIBFILEIO_API
#endif
#endif

#include "../FileIO/FileIO.h"
#include "../FileIO/SingleFrame.h"
#include "../FileIO/MovieStack.h"

extern "C"
LIBFILEIO_API int CanReadAsSingleFrame(char* aFilename, int* aWidth, int* aHeight, DataType_enum* aDataType);

extern "C"
LIBFILEIO_API int GetImageData(char* aFilename, void* aData, int* needsFlipOnAxis, int* isPlanar);

extern "C"
LIBFILEIO_API int CanReadAsMovieStack(char* aFilename, int* aWidth, int* aHeight, int* aFrameCount, DataType_enum* aDataType);

extern "C"
LIBFILEIO_API int OpenMovieStack(char* aFilename, void** aHandle, int* needsFlipOnAxis, float* aPixelsize);

extern "C"
LIBFILEIO_API int GetData(void* aHandle, int idx, void** aData);

extern "C"
LIBFILEIO_API int CloseMovieStack(void* aHandle);

extern "C"
LIBFILEIO_API int CanWriteAsTIFF(int aWidth, int aHeight, DataType_enum aDataType);

extern "C"
LIBFILEIO_API int WriteTIFF(char* aFileName, int aDimX, int aDimY, float aPixelSize, DataType_enum aDatatype, void * aData);

extern "C"
LIBFILEIO_API int CanWriteAsEM(int aWidth, int aHeight, int aDepth, DataType_enum aDataType);

extern "C"
LIBFILEIO_API int WriteEM(char* aFileName, int aDimX, int aDimY, int aDimZ, float aPixelSize, DataType_enum aDatatype, void * aData);


