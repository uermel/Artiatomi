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


#ifndef SERFILE_H
#define SERFILE_H

#include "../Basics/Default.h"
#include "FileReader.h"


//! Types of pixel in image
enum SerDataType_Enum : ushort
{
	//! unsigned bytes
	SERTYPE_UI1 = 1,
	//! unsigned short integer
	SERTYPE_UI2 = 2,
	//! unsigned int
	SERTYPE_UI4 = 3,
	//! signed bytes
	SERTYPE_I1 = 4,
	//! signed short integers
	SERTYPE_I2 = 5,
	//! signed integers
	SERTYPE_I4 = 6,
	//! float
	SERTYPE_F4 = 7,
	//! double
	SERTYPE_F8 = 8,
	//! complex floats
	SERTYPE_CF4 = 9,
	//! complex doubles
	SERTYPE_CF8 = 10
};



//! Header of a SER file
#ifdef _USE_WINDOWS_COMPILER_SETTINGS
#pragma pack(push)
#pragma pack(1)
#endif
struct struct_SerHeader
{
	//! Byte ordering indicator - value 0x4949 ('II') indicates little-endian
	ushort ByteOrder;
	//! Series identification word - value 0x0197 indicates ES Vision Series Data File
	ushort SeriesID;
	//! Version number word - indicates the version of the Series Data File. Should be 0x0210 or 0x220
	ushort SeriesVersion;

	//! Indicates the type of data object stored at each element in the series
	uint DataTypeID;

	//! Indicates the type of tab stored at each element in the series.
	uint TagypeID;

	//! Indicates the total number of data elements and tags referred to by the Dimension Array. Equals the product of the dimension sizes, and corresponds to the total number of addressable indices in the series.
	uint TotalNumberElements;

	//! Indicates the number of valid Data elements and Tags in the series data file.
	uint ValidNumberElements;

	//! Indicates the absolute offset (in bytes) in the Series Data File of the beginning of the Data Offset Array.
	ulong64 OffsetArrayOffset;

	//! Indicates the number of dimensions of the Series Data File. This indicates the number of dimensions of the indices, NOT the number of dimensions of the data.
	uint NumberDimensions;

} 
#if defined(_USE_LINUX_COMPILER_SETTINGS) || defined(_USE_APPLE_COMPILER_SETTINGS)
__attribute__((packed))
#endif
;
typedef struct struct_SerHeader SerHeader;

//! Dimension header of a SER file
struct struct_SerDimensionArray
{
	//! Indicates the number of elements in this dimension.
	uint DimensionSize;
	//! Indicates the calibration value at element CalibrationElement.
	double CalibrationOffset;
	//! Indicates the calibration delta between elements of the series.
	double CalibrationDelta;

	//! Indicates the element in the series which has a calibration value of CalibrationOffset.
	uint CalibrationElement;

	//! Indicates the length of the Description string
	uint DescriptionLength;

	//! String of length DescriptionLength which describes this dimension.
	char* Description;

	//! Indicates the length of the Units string.
	uint UnitsLength;

	//! String of length UnitsLength which is the name of units in this dimension.
	char* Units;
}
#if defined(_USE_LINUX_COMPILER_SETTINGS) || defined(_USE_APPLE_COMPILER_SETTINGS)
__attribute__((packed))
#endif
;
typedef struct struct_SerDimensionArray SerDimensionArray;


//! 2D element format header of a SER file
struct struct_Ser2DElementFormat
{
	//! Indicates the calibration value at element CalibrationElement in the X - direction.
	double CalibrationOffsetX;
	//! Indicates the calibration delta between elements of the array in the X-direction.
	double CalibrationDeltaX;
	//! Indicates the element in the array in the X-direction which has a calibration value of CalibrationOffset.
	uint CalibrationElementX;

	//! Indicates the calibration value at element CalibrationElement in the Y-direction.
	double CalibrationOffsetY;
	//! Indicates the calibration delta between elements of the array in the Y-direction.
	double CalibrationDeltaY;
	//! Indicates the element in the array in the Y-direction which has a calibration value of CalibrationOffset.
	uint CalibrationElementY;

	//! Indicates the type of data stored at each element of the array.
	SerDataType_Enum DataType;

	//! Indicates the number of elements in the array in the X-direction (the array width).
	uint ArraySizeX;

	//! Indicates the number of elements in the array in the Y - direction(the array height).
	uint ArraySizeY;

	//! The actual data values.
	void* Data;
}
#if defined(_USE_LINUX_COMPILER_SETTINGS) || defined(_USE_APPLE_COMPILER_SETTINGS)
__attribute__((packed))
#endif
;
#ifdef _USE_WINDOWS_COMPILER_SETTINGS
#pragma pack(pop)
#endif
typedef struct struct_Ser2DElementFormat Ser2DElement;

//!  SERFile represents a FEI *.ser file in memory and maps contained information to the default internal Image format. 
/*!
SerFile gives access to header infos, image data.
\author Michael Kunz
\date   September 2016
\version 1.0
*/
class SERFile : public FileReader
{
public:
	SERFile(std::string filename);
	~SERFile();

	//! Opens the file File#mFileName and reads the entire content.
	/*!
	\throw FileIOException
	*/
	bool OpenAndRead();

	//! Opens the file File#mFileName and reads only the file header.
	/*!
	\throw FileIOException
	*/
	bool OpenAndReadHeader();

	//! Converts from SER data type enum to internal data type
	DataType_enum GetDataType();

	//! Returns the size of the data block. If the header is not yet read, it will return 0.
	size_t GetDataSize();

	//! Returns a reference to the inner SER file header.
	SerHeader& GetFileHeader();

	//! Returns a reference to the inner SER dimension header.
	SerDimensionArray& GetDimensionHeader();

	//! Returns a reference to the inner SER element header.
	Ser2DElement& GetElementHeader();

	//! Returns the inner data pointer.
	void* GetData();

	float GetPixelSizeX();
	float GetPixelSizeY();
private:
	SerHeader _fileHeader;
	SerDimensionArray _dimensionArray;
	Ser2DElement _element;
	uint _GetDataTypeSize(SerDataType_Enum aDataType);
	void InverseEndianessHeader();
	void InverseEndianessDimHeader();
	void InverseEndianessElementHeader();
	void InverseEndianessData();
};

#endif