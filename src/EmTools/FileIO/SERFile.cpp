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


#include "SERFile.h"
#include <string.h>
#include <math.h>

SERFile::SERFile(string aFileName)
	: FileReader(aFileName)
{
	memset(&_fileHeader, 0, sizeof(_fileHeader));
	memset(&_dimensionArray, 0, sizeof(_dimensionArray));
	memset(&_element, 0, sizeof(_element));
}

SERFile::~SERFile()
{
	if (_dimensionArray.Description)
		delete[] _dimensionArray.Description;
	_dimensionArray.Description = NULL;

	if (_dimensionArray.Units)
		delete[] _dimensionArray.Units;
	_dimensionArray.Units = NULL;

	// Delete existing data
	if (_element.Data)
		delete[] (char *)_element.Data;

	_element.Data = NULL;
}

uint SERFile::_GetDataTypeSize(SerDataType_Enum aDataType)
{
	switch (aDataType)
	{
	case SERTYPE_UI1:
		return 1;
	case SERTYPE_UI2:
		return 2;
	case SERTYPE_UI4:
		return 4;
	case SERTYPE_I1:
		return 1;
	case SERTYPE_I2:
		return 2;
	case SERTYPE_I4:
		return 4;
	case SERTYPE_F4:
		return 4;
	case SERTYPE_F8:
		return 8;
	case SERTYPE_CF4:
		return 8;
	case SERTYPE_CF8:
		return 16;
	}
	return 0;
}

bool SERFile::OpenAndRead()
{
	bool res = FileReader::OpenRead();

	if (!res)
	{
		FileReader::CloseRead();
		//Allocations, if any, will be freed in destructor, just leave...
		throw FileIOException(FileReader::mFileName, "Cannot open file for reading.");
	}

	memset(&_fileHeader, 0, sizeof(_fileHeader));
	Read((char*)&_fileHeader, sizeof(_fileHeader));

	bool needEndianessInverse = _fileHeader.ByteOrder != 0x4949;
	if (needEndianessInverse)
		InverseEndianessHeader();

	if (_fileHeader.SeriesVersion != 0x220)
	{
		FileReader::CloseRead();
		//Allocations, if any, will be freed in destructor, just leave...
		throw FileIOException(FileReader::mFileName, "Cannot read old file version.");
	}

	Read((char*)&_dimensionArray, 4 + 8 + 8 + 4 + 4);
	if (needEndianessInverse)
	{
		InverseEndianessDimHeader();
		uint tempui = _dimensionArray.DescriptionLength;
		FileReader::Endian_swap(tempui);
		_dimensionArray.DescriptionLength = tempui;
	}
	if (_dimensionArray.DescriptionLength > 0)
	{
		_dimensionArray.Description = new char[_dimensionArray.DescriptionLength];
		Read((char*)_dimensionArray.Description, _dimensionArray.DescriptionLength);
	}
	Read((char*)&_dimensionArray.UnitsLength, 4);
	if (needEndianessInverse)
	{
		uint tempui = _dimensionArray.UnitsLength;
		FileReader::Endian_swap(tempui);
		_dimensionArray.UnitsLength = tempui;
	}
	if (_dimensionArray.UnitsLength > 0)
	{
		_dimensionArray.Units = new char[_dimensionArray.UnitsLength];
		Read((char*)_dimensionArray.Units, _dimensionArray.UnitsLength);
	}
	//size_t t1 = FileReader::Tell();
	FileReader::mFile->seekg(_fileHeader.OffsetArrayOffset, ios_base::beg);

	ulong64 offset;
	Read((char*)&offset, 8);
	if (needEndianessInverse)
	{
		FileReader::Endian_swap(offset);
	}
	//size_t t2 = FileReader::Tell();
	FileReader::mFile->seekg(offset, ios_base::beg);

	Read((char*)&_element, sizeof(_element) - sizeof(size_t));

	/*double t4 = _element.CalibrationDeltaX * pow(10, 9);
	double t5 = _element.CalibrationDeltaY * pow(10, 9);*/

	if (needEndianessInverse)
	{
		InverseEndianessElementHeader();
	}

	_element.Data = new char[_element.ArraySizeX * _element.ArraySizeY * _GetDataTypeSize(_element.DataType)];

	ReadWithStatus((char*)_element.Data, _element.ArraySizeX * _element.ArraySizeY * _GetDataTypeSize(_element.DataType));

	if (needEndianessInverse)
	{
		InverseEndianessData();
	}

	bool ok = FileReader::mFile->good();
	/*size_t pos = FileReader::Tell();
	FileReader::Seek(0, ios_base::end);
	size_t size = FileReader::Tell();
	size_t diff = size - pos;*/
	FileReader::CloseRead();

	if (!ok)
	{
		throw FileIOException(FileReader::mFileName, "This is not a proper SER file.");
	}

	return ok;
}

bool SERFile::OpenAndReadHeader()
{
	bool res = FileReader::OpenRead();

	if (!res)
	{
		return false;
	}

	memset(&_fileHeader, 0, sizeof(_fileHeader));
	Read((char*)&_fileHeader, sizeof(_fileHeader));

	bool needEndianessInverse = _fileHeader.ByteOrder != 0x4949;
	if (needEndianessInverse)
		InverseEndianessHeader();

	if (_fileHeader.SeriesVersion != 0x220)
	{
		return false;
	}

	Read((char*)&_dimensionArray, 4 + 8 + 8 + 4 + 4);
	if (needEndianessInverse)
	{
		InverseEndianessDimHeader();
		uint tempui = _dimensionArray.DescriptionLength;
		FileReader::Endian_swap(tempui);
		_dimensionArray.DescriptionLength = tempui;
	}
	if (_dimensionArray.DescriptionLength > 0)
	{
		_dimensionArray.Description = new char[_dimensionArray.DescriptionLength];
		Read((char*)_dimensionArray.Description, _dimensionArray.DescriptionLength);
	}
	Read((char*)&_dimensionArray.UnitsLength, 4);
	if (needEndianessInverse)
	{
		uint tempui = _dimensionArray.UnitsLength;
		FileReader::Endian_swap(tempui);
		_dimensionArray.UnitsLength = tempui;
	}
	if (_dimensionArray.UnitsLength > 0)
	{
		_dimensionArray.Units = new char[_dimensionArray.UnitsLength];
		Read((char*)_dimensionArray.Units, _dimensionArray.UnitsLength);
	}

	FileReader::mFile->seekg(_fileHeader.OffsetArrayOffset, ios_base::beg);

	ulong64 offset;
	Read((char*)&offset, 8);
	if (needEndianessInverse)
	{
		FileReader::Endian_swap(offset);
	}
	FileReader::mFile->seekg(offset, ios_base::beg);

	Read((char*)&_element, sizeof(_element) - sizeof(size_t));

	if (needEndianessInverse)
	{
		InverseEndianessElementHeader();
	}

	bool ok = FileReader::mFile->good();
	FileReader::CloseRead();

	if (!ok)
	{
		return false;
	}

	return ok;
}

DataType_enum SERFile::GetDataType()
{
	//Output only if header was set.
	if (_element.DataType == 0) return DT_UNKNOWN;

	switch (_element.DataType)
	{
	case SERTYPE_UI1:
		return DT_UCHAR;
	case SERTYPE_UI2:
		return DT_USHORT;
	case SERTYPE_UI4:
		return DT_UINT;
	case SERTYPE_I1:
		return DT_CHAR;
	case SERTYPE_I2:
		return DT_SHORT;
	case SERTYPE_I4:
		return DT_INT;
	case SERTYPE_F4:
		return DT_FLOAT;
	case SERTYPE_F8:
		return DT_DOUBLE;
	case SERTYPE_CF4:
		return DT_FLOAT2;
	case SERTYPE_CF8:
		return DT_DOUBLE2;
	default:
		return DT_UNKNOWN;
	}
}

size_t SERFile::GetDataSize()
{
	//Header default values are 0 --> return 0 if not yet read from file or set manually.
	size_t sizeData = (size_t)_element.ArraySizeX * (size_t)_element.ArraySizeY;
	sizeData *= (size_t)_GetDataTypeSize(_element.DataType);
	return sizeData;
}

SerHeader& SERFile::GetFileHeader()
{
	return _fileHeader;
}

SerDimensionArray& SERFile::GetDimensionHeader()
{
	return _dimensionArray;
}

Ser2DElement& SERFile::GetElementHeader()
{
	return _element;
}

void* SERFile::GetData()
{
	return _element.Data;
}

void SERFile::InverseEndianessHeader()
{
	uint tempui;
	ulong64 tempul;
	ushort tempus;
	tempui = _fileHeader.DataTypeID;
	FileReader::Endian_swap(tempui);
	_fileHeader.DataTypeID = tempui;
	tempui = _fileHeader.NumberDimensions;
	FileReader::Endian_swap(tempui);
	_fileHeader.NumberDimensions = tempui;
	tempul = _fileHeader.OffsetArrayOffset;
	FileReader::Endian_swap(tempul);
	_fileHeader.OffsetArrayOffset = tempul;
	tempus = _fileHeader.SeriesID;
	FileReader::Endian_swap(tempus);
	_fileHeader.SeriesID = tempus;
	tempus = _fileHeader.SeriesVersion;
	FileReader::Endian_swap(tempus);
	_fileHeader.SeriesVersion = tempus;
	tempui = _fileHeader.TagypeID;
	FileReader::Endian_swap(tempui);
	_fileHeader.TagypeID = tempui;
	tempui = _fileHeader.TotalNumberElements;
	FileReader::Endian_swap(tempui);
	_fileHeader.TotalNumberElements = tempui;
	tempui = _fileHeader.ValidNumberElements;
	FileReader::Endian_swap(tempui);
	_fileHeader.ValidNumberElements = tempui;
}

void SERFile::InverseEndianessDimHeader()
{
	double tempd = _dimensionArray.CalibrationDelta;
	FileReader::Endian_swap(tempd);
	_dimensionArray.CalibrationDelta = tempd;
	uint tempui = _dimensionArray.CalibrationElement;
	FileReader::Endian_swap(tempui);
	_dimensionArray.CalibrationElement = tempui;
	tempd = _dimensionArray.CalibrationOffset;
	FileReader::Endian_swap(tempd);
	_dimensionArray.CalibrationOffset = tempd;
}

void SERFile::InverseEndianessElementHeader()
{
	uint tempui = _element.ArraySizeX;
	FileReader::Endian_swap(tempui);
	_element.ArraySizeX = tempui;
	tempui = _element.ArraySizeY;
	FileReader::Endian_swap(tempui);
	tempui = _element.ArraySizeY = tempui;
	double tempd = _element.CalibrationDeltaX;
	FileReader::Endian_swap(tempd);
	_element.CalibrationDeltaX = tempd;
	tempd = _element.CalibrationDeltaY;
	FileReader::Endian_swap(tempd);
	_element.CalibrationDeltaY = tempd;
	tempui = _element.CalibrationElementX;
	FileReader::Endian_swap(tempui);
	_element.CalibrationElementX = tempui;
	tempui = _element.CalibrationElementY;
	FileReader::Endian_swap(tempui);
	_element.CalibrationElementY = tempui;
	tempd = _element.CalibrationOffsetX;
	FileReader::Endian_swap(tempd);
	_element.CalibrationOffsetX = tempd;
	tempd = _element.CalibrationOffsetY;
	FileReader::Endian_swap(tempd);
	_element.CalibrationOffsetY = tempd;
	ushort temp = _element.DataType;
	FileReader::Endian_swap(temp);
	_element.DataType = (SerDataType_Enum)temp;
}

void SERFile::InverseEndianessData()
{
	size_t dataSize = (size_t)_element.ArraySizeX * (size_t)_element.ArraySizeY;
	short* d_us;
	int* d_i;
	ulong64* d_ul;

	if (_element.DataType == SERTYPE_CF4 || _element.DataType == SERTYPE_CF8)
	{
		switch (_GetDataTypeSize(_element.DataType))
		{
		case 8:
			d_i = reinterpret_cast<int*>(_element.Data);

#pragma omp for
			for (long long i = 0; i < dataSize * 2; i++)
			{
				FileReader::Endian_swap(d_i[i]);
			}
			break;
		case 16:
			d_ul = reinterpret_cast<ulong64*>(_element.Data);

#pragma omp for
			for (long long i = 0; i < dataSize * 2; i++)
			{
				FileReader::Endian_swap(d_ul[i]);
			}
			break;
		default:
			break;
		}
	}
	else
	{
		switch (_GetDataTypeSize(_element.DataType))
		{
		case 2:
			//signed and unsigned short are the same
			d_us = reinterpret_cast<short*>(_element.Data);

#pragma omp for
			for (long long i = 0; i < dataSize; i++)
			{
				FileReader::Endian_swap(d_us[i]);
			}
			break;
		case 4:
			d_i = reinterpret_cast<int*>(_element.Data);

#pragma omp for
			for (long long i = 0; i < dataSize; i++)
			{
				FileReader::Endian_swap(d_i[i]);
			}
			break;
		case 8:
			d_ul = reinterpret_cast<ulong64*>(_element.Data);

#pragma omp for
			for (long long i = 0; i < dataSize; i++)
			{
				FileReader::Endian_swap(d_ul[i]);
			}
			break;
		default:
			break;
		}
	}
}

float SERFile::GetPixelSizeX()
{
	double val = _element.CalibrationDeltaX * pow(10.0, 9.0); //convert to nm --> 10^9 seems to fit
	return (float)val;
}

float SERFile::GetPixelSizeY()
{
	double val = _element.CalibrationDeltaY * pow(10.0, 9.0); //convert to nm --> 10^9 seems to fit
	return (float)val;
}
