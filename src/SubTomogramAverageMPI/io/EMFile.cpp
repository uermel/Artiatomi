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


#include "EMFile.h"

EMFile::EMFile(string aFileName)
	: FileReader(aFileName), FileWriter(aFileName), _dataStartPosition(0)
{
	memset(&_fileHeader, 0, sizeof(_fileHeader));
}

EMFile::EMFile(string aFileName, const Image& aImage)
	: FileReader(aFileName), FileWriter(aFileName), Image(aImage), _dataStartPosition(0)
{
	memset(&_fileHeader, 0, sizeof(_fileHeader));

	WriteInfoToHeader();
}

uint EMFile::_GetDataTypeSize(EmDataType_Enum aDataType)
{
	switch (aDataType)
	{
	case EMDATATYPE_BYTE:
		return 1;
	case EMDATATYPE_SHORT:
		return 3;
	case EMDATATYPE_INT:
		return 4;
	case EMDATATYPE_FLOAT:
		return 4;
	case EMDATATYPE_COMPLEX:
		return 8;
	case EMDATATYPE_DOUBLE:
		return 8;
	}
	return 0;
}

EMFile* EMFile::CreateEMFile(string aFileNameBase, int index)
{
	stringstream ss;
	ss << aFileNameBase << index << ".em";
	return new EMFile(ss.str());
}

bool EMFile::OpenAndRead()
{
	bool res = FileReader::OpenRead();
	if (!res)
	{
		throw FileIOException(FileReader::mFileName, "Cannot open file for reading.");
	}

	EmMachine_Enum machine = (EmMachine_Enum)FileReader::ReadI1();
	FileReader::mIsLittleEndian = (machine == EMMACHINE_PC);

	FileReader::mFile->seekp(0);

	if (FileReader::mIsLittleEndian)
	{
		FileReader::mFile->read((char*) &_fileHeader, sizeof(_fileHeader));

		_dataStartPosition = (uint)FileReader::mFile->tellg();

		size_t sizeData = GetDataSize();

		_data = new char[sizeData];

		FileReader::mFile->read(_data, sizeData);
		bool ok = FileReader::mFile->good();
		FileReader::CloseRead();
		if (!ok)
		{
			throw FileIOException(FileReader::mFileName, "This is not a proper EM file.");
		}
		return ok;
	}

	if (machine == EMMACHINE_OS9
		|| machine == EMMACHINE_VAX
		|| machine == EMMACHINE_CONVEX
		|| machine == EMMACHINE_SGI
		|| machine == EMMACHINE_MAC)
	{
		throw FileIOException(FileReader::mFileName, "Big endian files are not supported.");
	}

	throw FileIOException(FileReader::mFileName, "This is not a proper EM file.");
}

bool EMFile::OpenAndReadHeader()
{
	bool res = FileReader::OpenRead();
	if (!res)
	{
		throw FileIOException(FileReader::mFileName, "Cannot open file for reading.");
	}

	EmMachine_Enum machine = (EmMachine_Enum)FileReader::ReadI1();
	FileReader::mIsLittleEndian = (machine == EMMACHINE_PC);

	FileReader::mFile->seekp(0);

	if (FileReader::mIsLittleEndian)
	{
		FileReader::mFile->read((char*) &_fileHeader, sizeof(_fileHeader));

		_dataStartPosition = (uint)FileReader::mFile->tellg();

		bool ok = FileReader::mFile->good();
		FileReader::CloseRead();
		if (!ok)
		{
			throw FileIOException(FileReader::mFileName, "This is not a proper EM file.");
		}
		return ok;
	}

	if (machine == EMMACHINE_OS9
		|| machine == EMMACHINE_VAX
		|| machine == EMMACHINE_CONVEX
		|| machine == EMMACHINE_SGI
		|| machine == EMMACHINE_MAC)
	{
		throw FileIOException(FileReader::mFileName, "Big endian files are not supported.");
	}

	throw FileIOException(FileReader::mFileName, "This is not a proper EM file.");
}

bool EMFile::OpenAndWrite()
{
	bool res = FileWriter::OpenWrite();
	if (!res)
	{
		throw FileIOException(FileReader::mFileName, "Cannot open file for writing.");
	}

	//make shure that file header is set to little endian:
	_fileHeader.MachineCoding = EMMACHINE_PC;

	FileWriter::mIsLittleEndian = true;
	FileWriter::mFile->seekp(0);

	FileWriter::mFile->write((char*) &_fileHeader, sizeof(_fileHeader));
	size_t sizeData = GetDataSize();

	FileWriter::mFile->write(_data, sizeData);
	bool ok = FileWriter::mFile->good();
	FileWriter::CloseWrite();
	if (!ok)
	{
		throw FileIOException(FileReader::mFileName, "An error occurred while writing to file.");
	}
	return ok;

}

void EMFile::SetDataType(FileDataType_enum aType)
{
	switch (aType)
	{
	case FDT_UCHAR:
		_fileHeader.DataType = EMDATATYPE_BYTE;
		break;
	case FDT_USHORT:
		_fileHeader.DataType = EMDATATYPE_SHORT;
		break;
	case FDT_UINT:
		_fileHeader.DataType = EMDATATYPE_INT;
		break;
	case FDT_FLOAT:
		_fileHeader.DataType = EMDATATYPE_FLOAT;
		break;
	case FDT_DOUBLE:
		_fileHeader.DataType = EMDATATYPE_DOUBLE;
		break;
	case FDT_CHAR:
		_fileHeader.DataType = EMDATATYPE_BYTE;
		break;
	case FDT_SHORT:
		_fileHeader.DataType = EMDATATYPE_SHORT;
		break;
	case FDT_INT:
		_fileHeader.DataType = EMDATATYPE_INT;
		break;
	case FDT_FLOAT2:
		_fileHeader.DataType = EMDATATYPE_COMPLEX;
		break;
	default:
		cout << "ERROR: The given data type is not supported!\n";
	}
}

/*!
*/
FileDataType_enum EMFile::GetDataType()
{
	//Output only if header was set.
	if (_fileHeader.MachineCoding == 0) return FDT_UNKNOWN;

	//EM format cannot distinguish signed/unsigned! We use only unsigned.
	switch (_fileHeader.DataType)
	{
	case EMDATATYPE_BYTE:
		return FDT_UCHAR;
	case EMDATATYPE_SHORT:
		return FDT_USHORT;
	case EMDATATYPE_INT:
		return FDT_UINT;
	case EMDATATYPE_FLOAT:
		return FDT_FLOAT;
	case EMDATATYPE_COMPLEX:
		return FDT_FLOAT2;
	case EMDATATYPE_DOUBLE:
		return FDT_DOUBLE;
	default:
		return FDT_UNKNOWN;
	}
}

size_t EMFile::GetDataSize()
{
	//Header default values are 0 --> return 0 if not yet read from file or set manually.
	size_t sizeData = (size_t)_fileHeader.DimX * (size_t)_fileHeader.DimY * (size_t)_fileHeader.DimZ;
	sizeData *= (size_t)_GetDataTypeSize(_fileHeader.DataType);
	return sizeData;
}

EmHeader& EMFile::GetFileHeader()
{
	return _fileHeader;
}

void EMFile::SetFileHeader(EmHeader& aHeader)
{
	memcpy(&_fileHeader, &aHeader, sizeof(_fileHeader));
}

char* EMFile::GetData()
{
	return _data;
}

char* EMFile::GetProjection(uint aIndex)
{
	if (DimX == 0) return NULL;

	bool res = FileReader::OpenRead();
	if (!res)
	{
		throw FileIOException(FileReader::mFileName, "Cannot open file for reading.");
	}

	size_t projectionSize = DimX * DimY * _GetDataTypeSize(_fileHeader.DataType);

	FileReader::mFile->seekg(_dataStartPosition + aIndex * projectionSize, ios_base::beg);

	char* data = new char[projectionSize];
	FileReader::mFile->read(data, projectionSize);

	bool ok = FileReader::mFile->good();
	FileReader::CloseRead();

	if (!ok)
	{
		throw FileIOException(FileReader::mFileName, "This is not a proper EM file.");
	}

	return data;
}

void EMFile::SetData(char* aData)
{
	_data = aData;
}

void EMFile::ReadHeaderInfo()
{
	//Check if header is already read from file
	if (_fileHeader.DimX == 0) return;
	ClearImageMetadata();

	DimX = _fileHeader.DimX;
	DimY = _fileHeader.DimY;
	DimZ = _fileHeader.DimZ;

	DataType = GetDataType();

	Cs = _fileHeader.Cs;
	Voltage = _fileHeader.Voltage;

	AllocArrays(DimZ);

	for (int i = 0; i < DimZ; i++)
	{
		TiltAlpha[i] = _fileHeader.Tiltangle / 1000.0f;

		Defocus[i] = (float)_fileHeader.Defocus;
		ExposureTime[i] = _fileHeader.ExposureTime / 1000.0f;
		TiltAxis[i] = (float)_fileHeader.Tiltaxis;
		PixelSize[i] = _fileHeader.Pixelsize / 1000.0f;
		Magnification[i] = (float)_fileHeader.Magnification;
	}
}

void EMFile::WriteInfoToHeader()
{
	memset(&_fileHeader, 0, sizeof(_fileHeader));

	_fileHeader.MachineCoding = EMMACHINE_PC;

	SetDataType(DataType);

	_fileHeader.DimX = DimX;
	_fileHeader.DimY = DimY;
	_fileHeader.DimZ = DimZ;

	_fileHeader.Cs = Cs;
	_fileHeader.Voltage = Voltage;

	_fileHeader.Defocus = (int)Defocus[0];

	_fileHeader.Tiltangle = (int)(TiltAlpha[0] * 1000.0f);

	_fileHeader.ExposureTime = (int)(ExposureTime[0] * 1000.0f);
	_fileHeader.Tiltaxis = (int)TiltAxis[0];
	_fileHeader.Pixelsize = (int)(PixelSize[0] * 1000.0f);
	_fileHeader.Magnification = (int)Magnification[0];

}

void emwrite(string aFileName, float* data, int width, int height)
{
    float* dat = new float[width * height];
    for (int i = 0; i < width; i++)
        for (int j = 0; j < height; j++)
            //dat[j * width + i] = data[(height - i -1) * height + j];
            dat[j * width + i] = data[j * width + i];
    EMFile em(aFileName);
    em.GetFileHeader().DimX = width;
    em.GetFileHeader().DimY = height;
    em.GetFileHeader().DimZ = 1;
    em.SetDataType(FDT_FLOAT);
    em.SetData((char*) dat);
    em.OpenAndWrite();
}

void emread(string aFileName, float*& data, int& width, int& height)
{
    EMFile em(aFileName);
    em.OpenAndRead();
    width = em.GetFileHeader().DimX;
    height = em.GetFileHeader().DimY;

    char* datac = em.GetData();
    float* dataf = (float*) datac;

    data = new float[width * height];
    for (int i = 0; i < width; i++)
        for (int j = 0; j < height; j++)
            //dat[j * width + i] = data[(height - i -1) * height + j];
            data[j * width + i] = -dataf[j * width + i];
}


void emwrite(string aFileName, float* data, int width, int height, int depth)
{
    EMFile em(aFileName);
    em.GetFileHeader().DimX = width;
    em.GetFileHeader().DimY = height;
    em.GetFileHeader().DimZ = depth;
    em.SetDataType(FDT_FLOAT);
    em.SetData((char*) data);
    em.OpenAndWrite();
	em.SetData(NULL);
}

void emread(string aFileName, float*& data, int& width, int& height, int& depth)
{
    EMFile em(aFileName);
    em.OpenAndRead();
    width = em.GetFileHeader().DimX;
    height = em.GetFileHeader().DimY;

    char* datac = em.GetData();
    float* dataf = (float*) datac;

	data = dataf;

	em.SetData(NULL);
}