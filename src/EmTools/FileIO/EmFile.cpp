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


#include "EmFile.h"
#include <string.h>

EmFile::EmFile(string aFileName)
	: FileReader(aFileName), FileWriter(aFileName), _dataStartPosition(0), _data(NULL)
{
	memset(&_fileHeader, 0, sizeof(_fileHeader));
}

EmFile::~EmFile()
{	
	// Delete existing data
	if (_data)
		delete[] (char *)_data;

	_data = NULL;
}

uint EmFile::_GetDataTypeSize(EmDataType_Enum aDataType)
{
	switch (aDataType)
	{
	case EMDATATYPE_BYTE:
		return 1;
	case EMDATATYPE_SHORT:
		return 2;
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

bool EmFile::OpenAndRead()
{
	bool res = FileReader::OpenRead();
	bool needEndianessInverse = false;

	if (!res)
	{
		throw FileIOException(FileReader::mFileName, "Cannot open file for reading.");
	}


	Read((char*)&_fileHeader, sizeof(_fileHeader));

	if (_fileHeader.MachineCoding != EMMACHINE_PC)
	{
		throw FileIOException(FileReader::mFileName, "Only EM files with PC machine coding are supported.");
	}

	_dataStartPosition = (uint)FileReader::mFile->tellg(); //should be 512!

	size_t sizeData = GetDataSize();

	_data = new char[sizeData];

	Read(reinterpret_cast<char*>(_data), sizeData);
	bool ok = FileReader::mFile->good();
	FileReader::CloseRead();

	if (!ok)
	{
		throw FileIOException(FileReader::mFileName, "This is not a proper Em file.");
	}

	return ok;
}

bool EmFile::OpenAndReadHeader()
{
	bool res = FileReader::OpenRead();

	if (!res)
	{
		return false;
	}

	memset(&_fileHeader, 0, sizeof(_fileHeader));
	Read((char*)&_fileHeader, sizeof(_fileHeader));

	_dataStartPosition = (uint)FileReader::mFile->tellg(); //should be 512!


	bool ok = FileReader::mFile->good();
	FileReader::CloseRead();

	if (!ok)
	{
		return false;
	}

	return ok;
}

bool EmFile::OpenAndWrite()
{
	bool res = FileWriter::OpenWrite(true);
	if (!res)
	{
		throw FileIOException(FileReader::mFileName, "Cannot open file for writing.");
	}

	//make shure that file header is set to little endian:
	_fileHeader.MachineCoding = EMMACHINE_PC;
	FileWriter::mIsLittleEndian = true;
	FileWriter::mFile->seekp(0);

	FileWriter::mFile->write((char*)&_fileHeader, sizeof(_fileHeader));
	size_t sizeData = GetDataSize();

	FileWriter::mFile->write((char*)_data, sizeData);
	bool ok = FileWriter::mFile->good();
	FileWriter::CloseWrite();
	if (!ok)
	{
		throw FileIOException(FileReader::mFileName, "An error occurred while writing to file.");
	}
	return ok;

}

bool EmFile::OpenAndWriteHeader()
{
	//don't delete previous file
	bool res = FileWriter::OpenWrite(false);
	if (!res)
	{
		throw FileIOException(FileReader::mFileName, "Cannot open file for writing.");
	}

	//make shure that file header is set to little endian:
	_fileHeader.MachineCoding = EMMACHINE_PC;

	FileWriter::mIsLittleEndian = true;
	FileWriter::mFile->seekp(0);

	FileWriter::mFile->write((char*)&_fileHeader, sizeof(_fileHeader));

	bool ok = FileWriter::mFile->good();
	FileWriter::CloseWrite();
	if (!ok)
	{
		throw FileIOException(FileReader::mFileName, "An error occurred while writing to file.");
	}
	return ok;
}

void EmFile::AddSlice(string aFileName, float* data, int width, int height)
{
	bool exists = false;
	if (FILE* file = fopen(aFileName.c_str(), "r"))
	{
		fclose(file);
		exists = true;
	}

	if (!exists)
	{
		emwrite(aFileName, data, width, height);
	}
	else
	{
		EmFile em(aFileName);
		em.OpenAndReadHeader();

		if (em.GetFileHeader().DimX != width ||
			em.GetFileHeader().DimY != height)
		{
			throw FileIOException(aFileName, "Image dimensions do not fit.");
		}
		em.GetFileHeader().DimZ++;
		
		em.OpenAndWriteHeader();

		//append data
		em.FileWriter::mFile->open(aFileName.c_str(), fstream::out | fstream::binary | fstream::app);
		bool ok = em.FileWriter::mFile->is_open() && em.FileWriter::mFile->good();
		if (!ok)
		{
			throw FileIOException(aFileName, "Could not append slice to file.");
		}

		em.FileWriter::mFile->write((char*)data, width * height * sizeof(float));
		ok = em.FileWriter::mFile->is_open() && em.FileWriter::mFile->good();
		if (!ok)
		{
			throw FileIOException(aFileName, "Could not append slice to file.");
		}
		em.FileWriter::mFile->close();
	}
}

bool EmFile::CanWriteAsEM(int aDimX, int aDimY, int aDimZ, DataType_enum aDatatype)
{
	if (aDimX < 32767 && aDimY < 32767 && aDimZ < 32767 && aDimX > 0 && aDimY > 0 && aDimZ > 0)
	{
		if (aDatatype == DataType_enum::DT_UCHAR  ||
			aDatatype == DataType_enum::DT_SHORT  ||
			aDatatype == DataType_enum::DT_INT    ||
			aDatatype == DataType_enum::DT_FLOAT  ||
			aDatatype == DataType_enum::DT_FLOAT2 ||
			aDatatype == DataType_enum::DT_DOUBLE)
		{
			return true;
		}
	}
	return false;;
}

bool EmFile::SetHeaderData(EmHeader& header, int aDimX, int aDimY, int aDimZ, float aPixelSize, DataType_enum aDatatype)
{
	memset(&header, 0, sizeof(header));

	switch (aDatatype)
	{
	case DT_UNKNOWN:
		return false;
	case DT_UCHAR:
		header.DataType = EMDATATYPE_BYTE;
		break;
	case DT_USHORT:
		return false;
		break;
	case DT_UINT:
		return false;
	case DT_ULONG:
		return false;
	case DT_FLOAT:
		header.DataType = EMDATATYPE_FLOAT;
		break;
	case DT_DOUBLE:
		header.DataType = EMDATATYPE_DOUBLE;
		break;
	case DT_CHAR:
		return false;
	case DT_SHORT:
		header.DataType = EMDATATYPE_SHORT;
		break;
	case DT_INT:
		header.DataType = EMDATATYPE_INT;
		break;
	case DT_LONG:
		return false;
	case DT_FLOAT2:
		header.DataType = EMDATATYPE_COMPLEX;
		break;
	case DT_SHORT2:
		return false;
	default:
		return false;
		break;
	}
	header.MachineCoding = EMMACHINE_PC;
	header.DimX = aDimX;
	header.DimY = aDimY;
	header.DimZ = aDimZ;

	header.ObjectPixelSize = (uint)(aPixelSize * 1000);
	string message = "File created by EmTools";
	strncpy(header.Comment, message.c_str(), 80);
	//strcpy_s(header.Comment, message.size(), message.c_str());

	return true;
}

bool EmFile::InitHeader(string aFileName, int aDimX, int aDimY, float aPixelSize, DataType_enum aDatatype)
{
	bool res;
	EmHeader header;
	res = SetHeaderData(header, aDimX, aDimY, 1, aPixelSize, aDatatype);
	if (!res)
	{
		throw FileIOException(aFileName, "Cannot set header data.");
	}


	FileWriter file(aFileName, true);

	res = file.OpenWrite(true);
	if (!res)
	{
		throw FileIOException(aFileName, "Cannot open file for writing.");
	}

	file.Seek(0, ios_base::beg);

	file.Write(&header, sizeof(header));

	file.CloseWrite();
	return true;
}

bool EmFile::InitHeader(string aFileName, int aDimX, int aDimY, int aDimZ, float aPixelSize, DataType_enum aDatatype)
{
	bool res;
	EmHeader header;
	res = SetHeaderData(header, aDimX, aDimY, aDimZ, aPixelSize, aDatatype);
	if (!res)
	{
		throw FileIOException(aFileName, "Cannot set header data.");
	}


	FileWriter file(aFileName, true);

	res = file.OpenWrite(true);
	if (!res)
	{
		throw FileIOException(aFileName, "Cannot open file for writing.");
	}

	file.Seek(0, ios_base::beg);

	file.Write(&header, sizeof(header));

	file.CloseWrite();
	return true;
}

bool EmFile::InitHeader(string aFileName, EmHeader& aHeader)
{
	bool res;

	FileWriter file(aFileName, true);

	res = file.OpenWrite(true);
	if (!res)
	{
		throw FileIOException(aFileName, "Cannot open file for writing.");
	}

	file.Seek(0, ios_base::beg);

	file.Write(&aHeader, sizeof(aHeader));

	file.CloseWrite();
	return true;
}

bool EmFile::WriteRawData(string aFileName, void* aData, size_t aSize)
{
	EmFile Em(aFileName);
	Em.OpenAndReadHeader();

	uint dataStartPosition = Em._dataStartPosition;

	bool res;

	FileWriter file(aFileName, true);

	res = file.OpenWrite(false);
	if (!res)
	{
		throw FileIOException(aFileName, "Cannot open file for writing.");
	}

	file.Seek(dataStartPosition, ios_base::beg);
	file.Write(aData, aSize);
	file.CloseWrite();
	return true;
}

DataType_enum EmFile::GetDataType()
{
	//Output only if header was set.
	if (_fileHeader.DimX == 0) return DT_UNKNOWN;

	switch (_fileHeader.DataType)
	{
	case EMDATATYPE_BYTE:
		return DT_UCHAR;
	case EMDATATYPE_SHORT:
		return DT_SHORT;
	case EMDATATYPE_INT:
		return DT_INT;
	case EMDATATYPE_FLOAT:
		return DT_FLOAT;
	case EMDATATYPE_COMPLEX:
		return DT_FLOAT2;
	case EMDATATYPE_DOUBLE:
		return DT_DOUBLE;
	default:
		return DT_UNKNOWN;
	}
}

size_t EmFile::GetDataSize()
{
	//Header default values are 0 --> return 0 if not yet read from file or set manually.
	size_t sizeData = (size_t)_fileHeader.DimX * (size_t)_fileHeader.DimY * (size_t)_fileHeader.DimZ;
	sizeData *= (size_t)_GetDataTypeSize(_fileHeader.DataType);
	return sizeData;
}

EmHeader& EmFile::GetFileHeader()
{
	return _fileHeader;
}

void* EmFile::GetData()
{
	return _data;
}

void* EmFile::GetData(size_t aIndex)
{
	if (!_data)
		return NULL;

	if (aIndex >= _fileHeader.DimZ)
		return NULL;

	size_t projectionSize = (size_t)_fileHeader.DimX * (size_t)_fileHeader.DimY * (size_t)_GetDataTypeSize(_fileHeader.DataType);
	return reinterpret_cast<char*>(_data) + aIndex * projectionSize;
}

void emwrite(string aFileName, float* data, int width, int height)
{
	EmFile::InitHeader(aFileName, width, height, 0, DT_FLOAT);
	EmFile::WriteRawData(aFileName, data, (size_t)width * (size_t)height * sizeof(float));
	//EmFile em(aFileName);
	//em.GetFileHeader().DimX = width;
	//em.GetFileHeader().DimY = height;
	//em.GetFileHeader().DimZ = 1;
	//em.SetDataType(DT_FLOAT);
	//em.SetData((char*)data);
	//em.OpenAndWrite();
	////Set inner pointer NULL to avoid that it gets deleted!
	//em.SetData(NULL);
}

void emwrite(string aFileName, float* data, int width, int height, int depth)
{
	EmFile::InitHeader(aFileName, width, height, depth, 0, DT_FLOAT);
	EmFile::WriteRawData(aFileName, data, (size_t)width * (size_t)height * (size_t)depth * sizeof(float));
	//float* dat = new float[width * height];
	//for (int i = 0; i < width; i++)
	//    for (int j = 0; j < height; j++)
	//        //dat[j * width + i] = data[(height - i -1) * height + j];
	//        dat[j * width + i] = data[j * width + i];
	//EmFile em(aFileName);
	//em.GetFileHeader().DimX = width;
	//em.GetFileHeader().DimY = height;
	//em.GetFileHeader().DimZ = depth;
	//em.SetDataType(FDT_FLOAT);
	//em.SetData((char*)data);
	//em.OpenAndWrite();
	////Set inner pointer NULL to avoid that it gets deleted!
	//em.SetData(NULL);
}