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


#include "MRCFile.h"
#include <string.h>

MRCFile::MRCFile(string aFileName)
	: FileReader(aFileName), FileWriter(aFileName), _dataStartPosition(0), _data(NULL)
{
	memset(&_fileHeader, 0, sizeof(_fileHeader));
	_extHeaders = NULL;
}

MRCFile::~MRCFile()
{
	if (_extHeaders)
		delete[] _extHeaders;
	_extHeaders = NULL;

	// Delete existing data
	if (_data)
		delete[] (char *)_data;

	_data = NULL;
}

uint MRCFile::_GetDataTypeSize(MrcMode_Enum aDataType)
{
	switch (aDataType)
	{
	case MRCMODE_UI1:
		return 1;
	case MRCMODE_I2:
		return 2;
	case MRCMODE_UI2:
		return 2;
	case MRCMODE_F:
		return 4;
	case MRCMODE_CUI2:
		return 4;
	case MRCMODE_CF:
		return 4;
	case MRCMODE_RGB:
		return 3;
	}
	return 0;
}

bool MRCFile::OpenAndRead()
{
	bool res = FileReader::OpenRead();
	bool needEndianessInverse = false;

	if (!res)
	{
		FileReader::CloseRead();
		//Allocations, if any, will be freed in destructor, just leave...
		throw FileIOException(FileReader::mFileName, "Cannot open file for reading.");
	}


	Read((char*)&_fileHeader, sizeof(_fileHeader));

	if (_fileHeader.NX > 32000) //image sizes > 32000 indicate big endianess
	{
		InverseEndianessHeader();
		needEndianessInverse = true;
	}

	//Check if extended headers exist:
	bool extExist = _fileHeader.NEXT > 0 && !(_fileHeader.NEXT % sizeof(MrcExtendedHeader)); //default extended headers
	bool imodHeader = ((_fileHeader.NREAL & 1) && _fileHeader.NINT > 1 && _fileHeader.NINT < 16) && _fileHeader.IMODStamp == 1146047817; //Current SerialEM format with 61 is a bitpattern, 14 is width in bytes

	if (extExist && !imodHeader)
	{
		int extHeaderCount = _fileHeader.NEXT / sizeof(MrcExtendedHeader);
		if (_extHeaders)
			delete[] _extHeaders;
		_extHeaders = new MrcExtendedHeader[extHeaderCount];

		for (int i = 0; i < extHeaderCount; i++)
			Read((char*)&_extHeaders[i], sizeof(MrcExtendedHeader));

		if (needEndianessInverse)
		{
			//These strange files from Digital Micrograph are note entirely big endian...
			//InverseEndianessExtHeaders();
		}
	}


	//new SeriealEM ST-Format
	if (imodHeader) 
	{
		char temp[256];
		int extHeaderCount = _fileHeader.NZ;
		if (_extHeaders)
			delete[] _extHeaders;
		_extHeaders = new MrcExtendedHeader[extHeaderCount];
		memset(_extHeaders, 0, sizeof(MrcExtendedHeader) * extHeaderCount);

		for (int image = 0; image < extHeaderCount; image++)
		{
			Read(temp, _fileHeader.NINT);
			short* temp2 = (short*)temp;
			float ang = temp2[0] / 100.0f;
			_extHeaders[image].a_tilt = ang;
		}
	}
	
	_dataStartPosition = (uint)sizeof(_fileHeader) + _fileHeader.NEXT;
	FileReader::Seek(_dataStartPosition, ios::beg);
	//uint check = (uint)FileReader::mFile->tellg();

	size_t sizeData = GetDataSize();

	_data = new char[sizeData];

	ReadWithStatus(reinterpret_cast<char*>(_data), sizeData);
	bool ok = FileReader::mFile->good();
	FileReader::CloseRead();

	if (!ok)
	{
		throw FileIOException(FileReader::mFileName, "This is not a proper MRC file.");
	}

	if (needEndianessInverse)
		InverseEndianessData();

	return ok;
}

bool MRCFile::OpenAndReadHeader()
{
	bool res = FileReader::OpenRead();
	bool needEndianessInverse = false;

	if (!res)
	{
		return false;
	}

	memset(&_fileHeader, 0, sizeof(_fileHeader));
	Read((char*) &_fileHeader, sizeof(_fileHeader));

	if (_fileHeader.NX > 32000) //image sizes > 32000 indicate big endianess
	{
		InverseEndianessHeader();
		needEndianessInverse = true;
	}

	//Check if extended headers exist:
	bool extExist = _fileHeader.NEXT > 0 && !(_fileHeader.NEXT % sizeof(MrcExtendedHeader));
	bool imodHeader = ((_fileHeader.NREAL & 1) && _fileHeader.NINT > 1 && _fileHeader.NINT < 16) && _fileHeader.IMODStamp == 1146047817; //Current SerialEM format with 61 is a bitpattern, 14 is width in bytes

	if (extExist && !imodHeader)
	{
		int extHeaderCount = _fileHeader.NEXT / sizeof(MrcExtendedHeader);
		if (_extHeaders)
			delete[] _extHeaders;

		_extHeaders = new MrcExtendedHeader[extHeaderCount];

		for (int i = 0; i < extHeaderCount; i++)
			Read((char*) &_extHeaders[i], sizeof(MrcExtendedHeader));

		if (needEndianessInverse)
		{
			//These strange files from Digital Micrograph are note entirely big endian...
			//InverseEndianessExtHeaders();
		}
	}


	//new SeriealEM ST-Format
	if (imodHeader) //Current SerialEM format with 61 is a bitpattern, 14 is width in bytes
	{
		char temp[256];
		int extHeaderCount = _fileHeader.NZ;
		if (_extHeaders)
			delete[] _extHeaders;
		_extHeaders = new MrcExtendedHeader[extHeaderCount];
		memset(_extHeaders, 0, sizeof(MrcExtendedHeader) * extHeaderCount);

		for (int image = 0; image < extHeaderCount; image++)
		{
			Read(temp, _fileHeader.NINT);
			short* temp2 = (short*)temp;
			float ang = temp2[0] / 100.0f;
			_extHeaders[image].a_tilt = ang;
		}
	}

	size_t checkSize = 0;
	_dataStartPosition = (uint)sizeof(_fileHeader) + _fileHeader.NEXT;// (uint)FileReader::mFile->tellg();
	FileReader::mFile->seekg(0, ios::end);
	checkSize =FileReader::mFile->tellg();

	size_t sizeData = GetDataSize();

	bool ok = FileReader::mFile->good();
	ok = ok && (sizeData == (checkSize - _dataStartPosition));

	FileReader::CloseRead();

	if (!ok)
	{
		return false;
	}

	return ok;
}

bool MRCFile::SetHeaderData(MrcHeader& header, int aDimX, int aDimY, int aDimZ, float aPixelSize, DataType_enum aDatatype, bool aSetExtended)
{
	memset(&header, 0, sizeof(header));

	switch (aDatatype)
	{
	case DT_UNKNOWN:
		return false;
	case DT_UCHAR:
		header.MODE = MRCMODE_UI1;
		break;
	case DT_USHORT:
		header.MODE = MRCMODE_UI2;
		break;
	case DT_UINT:
		return false;
	case DT_ULONG:
		return false;
	case DT_FLOAT:
		header.MODE = MRCMODE_F;
		break;
	case DT_DOUBLE:
		return false;
	case DT_CHAR:
		header.MODE = MRCMODE_UI1;
		header.IMODStamp = MRC_IMODFLAG;
		header.IMODFlags = 1;
		break;
	case DT_SHORT:
		header.MODE = MRCMODE_I2;
		break;
	case DT_INT:
		return false;
	case DT_LONG:
		return false;
	case DT_FLOAT2:
		header.MODE = MRCMODE_CF;
		break;
	case DT_SHORT2:
		header.MODE = MRCMODE_CUI2;
		break;
	default:
		break;
	}

	header.NX = aDimX;
	header.NY = aDimY;
	header.NZ = aDimZ;

	header.MX = aDimX;
	header.MY = aDimY;
	header.MZ = aDimZ;

	header.Xlen = aDimX * aPixelSize * 10.0f; //nm to Angstrom = factor 10
	header.Ylen = aDimY * aPixelSize * 10.0f;
	header.Zlen = aDimZ * aPixelSize * 10.0f;

	header.ALPHA = 90;
	header.BETA = 90;
	header.GAMMA = 90;

	header.MAPC = MRCAXIS_X;
	header.MAPR = MRCAXIS_Y;
	header.MAPS = MRCAXIS_Z;

	if (aSetExtended)
	{
		header.NEXT = 1024 * sizeof(MrcExtendedHeader);
		header.NREAL = 32;
	}

	header.CMAP[0] = 'M';
	header.CMAP[1] = 'A';
	header.CMAP[2] = 'P';
	header.CMAP[3] = 0;

	header.STAMP = 16708;

	header.NLABL = 1;
	std::string message = "File created by EmTools";

	strncpy(header.LABELS[0], message.c_str(), sizeof(header.LABELS[0]));
	//strcpy_s(header.LABELS[0], sizeof(header.LABELS[0]), message.c_str());

	return true;
}

bool MRCFile::InitHeaders(string aFileName, int aDimX, int aDimY, float aPixelSize, DataType_enum aDatatype, bool aSetExtended)
{
	bool res;
	MrcHeader header;
	res = SetHeaderData(header, aDimX, aDimY, 0, aPixelSize, aDatatype, aSetExtended);
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

	if (aSetExtended)
	{
		MrcExtendedHeader extHeader;
		memset(&extHeader, 0, sizeof(extHeader));
		for (size_t i = 0; i < 1024; i++)
		{
			file.Write(&extHeader, sizeof(extHeader));
		}
	}

	file.CloseWrite();
	return true;
}

bool MRCFile::InitHeaders(string aFileName, int aDimX, int aDimY, int aDimZ, float aPixelSize, DataType_enum aDatatype, bool aSetExtended)
{
	bool res;
	MrcHeader header;
	res = SetHeaderData(header, aDimX, aDimY, aDimZ, aPixelSize, aDatatype, aSetExtended);
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

	if (aSetExtended)
	{
		MrcExtendedHeader extHeader;
		memset(&extHeader, 0, sizeof(extHeader));
		for (size_t i = 0; i < 1024; i++)
		{
			file.Write(&extHeader, sizeof(extHeader));
		}
	}

	file.CloseWrite();
	return true;
}

bool MRCFile::InitHeaders(string aFileName, MrcHeader& aHeader)
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

	bool extExist = aHeader.NEXT > 0 && !(aHeader.NEXT % sizeof(MrcExtendedHeader));

	if (extExist)
	{
		MrcExtendedHeader extHeader;
		memset(&extHeader, 0, sizeof(extHeader));
		for (size_t i = 0; i < 1024; i++)
		{
			file.Write(&extHeader, sizeof(extHeader));
		}
	}

	file.CloseWrite();
	return true;
}

bool MRCFile::AddPlaneToMRCFile(string aFileName, DataType_enum aDatatype, void* aData, float tiltAngle)
{
	MRCFile mrc(aFileName);
	mrc.OpenAndReadHeader();

	uint dataStartPosition = mrc._dataStartPosition;
	MrcHeader header = mrc._fileHeader;
	bool writeTileAngle = dataStartPosition == 1024 * sizeof(MrcExtendedHeader) + sizeof(MrcHeader);

	uint actualZ = header.NZ;

	if (aDatatype != mrc.GetDataType())
	{
		throw FileIOException(aFileName, "Cannot append image plane to MRC file: datatyps don't match.");
	}

	bool res;

	FileWriter file(aFileName, true);

	res = file.OpenWrite(false);
	if (!res)
	{
		throw FileIOException(aFileName, "Cannot open file for writing.");
	}

	file.Seek(dataStartPosition, ios_base::beg);
	file.Seek(mrc.GetDataSize(), ios_base::cur);

	file.Write(aData, (size_t)header.NX * (size_t)header.NY * (size_t)mrc._GetDataTypeSize(header.MODE));

	if (writeTileAngle)
	{
		file.Seek(sizeof(header) + header.NZ * sizeof(MrcExtendedHeader), ios_base::beg);
		//first entry of extended header is the alpha tilt:
		file.WriteLE(tiltAngle);
	}

	header.NZ++;
	header.MZ++;
	header.Zlen = header.Xlen / header.NX * header.NZ;

	file.Seek(0, ios_base::beg);

	file.Write(&header, sizeof(MrcHeader));

	file.CloseWrite();
	return true;
}

bool MRCFile::WriteRawData(string aFileName, void* aData, size_t aSize)
{
	MRCFile mrc(aFileName);
	mrc.OpenAndReadHeader();

	uint dataStartPosition = mrc._dataStartPosition;
	
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

DataType_enum MRCFile::GetDataType()
{
	//Output only if header was set.
	if (_fileHeader.NX == 0) return DT_UNKNOWN;

	switch (_fileHeader.MODE)
	{
	case MRCMODE_UI1:
		return DT_UCHAR;
	case MRCMODE_I2:
		return DT_SHORT;
	case MRCMODE_F:
		return DT_FLOAT;
	case MRCMODE_CUI2:
		return DT_SHORT2;
	case MRCMODE_CF:
		return DT_FLOAT2;
	case MRCMODE_UI2:
		return DT_USHORT;
	case MRCMODE_RGB:
		return DT_RGB;
	default:
		return DT_UNKNOWN;
	}
}

size_t MRCFile::GetDataSize()
{
	//Header default values are 0 --> return 0 if not yet read from file or set manually.
	size_t sizeData = (size_t)_fileHeader.NX * (size_t)_fileHeader.NY * (size_t)_fileHeader.NZ;
	sizeData *= (size_t)_GetDataTypeSize(_fileHeader.MODE);
	return sizeData;
}

MrcHeader& MRCFile::GetFileHeader()
{
	return _fileHeader;
}

void* MRCFile::GetData()
{
	return _data;
}

void* MRCFile::GetData(size_t aIndex)
{
	if (!_data)
		return NULL;

	if (aIndex >= _fileHeader.NZ)
		return NULL;

	size_t projectionSize = (size_t)_fileHeader.NX * (size_t)_fileHeader.NY * (size_t)_GetDataTypeSize(_fileHeader.MODE);
	return reinterpret_cast<char*>(_data) + aIndex * projectionSize;
}

void* MRCFile::GetProjection(uint aIndex)
{
	if (_fileHeader.NX == 0) return NULL;

	bool res = FileReader::OpenRead();
	if (!res)
	{
		throw FileIOException(FileReader::mFileName, "Cannot open file for reading.");
	}

	size_t projectionSize = (size_t)_fileHeader.NX * (size_t)_fileHeader.NY * (size_t)_GetDataTypeSize(_fileHeader.MODE);

	FileReader::mFile->seekg(_dataStartPosition + aIndex * projectionSize, ios_base::beg);

	char* data = new char[projectionSize];
	Read(data, projectionSize);

	bool ok = FileReader::mFile->good();
	FileReader::CloseRead();

	if (!ok)
	{
		throw FileIOException(FileReader::mFileName, "This is not a proper MRC file.");
	}

	return data;
}

float MRCFile::GetPixelsize()
{
	if (_fileHeader.NX == 0 || _fileHeader.Xlen == 0)
		return 0;

	return _fileHeader.Xlen / (float)_fileHeader.NX / 10.0f; //A to nm
}

MrcExtendedHeader* MRCFile::GetFileExtHeaders()
{
	return _extHeaders;
}

void MRCFile::InverseEndianessHeader()
{
	FileReader::Endian_swap(_fileHeader.NX);
	FileReader::Endian_swap(_fileHeader.NY);
	FileReader::Endian_swap(_fileHeader.NZ);

	uint mode = _fileHeader.MODE;
	FileReader::Endian_swap(mode);
	_fileHeader.MODE = (MrcMode_Enum)mode;

	FileReader::Endian_swap(_fileHeader.NXSTART);
	FileReader::Endian_swap(_fileHeader.NYSTART);
	FileReader::Endian_swap(_fileHeader.NZSTART);
	FileReader::Endian_swap(_fileHeader.MX);
	FileReader::Endian_swap(_fileHeader.MY);
	FileReader::Endian_swap(_fileHeader.MZ);

	FileReader::Endian_swap(_fileHeader.Xlen);
	FileReader::Endian_swap(_fileHeader.Ylen);
	FileReader::Endian_swap(_fileHeader.Zlen);

	FileReader::Endian_swap(_fileHeader.ALPHA);
	FileReader::Endian_swap(_fileHeader.BETA);
	FileReader::Endian_swap(_fileHeader.GAMMA);

	mode = _fileHeader.MAPC;
	FileReader::Endian_swap(mode);
	_fileHeader.MAPC = (MrcAxis_Enum)mode;

	mode = _fileHeader.MAPR;
	FileReader::Endian_swap(mode);
	_fileHeader.MAPR = (MrcAxis_Enum)mode;

	mode = _fileHeader.MAPS;
	FileReader::Endian_swap(mode);
	_fileHeader.MAPS = (MrcAxis_Enum)mode;

	FileReader::Endian_swap(_fileHeader.AMIN);
	FileReader::Endian_swap(_fileHeader.AMAX);
	FileReader::Endian_swap(_fileHeader.AMEAN);

	FileReader::Endian_swap(_fileHeader.ISPG);
	FileReader::Endian_swap(_fileHeader.NSYMBT);

	FileReader::Endian_swap(_fileHeader.NEXT);

	FileReader::Endian_swap(_fileHeader.CREATEID);

	FileReader::Endian_swap(_fileHeader.NINT);
	FileReader::Endian_swap(_fileHeader.NREAL);

	FileReader::Endian_swap(_fileHeader.TILTANGLES[0]);
	FileReader::Endian_swap(_fileHeader.TILTANGLES[1]);
	FileReader::Endian_swap(_fileHeader.TILTANGLES[2]);
	FileReader::Endian_swap(_fileHeader.TILTANGLES[3]);
	FileReader::Endian_swap(_fileHeader.TILTANGLES[4]);
	FileReader::Endian_swap(_fileHeader.TILTANGLES[5]);
	FileReader::Endian_swap(_fileHeader.XORIGIN);
	FileReader::Endian_swap(_fileHeader.YORIGIN);
	FileReader::Endian_swap(_fileHeader.ZORIGIN);
	FileReader::Endian_swap(_fileHeader.RMS);
	FileReader::Endian_swap(_fileHeader.NLABL);
}

void MRCFile::InverseEndianessExtHeaders()
{
	if (_extHeaders)
	{
		int extHeaderCount = _fileHeader.NEXT / sizeof(float);
		float* headerAsFloats = reinterpret_cast<float*>(_extHeaders);

		for (int i = 0; i < extHeaderCount; i++)
			FileReader::Endian_swap(headerAsFloats[i]);
	}
}

void MRCFile::InverseEndianessData()
{
	size_t dataSize = (size_t)_fileHeader.NX * (size_t)_fileHeader.NY * (size_t)_fileHeader.NZ;
	short* d_us;
	float* d_f;
	switch (_fileHeader.MODE)
	{
	case MRCMODE_I2:
	case MRCMODE_UI2:
		//signed and unsigned short are the same
		d_us = reinterpret_cast<short*>(_data);
		
		#pragma omp for
		for (long long i = 0; i < dataSize; i++)
		{
			FileReader::Endian_swap(d_us[i]);
		}
		break;
	case MRCMODE_F:
		d_f = reinterpret_cast<float*>(_data);
		
		#pragma omp for
		for (long long i = 0; i < dataSize; i++)
		{
			FileReader::Endian_swap(d_f[i]);
		}
		break;
	default:
		break;
	}
}
