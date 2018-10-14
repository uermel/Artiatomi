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
#include "../utils/Config.h"

MRCFile::MRCFile(string aFileName)
	: ProjectionSource(aFileName), _dataStartPosition(0), _projectionCache(0)
{
	memset(&_fileHeader, 0, sizeof(_fileHeader));
	_extHeaders = NULL;
}

MRCFile::MRCFile(string aFileName, const Image& aImage)
	: ProjectionSource(aFileName, aImage), _dataStartPosition(0), _projectionCache(0)
{
	memset(&_fileHeader, 0, sizeof(_fileHeader));
	_extHeaders = NULL;

	WriteInfoToHeader();
}

MRCFile::~MRCFile()
{
	if (_extHeaders)
		delete[] _extHeaders;
	if (_projectionCache)
        for (int i = 0; i < DimZ; i++)
            if (_projectionCache[i])
                delete[] _projectionCache[i];
	_extHeaders = NULL;
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
	}
	return 0;
}

bool MRCFile::OpenAndRead()
{
	bool res = FileReader::OpenRead();
	if (!res)
	{
		throw FileIOException(FileReader::mFileName, "Cannot open file for reading.");
	}


	FileReader::mFile->read((char*) &_fileHeader, sizeof(_fileHeader));

	//Check if extended headers exist:
	bool extExist = _fileHeader.NEXT > 0 && !(_fileHeader.NEXT % sizeof(MrcExtendedHeader));

	if (extExist)
	{
		int extHeaderCount = _fileHeader.NEXT / sizeof(MrcExtendedHeader);
		_extHeaders = new MrcExtendedHeader[extHeaderCount];

		for (int i = 0; i < extHeaderCount; i++)
			FileReader::mFile->read((char*) &_extHeaders[i], sizeof(MrcExtendedHeader));
	}

	_dataStartPosition = (uint)FileReader::mFile->tellg();

	size_t sizeData = GetDataSize();

	_data = new char[sizeData];

	FileReader::mFile->read(_data, sizeData);
	bool ok = FileReader::mFile->good();
	FileReader::CloseRead();

	if (!ok)
	{
		throw FileIOException(FileReader::mFileName, "This is not a proper MRC file.");
	}
	ReadHeaderInfo();
	return ok;
}

bool MRCFile::OpenAndReadHeader()
{
	bool res = FileReader::OpenRead();
	if (!res)
	{
		throw FileIOException(FileReader::mFileName, "Cannot open file for reading.");
	}


	FileReader::mFile->read((char*) &_fileHeader, sizeof(_fileHeader));

	//Check if extended headers exist:
	bool extExist = _fileHeader.NEXT > 0 && !(_fileHeader.NEXT % sizeof(MrcExtendedHeader));

	if (extExist)
	{
		int extHeaderCount = _fileHeader.NEXT / sizeof(MrcExtendedHeader);
		_extHeaders = new MrcExtendedHeader[extHeaderCount];

		for (int i = 0; i < extHeaderCount; i++)
			FileReader::mFile->read((char*) &_extHeaders[i], sizeof(MrcExtendedHeader));
	}

	//new serialEM ST-Format
	if (((_fileHeader.NREAL & 1) && _fileHeader.NINT > 1 && _fileHeader.NINT < 16) && _fileHeader.imodStamp == 1146047817) //Current SerialEM format with 61 is a bitpattern, 14 is width in bytes
    {
		char temp[256];
		int extHeaderCount = _fileHeader.NZ;
		_extHeaders = new MrcExtendedHeader[extHeaderCount];
		memset(_extHeaders, 0, sizeof(MrcExtendedHeader) * extHeaderCount);

        for (int image = 0; image < extHeaderCount; image++)
        {
			FileReader::mFile->read(temp, _fileHeader.NINT);
			short* temp2 = (short*)temp;
            float ang = temp2[0] / 100.0f;
            _extHeaders[image].a_tilt = ang;
        }
    }

	_dataStartPosition = (uint)sizeof(_fileHeader) + _fileHeader.NEXT;// (uint)FileReader::mFile->tellg();

	bool ok = FileReader::mFile->good();
	FileReader::CloseRead();

	if (!ok)
	{
		throw FileIOException(FileReader::mFileName, "This is not a proper MRC file.");
	}

	ReadHeaderInfo();
	return ok;
}

bool MRCFile::OpenAndWrite()
{
	bool res = FileWriter::OpenWrite();
	if (!res)
	{
		throw FileIOException(FileReader::mFileName, "Cannot open file for writing.");
	}

	//make shure that file header is set to little endian:
	FileWriter::mIsLittleEndian = true;
	FileWriter::mFile->seekp(0);

	FileWriter::mFile->write((char*) &_fileHeader, sizeof(_fileHeader));
	if (_fileHeader.NEXT > 0)
	{
		FileWriter::mFile->write((char*) _extHeaders, _fileHeader.NEXT);
	}
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

void MRCFile::SetDataType(FileDataType_enum aType)
{
	switch (aType)
	{
	case FDT_UCHAR:
		_fileHeader.MODE = MRCMODE_UI1;
		break;
	case FDT_USHORT:
		_fileHeader.MODE = MRCMODE_UI2;
		break;
	case FDT_FLOAT:
		_fileHeader.MODE = MRCMODE_F;
		break;
	case FDT_CHAR:
		_fileHeader.MODE = MRCMODE_UI1;
		break;
	case FDT_SHORT:
		_fileHeader.MODE = MRCMODE_I2;
		break;
	case FDT_FLOAT2:
		_fileHeader.MODE = MRCMODE_CF;
		break;
	default:
		cout << "ERROR: The given data type is not supported!\n";
	}
}

FileDataType_enum MRCFile::GetDataType()
{
	//Output only if header was set.
	if (_fileHeader.NX == 0) return FDT_UNKNOWN;

	switch (_fileHeader.MODE)
	{
	case MRCMODE_UI1:
		return FDT_UCHAR;
	case MRCMODE_I2:
		return FDT_SHORT;
	case MRCMODE_F:
		return FDT_FLOAT;
	case MRCMODE_CUI2:
		return FDT_SHORT2;
	case MRCMODE_CF:
		return FDT_FLOAT2;
	case MRCMODE_UI2:
		return FDT_USHORT;
	default:
		return FDT_UNKNOWN;
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

void MRCFile::SetFileHeader(MrcHeader& aHeader)
{
	memcpy(&_fileHeader, &aHeader, sizeof(_fileHeader));

	if (!_extHeaders)
		_fileHeader.NEXT = 0;
}

char* MRCFile::GetData()
{
	return _data;
}

char* MRCFile::GetProjection(uint aIndex)
{
	if (DimX == 0) return NULL;
	if (_projectionCache == NULL)
	{
        _projectionCache = new float*[DimZ];
        for (int i = 0; i < DimZ; i++)
            _projectionCache[i] = NULL;
	}

    if (_projectionCache[aIndex] == NULL)
    {
        _projectionCache[aIndex] = new float[DimX * DimY];

        bool res = FileReader::OpenRead();
        if (!res)
        {
            throw FileIOException(FileReader::mFileName, "Cannot open file for reading.");
        }

        size_t projectionSize = DimX * DimY * _GetDataTypeSize(_fileHeader.MODE);

        FileReader::mFile->seekg(_dataStartPosition + aIndex * projectionSize, ios_base::beg);

        //char* data = new char[projectionSize];
        FileReader::mFile->read((char*)_projectionCache[aIndex], projectionSize);

        bool ok = FileReader::mFile->good();
        FileReader::CloseRead();

        if (!ok)
        {
            throw FileIOException(FileReader::mFileName, "This is not a proper MRC file.");
        }

        return (char*)_projectionCache[aIndex];
    }
    else
        return (char*)_projectionCache[aIndex];
}

float* MRCFile::GetProjectionFloat(uint aIndex)
{
	char* data = (char*)GetProjection(aIndex);
	if (data == NULL) return NULL;

	float* fdata = new float[DimX * DimY];

	uint pixelSize = Image::GetPixelSizeInBytes();

	if (pixelSize == 1)//MRC is then uchar
		for (int i = 0; i < DimX * DimY; i++)
		{
			fdata[i] = (float)data[i] / 255.0f;
		}
	else if (pixelSize == 2) //MRC is then short but in our case always ushort
	{
		ushort* usdata = (ushort*) data;
		for (int i = 0; i < DimX * DimY; i++)
		{
			fdata[i] = (float)usdata[i] / 65535.0f;
		}
	}
	else if (pixelSize == 4) //MRC is then already float
	{
	    delete[] fdata;
		return (float*) data;
	}
	delete[] data;
	return fdata;
}

float* MRCFile::GetProjectionInvertFloat(uint aIndex)
{
	char* data = (char*)GetProjection(aIndex);
	if (data == NULL) return NULL;

	float* fdata = new float[DimX * DimY];

	uint pixelSize = Image::GetPixelSizeInBytes();

	if (pixelSize == 1)//MRC is then uchar
		for (int i = 0; i < DimX * DimY; i++)
		{
			fdata[i] = (float)(255 - data[i]) / 255.0f;
		}
	else if (pixelSize == 2) //MRC is then short but in our case always ushort
	{
		ushort* usdata = (ushort*) data;
		float maxVal;
		float normVal;// = Configuration::Config::GetConfig().ProjNormVal;
		float min = 1000000000.0f;
		float max = -1000000000.0f;
//		if (Configuration::Config::GetConfig().Filtered)
//		{
//            maxVal = 32767.0f;
//            normVal = 1.0f;
//		    short* sdata = (short*) data;
//		    for (int i = 0; i < DimX * DimY; i++)
//            {
//                fdata[i] = (32768.0f - ((float)sdata[i])) / 32768.0f * normVal;
//                //fdata[i] = (maxVal - ((float)usdata[i])) / maxVal * normVal;
//                //fdata[i] = std::max(fdata[i], 0.0f);
//                //max = std::max(max, fdata[i]);
//                //min = std::min(min, fdata[i]);
//            }
//        }
//		else
        {
            maxVal = 65535.0f;
            for (int i = 0; i < DimX * DimY; i++)
            {
                normVal = maxVal;
                fdata[i] = (maxVal - ((float)usdata[i] * maxVal / normVal)) / maxVal;
                //fdata[i] = std::max(fdata[i], 0.0f);
                //max = std::max(max, fdata[i]);
                //min = std::min(min, fdata[i]);
            }
        }
        //printf("Check Min/Max: %f, %f\n", min, max);
	}
	else if (pixelSize == 4) //MRC is then already float
	{
	    for (int i = 0; i < DimX * DimY; i++)
		{
			fdata[i] = 1.0f - data[i];
		}
	}
	delete[] data;
	return fdata;
}

void MRCFile::SetData(char* aData)
{
	_data = aData;
}

void MRCFile::SetFileExtHeader(MrcExtendedHeader* aHeader, int aCount)
{
	_fileHeader.NEXT = aCount * sizeof(MrcExtendedHeader);
	if (_extHeaders)
		delete[] _extHeaders;

	_extHeaders = new MrcExtendedHeader[aCount];
	memcpy(_extHeaders, aHeader, _fileHeader.NEXT);
}

MrcExtendedHeader* MRCFile::GetFileExtHeaders()
{
	return _extHeaders;
}

void MRCFile::ReadHeaderInfo()
{
	//Check if header is already read from file
	if (_fileHeader.NX == 0) return;
	ClearImageMetadata();

	DimX = _fileHeader.NX;
	DimY = _fileHeader.NY;
	DimZ = _fileHeader.NZ;

	DataType = GetDataType();

	CellSizeX = _fileHeader.Xlen;
	CellSizeY = _fileHeader.Ylen;
	CellSizeZ = _fileHeader.Zlen;

	CellAngleAlpha = _fileHeader.ALPHA;
	CellAngleBeta = _fileHeader.BETA;
	CellAngleGamma = _fileHeader.GAMMA;

	OriginX = _fileHeader.XORIGIN;
	OriginY = _fileHeader.YORIGIN;
	OriginZ = _fileHeader.ZORIGIN;

	RMS = _fileHeader.RMS;

	AllocArrays(DimZ);
	if (_fileHeader.NEXT < sizeof(MrcExtendedHeader) * DimZ && !((_fileHeader.NREAL & 1) && _fileHeader.NINT > 1 && _fileHeader.NINT < 16 && _fileHeader.imodStamp == 1146047817))
	{
		for (int i = 0; i < DimZ; i++)
		{
			PixelSize[i] = _fileHeader.Xlen / float(_fileHeader.NX);
		}
		return;
	};

	for (int i = 0; i < DimZ; i++)
	{
		MinValue[i] = _fileHeader.AMIN;
		MaxValue[i] = _fileHeader.AMAX;
		MeanValue[i] = _fileHeader.AMEAN;

		TiltAlpha[i] = _extHeaders[i].a_tilt;
		TiltBeta[i] = _extHeaders[i].b_tilt;

		StageXPosition[i] = _extHeaders[i].x_stage;
		StageYPosition[i] = _extHeaders[i].y_stage;
		StageZPosition[i] = _extHeaders[i].z_stage;

		ShiftX[i] = _extHeaders[i].x_shift;
		ShiftY[i] = _extHeaders[i].y_shift;
		ShiftZ[i] = 0;//_extHeaders[i].z_shift;

		Defocus[i] = _extHeaders[i].defocus;
		ExposureTime[i] = _extHeaders[i].exp_time;
		TiltAxis[i] = _extHeaders[i].tilt_axis;
		PixelSize[i] = _fileHeader.Xlen / float(_fileHeader.NX); //_extHeaders[i].pixel_size;
		Magnification[i] = _extHeaders[i].magnification;
		Binning[i] = _extHeaders[i].binning;
		AppliedDefocus[i] = _extHeaders[i].appliedDefocus;
		Voltage = (int)_extHeaders[0].ht;
	}

}

void MRCFile::WriteInfoToHeader()
{
	//Check if extended headers are already allocated:
	if (_extHeaders == NULL)
		_extHeaders = new MrcExtendedHeader[1024];

	memset(&_fileHeader, 0, sizeof(_fileHeader));
	_fileHeader.NEXT = 1024 * sizeof(MrcExtendedHeader);
	memset(_extHeaders, 0, _fileHeader.NEXT);

	_fileHeader.NX = DimX;
	_fileHeader.NY = DimY;
	_fileHeader.NZ = DimZ;
	_fileHeader.MX = DimX;
	_fileHeader.MY = DimY;
	_fileHeader.MZ = DimZ;

	_fileHeader.MAPC = MRCAXIS_X;
	_fileHeader.MAPR = MRCAXIS_Y;
	_fileHeader.MAPS = MRCAXIS_Z;

	SetDataType(DataType);

	_fileHeader.Xlen = CellSizeX;
	_fileHeader.Ylen = CellSizeY;
	_fileHeader.Zlen = CellSizeZ;

	_fileHeader.ALPHA = CellAngleAlpha;
	_fileHeader.BETA  = CellAngleBeta;
	_fileHeader.GAMMA = CellAngleGamma;

	_fileHeader.XORIGIN = OriginX;
	_fileHeader.YORIGIN = OriginY;
	_fileHeader.ZORIGIN = OriginZ;

	memcpy(_fileHeader.CMAP, "MAP\0", 4);

	_fileHeader.RMS = RMS;

	for (int i = 0; i < DimZ; i++)
	{
		if (i == 0)
			_fileHeader.AMIN = MinValue[i];
		else
			_fileHeader.AMIN = min(_fileHeader.AMIN, MinValue[i]);

		if (i == 0)
			_fileHeader.AMAX = MaxValue[i];
		else
			_fileHeader.AMAX = max(_fileHeader.AMAX, MaxValue[i]);
		_fileHeader.AMEAN += MeanValue[i];

		_extHeaders[i].a_tilt = TiltAlpha[i];
		_extHeaders[i].b_tilt = TiltBeta[i];

		_extHeaders[i].x_stage = StageXPosition[i];
		_extHeaders[i].y_stage = StageYPosition[i];
		_extHeaders[i].z_stage = StageZPosition[i];

		_extHeaders[i].x_shift = ShiftX[i];
		_extHeaders[i].y_shift = ShiftY[i];
		//_extHeaders[i].z_shift = ShiftZ[i];

		_extHeaders[i].defocus = Defocus[i];
		_extHeaders[i].exp_time = ExposureTime[i];
		_extHeaders[i].tilt_axis = TiltAxis[i];
		_extHeaders[i].pixel_size = PixelSize[i];
		_extHeaders[i].magnification = Magnification[i];
		_extHeaders[i].binning = Binning[i];
		_extHeaders[i].appliedDefocus = AppliedDefocus[i];
		_extHeaders[0].ht = (float)Voltage;
	}
	_fileHeader.AMEAN /= (float)DimZ;
}
