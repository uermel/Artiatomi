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


#include "Volume.h"

Volume::Volume(string aFileName, void(*readStatusCallback)(FileReader::FileReaderStatus)) : ImageBase(), _file(NULL), _fileType(FT_NONE)
{
	if (!CanReadFile(aFileName, _fileType))
	{
		throw FileIOException(aFileName, "This file doesn't seem to be a volume image file.");
	}

	if (_fileType == FT_MRC)
	{
		_file = new MRCFile(aFileName);
		MRCFile* temp = reinterpret_cast<MRCFile*>(_file);
		temp->readStatusCallback = readStatusCallback;
		bool erg = temp->OpenAndRead();
		temp->readStatusCallback = NULL;
		if (!erg)
		{
			throw FileIOException(aFileName, "Could not read image file.");
		}
	}

	if (_fileType == FT_EM)
	{
		_file = new EmFile(aFileName);
		EmFile* temp = reinterpret_cast<EmFile*>(_file);
		temp->readStatusCallback = readStatusCallback;
		bool erg = temp->OpenAndRead();
		temp->readStatusCallback = NULL;
		if (!erg)
		{
			throw FileIOException(aFileName, "Could not read image file.");
		}
	}
	if (readStatusCallback)
	{
		FileReader::FileReaderStatus status;
		status.bytesRead = 1;
		status.bytesToRead = 1;
		(*readStatusCallback)(status);
	}
}

Volume::~Volume()
{
	if (_file)
	{
		if (_fileType == FT_MRC)
		{
			delete reinterpret_cast<MRCFile*>(_file);
		}

		if (_fileType == FT_EM)
		{
			delete reinterpret_cast<EmFile*>(_file);
		};
	}

	_file = NULL;
}


ImageType_enum Volume::GetImageType()
{
	return ImageType_enum::IT_VOLUME;
}

FileType_enum Volume::GetFileType()
{
	return _fileType;
}

bool Volume::NeedsFlipOnYAxis()
{
	if (_fileType == FT_MRC)
	{
		return false;
	}

	if (_fileType == FT_EM)
	{
		return false;
	}

	return false;
}

DataType_enum Volume::GetFileDataType()
{
	if (_fileType == FT_MRC)
	{
		MRCFile* temp = reinterpret_cast<MRCFile*>(_file);
		return temp->GetDataType();
	}

	if (_fileType == FT_EM)
	{
		EmFile* temp = reinterpret_cast<EmFile*>(_file);
		return temp->GetDataType();
	}

	return DataType_enum::DT_UNKNOWN;
}

bool Volume::CanReadFile(string aFilename, FileType_enum& fileType)
{
	int width, height, depth;
	DataType_enum dt;
	return CanReadFile(aFilename, fileType, width, height, depth, dt);
}

bool Volume::CanReadFile(string aFilename, FileType_enum& fileType, int& aWidth, int& aHeight, int& aDepth, DataType_enum& aDataType)
{
	//first check filename ending
	fileType = ImageBase::GuessFileTypeFromEnding(aFilename);
	
	if (fileType == FT_MRC)
	{
		MRCFile mrc(aFilename);
		if (mrc.OpenAndReadHeader())
		{
			uint dimZ = mrc.GetFileHeader().NZ;
			if (dimZ > 1)
			{
				aWidth = mrc.GetFileHeader().NX;
				aHeight = mrc.GetFileHeader().NY;
				aDepth = dimZ;
				aDataType = mrc.GetDataType();
				return true;
			}
		}
		return false;
	}

	if (fileType == FT_EM)
	{
		EmFile em(aFilename);
		if (em.OpenAndReadHeader())
		{
			int dimZ = em.GetFileHeader().DimZ;
			if (dimZ > 1)
			{
				aWidth = em.GetFileHeader().DimX;
				aHeight = em.GetFileHeader().DimY;
				aDepth = dimZ;
				aDataType = em.GetDataType();
				return true;
			}
		}
		return false;
	}

	return false;
}

bool Volume::CanReadFile(string aFilename)
{
	FileType_enum fileType;

	return 	CanReadFile(aFilename, fileType);
}

void* Volume::GetData()
{
	if (_fileType == FT_MRC)
	{
		MRCFile* temp = reinterpret_cast<MRCFile*>(_file);
		return temp->GetData();
	}

	if (_fileType == FT_EM)
	{
		EmFile* temp = reinterpret_cast<EmFile*>(_file);
		return temp->GetData();
	}

	return NULL;
}

uint Volume::GetWidth()
{
	if (_fileType == FT_MRC)
	{
		MRCFile* temp = reinterpret_cast<MRCFile*>(_file);
		return temp->GetFileHeader().NX;
	}

	if (_fileType == FT_EM)
	{
		EmFile* temp = reinterpret_cast<EmFile*>(_file);
		return temp->GetFileHeader().DimX;
	}

	return 0;
}

uint Volume::GetHeight()
{
	if (_fileType == FT_MRC)
	{
		MRCFile* temp = reinterpret_cast<MRCFile*>(_file);
		return temp->GetFileHeader().NY;
	}

	if (_fileType == FT_EM)
	{
		EmFile* temp = reinterpret_cast<EmFile*>(_file);
		return temp->GetFileHeader().DimY;
	}

	return 0;
}

uint Volume::GetDepth()
{
	if (_fileType == FT_MRC)
	{
		MRCFile* temp = reinterpret_cast<MRCFile*>(_file);
		return temp->GetFileHeader().NZ;
	}

	if (_fileType == FT_EM)
	{
		EmFile* temp = reinterpret_cast<EmFile*>(_file);
		return temp->GetFileHeader().DimZ;
	}

	return 0;
}

float Volume::GetPixelSize()
{
	if (_fileType == FT_MRC)
	{
		MRCFile* temp = reinterpret_cast<MRCFile*>(_file);
		return temp->GetPixelsize();
	}

	if (_fileType == FT_EM)
	{
		return 1;
	}

	return 0;
}

Volume* Volume::CreateInstance(string aFileName, void(*readStatusCallback)(FileReader::FileReaderStatus))
{
	return new Volume(aFileName, readStatusCallback);
}