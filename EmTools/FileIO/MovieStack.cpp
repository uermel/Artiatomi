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


#include "MovieStack.h"

MovieStack::MovieStack(string aFileName, void(*readStatusCallback)(FileReader::FileReaderStatus)) : ImageBase(), _file(NULL), _fileType(FT_NONE)
{
	if (!CanReadFile(aFileName, _fileType))
	{
		throw FileIOException(aFileName, "This file doesn't seem to be a movie stack image file.");
	}

	if (_fileType == FT_DM3)
	{
		_file = new Dm3File(aFileName);
		Dm3File* temp = reinterpret_cast<Dm3File*>(_file);
		bool erg = temp->OpenAndRead();
		if (!erg)
		{
			throw FileIOException(aFileName, "Could not read image file.");
		}
	}

	if (_fileType == FT_DM4)
	{
		_file = new Dm4File(aFileName);
		Dm4File* temp = reinterpret_cast<Dm4File*>(_file);
		bool erg = temp->OpenAndRead();
		if (!erg)
		{
			throw FileIOException(aFileName, "Could not read image file.");
		}
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

MovieStack::~MovieStack()
{
	if (_file)
	{
		if (_fileType == FT_DM4)
		{
			delete reinterpret_cast<Dm4File*>(_file);
		}

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


ImageType_enum MovieStack::GetImageType()
{
	return ImageType_enum::IT_MOVIESTACK;
}

FileType_enum MovieStack::GetFileType()
{
	return _fileType;
}

bool MovieStack::NeedsFlipOnYAxis()
{
	if (_fileType == FT_DM4)
	{
		return true;
	}

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

DataType_enum MovieStack::GetFileDataType()
{
	if (_fileType == FT_DM4)
	{
		Dm4File* temp = reinterpret_cast<Dm4File*>(_file);
		return temp->GetDataType();
	}

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

bool MovieStack::CanReadFile(string aFilename, FileType_enum& fileType)
{
	int width, height, frameCount;
	DataType_enum dt;
	return CanReadFile(aFilename, fileType, width, height, frameCount, dt);
}

bool MovieStack::CanReadFile(string aFilename, FileType_enum& fileType, int& aWidth, int& aHeight, int& aFrameCount, DataType_enum& aDataType)
{
	//first check filename ending
	fileType = ImageBase::GuessFileTypeFromEnding(aFilename);

	if (fileType == FT_DM3)
	{
		//We never had any dm3 files with movie frames (dose fractionation)...
		return false;
	}

	if (fileType == FT_DM4)
	{
		Dm4File dm4(aFilename);
		if (dm4.OpenAndReadHeader())
		{
			uint dimZ = dm4.GetImageDimensionZ();
			if (dimZ > 1)
			{
				aWidth = dm4.GetImageDimensionX();
				aHeight = dm4.GetImageDimensionY();
				aFrameCount = dimZ;
				aDataType = dm4.GetDataType();
				return true;
			}
		}
		return false;
	}

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
				aFrameCount = dimZ;
				aDataType = mrc.GetDataType();
				return true;
			}
		}
		return false;
	}

	if (fileType == FT_EM)
	{
		//we never had this case, but who knows who might need it...
		EmFile em(aFilename);
		if (em.OpenAndReadHeader())
		{
			int dimZ = em.GetFileHeader().DimZ;
			if (dimZ > 1)
			{
				aWidth = em.GetFileHeader().DimX;
				aHeight = em.GetFileHeader().DimY;
				aFrameCount = dimZ;
				aDataType = em.GetDataType();
				return true;
			}
		}
		return false;
	}

	return false;
}

bool MovieStack::CanReadFile(string aFilename)
{
	FileType_enum fileType;

	return 	CanReadFile(aFilename, fileType);
}

void* MovieStack::GetData(size_t idx)
{
	if (_fileType == FT_DM4)
	{
		Dm4File* temp = reinterpret_cast<Dm4File*>(_file);

		if (idx >= temp->GetImageDimensionZ())
			return NULL;

		size_t imageSize = temp->GetImageSizeInBytes();
		char* data = reinterpret_cast<char*>(temp->GetImageData()) + idx * imageSize;
		return data;
	}

	if (_fileType == FT_MRC)
	{
		MRCFile* temp = reinterpret_cast<MRCFile*>(_file);
		return temp->GetData(idx);
	}

	if (_fileType == FT_EM)
	{
		EmFile* temp = reinterpret_cast<EmFile*>(_file);
		return temp->GetData(idx);
	}

	return NULL;
}

uint MovieStack::GetWidth()
{
	if (_fileType == FT_DM4)
	{
		Dm4File* temp = reinterpret_cast<Dm4File*>(_file);
		return temp->GetImageDimensionX();
	}

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

uint MovieStack::GetHeight()
{
	if (_fileType == FT_DM4)
	{
		Dm4File* temp = reinterpret_cast<Dm4File*>(_file);
		return temp->GetImageDimensionY();
	}

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

uint MovieStack::GetImageCount()
{
	if (_fileType == FT_DM4)
	{
		Dm4File* temp = reinterpret_cast<Dm4File*>(_file);
		return temp->GetImageDimensionZ();
	}

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

float MovieStack::GetPixelSize()
{
	if (_fileType == FT_DM4)
	{
		Dm4File* temp = reinterpret_cast<Dm4File*>(_file);
		return temp->GetPixelSizeX();
	}

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

MovieStack* MovieStack::CreateInstance(string aFileName, void(*readStatusCallback)(FileReader::FileReaderStatus))
{
	return new MovieStack(aFileName, readStatusCallback);
}