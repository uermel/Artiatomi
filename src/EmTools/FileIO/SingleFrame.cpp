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


#include "SingleFrame.h"

SingleFrame::SingleFrame(string aFileName, void(*readStatusCallback)(FileReader::FileReaderStatus)) : ImageBase(), _file(NULL), _fileType(FT_NONE)
{
	if (!CanReadFile(aFileName, _fileType))
	{
		throw FileIOException(aFileName, "This file doesn't seem to be a single frame image file.");
	}
	
	if (_fileType == FT_DM3)
	{
		_file = new Dm3File(aFileName);
		Dm3File* temp = reinterpret_cast<Dm3File*>(_file);
		temp->readStatusCallback = readStatusCallback;
		bool erg = temp->OpenAndRead();
		temp->readStatusCallback = NULL;
		if (!erg)
		{
			throw FileIOException(aFileName, "Could not read image file.");
		}
	}

	if (_fileType == FT_DM4)
	{
		_file = new Dm4File(aFileName);
		Dm4File* temp = reinterpret_cast<Dm4File*>(_file);
		temp->readStatusCallback = readStatusCallback;
		bool erg = temp->OpenAndRead();
		temp->readStatusCallback = NULL;
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

	if (_fileType == FT_SER)
	{
		_file = new SERFile(aFileName);
		SERFile* temp = reinterpret_cast<SERFile*>(_file);
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

	if (_fileType == FT_TIFF)
	{
		_file = new TIFFFile(aFileName);
		TIFFFile* temp = reinterpret_cast<TIFFFile*>(_file);
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
		//make sure at the end of all reading, that we have 100% status...
		FileReader::FileReaderStatus status;
		status.bytesToRead = 1;
		status.bytesRead = 1;
		(*readStatusCallback)(status);
	}
}

SingleFrame::~SingleFrame()
{
	if (_file)
	{
		if (_fileType == FT_DM3)
		{
			delete reinterpret_cast<Dm3File*>(_file);
		}

		if (_fileType == FT_DM4)
		{
			delete reinterpret_cast<Dm4File*>(_file);
		}

		if (_fileType == FT_MRC)
		{
			delete reinterpret_cast<MRCFile*>(_file);
		}

		if (_fileType == FT_SER)
		{
			delete reinterpret_cast<SERFile*>(_file);
		}

		if (_fileType == FT_EM)
		{
			delete reinterpret_cast<EmFile*>(_file);
		};

		if (_fileType == FT_TIFF)
		{
			delete reinterpret_cast<TIFFFile*>(_file);
		};
	}
	

	_file = NULL;
}


ImageType_enum SingleFrame::GetImageType()
{
	return ImageType_enum::IT_SINGLEFRAME;
}

FileType_enum SingleFrame::GetFileType()
{
	return _fileType;
}

bool SingleFrame::NeedsFlipOnYAxis()
{
	if (_fileType == FT_DM3)
	{
		return true;
	}

	if (_fileType == FT_DM4)
	{
		return true;
	}

	if (_fileType == FT_MRC)
	{
		return false;
	}

	if (_fileType == FT_SER)
	{
		return false;
	}

	if (_fileType == FT_EM)
	{
		return false;
	}

	if (_fileType == FT_TIFF)
	{
		TIFFFile* temp = reinterpret_cast<TIFFFile*>(_file);
		return temp->NeedsFlipOnYAxis();
	}

	return false;
}

DataType_enum SingleFrame::GetFileDataType()
{
	if (_fileType == FT_DM3)
	{
		Dm3File* temp = reinterpret_cast<Dm3File*>(_file);
		return temp->GetDataType();		
	}

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

	if (_fileType == FT_SER)
	{
		SERFile* temp = reinterpret_cast<SERFile*>(_file);
		return temp->GetDataType();
	}

	if (_fileType == FT_EM)
	{
		EmFile* temp = reinterpret_cast<EmFile*>(_file);
		return temp->GetDataType();
	}

	if (_fileType == FT_TIFF)
	{
		TIFFFile* temp = reinterpret_cast<TIFFFile*>(_file);
		return temp->GetDataType();
	}

	return DataType_enum::DT_UNKNOWN;
}

bool SingleFrame::CanReadFile(string aFilename, FileType_enum& fileType)
{
	int width, height;
	DataType_enum dt;
	return CanReadFile(aFilename, fileType, width, height, dt);
}

bool SingleFrame::CanReadFile(string aFilename, FileType_enum& fileType, int& aWidth, int& aHeight, DataType_enum& aDataType)
{
	//first check filename ending
	fileType = ImageBase::GuessFileTypeFromEnding(aFilename);

	if (fileType == FT_DM3)
	{
		Dm3File dm3(aFilename);
		if (dm3.OpenAndReadHeader())
		{
			aWidth = dm3.GetImageDimensionX();
			aHeight = dm3.GetImageDimensionY();
			aDataType = dm3.GetDataType();
			return true;
		}
		return false;
	}

	if (fileType == FT_DM4)
	{
		Dm4File dm4(aFilename);
		if (dm4.OpenAndReadHeader())
		{
			uint dimZ = dm4.GetImageDimensionZ();
			if (dimZ == 1 || dimZ == 0)
			{
				aWidth = dm4.GetImageDimensionX();
				aHeight = dm4.GetImageDimensionY();
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
			if (dimZ == 1 || dimZ == 0)
			{
				aWidth = mrc.GetFileHeader().NX;
				aHeight = mrc.GetFileHeader().NY;
				aDataType = mrc.GetDataType();
				return true;
			}
		}
		return false;
	}

	if (fileType == FT_SER)
	{
		SERFile ser(aFilename);
		if (ser.OpenAndReadHeader())
		{
			if (ser.GetFileHeader().DataTypeID == 0x4122)
			{
				aWidth = ser.GetElementHeader().ArraySizeX;
				aHeight = ser.GetElementHeader().ArraySizeY;
				aDataType = ser.GetDataType();
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
			if (em.GetFileHeader().DimZ == 1)
			{
				aWidth = em.GetFileHeader().DimX;
				aHeight = em.GetFileHeader().DimY;
				aDataType = em.GetDataType();
				return true;
			}
		}
		return false;
	}

	if (fileType == FT_TIFF)
	{
		TIFFFile tiff(aFilename);
		if (tiff.OpenAndReadHeader())
		{			
			aWidth = tiff.GetWidth();
			aHeight = tiff.GetHeight();
			aDataType = tiff.GetDataType();
			return true;
		}
		return false;
	}

	return false;
}

bool SingleFrame::CanReadFile(string aFilename)
{
	FileType_enum fileType;
	int width, height;
	DataType_enum dt;

	return 	CanReadFile(aFilename, fileType, width, height, dt);
}

void* SingleFrame::GetData()
{
	if (_fileType == FT_DM3)
	{
		Dm3File* temp = reinterpret_cast<Dm3File*>(_file);
		return temp->GetImageData();
	}

	if (_fileType == FT_DM4)
	{
		Dm4File* temp = reinterpret_cast<Dm4File*>(_file);
		return temp->GetImageData();
	}

	if (_fileType == FT_MRC)
	{
		MRCFile* temp = reinterpret_cast<MRCFile*>(_file);
		return temp->GetData();
	}

	if (_fileType == FT_SER)
	{
		SERFile* temp = reinterpret_cast<SERFile*>(_file);
		return temp->GetData();
	}

	if (_fileType == FT_EM)
	{
		EmFile* temp = reinterpret_cast<EmFile*>(_file);
		return temp->GetData();
	}

	if (_fileType == FT_TIFF)
	{
		TIFFFile* temp = reinterpret_cast<TIFFFile*>(_file);
		return temp->GetData();
	}

	return NULL;
}

uint SingleFrame::GetWidth()
{
	if (_fileType == FT_DM3)
	{
		Dm3File* temp = reinterpret_cast<Dm3File*>(_file);
		return temp->GetImageDimensionX();
	}

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

	if (_fileType == FT_SER)
	{
		SERFile* temp = reinterpret_cast<SERFile*>(_file);
		return temp->GetElementHeader().ArraySizeX;
	}

	if (_fileType == FT_EM)
	{
		EmFile* temp = reinterpret_cast<EmFile*>(_file);
		return temp->GetFileHeader().DimX;
	}

	if (_fileType == FT_TIFF)
	{
		TIFFFile* temp = reinterpret_cast<TIFFFile*>(_file);
		return temp->GetWidth();
	}

	return 0;
}

uint SingleFrame::GetHeight()
{
	if (_fileType == FT_DM3)
	{
		Dm3File* temp = reinterpret_cast<Dm3File*>(_file);
		return temp->GetImageDimensionY();
	}

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

	if (_fileType == FT_SER)
	{
		SERFile* temp = reinterpret_cast<SERFile*>(_file);
		return temp->GetElementHeader().ArraySizeY;
	}

	if (_fileType == FT_EM)
	{
		EmFile* temp = reinterpret_cast<EmFile*>(_file);
		return temp->GetFileHeader().DimY;
	}

	if (_fileType == FT_TIFF)
	{
		TIFFFile* temp = reinterpret_cast<TIFFFile*>(_file);
		return temp->GetHeight();
	}

	return 0;
}

float SingleFrame::GetPixelSize()
{
	if (_fileType == FT_DM3)
	{
		Dm3File* temp = reinterpret_cast<Dm3File*>(_file);
		return temp->GetPixelSizeX();
	}

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

	if (_fileType == FT_SER)
	{
		SERFile* temp = reinterpret_cast<SERFile*>(_file);
		return temp->GetPixelSizeX();
	}

	if (_fileType == FT_EM)
	{
		return 0;
	}

	if (_fileType == FT_TIFF)
	{
		TIFFFile* temp = reinterpret_cast<TIFFFile*>(_file);
		return temp->GetPixelSize();
	}

	return 0;
}

bool SingleFrame::GetIsPlanar()
{
	if (_fileType == FT_TIFF)
	{
		TIFFFile* temp = reinterpret_cast<TIFFFile*>(_file);
		return temp->GetIsPlanar();
	}
	return false;
}

SingleFrame* SingleFrame::CreateInstance(string aFileName, void(*readStatusCallback)(FileReader::FileReaderStatus))
{
	return new SingleFrame(aFileName, readStatusCallback);
}