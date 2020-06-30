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


#include "TiltSeries.h"
#include <iomanip>
#include <sstream>
#include <string.h>

TiltSeries::TiltSeries(string aFileName, void(*readStatusCallback)(FileReader::FileReaderStatus)) : ImageBase(), _files(), _fileType(FT_NONE)
{
	if (!CanReadFile(aFileName, _fileType))
	{
		throw FileIOException(aFileName, "This file doesn't seem to be a tilt series file.");
	}

	if (_fileType == FT_DM3)
	{
		vector<string> names = GetMultiFilesFromFilename(aFileName);

		for (size_t i = 0; i < names.size(); i++)
		{
			Dm3File* temp = new Dm3File(names[i]);
			bool erg = temp->OpenAndRead();
			if (!erg)
			{
				throw FileIOException(names[i], "Could not read image file.");
			}
			if (readStatusCallback) 
			{
				FileReader::FileReaderStatus status;
				status.bytesRead = i + 1;
				status.bytesToRead = names.size();
				(*readStatusCallback)(status);
			}
			_files.push_back(temp);
		}
	}

	if (_fileType == FT_DM4)
	{
		vector<string> names = GetMultiFilesFromFilename(aFileName);

		for (size_t i = 0; i < names.size(); i++)
		{
			Dm4File* temp = new Dm4File(names[i]);
			bool erg = temp->OpenAndRead();
			if (!erg)
			{
				throw FileIOException(names[i], "Could not read image file.");
			}
			if (readStatusCallback)
			{
				FileReader::FileReaderStatus status;
				status.bytesRead = i + 1;
				status.bytesToRead = names.size();
				(*readStatusCallback)(status);
			}
			_files.push_back(temp);
		}
	}

	if (_fileType == FT_MRC)
	{
		MRCFile* temp = new MRCFile(aFileName);
		temp->readStatusCallback = readStatusCallback;
		bool erg = temp->OpenAndRead();
		temp->readStatusCallback = NULL;
		if (!erg)
		{
			throw FileIOException(aFileName, "Could not read image file.");
		}
		_files.push_back(temp);
	}

	if (_fileType == FT_EM)
	{
		EmFile* temp = new EmFile(aFileName);
		temp->readStatusCallback = readStatusCallback;
		bool erg = temp->OpenAndRead();
		temp->readStatusCallback = NULL;
		if (!erg)
		{
			throw FileIOException(aFileName, "Could not read image file.");
		}
		_files.push_back(temp);
	}
	if (readStatusCallback)
	{
		FileReader::FileReaderStatus status;
		status.bytesRead = 1;
		status.bytesToRead = 1;
		(*readStatusCallback)(status);
	}
}

TiltSeries::~TiltSeries()
{
	for (size_t i = 0; i < _files.size(); i++)
	{
		if (_files[i])
		{
			if (_fileType == FT_DM3)
			{
				delete reinterpret_cast<Dm3File*>(_files[i]);
			}

			if (_fileType == FT_DM4)
			{
				delete reinterpret_cast<Dm4File*>(_files[i]);
			}

			if (_fileType == FT_MRC)
			{
				delete reinterpret_cast<MRCFile*>(_files[i]);
			}

			if (_fileType == FT_EM)
			{
				delete reinterpret_cast<EmFile*>(_files[i]);
			};
		}
		_files[i] = NULL;
	}
	_files.clear();
}


ImageType_enum TiltSeries::GetImageType()
{
	return ImageType_enum::IT_TILTSERIES;
}

FileType_enum TiltSeries::GetFileType()
{
	return _fileType;
}

bool TiltSeries::NeedsFlipOnYAxis()
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

	if (_fileType == FT_EM)
	{
		return false;
	}

	return false;
}

DataType_enum TiltSeries::GetFileDataType()
{
	if (_files.size() < 1)
		return DataType_enum::DT_UNKNOWN;

	if (!_files[0])
		return DataType_enum::DT_UNKNOWN;

	if (_fileType == FT_DM3)
	{
		Dm3File* temp = reinterpret_cast<Dm3File*>(_files[0]);
		return temp->GetDataType();
	}

	if (_fileType == FT_DM4)
	{
		Dm4File* temp = reinterpret_cast<Dm4File*>(_files[0]);
		return temp->GetDataType();
	}

	if (_fileType == FT_MRC)
	{
		MRCFile* temp = reinterpret_cast<MRCFile*>(_files[0]);
		return temp->GetDataType();
	}

	if (_fileType == FT_EM)
	{
		EmFile* temp = reinterpret_cast<EmFile*>(_files[0]);
		return temp->GetDataType();
	}

	return DataType_enum::DT_UNKNOWN;
}

bool TiltSeries::CanReadFile(string aFilename, FileType_enum& fileType)
{
	int width, height, count;
	DataType_enum dt;

	return 	CanReadFile(aFilename, fileType, width, height, count, dt);
}

bool TiltSeries::CanReadFile(string aFilename, FileType_enum & fileType, int & aWidth, int & aHeight, int & aImageCount, DataType_enum & aDataType)
{
	//first check filename ending
	fileType = ImageBase::GuessFileTypeFromEnding(aFilename);

	if (fileType == FT_DM3 && FileIsPartOfMultiFiles(aFilename))
	{
		Dm3File dm3(aFilename);
		if (dm3.OpenAndReadHeader())
		{
			aWidth = dm3.GetImageDimensionX();
			aHeight = dm3.GetImageDimensionY();
			int nothing;
			aImageCount = CountFilesInStack(aFilename, nothing);
			aDataType = dm3.GetDataType();
			return true;
		}
		return false;
	}

	if (fileType == FT_DM4 && FileIsPartOfMultiFiles(aFilename))
	{
		Dm4File dm4(aFilename);
		if (dm4.OpenAndReadHeader())
		{
			uint dimZ = dm4.GetImageDimensionZ();
			if (dimZ == 1 || dimZ == 0)
			{
				aWidth = dm4.GetImageDimensionX();
				aHeight = dm4.GetImageDimensionY();
				int nothing;
				aImageCount = CountFilesInStack(aFilename, nothing);
				aDataType = dm4.GetDataType();
				return true;
			}
			else
			{
				return false;
			}
		}
		return false;
	}

        if (fileType == FT_DM4 && !FileIsPartOfMultiFiles(aFilename))
        {
                Dm4File dm4(aFilename);
                if (dm4.OpenAndReadHeader())
                {
                        uint dimZ = dm4.GetImageDimensionZ();
                        if (dimZ > 1)
                        {
                                aWidth = dm4.GetImageDimensionX();
                                aHeight = dm4.GetImageDimensionY();
                                aImageCount = dimZ;
                                aDataType = dm4.GetDataType();
                                return true;
                        }
                        else
                        {
                                return false;
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
				aImageCount = dimZ;
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
				aImageCount = dimZ;
				aDataType = em.GetDataType();
				return true;
			}
		}
		return false;
	}

	return false;
}

bool TiltSeries::CanReadFile(string aFilename)
{
	FileType_enum fileType;
	int width, height, count;
	DataType_enum dt;

	return 	CanReadFile(aFilename, fileType, width, height, count, dt);
}

void* TiltSeries::GetData(size_t idx)
{
	if (_fileType == FT_DM3)
	{
		if (idx >= _files.size())
		{
			return NULL;
		}
		Dm3File* temp = reinterpret_cast<Dm3File*>(_files[idx]);
		return temp->GetImageData();
	}

	if (_fileType == FT_DM4)
	{
		if (idx >= _files.size())
		{
			return NULL;
		}
		Dm4File* temp = reinterpret_cast<Dm4File*>(_files[idx]);
		return temp->GetImageData();
	}

	if (_fileType == FT_MRC)
	{
		MRCFile* temp = reinterpret_cast<MRCFile*>(_files[0]);
		return temp->GetData(idx);
	}

	if (_fileType == FT_EM)
	{
		EmFile* temp = reinterpret_cast<EmFile*>(_files[0]);
		return temp->GetData(idx);
	}

	return NULL;
}

uint TiltSeries::GetWidth()
{
	if (_files.size() < 1)
		return 0;

	if (!_files[0])
		return 0;

	if (_fileType == FT_DM3)
	{
		Dm3File* temp = reinterpret_cast<Dm3File*>(_files[0]);
		return temp->GetImageDimensionX();
	}

	if (_fileType == FT_DM4)
	{
		Dm4File* temp = reinterpret_cast<Dm4File*>(_files[0]);
		return temp->GetImageDimensionX();
	}

	if (_fileType == FT_MRC)
	{
		MRCFile* temp = reinterpret_cast<MRCFile*>(_files[0]);
		return temp->GetFileHeader().NX;
	}

	if (_fileType == FT_EM)
	{
		EmFile* temp = reinterpret_cast<EmFile*>(_files[0]);
		return temp->GetFileHeader().DimX;
	}

	return 0;
}

uint TiltSeries::GetHeight()
{
	if (_files.size() < 1)
		return 0;

	if (!_files[0])
		return 0;

	if (_fileType == FT_DM3)
	{
		Dm3File* temp = reinterpret_cast<Dm3File*>(_files[0]);
		return temp->GetImageDimensionY();
	}

	if (_fileType == FT_DM4)
	{
		Dm4File* temp = reinterpret_cast<Dm4File*>(_files[0]);
		return temp->GetImageDimensionY();
	}

	if (_fileType == FT_MRC)
	{
		MRCFile* temp = reinterpret_cast<MRCFile*>(_files[0]);
		return temp->GetFileHeader().NY;
	}

	if (_fileType == FT_EM)
	{
		EmFile* temp = reinterpret_cast<EmFile*>(_files[0]);
		return temp->GetFileHeader().DimY;
	}

	return 0;
}

uint TiltSeries::GetImageCount()
{
	if (_files.size() < 1)
		return 0;

	if (!_files[0])
		return 0;

	if (_fileType == FT_DM3)
	{
		return (uint)_files.size();
	}

	if (_fileType == FT_DM4)
	{
		return (uint)_files.size();
	}

	if (_fileType == FT_MRC)
	{
		MRCFile* temp = reinterpret_cast<MRCFile*>(_files[0]);
		return temp->GetFileHeader().NZ;
	}

	if (_fileType == FT_EM)
	{
		EmFile* temp = reinterpret_cast<EmFile*>(_files[0]);
		return temp->GetFileHeader().DimZ;
	}

	return 0;
}

float TiltSeries::GetPixelSize()
{
	if (_fileType == FT_DM3)
	{
		Dm3File* temp = reinterpret_cast<Dm3File*>(_files[0]);
		return temp->GetPixelSizeX();
	}

	if (_fileType == FT_DM4)
	{
		Dm4File* temp = reinterpret_cast<Dm4File*>(_files[0]);
		return temp->GetPixelSizeX();
	}

	if (_fileType == FT_MRC)
	{
		MRCFile* temp = reinterpret_cast<MRCFile*>(_files[0]);
		return temp->GetPixelsize();
	}

	if (_fileType == FT_EM)
	{
		return 1;
	}

	return 0;
}

float TiltSeries::GetTiltAngle(size_t idx)
{
	if (!_files[0])
		return 0;


	if (_fileType == FT_MRC)
	{
		MRCFile* temp = reinterpret_cast<MRCFile*>(_files[0]);
		if (idx >= temp->GetFileHeader().NZ)
                        return 0.0f;
                MrcExtendedHeader* extHeaders = temp->GetFileExtHeaders();
                if (extHeaders != NULL)
                {
                    return temp->GetFileExtHeaders()[idx].a_tilt;
                }
                else
                {
                    return 0.0f;
                }
	}

	if (_fileType == FT_DM4)
	{
		if (idx >= _files.size())
			return 0;
		Dm4File* temp = reinterpret_cast<Dm4File*>(_files[idx]);
		return temp->GetTiltAngle();

	}
	if (_fileType == FT_DM3)
	{
		if (idx >= _files.size())
			return 0;
		Dm3File* temp = reinterpret_cast<Dm3File*>(_files[idx]);
		return temp->GetTiltAngle();
	}
	return 0;
}


bool TiltSeries::FileIsPartOfMultiFiles(string aFileName)
{
	int nothing;
	int count = CountFilesInStack(aFileName, nothing);

	return count > 1;
}

vector<string> TiltSeries::GetMultiFilesFromFilename(string aFileName)
{
	int firstIndex;
	int count = CountFilesInStack(aFileName, firstIndex);

	vector<string> result;
	
	if (count > 1)
	{
		for (int i = firstIndex; i < count + firstIndex; i++)
		{
			string file = GetFileNameFromIndex(i, aFileName);
			result.push_back(file);
		}
	}
	else
	{
		result.push_back(aFileName);
	}

	return result;
}


bool TiltSeries::fexists(string aFileName)
{
	ifstream ifile(aFileName.c_str());
	return !!ifile;
}

string TiltSeries::GetStringFromInt(int aInt)
{
	ostringstream ss;
	ss << setfill('0') << setw(4);
	ss << aInt;
	return ss.str();
}

string TiltSeries::GetFileNameFromIndex(int aIndex, std::string aFileName)
{
	string fileEnding(".dm4");
	const int count0 = 4;
	size_t pos = aFileName.find_last_of(fileEnding);
	if (pos == string::npos)
	{
		fileEnding = ".dm3";
		pos = aFileName.find_last_of(fileEnding);
		if (pos == string::npos)
			return "";
	}

	//Substract length of search string:
	pos -= fileEnding.length() + 3;
	return aFileName.replace(pos, count0, GetStringFromInt(aIndex));
}

int TiltSeries::CountFilesInStack(std::string aFileName, int& aFirstIndex)
{
	bool notFinished = true;
	int fileCounter = 0;
	bool firstFound = false;
	aFirstIndex = -1;
	int notUnlimited = 0;

	while (notFinished)
	{
		string file = GetFileNameFromIndex(fileCounter, aFileName);
		if (!fexists(file))
		{
			if (firstFound)
			{
				notFinished = false;
				fileCounter--;
			}
			else
			{
				notUnlimited++;
			}
		}
		else
		{
			if (!firstFound)
				aFirstIndex = fileCounter;
			firstFound = true;
		}
		fileCounter++;
		if (notUnlimited > 100) return 0;
	}
        return fileCounter - aFirstIndex;
}

TiltSeries* TiltSeries::CreateInstance(string aFileName, void(*readStatusCallback)(FileReader::FileReaderStatus))
{
	return new TiltSeries(aFileName, readStatusCallback);
}
