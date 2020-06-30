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


#include "Dm3FileStack.h"
#include <iomanip>
#include <sstream>
#include <string.h>

using namespace std;

Dm3FileStack::Dm3FileStack(string aFileName)
	: FileReader(""), _dm3files(0), _fileCount(0)
{
	_fileCount = CountFilesInStack(aFileName);
	for (int i = 0; i < _fileCount; i++)
	{
		Dm3File* dm3 = new Dm3File(GetFileNameFromIndex(i, aFileName));
		_dm3files.push_back(dm3);
	}
}

Dm3FileStack::~Dm3FileStack()
{
	for (uint i = 0; i < _dm3files.size(); i++)
	{
		delete _dm3files[i];
	}

}


bool Dm3FileStack::fexists(string aFileName)
{
  ifstream ifile(aFileName.c_str());
  return !!ifile;
}

string Dm3FileStack::GetStringFromInt(int aInt)
{
	ostringstream ss;
	ss << setfill('0') << setw(4);
	ss << aInt;
	return ss.str();
}

string Dm3FileStack::GetFileNameFromIndex(int aIndex, std::string aFileName)
{
	string fileEnding("0000.dm3");
	const int count0 = 4;
	size_t pos = aFileName.find_last_of(fileEnding);
	if (pos == string::npos) return "";

	//Substract length of search string:
	pos -= fileEnding.length()-1;
	return aFileName.replace(pos, count0, GetStringFromInt(aIndex));
}

int Dm3FileStack::CountFilesInStack(std::string aFileName)
{
	bool notFinished = true;
	int fileCounter = 0;
	while (notFinished)
	{
		string file = GetFileNameFromIndex(fileCounter, aFileName);
		if (!fexists(file))
		{
			notFinished = false;
			fileCounter--;
		}
		fileCounter++;
	}
	return fileCounter;
}


bool Dm3FileStack::OpenAndRead()
{
    if (_fileCount == 0) return false;
	bool res = true;

	if (_data)
		delete[] _data;
	_data = NULL;

	int imgsize = 0;

	for (int i = 0; i < _fileCount; i++)
	{
		res &= _dm3files[i]->OpenAndRead();

		if (i == 0)
		{
			imgsize = _dm3files[0]->GetImageSizeInBytes();
			_data = new char[(long long)imgsize * (long long)_fileCount];
		}
		char* img = _dm3files[i]->GetImageData();
		memcpy(_data + (long long)imgsize * (long long)i, img, (long long)imgsize);
		_dm3files[i]->GetImageDataDir()->FindTag("Data")->FreeData();
		_dm3files[i]->_data = NULL;
	}

	if (!res)
	{
		delete[] _data;
		_data = NULL;
	}
	return res;
}

DataType_enum Dm3FileStack::GetDataType()
{
	if (_dm3files.size() < 1) return DT_UNKNOWN;
	return _dm3files[0]->GetDataType();
}



void Dm3FileStack::ReadHeaderInfo()
{
	if (_dm3files.size() < 1) return;


	ClearImageMetadata();

	DimX = _dm3files[0]->GetImageDimensionX();
	DimY = _dm3files[0]->GetImageDimensionY();
	DimZ = _fileCount;

	CellSizeX = _dm3files[0]->GetPixelSizeX();
	CellSizeY = _dm3files[0]->GetPixelSizeY();
	CellSizeZ = _dm3files[0]->GetPixelSizeY();

	DataType = _dm3files[0]->GetDataType();

	Cs = _dm3files[0]->GetCs();
	Voltage = _dm3files[0]->GetVoltage();

	AllocArrays(_fileCount);

	for (int i = 0; i < _fileCount; i++)
	{
		TiltAlpha[i] = _dm3files[i]->GetTiltAngle();

		Defocus[i] = 0;
		ExposureTime[i] = _dm3files[i]->GetExposureTime();
		TiltAxis[i] = 0;
		PixelSize[i] = _dm3files[i]->GetPixelSizeX();
		Magnification[i] = (float)_dm3files[i]->GetMagnification();
	}
}

void Dm3FileStack::WriteInfoToHeader()
{
	//We can't write dm3 files...
}

char* Dm3FileStack::GetData()
{
	return _data;
}
