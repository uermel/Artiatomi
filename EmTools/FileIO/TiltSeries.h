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


#ifndef TILTSERIES_H
#define TILTSERIES_H

#include "FileIO.h"
#include "ImageBase.h"
#include <vector>

using namespace std;

class TiltSeries : public ImageBase
{
public:
	TiltSeries(string aFileName, void(*readStatusCallback)(FileReader::FileReaderStatus) = NULL);
	~TiltSeries();

	virtual ImageType_enum GetImageType();
	virtual FileType_enum GetFileType();
	virtual DataType_enum GetFileDataType();
	virtual bool NeedsFlipOnYAxis();

	static bool CanReadFile(string aFilename);
	static bool CanReadFile(string aFilename, FileType_enum& fileType);
	static bool CanReadFile(string aFilename, FileType_enum& fileType, int& aWidth, int& aHeight, int &aImageCount, DataType_enum& aDataType);

	void* GetData(size_t idx);
	uint GetWidth();
	uint GetHeight();
	uint GetImageCount();
	float GetPixelSize();
	float GetTiltAngle(size_t idx);

	static TiltSeries* CreateInstance(string aFileName, void(*readStatusCallback)(FileReader::FileReaderStatus) = NULL);
private:

	FileType_enum _fileType;
	vector<void*> _files;

	static bool FileIsPartOfMultiFiles(string aFileName);
	static vector<string> GetMultiFilesFromFilename(string aFileName);
	static int CountFilesInStack(std::string aFileName, int& aFirstIndex);
	static bool fexists(string aFileName);
	static string GetStringFromInt(int aInt);
	static string GetFileNameFromIndex(int aIndex, string aFileName);

};

#endif