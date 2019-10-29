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


#ifndef DM4FILE_H
#define DM4FILE_H

#include "../Basics/Default.h"
#include "FileReader.h"
#include "Dm4FileTagDirectory.h"

//!  Dm4File represents a gatan *.dm4 file in memory and maps contained information to the default internal Image format.
/*!
	Dm4File gives access to header infos, image data.
	\author Michael Kunz
	\date   September 2011
	\version 1.0
*/
class Dm4File : public FileReader
{
public:
	Dm4File(std::string filename);
	Dm4File(istream* stream);
	virtual ~Dm4File();

	uint version;
	ulong64 fileSize;
	char* _data;

	Dm4FileTagDirectory* root;

	uint GetPixelDepthInBytes();
	uint GetImageSizeInBytes();
	char* GetImageData();
	uint GetImageDimensionX();
	uint GetImageDimensionY();
	uint GetImageDimensionZ();
	void* GetThumbnailData();
	uint GetThumbnailDimensionX();
	uint GetThumbnailDimensionY();
	float GetPixelSizeX();
	float GetPixelSizeY();
	float GetExposureTime();
	string GetAcquisitionDate();
	string GetAcquisitionTime();
	int GetCs();
	int GetVoltage();
	int GetMagnification();
	float GetTiltAngle(int aIndex);
	float GetTiltAngle();
	bool OpenAndRead();
	bool OpenAndReadHeader();
	DataType_enum GetDataType();

private:
	Dm4FileTagDirectory* GetImageDataDir();
	Dm4FileTagDirectory* GetImageTagsDir();
	Dm4FileTagDirectory* GetThumbnailDataDir();
	Dm4FileTagDirectory* GetThumbnailTagsDir();

	friend class Dm4FileStack;
};

#endif
