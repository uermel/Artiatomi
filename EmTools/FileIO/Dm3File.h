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


#ifndef DM3FILE_H
#define DM3FILE_H

#include "../Basics/Default.h"
#include "FileReader.h"
#include "Dm3FileTagDirectory.h"

//!  Dm3File represents a gatan *.dm3 file in memory and maps contained information to the default internal Image format. 
/*!
	Dm3File gives access to header infos, image data.
	\author Michael Kunz
	\date   September 2011
	\version 1.0
*/
class Dm3File : public FileReader
{
public:
	Dm3File(std::string filename);
	Dm3File(fstream* stream);
	Dm3File(istream* stream);
	virtual ~Dm3File();

	uint version;
	uint fileSize;
	void* _data;

	Dm3FileTagDirectory* root;

	uint GetPixelDepthInBytes();
	uint GetImageSizeInBytes();
	void* GetImageData();
	uint GetImageDimensionX();
	uint GetImageDimensionY();
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
	Dm3FileTagDirectory* GetImageDataDir();
	Dm3FileTagDirectory* GetImageTagsDir();

	friend class Dm3FileStack;
};

#endif