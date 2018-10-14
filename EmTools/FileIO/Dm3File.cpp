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


#include "Dm3File.h"

using namespace std;

Dm3File::Dm3File(string aFilename) : FileReader(aFilename), version(0), _data(NULL)
{
	root = NULL;
}

Dm3File::Dm3File(fstream* aStream) : FileReader(aStream), version(0), _data(NULL)
{
	root = NULL;
}

Dm3File::Dm3File(istream* aStream) : FileReader(aStream, true), version(0), _data(NULL)
{
	root = NULL;
}

Dm3File::~Dm3File()
{
	if (root)
		delete root;
	root = NULL;
}

uint Dm3File::GetPixelDepthInBytes()
{
	Dm3FileTagDirectory* dataDir = GetImageDataDir();
	if (dataDir == NULL) return 0;

	Dm3FileTag* pixelDepth = dataDir->FindTag("PixelDepth");
	if (pixelDepth == NULL) return 0;

	return pixelDepth->GetSingleValueInt(pixelDepth->InfoArray[0]);

}
uint Dm3File::GetImageSizeInBytes()
{
	return GetPixelDepthInBytes() * GetImageDimensionX() * GetImageDimensionY();
}
uint Dm3File::GetImageDimensionX()
{
	Dm3FileTagDirectory* dataDir = GetImageDataDir();
	if (dataDir == NULL) return 0;
	Dm3FileTagDirectory* dimDir = dataDir->FindTagDir("Dimensions");
	if (dimDir == NULL) return 0;
	Dm3FileTag* dimX = dimDir->Tags[0];

	return dimX->GetSingleValueInt(dimX->InfoArray[0]);
}
uint Dm3File::GetImageDimensionY()
{
	Dm3FileTagDirectory* dataDir = GetImageDataDir();
	if (dataDir == NULL) return 0;
	Dm3FileTagDirectory* dimDir = dataDir->FindTagDir("Dimensions");
	if (dimDir == NULL) return 0;
	Dm3FileTag* dimY = dimDir->Tags[1];

	return dimY->GetSingleValueInt(dimY->InfoArray[0]);
}
void* Dm3File::GetImageData()
{
	Dm3FileTagDirectory* dataDir = GetImageDataDir();
	if (dataDir == NULL) return NULL;

	Dm3FileTag* data = dataDir->FindTag("Data");
	if (data == NULL) return NULL;

	return data->Data;
}


Dm3FileTagDirectory* Dm3File::GetImageDataDir()
{
	Dm3FileTagDirectory* thumbnail = root->FindTagDir("Thumbnails");
	int imgIndex = -1;
	Dm3FileTagDirectory* img_thumbnail;

	if (thumbnail != NULL)
	{
		//Find largest thumbnail = the image
		int thumbSize = 0;
		for	(uint i = 0; i < thumbnail->TagDirs.size(); i++)
		{
			if (thumbnail->TagDirs[i]->FindTag("SourceSize_Pixels")->GetStructValueAsInt(0) > thumbSize)
			{
				thumbSize = thumbnail->TagDirs[i]->FindTag("SourceSize_Pixels")->GetStructValueAsInt(0);
				imgIndex = *((int*)(thumbnail->TagDirs[i]->FindTag("ImageIndex")->Data)) + 1; //Warum auch immer +1...
			}
		}

		img_thumbnail = root->FindTagDir("ImageList");

		if (img_thumbnail == NULL) return NULL;

		return img_thumbnail->TagDirs[imgIndex]->FindTagDir("ImageData");
	}
	return NULL;
}

Dm3FileTagDirectory* Dm3File::GetImageTagsDir()
{
	Dm3FileTagDirectory* thumbnail = root->FindTagDir("Thumbnails");
	int imgIndex = -1;
	Dm3FileTagDirectory* img_thumbnail;

	if (thumbnail != NULL)
	{
		//Find largest thumbnail = the image
		int thumbSize = 0;
		for	(uint i = 0; i < thumbnail->TagDirs.size(); i++)
		{
			if (thumbnail->TagDirs[i]->FindTag("SourceSize_Pixels")->GetStructValueAsInt(0) > thumbSize)
			{
				thumbSize = thumbnail->TagDirs[i]->FindTag("SourceSize_Pixels")->GetStructValueAsInt(0);
				imgIndex = *((int*)(thumbnail->TagDirs[i]->FindTag("ImageIndex")->Data)) + 1; //Warum auch immer +1...
			}
		}

		img_thumbnail = root->FindTagDir("ImageList");

		if (img_thumbnail == NULL) return NULL;

		return img_thumbnail->TagDirs[imgIndex]->FindTagDir("ImageTags");
	}
	return NULL;
}

bool Dm3File::OpenAndRead()
{
	bool res = FileReader::OpenRead();

	/*if (!res)
	{
		return false;
	}*/

	version = ReadUI4BE();
	fileSize = ReadUI4BE();
	mIsLittleEndian = ReadUI4BE() != 0;
	if (isFileStream)
		root = new Dm3FileTagDirectory(mFile, mIsLittleEndian);
	else
		root = new Dm3FileTagDirectory(mIStream, mIsLittleEndian);
	_data = GetImageData();
	return true;
}

bool Dm3File::OpenAndReadHeader()
{
	bool res = FileReader::OpenRead();

	if (!res)
	{
		return false;
	}

	version = ReadUI4BE();
	fileSize = ReadUI4BE();
	mIsLittleEndian = ReadUI4BE() != 0;
	if (isFileStream)
		root = new Dm3FileTagDirectory(mFile, mIsLittleEndian, false, true);
	else
		root = new Dm3FileTagDirectory(mIStream, mIsLittleEndian, false);
	//Test();
	//_data = GetImageData();
	return true;
}


//! Converts from em data type enum to internal data type
/*!
*/
DataType_enum Dm3File::GetDataType()
{
	Dm3FileTagDirectory* dataDir = GetImageDataDir();
	if (dataDir == NULL) return DT_UNKNOWN;
	Dm3FileTag* typeTag = dataDir->FindTag("DataType");
	if (typeTag == NULL) return DT_UNKNOWN;

	int type = typeTag->GetSingleValueInt(typeTag->InfoArray[0]);

	switch (type)
	{

	case DTT_UNKNOWN:
		return DT_UNKNOWN;
	case DTT_I2:
		return DT_SHORT;
	case DTT_F4:
		return DT_FLOAT;
	case DTT_C8:
		return DT_UNKNOWN;
	case DTT_OBSOLETE:
		return DT_UNKNOWN;
	case DTT_C4:
		return DT_UNKNOWN;
	case DTT_UI1:
		return DT_UCHAR;
	case DTT_I4:
		return DT_INT;
	case DTT_RGB_4UI1:
		return DT_UNKNOWN;
	case DTT_I1:
		return DT_CHAR;
	case DTT_UI2:
		return DT_USHORT;
	case DTT_UI4:
		return DT_UINT;
	case DTT_F8:
		return DT_DOUBLE;
	case DTT_C16:
		return DT_UNKNOWN;
	case DTT_BINARY:
		return DT_CHAR;
	case DTT_RGBA_4UI1:
		return DT_UNKNOWN;
	}
	return DT_UNKNOWN;
}

float Dm3File::GetExposureTime()
{
	Dm3FileTagDirectory* tagsDir = GetImageTagsDir();
	if (tagsDir == NULL) return 0;
	Dm3FileTagDirectory* acqDir = tagsDir->FindTagDir("Acquisition");
	if (acqDir == NULL) return 0;
	Dm3FileTagDirectory* parDir = acqDir->FindTagDir("Parameters");
	if (parDir == NULL) return 0;
	Dm3FileTagDirectory* hilDir = parDir->FindTagDir("High Level");
	if (hilDir == NULL) return 0;

	Dm3FileTag* exp = hilDir->FindTag("Exposure (s)");

	return exp->GetSingleValueFloat(exp->InfoArray[0]);
}

float Dm3File::GetPixelSizeX()
{
	Dm3FileTagDirectory* dataDir = GetImageDataDir();
	if (dataDir == NULL) //might be a TIFF embedded DM3
	{
		/*Dm3FileTagDirectory* level1 = root->FindTagDir("Acquisition");
		if (!level1) return 0;
		Dm3FileTagDirectory* level2 = level1->FindTagDir("Frame");
		if (!level2) return 0;
		Dm3FileTagDirectory* level3 = level2->FindTagDir("CCD");
		if (!level3) return 0;
		Dm3FileTag* tag = level3->FindTag("Pixel Size (um)");
		if (!tag) return 0;
		return tag->GetSingleValueFloat(tag->InfoArray[0]);*/

		return 0;
	}
	
	Dm3FileTagDirectory* calDir = dataDir->FindTagDir("Calibrations");
	if (calDir == NULL) return 0;
	Dm3FileTagDirectory* dimDir = calDir->FindTagDir("Dimension");
	if (dimDir == NULL) return 0;
	if (dimDir->TagDirs.size() < 2) return 0;

	Dm3FileTag* scaleX = dimDir->TagDirs[0]->FindTag("Scale");
	Dm3FileTag* unit = dimDir->TagDirs[0]->FindTag("Units");
	float scaleUnit = 1;
	if (unit)
	{
		std::string s = unit->GetSingleValueString(unit->InfoArray[1]);
		if (s == "µm")
			scaleUnit = 1000;
		if (s == "nm")
			scaleUnit = 1;
	}

	return scaleX->GetSingleValueFloat(scaleX->InfoArray[0]) * scaleUnit;
}

float Dm3File::GetPixelSizeY()
{
	Dm3FileTagDirectory* dataDir = GetImageDataDir();
	if (dataDir == NULL) return 0;
	Dm3FileTagDirectory* calDir = dataDir->FindTagDir("Calibrations");
	if (calDir == NULL) return 0;
	Dm3FileTagDirectory* dimDir = calDir->FindTagDir("Dimension");
	if (dimDir == NULL) return 0;
	if (dimDir->TagDirs.size() < 2) return 0;

	Dm3FileTag* scaleX = dimDir->TagDirs[1]->FindTag("Scale");
	Dm3FileTag* unit = dimDir->TagDirs[0]->FindTag("Units");
	float scaleUnit = 1;
	if (unit)
	{
		std::string s = unit->GetSingleValueString(unit->InfoArray[1]);
		if (s == "µm")
			scaleUnit = 1000;
		if (s == "nm")
			scaleUnit = 1;
	}

	return scaleX->GetSingleValueFloat(scaleX->InfoArray[0]) * scaleUnit;
}

string Dm3File::GetAcquisitionDate()
{
	Dm3FileTagDirectory* tagsDir = GetImageTagsDir();
	if (tagsDir == NULL) return 0;
	Dm3FileTagDirectory* datDir = tagsDir->FindTagDir("DataBar");
	if (datDir == NULL) return 0;

	Dm3FileTag* date = datDir->FindTag("Acquisition Date");

	return date->GetSingleValueString(date->InfoArray[1]);
}

string Dm3File::GetAcquisitionTime()
{
	Dm3FileTagDirectory* tagsDir = GetImageTagsDir();
	if (tagsDir == NULL) return 0;
	Dm3FileTagDirectory* datDir = tagsDir->FindTagDir("DataBar");
	if (datDir == NULL) return 0;

	Dm3FileTag* time = datDir->FindTag("Acquisition Time");

	return time->GetSingleValueString(time->InfoArray[1]);
}

int Dm3File::GetCs()
{
	Dm3FileTagDirectory* tagsDir = GetImageTagsDir();
	if (tagsDir == NULL) return 0;
	Dm3FileTagDirectory* micDir = tagsDir->FindTagDir("Microscope Info");
	if (micDir == NULL) return 0;

	Dm3FileTag* cs = micDir->FindTag("Cs(mm)");

	return cs->GetSingleValueInt(cs->InfoArray[0]);
}

int Dm3File::GetVoltage()
{
	Dm3FileTagDirectory* tagsDir = GetImageTagsDir();
	if (tagsDir == NULL) return 0;
	Dm3FileTagDirectory* micDir = tagsDir->FindTagDir("Microscope Info");
	if (micDir == NULL) return 0;

	Dm3FileTag* v = micDir->FindTag("Voltage");

	return v->GetSingleValueInt(v->InfoArray[0]);
}

int Dm3File::GetMagnification()
{
	Dm3FileTagDirectory* tagsDir = GetImageTagsDir();
	if (tagsDir == NULL) return 0;
	Dm3FileTagDirectory* micDir = tagsDir->FindTagDir("Microscope Info");
	if (micDir == NULL) return 0;

	Dm3FileTag* mag = micDir->FindTag("Actual Magnification");

	return mag->GetSingleValueInt(mag->InfoArray[0]);
}

float Dm3File::GetTiltAngle(int aIndex)
{
	//Meta Data.Dimension info.2.Data.0
	/*Dm3FileTagDirectory* tagsDir = GetImageTagsDir();
	if (tagsDir == NULL) return 0;
	Dm3FileTagDirectory* metDir = tagsDir->FindTagDir("Meta Data");
	if (metDir == NULL) return 0;
	Dm3FileTagDirectory* dimDir = metDir->FindTagDir("Dimension info");
	if (dimDir == NULL) return 0;
	Dm3FileTagDirectory* zwoDir = dimDir->FindTagDir("2");
	if (zwoDir == NULL) return 0;
	Dm3FileTagDirectory* datDir = zwoDir->FindTagDir("Data");
	if (datDir == NULL) return 0;

	stringstream ss();
	ss << aIndex;

	Dm3FileTag* null = datDir->FindTag(ss.str());
	if (null == NULL) return 0;
	return null->GetSingleValueFloat(null->InfoArray[0]);*/

	Dm3FileTagDirectory* tagsDir = GetImageTagsDir();
	if (tagsDir == NULL) return 0;
	Dm3FileTagDirectory* tomDir = tagsDir->FindTagDir("Tomography");
	if (tomDir == NULL) return 0;
	Dm3FileTagDirectory* traDir = tomDir->FindTagDir("Tracking data");
	if (traDir == NULL) return 0;
	Dm3FileTagDirectory* onlDir = traDir->FindTagDir("Online tracking data");
	if (onlDir == NULL) return 0;
	if (onlDir->TagDirs.size() <= (uint)aIndex) return 0;

	Dm3FileTag* tilt = onlDir->TagDirs[aIndex]->FindTag("stage angle (degree)");
	if (tilt == NULL) return 0;
	return tilt->GetSingleValueFloat(tilt->InfoArray[0]);
}

float Dm3File::GetTiltAngle()
{
	//Meta Data.Dimension info.2.Data.0
	Dm3FileTagDirectory* tagsDir = GetImageTagsDir();
	if (tagsDir == NULL) return 0;
	Dm3FileTagDirectory* metDir = tagsDir->FindTagDir("Meta Data");
	if (metDir == NULL) return 0;
	Dm3FileTagDirectory* dimDir = metDir->FindTagDir("Dimension info");
	if (dimDir == NULL) return 0;
	Dm3FileTagDirectory* zwoDir = dimDir->FindTagDir("2");
	if (zwoDir == NULL) return 0;
	Dm3FileTagDirectory* datDir = zwoDir->FindTagDir("Data");
	if (datDir == NULL) return 0;

	if (datDir->Tags.size() < 1) return 0;
	Dm3FileTag* null = datDir->Tags[0];
	if (null == NULL) return 0;
	return null->GetSingleValueFloat(null->InfoArray[0]);

	/*Dm3FileTagDirectory* tagsDir = GetImageTagsDir();
	if (tagsDir == NULL) return 0;
	Dm3FileTagDirectory* tomDir = tagsDir->FindTagDir("Tomography");
	if (tomDir == NULL) return 0;
	Dm3FileTagDirectory* traDir = tomDir->FindTagDir("Tracking data");
	if (traDir == NULL) return 0;
	Dm3FileTagDirectory* onlDir = traDir->FindTagDir("Online tracking data");
	if (onlDir == NULL) return 0;
	if (onlDir->TagDirs.size() <= aIndex) return 0;

	Dm3FileTag* tilt = onlDir->TagDirs[aIndex]->FindTag("stage angle (degree)");
	if (tilt == NULL) return 0;
	return tilt->GetSingleValueFloat(tilt->InfoArray[0]);*/
}
