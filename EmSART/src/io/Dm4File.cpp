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


#include "Dm4File.h"

using namespace std;

//bool Dm4File::Test2(Dm4FileTagDirectory* dir, int& count, string& ort)
//{
//	for (int i = 0; i < dir->TagDirs.size(); i++)
//	{
//		Test2(dir->TagDirs[i], count, ort);
//		if (dir->TagDirs[i]->Name == "ImageData")
//			ort += " - " + dir->Name + " - ";
//	}
//
//	for (int i = 0; i < dir->Tags.size(); i++)
//	{
//		if (dir->Tags[i]->Name == "Data")
//		{
//			count++;
//			ort += ";" + dir->Name + ";";
//		}
//	}
//	return 0;
//}
//
//bool Dm4File::Test()
//{
//	int count = 0;
//	string ort;
//	count = 0;
//	ort = "";
//
//
//	uint a = GetPixelDepthInBytes();
//	uint b = GetImageSizeInBytes();
//	char* c = GetImageData();
//	uint d = GetImageDimensionX();
//	uint e = GetImageDimensionY();
//	float f = GetPixelSizeX();
//	float g =  GetPixelSizeY();
//	float h = GetExposureTime();
//	string i = GetAcquisitionDate();
//	string j = GetAcquisitionTime();
//	int k = GetCs();
//	int l = GetVoltage();
//	int m = GetMagnification();
//	float n = GetTiltAngle(0);
//	float o = GetTiltAngle();
//	FileDataType_enum p = GetDataType();
//
//
//
//
//
//
//
//
//
//
//	Dm4FileTagDirectory* dir = root->TagDirs[4];
//	Dm4FileTagDirectory* imageList = root->FindTagDir("ImageList");
//
//
//	int maxDimX = 0;
//	Dm4FileTagDirectory* image = NULL;
//	for (int i = 0; i < imageList->TagDirs.size(); i++)
//	{
//		Dm4FileTagDirectory* tmp = imageList->TagDirs[i];
//		if (tmp == NULL) continue;
//		Dm4FileTagDirectory* imgData = tmp->FindTagDir("ImageData");
//		if (imgData == NULL) continue;
//		Dm4FileTagDirectory* dimensions = imgData->FindTagDir("Dimensions");
//		if (dimensions == NULL) continue;
//		int dim = dimensions->Tags[0]->GetSingleValueInt(DTTI4);
//		if (dim > maxDimX)
//		{
//			maxDimX = dim;
//			image = tmp;
//		}
//	}
//	if (image == NULL) return 0;
//
//
//	dir = dir->TagDirs[1];
//
//	Test2(dir->TagDirs[0], count, ort);
//	count = 0;
//	ort = "";
//	Test2(dir->TagDirs[1], count, ort);
//	count = 0;
//	ort = "";
//	Test2(root->TagDirs[2], count, ort);
//	count = 0;
//	ort = "";
//	Test2(root->TagDirs[3], count, ort);
//	count = 0;
//	ort = "";
//	Test2(root->TagDirs[4], count, ort);
//	count = 0;
//	ort = "";
//	Test2(root->TagDirs[5], count, ort);
//	return 0;
//}

Dm4File::Dm4File(string aFilename) : FileReader(aFilename), Image(), version(0)
{
	root = NULL;
}

Dm4File::~Dm4File()
{
	if (root)
		delete root;
	root = NULL;
	//gets deleted in tagDir
	_data = NULL;
}

uint Dm4File::GetPixelDepthInBytes()
{
	Dm4FileTagDirectory* dataDir = GetImageDataDir();
	if (dataDir == NULL) return 0;

	Dm4FileTag* pixelDepth = dataDir->FindTag("PixelDepth");
	if (pixelDepth == NULL) return 0;

	return pixelDepth->GetSingleValueInt(pixelDepth->InfoArray[0]);

}
uint Dm4File::GetImageSizeInBytes()
{
	return GetPixelDepthInBytes() * GetImageDimensionX() * GetImageDimensionY();
}
uint Dm4File::GetImageDimensionX()
{
	//printf("_1:\n");
	Dm4FileTagDirectory* dataDir = GetImageDataDir();
	//printf("_2:\n");
	if (dataDir == NULL) return 0;
	Dm4FileTagDirectory* dimDir = dataDir->FindTagDir("Dimensions");
	if (dimDir == NULL) return 0;
	Dm4FileTag* dimX = dimDir->Tags[0];

	return dimX->GetSingleValueInt(dimX->InfoArray[0]);
}
uint Dm4File::GetThumbnailDimensionX()
{
	Dm4FileTagDirectory* dataDir = GetThumbnailDataDir();
	if (dataDir == NULL) return 0;
	Dm4FileTagDirectory* dimDir = dataDir->FindTagDir("Dimensions");
	if (dimDir == NULL) return 0;
	Dm4FileTag* dimX = dimDir->Tags[0];

	return dimX->GetSingleValueInt(dimX->InfoArray[0]);
}
uint Dm4File::GetImageDimensionY()
{
	Dm4FileTagDirectory* dataDir = GetImageDataDir();
	if (dataDir == NULL) return 0;
	Dm4FileTagDirectory* dimDir = dataDir->FindTagDir("Dimensions");
	if (dimDir == NULL) return 0;
	Dm4FileTag* dimY = dimDir->Tags[1];

	return dimY->GetSingleValueInt(dimY->InfoArray[0]);
}
uint Dm4File::GetThumbnailDimensionY()
{
	Dm4FileTagDirectory* dataDir = GetThumbnailDataDir();
	if (dataDir == NULL) return 0;
	Dm4FileTagDirectory* dimDir = dataDir->FindTagDir("Dimensions");
	if (dimDir == NULL) return 0;
	Dm4FileTag* dimY = dimDir->Tags[1];

	return dimY->GetSingleValueInt(dimY->InfoArray[0]);
}
char* Dm4File::GetImageData()
{
	Dm4FileTagDirectory* dataDir = GetImageDataDir();
	if (dataDir == NULL) return NULL;

	Dm4FileTag* data = dataDir->FindTag("Data");
	if (data == NULL) return NULL;

	return data->Data;
}
char* Dm4File::GetThumbnailData()
{
	Dm4FileTagDirectory* dataDir = GetThumbnailDataDir();
	if (dataDir == NULL) return NULL;

	Dm4FileTag* data = dataDir->FindTag("Data");
	if (data == NULL) return NULL;

	return data->Data;
}


Dm4FileTagDirectory* Dm4File::GetImageDataDir()
{
	//printf("__1: %p\n", root);
	Dm4FileTagDirectory* imageList = root->FindTagDir("ImageList");
	//printf("__2\n");
	if (imageList == NULL) return NULL;
	//rintf("__3\n");

	int maxDimX = 0;
	Dm4FileTagDirectory* imageData = NULL;

	//printf("__4\n");
	for (int i = 0; i < imageList->TagDirs.size(); i++)
	{
		Dm4FileTagDirectory* tmp = imageList->TagDirs[i];
	//printf("__5\n");
		if (tmp == NULL) continue;
		Dm4FileTagDirectory* imgData = tmp->FindTagDir("ImageData");
	//printf("__6\n");
		if (imgData == NULL) continue;
		Dm4FileTagDirectory* dimensions = imgData->FindTagDir("Dimensions");
	//printf("__7\n");
		if (dimensions == NULL) continue;
	//printf("__8\n");
		int dim = dimensions->Tags[0]->GetSingleValueInt(DTTI4);
	//printf("__9\n");
		if (dim > maxDimX)
		{
			maxDimX = dim;
			imageData = imgData;
		}
	}

	return imageData;
}


Dm4FileTagDirectory* Dm4File::GetThumbnailDataDir()
{
	Dm4FileTagDirectory* imageList = root->FindTagDir("ImageList");
	if (imageList == NULL) return NULL;

	int minDimX = 2000000;
	Dm4FileTagDirectory* imageData = NULL;

	for (int i = 0; i < imageList->TagDirs.size(); i++)
	{
		Dm4FileTagDirectory* tmp = imageList->TagDirs[i];
		if (tmp == NULL) continue;
		Dm4FileTagDirectory* imgData = tmp->FindTagDir("ImageData");
		if (imgData == NULL) continue;
		Dm4FileTagDirectory* dimensions = imgData->FindTagDir("Dimensions");
		if (dimensions == NULL) continue;
		int dim = dimensions->Tags[0]->GetSingleValueInt(DTTI4);
		if (dim < minDimX)
		{
			minDimX = dim;
			imageData = imgData;
		}
	}

	return imageData;
}

Dm4FileTagDirectory* Dm4File::GetImageTagsDir()
{
	Dm4FileTagDirectory* imageList = root->FindTagDir("ImageList");
	if (imageList == NULL) return NULL;

	int maxDimX = 0;
	Dm4FileTagDirectory* imageTags = NULL;

	for (int i = 0; i < imageList->TagDirs.size(); i++)
	{
		Dm4FileTagDirectory* tmp = imageList->TagDirs[i];
		if (tmp == NULL) continue;
		Dm4FileTagDirectory* imgData = tmp->FindTagDir("ImageData");
		if (imgData == NULL) continue;
		Dm4FileTagDirectory* imgTags = tmp->FindTagDir("ImageTags");
		if (imgTags == NULL) continue;
		Dm4FileTagDirectory* dimensions = imgData->FindTagDir("Dimensions");
		if (dimensions == NULL) continue;
		int dim = dimensions->Tags[0]->GetSingleValueInt(DTTI4);
		if (dim > maxDimX)
		{
			maxDimX = dim;
			imageTags = imgTags;
		}
	}

	return imageTags;
}

Dm4FileTagDirectory* Dm4File::GetThumbnailTagsDir()
{
	Dm4FileTagDirectory* imageList = root->FindTagDir("ImageList");
	if (imageList == NULL) return NULL;

	int minDimX = 2000000;
	Dm4FileTagDirectory* imageTags = NULL;

	for (int i = 0; i < imageList->TagDirs.size(); i++)
	{
		Dm4FileTagDirectory* tmp = imageList->TagDirs[i];
		if (tmp == NULL) continue;
		Dm4FileTagDirectory* imgData = tmp->FindTagDir("ImageData");
		if (imgData == NULL) continue;
		Dm4FileTagDirectory* imgTags = tmp->FindTagDir("ImageTags");
		if (imgTags == NULL) continue;
		Dm4FileTagDirectory* dimensions = imgData->FindTagDir("Dimensions");
		if (dimensions == NULL) continue;
		int dim = dimensions->Tags[0]->GetSingleValueInt(DTTI4);
		if (dim < minDimX)
		{
			minDimX = dim;
			imageTags = imgTags;
		}
	}

	return imageTags;
}

bool Dm4File::OpenAndRead()
{
	bool res = FileReader::OpenRead();

	if (!res)
	{
		return false;
	}

	version = ReadUI4BE();
	fileSize = ReadUI8BE();
	mIsLittleEndian = ReadUI4BE() != 0;

	root = new Dm4FileTagDirectory(mFile, mIsLittleEndian);
	//Test();
	_data = GetImageData();
	return true;
}

bool Dm4File::OpenAndReadThumbnail()
{
	bool res = FileReader::OpenRead();

	if (!res)
	{
		return false;
	}

	version = ReadUI4BE();
	fileSize = ReadUI8BE();
	mIsLittleEndian = ReadUI4BE() != 0;

	root = new Dm4FileTagDirectory(mFile, mIsLittleEndian, false, true);
	//Test();
	//_data = GetImageData();
	return true;
}


//! Converts from em data type enum to internal data type
/*!
*/
FileDataType_enum Dm4File::GetDataType()
{
	Dm4FileTagDirectory* dataDir = GetImageDataDir();
	if (dataDir == NULL) return FDT_UNKNOWN;
	Dm4FileTag* typeTag = dataDir->FindTag("DataType");
	if (typeTag == NULL) return FDT_UNKNOWN;

	int type = typeTag->GetSingleValueInt(typeTag->InfoArray[0]);

	switch (type)
	{

	case DTT_UNKNOWN:
		return FDT_UNKNOWN;
	case DTT_I2:
		return FDT_SHORT;
	case DTT_F4:
		return FDT_FLOAT;
	case DTT_C8:
		return FDT_UNKNOWN;
	case DTT_OBSOLETE:
		return FDT_UNKNOWN;
	case DTT_C4:
		return FDT_UNKNOWN;
	case DTT_UI1:
		return FDT_UCHAR;
	case DTT_I4:
		return FDT_INT;
	case DTT_RGB_4UI1:
		return FDT_UNKNOWN;
	case DTT_I1:
		return FDT_CHAR;
	case DTT_UI2:
		return FDT_USHORT;
	case DTT_UI4:
		return FDT_UINT;
	case DTT_F8:
		return FDT_DOUBLE;
	case DTT_C16:
		return FDT_UNKNOWN;
	case DTT_BINARY:
		return FDT_CHAR;
	case DTT_RGBA_4UI1:
		return FDT_UNKNOWN;
	}
	return FDT_UNKNOWN;
}

void Dm4File::ReadHeaderInfo()
{
	//Check if file is already read
	if (root == NULL) return;

	ClearImageMetadata();

	DimX = GetImageDimensionX();
	DimY = GetImageDimensionY();
	DimZ = 1;

	CellSizeX = GetPixelSizeX();
	CellSizeY = GetPixelSizeY();
	CellSizeZ = 0;

	DataType = GetDataType();

	Cs = GetCs();
	Voltage = GetVoltage();

	AllocArrays(DimZ);

	for (int i = 0; i < DimZ; i++)
	{
		TiltAlpha[i] = GetTiltAngle();

		Defocus[i] = 0;
		ExposureTime[i] = GetExposureTime();
		TiltAxis[i] = 0;
		PixelSize[i] = GetPixelSizeX();
		Magnification[i] = (float)GetMagnification();
	}
}

void Dm4File::WriteInfoToHeader()
{
	//We can't write dm4 files...
}

float Dm4File::GetExposureTime()
{
	Dm4FileTagDirectory* tagsDir = GetImageTagsDir();
	if (tagsDir == NULL) return 0;
	Dm4FileTagDirectory* acqDir = tagsDir->FindTagDir("Acquisition");
	if (acqDir == NULL) return 0;
	Dm4FileTagDirectory* parDir = acqDir->FindTagDir("Parameters");
	if (parDir == NULL) return 0;
	Dm4FileTagDirectory* hilDir = parDir->FindTagDir("High Level");
	if (hilDir == NULL) return 0;

	Dm4FileTag* exp = hilDir->FindTag("Exposure (s)");

	return exp->GetSingleValueFloat(exp->InfoArray[0]);
}

float Dm4File::GetPixelSizeX()
{
	Dm4FileTagDirectory* dataDir = GetImageDataDir();
	if (dataDir == NULL) return 0;
	Dm4FileTagDirectory* calDir = dataDir->FindTagDir("Calibrations");
	if (calDir == NULL) return 0;
	Dm4FileTagDirectory* dimDir = calDir->FindTagDir("Dimension");
	if (dimDir == NULL) return 0;
	if (dimDir->TagDirs.size() < 2) return 0;

	Dm4FileTag* scaleX = dimDir->TagDirs[0]->FindTag("Scale");

	//Convert nm to Angstrom: * 10
	return scaleX->GetSingleValueFloat(scaleX->InfoArray[0]) * 10.0f;
}

float Dm4File::GetPixelSizeY()
{
	Dm4FileTagDirectory* dataDir = GetImageDataDir();
	if (dataDir == NULL) return 0;
	Dm4FileTagDirectory* calDir = dataDir->FindTagDir("Calibrations");
	if (calDir == NULL) return 0;
	Dm4FileTagDirectory* dimDir = calDir->FindTagDir("Dimension");
	if (dimDir == NULL) return 0;
	if (dimDir->TagDirs.size() < 2) return 0;

	Dm4FileTag* scaleX = dimDir->TagDirs[1]->FindTag("Scale");

	//Convert nm to Angstrom: * 10
	return scaleX->GetSingleValueFloat(scaleX->InfoArray[0]) * 10.0f;
}

string Dm4File::GetAcquisitionDate()
{
	Dm4FileTagDirectory* tagsDir = GetImageTagsDir();
	if (tagsDir == NULL) return 0;
	Dm4FileTagDirectory* datDir = tagsDir->FindTagDir("DataBar");
	if (datDir == NULL) return 0;

	Dm4FileTag* date = datDir->FindTag("Acquisition Date");

	return date->GetSingleValueString(date->InfoArray[1]);
}

string Dm4File::GetAcquisitionTime()
{
	Dm4FileTagDirectory* tagsDir = GetImageTagsDir();
	if (tagsDir == NULL) return 0;
	Dm4FileTagDirectory* datDir = tagsDir->FindTagDir("DataBar");
	if (datDir == NULL) return 0;

	Dm4FileTag* time = datDir->FindTag("Acquisition Time");

	return time->GetSingleValueString(time->InfoArray[1]);
}

int Dm4File::GetCs()
{
	Dm4FileTagDirectory* tagsDir = GetImageTagsDir();
	if (tagsDir == NULL) return 0;
	Dm4FileTagDirectory* micDir = tagsDir->FindTagDir("Microscope Info");
	if (micDir == NULL) return 0;

	Dm4FileTag* cs = micDir->FindTag("Cs(mm)");

	return cs->GetSingleValueInt(cs->InfoArray[0]);
}

int Dm4File::GetVoltage()
{
	Dm4FileTagDirectory* tagsDir = GetImageTagsDir();
	if (tagsDir == NULL) return 0;
	Dm4FileTagDirectory* micDir = tagsDir->FindTagDir("Microscope Info");
	if (micDir == NULL) return 0;

	Dm4FileTag* v = micDir->FindTag("Voltage");

	return v->GetSingleValueInt(v->InfoArray[0]);
}

int Dm4File::GetMagnification()
{
	Dm4FileTagDirectory* tagsDir = GetImageTagsDir();
	if (tagsDir == NULL) return 0;
	Dm4FileTagDirectory* micDir = tagsDir->FindTagDir("Microscope Info");
	if (micDir == NULL) return 0;

	Dm4FileTag* mag = micDir->FindTag("Actual Magnification");

	return mag->GetSingleValueInt(mag->InfoArray[0]);
}

float Dm4File::GetTiltAngle(int aIndex)
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

	Dm4FileTagDirectory* tagsDir = GetImageTagsDir();
	if (tagsDir == NULL) return 0;
	Dm4FileTagDirectory* tomDir = tagsDir->FindTagDir("Tomography");
	if (tomDir == NULL) return 0;
	Dm4FileTagDirectory* traDir = tomDir->FindTagDir("Tracking data");
	if (traDir == NULL) return 0;
	Dm4FileTagDirectory* onlDir = traDir->FindTagDir("Online tracking data");
	if (onlDir == NULL) return 0;
	if (onlDir->TagDirs.size() <= (uint)aIndex) return 0;

	Dm4FileTag* tilt = onlDir->TagDirs[aIndex]->FindTag("stage angle (degree)");
	if (tilt == NULL) return 0;
	return tilt->GetSingleValueFloat(tilt->InfoArray[0]);
}

float Dm4File::GetTiltAngle()
{
	//Meta Data.Dimension info.2.Data.0
	Dm4FileTagDirectory* tagsDir = GetImageTagsDir();
	if (tagsDir == NULL) return 0;
	Dm4FileTagDirectory* metDir = tagsDir->FindTagDir("Meta Data");
	if (metDir == NULL) return 0;
	Dm4FileTagDirectory* dimDir = metDir->FindTagDir("Dimension info");
	if (dimDir == NULL) return 0;
	Dm4FileTagDirectory* zwoDir = dimDir->FindTagDir("2");
	if (zwoDir == NULL) return 0;
	Dm4FileTagDirectory* datDir = zwoDir->FindTagDir("Data");
	if (datDir == NULL) return 0;

	if (datDir->Tags.size() < 1) return 0;
	Dm4FileTag* null = datDir->Tags[0];
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
