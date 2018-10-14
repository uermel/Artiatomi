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


#ifndef MRCFILE_H
#define MRCFILE_H

#include "mrcHeader.h"
#include "ProjectionSource.h"

using namespace std;

//!  MRCFile represents a *.mrc file in memory and maps contained information to the default internal Image format.
/*!
	MRCFile gives access to header infos, volume data and single projections.
	\author Michael Kunz
	\date   October 2012
	\version 2.0
*/
class MRCFile : public ProjectionSource
{
private:
	MrcHeader _fileHeader;
	MrcExtendedHeader* _extHeaders;
	uint _GetDataTypeSize(MrcMode_Enum aDataType);
	uint _dataStartPosition;
	float** _projectionCache;


public:
	//! Creates a new MRCFile instance. The file name is only set internally; the file itself keeps untouched.
	MRCFile(string aFileName);
	//! Creates a new MRCFile instance and copies information from an existing Image. The file name is only set internally; the file itself keeps untouched.
	MRCFile(string aFileName, const Image& aImage);
	~MRCFile();

	//! Opens the file File#mFileName and reads the entire content.
	/*!
		\throw FileIOException
	*/
	bool OpenAndRead();

	//! Opens the file File#mFileName and reads only the file header.
	/*!
		\throw FileIOException
	*/
	bool OpenAndReadHeader();

	//! Opens the file File#mFileName and writes the entire content.
	/*!
		\throw FileIOException
	*/
	bool OpenAndWrite();

	//! Converts from MRC data type enum to internal data type
	/*!
		MRCFile::GetDataType dows not take into account if the data type is unsigned or signed as the
		MRC file format cannot distinguish them.
	*/
	FileDataType_enum GetDataType();

	//! Converts from internal data type enum to mrc data type and sets the flag in the file header
	/*!
		MRCFile::SetDataType dows not take into account if the data type is unsigned or signed as the
		MRC file format cannot distinguish them.
	*/
	void SetDataType(FileDataType_enum aType);

	//! Returns the size of the data block. If the header is not yet read, it will return 0.
	size_t GetDataSize();

	//! Creates a copy for this mrcfile from \p aHeader.
	void SetFileHeader(MrcHeader& aHeader);
	//! Returns a reference to the inner mrc file header.
	MrcHeader& GetFileHeader();

	//! Sets the extended header section.
	void SetFileExtHeader(MrcExtendedHeader* aHeader, int aCount);

	//! Returns a pointer to the extended header section.
	MrcExtendedHeader* GetFileExtHeaders();

	//! Sets the inner data pointer to \p aData.
	void SetData(char* aData);

	//! Returns the inner data pointer.
	char* GetData();

	//! Reads the projection with index \p aIndex from file and returns it's pointer.
	char* GetProjection(uint aIndex);

	//! Reads the projection converted to float with index \p aIndex from file and returns it's pointer.
	float* GetProjectionFloat(uint aIndex);

	//! Reads the projection converted to float with index \p aIndex from file and returns it's pointer. Values are inverted.
	float* GetProjectionInvertFloat(uint aIndex);

	//! Reads the information from the mrc file header and stores in general Image format.
	void ReadHeaderInfo();

	//! Reads the information from the general Image format and stores it in the mrc file header.
	void WriteInfoToHeader();
};

#endif
