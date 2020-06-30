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


#ifndef EMFILE_H
#define EMFILE_H

#include "IODefault.h"
#include "emHeader.h"
#include "FileReader.h"
#include "FileWriter.h"
#include "Image.h"

using namespace std;

//!  EMFile represents a *.em file in memory and maps contained information to the default internal Image format.
/*!
	EMFile gives access to header infos, volume data and single projections.
	\author Michael Kunz
	\date   September 2011
	\version 1.0
*/
class EMFile : public FileReader, public FileWriter, public Image
{
protected:
	EmHeader _fileHeader;
	uint _GetDataTypeSize(EmDataType_Enum aDataType);
	uint _dataStartPosition;

public:
	//! Creates a new EMFile instance. The file name is only set internally; the file itself keeps untouched.
	EMFile(string aFileName);
	//! Creates a new MRCFile instance and copies information from an existing Image. The file name is only set internally; the file itself keeps untouched.
	EMFile(string aFileName, const Image& aImage);

	EMFile* CreateEMFile(string aFileNameBase, int index);

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

	//! Converts from em data type enum to internal data type
	/*!
		EMFile::SetDataType dows not take into account if the data type is unsigned or signed as the
		EM file format cannot distinguish them.
	*/
	FileDataType_enum GetDataType();

	//! Converts from internal data type enum to em data type and sets the flag in the file header
	/*!
		EMFile::SetDataType dows not take into account if the data type is unsigned or signed as the
		EM file format cannot distinguish them.
	*/
	void SetDataType(FileDataType_enum aType);

	//! Returns the size of the data block. If the header is not yet read, it will return 0.
	size_t GetDataSize();

	//! Creates a copy for this em-file from \p aHeader.
	void SetFileHeader(EmHeader& aHeader);

	//! Returns a reference to the inner em file header.
	EmHeader& GetFileHeader();

	//! Sets the inner data pointer to \p aData.
	void SetData(char* aData);
	//! Returns the inner data pointer.
	char* GetData();

	//! Reads the projection with index \p aIndex from file and returns it's pointer.
	char* GetProjection(uint aIndex);

	//! Reads the information from the em file header and stores in general Image format.
	void ReadHeaderInfo();
	//! Reads the information from the general Image format and stores it in the em file header.
	void WriteInfoToHeader();

};

void emwrite(string aFileName, float* data, int width, int height);
void emread(string aFileName, float*& data, int& width, int& height);
void emwrite(string aFileName, float* data, int width, int height, int depth);
void emread(string aFileName, float*& data, int& width, int& height, int &depth);

#endif
