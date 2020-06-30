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

#include "../Basics/Default.h"
#include "EmHeader.h"
#include "FileReader.h"
#include "FileWriter.h"


using namespace std;

//!  EmFile represents a *.Em file in memory and maps contained information to the default internal Image format. 
/*!
EmFile gives access to header infos, volume data and single projections.
\author Michael Kunz
\date   September 2011
\version 1.0
*/
class EmFile : public FileReader, public FileWriter
{
protected:
	EmHeader _fileHeader;
	uint _GetDataTypeSize(EmDataType_Enum aDataType);
	uint _dataStartPosition;
	void* _data;

public:
	//! Creates a new EmFile instance. The file name is only set internally; the file itself keeps untouched.
	EmFile(string aFileName);

	~EmFile();

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

	//! Opens the file File#mFileName and writes only the header.
	/*!
	\throw FileIOException
	*/
	bool OpenAndWriteHeader();

	//! Determines if a given image dimension and datatype can be written to a EM file
	static bool CanWriteAsEM(int aDimX, int aDimY, int aDimZ, DataType_enum aDatatype);

	//! Opens the file File#mFileName and writes the header.
	static bool InitHeader(string aFileName, int aDimX, int aDimY, float aPixelSize, DataType_enum aDatatype);

	//! Opens the file File#mFileName and writes the header.
	static bool InitHeader(string aFileName, int aDimX, int aDimY, int aDimZ, float aPixelSize, DataType_enum aDatatype);

	//! Opens the file File#mFileName and writes the header.
	static bool InitHeader(string aFileName, EmHeader& header);

	//! Opens the file File#mFileName and writes the data after the previously written header.
	static bool WriteRawData(string aFileName, void* aData, size_t aSize);

	//! Initialises a header structure
	static bool SetHeaderData(EmHeader& header, int aDimX, int aDimY, int aDimZ, float aPixelSize, DataType_enum aDatatype);

	//! Converts from Em data type enum to internal data type
	/*!
	EmFile::GetDataType dows not take into account if the data type is unsigned or signed as the
	Em file format cannot distinguish them.
	*/
	DataType_enum GetDataType();

	//! Returns the size of the data block. If the header is not yet read, it will return 0.
	size_t GetDataSize();

	//! Returns a reference to the inner Em file header.
	EmHeader& GetFileHeader();

	//! Returns the inner data pointer.
	void* GetData();

	//! Returns the inner data pointer shifted to image plane idx.
	void* GetData(size_t idx);

	static void AddSlice(string aFileName, float* data, int width, int height);
};

void emwrite(string aFileName, float* data, int width, int height);
void emwrite(string aFileName, float* data, int width, int height, int depth);

#endif