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

#include "../Basics/Default.h"
#include "mrcHeader.h"
#include "FileReader.h"
#include "FileWriter.h"

using namespace std;

//!  MRCFile represents a *.mrc file in memory and maps contained information to the default internal Image format. 
/*!
	MRCFile gives access to header infos, volume data and single projections.
	\author Michael Kunz
	\date   September 2011
	\version 1.0
*/
class MRCFile : public FileReader, public FileWriter
{
private:
	MrcHeader _fileHeader;
	MrcExtendedHeader* _extHeaders;
	uint _GetDataTypeSize(MrcMode_Enum aDataType);
	uint _dataStartPosition;
	void* _data;

	void InverseEndianessHeader();
	void InverseEndianessExtHeaders();
	void InverseEndianessData();

public:
	//! Creates a new MRCFile instance. The file name is only set internally; the file itself keeps untouched.
	MRCFile(string aFileName);

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
	static bool InitHeaders(string aFileName, int aDimX, int aDimY, float aPixelSize, DataType_enum aDatatype, bool aSetExtended);

	static bool InitHeaders(string aFileName, int aDimX, int aDimY, int aDimZ, float aPixelSize, DataType_enum aDatatype, bool aSetExtended);

	static bool InitHeaders(string aFileName, MrcHeader& header);

	static bool AddPlaneToMRCFile(string aFileName, DataType_enum aDatatype, void* aData, float tiltAngle);

	static bool WriteRawData(string aFileName, void* aData, size_t aSize);

	static bool SetHeaderData(MrcHeader& header, int aDimX, int aDimY, int aDimZ, float aPixelSize, DataType_enum aDatatype, bool aSetExtended);

	//! Converts from MRC data type enum to internal data type
	/*!
		MRCFile::GetDataType dows not take into account if the data type is unsigned or signed as the
		MRC file format cannot distinguish them.
	*/ 
	DataType_enum GetDataType();
	
	//! Returns the size of the data block. If the header is not yet read, it will return 0.
	size_t GetDataSize();

	//! Returns a reference to the inner mrc file header.
	MrcHeader& GetFileHeader();
	
	//! Returns a pointer to the extended header section.
	MrcExtendedHeader* GetFileExtHeaders();

	//! Returns the inner data pointer.
	void* GetData();

	//! Returns the inner data pointer shifted to image plane idx.
	void* GetData(size_t idx);

	//! Reads the projection with index \p aIndex from file and returns it's pointer.
	void* GetProjection(uint aIndex);

	//! Returns the pixel size in nm as given in the file header as total width / number of pixels / 10.
	float GetPixelsize();
};

#endif