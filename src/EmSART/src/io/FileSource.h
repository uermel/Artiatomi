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


#ifndef FILESOURCE_H
#define FILESOURCE_H

#include "ProjectionSource.h"
#include <TiltSeries.h>

using namespace std;

//!  MRCFile represents a *.mrc file in memory and maps contained information to the default internal Image format.
/*!
	MRCFile gives access to header infos, volume data and single projections.
	\author Michael Kunz
	\date   October 2012
	\version 2.0
*/
class FileSource : public ProjectionSource
{
private:
	float** _projectionCache;
	TiltSeries* _ts;
	string _filename;

	static void FileLoadStatusUpdate(FileReader::FileReaderStatus status);

public:
	//! Creates a new MRCFile instance. The file name is only set internally; the file itself keeps untouched.
	FileSource(string aFilename);
	~FileSource();

	//! Opens the file File#mFileName and reads the entire content.
	/*!
		\throw FileIOException
	*/
	//bool OpenAndRead();

	//! Converts from MRC data type enum to internal data type
	/*!
		MRCFile::GetDataType dows not take into account if the data type is unsigned or signed as the
		MRC file format cannot distinguish them.
	*/
	DataType_enum GetDataType();

	//void SetDataType(FileDataType_enum aType);

	////! Returns the size of the data block. If the header is not yet read, it will return 0.
	//size_t GetDataSize();

	////! Returns the inner data pointer.
	//char* GetData();

	//! Reads the projection with index \p aIndex from file and returns it's pointer.
	char* GetProjection(uint aIndex);

	////! Reads the projection converted to float with index \p aIndex from file and returns it's pointer.
	//float* GetProjectionFloat(uint aIndex);

	////! Reads the projection converted to float with index \p aIndex from file and returns it's pointer. Values are inverted.
	//float* GetProjectionInvertFloat(uint aIndex);

	/*void ReadHeaderInfo();
	void WriteInfoToHeader();
	bool OpenAndWrite();*/

	float GetPixelSize();
	int GetWidth();
	int GetHeight();
	int GetProjectionCount();
};

#endif
