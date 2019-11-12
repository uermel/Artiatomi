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


#ifndef PROJECTIONSOURCE_H
#define PROJECTIONSOURCE_H

#include "IODefault.h"
//#include <TiltSeries.h>
//#include "FileReader.h"
//#include "FileWriter.h"
//#include "Image.h"

using namespace std;

//!  MRCFile represents a *.mrc file in memory and maps contained information to the default internal Image format.
/*!
	MRCFile gives access to header infos, volume data and single projections.
	\author Michael Kunz
	\date   October 2012
	\version 1.0
*/
class ProjectionSource //: public FileReader, public FileWriter, public Image
{
private:

public:
	//ProjectionSource(string aFileName, const Image& aImage) : FileReader(aFileName), FileWriter(aFileName), Image(aImage) {};
	//ProjectionSource(string aFileName)
	//{
	//}// : FileReader(aFileName), FileWriter(aFileName) {};

	//! Opens the file File#mFileName and reads the entire content.
	/*!
		\throw FileIOException
	*/
	//virtual bool OpenAndRead() = 0;

	//! Converts from MRC data type enum to internal data type
	/*!
		MRCFile::GetDataType dows not take into account if the data type is unsigned or signed as the
		MRC file format cannot distinguish them.
	*/
	virtual DataType_enum GetDataType() = 0;

	////! Returns the size of the data block. If the header is not yet read, it will return 0.
	//virtual size_t GetDataSize() = 0;

	////! Returns the inner data pointer.
	//virtual char* GetData() = 0;

	//! Reads the projection with index \p aIndex from file and returns it's pointer.
	virtual char* GetProjection(uint aIndex) = 0;

	////! Reads the projection converted to float with index \p aIndex from file and returns it's pointer.
	//virtual float* GetProjectionFloat(uint aIndex) = 0;

	////! Reads the projection converted to float with index \p aIndex from file and returns it's pointer. Values are inverted.
	//virtual float* GetProjectionInvertFloat(uint aIndex) = 0;

	virtual float GetPixelSize() = 0;
	virtual int GetWidth() = 0;
	virtual int GetHeight() = 0;
	virtual int GetProjectionCount() = 0;
};

#endif
