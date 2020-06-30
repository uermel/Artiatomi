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


#include "ImageBase.h"
#include "../MKLog/MKLog.h"

using namespace std;

ImageBase::ImageBase(){}
ImageBase::~ImageBase(){}

FileType_enum ImageBase::GuessFileTypeFromEnding(string aFilename)
{
	string::size_type dotPos = aFilename.find_last_of('.');
	if (dotPos == string::npos)
	{
		throw FileIOException(aFilename, "Cannot determine filetype from file ending.");
	}

	string ending = aFilename.substr(dotPos);

	MKLOG("Found file ending: " + ending);

	if (ending == ".mrc")
		return FileType_enum::FT_MRC;
	if (ending == ".mrcs")
		return FileType_enum::FT_MRC;
	if (ending == ".st")
		return FileType_enum::FT_MRC;
	if (ending == ".rec")
		return FileType_enum::FT_MRC;
	if (ending == ".dm4")
		return FileType_enum::FT_DM4;
	if (ending == ".dm3")
		return FileType_enum::FT_DM3;
	if (ending == ".em")
		return FileType_enum::FT_EM;
	if (ending == ".ser")
		return FileType_enum::FT_SER;
	if (ending == ".tif")
		return FileType_enum::FT_TIFF;
	if (ending == ".tiff")
		return FileType_enum::FT_TIFF;
	if (ending == ".lsm")
		return FileType_enum::FT_TIFF;

	throw FileIOException(aFilename, "Cannot determine filetype from file ending.");
}