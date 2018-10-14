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


#ifndef DM4FILESTACK_H
#define DM4FILESTACK_H

#include "../../Basics/Default.h"
#include "Dm4File.h"
#include "FileReader.h"

//!  Dm4FileStack represents a tilt series in gatan's dm4 format.
/*!
	Dm4FileStack reads all projections with the same name base into one volume in memory.
	The provided file name must end with file index "0000".
	\author Michael Kunz
	\date   September 2011
	\version 1.0
*/
class Dm4FileStack : public FileReader
{
private:
	bool fexists(std::string filename);
	std::string GetStringFromInt(int aInt);
	std::string GetFileNameFromIndex(int aIndex, std::string aFileName);
	int CountFilesInStack(std::string aFileName);
	int _fileCount;
	int _firstIndex;

	std::vector<Dm4File*> _dm4files;

public:
	Dm4FileStack(std::string aFileName);
	virtual ~Dm4FileStack();

	bool OpenAndRead();
	DataType_enum GetDataType();
	void ReadHeaderInfo();
	void WriteInfoToHeader();
	char* GetData();

};

#endif
