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


#ifndef DM3FILETAGDIRECTORY_H
#define DM3FILETAGDIRECTORY_H

#include "../Basics/Default.h"
#include "FileReader.h"
#include "Dm3FileTag.h"
#include <sstream>

using namespace std;

#define TAG_ID 0x15
#define TAGDIR_ID 0x14
#define DIREOF 0x0
#define TAGSTRUCT 0x0f
#define TAGARRAY 0x14


//!  Dm3FileTagDirectory represents a gatan *.dm3 tag directory. 
/*!
	\author Michael Kunz
	\date   September 2011
	\version 1.0
*/
class Dm3FileTagDirectory : public FileReader
{
public: 
	Dm3FileTagDirectory(fstream* aStream, bool aIsLittleEndian);
	Dm3FileTagDirectory(fstream* aStream, bool aIsLittleEndian, bool nonRoot);
	Dm3FileTagDirectory(istream* aStream, bool aIsLittleEndian);
	Dm3FileTagDirectory(istream* aStream, bool aIsLittleEndian, bool nonRoot);
	Dm3FileTagDirectory(fstream* aStream, bool aIsLittleEndian, bool nonRoot, bool headerOnly);
	virtual ~Dm3FileTagDirectory();

	bool sorted;
	bool closed;
	uint countTags;
	string Name;
	ushort LengthName;

	vector<Dm3FileTag*> Tags;
	vector<Dm3FileTagDirectory*> TagDirs;
	
	void Print(ostream& stream, uint id, string pre = "");
	Dm3FileTag* FindTag(string aName);
	Dm3FileTagDirectory* FindTagDir(string aName);

	bool OpenAndRead();
	DataType_enum GetDataType();
private:
	void readTag();
	void readTagHeaderOnly();
};

#endif