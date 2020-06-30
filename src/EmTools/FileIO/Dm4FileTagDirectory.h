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


#ifndef DM4FILETAGDIRECTORY_H
#define DM4FILETAGDIRECTORY_H

#include "../Basics/Default.h"
#include "FileReader.h"
#include "Dm4FileTag.h"
#include <sstream>

using namespace std;

#define TAG_ID 0x15
#define TAGDIR_ID 0x14
#define DIREOF 0x0
#define TAGSTRUCT 0x0f
#define TAGARRAY 0x14


//!  Dm4FileTagDirectory represents a gatan *.dm4 tag directory.
/*!
	\author Michael Kunz
	\date   September 2011
	\version 1.0
*/
class Dm4FileTagDirectory : public FileReader
{
public:
	Dm4FileTagDirectory(fstream* aStream, bool aIsLittleEndian);
	Dm4FileTagDirectory(fstream* aStream, bool aIsLittleEndian, bool nonRoot);
	Dm4FileTagDirectory(fstream* aStream, bool aIsLittleEndian, bool nonRoot, bool thumbnailOnly);
	Dm4FileTagDirectory(istream* aStream, bool aIsLittleEndian);
	Dm4FileTagDirectory(istream* aStream, bool aIsLittleEndian, bool nonRoot);
	virtual ~Dm4FileTagDirectory();

	bool sorted;
	bool closed;
	uint countTags;
	string Name;
	ushort LengthName;

	vector<Dm4FileTag*> Tags;
	vector<Dm4FileTagDirectory*> TagDirs;

	void Print(ostream& stream, uint id, string pre = "");
	Dm4FileTag* FindTag(string aName);
	Dm4FileTagDirectory* FindTagDir(string aName);

	bool OpenAndRead();
	DataType_enum GetDataType();
private:
	void readTag();
	void readTagHeaderOnly();
	void readTagThumbnail();
};

#endif
