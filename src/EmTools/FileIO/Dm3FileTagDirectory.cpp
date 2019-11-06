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


#include "Dm3FileTagDirectory.h"

using namespace std;

Dm3FileTagDirectory::Dm3FileTagDirectory(fstream* aStream, bool aIsLittleEndian)
	: FileReader(aStream, aIsLittleEndian), Tags(), TagDirs()
{
	sorted = !!ReadUI1();
	closed = !!ReadUI1();
	countTags = ReadUI4BE();
	LengthName = 0;
	Name = "root";
	
	uchar next;// = readUI1();

	for (uint i = 0; i < countTags; i++)
	{
		next = ReadUI1();
		if (next == TAG_ID) 
		{
			readTag();
		}

		if (next == TAGDIR_ID) 
		{
			Dm3FileTagDirectory* dir = new Dm3FileTagDirectory(aStream, aIsLittleEndian, true);
			TagDirs.push_back(dir);
		}
	}
}

Dm3FileTagDirectory::Dm3FileTagDirectory(fstream* aStream, bool aIsLittleEndian, bool nonRoot)
	: FileReader(aStream, aIsLittleEndian), Tags(), TagDirs()
{
	if (nonRoot)
	{
		LengthName = ReadUI2BE();
		Name = ReadStr(LengthName);
		sorted = !!ReadUI1();
		closed = !!ReadUI1();
		countTags = ReadUI4BE();

		uchar next;// = readUI1();

		for (uint i = 0; i < countTags; i++)
		{
			next = ReadUI1();
			if (next == TAG_ID)
			{
				readTag();
			}

			if (next == TAGDIR_ID)
			{
				Dm3FileTagDirectory* dir = new Dm3FileTagDirectory(aStream, aIsLittleEndian, true);
				TagDirs.push_back(dir);
			}
		}
	}
	else
	{
		sorted = !!ReadUI1();
		closed = !!ReadUI1();
		countTags = ReadUI4BE();
		LengthName = 0;
		Name = "root";

		uchar next;// = readUI1();

		for (uint i = 0; i < countTags; i++)
		{
			next = ReadUI1();
			if (next == TAG_ID)
			{
				readTag();
			}

			if (next == TAGDIR_ID)
			{
				Dm3FileTagDirectory* dir = new Dm3FileTagDirectory(aStream, aIsLittleEndian, true);
				TagDirs.push_back(dir);
			}
		}
	}
}

Dm3FileTagDirectory::Dm3FileTagDirectory(istream* aStream, bool aIsLittleEndian)
	: FileReader(aStream, aIsLittleEndian), Tags(), TagDirs()
{
	sorted = !!ReadUI1();
	closed = !!ReadUI1();
	countTags = ReadUI4BE();
	LengthName = 0;
	Name = "root";

	uchar next;// = readUI1();

	for (uint i = 0; i < countTags; i++)
	{
		next = ReadUI1();
		if (next == TAG_ID)
		{
			readTag();
		}

		if (next == TAGDIR_ID)
		{
			Dm3FileTagDirectory* dir = new Dm3FileTagDirectory(aStream, aIsLittleEndian, true);
			TagDirs.push_back(dir);
		}
	}
}

Dm3FileTagDirectory::Dm3FileTagDirectory(istream* aStream, bool aIsLittleEndian, bool nonRoot)
	: FileReader(aStream, aIsLittleEndian), Tags(), TagDirs()
{
	if (nonRoot)
	{
		LengthName = ReadUI2BE();
		Name = ReadStr(LengthName);
		sorted = !!ReadUI1();
		closed = !!ReadUI1();
		countTags = ReadUI4BE();

		uchar next;// = readUI1();

		for (uint i = 0; i < countTags; i++)
		{
			next = ReadUI1();
			if (next == TAG_ID)
			{
				readTag();
			}

			if (next == TAGDIR_ID)
			{
				Dm3FileTagDirectory* dir = new Dm3FileTagDirectory(aStream, aIsLittleEndian, true);
				TagDirs.push_back(dir);
			}
		}
	}
	else
	{
		sorted = !!ReadUI1();
		closed = !!ReadUI1();
		countTags = ReadUI4BE();
		LengthName = 0;
		Name = "root";

		uchar next;// = readUI1();

		for (uint i = 0; i < countTags; i++)
		{
			next = ReadUI1();
			if (next == TAG_ID)
			{
				readTag();
			}

			if (next == TAGDIR_ID)
			{
				Dm3FileTagDirectory* dir = new Dm3FileTagDirectory(aStream, aIsLittleEndian, true);
				TagDirs.push_back(dir);
			}
		}
	}
}

Dm3FileTagDirectory::Dm3FileTagDirectory(fstream* aStream, bool aIsLittleEndian, bool nonRoot, bool headerOnly)
	: FileReader(aStream, aIsLittleEndian), Tags(), TagDirs()
{
	if (nonRoot)
	{
		LengthName = ReadUI2BE();
		Name = ReadStr(LengthName);
		sorted = !!ReadUI1();
		closed = !!ReadUI1();
		countTags = ReadUI4BE();

		uchar next;// = readUI1();

		for (uint i = 0; i < countTags; i++)
		{
			next = ReadUI1();
			if (next == TAG_ID)
			{
				if (headerOnly)
				{
					readTagHeaderOnly();
				}
				else
				{
					readTag();
				}
			}

			if (next == TAGDIR_ID)
			{
				Dm3FileTagDirectory* dir = new Dm3FileTagDirectory(aStream, aIsLittleEndian, true);
				TagDirs.push_back(dir);
			}
		}
	}
	else
	{
		sorted = !!ReadUI1();
		closed = !!ReadUI1();
		countTags = ReadUI4BE();
		LengthName = 0;
		Name = "root";

		uchar next;// = readUI1();

		for (uint i = 0; i < countTags; i++)
		{
			next = ReadUI1();
			if (next == TAG_ID)
			{
				if (headerOnly)
				{
					readTagHeaderOnly();
				}
				else
				{
					readTag();
				}
			}

			if (next == TAGDIR_ID)
			{
				Dm3FileTagDirectory* dir = new Dm3FileTagDirectory(aStream, aIsLittleEndian, true);
				TagDirs.push_back(dir);
			}
		}
	}
}

Dm3FileTagDirectory::~Dm3FileTagDirectory()
{	
	//The TagDirectory class should not free the mFile pointer.
	//This is done in the Dm3File class.
	mFile = NULL;
	for (uint i = 0; i < Tags.size(); i++)
		delete Tags[i];

	for (uint i = 0; i < TagDirs.size(); i++)
		delete TagDirs[i];
}

void Dm3FileTagDirectory::readTag()
{
	ushort lengthTagName = ReadUI2BE();
	Dm3FileTag* tag = new Dm3FileTag();
	tag->Name = ReadStr(lengthTagName);
	if(tag->Name == "Data")
		tag->Name = "Data";
	string percentsSigns = ReadStr(4);
	tag->SizeInfoArray = ReadUI4BE();
	tag->InfoArray = new uint[tag->SizeInfoArray];
	for (uint i = 0; i < tag->SizeInfoArray; i++)
	{
		tag->InfoArray[i] = ReadUI4BE();
	}
			
	//cout << tag->Name << endl;
	uint SizeData = 0;

	//Detect Tag type:
	if (tag->SizeInfoArray == 1) //Single Entry tag
	{
		tag->tagType = SingleEntryTag;
		SizeData = tag->GetSize(tag->InfoArray[0]);
		tag->Data = new char[SizeData];
		Read(tag->Data, SizeData);
	}

	if (tag->SizeInfoArray > 2)
	{
		if (tag->InfoArray[0] == TAGSTRUCT) //Group of data (struct) tag
		{
			tag->tagType = StructTag;
			for (uint i = 0; i < tag->InfoArray[2]; i++)
				SizeData += tag->GetSize(tag->InfoArray[i * 2 + 4]);

			tag->Data = new char[SizeData];
			Read(tag->Data, SizeData);
		}

		if (tag->InfoArray[0] == TAGARRAY && tag->InfoArray[1] != TAGSTRUCT) //Array tag
		{
			tag->tagType = ArrayTag;
			SizeData = tag->GetSize(tag->InfoArray[1]); //array element size
			SizeData *= tag->InfoArray[2]; //Number of elements in array

			tag->Data = new char[SizeData];
			Read(tag->Data, SizeData);
		}

		if (tag->InfoArray[0] == TAGARRAY && tag->InfoArray[1] == TAGSTRUCT) //Array of group tag
		{
			tag->tagType = ArrayStructTag;
			uint entriesInGroup = tag->InfoArray[3];
					
			for (uint i = 0; i < entriesInGroup; i++)
				SizeData += tag->GetSize(tag->InfoArray[i * 2 + 5]);

			SizeData *= tag->InfoArray[tag->SizeInfoArray - 1]; //Number of elements in array

			tag->Data = new char[SizeData];
			Read(tag->Data, SizeData);
		}
	}
	tag->SizeData = SizeData;
	Tags.push_back(tag);
}

void Dm3FileTagDirectory::readTagHeaderOnly()
{
	ushort lengthTagName = ReadUI2BE();
	Dm3FileTag* tag = new Dm3FileTag();
	tag->Name = ReadStr(lengthTagName);
	if(tag->Name == "Data")
		tag->Name = "Data";
	string percentsSigns = ReadStr(4);
	tag->SizeInfoArray = ReadUI4BE();
	tag->InfoArray = new uint[tag->SizeInfoArray];
	for (uint i = 0; i < tag->SizeInfoArray; i++)
	{
		tag->InfoArray[i] = ReadUI4BE();
	}
			
	//cout << tag->Name << endl;
	uint SizeData = 0;

	//Detect Tag type:
	if (tag->SizeInfoArray == 1) //Single Entry tag
	{
		tag->tagType = SingleEntryTag;
		SizeData = tag->GetSize(tag->InfoArray[0]);
		//if data chunk is larger than a few bytes, we skip it in header only mode to save reading time
		if (SizeData > 128)
		{
			tag->Data = NULL;
			mFile->seekg(SizeData, ios_base::cur);
		}
		else
		{
			tag->Data = new char[SizeData];
			Read(tag->Data, SizeData);
		}
	}

	if (tag->SizeInfoArray > 2)
	{
		if (tag->InfoArray[0] == TAGSTRUCT) //Group of data (struct) tag
		{
			tag->tagType = StructTag;
			for (uint i = 0; i < tag->InfoArray[2]; i++)
				SizeData += tag->GetSize(tag->InfoArray[i * 2 + 4]);

			//if data chunk is larger than a few bytes, we skip it in header only mode to save reading time
			if (SizeData > 128)
			{
				tag->Data = NULL;
				mFile->seekg(SizeData, ios_base::cur);
			}
			else
			{
				tag->Data = new char[SizeData];
				Read(tag->Data, SizeData);
			}
		}

		if (tag->InfoArray[0] == TAGARRAY && tag->InfoArray[1] != TAGSTRUCT) //Array tag
		{
			tag->tagType = ArrayTag;
			SizeData = tag->GetSize(tag->InfoArray[1]); //array element size
			SizeData *= tag->InfoArray[2]; //Number of elements in array

			//if data chunk is larger than a few bytes, we skip it in header only mode to save reading time
			if (SizeData > 128)
			{
				tag->Data = NULL;
				mFile->seekg(SizeData, ios_base::cur);
			}
			else
			{
				tag->Data = new char[SizeData];
				Read(tag->Data, SizeData);
			}
		}

		if (tag->InfoArray[0] == TAGARRAY && tag->InfoArray[1] == TAGSTRUCT) //Array of group tag
		{
			tag->tagType = ArrayStructTag;
			uint entriesInGroup = tag->InfoArray[3];
					
			for (uint i = 0; i < entriesInGroup; i++)
				SizeData += tag->GetSize(tag->InfoArray[i * 2 + 5]);

			SizeData *= tag->InfoArray[tag->SizeInfoArray - 1]; //Number of elements in array

			//if data chunk is larger than a few bytes, we skip it in header only mode to save reading time
			if (SizeData > 128)
			{
				tag->Data = NULL;
				mFile->seekg(SizeData, ios_base::cur);
			}
			else
			{
				tag->Data = new char[SizeData];
				Read(tag->Data, SizeData);
			}
		}
	}
	tag->SizeData = SizeData;
	Tags.push_back(tag);
}

void Dm3FileTagDirectory::Print(ostream& stream, uint id, string pre)
{
	
	size_t size = Tags.size();
	for (size_t i = 0; i < size; i++)
	{
		stream << pre << ".";

		if (Name.size() == 0)
		{
			char tmp[100];
			snprintf(tmp, 100, "<%d>", id);
			//sprintf_s(tmp, 100, "<%d>", id);
			stream << string(tmp);
		}
		else
			stream << Name;

		if (Tags[i]->Name.size() == 0)
			stream << "." << "Tag[" << i << "] = ";
		else
			stream << "." << Tags[i]->Name << " = ";
		
		stream << *Tags[i] << ": " << Tags[i]->SizeData / 1024;//;
		stream << endl;
	}

	size = TagDirs.size();

	for (int i = 0; i < size; i++)
	{
		if (Name.size() == 0)
		{
			char tmp[100];
			snprintf(tmp, 100, "%d", id);
			TagDirs[i]->Print(stream, i, pre + "." + string(tmp));
		}
		else
			TagDirs[i]->Print(stream, i, pre + "." + Name);
		
		
	}

}

Dm3FileTag* Dm3FileTagDirectory::FindTag(string aName)
{
	Dm3FileTag* foundTag = NULL;
	for (uint i = 0; i < Tags.size(); i++)
	{
		if (Tags[i]->Name == aName) foundTag = Tags[i];
	}
	return foundTag;
}

Dm3FileTagDirectory* Dm3FileTagDirectory::FindTagDir(string aName)
{
	Dm3FileTagDirectory* foundTagDir = NULL;
	for (uint i = 0; i < TagDirs.size(); i++)
	{
		if (TagDirs[i]->Name == aName) foundTagDir = TagDirs[i];
	}
	return foundTagDir;
}

bool Dm3FileTagDirectory::OpenAndRead()
{	
	return false;	
}

DataType_enum Dm3FileTagDirectory::GetDataType()
{
	return DT_UNKNOWN;
}