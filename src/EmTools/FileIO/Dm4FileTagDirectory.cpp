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


#include "Dm4FileTagDirectory.h"

using namespace std;

bool foundRGBAType = false;
bool foundImageData = false;
bool foundImageSize = false;
bool foundImageSize2 = false;
bool foundRGBATypeDir = false;
bool foundImageDataDir = false;
bool foundImageSizeDir = false;

Dm4FileTagDirectory::Dm4FileTagDirectory(fstream* aStream, bool aIsLittleEndian)
	: FileReader(aStream, aIsLittleEndian), Tags(), TagDirs()
{
	sorted = !!ReadUI1();
	closed = !!ReadUI1();
	//int test = ReadUI8BE();
	countTags = (uint)ReadUI8BE();
	LengthName = 0;
	Name = "root";

	//cout << Name << endl;
	uchar next;// = readUI1();

	int NoEndlessLoops = 0;

	for (uint i = 0; i < countTags; i++)
	{
		next = ReadUI1();
		if (next == TAG_ID)
		{
			//cout << "TAG: ";
			readTag();
		}

		if (next == TAGDIR_ID)
		{
			//cout << "TAG DIR: ";
			Dm4FileTagDirectory* dir = new Dm4FileTagDirectory(aStream, aIsLittleEndian, true);
			TagDirs.push_back(dir);
			//next = readUI1();
		}
	}

	//while (next != DIREOF && next != 0xCC && NoEndlessLoops < 10000)
	//{
	//	if (next == TAG_ID)
	//	{
	//		//cout << "TAG: ";
	//		readTag();
	//		next = readUI1();
	//	}

	//	if (next == TAGDIR_ID)
	//	{
	//		//cout << "TAG DIR: ";
	//		Dm3FileTagDirectory* dir = new Dm3FileTagDirectory(aStream, aIsLittleEndian, true);
	//		TagDirs.push_back(dir);
	//		next = readUI1();
	//	}
	//	NoEndlessLoops++;
	//}
}

Dm4FileTagDirectory::Dm4FileTagDirectory(fstream* aStream, bool aIsLittleEndian, bool nonRoot)
	: FileReader(aStream, aIsLittleEndian), Tags(), TagDirs()
{
	LengthName = ReadUI2BE();
	Name = ReadStr(LengthName);
	sorted = !!ReadUI1();
	closed = !!ReadUI1();
	ulong64 test = ReadUI8BE();
	countTags = (uint)ReadUI8BE();
	//cout << Name << endl;

	uchar next;// = readUI1();
	int NoEndlessLoops = 0;

	for (uint i = 0; i < countTags; i++)
	{
		next = ReadUI1();
		if (next == TAG_ID)
		{
			//cout << "TAG: ";
			readTag();
		}

		if (next == TAGDIR_ID)
		{
			//cout << "TAG DIR: ";
			Dm4FileTagDirectory* dir = new Dm4FileTagDirectory(aStream, aIsLittleEndian, true);
			TagDirs.push_back(dir);
			//next = readUI1();
		}
	}
	//while (next != DIREOF && next != 0xCC && NoEndlessLoops < 10000)
	//{
	//	if (next == TAG_ID)
	//	{
	//		//cout << "TAG: ";
	//		readTag();
	//		next = readUI1();
	//	}

	//	if (next == TAGDIR_ID)
	//	{
	//		//cout << "TAG DIR: ";
	//		Dm3FileTagDirectory* dir = new Dm3FileTagDirectory(aStream, aIsLittleEndian, true);
	//		TagDirs.push_back(dir);
	//		next = readUI1();
	//	}
	//	NoEndlessLoops++;
	//}
}

Dm4FileTagDirectory::Dm4FileTagDirectory(fstream* aStream, bool aIsLittleEndian, bool nonRoot, bool headerOnly)
	: FileReader(aStream, aIsLittleEndian), Tags(), TagDirs()
{
    if (nonRoot)
    {
        LengthName = ReadUI2BE();
        Name = ReadStr(LengthName);
        sorted = !!ReadUI1();
        closed = !!ReadUI1();
        ulong64 test = ReadUI8BE();
        countTags = (uint)ReadUI8BE();
        //cout << Name << endl;

        uchar next;// = readUI1();
        int NoEndlessLoops = 0;

        if (Name == "Dimensions")
            foundImageSizeDir = true;
        if (Name == "ImageData")
            foundImageDataDir = true;
        if (Name == "ImageData")
            foundRGBATypeDir = true;
        for (uint i = 0; i < countTags; i++)
        {

            next = ReadUI1();
            if (next == TAG_ID)
            {
                //cout << "TAG: ";
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
                //cout << "TAG DIR: ";
                Dm4FileTagDirectory* dir = new Dm4FileTagDirectory(aStream, aIsLittleEndian, true, headerOnly);
                TagDirs.push_back(dir);
                //next = readUI1();
            }

            
        }
    }
    else
    {
        sorted = !!ReadUI1();
        closed = !!ReadUI1();
        //int test = ReadUI8BE();
        countTags = (uint)ReadUI8BE();
        LengthName = 0;
        Name = "root";

        //cout << Name << endl;
        uchar next;// = readUI1();

        int NoEndlessLoops = 0;
        bool done = false;
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
                //cout << "TAG DIR: ";

				Dm4FileTagDirectory* dir = new Dm4FileTagDirectory(aStream, aIsLittleEndian, true, headerOnly);
                TagDirs.push_back(dir);
                //next = readUI1();
            }
        }
    }
}



Dm4FileTagDirectory::Dm4FileTagDirectory(istream* aStream, bool aIsLittleEndian)
	: FileReader(aStream, aIsLittleEndian), Tags(), TagDirs()
{
	sorted = !!ReadUI1();
	closed = !!ReadUI1();
	//int test = ReadUI8BE();
	countTags = (uint)ReadUI8BE();
	LengthName = 0;
	Name = "root";

	//cout << Name << endl;
	uchar next;// = readUI1();

	int NoEndlessLoops = 0;

	for (uint i = 0; i < countTags; i++)
	{
		next = ReadUI1();
		if (next == TAG_ID)
		{
			//cout << "TAG: ";
			readTag();
		}

		if (next == TAGDIR_ID)
		{
			//cout << "TAG DIR: ";
			Dm4FileTagDirectory* dir = new Dm4FileTagDirectory(aStream, aIsLittleEndian, true);
			TagDirs.push_back(dir);
			//next = readUI1();
		}
	}
}

Dm4FileTagDirectory::Dm4FileTagDirectory(istream* aStream, bool aIsLittleEndian, bool nonRoot)
	: FileReader(aStream, aIsLittleEndian), Tags(), TagDirs()
{
	LengthName = ReadUI2BE();
	Name = ReadStr(LengthName);
	sorted = !!ReadUI1();
	closed = !!ReadUI1();
	ulong64 test = ReadUI8BE();
	countTags = (uint)ReadUI8BE();
	//cout << Name << endl;

	uchar next;// = readUI1();
	int NoEndlessLoops = 0;

	for (uint i = 0; i < countTags; i++)
	{
		next = ReadUI1();
		if (next == TAG_ID)
		{
			//cout << "TAG: ";
			readTag();
		}

		if (next == TAGDIR_ID)
		{
			//cout << "TAG DIR: ";
			Dm4FileTagDirectory* dir = new Dm4FileTagDirectory(aStream, aIsLittleEndian, true);
			TagDirs.push_back(dir);
			//next = readUI1();
		}
	}
}

Dm4FileTagDirectory::~Dm4FileTagDirectory()
{
	//The TagDirectory class should not free the mFile pointer.
	//This is done in the Dm3File class.
	mFile = NULL;
	for (uint i = 0; i < Tags.size(); i++)
		delete Tags[i];

	for (uint i = 0; i < TagDirs.size(); i++)
		delete TagDirs[i];
}

void Dm4FileTagDirectory::readTag()
{
	ushort lengthTagName = ReadUI2BE();
	Dm4FileTag* tag = new Dm4FileTag();
	tag->Name = ReadStr(lengthTagName);
	if(tag->Name == "Data")
		tag->Name = "Data";
	ulong64 xyz = ReadUI8BE();
	string percentsSigns = ReadStr(4);
	tag->SizeInfoArray = ReadUI8BE();
	tag->InfoArray = new ulong64[tag->SizeInfoArray];
	for (ulong64 i = 0; i < tag->SizeInfoArray; i++)
	{
		tag->InfoArray[i] = ReadUI8BE();
	}

	//cout << tag->Name << endl;
	ulong64 SizeData = 0;

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
			ulong64 entriesInGroup = tag->InfoArray[3];

			for (ulong64 i = 0; i < entriesInGroup; i++)
				SizeData += tag->GetSize(tag->InfoArray[i * 2 + 5]);

			SizeData *= tag->InfoArray[tag->SizeInfoArray - 1]; //Number of elements in array

			tag->Data = new char[SizeData];
			Read(tag->Data, SizeData);
		}
	}
	tag->SizeData = SizeData;
	Tags.push_back(tag);
}

void Dm4FileTagDirectory::readTagHeaderOnly()
{
	ushort lengthTagName = ReadUI2BE();
	Dm4FileTag* tag = new Dm4FileTag();
	tag->Name = ReadStr(lengthTagName);
	if (tag->Name == "Data")
		tag->Name = "Data";
	ulong64 xyz = ReadUI8BE();
	string percentsSigns = ReadStr(4);
	tag->SizeInfoArray = ReadUI8BE();
	tag->InfoArray = new ulong64[tag->SizeInfoArray];
	for (ulong64 i = 0; i < tag->SizeInfoArray; i++)
	{
		tag->InfoArray[i] = ReadUI8BE();
	}

	//cout << tag->Name << endl;
	ulong64 SizeData = 0;

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
			ulong64 entriesInGroup = tag->InfoArray[3];

			for (ulong64 i = 0; i < entriesInGroup; i++)
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

void Dm4FileTagDirectory::readTagThumbnail()
{
	ushort lengthTagName = ReadUI2BE();
	Dm4FileTag* tag = new Dm4FileTag();
	tag->Name = ReadStr(lengthTagName);



	ulong64 xyz = ReadUI8BE();
	string percentsSigns = ReadStr(4);
	tag->SizeInfoArray = ReadUI8BE();
	tag->InfoArray = new ulong64[tag->SizeInfoArray];
	for (ulong64 i = 0; i < tag->SizeInfoArray; i++)
	{
		tag->InfoArray[i] = ReadUI8BE();
	}

	//cout << tag->Name << endl;
	ulong64 SizeData = 0;

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
			ulong64 entriesInGroup = tag->InfoArray[3];

			for (ulong64 i = 0; i < entriesInGroup; i++)
				SizeData += tag->GetSize(tag->InfoArray[i * 2 + 5]);

			SizeData *= tag->InfoArray[tag->SizeInfoArray - 1]; //Number of elements in array

			tag->Data = new char[SizeData];
			Read(tag->Data, SizeData);
		}
	}
	tag->SizeData = SizeData;
	Tags.push_back(tag);

	//if (tag->Name == "DataType")
	{
	    if (foundRGBATypeDir)
	    if (tag->GetSingleValueInt(tag->InfoArray[0]) == DTT_RGBA_4UI1)
            foundRGBAType = true;

        if (foundImageSize)
	    if (tag->GetSingleValueInt(tag->InfoArray[0]) == 384)
            foundImageSize2 = true;
        if (foundImageSizeDir)
	    if (tag->GetSingleValueInt(tag->InfoArray[0]) == 384)
            foundImageSize = true;

        if (foundImageDataDir)
	    if (tag->Name == "Data")
            foundImageData = true;
	}
//    return false;
}

void Dm4FileTagDirectory::Print(ostream& stream, uint id, string pre)
{

	size_t size = Tags.size();
	for (size_t i = 0; i < size; i++)
	{
		stream << pre << ".";

		if (Name.size() == 0)
		{
			char tmp[100];
			snprintf(tmp,100, "<%d>", id);
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

Dm4FileTag* Dm4FileTagDirectory::FindTag(string aName)
{
	Dm4FileTag* foundTag = NULL;
	for (uint i = 0; i < Tags.size(); i++)
	{
		if (Tags[i]->Name == aName) foundTag = Tags[i];
	}
	return foundTag;
}

Dm4FileTagDirectory* Dm4FileTagDirectory::FindTagDir(string aName)
{
	Dm4FileTagDirectory* foundTagDir = NULL;
	for (uint i = 0; i < TagDirs.size(); i++)
	{
		if (TagDirs[i]->Name == aName) foundTagDir = TagDirs[i];
	}
	return foundTagDir;
}

bool Dm4FileTagDirectory::OpenAndRead()
{
	return false;
}

DataType_enum Dm4FileTagDirectory::GetDataType()
{
	return DT_UNKNOWN;
}
