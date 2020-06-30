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


#include "FileReader.h"

FileReader::FileReader(string aFileName)
	: File(aFileName, true)
{

}

FileReader::FileReader(string aFileName, bool aIsLittleEndian)
	: File(aFileName, aIsLittleEndian)
{

}
FileReader::FileReader(fstream* aStream)
	: File(aStream, true)
{

}

FileReader::FileReader(fstream* aStream, bool aIsLittleEndian)
	: File(aStream, aIsLittleEndian)
{

}

bool FileReader::OpenRead()
{
	mFile->open(mFileName.c_str(), ios_base::in | ios_base::binary);
	return mFile->is_open() && mFile->good();
}

void FileReader::CloseRead()
{
	mFile->close();
}

long64 FileReader::ReadI8LE()
{
	long64 temp;
	mFile->read((char*)&temp, 8);

	if (!mIsLittleEndian) Endian_swap(temp);

	return temp;
}

long64 FileReader::ReadI8BE()
{
	long64 temp;
	mFile->read((char*)&temp, 8);

	if (mIsLittleEndian) Endian_swap(temp);

	return temp;
}

int FileReader::ReadI4LE()
{
	int temp;
	mFile->read((char*)&temp, 4);

	if (!mIsLittleEndian) Endian_swap(temp);

	return temp;
}

int FileReader::ReadI4BE()
{
	int temp;
	mFile->read((char*)&temp, 4);

	if (mIsLittleEndian) Endian_swap(temp);

	return temp;
}

short FileReader::ReadI2LE()
{
	short temp;
	mFile->read((char*)&temp, 2);

	if (!mIsLittleEndian) Endian_swap(temp);

	return temp;
}

short FileReader::ReadI2BE()
{
	short temp;
	mFile->read((char*)&temp, 2);

	if (mIsLittleEndian) Endian_swap(temp);

	return temp;
}

char FileReader::ReadI1()
{
	char temp;
	mFile->read((char*)&temp, 1);

	return temp;
}

ulong64 FileReader::ReadUI8LE()
{
	ulong64 temp;
	mFile->read((char*)&temp, 8);

	if (!mIsLittleEndian) Endian_swap(temp);

	return temp;
}

ulong64 FileReader::ReadUI8BE()
{
	ulong64 temp;
	mFile->read((char*)&temp, 8);

	if (mIsLittleEndian) Endian_swap(temp);

	return temp;
}

uint FileReader::ReadUI4LE()
{
	uint temp;
	mFile->read((char*)&temp, 4);

	if (!mIsLittleEndian) Endian_swap(temp);

	return temp;
}

uint FileReader::ReadUI4BE()
{
	uint temp;
	mFile->read((char*)&temp, 4);

	if (mIsLittleEndian) Endian_swap(temp);

	return temp;
}

ushort FileReader::ReadUI2LE()
{
	ushort temp;
	mFile->read((char*)&temp, 2);

	if (!mIsLittleEndian) Endian_swap(temp);

	return temp;
}

ushort FileReader::ReadUI2BE()
{
	ushort temp;
	mFile->read((char*)&temp, 2);

	if (mIsLittleEndian) Endian_swap(temp);

	return temp;
}

uchar FileReader::ReadUI1()
{
	uchar temp;
	mFile->read((char*)&temp, 1);

	return temp;
}

double FileReader::ReadF8LE()
{
	double temp;
	mFile->read((char*)&temp, 8);

	if (!mIsLittleEndian) Endian_swap(temp);

	return temp;
}

double FileReader::ReadF8BE()
{
	double temp;
	mFile->read((char*)&temp, 8);

	if (mIsLittleEndian) Endian_swap(temp);

	return temp;
}

float FileReader::ReadF4LE()
{
	float temp;
	mFile->read((char*)&temp, 4);

	if (!mIsLittleEndian) Endian_swap(temp);

	return temp;
}

float FileReader::ReadF4BE()
{
	float temp;
	mFile->read((char*)&temp, 4);

	if (!mIsLittleEndian) Endian_swap(temp);

	return temp;
}

string FileReader::ReadStr(int aCount)
{	
	if (aCount == 0) return string("");
	char* nameTemp = new char[aCount + 1];
	nameTemp[aCount] = '\0';
	mFile->read(nameTemp, aCount);
	string ret(nameTemp);
	delete[] nameTemp;
	return ret;
}

string FileReader::ReadStrUTF(int aCount)
{
	return NULL;
}