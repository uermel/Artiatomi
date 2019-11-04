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
#include "../MKLog/MKLog.h"

FileReader::FileReader(string aFileName)
	: File(aFileName, true), readStatusCallback(NULL), isFileStream(true), mIStream(NULL)
{

}

FileReader::FileReader(string aFileName, bool aIsLittleEndian)
	: File(aFileName, aIsLittleEndian), readStatusCallback(NULL), isFileStream(true), mIStream(NULL)
{

}
FileReader::FileReader(fstream* aStream)
	: File(aStream, true), readStatusCallback(NULL), isFileStream(true), mIStream(NULL)
{

}

FileReader::FileReader(fstream* aStream, bool aIsLittleEndian)
	: File(aStream, aIsLittleEndian), readStatusCallback(NULL), isFileStream(true), mIStream(NULL)
{

}

FileReader::FileReader(istream* aStream, bool aIsLittleEndian)
	: File(NULL, aIsLittleEndian), readStatusCallback(NULL), isFileStream(false), mIStream(aStream)
{
	cout << "test";
}

bool FileReader::OpenRead()
{
	if (!isFileStream) return true;
	mFile->open(mFileName.c_str(), ios_base::in | ios_base::binary);
	bool status = mFile->is_open() && mFile->good();
	MKLOG("FileReader opened file " + mFileName + ". Status: " + (status ? "GOOD" : "BAD"));
	return status;
}

void FileReader::CloseRead()
{
	if (!isFileStream) return;
	MKLOG("FileReader closed file " + mFileName + ".");
	mFile->close();
}

long64 FileReader::ReadI8LE()
{
	long64 temp;
	if (isFileStream)
		mFile->read((char*)&temp, 8);
	else
		mIStream->read((char*)&temp, 8);

	if (!mIsLittleEndian) Endian_swap(temp);

	return temp;
}

long64 FileReader::ReadI8BE()
{
	long64 temp;
	if (isFileStream)
		mFile->read((char*)&temp, 8);
	else
		mIStream->read((char*)&temp, 8);

	if (mIsLittleEndian) Endian_swap(temp);

	return temp;
}

int FileReader::ReadI4LE()
{
	int temp;
	if (isFileStream)
		mFile->read((char*)&temp, 4);
	else
		mIStream->read((char*)&temp, 4);

	if (!mIsLittleEndian) Endian_swap(temp);

	return temp;
}

int FileReader::ReadI4BE()
{
	int temp;
	if (isFileStream)
		mFile->read((char*)&temp, 4);
	else
		mIStream->read((char*)&temp, 4);

	if (mIsLittleEndian) Endian_swap(temp);

	return temp;
}

short FileReader::ReadI2LE()
{
	short temp;
	if (isFileStream)
		mFile->read((char*)&temp, 2);
	else
		mIStream->read((char*)&temp, 2);

	if (!mIsLittleEndian) Endian_swap(temp);

	return temp;
}

short FileReader::ReadI2BE()
{
	short temp;
	if (isFileStream)
		mFile->read((char*)&temp, 2);
	else
		mIStream->read((char*)&temp, 2);

	if (mIsLittleEndian) Endian_swap(temp);

	return temp;
}

char FileReader::ReadI1()
{
	char temp;
	if (isFileStream)
		mFile->read((char*)&temp, 1);
	else
		mIStream->read((char*)&temp, 1);

	return temp;
}

ulong64 FileReader::ReadUI8LE()
{
	ulong64 temp;
	if (isFileStream)
		mFile->read((char*)&temp, 8);
	else
		mIStream->read((char*)&temp, 8);

	if (!mIsLittleEndian) Endian_swap(temp);

	return temp;
}

ulong64 FileReader::ReadUI8BE()
{
	ulong64 temp;
	if (isFileStream)
		mFile->read((char*)&temp, 8);
	else
		mIStream->read((char*)&temp, 8);

	if (mIsLittleEndian) Endian_swap(temp);

	return temp;
}

uint FileReader::ReadUI4LE()
{
	uint temp;
	if (isFileStream)
		mFile->read((char*)&temp, 4);
	else
		mIStream->read((char*)&temp, 4);

	if (!mIsLittleEndian) Endian_swap(temp);

	return temp;
}

uint FileReader::ReadUI4BE()
{
	uint temp;
	if (isFileStream)
		mFile->read((char*)&temp, 4);
	else
		mIStream->read((char*)&temp, 4);

	if (mIsLittleEndian) Endian_swap(temp);

	return temp;
}

ushort FileReader::ReadUI2LE()
{
	ushort temp;
	if (isFileStream)
		mFile->read((char*)&temp, 2);
	else
		mIStream->read((char*)&temp, 2);

	if (!mIsLittleEndian) Endian_swap(temp);

	return temp;
}

ushort FileReader::ReadUI2BE()
{
	ushort temp;
	if (isFileStream)
		mFile->read((char*)&temp, 2);
	else
		mIStream->read((char*)&temp, 2);

	if (mIsLittleEndian) Endian_swap(temp);

	return temp;
}

uchar FileReader::ReadUI1()
{
	uchar temp;
	if (isFileStream)
		mFile->read((char*)&temp, 1);
	else
		mIStream->read((char*)&temp, 1);

	return temp;
}

double FileReader::ReadF8LE()
{
	double temp;
	if (isFileStream)
		mFile->read((char*)&temp, 8);
	else
		mIStream->read((char*)&temp, 8);

	if (!mIsLittleEndian) Endian_swap(temp);

	return temp;
}

double FileReader::ReadF8BE()
{
	double temp;
	if (isFileStream)
		mFile->read((char*)&temp, 8);
	else
		mIStream->read((char*)&temp, 8);

	if (mIsLittleEndian) Endian_swap(temp);

	return temp;
}

float FileReader::ReadF4LE()
{
	float temp;
	if (isFileStream)
		mFile->read((char*)&temp, 4);
	else
		mIStream->read((char*)&temp, 4);

	if (!mIsLittleEndian) Endian_swap(temp);

	return temp;
}

float FileReader::ReadF4BE()
{
	float temp;
	if (isFileStream)
		mFile->read((char*)&temp, 4);
	else
		mIStream->read((char*)&temp, 4);

	if (!mIsLittleEndian) Endian_swap(temp);

	return temp;
}

string FileReader::ReadStr(int aCount)
{	
	if (aCount == 0) return string("");
	char* nameTemp = new char[aCount + 1];
	nameTemp[aCount] = '\0';

	if (isFileStream)
		mFile->read(nameTemp, aCount);
	else
		mIStream->read(nameTemp, aCount);
	
	string ret(nameTemp);
	delete[] nameTemp;
	return ret;
}

string FileReader::ReadStrUTF(int aCount)
{
	return NULL;
}

void FileReader::Read(char* dest, size_t count)
{
	if (isFileStream)
		mFile->read(dest, count);
	else
		mIStream->read(dest, count);	
}

void FileReader::ReadWithStatus(char* dest, size_t count)
{
	if (count <= FILEREADER_CHUNK_SIZE)
	{
		if (isFileStream)
			mFile->read(dest, count);
		else
			mIStream->read(dest, count);

		if (readStatusCallback)
		{
			FileReaderStatus status;
			status.bytesToRead = count;
			status.bytesRead = count;
			(*readStatusCallback)(status);
		}
	}
	else
	{
		for (size_t sizeRead = 0; sizeRead < count; sizeRead +=FILEREADER_CHUNK_SIZE)
		{
			size_t sizeToRead = FILEREADER_CHUNK_SIZE;
			if (sizeRead + sizeToRead > count)
				sizeToRead = count - sizeRead;
			if (isFileStream)
				mFile->read(dest + sizeRead, sizeToRead);
			else
				mIStream->read(dest + sizeRead, sizeToRead);

			if (readStatusCallback)
			{
				FileReaderStatus status;
				status.bytesToRead = count;
				status.bytesRead = sizeRead + sizeToRead;
				(*readStatusCallback)(status);
			}
		}
	}
}

void FileReader::Seek(size_t pos, ios_base::seekdir dir)
{
	if (isFileStream)
		mFile->seekg(pos, dir);
	else
		mIStream->seekg(pos, dir);
	
}

size_t FileReader::Tell()
{
	if (isFileStream)
		return mFile->tellg();
	else
		return mIStream->tellg();
}

void FileReader::DeleteData(void * data, DataType_enum dataType) 
{

	// Delete existing data
	if (data) {
		// Just need to cast pointer to type with same size as intended
		switch (dataType)
		{	
			// Image unsigned bytes
			case DT_UCHAR:
				delete[] (uchar *)data;
			// Image unsigned shorts
			case DT_USHORT: 
				delete[] (ushort *)data;
			// Image unsigned long ints
			case DT_UINT: 
				delete[] (uint *)data;
			// Image unsigned long long ints
			case DT_ULONG: 
				delete[] (ulong64 *)data;
			// Image floats
			case DT_FLOAT:
				delete[] (float *)data;
			// Double-precision float
			case DT_DOUBLE:
				delete[] (double *)data;
			// Signed byte
			case DT_CHAR:
				delete[] (signed char *)data;
			// Image signed short integer
			case DT_SHORT:
				delete[] (short *)data;
			// Image signed int
			case DT_INT:
				delete[] (long *)data;
			// Image signed long long int
			case DT_LONG:
				delete[] (long64 *)data;
			// 8 bytes (two floats for one pixel in TIFF)
			case DT_FLOAT2:
				delete[] (float2_t *)data;
			// 4 bytes (two signed shorts)
			case DT_SHORT2:
				delete[] (short2_t *)data;
			// 6 bytes (3 signed shorts)
			case DT_SHORT3:
				delete[] (short3_t *)data;
			// 8 bytes (4 signed shorts)
			case DT_SHORT4:
				delete[] (short4_t *)data;
			// 16 bytes (2 doubles)
			case DT_DOUBLE2:
				delete[] (double2_t *)data;
			// 12 bytes (3 floats)
			case DT_FLOAT3:
				delete[] (float3_t *)data;
			// 16 bytes (4 floats)
			case DT_FLOAT4:
				delete[] (float4_t *)data;
			// Two bytes
			case DT_UCHAR2:
				delete[] (uchar2_t *)data;
			// 3 bytes
			case DT_UCHAR3:
				delete[] (uchar3_t *)data;
			// 4 bytes
			case DT_UCHAR4:
				delete[] (uchar4_t *)data;
			// 2 unsigned shorts for one pixel in TIFF
			case DT_USHORT2:
				delete[] (ushort2_t *)data;
			// 3 unsigned shorts for one pixel in TIFF
			case DT_USHORT3:
				delete[] (ushort3_t *)data;
			// 4 unsigned shorts for one pixel in TIFF
			case DT_USHORT4:
				delete[] (ushort4_t *)data;
			// Image RGB data - 3 unsigned chars
			case DT_RGB:
				delete[] (rgb_t *)data;
			default:
				// If no data type, just use char * for now
				// TODO: figure out best way to handle this
				delete[] (char *)data;
		}
	}
}