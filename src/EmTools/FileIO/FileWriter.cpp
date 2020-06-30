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


#include "FileWriter.h"
#include "../MKLog/MKLog.h"

FileWriter::FileWriter(string aFileName)
	: File(aFileName, true)
{

}

FileWriter::FileWriter(string aFileName, bool aIsLittleEndian)
	: File(aFileName, aIsLittleEndian)
{

}
FileWriter::FileWriter(fstream* aStream)
	: File(aStream, true)
{

}

FileWriter::FileWriter(fstream* aStream, bool aIsLittleEndian)
	: File(aStream, aIsLittleEndian)
{

}

bool FileWriter::OpenWrite(bool createNew)
{
	if (createNew)
		mFile->open(mFileName.c_str(), ios_base::in | ios_base::out | ios_base::binary | ios_base::trunc);
	else
		mFile->open(mFileName.c_str(), ios_base::in | ios_base::out | ios_base::binary);
	bool status = mFile->is_open() && mFile->good();
	MKLOG("FileWriter opened file " + mFileName + ". Status: " + (status ? "GOOD" : "BAD"));
	return status;
}

void FileWriter::CloseWrite()
{
	MKLOG("FileWriter closed file " + mFileName + ".");
	mFile->close();
}


void FileWriter::WriteBE(ulong64& aX)
{	
	if (mIsLittleEndian) Endian_swap(aX);

	mFile->write((char*)&aX, 8);
}

void FileWriter::WriteLE(ulong64& aX)
{
	if (!mIsLittleEndian) Endian_swap(aX);

	mFile->write((char*)&aX, 8);
}

void FileWriter::WriteBE(uint& aX)
{
	if (mIsLittleEndian) Endian_swap(aX);

	mFile->write((char*)&aX, 4);
}

void FileWriter::WriteLE(uint& aX)
{
	if (!mIsLittleEndian) Endian_swap(aX);

	mFile->write((char*)&aX, 4);
}

void FileWriter::WriteBE(ushort& aX)
{
	if (mIsLittleEndian) Endian_swap(aX);

	mFile->write((char*)&aX, 2);
}

void FileWriter::WriteLE(ushort& aX)
{
	if (!mIsLittleEndian) Endian_swap(aX);

	mFile->write((char*)&aX, 2);
}

void FileWriter::Write(uchar& aX)
{
	mFile->write((char*)&aX, 1);
}

void FileWriter::WriteBE(long64& aX)
{
	if (mIsLittleEndian) Endian_swap(aX);

	mFile->write((char*)&aX, 8);
}

void FileWriter::WriteLE(long64& aX)
{
	if (!mIsLittleEndian) Endian_swap(aX);

	mFile->write((char*)&aX, 8);
}

void FileWriter::WriteBE(int& aX)
{
	if (mIsLittleEndian) Endian_swap(aX);

	mFile->write((char*)&aX, 4);
}

void FileWriter::WriteLE(int& aX)
{
	if (!mIsLittleEndian) Endian_swap(aX);

	mFile->write((char*)&aX, 4);
}

void FileWriter::WriteBE(short& aX)
{
	if (mIsLittleEndian) Endian_swap(aX);

	mFile->write((char*)&aX, 2);
}

void FileWriter::WriteLE(short& aX)
{
	if (!mIsLittleEndian) Endian_swap(aX);

	mFile->write((char*)&aX, 2);
}

void FileWriter::Write(char& aX)
{
	mFile->write((char*)&aX, 1);
}

void FileWriter::WriteBE(double& aX)
{
	if (mIsLittleEndian) Endian_swap(aX);

	mFile->write((char*)&aX, 8);
}

void FileWriter::WriteLE(double& aX)
{
	if (!mIsLittleEndian) Endian_swap(aX);

	mFile->write((char*)&aX, 8);
}

void FileWriter::WriteBE(float& aX)
{
	if (mIsLittleEndian) Endian_swap(aX);

	mFile->write((char*)&aX, 4);
}

void FileWriter::WriteLE(float& aX)
{
	if (!mIsLittleEndian) Endian_swap(aX);

	mFile->write((char*)&aX, 4);
}

void FileWriter::Write(void* aX, size_t aCount)
{
	mFile->write(reinterpret_cast<char*>(aX), aCount);
}

void FileWriter::Seek(size_t pos, ios_base::seekdir dir)
{
	mFile->seekp(pos, dir);
}

size_t FileWriter::Tell()
{
	return mFile->tellp();
}

void FileWriter::Write(char* src, size_t count)
{
	mFile->write(src, count);
}

void FileWriter::WriteWithStatus(char* src, size_t count)
{
	if (count <= FILEWRITER_CHUNK_SIZE)
	{
		mFile->read(src, count);
		if (writeStatusCallback)
			(*writeStatusCallback)(100);
	}
	else
	{
		for (size_t sizeWrite = 0; sizeWrite < count; sizeWrite += FILEWRITER_CHUNK_SIZE)
		{
			size_t sizeToWrite = FILEWRITER_CHUNK_SIZE;
			if (sizeWrite + sizeToWrite > count)
				sizeToWrite = count - sizeWrite;

			mFile->read(src + sizeWrite, sizeToWrite);

			if (writeStatusCallback)
				(*writeStatusCallback)((int)(((sizeWrite + sizeToWrite) * 100) / count));
		}
	}
}