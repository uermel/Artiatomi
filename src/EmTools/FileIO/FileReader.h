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


#ifndef FILEREADER_H
#define FILEREADER_H

#include "../Basics/Default.h"
#include "File.h"
#include <istream>

using namespace std;

#define FILEREADER_CHUNK_SIZE (10*1024*1024) //10MB

//!  FileReader provides endianess independent file read methods. 
/*!
	FileReader is an abstract class.
	\author Michael Kunz
	\date   September 2011
	\version 1.0
*/
class FileReader : public File
{
protected:
	bool isFileStream;
	istream* mIStream;

public:
	struct FileReaderStatus_struct
	{
		size_t bytesToRead;
		size_t bytesRead;
	};
	typedef struct FileReaderStatus_struct FileReaderStatus;

	//! Creates a new FileReader instance for file \p aFileName. File endianess is set to little Endian.
	FileReader(string aFileName);
	//! Creates a new FileReader instance for file \p aFileName. File endianess is set to \p aIsLittleEndian.
	FileReader(string aFileName, bool aIsLittleEndian);
	//! Creates a new FileReader instance for file stream \p aStream. File endianess is set to little Endian.
	FileReader(fstream* aStream);
	//! Creates a new FileReader instance for file stream \p aStream. File endianess is set to \p aIsLittleEndian.
	FileReader(fstream* aStream, bool aIsLittleEndian);

	FileReader(istream* aStream, bool aIsLittleEndian);
	
	virtual bool OpenAndRead() = 0;
	virtual DataType_enum GetDataType() = 0;
	
	void(*readStatusCallback)(FileReaderStatus );

protected:
	long64 ReadI8LE();
	long64 ReadI8BE();
	int ReadI4LE();
	int ReadI4BE();
	short ReadI2LE();
	short ReadI2BE();
	char ReadI1();
	ulong64 ReadUI8LE();
	ulong64 ReadUI8BE();
	uint ReadUI4LE();
	uint ReadUI4BE();
	ushort ReadUI2LE();
	ushort ReadUI2BE();
	uchar ReadUI1();
	float ReadF4LE();
	float ReadF4BE();
	double ReadF8LE();
	double ReadF8BE();
	string ReadStr(int aCount);
	string ReadStrUTF(int aCount);
	bool OpenRead();
	void CloseRead();
	void Read(char* dest, size_t count);
	void ReadWithStatus(char* dest, size_t count);

	void Seek(size_t pos, ios_base::seekdir dir);
	size_t Tell();

	friend class ImageFileDirectoryEntry;
	friend class StringImageFileDirectory;
	friend class ByteImageFileDirectory;
	friend class SByteImageFileDirectory;
	friend class UShortImageFileDirectory;
	friend class ShortImageFileDirectory;
	friend class UIntImageFileDirectory;
	friend class IntImageFileDirectory;
	friend class RationalImageFileDirectory;
	friend class SRationalImageFileDirectory;
	friend class FloatImageFileDirectory;
	friend class DoubleImageFileDirectory;

	friend class IFDImageLength;
	friend class IFDImageWidth;
	friend class IFDRowsPerStrip;
	friend class IFDStripByteCounts;
	friend class IFDStripOffsets;

	friend class ImageFileDirectory;
};

#endif