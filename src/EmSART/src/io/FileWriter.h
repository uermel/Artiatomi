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


#ifndef FILEWRITER_H
#define FILEWRITER_H

#include "IODefault.h"
#include "File.h"

using namespace std;

//!  FileWriter provides endianess independent file write methods. 
/*!
	FileWriter is an abstract class.
	\author Michael Kunz
	\date   September 2011
	\version 1.0
*/
class FileWriter : public File
{
public:
	//! Creates a new FileWriter instance for file \p aFileName. File endianess is set to little Endian.
	FileWriter(string aFileName);
	//! Creates a new FileWriter instance for file \p aFileName. File endianess is set to \p aIsLittleEndian.
	FileWriter(string aFileName, bool aIsLittleEndian);
	//! Creates a new FileWriter instance for file stream \p aStream. File endianess is set to little Endian.
	FileWriter(fstream* aStream);
	//! Creates a new FileWriter instance for file stream \p aStream. File endianess is set to \p aIsLittleEndian.
	FileWriter(fstream* aStream, bool aIsLittleEndian);
	
	virtual bool OpenAndWrite() = 0;
	virtual void SetDataType(FileDataType_enum aType) = 0;

protected:	
	void WriteBE(ulong64& aX);
	void WriteLE(ulong64& aX);
	void WriteBE(uint& aX);
	void WriteLE(uint& aX);
	void WriteBE(ushort& aX);
	void WriteLE(ushort& aX);
	void Write(uchar& aX);
	void WriteBE(long64& aX);
	void WriteLE(long64& aX);
	void WriteBE(int& aX);
	void WriteLE(int& aX);
	void WriteBE(short& aX);
	void WriteLE(short& aX);
	void Write(char& aX);
	void WriteBE(double& aX);
	void WriteLE(double& aX);
	void WriteBE(float& aX);
	void WriteLE(float& aX);
	void Write(char* aX, uint aCount);

	bool OpenWrite(bool append = false);
	void CloseWrite();
	bool OpenAndWriteAppend(char* aData, uint aCount);
};

#endif