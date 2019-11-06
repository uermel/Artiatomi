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

#include "IODefault.h"
#include "File.h"

using namespace std;

//!  FileReader provides endianess independent file read methods. 
/*!
	FileReader is an abstract class.
	\author Michael Kunz
	\date   September 2011
	\version 1.0
*/
class FileReader : public File
{
public:
	//! Creates a new FileReader instance for file \p aFileName. File endianess is set to little Endian.
	FileReader(string aFileName);
	//! Creates a new FileReader instance for file \p aFileName. File endianess is set to \p aIsLittleEndian.
	FileReader(string aFileName, bool aIsLittleEndian);
	//! Creates a new FileReader instance for file stream \p aStream. File endianess is set to little Endian.
	FileReader(fstream* aStream);
	//! Creates a new FileReader instance for file stream \p aStream. File endianess is set to \p aIsLittleEndian.
	FileReader(fstream* aStream, bool aIsLittleEndian);
	
	virtual bool OpenAndRead() = 0;
	virtual FileDataType_enum GetDataType() = 0;
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

};

#endif