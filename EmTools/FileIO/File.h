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


#ifndef FILE_H
#define FILE_H

#include "../Basics/Default.h"
#include "FileIOException.h"

using namespace std;

//!  File is a very simple class wrapping a file stream. 
/*!
	File only wrapps a file stream and provides a filename and endian-swap methods.
	\author Michael Kunz
	\date   September 2011
	\version 1.0
*/
class File
{
protected:
	fstream* mFile;
	string mFileName;

	bool mIsLittleEndian;

	void Endian_swap(ushort& x);
	void Endian_swap(uint& x);
	void Endian_swap(ulong64& x);
	void Endian_swap(short& x);
	void Endian_swap(int& x);
	void Endian_swap(long64& x);
	void Endian_swap(double& x);
	void Endian_swap(float& x);

public:
	//! Creates a new instance of File with name \p aFileName and endianess \p aIsLittleEndian.
	File(string aFileName, bool aIsLittleEndian);
	//! Creates a new instance of File with filestram given by \p aStream and endianess \p aIsLittleEndian.
	File(fstream* aStream, bool aIsLittleEndian);
	virtual ~File();


};

#endif