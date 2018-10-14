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


#include "File.h"

File::File(string aFileName, bool aIsLittleEndian)
	: mFileName(aFileName), mIsLittleEndian(aIsLittleEndian)
{
	mFile = new fstream();
}

File::File(fstream* aStream, bool aIsLittleEndian)
	: mFileName(), mIsLittleEndian(aIsLittleEndian)
{
	mFile = aStream;
}

File::~File()
{
	if (mFile)
		delete mFile;
	mFile = NULL;
}

void File::Endian_swap(ushort& x)
{
	x = (x>>8) | 
		(x<<8);
}

void File::Endian_swap(uint& x)
{
	x =
	 (x>>24) | 
		((x<<8) & 0x00FF0000) |
		((x>>8) & 0x0000FF00) |
		(x<<24);
}

void File::Endian_swap(ulong64& x)
{
	x = (x>>56) | 
		((x<<40) & 0x00FF000000000000) |
		((x<<24) & 0x0000FF0000000000) |
		((x<<8)  & 0x000000FF00000000) |
		((x>>8)  & 0x00000000FF000000) |
		((x>>24) & 0x0000000000FF0000) |
		((x>>40) & 0x000000000000FF00) |
		(x<<56);
}

void File::Endian_swap(short& x)
{
	ushort temp = *((ushort*)&x);
	temp = (temp >>8) |
		(temp <<8);
	x = *((short*)&temp);
}

void File::Endian_swap(int& x)
{
	uint temp = *((uint*)&x);
	temp = (temp>>24) |
		((temp <<8) & 0x00FF0000) |
		((temp >>8) & 0x0000FF00) |
		(temp <<24);
	x = *((int*)&temp);
}

void File::Endian_swap(long64& x)
{
	ulong64 temp = *((ulong64*)&x);
	temp = (temp >>56) |
		((temp <<40) & 0x00FF000000000000) |
		((temp <<24) & 0x0000FF0000000000) |
		((temp <<8)  & 0x000000FF00000000) |
		((temp >>8)  & 0x00000000FF000000) |
		((temp >>24) & 0x0000000000FF0000) |
		((temp >>40) & 0x000000000000FF00) |
		(temp <<56);
	x = *((ulong64*)&temp);
}

void File::Endian_swap(float& x)
{
	uint i = * (uint*) &x;
	i = (i>>24) | 
		((i<<8) & 0x00FF0000) |
		((i>>8) & 0x0000FF00) |
		(i<<24);
	x = * (float *) &i;
}

void File::Endian_swap(double& x)
{
	ulong64 i = * (ulong64*) &x;
	i = (i>>56) | 
		((i<<40) & 0x00FF000000000000) |
		((i<<24) & 0x0000FF0000000000) |
		((i<<8)  & 0x000000FF00000000) |
		((i>>8)  & 0x00000000FF000000) |
		((i>>24) & 0x0000000000FF0000) |
		((i>>40) & 0x000000000000FF00) |
		(i<<56);
	x = * (double *) &i;
}
