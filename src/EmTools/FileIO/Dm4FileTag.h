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


#ifndef DM4FILETAG_H
#define DM4FILETAG_H

#include "../Basics/Default.h"
#include "Dm3_4_Enums.h"

//!  Dm4FileTag represents a gatan *.dm4 tag. 
/*!
	\author Michael Kunz
	\date   September 2011
	\version 1.0
*/
class Dm4FileTag
{
public: 
	Dm4FileTag();
	virtual ~Dm4FileTag();

	std::string Name;
	ulong64 SizeInfoArray;
	ulong64* InfoArray; 
	char* Data;
	ulong64 SizeData;
	TagTypes tagType;

	uint GetSize(ulong64 typeId);
	uint GetSizePixel(ulong64 typeId);
	//void PrintValues(std::ostream& aStream);
	
	friend std::ostream& operator<<(std::ostream& stream, Dm4FileTag& tag);
	
	void PrintSingleValue(std::ostream& aStream, ulong64 aType, uint aStartPos = 0);
	int GetSingleValueInt(ulong64 aType, uint aStartPos = 0);
	float GetSingleValueFloat(ulong64 aType, uint aStartPos = 0);
	double GetSingleValueDouble(ulong64 aType, uint aStartPos = 0);
	std::string GetSingleValueString(ulong64 aType, uint aStartPos = 0);

	int GetStructValueAsInt(uint aIndex);
	void FreeData();
	/*uint GetValueAsUint();
	float GetValueAsFloat();
	double GetValueAsDouble();*/
	

};


#endif