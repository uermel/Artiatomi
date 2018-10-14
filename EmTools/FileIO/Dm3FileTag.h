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


#ifndef DM3FILETAG_H
#define DM3FILETAG_H

#include "../Basics/Default.h"
#include "Dm3_4_Enums.h"

//!  Dm3FileTag represents a gatan *.dm3 tag. 
/*!
	\author Michael Kunz
	\date   September 2011
	\version 1.0
*/
class Dm3FileTag
{
public: 
	Dm3FileTag();
	virtual ~Dm3FileTag();

	std::string Name;
	uint SizeInfoArray;
	uint* InfoArray; 
	char* Data;
	uint SizeData;
	TagTypes tagType;

	uint GetSize(uint typeId);
	uint GetSizePixel(uint typeId);
	//void PrintValues(std::ostream& aStream);
	
	friend std::ostream& operator<<(std::ostream& stream, Dm3FileTag& tag);
	
	void PrintSingleValue(std::ostream& aStream, uint aType, uint aStartPos = 0);
	int GetSingleValueInt(uint aType, uint aStartPos = 0);
	float GetSingleValueFloat(uint aType, uint aStartPos = 0);
	double GetSingleValueDouble(uint aType, uint aStartPos = 0);
	std::string GetSingleValueString(uint aType, uint aStartPos = 0);

	int GetStructValueAsInt(uint aIndex);
	void FreeData();
	/*uint GetValueAsUint();
	float GetValueAsFloat();
	double GetValueAsDouble();*/
	

};


#endif