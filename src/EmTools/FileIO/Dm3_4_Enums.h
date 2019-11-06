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


#ifndef DM3_4_ENUMS_H
#define DM3_4_ENUMS_H

enum DataTypesTag
{
	DTTI2 = 2,
	DTTI4 = 3,
	DTTUI2 = 4,
	DTTUI4 = 5,
	DTTF4 = 6,
	DTTF8 = 7,
	DTTI1 = 8,
	DTTCHAR = 9,
	DTTI1_2 = 10,
	DTTUI8 = 11
};

enum DataTypesImage
{
	DTT_UNKNOWN = 0,
	DTT_I2 = 1,
	DTT_F4 = 2,
	DTT_C8 = 3,
	DTT_OBSOLETE = 4,
	DTT_C4 = 5,
	DTT_UI1 = 6,
	DTT_I4 = 7,
	DTT_RGB_4UI1 = 8,
	DTT_I1 = 9,
	DTT_UI2 = 10,
	DTT_UI4 = 11,
	DTT_F8 = 12,
	DTT_C16 = 13,
	DTT_BINARY = 14,
	DTT_RGBA_4UI1 = 23
};

enum TagTypes
{
	SingleEntryTag,
	StructTag,
	ArrayTag,
	ArrayStructTag
};

enum DataTypes
{
	DTI2 = 2,
	DTI4 = 3,
	DTUI2 = 4,
	DTUI4 = 5,
	DTF4 = 6,
	DTF8 = 7,
	DTI1 = 8,
	DTCHAR = 9,
	DTI1_2 = 10,
	DTSTRUCT = 15,
	DTSTR = 18,
	DTARRAY = 20
};

#endif