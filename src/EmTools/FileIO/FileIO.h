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


#ifndef FILEIO_H
#define FILEIO_H

#include "../Basics/Default.h"

enum FileType_enum
{
	FT_DM3,
	FT_DM4,
	FT_MRC,
	FT_EM,
	FT_SER,
	FT_TIFF,
	FT_NONE
};

#include "MRCFile.h"
#include "Dm3File.h"
#include "Dm4File.h"
#include "SERFile.h"
#include "EmFile.h"
#include "TIFFFile.h"

#endif
