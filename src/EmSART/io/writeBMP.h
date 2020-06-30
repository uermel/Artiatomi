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


#ifndef WRITEBMP_H
#define WRITEBMP_H

#include "IODefault.h"



#define BITMAP_SIGNATURE 'MB'

typedef struct {
	unsigned int Size;
	unsigned int Reserved;
	unsigned int BitsOffset;
} BITMAP_FILEHEADER;

#define BITMAP_FILEHEADER_SIZE 14

typedef struct {
	unsigned int HeaderSize;
	int Width;
	int Height;
	unsigned short int Planes;
	unsigned short int BitCount;
	unsigned int Compression;
	unsigned int SizeImage;
	int PelsPerMeterX;
	int PelsPerMeterY;
	unsigned int ClrUsed;
	unsigned int ClrImportant;
} BITMAP_HEADER;


typedef struct {
	unsigned char Blue;
	unsigned char Green;
	unsigned char Red;
} Pixel;

//! Small and simple windows bitmap write function. Normalizes values to min/max. Float values
//Small and simple windows bitmap write function. Normalizes values to min/max. Float values
void writeBMP(std::string filename, float* aData, uint aWidth, uint aHeight);

//! Small and simple windows bitmap write function. Normalizes values to min/max. Double values
//Small and simple windows bitmap write function. Normalizes values to min/max. Double values
void writeBMP(std::string filename, double* aData, uint aWidth, uint aHeight);

//! Small and simple windows bitmap write function. Normalizes values to min/max. Int values
//Small and simple windows bitmap write function. Normalizes values to min/max. Int values
void writeBMP(std::string filename, int* aData, uint aWidth, uint aHeight);

//! Small and simple windows bitmap write function. Normalizes values to min/max. Ushort values
//Small and simple windows bitmap write function. Normalizes values to min/max. Ushort values
void writeBMP(std::string filename, ushort* aData, uint aWidth, uint aHeight);

//! Small and simple windows bitmap write function. Normalizes values to min/max. Char values
//Small and simple windows bitmap write function. Normalizes values to min/max. Char values
void writeBMP(std::string filename, char* aData, uint aWidth, uint aHeight);

#endif //WRITEBMP_H