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


#include "writeBMP.h"

using namespace std;

void writeBMP(string filename, float* aData, uint aWidth, uint aHeight)
{
	uint fitX, fitY;
	fitX = aWidth * 3;
	while(fitX % 4)
	{
	    fitX += 1;
	}
	 //+ (aWidth * 3) % 4;
	fitY = aHeight;
	float Min = FLT_MAX, Max = -FLT_MAX;
    //printf("FitX: %d\n",fitX);
    double mean = 0;
	for (uint x = 0; x < aWidth; x++)
	for (uint y = 0; y < aHeight; y++)
	{
	    //if (isinf(aData[x + y * aWidth])) aData[x + y * aWidth] = 123;
		Min = min(aData[x + y * aWidth], Min);
		Max = max(aData[x + y * aWidth], Max);
		//if (x > 1024 && x < 3096 && y > 1024 && y < 3096)
            mean += aData[x + y * aWidth];

		/*if (aData[x + y * aWidth] > 10000)
			printf("Error in Pixel: %d x %d\n", x, y);*/
	}

	mean = mean / aWidth / aHeight;
    printf("Min: %f, Max: %f, Mean: %f\n", Min, Max, (float)(mean));
	unsigned char* pixels = new unsigned char[fitX * fitY];
	memset(pixels, 0, fitX * fitY);
	
	/*Max = 3 * mean;
	Max = 50;*/

	for (uint i = 0; i < fitX-2; i+= 3)
	for (uint j = 0; j < fitY; j+= 1)
	{
		Pixel* pixel = (Pixel*)(pixels + (i + j * fitX));

		uint jInv = fitY - j - 1;

        /*if (aData[i / 3 + jInv * aWidth] < Min) aData[i / 3 + jInv * aWidth] = Min;
        if (aData[i / 3 + jInv * aWidth] > Max) aData[i / 3 + jInv * aWidth] = Max;*/

		pixel->Blue  = (unsigned char)((aData[i / 3 + jInv * aWidth] - Min) / (Max - Min) * 255.0f);
		pixel->Green = (unsigned char)((aData[i / 3 + jInv * aWidth] - Min) / (Max - Min) * 255.0f);
		pixel->Red   = (unsigned char)((aData[i / 3 + jInv * aWidth] - Min) / (Max - Min) * 255.0f);
	}

	BITMAP_FILEHEADER fileheader;
	BITMAP_HEADER bmpheader;

	fileheader.Size = BITMAP_FILEHEADER_SIZE + sizeof(bmpheader) + fitX * fitY;
	fileheader.Reserved = 0;
	fileheader.BitsOffset = BITMAP_FILEHEADER_SIZE + sizeof(bmpheader);

	ofstream bmp(filename.c_str(), ios::binary);
	bmp.write("BM", 2);
	bmp.write((char*)&fileheader, 12);

	memset(&bmpheader, 0, sizeof(bmpheader));

	bmpheader.HeaderSize = sizeof(bmpheader);
	bmpheader.Width = aWidth;
	bmpheader.Height = aHeight;
	bmpheader.Planes = 1;
	bmpheader.BitCount = 24;
	bmpheader.SizeImage = fitX * fitY * sizeof(Pixel);

	bmp.write((char*)&bmpheader, sizeof(bmpheader));
	bmp.write((char*)pixels, fitX * fitY);

	bmp.close();

}

void writeBMP(string filename, double* aData, uint aWidth, uint aHeight)
{
	uint fitX, fitY;
	fitX = aWidth * 3 + (aWidth * 3) % 4;
	fitY = aHeight;
	double Min = DBL_MAX, Max = -DBL_MAX;


	for (uint x = 0; x < aWidth; x++)
	for (uint y = 0; y < aHeight; y++)
	{
		Min = min(aData[x + y * aWidth], Min);
		Max = max(aData[x + y * aWidth], Max);
	}

	unsigned char* pixels = new unsigned char[fitX * fitY];
	memset(pixels, 0, fitX * fitY);

	for (uint i = 0; i < fitX-2; i+= 3)
	for (uint j = 0; j < fitY; j+= 1)
	{
		Pixel* pixel = (Pixel*)(pixels + (i + j * fitX));

		uint jInv = fitY - j - 1;

		pixel->Blue  = (unsigned char)((aData[i / 3 + jInv * aWidth] - Min) / (Max - Min) * 255.0f);
		pixel->Green = (unsigned char)((aData[i / 3 + jInv * aWidth] - Min) / (Max - Min) * 255.0f);
		pixel->Red   = (unsigned char)((aData[i / 3 + jInv * aWidth] - Min) / (Max - Min) * 255.0f);
	}

	BITMAP_FILEHEADER fileheader;
	BITMAP_HEADER bmpheader;

	fileheader.Size = BITMAP_FILEHEADER_SIZE + sizeof(bmpheader) + fitX * fitY;
	fileheader.Reserved = 0;
	fileheader.BitsOffset = BITMAP_FILEHEADER_SIZE + sizeof(bmpheader);

	ofstream bmp(filename.c_str(), ios::binary);
	bmp.write("BM", 2);
	bmp.write((char*)&fileheader, 12);

	memset(&bmpheader, 0, sizeof(bmpheader));

	bmpheader.HeaderSize = sizeof(bmpheader);
	bmpheader.Width = aWidth;
	bmpheader.Height = aHeight;
	bmpheader.Planes = 1;
	bmpheader.BitCount = 24;
	bmpheader.SizeImage = fitX * fitY * sizeof(Pixel);

	bmp.write((char*)&bmpheader, sizeof(bmpheader));
	bmp.write((char*)pixels, fitX * fitY);

	bmp.close();

}

void writeBMP(string filename, int* aData, uint aWidth, uint aHeight)
{
	uint fitX, fitY;
	fitX = aWidth * 3 + (aWidth * 3) % 4;
	fitY = aHeight;
	float Min = FLT_MAX, Max = -FLT_MAX;


	for (uint x = 0; x < aWidth; x++)
	for (uint y = 0; y < aHeight; y++)
	{
		Min = min((float)aData[x + y * aWidth], Min);
		Max = max((float)aData[x + y * aWidth], Max);
	}
	//std::cout << "Min: " << Min << " Max: " << Max << endl;

	unsigned char* pixels = new unsigned char[fitX * fitY];
	memset(pixels, 0, fitX * fitY);

	for (uint i = 0; i < fitX-2; i+= 3)
	for (uint j = 0; j < fitY; j+= 1)
	{
		Pixel* pixel = (Pixel*)(pixels + (i + j * fitX));

		uint jInv = fitY - j - 1;

		pixel->Blue  = (unsigned char)(((float)aData[i / 3 + jInv * aWidth] - Min) / (Max - Min) * 255.0f);
		pixel->Green = (unsigned char)(((float)aData[i / 3 + jInv * aWidth] - Min) / (Max - Min) * 255.0f);
		pixel->Red   = (unsigned char)(((float)aData[i / 3 + jInv * aWidth] - Min) / (Max - Min) * 255.0f);
	}

	BITMAP_FILEHEADER fileheader;
	BITMAP_HEADER bmpheader;

	fileheader.Size = BITMAP_FILEHEADER_SIZE + sizeof(bmpheader) + fitX * fitY;
	fileheader.Reserved = 0;
	fileheader.BitsOffset = BITMAP_FILEHEADER_SIZE + sizeof(bmpheader);

	ofstream bmp(filename.c_str(), ios::binary);
	bmp.write("BM", 2);
	bmp.write((char*)&fileheader, 12);

	memset(&bmpheader, 0, sizeof(bmpheader));

	bmpheader.HeaderSize = sizeof(bmpheader);
	bmpheader.Width = aWidth;
	bmpheader.Height = aHeight;
	bmpheader.Planes = 1;
	bmpheader.BitCount = 24;
	bmpheader.SizeImage = fitX * fitY * sizeof(Pixel);

	bmp.write((char*)&bmpheader, sizeof(bmpheader));
	bmp.write((char*)pixels, fitX * fitY);

	bmp.close();

}

void writeBMP(string filename, ushort* aData, uint aWidth, uint aHeight)
{
	uint fitX, fitY;
	fitX = aWidth * 3 + (aWidth * 3) % 4;
	fitY = aHeight;
	float Min = FLT_MAX, Max = -FLT_MAX;


	for (uint x = 0; x < aWidth; x++)
	for (uint y = 0; y < aHeight; y++)
	{
		Min = min((float)aData[x + y * aWidth], Min);
		Max = max((float)aData[x + y * aWidth], Max);
	}
    printf("Min: %f, Max: %f\n", Min, Max);
	//std::cout << "Min: " << Min << " Max: " << Max << endl;

	unsigned char* pixels = new unsigned char[fitX * fitY];
	memset(pixels, 0, fitX * fitY);

	for (uint i = 0; i < fitX-2; i+= 3)
	for (uint j = 0; j < fitY; j+= 1)
	{
		Pixel* pixel = (Pixel*)(pixels + (i + j * fitX));

		uint jInv = fitY - j - 1;

		pixel->Blue  = (unsigned char)(((float)aData[i / 3 + jInv * aWidth] - Min) / (Max - Min) * 255.0f);
		pixel->Green = (unsigned char)(((float)aData[i / 3 + jInv * aWidth] - Min) / (Max - Min) * 255.0f);
		pixel->Red   = (unsigned char)(((float)aData[i / 3 + jInv * aWidth] - Min) / (Max - Min) * 255.0f);
	}

	BITMAP_FILEHEADER fileheader;
	BITMAP_HEADER bmpheader;

	fileheader.Size = BITMAP_FILEHEADER_SIZE + sizeof(bmpheader) + fitX * fitY;
	fileheader.Reserved = 0;
	fileheader.BitsOffset = BITMAP_FILEHEADER_SIZE + sizeof(bmpheader);

	ofstream bmp(filename.c_str(), ios::binary);
	bmp.write("BM", 2);
	bmp.write((char*)&fileheader, 12);

	memset(&bmpheader, 0, sizeof(bmpheader));

	bmpheader.HeaderSize = sizeof(bmpheader);
	bmpheader.Width = aWidth;
	bmpheader.Height = aHeight;
	bmpheader.Planes = 1;
	bmpheader.BitCount = 24;
	bmpheader.SizeImage = fitX * fitY * sizeof(Pixel);

	bmp.write((char*)&bmpheader, sizeof(bmpheader));
	bmp.write((char*)pixels, fitX * fitY);

	bmp.close();

}

void writeBMP(string filename, char* aData, uint aWidth, uint aHeight)
{
	uint fitX, fitY;
	fitX = aWidth * 3 + (aWidth * 3) % 4;
	fitY = aHeight;


	unsigned char* pixels = new unsigned char[fitX * fitY];
	memset(pixels, 0, fitX * fitY);

	for (uint i = 0; i < fitX-2; i+= 3)
	for (uint j = 0; j < fitY; j+= 1)
	{
		Pixel* pixel = (Pixel*)(pixels + (i + j * fitX));

		uint jInv = fitY - j - 1;

		pixel->Blue  = (unsigned char)(aData[i / 3 + jInv * aWidth]);
		pixel->Green = (unsigned char)(aData[i / 3 + jInv * aWidth]);
		pixel->Red   = (unsigned char)(aData[i / 3 + jInv * aWidth]);
	}

	BITMAP_FILEHEADER fileheader;
	BITMAP_HEADER bmpheader;

	fileheader.Size = BITMAP_FILEHEADER_SIZE + sizeof(bmpheader) + fitX * fitY;
	fileheader.Reserved = 0;
	fileheader.BitsOffset = BITMAP_FILEHEADER_SIZE + sizeof(bmpheader);

	ofstream bmp(filename.c_str(), ios::binary);
	bmp.write("BM", 2);
	bmp.write((char*)&fileheader, 12);

	memset(&bmpheader, 0, sizeof(bmpheader));

	bmpheader.HeaderSize = sizeof(bmpheader);
	bmpheader.Width = aWidth;
	bmpheader.Height = aHeight;
	bmpheader.Planes = 1;
	bmpheader.BitCount = 24;
	bmpheader.SizeImage = fitX * fitY * sizeof(Pixel);

	bmp.write((char*)&bmpheader, sizeof(bmpheader));
	bmp.write((char*)pixels, fitX * fitY);

	bmp.close();

}
