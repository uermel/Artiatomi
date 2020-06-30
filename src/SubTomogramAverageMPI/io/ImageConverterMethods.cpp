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


#include "Image.h"


	uchar* Image::ConvertPixelTypeUCHAR(ushort* in, int aAdd)
	{
		uchar* tmp = new uchar[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (uchar)(in[i] + aAdd);
		}
		return tmp;
	}
	uchar* Image::ConvertPixelTypeUCHAR(uint* in, int aAdd)
	{
		uchar* tmp = new uchar[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (uchar)(in[i] + aAdd);
		}
		return tmp;
	}
	uchar* Image::ConvertPixelTypeUCHAR(ulong64* in, int aAdd)
	{
		uchar* tmp = new uchar[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (uchar)(in[i] + aAdd);
		}
		return tmp;
	}
	uchar* Image::ConvertPixelTypeUCHAR(float* in, int aAdd)
	{
		uchar* tmp = new uchar[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (uchar)(in[i] + aAdd);
		}
		return tmp;
	}
	uchar* Image::ConvertPixelTypeUCHAR(double* in, int aAdd)
	{
		uchar* tmp = new uchar[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (uchar)(in[i] + aAdd);
		}
		return tmp;
	}
	uchar* Image::ConvertPixelTypeUCHAR(char* in, int aAdd)
	{
		uchar* tmp = new uchar[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (uchar)(in[i] + aAdd);
		}
		return tmp;
	}
	uchar* Image::ConvertPixelTypeUCHAR(short* in, int aAdd)
	{
		uchar* tmp = new uchar[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (uchar)(in[i] + aAdd);
		}
		return tmp;
	}
	uchar* Image::ConvertPixelTypeUCHAR(int* in, int aAdd)
	{
		uchar* tmp = new uchar[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (uchar)(in[i] + aAdd);
		}
		return tmp;
	}
	uchar* Image::ConvertPixelTypeUCHAR(long* in, int aAdd)
	{
		uchar* tmp = new uchar[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (uchar)(in[i] + aAdd);
		}
		return tmp;
	}
	
	ushort* Image::ConvertPixelTypeUSHORT(uchar* in, int aAdd)
	{
		ushort* tmp = new ushort[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (ushort)(in[i] + aAdd);
		}
		return tmp;
	}
	ushort* Image::ConvertPixelTypeUSHORT(uint* in, int aAdd)
	{
		ushort* tmp = new ushort[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (ushort)(in[i] + aAdd);
		}
		return tmp;
	}
	ushort* Image::ConvertPixelTypeUSHORT(ulong64* in, int aAdd)
	{
		ushort* tmp = new ushort[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (ushort)(in[i] + aAdd);
		}
		return tmp;
	}
	ushort* Image::ConvertPixelTypeUSHORT(float* in, int aAdd)
	{
		ushort* tmp = new ushort[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (ushort)(in[i] + aAdd);
		}
		return tmp;
	}
	ushort* Image::ConvertPixelTypeUSHORT(double* in, int aAdd)
	{
		ushort* tmp = new ushort[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (ushort)(in[i] + aAdd);
		}
		return tmp;
	}
	ushort* Image::ConvertPixelTypeUSHORT(char* in, int aAdd)
	{
		ushort* tmp = new ushort[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (ushort)(in[i] + aAdd);
		}
		return tmp;
	}
	ushort* Image::ConvertPixelTypeUSHORT(short* in, int aAdd)
	{
		ushort* tmp = new ushort[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (ushort)(in[i] + aAdd);
		}
		return tmp;
	}
	ushort* Image::ConvertPixelTypeUSHORT(int* in, int aAdd)
	{
		ushort* tmp = new ushort[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (ushort)(in[i] + aAdd);
		}
		return tmp;
	}
	ushort* Image::ConvertPixelTypeUSHORT(long* in, int aAdd)
	{
		ushort* tmp = new ushort[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (ushort)(in[i] + aAdd);
		}
		return tmp;
	}
	
	uint* Image::ConvertPixelTypeUINT(uchar* in, int aAdd)
	{
		uint* tmp = new uint[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (uint)(in[i] + aAdd);
		}
		return tmp;
	}
	uint* Image::ConvertPixelTypeUINT(ushort* in, int aAdd)
	{
		uint* tmp = new uint[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (uint)(in[i] + aAdd);
		}
		return tmp;
	}
	uint* Image::ConvertPixelTypeUINT(ulong64* in, int aAdd)
	{
		uint* tmp = new uint[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (uint)(in[i] + aAdd);
		}
		return tmp;
	}
	uint* Image::ConvertPixelTypeUINT(float* in, int aAdd)
	{
		uint* tmp = new uint[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (uint)(in[i] + aAdd);
		}
		return tmp;
	}
	uint* Image::ConvertPixelTypeUINT(double* in, int aAdd)
	{
		uint* tmp = new uint[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (uint)(in[i] + aAdd);
		}
		return tmp;
	}
	uint* Image::ConvertPixelTypeUINT(char* in, int aAdd)
	{
		uint* tmp = new uint[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (uint)(in[i] + aAdd);
		}
		return tmp;
	}
	uint* Image::ConvertPixelTypeUINT(short* in, int aAdd)
	{
		uint* tmp = new uint[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (uint)(in[i] + aAdd);
		}
		return tmp;
	}
	uint* Image::ConvertPixelTypeUINT(int* in, int aAdd)
	{
		uint* tmp = new uint[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (uint)(in[i] + aAdd);
		}
		return tmp;
	}
	uint* Image::ConvertPixelTypeUINT(long* in, int aAdd)
	{
		uint* tmp = new uint[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (uint)(in[i] + aAdd);
		}
		return tmp;
	}
	
	ulong64* Image::ConvertPixelTypeULONG(uchar* in, int aAdd)
	{
		ulong64* tmp = new ulong64[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (ulong64)(in[i] + aAdd);
		}
		return tmp;
	}
	ulong64* Image::ConvertPixelTypeULONG(ushort* in, int aAdd)
	{
		ulong64* tmp = new ulong64[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (ulong64)(in[i] + aAdd);
		}
		return tmp;
	}
	ulong64* Image::ConvertPixelTypeULONG(uint* in, int aAdd)
	{
		ulong64* tmp = new ulong64[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (ulong64)(in[i] + aAdd);
		}
		return tmp;
	}
	ulong64* Image::ConvertPixelTypeULONG(float* in, int aAdd)
	{
		ulong64* tmp = new ulong64[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (ulong64)(in[i] + aAdd);
		}
		return tmp;
	}
	ulong64* Image::ConvertPixelTypeULONG(double* in, int aAdd)
	{
		ulong64* tmp = new ulong64[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (ulong64)(in[i] + aAdd);
		}
		return tmp;
	}
	ulong64* Image::ConvertPixelTypeULONG(char* in, int aAdd)
	{
		ulong64* tmp = new ulong64[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (ulong64)(in[i] + aAdd);
		}
		return tmp;
	}
	ulong64* Image::ConvertPixelTypeULONG(short* in, int aAdd)
	{
		ulong64* tmp = new ulong64[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (ulong64)(in[i] + aAdd);
		}
		return tmp;
	}
	ulong64* Image::ConvertPixelTypeULONG(int* in, int aAdd)
	{
		ulong64* tmp = new ulong64[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (ulong64)(in[i] + aAdd);
		}
		return tmp;
	}
	ulong64* Image::ConvertPixelTypeULONG(long* in, int aAdd)
	{
		ulong64* tmp = new ulong64[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (ulong64)(in[i] + aAdd);
		}
		return tmp;
	}
	
	float* Image::ConvertPixelTypeFLOAT(uchar* in, int aAdd)
	{
		float* tmp = new float[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (float)(in[i] + aAdd);
		}
		return tmp;
	}
	float* Image::ConvertPixelTypeFLOAT(ushort* in, int aAdd)
	{
		float* tmp = new float[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (float)(in[i] + aAdd);
		}
		return tmp;
	}
	float* Image::ConvertPixelTypeFLOAT(uint* in, int aAdd)
	{
		float* tmp = new float[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (float)(in[i] + aAdd);
		}
		return tmp;
	}
	float* Image::ConvertPixelTypeFLOAT(ulong64* in, int aAdd)
	{
		float* tmp = new float[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (float)(in[i] + aAdd);
		}
		return tmp;
	}
	float* Image::ConvertPixelTypeFLOAT(double* in, int aAdd)
	{
		float* tmp = new float[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (float)(in[i] + aAdd);
		}
		return tmp;
	}
	float* Image::ConvertPixelTypeFLOAT(char* in, int aAdd)
	{
		float* tmp = new float[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (float)(in[i] + aAdd);
		}
		return tmp;
	}
	float* Image::ConvertPixelTypeFLOAT(short* in, int aAdd)
	{
		float* tmp = new float[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (float)(in[i] + aAdd);
		}
		return tmp;
	}
	float* Image::ConvertPixelTypeFLOAT(int* in, int aAdd)
	{
		float* tmp = new float[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (float)(in[i] + aAdd);
		}
		return tmp;
	}
	float* Image::ConvertPixelTypeFLOAT(long* in, int aAdd)
	{
		float* tmp = new float[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (float)(in[i] + aAdd);
		}
		return tmp;
	}
	
	double* Image::ConvertPixelTypeDOUBLE(uchar* in, int aAdd)
	{
		double* tmp = new double[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (double)(in[i] + aAdd);
		}
		return tmp;
	}
	double* Image::ConvertPixelTypeDOUBLE(ushort* in, int aAdd)
	{
		double* tmp = new double[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (double)(in[i] + aAdd);
		}
		return tmp;
	}
	double* Image::ConvertPixelTypeDOUBLE(uint* in, int aAdd)
	{
		double* tmp = new double[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (double)(in[i] + aAdd);
		}
		return tmp;
	}
	double* Image::ConvertPixelTypeDOUBLE(ulong64* in, int aAdd)
	{
		double* tmp = new double[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (double)(in[i] + aAdd);
		}
		return tmp;
	}
	double* Image::ConvertPixelTypeDOUBLE(float* in, int aAdd)
	{
		double* tmp = new double[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (double)(in[i] + aAdd);
		}
		return tmp;
	}
	double* Image::ConvertPixelTypeDOUBLE(char* in, int aAdd)
	{
		double* tmp = new double[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (double)(in[i] + aAdd);
		}
		return tmp;
	}
	double* Image::ConvertPixelTypeDOUBLE(short* in, int aAdd)
	{
		double* tmp = new double[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (double)(in[i] + aAdd);
		}
		return tmp;
	}
	double* Image::ConvertPixelTypeDOUBLE(int* in, int aAdd)
	{
		double* tmp = new double[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (double)(in[i] + aAdd);
		}
		return tmp;
	}
	double* Image::ConvertPixelTypeDOUBLE(long* in, int aAdd)
	{
		double* tmp = new double[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (double)(in[i] + aAdd);
		}
		return tmp;
	}
	
	char* Image::ConvertPixelTypeCHAR(uchar* in, int aAdd)
	{
		char* tmp = new char[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (char)(in[i] + aAdd);
		}
		return tmp;
	}
	char* Image::ConvertPixelTypeCHAR(ushort* in, int aAdd)
	{
		char* tmp = new char[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (char)(in[i] + aAdd);
		}
		return tmp;
	}
	char* Image::ConvertPixelTypeCHAR(uint* in, int aAdd)
	{
		char* tmp = new char[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (char)(in[i] + aAdd);
		}
		return tmp;
	}
	char* Image::ConvertPixelTypeCHAR(ulong64* in, int aAdd)
	{
		char* tmp = new char[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (char)(in[i] + aAdd);
		}
		return tmp;
	}
	char* Image::ConvertPixelTypeCHAR(float* in, int aAdd)
	{
		char* tmp = new char[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (char)(in[i] + aAdd);
		}
		return tmp;
	}
	char* Image::ConvertPixelTypeCHAR(double* in, int aAdd)
	{
		char* tmp = new char[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (char)(in[i] + aAdd);
		}
		return tmp;
	}
	char* Image::ConvertPixelTypeCHAR(short* in, int aAdd)
	{
		char* tmp = new char[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (char)(in[i] + aAdd);
		}
		return tmp;
	}
	char* Image::ConvertPixelTypeCHAR(int* in, int aAdd)
	{
		char* tmp = new char[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (char)(in[i] + aAdd);
		}
		return tmp;
	}
	char* Image::ConvertPixelTypeCHAR(long* in, int aAdd)
	{
		char* tmp = new char[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (char)(in[i] + aAdd);
		}
		return tmp;
	}
	
	short* Image::ConvertPixelTypeSHORT(uchar* in, int aAdd)
	{
		short* tmp = new short[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (short)(in[i] + aAdd);
		}
		return tmp;
	}
	short* Image::ConvertPixelTypeSHORT(ushort* in, int aAdd)
	{
		short* tmp = new short[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (short)(in[i] + aAdd);
		}
		return tmp;
	}
	short* Image::ConvertPixelTypeSHORT(uint* in, int aAdd)
	{
		short* tmp = new short[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (short)(in[i] + aAdd);
		}
		return tmp;
	}
	short* Image::ConvertPixelTypeSHORT(ulong64* in, int aAdd)
	{
		short* tmp = new short[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (short)(in[i] + aAdd);
		}
		return tmp;
	}
	short* Image::ConvertPixelTypeSHORT(float* in, int aAdd)
	{
		short* tmp = new short[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (short)(in[i] + aAdd);
		}
		return tmp;
	}
	short* Image::ConvertPixelTypeSHORT(double* in, int aAdd)
	{
		short* tmp = new short[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (short)(in[i] + aAdd);
		}
		return tmp;
	}
	short* Image::ConvertPixelTypeSHORT(char* in, int aAdd)
	{
		short* tmp = new short[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (short)(in[i] + aAdd);
		}
		return tmp;
	}
	short* Image::ConvertPixelTypeSHORT(int* in, int aAdd)
	{
		short* tmp = new short[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (short)(in[i] + aAdd);
		}
		return tmp;
	}
	short* Image::ConvertPixelTypeSHORT(long* in, int aAdd)
	{
		short* tmp = new short[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (short)(in[i] + aAdd);
		}
		return tmp;
	}
	
	int* Image::ConvertPixelTypeINT(uchar* in, int aAdd)
	{
		int* tmp = new int[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (int)(in[i] + aAdd);
		}
		return tmp;
	}
	int* Image::ConvertPixelTypeINT(ushort* in, int aAdd)
	{
		int* tmp = new int[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (int)(in[i] + aAdd);
		}
		return tmp;
	}
	int* Image::ConvertPixelTypeINT(uint* in, int aAdd)
	{
		int* tmp = new int[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (int)(in[i] + aAdd);
		}
		return tmp;
	}
	int* Image::ConvertPixelTypeINT(ulong64* in, int aAdd)
	{
		int* tmp = new int[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (int)(in[i] + aAdd);
		}
		return tmp;
	}
	int* Image::ConvertPixelTypeINT(float* in, int aAdd)
	{
		int* tmp = new int[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (int)(in[i] + aAdd);
		}
		return tmp;
	}
	int* Image::ConvertPixelTypeINT(double* in, int aAdd)
	{
		int* tmp = new int[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (int)(in[i] + aAdd);
		}
		return tmp;
	}
	int* Image::ConvertPixelTypeINT(char* in, int aAdd)
	{
		int* tmp = new int[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (int)(in[i] + aAdd);
		}
		return tmp;
	}
	int* Image::ConvertPixelTypeINT(short* in, int aAdd)
	{
		int* tmp = new int[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (int)(in[i] + aAdd);
		}
		return tmp;
	}
	int* Image::ConvertPixelTypeINT(long* in, int aAdd)
	{
		int* tmp = new int[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (int)(in[i] + aAdd);
		}
		return tmp;
	}
	
	long* Image::ConvertPixelTypeLONG(uchar* in, int aAdd)
	{
		long* tmp = new long[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (long)(in[i] + aAdd);
		}
		return tmp;
	}
	long* Image::ConvertPixelTypeLONG(ushort* in, int aAdd)
	{
		long* tmp = new long[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (long)(in[i] + aAdd);
		}
		return tmp;
	}
	long* Image::ConvertPixelTypeLONG(uint* in, int aAdd)
	{
		long* tmp = new long[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (long)(in[i] + aAdd);
		}
		return tmp;
	}
	long* Image::ConvertPixelTypeLONG(ulong64* in, int aAdd)
	{
		long* tmp = new long[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (long)(in[i] + aAdd);
		}
		return tmp;
	}
	long* Image::ConvertPixelTypeLONG(float* in, int aAdd)
	{
		long* tmp = new long[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (long)(in[i] + aAdd);
		}
		return tmp;
	}
	long* Image::ConvertPixelTypeLONG(double* in, int aAdd)
	{
		long* tmp = new long[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (long)(in[i] + aAdd);
		}
		return tmp;
	}
	long* Image::ConvertPixelTypeLONG(char* in, int aAdd)
	{
		long* tmp = new long[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (long)(in[i] + aAdd);
		}
		return tmp;
	}
	long* Image::ConvertPixelTypeLONG(short* in, int aAdd)
	{
		long* tmp = new long[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (long)(in[i] + aAdd);
		}
		return tmp;
	}
	long* Image::ConvertPixelTypeLONG(int* in, int aAdd)
	{
		long* tmp = new long[DimX * DimY * DimZ];
		for (int i = 0; i < DimX * DimY * DimZ; i++)
		{
			tmp[i] = (long)(in[i] + aAdd);
		}
		return tmp;
	}