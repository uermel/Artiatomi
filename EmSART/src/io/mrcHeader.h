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


#ifndef MRCHEADER_H
#define MRCHEADER_H
#include "IODefault.h"

//! Types of pixel in image
enum MrcMode_Enum : uint
{
	//! Image unsigned bytes
	MRCMODE_UI1  = 0,
	//! Image signed short integer
	MRCMODE_I2   = 1,
	//! Image float
	MRCMODE_F    = 2,
	//! Complex short
	MRCMODE_CUI2 = 3,
	//! Complex float
	MRCMODE_CF   = 4,
	//! Image unsigned shorts
	MRCMODE_UI2 = 6,
	//! half float
	MRCMODE_HALF   = 64
};

//! X, Y or Z Axis
enum MrcAxis_Enum : uint
{
	//! X axis
	MRCAXIS_X = 1,
	//! Y axis
	MRCAXIS_Y = 2,
	//! Z axis
	MRCAXIS_Z = 3
};

//! Type
enum MrcIDType_Enum : ushort
{
	//! Mono
	MRCIDTYPE_MONO = 0,
	//! Tilt
	MRCIDTYPE_TILT = 1,
	//! Tilits
	MRCIDTYPE_TILTS = 2,
	//! Lina
	MRCIDTYPE_LINA = 3,
	//! Lins
	MRCIDTYPE_LINS = 4
};

//! Header of a mrc file (1024 bytes)
struct struct_MrcHeader
{
	//! number of Columns    (fastest changing in map)
	uint NX;
	//! number of Rows
	uint NY;
	//! number of Sections   (slowest changing in map)
	uint NZ;

	//! Types of pixel in image
	MrcMode_Enum MODE;

	//! Number of first COLUMN  in map (Default = 0)
	uint NXSTART;
	//! Number of first ROW  in map (Default = 0)
	uint NYSTART;
	//! Number of first SECTION  in map (Default = 0)
	uint NZSTART;
	//! Number of intervals along X
	uint MX;
	//! Number of intervals along Y
	uint MY;
	//! Number of intervals along Z
	uint MZ;

	//! Cell Dimensions (Angstroms)
	float Xlen;
	//! Cell Dimensions (Angstroms)
	float Ylen;
	//! Cell Dimensions (Angstroms)
	float Zlen;

	//! Cell Angles (Degrees)
	float ALPHA;
	//! Cell Angles (Degrees)
	float BETA;
	//! Cell Angles (Degrees)
	float GAMMA;

	//! Which axis corresponds to Columns  (1,2,3 for X,Y,Z)
	MrcAxis_Enum MAPC;
	//! Which axis corresponds to Rows  (1,2,3 for X,Y,Z)
	MrcAxis_Enum MAPR;
	//! Which axis corresponds to Sections  (1,2,3 for X,Y,Z)
	MrcAxis_Enum MAPS;

	//! Minimum density value
	float AMIN;
	//! Maximum density value
	float AMAX;
	//! Mean density value (Average)
	float AMEAN;

	//! Space group number       (0 for images)
	ushort ISPG;
	//! Number of bytes used for storing symmetry operators
	ushort NSYMBT;

	//! Number of bytes in extended header
	uint NEXT;

	//! Creator ID
	ushort CREATEID;
	//! Not used. All set to zero by default
	char EXTRA[30];

	//! Number of integer per section
	ushort NINT;
	//! Number of reals per section
	ushort NREAL;
	//! Not used. All set to zero by default
	char EXTRA2[20];

	//1146047817 indicates that file was created by IMOD or other software that uses bit flags in the following field
    uint imodStamp;

    //Bit flags:
    //1 = bytes are stored as signed
    //2 = pixel spacing was set from size in extended header 
    //4 = origin is stored with sign inverted from definition below
    uint imodFlags;

	//! 0=mono, 1=tilt, 2=tilts, 3=lina, 4=lins
	MrcIDType_Enum IDTYPE;
	//!
	ushort LENS;
	//!
	ushort ND1;
	//!
	ushort ND2;
	//!
	ushort VD1;
	//!
	ushort VD2;

	//!
	float TILTANGLES[6];
	//! X origin
	float XORIGIN;
	//! Y origin
	float YORIGIN;
	//! Z origin
	float ZORIGIN;
	//! Contains "MAP "
	char CMAP[4];
	//!
	char STAMP[4];
	//!
	float RMS;
	//! Number of labels being used
	uint NLABL;
	//! 10 labels of 80 character
	char LABELS[10][80];
};
typedef struct struct_MrcHeader MrcHeader;

/*!
	Extended Header (FEI format and IMOD format)
	The extended header contains the information about a maximum of 1024 images.
	Each section is 128 bytes long. The extended header is thus 1024 * 128 bytes
	(always the same length, regardless of how many images are present
*/
//! Extended mrc file header
struct struct_MrcExtendedHeader
{
	//! Alpha tilt (deg)
	float a_tilt;
	//! Beta tilt (deg)
	float b_tilt;
	//! Stage x position (Unit=m. But if value>1, unit=???m)
	float x_stage;
	//! y_stage  Stage y position (Unit=m. But if value>1, unit=???m)
	float y_stage;
	//! z_stage  Stage z position (Unit=m. But if value>1, unit=???m)
	float z_stage;
	//! x_shift  Image shift x (Unit=m. But if value>1, unit=???m)
	float x_shift;
	//! y_shift  Image shift y (Unit=m. But if value>1, unit=???m)
	float y_shift;
	//! Image shift z (Unit=m. But if value>1, unit=???m)
	//float z_shift;
	//! Defocus Unit=m. But if value>1, unit=???m)
	float defocus;
	//! Exposure time (s)
	float exp_time;
	//! Mean value of image
	float mean_int;
	//! Tilt axis (deg)
	float tilt_axis;
	//! Pixel size of image (m)
	float pixel_size;
	//! Magnification used
	float magnification;
	//! Value of the high tension in SI units (volts)
    float ht;
	//! The binning of the CCD or STEM acquisition
    float binning;
	//! The intended application defocus in SI units (meters), as defined for example in the tomography parameters view.
    float appliedDefocus;
	//! Not used (filling up to 128 bytes)
	float remainder[16];
};
typedef struct struct_MrcExtendedHeader MrcExtendedHeader;

#endif
