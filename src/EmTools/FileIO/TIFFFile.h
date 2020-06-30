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


#ifndef TIFFIMAGE_H
#define TIFFIMAGE_H

#include "../Basics/Default.h"
#include "EmHeader.h"
#include "FileReader.h"
#include "FileWriter.h"


struct TiffFileHeader
{
	ushort BytaOrder;
	ushort ID;
	uint OffsetToIFD;
};

enum TiffType_enum : ushort
{
	TIFFT_BYTE = 1,
	TIFFT_ASCII = 2,
	TIFFT_SHORT = 3,
	TIFFT_LONG = 4,
	TIFFT_RATIONAL = 5,
	TIFFT_SBYTE = 6,
	TIFFT_UNDEFINED = 7,
	TIFFT_SSHORT = 8,
	TIFFT_SLONG = 9,
	TIFFT_SRATIONAL = 10,
	TIFFT_FLOAT = 11,
	TIFFT_DOUBLE = 12
};

struct TiffTag
{
	ushort TagID;
	TiffType_enum Type;
	uint Count;
	union
	{
		char CharVal;
		uchar UCharVal;
		short ShortVal;
		ushort UShortVal;
		int IntVal;
		uint UIntVal;
	} Offset;
};

size_t GetTiffTypeSizeInBytes(TiffType_enum aType);

class Rational
{
public:
	uint nominator;
	uint denominator;

	Rational(uint aNominator, uint aDenominator);
	Rational(uint aValues[2]);
	Rational();

	double GetValue();
};

class SRational
{
public:
	int nominator;
	int denominator;

	SRational(int aNominator, int aDenominator);
	SRational(int aValues[2]);
	SRational();

	double GetValue();
};

//!  TIFFFile represents a *.Em file in memory and maps contained information to the default internal Image format. 
/*!
TIFFFile gives access to header infos, volume data and single projections.
\author Michael Kunz
\date   September 2011
\version 1.0
*/
class ImageFileDirectory;

enum SampleFormatEnum : short
{
	SAMPLEFORMAT_UINT = 1,
	SAMPLEFORMAT_INT = 2,
	SAMPLEFORMAT_IEEEFP = 3,
	SAMPLEFORMAT_VOID = 4,
	SAMPLEFORMAT_COMPLEXINT = 5,
	SAMPLEFORMAT_COMPLEXIEEEFP = 6
};

class TIFFFile : public FileReader, public FileWriter
{
private:
	TiffFileHeader _fileHeader;
	uint _dataStartPosition;
	void* _data;
	std::vector<ImageFileDirectory*> _imageFileDirectories;
	uint _width, _height, _bitsPerSample, _samplesPerPixel;
	float _pixelSize;
	float _magnification;
	bool _needsFlipOnY;
	bool _isPlanar;
	SampleFormatEnum _sampleFormat;
	DataType_enum GetDataTypeUnsigned();
	DataType_enum GetDataTypeSigned();
	DataType_enum GetDataTypeFloat();
	DataType_enum GetDataTypeComplex();
	DataType_enum GetDataTypeComplexFloat();

public:
	//! Creates a new EmFile instance. The file name is only set internally; the file itself keeps untouched.
	TIFFFile(string aFileName);

	~TIFFFile();

	//! Opens the file File#mFileName and reads the entire content.
	/*!
	\throw FileIOException
	*/
	bool OpenAndRead();

	//! Opens the file File#mFileName and reads only the file header.
	/*!
	\throw FileIOException
	*/
	bool OpenAndReadHeader();

	//! Opens the file File#mFileName and writes the entire content.
	/*!
	\throw FileIOException
	*/
	static bool WriteTIFF(string aFileName, int aDimX, int aDimY, float aPixelSize, DataType_enum aDatatype, void* aData);

	//! Determines if a given image dimension and datatype can be written to a TIFF file
	static bool CanWriteAsTIFF(int aDimX, int aDimY, DataType_enum aDatatype);

	//! Converts from Em data type enum to internal data type
	/*!
	EmFile::GetDataType dows not take into account if the data type is unsigned or signed as the
	Em file format cannot distinguish them.
	*/
	DataType_enum GetDataType();

	//! Returns the size of the data block. If the header is not yet read, it will return 0.
	size_t GetDataSize();

	//! Returns a reference to the inner Em file header.
	TiffFileHeader& GetFileHeader();

	//! Returns the first image plane.
	void* GetData();

	uint GetWidth();
	uint GetHeight();
	uint BitsPerSample();
	uint SamplesPerPixel();

	bool NeedsFlipOnYAxis();
	bool GetIsPlanar();

	float GetPixelSize();
};

class ImageFileDirectoryEntry
{
protected:
	TIFFFile* mTifffile;
	TiffTag mTag;

	ImageFileDirectoryEntry(TIFFFile* aFile, ushort aTagID);

	void EndianSwap(uint& x);
public:
	ImageFileDirectoryEntry(TIFFFile* aFile);
	static ImageFileDirectoryEntry* CreateFileDirectoryEntry(TIFFFile* aFile);

	friend class ImageFileDirectory;
};

class StringImageFileDirectory : public ImageFileDirectoryEntry
{
protected: 
	std::string mValue;
public:
	StringImageFileDirectory(TIFFFile* aFile, ushort aTagID);
};

class ByteImageFileDirectory : public ImageFileDirectoryEntry
{
protected:
	uchar* mValue;
public:
	ByteImageFileDirectory(TIFFFile* aFile, ushort aTagID);
	~ByteImageFileDirectory();
};

class SByteImageFileDirectory : public ImageFileDirectoryEntry
{
protected:
	char* mValue;
public:
	SByteImageFileDirectory(TIFFFile* aFile, ushort aTagID);
	~SByteImageFileDirectory();
};

class UShortImageFileDirectory : public ImageFileDirectoryEntry
{
protected:
	ushort* mValue;
public:
	UShortImageFileDirectory(TIFFFile* aFile, ushort aTagID);
	~UShortImageFileDirectory();
};

class ShortImageFileDirectory : public ImageFileDirectoryEntry
{
protected:
	short* mValue;
public:
	ShortImageFileDirectory(TIFFFile* aFile, ushort aTagID);
	~ShortImageFileDirectory();
};

class UIntImageFileDirectory : public ImageFileDirectoryEntry
{
protected:
	uint* mValue;
public:
	UIntImageFileDirectory(TIFFFile* aFile, ushort aTagID);
	~UIntImageFileDirectory();
};

class IntImageFileDirectory : public ImageFileDirectoryEntry
{
protected:
	int* mValue;
public:
	IntImageFileDirectory(TIFFFile* aFile, ushort aTagID);
	~IntImageFileDirectory();
};

class RationalImageFileDirectory : public ImageFileDirectoryEntry
{
protected:
	Rational* mValue;
public:
	RationalImageFileDirectory(TIFFFile* aFile, ushort aTagID);
	~RationalImageFileDirectory();
};

class SRationalImageFileDirectory : public ImageFileDirectoryEntry
{
protected:
	SRational* mValue;
public:
	SRationalImageFileDirectory(TIFFFile* aFile, ushort aTagID);
	~SRationalImageFileDirectory();
};

class FloatImageFileDirectory : public ImageFileDirectoryEntry
{
protected:
	float* mValue;
public:
	FloatImageFileDirectory(TIFFFile* aFile, ushort aTagID);
	~FloatImageFileDirectory();
};

class DoubleImageFileDirectory : public ImageFileDirectoryEntry
{
protected:
	double* mValue;
public:
	DoubleImageFileDirectory(TIFFFile* aFile, ushort aTagID);
	~DoubleImageFileDirectory();
};

class ImageFileDirectory
{
private:
	TIFFFile* mTifffile;
	ushort mEntryCount;
	std::vector<ImageFileDirectoryEntry*> mEntries;

public:
	ImageFileDirectory(TIFFFile* aFile);
	~ImageFileDirectory();

	ImageFileDirectoryEntry* GetEntry(ushort tagID);
};

class IFDImageLength : public ImageFileDirectoryEntry
{
	uint mValue;

public:
	static const ushort TagID = 257;
	static const std::string TagName;
	
	IFDImageLength(TIFFFile* aFile, ushort aTagID);

	uint Value();
};

class IFDImageWidth : public ImageFileDirectoryEntry
{
	uint mValue;

public:
	static const ushort TagID = 256;
	static const string TagName;

	IFDImageWidth(TIFFFile* aFile, ushort aTagID);

	uint Value();
};

class IFDRowsPerStrip : public ImageFileDirectoryEntry
{
	uint mValue;

public:
	static const ushort TagID = 278;
	static const string TagName;

	IFDRowsPerStrip(TIFFFile* aFile, ushort aTagID);

	uint Value();
};

class IFDStripByteCounts : public ImageFileDirectoryEntry
{
	uint* mValue;

public:
	static const ushort TagID = 279;
	static const string TagName;

	IFDStripByteCounts(TIFFFile* aFile, ushort aTagID);
	~IFDStripByteCounts();

	uint* Value();
	size_t ValueCount();
};

class IFDStripOffsets : public ImageFileDirectoryEntry
{
	uint* mValue;

public:
	static const ushort TagID = 273;
	static const string TagName;

	IFDStripOffsets(TIFFFile* aFile, ushort aTagID);
	~IFDStripOffsets();

	uint* Value();
	size_t ValueCount();
};

class IFDArtist : public StringImageFileDirectory
{
public:
	static const ushort TagID = 315;
	static const string TagName;

	IFDArtist(TIFFFile* aFile, ushort aTagID);
	
	std::string Value();
};

class IFDCopyright : public StringImageFileDirectory
{
public:
	static const ushort TagID = 33432;
	static const string TagName;

	IFDCopyright(TIFFFile* aFile, ushort aTagID);
	
	std::string Value();
};

class IFDDateTime : public StringImageFileDirectory
{
public:
	static const ushort TagID = 306;
	static const string TagName;

	IFDDateTime(TIFFFile* aFile, ushort aTagID);
	
	std::string Value();
};

class IFDHostComputer : public StringImageFileDirectory
{
public:
	static const ushort TagID = 316;
	static const string TagName;

	IFDHostComputer(TIFFFile* aFile, ushort aTagID);
	
	std::string Value();
};

class IFDImageDescription : public StringImageFileDirectory
{
public:
	static const ushort TagID = 270;
	static const string TagName;

	IFDImageDescription(TIFFFile* aFile, ushort aTagID);
	
	std::string Value();
};

class IFDModel : public StringImageFileDirectory
{
public:
	static const ushort TagID = 272;
	static const string TagName;

	IFDModel(TIFFFile* aFile, ushort aTagID);
	
	std::string Value();
};

class IFDMake : public StringImageFileDirectory
{
public:
	static const ushort TagID = 271;
	static const string TagName;

	IFDMake(TIFFFile* aFile, ushort aTagID);
	
	std::string Value();
};

class IFDSoftware : public StringImageFileDirectory
{
public:
	static const ushort TagID = 305;
	static const string TagName;

	IFDSoftware(TIFFFile* aFile, ushort aTagID);
	
	std::string Value();
};

class IFDBitsPerSample : public UShortImageFileDirectory
{
public:
	static const ushort TagID = 258;
	static const string TagName;

	IFDBitsPerSample(TIFFFile* aFile, ushort aTagID);

	ushort Value(size_t aIdx);
};

class IFDCellLength : public UShortImageFileDirectory
{
public:
	static const ushort TagID = 265;
	static const string TagName;

	IFDCellLength(TIFFFile* aFile, ushort aTagID);

	ushort Value();
};

class IFDCellWidth : public UShortImageFileDirectory
{
public:
	static const ushort TagID = 264;
	static const string TagName;

	IFDCellWidth(TIFFFile* aFile, ushort aTagID);

	ushort Value();
};

class IFDColorMap : public UShortImageFileDirectory
{
public:
	static const ushort TagID = 320;
	static const std::string TagName;

	IFDColorMap(TIFFFile* aFile, ushort aTagID);

	ushort Value();
};

class IFDCompression : public UShortImageFileDirectory
{
public:
	enum TIFFCompression : ushort
	{
		NoCompression = 1,
		CCITTGroup3 = 2,
		PackBits = 32773
	};

	static const ushort TagID = 259;
	static const std::string TagName;

	IFDCompression(TIFFFile* aFile, ushort aTagID);

	TIFFCompression Value();
};

class IFDExtraSamples : public UShortImageFileDirectory
{
public:
	static const ushort TagID = 338;
	static const std::string TagName;

	IFDExtraSamples(TIFFFile* aFile, ushort aTagID);

	ushort Value();
};

class IFDFillOrder : public UShortImageFileDirectory
{
public:
	static const ushort TagID = 226;
	static const std::string TagName;

	IFDFillOrder(TIFFFile* aFile, ushort aTagID);

	ushort Value();
};

class IFDFreeByteCounts : public UIntImageFileDirectory
{
public:
	static const ushort TagID = 289;
	static const std::string TagName;

	IFDFreeByteCounts(TIFFFile* aFile, ushort aTagID);

	uint Value();
};

class IFDFreeOffsets : public UIntImageFileDirectory
{
public:
	static const ushort TagID = 288;
	static const std::string TagName;

	IFDFreeOffsets(TIFFFile* aFile, ushort aTagID);

	uint Value();
};

class IFDGrayResponseCurve : public UShortImageFileDirectory
{
public:
	static const ushort TagID = 291;
	static const std::string TagName;

	IFDGrayResponseCurve(TIFFFile* aFile, ushort aTagID);

	ushort* Value();
	size_t ValueCount();
};

class IFDGrayResponseUnit : public UShortImageFileDirectory
{
public:
	static const ushort TagID = 290;
	static const std::string TagName;

	IFDGrayResponseUnit(TIFFFile* aFile, ushort aTagID);

	ushort Value();
};

class IFDMaxSampleValue : public UShortImageFileDirectory
{
public:
	static const ushort TagID = 281;
	static const std::string TagName;

	IFDMaxSampleValue(TIFFFile* aFile, ushort aTagID);

	ushort Value();
};

class IFDMinSampleValue : public UShortImageFileDirectory
{
public:
	static const ushort TagID = 280;
	static const std::string TagName;

	IFDMinSampleValue(TIFFFile* aFile, ushort aTagID);

	ushort Value();
};

class IFDNewSubfileType : public UIntImageFileDirectory
{
public:
	static const ushort TagID = 254;
	static const std::string TagName;

	IFDNewSubfileType(TIFFFile* aFile, ushort aTagID);

	uint Value();
};

class IFDOrientation : public UShortImageFileDirectory
{
public:
	enum TiffOrientation
	{
		TOPLEFT = 1,
		TOPRIGHT = 2,
		BOTRIGHT = 3,
		BOTLEFT = 4,
		LEFTTOP = 5,
		RIGHTTOP = 6,
		RIGHTBOT = 7,
		LEFTBOT = 8
	};

	static const ushort TagID = 274;
	static const std::string TagName;

	IFDOrientation(TIFFFile* aFile, ushort aTagID);

	TiffOrientation Value();
};

class IFDPhotometricInterpretation : public UShortImageFileDirectory
{
public:
	enum TIFFPhotometricInterpretation : ushort
	{
		WhiteIsZero = 0,
		BlackIsZero = 1,
		RGB = 2,
		Palette = 3,
		TransparencyMask = 4
	};

	static const ushort TagID = 262;
	static const std::string TagName;

	IFDPhotometricInterpretation(TIFFFile* aFile, ushort aTagID);

	TIFFPhotometricInterpretation Value();
};

class IFDPlanarConfiguration : public UShortImageFileDirectory
{
public:
	enum TIFFPlanarConfigurartion : ushort
	{
		Chunky = 1,
		Planar = 2
	};

	static const ushort TagID = 284;
	static const std::string TagName;

	IFDPlanarConfiguration(TIFFFile* aFile, ushort aTagID);

	TIFFPlanarConfigurartion Value();
};

class IFDResolutionUnit : public UShortImageFileDirectory
{
public:
	enum TIFFResolutionUnit
	{
		None = 1,
		Inch = 2,
		Centimeter = 3
	};

	static const ushort TagID = 296;
	static const std::string TagName;

	IFDResolutionUnit(TIFFFile* aFile, ushort aTagID);

	TIFFResolutionUnit Value();
};

class IFDSamplesPerPixel : public UShortImageFileDirectory
{
public:
	static const ushort TagID = 277;
	static const std::string TagName;

	IFDSamplesPerPixel(TIFFFile* aFile, ushort aTagID);

	ushort Value();
};

class IFDSampleFormat : public ShortImageFileDirectory
{
public:

	static const ushort TagID = 339;
	static const std::string TagName;

	IFDSampleFormat(TIFFFile* aFile, ushort aTagID);

	SampleFormatEnum Value();
};

class IFDSubfileType : public UShortImageFileDirectory
{
public:
	static const ushort TagID = 255;
	static const std::string TagName;

	IFDSubfileType(TIFFFile* aFile, ushort aTagID);

	ushort Value();
};

class IFDThreshholding : public UShortImageFileDirectory
{
public:
	static const ushort TagID = 263;
	static const std::string TagName;

	IFDThreshholding(TIFFFile* aFile, ushort aTagID);

	ushort Value();
};

class IFDXResolution : public RationalImageFileDirectory
{
public:
	static const ushort TagID = 282;
	static const std::string TagName;

	IFDXResolution(TIFFFile* aFile, ushort aTagID);

	Rational Value();
};

class IFDYResolution : public RationalImageFileDirectory
{
public:
	static const ushort TagID = 283;
	static const std::string TagName;

	IFDYResolution(TIFFFile* aFile, ushort aTagID);

	Rational Value();
};

class IFDGatan65006 : public DoubleImageFileDirectory
{
public:
	static const ushort TagID = 65006;
	static const std::string TagName;

	IFDGatan65006(TIFFFile* aFile, ushort aTagID);

	double Value();
};

class IFDGatan65007 : public DoubleImageFileDirectory
{
public:
	static const ushort TagID = 65007;
	static const std::string TagName;

	IFDGatan65007(TIFFFile* aFile, ushort aTagID);

	double Value();
};

class IFDGatan65009 : public DoubleImageFileDirectory
{
public:
	static const ushort TagID = 65009;
	static const std::string TagName;

	IFDGatan65009(TIFFFile* aFile, ushort aTagID);

	double Value();
};

class IFDGatan65010 : public DoubleImageFileDirectory
{
public:
	static const ushort TagID = 65010;
	static const std::string TagName;

	IFDGatan65010(TIFFFile* aFile, ushort aTagID);

	double Value();
};

class IFDGatan65015 : public IntImageFileDirectory
{
public:
	static const ushort TagID = 65015;
	static const std::string TagName;

	IFDGatan65015(TIFFFile* aFile, ushort aTagID);

	int Value();
};

class IFDGatan65016 : public IntImageFileDirectory
{
public:
	static const ushort TagID = 65016;
	static const std::string TagName;

	IFDGatan65016(TIFFFile* aFile, ushort aTagID);

	int Value();
};

class IFDGatan65024 : public DoubleImageFileDirectory
{
public:
	static const ushort TagID = 65024;
	static const std::string TagName;

	IFDGatan65024(TIFFFile* aFile, ushort aTagID);

	double Value();
};

class IFDGatan65025 : public DoubleImageFileDirectory
{
public:
	static const ushort TagID = 65025;
	static const std::string TagName;

	IFDGatan65025(TIFFFile* aFile, ushort aTagID);

	double Value();
};

class IFDGatan65026 : public IntImageFileDirectory
{
public:
	static const ushort TagID = 65026;
	static const std::string TagName;

	IFDGatan65026(TIFFFile* aFile, ushort aTagID);

	int Value();
};

class IFDGatan65027 : public ByteImageFileDirectory
{
public:
	static const ushort TagID = 65027;
	static const std::string TagName;

	IFDGatan65027(TIFFFile* aFile, ushort aTagID);

	uchar* Value();
	uint ValueCount();
};

class IFDLSM34412 : public ByteImageFileDirectory
{

public:
#ifdef _USE_WINDOWS_COMPILER_SETTINGS
#pragma pack(push)
#pragma pack(1)
#endif
	struct IFDInfo
	{
		int MagicNumber;// = hex(int(_temp[0]))
		int StructureSize;// = int(_temp[1])
		int DimensionX;// = int(_temp[2])
		int DimensionY;// = int(_temp[3])
		int DimensionZ;// = int(_temp[4])
		int Unknown1[3];
		int ThumbnailX;// = int(_temp[8])
		int ThumbnailY;// = int(_temp[9])
		double VoxelSizeX;// = _temp[0]
		double VoxelSizeY;// = _temp[1]
		double VoxelSizeZ;// = _temp[2]
		int ScanType;// = int(_temp[0])
		int DataType;// = int(_temp[1])
		int OffsetVectorOverlay;// = int(_temp[2])
		int OffsetInputLut;// = int(_temp[3])
		int OffsetOutputLut;// = int(_temp[4])
		int OffsetChannelColor;// = int(_temp[5])
		double TimeInterval;// = int(_temp[0]) ## float 64 <--a changer
		int OffsetChannelDataType;// = int(_temp[0])
		int OffsetScanInformation;// = int(_temp[1])
		int OffsetKsData;// = int(_temp[2])
		int OffsetTimeStamps;// = int(_temp[3])
		int OffsetEventList;// = int(_temp[4])
		int OffsetROI;// = int(_temp[5])
		int OffsetBleachROI;// = int(_temp[6])
		int OffsetNextRecording;// = int(_temp[7])
		int Reserved;// = int(_temp[8])
	}
#ifdef _USE_LINUX_COMPILER_SETTINGS
	__attribute__((packed)) struct_IFDInfo_
#endif
		;
#ifdef _USE_WINDOWS_COMPILER_SETTINGS
#pragma pack(pop)
#endif
	static const ushort TagID = 34412;
	static const std::string TagName;

	IFDLSM34412(TIFFFile* aFile, ushort aTagID);

	IFDInfo* Value();
	uint ValueCount();
};

#endif