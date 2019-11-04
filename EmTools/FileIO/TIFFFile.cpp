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


#include "TIFFFile.h"
#include "Dm3FileTagDirectory.h"
#include <streambuf>
#include <iostream>
#include <istream>
#include <string>
#include <cstring>
#include "Dm3File.h"
#include "Dm4File.h"
#include <math.h>

size_t GetTiffTypeSizeInBytes(TiffType_enum aType)
{
	switch (aType)
	{
	case TIFFT_BYTE:
		return 1;
	case TIFFT_ASCII:
		return 1;
	case TIFFT_SHORT:
		return 2;
	case TIFFT_LONG:
		return 4;
	case TIFFT_RATIONAL:
		return 8;
	case TIFFT_SBYTE:
		return 1;
	case TIFFT_UNDEFINED:
		return 1;
	case TIFFT_SSHORT:
		return 2;
	case TIFFT_SLONG:
		return 4;
	case TIFFT_SRATIONAL:
		return 8;
	case TIFFT_FLOAT:
		return 4;
	case TIFFT_DOUBLE:
		return 8;
	default:
		return 0;
	};
}

Rational::Rational(uint aNominator, uint aDenominator)
	: nominator(aNominator), denominator(aDenominator)
{
}

Rational::Rational(uint aValues[2])
	: nominator(aValues[0]), denominator(aValues[1])
{
}

Rational::Rational()
	: nominator(0), denominator(1)
{
}

double Rational::GetValue()
{
	return nominator / (double)denominator;
}

SRational::SRational(int aNominator, int aDenominator)
	: nominator(aNominator), denominator(aDenominator)
{
}

SRational::SRational(int aValues[2])
	: nominator(aValues[0]), denominator(aValues[1])
{
}

SRational::SRational()
	: nominator(0), denominator(1)
{
}

double SRational::GetValue()
{
	return nominator / (double)denominator;
}

ImageFileDirectoryEntry::ImageFileDirectoryEntry(TIFFFile * aFile, ushort aTagID)
	: mTifffile(aFile), mTag()
{
	mTag.TagID = aTagID;
	mTag.Type = (TiffType_enum)mTifffile->ReadUI2LE();
	mTag.Count = mTifffile->ReadUI4LE();
	mTag.Offset.UIntVal = mTifffile->ReadUI4LE();
}

void ImageFileDirectoryEntry::EndianSwap(uint & x)
{
	x =
		(x >> 24) |
		((x << 8) & 0x00FF0000) |
		((x >> 8) & 0x0000FF00) |
		(x << 24);
}

ImageFileDirectoryEntry::ImageFileDirectoryEntry(TIFFFile * aFile)
	: mTifffile(aFile), mTag()
{
	mTifffile->Read((char*)&mTag, sizeof(mTag));
}

ImageFileDirectoryEntry* ImageFileDirectoryEntry::CreateFileDirectoryEntry(TIFFFile * aFile)
{
	ushort tagID = aFile->ReadUI2LE();
	switch (tagID)
	{
	case IFDArtist::TagID:
		return new IFDArtist(aFile, tagID);
	case IFDBitsPerSample::TagID:
		return new IFDBitsPerSample(aFile, tagID);
	case IFDCellLength::TagID:
		return new IFDCellLength(aFile, tagID);
	case IFDCellWidth::TagID:
		return new IFDCellWidth(aFile, tagID);
	case IFDColorMap::TagID:
		return new IFDColorMap(aFile, tagID);
	case IFDCompression::TagID:
		return new IFDCompression(aFile, tagID);
	case IFDCopyright::TagID:
		return new IFDCopyright(aFile, tagID);
	case IFDDateTime::TagID:
		return new IFDDateTime(aFile, tagID);
	case IFDExtraSamples::TagID:
		return new IFDExtraSamples(aFile, tagID);
	case IFDFillOrder::TagID:
		return new IFDFillOrder(aFile, tagID);
	case IFDFreeByteCounts::TagID:
		return new IFDFreeByteCounts(aFile, tagID);
	case IFDFreeOffsets::TagID:
		return new IFDFreeOffsets(aFile, tagID);
	case IFDGrayResponseCurve::TagID:
		return new IFDGrayResponseCurve(aFile, tagID);
	case IFDGrayResponseUnit::TagID:
		return new IFDGrayResponseUnit(aFile, tagID);
	case IFDHostComputer::TagID:
		return new IFDHostComputer(aFile, tagID);
	case IFDImageDescription::TagID:
		return new IFDImageDescription(aFile, tagID);
	case IFDImageLength::TagID:
		return new IFDImageLength(aFile, tagID);
	case IFDImageWidth::TagID:
		return new IFDImageWidth(aFile, tagID);
	case IFDMake::TagID:
		return new IFDMake(aFile, tagID);
	case IFDMaxSampleValue::TagID:
		return new IFDMaxSampleValue(aFile, tagID);
	case IFDMinSampleValue::TagID:
		return new IFDMinSampleValue(aFile, tagID);
	case IFDModel::TagID:
		return new IFDModel(aFile, tagID);
	case IFDNewSubfileType::TagID:
		return new IFDNewSubfileType(aFile, tagID);
	case IFDOrientation::TagID:
		return new IFDOrientation(aFile, tagID);
	case IFDPhotometricInterpretation::TagID:
		return new IFDPhotometricInterpretation(aFile, tagID);
	case IFDPlanarConfiguration::TagID:
		return new IFDPlanarConfiguration(aFile, tagID);
	case IFDResolutionUnit::TagID:
		return new IFDResolutionUnit(aFile, tagID);
	case IFDRowsPerStrip::TagID:
		return new IFDRowsPerStrip(aFile, tagID);
	case IFDSamplesPerPixel::TagID:
		return new IFDSamplesPerPixel(aFile, tagID);
	case IFDSampleFormat::TagID:
		return new IFDSampleFormat(aFile, tagID);
	case IFDSoftware::TagID:
		return new IFDSoftware(aFile, tagID);
	case IFDStripByteCounts::TagID:
		return new IFDStripByteCounts(aFile, tagID);
	case IFDStripOffsets::TagID:
		return new IFDStripOffsets(aFile, tagID);
	case IFDSubfileType::TagID:
		return new IFDSubfileType(aFile, tagID);
	case IFDThreshholding::TagID:
		return new IFDThreshholding(aFile, tagID);
	case IFDXResolution::TagID:
		return new IFDXResolution(aFile, tagID);
	case IFDYResolution::TagID:
		return new IFDYResolution(aFile, tagID);
	case IFDGatan65006::TagID:
		return new IFDGatan65006(aFile, tagID);
	case IFDGatan65007::TagID:
		return new IFDGatan65007(aFile, tagID);
	case IFDGatan65009::TagID:
		return new IFDGatan65009(aFile, tagID);
	case IFDGatan65010::TagID:
		return new IFDGatan65010(aFile, tagID);
	case IFDGatan65015::TagID:
		return new IFDGatan65015(aFile, tagID);
	case IFDGatan65016::TagID:
		return new IFDGatan65016(aFile, tagID);
	case IFDGatan65024::TagID:
		return new IFDGatan65024(aFile, tagID);
	case IFDGatan65025::TagID:
		return new IFDGatan65025(aFile, tagID);
	case IFDGatan65026::TagID:
		return new IFDGatan65026(aFile, tagID);
	case IFDGatan65027::TagID:
		return new IFDGatan65027(aFile, tagID);
	case IFDLSM34412::TagID:
		return new IFDLSM34412(aFile, tagID);
	default:
		return new ImageFileDirectoryEntry(aFile, tagID);
	}
}

StringImageFileDirectory::StringImageFileDirectory(TIFFFile * aFile, ushort aTagID)
	: ImageFileDirectoryEntry(aFile, aTagID)
{
	if (GetTiffTypeSizeInBytes(mTag.Type) * mTag.Count <= 4)
	{
		uint temp = mTag.Offset.UIntVal;

		if (!aFile->FileReader::mIsLittleEndian)
		{			
			EndianSwap(temp);
		}
		
		char* text = new char[mTag.Count + 1];
		char* vals = (char*)&temp;
		for (size_t i = 0; i < mTag.Count; i++)
		{
			text[i] = vals[i];
		}
		text[mTag.Count] = 0;
		mValue = std::string(text);
		delete[] text;
	}
	else
	{
		uint currentOffset = (uint)aFile->FileReader::Tell(); //safe as TIFF are < 4 GByte...
		aFile->FileReader::Seek(mTag.Offset.UIntVal, std::ios_base::beg);
		mValue = mTifffile->ReadStr(mTag.Count);
		aFile->FileReader::Seek(currentOffset, std::ios_base::beg);
	}
}

ByteImageFileDirectory::ByteImageFileDirectory(TIFFFile * aFile, ushort aTagID)
	: ImageFileDirectoryEntry(aFile, aTagID), mValue(NULL)
{
	if (GetTiffTypeSizeInBytes(mTag.Type) * mTag.Count <= 4)
	{
		uint temp = mTag.Offset.UIntVal;

		if (!aFile->FileReader::mIsLittleEndian)
		{
			EndianSwap(temp);
		}

		mValue = new uchar[mTag.Count];
		uchar* vals = (uchar*)&temp;
		for (size_t i = 0; i < mTag.Count; i++)
		{
			mValue[i] = vals[i];
		}
	}
	else
	{
		uint currentOffset = (uint)aFile->FileReader::Tell(); //safe as TIFF are < 4 GByte...
		aFile->FileReader::Seek(mTag.Offset.UIntVal, std::ios_base::beg);
		mValue = new uchar[mTag.Count];
		mTifffile->Read((char*)mValue, mTag.Count * GetTiffTypeSizeInBytes(mTag.Type));
		aFile->FileReader::Seek(currentOffset, std::ios_base::beg);
	}
}

ByteImageFileDirectory::~ByteImageFileDirectory()
{
	if (mValue)
		delete[] mValue;
}

SByteImageFileDirectory::SByteImageFileDirectory(TIFFFile * aFile, ushort aTagID)
	: ImageFileDirectoryEntry(aFile, aTagID), mValue(NULL)
{
	if (GetTiffTypeSizeInBytes(mTag.Type) * mTag.Count <= 4)
	{
		uint temp = mTag.Offset.UIntVal;

		if (!aFile->FileReader::mIsLittleEndian)
		{
			EndianSwap(temp);
		}

		mValue = new char[mTag.Count];
		char* vals = (char*)&temp;
		for (size_t i = 0; i < mTag.Count; i++)
		{
			mValue[i] = vals[i];
		}
	}
	else
	{
		uint currentOffset = (uint)aFile->FileReader::Tell(); //safe as TIFF are < 4 GByte...
		aFile->FileReader::Seek(mTag.Offset.UIntVal, std::ios_base::beg);
		mValue = new char[mTag.Count];
		mTifffile->Read((char*)mValue, mTag.Count * GetTiffTypeSizeInBytes(mTag.Type));
		aFile->FileReader::Seek(currentOffset, std::ios_base::beg);
	}
}

SByteImageFileDirectory::~SByteImageFileDirectory()
{
	if (mValue)
		delete[] mValue;
}

UShortImageFileDirectory::UShortImageFileDirectory(TIFFFile * aFile, ushort aTagID)
	: ImageFileDirectoryEntry(aFile, aTagID), mValue(NULL)
{
	if (GetTiffTypeSizeInBytes(mTag.Type) * mTag.Count <= 4)
	{
		uint temp = mTag.Offset.UIntVal;

		if (!aFile->FileReader::mIsLittleEndian)
		{
			EndianSwap(temp);
		}

		mValue = new ushort[mTag.Count];
		ushort* vals = (ushort*)&temp;
		for (size_t i = 0; i < mTag.Count; i++)
		{
			mValue[i] = vals[i];
		}
	}
	else
	{
		uint currentOffset = (uint)aFile->FileReader::Tell(); //safe as TIFF are < 4 GByte...
		aFile->FileReader::Seek(mTag.Offset.UIntVal, std::ios_base::beg);
		mValue = new ushort[mTag.Count];
		mTifffile->Read((char*)mValue, mTag.Count * GetTiffTypeSizeInBytes(mTag.Type));
		aFile->FileReader::Seek(currentOffset, std::ios_base::beg);
	}
}

UShortImageFileDirectory::~UShortImageFileDirectory()
{
	if (mValue)
		delete[] mValue;
}

ShortImageFileDirectory::ShortImageFileDirectory(TIFFFile * aFile, ushort aTagID)
	: ImageFileDirectoryEntry(aFile, aTagID), mValue(NULL)
{
	if (GetTiffTypeSizeInBytes(mTag.Type) * mTag.Count <= 4)
	{
		uint temp = mTag.Offset.UIntVal;

		if (!aFile->FileReader::mIsLittleEndian)
		{
			EndianSwap(temp);
		}

		mValue = new short[mTag.Count];
		short* vals = (short*)&temp;
		for (size_t i = 0; i < mTag.Count; i++)
		{
			mValue[i] = vals[i];
		}
	}
	else
	{
		uint currentOffset = (uint)aFile->FileReader::Tell(); //safe as TIFF are < 4 GByte...
		aFile->FileReader::Seek(mTag.Offset.UIntVal, std::ios_base::beg);
		mValue = new short[mTag.Count];
		mTifffile->Read((char*)mValue, mTag.Count * GetTiffTypeSizeInBytes(mTag.Type));
		aFile->FileReader::Seek(currentOffset, std::ios_base::beg);
	}
}

ShortImageFileDirectory::~ShortImageFileDirectory()
{
	if (mValue)
		delete[] mValue;
}

UIntImageFileDirectory::UIntImageFileDirectory(TIFFFile * aFile, ushort aTagID)
	: ImageFileDirectoryEntry(aFile, aTagID), mValue(NULL)
{
	if (GetTiffTypeSizeInBytes(mTag.Type) * mTag.Count <= 4)
	{
		uint temp = mTag.Offset.UIntVal;

		if (!aFile->FileReader::mIsLittleEndian)
		{
			EndianSwap(temp);
		}

		mValue = new uint[mTag.Count];
		uint* vals = (uint*)&temp;
		for (size_t i = 0; i < mTag.Count; i++)
		{
			mValue[i] = vals[i];
		}
	}
	else
	{
		uint currentOffset = (uint)aFile->FileReader::Tell(); //safe as TIFF are < 4 GByte...
		aFile->FileReader::Seek(mTag.Offset.UIntVal, std::ios_base::beg);
		mValue = new uint[mTag.Count];
		mTifffile->Read((char*)mValue, mTag.Count * GetTiffTypeSizeInBytes(mTag.Type));
		aFile->FileReader::Seek(currentOffset, std::ios_base::beg);
	}
}

UIntImageFileDirectory::~UIntImageFileDirectory()
{
	if (mValue)
		delete[] mValue;
}

IntImageFileDirectory::IntImageFileDirectory(TIFFFile * aFile, ushort aTagID)
	: ImageFileDirectoryEntry(aFile, aTagID), mValue(NULL)
{
	if (GetTiffTypeSizeInBytes(mTag.Type) * mTag.Count <= 4)
	{
		uint temp = mTag.Offset.UIntVal;

		if (!aFile->FileReader::mIsLittleEndian)
		{
			EndianSwap(temp);
		}

		mValue = new int[mTag.Count];
		int* vals = (int*)&temp;
		for (size_t i = 0; i < mTag.Count; i++)
		{
			mValue[i] = vals[i];
		}
	}
	else
	{
		uint currentOffset = (uint)aFile->FileReader::Tell(); //safe as TIFF are < 4 GByte...
		aFile->FileReader::Seek(mTag.Offset.UIntVal, std::ios_base::beg);
		mValue = new int[mTag.Count];
		mTifffile->Read((char*)mValue, mTag.Count * GetTiffTypeSizeInBytes(mTag.Type));
		aFile->FileReader::Seek(currentOffset, std::ios_base::beg);
	}
}

IntImageFileDirectory::~IntImageFileDirectory()
{
	if (mValue)
		delete[] mValue;
}

FloatImageFileDirectory::FloatImageFileDirectory(TIFFFile * aFile, ushort aTagID)
	: ImageFileDirectoryEntry(aFile, aTagID), mValue(NULL)
{
	if (GetTiffTypeSizeInBytes(mTag.Type) * mTag.Count <= 4)
	{
		uint temp = mTag.Offset.UIntVal;

		if (!aFile->FileReader::mIsLittleEndian)
		{
			EndianSwap(temp);
		}

		mValue = new float[mTag.Count];
		float* vals = (float*)&temp;
		for (size_t i = 0; i < mTag.Count; i++)
		{
			mValue[i] = vals[i];
		}
	}
	else
	{
		uint currentOffset = (uint)aFile->FileReader::Tell(); //safe as TIFF are < 4 GByte...
		aFile->FileReader::Seek(mTag.Offset.UIntVal, std::ios_base::beg);
		mValue = new float[mTag.Count];
		mTifffile->Read((char*)mValue, mTag.Count * GetTiffTypeSizeInBytes(mTag.Type));
		aFile->FileReader::Seek(currentOffset, std::ios_base::beg);
	}
}

FloatImageFileDirectory::~FloatImageFileDirectory()
{
	if (mValue)
		delete[] mValue;
}

DoubleImageFileDirectory::DoubleImageFileDirectory(TIFFFile * aFile, ushort aTagID)
	: ImageFileDirectoryEntry(aFile, aTagID), mValue(NULL)
{	
	uint currentOffset = (uint)aFile->FileReader::Tell(); //safe as TIFF are < 4 GByte...
	aFile->FileReader::Seek(mTag.Offset.UIntVal, std::ios_base::beg);
	mValue = new double[mTag.Count];
	mTifffile->Read((char*)mValue, mTag.Count * GetTiffTypeSizeInBytes(mTag.Type));
	aFile->FileReader::Seek(currentOffset, std::ios_base::beg);	
}

DoubleImageFileDirectory::~DoubleImageFileDirectory()
{
	if (mValue)
		delete[] mValue;
}

RationalImageFileDirectory::RationalImageFileDirectory(TIFFFile * aFile, ushort aTagID)
	: ImageFileDirectoryEntry(aFile, aTagID), mValue(NULL)
{
	uint currentOffset = (uint)aFile->FileReader::Tell(); //safe as TIFF are < 4 GByte...
	aFile->FileReader::Seek(mTag.Offset.UIntVal, std::ios_base::beg);
	mValue = new Rational[mTag.Count];
	mTifffile->Read((char*)mValue, mTag.Count * GetTiffTypeSizeInBytes(mTag.Type));
	aFile->FileReader::Seek(currentOffset, std::ios_base::beg);
}

RationalImageFileDirectory::~RationalImageFileDirectory()
{
	if (mValue)
		delete[] mValue;
}

SRationalImageFileDirectory::SRationalImageFileDirectory(TIFFFile * aFile, ushort aTagID)
	: ImageFileDirectoryEntry(aFile, aTagID), mValue(NULL)
{
	uint currentOffset = (uint)aFile->FileReader::Tell(); //safe as TIFF are < 4 GByte...
	aFile->FileReader::Seek(mTag.Offset.UIntVal, std::ios_base::beg);
	mValue = new SRational[mTag.Count];
	mTifffile->Read((char*)mValue, mTag.Count * GetTiffTypeSizeInBytes(mTag.Type));
	aFile->FileReader::Seek(currentOffset, std::ios_base::beg);
}

SRationalImageFileDirectory::~SRationalImageFileDirectory()
{
	if (mValue)
		delete[] mValue;
}









const std::string IFDImageLength::TagName = "Image length";
const std::string IFDImageWidth::TagName = "Image width";
const std::string IFDRowsPerStrip::TagName = "Rows per strip";
const std::string IFDStripByteCounts::TagName = "Strip byte counts";
const std::string IFDStripOffsets::TagName = "Strip offsets";
const std::string IFDArtist::TagName = "Artist";
const std::string IFDCopyright::TagName = "Copyright";
const std::string IFDDateTime::TagName = "Date/Time";
const std::string IFDHostComputer::TagName = "Host computer";
const std::string IFDImageDescription::TagName = "Image description";
const std::string IFDModel::TagName = "Model";
const std::string IFDMake::TagName = "Make";
const std::string IFDSoftware::TagName = "Software";
const std::string IFDBitsPerSample::TagName = "Bits per sample";
const std::string IFDCellLength::TagName = "Cell length";
const std::string IFDCellWidth::TagName = "Cell width";
const std::string IFDColorMap::TagName = "Color map";
const std::string IFDCompression::TagName = "Compression";
const std::string IFDExtraSamples::TagName = "Extra samples";
const std::string IFDFillOrder::TagName = "Fill order";
const std::string IFDFreeByteCounts::TagName = "Free byte counts";
const std::string IFDFreeOffsets::TagName = "Free offsets";
const std::string IFDGrayResponseCurve::TagName = "Gray response curve";
const std::string IFDGrayResponseUnit::TagName = "Gray response unit";
const std::string IFDMaxSampleValue::TagName = "Max sample value";
const std::string IFDMinSampleValue::TagName = "Min sample value";
const std::string IFDNewSubfileType::TagName = "New subfile type";
const std::string IFDOrientation::TagName = "Orientation";
const std::string IFDPhotometricInterpretation::TagName = "Photometric interpretation";
const std::string IFDPlanarConfiguration::TagName = "Planar configuration";
const std::string IFDResolutionUnit::TagName = "Resolution unit";
const std::string IFDSamplesPerPixel::TagName = "Samples per pixel";
const std::string IFDSampleFormat::TagName = "Samples format";
const std::string IFDSubfileType::TagName = "Subfile type";
const std::string IFDThreshholding::TagName = "Threshholding";
const std::string IFDXResolution::TagName = "X-Resolution";
const std::string IFDYResolution::TagName = "Y-Resolution";
const std::string IFDGatan65006::TagName = "IFDGatan65006";
const std::string IFDGatan65007::TagName = "IFDGatan65007";
const std::string IFDGatan65009::TagName = "IFDGatan65009";
const std::string IFDGatan65010::TagName = "IFDGatan65010";
const std::string IFDGatan65015::TagName = "IFDGatan65015";
const std::string IFDGatan65016::TagName = "IFDGatan65016";
const std::string IFDGatan65024::TagName = "IFDGatan65024";
const std::string IFDGatan65025::TagName = "IFDGatan65025";
const std::string IFDGatan65026::TagName = "IFDGatan65026";
const std::string IFDGatan65027::TagName = "IFDGatan65027";
const std::string IFDLSM34412::TagName = "IFDLSM34412";

IFDImageLength::IFDImageLength(TIFFFile * aFile, ushort aTagID) :
	ImageFileDirectoryEntry(aFile, aTagID)
{
	if (GetTiffTypeSizeInBytes(mTag.Type) == 2 && !aFile->FileReader::mIsLittleEndian)
	{
		uint* ptr = &mTag.Offset.UIntVal;
		ushort* ptrUS = (ushort*)ptr;
		mValue = ptrUS[4 / GetTiffTypeSizeInBytes(mTag.Type) - 1];		
	}
	else
	{
		mValue = mTag.Offset.UIntVal;
	}
}

uint IFDImageLength::Value()
{
	return mValue;
}

IFDImageWidth::IFDImageWidth(TIFFFile * aFile, ushort aTagID) :
	ImageFileDirectoryEntry(aFile, aTagID)
{
	if (GetTiffTypeSizeInBytes(mTag.Type) == 2 && !aFile->FileReader::mIsLittleEndian)
	{
		uint* ptr = &mTag.Offset.UIntVal;
		ushort* ptrUS = (ushort*)ptr;
		mValue = ptrUS[4 / GetTiffTypeSizeInBytes(mTag.Type) - 1];
	}
	else
	{
		mValue = mTag.Offset.UIntVal;
	}
}

uint IFDImageWidth::Value()
{
	return mValue;
}

IFDRowsPerStrip::IFDRowsPerStrip(TIFFFile * aFile, ushort aTagID) :
	ImageFileDirectoryEntry(aFile, aTagID)
{
	if (GetTiffTypeSizeInBytes(mTag.Type) == 2 && !aFile->FileReader::mIsLittleEndian)
	{
		uint* ptr = &mTag.Offset.UIntVal;
		ushort* ptrUS = (ushort*)ptr;
		mValue = ptrUS[4 / GetTiffTypeSizeInBytes(mTag.Type) - 1];
	}
	else
	{
		mValue = mTag.Offset.UIntVal;
	}
}

uint IFDRowsPerStrip::Value()
{
	return mValue;
}

IFDStripByteCounts::IFDStripByteCounts(TIFFFile * aFile, ushort aTagID) :
	ImageFileDirectoryEntry(aFile, aTagID), mValue(NULL)
{
	if (GetTiffTypeSizeInBytes(mTag.Type) == 2)
	{
		if (GetTiffTypeSizeInBytes(mTag.Type) * mTag.Count <= 4)
		{
			uint* ptr = &mTag.Offset.UIntVal;
			ushort* ptrUS = (ushort*)ptr;
			mValue = new uint[mTag.Count];
			for (uint i = 0; i < mTag.Count; i++)
			{
				if (!aFile->FileReader::mIsLittleEndian)
					mValue[i] = ptrUS[4 / GetTiffTypeSizeInBytes(mTag.Type) - i - 1];
				else
					mValue[i] = ptrUS[i];
			}
		}
		else
		{
			size_t currentOffset = aFile->FileReader::Tell();
			aFile->FileReader::Seek(mTag.Offset.UIntVal, std::ios_base::beg);

			mValue = new uint[mTag.Count];
			for (uint i = 0; i < mTag.Count; i++)
			{
				mValue[i] = aFile->ReadUI2LE();
			}
			aFile->FileReader::Seek(currentOffset, std::ios_base::beg);
		}
	}
	else
	{
		if (GetTiffTypeSizeInBytes(mTag.Type) * mTag.Count <= 4)
		{
			mValue = new uint[mTag.Count];
			mValue[0] = mTag.Offset.UIntVal;
		}
		else
		{
			size_t currentOffset = aFile->FileReader::Tell();
			aFile->FileReader::Seek(mTag.Offset.UIntVal, std::ios_base::beg);

			mValue = new uint[mTag.Count];
			for (uint i = 0; i < mTag.Count; i++)
			{
				mValue[i] = aFile->ReadUI4LE();
			}
			aFile->FileReader::Seek(currentOffset, std::ios_base::beg);
		}
	}
}

IFDStripByteCounts::~IFDStripByteCounts()
{
	if (mValue)
		delete[] mValue;
}

uint* IFDStripByteCounts::Value()
{
	return mValue;
}

size_t IFDStripByteCounts::ValueCount()
{
	return mTag.Count;
}

IFDStripOffsets::IFDStripOffsets(TIFFFile * aFile, ushort aTagID) :
	ImageFileDirectoryEntry(aFile, aTagID), mValue(NULL)
{
	if (GetTiffTypeSizeInBytes(mTag.Type) == 2)
	{
		if (GetTiffTypeSizeInBytes(mTag.Type) * mTag.Count <= 4)
		{
			uint* ptr = &mTag.Offset.UIntVal;
			ushort* ptrUS = (ushort*)ptr;
			mValue = new uint[mTag.Count];
			for (uint i = 0; i < mTag.Count; i++)
			{
				if (!aFile->FileReader::mIsLittleEndian)
					mValue[i] = ptrUS[4 / GetTiffTypeSizeInBytes(mTag.Type) - i - 1];
				else
					mValue[i] = ptrUS[i];
			}
		}
		else
		{
			size_t currentOffset = aFile->FileReader::Tell();
			aFile->FileReader::Seek(mTag.Offset.UIntVal, std::ios_base::beg);

			mValue = new uint[mTag.Count];
			for (uint i = 0; i < mTag.Count; i++)
			{
				mValue[i] = aFile->ReadUI2LE();
			}
			aFile->FileReader::Seek(currentOffset, std::ios_base::beg);
		}
	}
	else
	{
		if (GetTiffTypeSizeInBytes(mTag.Type) * mTag.Count <= 4)
		{
			mValue = new uint[mTag.Count];
			mValue[0] = mTag.Offset.UIntVal;
		}
		else
		{
			size_t currentOffset = aFile->FileReader::Tell();
			aFile->FileReader::Seek(mTag.Offset.UIntVal, std::ios_base::beg);

			mValue = new uint[mTag.Count];
			for (uint i = 0; i < mTag.Count; i++)
			{
				mValue[i] = aFile->ReadUI4LE();
			}
			aFile->FileReader::Seek(currentOffset, std::ios_base::beg);
		}
	}
}

IFDStripOffsets::~IFDStripOffsets()
{
	if (mValue)
		delete[] mValue;
}

uint* IFDStripOffsets::Value()
{
	return mValue;
}

size_t IFDStripOffsets::ValueCount()
{
	return mTag.Count;
}

IFDArtist::IFDArtist(TIFFFile * aFile, ushort aTagID) :
	StringImageFileDirectory(aFile, aTagID)
{
}

std::string IFDArtist::Value()
{
	return mValue;
}

IFDCopyright::IFDCopyright(TIFFFile * aFile, ushort aTagID) :
	StringImageFileDirectory(aFile, aTagID)
{
}

std::string IFDCopyright::Value()
{
	return mValue;
}

IFDDateTime::IFDDateTime(TIFFFile * aFile, ushort aTagID) :
	StringImageFileDirectory(aFile, aTagID)
{
}

std::string IFDDateTime::Value()
{
	return mValue;
}

IFDHostComputer::IFDHostComputer(TIFFFile * aFile, ushort aTagID) :
	StringImageFileDirectory(aFile, aTagID)
{
}

std::string IFDHostComputer::Value()
{
	return mValue;
}

IFDImageDescription::IFDImageDescription(TIFFFile * aFile, ushort aTagID) :
	StringImageFileDirectory(aFile, aTagID)
{
}

std::string IFDImageDescription::Value()
{
	return mValue;
}

IFDModel::IFDModel(TIFFFile * aFile, ushort aTagID) :
	StringImageFileDirectory(aFile, aTagID)
{
}

std::string IFDModel::Value()
{
	return mValue;
}

IFDMake::IFDMake(TIFFFile * aFile, ushort aTagID) :
	StringImageFileDirectory(aFile, aTagID)
{
}

std::string IFDMake::Value()
{
	return mValue;
}

IFDSoftware::IFDSoftware(TIFFFile * aFile, ushort aTagID) :
	StringImageFileDirectory(aFile, aTagID)
{
}

std::string IFDSoftware::Value()
{
	return mValue;
}

IFDBitsPerSample::IFDBitsPerSample(TIFFFile * aFile, ushort aTagID) :
	UShortImageFileDirectory(aFile, aTagID)
{
}

ushort IFDBitsPerSample::Value(size_t aIdx)
{
	if (aIdx >= mTag.Count)
		return 0;
	return mValue[aIdx];
}

IFDCellLength::IFDCellLength(TIFFFile * aFile, ushort aTagID) :
	UShortImageFileDirectory(aFile, aTagID)
{
}

ushort IFDCellLength::Value()
{
	return mValue[0];
}

IFDCellWidth::IFDCellWidth(TIFFFile * aFile, ushort aTagID) :
	UShortImageFileDirectory(aFile, aTagID)
{
}

ushort IFDCellWidth::Value()
{
	return mValue[0];
}

IFDColorMap::IFDColorMap(TIFFFile * aFile, ushort aTagID) :
	UShortImageFileDirectory(aFile, aTagID)
{
}

ushort IFDColorMap::Value()
{
	return mValue[0];
}

IFDCompression::IFDCompression(TIFFFile * aFile, ushort aTagID) :
	UShortImageFileDirectory(aFile, aTagID)
{
}

IFDCompression::TIFFCompression IFDCompression::Value()
{
	return (TIFFCompression)mValue[0];
}

IFDExtraSamples::IFDExtraSamples(TIFFFile * aFile, ushort aTagID) :
	UShortImageFileDirectory(aFile, aTagID)
{
}

ushort IFDExtraSamples::Value()
{
	return mValue[0];
}

IFDFillOrder::IFDFillOrder(TIFFFile * aFile, ushort aTagID) :
	UShortImageFileDirectory(aFile, aTagID)
{
}

ushort IFDFillOrder::Value()
{
	return mValue[0];
}

IFDFreeByteCounts::IFDFreeByteCounts(TIFFFile * aFile, ushort aTagID) :
	UIntImageFileDirectory(aFile, aTagID)
{
}

uint IFDFreeByteCounts::Value()
{
	return mValue[0];
}

IFDFreeOffsets::IFDFreeOffsets(TIFFFile * aFile, ushort aTagID) :
	UIntImageFileDirectory(aFile, aTagID)
{
}

uint IFDFreeOffsets::Value()
{
	return mValue[0];
}

IFDGrayResponseCurve::IFDGrayResponseCurve(TIFFFile * aFile, ushort aTagID) :
	UShortImageFileDirectory(aFile, aTagID)
{
}

ushort* IFDGrayResponseCurve::Value()
{
	return mValue;
}

size_t IFDGrayResponseCurve::ValueCount()
{
	return mTag.Count;
}

IFDGrayResponseUnit::IFDGrayResponseUnit(TIFFFile * aFile, ushort aTagID) :
	UShortImageFileDirectory(aFile, aTagID)
{
}

ushort IFDGrayResponseUnit::Value()
{
	return mValue[0];
}

IFDMaxSampleValue::IFDMaxSampleValue(TIFFFile * aFile, ushort aTagID) :
	UShortImageFileDirectory(aFile, aTagID)
{
}

ushort IFDMaxSampleValue::Value()
{
	return mValue[0];
}

IFDMinSampleValue::IFDMinSampleValue(TIFFFile * aFile, ushort aTagID) :
	UShortImageFileDirectory(aFile, aTagID)
{
}

ushort IFDMinSampleValue::Value()
{
	return mValue[0];
}

IFDNewSubfileType::IFDNewSubfileType(TIFFFile * aFile, ushort aTagID) :
	UIntImageFileDirectory(aFile, aTagID)
{
}

uint IFDNewSubfileType::Value()
{
	return mValue[0];
}

IFDOrientation::IFDOrientation(TIFFFile * aFile, ushort aTagID) :
	UShortImageFileDirectory(aFile, aTagID)
{
}

IFDOrientation::TiffOrientation IFDOrientation::Value()
{
	return (TiffOrientation)mValue[0];
}

IFDPhotometricInterpretation::IFDPhotometricInterpretation(TIFFFile * aFile, ushort aTagID) :
	UShortImageFileDirectory(aFile, aTagID)
{
}

IFDPhotometricInterpretation::TIFFPhotometricInterpretation IFDPhotometricInterpretation::Value()
{
	return (TIFFPhotometricInterpretation)mValue[0];
}

IFDPlanarConfiguration::IFDPlanarConfiguration(TIFFFile * aFile, ushort aTagID) :
	UShortImageFileDirectory(aFile, aTagID)
{
}

IFDPlanarConfiguration::TIFFPlanarConfigurartion IFDPlanarConfiguration::Value()
{
	return (TIFFPlanarConfigurartion)mValue[0];
}

IFDResolutionUnit::IFDResolutionUnit(TIFFFile * aFile, ushort aTagID) :
	UShortImageFileDirectory(aFile, aTagID)
{
}

IFDResolutionUnit::TIFFResolutionUnit IFDResolutionUnit::Value()
{
	return (TIFFResolutionUnit)mValue[0];
}

IFDSamplesPerPixel::IFDSamplesPerPixel(TIFFFile * aFile, ushort aTagID) :
	UShortImageFileDirectory(aFile, aTagID)
{
}

ushort IFDSamplesPerPixel::Value()
{
	return mValue[0];
}

IFDSampleFormat::IFDSampleFormat(TIFFFile * aFile, ushort aTagID) :
	ShortImageFileDirectory(aFile, aTagID)
{
}

SampleFormatEnum IFDSampleFormat::Value()
{
	return (SampleFormatEnum)mValue[0];
}

IFDSubfileType::IFDSubfileType(TIFFFile * aFile, ushort aTagID) :
	UShortImageFileDirectory(aFile, aTagID)
{
}

ushort IFDSubfileType::Value()
{
	return mValue[0];
}

IFDThreshholding::IFDThreshholding(TIFFFile * aFile, ushort aTagID) :
	UShortImageFileDirectory(aFile, aTagID)
{
}

ushort IFDThreshholding::Value()
{
	return mValue[0];
}

IFDXResolution::IFDXResolution(TIFFFile * aFile, ushort aTagID) :
	RationalImageFileDirectory(aFile, aTagID)
{
}

Rational IFDXResolution::Value()
{
	return mValue[0];
}

IFDYResolution::IFDYResolution(TIFFFile * aFile, ushort aTagID) :
	RationalImageFileDirectory(aFile, aTagID)
{
}

Rational IFDYResolution::Value()
{
	return mValue[0];
}

IFDGatan65006::IFDGatan65006(TIFFFile * aFile, ushort aTagID) :
	DoubleImageFileDirectory(aFile, aTagID)
{
}

double IFDGatan65006::Value()
{
	return mValue[0];
}

IFDGatan65007::IFDGatan65007(TIFFFile * aFile, ushort aTagID) :
	DoubleImageFileDirectory(aFile, aTagID)
{
}

double IFDGatan65007::Value()
{
	return mValue[0];
}

IFDGatan65009::IFDGatan65009(TIFFFile * aFile, ushort aTagID) :
	DoubleImageFileDirectory(aFile, aTagID)
{
}

double IFDGatan65009::Value()
{
	return mValue[0];
}

IFDGatan65010::IFDGatan65010(TIFFFile * aFile, ushort aTagID) :
	DoubleImageFileDirectory(aFile, aTagID)
{
}

double IFDGatan65010::Value()
{
	return mValue[0];
}

IFDGatan65015::IFDGatan65015(TIFFFile * aFile, ushort aTagID) :
	IntImageFileDirectory(aFile, aTagID)
{
}

int IFDGatan65015::Value()
{
	return mValue[0];
}

IFDGatan65016::IFDGatan65016(TIFFFile * aFile, ushort aTagID) :
	IntImageFileDirectory(aFile, aTagID)
{
}

int IFDGatan65016::Value()
{
	return mValue[0];
}

IFDGatan65024::IFDGatan65024(TIFFFile * aFile, ushort aTagID) :
	DoubleImageFileDirectory(aFile, aTagID)
{
}

double IFDGatan65024::Value()
{
	return mValue[0];
}

IFDGatan65025::IFDGatan65025(TIFFFile * aFile, ushort aTagID) :
	DoubleImageFileDirectory(aFile, aTagID)
{
}

double IFDGatan65025::Value()
{
	return mValue[0];
}

IFDGatan65026::IFDGatan65026(TIFFFile * aFile, ushort aTagID) :
	IntImageFileDirectory(aFile, aTagID)
{
}

int IFDGatan65026::Value()
{
	return mValue[0];
}

IFDGatan65027::IFDGatan65027(TIFFFile * aFile, ushort aTagID) :
	ByteImageFileDirectory(aFile, aTagID)
{
}

uchar* IFDGatan65027::Value()
{
	return mValue;
}

uint IFDGatan65027::ValueCount()
{
	return mTag.Count;
}

IFDLSM34412::IFDLSM34412(TIFFFile * aFile, ushort aTagID) :
	ByteImageFileDirectory(aFile, aTagID)
{
}

IFDLSM34412::IFDInfo* IFDLSM34412::Value()
{
	size_t check = sizeof(IFDLSM34412::IFDInfo);
	return (IFDLSM34412::IFDInfo*)mValue;
}

uint IFDLSM34412::ValueCount()
{
	return mTag.Count;
}

ImageFileDirectory::ImageFileDirectory(TIFFFile * aFile)
	: mTifffile(aFile), mEntryCount(0), mEntries()
{
	mEntryCount = mTifffile->ReadUI2LE();
	for (ushort i = 0; i < mEntryCount; i++)
	{
		ImageFileDirectoryEntry* entry = ImageFileDirectoryEntry::CreateFileDirectoryEntry(mTifffile);
		mEntries.push_back(entry);
	}
}

ImageFileDirectory::~ImageFileDirectory()
{
	for (size_t i = 0; i < mEntries.size(); i++)
	{
		delete mEntries[i];
	}
	mEntries.clear();
}

ImageFileDirectoryEntry * ImageFileDirectory::GetEntry(ushort tagID)
{
	for (size_t i = 0; i < mEntries.size(); i++)
	{
		if (mEntries[i]->mTag.TagID == tagID)
			return mEntries[i];
	}

	return NULL;
}

TIFFFile::TIFFFile(string aFileName)
	: FileReader(aFileName), FileWriter(aFileName), _dataStartPosition(0), _data(NULL), _imageFileDirectories(),
	_width(0), _height(0), _bitsPerSample(0), _samplesPerPixel(0), _needsFlipOnY(true), _isPlanar(false), _sampleFormat(SAMPLEFORMAT_UINT), _pixelSize(0)
{
	memset(&_fileHeader, 0, sizeof(_fileHeader));
}

TIFFFile::~TIFFFile()
{
	// Need to figure out type of data array first
	DataType_enum dataType = this->GetDataType();

	// Free data block
	FileReader::DeleteData(_data, dataType);
	
	_data = NULL;

	for (size_t i = 0; i < _imageFileDirectories.size(); i++)
	{
		delete _imageFileDirectories[i];
	}
	_imageFileDirectories.clear();
}

struct membuf : std::streambuf
{
	membuf(char* begin, char* end) {
		this->setg(begin, begin, end);
	}
};


bool TIFFFile::OpenAndRead()
{
	bool res = FileReader::OpenRead();
	bool needEndianessInverse = false;

	if (!res)
	{
		FileReader::CloseRead();
		throw FileIOException(FileReader::mFileName, "Cannot open file for reading.");
	}
	
	Read((char*)&_fileHeader, sizeof(_fileHeader));

	if (_fileHeader.BytaOrder == 0x4949) //little Endian
	{
		needEndianessInverse = false; //We are on ittle Endian hardware...
	}
	else if (_fileHeader.BytaOrder == 0x4D4D) //big Endian
	{
		needEndianessInverse = true;
	}
	else
	{
		FileReader::CloseRead();
		//nothing allocated so far, just leave...
		throw FileIOException(FileReader::mFileName, "File doesn't seem to be a TIFF file.");
	}

	if (needEndianessInverse)
	{
		FileReader::Endian_swap(_fileHeader.ID);
		FileReader::Endian_swap(_fileHeader.OffsetToIFD);
		FileReader::mIsLittleEndian = false;
	}

	if (_fileHeader.ID != 42)
	{
		FileReader::CloseRead();
		//nothing allocated so far, just leave...
		throw FileIOException(FileReader::mFileName, "File doesn't seem to be a TIFF file.");
	}

	bool ok = FileReader::mFile->good();

	if (!ok)
	{
		FileReader::CloseRead();
		//nothing allocated so far, just leave...
		throw FileIOException(FileReader::mFileName, "This is not a proper TIFF file.");
	}

	FileReader::Seek(_fileHeader.OffsetToIFD, std::ios_base::beg);

	while (true)
	{
		ImageFileDirectory* ifd = new ImageFileDirectory(this);
		_imageFileDirectories.push_back(ifd);
		uint offsetToNext = ReadUI4LE();
		if (offsetToNext == 0)
			break;
		FileReader::Seek(offsetToNext, std::ios_base::beg);
	}

	//Raw Data:
	ImageFileDirectory* first = _imageFileDirectories[0];
	IFDCompression* compressionIFD = (IFDCompression*)first->GetEntry(IFDCompression::TagID);
	if (!compressionIFD)
	{
		FileReader::CloseRead();
		//Allocations will be freed in destructor, just leave...
		throw FileIOException(FileReader::mFileName, "This is not a proper TIFF file.");
	}
	if (compressionIFD->Value() != 1)
	{
		FileReader::CloseRead();
		//Allocations will be freed in destructor, just leave...
		throw FileIOException(FileReader::mFileName, "Cannot read compressed TIFF files.");
	}

	IFDStripOffsets* offsetIFD = (IFDStripOffsets*)first->GetEntry(IFDStripOffsets::TagID);
	if (!offsetIFD)
	{
		FileReader::CloseRead();
		//Allocations will be freed in destructor, just leave...
		throw FileIOException(FileReader::mFileName, "This is not a proper TIFF file.");
	}
	IFDImageWidth* widthIFD = (IFDImageWidth*)first->GetEntry(IFDImageWidth::TagID);
	if (!widthIFD)
	{
		FileReader::CloseRead();
		//Allocations will be freed in destructor, just leave...
		throw FileIOException(FileReader::mFileName, "This is not a proper TIFF file.");
	}
	IFDImageLength* heightIFD = (IFDImageLength*)first->GetEntry(IFDImageLength::TagID);
	if (!heightIFD)
	{
		FileReader::CloseRead();
		//Allocations will be freed in destructor, just leave...
		throw FileIOException(FileReader::mFileName, "This is not a proper TIFF file.");
	}
	IFDBitsPerSample* BPSIFD = (IFDBitsPerSample*)first->GetEntry(IFDBitsPerSample::TagID);
	if (!BPSIFD)
	{
		FileReader::CloseRead();
		//Allocations will be freed in destructor, just leave...
		throw FileIOException(FileReader::mFileName, "This is not a proper TIFF file.");
	}
	IFDStripByteCounts* SBCIFD = (IFDStripByteCounts*)first->GetEntry(IFDStripByteCounts::TagID);
	if (!SBCIFD)
	{
		FileReader::CloseRead();
		//Allocations will be freed in destructor, just leave...
		throw FileIOException(FileReader::mFileName, "This is not a proper TIFF file.");
	}
	IFDSamplesPerPixel* SPPIFD = (IFDSamplesPerPixel*)first->GetEntry(IFDSamplesPerPixel::TagID);
	/*if (!BPSIFD)
	{
		throw FileIOException(FileReader::mFileName, "This is not a proper TIFF file.");
	}*/
	IFDSampleFormat* sampleFormatIFD = (IFDSampleFormat*)first->GetEntry(IFDSampleFormat::TagID);
	if (sampleFormatIFD)
	{
		_sampleFormat = sampleFormatIFD->Value();
	}
	IFDPlanarConfiguration* planarIFD = (IFDPlanarConfiguration*)first->GetEntry(IFDPlanarConfiguration::TagID);
	if (planarIFD)
	{
		_isPlanar = planarIFD->Value() == IFDPlanarConfiguration::Planar;
	}
	IFDOrientation* orientationIFD = (IFDOrientation*)first->GetEntry(IFDOrientation::TagID);
	if (!orientationIFD)
	{
		_needsFlipOnY = false;
	}
	else
	{
		IFDOrientation::TiffOrientation orient = orientationIFD->Value();

		switch (orient)
		{
		case IFDOrientation::TOPLEFT:
		case IFDOrientation::TOPRIGHT:
		case IFDOrientation::LEFTTOP:
		case IFDOrientation::RIGHTTOP:
			_needsFlipOnY = false;
			break;
		case IFDOrientation::BOTRIGHT:
		case IFDOrientation::BOTLEFT:
		case IFDOrientation::RIGHTBOT:
		case IFDOrientation::LEFTBOT:
			_needsFlipOnY = true;
			break;
		default:
			_needsFlipOnY = false;
			break;
		}
	}

	//Check if File is a GATAN file, then it is flipped:
	IFDGatan65027* ifdGatan = ((IFDGatan65027*)first->GetEntry(IFDGatan65027::TagID));
	if (ifdGatan)
	{
		_needsFlipOnY = true;
		uchar* Gatan10 = ((IFDGatan65027*)first->GetEntry(IFDGatan65027::TagID))->Value();
		uint sizeGatan = ((IFDGatan65027*)first->GetEntry(IFDGatan65027::TagID))->ValueCount();
		membuf sbuf((char*)Gatan10, (char*)Gatan10 + sizeGatan);
		std::istream in(&sbuf);

		if (Gatan10[3] == 3)
		{
			Dm3File dm3(&in);
			dm3.OpenAndRead();
			float pix = dm3.GetPixelSizeX();
		}
		else if (Gatan10[3] == 4)
		{
			Dm4File dm4(&in);
			dm4.OpenAndRead();
			float pix = dm4.GetPixelSizeX();
		}
	}

	//Check if File is a LSM file:
	IFDLSM34412* ifdLSM = ((IFDLSM34412*)first->GetEntry(IFDLSM34412::TagID));
	if (ifdLSM)
	{
		IFDLSM34412::IFDInfo* info = ifdLSM->Value();
		_pixelSize = (float)(info->VoxelSizeX * pow(10,9));
	}
	/*if (offsetIFD->ValueCount() != 1)
	{
		throw FileIOException(FileReader::mFileName, "Cannt read multi-strip TIFF images.");
	}*/

	
	/*uchar* Gatan10 = ((IFDGatan65027*)first->GetEntry(IFDGatan65027::TagID))->Value();
	uint sizeGatan = ((IFDGatan65027*)first->GetEntry(IFDGatan65027::TagID))->ValueCount();
	membuf sbuf((char*)Gatan10, (char*)Gatan10 + sizeGatan);
	std::istream in(&sbuf);

	Dm3File dm3(&in);
	dm3.OpenAndRead();
	float pix = dm3.GetPixelSizeX();

	uint* t = (uint*)Gatan10;
	FileReader::Endian_swap(t[0]);
	FileReader::Endian_swap(t[1]);
	FileReader::Endian_swap(t[2]);
	FileReader::Endian_swap(t[3]);
	FileReader::Endian_swap(t[4]);
	FileReader::Endian_swap(t[5]);*/
	


	_width = widthIFD->Value();
	_height = heightIFD->Value();
	size_t pixelSizeInBytes = 0;
	if (!SPPIFD)
		_samplesPerPixel = 1;
	else
		_samplesPerPixel = SPPIFD->Value();

	_bitsPerSample = BPSIFD->Value(0);
	
	for (size_t i = 0; i < _samplesPerPixel; i++)
	{
		ushort sampleSize = BPSIFD->Value(i);
		if (sampleSize % 8 != 0)
		{
			FileReader::CloseRead();
			//Allocations will be freed in destructor, just leave...
			throw FileIOException("Cannot read TIFF files with not byte aligned pixel sizes.");
		}
		pixelSizeInBytes += sampleSize / 8;
	}

	//check consistency:
	size_t bytesToRead = 0;
	size_t minStripes = SBCIFD->ValueCount();

	if (offsetIFD->ValueCount() < SBCIFD->ValueCount())
		minStripes = offsetIFD->ValueCount();

	for (size_t stripe = 0; stripe < minStripes; stripe++)
	{
		bytesToRead += SBCIFD->Value()[stripe];
	}

	if (bytesToRead < _width * _height * pixelSizeInBytes)
	{
		throw FileIOException("Cannot read TIFF file: image stripes don't seem to fit image dimensions.");
	}

	_data = new char[_width * _height * pixelSizeInBytes];
	size_t offsetInData = 0;
	
	for (size_t stripe = 0; stripe < minStripes; stripe++)
	{
		uint offset = offsetIFD->Value()[stripe];
		size_t toRead = SBCIFD->Value()[stripe];
		FileReader::Seek(offset, std::ios_base::beg);

		if (offsetInData + toRead > _width * _height * pixelSizeInBytes)
		{
			toRead = _width * _height * pixelSizeInBytes - offsetInData;
		}
		if (toRead > 0)
			Read(((char*)_data) + offsetInData, toRead);

		offsetInData += toRead;
	}


	ok = FileReader::mFile->good();
	FileReader::CloseRead();

	if (!ok)
	{
		throw FileIOException(FileReader::mFileName, "This is not a proper TIFF file.");
	}
	

	return ok;
}

bool TIFFFile::OpenAndReadHeader()
{

	bool res = FileReader::OpenRead();
	bool needEndianessInverse = false;

	if (!res)
	{
		FileReader::CloseRead();
		throw FileIOException(FileReader::mFileName, "Cannot open file for reading.");
	}

	Read((char*)&_fileHeader, sizeof(_fileHeader));

	if (_fileHeader.BytaOrder == 0x4949) //little Endian
	{
		needEndianessInverse = false; //We are on ittle Endian hardware...
	}
	else if (_fileHeader.BytaOrder == 0x4D4D) //big Endian
	{
		needEndianessInverse = true;
	}
	else
	{
		FileReader::CloseRead();
		//Allocations, if any, will be freed in destructor, just leave...
		throw FileIOException(FileReader::mFileName, "File doesn't seem to be a TIFF file.");
	}

	if (needEndianessInverse)
	{
		FileReader::Endian_swap(_fileHeader.ID);
		FileReader::Endian_swap(_fileHeader.OffsetToIFD);
		FileReader::mIsLittleEndian = false;
	}

	if (_fileHeader.ID != 42)
	{
		FileReader::CloseRead();
		//Allocations, if any, will be freed in destructor, just leave...
		throw FileIOException(FileReader::mFileName, "File doesn't seem to be a TIFF file.");
	}

	bool ok = FileReader::mFile->good();

	if (!ok)
	{
		FileReader::CloseRead();
		//Allocations, if any, will be freed in destructor, just leave...
		throw FileIOException(FileReader::mFileName, "This is not a proper TIFF file.");
	}

	FileReader::Seek(_fileHeader.OffsetToIFD, std::ios_base::beg);

	while (true)
	{
		ImageFileDirectory* ifd = new ImageFileDirectory(this);
		_imageFileDirectories.push_back(ifd);
		uint offsetToNext = ReadUI4LE();
		if (offsetToNext == 0)
			break;
		FileReader::Seek(offsetToNext, std::ios_base::beg);
	}

	//Raw Data:
	ImageFileDirectory* first = _imageFileDirectories[0];
	IFDCompression* compressionIFD = (IFDCompression*)first->GetEntry(IFDCompression::TagID);
	if (!compressionIFD)
	{
		FileReader::CloseRead();
		//Allocations, if any, will be freed in destructor, just leave...
		throw FileIOException(FileReader::mFileName, "This is not a proper TIFF file.");
	}
	if (compressionIFD->Value() != 1)
	{
		FileReader::CloseRead();
		//Allocations, if any, will be freed in destructor, just leave...
		throw FileIOException(FileReader::mFileName, "Cannot read compressed TIFF files.");
	}

	IFDStripOffsets* offsetIFD = (IFDStripOffsets*)first->GetEntry(IFDStripOffsets::TagID);
	if (!offsetIFD)
	{
		FileReader::CloseRead();
		//Allocations, if any, will be freed in destructor, just leave...
		throw FileIOException(FileReader::mFileName, "This is not a proper TIFF file.");
	}
	IFDImageWidth* widthIFD = (IFDImageWidth*)first->GetEntry(IFDImageWidth::TagID);
	if (!widthIFD)
	{
		FileReader::CloseRead();
		//Allocations, if any, will be freed in destructor, just leave...
		throw FileIOException(FileReader::mFileName, "This is not a proper TIFF file.");
	}
	IFDImageLength* heightIFD = (IFDImageLength*)first->GetEntry(IFDImageLength::TagID);
	if (!heightIFD)
	{
		FileReader::CloseRead();
		//Allocations, if any, will be freed in destructor, just leave...
		throw FileIOException(FileReader::mFileName, "This is not a proper TIFF file.");
	}
	IFDBitsPerSample* BPSIFD = (IFDBitsPerSample*)first->GetEntry(IFDBitsPerSample::TagID);
	if (!BPSIFD)
	{
		FileReader::CloseRead();
		//Allocations, if any, will be freed in destructor, just leave...
		throw FileIOException(FileReader::mFileName, "This is not a proper TIFF file.");
	}
	IFDSamplesPerPixel* SPPIFD = (IFDSamplesPerPixel*)first->GetEntry(IFDSamplesPerPixel::TagID);
	IFDSampleFormat* sampleFormatIFD = (IFDSampleFormat*)first->GetEntry(IFDSampleFormat::TagID);
	if (sampleFormatIFD)
	{
		_sampleFormat = sampleFormatIFD->Value();
	}
	/*if (!BPSIFD)
	{
		throw FileIOException(FileReader::mFileName, "This is not a proper TIFF file.");
	}*/
	/*if (offsetIFD->ValueCount() != 1)
	{
	throw FileIOException(FileReader::mFileName, "Cannt read multi-strip TIFF images.");
	}*/

	IFDOrientation* orientationIFD = (IFDOrientation*)first->GetEntry(IFDOrientation::TagID);
	if (!orientationIFD)
	{
		_needsFlipOnY = false;
	}
	else
	{
		IFDOrientation::TiffOrientation orient = orientationIFD->Value();

		switch (orient)
		{
		case IFDOrientation::TOPLEFT:
		case IFDOrientation::TOPRIGHT:
		case IFDOrientation::LEFTTOP:
		case IFDOrientation::RIGHTTOP:
			_needsFlipOnY = false;
			break;
		case IFDOrientation::BOTRIGHT:
		case IFDOrientation::BOTLEFT:
		case IFDOrientation::RIGHTBOT:
		case IFDOrientation::LEFTBOT:
			_needsFlipOnY = true;
			break;
		default:
			_needsFlipOnY = false;
			break;
		}
	}

	//Check if File is a GATAN file, then it is flipped:
	IFDGatan65027* ifdGatan = ((IFDGatan65027*)first->GetEntry(IFDGatan65027::TagID));
	if (ifdGatan)
	{
		_needsFlipOnY = true;
	}


	_width = widthIFD->Value();
	_height = heightIFD->Value();
	uint offset = offsetIFD->Value()[0];
	size_t pixelSizeInBytes = 0;
	if (!SPPIFD)
		_samplesPerPixel = 1;
	else
		_samplesPerPixel = SPPIFD->Value();
	
	_bitsPerSample = BPSIFD->Value(0);

	for (size_t i = 0; i < _samplesPerPixel; i++)
	{
		ushort sampleSize = BPSIFD->Value(i);
		if (sampleSize % 8 != 0)
		{
			FileReader::CloseRead();
			//Allocations, if any, will be freed in destructor, just leave...
			throw FileIOException("Cannot read TIFF files with not byte aligned pixel sizes.");
		}
		pixelSizeInBytes += sampleSize / 8;
	}

	ok = FileReader::mFile->good();
	FileReader::CloseRead();

	if (!ok)
	{
		throw FileIOException(FileReader::mFileName, "This is not a proper TIFF file.");
	}


	return ok;
}

bool TIFFFile::WriteTIFF(string aFileName, int aDimX, int aDimY, float aPixelSize, DataType_enum aDatatype, void * aData)
{
	if (aDatatype == DataType_enum::DT_SHORT || aDatatype == DataType_enum::DT_UCHAR)
	{
		size_t dataSize = aDimX * aDimY * GetDataTypeSize(aDatatype);

		FileWriter file(aFileName, true);
		file.OpenWrite(true);

		TiffFileHeader fileheader; //8 bytes
		fileheader.BytaOrder = 0x4949;
		fileheader.ID = 42;
		fileheader.OffsetToIFD = sizeof(fileheader);
		ushort entryCount = 11; //2 bytes
		TiffTag ImageWidth; //12 bytes each
		TiffTag ImageLength;
		TiffTag BitsPerSample;
		TiffTag Compression;
		TiffTag PhotometricInterpretation;
		TiffTag RowsPerStrip;
		TiffTag StripByteCounts;
		TiffTag XResolution; //->set offset to 146
		TiffTag YResolution; //->set offset to 146+8
		TiffTag StripOffsets;//->set offset to 146+16
		TiffTag ResolutionUnit;
		uint endOfEntries = 0; //4 bytes
		//total header size is 8 + 2 + 12 * 11 + 4 = 146

		ImageWidth.Count = 1;
		ImageWidth.Offset.UIntVal = aDimX;
		ImageWidth.TagID = IFDImageWidth::TagID;
		ImageWidth.Type = TiffType_enum::TIFFT_LONG;

		ImageLength.Count = 1;
		ImageLength.Offset.UIntVal = aDimY;
		ImageLength.TagID = IFDImageLength::TagID;
		ImageLength.Type = TiffType_enum::TIFFT_LONG;

		BitsPerSample.Count = 1;
		BitsPerSample.Offset.UIntVal = 0;
		BitsPerSample.Offset.UShortVal = (ushort)GetDataTypeSize(aDatatype) * 8;
		BitsPerSample.TagID = IFDBitsPerSample::TagID;
		BitsPerSample.Type = TiffType_enum::TIFFT_SHORT;

		Compression.Count = 1;
		Compression.Offset.UIntVal = 0;
		Compression.Offset.UShortVal = IFDCompression::TIFFCompression::NoCompression;
		Compression.TagID = IFDCompression::TagID;
		Compression.Type = TiffType_enum::TIFFT_SHORT;

		PhotometricInterpretation.Count = 1;
		PhotometricInterpretation.Offset.UIntVal = 0;
		PhotometricInterpretation.Offset.UShortVal = IFDPhotometricInterpretation::TIFFPhotometricInterpretation::BlackIsZero;
		PhotometricInterpretation.TagID = IFDPhotometricInterpretation::TagID;
		PhotometricInterpretation.Type = TiffType_enum::TIFFT_SHORT;

		RowsPerStrip.Count = 1;
		RowsPerStrip.Offset.UIntVal = aDimY;
		RowsPerStrip.TagID = IFDRowsPerStrip::TagID;
		RowsPerStrip.Type = TiffType_enum::TIFFT_LONG;

		StripByteCounts.Count = 1;
		StripByteCounts.Offset.UIntVal = (uint)dataSize;
		StripByteCounts.TagID = IFDStripByteCounts::TagID;
		StripByteCounts.Type = TiffType_enum::TIFFT_LONG;

		ResolutionUnit.Count = 1;
		ResolutionUnit.Offset.UIntVal = 0;
		ResolutionUnit.Offset.UShortVal = IFDResolutionUnit::TIFFResolutionUnit::None;
		ResolutionUnit.TagID = IFDResolutionUnit::TagID;
		ResolutionUnit.Type = TiffType_enum::TIFFT_SHORT;

		XResolution.Count = 1;
		XResolution.Offset.UIntVal = 146;
		XResolution.TagID = IFDXResolution::TagID;
		XResolution.Type = TiffType_enum::TIFFT_RATIONAL;

		YResolution.Count = 1;
		YResolution.Offset.UIntVal = 146 + 8;
		YResolution.TagID = IFDYResolution::TagID;
		YResolution.Type = TiffType_enum::TIFFT_RATIONAL;

		StripOffsets.Count = 1;
		StripOffsets.Offset.UIntVal = 146 + 16;
		StripOffsets.TagID = IFDStripOffsets::TagID;
		StripOffsets.Type = TiffType_enum::TIFFT_LONG;


		Rational xRes(1, 72);
		Rational yRes(1, 72);


		file.Write(&fileheader, sizeof(fileheader));
		file.Write(&entryCount, 2);
		file.Write(&ImageWidth, sizeof(TiffTag));
		file.Write(&ImageLength, sizeof(TiffTag));
		file.Write(&BitsPerSample, sizeof(TiffTag));
		file.Write(&Compression, sizeof(TiffTag));
		file.Write(&PhotometricInterpretation, sizeof(TiffTag));
		file.Write(&RowsPerStrip, sizeof(TiffTag));
		file.Write(&StripByteCounts, sizeof(TiffTag));
		file.Write(&XResolution, sizeof(TiffTag));
		file.Write(&YResolution, sizeof(TiffTag));
		file.Write(&StripOffsets, sizeof(TiffTag));
		file.Write(&ResolutionUnit, sizeof(TiffTag));
		file.Write(&endOfEntries, 4);
		file.Write(&xRes, sizeof(Rational));
		file.Write(&yRes, sizeof(Rational));

		file.Write(aData, dataSize);
		file.CloseWrite();
		return true;
	}
	else if (aDatatype == DataType_enum::DT_SHORT3 || aDatatype == DataType_enum::DT_UCHAR3)
	{
		size_t dataSize = aDimX * aDimY * GetDataTypeSize(aDatatype);

		FileWriter file(aFileName, true);
		file.OpenWrite(true);

		TiffFileHeader fileheader; //8 bytes
		fileheader.BytaOrder = 0x4949;
		fileheader.ID = 42;
		fileheader.OffsetToIFD = sizeof(fileheader);
		ushort entryCount = 12; //2 bytes
		TiffTag ImageWidth; //12 bytes each
		TiffTag ImageLength;
		TiffTag BitsPerSample;
		TiffTag SamplesPerPixel;
		TiffTag Compression;
		TiffTag PhotometricInterpretation;
		TiffTag RowsPerStrip;
		TiffTag StripByteCounts;
		TiffTag XResolution; //->set offset to 158
		TiffTag YResolution; //->set offset to 158+8
		TiffTag StripOffsets;//->set offset to 158+16
		TiffTag ResolutionUnit;
		uint endOfEntries = 0; //4 bytes
							   //total header size is 8 + 2 + 12 * 12 + 4 + 4*2 = 158

		ImageWidth.Count = 1;
		ImageWidth.Offset.UIntVal = aDimX;
		ImageWidth.TagID = IFDImageWidth::TagID;
		ImageWidth.Type = TiffType_enum::TIFFT_LONG;

		ImageLength.Count = 1;
		ImageLength.Offset.UIntVal = aDimY;
		ImageLength.TagID = IFDImageLength::TagID;
		ImageLength.Type = TiffType_enum::TIFFT_LONG;

		ushort BitsPerSampleValues[4];
		BitsPerSampleValues[0] = (ushort)(GetDataTypeSize(aDatatype) * 8) / 3;
		BitsPerSampleValues[1] = (ushort)(GetDataTypeSize(aDatatype) * 8) / 3;
		BitsPerSampleValues[2] = (ushort)(GetDataTypeSize(aDatatype) * 8) / 3;
		BitsPerSampleValues[3] = 0;
		BitsPerSample.Count = 3;
		BitsPerSample.Offset.UIntVal = 158 + 16; // Put it behind the gray valued header...
		//BitsPerSample.Offset.UShortVal = GetDataTypeSize(aDatatype) * 8;
		BitsPerSample.TagID = IFDBitsPerSample::TagID;
		BitsPerSample.Type = TiffType_enum::TIFFT_SHORT;

		SamplesPerPixel.Count = 1;
		SamplesPerPixel.Offset.UIntVal = 0;
		SamplesPerPixel.Offset.UShortVal = 3;
		SamplesPerPixel.TagID = IFDSamplesPerPixel::TagID;
		SamplesPerPixel.Type = TiffType_enum::TIFFT_SHORT;

		Compression.Count = 1;
		Compression.Offset.UIntVal = 0;
		Compression.Offset.UShortVal = IFDCompression::TIFFCompression::NoCompression;
		Compression.TagID = IFDCompression::TagID;
		Compression.Type = TiffType_enum::TIFFT_SHORT;

		PhotometricInterpretation.Count = 1;
		PhotometricInterpretation.Offset.UIntVal = 0;
		PhotometricInterpretation.Offset.UShortVal = IFDPhotometricInterpretation::TIFFPhotometricInterpretation::RGB;
		PhotometricInterpretation.TagID = IFDPhotometricInterpretation::TagID;
		PhotometricInterpretation.Type = TiffType_enum::TIFFT_SHORT;

		RowsPerStrip.Count = 1;
		RowsPerStrip.Offset.UIntVal = aDimY;
		RowsPerStrip.TagID = IFDRowsPerStrip::TagID;
		RowsPerStrip.Type = TiffType_enum::TIFFT_LONG;

		StripByteCounts.Count = 1;
		StripByteCounts.Offset.UIntVal = (uint)dataSize;
		StripByteCounts.TagID = IFDStripByteCounts::TagID;
		StripByteCounts.Type = TiffType_enum::TIFFT_LONG;

		ResolutionUnit.Count = 1;
		ResolutionUnit.Offset.UIntVal = 0;
		ResolutionUnit.Offset.UShortVal = IFDResolutionUnit::TIFFResolutionUnit::None;
		ResolutionUnit.TagID = IFDResolutionUnit::TagID;
		ResolutionUnit.Type = TiffType_enum::TIFFT_SHORT;

		XResolution.Count = 1;
		XResolution.Offset.UIntVal = 158;
		XResolution.TagID = IFDXResolution::TagID;
		XResolution.Type = TiffType_enum::TIFFT_RATIONAL;

		YResolution.Count = 1;
		YResolution.Offset.UIntVal = 158 + 8;
		YResolution.TagID = IFDYResolution::TagID;
		YResolution.Type = TiffType_enum::TIFFT_RATIONAL;

		StripOffsets.Count = 1;
		StripOffsets.Offset.UIntVal = 158 + 16 + 4 * 2;
		StripOffsets.TagID = IFDStripOffsets::TagID;
		StripOffsets.Type = TiffType_enum::TIFFT_LONG;


		Rational xRes(1, 72);
		Rational yRes(1, 72);


		file.Write(&fileheader, sizeof(fileheader));
		file.Write(&entryCount, 2);
		file.Write(&ImageWidth, sizeof(TiffTag));
		file.Write(&ImageLength, sizeof(TiffTag));
		file.Write(&BitsPerSample, sizeof(TiffTag));
		file.Write(&SamplesPerPixel, sizeof(TiffTag));
		file.Write(&Compression, sizeof(TiffTag));
		file.Write(&PhotometricInterpretation, sizeof(TiffTag));
		file.Write(&RowsPerStrip, sizeof(TiffTag));
		file.Write(&StripByteCounts, sizeof(TiffTag));
		file.Write(&XResolution, sizeof(TiffTag));
		file.Write(&YResolution, sizeof(TiffTag));
		file.Write(&StripOffsets, sizeof(TiffTag));
		file.Write(&ResolutionUnit, sizeof(TiffTag));
		file.Write(&endOfEntries, 4);
		file.Write(&xRes, sizeof(Rational));
		file.Write(&yRes, sizeof(Rational));
		file.Write(BitsPerSampleValues, sizeof(BitsPerSampleValues));

		size_t test = file.Tell();

		file.Write(aData, dataSize);
		file.CloseWrite();
		return true;
	}
	else if (aDatatype == DataType_enum::DT_SHORT4 || aDatatype == DataType_enum::DT_UCHAR4)
	{
		size_t dataSize = aDimX * aDimY * GetDataTypeSize(aDatatype);

		FileWriter file(aFileName, true);
		file.OpenWrite(true);

		TiffFileHeader fileheader; //8 bytes
		fileheader.BytaOrder = 0x4949;
		fileheader.ID = 42;
		fileheader.OffsetToIFD = sizeof(fileheader);
		ushort entryCount = 12; //2 bytes
		TiffTag ImageWidth; //12 bytes each
		TiffTag ImageLength;
		TiffTag BitsPerSample;
		TiffTag SamplesPerPixel;
		TiffTag Compression;
		TiffTag PhotometricInterpretation;
		TiffTag RowsPerStrip;
		TiffTag StripByteCounts;
		TiffTag XResolution; //->set offset to 158
		TiffTag YResolution; //->set offset to 158+8
		TiffTag StripOffsets;//->set offset to 158+16
		TiffTag ResolutionUnit;
		uint endOfEntries = 0; //4 bytes
							   //total header size is 8 + 2 + 12 * 12 + 4 + 4*2 = 158

		ImageWidth.Count = 1;
		ImageWidth.Offset.UIntVal = aDimX;
		ImageWidth.TagID = IFDImageWidth::TagID;
		ImageWidth.Type = TiffType_enum::TIFFT_LONG;

		ImageLength.Count = 1;
		ImageLength.Offset.UIntVal = aDimY;
		ImageLength.TagID = IFDImageLength::TagID;
		ImageLength.Type = TiffType_enum::TIFFT_LONG;

		ushort BitsPerSampleValues[4];
		BitsPerSampleValues[0] = (ushort)(GetDataTypeSize(aDatatype) * 8) / 4;
		BitsPerSampleValues[1] = (ushort)(GetDataTypeSize(aDatatype) * 8) / 4;
		BitsPerSampleValues[2] = (ushort)(GetDataTypeSize(aDatatype) * 8) / 4;
		BitsPerSampleValues[3] = (ushort)(GetDataTypeSize(aDatatype) * 8) / 4;
		BitsPerSample.Count = 4;
		BitsPerSample.Offset.UIntVal = 158 + 16; // Put it behind the gray valued header...
												 //BitsPerSample.Offset.UShortVal = GetDataTypeSize(aDatatype) * 8;
		BitsPerSample.TagID = IFDBitsPerSample::TagID;
		BitsPerSample.Type = TiffType_enum::TIFFT_SHORT;

		SamplesPerPixel.Count = 1;
		SamplesPerPixel.Offset.UIntVal = 0;
		SamplesPerPixel.Offset.UShortVal = 4;
		SamplesPerPixel.TagID = IFDSamplesPerPixel::TagID;
		SamplesPerPixel.Type = TiffType_enum::TIFFT_SHORT;

		Compression.Count = 1;
		Compression.Offset.UIntVal = 0;
		Compression.Offset.UShortVal = IFDCompression::TIFFCompression::NoCompression;
		Compression.TagID = IFDCompression::TagID;
		Compression.Type = TiffType_enum::TIFFT_SHORT;

		PhotometricInterpretation.Count = 1;
		PhotometricInterpretation.Offset.UIntVal = 0;
		PhotometricInterpretation.Offset.UShortVal = IFDPhotometricInterpretation::TIFFPhotometricInterpretation::RGB;
		PhotometricInterpretation.TagID = IFDPhotometricInterpretation::TagID;
		PhotometricInterpretation.Type = TiffType_enum::TIFFT_SHORT;

		RowsPerStrip.Count = 1;
		RowsPerStrip.Offset.UIntVal = aDimY;
		RowsPerStrip.TagID = IFDRowsPerStrip::TagID;
		RowsPerStrip.Type = TiffType_enum::TIFFT_LONG;

		StripByteCounts.Count = 1;
		StripByteCounts.Offset.UIntVal = (uint)dataSize;
		StripByteCounts.TagID = IFDStripByteCounts::TagID;
		StripByteCounts.Type = TiffType_enum::TIFFT_LONG;

		ResolutionUnit.Count = 1;
		ResolutionUnit.Offset.UIntVal = 0;
		ResolutionUnit.Offset.UShortVal = IFDResolutionUnit::TIFFResolutionUnit::None;
		ResolutionUnit.TagID = IFDResolutionUnit::TagID;
		ResolutionUnit.Type = TiffType_enum::TIFFT_SHORT;

		XResolution.Count = 1;
		XResolution.Offset.UIntVal = 158;
		XResolution.TagID = IFDXResolution::TagID;
		XResolution.Type = TiffType_enum::TIFFT_RATIONAL;

		YResolution.Count = 1;
		YResolution.Offset.UIntVal = 158 + 8;
		YResolution.TagID = IFDYResolution::TagID;
		YResolution.Type = TiffType_enum::TIFFT_RATIONAL;

		StripOffsets.Count = 1;
		StripOffsets.Offset.UIntVal = 158 + 16 + 4 * 2;
		StripOffsets.TagID = IFDStripOffsets::TagID;
		StripOffsets.Type = TiffType_enum::TIFFT_LONG;


		Rational xRes(1, 72);
		Rational yRes(1, 72);


		file.Write(&fileheader, sizeof(fileheader));
		file.Write(&entryCount, 2);
		file.Write(&ImageWidth, sizeof(TiffTag));
		file.Write(&ImageLength, sizeof(TiffTag));
		file.Write(&BitsPerSample, sizeof(TiffTag));
		file.Write(&SamplesPerPixel, sizeof(TiffTag));
		file.Write(&Compression, sizeof(TiffTag));
		file.Write(&PhotometricInterpretation, sizeof(TiffTag));
		file.Write(&RowsPerStrip, sizeof(TiffTag));
		file.Write(&StripByteCounts, sizeof(TiffTag));
		file.Write(&XResolution, sizeof(TiffTag));
		file.Write(&YResolution, sizeof(TiffTag));
		file.Write(&StripOffsets, sizeof(TiffTag));
		file.Write(&ResolutionUnit, sizeof(TiffTag));
		file.Write(&endOfEntries, 4);
		file.Write(&xRes, sizeof(Rational));
		file.Write(&yRes, sizeof(Rational));
		file.Write(BitsPerSampleValues, sizeof(BitsPerSampleValues));

		file.Write(aData, dataSize);
		file.CloseWrite();
		return true;
	}
	return false;
}

bool TIFFFile::CanWriteAsTIFF(int aDimX, int aDimY, DataType_enum aDatatype)
{
	if (aDimX < 32767 && aDimY < 32767 && aDimX > 0 && aDimY > 0)
	{
		if (aDatatype == DataType_enum::DT_UCHAR ||
			aDatatype == DataType_enum::DT_UCHAR3 ||
			aDatatype == DataType_enum::DT_UCHAR4 ||
			aDatatype == DataType_enum::DT_SHORT ||
			aDatatype == DataType_enum::DT_SHORT3 ||
			aDatatype == DataType_enum::DT_SHORT4)
		{
			return true;
		}
	}
	return false;
}

DataType_enum TIFFFile::GetDataTypeUnsigned()
{
	if (_samplesPerPixel == 1)
	{
		if (_bitsPerSample == 8)
			return DataType_enum::DT_UCHAR;
		if (_bitsPerSample == 16)
			return DataType_enum::DT_USHORT;
		if (_bitsPerSample == 32)
			return DataType_enum::DT_UINT;
	}
	if (_samplesPerPixel == 2)
	{
		if (_bitsPerSample == 8)
			return DataType_enum::DT_UCHAR2;
		if (_bitsPerSample == 16)
			return DataType_enum::DT_USHORT2;
	}
	if (_samplesPerPixel == 3)
	{
		if (_bitsPerSample == 8)
			return DataType_enum::DT_UCHAR3;
		if (_bitsPerSample == 16)
			return DataType_enum::DT_USHORT3;
	}
	if (_samplesPerPixel == 4)
	{
		if (_bitsPerSample == 8)
			return DataType_enum::DT_UCHAR4;
		if (_bitsPerSample == 16)
			return DataType_enum::DT_USHORT4;
	}
	return DataType_enum::DT_UNKNOWN;
}

DataType_enum TIFFFile::GetDataTypeSigned()
{
	if (_samplesPerPixel == 1)
	{
		if (_bitsPerSample == 8)
			return DataType_enum::DT_CHAR;
		if (_bitsPerSample == 16)
			return DataType_enum::DT_SHORT;
		if (_bitsPerSample == 32)
			return DataType_enum::DT_INT;
	}
	if (_samplesPerPixel == 2)
	{
		if (_bitsPerSample == 16)
			return DataType_enum::DT_SHORT2;
	}
	if (_samplesPerPixel == 3)
	{
		if (_bitsPerSample == 16)
			return DataType_enum::DT_SHORT3;
	}
	if (_samplesPerPixel == 4)
	{
		if (_bitsPerSample == 16)
			return DataType_enum::DT_SHORT4;
	}
	return DataType_enum::DT_UNKNOWN;
}

DataType_enum TIFFFile::GetDataTypeFloat()
{
	if (_samplesPerPixel == 1)
	{
		if (_bitsPerSample == 32)
			return DataType_enum::DT_FLOAT;
	}
	if (_samplesPerPixel == 2)
	{
		if (_bitsPerSample == 32)
			return DataType_enum::DT_FLOAT2;
	}
	if (_samplesPerPixel == 3)
	{
		if (_bitsPerSample == 32)
			return DataType_enum::DT_FLOAT3;
	}
	if (_samplesPerPixel == 4)
	{
		if (_bitsPerSample == 32)
			return DataType_enum::DT_FLOAT4;
	}
	return DataType_enum::DT_UNKNOWN;
}

DataType_enum TIFFFile::GetDataTypeComplex()
{
	if (_samplesPerPixel == 1)
	{
		if (_bitsPerSample == 32)
			return DataType_enum::DT_SHORT2;
	}
	return DataType_enum::DT_UNKNOWN;
}

DataType_enum TIFFFile::GetDataTypeComplexFloat()
{
	if (_samplesPerPixel == 1)
	{
		if (_bitsPerSample == 32)
			return DataType_enum::DT_FLOAT2;
	}
	return DataType_enum::DT_UNKNOWN;
}

DataType_enum TIFFFile::GetDataType()
{
	switch (_sampleFormat)
	{
	case SAMPLEFORMAT_UINT:
		return GetDataTypeUnsigned();
	case SAMPLEFORMAT_INT:
		return GetDataTypeSigned();
	case SAMPLEFORMAT_IEEEFP:
		return GetDataTypeFloat();
	case SAMPLEFORMAT_VOID:
		return DataType_enum::DT_UNKNOWN;
	case SAMPLEFORMAT_COMPLEXINT:
		return GetDataTypeComplex();
	case SAMPLEFORMAT_COMPLEXIEEEFP:
		return GetDataTypeComplexFloat();
	default:
		return DataType_enum::DT_UNKNOWN;
	}
	
}

size_t TIFFFile::GetDataSize()
{
	return _width * _height * _samplesPerPixel * _bitsPerSample / 8;
}

TiffFileHeader & TIFFFile::GetFileHeader()
{
	return _fileHeader;
}

void * TIFFFile::GetData()
{
	return _data;
}

uint TIFFFile::GetWidth()
{
	return _width;
}

uint TIFFFile::GetHeight()
{
	return _height;
}

uint TIFFFile::BitsPerSample()
{
	return _bitsPerSample;
}

uint TIFFFile::SamplesPerPixel()
{
	return _samplesPerPixel;
}

bool TIFFFile::NeedsFlipOnYAxis()
{
	return _needsFlipOnY;
}

bool TIFFFile::GetIsPlanar()
{
	return _isPlanar;
}

float TIFFFile::GetPixelSize()
{
	return _pixelSize;
}
