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


#include "Dm4FileTag.h"

using namespace std;

Dm4FileTag::Dm4FileTag()
{
	InfoArray = NULL;
	Data = NULL;
}

Dm4FileTag::~Dm4FileTag()
{
	if (InfoArray)
		delete[] InfoArray;
	InfoArray = NULL;
	if (Data)
		delete[] Data;
	Data = NULL;
}

uint Dm4FileTag::GetSize(uint typeID)
{
	uint totalSize = 0;
	switch (typeID)
		{
		case DTTI2:
			totalSize += 2;
			break;
		case DTTI4:
			totalSize += 4;
			break;
		case DTTUI2:
			totalSize += 2;
			break;
		case DTTUI4:
			totalSize += 4;
			break;
		case DTTF4:
			totalSize += 4;
			break;
		case DTTF8:
			totalSize += 8;
			break;
		case DTTI1:
			totalSize += 1;
			break;
		case DTTCHAR:
			totalSize += 1;
			break;
		case DTTI1_2:
			totalSize += 1;
			break;
		case DTTUI8:
			totalSize += 8;
			break;
		}
	return totalSize;
}

uint Dm4FileTag::GetSizePixel(uint typeID)
{
	uint totalSize = 0;
	switch (typeID)
		{
		case DTT_UNKNOWN: 
			totalSize += 0;
			break;
		case DTT_I2: 
			totalSize += 2;
			break;
		case DTT_F4: 
			totalSize += 4;
			break;
		case DTT_C8: 
			totalSize += 8;
			break;
		case DTT_OBSOLETE: 
			totalSize += 0;
			break;
		case DTT_C4: 
			totalSize += 4;
			break;
		case DTT_UI1: 
			totalSize += 1;
			break;
		case DTT_I4: 
			totalSize += 4;
			break;
		case DTT_RGB_4UI1: 
			totalSize += 4;
			break;
		case DTT_I1: 
			totalSize += 1;
			break;
		case DTT_UI2: 
			totalSize += 2;
			break;
		case DTT_UI4: 
			totalSize += 4;
			break;
		case DTT_F8: 
			totalSize += 8;
			break;
		case DTT_C16: 
			totalSize += 16;
			break;
		case DTT_BINARY: 
			totalSize += 1;
			break;
		case DTT_RGBA_4UI1: 
			totalSize += 4;
			break;
		}
	return totalSize;
}

//void Dm4FileTag::PrintValues(ostream& aStream)
//{
//	
//}

void Dm4FileTag::PrintSingleValue(ostream& aStream, uint aType, uint aStartPos)
{
	switch (aType)
	{
		case DTTI2:
			aStream << (*(short*)&(this->Data[aStartPos]));
			break;
		case DTTI4:
			aStream << (*(int*)&(this->Data[aStartPos]));
			break;
		case DTTUI2:
			aStream << (*(ushort*)&(this->Data[aStartPos]));
			break;
		case DTTUI4:
			aStream << (*(uint*)&(this->Data[aStartPos]));
			break;
		case DTTF4:
			aStream << (*(float*)&(this->Data[aStartPos]));
			break;
		case DTTF8:
			aStream << (*(double*)&(this->Data[aStartPos]));
			break;
		case DTTI1:
			aStream << (*(char*)&(this->Data[aStartPos]));
			break;
		case DTTCHAR:
			aStream << (*(char*)&(this->Data[aStartPos]));
			break;
		case DTTI1_2:
			aStream << (*(char*)&(this->Data[aStartPos]));
			break;
		case DTTUI8:
			aStream << (*(ulong64*)&(this->Data[aStartPos]));
			break;
	//case DTT_I2:
	//	aStream << (*(short*)&(this->Data[aStartPos]));
	//	break;
	//case DTT_F4:
	//	aStream << (*(float*)&(this->Data[aStartPos]));
	//	break;
	//case DTT_C8:
	//	aStream << (*(float*)&(this->Data[aStartPos])) 
	//		    << (*(float*)&(this->Data[aStartPos + 4]));
	//	break;
	//case DTT_OBSOLETE:
	//	// NOTHING
	//	break;
	//case DTT_C4:
	//	aStream << (*(short*)&(this->Data[aStartPos])) 
	//		    << (*(short*)&(this->Data[aStartPos + 2]));
	//	break;
	//case DTT_UI1:
	//	aStream << (*(uchar*)&(this->Data[aStartPos]));
	//	break;
	//case DTT_I4:
	//	aStream << (*(int*)&(this->Data[aStartPos]));
	//	break;
	//case DTT_RGB_4UI1:
	//	aStream << "[";
	//	aStream << (*(uchar*)&(this->Data[aStartPos]))   << ";";
	//	aStream << (*(uchar*)&(this->Data[aStartPos+1])) << ";";
	//	aStream << (*(uchar*)&(this->Data[aStartPos+2])) << ";";
	//	aStream << (*(uchar*)&(this->Data[aStartPos+3])) << "]";
	//	break;
	//case DTT_I1:
	//	aStream << (*(char*)&(this->Data[aStartPos]));
	//	break;
	//case DTT_UI2:
	//	aStream << (*(ushort*)&(this->Data[aStartPos]));
	//	break;
	//case DTT_UI4:
	//	aStream << (*(uint*)&(this->Data[aStartPos]));
	//	break;
	//case DTT_F8:
	//	aStream << (*(double*)&(this->Data[aStartPos]));
	//	break;
	//case DTT_C16:
	//	aStream << (*(double*)&(this->Data[aStartPos])) 
	//		    << (*(double*)&(this->Data[aStartPos + 8]));
	//	break;
	//case DTT_BINARY:
	//	aStream << ((*(bool*)&(this->Data[aStartPos])) ? "True" : "False");
	//	break;
	//case DTT_RGBA_4UI1:
	//	aStream << "[";
	//	aStream << (*(uchar*)&(this->Data[aStartPos]))   << ";";
	//	aStream << (*(uchar*)&(this->Data[aStartPos+1])) << ";";
	//	aStream << (*(uchar*)&(this->Data[aStartPos+2])) << ";";
	//	aStream << (*(uchar*)&(this->Data[aStartPos+3])) << "]";
	//	break;
	}
}

int Dm4FileTag::GetSingleValueInt(uint aType, uint aStartPos)
{
	switch (aType)
	{
		case DTTI2:
			return (int) (*(short*)&(this->Data[aStartPos]));
		case DTTI4:
			return (int) (*(int*)&(this->Data[aStartPos]));
		case DTTUI2:
			return (int) (*(ushort*)&(this->Data[aStartPos]));
		case DTTUI4:
			return (int) (*(uint*)&(this->Data[aStartPos]));
		case DTTF4:
			return (int) (*(float*)&(this->Data[aStartPos]));
		case DTTF8:
			return (int) (*(double*)&(this->Data[aStartPos]));
		case DTTI1:
			return (int) (*(char*)&(this->Data[aStartPos]));
		case DTTCHAR:
			return (int) (*(char*)&(this->Data[aStartPos]));
		case DTTI1_2:
			return (int) (*(char*)&(this->Data[aStartPos]));
		case DTTUI8:
			return (int) (*(ulong64*)&(this->Data[aStartPos]));
	/*case DTT_I2:
		return (int) (*(short*)&(this->Data[aStartPos]));
	case DTT_F4:
		return (int) (*(float*)&(this->Data[aStartPos]));
	case DTT_C8:
		return 0;
	case DTT_OBSOLETE:
		return 0;
	case DTT_C4:
		return 0;
	case DTT_UI1:
		return (int) (*(uchar*)&(this->Data[aStartPos]));
	case DTT_I4:
		return (int) (*(int*)&(this->Data[aStartPos]));
	case DTT_RGB_4UI1:
		return (int) (*(int*)&(this->Data[aStartPos]));
	case DTT_I1:
		return (int) (*(char*)&(this->Data[aStartPos]));
	case DTT_UI2:
		return (int) (*(ushort*)&(this->Data[aStartPos]));
	case DTT_UI4:
		return (int) (*(uint*)&(this->Data[aStartPos]));
	case DTT_F8:
		return 0;
	case DTT_C16:
		return 0;
	case DTT_BINARY:
		return (int) (*(bool*)&(this->Data[aStartPos]));
	case DTT_RGBA_4UI1:
		return (int) (*(int*)&(this->Data[aStartPos]));*/
	}
	return 0;
}

float Dm4FileTag::GetSingleValueFloat(uint aType, uint aStartPos)
{
	switch (aType)
	{
		case DTTI2:
			return (float) (*(short*)&(this->Data[aStartPos]));
		case DTTI4:
			return (float) (*(int*)&(this->Data[aStartPos]));
		case DTTUI2:
			return (float) (*(ushort*)&(this->Data[aStartPos]));
		case DTTUI4:
			return (float) (*(uint*)&(this->Data[aStartPos]));
		case DTTF4:
			return (float) (*(float*)&(this->Data[aStartPos]));
		case DTTF8:
			return (float) (*(double*)&(this->Data[aStartPos]));
		case DTTI1:
			return (float) (*(char*)&(this->Data[aStartPos]));
		case DTTCHAR:
			return (float) (*(char*)&(this->Data[aStartPos]));
		case DTTI1_2:
			return (float) (*(char*)&(this->Data[aStartPos]));
		case DTTUI8:
			return (float) (*(ulong64*)&(this->Data[aStartPos]));
	/*case DTT_I2:
		return (float) (*(short*)&(this->Data[aStartPos]));
	case DTT_F4:
		return (float) (*(float*)&(this->Data[aStartPos]));
	case DTT_C8:
		return 0;
	case DTT_OBSOLETE:
		return 0;
	case DTT_C4:
		return 0;
	case DTT_UI1:
		return (float) (*(uchar*)&(this->Data[aStartPos]));
	case DTT_I4:
		return (float) (*(int*)&(this->Data[aStartPos]));
	case DTT_RGB_4UI1:
		return 0;
	case DTT_I1:
		return (float) (*(char*)&(this->Data[aStartPos]));
	case DTT_UI2:
		return (float) (*(ushort*)&(this->Data[aStartPos]));
	case DTT_UI4:
		return (float) (*(uint*)&(this->Data[aStartPos]));
	case DTT_F8:
		return (float) (*(double*)&(this->Data[aStartPos]));
	case DTT_C16:
		return 0;
	case DTT_BINARY:
		return (float) (*(bool*)&(this->Data[aStartPos]));
	case DTT_RGBA_4UI1:
		return 0;*/
	}
	return 0;
}

double Dm4FileTag::GetSingleValueDouble(uint aType, uint aStartPos)
{
	switch (aType)
	{
		case DTTI2:
			return (double) (*(short*)&(this->Data[aStartPos]));
		case DTTI4:
			return (double) (*(int*)&(this->Data[aStartPos]));
		case DTTUI2:
			return (double) (*(ushort*)&(this->Data[aStartPos]));
		case DTTUI4:
			return (double) (*(uint*)&(this->Data[aStartPos]));
		case DTTF4:
			return (double) (*(float*)&(this->Data[aStartPos]));
		case DTTF8:
			return (double) (*(double*)&(this->Data[aStartPos]));
		case DTTI1:
			return (double) (*(char*)&(this->Data[aStartPos]));
		case DTTCHAR:
			return (double) (*(char*)&(this->Data[aStartPos]));
		case DTTI1_2:
			return (double) (*(char*)&(this->Data[aStartPos]));
		case DTTUI8:
			return (double) (*(ulong64*)&(this->Data[aStartPos]));
	/*case DTT_I2:
		return (double) (*(short*)&(this->Data[aStartPos]));
	case DTT_F4:
		return (double) (*(float*)&(this->Data[aStartPos]));
	case DTT_C8:
		return 0;
	case DTT_OBSOLETE:
		return 0;
	case DTT_C4:
		return 0;
	case DTT_UI1:
		return (double) (*(uchar*)&(this->Data[aStartPos]));
	case DTT_I4:
		return (double) (*(int*)&(this->Data[aStartPos]));
	case DTT_RGB_4UI1:
		return 0;
	case DTT_I1:
		return (double) (*(char*)&(this->Data[aStartPos]));
	case DTT_UI2:
		return (double) (*(ushort*)&(this->Data[aStartPos]));
	case DTT_UI4:
		return (double) (*(uint*)&(this->Data[aStartPos]));
	case DTT_F8:
		return (double) (*(double*)&(this->Data[aStartPos]));
	case DTT_C16:
		return 0;
	case DTT_BINARY:
		return (double) (*(bool*)&(this->Data[aStartPos]));
	case DTT_RGBA_4UI1:
		return 0;*/
	}
	return 0;
}

string Dm4FileTag::GetSingleValueString(uint aType, uint aStartPos)
{
	switch (aType)
	{
	case DTTUI2:
	case DTTI1:
	case DTTCHAR:
	case DTTI1_2:
		if (tagType == ArrayTag)
		{
			if (GetSize(InfoArray[1]) == 1)
			{
				//Normal string
				string str;
				for (uint i = 0; i < SizeData; i+=1) 
					str += *(Data+i); 
				return str;
			}
			else
			{
				//Unicode string
				string str;
				for (uint i = 0; i < SizeData; i+=2) 
					str += *(Data+i); 
				return str;
			}
		}
	}
	return "";
}


ostream& operator<<(ostream &stream, Dm4FileTag& tag)
{
	uint bytesRead = 0;
	uint elemSize;

	if (tag.Name == "Name" || tag.Name == "Class Name" || tag.Name == "kind" || tag.Name == "Processing" || tag.Name == "Set Name"
		 || tag.Name == "Frame Combine Style" || tag.Name == "Parameter Set Name" || tag.Name == "Parameter Set Tag Path" || tag.Name == "ClassName"
		 || tag.Name == "Acquisition Date" || tag.Name == "Acquisition Time" || tag.Name == "Acquisition Time (OS)" || tag.Name == "Device Name"
		 || tag.Name == "Aperture label" || tag.Name == "Instrument name" || tag.Name == "Illumination Mode" || tag.Name == "Imaging Mode" 
		 || tag.Name == "Label" || tag.Name == "Tracking correction Method" || tag.Name == "CLUTName" || tag.Name == "Mode Name" || tag.Name == "View Name"
		 || tag.Name == "Viewer Class" || tag.Name == "Operation Mode" || tag.Name == "Tag path" || tag.Name == "Acquisition start" 
		 || tag.Name == "Acquisition finish" || tag.Name == "Tomography Modality Name" || tag.Name == "Source" || tag.Name == "Illumination mode"
		 || tag.Name == "Imaging optics mode" || tag.Name == "XY tracking algorithm" || tag.Name == "XY tracking device" || tag.Name == "Z tracking algorithm"
		 || tag.Name == "Z tracking device" || tag.Name == "Exposure time increasing model" || tag.Name == "Tilt direction" || tag.Name == "Tilt model" 
		 || tag.Name == "Tracking correction method")
	{
		if (tag.tagType == ArrayTag)
		{
			if (tag.GetSize(tag.InfoArray[1]) == 1)
			{
				//Normal string
				string str;
				for (uint i = 0; i < tag.SizeData; i+=1) 
					str += *(tag.Data+i); 
				stream << str;
			}
			else
			{
				//Unicode string
				string str;
				for (uint i = 0; i < tag.SizeData; i+=2) 
					str += *(tag.Data+i); 
				stream << str;
			}
			return stream;
		}
	}

	
	switch (tag.tagType)
	{
	case SingleEntryTag:
		tag.PrintSingleValue(stream, tag.InfoArray[0], 0);
		break;
	case StructTag:
		stream << "{";
		
		for (uint i = 0; i < tag.InfoArray[2]; i++)
		{
			tag.PrintSingleValue(stream, tag.InfoArray[i * 2 + 4], bytesRead);
			bytesRead += tag.GetSize(tag.InfoArray[i * 2 + 4]);
			if (i < tag.InfoArray[2] - 1) stream << "; ";
		}
		
		stream << "}";
		break;
	case ArrayTag:
		elemSize = tag.GetSize(tag.InfoArray[1]);
		stream << "[";
		
		if (tag.SizeData < 40)
			for (uint i = 0; i < tag.InfoArray[2]; i++)
			{
				tag.PrintSingleValue(stream, tag.InfoArray[1], bytesRead);
				bytesRead += elemSize;
				if (i < tag.InfoArray[2] - 1) stream << ", ";
			}
		else
			for (uint i = 0; i < 10; i++)
			{
				tag.PrintSingleValue(stream, tag.InfoArray[1], bytesRead);
				bytesRead += elemSize;
				if (i < tag.InfoArray[2] - 1) stream << ", ";
			}
		
		stream << "]";
		break;
	case ArrayStructTag:
		//That's only crap here...
		break;

	}

	return stream;
}


int Dm4FileTag::GetStructValueAsInt(uint aIndex)
{
	if (tagType != StructTag) return -1;

	if (aIndex > InfoArray[2]) return -1;

	uint bytesRead = 0;

	for (int i = 0; i < (int)aIndex-1; i++)
	{
		bytesRead += GetSize(InfoArray[i * 2 + 4]);
	}
	
	return GetSingleValueInt(InfoArray[aIndex * 2 + 4], bytesRead);
}

void Dm4FileTag::FreeData()
{
	delete[] Data;
	Data = NULL;
}
