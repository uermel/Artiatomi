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


#include "HostSinkElement2D.h"
#include <exception>


HostSinkElement2D::HostSinkElement2D() :
	mDestDataPointer(NULL), IFilterElementSink2D(1)
{
}

bool HostSinkElement2D::CanConnect(IFilterElementBase * aFilter)
{
	throw std::runtime_error("NotImplemented yet!");
}

BufferRequestSet HostSinkElement2D::GetBufferRequests()
{
	return mBufferRequests;
}

void HostSinkElement2D::Allocate()
{
	//Nothing to allocate
}

void HostSinkElement2D::Prepare()
{
	/*IFilterElementSource2D* source = dynamic_cast<IFilterElementSource2D*>(mPrecessor[0]);
	if (!source)
	{
		throw runtime_error("");
	}*/
//	mBuffer = source->GetOutputImageBuffer(mParameter.PrecessorBufferIndices[0]);
}

void HostSinkElement2D::SetComputeParameters(IComputeParameters * aParameters)
{
	HostSinkElement2DComputeParameters* compParams = dynamic_cast<HostSinkElement2DComputeParameters*>(aParameters);
	if (!compParams)
	{
		return;
	}
	mDestDataPointer = compParams->DestPointer;
}

void HostSinkElement2D::GetComputeParameters(IComputeParameters * aParameters)
{
	HostSinkElement2DComputeParameters* compParams = dynamic_cast<HostSinkElement2DComputeParameters*>(aParameters);
	if (!compParams)
	{
		return;
	}
	compParams->DestPointer = mDestDataPointer;
}

 void HostSinkElement2D::SetInputImageBuffer(std::shared_ptr<BufferRequest> aBuffer, size_t aIndex)
{
	if (aIndex != 0)
	{
		throw std::runtime_error("aIndex is out of bounds!");
	}
	mInputBuffer = aBuffer;
}

void HostSinkElement2D::Execute(FilterROI aRoi)
{
	ExecuteOne(aRoi);
}

void HostSinkElement2D::ExecuteOne(FilterROI aRoi)
{
	if (!mROI.Contains(aRoi))
	{
		throw std::runtime_error("ROI mismatch!");
	}

	size_t typeSize = GetDataTypeSize(mInputParameter[0].DataType);
	//pitched and ROI based copy:
	for (int y = aRoi.Top(); y <= aRoi.Bottom(); y++)
	{
		memcpy((char*)mDestDataPointer + y * mInputParameter[0].Size.width * typeSize + aRoi.Left() * typeSize,
			(char*)mInputBuffer->mPtr + y * mInputBuffer->mAllocatedPitch + aRoi.Left() * typeSize,
			aRoi.width * typeSize);
	}
}

void HostSinkElement2D::SetInputParameters(FilterParameter& aParameter, size_t aIndex)
{
	if (aIndex != 0)
	{
		throw std::runtime_error("aIndex is out of bounds!");
	}
	mInputParameter[0] = aParameter;
	mSize = aParameter.Size;
	mROI = FilterROI(0, 0, aParameter.Size);
}

FilterParameter HostSinkElement2D::GetOutputParameters(size_t aIndex)
{
	if (aIndex != 0)
	{
		throw std::runtime_error("aIndex is out of bounds!");
	}
	return mInputParameter[0];
}
