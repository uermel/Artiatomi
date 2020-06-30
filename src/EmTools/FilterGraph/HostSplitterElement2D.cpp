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


#include "HostSplitterElement2D.h"
#include <exception>


HostSplitterElement2D::HostSplitterElement2D()
	: IFilterElement2D(1, 2)
{
}

bool HostSplitterElement2D::CanConnect(IFilterElementBase * aFilter)
{
	throw std::runtime_error("NotImplemented yet!");
}

BufferRequestSet HostSplitterElement2D::GetBufferRequests()
{
	return mBufferRequests;
}

void HostSplitterElement2D::Allocate()
{
	MemoryPool::Get()->FreeAllocations(mBufferRequests);
	mBufferRequests.Clear();

	//We need a second buffer for the second output:
	mOuput2Buffer = std::make_shared<BufferRequest>(BT_DefaultHost, mOutputParameter[1].DataType, mOutputParameter[1].Size.width, mOutputParameter[1].Size.height);
	mBufferRequests.AddBufferRequest(mOuput2Buffer);
}

void HostSplitterElement2D::Prepare()
{
	//Nothing to prepare, propagate to successors:
	for (auto elem : mSuccessor)
	{
		elem->Prepare();
	}
}

void HostSplitterElement2D::SetComputeParameters(IComputeParameters * aParameters)
{
	HostSplitterElement2DComputeParameters* compParams = dynamic_cast<HostSplitterElement2DComputeParameters*>(aParameters);
	if (!compParams)
	{
		return;
	}
	//mSrcDataPointer = compParams->SrcPointer;
}

void HostSplitterElement2D::GetComputeParameters(IComputeParameters * aParameters)
{
	HostSplitterElement2DComputeParameters* compParams = dynamic_cast<HostSplitterElement2DComputeParameters*>(aParameters);
	if (!compParams)
	{
		return;
	}
	//compParams->SrcPointer = mSrcDataPointer;
}

void HostSplitterElement2D::SetInputImageBuffer(std::shared_ptr<BufferRequest> aBuffer, size_t aIndex)
{
	if (aIndex != 0)
	{
		throw std::runtime_error("aIndex is out of bounds!");
	}
	mInputOuput1Buffer = aBuffer;
}

std::shared_ptr<BufferRequest> HostSplitterElement2D::GetOutputImageBuffer(size_t aIndex)
{
	if (aIndex > 1)
	{
		throw std::runtime_error("aIndex is out of bounds!");
	}
	if (aIndex == 0)
	{
		return mInputOuput1Buffer;
	}
	else
	{
		return mOuput2Buffer;
	}
}

void HostSplitterElement2D::Execute(FilterROI aRoi)
{
	ExecuteOne(aRoi);

	//propagate to successors:
	for (auto elem : mSuccessor)
	{
		IFilterElement2DBase* temp = dynamic_cast<IFilterElement2DBase*>(elem);
		if (temp)
		{
			temp->Execute(aRoi);
		}
	}
}

void HostSplitterElement2D::ExecuteOne(FilterROI aRoi)
{
	if (!mROI.Contains(aRoi))
	{
		throw std::runtime_error("ROI mismatch!");
	}

	size_t typeSize = GetDataTypeSize(mOutputParameter[0].DataType);
	//pitched and ROI based copy:
	for (int y = aRoi.Top(); y <= aRoi.Bottom(); y++)
	{
		memcpy((char*)mInputOuput1Buffer->mPtr + y * mInputOuput1Buffer->mAllocatedPitch + aRoi.Left() * typeSize,
			(char*)mOuput2Buffer->mPtr + y * mOuput2Buffer->mAllocatedPitch + aRoi.Left() * typeSize,
			aRoi.width * typeSize);
	}
}

void HostSplitterElement2D::SetInputParameters(FilterParameter& aParameter, size_t aIndex)
{
	if (aIndex != 0)
	{
		throw std::runtime_error("aIndex is out of bounds!");
	}
	//Both outputs are identical to input...
	mInputParameter[0] = aParameter;
	mOutputParameter[0] = aParameter;
	mOutputParameter[1] = aParameter;
	mSize = aParameter.Size;
	mROI = FilterROI(0, 0, aParameter.Size);
}

FilterParameter HostSplitterElement2D::GetOutputParameters(size_t aIndex)
{
	if (aIndex > 1)
	{
		throw std::runtime_error("aIndex is out of bounds!");
	}
	return mOutputParameter[aIndex];
}

