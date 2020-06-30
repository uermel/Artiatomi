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


#include "IFilterElement.h"



IFilterElementSinkBase::IFilterElementSinkBase(size_t aInputCount)
	: mInputParameter(new FilterParameter[aInputCount])
{
}

IFilterElementSinkBase::~IFilterElementSinkBase()
{
	if (mInputParameter)
		delete[] mInputParameter;
	mInputParameter = NULL;
}

void IFilterElementSinkHiddenMembers::GetPreviousFilters(size_t * count, IFilterElementSourceBase *** filterList)
{
	if (mPrecessor.size() > 0)
	{
		*count = mPrecessor.size();
		*filterList = &mPrecessor[0];
	}
	else
	{
		*count = 0;
		*filterList = NULL;
	}
}

IFilterElementSinkHiddenMembers::IFilterElementSinkHiddenMembers(): mPrecessorCounter(0)
{
}

size_t IFilterElementSinkHiddenMembers::AddPreviousFilter(IFilterElementSourceBase * aFilter)
{
	mPrecessor.push_back(aFilter);
	mPrecessorIndices.push_back(std::pair<size_t, size_t>(mPrecessorCounter, mPrecessor.size() - 1));
	mPrecessorCounter++;
	return mPrecessorCounter - 1;
}

IFilterElementSourceBase::IFilterElementSourceBase(size_t aOutputCount)
	: mOutputParameter(new FilterParameter[aOutputCount]), mSuccessorCounter(0)
{
}

IFilterElementSourceBase::~IFilterElementSourceBase()
{
	if (mOutputParameter)
		delete[] mOutputParameter;
	mOutputParameter = NULL;
}

void IFilterElementSourceBase::GetNextFilters(size_t * count, IFilterElementBase *** filterList)
{
	if (mSuccessor.size() > 0)
	{
		*count = mSuccessor.size();
		*filterList = &mSuccessor[0];
	}
	else
	{
		*count = 0;
		*filterList = NULL;
	}
}

size_t IFilterElementSourceBase::AddNextFilter(IFilterElementSink * aFilter)
{
	mSuccessor.push_back(aFilter);
	mSuccessorIndices.push_back(std::pair<size_t, size_t>(mSuccessorCounter, mSuccessor.size() - 1));
	size_t addedIndex = aFilter->AddPreviousFilter(this);
	mPrecessorIndexMapper.push_back(std::pair<size_t, size_t>(mSuccessorCounter, addedIndex));
	mSuccessorCounter++;
	return mSuccessorCounter - 1;
}

size_t IFilterElementSourceBase::AddNextFilter(IFilterElement * aFilter)
{
	mSuccessor.push_back(aFilter);
	mSuccessorIndices.push_back(std::pair<size_t, size_t>(mSuccessorCounter, mSuccessor.size() - 1));
	size_t addedIndex = aFilter->AddPreviousFilter(this);
	mPrecessorIndexMapper.push_back(std::pair<size_t, size_t>(mSuccessorCounter, addedIndex));
	mSuccessorCounter++;
	return mSuccessorCounter - 1;
}

void IFilterElementSinkHiddenMembers::RemovePreviousFilter(size_t aIndex)
{
	size_t indexToRemove = 0;
	bool found = 0;
	for (size_t i = 0; i < mPrecessorIndices.size(); i++)
	{
		if (mPrecessorIndices[i].first == aIndex)
		{
			indexToRemove = mPrecessorIndices[i].second;
			mPrecessorIndices.erase(mPrecessorIndices.begin() + i);
			found = true;
			break;
		}
	}

	if (!found)
		throw std::runtime_error("Index out of boundaries!");

	for (size_t i = 0; i < mPrecessorIndices.size(); i++)
	{
		if (mPrecessorIndices[i].second > indexToRemove)
		{
			mPrecessorIndices[i].second--;
		}
	}

	mPrecessor.erase(mPrecessor.begin() + indexToRemove);
}

void IFilterElementSourceBase::RemoveNextFilter(size_t aIndex)
{
	size_t indexToRemove = 0;
	bool found = false;
	for (size_t i = 0; i < mSuccessorIndices.size(); i++)
	{
		if (mSuccessorIndices[i].first == aIndex)
		{
			indexToRemove = mSuccessorIndices[i].second;
			mSuccessorIndices.erase(mSuccessorIndices.begin() + i);
			found = true;
			break;
		}
	}

	if (!found)
		throw std::runtime_error("Index out of boundaries!");

	size_t precessor;
	for (size_t i = 0; i < mPrecessorIndexMapper.size(); i++)
	{
		if (mPrecessorIndexMapper[i].first == aIndex)
		{
			precessor = mPrecessorIndexMapper[i].second;
			mPrecessorIndexMapper.erase(mPrecessorIndexMapper.begin() + i);
		}
	}

	for (size_t i = 0; i < mSuccessorIndices.size(); i++)
	{
		if (mSuccessorIndices[i].second > indexToRemove)
		{
			mSuccessorIndices[i].second--;
		}
	}

	IFilterElementSinkHiddenMembers* temp = dynamic_cast<IFilterElementSinkHiddenMembers*>(mSuccessor[indexToRemove]);
	if (temp != NULL) //should NEVER be NULL!
	{
		temp->RemovePreviousFilter(precessor);
	}
	
	mSuccessor.erase(mSuccessor.begin() + indexToRemove);
}

IFilterElement::IFilterElement(size_t aInputCount, size_t aOutputCount)
	: IFilterElementSinkBase(aInputCount), IFilterElementSourceBase(aOutputCount)
{
}

IFilterElementSink2D::IFilterElementSink2D(size_t aInputCount)
	: IFilterElementSink(aInputCount)
{
}

IFilterElementSource2D::IFilterElementSource2D(size_t aOutputCount)
	: IFilterElementSource(aOutputCount)
{
}

IFilterElement2D::IFilterElement2D(size_t aInputCount, size_t aOutputCount)
	: IFilterElement(aInputCount, aOutputCount)
{
}

IFilterElement3D::IFilterElement3D(size_t aInputCount, size_t aOutputCount)
	: IFilterElement(aInputCount, aOutputCount), mSize(0, 0, 0), mROI(0, 0, 0, 0, 0, 0)
{
}

IFilterElementSink::IFilterElementSink(size_t aInputCount)
	: IFilterElementSinkBase(aInputCount)
{
}

IFilterElementSource::IFilterElementSource(size_t aOutputCount) :
	IFilterElementSourceBase(aOutputCount)
{
}

IFilterElement2DBase::IFilterElement2DBase()
	: mSize(0,0), mROI(0, 0, 0, 0)
{
}
