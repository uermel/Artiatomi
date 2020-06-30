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


#ifndef IFILTERELEMENT_H
#define IFILTERELEMENT_H

#include "../Basics/Default.h"
#include "../MemoryPool/BufferRequestSet.h"
#include "../MemoryPool/BufferRequest.h"
#include "../MemoryPool/MemoryPool.h"
#include "FilterROI.h"
#include "FilterSize.h"
#include "FilterParameter.h"
#include "FilterROI3D.h"
#include "FilterSize3D.h"
#include "FilterParameter3D.h"
#include "IComputeParameters.h"
#include <vector>

//! Base class for all Filter elements
class IFilterElementBase
{
protected:
	BufferRequestSet mBufferRequests;

public:
	//! Test if aFilter can connect as a successor to this filter element
	virtual bool CanConnect(IFilterElementBase* aFilter) = 0;
	//! Returns the BufferRequestSet of this Filter element necessary for operation
	virtual BufferRequestSet GetBufferRequests() = 0;

	//! Determines the size and type of memory allocations needed for this filter elemnt
	virtual void Allocate() = 0;

	//! Computes eventually some static data, not to be recomputed with each execution, e.g. a look-up table
	virtual void Prepare() = 0;

	//! Sets compute parameters for this filter, like filter strength, number of iteratios etc.
	virtual void SetComputeParameters(IComputeParameters* aParameters) = 0;
	virtual void GetComputeParameters(IComputeParameters* aParameters) = 0;
};

class IFilterElementSourceBase;

//! Management members for filter tree structure: The successor must delete precessors etc...
class IFilterElementSinkHiddenMembers
{
public:
	IFilterElementSinkHiddenMembers();

	std::vector<IFilterElementSourceBase*> mPrecessor;
	std::vector<std::pair<size_t, size_t> > mPrecessorIndices;
	size_t mPrecessorCounter;
	size_t AddPreviousFilter(IFilterElementSourceBase* aFilter);
	void RemovePreviousFilter(size_t aIndex);

	//! Returns a list of all directly connected previous filters
	virtual void GetPreviousFilters(size_t* count, IFilterElementSourceBase*** filterList);
};

//! Management members for filter tree structure: The successor must delete precessors etc...
class IFilterElementSinkBase : public IFilterElementSinkHiddenMembers
{
protected:
	FilterParameter* mInputParameter;

public:
	//! aInputCount determines the number of input buffers used by this filter
	IFilterElementSinkBase(size_t aInputCount);
	~IFilterElementSinkBase();

	//! Sets the buffer used as input buffer for input index aIndex
	virtual void SetInputImageBuffer(std::shared_ptr<BufferRequest> aBuffer, size_t aIndex) = 0;
};

//! A sink filter is a filter with at least one output buffer: buffer count is specified in the constructor
class IFilterElementSink : public IFilterElementBase, public IFilterElementSinkBase
{
public:
	IFilterElementSink(size_t aInputCount);
};

class IFilterElement;

//! A source filter is a filter with at least one input buffer: buffer count is specified in the constructor
class IFilterElementSourceBase
{
protected:
	FilterParameter* mOutputParameter;
	std::vector<IFilterElementBase*> mSuccessor;
	std::vector<std::pair<size_t, size_t> > mSuccessorIndices;
	size_t mSuccessorCounter;
	std::vector<std::pair<size_t, size_t> > mPrecessorIndexMapper;

public:
	//! aOutputCount determines the number of output buffers used by this filter
	IFilterElementSourceBase(size_t aOutputCount);
	~IFilterElementSourceBase();

	//! Returns a list of all directly connected successing filters
	virtual void GetNextFilters(size_t* count, IFilterElementBase*** filterList);
	//! Appends a succeeding filter to this instance, internally adds this instance as precessor to the succeding filter
	size_t AddNextFilter(IFilterElementSink* aFilter);
	//! Appends a succeeding filter to this instance, internally adds this instance as precessor to the succeding filter
	size_t AddNextFilter(IFilterElement* aFilter);
	//! Removes aIndex of the succeeding filters from this instance and also deletes the internal precessor of the succeeding filter.
	void RemoveNextFilter(size_t aIndex);

	//! Returns the buffer of the outputBuffer with index aIndex
	virtual std::shared_ptr<BufferRequest> GetOutputImageBuffer(size_t aIndex) = 0;

};



//! A source filter is a filter with at least one input buffer: buffer count is specified in the constructor
class IFilterElementSource : public IFilterElementBase, public IFilterElementSourceBase
{
public:
	//! aOutputCount determines the number of output buffers used by this filter
	IFilterElementSource(size_t aOutputCount);
};

//! A coombination of input and output filter: i.e. a default filter element that computes an output from some input buffers
class IFilterElement : public IFilterElementBase, public IFilterElementSourceBase, public IFilterElementSinkBase
{
public:
	//! aInputCount defines the number of input buffers, aOutputCount the number of outputs.
	IFilterElement(size_t aInputCount, size_t aOutputCount);
};

class IFilterElement2DBase
{
public:
	IFilterElement2DBase();

	//! Executes the actual filter, after termination the output buffer contains the computed values. Calls execute on all following filters.
	virtual void Execute(FilterROI aRoi) = 0;

	//! Executes the actual filter, after termination the output buffer contains the computed values.
	virtual void ExecuteOne(FilterROI aRoi) = 0;

	//! Sets the input parameters, such as image size and type
	virtual void SetInputParameters(FilterParameter& aParameter, size_t aIndex) = 0;
	//! Gets the output paramteres for output with index aIndex, depending on the input parameters.
	virtual FilterParameter GetOutputParameters(size_t aIndex) = 0;

protected:
	FilterSize mSize;
	FilterROI mROI;
};

//! Filter sink for two-dimensional image data
class IFilterElementSink2D : public IFilterElementSink, public IFilterElement2DBase
{
public:
	//! aInputCount determines the number of input buffers used by this filter
	IFilterElementSink2D(size_t aInputCount);
};

//! Source filter for two-dimensional image data
class IFilterElementSource2D : public IFilterElementSource, public IFilterElement2DBase
{
public:
	//! aOutputCount determines the number of output buffers used by this filter
	IFilterElementSource2D(size_t aOutputCount);
};

//! Filter for two-dimensional image data
class IFilterElement2D : public IFilterElement, public IFilterElement2DBase
{
public:
	//! aInputCount determines the number of input buffers, aOutputCount determines the number of output buffers used by this filter
	IFilterElement2D(size_t aInputCount, size_t aOutputCount);
};

//! Filter for three-dimensional image data
class IFilterElement3D : public IFilterElement
{
public:
	IFilterElement3D(size_t aInputCount, size_t aOutputCount);

	virtual void Execute(FilterROI3D aRoi) = 0;

	virtual void SetInputParameters(FilterParameter3D& aParameter, size_t aIndex) = 0;
	virtual FilterParameter3D GetOutputParameters(size_t aIndex) = 0;

protected:
	FilterSize3D mSize;
	FilterROI3D mROI;
};

#endif

