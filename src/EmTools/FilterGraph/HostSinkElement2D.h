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


#ifndef HOSTSINKELEMENT2D_H
#define HOSTSINKELEMENT2D_H

#include "IFilterElement.h"

class HostSinkElement2DComputeParameters : public IComputeParameters
{
public:
	void* DestPointer;
};

class HostSinkElement2D : public IFilterElementSink2D
{
private:
	void* mDestDataPointer;
	std::shared_ptr<BufferRequest> mInputBuffer;

public:
	HostSinkElement2D();

	virtual bool CanConnect(IFilterElementBase* aFilter);
	virtual BufferRequestSet GetBufferRequests();

	virtual void Allocate();

	virtual void Prepare();

	virtual void SetComputeParameters(IComputeParameters* aParameters);
	virtual void GetComputeParameters(IComputeParameters* aParameters);

	virtual void SetInputImageBuffer(std::shared_ptr<BufferRequest> aBuffer, size_t aIndex);

	virtual void Execute(FilterROI aRoi);
	virtual void ExecuteOne(FilterROI aRoi);

	virtual void SetInputParameters(FilterParameter& aParameter, size_t aIndex);
	virtual FilterParameter GetOutputParameters(size_t aIndex);
};

#endif