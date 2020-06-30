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


// FilterGraphTest.cpp : Definiert den Einstiegspunkt für die Konsolenanwendung.
//

#include "stdafx.h"
#include "../FilterGraph/FilterROI.h"
#include "../FilterGraph/FilterSize.h"
#include "../FilterGraph/FilterPoint2D.h"

#include "../FilterGraph/HostSourceElement2D.h"
#include "../FilterGraph/HostSinkElement2D.h"
#include "../FilterGraph/HostSplitterElement2D.h"

int main()
{
	/*FilterPoint2D p(2, 3);
	FilterPoint2D p2 = p * 2;
	FilterPoint2D p3 = 3 * (p + p2);

	FilterSize s(10, 20);
	FilterSize s2;
	FilterSize s3(100, 200);

	s2 = s + s3;
	s *= 5;

	FilterROI r1(10, 20, 30, 40);
	FilterROI r2 = 10 * r1;
	FilterROI r3 = r2 - r1;
	FilterROI rett = r3;
	r3.Deflate(5);

	FilterROI small(50, 50, 10, 10);
	FilterROI big(0, 0, 100, 100);
	FilterPoint2D tp(60, 60);

	bool t1 = big.IntersectsWith(small);
	bool t2 = small.Contains(tp);*/
	int w = 10, h = 10;
	float* data = new float[w * h];
	float* data1 = new float[w * h];
	float* data2 = new float[w * h];
	for (size_t i = 0; i < w*h; i++)
	{
		data[i] = (float)i;
	}
	FilterROI roi;
	roi.x = 0;
	roi.y = 0;
	roi.width = w;
	roi.height = h;
	FilterParameter params;
	params.DataType = DT_FLOAT;
	params.Size = roi.Size();
	FilterROI roi2 = roi;
	roi2.x = 1;
	roi2.y = 1;
	roi2.width = 9;
	roi2.height = 9;


	HostSourceElement2DComputeParameters Cparams;
	HostSourceElement2D source;
	HostSplitterElement2D splitter;
	HostSinkElement2D sink1;
	HostSinkElement2D sink2;
	Cparams.SrcPointer = data;
	size_t idx2 = source.AddNextFilter(&splitter);
	size_t idx3 = splitter.AddNextFilter(&sink1);
	size_t idx4 = splitter.AddNextFilter(&sink2);

	HostSplitterElement2DComputeParameters CParamsS;

	HostSinkElement2DComputeParameters Cparams1;
	Cparams1.DestPointer = data1;
	HostSinkElement2DComputeParameters Cparams2;
	Cparams2.DestPointer = data2;

	source.SetInputParameters(params, 0);
	source.SetComputeParameters(&Cparams);
	source.Allocate();
	BufferRequestSet setSource = source.GetBufferRequests();
	std::shared_ptr<BufferRequest> req = source.GetOutputImageBuffer(0);

	MemoryPool::Get()->Allocate(setSource);

	splitter.SetComputeParameters(&CParamsS);
	splitter.SetInputParameters(params, 0);
	splitter.Allocate();
	BufferRequestSet setSplitter = splitter.GetBufferRequests();
	splitter.SetInputImageBuffer(req, 0);
	std::shared_ptr<BufferRequest> req1 = splitter.GetOutputImageBuffer(0);
	std::shared_ptr<BufferRequest> req2 = splitter.GetOutputImageBuffer(1);

	MemoryPool::Get()->Allocate(setSplitter);

	sink1.SetComputeParameters(&Cparams1);
	sink1.SetInputParameters(params, 0);
	sink1.Allocate();
	sink1.SetInputImageBuffer(req1, 0);

	sink2.SetComputeParameters(&Cparams2);
	sink2.SetInputParameters(params, 0);
	sink2.Allocate();
	sink2.SetInputImageBuffer(req2, 0);

	source.Prepare();
	source.Execute(roi2);

	/*splitter.Prepare();
	splitter.Execute(roi);

	sink1.Prepare();
	sink1.Execute(roi);

	sink2.Prepare();
	sink2.Execute(roi);*/
    return 0;
}

