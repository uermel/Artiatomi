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


// MemoryPoolTest.cpp : Definiert den Einstiegspunkt für die Konsolenanwendung.
//

#include "stdafx.h"
#include "../MemoryPool/MemoryPool.h"
#include "../MKLog/MKLog.h"

using namespace std;

int main()
{

	MKLog::Init("C:\\Users\\Michael Kunz\\Source\\Repos\\EmTools\\myLogFile.log");




	//BufferRequest* hbr;
	{
		BufferRequestSet brs1;
		BufferRequestSet brs2;
		{
			shared_ptr<BufferRequest> host1 = make_shared<BufferRequest>(BT_DefaultHost, DT_FLOAT, 100);
			shared_ptr<BufferRequest> host2 = make_shared<BufferRequest>(BT_DefaultHost, DT_FLOAT, 200);
			shared_ptr<BufferRequest> host3 = make_shared<BufferRequest>(BT_DefaultHost, DT_FLOAT, 300);
			shared_ptr<BufferRequest> host4 = make_shared<BufferRequest>(BT_DefaultHost, DT_SHORT, 101);

			brs1.AddBufferRequest(host1);
			brs1.AddBufferRequest(host3);
			brs1.AddBufferRequest(host2);
			brs1.AddBufferRequest(host4);
			MKLOG("First allocation round:");
			MemoryPool::Get()->Allocate(brs1);
		}

		{
			shared_ptr<BufferRequest> host1 = make_shared<BufferRequest>(BT_DefaultHost, DT_FLOAT, 100);
			shared_ptr<BufferRequest> host3 = make_shared<BufferRequest>(BT_DefaultHost, DT_FLOAT, 300);
			shared_ptr<BufferRequest> host2 = make_shared<BufferRequest>(BT_DefaultHost, DT_FLOAT, 200);

			brs2.AddBufferRequest(host1);
			brs2.AddBufferRequest(host2);
			brs2.AddBufferRequest(host3);

			MKLOG("Second allocation round:");
			MemoryPool::Get()->Allocate(brs2);
		}

		MKLOG("First free round:");
		MemoryPool::Get()->FreeAllocations(brs1);
		MKLOG("Second free round:");
		MemoryPool::Get()->FreeAllocations(brs2);

		printf("");
	}
    return 0;
}

