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


#include "SimpleFileList.h"
#include <fstream>
#include <iostream>

using namespace std;

SimpleFileList::SimpleFileList(std::string aFilename)
{
	_filename = aFilename;

	ifstream fstream(_filename);

	string line;
	vector<string> lines;

	while (getline(fstream, line))
	{
		//ignore empty lines
		if (line.length() > 3)
		{
			lines.push_back(line);
		}
	}
}

std::vector<std::string> SimpleFileList::GetEntries()
{
	return _entries;
}
