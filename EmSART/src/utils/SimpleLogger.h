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


#ifndef SIMPLELOGGER_H
#define SIMPLELOGGER_H

#include "UtilsDefault.h"
#include "../io/FileIOException.h"
#include <ctime>

using namespace std;

class SimpleLogger
{
public:
	enum SimpleLogLevel { LOG_QUIET, LOG_ERROR, LOG_INFO, LOG_DEBUG };
	SimpleLogger(string aFilename, SimpleLogLevel aLevel, bool aOff);
	~SimpleLogger();

	friend SimpleLogger& operator<<(SimpleLogger& logger, const SimpleLogger::SimpleLogLevel& level);
	friend SimpleLogger& operator<<(SimpleLogger& logger, const char* val);
	friend SimpleLogger& operator<<(SimpleLogger& logger, const string& val);
	friend SimpleLogger& operator<<(SimpleLogger& logger, const int& val);
	friend SimpleLogger& operator<<(SimpleLogger& logger, bool val);
	friend SimpleLogger& operator<<(SimpleLogger& logger, const unsigned int& val);
	friend SimpleLogger& operator<<(SimpleLogger& logger, const float& val);
	friend SimpleLogger& operator<<(SimpleLogger& logger, const double& val);
	friend SimpleLogger& operator<<(SimpleLogger& logger, const dim3& val);
	friend SimpleLogger& operator<<(SimpleLogger& logger, const int3& val);
	friend SimpleLogger& operator<<(SimpleLogger& logger, const int4& val);
	friend SimpleLogger& operator<<(SimpleLogger& logger, const float3& val);
	friend SimpleLogger& operator<<(SimpleLogger& logger, const float4& val);
	friend SimpleLogger& operator<<(SimpleLogger& logger, const FileDataType_enum& val);
	friend SimpleLogger& operator<<(SimpleLogger& logger, ostream& (*f)(std::ostream&));

private:
	void printLevel();
	string _filename;
	SimpleLogLevel _level;
	SimpleLogLevel _currentLevel;
	ofstream _stream;
	bool _off;
	bool _isNewLine;
};

SimpleLogger& operator<<(SimpleLogger& logger, const SimpleLogger::SimpleLogLevel& level);
SimpleLogger& operator<<(SimpleLogger& logger, const char* val);
SimpleLogger& operator<<(SimpleLogger& logger, const string& val);
SimpleLogger& operator<<(SimpleLogger& logger, bool val);
SimpleLogger& operator<<(SimpleLogger& logger, const int& val);
SimpleLogger& operator<<(SimpleLogger& logger, const unsigned int& val);
SimpleLogger& operator<<(SimpleLogger& logger, const float& val);
SimpleLogger& operator<<(SimpleLogger& logger, const double& val);
SimpleLogger& operator<<(SimpleLogger& logger, const dim3& val);
SimpleLogger& operator<<(SimpleLogger& logger, const int3& val);
SimpleLogger& operator<<(SimpleLogger& logger, const int4& val);
SimpleLogger& operator<<(SimpleLogger& logger, const float3& val);
SimpleLogger& operator<<(SimpleLogger& logger, const float4& val);
SimpleLogger& operator<<(SimpleLogger& logger, const FileDataType_enum& val);
SimpleLogger& operator<<(SimpleLogger& logger, ostream& (*f)(std::ostream&) );

#endif