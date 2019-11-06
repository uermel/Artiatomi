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


#include "SimpleLogger.h"

SimpleLogger::SimpleLogger(string aFilename, SimpleLogLevel aLevel, bool aOff)
	:_filename(aFilename), _level(aLevel), _currentLevel(LOG_INFO), _off(aOff), _stream(!aOff ? aFilename.c_str() : ""),
	_isNewLine(true)
{
	if (!_off)
	{
		time_t     now = time(0);
		struct tm  tstruct;
		char       buf[80];
		tstruct = *localtime(&now);
	
		strftime(buf, sizeof(buf), "%A, %d. %B %Y - %X", &tstruct);

		_stream << "Reconstruction log file " << buf << endl;
	}
}

SimpleLogger::~SimpleLogger()
{
	if (!_off)
	{
		time_t     now = time(0);
		struct tm  tstruct;
		char       buf[80];
		tstruct = *localtime(&now);
	
		strftime(buf, sizeof(buf), "%A, %d. %B %Y - %X", &tstruct);

		_stream << "Log finished " << buf << endl << endl <<
			"--------------------------------------------------------------------------------" << endl << endl;
		_stream.close();
	}
}

void SimpleLogger::printLevel()
{
	if (_isNewLine)
	{
		switch (_currentLevel)
		{
		case LOG_ERROR:
			_stream << "[ERROR] ";
			break;
		case LOG_INFO:
			_stream << "[INFO]  ";
			break;
		case LOG_DEBUG:
			_stream << "[DEBUG] ";
			break;
		}
	}
	_isNewLine = false;
}

SimpleLogger& operator<<(SimpleLogger& logger, const SimpleLogger::SimpleLogLevel& level)
{
	if (!logger._off)
	{
		logger._currentLevel = level;
		logger._isNewLine = true;
	}
	return logger;
}

SimpleLogger& operator<<(SimpleLogger& logger, const string& val)
{
	if (logger._currentLevel >= logger._level && !logger._off)
	{
		logger.printLevel();
		logger._stream << val;
	}
	return logger;
}

SimpleLogger& operator<<(SimpleLogger& logger, const char* val)
{
	if (logger._currentLevel >= logger._level && !logger._off)
	{
		logger.printLevel();
		logger._stream << val;
	}
	return logger;
}

SimpleLogger& operator<<(SimpleLogger& logger, const FileDataType_enum& val)
{
	if (logger._currentLevel >= logger._level && !logger._off)
	{
		logger.printLevel();
		switch (val)
		{
		case FDT_UNKNOWN: 
			logger._stream << "UNKNOWN";
			break;
		case FDT_UCHAR: 
			logger._stream << "UCHAR";
			break;
		case FDT_USHORT: 
			logger._stream << "USHORT";
			break;
		case FDT_UINT: 
			logger._stream << "UINT32";
			break;
		case FDT_ULONG: 
			logger._stream << "UINT64";
			break;
		case FDT_FLOAT: 
			logger._stream << "FLOAT";
			break;
		case FDT_DOUBLE: 
			logger._stream << "DOUBLE";
			break;
		case FDT_CHAR: 
			logger._stream << "CHAR";
			break;
		case FDT_SHORT: 
			logger._stream << "SHORT";
			break;
		case FDT_INT: 
			logger._stream << "INT32";
			break;
		case FDT_LONG: 
			logger._stream << "INT64";
			break;
		case FDT_FLOAT2: 
			logger._stream << "FLOAT2";
			break;
		case FDT_SHORT2: 
			logger._stream << "SHORT2";
			break;
		}
	}
	return logger;
}

SimpleLogger& operator<<(SimpleLogger& logger, const int& val)
{
	if (logger._currentLevel >= logger._level && !logger._off)
	{
		logger.printLevel();
		logger._stream << val;
	}
	return logger;
}

SimpleLogger& operator<<(SimpleLogger& logger, bool val)
{
	if (logger._currentLevel >= logger._level && !logger._off)
	{
		logger.printLevel();
		logger._stream << val ? "true" : "false";
	}
	return logger;
}

SimpleLogger& operator<<(SimpleLogger& logger, const unsigned int& val)
{
	if (logger._currentLevel >= logger._level && !logger._off)
	{
		logger.printLevel();
		logger._stream << val;
	}
	return logger;
}

SimpleLogger& operator<<(SimpleLogger& logger, const float& val)
{
	if (logger._currentLevel >= logger._level && !logger._off)
	{
		logger.printLevel();
		logger._stream << val;
	}
	return logger;
}

SimpleLogger& operator<<(SimpleLogger& logger, const double& val)
{
	if (logger._currentLevel >= logger._level && !logger._off)
	{
		logger.printLevel();
		logger._stream << val;
	}
	return logger;
}

SimpleLogger& operator<<(SimpleLogger& logger, const dim3& val)
{
	if (logger._currentLevel >= logger._level && !logger._off)
	{
		logger.printLevel();
		logger._stream << "(" << val.x << ", " << val.y << ", " << val.z << ")";
	}
	return logger;
}

SimpleLogger& operator<<(SimpleLogger& logger, const int3& val)
{
	if (logger._currentLevel >= logger._level && !logger._off)
	{
		logger.printLevel();
		logger._stream << "(" << val.x << ", " << val.y << ", " << val.z << ")";
	}
	return logger;
}

SimpleLogger& operator<<(SimpleLogger& logger, const float3& val)
{
	if (logger._currentLevel >= logger._level && !logger._off)
	{
		logger.printLevel();
		logger._stream << "(" << val.x << ", " << val.y << ", " << val.z << ")";
	}
	return logger;
}

SimpleLogger& operator<<(SimpleLogger& logger, const int4& val)
{
	if (logger._currentLevel >= logger._level && !logger._off)
	{
		logger.printLevel();
		logger._stream << "(" << val.x << ", " << val.y << ", " << val.z << ", " << val.w << ")";
	}
	return logger;
}

SimpleLogger& operator<<(SimpleLogger& logger, const float4& val)
{
	if (logger._currentLevel >= logger._level && !logger._off)
	{
		logger.printLevel();
		logger._stream << "(" << val.x << ", " << val.y << ", " << val.z << ", " << val.w << ")";
	}
	return logger;
}

SimpleLogger& operator<<(SimpleLogger& logger, ostream& (*f)(std::ostream&))
{
    f(logger._stream);
	logger._isNewLine = true;
	return logger;
}