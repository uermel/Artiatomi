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


#include "MKLog.h"
#include <exception>

MKLog* MKLog::_instance = NULL;
std::mutex MKLog::_mutex;

MKLog::MKLog(std::string aFilename, logLevel_enum aLevel)
	: mLevel(aLevel), mLogFile(aFilename, std::ios_base::trunc)
{
	if (!mLogFile.good())
	{
		std::string message;
		message = "Could not open log file: " + aFilename;
		throw std::runtime_error(message.c_str());
	}
}

MKLog * MKLog::Get()
{
	//std::lock_guard<std::mutex> lock(_mutex);

	if (!_instance)
	{
		Init("Logfile.log");
		MKLOG("Logger not initialized! Using default log file...");
	}

	return _instance;
}

void MKLog::Init(std::string aFilename, logLevel_enum aLevel)
{
#ifdef _DEBUG
	std::lock_guard<std::mutex> lock(_mutex);

	if (!_instance)
	{
		_instance = new MKLog(aFilename, aLevel);
	}
	else
	{
		std::string message;
		message = "Logger already initilized!";
		throw std::runtime_error(message.c_str());
	}
#endif
}

void MKLog::SetLevel(logLevel_enum aLevel)
{
#ifdef _DEBUG
	std::lock_guard<std::mutex> lock(_mutex);

	mLevel = aLevel;
#endif
}

std::string MKLog::getLevelString(logLevel_enum aLevel)
{
	switch (aLevel)
	{
	case LL_DEBUG:
		return "[DEBUG] ";
	case LL_INFO:
		return "[INFO ] ";
	case LL_ERROR:
		return "[ERROR] ";
	case LL_OFF:
		return "[OFF  ] ";
	default:
		return std::string();
	}
}

void MKLog::Log(logLevel_enum aLevel, std::string aMessage)
{
#ifdef _DEBUG
	if (aLevel >= mLevel)
	{
		std::lock_guard<std::mutex> lock(_mutex);
		mLogFile << getLevelString(aLevel) << aMessage << std::endl;
	}
#endif
}

void MKLog::Log(logLevel_enum aLevel, const char * aMessage)
{
	Log(aLevel, std::string(aMessage));
}

template<typename ...Args>
inline void MKLog::Log(logLevel_enum aLevel, const char * fmt, const Args & ...args)
{
#ifdef _DEBUG
	if (aLevel >= mLevel)
	{
		char* buffer = new char[1024];
		snprintf(buffer, 1024, fmt, args...);
		Log(aLevel, buffer);
		delete[] buffer;
	}
#endif
}

//inline void MKLog::Log(logLevel_enum aLevel, const char * fmt, const char* args)
//{
//#ifdef _DEBUG
//	if (aLevel >= mLevel)
//	{
//		char* buffer = new char[1024];
//		snprintf(buffer, 1024, fmt, args);
//		Log(aLevel, buffer);
//		delete[] buffer;
//	}
//#endif
//}

template void MKLog::Log<unsigned long long>(logLevel_enum, const char*, const unsigned long long&);
template void MKLog::Log<long long>(logLevel_enum, const char*, const long long&);
template void MKLog::Log<unsigned long long, unsigned long long>(logLevel_enum, const char*, const unsigned long long&, const unsigned long long&);
template void MKLog::Log<long long, long long>(logLevel_enum, const char*, const long long&, const long long&);
template void MKLog::Log<long long, unsigned long long>(logLevel_enum, const char*, const long long&, const unsigned long long&);
template void MKLog::Log<unsigned long long, long long>(logLevel_enum, const char*, const unsigned long long&, const long long&);
template void MKLog::Log<unsigned long long, unsigned long long, unsigned long long>(logLevel_enum, const char*, const unsigned long long&, const unsigned long long&, const unsigned long long&);
template void MKLog::Log<int>(logLevel_enum, const char*, const int&);
template void MKLog::Log<uint>(logLevel_enum, const char*, const uint&);
template void MKLog::Log<int, int>(logLevel_enum, const char*, const int&, const int&);
template void MKLog::Log<char*>(logLevel_enum, const char*, char * const &);
template void MKLog::Log<uint, const char*, const char*>(logLevel_enum, const char*, const uint&, const char * const &, const char * const &);
template void MKLog::Log<char const*>(logLevel_enum, const char*, char const * const &);
template void MKLog::Log<uint, uint, uint, uint, uint, uint>(logLevel_enum, const char*, const uint&, const uint&, const uint&, const uint&, const uint&, const uint&);
template void MKLog::Log<char const *, char const *, char const *, char const *, char const *, char const *>(logLevel_enum, const char*, const char * const &, const char * const &, const char * const &, const char * const &, const char * const &, const char * const &);
template void MKLog::Log<char[19] >(logLevel_enum, const char*, char const (&array)[19]);
template void MKLog::Log<char[27] >(logLevel_enum, const char*, char const (&array)[27]);
template void MKLog::Log<char[23] >(logLevel_enum, const char*, char const (&array)[23]);
template void MKLog::Log<char[39] >(logLevel_enum, const char*, char const (&array)[39]);
template void MKLog::Log<char[26] >(logLevel_enum, const char*, char const (&array)[26]);
template void MKLog::Log<char[2], char[14], char[13]>(logLevel_enum, const char*, char const (&array)[2], char const (&array2)[14], char const (&array3)[13]);

template void MKLog::Log<int, float, float>(logLevel_enum, const char*, const int&, const float&, const float&);
template void MKLog::Log<int, float>(logLevel_enum, const char*, const int&, const float&);
template void MKLog::Log<float, int>(logLevel_enum, const char*, const float&, const int&);
template void MKLog::Log<int, int, int, int>(logLevel_enum, const char*, const int&, const int&, const int&, const int&);
template void MKLog::Log<int, int, int>(logLevel_enum, const char*, const int&, const int&, const int&);