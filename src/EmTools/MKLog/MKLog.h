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


#ifndef MKLOG_H
#define MKLOG_H

#include "../Basics/Default.h"
#include <mutex>
#include <string>
#include <fstream>

#ifdef _DEBUG
#define MKLOG(a, ...) (MKLog::Get()->Log(LL_DEBUG, a, __VA_ARGS__))
#define INITLOGGER(a,b) (MKLog::Init(a, b))
//#define INITLOGGER(a) (MKLog::Init(a))
#else
#define MKLOG(a, ...)
#define INITLOGGER(a,b)
//#define INITLOGGER(a)
#endif

enum logLevel_enum
{
	LL_DEBUG,
	LL_INFO,
	LL_ERROR,
	LL_OFF
};

class MKLog
{
public:
	static MKLog* Get();
	static void Init(std::string aFilename, logLevel_enum aLevel = LL_DEBUG);
	void SetLevel(logLevel_enum aLevel);
	void Log(logLevel_enum aLevel, const char* aMessage);
	void Log(logLevel_enum aLevel, std::string aMessage);
	template <typename... Args> void Log(logLevel_enum aLevel, const char* fmt, const Args&... args);


	//void Log(logLevel_enum aLevel, const char* fmt, const char* args);

private:
	static std::mutex _mutex;
	static MKLog* _instance;

	MKLog() = delete;
	MKLog(std::string aFilename, logLevel_enum aLevel = LL_DEBUG);
	MKLog(const MKLog&) = delete;
	MKLog(MKLog&&) = delete;

	logLevel_enum mLevel;
	std::ofstream mLogFile;

	std::string getLevelString(logLevel_enum aLevel);
};

#endif
