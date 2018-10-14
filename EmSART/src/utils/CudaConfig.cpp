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


#include "CudaConfig.h"
#include "log.h"

using namespace std;

namespace Configuration
{
	CudaConfig::CudaConfig(string aName, string aParentDebugInfo) {
		mDebugInfo = aParentDebugInfo + ", " + aName;
	}

	CudaConfig::CudaConfig(string aConfigFile, int argc, char** argv, char** appEnvp)
		:mConfigFileName(aConfigFile),
			FPKernelMaxReg(0),
			FPKernelName(),
			FPBlockSize(0),
			CompKernelMaxReg(0),
			CompKernelName(),
			CompBlockSize(0),
			BPKernelMaxReg(0),
			BPKernelName(),
			BPBlockSize(0),
			CompilerOutput(false),
			InfoOutput(false)
	{
		while (appEnvp && *appEnvp) {
			string envEntry = *appEnvp;
			size_t pos = envEntry.find('=');
			if (pos != string::npos) {
				string name = envEntry.substr(0, pos);
				string value = envEntry.substr(pos+1, string::npos);
				envSymbols[name] = value;
				logDebug(cout << "environment symbol: '" << name << "' = '" << value << "'" << endl);
			}
			++appEnvp;
		}


		mDebugInfo = aConfigFile;
		groupStack.push_front(this);

		ifstream in(aConfigFile.c_str());
		if (!in.good())
		{
			ConfigFileException ex(aConfigFile);
			throw ex;
		}

		char buff[1024];
		while (!in.eof())
		{
			in.getline(buff, 1024);
			string line=buff;
			if ( (line.length() > 2) && (line[0] != '#') && (line.find(')') == string::npos) ) {
				string name;
				string value;
				split(line, name, value, '=');

				if (value == "(") {
					logDebug(cout << "   config: new group '" << name << "'" << endl);
					CudaConfig* newGroup = new CudaConfig(name, mDebugInfo);
					groupStack.front()->groups[name] = newGroup;
					groupStack.push_front(newGroup);
				} else {
					for (list<CudaConfig*>::reverse_iterator i = groupStack.rbegin(); i != groupStack.rend(); ++i) {
						(*i)->symbolExpand(value);
					}
					envSymbolExpand(value);
					logDebug(cout << "   config: name = '" << name << "', value = '" << value << "'" << endl);
					groupStack.front()->add(name, value);
				}
			}
			if ( (line.length() > 0) && (line[0] != '#') && (line.find(')') != string::npos) ) {
				logDebug(cout << "   end of group" << endl);
				groupStack.pop_front();
			}
		}
		in.close();

		FPKernelMaxReg = GetInt("FPKernelMaxReg");
		FPKernelName = "march";
		FPBlockSize = GetDim3("FPBlockSize");

		SlicerKernelMaxReg = GetInt("SlicerKernelMaxReg");
		SlicerKernelName = "slicer";
		SlicerBlockSize = GetDim3("SlicerBlockSize");

		VolTravLenKernelMaxReg = GetInt("VolTravLenKernelMaxReg");
		VolTravLenKernelName = "volTraversalLength";
		VolTravLenBlockSize = GetDim3("VolTravLenBlockSize");

		CompKernelMaxReg = GetInt("CompKernelMaxReg");
		CompKernelName = "compare";
		CompBlockSize = GetDim3("CompBlockSize");

		BPKernelMaxReg = GetInt("BPKernelMaxReg");
		BPKernelName = "backProjection";
		BPBlockSize = GetDim3("BPBlockSize");

		CompilerOutput = GetBool("CompilerOuput");
		InfoOutput = GetBool("InfoOuput");
	}

    CudaConfig* CudaConfig::config = NULL;
	CudaConfig& CudaConfig::GetConfig(string aConfigFile, int argc, char** argv, char** appEnvp)
	{
            if (config == NULL)
            {
                    config = new CudaConfig(aConfigFile, argc, argv, appEnvp);
            }
            return *config;
    }

    CudaConfig& CudaConfig::GetConfig()
    {
            return *config;
    }

	CudaConfig::~CudaConfig() {
		for (map<string, CudaConfig*>::iterator i = groups.begin(); i != groups.end(); ++i) {
			delete i->second;
		}
	}

	void CudaConfig::add(string aName, string aValue) {
		symbols[aName] = aValue;
	}

	void CudaConfig::split(string in, string& left, string& right, char c) {
		size_t pos = in.find_first_of(c);
		if(pos == string::npos) {
			left = in;
			trim(left);
			right = "";
		} else if (pos <= 1) {
			left = "";
			right = in.substr(pos+1, string::npos);
			trim(right);
		} else {
			left = in.substr(0, pos);
			trim(left);
			right = in.substr(pos+1, string::npos);
			trim(right);
		}
	}

	void CudaConfig::trim(string& s) {
		while ( (s.length() > 1) && ( (s[0] == ' ') || (s[0] =='\t') ) ) {
			s = s.substr(1, string::npos);
		}
		while ( (s.length() > 1) &&
				( (s[s.length()-1] == ' ') ||
				  (s[s.length()-1] == '\t') ||
				  (s[s.length()-1] == '\n') ||
				  (s[s.length()-1] == '\r') ) ) {
			s = s.substr(0, s.length()-1);
		}
		if ( (s.length() > 1) && (s[0] == '"') ) {
			s = s.substr(1, string::npos);
		}
		if ( (s.length() > 1) && (s[s.length()-1] == '"') ) {
			s = s.substr(0, s.length()-1);
		}
	}

	void CudaConfig::symbolExpand(string& s) {
		symbolExpand(symbols, s);
	}

	void CudaConfig::envSymbolExpand(string& s) {
		symbolExpand(envSymbols, s);
	}

	void CudaConfig::symbolExpand(map<string, string>& symbols, string& s) {
		bool expanded;
		do {
			expanded = false;
			for (map<string, string>::iterator i = symbols.begin(); i != symbols.end(); ++i) {
				string search = "%" + i->first + "%";
				string replace = i->second;
				size_t pos = s.find(search);
				if (pos != string::npos) {
					expanded = true;
					s.replace(pos, search.length(), replace);
				}
			}
		} while (expanded);
	}

	string CudaConfig::GetString(string aName) {
		map<string, string>::iterator i = symbols.find(aName);
		if (i == symbols.end()) {
			logError(cout << "access of missing property '" << aName << "' (" << mDebugInfo << ")" << endl);
			ConfigPropertyException ex(mConfigFileName, aName);
			throw ex;
			//exit(4);
		}
		return i->second;
	}

	bool CudaConfig::GetBool(string aName) {
		string val = GetString(aName);

		if ( (val == "yes") ||
			 (val == "Yes") ||
			 (val == "YES") ||
			 (val == "true") ||
			 (val == "True") ||
			 (val == "TRUE"))
		{
			return true;
		}

		return false;
	}

	double CudaConfig::GetDouble(string aName) {
		string val = GetString(aName);
		stringstream ss(val);
		double retVal = 0;
		if ((ss >> retVal).fail())
		{
			ConfigValueException ex(mConfigFileName, aName, "Double");
			throw ex;
		}
		return retVal;
	}

	float CudaConfig::GetFloat(string aName) {
		string val = GetString(aName);
		stringstream ss(val);
		float retVal = 0;
		if ((ss >> retVal).fail())
		{
			ConfigValueException ex(mConfigFileName, aName, "Float");
			throw ex;
		}
		return retVal;
	}

	int CudaConfig::GetInt(string aName) {
		string val = GetString(aName);
		stringstream ss(val);
		int retVal = 0;
		if ((ss >> retVal).fail())
		{
			ConfigValueException ex(mConfigFileName, aName, "Integer");
			throw ex;
		}
		return retVal;
	}

	void CudaConfig::replaceChar(string& str, char replace, char by)
	{
		size_t size = str.size();

		for (size_t i = 0; i < size; i++)
		{
			if (str[i] == replace) str[i] = by;
		}
	}

	dim3 CudaConfig::GetDim3(string aName) {
		string val = GetString(aName);
		replaceChar(val, '(', ' ');
		replaceChar(val, '[', ' ');
		replaceChar(val, '{', ' ');
		replaceChar(val, '.', ' ');
		replaceChar(val, ',', ' ');
		replaceChar(val, ';', ' ');
		replaceChar(val, ')', ' ');
		replaceChar(val, ']', ' ');
		replaceChar(val, '}', ' ');
		stringstream ss(val);
		dim3 retVal;
		uint temp;
		if ((ss >> temp).fail())
		{
			ConfigValueException ex(mConfigFileName, aName, "dim3");
			throw ex;
		}
		retVal.x = temp;
		if ((ss >> temp).fail())
		{
			ConfigValueException ex(mConfigFileName, aName, "dim3");
			throw ex;
		}
		retVal.y = temp;
		if ((ss >> temp).fail())
		{
			ConfigValueException ex(mConfigFileName, aName, "dim3");
			throw ex;
		}
		retVal.z = temp;
		return retVal;
	}

	float3 CudaConfig::GetFloat3(string aName) {
		string val = GetString(aName);
		replaceChar(val, '(', ' ');
		replaceChar(val, '[', ' ');
		replaceChar(val, '{', ' ');
		replaceChar(val, '.', ' ');
		replaceChar(val, ',', ' ');
		replaceChar(val, ';', ' ');
		replaceChar(val, ')', ' ');
		replaceChar(val, ']', ' ');
		replaceChar(val, '}', ' ');
		stringstream ss(val);
		float3 retVal;
		float temp;
		if ((ss >> temp).fail())
		{
			ConfigValueException ex(mConfigFileName, aName, "float3");
			throw ex;
		}
		retVal.x = temp;
		if ((ss >> temp).fail())
		{
			ConfigValueException ex(mConfigFileName, aName, "float3");
			throw ex;
		}
		retVal.y = temp;
		if ((ss >> temp).fail())
		{
			ConfigValueException ex(mConfigFileName, aName, "float3");
			throw ex;
		}
		retVal.z = temp;
		return retVal;
	}
} //end namespace Configuration
