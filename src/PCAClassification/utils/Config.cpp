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


#include "Config.h"

using namespace std;

namespace Configuration
{
	Config::Config(string aName, string aParentDebugInfo) {
		mDebugInfo = aParentDebugInfo + ", " + aName;
	}

	Config::Config(string aConfigFile, int argc, char** argv, int mpiPart, char** appEnvp)
		:mConfigFileName(aConfigFile),
		logAllowed(mpiPart == 0),
		CudaDeviceIDs(),
		MotiveList(),
		ScaleMotivelistShift(1.0f),
		Particles(),
		NamingConv(MotiveList::NamingConvention_enum::NC_ParticleOnly),
		WedgeFile(),
		SingleWedge(false),
		Mask(),
		LowPass(0),
		HighPass(0),
		Sigma(0),
		FilterFileName(),
		UseFilterVolume(false),
		NumberOfClasses(-1),
		NumberOfEigenVectors(-1),
		BlockSize(1),
		ComputeCovVarMatrix(true),
		ComputeAverageParticle(true)
	{
		while (appEnvp && *appEnvp) {
			string envEntry = *appEnvp;
			size_t pos = envEntry.find('=');
			if (pos != string::npos) {
				string name = envEntry.substr(0, pos);
				string value = envEntry.substr(pos+1, string::npos);
				envSymbols[name] = value;
			}
			++appEnvp;
		}

		for (int i = 1; i < argc; i++)
		{
			string str(argv[i]);
			if (str == "-u")
			{
				aConfigFile = string(argv[i+1]);
				mConfigFileName = aConfigFile;
				i++;
			}
			else if (str == "-c")
			{
				stringstream ss(argv[i+1]);
				ss >> NumberOfClasses;
				i++;
			}
			else if (str == "-e")
			{
				stringstream ss(argv[i+1]);
				ss >> NumberOfEigenVectors;
				i++;
			}
			else
			{
				if (logAllowed) 
				{
					cout << endl;
					cout << "Usage: " << argv[0] << endl;
					cout << "    The following optional options override the configuration file:" << endl;
					cout << "    Options: " << endl;
					cout << "    -c VALUE:   Number of classes." << endl;
					cout << "    -e VALUE:   Number of eigen vectors." << endl;
					cout << "    -h:         Show this text." << endl;


					cout << ("\nPress <Enter> to exit...");
				}
				char c = cin.get();
				exit(-1);
			}
		}
        if (logAllowed)  printf("[%s] ... ", aConfigFile.c_str());
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
					Config* newGroup = new Config(name, mDebugInfo);
					groupStack.front()->groups[name] = newGroup;
					groupStack.push_front(newGroup);
				} else {
					for (list<Config*>::reverse_iterator i = groupStack.rbegin(); i != groupStack.rend(); ++i) {
						(*i)->symbolExpand(value);
					}
					envSymbolExpand(value);
					groupStack.front()->add(name, value);
				}
			}
			if ( (line.length() > 0) && (line[0] != '#') && (line.find(')') != string::npos) ) {
				groupStack.pop_front();
			}
		}
		in.close();

		CudaDeviceIDs = GetVectorInt("CudaDeviceID");

		MotiveList = GetString("MotiveList");
		ScaleMotivelistShift = GetFloat("ScaleMotivelistShift");
		WedgeFile = GetString("WedgeFile");
		SingleWedge = GetBool("SingleWedge");
		Particles = GetString("Particles");
		Mask = GetString("Mask");

		LowPass = GetInt("LowPass");
		HighPass = GetInt("HighPass");
		Sigma = GetInt("Sigma");

		string nc = GetStringOptional("NamingConvention");
		NamingConv = MotiveList::NamingConvention_enum::NC_ParticleOnly;
		if (nc == "TomoParticle" || nc == "Tomo_Particle")
		{
			NamingConv = MotiveList::NamingConvention_enum::NC_TomogramParticle;
		}

		if (NumberOfClasses < 0)
		{
			NumberOfClasses = GetInt("NumberOfClasses");
		}

		if (NumberOfEigenVectors < 0)
		{
			NumberOfEigenVectors = GetInt("NumberOfEigenVectors");
		}

		UseFilterVolume = GetBool("UseFilterVolume", false);

		if (UseFilterVolume)
		{
			FilterFileName = GetString("FilterFileName");
		}

		ComputeCovVarMatrix = GetBool("ComputeCovVarMatrix");
		
		CovVarMatrixFilename = GetString("CovVarMatrixFile");

		WeightMatrixFile = GetString("WeightMatrixFile");

		EigenImages = GetStringOptional("EigenImages");

		EigenValues = GetStringOptional("EigenValues");

		EigenVectors = GetStringOptional("EigenVectors");

		ComputeAverageParticle = GetBool("ComputeAverageParticle");

		AverageParticleFile = GetStringOptional("AverageParticleFile");
		if (!ComputeAverageParticle)
		{
			AverageParticleFile = GetString("AverageParticleFile");
		}

		ClassFileName = GetString("ClassFileName");

		BlockSize = GetInt("BlockSize");

	}
    Config* Config::config = NULL;
	Config& Config::GetConfig(string aConfigFile, int argc, char** argv, int mpiPart, char** appEnvp)
	{
        if (config == NULL)
        {
                config = new Config(aConfigFile, argc, argv, mpiPart, appEnvp);
        }
        return *config;
    }

    Config& Config::GetConfig()
    {
        return *config;
    }

	Config::~Config() {
		for (map<string, Config*>::iterator i = groups.begin(); i != groups.end(); ++i) {
			delete i->second;
		}
	}

	void Config::add(string aName, string aValue) {
		symbols[aName] = aValue;
	}

	void Config::split(string in, string& left, string& right, char c) {
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

	void Config::trim(string& s) {
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

	void Config::symbolExpand(string& s) {
		symbolExpand(symbols, s);
	}

	void Config::envSymbolExpand(string& s) {
		symbolExpand(envSymbols, s);
	}

	void Config::symbolExpand(map<string, string>& symbols, string& s) {
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

	string Config::GetString(string aName) {
		map<string, string>::iterator i = symbols.find(aName);
		if (i == symbols.end()) {
			ConfigPropertyException ex(mConfigFileName, aName);
			throw ex;
			//exit(4);
		}
		return i->second;
	}

	string Config::GetStringOptional(string aName) {
		map<string, string>::iterator i = symbols.find(aName);
		if (i == symbols.end()) {
			/*if (logAllowed) logError(cout << "access of missing property '" << aName << "' (" << mDebugInfo << ")" << endl);
			ConfigPropertyException ex(mConfigFileName, aName);
			throw ex;*/
			//exit(4);
			return string();
		}
		return i->second;
	}

	bool Config::GetBool(string aName) {
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

	bool Config::GetBool(string aName, bool defaultVal) {
		string val = GetStringOptional(aName);

		if (val.empty()) return defaultVal;

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

	double Config::GetDouble(string aName) {
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

	float Config::GetFloat(string aName) {
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

	float Config::GetFloat(string aName, float defaultVal) {
		string val = GetStringOptional(aName);
		stringstream ss(val);
		float retVal = 0;
		if ((ss >> retVal).fail())
		{
			return defaultVal;
		}
		return retVal;
	}

	int Config::GetInt(string aName) {
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

	void Config::replaceChar(string& str, char replace, char by)
	{
		size_t size = str.size();

		for (size_t i = 0; i < size; i++)
		{
			if (str[i] == replace) str[i] = by;
		}
	}

	dim3 Config::GetDim3(string aName) {
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

	float2 Config::GetFloat2(string aName) {
		string val = GetString(aName);
		replaceChar(val, '(', ' ');
		replaceChar(val, '[', ' ');
		replaceChar(val, '{', ' ');
		replaceChar(val, ';', ' ');
		replaceChar(val, ')', ' ');
		replaceChar(val, ']', ' ');
		replaceChar(val, '}', ' ');
		stringstream ss(val);
		float2 retVal;
		float temp;
		if ((ss >> temp).fail())
		{
			ConfigValueException ex(mConfigFileName, aName, "float2");
			throw ex;
		}
		retVal.x = temp;
		if ((ss >> temp).fail())
		{
			ConfigValueException ex(mConfigFileName, aName, "float2");
			throw ex;
		}
		retVal.y = temp;
		
		return retVal;
	}


	float2 Config::GetFloat2(string aName, float2 defaultVal) {
		string val = GetStringOptional(aName);
		replaceChar(val, '(', ' ');
		replaceChar(val, '[', ' ');
		replaceChar(val, '{', ' ');
		replaceChar(val, ';', ' ');
		replaceChar(val, ')', ' ');
		replaceChar(val, ']', ' ');
		replaceChar(val, '}', ' ');
		stringstream ss(val);
		float2 retVal;
		float temp;
		if ((ss >> temp).fail())
		{
			return defaultVal;
		}
		retVal.x = temp;
		if ((ss >> temp).fail())
		{
			return defaultVal;
		}
		retVal.y = temp;

		return retVal;
	}

	float3 Config::GetFloat3(string aName) {
		string val = GetString(aName);
		replaceChar(val, '(', ' ');
		replaceChar(val, '[', ' ');
		replaceChar(val, '{', ' ');
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

	float3 Config::GetFloatOrFloat3(string aName) {
		string val = GetString(aName);
		replaceChar(val, '(', ' ');
		replaceChar(val, '[', ' ');
		replaceChar(val, '{', ' ');
		replaceChar(val, ';', ' ');
		replaceChar(val, ')', ' ');
		replaceChar(val, ']', ' ');
		replaceChar(val, '}', ' ');
		stringstream ss(val);
		float3 retVal;
		float temp;
		if ((ss >> temp).fail())
		{
			ConfigValueException ex(mConfigFileName, aName, "float or float3");
			throw ex;
		}
		retVal.x = temp;
		if ((ss >> temp).fail())
		{
			retVal.y = retVal.x;
			retVal.z = retVal.x;
			return retVal;
		}
		retVal.y = temp;
		if ((ss >> temp).fail())
		{
			ConfigValueException ex(mConfigFileName, aName, "float or float3");
			throw ex;
		}
		retVal.z = temp;
		return retVal;
	}

	float4 Config::GetFloat4(string aName, float4 defaultVal) {
		string val;
		val = GetStringOptional(aName);
		if (val.empty())
		{
			//cout << "Using default values for BetaFac!" << endl;
			return defaultVal;
		}
		
		replaceChar(val, '(', ' ');
		replaceChar(val, '[', ' ');
		replaceChar(val, '{', ' ');
		replaceChar(val, ';', ' ');
		replaceChar(val, ')', ' ');
		replaceChar(val, ']', ' ');
		replaceChar(val, '}', ' ');
		stringstream ss(val);
		float4 retVal;
		float temp;
		if ((ss >> temp).fail())
		{
			return defaultVal;
		}
		retVal.x = temp;
		if ((ss >> temp).fail())
		{
			return defaultVal;
		}
		retVal.y = temp;
		if ((ss >> temp).fail())
		{
			return defaultVal;
		}
		retVal.z = temp;
		if ((ss >> temp).fail())
		{
			return defaultVal;
		}
		retVal.w = temp;
		return retVal;
	}

	vector<int> Config::GetVectorInt(string aName) {
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

		vector<int> retVal;
		int temp;
		while(!(ss >> temp).fail())
		{
			retVal.push_back(temp);
		}

		if (retVal.size() < 1)
		{
			ConfigValueException ex(mConfigFileName, aName, "vector<int>");
			throw ex;
		}
		return retVal;
	}

	vector<float> Config::GetVectorFloat(string aName) {
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

		vector<float> retVal;
		float temp;
		while(!(ss >> temp).fail())
		{
			retVal.push_back(temp);
		}

		if (retVal.size() < 1)
		{
			ConfigValueException ex(mConfigFileName, aName, "vector<float>");
			throw ex;
		}
		return retVal;
	}

	string Config::GetConfigFileName()
	{
		return mConfigFileName;
	}
} //end namespace Configuration
