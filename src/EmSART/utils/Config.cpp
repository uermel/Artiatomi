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
#include "log.h"

using namespace std;

namespace Configuration
{
	Config::Config(string aName, string aParentDebugInfo) {
		mDebugInfo = aParentDebugInfo + ", " + aName;
	}

	Config::Config(string aConfigFile, int argc, char** argv, int mpiPart, char** appEnvp)
		:mConfigFileName(aConfigFile),
		CudaDeviceIDs(),
		ProjectionFile(),
		OutVolumeFile(),
		MarkerFile(),
		Lambda(-1.f),
		AddTiltAngle(0.f),
		Iterations(-1),
		RecDimensions(0),
		PsiAngle(0),
		PhiAngle(-1),
		UseFixPsiAngle(false),
		VolumeShift(make_float3(0)),
		ReferenceMarker(0),
		OverSampling(-1),
		VoxelSize(make_float3(-1, -1, -1)),
		DimLength(make_float2(-1, -1)),
		CutLength(make_float2(-1, -1)),
		Crop(make_float4(0, 0, 0, 0)),
		CropDim(make_float4(0, 0, 0, 0)),
		SkipFilter(false),
		fourFilterLP(800),
		fourFilterLPS(200),
		fourFilterHP(0),
		fourFilterHPS(0),
		SIRTCount(1),
		CtfFile(),
		BadPixelValue(10),
		CorrectBadPixels(true),
		FP16Volume(false),
		WriteVolumeAsFP16(false),
		ProjectionScaleFactor(1),
		logAllowed(mpiPart == 0),
		WBP_NoSART(false),
		WBPFilter(FM_RAMP),
		Cs(2.7f),
		Voltage(300),
		MagAnisotropyAmount(1.0f),
		MagAnisotropyAngleInDeg(0.0f),
		IgnoreZShiftForCTF(false),
		CTFSliceThickness(50.0f),
		DebugImages(false),
		DownWeightTiltsForWBP(false),
		DoseWeighting(false),
		PhaseFlipOnly(false),
		WienerFilterNoiseLevel(0.1f)
			//ProjNormVal(-1)
#ifdef REFINE_MODE
			,
			SizeSubVol(0),
			VoxelSizeSubVol(0),
			MotiveList(),
			Reference(),
			MaxShift(0),
			ShiftOutputFile(),
			GroupMode(MotiveList::GroupMode_enum::GM_BYGROUP),
			GroupSize(0),
			MaxDistance(0),
			CCMapFileName(),
			SpeedUpDistance(0),
			ScaleMotivelistShift(1),
			ScaleMotivelistPosition(1),
			SupportingReferences(),
			SupportingMotiveLists(),
			MultiPeakDetection(false)
#endif
#ifdef SUBVOLREC_MODE
			,
			SizeSubVol(0),
			VoxelSizeSubVol(0),
			MotiveList(),
			SubVolPath(),
			ShiftInputFile(),
			BatchSize(0),
			NamingConv(MotiveList::NamingConvention_enum::NC_ParticleOnly),
			ScaleMotivelistShift(1),
			ScaleMotivelistPosition(1)
#endif
	{
		while (appEnvp && *appEnvp) {
			string envEntry = *appEnvp;
			size_t pos = envEntry.find('=');
			if (pos != string::npos) {
				string name = envEntry.substr(0, pos);
				string value = envEntry.substr(pos+1, string::npos);
				envSymbols[name] = value;
				if (logAllowed) logDebug(cout << "environment symbol: '" << name << "' = '" << value << "'" << endl);
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
			else if (str == "-p")
			{
				ProjectionFile = string(argv[i+1]);
				i++;
			}
			else if (str == "-o")
			{
				OutVolumeFile = string(argv[i+1]);
				i++;
			}
			else if (str == "-m")
			{
				MarkerFile = string(argv[i+1]);
				i++;
			}
			else if (str == "-l")
			{
				stringstream ss(argv[i+1]);
				ss >> Lambda;
				i++;
			}
			else if (str == "-i")
			{
				stringstream ss(argv[i+1]);
				ss >> Iterations;
				i++;
			}
			else if (str == "-psi")
			{
				stringstream ss(argv[i+1]);
				ss >> PsiAngle;
				UseFixPsiAngle = true;
				i++;
			}
			else if (str == "-vs")
			{
				stringstream ss(argv[i+1]);
				ss >> VolumeShift.x;
				stringstream ss2(argv[i+2]);
				ss2 >> VolumeShift.y;
				stringstream ss3(argv[i+3]);
				ss3 >> VolumeShift.z;
				i+= 3;
			}
			else if (str == "-r")
			{
				stringstream ss(argv[i+1]);
				ss >> ReferenceMarker;
				i++;
			}
			else if (str == "-os")
			{
				stringstream ss(argv[i+1]);
				ss >> OverSampling;
				i++;
			}
			else if (str == "-phi")
			{
				stringstream ss(argv[i+1]);
				ss >> PhiAngle;
				i++;
			}
			/*else if (str == "-VS")
			{
				stringstream ss(argv[i+1]);
				ss >> VoxelSize;
				i++;
			}
			else if (str == "-dl")
			{
				stringstream ss(argv[i+1]);
				ss >> DimLength;
				i++;
			}
			else if (str == "-cl")
			{
				stringstream ss(argv[i+1]);
				ss >> CutLength;
				i++;
			}*/
			else if (str == "-ctf")
			{
				CtfFile = string(argv[i+1]);
				i++;
			}
			else if (str == "-log")
			{
				i++;
			}
			else if (str == "-debug")
			{
				DebugImages = true;
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
					cout << "    -u FILENAME:   Use a user defined configuration file." << endl;
					cout << "    -p FILENAME:   Filename containing the projections." << endl;
					cout << "    -o FILENAME:   Filename where to store the final volume." << endl;
					cout << "    -m FILENAME:   Filename containing the marker positions." << endl;
					cout << "    -l VALUE:      Relaxation factor lambda [0..1]." << endl;
					cout << "    -i VALUE:      Number of iterations." << endl;
					cout << "    -psi VALUE:    Use a fixed value for angle PSI at VALUE." << endl;
					cout << "    -phi VALUE:    Set value for angle PHI to VALUE." << endl;
					cout << "    -vs VALUE:     Shift volume in x, y, z direction by VALUE.(xyz)." << endl;
					cout << "    -r VALUE:      Reference Marker." << endl;
					cout << "    -os VALUE:     Oversampling used in back projection." << endl;
					//cout << "    -VS VALUE:     Voxel size. Factor relative to pixel size." << endl;
					cout << "    -ctf FILENAME: Use a user defined configuration file." << endl;
					cout << "    -log FILENAME: Write reconstruction info to log file." << endl;
					cout << "    -h:            Show this text." << endl;


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
					if (logAllowed)  logDebug(cout << "   config: new group '" << name << "'" << endl);
					Config* newGroup = new Config(name, mDebugInfo);
					groupStack.front()->groups[name] = newGroup;
					groupStack.push_front(newGroup);
				} else {
					for (list<Config*>::reverse_iterator i = groupStack.rbegin(); i != groupStack.rend(); ++i) {
						(*i)->symbolExpand(value);
					}
					envSymbolExpand(value);
					if (logAllowed) logDebug(cout << "   config: name = '" << name << "', value = '" << value << "'" << endl);
					groupStack.front()->add(name, value);
				}
			}
			if ( (line.length() > 0) && (line[0] != '#') && (line.find(')') != string::npos) ) {
				if (logAllowed) logDebug(cout << "   end of group" << endl);
				groupStack.pop_front();
			}
		}
		in.close();

		CudaDeviceIDs = GetVectorInt("CudaDeviceID");
		
		if (ProjectionFile.size() < 1)
			ProjectionFile = GetString("ProjectionFile");
		if (OutVolumeFile.size() < 1)
			OutVolumeFile = GetString("OutVolumeFile");
		if (MarkerFile.size() < 1)
			MarkerFile = GetString("MarkerFile");
		if (Lambda < 0)
			Lambda = GetFloat("Lambda");
		if (Iterations < 0)
			Iterations = GetInt("Iterations");
		RecDimensions = GetDim3("RecDimesions");
		if (UseFixPsiAngle == false)
            UseFixPsiAngle = GetBool("UseFixPsiAngle");
        if (UseFixPsiAngle)
            PsiAngle = GetFloat("PsiAngle");
        if ((VolumeShift.x == 0) && (VolumeShift.y == 0) && (VolumeShift.z  == 0))
            VolumeShift = GetFloat3("VolumeShift");
		if (ReferenceMarker < 0)
			ReferenceMarker = GetInt("ReferenceMarker");
		if (OverSampling < 0)
			OverSampling = GetInt("OverSampling");
		if (PhiAngle < 0)
			PhiAngle = GetFloat("PhiAngle");
		VoxelSize = GetFloatOrFloat3("VoxelSize");
		
		AddTiltAngle = GetFloat("AddTiltAngle", 0.0f);	
		AddTiltXAngle = GetFloat("AddTiltXAngle", 0.0f);	
		DimLength = GetFloat2("DimLength");		
		CutLength = GetFloat2("CutLength");	
		Crop = GetFloat4("Crop", make_float4(0, 0, 0, 0));
		CropDim = GetFloat4("CropDim", make_float4(0, 0, 0, 0));
		SkipFilter = GetBool("SkipFilter");
		fourFilterLP = GetInt("fourFilterLP");
		fourFilterLPS = GetInt("fourFilterLPS");
		fourFilterHP = GetInt("fourFilterHP");
		fourFilterHPS = GetInt("fourFilterHPS");
		SIRTCount = GetInt("SIRTCount");
		CTFBetaFac = GetFloat4("CTFBetaFac", make_float4(200, 0, 0.008f, 0));
		FP16Volume = GetBool("FP16Volume", false);
		WriteVolumeAsFP16 = GetBool("WriteVolumeAsFP16", false);
		ProjectionScaleFactor = GetFloat("ProjectionScaleFactor" , 1.0f);
		WBP_NoSART = GetBool("WBP", false);	

		DoseWeighting = GetBool("DoseWeighting", false);
		if (DoseWeighting)
		{
			AccumulatedDose = GetVectorFloat("AccumulatedDose");
		}
		DownWeightTiltsForWBP = GetBool("DownWeightTiltsForWBP", false);
		SwitchCTFDirectionForIMOD = GetBool("SwitchCTFDirectionForIMOD", false);
		PhaseFlipOnly = GetBool("PhaseFlipOnly", false);
		WienerFilterNoiseLevel = GetFloat("WienerFilterNoiseLevel", 0.1f);

		string f = GetStringOptional("WBPFilter");
		if (f.length() > 0)
		{
			if (f == "Ramp" || f == "RAMP" || f == "ramp")
				WBPFilter = FM_RAMP;
			if (f == "Exact" || f == "EXACT" || f == "exact")
				WBPFilter = FM_EXACT;
			if (f == "Contrast2" || f == "CONTRAST2" || f == "contrast2")
				WBPFilter = FM_CONTRAST2;
			if (f == "Contrast10" || f == "CONTRAST10" || f == "contrast10")
				WBPFilter = FM_CONTRAST10;
			if (f == "Contrast30" || f == "CONTRAST30" || f == "contrast30")
				WBPFilter = FM_CONTRAST30;
		}
		//ProjNormVal = GetFloat("ProjNormVal");
		//Filtered = GetBool("Filtered");
		if (GetBool("CtfMode"))
			CtfMode = CTFM_YES;
		else
			CtfMode = CTFM_NO;

		if (CtfMode == CTFM_YES && CtfFile.length() < 1)
			CtfFile = GetString("CtfFile");
		
		if (CtfMode == CTFM_YES)
		{
			Cs = GetFloat("Cs");
			Voltage = GetFloat("Voltage");
		}

		BadPixelValue = GetFloat("BadPixelValue");
		CorrectBadPixels = GetBool("CorrectBadPixels");

		ProjectionNormalization = PNM_MEAN;
		string ProjNorm = GetStringOptional("ProjectionNormalization");
		if (ProjNorm == "STD" || ProjNorm == "std" || ProjNorm == "StandardDeviation")
			ProjectionNormalization = PNM_STANDARD_DEV;
		if (ProjNorm == "NONE" || ProjNorm == "none" || ProjNorm == "None")
			ProjectionNormalization = PNM_NONE;

		float2 magIsoDef;
		magIsoDef.x = 1.0f;
		magIsoDef.y = 0.0f;
		float2 magIso = GetFloat2("MagAnisotropy", magIsoDef);
		MagAnisotropyAmount = magIso.x;
		MagAnisotropyAngleInDeg = magIso.y;

		IgnoreZShiftForCTF = GetBool("IgnoreZShiftForCTF", false);
		CTFSliceThickness = GetFloat("CTFSliceThickness", 50.0f);
		
#ifdef REFINE_MODE
		SizeSubVol = GetInt("SizeSubVol");
		VoxelSizeSubVol = GetFloat("VoxelSizeSubVol");
		MotiveList = GetString("MotiveList");
		Reference = GetString("Reference");
		MaxShift = GetInt("MaxShift");
		ShiftOutputFile = GetString("ShiftOutputFile");
		string gm = GetString("GroupMode");

		if (gm == "ByGroup")
		{
			GroupMode = MotiveList::GroupMode_enum::GM_BYGROUP;
		}
		else if(gm == "MaxDistance")
		{
			GroupMode = MotiveList::GroupMode_enum::GM_MAXDIST;
		}
		else if(gm == "MaxCount")
		{
			GroupMode = MotiveList::GroupMode_enum::GM_MAXCOUNT;
		}
		else
		{
			ConfigValueException ex(mConfigFileName, "GroupMode", "ByGroup, MaxDistance or MaxCount");
			throw ex;
		}

		if (GroupMode == MotiveList::GroupMode_enum::GM_MAXDIST)
		{
			MaxDistance = GetFloat("MaxDistance");
		}
		if (GroupMode == MotiveList::GroupMode_enum::GM_MAXCOUNT)
		{
			GroupSize = GetInt("GroupSize");
		}
		CCMapFileName = GetStringOptional("CCMapFileName");
		SpeedUpDistance = GetInt("SpeedUpDistance");
		ScaleMotivelistShift = GetFloat("ScaleMotivelistShift");
		ScaleMotivelistPosition = GetFloat("ScaleMotivelistPosition");
		MultiPeakDetection = GetBool("MultiPeakDetection");

		{
			string temp = GetStringOptional("SupportingReferences");
			if (temp.length() > 3)
			{
				stringstream ss(temp);
				string item;
				while (getline(ss, item, ';'))
				{
					if (item.length() > 3)
						SupportingReferences.push_back(item);
				}

				temp = GetString("SupportingMotiveLists");
				if (temp.length() > 3)
				{
					stringstream ss(temp);
					string item;
					while (getline(ss, item, ';'))
					{
						if (item.length() > 3)
							SupportingMotiveLists.push_back(item);
					}
				}

				MaxDistanceSupport = GetFloat("MaxDistanceSupport");
			}

		}
#endif
#ifdef SUBVOLREC_MODE
		SizeSubVol = GetInt("SizeSubVol");
		VoxelSizeSubVol = GetFloat("VoxelSizeSubVol");
		MotiveList = GetString("MotiveList");
		SubVolPath = GetString("SubVolPath");
		ShiftInputFile = GetString("ShiftInputFile");
		BatchSize = GetInt("BatchSize");
		MaxShift = GetInt("MaxShift");

		string nc = GetStringOptional("NamingConvention");
		NamingConv = MotiveList::NamingConvention_enum::NC_ParticleOnly;
		if (nc == "TomoParticle" || nc == "Tomo_Particle")
		{
			NamingConv = MotiveList::NamingConvention_enum::NC_TomogramParticle;
		}
		ScaleMotivelistShift = GetFloat("ScaleMotivelistShift");
		ScaleMotivelistPosition = GetFloat("ScaleMotivelistPosition");
#endif
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
			if (logAllowed) logError(cout << "access of missing property '" << aName << "' (" << mDebugInfo << ")" << endl);
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

	Config::FILE_SAVE_MODE Config::GetFileSaveMode()
	{
		size_t lastMatchPos = OutVolumeFile.rfind(".em");
		bool isEnding = lastMatchPos != std::string::npos;

		if (isEnding) return FSM_EM;

		lastMatchPos = OutVolumeFile.rfind(".mrc");
		isEnding = lastMatchPos != std::string::npos;

		if (isEnding) return FSM_MRC;

		lastMatchPos = OutVolumeFile.rfind(".rec");
		isEnding = lastMatchPos != std::string::npos;

		if (isEnding) return FSM_MRC;

		return FSM_RAW;
	}

	Config::FILE_READ_MODE Config::GetFileReadMode()
	{
		size_t lastMatchPos = ProjectionFile.rfind(".dm4");
		bool isEnding = lastMatchPos != std::string::npos;

		if (isEnding) return FRM_DM4;

		lastMatchPos = OutVolumeFile.rfind(".mrc");
		isEnding = lastMatchPos != std::string::npos;

		if (isEnding) return FRM_MRC;

		lastMatchPos = OutVolumeFile.rfind(".st");
		isEnding = lastMatchPos != std::string::npos;

		if (isEnding) return FRM_MRC;

		return FRM_MRC;
	}

	string Config::GetConfigFileName()
	{
		return mConfigFileName;
	}
} //end namespace Configuration
