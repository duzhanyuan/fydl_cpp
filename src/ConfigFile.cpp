#include "ConfigFile.h"
#include "StringArray.h"
using namespace fydl; 
#include <fstream>
using namespace std; 


//////////////////////////////////////////////////////////////////////////////////////////
// Construction & Destruction 

ConfigFile::ConfigFile()
{
}


ConfigFile::~ConfigFile()
{
}


//////////////////////////////////////////////////////////////////////////////////////////
// Operations 

bool ConfigFile::Read(const char* sConfigFile, const char* sSep, const char* sSubSep)
{
	ifstream ifs(sConfigFile); 
	if(!ifs.is_open())
		return false; 

	string str, strkey, strvals, strval; 
	map<string,vector<string> >::iterator iter_map; 
	
	m_mapKeyToValues.clear(); 
	while(!ifs.eof())
	{
		std::getline(ifs, str); 
		if(str.empty())
			continue;
		if(str.at(0) == '#')
			continue; 
		StringArray array(str.c_str(), sSep); 
		if(array.Count() != 2)
			continue; 
		strkey = array.GetString(0);
		IgnoreSpace(strkey); 
		iter_map = m_mapKeyToValues.find(strkey); 
		if(iter_map == m_mapKeyToValues.end())
		{
			vector<string> vtr_vals; 
			strvals = array.GetString(1);
			StringArray ar(strvals.c_str(), sSubSep); 
			for(int32_t i = 0; i < ar.Count(); i++) 
			{
				strval = ar.GetString(i); 	
				IgnoreSpace(strval); 
				vtr_vals.push_back(strval); 
			}	
			m_mapKeyToValues.insert(pair<string,vector<string> >(strkey, vtr_vals));	
		}
		else
		{
			strvals = array.GetString(1);
			StringArray ar(strvals.c_str(), sSubSep); 
			for(int32_t i = 0; i < ar.Count(); i++) 
			{
				strval = ar.GetString(i); 	
				IgnoreSpace(strval); 
				iter_map->second.push_back(strval); 
			}	
		}
	}

	ifs.close(); 
	return true; 
}


int32_t ConfigFile::ValCnt(const char* sKey)
{
	map<string,vector<string> >::iterator iter_map = m_mapKeyToValues.find(sKey); 
	if(iter_map == m_mapKeyToValues.end())
		return 0; 
	return (int32_t)iter_map->second.size(); 
}


string ConfigFile::GetVal_asString(const char* sKey, const int32_t nIdx, const char* sDefault)
{
	map<string,vector<string> >::iterator iter_map = m_mapKeyToValues.find(sKey); 
	if(iter_map == m_mapKeyToValues.end())
		return string(sDefault); 
	if(nIdx < 0 || nIdx >= (int32_t)iter_map->second.size())
		return string(sDefault); 
	return string(iter_map->second[nIdx]); 
}


int32_t ConfigFile::GetVal_asInt(const char* sKey, const int32_t nIdx, const int32_t nDefault)
{
	map<string,vector<string> >::iterator iter_map = m_mapKeyToValues.find(sKey); 
	if(iter_map == m_mapKeyToValues.end())
		return nDefault; 
	if(nIdx < 0 || nIdx >= (int32_t)iter_map->second.size())
		return nDefault; 
	int32_t val; 
	sscanf(iter_map->second[nIdx].c_str(), "%d", &val); 
	return val; 
}


double ConfigFile::GetVal_asFloat(const char* sKey, const int32_t nIdx, const double dDefault)
{
	map<string,vector<string> >::iterator iter_map = m_mapKeyToValues.find(sKey); 
	if(iter_map == m_mapKeyToValues.end())
		return dDefault; 
	if(nIdx < 0 || nIdx >= (int32_t)iter_map->second.size())
		return dDefault; 
	double val; 
	sscanf(iter_map->second[nIdx].c_str(), "%lf", &val); 
	return val; 
}


bool ConfigFile::GetVal_asBool(const char* sKey, const int32_t nIdx, const bool bDefault)
{
	map<string,vector<string> >::iterator iter_map = m_mapKeyToValues.find(sKey); 
	if(iter_map == m_mapKeyToValues.end())
		return bDefault; 
	if(nIdx < 0 || nIdx >= (int32_t)iter_map->second.size())
		return bDefault; 
	if(iter_map->second[nIdx] == "true")
		return true;
	else
		return false;
}


//////////////////////////////////////////////////////////////////////////////////////////
// Internal Operations 

void ConfigFile::IgnoreSpace(string& str)
{
	int32_t nOffset = 0; 
	while(nOffset < (int32_t)str.length()) 
	{
		if(str.at(nOffset) != ' ' && str.at(nOffset) != '\t' && str.at(nOffset) != '\r')
			break;
		nOffset++;
	}
	str = str.substr(nOffset, str.length() - nOffset);

	nOffset = str.length() - 1;
	while(nOffset >= 0)
	{
		if(str.at(nOffset) != ' ' && str.at(nOffset) != '\t' && str.at(nOffset) != '\r')
			break;
		nOffset--;
	}
	str = str.substr(0, nOffset+1);
}


