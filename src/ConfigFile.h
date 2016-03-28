#ifndef _FYDL_CONFIG_FILE_H 
#define _FYDL_CONFIG_FILE_H 

#include <string> 
#include <map> 
#include <vector> 
using namespace std; 
#include <stdint.h> 

namespace fydl
{

// CLASS
//	ConfigFile - config file operations
// 
// DESCRIPTION
//	This class supports the config file parsing and value transformation
//
class ConfigFile
{
public: 
	ConfigFile(); 
	virtual ~ConfigFile(); 

	// NAME
	//	Read - open and read config file
	//
	// DESCRIPTION
	//	sConfigFile: config file 
	//	sSep: the separator between key and values
	//	sSubSep: the sub-separator between values
	//
	// RETRUN
	//	return true for success, false for some errors
	bool Read(const char* sConfigFile, const char* sSep = ":", const char* sSubSep = ","); 

	// NAME
	//	ValCnt - get the number of values
	//
	// DESCRIPTION
	//	sKey: key	
	//
	// RETURN
	//	number of values
	int32_t ValCnt(const char* sKey);

	// NAME
	//	GetVal_asString - get the value in string format
	//	GetVal_asInt - get the value in integer format
	//	GetVal_asFloat - get the value in float format
	//	GetVal_asBool - get the value in boolean format
	//
	// DESCRIPTION
	//	sKey: key	
	//	nIdx: the indes of the value
	//	sDefault, nDefault, dDefault, bDefault: default value
	string GetVal_asString(const char* sKey, const int32_t nIdx = 0, const char* sDefault = NULL);
	int32_t GetVal_asInt(const char* sKey, const int32_t nIdx = 0, const int32_t nDefault = 0);
	double GetVal_asFloat(const char* sKey, const int32_t nIdx = 0, const double dDefault = 0.0);
	bool GetVal_asBool(const char* sKey, const int32_t nIdx = 0, const bool bDefault = false);

private:
	// ignore the space (blank, space, tab...) in the config string
	void IgnoreSpace(string& str);	
	map<string,vector<string> > m_mapKeyToValues;   // Key->Values  
};  

}

#endif /* _FYDL_CONFIG_FILE_H */ 


