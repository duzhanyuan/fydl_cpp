#ifndef _FYDL_STRING_ARRAY_H 
#define _FYDL_STRING_ARRAY_H 

#include <string> 
#include <vector> 
using namespace std; 
#include <stdint.h> 

namespace fydl
{

class StringArray 
{
public: 
	StringArray(const char* sStr, const char* sDilm);   
	virtual ~StringArray(); 
	
	string GetString(const uint32_t unIdx) const; 
	uint32_t Count() const; 

private: 
	void Decompose(const char* sStr, const char* sDilm); 
	vector<string> m_vtrString;
};  

}

#endif /* _FYDL_STRING_ARRAY_H */ 

