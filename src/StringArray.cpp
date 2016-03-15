#include "StringArray.h" 
using namespace fydl; 
#include <string.h>


////////////////////////////////////////////////////////////////////////////
// Construction & Destruction 

StringArray::StringArray(const char* sStr, const char* sDilm) 
{ 
	Decompose(sStr, sDilm);  
} 


StringArray::~StringArray() 
{ 
}  


////////////////////////////////////////////////////////////////////////////
// Operations  

string StringArray::GetString(const uint32_t unIdx) const 
{ 
	if(unIdx >= (uint32_t)m_vtrString.size()) 
		return string(""); 
	return m_vtrString[unIdx];  
}  


uint32_t StringArray::Count() const 
{ 
	return (uint32_t)m_vtrString.size();  
}  


////////////////////////////////////////////////////////////////////////////
// Internal Operations 

void StringArray::Decompose(const char* sStr, const char* sDilm) 
{ 
	string str(sStr);
       	str += string(sDilm) + string("EOF");	
	int32_t nLen = str.length(); 	
	int32_t nOffset = 0, nPos; 
	while(nOffset < nLen)
	{ 
		nPos = str.find(sDilm, nOffset); 
		if(nPos == string::npos)  
			nPos = str.length();
		if(nPos == nOffset)
		{
			m_vtrString.push_back(string("")); 
		}
		else
		{		       
			m_vtrString.push_back(str.substr(nOffset, nPos - nOffset)); 
		}
		nOffset = nPos + strlen(sDilm); 
	}
	m_vtrString.pop_back(); 
}  


