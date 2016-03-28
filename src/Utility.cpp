#include "Utility.h"
using namespace fydl; 
#include <iostream>
#include <algorithm>
using namespace std; 
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>


//////////////////////////////////////////////////////////////////////////////////////////
// Construction & Destruction 

Utility::Utility()
{
}


Utility::~Utility()
{
}


vector<int32_t> Utility::m_vtr;


//////////////////////////////////////////////////////////////////////////////////////////
// Operations 

int32_t Utility::Random()
{
	if(m_vtr.empty())
	{
		for(int32_t k = 1; k < 10; k++) 
			m_vtr.push_back(k*10+k); 	
	}
	random_shuffle(m_vtr.begin(), m_vtr.end()); 
	
	timeval now;
	gettimeofday(&now, NULL);	
	srand(now.tv_sec + now.tv_usec + m_vtr[0]); 
	return rand(); 
}


double Utility::RandUni(const double left, const double right)
{
	return (double)Random() / (RAND_MAX + 1.0) * (right - left) + left;
}


double Utility::RandNormal(const double mu, const double sigma)
{
	double u1 = RandUni(); 
	double u2 = RandUni(); 
	double z = sqrt(0.0 - 2.0 * log(u1)) * cos(2.0 * M_PI * u2);  
	
	return sigma * z + mu; 
}


int32_t Utility::RandBinomial(const int32_t n, const double p)
{
	if(p < 0 || p > 1.0)
		return 0; 
	int32_t c = 0; 
	for(int32_t i = 0; i < n; i++) 
	{
		if(RandUni() < p)
			c++; 
	}

	return c; 
}


