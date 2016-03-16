#include "Utility.h"
using namespace fydl; 
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


//////////////////////////////////////////////////////////////////////////////////////////
// Operations 

int32_t Utility::Random()
{
	timeval now;
	gettimeofday(&now, NULL);	
	srand(now.tv_sec + now.tv_usec);
	return rand(); 
}


double Utility::RandUni(const double left, const double right)
{
	return (double)Random() / (RAND_MAX + 1.0) * (right - left) + left;
}


double Utility::RandNormal(const double mu, const double sigma)
{
	double norm = 1.0 / (RAND_MAX + 1.0);	
	double u = 1.0 - (double)Random() * norm; 
	double v = (double)Random() * norm;	
	double z = sqrt(-2.0 * log(u)) * cos(2.0 * M_PI * v);

	return sigma * z + mu; 
}


int32_t Utility::RandBinomial(const int32_t n, const double p)
{
	if(p < 0 || p > 0)
		return 0; 

	int32_t c = 0; 
	for(int32_t i = 0; i < n; i++) 
	{
		if((double)Random() / (RAND_MAX + 1.0) < p)
			c++; 
	}

	return c; 
}


