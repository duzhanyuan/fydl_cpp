#ifndef _FYDL_UTILITY_H 
#define _FYDL_UTILITY_H 

#include <stdint.h>


namespace fydl
{

class Utility
{
private: 
	Utility(); 
	virtual ~Utility(); 

	// generate random value
	static int32_t Random(); 	

public: 
	// NAME
	//	RandUni - generate random value based on uniform distribution
	//
	// DESCRIPTION
	//	left, right: bottom and top of the range
	//
	// RETURN
	//	random value
	static double RandUni(const double left = 0.0, const double right = 1.0);
	
	// NAME	
	//	RandNormal - generate random value based on normal distribution
	//
	// DESCRIPTION
	//	mu, sigma: mean and standard deviation the normal distribution
	//
	// RETURN
	//	random value
	static double RandNormal(const double mu = 0.0, const double sigma = 1.0);

	// NAME
	//	RandBinomial - generate random value based on binomial distribution
	//
	// DESCRIPTION
	//	n: number of trials
	//	p: the probability of success
	//
	// RETURN
	//	random value
	static int32_t RandBinomial(const int32_t n, const double p = 0.5);   

};

}

#endif /* _FYDL_UTILITY_H */ 


