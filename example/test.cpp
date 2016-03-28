#include <iostream>
#include <fstream>
using namespace std; 
#include "fydl.h"
using namespace fydl; 
#include <stdio.h>
#include <string.h>
#include <math.h>

void Test_Utility_RandUni()
{
	cout<<"== Test_Utility_RandUni =="<<endl; 
	int32_t st[20]; 
	memset(st, 0, sizeof(int32_t) * 20); 
	double cnt = 9999; 
	double r; 
	
	for(int32_t i = 0; i < cnt; i++)  
	{
		r = Utility::RandUni(0.0, 1.0);
		st[(int32_t)(r * 20.0) % 20] += 1; 
	}	

	for(int32_t k = 0; k < 20; k++) 
	{
		cout<<k<<"\t"; 
		for(int32_t i = 0; i < st[k] * 300 / cnt; i++) 
			cout<<"*"; 
		cout<<"  "<<st[k]<<endl; 
	}
	cout<<endl; 
}


void Test_Utility_RandNormal()
{
	cout<<"== Test_Utility_RandNormal =="<<endl; 
	int32_t st[20]; 
	memset(st, 0, sizeof(int32_t) * 20); 
	double cnt = 9999; 
	double r; 
	
	for(int32_t i = 0; i < cnt; i++)  
	{
		r = Utility::RandNormal(10.0, 2.5); 
		if(r > 0.0 && r < 20.0)
			st[(int32_t)r] += 1; 
	}	

	for(int32_t k = 0; k < 20; k++) 
	{
		cout<<k<<"\t"; 
		for(int32_t i = 0; i < st[k] * 300 / cnt; i++) 
			cout<<"*"; 
		cout<<"  "<<st[k]<<endl; 
	}
	cout<<endl; 
}



int main()
{
	Test_Utility_RandUni();
	Test_Utility_RandNormal();
	return 0; 
}


