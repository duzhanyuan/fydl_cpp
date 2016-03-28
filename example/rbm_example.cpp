// rbm_example.cpp
//
// example of RBM
//
// AUTHOR
//	fengyoung (fengyoung82@sina.com)
// 
// HISTORY
//	v1.0 2016-03-18
//

#include <iostream>
#include <fstream>
using namespace std; 
#include "fydl.h"
using namespace fydl; 
#include <stdio.h>
#include <string.h>
#include <math.h>



void PrintHelp()
{
	cout<<"Usage: rbm_example [--train <config_file> <training_patterns_file> <out_model_file>"<<endl;   
	cout<<"                   [--test <model_file> <testing_patterns_file>]"<<endl;   
}


bool ReadPatterns(vector<Pattern*>& vtrPatts, const char* sPattFile)
{
	ifstream ifs(sPattFile);
	if(!ifs.is_open())
		return false; 
	string str; 
	Pattern* ppatt = NULL; 
	while(!ifs.eof())
	{
		std::getline(ifs, str); 
		if(str.empty())
			continue; 
		ppatt = new Pattern(); 	
		if(!ppatt->FromString(str.c_str()))
		{
			delete ppatt; 
			continue; 
		}
		vtrPatts.push_back(ppatt); 
	}
	ifs.close(); 
	if(vtrPatts.empty())
		return false; 
	return true; 
}


void TrainDemo(const char* sConfigFile, const char* sPattFile, const char* sModelFile)
{
	cout<<"== RBM Example: Training =="<<endl; 

	vector<Pattern*> vtr_patts; 
	if(!ReadPatterns(vtr_patts, sPattFile))
	{
		cout<<"failed to open training patterns file "<<sPattFile<<endl; 
		return; 
	}

	RBM rbm; 
	if(!rbm.InitFromConfig(sConfigFile, vtr_patts[0]->m_nXCnt))
	{
		cout<<"failed to initialize the RBM from config file "<<sConfigFile<<endl; 
		return; 
	}

	TypeDefs::Print_RBMLearningParamsT(cout, rbm.GetLearningParams()); 
	cout<<"--"<<endl; 
	TypeDefs::Print_RBMParamsT(cout, rbm.GetArchParams()); 
	cout<<"==========================="<<endl; 

	rbm.Train(vtr_patts); 

	if(rbm.Save(sModelFile) != _FYDL_SUCCESS)
		cout<<"failed to save the RBM model to "<<sModelFile<<endl; 

	for(size_t i = 0; i < vtr_patts.size(); i++)
		delete vtr_patts[i]; 
	vtr_patts.clear(); 
}


void TestDemo(const char* sModelFile, const char* sPattFile)
{
	cout<<"== RBM Example: Testing =="<<endl; 
	
	RBM rbm; 
	int32_t ret = rbm.Load(sModelFile); 
	if(ret != _FYDL_SUCCESS)
	{
		cout<<"failed to load the RBM model from "<<sModelFile<<endl; 
		cout<<"error code is "<<ret<<endl; 
		return;
	}

	TypeDefs::Print_RBMLearningParamsT(cout, rbm.GetLearningParams()); 
	cout<<"--"<<endl; 
	TypeDefs::Print_RBMParamsT(cout, rbm.GetArchParams()); 
	cout<<"=========================="<<endl; 

	vector<Pattern*> vtr_patts; 
	if(!ReadPatterns(vtr_patts, sPattFile))
	{
		cout<<"failed to open training patterns file "<<sPattFile<<endl; 
		return; 
	}

	int32_t x_len = vtr_patts[0]->m_nXCnt;	
	double* xr = new double[x_len]; 
	int32_t patts = (int32_t)vtr_patts.size(); 
	double error, rmse = 0.0; 

	for(int32_t i = 0; i < patts; i++) 
	{
		error = rbm.Reconstruct(xr, vtr_patts[i]->m_x, x_len); 
		rmse += error; 

		printf("(%d) [%s] -> [%s] | error: %.12g\n", 
				i+1, 
				Pattern::ArrayToString(vtr_patts[i]->m_x, vtr_patts[i]->m_nXCnt).c_str(), 
				Pattern::ArrayToString(xr, x_len).c_str(), 
				error);
		delete vtr_patts[i]; 
	}
	rmse = sqrt(rmse / (double)patts); 
	vtr_patts.clear(); 

	printf("Pattern_Cnt: %d, rmse: %.12g\n", 
			patts, rmse);
}


int main(int argc, char** argv)
{
	if(argc != 4 && argc != 5)
	{
		PrintHelp(); 
		return -1; 	
	}

	if(strcmp(argv[1], "--train") == 0)
	{
		if(argc != 5)
		{
			PrintHelp(); 
			return -1;
		}
		TrainDemo(argv[2], argv[3], argv[4]); 
	}
	else if(strcmp(argv[1], "--test") == 0)
	{
		if(argc != 4)
		{
			PrintHelp(); 
			return -1;
		}
		TestDemo(argv[2], argv[3]); 
	}
	else
	{
		PrintHelp(); 
		return -1;
	}

	return 0; 
}


