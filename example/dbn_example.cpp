// dbn_example.cpp
//
// example of DBN
//
// AUTHOR
//	fengyoung (fengyoung82@sina.com)
// 
// HISTORY
//	v1.0 2016-03-22
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
	cout<<"Usage: dbn_example [--train <config_file> <training_patterns_file> <out_model_file>"<<endl;   
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
	cout<<"== DBN Example: Training =="<<endl; 
	vector<Pattern*> vtr_patts; 
	if(!ReadPatterns(vtr_patts, sPattFile))
	{
		cout<<"failed to open training patterns file "<<sPattFile<<endl; 
		return; 
	}

	DBN dbn; 
	if(!dbn.InitFromConfig(sConfigFile, vtr_patts[0]->m_nXCnt, vtr_patts[0]->m_nYCnt))
	{
		cout<<"failed to initialize the DBN from config file "<<sConfigFile<<endl; 
		return; 
	}

	TypeDefs::Print_DBNLearningParamsT(cout, dbn.GetLearningParams()); 
	cout<<"--"<<endl; 
	TypeDefs::Print_DBNParamsT(cout, dbn.GetArchParams()); 
	cout<<"==========================="<<endl; 

	dbn.Train(vtr_patts); 

	if(dbn.Save(sModelFile) != _FYDL_SUCCESS)
		cout<<"failed to save the DBN model to "<<sModelFile<<endl; 

	for(size_t i = 0; i < vtr_patts.size(); i++)
		delete vtr_patts[i]; 
	vtr_patts.clear(); 
}


void TestDemo(const char* sModelFile, const char* sPattFile)
{
	cout<<"== DBN Example: Testing =="<<endl; 
	
	DBN dbn;
	int32_t ret = dbn.Load(sModelFile); 
	if(ret != _FYDL_SUCCESS)
	{
		cout<<"failed to load the DBN model from "<<sModelFile<<endl; 
		cout<<"error code is "<<ret<<endl; 
		return;
	}

	TypeDefs::Print_DBNLearningParamsT(cout, dbn.GetLearningParams()); 
	cout<<"--"<<endl; 
	TypeDefs::Print_DBNParamsT(cout, dbn.GetArchParams()); 
	cout<<"=========================="<<endl; 

	vector<Pattern*> vtr_patts; 
	if(!ReadPatterns(vtr_patts, sPattFile))
	{
		cout<<"failed to open training patterns file "<<sPattFile<<endl; 
		return; 
	}

	int32_t y_len = vtr_patts[0]->m_nYCnt;	
	double* y = new double[y_len]; 
	int32_t patts = (int32_t)vtr_patts.size(); 
	int32_t correct = 0, success = 0; 
	double error;
	double rmse = 0;

	for(int32_t i = 0; i < patts; i++) 
	{
		ret = dbn.Predict(y, y_len, vtr_patts[i]->m_x, vtr_patts[i]->m_nXCnt); 
		if(ret == _FYDL_SUCCESS)
		{
			success += 1;  
			error = Pattern::Error(y, vtr_patts[i]->m_y, y_len);
			rmse += error; 
			if(Pattern::MaxOff(y, y_len) == Pattern::MaxOff(vtr_patts[i]->m_y, vtr_patts[i]->m_nYCnt))
				correct += 1;
			else
				printf("** ");

			printf("(%d) [%s] -> [%s] | error: %.12g\n", 
					i+1, 
					Pattern::ArrayToString(vtr_patts[i]->m_y, vtr_patts[i]->m_nYCnt).c_str(), 
					Pattern::ArrayToString(y, y_len).c_str(), 
					error);
		}
		else
		{
			printf("** (%d) failed, ret is %d\n", i+1, ret);
		}
		delete vtr_patts[i]; 
	}
	rmse = sqrt(rmse / (double)success); 
	vtr_patts.clear(); 

	printf("Recall: %.6g%%, Precision: %.6g%%, RMSE: %.12g\n", 
			(double)(success * 100) / (double)patts, 
			(double)(correct * 100) / (double)(success), 
			rmse);

	dbn.Save("0101.txt");
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


