#include "TypeDefs.h"
#include "StringArray.h"
#include "Activation.h"
using namespace fydl;
#include <stdio.h>



//////////////////////////////////////////////////////////////////////////////////////////
// Construction & Destruction 

TypeDefs::TypeDefs()
{
}


TypeDefs::~TypeDefs()
{
}


//////////////////////////////////////////////////////////////////////////////////////////
// Operations 

string TypeDefs::RegulaName(const ERegula eRegula)
{
	switch(eRegula)
	{
		case _REGULA_L1:
			return "L1";
		case _REGULA_L2:
			return "L2";
		default:
			break;	
	}
	return "none";
}


ERegula TypeDefs::RegulaType(const char* sRegulaName)
{
	if(strcmp(sRegulaName, "L1") == 0)
		return _REGULA_L1; 
	else if(strcmp(sRegulaName, "L2") == 0)
		return _REGULA_L2; 
	else
		return _REGULA_NONE; 
}


string TypeDefs::ActName(const EActType eActType)
{
	switch(eActType)
	{
		case _ACT_SIGMOID:
			return "sigmoid";
		case _ACT_TANH:
			return "tanh";
		case _ACT_RELU:
			return "relu";
		case _ACT_SOFTMAX: 
			return "softmax";
		default:
			break;
	}
	return "none";
}


EActType TypeDefs::ActType(const char* sActTypeName)
{
	if(strcmp(sActTypeName, "sigmoid") == 0)
		return _ACT_SIGMOID; 
	else if(strcmp(sActTypeName, "tanh") == 0)
		return _ACT_TANH; 
	else if(strcmp(sActTypeName, "relu") == 0)
		return _ACT_RELU; 
	else if(strcmp(sActTypeName, "softmax") == 0)
		return _ACT_SOFTMAX; 
	else
		return _ACT_NONE; 
}


string TypeDefs::RBMName(const ERBMType eRBMType)
{
	switch(eRBMType)
	{
		case _GAUSS_BERNOULLI_RBM: 
			return "GB-RBM"; 
		case _BINOMIAL_BERNOULLI_RBM: 
			return "BB-RBM"; 
		default:
			break; 
	}
	return "Unknown";
}


ERBMType TypeDefs::RBMType(const char* sRBMName)
{
	if(strcmp(sRBMName, "GB-RBM") == 0)
		return _GAUSS_BERNOULLI_RBM; 
	else if(strcmp(sRBMName, "BB-RBM") == 0)
		return _BINOMIAL_BERNOULLI_RBM; 
	else
		return _UNKNOWN_RBM;
}



void TypeDefs::Print_PerceptronLearningParamsT(ostream& os, const PerceptronLearningParamsT perceptronLearningParamsT)
{
	os<<"Regula:"<<RegulaName(perceptronLearningParamsT.regula)<<endl; 
	os<<"MiniBatch:"<<perceptronLearningParamsT.mini_batch<<endl; 
	os<<"Iterations:"<<perceptronLearningParamsT.iterations<<endl; 
	os<<"LearningRate:"<<perceptronLearningParamsT.learning_rate<<endl; 
	os<<"RateDecay:"<<perceptronLearningParamsT.rate_decay<<endl; 
	os<<"Epsilon:"<<perceptronLearningParamsT.epsilon<<endl; 
}


bool TypeDefs::Read_PerceptronLearningParamsT(PerceptronLearningParamsT& perceptronLearningParamsT, istream& is)
{
	int32_t cnt = 0;
	string str; 
	while(cnt < 6)
	{
		std::getline(is, str); 
		StringArray ar(str.c_str(), ":"); 
		if(ar.Count() != 2)
			return false;
		if(ar.GetString(0) == "Regula")
			perceptronLearningParamsT.regula = RegulaType(ar.GetString(1).c_str()); 
		else if(ar.GetString(0) == "MiniBatch")
			sscanf(ar.GetString(1).c_str(), "%d", &(perceptronLearningParamsT.mini_batch));
		else if(ar.GetString(0) == "Iterations")
			sscanf(ar.GetString(1).c_str(), "%d", &(perceptronLearningParamsT.iterations));
		else if(ar.GetString(0) == "LearningRate")
			sscanf(ar.GetString(1).c_str(), "%lf", &(perceptronLearningParamsT.learning_rate));
		else if(ar.GetString(0) == "RateDecay")
			sscanf(ar.GetString(1).c_str(), "%lf", &(perceptronLearningParamsT.rate_decay));
		else if(ar.GetString(0) == "Epsilon")
			sscanf(ar.GetString(1).c_str(), "%lf", &(perceptronLearningParamsT.epsilon));
		else
			return false;
		cnt++; 
	}
	return true; 
}


void TypeDefs::Print_PerceptronParamsT(ostream& os, const PerceptronParamsT perceptronParamsT)
{
	os<<"Input:"<<perceptronParamsT.input-1<<endl; 
	os<<"Output:"<<perceptronParamsT.output<<endl; 
	os<<"Activation:"<<ActName(perceptronParamsT.act_output)<<endl; 
}


bool TypeDefs::Read_PerceptronParamsT(PerceptronParamsT& perceptronParamsT, istream& is)
{
	int32_t cnt = 0;
	string str; 
	while(cnt < 3)
	{
		std::getline(is, str); 
		StringArray ar(str.c_str(), ":"); 
		if(ar.Count() != 2) 
			return false; 
		if(ar.GetString(0) == "Input")
		{
			sscanf(ar.GetString(1).c_str(), "%d", &(perceptronParamsT.input));
			perceptronParamsT.input += 1; // add 1 for bias nodes
		}
		else if(ar.GetString(0) == "Output")
			sscanf(ar.GetString(1).c_str(), "%d", &(perceptronParamsT.output));
		else if(ar.GetString(0) == "Activation")
			perceptronParamsT.act_output = ActType(ar.GetString(1).c_str());
		else
			return false;
		cnt++; 
	}
	return true; 
}


void TypeDefs::Print_MLPParamsT(ostream& os, const MLPParamsT mlpParamsT)
{
	os<<"Input:"<<mlpParamsT.input-1<<endl; 
	os<<"Output:"<<mlpParamsT.output<<endl; 
	for(size_t i = 0; i < mlpParamsT.vtr_hidden.size(); i++) 
	{
		if(i == 0)
			os<<"Hiddens:"<<mlpParamsT.vtr_hidden[i]; 
		else
			os<<","<<mlpParamsT.vtr_hidden[i]; 
	}
	os<<endl;
	os<<"ActHidden:"<<ActName(mlpParamsT.act_hidden)<<endl; 
	os<<"ActOutput:"<<ActName(mlpParamsT.act_output)<<endl; 
}


bool TypeDefs::Read_MLPParamsT(MLPParamsT& mlpParamsT, istream& is)
{
	int32_t cnt = 0, hidden;
	string str;
	while(cnt < 5)
	{
		std::getline(is, str); 
		StringArray ar(str.c_str(), ":"); 
		if(ar.Count() != 2) 
			return false; 
		if(ar.GetString(0) == "Input")
		{
			sscanf(ar.GetString(1).c_str(), "%d", &(mlpParamsT.input));
			mlpParamsT.input += 1; // add 1 for bias nodes
		}
		else if(ar.GetString(0) == "Output")
			sscanf(ar.GetString(1).c_str(), "%d", &(mlpParamsT.output));
		else if(ar.GetString(0) == "Hiddens")
		{
			mlpParamsT.vtr_hidden.clear(); 
			StringArray array(ar.GetString(1).c_str(), ","); 
			for(int32_t i = 0; i < array.Count(); i++) 
			{
				sscanf(array.GetString(i).c_str(), "%d", &hidden); 
				mlpParamsT.vtr_hidden.push_back(hidden); 
			}
		}
		else if(ar.GetString(0) == "ActHidden")
			mlpParamsT.act_hidden = ActType(ar.GetString(1).c_str());
		else if(ar.GetString(0) == "ActOutput")
			mlpParamsT.act_output = ActType(ar.GetString(1).c_str());
		else
			return false;
		cnt++; 
	}
	return true; 
}


void TypeDefs::Print_RBMLearningParamsT(ostream& os, const RBMLearningParamsT rbmLearningParamsT)
{
	os<<"GibbsSteps:"<<rbmLearningParamsT.gibbs_steps<<endl; 
	os<<"MiniBatch:"<<rbmLearningParamsT.mini_batch<<endl; 
	os<<"Iterations:"<<rbmLearningParamsT.iterations<<endl; 
	os<<"LearningRate:"<<rbmLearningParamsT.learning_rate<<endl; 
	os<<"RateDecay:"<<rbmLearningParamsT.rate_decay<<endl; 
	os<<"Epsilon:"<<rbmLearningParamsT.epsilon<<endl; 
}


bool TypeDefs::Read_RBMLearningParamsT(RBMLearningParamsT& rbmLearningParamsT, istream& is)
{
	int32_t cnt = 0;
	string str; 
	while(cnt < 6)
	{
		std::getline(is, str); 
		StringArray ar(str.c_str(), ":"); 
		if(ar.Count() != 2)
			return false;
		if(ar.GetString(0) == "GibbsSteps")
			sscanf(ar.GetString(1).c_str(), "%d", &(rbmLearningParamsT.gibbs_steps));
		else if(ar.GetString(0) == "MiniBatch")
			sscanf(ar.GetString(1).c_str(), "%d", &(rbmLearningParamsT.mini_batch));
		else if(ar.GetString(0) == "Iterations")
			sscanf(ar.GetString(1).c_str(), "%d", &(rbmLearningParamsT.iterations));
		else if(ar.GetString(0) == "LearningRate")
			sscanf(ar.GetString(1).c_str(), "%lf", &(rbmLearningParamsT.learning_rate));
		else if(ar.GetString(0) == "RateDecay")
			sscanf(ar.GetString(1).c_str(), "%lf", &(rbmLearningParamsT.rate_decay));
		else if(ar.GetString(0) == "Epsilon")
			sscanf(ar.GetString(1).c_str(), "%lf", &(rbmLearningParamsT.epsilon));
		else
			return false;
		cnt++; 
	}
	return true; 
}


void TypeDefs::Print_RBMParamsT(ostream& os, const RBMParamsT rbmParamsT)
{
	os<<"RBMType:"<<RBMName(rbmParamsT.rbm_type)<<endl; 
	os<<"Visible:"<<rbmParamsT.visible<<endl; 
	os<<"Hidden:"<<rbmParamsT.hidden<<endl; 
}


bool TypeDefs::Read_RBMParamsT(RBMParamsT& rbmParamsT, istream& is)
{
	int32_t cnt = 0;
	string str; 
	while(cnt < 3)
	{
		std::getline(is, str); 
		StringArray ar(str.c_str(), ":"); 
		if(ar.Count() != 2) 
			return false; 
		if(ar.GetString(0) == "RBMType")
			rbmParamsT.rbm_type = RBMType(ar.GetString(1).c_str()); 
		else if(ar.GetString(0) == "Visible")
			sscanf(ar.GetString(1).c_str(), "%d", &(rbmParamsT.visible));
		else if(ar.GetString(0) == "Hidden")
			sscanf(ar.GetString(1).c_str(), "%d", &(rbmParamsT.hidden));
		else
			return false;
		cnt++; 
	}
	return true; 
}


void TypeDefs::Print_DBNLearningParamsT(ostream& os, const DBNLearningParamsT dbnLearningParamsT)
{
	os<<"RBMs_GibbsSteps:"<<dbnLearningParamsT.rbm_learning_params.gibbs_steps<<endl; 
	os<<"RBMs_MiniBatch:"<<dbnLearningParamsT.rbm_learning_params.mini_batch<<endl; 
	os<<"RBMs_Iterations:"<<dbnLearningParamsT.rbm_learning_params.iterations<<endl; 
	os<<"RBMs_LearningRate:"<<dbnLearningParamsT.rbm_learning_params.learning_rate<<endl; 
	os<<"RBMs_RateDecay:"<<dbnLearningParamsT.rbm_learning_params.rate_decay<<endl; 
	os<<"RBMs_Epsilon:"<<dbnLearningParamsT.rbm_learning_params.epsilon<<endl; 
	os<<"MLP_Regula:"<<RegulaName(dbnLearningParamsT.mlp_learning_params.regula)<<endl; 
	os<<"MLP_MiniBatch:"<<dbnLearningParamsT.mlp_learning_params.mini_batch<<endl; 
	os<<"MLP_Iterations:"<<dbnLearningParamsT.mlp_learning_params.iterations<<endl; 
	os<<"MLP_LearningRate:"<<dbnLearningParamsT.mlp_learning_params.learning_rate<<endl; 
	os<<"MLP_RateDecay:"<<dbnLearningParamsT.mlp_learning_params.rate_decay<<endl; 
	os<<"MLP_Epsilon:"<<dbnLearningParamsT.mlp_learning_params.epsilon<<endl; 
}


bool TypeDefs::Read_DBNLearningParamsT(DBNLearningParamsT& dbnLearningParamsT, istream& is)
{
	int32_t cnt = 0;
	string str; 
	while(cnt < 12)
	{
		std::getline(is, str); 
		StringArray ar(str.c_str(), ":"); 
		if(ar.Count() != 2) 
			return false; 
		if(ar.GetString(0) == "RBMs_GibbsSteps")
			sscanf(ar.GetString(1).c_str(), "%d", &(dbnLearningParamsT.rbm_learning_params.gibbs_steps));
		else if(ar.GetString(0) == "RBMs_MiniBatch")
			sscanf(ar.GetString(1).c_str(), "%d", &(dbnLearningParamsT.rbm_learning_params.mini_batch));
		else if(ar.GetString(0) == "RBMs_Iterations")
			sscanf(ar.GetString(1).c_str(), "%d", &(dbnLearningParamsT.rbm_learning_params.iterations));
		else if(ar.GetString(0) == "RBMs_LearningRate")
			sscanf(ar.GetString(1).c_str(), "%lf", &(dbnLearningParamsT.rbm_learning_params.learning_rate));
		else if(ar.GetString(0) == "RBMs_RateDecay")
			sscanf(ar.GetString(1).c_str(), "%lf", &(dbnLearningParamsT.rbm_learning_params.rate_decay));
		else if(ar.GetString(0) == "RBMs_Epsilon")
			sscanf(ar.GetString(1).c_str(), "%lf", &(dbnLearningParamsT.rbm_learning_params.epsilon));
		else if(ar.GetString(0) == "MLP_Regula")
			dbnLearningParamsT.mlp_learning_params.regula = RegulaType(ar.GetString(1).c_str()); 
		else if(ar.GetString(0) == "MLP_MiniBatch")
			sscanf(ar.GetString(1).c_str(), "%d", &(dbnLearningParamsT.mlp_learning_params.mini_batch));
		else if(ar.GetString(0) == "MLP_Iterations")
			sscanf(ar.GetString(1).c_str(), "%d", &(dbnLearningParamsT.mlp_learning_params.iterations));
		else if(ar.GetString(0) == "MLP_LearningRate")
			sscanf(ar.GetString(1).c_str(), "%lf", &(dbnLearningParamsT.mlp_learning_params.learning_rate));
		else if(ar.GetString(0) == "MLP_RateDecay")
			sscanf(ar.GetString(1).c_str(), "%lf", &(dbnLearningParamsT.mlp_learning_params.rate_decay));
		else if(ar.GetString(0) == "MLP_Epsilon")
			sscanf(ar.GetString(1).c_str(), "%lf", &(dbnLearningParamsT.mlp_learning_params.epsilon));
		else
			return false;
		cnt++; 
	}
	return true; 
}


void TypeDefs::Print_DBNParamsT(ostream& os, const DBNParamsT dbnParamsT)
{
	os<<"Input:"<<dbnParamsT.input<<endl; 
	os<<"Output:"<<dbnParamsT.output<<endl; 
	os<<"RBMs_Type:"<<RBMName(dbnParamsT.rbms_type)<<endl; 
	for(size_t i = 0; i < dbnParamsT.vtr_rbms_hidden.size(); i++) 
	{
		if(i == 0)
			os<<"RBMs_Hiddens:"<<dbnParamsT.vtr_rbms_hidden[i]; 
		else
			os<<","<<dbnParamsT.vtr_rbms_hidden[i]; 
	}
	os<<endl; 
	for(size_t i = 0; i < dbnParamsT.vtr_mlp_hidden.size(); i++) 
	{
		if(i == 0)
			os<<"MLP_Hiddens:"<<dbnParamsT.vtr_mlp_hidden[i]; 
		else
			os<<","<<dbnParamsT.vtr_mlp_hidden[i]; 
	}
	os<<endl; 
	os<<"MLP_ActHidden:"<<ActName(dbnParamsT.mlp_act_hidden)<<endl; 
	os<<"MLP_ActOutput:"<<ActName(dbnParamsT.mlp_act_output)<<endl; 
}


bool TypeDefs::Read_DBNParamsT(DBNParamsT& dbnParamsT, istream& is)
{
	int32_t cnt = 0, hidden;
	string str; 
	while(cnt < 7)
	{
		std::getline(is, str); 
		StringArray ar(str.c_str(), ":"); 
		if(ar.Count() != 2) 
			return false; 
		else if(ar.GetString(0) == "Input")
			sscanf(ar.GetString(1).c_str(), "%d", &(dbnParamsT.input)); 
		else if(ar.GetString(0) == "Output")
			sscanf(ar.GetString(1).c_str(), "%d", &(dbnParamsT.output)); 
		else if(ar.GetString(0) == "RBMs_Type")
			dbnParamsT.rbms_type = RBMType(ar.GetString(1).c_str()); 
		else if(ar.GetString(0) == "RBMs_Hiddens")
		{
			dbnParamsT.vtr_rbms_hidden.clear(); 
			StringArray array(ar.GetString(1).c_str(), ","); 
			for(int32_t i = 0; i < array.Count(); i++) 
			{
				sscanf(array.GetString(i).c_str(), "%d", &hidden); 
				dbnParamsT.vtr_rbms_hidden.push_back(hidden); 
			}
		}
		else if(ar.GetString(0) == "MLP_Hiddens")
		{
			dbnParamsT.vtr_mlp_hidden.clear(); 
			StringArray array(ar.GetString(1).c_str(), ","); 
			for(int32_t i = 0; i < array.Count(); i++) 
			{
				sscanf(array.GetString(i).c_str(), "%d", &hidden); 
				dbnParamsT.vtr_mlp_hidden.push_back(hidden); 
			}
		}
		else if(ar.GetString(0) == "MLP_ActHidden")
			dbnParamsT.mlp_act_hidden = ActType(ar.GetString(1).c_str()); 
		else if(ar.GetString(0) == "MLP_ActOutput")
			dbnParamsT.mlp_act_output = ActType(ar.GetString(1).c_str()); 
		else
			return false; 
		cnt++; 	
	}
	return true; 
}



