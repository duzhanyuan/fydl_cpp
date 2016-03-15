#include "TypeDefs.h"
#include "Activation.h"
using namespace fydl;


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


void TypeDefs::PrintLearningParams(ostream& os, const LearningParamsT paramsLearning)
{
	os<<"Regula:       "<<RegulaName(paramsLearning.regula)<<endl;	
	os<<"MiniBatch:    "<<paramsLearning.mini_batch<<endl; 
	os<<"Iterations:   "<<paramsLearning.iterations<<endl; 
	os<<"LearningRate: "<<paramsLearning.learning_rate<<endl; 
	os<<"RateDecay:    "<<paramsLearning.rate_decay<<endl; 
	os<<"Epsilon:      "<<paramsLearning.epsilon<<endl; 
}


void TypeDefs::PrintMLPNNParamsT(ostream& os, const MLPNNParamsT paramsMLPNN)
{
	os<<"Input:     "<<paramsMLPNN.input<<endl; 
	os<<"Output:    "<<paramsMLPNN.output<<endl; 
	os<<"Hiddens:   ";
	for(size_t i = 0; i < paramsMLPNN.vtr_hidden.size(); i++) 
	{
		if(i == 0)
			os<<paramsMLPNN.vtr_hidden[i];
		else
			os<<", "<<paramsMLPNN.vtr_hidden[i];
	}
	os<<endl; 
	os<<"HiddenActivation: "<<ActName(paramsMLPNN.act_hidden)<<endl; 
	os<<"OutputActivation: "<<ActName(paramsMLPNN.act_output)<<endl; 
}


void TypeDefs::PrintPerceptronParamsT(ostream& os, const PerceptronParamsT paramsPerceptron)
{
	os<<"Input:     "<<paramsPerceptron.input<<endl; 
	os<<"Output:    "<<paramsPerceptron.output<<endl; 
	os<<"OutputActivation: "<<ActName(paramsPerceptron.act_output)<<endl; 
}


