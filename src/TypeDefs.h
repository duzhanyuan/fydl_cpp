#ifndef _FYDL_TYPEDEFS_H 
#define _FYDL_TYPEDEFS_H 

#include <string>
#include <vector>
#include <iostream>
using namespace std; 
#include <string.h>
#include <stdint.h>


namespace fydl
{

// activation types
enum EActType
{
	_ACT_SIGMOID,
	_ACT_TANH,
	_ACT_RELU,
	_ACT_SOFTMAX,
	_ACT_NONE
}; 


// regularization types
enum ERegula
{
	_REGULA_L1,		// L1 
	_REGULA_L2,		// L2
	_REGULA_NONE	// no regularization
};


// learning parameters
typedef struct _learning_params_t
{
	ERegula regula;			// regularization type
	int32_t mini_batch;		// mini_batch; 0 for batch-GD, 1 for SGD, greater than 1 for mini-batch
	int32_t iterations;		// maximal iteration number
	double learning_rate;	// learning rate
	double rate_decay;		// decay of learning rate
	double epsilon;			// threshold of RMSE, used for iteration stopping
} LearningParamsT;


// architecture parameters of MLP
typedef struct _mlp_nn_params_t
{
	int32_t input;				// number of input nodes
	int32_t output;				// number of output nodes
	vector<int32_t> vtr_hidden;		// numbers of hidden nodes for each hidden layer
	EActType act_hidden;			// activation of hidden layer
	EActType act_output;			// activation of output layer
} MLPNNParamsT;


// architecture parameters of perceptron
typedef struct _perceptron_params_t
{
	int32_t input;			// number of input nodes
	int32_t output;			// number of output nodes
	EActType act_output;	// activation of output layer
} PerceptronParamsT;



class TypeDefs
{
private: 
	TypeDefs(); 
	virtual ~TypeDefs(); 

public:
	// Transform regularization type to name string
	static string RegulaName(const ERegula eRegula);
	// Retransform regularization name string to type
	static ERegula RegulaType(const char* sRegulaName);

	// Tansform activation type to name string 
	static string ActName(const EActType eActType); 	
	// Retansform activation name string to type 
	static EActType ActType(const char* sActTypeName); 	

	// Print learning parameters
	static void Print_LearningParamsT(ostream& os, const LearningParamsT paramsLearning); 

	// Print parameters of MLP
	static void Print_MLPNNParamsT(ostream& os, const MLPNNParamsT paramsMLPNN);
	
	// Print parameters of perceptron
	static void Print_PerceptronParamsT(ostream& os, const PerceptronParamsT paramsPerceptron); 
}; 


}

#endif /* _FYDL_TYPEDEFS_H */ 


