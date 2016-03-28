#ifndef _FYDL_TYPEDEFS_H 
#define _FYDL_TYPEDEFS_H 

#include <string>
#include <vector>
#include <map>
#include <iostream>
using namespace std; 
#include <string.h>
#include <stdint.h>

// return value
#define _FYDL_SUCCESS	0				// success
#define _FYDL_ERROR_INPUT_NULL	-1		// the input parameters is null
#define _FYDL_ERROR_WRONG_LEN	-2		// length error
#define _FYDL_ERROR_MODEL_NULL	-3		// the model is null
#define _FYDL_ERROR_FILE_OPEN	-4		// failed to open file
#define _FYDL_ERROR_NOT_MODEL_FILE	-5	// is not model file
#define _FYDL_ERROR_WEIGHT_MISALIGNMENT	-6	// weight is not aligned
#define _FYDL_ERROR_LAYERS_MISMATCHING	-7	// depth is not matched
#define _FYDL_ERROR_LERANING_PARAMS		-8
#define _FYDL_ERROR_ACH_PARAMS			-9
#define _FYDL_ERROR_MODEL_DATA			-10


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


//////////////////////////////////////////////////////////////////
// typedefs of Perceptron & MLP

// learning parameters of perceptron & MLP
typedef struct _perceptron_learning_params_t
{
	ERegula regula;			// regularization type
	int32_t mini_batch;		// mini_batch; 0 for batch-GD, 1 for SGD, greater than 1 for mini-batch
	int32_t iterations;		// maximal iteration number
	double learning_rate;	// learning rate
	double rate_decay;		// decay of learning rate
	double epsilon;			// threshold of RMSE, used for iteration stopping
} PerceptronLearningParamsT, MLPLearningParamsT; 


// architecture parameters of perceptron
typedef struct _perceptron_params_t
{
	int32_t input;			// number of input nodes
	int32_t output;			// number of output nodes
	EActType act_output;	// activation of output layer
} PerceptronParamsT;


// architecture parameters of MLP
typedef struct _mlp_params_t
{
	int32_t input;			// number of input nodes
	int32_t output;			// number of output nodes
	vector<int32_t> vtr_hidden;		// numbers of hidden nodes for each hidden layer
	EActType act_hidden;			// activation of hidden layer
	EActType act_output;			// activation of output layer
} MLPParamsT;



//////////////////////////////////////////////////////////////////
// typedefs of RBM

// RBM type 
enum ERBMType
{
	_GAUSS_BERNOULLI_RBM,       // Gauss-Bernoulli RBM, for contiuous input
	_BINOMIAL_BERNOULLI_RBM,    // Binomial-Bernoulli RBM, for discrete input
	_UNKNOWN_RBM
};


// learning parameters of RBM
typedef struct _rbm_learning_params_t
{
	int32_t gibbs_steps;    // number of gibbs sample steps
	int32_t mini_batch;		// mini_batch;
	int32_t iterations;     // maximal iteration number
	double learning_rate;   // learning rate
	double rate_decay;      // decay of learning rate
	double epsilon;         // threshold of RMSE, used for iteration stopping
} RBMLearningParamsT;


// architecture parameters of RBM
typedef struct _rbm_params_t
{
	ERBMType rbm_type;	// RBM type
	int32_t visible;	// number of visible neurons
	int32_t hidden;		// number of hidden neurons
} RBMParamsT;


//////////////////////////////////////////////////////////////////
// typedefs of DBN

// learning parameters of DBN
typedef struct _dbn_learning_params_t
{
	RBMLearningParamsT rbm_learning_params;		// learning parameters of RBMs, which on the bottom of DBN
	MLPLearningParamsT mlp_learning_params;		// learning parameters of MLP, which on the top of DBN
} DBNLearningParamsT;


// architecture parameters of DBN
typedef struct _dbn_params_t
{
	int32_t input;				// number of input nodes
	int32_t output;					// number of output nodes
	// for RBMs
	ERBMType rbms_type;			// type of RBMs	
	vector<int32_t> vtr_rbms_hidden;	// numbers of hidden nodes for each RBM, lower hidden layer would be visible layer of upper RBM  	
	// for MLP
	vector<int32_t> vtr_mlp_hidden;		// numbers of hidden nodes for each hidden layer of MLP
	EActType mlp_act_hidden;			// activation of hidden layer of MLP
	EActType mlp_act_output;			// activation of output layer of MLP
} DBNParamsT; 


//////////////////////////////////////////////////////////////////
// class TypeDefs

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

	// Tansform RBM type to name string 
	static string RBMName(const ERBMType eRBMType); 
	// Retansform RBM name string to type 
	static ERBMType RBMType(const char* sRBMName); 	

	static void Print_PerceptronLearningParamsT(ostream& os, const PerceptronLearningParamsT perceptronLearningParamsT); 
	static bool Read_PerceptronLearningParamsT(PerceptronLearningParamsT& perceptronLearningParamsT, istream& is); 

	static void Print_PerceptronParamsT(ostream& os, const PerceptronParamsT perceptronParamsT); 
	static bool Read_PerceptronParamsT(PerceptronParamsT& perceptronParamsT, istream& is); 

	static void Print_MLPParamsT(ostream& os, const MLPParamsT mlpParamsT); 
	static bool Read_MLPParamsT(MLPParamsT& mlpParamsT, istream& is); 
	
	static void Print_RBMLearningParamsT(ostream& os, const RBMLearningParamsT rbmLearningParamsT);
	static bool Read_RBMLearningParamsT(RBMLearningParamsT& rbmLearningParamsT, istream& is);

	static void Print_RBMParamsT(ostream& os, const RBMParamsT rbmParamsT); 
	static bool Read_RBMParamsT(RBMParamsT& rbmParamsT, istream& is); 

	static void Print_DBNLearningParamsT(ostream& os, const DBNLearningParamsT dbnLearningParamsT);
	static bool Read_DBNLearningParamsT(DBNLearningParamsT& dbnLearningParamsT, istream& is);

	static void Print_DBNParamsT(ostream& os, const DBNParamsT dbnParamsT); 
	static bool Read_DBNParamsT(DBNParamsT& dbnParamsT, istream& is); 

}; 


}

#endif /* _FYDL_TYPEDEFS_H */ 


