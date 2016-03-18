// MLP.h
//
// MultiLayer Perceptron (MLP) neural network consists one input layer, one output layer and some hidden layers. 
//
// In the MLP architecture, the number of input nodes equls the number of input signal add 1, the "1" for bias.
// The number of output nodes equls the number of output signal 
// The number of hidden layers and each layer node number are user defined
// Nodes on hidden and output layers take activation functions   
//
// IMPORTANT
// (1) when using the logit (sigmoid) activation function for the output layer make sure y values are scaled from 0 to 1
// (2) when using the tanh activation function for the output layer make sure y values are scaled from 0 to 1
// (3) sigmoid function is suggested in output layer for binary classification, and softmax is suggested for multiple classfication
// (4) the output activation can be set as 'none' for regression
// (5) tanh is recommended for hidden layers to replace sigmoid 
//
// AUTHOR
//	fengyoung (fengyoung82@sina.com)
// 
// HISTORY
//	v1.0 2016-03-14
//

#ifndef _FYDL_MLP_H 
#define _FYDL_MLP_H 

#include <string>
#include <vector>
using namespace std;
#include <stdint.h>
#include "TypeDefs.h"
#include "Pattern.h"
#include "Matrix.h"
#include "Activation.h"


// return value
#define _MLP_SUCCESS	0				// success
#define _MLP_ERROR_INPUT_NULL	-1		// the input parameters is null
#define _MLP_ERROR_WRONG_LEN	-2		// length error
#define _MLP_ERROR_MODEL_NULL	-3		// the model is null
#define _MLP_ERROR_FILE_OPEN	-4		// failed to open file
#define _MLP_ERROR_NOT_MODEL_FILE	-5	// is not MLP model file
#define _MLP_ERROR_WEIGHT_MISALIGNMENT	-6	// weight is not aligned
#define _MLP_ERROR_LAYERS_MISMATCHING	-7	// depth is not matched


namespace fydl
{

// CLASS
//	MLP - neural network based on MLP
//
// DESCRIPTION
//	This MLP neural network is based on back-propagation algorithm. 
//	The training supports online(SGD), mini-batch(MSGD) and batch(GD) mode
//
class MLP
{
public: 
	MLP();
	virtual ~MLP();

	// NAME
	//	Init - initialize the MLP parameters, including learning parameters and MLP architecture parameters 
	//	InitFromConfig - initialize the MLP parameters by values read from config file
	// 
	// DESCRIPTION
	//	nInput: number of input signal 
	//	vtrHidden: number of hidden nodes for each hidden layer, in bottom-up order
	//	nOnput: number of output signal 
	//	eActHidden, eActOutput: activation types of hidden and output layers
	//	pLearningParamsT: learning parameters
	//	sConfigFile: config file
	//
	// RETURN
	//	true for success, false for some errors
	void Init(const int32_t nInput, vector<int32_t>& vtrHidden, const int32_t nOutput, 
			const EActType eActHidden = _ACT_RELU, const EActType eActOutput = _ACT_SIGMOID, 
			const LearningParamsT* pLearningParamsT = NULL); 
	bool InitFromConfig(const char* sConfigFile, const int32_t nInput, const int32_t nOutput);

	// NAME
	//	Train - train MLP model by patterns
	// 
	// DESCRIPTION
	//	vtrPatts: list of training patterns
	void Train(vector<Pattern*>& vtrPatts);

	// NAME
	//	Predict - calculate prediction of input signal based on current MLP model
	// 
	// DESCRIPTION
	//	y: output parameter, the prediction result
	//	y_len: size of y
	//	x: input signal
	//	x_len: size of x
	// 
	// RETURN
	//	Return _MLP_SUCCESS for success, others for some errors
	int32_t Predict(double* y, const int32_t y_len, const double* x, const int32_t x_len); 

	// NAME
	//	Save - save current MLP model to file
	//
	// DESCRIPTION
	//	sFile - MLP model file
	// 
	// RETURN
	//	true for success, false for some errors
	int32_t Save(const char* sFile, const char* sTitle = "MLP Neural Network"); 
	
	// NAME
	//	Load - load MLP model from file to construct current object
	//
	// DESCRIPTION
	//	sFile - MLP model file
	// 
	// RETURN
	//	true for success, false for some errors
	int32_t Load(const char* sFile, const char* sCheckTitle = NULL); 

	// Get learning parameters
	LearningParamsT GetLearningParams(); 
	
	// Get architecture parameters 
	MLPNNParamsT GetMLPNNParams(); 

private:
	// Create inner objects, allocate menory
	void Create();
	// Release inner objects
	void Release();

	// NAME
	//	FeedForward - forward phase
	// 
	// DESCRIPTION
	//	in_vals: input signal
	//	in_len: size of input signal
	void FeedForward(const double* in_vals, const int32_t in_len); 
	
	// NAME
	//	BackPropagate - backword phase
	//	
	// DESCRIPTION
	//	out_vals: output signal
	//	out_len: size of output signal
	//
	// RETURN
	//	The error (difference) between output layer and output signal 
	double BackPropagate(const double* out_vals, const int32_t out_len); 
	
	// NAME
	//	ActivateForward - forward activation, activate upper layer by lower layer
	//
	// DESCRIPTION
	//	up_a: out parameter, upper layer
	//	up_size: size of upper layer
	//	low_a: lower layer
	//	low_size: size of lower layer
	//	w: transform matrix
	//	up_act_type: activation function type of upper layer
	void ActivateForward(double* up_a, const int32_t up_size, const double* low_a, const int32_t low_size, 
			Matrix& w, const EActType up_act_type); 

	// NAME
	//	UpdateTransformMatrices - update all transform matrices in MLP
	// 
	// DESCRIPTION
	//	After transform matrices updating, elements of all change matrices should be set as 0
	//	
	//	learning_rate: learning rate
	void UpdateTransformMatrices(const double learning_rate); 

	// NAME
	//	Validation - validate current MLP model
	// 
	// DESCRIPTION	
	//	vtrPatts: training patterns list
	//	nBackCnt: number of validation patterns, which are at back of the list
	// 
	// RETRUN
	//	The precision and RMSE
	pair<double, double> Validation(vector<Pattern*>& vtrPatts, const int32_t nBackCnt);

private: 
	LearningParamsT m_paramsLearning;	// learning parameters
	int32_t m_nIters;					// real iteration times
	MLPNNParamsT m_paramsNN;			// architecture parameters of MLP

	Matrix* m_whs;	// transform matrices of hidden layers
	Matrix m_wo;	// transform matrix of output layer

	double* m_ai;	// input layer
	double** m_ahs;	// hidden layers
	double* m_ao;	// output layer

	double** m_dhs;	// delta arrays of hidden layers
	double* m_do;	// delta array of output layer
	Matrix* m_chs;	// change matrices of hidden layers
	Matrix m_co;	// change matrix of output layer
}; 

}

#endif /* _FYDL_MLP_H */


