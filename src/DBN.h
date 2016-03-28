// DBN.h
//
// Deep Belief Network (DBN) is made up by some RBMs and one MLP
//
// RBMs in DBN is used for feature detecting, the training of them is unsuppervised, it is thought as the pre-train of MLP
// MLP in DBN is used for classification or regression, the training of MLP is suppervised, it is thought as fine-tuning of DBN
//
// AUTHOR
//	fengyoung (fengyoung82@sina.com)
// 
// HISTORY
//	v1.0 2016-03-22
//

#ifndef _FYDL_DBN_H 
#define _FYDL_DBN_H 

#include <stdint.h>
#include "RBM.h"
#include "MLP.h"


namespace fydl
{

// CLASS
//	DBN - Deep Belief Network
//
// DESCRIPTION
//
class DBN 
{
public:
	DBN(); 
	virtual ~DBN();
	
	// NAME
	//	Init - initialize the DBN parameters, including learning parameters and DBN architecture parameters 
	//	InitFromConfig - initialize the DBN parameters by values read from config file
	// 
	// DESCRIPTION
	//	dbnParamsT: architecture parameters of DBN
	//	dbnLearningParamsT: learning parameters
	//	sConfigFile: config file
	//	nInput: number of input signal 
	//	nOnput: number of output signal 
	//
	// RETURN
	//	true for success, false for some errors
	void Init(const DBNParamsT dbnParamsT, const DBNLearningParamsT dbnLearningParamsT); 
	bool InitFromConfig(const char* sConfigFile, const int32_t nInput, const int32_t nOutput);
	
	// NAME
	//	Train - train DBN model by patterns
	// 
	// DESCRIPTION
	//	vtrPatts: list of training patterns
	void Train(vector<Pattern*>& vtrPatts);

	// NAME
	//	Predict - calculate prediction of input signal based on current DBN model
	// 
	// DESCRIPTION
	//	y: output parameter, the prediction result
	//	y_len: size of y
	//	x: input signal
	//	x_len: size of x
	// 
	// RETURN
	//	Return _FYDL_SUCCESS for success, others for some errors
	int32_t Predict(double* y, const int32_t y_len, const double* x, const int32_t x_len); 

	// NAME
	//	Save - save current MLP model to file
	//	Load - load MLP model from file to construct current object
	//
	// DESCRIPTION
	//	sFile - MLP model file
	// 
	// RETURN
	//	Return _FYDL_SUCCESS for success, others for some errors
	int32_t Save(const char* sFile, const char* sTitle = "Deep Belief Network (DBN)"); 
	int32_t Load(const char* sFile, const char* sCheckTitle = "Deep Belief Network (DBN)"); 

	// Get learning parameters
	DBNLearningParamsT GetLearningParams(); 
	
	// Get architecture parameters 
	DBNParamsT GetArchParams(); 

private: 
	// Create model variables, allocate menory for them 
	void Create();
	// Release inner objects
	void Release();
	// Create assistant variables of one RBM, allocate menory for them 
	void CreateAssistant(); 
	// Release assistant variables of one RBM
	void ReleaseAssistant(); 


	void PreTrain(vector<Pattern*>& vtrPatts);
	void FineTuning(vector<Pattern*>& vtrPatts);

	int32_t Generate(double** ao, const double* x, const int32_t out_rbm_idx);

	// NAME
	//	Validation - validate current DBN model
	// 
	// DESCRIPTION	
	//	vtrPatts: training patterns list
	//	nBackCnt: number of validation patterns, which are at back of the list
	// 
	// RETRUN
	//	The precision and RMSE
	pair<double, double> Validation(vector<Pattern*>& vtrPatts, const int32_t nBackCnt);

private: 
	DBNLearningParamsT m_paramsLearning;	// learning parameters of DBN
	DBNParamsT m_paramsDBN; 				// architecture parameters of DBN

	RBM* m_rbms;
	MLP m_mlp;

	double** m_rbms_ao;
};

}


#endif /* _FYDL_DBN_H */


