// RBM.h
//
// Restricted Boltzmann Machine (RBM) consists one visible layer and one hidden layer 
//
// In the RBM there are only connections between visible and hidden layers and no connections among units in the same layer.
// The number of visible nodes equls the number of output signal. 
// The number of hidden nodes are user defined.
// Connections between visible neurons and hidden neurons are undirected, and each of them takes sigmoid as activation function.
//
// IMPORTANT
// (1) when the input signal is discrete, the type of the RBM should be set as 'BB-RBM' (Binomial-Bernoulli RBM) 
// (2) when the input signal is continuous, the type of the RBM should be set as 'GB-RBM' (Gauss-Bernoulli RBM) 
//
// AUTHOR
//	fengyoung (fengyoung82@sina.com)
// 
// HISTORY
//	v1.0 2016-03-18
//

#ifndef _FYDL_RBM_H 
#define _FYDL_RBM_H 

#include <vector>
using namespace std; 
#include <stdint.h>
#include "Matrix.h"
#include "TypeDefs.h"
#include "Pattern.h"


namespace fydl
{

// CLASS
//	RBM - Restricted Boltzmann Machine 
//
// DESCRIPTION
//	This RBM is based on Contrastive Divergence (CD) algorithm.
//
class RBM
{
public:
	RBM(); 
	virtual ~RBM();

	// NAME
	//	Init - initialize the RBM parameters, including learning parameters and RBM architecture parameters 
	//	InitFromConfig - initialize the RBM parameters by values read from config file
	// 
	// DESCRIPTION
	//	rbmParamsT: architecture parameters of RBM
	//	rbmLearningParamsT: learning parameters
	//	sConfigFile: config file
	//
	// RETURN
	//	true for success, false for some errors
	void Init(const RBMParamsT rbmParamsT, const RBMLearningParamsT rbmLearningParamsT); 
	bool InitFromConfig(const char* sConfigFile, const int32_t nVisible); 

	// NAME
	//	Train - train RBM model by patterns
	// 
	// DESCRIPTION
	//	vtrPatts: list of training patterns
	void Train(vector<Pattern*>& vtrPatts);
	
	// NAME
	//	Reconstruct - reconstruct input signl based on current RBM model
	// 
	// DESCRIPTION
	//	xr: output parameter, reconstructed signal
	//	x: input signal
	//	x_len: size of x
	// 
	// RETURN
	//	error betwen input and reconstructed signal
	double Reconstruct(double* xr, const double* x, const int32_t len); 

	// NAME
	//	PropagateForward - forward propagation, transform input signal to output signal based on current RBM model
	// 
	// DESCRIPTION
	//	y: output parameter, the output signal
	//	y_len: size of y
	//	x: input signal
	//	x_len: size of x
	// 
	// RETURN
	//	Return _FYDL_SUCCESS for success, others for some errors
	int32_t PropagateForward(double* y, const int32_t y_len, const double* x, const int32_t x_len); 

	// NAME
	//	Save - save current RBM model to file
	//	Load - load RBM model from file to construct current object
	//
	// DESCRIPTION
	//	sFile - RBM model file
	//	sTitle - title of the model file, which would be print as the first line of the file 	
	//	sCheckTitle - title of the model file, if set as NULL, no checking
	// 
	// RETURN
	//	Return _FYDL_SUCCESS for success, others for some errors
	int32_t Save(const char* sFile, const char* sTitle = "Restricted Boltzmann Machine (RBM)");  
	int32_t Load(const char* sFile, const char* sCheckTitle = "Restricted Boltzmann Machine (RBM)"); 

	// Get learning parameters
	RBMLearningParamsT GetLearningParams(); 

	// Get architecture parameters 
	RBMParamsT GetArchParams();	

	// NAME
	//	CDk_Step - one step of CD-k
	//
	// DESCRIPTION
	//	x - input signal (independent variables of pattern)
	//	x_len - length of x
	//	bFirst - if it is the first time to call this fucntion 
	//
	// RETURN
	//	The error in this training step
	double CDk_Step(const double* x, const int32_t x_len, const bool bFirst = false);
	
	// NAME
	//	ModelUpdate - update model parameters such as transform matrix, visible bias and hidden bias
	// 
	// DESCRIPTION
	//	After transform matrices updating, elements of change matrix and arrays should be set as 0
	//	
	//	dLearningRate: learning rate
	void ModelUpdate(const double dLearningRate); 

private: 
	// Create model variables, allocate menory for them 
	void Create();
	// Release inner objects
	void Release();
	// Create assistant variables, allocate menory for them 
	void CreateAssistant();
	// Release assistant variables
	void ReleaseAssistant(); 
	
	// NAME
	//	ContrastiveDivergrence - Contrastive Divergence (CD) algorithm
	//
	// DESCRIPTION
	//	in_vals: input signal
	//	in_len: size of input signal
	//
	// RETURN
	//	the error between input and reconstructed signal 
	double ContrastiveDivergrence(const double* in_vals, const int32_t in_len); 
	
	// NAME
	//	Sample_HbyV - take samples of hidden neurons, according to the distribution P(h=1|v)   
	//	Sample_VbyH - take samples of visible neurons, according to the distribution P(v=x|h)
	//
	// DESCRITPTION
	//	h: means of hidden neourns 
	//	hs: the sample of the hidden layer
	//	hbias: bias of hidden layer
	//	v: mean of visible neurons
	//	vs: the sample of the visible layer
	//	vbias: bias of visible layer
	//	w: transform matrix between visible and hidden layers  
	//	eRBMType: type of RBM
	void Sample_HbyV(double* h, double* hs, const double* hbias, const double* vs, Matrix& w); 
	void Sample_VbyH(double* v, double* vs, const double* vbias, const double* hs, Matrix& w, const ERBMType eRBMType); 

	// NAME
	//	PropagateUp - propagate from visible layer to hidden layer
	//	PropagateDown - propagate from hidden layer to visible layer
	//
	// DESCRITPTION
	//	h: hidden layer
	//	hbias: bias of hidden layer
	//	v: visible layer
	//	vbias: bias of visible layer
	//	w: transform matrix between visible and hidden layers  
	//	eRBMType: type of RBM
	void PropagateUp(double* h, const double* hbias, const double* v, Matrix& w);
	void PropagateDown(double* v, const double* vbias, const double* h, Matrix& w, const ERBMType eRBMType); 


public: 
	// model variables
	Matrix m_w;			// transform matrix between visible and hidden layers  
	double* m_vbias;	// bias of visible layer
	double* m_hbias;	// bias of hidden layer

private: 
	RBMLearningParamsT m_paramsLearning;	// learning parameters of RBM
	RBMParamsT m_paramsRBM;					// architecture parameters of RBM

	// training assistant variables
	Matrix m_cw;		// change matrix of transform matrix
	double* m_cvb;		// change array of visible bias	
	double* m_chb;		// change array of hidden bias	
	double* m_v0;
	double* m_v1;
	double* m_v1_s;
	double* m_h0;
	double* m_h0_s;
	double* m_h1;
	double* m_h1_s; 

	int32_t m_nPattCnt; 
};

}

#endif /* _FYDL_RBM_H */



