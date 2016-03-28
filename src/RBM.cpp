#include "RBM.h"
#include "Activation.h"
#include "Utility.h"
#include "Timer.h"
#include "StringArray.h"
#include "ConfigFile.h"
using namespace fydl; 
#include <algorithm>
#include <fstream>
#include <iostream>
using namespace std; 
#include <math.h>


////////////////////////////////////////////////////////////////////////////////////////////
// Construction & Destruction  

RBM::RBM()
{
	m_vbias = NULL; 
	m_hbias = NULL; 

	m_cvb = NULL; 
	m_chb = NULL; 
	m_v0 = NULL;
	m_v1 = NULL;
	m_v1_s = NULL;
	m_h0 = NULL;
	m_h0_s = NULL;
	m_h1 = NULL;
	m_h1_s = NULL;
}


RBM::~RBM()
{
	Release(); 
}


////////////////////////////////////////////////////////////////////////////////////////////
// Operations 

void RBM::Init(const RBMParamsT rbmParamsT, const RBMLearningParamsT rbmLearningParamsT)
{
	Release(); 
	m_paramsRBM = rbmParamsT; 
	m_paramsLearning = rbmLearningParamsT; 
	Create(); 
}


bool RBM::InitFromConfig(const char* sConfigFile, const int32_t nVisible)
{
	ConfigFile conf_file; 
	if(!conf_file.Read(sConfigFile)) 
		return false; 
	Release(); 
	
	m_paramsRBM.rbm_type = TypeDefs::RBMType(conf_file.GetVal_asString("RBMType").c_str()); 
	if(m_paramsRBM.rbm_type == _UNKNOWN_RBM) 
		return false; 
	m_paramsRBM.visible = nVisible;
	m_paramsRBM.hidden = conf_file.GetVal_asInt("Hidden"); 

	m_paramsLearning.gibbs_steps = conf_file.GetVal_asInt("GibbsSteps"); 
	m_paramsLearning.mini_batch = conf_file.GetVal_asInt("MiniBatch"); 
	m_paramsLearning.iterations = conf_file.GetVal_asInt("Iterations"); 
	m_paramsLearning.learning_rate = conf_file.GetVal_asFloat("LearningRate"); 
	m_paramsLearning.rate_decay = conf_file.GetVal_asFloat("RateDecay"); 
	m_paramsLearning.epsilon = conf_file.GetVal_asFloat("Epsilon"); 
	
	Create(); 

	return true; 
}


void RBM::Train(vector<Pattern*>& vtrPatts)
{
	double learning_rate = m_paramsLearning.learning_rate;	// learning rate, it would be update after every iteration
	double error, rmse; 
	Timer timer;		// timer

	// create assistant variables
	CreateAssistant(); 

	for(int32_t t = 0; t < m_paramsLearning.iterations; t++)
	{
		error = 0.0;

		timer.Start(); 
		// shuffle pattens 
		random_shuffle(vtrPatts.begin(), vtrPatts.end());
		// 
		for(int32_t p = 0; p < (int32_t)vtrPatts.size(); p++) 
		{
			// CD-k	
			error += ContrastiveDivergrence(vtrPatts[p]->m_x, vtrPatts[p]->m_nXCnt); 
			m_nPattCnt++; 

			if(m_paramsLearning.mini_batch > 0)	// online or mini-batch
			{
				if(m_nPattCnt >= m_paramsLearning.mini_batch)
					ModelUpdate(learning_rate); 
			}		
		}
		if(m_paramsLearning.mini_batch == 0)	// batch 
			ModelUpdate(learning_rate); 
		
		rmse = sqrt(error / (double)(vtrPatts.size()));
		timer.Stop(); 	
	
		printf("iter %d | learning_rate: %.6g | error: %.6g | rmse: %.6g | time_cost(s): %.3f\n", 
				t+1, learning_rate, error, rmse, timer.GetLast_asSec()); 
		learning_rate = learning_rate * (learning_rate / (learning_rate + (learning_rate * m_paramsLearning.rate_decay)));	

		if(rmse <= m_paramsLearning.epsilon)
			break;
	}
}


double RBM::Reconstruct(double* xr, const double* x, const int32_t len)
{
	if(!xr || !x || len != m_paramsRBM.visible) 
		throw "RBM::Reconstruct() ERROR: Incorrect input parameters!"; 
	if(m_w.IsNull() || !m_vbias || !m_hbias)
		throw "RBM::Reconstruct() ERROR: Model is NULL"; 

	double* h = new double[m_paramsRBM.hidden];  
	double* hs = new double[m_paramsRBM.hidden];  
	double* v = new double[m_paramsRBM.visible]; 	

	Sample_HbyV(h, hs, m_hbias, x, m_w); 
	Sample_VbyH(v, xr, m_vbias, hs, m_w, m_paramsRBM.rbm_type);
	
	delete h; 
	delete hs; 
	delete v; 

	return Pattern::Error(x, xr, len); 
}


int32_t RBM::PropagateForward(double* y, const int32_t y_len, const double* x, const int32_t x_len)
{
	if(!y || !x)
		return _FYDL_ERROR_INPUT_NULL;
	if(y_len != m_paramsRBM.hidden || x_len != m_paramsRBM.visible) 
		return _FYDL_ERROR_WRONG_LEN;
	if(m_w.IsNull() || !m_vbias || !m_hbias)
		return _FYDL_ERROR_MODEL_NULL;
	
	PropagateUp(y, m_hbias, x, m_w); 
	
	return _FYDL_SUCCESS;
}


int32_t RBM::Save(const char* sFile, const char* sTitle)
{
	if(m_w.IsNull() || !m_vbias || !m_hbias)
		return _FYDL_ERROR_MODEL_NULL; 
	ofstream ofs(sFile); 
	if(!ofs.is_open())
		return _FYDL_ERROR_FILE_OPEN;

	ofs<<"** "<<sTitle<<" **"<<endl; 
	ofs<<endl;

	// save learning parameters
	ofs<<"@learning_params"<<endl; 
	TypeDefs::Print_RBMLearningParamsT(ofs, m_paramsLearning); 
	ofs<<endl; 
	
	// save architecture parameters of RBM
	ofs<<"@architecture_params"<<endl; 
	TypeDefs::Print_RBMParamsT(ofs, m_paramsRBM); 
	ofs<<endl; 
	
	// save transtorm matrix
	ofs<<"@weight"<<endl; 
	Matrix::Print_Matrix(ofs, m_w);
	ofs<<endl; 

	// save visible bias 
	ofs<<"@visible_bias"<<endl; 
	Pattern::Print_Array(ofs, m_vbias, m_paramsRBM.visible);
	ofs<<endl; 
	
	// save hidden bias 
	ofs<<"@hidden_bias"<<endl; 
	Pattern::Print_Array(ofs, m_hbias, m_paramsRBM.hidden);
	ofs<<endl; 

	ofs.close(); 
	return _FYDL_SUCCESS; 
}


int32_t RBM::Load(const char* sFile, const char* sCheckTitle)
{
	Release();

	ifstream ifs(sFile);  
	if(!ifs.is_open())
		return _FYDL_ERROR_FILE_OPEN;
	Release(); 

	string str; 

	if(sCheckTitle)
	{
		std::getline(ifs, str); 
		if(str.find(sCheckTitle) == string::npos) 
		{
			ifs.close();
			return _FYDL_ERROR_NOT_MODEL_FILE;
		}
	}
	
	while(!ifs.eof())
	{
		std::getline(ifs, str);
		if(str.empty())
			continue; 
		else if(str == "@learning_params")
		{
			if(!TypeDefs::Read_RBMLearningParamsT(m_paramsLearning, ifs))
				return _FYDL_ERROR_LERANING_PARAMS;
		}
		else if(str == "@architecture_params")
		{
			if(!TypeDefs::Read_RBMParamsT(m_paramsRBM, ifs))
				return _FYDL_ERROR_ACH_PARAMS;
			Create(); 
		}
		else if(str == "@weight")
		{
			if(!Matrix::Read_Matrix(m_w, ifs))
				return _FYDL_ERROR_MODEL_DATA;
		}
		else if(str == "@visible_bias")
		{
			if(!Pattern::Read_Array(m_vbias, m_paramsRBM.visible, ifs))
				return _FYDL_ERROR_MODEL_DATA;
		}
		else if(str == "@hidden_bias")
		{
			if(!Pattern::Read_Array(m_hbias, m_paramsRBM.hidden, ifs))
				return _FYDL_ERROR_MODEL_DATA;
		}
	}

	ifs.close(); 
	return _FYDL_SUCCESS; 
}


RBMLearningParamsT RBM::GetLearningParams()
{
	return m_paramsLearning; 
}


RBMParamsT RBM::GetArchParams() 
{
	return m_paramsRBM; 
}


double RBM::CDk_Step(const double* x, const int32_t x_len, const bool bFirst)
{
	if(bFirst)
		CreateAssistant(); 
	// CD-k
	double error = ContrastiveDivergrence(x, x_len); 
	m_nPattCnt++; 
	
	return error; 
}


void RBM::ModelUpdate(const double dLearningRate)
{
	// update the transform matrix
	for(int32_t i = 0; i < m_paramsRBM.visible; i++) 
	{
		for(int32_t j = 0; j < m_paramsRBM.hidden; j++) 
		{
			m_w[i][j] += dLearningRate * m_cw[i][j] / (double)m_nPattCnt; 
			m_cw[i][j] = 0.0; 
		}
	}
	// update visible bias
	for(int32_t i = 0; i < m_paramsRBM.visible; i++) 
	{
		m_vbias[i] += dLearningRate * m_cvb[i] / (double)m_nPattCnt; 
		m_cvb[i] = 0.0; 
	}
	// update hidden bias
	for(int32_t j = 0; j < m_paramsRBM.hidden; j++) 
	{
		m_hbias[j] += dLearningRate * m_chb[j] / (double)m_nPattCnt;  
		m_chb[j] = 0.0; 
	}
	
	m_nPattCnt = 0;
}


////////////////////////////////////////////////////////////////////////////////////////////
// Internal Operations 

void RBM::Create()
{
	// create transform matrix 
	m_w.Create(m_paramsRBM.visible, m_paramsRBM.hidden); 
	Activation::InitTransformMatrix(m_w, _ACT_SIGMOID); 

	// create the bias array of visible layer
	m_vbias = new double[m_paramsRBM.visible];
	for(int32_t i = 0; i < m_paramsRBM.visible; i++) 
		m_vbias[i] = 0.0; 

	// create the bias array of hidden layer
	m_hbias = new double[m_paramsRBM.hidden];
	for(int32_t j = 0; j < m_paramsRBM.hidden; j++) 
		m_hbias[j] = 0.0; 
	
	m_nPattCnt = 0; 
}


void RBM::Release()
{
	if(m_vbias)
	{
		delete m_vbias;
		m_vbias = NULL; 
	}
	if(m_hbias)
	{
		delete m_hbias;
		m_hbias = NULL; 
	}
	ReleaseAssistant();
}


void RBM::CreateAssistant()
{
	ReleaseAssistant();
	
	m_cw.Create(m_paramsRBM.visible, m_paramsRBM.hidden); 
	m_cw.Init(0.0); 
	m_cvb = new double[m_paramsRBM.visible];
	for(int32_t i = 0; i < m_paramsRBM.visible; i++)
		m_cvb[i] = 0.0; 
	m_chb = new double[m_paramsRBM.hidden]; 
	for(int32_t j = 0; j < m_paramsRBM.hidden; j++)
		m_chb[j] = 0.0; 

	m_v0 = new double[m_paramsRBM.visible];
	m_v1 = new double[m_paramsRBM.visible];
	m_v1_s = new double[m_paramsRBM.visible];
	m_h0 = new double[m_paramsRBM.hidden];
	m_h0_s = new double[m_paramsRBM.hidden];
	m_h1 = new double[m_paramsRBM.hidden];
	m_h1_s = new double[m_paramsRBM.hidden];
	
	m_nPattCnt = 0; 
}


void RBM::ReleaseAssistant()
{
	if(m_cvb)
	{
		delete m_cvb; 
		m_cvb = NULL; 
	}
	if(m_chb)
	{
		delete m_chb; 
		m_chb = NULL; 
	}
	if(m_v0)
	{
		delete m_v0; 
		m_v0 = NULL; 
	}
	if(m_v1)
	{
		delete m_v1; 
		m_v1 = NULL; 
	}
	if(m_v1_s)
	{
		delete m_v1_s; 
		m_v1_s = NULL; 
	}
	if(m_h0)
	{
		delete m_h0; 
		m_h0 = NULL; 
	}
	if(m_h0_s)
	{
		delete m_h0_s; 
		m_h0_s = NULL; 
	}
	if(m_h1)
	{
		delete m_h1; 
		m_h1 = NULL; 
	}
	if(m_h1_s)
	{
		delete m_h1_s; 
		m_h1_s = NULL; 
	}
}


double RBM::ContrastiveDivergrence(const double* in_vals, const int32_t in_len) 
{
	if(in_len != m_paramsRBM.visible)
		throw "RBM::ContrastiveDivergrence() ERROR: Wrong length of \'in_vals\'";

	// activte the visible layer
	for(int32_t i = 0; i < in_len; i++) 
		m_v0[i] = in_vals[i]; 

	// the first sampling of hidden neurons from visible 
	Sample_HbyV(m_h0, m_h0_s, m_hbias, m_v0, m_w); 

	// gibbs sampling 
	for(int32_t k = 0; k < m_paramsLearning.gibbs_steps; k++) 
	{
		if(k == 0)
			Sample_VbyH(m_v1, m_v1_s, m_vbias, m_h0_s, m_w, m_paramsRBM.rbm_type); 
		else
			Sample_VbyH(m_v1, m_v1_s, m_vbias, m_h1_s, m_w, m_paramsRBM.rbm_type); 
		Sample_HbyV(m_h1, m_h1_s, m_hbias, m_v1_s, m_w); 
	}

	// update the change of transform matrix
	for(int32_t i = 0; i < m_paramsRBM.visible; i++) 
	{
		for(int32_t j = 0; j < m_paramsRBM.hidden; j++) 
			m_cw[i][j] += m_v0[i] * m_h0[j] - m_v1_s[i] * m_h1[j]; 
	}
	// update the change of visible bias
	for(int32_t i = 0; i < m_paramsRBM.visible; i++) 
		m_cvb[i] = m_v0[i] - m_v1_s[i];	
	// update the change of hidden bias
	for(int32_t j = 0; j < m_paramsRBM.hidden; j++) 
		m_chb[j] = m_h0[j] - m_h1[j];	

	return Pattern::Error(in_vals, m_v1_s, in_len);
}


void RBM::Sample_HbyV(double* h, double* hs, const double* hbias, const double* vs, Matrix& w)
{
	PropagateUp(h, hbias, vs, w); 
	for(int32_t j = 0; j < w.Cols(); j++) 
		hs[j] = (double)Utility::RandBinomial(1, h[j]); 
}


void RBM::Sample_VbyH(double* v, double* vs, const double* vbias, const double* hs, Matrix& w, const ERBMType eRBMType)
{
	PropagateDown(v, vbias, hs, w, eRBMType);
	for(int32_t i = 0; i < w.Rows(); i++) 
	{
		if(eRBMType == _GAUSS_BERNOULLI_RBM)	
			vs[i] = Utility::RandNormal(v[i], 1.0); 
		else if(eRBMType == _BINOMIAL_BERNOULLI_RBM)
			vs[i] = (double)Utility::RandBinomial(1, v[i]); 
	}
}


void RBM::PropagateUp(double* h, const double* hbias, const double* v, Matrix& w)
{
	double sum; 
	for(int32_t j = w.Cols(); j < w.Cols(); j++) 
	{
		sum = hbias[j]; 
		for(int32_t i = 0; i < w.Rows(); i++) 
			sum += w[i][j] * v[i]; 
		h[j] = Activation::Sigmoid(sum); 
	}
}


void RBM::PropagateDown(double* v, const double* vbias, const double* h, Matrix& w, const ERBMType eRBMType) 
{
	double sum; 
	for(int32_t i = 0; i < w.Rows(); i++) 
	{
		sum = vbias[i]; 
		for(int32_t j = 0; j < w.Cols(); j++) 
			sum += w[i][j] * h[j]; 
	
		if(eRBMType == _GAUSS_BERNOULLI_RBM)	
			v[i] = sum; 
		else if(eRBMType == _BINOMIAL_BERNOULLI_RBM)
			v[i] = Activation::Sigmoid(sum); 
	}
}


