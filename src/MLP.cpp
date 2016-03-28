#include "MLP.h"
#include "Timer.h"
#include "StringArray.h"
#include "ConfigFile.h"
using namespace fydl; 
#include <algorithm>
#include <fstream>
#include <iostream>
using namespace std; 
#include <math.h>
#include <stdio.h>


////////////////////////////////////////////////////////////////////////////////////////////
// Construction & Destruction  

MLP::MLP()
{
	m_whs = NULL; 

	m_ai = NULL; 
	m_ahs = NULL; 
	m_ao = NULL;

	m_dhs = NULL;
	m_do = NULL;
	m_chs = NULL;	
}


MLP::~MLP()
{
	Release(); 
}


////////////////////////////////////////////////////////////////////////////////////////////
// Operations 

void MLP::Init(const MLPParamsT mlpParamsT, const MLPLearningParamsT mlpLearningParamsT)
{
	Release(); 

	m_paramsMLP = mlpParamsT; 
	m_paramsMLP.input += 1;		// add 1 for bias nodes 
	m_paramsLearning = mlpLearningParamsT; 

	Create(); 
}


bool MLP::InitFromConfig(const char* sConfigFile, const int32_t nInput, const int32_t nOutput)
{
	ConfigFile conf_file; 
	if(!conf_file.Read(sConfigFile)) 
		return false; 
	Release(); 

	m_paramsMLP.input = nInput + 1;		// add 1 for bias nodes
	m_paramsMLP.output = nOutput;
	m_paramsMLP.vtr_hidden.clear(); 
	for(int32_t i = 0; i < conf_file.ValCnt("Hiddens"); i++) 
		m_paramsMLP.vtr_hidden.push_back(conf_file.GetVal_asInt("Hiddens", i)); 
	if(m_paramsMLP.vtr_hidden.empty())
		return false; 
	m_paramsMLP.act_hidden = TypeDefs::ActType(conf_file.GetVal_asString("ActHidden").c_str());   
	m_paramsMLP.act_output = TypeDefs::ActType(conf_file.GetVal_asString("ActOutput").c_str());   
	if(m_paramsMLP.act_hidden == _ACT_NONE || m_paramsMLP.act_output == _ACT_NONE)
		return false; 

	m_paramsLearning.regula = TypeDefs::RegulaType(conf_file.GetVal_asString("Regula").c_str()); 
	m_paramsLearning.mini_batch = conf_file.GetVal_asInt("MiniBatch"); 
	m_paramsLearning.iterations = conf_file.GetVal_asInt("Iterations"); 
	m_paramsLearning.learning_rate = conf_file.GetVal_asFloat("LearningRate"); 
	m_paramsLearning.rate_decay = conf_file.GetVal_asFloat("RateDecay"); 
	m_paramsLearning.epsilon = conf_file.GetVal_asFloat("Epsilon"); 

	Create(); 
	return true; 
}


void MLP::Train(vector<Pattern*>& vtrPatts)
{
	int32_t cross_cnt = (int32_t)vtrPatts.size() / 20;			// 5% patterns for corss validation
	int32_t train_cnt = (int32_t)vtrPatts.size() - cross_cnt;	// 95% patterns for training
	double learning_rate = m_paramsLearning.learning_rate;	// learning rate, it would be update after every iteration
	double error, rmse;	// training error and RMSE in one iteration
	pair<double,double> validation;	// precision and RMSE of validation 
	Timer timer;		// timer

	// create assistant variables for training
	CreateAssistant();

	// shuffle pattens 
	random_shuffle(vtrPatts.begin(), vtrPatts.end());

	for(int32_t t = 0; t < m_paramsLearning.iterations; t++) 
	{
		error = 0.0; 	

		timer.Start(); 	

		// shuffle training patterns
		random_shuffle(vtrPatts.begin(), vtrPatts.end() - cross_cnt);

		for(int32_t p = 0; p < train_cnt; p++) 
		{
			// forward & backward phase
			FeedForward(vtrPatts[p]->m_x, vtrPatts[p]->m_nXCnt); 
			error += BackPropagate(vtrPatts[p]->m_y, vtrPatts[p]->m_nYCnt); 
			m_nPattCnt++; 

			if(m_paramsLearning.mini_batch > 0)	// online or mini-batch
			{
				if(m_nPattCnt >= m_paramsLearning.mini_batch)
					ModelUpdate(learning_rate); 
			}
		}	

		if(m_paramsLearning.mini_batch == 0)	// batch 
			ModelUpdate(learning_rate); 

		validation = Validation(vtrPatts, cross_cnt); 
		rmse = sqrt(error / (double)(train_cnt));
		
		timer.Stop(); 	

		printf("iter %d | learning_rate: %.6g | error: %.6g | rmse: %.6g | validation(pr & rmse): %.4g%% & %.6g | time_cost(s): %.3f\n", 
				t+1, learning_rate, error, rmse, validation.first * 100.0, validation.second, timer.GetLast_asSec()); 
		learning_rate = learning_rate * (learning_rate / (learning_rate + (learning_rate * m_paramsLearning.rate_decay)));	

		if(rmse <= m_paramsLearning.epsilon)
			break;
	}
}


int32_t MLP::Predict(double* y, const int32_t y_len, const double* x, const int32_t x_len)
{
	if(!y || !x)
		return _FYDL_ERROR_INPUT_NULL;
	if(y_len != m_paramsMLP.output || x_len != m_paramsMLP.input - 1)
		return _FYDL_ERROR_WRONG_LEN;
	if(!m_whs || m_wo.IsNull())
		return _FYDL_ERROR_MODEL_NULL;

	int32_t hl = (int32_t)m_paramsMLP.vtr_hidden.size();	// number of hidden layers
	// for thread save, do not use inner layer
	double* ai = new double[m_paramsMLP.input];		// input layer
	double** ahs = new double*[hl];			// hidden layers

	// activate input layer
	for(int32_t i = 0; i < x_len; i++) 
		ai[i] = x[i]; 
	ai[x_len] = 1.0;	// set bias

	// activate hidden layer-by-layer
	for(int32_t h = 0; h < hl; h++) 
	{
		ahs[h] = new double[m_paramsMLP.vtr_hidden[h]]; 
		if(h == 0)	// activate the first hidden layer by input layer
			ActivateForward(ahs[h], m_paramsMLP.vtr_hidden[h], ai, m_paramsMLP.input, m_whs[h], m_paramsMLP.act_hidden); 
		else	// activate the upper layer by lower layer
			ActivateForward(ahs[h], m_paramsMLP.vtr_hidden[h], ahs[h-1], m_paramsMLP.vtr_hidden[h-1], m_whs[h], m_paramsMLP.act_hidden); 
	}
	// activate output 
	ActivateForward(y, y_len, ahs[hl-1], m_paramsMLP.vtr_hidden[hl-1], m_wo, m_paramsMLP.act_output); 

	delete ai; 
	for(int32_t h = 0; h < hl; h++) 
		delete ahs[h];
	delete ahs;

	return _FYDL_SUCCESS; 	
}


int32_t MLP::Save(const char* sFile, const char* sTitle)
{
	if(!m_whs || m_wo.IsNull())
		return _FYDL_ERROR_MODEL_NULL; 
	ofstream ofs(sFile); 
	if(!ofs.is_open())
		return _FYDL_ERROR_FILE_OPEN;
	
	int32_t hl = (int32_t)m_paramsMLP.vtr_hidden.size();	// number of hidden layers

	ofs<<"** "<<sTitle<<" **"<<endl; 
	ofs<<endl;

	// save learning parameters
	ofs<<"@learning_params"<<endl; 
	TypeDefs::Print_PerceptronLearningParamsT(ofs, m_paramsLearning); 
	ofs<<endl; 
	
	// save architecture parameters of RBM
	ofs<<"@architecture_params"<<endl; 
	TypeDefs::Print_MLPParamsT(ofs, m_paramsMLP); 
	ofs<<endl; 

	// save transtorm matrix
	for(int32_t h = 0; h < hl; h++) 
	{
		ofs<<"@weight_hidden_"<<h<<endl; 
		Matrix::Print_Matrix(ofs, m_whs[h]);
		ofs<<endl; 
	}
	ofs<<"@weight_output"<<endl; 
	Matrix::Print_Matrix(ofs, m_wo); 
	ofs<<endl; 

	ofs.close(); 
	return _FYDL_SUCCESS; 
}


int32_t MLP::Load(const char* sFile, const char* sCheckTitle)
{
	Release();

	ifstream ifs(sFile);  
	if(!ifs.is_open())
		return _FYDL_ERROR_FILE_OPEN;
	Release(); 

	string str; 
	int32_t idx; 

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
			if(!TypeDefs::Read_PerceptronLearningParamsT(m_paramsLearning, ifs))
				return _FYDL_ERROR_LERANING_PARAMS;
		}
		else if(str == "@architecture_params")
		{
			if(!TypeDefs::Read_MLPParamsT(m_paramsMLP, ifs))
				return _FYDL_ERROR_ACH_PARAMS;
			Create(); 
		}
		else if(str.find("@weight_hidden_") == 0)
		{
			StringArray ar(str.c_str(), "_"); 
			sscanf(ar.GetString(ar.Count()-1).c_str(), "%d", &idx); 
			if(!Matrix::Read_Matrix(m_whs[idx], ifs))
				return _FYDL_ERROR_MODEL_DATA;
		}
		else if(str == "@weight_output")
		{
			if(!Matrix::Read_Matrix(m_wo, ifs))
				return _FYDL_ERROR_MODEL_DATA;
		}
	}
	
	ifs.close(); 
	return _FYDL_SUCCESS; 
}


MLPLearningParamsT MLP::GetLearningParams()
{
	return m_paramsLearning; 
}


MLPParamsT MLP::GetArchParams()
{
	return m_paramsMLP; 
}


double MLP::FfBp_Step(const double* y, const int32_t y_len, const double* x, const int32_t x_len, const bool bFirst)
{
	if(bFirst)
		CreateAssistant();
	// forward & backward phase
	FeedForward(x, x_len); 
	double error = BackPropagate(y, y_len); 
	m_nPattCnt++; 
	
	return error; 
}


void MLP::ModelUpdate(const double learning_rate)
{
	int32_t hl = (int32_t)m_paramsMLP.vtr_hidden.size();	// number of hidden layers

	// update the transform matrix of output layer (m_wo)
	for(int32_t i = 0; i < m_paramsMLP.vtr_hidden[hl-1]; i++) 
	{
		for(int32_t j = 0; j < m_paramsMLP.output; j++) 
		{
			m_wo[i][j] -= learning_rate * (m_co[i][j] / (double)m_nPattCnt + Activation::DActRegula(m_wo[i][j], m_paramsLearning.regula));
			m_co[i][j] = 0.0; 
		}
	}

	// update the transform matrices of hidden layers (m_whs)
	for(int32_t h = hl - 1; h > 0; h--)
	{
		for(int32_t i = 0; i < m_paramsMLP.vtr_hidden[h-1]; i++)
		{
			for(int32_t j = 0; j < m_paramsMLP.vtr_hidden[h]; j++)
			{
				m_whs[h][i][j] -= learning_rate * (m_chs[h][i][j] / (double)m_nPattCnt + Activation::DActRegula(m_whs[h][i][j], m_paramsLearning.regula));
				m_chs[h][i][j] = 0.0; 
			}
		}
	}
	for(int32_t i = 0; i < m_paramsMLP.input; i++) 
	{
		for(int32_t j = 0; j < m_paramsMLP.vtr_hidden[0]; j++)
		{
			m_whs[0][i][j] -= learning_rate * (m_chs[0][i][j] / (double)m_nPattCnt + Activation::DActRegula(m_whs[0][i][j], m_paramsLearning.regula));
			m_chs[0][i][j] = 0.0; 
		}
	}
	
	m_nPattCnt = 0;
}


////////////////////////////////////////////////////////////////////////////////////////////
// Internal Operations 

void MLP::Create()
{
	int32_t hl = (int32_t)m_paramsMLP.vtr_hidden.size();	// number of hidden layers

	// create transform matrices of hidden layers 
	m_whs = new Matrix[hl];
	for(int32_t h = 0; h < hl; h++)
	{
		if(h == 0)	
			m_whs[h].Create(m_paramsMLP.input, m_paramsMLP.vtr_hidden[h]);
		else	
			m_whs[h].Create(m_paramsMLP.vtr_hidden[h-1], m_paramsMLP.vtr_hidden[h]);
		Activation::InitTransformMatrix(m_whs[h], m_paramsMLP.act_hidden); 
	}	
	
	// create transform matrices of output layer
	m_wo.Create(m_paramsMLP.vtr_hidden[hl-1], m_paramsMLP.output);
	Activation::InitTransformMatrix(m_wo, m_paramsMLP.act_output); 
	
	m_nPattCnt = 0;
}


void MLP::Release()
{
	if(m_whs)
	{
		delete [] m_whs; 
		m_whs = NULL; 
	}
	ReleaseAssistant();
}


void MLP::CreateAssistant()
{
	ReleaseAssistant();
	
	int32_t hl = (int32_t)m_paramsMLP.vtr_hidden.size();	// number of hidden layers
	
	// create change matrices of hidden layers
	m_chs = new Matrix[hl];
	for(int32_t h = 0; h < hl; h++)
	{
		if(h == 0)	
			m_chs[h].Create(m_paramsMLP.input, m_paramsMLP.vtr_hidden[h]);
		else	
			m_chs[h].Create(m_paramsMLP.vtr_hidden[h-1], m_paramsMLP.vtr_hidden[h]);
		m_chs[h].Init(0.0); 
	}
	// create change matrix of output layer
	m_co.Create(m_paramsMLP.vtr_hidden[hl-1], m_paramsMLP.output);
	m_co.Init(0.0);

	// create input layer
	m_ai = new double[m_paramsMLP.input];
	// create hidden layers
	m_ahs = new double*[hl];
	for(int32_t h = 0; h < hl; h++)
		m_ahs[h] = new double[m_paramsMLP.vtr_hidden[h]];
	// create output layer
	m_ao = new double[m_paramsMLP.output];

	// create delta arrays of hidden layers
	m_dhs = new double*[hl]; 
	for(int32_t h = 0; h < hl; h++)
		m_dhs[h] = new double[m_paramsMLP.vtr_hidden[h]];
	// create delta array of output layers
	m_do = new double[m_paramsMLP.output];
	
	m_nPattCnt = 0; 
}


void MLP::ReleaseAssistant()
{
	int32_t hl = (int32_t)m_paramsMLP.vtr_hidden.size();	// number of hidden layers

	if(m_ai)
	{
		delete m_ai; 
		m_ai = NULL; 
	}

	if(m_ahs)
	{
		for(int32_t h = 0; h < hl; h++) 
			delete m_ahs[h]; 
		delete m_ahs; 
		m_ahs = NULL; 
	}
	if(m_ao)
	{
		delete m_ao; 
		m_ao = NULL; 
	}

	if(m_dhs)
	{
		for(int32_t h = 0; h < hl; h++) 
			delete m_dhs[h]; 
		delete m_dhs; 
		m_dhs = NULL; 
	}
	if(m_do)
	{
		delete m_do; 
		m_do = NULL; 
	}

	if(m_chs)
	{
		delete [] m_chs; 
		m_chs = NULL; 
	}
}


void MLP::FeedForward(const double* in_vals, const int32_t in_len)
{
	if(!in_vals || in_len != m_paramsMLP.input - 1)
		throw "MLP::FeedForward() ERROR: Wrong length of \'in_vals\'!"; 

	// activate input layer
	for(int32_t i = 0; i < in_len; i++) 
		m_ai[i] = in_vals[i];
	m_ai[in_len] = 1.0;		// set bias

	// activate hidden layer-by-layer
	int32_t hl = (int32_t)m_paramsMLP.vtr_hidden.size();	// number of hidden layers
	for(int32_t h = 0; h < hl; h++) 
	{
		if(h == 0)	// activate the first hidden layer by input layer
			ActivateForward(m_ahs[h], m_paramsMLP.vtr_hidden[h], m_ai, m_paramsMLP.input, m_whs[h], m_paramsMLP.act_hidden); 
		else	// activate the upper layer by lower layer
			ActivateForward(m_ahs[h], m_paramsMLP.vtr_hidden[h], m_ahs[h-1], m_paramsMLP.vtr_hidden[h-1], m_whs[h], m_paramsMLP.act_hidden); 
	}

	// activate output layer
	ActivateForward(m_ao, m_paramsMLP.output, m_ahs[hl-1], m_paramsMLP.vtr_hidden[hl-1], m_wo, m_paramsMLP.act_output); 
}


double MLP::BackPropagate(const double* out_vals, const int32_t out_len)
{
	if(!out_vals || out_len != m_paramsMLP.output)
		throw "MLP::BackPropagate() ERROR: Wrong length of \'out_vals\'!"; 

	double error = 0.0; 
	int32_t hl = (int32_t)m_paramsMLP.vtr_hidden.size();	// number of hidden layers

	// caculate delta and error of output layer
	for(int32_t j = 0; j < m_paramsMLP.output; j++) 
	{
		m_do[j] = m_ao[j] - out_vals[j]; 
		error += m_do[j] * m_do[j];  
	}

	// output layer back propagate
	for(int32_t i = 0; i < m_paramsMLP.vtr_hidden[hl-1]; i++)
	{
		m_dhs[hl-1][i] = 0.0;   
		for(int32_t j = 0; j < m_paramsMLP.output; j++)
			m_dhs[hl-1][i] += m_do[j] * m_wo[i][j]; 
	}
	for(int32_t h = hl - 2; h >= 0; h--)
	{
		for(int32_t i = 0; i < m_paramsMLP.vtr_hidden[h]; i++) 
		{
			m_dhs[h][i] = 0.0;
			for(int32_t j = 0; j < m_paramsMLP.vtr_hidden[h+1]; j++) 
				m_dhs[h][i] += m_dhs[h+1][j] * m_whs[h+1][i][j];  
		}	
	}

	// update change matrices
	for(int32_t j = 0; j < m_paramsMLP.output; j++)
	{ // m_co
		for(int32_t i = 0; i < m_paramsMLP.vtr_hidden[hl-1]; i++)
			m_co[i][j] += m_do[j] * Activation::DActivate(m_ao[j], m_paramsMLP.act_output) * m_ahs[hl-1][i];  
	}
	for(int32_t h = hl - 1; h > 0; h--)
	{ // m_whs
		for(int32_t j = 0; j < m_paramsMLP.vtr_hidden[h]; j++) 
		{
			for(int32_t i = 0; i < m_paramsMLP.vtr_hidden[h-1]; i++) 
				m_chs[h][i][j] += m_dhs[h][j] * Activation::DActivate(m_ahs[h][j], m_paramsMLP.act_hidden) * m_ahs[h-1][i]; 	
		}
	}
	for(int32_t j = 0; j < m_paramsMLP.vtr_hidden[0]; j++) 
	{ // m_whs[0]
		for(int32_t i = 0; i < m_paramsMLP.input; i++) 
			m_chs[0][i][j] += m_dhs[0][j] * Activation::DActivate(m_ahs[0][j], m_paramsMLP.act_hidden) * m_ai[i]; 
	}

	return error / double(m_paramsMLP.output); 
}


void MLP::ActivateForward(double* up_a, const int32_t up_size, const double* low_a, const int32_t low_size, 
		Matrix& w, const EActType up_act_type)
{
	double sum;
	double e = 0.0; 

	for(int32_t j = 0; j < up_size; j++) 
	{
		sum = 0.0; 
		for(int32_t i = 0; i < low_size; i++) 
			sum += low_a[i] * w[i][j]; 
		up_a[j] = Activation::Activate(sum, up_act_type);  

		if(up_act_type == _ACT_SOFTMAX)
			e += up_a[j]; 
	}

	// for softmax
	if(up_act_type == _ACT_SOFTMAX)
	{
		for(int32_t j = 0; j < up_size; j++) 
			up_a[j] /= e; 
	}
}


pair<double, double> MLP::Validation(vector<Pattern*>& vtrPatts, const int32_t nBackCnt)
{
	double error = 0.0; 
	int32_t correct = 0, total = 0; 
	int32_t patts = (int32_t)vtrPatts.size(); 
	double* y = new double[m_paramsMLP.output];
	int32_t k; 

	for(int32_t i = 0; i < nBackCnt && i < patts; i++) 
	{
		k = patts-1-i;
		Predict(y, m_paramsMLP.output, vtrPatts[k]->m_x, vtrPatts[k]->m_nXCnt); 
		if(Pattern::MaxOff(y, m_paramsMLP.output) == Pattern::MaxOff(vtrPatts[k]->m_y, vtrPatts[k]->m_nYCnt))
			correct += 1; 	
		error += Pattern::Error(y, vtrPatts[k]->m_y, m_paramsMLP.output);  	
		total += 1; 	
	}

	delete y;

	double pr = (double)correct / (double)total;
	double rmse = sqrt(error / (double)total);

	return pair<double,double>(pr, rmse); 
}



