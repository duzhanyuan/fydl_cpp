#include "MLP_NeuralNetwork.h"
#include "Timer.h"
#include "StringArray.h"
using namespace fydl; 
#include <algorithm>
#include <fstream>
#include <iostream>
using namespace std; 
#include <math.h>
#include <stdio.h>


////////////////////////////////////////////////////////////////////////////////////////////
// Construction & Destruction  

MLP_NeuralNetwork::MLP_NeuralNetwork()
{
	m_paramsLearning.regula = _REGULA_L1;
	m_paramsLearning.mini_batch = 0; 
	m_paramsLearning.iterations = 50;
	m_paramsLearning.learning_rate = 0.5;
	m_paramsLearning.rate_decay = 0.01;
	m_paramsLearning.epsilon = 0.01;
	m_nIters = 0; 

	m_paramsNN.act_hidden = _ACT_TANH;
	m_paramsNN.act_output = _ACT_SIGMOID;

	m_whs = NULL; 

	m_ai = NULL; 
	m_ahs = NULL; 
	m_ao = NULL;

	m_dhs = NULL;
	m_do = NULL;
	m_chs = NULL;
}


MLP_NeuralNetwork::~MLP_NeuralNetwork()
{
	Release(); 
}


////////////////////////////////////////////////////////////////////////////////////////////
// Operations 

void MLP_NeuralNetwork::Init(const int32_t nInput, vector<int32_t>& vtrHidden, const int32_t nOutput, 
		const EActType eActHidden, const EActType eActOutput, const LearningParamsT* pLearningParamsT)
{
	Release(); 

	m_paramsNN.input = nInput + 1;	// add 1 for bias nodes 
	m_paramsNN.vtr_hidden = vtrHidden; 
	m_paramsNN.output = nOutput; 
	m_paramsNN.act_hidden = eActHidden;	
	m_paramsNN.act_output = eActOutput;
	if(pLearningParamsT)
		m_paramsLearning = *pLearningParamsT; 

	Create(); 
}


bool MLP_NeuralNetwork::InitFromConfig(const char* sConfigFile, const int32_t nInput, const int32_t nOutput)
{
	if(!sConfigFile)
		return false; 
	ifstream ifs(sConfigFile);
	if(!ifs.is_open())
		return false; 

	Release(); 

	m_paramsNN.input = nInput + 1;		// add 1 for bias nodes
	m_paramsNN.output = nOutput;

	int32_t hidden; 
	string str; 
	while(!ifs.eof())
	{
		std::getline(ifs, str); 
		if(str.empty())
			continue; 
		if(str.at(0) == '#')
			continue; 

		StringArray ar(str.c_str(), ":"); 
		if(ar.Count() != 2)
			continue; 

		if(ar.GetString(0) == "Regula")
			m_paramsLearning.regula = TypeDefs::RegulaType(ar.GetString(1).c_str()); 
		else if(ar.GetString(0) == "MiniBatch")
			sscanf(ar.GetString(1).c_str(), "%d", &m_paramsLearning.mini_batch);		
		else if(ar.GetString(0) == "Iterations")
			sscanf(ar.GetString(1).c_str(), "%d", &m_paramsLearning.iterations);		
		else if(ar.GetString(0) == "LearningRate")
			sscanf(ar.GetString(1).c_str(), "%lf", &m_paramsLearning.learning_rate);		
		else if(ar.GetString(0) == "RateDecay")
			sscanf(ar.GetString(1).c_str(), "%lf", &m_paramsLearning.rate_decay);		
		else if(ar.GetString(0) == "Epsilon")
			sscanf(ar.GetString(1).c_str(), "%lf", &m_paramsLearning.epsilon);		
		else if(ar.GetString(0) == "Hiddens")
		{
			StringArray array(ar.GetString(1).c_str(), ","); 
			for(int32_t i = 0; i < (int32_t)array.Count(); i++) 
			{
				sscanf(array.GetString(i).c_str(), "%d", &hidden); 
				m_paramsNN.vtr_hidden.push_back(hidden); 
			}
		}
		else if(ar.GetString(0) == "HiddenActivation")
			m_paramsNN.act_hidden = TypeDefs::ActType(ar.GetString(1).c_str()); 
		else if(ar.GetString(0) == "OutputActivation")
			m_paramsNN.act_output = TypeDefs::ActType(ar.GetString(1).c_str()); 
	}
	Create(); 

	ifs.close();
	return true; 
}


void MLP_NeuralNetwork::Train(vector<Pattern*>& vtrPatts)
{
	int32_t patt_cnt = (int32_t)vtrPatts.size(); 	// number of patterns
	int32_t cross_cnt = patt_cnt / 20;			// 5% patterns for corss validation
	int32_t train_cnt = patt_cnt - cross_cnt;	// 95% patterns for training
	double learning_rate = m_paramsLearning.learning_rate;	// learning rate, it would be update after every iteration
	double error, rmse;	// training error and RMSE in one iteration
	pair<double,double> validation;	// precision and RMSE of validation 
	Timer timer;		// timer

	// shuffle pattens 
	random_shuffle(vtrPatts.begin(), vtrPatts.end());

	m_nIters = 0; 
	while(m_nIters < m_paramsLearning.iterations)
	{
		m_nIters++; 
		printf("iter %d ", m_nIters);
		error = 0; 	

		timer.Start(); 	

		// shuffle training patterns
		random_shuffle(vtrPatts.begin(), vtrPatts.end() - cross_cnt);

		for(int32_t p = 0; p < train_cnt; p++) 
		{
			// forward & backward phase
			FeedForward(vtrPatts[p]->m_x, vtrPatts[p]->m_nXCnt); 
			error += BackPropagate(vtrPatts[p]->m_y, vtrPatts[p]->m_nYCnt); 

			if(m_paramsLearning.mini_batch > 0)	// online or mini-batch
			{
				if((p+1) % m_paramsLearning.mini_batch == 0)
					UpdateWeightMats(learning_rate);
			}
		}	

		if(m_paramsLearning.mini_batch == 0)	// batch 
			UpdateWeightMats(learning_rate);

		validation = Validation(vtrPatts, cross_cnt); 
		rmse = sqrt(error / (double)(train_cnt));
		
		timer.Stop(); 	

		printf("| learning_rate: %.6g | error: %.6g | rmse: %.6g | validation(pr & rmse): %.4g%% & %.6g | time_cost(s): %.3f\n", 
				learning_rate, error, rmse, validation.first * 100.0, validation.second, timer.GetLast_asSec()); 
		learning_rate = learning_rate * (learning_rate / (learning_rate + (learning_rate * m_paramsLearning.rate_decay)));	

		if(rmse <= m_paramsLearning.epsilon)
			break;
	}
}


int32_t MLP_NeuralNetwork::Predict(double* y, const int32_t y_len, const double* x, const int32_t x_len)
{
	if(!y || !x)
		return _MLP_ERROR_INPUT_NULL;
	if(y_len != m_paramsNN.output || x_len != m_paramsNN.input - 1)
		return _MLP_ERROR_WRONG_LEN;
	if(!m_whs || m_wo.IsNull())
		return _MLP_ERROR_MODEL_NULL;

	int32_t hl = (int32_t)m_paramsNN.vtr_hidden.size();	// number of hidden layers
	// for thread save, do not use inner layer
	double* ai = new double[m_paramsNN.input];		// input layer
	double** ahs = new double*[hl];			// hidden layers

	// activate input layer
	for(int32_t i = 0; i < x_len; i++) 
		ai[i] = x[i]; 
	ai[x_len] = 1.0;	// set bias

	// activate hidden layer-by-layer
	for(int32_t h = 0; h < hl; h++) 
	{
		ahs[h] = new double[m_paramsNN.vtr_hidden[h]]; 
		if(h == 0)	// activate the first hidden layer by input layer
			ActivateForward(ahs[h], m_paramsNN.vtr_hidden[h], ai, m_paramsNN.input, m_whs[h], m_paramsNN.act_hidden); 
		else	// activate the upper layer by lower layer
			ActivateForward(ahs[h], m_paramsNN.vtr_hidden[h], ahs[h-1], m_paramsNN.vtr_hidden[h-1], m_whs[h], m_paramsNN.act_hidden); 
	}
	// activate output 
	ActivateForward(y, y_len, ahs[hl-1], m_paramsNN.vtr_hidden[hl-1], m_wo, m_paramsNN.act_output); 

	delete ai; 
	for(int32_t h = 0; h < hl; h++) 
		delete ahs[h];
	delete ahs;

	return _MLP_SUCCESS; 	
}


int32_t MLP_NeuralNetwork::Save(const char* sFile, const char* sTitle)
{
	if(!m_whs || m_wo.IsNull())
		return _MLP_ERROR_MODEL_NULL; 
	FILE* fp = fopen(sFile, "w"); 
	if(!fp)
		return _MLP_ERROR_FILE_OPEN;

	int32_t hl = (int32_t)m_paramsNN.vtr_hidden.size();	// number of hidden layers

	fprintf(fp, "** %s **\n", sTitle);
	fprintf(fp, "\n"); 
	fprintf(fp, "regula:%s\n", TypeDefs::RegulaName(m_paramsLearning.regula).c_str()); 
	fprintf(fp, "mini_batch:%d\n", m_paramsLearning.mini_batch); 
	fprintf(fp, "max_iters:%d\n", m_paramsLearning.iterations); 
	fprintf(fp, "real_iters:%d\n", m_nIters); 
	fprintf(fp, "learning_rate:%.6g\n", m_paramsLearning.learning_rate); 
	fprintf(fp, "rate_decay:%.6g\n", m_paramsLearning.rate_decay); 
	fprintf(fp, "epsilon:%.6g\n", m_paramsLearning.epsilon); 
	fprintf(fp, "\n"); 
	fprintf(fp, "layers:%d\n", hl + 2); 
	fprintf(fp, "input:%d\n", m_paramsNN.input-1); 
	for(int32_t h = 0; h < hl; h++) 
	{
		if(h == 0)
			fprintf(fp, "hidden:%d", m_paramsNN.vtr_hidden[h]);
		else
			fprintf(fp, ",%u", m_paramsNN.vtr_hidden[h]); 
	}
	fprintf(fp, "\n"); 
	fprintf(fp, "output:%d\n", m_paramsNN.output); 
	fprintf(fp, "hidden_activation:%s\n", TypeDefs::ActName(m_paramsNN.act_hidden).c_str()); 
	fprintf(fp, "output_activation:%s\n\n", TypeDefs::ActName(m_paramsNN.act_output).c_str()); 
	fprintf(fp, "\n"); 

	for(int32_t h = 0; h < hl; h++) 
	{
		fprintf(fp, "@weight_hidden_%u\n", h); 
		//m_whs[h].Sparsification(); 
		for(int32_t i = 0; i < (int32_t)m_whs[h].Rows(); i++) 
		{
			for(int32_t j = 0; j < (int32_t)m_whs[h].Cols(); j++) 
			{
				if(j == 0)
					fprintf(fp, "%.12g", m_whs[h][i][j]); 
				else
					fprintf(fp, ",%.12g", m_whs[h][i][j]); 
			}
			fprintf(fp, "\n"); 
		}
		fprintf(fp, "\n"); 
	}

	fprintf(fp, "@weight_output\n"); 
	//m_wo.Sparsification(); 
	for(int32_t i = 0; i < (int32_t)m_wo.Rows(); i++)
	{
		for(int32_t j = 0; j < (int32_t)m_wo.Cols(); j++)
		{
			if(j == 0)
				fprintf(fp, "%.12g", m_wo[i][j]); 
			else
				fprintf(fp, ",%.12g", m_wo[i][j]); 
		}
		fprintf(fp, "\n"); 
	}
	fprintf(fp, "\n"); 

	fclose(fp);
	return _MLP_SUCCESS; 
}


int32_t MLP_NeuralNetwork::Load(const char* sFile, const char* sCheckTitle)
{
	ifstream ifs(sFile);  
	if(!ifs.is_open())
		return _MLP_ERROR_FILE_OPEN;
	Release(); 

	string str; 
	int32_t step = 0, wo_off = 0, layers, cur_h; 
	vector<int32_t> vtr_hoffs; 
	bool create_flag = false; 

	if(sCheckTitle)
	{
		std::getline(ifs, str); 
		if(str.find(sCheckTitle) == string::npos) 
		{
			ifs.close();
			return _MLP_ERROR_NOT_MODEL_FILE;
		}
	}

	while(!ifs.eof())
	{
		std::getline(ifs, str);
		if(str.empty())
			continue; 
		else if(str.find("@weight_hidden") == 0)
		{
			step = 1; 
			StringArray ar(str.c_str(), "_"); 
			sscanf(ar.GetString(ar.Count()-1).c_str(), "%u", &cur_h); 
			continue; 	
		}
		else if(str == "@weight_output")
		{
			step = 2; 
			continue; 	
		}

		if(step == 0)
		{
			StringArray array(str.c_str(), ":");
			if(array.Count() != 2)
				continue; 
			if(array.GetString(0) == "regula")
				m_paramsLearning.regula = TypeDefs::RegulaType(array.GetString(1).c_str()); 
			if(array.GetString(0) == "mini_batch")
				sscanf(array.GetString(1).c_str(), "%d", &m_paramsLearning.mini_batch); 
			if(array.GetString(0) == "max_iters")
				sscanf(array.GetString(1).c_str(), "%d", &m_paramsLearning.iterations); 
			if(array.GetString(0) == "real_iters")
				sscanf(array.GetString(1).c_str(), "%d", &m_nIters); 
			if(array.GetString(0) == "learning_rate")
				sscanf(array.GetString(1).c_str(), "%lf", &m_paramsLearning.learning_rate); 
			if(array.GetString(0) == "rate_decay")
				sscanf(array.GetString(1).c_str(), "%lf", &m_paramsLearning.rate_decay); 
			if(array.GetString(0) == "epsilon")
				sscanf(array.GetString(1).c_str(), "%lf", &m_paramsLearning.epsilon); 

			if(array.GetString(0) == "layers")
				sscanf(array.GetString(1).c_str(), "%d", &layers); 
			if(array.GetString(0) == "input")
			{
				sscanf(array.GetString(1).c_str(), "%d", &m_paramsNN.input); 
				m_paramsNN.input += 1;	// add 1 for bias nodes 
			}
			if(array.GetString(0) == "hidden")
			{
				StringArray ar(array.GetString(1).c_str(), ","); 
				if((int32_t)ar.Count() + 2 != layers)
				{
					ifs.close(); 
					return _MLP_ERROR_LAYERS_MISMATCHING;
				}
				int32_t hidden; 
				for(int32_t h = 0; h < (int32_t)ar.Count(); h++) 
				{
					sscanf(ar.GetString(h).c_str(), "%d", &hidden); 
					m_paramsNN.vtr_hidden.push_back(hidden); 
					vtr_hoffs.push_back(0); 	
				}
			}
			if(array.GetString(0) == "output")
				sscanf(array.GetString(1).c_str(), "%d", &m_paramsNN.output); 
			if(array.GetString(0) == "hidden_activation")
				m_paramsNN.act_hidden = TypeDefs::ActType(array.GetString(1).c_str()); 
			if(array.GetString(0) == "output_activation")
				m_paramsNN.act_output = TypeDefs::ActType(array.GetString(1).c_str()); 
		}
		else if(step == 1)
		{ // for m_whs
			if(!create_flag)
			{
				Create(); 
				create_flag = true; 
			}
			StringArray array(str.c_str(), ","); 
			if((int32_t)array.Count() != m_paramsNN.vtr_hidden[cur_h])	
			{
				ifs.close();
				return _MLP_ERROR_WEIGHT_MISALIGNMENT;
			}
			int32_t row, col; 
			for(int32_t k = 0; k < (int32_t)array.Count(); k++) 
			{
				row = vtr_hoffs[cur_h] / m_whs[cur_h].Cols();
				col = vtr_hoffs[cur_h] % m_whs[cur_h].Cols();
				sscanf(array.GetString(k).c_str(), "%lf", &(m_whs[cur_h][row][col])); 
				vtr_hoffs[cur_h] += 1; 
			}
		}
		else
		{ // for m_wo
			if(!create_flag)
			{
				Create(); 
				create_flag = true; 
			}
			StringArray array(str.c_str(), ","); 
			if((int32_t)array.Count() != m_paramsNN.output)
			{
				ifs.close(); 
				return _MLP_ERROR_WEIGHT_MISALIGNMENT;
			}
			int32_t row, col; 	
			for(int32_t k = 0; k < (int32_t)array.Count(); k++) 
			{
				row = wo_off / m_wo.Cols(); 
				col = wo_off % m_wo.Cols(); 
				sscanf(array.GetString(k).c_str(), "%lf", &(m_wo[row][col]));
				wo_off += 1; 
			}
		}
	}
	ifs.close(); 

	if(wo_off != (m_paramsNN.vtr_hidden[layers-3]) * m_paramsNN.output)
		return _MLP_ERROR_WEIGHT_MISALIGNMENT;
	if(vtr_hoffs[0] != (m_paramsNN.input) * m_paramsNN.vtr_hidden[0])
		return _MLP_ERROR_WEIGHT_MISALIGNMENT;
	for(int32_t h = 1; h < layers - 2; h++) 
	{
		if(vtr_hoffs[h] != (m_paramsNN.vtr_hidden[h-1]) * m_paramsNN.vtr_hidden[h])
		{
			return _MLP_ERROR_WEIGHT_MISALIGNMENT;
		}
	}
	return _MLP_SUCCESS; 
}


LearningParamsT MLP_NeuralNetwork::GetLearningParams()
{
	return m_paramsLearning; 
}


MLPNNParamsT MLP_NeuralNetwork::GetMLPNNParams()
{
	return m_paramsNN; 
}


////////////////////////////////////////////////////////////////////////////////////////////
// Internal Operations 


void MLP_NeuralNetwork::Create()
{
	int32_t hl = (int32_t)m_paramsNN.vtr_hidden.size();	// number of hidden layers

	// create transform matrices and change matrices of hidden layers
	m_whs = new Matrix[hl];
	m_chs = new Matrix[hl];
	for(int32_t h = 0; h < hl; h++)
	{
		if(h == 0)	
		{ 
			m_whs[h].Create(m_paramsNN.input, m_paramsNN.vtr_hidden[h]);
			m_chs[h].Create(m_paramsNN.input, m_paramsNN.vtr_hidden[h]);
		}
		else	
		{ 
			m_whs[h].Create(m_paramsNN.vtr_hidden[h-1], m_paramsNN.vtr_hidden[h]);
			m_chs[h].Create(m_paramsNN.vtr_hidden[h-1], m_paramsNN.vtr_hidden[h]);
		}
		Activation::InitTransformMatrix(m_whs[h], m_paramsNN.act_hidden); 
		m_chs[h].Init(0.0); 
	}
	// create transform matrix and change matrix of output layer
	m_wo.Create(m_paramsNN.vtr_hidden[hl-1], m_paramsNN.output);
	m_co.Create(m_paramsNN.vtr_hidden[hl-1], m_paramsNN.output);
	Activation::InitTransformMatrix(m_wo, m_paramsNN.act_output); 
	m_co.Init(0.0);

	// create input layer
	m_ai = new double[m_paramsNN.input];
	// create hidden layers
	m_ahs = new double*[hl];
	for(int32_t h = 0; h < hl; h++)
		m_ahs[h] = new double[m_paramsNN.vtr_hidden[h]];
	// create output layer
	m_ao = new double[m_paramsNN.output];

	// create delta arrays of hidden layers
	m_dhs = new double*[hl]; 
	for(int32_t h = 0; h < hl; h++)
		m_dhs[h] = new double[m_paramsNN.vtr_hidden[h]];
	// create delta array of output layers
	m_do = new double[m_paramsNN.output];
}


void MLP_NeuralNetwork::Release()
{
	int32_t hl = (int32_t)m_paramsNN.vtr_hidden.size();	// number of hidden layers

	if(m_whs)
	{
		delete [] m_whs; 
		m_whs = NULL; 
	}

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


void MLP_NeuralNetwork::FeedForward(const double* in_vals, const int32_t in_len)
{
	if(!in_vals || in_len != m_paramsNN.input - 1)
		throw "MLP_NeuralNetwork::FeedForward() ERROR: Wrong length of \'in_vals\'!"; 

	// activate input layer
	for(int32_t i = 0; i < in_len; i++) 
		m_ai[i] = in_vals[i];
	m_ai[in_len] = 1.0;		// set bias

	// activate hidden layer-by-layer
	int32_t hl = (int32_t)m_paramsNN.vtr_hidden.size();	// number of hidden layers
	for(int32_t h = 0; h < hl; h++) 
	{
		if(h == 0)	// activate the first hidden layer by input layer
			ActivateForward(m_ahs[h], m_paramsNN.vtr_hidden[h], m_ai, m_paramsNN.input, m_whs[h], m_paramsNN.act_hidden); 
		else	// activate the upper layer by lower layer
			ActivateForward(m_ahs[h], m_paramsNN.vtr_hidden[h], m_ahs[h-1], m_paramsNN.vtr_hidden[h-1], m_whs[h], m_paramsNN.act_hidden); 
	}

	// activate output layer
	ActivateForward(m_ao, m_paramsNN.output, m_ahs[hl-1], m_paramsNN.vtr_hidden[hl-1], m_wo, m_paramsNN.act_output); 
}


double MLP_NeuralNetwork::BackPropagate(const double* out_vals, const int32_t out_len)
{
	if(!out_vals || out_len != m_paramsNN.output)
		throw "MLP_NeuralNetwork::BackPropagate() ERROR: Wrong length of \'out_vals\'!"; 

	double error = 0.0; 
	int32_t hl = (int32_t)m_paramsNN.vtr_hidden.size();	// number of hidden layers

	// caculate delta and error of output layer
	for(int32_t j = 0; j < m_paramsNN.output; j++) 
	{
		m_do[j] = m_ao[j] - out_vals[j]; 
		error += m_do[j] * m_do[j];  
	}

	// output layer back propagate
	for(int32_t i = 0; i < m_paramsNN.vtr_hidden[hl-1]; i++)
	{
		m_dhs[hl-1][i] = 0.0;   
		for(int32_t j = 0; j < m_paramsNN.output; j++)
			m_dhs[hl-1][i] += m_do[j] * m_wo[i][j]; 
	}
	for(int32_t h = hl - 2; h >= 0; h--)
	{
		for(int32_t i = 0; i < m_paramsNN.vtr_hidden[h]; i++) 
		{
			m_dhs[h][i] = 0.0;
			for(int32_t j = 0; j < m_paramsNN.vtr_hidden[h+1]; j++) 
				m_dhs[h][i] += m_dhs[h+1][j] * m_whs[h+1][i][j];  
		}	
	}

	// update change matrices
	for(int32_t j = 0; j < m_paramsNN.output; j++)
	{ // m_co
		for(int32_t i = 0; i < m_paramsNN.vtr_hidden[hl-1]; i++)
			m_co[i][j] += m_do[j] * Activation::DActivate(m_ao[j], m_paramsNN.act_output) * m_ahs[hl-1][i];  
	}
	for(int32_t h = hl - 1; h > 0; h--)
	{ // m_whs
		for(int32_t j = 0; j < m_paramsNN.vtr_hidden[h]; j++) 
		{
			for(int32_t i = 0; i < m_paramsNN.vtr_hidden[h-1]; i++) 
				m_chs[h][i][j] += m_dhs[h][j] * Activation::DActivate(m_ahs[h][j], m_paramsNN.act_hidden) * m_ahs[h-1][i]; 	
		}
	}
	for(int32_t j = 0; j < m_paramsNN.vtr_hidden[0]; j++) 
	{ // m_whs[0]
		for(int32_t i = 0; i < m_paramsNN.input; i++) 
			m_chs[0][i][j] += m_dhs[0][j] * Activation::DActivate(m_ahs[0][j], m_paramsNN.act_hidden) * m_ai[i]; 
	}

	return error / double(m_paramsNN.output); 
}


void MLP_NeuralNetwork::ActivateForward(double* up_a, const int32_t up_size, const double* low_a, const int32_t low_size, 
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


void MLP_NeuralNetwork::UpdateWeightMats(const double learning_rate)
{
	int32_t hl = (int32_t)m_paramsNN.vtr_hidden.size();	// number of hidden layers

	// update the transform matrix of output layer (m_wo)
	for(int32_t i = 0; i < m_paramsNN.vtr_hidden[hl-1]; i++) 
	{
		for(int32_t j = 0; j < m_paramsNN.output; j++) 
		{
			m_wo[i][j] -= learning_rate * (m_co[i][j] + Activation::DActRegula(m_wo[i][j], m_paramsLearning.regula));
			m_co[i][j] = 0.0; 
		}
	}

	// update the transform matrices of hidden layers (m_whs)
	for(int32_t h = hl - 1; h > 0; h--)
	{
		for(int32_t i = 0; i < m_paramsNN.vtr_hidden[h-1]; i++)
		{
			for(int32_t j = 0; j < m_paramsNN.vtr_hidden[h]; j++)
			{
				m_whs[h][i][j] -= learning_rate * (m_chs[h][i][j] + Activation::DActRegula(m_whs[h][i][j], m_paramsLearning.regula));
				m_chs[h][i][j] = 0.0; 
			}
		}
	}
	for(int32_t i = 0; i < m_paramsNN.input; i++) 
	{
		for(int32_t j = 0; j < m_paramsNN.vtr_hidden[0]; j++)
		{
			m_whs[0][i][j] -= learning_rate * (m_chs[0][i][j] + Activation::DActRegula(m_whs[0][i][j], m_paramsLearning.regula));
			m_chs[0][i][j] = 0.0; 
		}
	}
}


pair<double, double> MLP_NeuralNetwork::Validation(vector<Pattern*>& vtrPatts, const int32_t nBackCnt)
{
	double error = 0.0; 
	int32_t correct = 0, total = 0; 
	int32_t patts = (int32_t)vtrPatts.size(); 
	double* y = new double[m_paramsNN.output];
	int32_t k; 

	for(int32_t i = 0; i < nBackCnt && i < patts; i++) 
	{
		k = patts-1-i;
		Predict(y, m_paramsNN.output, vtrPatts[k]->m_x, vtrPatts[k]->m_nXCnt); 
		if(Pattern::MaxOff(y, m_paramsNN.output) == Pattern::MaxOff(vtrPatts[k]->m_y, vtrPatts[k]->m_nYCnt))
			correct += 1; 	
		error += Pattern::Error(y, vtrPatts[k]->m_y, m_paramsNN.output);  	
		total += 1; 	
	}

	delete y;

	double pr = (double)correct / (double)total;
	double rmse = sqrt(error / (double)total);

	return pair<double,double>(pr, rmse); 
}



