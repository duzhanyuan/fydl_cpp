#include "Perceptron.h"
#include "Timer.h"
#include "StringArray.h"
#include "Activation.h"
using namespace fydl; 
#include <algorithm>
#include <fstream>
#include <iostream>
using namespace std; 
#include <math.h>
#include <stdio.h>


////////////////////////////////////////////////////////////////////////////////////////////
// Construction & Destruction  

Perceptron::Perceptron()
{
	m_paramsLearning.regula = _REGULA_L1;
	m_paramsLearning.mini_batch = 0; 
	m_paramsLearning.iterations = 50;
	m_paramsLearning.learning_rate = 0.5;
	m_paramsLearning.rate_decay = 0.01;
	m_paramsLearning.epsilon = 0.01;
	m_nIters = 0; 

	m_paramsPerceptron.act_output = _ACT_SIGMOID;

	m_ai = NULL; 
	m_ao = NULL; 

	m_do = NULL; 
}


Perceptron::~Perceptron()
{
	Release(); 
}


////////////////////////////////////////////////////////////////////////////////////////////
// Operations 

void Perceptron::Init(const int32_t nInput, const int32_t nOutput, const EActType eActOutput, const LearningParamsT* pLearningParamsT)
{
	Release(); 

	m_paramsPerceptron.input = nInput + 1; // add 1 for bias nodes 
	m_paramsPerceptron.output = nOutput; 
	m_paramsPerceptron.act_output = eActOutput;
	if(pLearningParamsT)
		m_paramsLearning = *pLearningParamsT; 

	Create(); 
}


bool Perceptron::InitFromConfig(const char* sConfigFile, const int32_t nInput, const int32_t nOutput)
{
	if(!sConfigFile)
		return false; 
	ifstream ifs(sConfigFile);
	if(!ifs.is_open())
		return false; 

	Release(); 

	m_paramsPerceptron.input = nInput + 1;		// add 1 for bias nodes
	m_paramsPerceptron.output = nOutput;

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
		else if(ar.GetString(0) == "OutputActivation")
			m_paramsPerceptron.act_output = TypeDefs::ActType(ar.GetString(1).c_str()); 
	}
	Create(); 

	ifs.close();
	return true; 
}


void Perceptron::Train(vector<Pattern*>& vtrPatts)
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
					UpdateTransformMatrix(learning_rate);
			}
		}	

		if(m_paramsLearning.mini_batch == 0)	// batch 
			UpdateTransformMatrix(learning_rate);
		
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


int32_t Perceptron::Predict(double* y, const int32_t y_len, const double* x, const int32_t x_len)
{
	if(!y || !x)
		return _PERCEPTRON_ERROR_INPUT_NULL;
	if(y_len != m_paramsPerceptron.output || x_len != m_paramsPerceptron.input - 1)
		return _PERCEPTRON_ERROR_WRONG_LEN;
	if(m_wo.IsNull())
		return _PERCEPTRON_ERROR_MODEL_NULL;

	// for thread save, do not use inner layer
	double* ai = new double[m_paramsPerceptron.input];		// input layer
	
	// activate input layer
	for(int32_t i = 0; i < x_len; i++) 
		ai[i] = x[i];
	ai[x_len] = 1.0;		// set bias

	// activate output 
	double sum; 	
	double e = 0.0; 
	for(int32_t j = 0; j < m_paramsPerceptron.output; j++)	
	{
		sum = 0.0; 	
		for(int32_t i = 0; i < m_paramsPerceptron.output; i++)
			sum += ai[i] * m_wo[i][j];
		y[j] = Activation::Activate(sum, m_paramsPerceptron.act_output);  

		if(m_paramsPerceptron.act_output == _ACT_SOFTMAX)
			e += y[j]; 
	}
	if(m_paramsPerceptron.act_output == _ACT_SOFTMAX)
	{
		for(int32_t j = 0; j < m_paramsPerceptron.output; j++)	
			y[j] /= e; 
	}

	delete ai; 
	
	return _PERCEPTRON_SUCCESS; 
}


int32_t Perceptron::Save(const char* sFile, const char* sTitle)
{
	if(m_wo.IsNull())
		return _PERCEPTRON_ERROR_MODEL_NULL; 
	FILE* fp = fopen(sFile, "w"); 
	if(!fp)
		return _PERCEPTRON_ERROR_FILE_OPEN;

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
	fprintf(fp, "input:%d\n", m_paramsPerceptron.input-1); 
	fprintf(fp, "output:%d\n", m_paramsPerceptron.output); 
	fprintf(fp, "output_activation:%s\n\n", TypeDefs::ActName(m_paramsPerceptron.act_output).c_str()); 
	fprintf(fp, "\n"); 

	fprintf(fp, "@weight\n"); 
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
	return _PERCEPTRON_SUCCESS; 
}


int32_t Perceptron::Load(const char* sFile, const char* sCheckTitle)
{
	ifstream ifs(sFile);  
	if(!ifs.is_open())
		return _PERCEPTRON_ERROR_FILE_OPEN;
	Release(); 

	string str; 
	int32_t step = 0, wo_off = 0; 
	bool create_flag = false; 

	if(sCheckTitle)
	{
		std::getline(ifs, str); 
		if(str.find(sCheckTitle) == string::npos) 
		{
			ifs.close();
			return _PERCEPTRON_ERROR_NOT_MODEL_FILE;
		}
	}

	while(!ifs.eof())
	{
		std::getline(ifs, str);
		if(str.empty())
			continue; 
		else if(str == "@weight")
		{
			step = 1; 
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

			if(array.GetString(0) == "input")
			{
				sscanf(array.GetString(1).c_str(), "%d", &m_paramsPerceptron.input); 
				m_paramsPerceptron.input += 1;	// add 1 for bias nodes 
			}
			if(array.GetString(0) == "output")
				sscanf(array.GetString(1).c_str(), "%d", &m_paramsPerceptron.output); 
			if(array.GetString(0) == "output_activation")
				m_paramsPerceptron.act_output = TypeDefs::ActType(array.GetString(1).c_str()); 
		}
		else
		{ // for m_wo
			if(!create_flag)
			{
				Create(); 
				create_flag = true; 
			}
			StringArray array(str.c_str(), ","); 
			if((int32_t)array.Count() != m_paramsPerceptron.output)
			{
				ifs.close(); 
				return _PERCEPTRON_ERROR_WEIGHT_MISALIGNMENT;
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

	if(wo_off != m_paramsPerceptron.input * m_paramsPerceptron.output)
		return _PERCEPTRON_ERROR_WEIGHT_MISALIGNMENT;

	return _PERCEPTRON_SUCCESS; 
}


LearningParamsT Perceptron::GetLearningParams()
{
	return m_paramsLearning; 
}


PerceptronParamsT Perceptron::GetPerceptronParams()
{
	return m_paramsPerceptron;	
}


////////////////////////////////////////////////////////////////////////////////////////////
// Internal Operations


void Perceptron::Create()
{
	// create transform matrix and change matrix
	m_wo.Create(m_paramsPerceptron.input, m_paramsPerceptron.output); 
	m_co.Create(m_paramsPerceptron.input, m_paramsPerceptron.output); 
	Activation::InitTransformMatrix(m_wo, m_paramsPerceptron.act_output); 
	m_co.Init(0.0);

	// create input layer
	m_ai = new double[m_paramsPerceptron.input];
	// create output layer
	m_ao = new double[m_paramsPerceptron.output];

	// create delta array
	m_do = new double[m_paramsPerceptron.output];
}


void Perceptron::Release()
{
	if(m_ai)
	{
		delete m_ai; 
		m_ai = NULL; 
	}
	if(m_ao)
	{
		delete m_ao; 
		m_ao = NULL; 
	}

	if(m_do)
	{
		delete m_do; 
		m_do = NULL; 
	}
}


void Perceptron::FeedForward(const double* in_vals, const int32_t in_len)
{
	if(!in_vals || in_len != m_paramsPerceptron.input - 1)
		throw "Perceptron::FeedForward() ERROR: Wrong length of \'in_vals\'!"; 

	// activate input layer
	for(int32_t i = 0; i < in_len; i++) 
		m_ai[i] = in_vals[i];
	m_ai[in_len] = 1.0;		// set bias

	// activate output layer
	double sum; 	
	double e = 0.0; 
	for(int32_t j = 0; j < m_paramsPerceptron.output; j++)	
	{
		sum = 0.0; 	
		for(int32_t i = 0; i < m_paramsPerceptron.output; i++)
			sum += m_ai[i] * m_wo[i][j];
		m_ao[j] = Activation::Activate(sum, m_paramsPerceptron.act_output);  

		if(m_paramsPerceptron.act_output == _ACT_SOFTMAX)
			e += m_ao[j]; 
	}
	if(m_paramsPerceptron.act_output == _ACT_SOFTMAX)
	{
		for(int32_t j = 0; j < m_paramsPerceptron.output; j++)	
			m_ao[j] /= e; 
	}
}


double Perceptron::BackPropagate(const double* out_vals, const int32_t out_len)
{
	if(!out_vals || out_len != m_paramsPerceptron.output)
		throw "Perceptron::BackPropagate() ERROR: Wrong length of \'out_vals\'!"; 

	double error = 0.0; 
	// calculate delta and error of output 
	for(int32_t j = 0; j < m_paramsPerceptron.output; j++) 
	{
		m_do[j] = m_ao[j] - out_vals[j]; 
		error += m_do[j] * m_do[j];  
	}

	// update change matrix
	for(int32_t j = 0; j < m_paramsPerceptron.output; j++)
	{ // m_co
		for(int32_t i = 0; i < m_paramsPerceptron.input; i++) 
			m_co[i][j] += m_do[j] * Activation::DActivate(m_ao[j], m_paramsPerceptron.act_output) * m_ai[i]; 
	}
	
	return error / double(m_paramsPerceptron.output); 
}


void Perceptron::UpdateTransformMatrix(const double learning_rate)
{
	// update transform matrix
	for(int32_t i = 0; i < m_paramsPerceptron.input; i++) 
	{
		for(int32_t j = 0; j < m_paramsPerceptron.output; j++) 
		{
			m_wo[i][j] -= learning_rate * (m_co[i][j] + Activation::DActRegula(m_wo[i][j], m_paramsLearning.regula));
			m_co[i][j] = 0.0; 
		}
	}
}


pair<double, double> Perceptron::Validation(vector<Pattern*>& vtrPatts, const int32_t nBackCnt)
{
	double error = 0.0; 
	int32_t correct = 0, total = 0; 
	int32_t patts = (int32_t)vtrPatts.size(); 
	double* y = new double[m_paramsPerceptron.output];
	int32_t k; 

	for(int32_t i = 0; i < nBackCnt && i < patts; i++) 
	{
		k = patts-1-i;
		Predict(y, m_paramsPerceptron.output, vtrPatts[k]->m_x, vtrPatts[k]->m_nXCnt); 
		if(Pattern::MaxOff(y, m_paramsPerceptron.output) == Pattern::MaxOff(vtrPatts[k]->m_y, vtrPatts[k]->m_nYCnt))
			correct += 1; 	
		error += Pattern::Error(y, vtrPatts[k]->m_y, m_paramsPerceptron.output);  	
		total += 1; 	
	}

	delete y;

	double pr = (double)correct / (double)total;
	double rmse = sqrt(error / (double)total);

	return pair<double,double>(pr, rmse); 
}


