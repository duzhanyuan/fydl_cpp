#include "DBN.h"
#include "ConfigFile.h"
#include "StringArray.h"
#include "Timer.h"
using namespace fydl; 
#include <algorithm>
#include <iostream>
#include <fstream>
using namespace std; 
#include <math.h>


////////////////////////////////////////////////////////////////////////////////////////////
// Construction & Destruction  

DBN::DBN()
{
	m_rbms = NULL;
	m_rbms_ao = NULL;
}


DBN::~DBN()
{
	Release(); 
}


////////////////////////////////////////////////////////////////////////////////////////////
// Operations
	
void DBN::Init(const DBNParamsT dbnParamsT, const DBNLearningParamsT dbnLearningParamsT)
{
	Release(); 

	m_paramsDBN = dbnParamsT; 
	m_paramsLearning = dbnLearningParamsT; 

	Create(); 
}


bool DBN::InitFromConfig(const char* sConfigFile, const int32_t nInput, const int32_t nOutput)
{
	ConfigFile conf_file; 
	if(!conf_file.Read(sConfigFile)) 
		return false; 
	Release(); 
	m_paramsDBN.input = nInput; 
	m_paramsDBN.output = nOutput;
	m_paramsDBN.rbms_type = TypeDefs::RBMType(conf_file.GetVal_asString("RBMs_Type").c_str());
	if(m_paramsDBN.rbms_type == _UNKNOWN_RBM)
		return false; 
	m_paramsDBN.vtr_rbms_hidden.clear(); 
	for(int32_t i = 0; i < conf_file.ValCnt("RBMs_Hiddens"); i++) 
		m_paramsDBN.vtr_rbms_hidden.push_back(conf_file.GetVal_asInt("RBMs_Hiddens", i)); 
	if(m_paramsDBN.vtr_rbms_hidden.empty())
		return false; 
	m_paramsDBN.vtr_mlp_hidden.clear(); 
	for(int32_t i = 0; i < conf_file.ValCnt("MLP_Hiddens"); i++) 
		m_paramsDBN.vtr_mlp_hidden.push_back(conf_file.GetVal_asInt("MLP_Hiddens", i)); 
	if(m_paramsDBN.vtr_mlp_hidden.empty())
		return false; 
	m_paramsDBN.mlp_act_hidden = TypeDefs::ActType(conf_file.GetVal_asString("MLP_ActHidden").c_str());
	m_paramsDBN.mlp_act_output = TypeDefs::ActType(conf_file.GetVal_asString("MLP_ActOutput").c_str());

	m_paramsLearning.rbm_learning_params.gibbs_steps = conf_file.GetVal_asInt("RBMs_GibbsSteps");
	m_paramsLearning.rbm_learning_params.mini_batch = conf_file.GetVal_asInt("RBMs_MiniBatch");
	m_paramsLearning.rbm_learning_params.iterations = conf_file.GetVal_asInt("RBMs_Iterations");
	m_paramsLearning.rbm_learning_params.learning_rate = conf_file.GetVal_asFloat("RBMs_LearningRate");
	m_paramsLearning.rbm_learning_params.rate_decay = conf_file.GetVal_asFloat("RBMs_RateDecay");
	m_paramsLearning.rbm_learning_params.epsilon= conf_file.GetVal_asFloat("RBMs_Epsilon");
	
	m_paramsLearning.mlp_learning_params.regula = TypeDefs::RegulaType(conf_file.GetVal_asString("MLP_Regula").c_str()); 
	m_paramsLearning.mlp_learning_params.mini_batch = conf_file.GetVal_asInt("MLP_MiniBatch");
	m_paramsLearning.mlp_learning_params.iterations = conf_file.GetVal_asInt("MLP_Iterations");
	m_paramsLearning.mlp_learning_params.learning_rate = conf_file.GetVal_asFloat("MLP_LearningRate");
	m_paramsLearning.mlp_learning_params.rate_decay = conf_file.GetVal_asFloat("MLP_RateDecay");
	m_paramsLearning.mlp_learning_params.epsilon= conf_file.GetVal_asFloat("MLP_Epsilon");

	Create(); 
	return true; 
}


void DBN::Train(vector<Pattern*>& vtrPatts)
{
	CreateAssistant(); 

	cout<<"** Pre-train **"<<endl; 
	PreTrain(vtrPatts); 
	cout<<"** Fine Tuning **"<<endl; 
	FineTuning(vtrPatts); 
	cout<<"** finish **"<<endl; 
}


int32_t DBN::Predict(double* y, const int32_t y_len, const double* x, const int32_t x_len)
{
	if(!y || !x)
		return _FYDL_ERROR_INPUT_NULL;
	if(y_len != m_paramsDBN.output || x_len != m_paramsDBN.input)
	{
		return _FYDL_ERROR_WRONG_LEN;
	}
	if(!m_mlp.m_whs || m_mlp.m_wo.IsNull() || !m_rbms)
		return _FYDL_ERROR_MODEL_NULL;

	int32_t rbms_num = (int32_t)m_paramsDBN.vtr_rbms_hidden.size();		// number of RBMs
	double** ao = new double*[rbms_num]; 
	for(int32_t r = 0; r < rbms_num; r++)
		ao[r] = new double[m_paramsDBN.vtr_rbms_hidden[r]]; 
	int32_t ret; 

	// generating
	ret = Generate(ao, x, rbms_num-1); 

	// prediction
	if(ret == _FYDL_SUCCESS)
		ret = m_mlp.Predict(y, y_len, ao[rbms_num-1], m_paramsDBN.vtr_rbms_hidden[rbms_num-1]); 

	for(int32_t r = 0; r < rbms_num; r++)
		delete ao[r];
	delete ao;

	return ret; 
}


int32_t DBN::Save(const char* sFile, const char* sTitle)
{
	if(!m_mlp.m_whs || m_mlp.m_wo.IsNull() || !m_rbms)
		return _FYDL_ERROR_MODEL_NULL; 
	ofstream ofs(sFile); 
	if(!ofs.is_open())
		return _FYDL_ERROR_FILE_OPEN;

	int32_t mlp_hl = (int32_t)m_paramsDBN.vtr_mlp_hidden.size();	// number of MLP hidden layers
	int32_t rbms_num = (int32_t)m_paramsDBN.vtr_rbms_hidden.size();	// number of RBMs
	int32_t v_len, h_len;

	ofs<<"** "<<sTitle<<" **"<<endl; 
	ofs<<endl;
	
	// save learning parameters
	ofs<<"@learning_params"<<endl; 
	TypeDefs::Print_DBNLearningParamsT(ofs, m_paramsLearning); 
	ofs<<endl; 
	
	// save architecture parameters of DBN
	ofs<<"@architecture_params"<<endl; 
	TypeDefs::Print_DBNParamsT(ofs, m_paramsDBN); 
	ofs<<endl; 

	////////////////////////////////////////////
	// save model variables of RBMs

	for(int32_t r = 0; r < rbms_num; r++) 
	{
		if(r == 0)
			v_len = m_paramsDBN.input; 
		else
			v_len = m_paramsDBN.vtr_rbms_hidden[r-1]; 
		h_len = m_paramsDBN.vtr_rbms_hidden[r]; 

		// save transtorm matrix
		ofs<<"@rbm_weight_"<<r<<endl; 
		Matrix::Print_Matrix(ofs, m_rbms[r].m_w);
		ofs<<endl; 
	
		// save visible bias 
		ofs<<"@rbm_visible_bias_"<<r<<endl; 
		Pattern::Print_Array(ofs, m_rbms[r].m_vbias, v_len); 
		ofs<<endl; 
		
		// save hidden bias 
		ofs<<"@rbm_hidden_bias_"<<r<<endl; 
		Pattern::Print_Array(ofs, m_rbms[r].m_hbias, h_len); 
		ofs<<endl; 
	}


	////////////////////////////////////////////
	// save model variables of MLP

	for(int32_t h = 0; h < mlp_hl; h++) 
	{
		ofs<<"@mlp_weight_hidden_"<<h<<endl; 
		Matrix::Print_Matrix(ofs, m_mlp.m_whs[h]);
		ofs<<endl; 
	}
	ofs<<"@mlp_weight_output"<<endl; 
	Matrix::Print_Matrix(ofs, m_mlp.m_wo); 
	ofs<<endl; 

	ofs.close(); 
	return _FYDL_SUCCESS; 
}


int32_t DBN::Load(const char* sFile, const char* sCheckTitle)
{
	Release();

	ifstream ifs(sFile);  
	if(!ifs.is_open())
		return _FYDL_ERROR_FILE_OPEN;
	Release(); 

	string str; 
	int32_t idx, v_len, h_len; 

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
			if(!TypeDefs::Read_DBNLearningParamsT(m_paramsLearning, ifs))
				return _FYDL_ERROR_LERANING_PARAMS;
		}
		else if(str == "@architecture_params")
		{
			if(!TypeDefs::Read_DBNParamsT(m_paramsDBN, ifs))
				return _FYDL_ERROR_ACH_PARAMS;
			Create(); 
		}
		else if(str.find("@rbm_weight_") == 0)
		{
			StringArray ar(str.c_str(), "_"); 
			sscanf(ar.GetString(ar.Count()-1).c_str(), "%d", &idx); 	
			
			if(!Matrix::Read_Matrix(m_rbms[idx].m_w, ifs))
				return _FYDL_ERROR_MODEL_DATA;
		}
		else if(str.find("@rbm_visible_bias_") == 0)
		{
			StringArray ar(str.c_str(), "_"); 
			sscanf(ar.GetString(ar.Count()-1).c_str(), "%d", &idx); 	
			if(idx == 0)
				v_len = m_paramsDBN.input; 
			else
				v_len = m_paramsDBN.vtr_rbms_hidden[idx-1]; 
			
			if(!Pattern::Read_Array(m_rbms[idx].m_vbias, v_len, ifs))
				return _FYDL_ERROR_MODEL_DATA;
		}	
		else if(str.find("@rbm_hidden_bias_") == 0)
		{
			StringArray ar(str.c_str(), "_"); 
			sscanf(ar.GetString(ar.Count()-1).c_str(), "%d", &idx); 	
			h_len = m_paramsDBN.vtr_rbms_hidden[idx];

			if(!Pattern::Read_Array(m_rbms[idx].m_hbias, h_len, ifs))
				return _FYDL_ERROR_MODEL_DATA;
		}	
		else if(str.find("@mlp_weight_hidden_") == 0)
		{
			StringArray ar(str.c_str(), "_"); 
			sscanf(ar.GetString(ar.Count()-1).c_str(), "%d", &idx); 	

			if(!Matrix::Read_Matrix(m_mlp.m_whs[idx], ifs))
				return _FYDL_ERROR_MODEL_DATA;
		}	
		else if(str == "@mlp_weight_output")
		{
			if(!Matrix::Read_Matrix(m_mlp.m_wo, ifs))
				return _FYDL_ERROR_MODEL_DATA;
		}	
	}
	
	ifs.close(); 	
	return _FYDL_SUCCESS; 
}


DBNLearningParamsT DBN::GetLearningParams()
{
	return m_paramsLearning;
}


DBNParamsT DBN::GetArchParams()
{
	return m_paramsDBN; 
}


////////////////////////////////////////////////////////////////////////////////////////////
// Internal Operations 

void DBN::Create()
{
	RBMParamsT rbmParamsT;		// architecture parameters of RBM
	MLPParamsT mlpParamsT;		// architecture parameters of MLP
	int32_t rbms_num = (int32_t)m_paramsDBN.vtr_rbms_hidden.size();	// number of RBMs 

	// create RBMs, which on the bottom of DBN
	m_rbms = new RBM[rbms_num];
	// initialize RBMs layer-by-layer on up-order
	for(int32_t r = 0; r < rbms_num; r++) 
	{
		rbmParamsT.rbm_type = m_paramsDBN.rbms_type; 
		if(r == 0)	
			rbmParamsT.visible = m_paramsDBN.input;
		else
			rbmParamsT.visible = m_paramsDBN.vtr_rbms_hidden[r-1];
		rbmParamsT.hidden = m_paramsDBN.vtr_rbms_hidden[r];
		// initialize the RBM
		m_rbms[r].Init(rbmParamsT, m_paramsLearning.rbm_learning_params); 
	}
	
	// get architecture parameters of MLP
	mlpParamsT.input = m_paramsDBN.vtr_rbms_hidden[rbms_num-1]; 
	mlpParamsT.output = m_paramsDBN.output; 
	mlpParamsT.vtr_hidden = m_paramsDBN.vtr_mlp_hidden; 
	mlpParamsT.act_hidden = m_paramsDBN.mlp_act_hidden; 
	mlpParamsT.act_output = m_paramsDBN.mlp_act_output; 
	// initialize the MLP, which on the top of DBN
	m_mlp.Init(mlpParamsT, m_paramsLearning.mlp_learning_params); 
}


void DBN::Release()
{
	if(m_rbms)
	{
		delete [] m_rbms; 
		m_rbms = NULL; 
	}
	
	ReleaseAssistant(); 
}


void DBN::CreateAssistant()
{
	ReleaseAssistant(); 

	int32_t rbms_num = (int32_t)m_paramsDBN.vtr_rbms_hidden.size();	// number of RBMs 
	m_rbms_ao = new double*[rbms_num]; 
	for(int32_t r = 0; r < rbms_num; r++) 
		m_rbms_ao[r] = new double[m_paramsDBN.vtr_rbms_hidden[r]]; 
}


void DBN::ReleaseAssistant()
{
	int32_t rbms_num = (int32_t)m_paramsDBN.vtr_rbms_hidden.size();	// number of RBMs 

	if(m_rbms_ao)
	{
		for(int32_t r = 0; r < rbms_num; r++) 
			delete m_rbms_ao[r];
		delete m_rbms_ao; 
		m_rbms_ao = NULL; 
	}
}


void DBN::PreTrain(vector<Pattern*>& vtrPatts)
{
	int32_t patt_cnt, ret, success; 
	double learning_rate;
	double error, rmse;	// training error and RMSE in one iteration
	int32_t rbms_num = (int32_t)m_paramsDBN.vtr_rbms_hidden.size();		// number of RBMs
	bool first; 
	Timer timer;		// timer

	for(int32_t r = 0; r < rbms_num; r++) 
	{
		patt_cnt  = 0; 
		first = true; 
		learning_rate = m_paramsLearning.rbm_learning_params.learning_rate;	// learning rate, it would be update after every iteration

		for(int32_t t = 0; t < m_paramsLearning.rbm_learning_params.iterations; t++) 
		{
			error = 0.0; 	
			success = 0; 
			
			timer.Start(); 	
		
			// shuffle pattens 
			random_shuffle(vtrPatts.begin(), vtrPatts.end());
		
			for(int32_t p = 0; p < (int32_t)vtrPatts.size(); p++) 
			{
				if(r == 0)
				{
					error += m_rbms[r].CDk_Step(vtrPatts[p]->m_x, vtrPatts[p]->m_nXCnt, first); 
					first = false; 
					patt_cnt++;
					success++; 
				}
				else
				{
					// generating
					ret = Generate(m_rbms_ao, vtrPatts[p]->m_x, r-1);
					// CD-k
					if(ret == _FYDL_SUCCESS)
					{
						error += m_rbms[r].CDk_Step(m_rbms_ao[r-1], m_paramsDBN.vtr_rbms_hidden[r-1], first); 
						first = false; 
						patt_cnt++;
						success++; 
					}
				}	
				
				if(m_paramsLearning.rbm_learning_params.mini_batch > 0)	// online or mini-batch
				{
					if(patt_cnt >= m_paramsLearning.rbm_learning_params.mini_batch)
					{
						m_rbms[r].ModelUpdate(learning_rate); 
						patt_cnt = 0; 	
					}
				}
			}

			if(m_paramsLearning.rbm_learning_params.mini_batch == 0)	// batch
			{
				m_rbms[r].ModelUpdate(learning_rate); 
				patt_cnt = 0; 	
			}
			
			rmse = sqrt(error / (double)success);

			timer.Stop();

			printf("RBM %d | iter %d | learning_rate: %.6g | error: %.6g | rmse: %.6g | time_cost(s): %.3f\n", 
				r, t+1, learning_rate, error, rmse, timer.GetLast_asSec()); 
			learning_rate = learning_rate * (learning_rate / (learning_rate + (learning_rate * m_paramsLearning.mlp_learning_params.rate_decay)));	
		
			if(rmse <= m_paramsLearning.rbm_learning_params.epsilon)
				break;
		}
	}
}


void DBN::FineTuning(vector<Pattern*>& vtrPatts)
{
	int32_t patt_cnt = 0; 
	int32_t cross_cnt = (int32_t)vtrPatts.size() / 20;			// 5% patterns for corss validation
	int32_t train_cnt = (int32_t)vtrPatts.size() - cross_cnt;	// 95% patterns for training
	double learning_rate = m_paramsLearning.mlp_learning_params.learning_rate;	// learning rate, it would be update after every iteration
	double error, rmse;	// training error and RMSE in one iteration
	pair<double,double> validation;	// precision and RMSE of validation 
	int32_t rbms_num = (int32_t)m_paramsDBN.vtr_rbms_hidden.size();		// number of RBMs
	int32_t ret; 
	bool first = true; 
	Timer timer;		// timer

	// shuffle pattens 
	random_shuffle(vtrPatts.begin(), vtrPatts.end());

	for(int32_t t = 0; t < m_paramsLearning.mlp_learning_params.iterations; t++) 
	{
		error = 0.0; 	

		timer.Start(); 	

		// shuffle training patterns
		random_shuffle(vtrPatts.begin(), vtrPatts.end() - cross_cnt);

		for(int32_t p = 0; p < train_cnt; p++) 
		{
			// generating
			ret = Generate(m_rbms_ao, vtrPatts[p]->m_x, rbms_num-1);

			// feed forward & back propagation step
			error += m_mlp.FfBp_Step(vtrPatts[p]->m_y, vtrPatts[p]->m_nYCnt, m_rbms_ao[rbms_num-1], m_paramsDBN.vtr_rbms_hidden[rbms_num-1], first); 
			first = false; 
			patt_cnt++; 

			if(m_paramsLearning.mlp_learning_params.mini_batch > 0)	// online or mini-batch
			{
				if(patt_cnt >= m_paramsLearning.mlp_learning_params.mini_batch)
				{
					m_mlp.ModelUpdate(learning_rate); 
					patt_cnt = 0; 	
				}
			}
		}
		if(m_paramsLearning.mlp_learning_params.mini_batch == 0)	// batch 
		{
			m_mlp.ModelUpdate(learning_rate); 
			patt_cnt = 0; 	
		}

		validation = Validation(vtrPatts, cross_cnt); 
		rmse = sqrt(error / (double)(train_cnt));
		
		timer.Stop(); 	

		printf("iter %d | learning_rate: %.6g | error: %.6g | rmse: %.6g | validation(pr & rmse): %.4g%% & %.6g | time_cost(s): %.3f\n", 
				t+1, learning_rate, error, rmse, validation.first * 100.0, validation.second, timer.GetLast_asSec()); 
		learning_rate = learning_rate * (learning_rate / (learning_rate + (learning_rate * m_paramsLearning.mlp_learning_params.rate_decay)));	

		if(rmse <= m_paramsLearning.mlp_learning_params.epsilon)
			break;
	}	
}


int32_t DBN::Generate(double** ao, const double* x, const int32_t out_rbm_idx)
{
	int32_t rbms_num = (int32_t)m_paramsDBN.vtr_rbms_hidden.size();		// number of RBMs
	if(out_rbm_idx < 0 || out_rbm_idx >= rbms_num)
		return _FYDL_ERROR_WRONG_LEN; 
	if(!ao || !x)
		return _FYDL_ERROR_INPUT_NULL; 
	if(!m_rbms)
		return _FYDL_ERROR_FILE_OPEN; 

	int32_t v_len, h_len, ret; 
	for(int32_t r = 0; r <= out_rbm_idx; r++) 
	{
		h_len = m_paramsDBN.vtr_rbms_hidden[r]; 
		if(r == 0)
		{
			v_len = m_paramsDBN.input;
			ret = m_rbms[r].PropagateForward(ao[r], h_len, x, v_len);
		}
		else
		{
			v_len = m_paramsDBN.vtr_rbms_hidden[r-1]; 
			ret = m_rbms[r].PropagateForward(ao[r], h_len, ao[r-1], v_len); 
		}
		if(ret != _FYDL_SUCCESS)
			break; 
	}
	
	return ret; 
}


pair<double, double> DBN::Validation(vector<Pattern*>& vtrPatts, const int32_t nBackCnt)
{
	double error = 0.0; 
	int32_t correct = 0, total = 0; 
	int32_t patts = (int32_t)vtrPatts.size(); 
	double* y = new double[m_paramsDBN.output];
	int32_t k; 

	for(int32_t i = 0; i < nBackCnt && i < patts; i++) 
	{
		k = patts-1-i;
		Predict(y, m_paramsDBN.output, vtrPatts[k]->m_x, vtrPatts[k]->m_nXCnt); 
		if(Pattern::MaxOff(y, m_paramsDBN.output) == Pattern::MaxOff(vtrPatts[k]->m_y, vtrPatts[k]->m_nYCnt))
			correct += 1; 
		error += Pattern::Error(y, vtrPatts[k]->m_y, m_paramsDBN.output);  	
		total += 1; 
	}

	delete y;

	double pr = (double)correct / (double)total;
	double rmse = sqrt(error / (double)total);

	return pair<double,double>(pr, rmse); 
}



