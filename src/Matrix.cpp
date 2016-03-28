#include "Matrix.h"
#include "Utility.h"
#include "StringArray.h"
using namespace fydl; 
#include <stdio.h>


//////////////////////////////////////////////////////////////////////////////////////////
// Construction & Destruction 

Matrix::Matrix() : m_nRows(0), m_nCols(0), m_data(NULL)
{
}


Matrix::Matrix(const int32_t nRows, const int32_t nCols) : m_nRows(nRows), m_nCols(nCols), m_data(NULL)
{
	m_data = new double*[nRows];
	for(int32_t i = 0; i < nRows; i++) 
		m_data[i] = new double[nCols]; 
}


Matrix::~Matrix()
{
	if(m_data)
	{
		for(int32_t i = 0; i < m_nRows; i++) 
			delete m_data[i]; 
		delete m_data; 
		m_data = NULL; 
	}
}


//////////////////////////////////////////////////////////////////////////////////////////
// Operations 

void Matrix::Create(const int32_t nRows, const int32_t nCols)
{
	if(m_data)
	{
		for(int32_t i = 0; i < m_nRows; i++) 
			delete m_data[i]; 
		delete m_data; 
		m_data = NULL; 
	}

	m_nRows = nRows; 
	m_nCols = nCols; 
	m_data = new double*[nRows];
	for(int32_t i = 0; i < nRows; i++) 
		m_data[i] = new double[nCols]; 
}


void Matrix::Init(const double dVal)
{
	for(int32_t i = 0; i < m_nRows; i++) 
	{
		for(int32_t j = 0; j < m_nCols; j++) 
			m_data[i][j] = dVal; 
	}
}


void Matrix::Init_RandUni(const double left, const double right)
{
	for(int32_t i = 0; i < m_nRows; i++) 
	{
		for(int32_t j = 0; j < m_nCols; j++) 
			m_data[i][j] = Utility::RandUni(left, right); 
	}
}


void Matrix::Init_RandNormal(const double mu, const double sigma)
{
	for(int32_t i = 0; i < m_nRows; i++) 
	{
		for(int32_t j = 0; j < m_nCols; j++) 
			m_data[i][j] = Utility::RandNormal(mu, sigma); 
	}
}


double* Matrix::operator [] (const int32_t nRow)
{
	if(nRow < 0 || nRow >= m_nRows)
		throw "Matrix::[] ERROR: index is out of bounds!"; 
	return m_data[nRow]; 
}


bool Matrix::IsNull()
{
	return (m_nRows == 0 || m_nCols == 0);  
}


int32_t Matrix::Rows()
{
	return m_nRows; 
}


int32_t Matrix::Cols()
{
	return m_nCols; 
}


void Matrix::Sparsification(const double dSpTh)
{
	for(int32_t i = 0; i < m_nRows; i++) 
	{
		for(int32_t j = 0; j < m_nCols; j++)
		{
			if(m_data[i][j] < dSpTh)
				m_data[i][j] = 0.0; 
		}
	}
}


void Matrix::Print_Matrix(ostream& os, Matrix& mat)
{
	for(int32_t i = 0; i < mat.Rows(); i++) 
	{
		for(int32_t j = 0; j < mat.Cols(); j++) 
		{
			if(j == 0)
				os<<mat[i][j];
			else
				os<<","<<mat[i][j];
		}
		os<<endl; 
	}
}


bool Matrix::Read_Matrix(Matrix& mat, istream& is)
{
	int32_t row = 0;
	string str; 
	while(row < mat.Rows())
	{
		std::getline(is, str); 
		StringArray ar(str.c_str(), ","); 
		if(ar.Count() != mat.Cols())
			return false;
		for(int32_t j = 0; j < mat.Cols(); j++) 
			sscanf(ar.GetString(j).c_str(), "%lf", &(mat[row][j]));  
		row++; 	
	}
	return true; 
}


