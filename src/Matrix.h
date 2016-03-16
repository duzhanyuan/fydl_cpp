#ifndef _FYDL_MATRIX_H 
#define _FYDL_MATRIX_H 

#include <string>
#include <vector>
using namespace std;
#include <stdint.h>
#include <sys/time.h>


namespace fydl
{

// CLASS
//	Matrix - definition of matrix 
//
class Matrix
{
public:
	Matrix(); 
	Matrix(const int32_t nRows, const int32_t nCols);
	virtual ~Matrix();

	// NAME
	//	Create - create the current matrix, allocate memory
	// 
	// DESCRIPTION
	//	nRows - number of rows
	//	nCols - numer of columes
	void Create(const int32_t nRows, const int32_t nCols); 

	// NAME
	//	Init - initialize the matrix, set elements as the same value 
	//	Init_RandUni - initialize the matrix, set elements based on uniform distribution 
	//	Init_RandNormal - initialize the matrix, set elements based on normal distribution
	// 
	// DESCRIPTION
	//	dVal: the initial value
	//	left, right: the minimal and maximal values of the uniform distribution
	//	mu, sigma - mean and standard deviation of the normal distribution
	void Init(const double dVal); 
	void Init_RandUni(const double left, const double right); 
	void Init_RandNormal(const double mu, const double sigma); 

	// Reload the subscrip
	double* operator [] (const int32_t row);

	// Determine whether the matrix is null
	bool IsNull(); 

	// Get the number of rows or columes 
	int32_t Rows(); 
	int32_t Cols(); 

	// Sparsificate the matrix
	void Sparsification(const double dSpTh = 0.000000000001); 

private: 
	int32_t m_nRows;	// number of rows 
	int32_t m_nCols;	// number of columes
	double** m_data;	// 2-d array, which is used for storing element values
};


}

#endif /* _FYDL_MATRIX_H */

