//=============================================================================
//	FILE:	tsne_g.h
//
//	DESC:	This file manages the TSNE gradient calcuation
//=============================================================================
#ifndef __TSNE_G_CU__
#define __TSNE_G_CU__

#include "util.h"
#include "math.h"
#include <vector>


//=============================================================================
//	Flags
//=============================================================================

//=============================================================================
//	Classes
//=============================================================================

template <class T>
class Memory;


//-----------------------------------------------------------------------------
//	PCA Handle Class
//
//	This class stores the PCA description information.
//-----------------------------------------------------------------------------
template <class T>
class tsnegHandle
{
	Memory<T>* m_pMem;
	Math<T>* m_pMath;
	cublasHandle_t m_cublas;
	curandGenerator_t m_curand;
	int m_nCurrentIteration;
	int m_nMaxIterations;
	unsigned int m_nN;				// number of rows in Data
	unsigned int m_nD;				// number of columns in Data.
	long m_hY;
	long m_hValP;
	long m_hdC;
	long m_hRowP;
	long m_hColP;

	T* m_pY_on_host;
	T* m_pValP_on_host;
	T* m_pPosF_on_host;
	T* m_pNegF_on_host;
	T* m_pRowP_on_host;
	T* m_pColP_on_host;
	T* m_pBuff_on_host;
	T* m_pdC_on_host;
	T  m_fTheta;
	T m_fMax;
	T m_fMin;

	long computeGradient(T* rowP, T* colP, T* valP, T* Y, unsigned int N, unsigned int D, T* dC, T fTheta);
	long evaluateError(T* rowP, T* colP, T* valP, T* Y, unsigned int N, unsigned int D, T fTheta, T* pfErr);
	long symmetrizeMatrix(T* row_P, T* col_P, T* val_P, unsigned int* pnRowCount);

public:	
	tsnegHandle(unsigned int nN, unsigned int nD, long hY, long hValP, long hRowP, long hColP, long hdC, T fTheta)
	{
		m_nN = nN;
		m_nD = nD;
		m_hY = hY;
		m_hValP = hValP;
		m_hRowP = hRowP;
		m_hColP = hColP;
		m_hdC = hdC;
		m_fTheta = fTheta;

		m_pY_on_host = NULL;
		m_pValP_on_host = NULL;
		m_pRowP_on_host = NULL;
		m_pColP_on_host = NULL;
		m_pPosF_on_host = NULL;
		m_pNegF_on_host = NULL;
		m_pBuff_on_host = NULL;
		m_pdC_on_host = NULL;

		m_pMem = NULL;
		m_pMath = NULL;
		m_cublas = NULL;
		m_curand = NULL;
		m_nCurrentIteration = 0;
		m_nMaxIterations = nN;	

		m_fMax = (sizeof(T) == 4) ? FLT_MAX : DBL_MAX;
		m_fMin = (sizeof(T) == 4) ? FLT_MIN : DBL_MIN;
	}

	// Allocates memory, pushes data to GPU.
	long Initialize(Memory<T>* pMem, Math<T>* pMath); 

	//	When nCurrentIteration == MaxIteration, done = TRUE.
	long SymmetrizeMatrix(unsigned int* pnRowCount);

	long ComputeGradient(bool bValPUpdated);
	long EvaluateError(T* pfErr);

	// Frees memory.
	long CleanUp();	
};


//=============================================================================
//	Inline Methods
//=============================================================================


#endif