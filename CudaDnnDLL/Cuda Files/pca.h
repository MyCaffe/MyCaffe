//=============================================================================
//	FILE:	pca.h
//
//	DESC:	This file manages the PCA algorithm
//=============================================================================
#ifndef __PCA_CU__
#define __PCA_CU__

#include "util.h"
#include "math.h"

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
class pcaHandle
{
	Memory<T>* m_pMem;
	Math<T>* m_pMath;
	cublasHandle_t m_cublas;
	curandGenerator_t m_curand;
	int m_nNaxIterations;	// maximum number of iterations to run.
	int m_nCurrentIteration;// current iteration.
	int m_nCurrentK;		// current component K that Run iterates on.
	int m_nN;				// number of rows in Data
	int m_nD;				// number of columns in Data.
	int m_nK;				// number of components.
	long m_hData;			// MxN matrix where each row contains one instance of length 'N'
							// So for example when using 60,000 28x28 images of MNIST,
							// M = 60000, N = 784 or 28*28 and K = 50 means that we are looking
							// for the top 50 principal components in the data.
	long m_hScores;			// MxK matrix containing the PCA scores - eg. the result
	long m_hLoads;			// NxK matrix containing the PCA loads.
	long m_hResiduals;		// MxN matrix containing residual values.
							// Work = Result * Loads' + Residual
	bool m_bOwnResiduals;
	long m_hEigenvalues;	// Kx1 matrix of eigenvalues.
	T* m_pfEigenvalues;
	bool m_bOwnEigenvalues; 
	long m_hMeanCenter;		// Mx1 mean centering values.
	T m_fA;
	T* m_pfR;
	T* m_pfP;
	T* m_pfT;
	T* m_pfU;

public:
	
	pcaHandle(int nMaxIteration, int nM, int nN, int nK, long hData, long hScoresResult, long hLoadsResult, long hResiduals = 0, long hEigenvalues = 0)
	{
		m_pfR = NULL;
		m_pfP = NULL;
		m_pfT = NULL;
		m_pfU = NULL;
		m_pMem = NULL;
		m_pMath = NULL;
		m_cublas = NULL;
		m_curand = NULL;
		m_nNaxIterations = nMaxIteration;	
		m_nN = nM;
		m_nD = nN;
		m_nK = nK;
		m_hData = hData;
		m_hScores = hScoresResult;
		m_hLoads = hLoadsResult;
		m_hResiduals = hResiduals;
		m_hEigenvalues = hEigenvalues;
		m_pfEigenvalues = NULL;
		m_hMeanCenter = 0;
		m_nCurrentIteration = 0;
		m_nCurrentK = 0;
		m_fA = 0;
		m_bOwnResiduals = (hResiduals != 0) ? false : true;
		m_bOwnEigenvalues = (hEigenvalues != 0) ? false : true;
	}

	long Initialize(Memory<T>* pMem, Math<T>* pMath); // Allocates memory, pushes data to GPU.


	// Runs 'nStep' of iterations on the current component.
	//	When nCurrentIteration == MaxIteration and nCurrentK == m_nK, done = TRUE.
	long Run(int nSteps, bool* pbDone, int* pnCurrentIteration, int* pnCurrentK);	
	long CleanUp();			// Frees memory.
};


//=============================================================================
//	Inline Methods
//=============================================================================


#endif