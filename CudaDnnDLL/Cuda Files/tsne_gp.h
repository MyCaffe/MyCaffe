//=============================================================================
//	FILE:	tsne_gp.h
//
//	DESC:	This file manages the TSNE gaussian perplexity calcuation
//=============================================================================
#ifndef __TSNE_GP_CU__
#define __TSNE_GP_CU__

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

template <class T>
class VpTree;

template <class T>
class DataPoint;


//-----------------------------------------------------------------------------
//	PCA Handle Class
//
//	This class stores the PCA description information.
//-----------------------------------------------------------------------------
template <class T>
class tsnegpHandle
{
	VpTree<T>* m_pTree;
	std::vector<DataPoint<T>*> m_rgObjX;
	Memory<T>* m_pMem;
	Math<T>* m_pMath;
	cublasHandle_t m_cublas;
	curandGenerator_t m_curand;
	int m_nMaxIterations;	// maximum number of iterations to run.
	int m_nCurrentIteration;// current iteration.
	unsigned int m_nN;				// number of rows in Data
	unsigned int m_nD;				// number of columns in Data.
	unsigned int m_nK;				
	long m_hX;
	long m_hCurP;
	long m_hValP;
	long m_hRowPonhost;
	long m_hColPonhost;
	T* m_pRowP;
	T* m_pColP;
	T* m_pX;
	T* m_pValP;
	T* m_pCurP;
	T m_fPerplexity;
	T m_fMax;
	T m_fMin;

public:
	
	tsnegpHandle(unsigned int nN, unsigned int nD, unsigned int nK, long hX, long hCurP, long hValP, long hRowPonhost, long hColPonhost, T fPerplexity)
		: m_rgObjX(nN, NULL)
	{
		m_pTree = NULL;
		m_pRowP = NULL;
		m_pColP = NULL;
		m_pX = NULL;
		m_pValP = NULL;
		m_pCurP = NULL;

		m_pMem = NULL;
		m_pMath = NULL;
		m_cublas = NULL;
		m_curand = NULL;
		m_nMaxIterations = nN;	
		m_nN = nN;
		m_nD = nD;
		m_nK = nK;
		m_hX = hX;
		m_hCurP = hCurP;
		m_hValP = hValP;
		m_hRowPonhost = hRowPonhost;
		m_hColPonhost = hColPonhost;
		m_fPerplexity = fPerplexity;
		m_fMax = (sizeof(T) == 4) ? FLT_MAX : DBL_MAX;
		m_fMin = (sizeof(T) == 4) ? FLT_MIN : DBL_MIN;
	}

	// Allocates memory, pushes data to GPU.
	long Initialize(Memory<T>* pMem, Math<T>* pMath); 

	//	When nCurrentIteration == MaxIteration, done = TRUE.
	long Run(bool* pbDone, int* pnCurrentIteration, int* pnMaxIteration);	

	// Frees memory.
	long CleanUp();	
};


//=============================================================================
//	Inline Methods
//=============================================================================


#endif