//=============================================================================
//	FILE:	cpd.h
//
//	DESC:	This file manages the CPD (change point detection) primitives.
//=============================================================================
#ifndef __CPD_CU__
#define __CPD_CU__

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
//	CPD Handle Class
//
//	This class stores the CPD description information.
//-----------------------------------------------------------------------------
template <class T>
class cpdHandle
{
	Memory<T>* m_pMem;
	Math<T>* m_pMath;
	long m_hT;
	long m_hD;
	long m_hWork;
	int m_nN;
	int m_nB;

public:
	
	cpdHandle()
	{
		m_pMem = NULL;
		m_pMath = NULL;
		m_hT = 0;
		m_hD = 0;
		m_hWork = 0;
		m_nN = 0;
		m_nB = 0;
	}

	long Initialize(Memory<T>* pMem, Math<T>* pMath)
	{
		m_pMem = pMem;
		m_pMath = pMath;
		return 0;
	}

	long CleanUp();
	long Set(int nN, int nB);
	long ComputeTvalueAt(int nT, int nTau, int nZ, long hZ, T* pfTVal);
	long ComputeSvalues(int nS, long hS, int nT, long hT);
};


//=============================================================================
//	Inline Methods
//=============================================================================


#endif