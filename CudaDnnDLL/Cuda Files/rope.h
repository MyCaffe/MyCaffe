//=============================================================================
//	FILE:	ROPE.h
//
//	DESC:	This file manages the (ROPE) positional encoding algorithm
//=============================================================================
#ifndef __ROPE_CU__
#define __ROPE_CU__

#include "util.h"
#include "math.h"
#include "memorycol.h"
#include "handlecol.h"
#include <vector>
#include <map>
#include <tuple>
#include <set>


//=============================================================================
//	Types
//=============================================================================

//=============================================================================
//	Classes
//=============================================================================

template <class T>
class RopeData;

template <class T>
class Memory;

using std::vector;
using std::map;
using std::tuple;
using std::set;

//-----------------------------------------------------------------------------
//	Rope Handle Class
//
//	This class stores the Rope description information.
//-----------------------------------------------------------------------------
template <class T>
class ropeHandle
{
	int m_nRefCount = 0;
	Memory<T>* m_pMem;
	MemoryCollection* m_pMemCol;
	Math<T>* m_pMath;
	RopeData<T>* m_pData;
	bool m_bOwner;

public:
	
	ropeHandle()
	{
		m_pData = NULL;
		m_bOwner = true;
	}

	int RefCount()
	{
		return m_nRefCount;
	}

	bool IsOwner()
	{
		return m_bOwner;
	}

	void SetOwner(bool bOwner)
	{
		m_bOwner = bOwner;
	}

	long Update(Memory<T>* pMem, Math<T>* pMath);

	long Initialize(int nGpuID, int nCount, int nBatch, int nSeqLen, int nHeads, int nDim, T fTheta);
	long CleanUp();

	long Forward(int n, long hXdata, long hYdata);
	long Backward(int n, long hXdata, long hYdiff, long hXdiff);
};


//=============================================================================
//	Inline Methods
//=============================================================================


#endif