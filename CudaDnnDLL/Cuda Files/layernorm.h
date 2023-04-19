//=============================================================================
//	FILE:	layernorm.h
//
//	DESC:	This file manages the layer normalization (layernorm) algorithm
//=============================================================================
#ifndef __LAYERNORM_CU__
#define __LAYERNORM_CU__

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
class LayerNormData;

template <class T>
class Memory;

using std::vector;
using std::map;
using std::tuple;
using std::set;

//-----------------------------------------------------------------------------
//	LAYERNORM Handle Class
//
//	This class stores the LAYERNORM description information.
//-----------------------------------------------------------------------------
template <class T>
class layernormHandle
{
	int m_nRefCount = 0;
	Memory<T>* m_pMem;
	MemoryCollection* m_pMemCol;
	Math<T>* m_pMath;
	LayerNormData<T>* m_pData;
	bool m_bOwner;

public:
	
	layernormHandle()
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

	long Initialize(int nGpuID, int nCount, int nOuterCount, int nChannels, int nInnerCount, T fEps);

	long CleanUp();

	long Forward(long hXdata, long hYdata);
	long Backward(long hYdata, long hYdiff, long hXdiff);
};


//=============================================================================
//	Inline Methods
//=============================================================================


#endif