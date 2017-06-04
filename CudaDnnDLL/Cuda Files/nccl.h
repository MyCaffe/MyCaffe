//=============================================================================
//	FILE:	nccl.h
//
//	DESC:	This file manages the nccl based multi-gpu communication
//=============================================================================
#ifndef __NCCL_CU__
#define __NCCL_CU__

#include "util.h"
#include "math.h"
#include "memorycol.h"
#include "handlecol.h"


//=============================================================================
//	Types
//=============================================================================

typedef enum {
	NCCL_SUM = 0,
	NCCL_PROD = 1,
	NCCL_MAX = 2,
	NCCL_MIN = 3
} NCCL_OP;


//=============================================================================
//	Classes
//=============================================================================

class Data;

template <class T>
class Memory;


//-----------------------------------------------------------------------------
//	NCCL Handle Class
//
//	This class stores the NCCL description information.
//-----------------------------------------------------------------------------
template <class T>
class ncclHandle
{
	int m_nRefCount = 0;
	int m_nGpuID;
	Memory<T>* m_pMem;
	MemoryCollection* m_pMemCol;
	Math<T>* m_pMath;
	Data* m_pData;
	bool m_bOwner;

	long isDisplayConnectedToGpu(int nGpuID, bool* pbIsDisplayOn);
	void setBufferSize(long lBufferCount);

public:
	
	ncclHandle()
	{
		m_pData = NULL;
		m_bOwner = true;
	}

	int GpuID()
	{
		return m_nGpuID;
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

	long Initialize(Memory<T>* pMem, Math<T>* pMath, int nGpuID, int nCount, int nRank, char* szId);
	long Update(Memory<T>* pMem, Math<T>* pMath);
	long CleanUp();

	LPCSTR GetErrorString(long lErr);

	long InitSingleProcess(long lBufferCount, int nCount, ncclHandle<T>* rgHandles[]);
	long InitMultiProcess(long lBufferCount);
	long Broadcast(long hStream, long hX, int nCount);
	long AllReduce(long hStream, long hX, int nCount, NCCL_OP op, T fScale);
};


//=============================================================================
//	Inline Methods
//=============================================================================


#endif