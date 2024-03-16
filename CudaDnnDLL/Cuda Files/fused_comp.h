//=============================================================================
//	FILE:	FUSEDCOMP.h
//
//	DESC:	This file manages the (FUSEDCOMP) fused computations.
//=============================================================================
#ifndef __FUSEDCOMP_CU__
#define __FUSEDCOMP_CU__

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

enum DataType
{
	DT_FLOAT = CUDNN_DATA_FLOAT,
	DT_DOUBLE = CUDNN_DATA_DOUBLE,
	DT_HALF = CUDNN_DATA_HALF
};

enum PreBuiltFusedComp
{
	PREBUILT_FUSED_COMP_NONE = 0,
	PREBUILT_FUSED_COMP_MATMUL = 1
};

enum FusedCompOp
{
	FUSED_COMP_OP_MATMUL = 1
};

enum HeurMode
{
	HEUR_MODE_NONE = -1,
	HEUR_MODE_INSTANT = CUDNN_HEUR_MODE_INSTANT,
	HEUR_MODE_A = CUDNN_HEUR_MODE_A,
	HEUR_MODE_FALLBACK = CUDNN_HEUR_MODE_FALLBACK,
	HEUR_MODE_B = CUDNN_HEUR_MODE_B
};

enum FusedCompSupport
{
	FUSED_COMP_SUPPORTED = 0,
	FUSED_COMP_NOT_SUPPORTED = 1
};

//=============================================================================
//	Classes
//=============================================================================

template <class T>
class FusedCompData;

template <class T>
class Memory;


//-----------------------------------------------------------------------------
//	FusedComp Handle Class
//
//	This class stores the FusedComp description information.
//-----------------------------------------------------------------------------
template <class T>
class fusedcompHandle
{
	int m_nRefCount = 0;
	Memory<T>* m_pMem;
	MemoryCollection* m_pMemCol;
	Math<T>* m_pMath;
	FusedCompData<T>* m_pData;
	bool m_bOwner;

public:
	
	fusedcompHandle()
	{
		m_pData = NULL;
		m_bOwner = true;
	}

	int RefCount()
	{
		return m_nRefCount;
	}

	void AddRef()
	{
		m_nRefCount++;
	}

	bool IsOwner()
	{
		return m_bOwner;
	}

	void SetOwner(bool bOwner)
	{
		m_bOwner = bOwner;
	}

	long Initialize(DataType dtIntermediate, DataType dtCompute, PreBuiltFusedComp preBuilt, long* phWokspace);
	long CleanUp();

	long AddTensor(long hSrcData, long nS1, long nS2, long nS3, long nS4, long* phTensorHandle);
	long GetTensor(long hDstData, long nS1, long nS2, long nS3, long nS4, long hTensorHandle);
	long AddOp(FusedCompOp nOp, long hTensor1, long hTensor2, long hTensor3, long hTensor4);
	long Build(HeurMode heur1, HeurMode heur2, long* phWorkspace);
	long Execute(long hWorkspace);
	long CheckSupport(FusedCompSupport* pSupport);
};


//=============================================================================
//	Inline Methods
//=============================================================================


#endif