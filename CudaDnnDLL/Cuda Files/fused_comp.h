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
	DT_FLOAT = 1,
	DT_DOUBLE = 2,
	DT_HALF = 3
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
	HEUR_MODE_A = 0,
	HEUR_MODE_B = 1,
	HEUR_MODE_FALLBACK = 2
};

enum FusedCompSupport
{
	FUSED_COMP_NOT_SUPPORTED = 0,
	FUSED_COMP_SUPPORTED = 1
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
	Math<T>* m_pMath;
	FusedCompData<T>* m_pData;
	bool m_bOwner;

public:
	
	fusedcompHandle()
	{
		m_pMem = NULL;
		m_pMath = NULL;
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

	long Update(Memory<T>* pMem, Math<T>* pMath);
	long Initialize(long hCuda, int nGpuID, DataType dtIo, DataType dtIntermediate, DataType dtCompute, PreBuiltFusedComp preBuilt, long* phWokspace);
	long CleanUp();

	long AddTensor(DataType dt, long nS1, long nS2, long nS3, long nS4, bool bTranspose, long* phTensorHandle, long* phTensorWorkspace);
	long GetTensor(long hTensorHandle, DataType* pdt, long* pnS1, long* pnS2, long* pnS3, long* pnS4, bool* pbTranspose);
	long AddOp(FusedCompOp nOp, DataType dtCompute, T fPadding, long hTensor1, long hTensor2, long hTensor3, long hTensor4, long* plIntermediateTensor);
	long Build(int nLocalID, HeurMode heur1, HeurMode heur2, long* phWorkspace);
	long Execute(int nLocalID, long hWorkspace, LONGLONG* rghTensor, LONGLONG* rghTensorData, LONGLONG* rghTensorWorkspaceData, long lCount);
};


//=============================================================================
//	Inline Methods
//=============================================================================


#endif