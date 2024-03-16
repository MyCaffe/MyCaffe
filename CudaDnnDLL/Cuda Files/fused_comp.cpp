//=============================================================================
//	FILE:	fused_comp.cu
//
//	DESC:	This file implements fused computation functions.
//=============================================================================

#include <limits>
#include <cudnn_frontend.h>

#include "util.h"
#include "fused_comp.h"
#include "memory.h"
#include <string>
#include <iostream>
#include <fstream>


//=============================================================================
//	Function Definitions
//=============================================================================

//=============================================================================
//	Private Classes
//=============================================================================

template <class T>
class FusedCompData
{
protected:
	Memory<T>* m_pMem;
	Math<T>* m_pMath;

public:

	FusedCompData(Memory<T>* pMem, Math<T>* pMath)
	{
		m_pMem = pMem;
		m_pMath = pMath;
	}

	~FusedCompData()
	{
	}

	Memory<T>* GetMemory()
	{
		return m_pMem;
	}

	Math<T>* GetMath()
	{
		return m_pMath;
	}

	LONG Initialize(DataType dtIntermediate, DataType dtCompute, PreBuiltFusedComp preBuilt, long* phWokspace);
	void CleanUp();

	LONG AddTensor(long hSrcData, long nS1, long nS2, long nS3, long nS4, long* phTensorHandle);
	LONG GetTensor(long hDstData, long nS1, long nS2, long nS3, long nS4, long hTensorHandle);
	LONG AddOp(FusedCompOp nOp, long hTensor1, long hTensor2, long hTensor3, long hTensor4);
	LONG Build(HeurMode heur1, HeurMode heur2, long* phWorkspace);
	LONG Execute(long hWorkspace);
	LONG CheckSupport(FusedCompSupport* pSupport);
};

//=============================================================================
//	Class Methods - FusedCompData
//=============================================================================

template <class T>
long FusedCompData<T>::Initialize(DataType dtIntermediate, DataType dtCompute, PreBuiltFusedComp preBuilt, long* phWokspace)
{
	return 0;
}

template long FusedCompData<double>::Initialize(DataType dtIntermediate, DataType dtCompute, PreBuiltFusedComp preBuilt, long* phWokspace);
template long FusedCompData<float>::Initialize(DataType dtIntermediate, DataType dtCompute, PreBuiltFusedComp preBuilt, long* phWokspace);


template <class T>
void FusedCompData<T>::CleanUp()
{
}

template void FusedCompData<double>::CleanUp();
template void FusedCompData<float>::CleanUp();


template <class T>
long FusedCompData<T>::AddTensor(long hSrcData, long nS1, long nS2, long nS3, long nS4, long* phTensorHandle)
{
	return 0;
}

template long FusedCompData<double>::AddTensor(long hSrcData, long nS1, long nS2, long nS3, long nS4, long* phTensorHandle);
template long FusedCompData<float>::AddTensor(long hSrcData, long nS1, long nS2, long nS3, long nS4, long* phTensorHandle);


template <class T>
long FusedCompData<T>::GetTensor(long hDstData, long nS1, long nS2, long nS3, long nS4, long hTensorHandle)
{
	return 0;
}

template long FusedCompData<double>::GetTensor(long hDstData, long nS1, long nS2, long nS3, long nS4, long hTensorHandle);
template long FusedCompData<float>::GetTensor(long hDstData, long nS1, long nS2, long nS3, long nS4, long hTensorHandle);


template <class T>
long FusedCompData<T>::AddOp(FusedCompOp nOp, long hTensor1, long hTensor2, long hTensor3, long hTensor4)
{
	return 0;
}

template long FusedCompData<double>::AddOp(FusedCompOp nOp, long hTensor1, long hTensor2, long hTensor3, long hTensor4);
template long FusedCompData<float>::AddOp(FusedCompOp nOp, long hTensor1, long hTensor2, long hTensor3, long hTensor4);


template <class T>
long FusedCompData<T>::Build(HeurMode heur1, HeurMode heur2, long* phWorkspace)
{
	return 0;
}

template long FusedCompData<double>::Build(HeurMode heur1, HeurMode heur2, long* phWorkspace);
template long FusedCompData<float>::Build(HeurMode heur1, HeurMode heur2, long* phWorkspace);


template <class T>
long FusedCompData<T>::Execute(long hWorkspace)
{
	return 0;
}

template long FusedCompData<double>::Execute(long hWorkspace);
template long FusedCompData<float>::Execute(long hWorkspace);


template <class T>
long FusedCompData<T>::CheckSupport(FusedCompSupport* pSupport)
{
	return 0;
}

template long FusedCompData<double>::CheckSupport(FusedCompSupport* pSupport);
template long FusedCompData<float>::CheckSupport(FusedCompSupport* pSupport);


//=============================================================================
//	Class Methods - LayerNorm
//=============================================================================

template <class T>
long fusedcompHandle<T>::Initialize(DataType dtIntermediate, DataType dtCompute, PreBuiltFusedComp preBuilt, long* phWorkspace)
{
	long lErr;

	if (lErr = m_pData->Initialize(dtIntermediate, dtCompute, preBuilt, phWorkspace))
	{
		m_pData->CleanUp();
		return lErr;
	}

	return 0;
}

template long fusedcompHandle<double>::Initialize(DataType dtIntermediate, DataType dtCompute, PreBuiltFusedComp preBuilt, long* phWokspace);
template long fusedcompHandle<float>::Initialize(DataType dtIntermediate, DataType dtCompute, PreBuiltFusedComp preBuilt, long* phWokspace);


template <class T>
long fusedcompHandle<T>::CleanUp()
{
	m_nRefCount--;

	if (m_nRefCount == 0)
	{
		if (m_pData != NULL)
		{
			m_pData->CleanUp();
			delete m_pData;
			m_pData = NULL;
		}
	}

	return 0;
}

template long fusedcompHandle<double>::CleanUp();
template long fusedcompHandle<float>::CleanUp();


template <class T>
long fusedcompHandle<T>::AddTensor(long hSrcData, long nS1, long nS2, long nS3, long nS4, long* phTensorHandle)
{
	return m_pData->AddTensor(hSrcData, nS1, nS2, nS3, nS4, phTensorHandle);
}

template long fusedcompHandle<double>::AddTensor(long hSrcData, long nS1, long nS2, long nS3, long nS4, long* phTensorHandle);
template long fusedcompHandle<float>::AddTensor(long hSrcData, long nS1, long nS2, long nS3, long nS4, long* phTensorHandle);


template <class T>
long fusedcompHandle<T>::GetTensor(long hDstData, long nS1, long nS2, long nS3, long nS4, long hTensorHandle)
{
	return m_pData->GetTensor(hDstData, nS1, nS2, nS3, nS4, hTensorHandle);
}

template long fusedcompHandle<double>::GetTensor(long hDstData, long nS1, long nS2, long nS3, long nS4, long hTensorHandle);
template long fusedcompHandle<float>::GetTensor(long hDstData, long nS1, long nS2, long nS3, long nS4, long hTensorHandle);


template <class T>
long fusedcompHandle<T>::AddOp(FusedCompOp nOp, long hTensor1, long hTensor2, long hTensor3, long hTensor4)
{
	return m_pData->AddOp(nOp, hTensor1, hTensor2, hTensor3, hTensor4);
}

template long fusedcompHandle<double>::AddOp(FusedCompOp nOp, long hTensor1, long hTensor2, long hTensor3, long hTensor4);
template long fusedcompHandle<float>::AddOp(FusedCompOp nOp, long hTensor1, long hTensor2, long hTensor3, long hTensor4);

template <class T>
long fusedcompHandle<T>::Build(HeurMode heur1, HeurMode heur2, long* phWorkspace)
{
	return m_pData->Build(heur1, heur2, phWorkspace);
}

template long fusedcompHandle<double>::Build(HeurMode heur1, HeurMode heur2, long* phWokspace);
template long fusedcompHandle<float>::Build(HeurMode heur1, HeurMode heur2, long* phWokspace);

template <class T>
long fusedcompHandle<T>::Execute(long hWorkspace)
{
	return m_pData->Execute(hWorkspace);
}

template long fusedcompHandle<double>::Execute(long hWorkspace);
template long fusedcompHandle<float>::Execute(long hWorkspace);

template <class T>
long fusedcompHandle<T>::CheckSupport(FusedCompSupport* pSupport)
{
	return m_pData->CheckSupport(pSupport);
}

template long fusedcompHandle<double>::CheckSupport(FusedCompSupport* pSupport);
template long fusedcompHandle<float>::CheckSupport(FusedCompSupport* pSupport);

// end