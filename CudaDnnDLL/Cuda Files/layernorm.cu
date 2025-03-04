//=============================================================================
//	FILE:	layernorm.cu
//
//	DESC:	This file implements the layer normalization (layernorm) algorithm
//=============================================================================

#include "util.h"
#include "layernorm.h"
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
class LayerNormMemory
{
	Memory<T>* m_pMem;
	MemoryCollection* m_pMemCol;
	bool m_bOwnHandle;

public:
	int m_nCount;
	int m_nOuterNum;
	int m_nChannels;
	int m_nInnerNum;
	long m_hData;
	long m_hDiff;

	LayerNormMemory(Memory<T>* pMem)
	{
		m_pMem = pMem;
		m_pMemCol = pMem->GetMemoryCollection();
		m_bOwnHandle = false;
		m_nCount = 0;
		m_nOuterNum = 0;
		m_nChannels = 0;
		m_nInnerNum = 0;
		m_hData = 0;
		m_hDiff = 0;
	}

	~LayerNormMemory()
	{
		CleanUp();
	}

	LONG Initialize(int nGpuID, int nCount, int nOuterNum, int nChannels, int nInnerNum);
	void CleanUp();

	LONG save(std::string strName, bool bSaveDiff = false);

	int count()
	{
		return m_nCount;
	}
	
	int outer_num()
	{
		return m_nOuterNum;
	}

	int channels()
	{
		return m_nChannels;
	}

	int inner_num()
	{
		return m_nInnerNum;
	}

	long gpu_data()
	{
		return m_hData;
	}

	long gpu_diff()
	{
		return m_hDiff;
	}
};

template <class T>
class LayerNormData
{
	Memory<T>* m_pMem;
	Math<T>* m_pMath;

public:
	int m_nCount;
	int m_nOuterNum;
	int m_nChannels;
	int m_nInnerNum;
	T m_fEps;
	LayerNormMemory<T> m_mu;
	LayerNormMemory<T> m_xmu;
	LayerNormMemory<T> m_xmusq;
	LayerNormMemory<T> m_var;
	LayerNormMemory<T> m_stdev;
	LayerNormMemory<T> m_stdevfull;
	LayerNormMemory<T> m_work;

	LayerNormData(Memory<T>* pMem, Math<T>* pMath) : m_mu(pMem), m_xmu(pMem), m_xmusq(pMem), m_var(pMem), m_stdev(pMem), m_stdevfull(pMem), m_work(pMem)
	{
		m_pMem = pMem;
		m_pMath = pMath;		
		m_nCount = 0;
		m_nOuterNum = 0;
		m_nChannels = 0;
		m_nInnerNum = 0;
		m_fEps = (T)1e-10;
	}

	~LayerNormData()
	{
		CleanUp();
	}

	Memory<T>* GetMemory()
	{
		return m_pMem;
	}

	Math<T>* GetMath()
	{
		return m_pMath;
	}

	LONG Initialize(int nGpuID, int nCount, int nOuterNum, int nChannels, int nInnerNum, T fEps);
	void CleanUp();

	LONG Forward(long hXdata, long hYdata);
	LONG Backward(long hYdata, long hYdiff, long hXdiff);
};


//=============================================================================
//	Class Methods - LayerNormMemory
//=============================================================================

template <class T>
LONG LayerNormMemory<T>::Initialize(int nGpuID, int nCount, int nOuterNum, int nChannels, int nInnerNum)
{
	LONG lErr;

	if (nCount != nOuterNum * nChannels * nInnerNum)
		return ERROR_PARAM_OUT_OF_RANGE;

	if (m_hData != NULL || m_nCount != nCount)
		CleanUp();

	if (m_hData == NULL)
	{
		if (lErr = m_pMem->AllocMemory(nGpuID, false, nCount, NULL, 0, &m_hData))
			return lErr;
	}

	if (m_hDiff == NULL)
	{
		if (lErr = m_pMem->AllocMemory(nGpuID, false, nCount, NULL, 0, &m_hDiff))
			return lErr;
	}

	m_bOwnHandle = true;
		
	m_nCount = nCount;
	m_nOuterNum = nOuterNum;
	m_nChannels = nChannels;
	m_nInnerNum = nInnerNum;
	
	return 0;
}

template LONG LayerNormMemory<double>::Initialize(int nGpuID, int nCount, int nOuterNum, int nChannels, int nInnerNum);
template LONG LayerNormMemory<float>::Initialize(int nGpuID, int nCount, int nOuterNum, int nChannels, int nInnerNum);


template <class T>
void LayerNormMemory<T>::CleanUp()
{
	if (m_hData != NULL)
	{
		if (m_bOwnHandle)
			m_pMem->FreeMemory(m_hData);
		m_hData = 0;
	}

	if (m_hDiff != NULL)
	{
		if (m_bOwnHandle)
			m_pMem->FreeMemory(m_hDiff);
		m_hDiff = 0;
	}

	m_nCount = 0;
	m_nOuterNum = 0;
	m_nChannels = 0;
	m_nInnerNum = 0;
}

template void LayerNormMemory<double>::CleanUp();
template void LayerNormMemory<float>::CleanUp();


template <class T>
LONG LayerNormMemory<T>::save(std::string strName, bool bSaveDiff)
{
	std::string strFile = "c:\\temp\\snap\\" + strName + ".npy";
	long hData = (bSaveDiff) ? m_hDiff : m_hData;

	return m_pMem->SaveToNumpy(strFile, hData, m_nOuterNum, m_nChannels, m_nInnerNum, 1);
}

template LONG LayerNormMemory<double>::save(std::string strName, bool bSaveDiff);
template LONG LayerNormMemory<float>::save(std::string strName, bool bSaveDiff);


//=============================================================================
//	Class Methods - LayerNormData
//=============================================================================

template <class T>
LONG LayerNormData<T>::Initialize(int nGpuID, int nCount, int nOuterNum, int nChannels, int nInnerNum, T fEps)
{
	LONG lErr;

	if (lErr = m_mu.Initialize(nGpuID, nCount, nOuterNum, nChannels, nInnerNum))
		return lErr;

	if (lErr = m_xmu.Initialize(nGpuID, nCount, nOuterNum, nChannels, nInnerNum))
		return lErr;
	
	if (lErr = m_xmusq.Initialize(nGpuID, nCount, nOuterNum, nChannels, nInnerNum))
		return lErr;

	if (lErr = m_var.Initialize(nGpuID, nCount, nOuterNum, nChannels, nInnerNum))
		return lErr;

	if (lErr = m_stdev.Initialize(nGpuID, nCount, nOuterNum, nChannels, nInnerNum))
		return lErr;

	if (lErr = m_stdevfull.Initialize(nGpuID, nCount, nOuterNum, nChannels, nInnerNum))
		return lErr;
	
	if (lErr = m_work.Initialize(nGpuID, nCount, nOuterNum, nChannels, nInnerNum))
		return lErr;

	m_nCount = nCount;
	m_nOuterNum = nOuterNum;
	m_nChannels = nChannels;
	m_nInnerNum = nInnerNum;
	m_fEps = fEps;

	return 0;
}

template LONG LayerNormData<double>::Initialize(int nGpuID, int nCount, int nOuterNum, int nChannels, int nInnerNum, double dfEps);
template LONG LayerNormData<float>::Initialize(int nGpuID, int nCount, int nOuterNum, int nChannels, int nInnerNum, float fEps);


template <class T>
void LayerNormData<T>::CleanUp()
{
	m_mu.CleanUp();
	m_xmu.CleanUp();
	m_xmusq.CleanUp();
	m_var.CleanUp();
	m_stdev.CleanUp();
	m_stdevfull.CleanUp();
	m_work.CleanUp();
	
	m_nCount = 0;
	m_nOuterNum = 0;
	m_nChannels = 0;
	m_nInnerNum = 0;
}

template void LayerNormData<double>::CleanUp();
template void LayerNormData<float>::CleanUp();


template <class T>
LONG LayerNormData<T>::Forward(long hXdata, long hYdata)
{
	LONG lErr;
	
	// mean = x.mean(dim-1, keepdim=True)
	if (lErr = m_pMath->channel_mean(m_nCount, m_nOuterNum, m_nChannels, m_nInnerNum, hXdata, m_mu.gpu_data(), 0))
		return lErr;
	
	// Copy mean values across all items in the channel.
	if (lErr = m_pMath->channel_fillfrom(m_nCount, m_nOuterNum, m_nChannels, m_nInnerNum, m_mu.gpu_data(), m_xmu.gpu_data(), 0))
		return lErr;
	
	// xmu = x - mean
	if (lErr = m_pMath->sub(m_nCount, hXdata, m_xmu.gpu_data(), m_xmu.gpu_data()))
		return lErr;

	// xmusq = xmu**2
	if (lErr = m_pMath->powx(m_nCount, m_xmu.gpu_data(), (T)2.0, m_xmusq.gpu_data()))
		return lErr;

	// var = xmusq.mean(dim-1, keepdim=True)
	if (lErr = m_pMath->channel_mean(m_nCount, m_nOuterNum, m_nChannels, m_nInnerNum, m_xmusq.gpu_data(), m_var.gpu_data(), 0))
		return lErr;
		
	// stdev = sqrt(var + eps)
	if (lErr = m_pMath->scale(m_nOuterNum * m_nChannels, (T)1.0, m_var.gpu_data(), m_stdev.gpu_data()))
		return lErr;
	
	if (lErr = m_pMath->add_scalar(m_nOuterNum * m_nChannels, m_fEps, m_stdev.gpu_data()))
		return lErr;

	if (lErr = m_pMath->sqrt(m_nOuterNum * m_nChannels, m_stdev.gpu_data(), m_stdev.gpu_data()))
		return lErr;

	// y = (x - mean) / std
	if (lErr = m_pMath->channel_fillfrom(m_nCount, m_nOuterNum, m_nChannels, m_nInnerNum, m_stdev.gpu_data(), m_stdevfull.gpu_data(), 0))
		return lErr;

	if (lErr = m_pMath->div(m_nCount, m_xmu.gpu_data(), m_stdevfull.gpu_data(), hYdata))
		return lErr;

	return 0;
}

template LONG LayerNormData<double>::Forward(long hXdata, long hYdata);
template LONG LayerNormData<float>::Forward(long hXdata, long hYdata);


template <class T>
LONG LayerNormData<T>::Backward(long hYdata, long hYdiff, long hXdiff)
{
	LONG lErr;

	// Multiply previous dx by dy (grad) 
	// dx1 = dx * dy
	if (lErr = m_pMath->mul(m_nCount, hYdata, hYdiff, m_work.gpu_diff()))
		return lErr;

	// Average (dx * dy) across channel, dx1 = dx1.mean()
	if (lErr = m_pMath->channel_mean(m_nCount, m_nOuterNum, m_nChannels, m_nInnerNum, m_work.gpu_diff(), m_var.gpu_diff(), 0))
		return lErr;

	// Average dy across channel, dx2 = dy.mean()
	if (lErr = m_pMath->channel_mean(m_nCount, m_nOuterNum, m_nChannels, m_nInnerNum, hYdiff, m_stdev.gpu_diff(), 0))
		return lErr;

	// Multiply previous dx with dx1 (average across channel of dx * dy)
	if (lErr = m_pMath->channel_fillfrom(m_nCount, m_nOuterNum, m_nChannels, m_nInnerNum, m_var.gpu_diff(), m_stdevfull.gpu_diff(), 0))
		return lErr;

	if (lErr = m_pMath->mul(m_nCount, hYdata, m_stdevfull.gpu_diff(), m_work.gpu_diff()))
		return lErr;

	// Add in dy average dx2
	if (lErr = m_pMath->channel_fillfrom(m_nCount, m_nOuterNum, m_nChannels, m_nInnerNum, m_stdev.gpu_diff(), m_stdevfull.gpu_diff(), 0))
		return lErr;

	if (lErr = m_pMath->add(m_nCount, m_work.gpu_diff(), m_stdevfull.gpu_diff(), m_work.gpu_diff(), 1))
		return lErr;

	// Subtract from original dy gradient
	// dy - ((dx * dx1) + dx2)
	if (lErr = m_pMath->sub(m_nCount, hYdiff, m_work.gpu_diff(), m_work.gpu_diff()))
		return lErr;

	// Divide by the original stdev std, dx = (dy - ((dx * dx1) + dx2))/std
	if (lErr = m_pMath->add_scalar(m_nOuterNum * m_nChannels, m_fEps, m_stdevfull.gpu_data()))
		return lErr;

	if (lErr = m_pMath->div(m_nCount, m_work.gpu_diff(), m_stdevfull.gpu_data(), hXdiff))
		return lErr;

	return 0;
}

template LONG LayerNormData<double>::Backward(long hYdata, long hYdiff, long hXdiff);
template LONG LayerNormData<float>::Backward(long hYdata, long hYdiff, long hXdiff);


//=============================================================================
//	Class Methods - LayerNorm
//=============================================================================

template <class T>
long layernormHandle<T>::Update(Memory<T>* pMem, Math<T>* pMath)
{
	m_pMem = pMem;
	m_pMath = pMath;
	m_nRefCount++;

	m_pData = new LayerNormData<T>(pMem, pMath);
	if (m_pData == NULL)
		return ERROR_MEMORY_OUT;

	return 0;
}

template long layernormHandle<double>::Update(Memory<double>* pMem, Math<double>* pMath);
template long layernormHandle<float>::Update(Memory<float>* pMem, Math<float>* pMath);


template <class T>
long layernormHandle<T>::Initialize(int nGpuID, int nCount, int nOuterCount, int nChannels, int nInnerCount, T fEps)
{
	long lErr;

	if (lErr = m_pData->Initialize(nGpuID, nCount, nOuterCount, nChannels, nInnerCount, fEps))
	{
		m_pData->CleanUp();
		return lErr;
	}

	return 0;
}

template long layernormHandle<double>::Initialize(int nGpuID, int nCount, int nOuterCount, int nChannels, int nInnerCount, double dfEps);
template long layernormHandle<float>::Initialize(int nGupID, int nCount, int nOuterCount, int nChannels, int nInnerCount, float fEps);


template <class T>
long layernormHandle<T>::CleanUp()
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

template long layernormHandle<double>::CleanUp();
template long layernormHandle<float>::CleanUp();


template <class T>
long layernormHandle<T>::Forward(long hXdata, long hYdata)
{
	return m_pData->Forward(hXdata, hYdata);
}

template long layernormHandle<double>::Forward(long hXdata, long hYdata);
template long layernormHandle<float>::Forward(long hXdata, long hYdata);


template <class T>
long layernormHandle<T>::Backward(long hYdata, long hYdiff, long hXdiff)
{
	return m_pData->Backward(hYdata, hYdiff, hXdiff);
}

template long layernormHandle<double>::Backward(long hYdata, long hYdiff, long hXdiff);
template long layernormHandle<float>::Backward(long hYdata, long hYdiff, long hXdiff);


// end