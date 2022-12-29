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
	LONG Backward(long hYdiff, long hXdiff);
};


//=============================================================================
//	Class Methods - LayerNormMemory
//=============================================================================

template <class T>
LONG LayerNormMemory<T>::Initialize(int nGpuID, int nCount, int nOuterNum, int nChannels, int nInnerNum)
{
	LONG lErr;
	T* pSrc = NULL;

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
	LONG lErr;
	std::ofstream wf("c:\\temp\\snap\\" + strName + ".npy", std::ios::out | std::ios::binary);
	if (!wf)
		return ERROR_FILE_NOT_FOUND;

	long hData = (bSaveDiff) ? m_hDiff : m_hData;
	int nOuterNum = m_nOuterNum;
	int nChannels = m_nChannels;
	int nInnerNum = m_nInnerNum;

	size_t lCount;
	T* pData = m_pMem->GetMemoryToHost(hData, &lCount);
	if (pData == NULL)
		return ERROR_MEMORY_NOT_FOUND;

	byte hdr[8];
	hdr[0] = (byte)0x93;
	hdr[1] = (byte)0x4E; // N
	hdr[2] = (byte)0x55; // U
	hdr[3] = (byte)0x4D; // M
	hdr[4] = (byte)0x50; // P
	hdr[5] = (byte)0x59; // Y
	hdr[6] = (byte)0x01;
	hdr[7] = (byte)0x00;
	wf.write((const char*)&hdr[0], 8);

	std::string strHeader = "{'descr': '<f4', 'fortran_order': False, 'shape': (";
	strHeader += std::to_string(nOuterNum);
	strHeader += ",";
	strHeader += std::to_string(nChannels);
	strHeader += ",";
	strHeader += std::to_string(nInnerNum);
	strHeader += ")";

	if (strHeader.length() < 117)
		strHeader += std::string(117 - strHeader.length(), ' ');
	strHeader += "\n";

	byte bLen = (byte)strHeader.length();
	wf.write((const char*)&bLen, 1);
	byte bVal = (byte)0x00;
	wf.write((const char*)&bVal, 1);

	for (int i = 0; i < strHeader.length(); i++)
	{
		bVal = (byte)strHeader[i];
		wf.write((const char*)&bVal, 1);
	}

	for (size_t lIdx = 0; lIdx < lCount; lIdx++)
	{
		float fVal = (float)pData[lIdx];
		wf.write((const char*)&fVal, 4);
	}

	m_pMem->FreeHost(pData);

	wf.close();
	if (!wf.good())
		return ERROR_FILE_CORRUPT;

	return 0;
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
	if (lErr = m_pMath->channel_sum(m_nCount, m_nOuterNum, m_nChannels, m_nInnerNum, hXdata, m_mu.gpu_data(), false))
		return lErr;

	if (lErr = m_pMath->scal(m_nCount, (T)1.0 / m_nInnerNum, m_mu.gpu_data()))
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
	if (lErr = m_pMath->channel_sum(m_nCount, m_nOuterNum, m_nChannels, m_nInnerNum, m_xmusq.gpu_data(), m_var.gpu_data(), false))
		return lErr;

	if (lErr = m_pMath->scal(m_nCount, (T)1.0 / m_nInnerNum, m_var.gpu_data()))
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
LONG LayerNormData<T>::Backward(long hYdiff, long hXdiff)
{
	LONG lErr;

	// y = (x - mean) / std
	// xmu' = y' / std
	if (lErr = m_pMath->div(m_nCount, hYdiff, m_stdevfull.gpu_data(), m_xmu.gpu_diff()))
		return lErr;
	
	// std' = y' * -xmu / std^2
	if (lErr = m_pMath->powx(m_nCount, m_stdevfull.gpu_data(), (T)2.0, m_stdevfull.gpu_diff()))
		return lErr;

	if (lErr = m_pMath->div(m_nCount, m_xmu.gpu_data(), m_stdevfull.gpu_diff(), m_work.gpu_diff()))
		return lErr;

	if (lErr = m_pMath->scal(m_nCount, -1, m_work.gpu_diff()))
		return lErr;

	if (lErr = m_pMath->mul(m_nCount, hYdiff, m_work.gpu_diff(), m_stdevfull.gpu_diff()))
		return lErr;

	// std' = channel_sum(stdfull')
	if (lErr = m_pMath->channel_sum(m_nCount, m_nOuterNum, m_nChannels, m_nInnerNum, m_stdevfull.gpu_diff(), m_stdev.gpu_diff(), false))
		return lErr;

	// var' = std' * 0.5 * std^-1
	if (lErr = m_pMath->set(m_nCount, m_work.gpu_diff(), -1, 0))
		return lErr;

	if (lErr = m_pMath->powx(m_nOuterNum * m_nChannels, m_stdev.gpu_data(), (T)-1.0, m_work.gpu_diff()))
		return lErr;

	if (lErr = m_pMath->scal(m_nOuterNum * m_nChannels, (T)0.5, m_work.gpu_diff()))
		return lErr;

	if (lErr = m_pMath->mul(m_nOuterNum * m_nChannels, m_stdev.gpu_diff(), m_work.gpu_diff(), m_var.gpu_diff()))
		return lErr;

	// xmusq' = 1 / n * var'
	if (lErr = m_pMath->channel_fillfrom(m_nCount, m_nOuterNum, m_nChannels, m_nInnerNum, m_var.gpu_diff(), m_xmusq.gpu_diff(), 0))
		return lErr;

	if (lErr = m_pMath->scal(m_nCount, (T)1.0 / m_nInnerNum, m_xmusq.gpu_diff()))
		return lErr;

	// xmu' = 2 * xmu * xmusq' + previous xmu' (xmu' = y' / std)
	if (lErr = m_pMath->mul(m_nCount, m_xmu.gpu_data(), m_xmusq.gpu_diff(), m_work.gpu_diff()))
		return lErr;

	if (lErr = m_pMath->scal(m_nCount, (T)2.0, m_work.gpu_diff()))
		return lErr;

	if (lErr = m_pMath->add(m_nCount, m_xmu.gpu_diff(), m_work.gpu_diff(), m_xmu.gpu_diff(), (T)1.0))
		return lErr;

	// x' = xmu'
	// mean' = -channel_sum(xmu')
	if (lErr = m_pMath->channel_sum(m_nCount, m_nOuterNum, m_nChannels, m_nInnerNum, m_xmu.gpu_diff(), m_mu.gpu_diff(), false))
		return lErr;

	if (lErr = m_pMath->scal(m_nOuterNum * m_nChannels, -1, m_mu.gpu_diff()))
		return lErr;

	// x' = 1 / n * mean' + x'
	if (lErr = m_pMath->channel_fillfrom(m_nCount, m_nOuterNum, m_nChannels, m_nInnerNum, m_mu.gpu_diff(), m_work.gpu_diff(), 0))
		return lErr;

	if (lErr = m_pMath->scal(m_nCount, (T)1.0 / m_nInnerNum, m_work.gpu_diff()))
		return lErr;

	if (lErr = m_pMath->add(m_nCount, m_xmu.gpu_diff(), m_work.gpu_diff(), hXdiff, (T)1.0))
		return lErr;

	return 0;
}

template LONG LayerNormData<double>::Backward(long hYdiff, long hXdiff);
template LONG LayerNormData<float>::Backward(long hYdiff, long hXdiff);


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
long layernormHandle<T>::Backward(long hYdiff, long hXdiff)
{
	return m_pData->Backward(hYdiff, hXdiff);
}

template long layernormHandle<double>::Backward(long hYdiff, long hXdiff);
template long layernormHandle<float>::Backward(long hYdiff, long hXdiff);


// end