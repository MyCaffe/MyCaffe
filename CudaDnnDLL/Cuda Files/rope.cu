//=============================================================================
//	FILE:	layernorm.cu
//
//	DESC:	This file implements the layer normalization (layernorm) algorithm
//=============================================================================

#include "util.h"
#include "rope.h"
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
class RopeData
{
	Memory<T>* m_pMem;
	Math<T>* m_pMath;
	T* m_pfFreqCos;
	T* m_pfFreqSin;
	T* m_pfXqr;
	T* m_pfXqi;
	T* m_pfXq_out_r;
	T* m_pfXq_out_i;
	T* m_pfX;
	T* m_pfY;
	int m_nCount;

public:
	int m_nGpuID;
	int m_nBatch;
	int m_nSeqLen;
	int m_nDim;
	T m_fTheta;

	RopeData(Memory<T>* pMem, Math<T>* pMath)
	{
		m_pMem = pMem;
		m_pMath = pMath;		
		m_nGpuID = 0;
		m_nBatch = 0;
		m_nSeqLen = 0;
		m_nDim = 0;
		m_fTheta = 0.0f;
		m_pfFreqCos = NULL;
		m_pfFreqSin = NULL;

		m_nCount = 0;
		m_pfX = NULL;
		m_pfY = NULL;
		m_pfXqr = NULL;
		m_pfXqi = NULL;
		m_pfXq_out_r = NULL;
		m_pfXq_out_i = NULL;
	}

	~RopeData()
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

	LONG Initialize(int nGpuID, int nCount, int nBatch, int nSeqLen, int nDim, T fTheta);
	void CleanUp();

	LONG allocHost(int nCount);

	LONG UpdateToHostMemory(int nCount, T* pfXdev);
	LONG UpdateToDevMemory(int nCount, T* pfYdev);

	LONG Forward(int n, long hXdata, long hYdata);
	LONG Backward(int n, long hYdata, long hYdiff, long hXdiff);
};


//=============================================================================
//	Class Methods - RopeData
//=============================================================================

template <class T>
LONG RopeData<T>::Initialize(int nGpuID, int nCount, int nBatch, int nSeqLen, int nDim, T fTheta)
{
	LONG lErr;

	CleanUp();

	m_nGpuID = nGpuID;
	m_nBatch = nBatch;
	m_nSeqLen = nSeqLen;
	m_nDim = nDim;
	m_fTheta = fTheta;

	nDim = m_nDim / 2;

	m_pfFreqCos = (T*)malloc(m_nSeqLen * nDim * sizeof(T));
	if (m_pfFreqCos == NULL)
		return ERROR_MEMORY_OUT;

	m_pfFreqSin = (T*)malloc(m_nSeqLen * nDim * sizeof(T));
	if (m_pfFreqSin == NULL)
	{
		CleanUp();
		return ERROR_MEMORY_OUT;
	}

	T* pfFreq = (T*)malloc(nDim * sizeof(T));
	if (pfFreq == NULL)
	{
		CleanUp();
		return ERROR_MEMORY_OUT;
	}

	int nIdx = 0;
	for (int i= 0; i < m_nDim; i+=2)
	{
		float fPower = (float)(i / (float)m_nDim);
		fPower = powf(fTheta, fPower);
		pfFreq[nIdx] = 1.0 / fPower;
		nIdx++;
	}

	for (int pos = 0; pos < m_nSeqLen; pos++)
	{
		for (int i = 0; i < nDim; i++)
		{
			T fPos = pfFreq[i] * pos;
			int nIdx = pos * nDim + i;
			m_pfFreqCos[nIdx] = cosf(fPos);
			m_pfFreqSin[nIdx] = sinf(fPos);
		}
	}

	free(pfFreq);
	pfFreq = NULL;

	if (lErr = allocHost(nCount))
	{
		CleanUp();
		return lErr;
	}

	return 0;
}

template LONG RopeData<double>::Initialize(int nGpuID, int nCount, int nBatch, int nSeqLen, int nDim, double fTheta);
template LONG RopeData<float>::Initialize(int nGpuID, int nCount, int nBatch, int nSeqLen, int nDim, float fTheta);


template <class T>
void RopeData<T>::CleanUp()
{
	if (m_pfFreqCos != NULL)
	{
		free(m_pfFreqCos);
		m_pfFreqCos = NULL;
	}

	if (m_pfFreqSin != NULL)
	{
		free(m_pfFreqSin);
		m_pfFreqSin = NULL;
	}

	if (m_pfXqr != NULL)
	{
		free(m_pfXqr);
		m_pfXqr = NULL;
	}

	if (m_pfXqi != NULL)
	{
		free(m_pfXqi);
		m_pfXqi = NULL;
	}

	if (m_pfXq_out_r != NULL)
	{
		free(m_pfXq_out_r);
		m_pfXq_out_r = NULL;
	}

	if (m_pfXq_out_i != NULL)
	{
		free(m_pfXq_out_i);
		m_pfXq_out_i = NULL;
	}

	if (m_pfX != NULL)
	{
		cudaFreeHost(m_pfX);
		m_pfX = NULL;
	}

	if (m_pfY != NULL)
	{
		cudaFreeHost(m_pfY);
		m_pfY = NULL;
	}
}

template void RopeData<double>::CleanUp();
template void RopeData<float>::CleanUp();

template <class T>
LONG RopeData<T>::allocHost(int nCount)
{
	if (nCount > m_nCount)
	{
		LONG lErr;

		if (m_pfX != NULL)
		{
			cudaFreeHost(m_pfX);
			m_pfX = NULL;
		}

		if (m_pfY != NULL)
		{
			cudaFreeHost(m_pfY);
			m_pfY = NULL;
		}

		if (lErr = cudaMallocHost(&m_pfX, nCount * sizeof(T)))
		{
			CleanUp();
			return ERROR_MEMORY_OUT;
		}

		if (lErr = cudaMallocHost(&m_pfY, nCount * sizeof(T)))
		{
			CleanUp();
			return ERROR_MEMORY_OUT;
		}

		if (m_pfXqr != NULL)
		{
			cudaFree(m_pfXqr);
			m_pfXqr = NULL;
		}

		if (m_pfXqi != NULL)
		{
			cudaFree(m_pfXqi);
			m_pfXqi = NULL;
		}

		if (m_pfXq_out_r != NULL)
		{
			cudaFree(m_pfXq_out_r);
			m_pfXq_out_r = NULL;
		}

		if (m_pfXq_out_i != NULL)
		{
			cudaFree(m_pfXq_out_i);
			m_pfXq_out_i = NULL;
		}

		int nSize = nCount / 2;
		m_pfXqr = (T*)malloc(nSize * sizeof(T));
		m_pfXqi = (T*)malloc(nSize * sizeof(T));
		m_pfXq_out_r = (T*)malloc(nSize * sizeof(T));
		m_pfXq_out_i = (T*)malloc(nSize * sizeof(T));

		if (m_pfXqr == NULL || m_pfXqi == NULL || m_pfXq_out_r == NULL || m_pfXq_out_i == NULL)
		{
			CleanUp();
			return ERROR_MEMORY_OUT;
		}

		m_nCount = nCount;
	}

	return 0;
}

template <class T>
LONG RopeData<T>::UpdateToHostMemory(int nCount, T* pfXdev)
{
	LONG lErr;

	if (lErr = allocHost(nCount))
		return lErr;

	if (lErr = cudaMemcpy(m_pfX, pfXdev, nCount * sizeof(T), cudaMemcpyDeviceToHost))
	{
		CleanUp();
		return lErr;
	}

	return 0;
}

template <class T>
LONG RopeData<T>::UpdateToDevMemory(int nCount, T* pfYdev)
{
	LONG lErr;

	if (lErr = cudaMemcpy(pfYdev, m_pfY, nCount * sizeof(T), cudaMemcpyHostToDevice))
	{
		CleanUp();
		return lErr;
	}

	return 0;
}

template <class T>
LONG RopeData<T>::Forward(int n, long hXdata, long hYdata)
{
	LONG lErr;
	MemoryItem* pX;
	MemoryItem* pY;
	MemoryCollection* pMemCol = m_pMem->GetMemoryCollection();

	if (lErr = pMemCol->GetData(hXdata, &pX))
		return lErr;

	if (lErr = pMemCol->GetData(hYdata, &pY))
		return lErr;

	if (lErr = UpdateToHostMemory(n, (T*)pX->Data()))
		return lErr;

	for (int i=0; i<n; i++)
	{
		if (i % 2 == 0)
			m_pfXqr[i / 2] = m_pfX[i];
		else
			m_pfXqi[i / 2] = m_pfX[i];
	}

	int nDim = m_nDim / 2;
	for (int i = 0; i < n / 2; i++)
	{
		int nIdx = (i / m_nDim) * nDim + i % nDim;
		float fCos = m_pfFreqCos[nIdx];
		float fSin = m_pfFreqSin[nIdx];

		m_pfXq_out_r[i] = m_pfXqr[i] * fCos - m_pfXqi[i] * fSin;
		m_pfXq_out_i[i] = m_pfXqr[i] * fSin + m_pfXqi[i] * fCos;
	}

	for (int i = 0; i < n; i++)
	{
		if (i % 2 == 0)
			m_pfY[i] = m_pfXq_out_r[i / 2];
		else
			m_pfY[i] = m_pfXq_out_i[i / 2];	
	}

	if (lErr = UpdateToDevMemory(n, (T*)pY->Data()))
		return lErr;

	return 0;
}

template LONG RopeData<double>::Forward(int n, long hXdata, long hYdata);
template LONG RopeData<float>::Forward(int n, long hXdata, long hYdata);


template <class T>
LONG RopeData<T>::Backward(int n, long hYdata, long hYdiff, long hXdiff)
{
	LONG lErr;

	return 0;
}

template LONG RopeData<double>::Backward(int n, long hYdata, long hYdiff, long hXdiff);
template LONG RopeData<float>::Backward(int n, long hYdata, long hYdiff, long hXdiff);


//=============================================================================
//	Class Methods - LayerNorm
//=============================================================================

template <class T>
long ropeHandle<T>::Update(Memory<T>* pMem, Math<T>* pMath)
{
	m_pMem = pMem;
	m_pMath = pMath;
	m_nRefCount++;

	m_pData = new RopeData<T>(pMem, pMath);
	if (m_pData == NULL)
		return ERROR_MEMORY_OUT;

	return 0;
}

template long ropeHandle<double>::Update(Memory<double>* pMem, Math<double>* pMath);
template long ropeHandle<float>::Update(Memory<float>* pMem, Math<float>* pMath);


template <class T>
long ropeHandle<T>::Initialize(int nGpuID, int nCount, int nBatch, int nSeqLen, int nDim, T fTheta)
{
	long lErr;

	if (lErr = m_pData->Initialize(nGpuID, nCount, nBatch, nSeqLen, nDim, fTheta))
	{
		m_pData->CleanUp();
		return lErr;
	}

	return 0;
}

template long ropeHandle<double>::Initialize(int nGpuID, int nCount, int nBatch, int nSeqLen, int nDim, double fTheta);
template long ropeHandle<float>::Initialize(int nGpuID, int nCount, int nBatch, int nSeqLen, int nDim, float fTheta);


template <class T>
long ropeHandle<T>::CleanUp()
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

template long ropeHandle<double>::CleanUp();
template long ropeHandle<float>::CleanUp();


template <class T>
long ropeHandle<T>::Forward(int n, long hXdata, long hYdata)
{
	return m_pData->Forward(n, hXdata, hYdata);
}

template long ropeHandle<double>::Forward(int n, long hXdata, long hYdata);
template long ropeHandle<float>::Forward(int n, long hXdata, long hYdata);


template <class T>
long ropeHandle<T>::Backward(int n, long hYdata, long hYdiff, long hXdiff)
{
	return m_pData->Backward(n, hYdata, hYdiff, hXdiff);
}

template long ropeHandle<double>::Backward(int n, long hYdata, long hYdiff, long hXdiff);
template long ropeHandle<float>::Backward(int n, long hYdata, long hYdiff, long hXdiff);


// end