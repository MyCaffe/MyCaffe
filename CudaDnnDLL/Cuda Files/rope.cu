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
protected:
	Memory<T>* m_pMem;
	Math<T>* m_pMath;

	T* m_pfFreqCos;
	T* m_pfFreqSin;
	T* m_pfXr;
	T* m_pfXi;
	T* m_pfX_out_r;
	T* m_pfX_out_i;
	int m_nCount;

public:
	int m_nGpuID;
	int m_nBatch;
	int m_nSeqLen;
	int m_nHeads;
	int m_nDim;
	T m_fTheta;

	RopeData(Memory<T>* pMem, Math<T>* pMath)
	{
		m_pMem = pMem;
		m_pMath = pMath;
		m_nGpuID = 0;
		m_nBatch = 0;
		m_nSeqLen = 0;
		m_nHeads = 0;
		m_nDim = 0;
		m_fTheta = 0.0f;
		m_pfFreqCos = NULL;
		m_pfFreqSin = NULL;

		m_nCount = 0;
		m_pfXr = NULL;
		m_pfXi = NULL;
		m_pfX_out_r = NULL;
		m_pfX_out_i = NULL;
	}

	~RopeData()
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

	virtual LONG Initialize(int nGpuID, int nCount, int nBatch, int nSeqLen, int nHeads, int nDim, T fTheta) = 0;
	virtual void CleanUp() = 0;

	virtual LONG Forward(int n, long hXdata, long hYdata, int nFreqOffset) = 0;
	virtual LONG Backward(int n, long hXdata, long hYdiff, long hXdiff) = 0;
};

template <class T>
class RopeDataCpu : public RopeData<T>
{
	T* m_pfX;
	T* m_pfY;

	LONG allocHost(int nCount);

	LONG updateToHostMemory(int nCount, T* pfXdev);
	LONG updateToDevMemory(int nCount, T* pfYdev);

public:
	RopeDataCpu(Memory<T>* pMem, Math<T>* pMath) : RopeData<T>(pMem, pMath)
	{
		m_pfX = NULL;
		m_pfY = NULL;
	}

	~RopeDataCpu()
	{
		CleanUp();
	}

	LONG Initialize(int nGpuID, int nCount, int nBatch, int nSeqLen, int nHeads, int nDim, T fTheta);
	void CleanUp();

	LONG Forward(int n, long hXdata, long hYdata, int nFreqOffset);
	LONG Backward(int n, long hXdata, long hYdiff, long hXdiff);
};

template <class T>
class RopeDataGpu : public RopeData<T>
{
	LONG allocGpu(int nCount);

public:

	RopeDataGpu(Memory<T>* pMem, Math<T>* pMath) : RopeData<T>(pMem, pMath)
	{
	}

	~RopeDataGpu()
	{
		CleanUp();
	}

	LONG Initialize(int nGpuID, int nCount, int nBatch, int nSeqLen, int nHeads, int nDim, T fTheta);
	void CleanUp();

	LONG Forward(int n, long hXdata, long hYdata, int nFreqOffset);
	LONG Backward(int n, long hXdata, long hYdiff, long hXdiff);
};

//=============================================================================
//	Class Methods - RopeDataCpu
//=============================================================================

template <class T>
LONG RopeDataCpu<T>::Initialize(int nGpuID, int nCount, int nBatch, int nSeqLen, int nHeads, int nDim, T fTheta)
{
	LONG lErr;

	CleanUp();

	m_nGpuID = nGpuID;
	m_nBatch = nBatch;
	m_nSeqLen = nSeqLen;
	m_nHeads = nHeads;
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
		T fPower = (T)(i / (T)m_nDim);
		fPower = (T)powf(fTheta, fPower);
		pfFreq[nIdx] = T(1.0) / fPower;
		nIdx++;
	}

	for (int pos = 0; pos < m_nSeqLen; pos++)
	{
		for (int i = 0; i < nDim; i++)
		{
			T fPos = pfFreq[i] * pos;
			int nIdx = pos * nDim + i;
			m_pfFreqCos[nIdx] = (T)cosf(fPos);
			m_pfFreqSin[nIdx] = (T)sinf(fPos);
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

template LONG RopeDataCpu<double>::Initialize(int nGpuID, int nCount, int nBatch, int nSeqLen, int nHeads, int nDim, double fTheta);
template LONG RopeDataCpu<float>::Initialize(int nGpuID, int nCount, int nBatch, int nSeqLen, int nHeads, int nDim, float fTheta);


template <class T>
void RopeDataCpu<T>::CleanUp()
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

	if (m_pfXr != NULL)
	{
		free(m_pfXr);
		m_pfXr = NULL;
	}

	if (m_pfXi != NULL)
	{
		free(m_pfXi);
		m_pfXi = NULL;
	}

	if (m_pfX_out_r != NULL)
	{
		free(m_pfX_out_r);
		m_pfX_out_r = NULL;
	}

	if (m_pfX_out_i != NULL)
	{
		free(m_pfX_out_i);
		m_pfX_out_i = NULL;
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

template void RopeDataCpu<double>::CleanUp();
template void RopeDataCpu<float>::CleanUp();

template <class T>
LONG RopeDataCpu<T>::allocHost(int nCount)
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

		if (m_pfXr != NULL)
		{
			cudaFree(m_pfXr);
			m_pfXr = NULL;
		}

		if (m_pfXi != NULL)
		{
			cudaFree(m_pfXi);
			m_pfXi = NULL;
		}

		if (m_pfX_out_r != NULL)
		{
			cudaFree(m_pfX_out_r);
			m_pfX_out_r = NULL;
		}

		if (m_pfX_out_i != NULL)
		{
			cudaFree(m_pfX_out_i);
			m_pfX_out_i = NULL;
		}

		int nSize = nCount / 2;
		m_pfXr = (T*)malloc(nSize * sizeof(T));
		m_pfXi = (T*)malloc(nSize * sizeof(T));
		m_pfX_out_r = (T*)malloc(nSize * sizeof(T));
		m_pfX_out_i = (T*)malloc(nSize * sizeof(T));

		if (m_pfXr == NULL || m_pfXi == NULL || m_pfX_out_r == NULL || m_pfX_out_i == NULL)
		{
			CleanUp();
			return ERROR_MEMORY_OUT;
		}

		m_nCount = nCount;
	}

	return 0;
}

template <class T>
LONG RopeDataCpu<T>::updateToHostMemory(int nCount, T* pfXdev)
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
LONG RopeDataCpu<T>::updateToDevMemory(int nCount, T* pfYdev)
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
LONG RopeDataCpu<T>::Forward(int n, long hXdata, long hYdata, int nFreqOffset)
{
	LONG lErr;
	MemoryItem* pX;
	MemoryItem* pY;
	MemoryCollection* pMemCol = m_pMem->GetMemoryCollection();

	if (lErr = pMemCol->GetData(hXdata, &pX))
		return lErr;

	if (lErr = pMemCol->GetData(hYdata, &pY))
		return lErr;

	if (lErr = updateToHostMemory(n, (T*)pX->Data()))
		return lErr;

	for (int i = 0; i < n; i++)
	{
		if (i % 2 == 0)
			m_pfXr[i / 2] = m_pfX[i];
		else
			m_pfXi[i / 2] = m_pfX[i];
	}

	T* pfFreqCos = m_pfFreqCos;
	if (nFreqOffset > 0)
		pfFreqCos += nFreqOffset;

	T* pfFreqSin = m_pfFreqSin;
	if (nFreqOffset > 0)
		pfFreqSin += nFreqOffset;

	int nDim = m_nDim / 2;
	int nFullDim = nDim * m_nHeads;
	for (int i = 0; i < n / 2; i++)
	{
		int nIdx1 = ((i / nFullDim) % nFullDim) * nDim;
		int nIdx2 = (i % nDim);
		int nIdx = nIdx1 + nIdx2;

		T fCos = pfFreqCos[nIdx];
		T fSin = pfFreqSin[nIdx];

		m_pfX_out_r[i] = m_pfXr[i] * fCos - m_pfXi[i] * fSin;
		m_pfX_out_i[i] = m_pfXr[i] * fSin + m_pfXi[i] * fCos;
	}

	for (int i = 0; i < n; i++)
	{
		if (i % 2 == 0)
			m_pfY[i] = m_pfX_out_r[i / 2];
		else
			m_pfY[i] = m_pfX_out_i[i / 2];	
	}

	if (lErr = updateToDevMemory(n, (T*)pY->Data()))
		return lErr;

	return 0;
}

template LONG RopeDataCpu<double>::Forward(int n, long hXdata, long hYdata, int nFreqOffset);
template LONG RopeDataCpu<float>::Forward(int n, long hXdata, long hYdata, int nFreqOffset);


template <class T>
LONG RopeDataCpu<T>::Backward(int n, long hXdata, long hYdiff, long hXdiff)
{
	LONG lErr;
	MemoryItem* pX;
	MemoryItem* pY;
	MemoryCollection* pMemCol = m_pMem->GetMemoryCollection();

	if (lErr = pMemCol->GetData(hXdiff, &pX))
		return lErr;

	if (lErr = pMemCol->GetData(hYdiff, &pY))
		return lErr;

	if (lErr = updateToHostMemory(n, (T*)pY->Data())) // y grads
		return lErr;

	for (int i = 0; i < n; i++)
	{
		if (i % 2 == 0)
			m_pfXr[i / 2] = m_pfX[i]; // y grads
		else
			m_pfXi[i / 2] = m_pfX[i]; // y grads
	}

	int nDim = m_nDim / 2;
	int nFullDim = nDim * m_nHeads;
	for (int i = 0; i < n / 2; i++)
	{
		int nIdx1 = ((i / nFullDim) % nFullDim) * nDim;
		int nIdx2 = (i % nDim);
		int nIdx = nIdx1 + nIdx2;

		T fCos = m_pfFreqCos[nIdx];
		T fSin = m_pfFreqSin[nIdx];

		m_pfX_out_r[i] = m_pfXr[i] * fCos + m_pfXi[i] * fSin; // y grads
		m_pfX_out_i[i] = -m_pfXr[i] * fSin + m_pfXi[i] * fCos; // y grads
	}

	for (int i = 0; i < n; i++)
	{
		if (i % 2 == 0)
			m_pfY[i] = m_pfX_out_r[i / 2]; // x grads
		else
			m_pfY[i] = m_pfX_out_i[i / 2]; // x grads
	}

	if (lErr = updateToDevMemory(n, (T*)pX->Data()))
		return lErr;

	return 0;
}

template LONG RopeDataCpu<double>::Backward(int n, long hXdata, long hYdiff, long hXdiff);
template LONG RopeDataCpu<float>::Backward(int n, long hXdata, long hYdiff, long hXdiff);

//=============================================================================
//	Class Methods - RopeDataCpu
//=============================================================================

template <class T>
LONG RopeDataGpu<T>::Initialize(int nGpuID, int nCount, int nBatch, int nSeqLen, int nHeads, int nDim, T fTheta)
{
	LONG lErr;

	CleanUp();

	m_nGpuID = nGpuID;
	m_nBatch = nBatch;
	m_nSeqLen = nSeqLen;
	m_nHeads = nHeads;
	m_nDim = nDim;
	m_fTheta = fTheta;

	nDim = m_nDim / 2;

	T* pfFreqCos = (T*)malloc(m_nSeqLen * nDim * sizeof(T));
	if (pfFreqCos == NULL)
		return ERROR_MEMORY_OUT;

	T* pfFreqSin = (T*)malloc(m_nSeqLen * nDim * sizeof(T));
	if (pfFreqSin == NULL)
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
	for (int i = 0; i < m_nDim; i += 2)
	{
		T fPower = (T)(i / (T)m_nDim);
		fPower = (T)powf(fTheta, fPower);
		pfFreq[nIdx] = T(1.0) / fPower;
		nIdx++;
	}

	for (int pos = 0; pos < m_nSeqLen; pos++)
	{
		for (int i = 0; i < nDim; i++)
		{
			T fPos = pfFreq[i] * pos;
			int nIdx = pos * nDim + i;
			pfFreqCos[nIdx] = (T)cosf(fPos);
			pfFreqSin[nIdx] = (T)sinf(fPos);
		}
	}

	if (lErr = cudaMalloc((void**)&m_pfFreqCos, m_nSeqLen * nDim * sizeof(T)))
	{
		free(pfFreq);
		free(pfFreqCos);
		free(pfFreqSin);
		CleanUp();
		return lErr;
	}

	if (lErr = cudaMemcpy(m_pfFreqCos, pfFreqCos, m_nSeqLen * nDim * sizeof(T), cudaMemcpyHostToDevice))
	{
		free(pfFreq);
		free(pfFreqCos);
		free(pfFreqSin);
		CleanUp();
		return lErr;
	}

	if (lErr = cudaMalloc((void**)&m_pfFreqSin, m_nSeqLen * nDim * sizeof(T)))
	{
		free(pfFreq);
		free(pfFreqCos);
		free(pfFreqSin);
		CleanUp();
		return lErr;
	}

	if (lErr = cudaMemcpy(m_pfFreqSin, pfFreqSin, m_nSeqLen * nDim * sizeof(T), cudaMemcpyHostToDevice))
	{
		free(pfFreq);
		free(pfFreqCos);
		free(pfFreqSin);
		CleanUp();
		return lErr;
	}

	free(pfFreq);
	pfFreq = NULL;

	free(pfFreqCos);
	pfFreqCos = NULL;

	free(pfFreqSin);
	pfFreqSin = NULL;

	if (lErr = allocGpu(nCount))
	{
		CleanUp();
		return lErr;
	}

	return 0;
}

template LONG RopeDataGpu<double>::Initialize(int nGpuID, int nCount, int nBatch, int nSeqLen, int nHeads, int nDim, double fTheta);
template LONG RopeDataGpu<float>::Initialize(int nGpuID, int nCount, int nBatch, int nSeqLen, int nHeads, int nDim, float fTheta);


template <class T>
void RopeDataGpu<T>::CleanUp()
{
	if (m_pfFreqCos != NULL)
	{
		cudaFree(m_pfFreqCos);
		m_pfFreqCos = NULL;
	}

	if (m_pfFreqSin != NULL)
	{
		cudaFree(m_pfFreqSin);
		m_pfFreqSin = NULL;
	}

	if (m_pfXr != NULL)
	{
		cudaFree(m_pfXr);
		m_pfXr = NULL;
	}

	if (m_pfXi != NULL)
	{
		cudaFree(m_pfXi);
		m_pfXi = NULL;
	}

	if (m_pfX_out_r != NULL)
	{
		cudaFree(m_pfX_out_r);
		m_pfX_out_r = NULL;
	}

	if (m_pfX_out_i != NULL)
	{
		cudaFree(m_pfX_out_i);
		m_pfX_out_i = NULL;
	}
}

template void RopeDataGpu<double>::CleanUp();
template void RopeDataGpu<float>::CleanUp();

template <class T>
LONG RopeDataGpu<T>::allocGpu(int nCount)
{
	if (nCount > m_nCount)
	{
		LONG lErr;

		if (m_pfXr != NULL)
		{
			cudaFree(m_pfXr);
			m_pfXr = NULL;
		}

		if (m_pfXi != NULL)
		{
			cudaFree(m_pfXi);
			m_pfXi = NULL;
		}

		if (m_pfX_out_r != NULL)
		{
			cudaFree(m_pfX_out_r);
			m_pfX_out_r = NULL;
		}

		if (m_pfX_out_i != NULL)
		{
			cudaFree(m_pfX_out_i);
			m_pfX_out_i = NULL;
		}

		int nSize = nCount / 2;
		if (lErr = cudaMalloc(&m_pfXr, nSize * sizeof(T)))
		{
			CleanUp();
			return ERROR_MEMORY_OUT;
		}

		if (lErr = cudaMalloc(&m_pfXi, nSize * sizeof(T)))
		{
			CleanUp();
			return ERROR_MEMORY_OUT;
		}

		if (lErr = cudaMalloc(&m_pfX_out_r, nSize * sizeof(T)))
		{
			CleanUp();
			return ERROR_MEMORY_OUT;
		}

		if (lErr = cudaMalloc(&m_pfX_out_i, nSize * sizeof(T)))
		{
			CleanUp();
			return ERROR_MEMORY_OUT;
		}

		if (m_pfXr == NULL || m_pfXi == NULL || m_pfX_out_r == NULL || m_pfX_out_i == NULL)
		{
			CleanUp();
			return ERROR_MEMORY_OUT;
		}

		m_nCount = nCount;
	}

	return 0;
}

template <typename T>
__global__ void copy_xToRI_kernel(const int n, const T* x, T* xr, T* xi)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n && i >= 0; i += blockDim.x * gridDim.x)
	{
		if (i % 2 == 0)
			xr[i / 2] = x[i];
		else
			xi[i / 2] = x[i];
	}
}

template <typename T>
__global__ void copy_computeRI_kernel(const int n, const int nDim, const int nFullDim, const T* xr, const T* xi, const T* xcos, const T* xsin, T* xro, T* xio)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n && i >= 0; i += blockDim.x * gridDim.x)
	{
		const int nIdx1 = ((i / nFullDim) % nFullDim) * nDim;
		const int nIdx2 = (i % nDim);
		const int nIdx = nIdx1 + nIdx2;

		const T fCos = xcos[nIdx];
		const T fSin = xsin[nIdx];

		xro[i] = xr[i] * fCos - xi[i] * fSin;
		xio[i] = xr[i] * fSin + xi[i] * fCos;
	}
}

template <typename T>
__global__ void copy_computeRI_grad_kernel(const int n, const int nDim, const int nFullDim, const T* xr, const T* xi, const T* xcos, const T* xsin, T* xro, T* xio)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n && i >= 0; i += blockDim.x * gridDim.x)
	{
		const int nIdx1 = ((i / nFullDim) % nFullDim) * nDim;
		const int nIdx2 = (i % nDim);
		const int nIdx = nIdx1 + nIdx2;

		const T fCos = xcos[nIdx];
		const T fSin = xsin[nIdx];

		xro[i] = xr[i] * fCos + xi[i] * fSin;
		xio[i] = -xr[i] * fSin + xi[i] * fCos;
	}
}

template <typename T>
__global__ void copy_riToY_kernel(const int n, const T* xr, const T* xi, T* y)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n && i >= 0; i += blockDim.x * gridDim.x)
	{
		if (i % 2 == 0)
			y[i] = xr[i / 2];
		else
			y[i] = xi[i / 2];
	}
}

template <class T>
LONG RopeDataGpu<T>::Forward(int n, long hXdata, long hYdata, int nFreqOffset)
{
	LONG lErr;
	MemoryItem* pX;
	MemoryItem* pY;
	MemoryCollection* pMemCol = m_pMem->GetMemoryCollection();

	if (lErr = pMemCol->GetData(hXdata, &pX))
		return lErr;

	if (lErr = pMemCol->GetData(hYdata, &pY))
		return lErr;

	T* pfX = (T*)pX->Data();
	T* pfY = (T*)pY->Data();

	copy_xToRI_kernel<T> << <CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS >> > (n, pfX, m_pfXr, m_pfXi);
	if (lErr = cudaStreamSynchronize(0))
		return lErr;

	int nDim = m_nDim / 2;
	int nFullDim = nDim * m_nHeads;

	T* pfFreqCos = m_pfFreqCos;
	if (nFreqOffset > 0)
		pfFreqCos += nFreqOffset;

	T* pfFreqSin = m_pfFreqSin;
	if (nFreqOffset > 0)
		pfFreqSin += nFreqOffset;

	copy_computeRI_kernel<T> << <CAFFE_GET_BLOCKS(n/2), CAFFE_CUDA_NUM_THREADS >> > (n/2, nDim, nFullDim, m_pfXr, m_pfXi, pfFreqCos, pfFreqSin, m_pfX_out_r, m_pfX_out_i);
	if (lErr = cudaStreamSynchronize(0))
		return lErr;

	copy_riToY_kernel<T> << <CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS >> > (n, m_pfX_out_r, m_pfX_out_i, pfY);
	if (lErr = cudaStreamSynchronize(0))
		return lErr;

	return 0;
}

template LONG RopeDataGpu<double>::Forward(int n, long hXdata, long hYdata, int nFreqOffset);
template LONG RopeDataGpu<float>::Forward(int n, long hXdata, long hYdata, int nFreqOffset);


template <class T>
LONG RopeDataGpu<T>::Backward(int n, long hXdata, long hYdiff, long hXdiff)
{
	LONG lErr;
	MemoryItem* pX;
	MemoryItem* pY;
	MemoryCollection* pMemCol = m_pMem->GetMemoryCollection();

	if (lErr = pMemCol->GetData(hXdiff, &pX))
		return lErr;

	if (lErr = pMemCol->GetData(hYdiff, &pY))
		return lErr;

	T* pfXdiff = (T*)pX->Data();
	T* pfYdiff = (T*)pY->Data();

	copy_xToRI_kernel<T> << <CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS >> > (n, pfYdiff, m_pfXr, m_pfXi);
	if (lErr = cudaStreamSynchronize(0))
		return lErr;

	int nDim = m_nDim / 2;
	int nFullDim = nDim * m_nHeads;

	copy_computeRI_grad_kernel<T> << <CAFFE_GET_BLOCKS(n / 2), CAFFE_CUDA_NUM_THREADS >> > (n / 2, nDim, nFullDim, m_pfXr, m_pfXi, m_pfFreqCos, m_pfFreqSin, m_pfX_out_r, m_pfX_out_i);
	if (lErr = cudaStreamSynchronize(0))
		return lErr;

	copy_riToY_kernel<T> << <CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS >> > (n, m_pfX_out_r, m_pfX_out_i, pfXdiff);
	if (lErr = cudaStreamSynchronize(0))
		return lErr;

	return 0;
}

template LONG RopeDataGpu<double>::Backward(int n, long hXdata, long hYdiff, long hXdiff);
template LONG RopeDataGpu<float>::Backward(int n, long hXdata, long hYdiff, long hXdiff);


//=============================================================================
//	Class Methods - LayerNorm
//=============================================================================

template <class T>
long ropeHandle<T>::Update(int nGpuID, Memory<T>* pMem, Math<T>* pMath)
{
	m_pMem = pMem;
	m_pMath = pMath;
	m_nRefCount++;

	if (nGpuID < 0)
		m_pData = new RopeDataCpu<T>(pMem, pMath);
	else
		m_pData = new RopeDataGpu<T>(pMem, pMath);

	if (m_pData == NULL)
		return ERROR_MEMORY_OUT;

	return 0;
}

template long ropeHandle<double>::Update(int nGpuID, Memory<double>* pMem, Math<double>* pMath);
template long ropeHandle<float>::Update(int nGpuID, Memory<float>* pMem, Math<float>* pMath);


template <class T>
long ropeHandle<T>::Initialize(int nGpuID, int nCount, int nBatch, int nSeqLen, int nHeads, int nDim, T fTheta)
{
	long lErr;

	if (lErr = m_pData->Initialize(nGpuID, nCount, nBatch, nSeqLen, nHeads, nDim, fTheta))
	{
		m_pData->CleanUp();
		return lErr;
	}

	return 0;
}

template long ropeHandle<double>::Initialize(int nGpuID, int nCount, int nBatch, int nSeqLen, int nHeads, int nDim, double fTheta);
template long ropeHandle<float>::Initialize(int nGpuID, int nCount, int nBatch, int nSeqLen, int nHeads, int nDim, float fTheta);


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
long ropeHandle<T>::Forward(int n, long hXdata, long hYdata, int nFreqOffset)
{
	return m_pData->Forward(n, hXdata, hYdata, nFreqOffset);
}

template long ropeHandle<double>::Forward(int n, long hXdata, long hYdata, int nFreqOffset);
template long ropeHandle<float>::Forward(int n, long hXdata, long hYdata, int nFreqOffset);


template <class T>
long ropeHandle<T>::Backward(int n, long hXdata, long hYdiff, long hXdiff)
{
	return m_pData->Backward(n, hXdata, hYdiff, hXdiff);
}

template long ropeHandle<double>::Backward(int n, long hXdata, long hYdiff, long hXdiff);
template long ropeHandle<float>::Backward(int n, long hXdata, long hYdiff, long hXdiff);


// end