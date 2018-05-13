//=============================================================================
//	FILE:	memory.h
//
//	DESC:	This file manages the GPU memory
//=============================================================================
#ifndef __MEMORY_CU__
#define __MEMORY_CU__

#include "util.h"
#include "handlecol.h"
#include "memorycol.h"
#include "memtest.h"
#include "pca.h"
#include "tsne_gp.h"
#include "tsne_g.h"
#include "nccl.h"
#include <vector>
#include <algorithm>


//=============================================================================
//	Flags
//=============================================================================

// Uncomment to use cudaMallocHost/cudaFreeHost (when commented out malloc/free are used)
#define USE_PINNED_HOST_MEM 1

//-----------------------------------------------------------------------------
//	 Modes
//-----------------------------------------------------------------------------
enum PoolingMethod
{
	MAX = CUDNN_POOLING_MAX,
	AVERAGE = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
};

enum ActivationMethod
{
	SIGMOID = CUDNN_ACTIVATION_SIGMOID,
	RELU = CUDNN_ACTIVATION_RELU,
	TANH = CUDNN_ACTIVATION_TANH
};



//=============================================================================
//	Classes
//=============================================================================

template <class T>
class HostBuffer
{
	private:
		long m_lCount;
		T* m_pMemory;

	public:
		HostBuffer(T* pMem, long lCount)
		{
			m_pMemory = pMem;
			m_lCount = lCount;
		}

		T* Data()
		{
			return m_pMemory;
		}

		long Count()
		{
			return m_lCount;
		}
};


//-----------------------------------------------------------------------------
//	Memory Class
//
//	The memory class implements manages underying memory used on the GPU device.
//-----------------------------------------------------------------------------
template <class T>
class Memory
{
	protected:
		std::vector<HostBuffer<T>*> m_rgActiveHostBuffers;
		MemoryCollection m_memory;
		MemoryCollection m_memoryPointers;
		HandleCollection<MAX_HANDLES> m_hostbuffers;
		HandleCollection<MAX_HANDLES> m_streams;
		HandleCollection<MAX_HANDLES> m_tensorDesc;
		HandleCollection<MAX_HANDLES> m_filterDesc;
		HandleCollection<MAX_HANDLES> m_convDesc;
		HandleCollection<MAX_HANDLES> m_poolDesc;
		HandleCollection<MAX_HANDLES> m_lrnDesc;
		HandleCollection<MAX_HANDLES> m_cudnn;
		HandleCollection<MIN_HANDLES> m_pca;
		HandleCollection<MIN_HANDLES> m_tsnegp;
		HandleCollection<MIN_HANDLES> m_tsneg;
		HandleCollection<MIN_HANDLES> m_memtest;
		HandleCollection<MIN_HANDLES> m_nccl;
		T m_tOne;
		T m_tZero;
#ifdef CUDNN_5
		HandleCollection<MAX_HANDLES> m_dropoutDesc;
		HandleCollection<MAX_HANDLES> m_activationDesc;
		long m_hGlobalActivationSigmoid;
		long m_hGlobalActivationRelu;
		long m_hGlobalActivationTanh;
#endif

	public:
		Memory();
		~Memory();

		MemoryCollection* GetMemoryCollection()
		{
			return &m_memory;
		}

		HandleCollection<MAX_HANDLES>* GetStreamCollection()
		{
			return &m_streams;
		}

		long CheckMemoryAttributes(long hSrc, int nSrcDeviceID, long hDst, int nDstDeviceID, bool* pbResult);
		long GetDeviceMemory(int nDeviceID, T* plTotal, T* plFree, T* plUsed, bool* pbEstimate);

		long AllocMemory(int nDeviceID, long lCount, T* pSrc, long hStream, long* phHandle);
		long FreeMemory(long hHandle);
		long GetMemory(long hHandle, MemoryItem** ppItem);
		long SetMemory(long hHandle, T* pSrc, long lCount, long hStream);
		long SetMemoryAt(long hHandle, T* pSrc, long lCount, int nOffset);

		T* GetMemoryToHost(long hHandle, long* plCount = NULL);
		long SetMemoryToHost(long hHandle, T* pDst);

		long AllocHostBuffer(long lCount, long* phHandle);
		long FreeHostBuffer(long hHandle);
		HostBuffer<T>* GetHostBuffer(long hHandle);
		long SetHostBuffer(long hHandle, long lCount, T* pData);
		bool IsHostBuffer(T* pf);

		long CopyToHost(long lCount, T* pDst, T* pSrc, bool bSrcOnDevice);
		long AllocHost(long lCount, T** ppDst, T* pSrc, bool bSrcOnDevice);
		long FreeHost(T* pDst);

		long AllocHost(LPTSTR* ppDst, LPTSTR pSrc);
		long FreeHost(LPTSTR pDst);

		long CreateMemoryPointer(long hData, long lOffset, long lCount, long* phHandle);
		long FreeMemoryPointer(long hData);

		long CreateStream(long* phHandle, bool bNonBlocking = false);
		long FreeStream(long hHandle);
		cudaStream_t GetStream(long hHandle);
		long SynchronizeStream(long hHandle);
		long SynchronizeThread();	

		long CreateCuDNN(long hStream, long* phHandle);
		long FreeCuDNN(long hHandle);
		cudnnHandle_t GetCuDNN(long hHandle);

		long CreateTensorDesc(long* phHandle);
		long FreeTensorDesc(long hHandle);
		cudnnTensorDescriptor_t GetTensorDesc(long hHandle);
		long SetTensorDesc(long hHandle, int n, int c, int h, int w, int stride_n, int stride_c, int stride_h, int stride_w);
		long AddTensor(long hHandle, T fAlpha, long hSrcDesc, long hSrc, int nSrcOffset, T fBeta, long hDstDesc, long hDst, int nDstOffset);

		long CreateFilterDesc(long* phHandle);
		long FreeFilterDesc(long hHandle);
		cudnnFilterDescriptor_t GetFilterDesc(long hHandle);
		long SetFilterDesc(long hHandle, int n, int c, int h, int w);

		long CreateConvolutionDesc(long* phHandle);
		long FreeConvolutionDesc(long hHandle);
		cudnnConvolutionDescriptor_t GetConvolutionDesc(long hHandle);
		long SetConvolutionDesc(long hHandle, int pad_h, int pad_w, int stride_h, int stride_w);
	    long GetConvolutionInfo(long hHandle, long hBottomDesc, long hFilterDesc, long hConvDesc, long hTopDesc, long lWsLimitInBytes, long* palgoFwd, long* plWsSizeFwd, long* palgoBwdFilter, long* plWsSizeBwdFilter, long* palgoBwdData, long* plWsSizeBwdData, int nPreferredFwdAlgo = -1);
		long ConvolutionForward(long hHandle, T fAlpha, long hBottomDesc, long hBottomData, int nBottomOffset, long hFilterDesc, long hWeight, int nWeightOffset, long hConvDesc, long algo, long hWorkspace, int nWorkspaceOffset, long lWorkspaceSize, T fBeta, long hTopDesc, long hTopData, int nTopOffset, bool bSyncStream);
		long ConvolutionBackwardBias(long hHandle, T fAlpha, long hTopDesc, long hTopDiff, int nTopOffset, T fBeta, long hBiasDesc, long hBiasDiff, int nBiasOffset, bool bSyncStream);
		long ConvolutionBackwardFilter(long hHandle, T fAlpha, long hBottomDesc, long hBottomData, int nBottomOffset, long hTopDesc, long hTopDiff, int nTopOffset, long hConvDesc, long algo, long hWorkspace, int nWorkspaceOffset, long lWorkspaceSiz, T fBeta, long hFilterDesc, long hWeightDiff, int nWeightOffsete, bool bSyncStream);
		long ConvolutionBackwardData(long hHandle, T fAlpha, long hFilterDesc, long hWeight, int nWeightOffset, long hTopDesc, long hTopDiff, int nTopOffset, long hConvDesc, long algo, long hWorkspace, int nWorkspaceOffset, long lWorkspaceSize, T fBeta, long hBottomDesc, long hBottomDiff, int nBottomOffset, bool bSyncStream);

		long CreatePoolingDesc(long* phHandle);
		long FreePoolingDesc(long hHandle);
		cudnnPoolingDescriptor_t GetPoolingDesc(long hHandle);
		long SetPoolingDesc(long hHandle, PoolingMethod method, int h, int w, int pad_h, int pad_w, int stride_h, int stride_w);
		long PoolingForward(long hHandle, long hPoolingDesc, T fAlpha, long hBottomDesc, long hBottomData, T fBeta, long hTopDesc, long hTopData);
		long PoolingBackward(long hHandle, long hPoolingDesc, T fAlpha, long hTopDataDesc, long hTopData, long hTopDiffDesc, long hTopDiff, long hBottomDataDesc, long hBottomData, T fBeta, long hBottomDiffDesc, long hBottomDiff);

		long CreateLRNDesc(long* phHandle);
		long FreeLRNDesc(long hHandle);
		cudnnLRNDescriptor_t GetLRNDesc(long hHandle);
		long SetLRNDesc(long hHandle, unsigned int nSize, T fAlpha, T fBeta, T fK);

		long LRNForwardCC(long hHandle, long hNormDesc, T fAlpha, long hBottomDesc, long hBottomData, T fBeta, long hTopDesc, long hTopData);
		long LRNBackwardCC(long hHandle, long hNormDesc, T fAlpha, long hTopDataDesc, long hTopData, long hTopDiffDesc, long hTopDiff, long hBottomDataDesc, long hBottomDadta, T fBeta, long hBottomDiffDesc, long hBottomDiff);
		long LCNForwardCC(long hHandle, long hNormDesc, T fAlpha, long hBottomDesc, long hBottomData, long hTemp1, long hTemp2, T fBeta, long hTopDesc, long hTopData);
		long LCNBackwardCC(long hHandle, long hNormDesc, T fAlpha, long hBottomDataDesc, long hBottomData, long hTopDiff, long hTemp1, long hTemp2, T fBeta, long hBottomDiffDesc, long hBottomDiff);
		
#ifdef CUDNN_5
		long CreateActivationDesc(long* phHandle);
		long FreeActivationDesc(long hHandle);
		cudnnActivationDescriptor_t GetActivationDesc(long hHandle);
		long SetActivationDesc(long hHandle, ActivationMethod method);

		long CreateDropoutDesc(long* phHandle);
		long FreeDropoutDesc(long hHandle);
		cudnnDropoutDescriptor_t GetDropoutDesc(long hHandle);
		long SetDropoutDesc(long hHandle, long hDropoutDesc, T fDropout, long hStates, long lSeed);
		long GetDropoutInfo(long hHandle, long hBottomDesc, unsigned long* plState, unsigned long* plReserved);
		long DropoutForward(long hHandle, long hDropoutDesc, long hBottomDesc, long hBottom, long hTopDesc, long hTop, long hReservedSpace);
		long DropoutBackward(long hHandle, long hDropoutDesc, long hTopDesc, long hTop, long hBottomDesc, long hBottom, long hReservedSpace);
#endif

		long TanhForward(long hHandle, T fAlpha, long hBottomDesc, long hBottomData, T fBeta, long hTopDesc, long hTopData);
		long TanhBackward(long hHandle, T fAlpha, long hTopDataDesc, long hTopData, long hTopDiffDesc, long hTopDiff, long hBottomDataDesc, long hBottomData, T fBeta, long hBottomDiffDesc, long hBottomDiff);

		long SigmoidForward(long hHandle, T fAlpha, long hBottomDesc, long hBottomData, T fBeta, long hTopDesc, long hTopData);
		long SigmoidBackward(long hHandle, T fAlpha, long hTopDataDesc, long hTopData, long hTopDiffDesc, long hTopDiff, long hBottomDataDesc, long hBottomData, T fBeta, long hBottomDiffDesc, long hBottomDiff);

		long ReLUForward(long hHandle, T fAlpha, long hBottomDesc, long hBottomData, T fBeta, long hTopDesc, long hTopData);
		long ReLUBackward(long hHandle, T fAlpha, long hTopDataDesc, long hTopData, long hTopDiffDesc, long hTopDiff, long hBottomDataDesc, long hBottomData, T fBeta, long hBottomDiffDesc, long hBottomDiff);

		long SoftmaxForward(long hHandle, T fAlpha, long hBottomDesc, long hBottomData, T fBeta, long hTopDesc, long hTopData);
		long SoftmaxBackward(long hHandle, T fAlpha, long hTopDataDesc, long hTopData, long hTopDiffDesc, long hTopDiff, T fBeta, long hBottomDiffDesc, long hBottomDiff);

		long CreatePCA(int nMaxIterations, int nM, int nN, int nK, long hData, long hScoresResult, long hLoadsResult, long hResiduals, long hEigenvalues, Math<T>* pMath, long* phHandle);
		long FreePCA(long hHandle);
		pcaHandle<T>* GetPCA(long hHandle);
		long RunPCA(long hHandle, int nSteps, bool* pbDone, int* pnCurrentIteration, int* pnCurrentK);

		long CreateTsneGaussianPerplexity(unsigned int nN, unsigned int nD, unsigned int nK, long hX, long hCurP, long hValP, long hRowPonhost, long hColPonhost, T fPerplexity, Math<T>* pMath, long *phHandle);
		long FreeTsneGaussianPerplexity(long hHandle);
		tsnegpHandle<T>* GetTsneGaussianPerplexity(long hHandle);
		long FindTsneGaussianPerplexity(long hHandle, bool* pbDone, int* pnCurrentIteration, int* pnMaxIteration);

		long CreateTsne(unsigned int nN, unsigned int nD, long hY, long hValP, long hRowP, long hColP, long hdC, T fTheta, Math<T>* pMath, long* phHandle);
		long FreeTsne(long hHandle);
		tsnegHandle<T>* GetTsne(long hHandle);
		long ComputeTsneGradient(long hHandle, bool bValPUpdated);
		long EvaluateTsneError(long hHandle, T* fErr);

		long CreateMemoryTest(T pfPctToAllocate, long* phHandle, size_t* pszTotalNumBlocks, T* pfMemAllocated, T* pfMemStartAddr, T* pfMemBlockSize);
		long FreeMemoryTest(long hHandle);
		memtestHandle<T>* GetMemoryTest(long hHandle);
		long RunMemoryTest(long hHandle, MEMTEST_TYPE memTestType, size_t szStartOffset, size_t szCount, long* plCount, T** ppfData, bool bVerbose, bool bWrite, bool bReadWrite, bool bRead);

		long CreateNCCL(int nGpuID, int nCount, int nRank, char* szId, Math<T>* pMath, long* phHandle);
		long FreeNCCL(long hHandle);
		ncclHandle<T>* GetNCCL(long hHandle);
		long Memory<T>::SetNCCL(ncclHandle<T>* pNccl, long* hHandle);
		long NcclInitSingleProcess(long lBufferCount, long rgNcclHandle[], int nCount);
		long NcclInitMultiProcess(long lBufferCount, long hNccl);
		long NcclBroadcast(long hNccl, long hStream, long hX, int nCount);
		long NcclAllReduce(long hNccl, long hStream, long hX, int nCount, NCCL_OP op, T fScale);
};


//=============================================================================
//	Inline Methods
//=============================================================================


//-----------------------------------------------------------------------------
//	Memory
//-----------------------------------------------------------------------------

template <class T>
inline long Memory<T>::CheckMemoryAttributes(long hSrc, int nSrcDeviceID, long hDst, int nDstDeviceID, bool* pbResult)
{
	LONG lErr;
	MemoryItem* pSrc;
	MemoryItem* pDst;

	*pbResult = false;

	if (lErr = m_memory.GetData(hSrc, &pSrc))
		return lErr;

	if (lErr = m_memory.GetData(hDst, &pDst))
		return lErr;

	cudaPointerAttributes srcAttrib;
	cudaPointerAttributes dstAttrib;

	if (lErr = cudaPointerGetAttributes(&srcAttrib, pSrc->Data()))
		return lErr;

	if (lErr = cudaPointerGetAttributes(&dstAttrib, pDst->Data()))
		return lErr;

	if (srcAttrib.device == nSrcDeviceID && 
		dstAttrib.device == nDstDeviceID)
		*pbResult = true;

	return 0;
}


template <class T>
long Memory<T>::CreateMemoryPointer(long hData, long lOffset, long lCount, long* phHandle)
{
	long lErr;
	long lSize = m_memory.GetSize(lCount, sizeof(T));
	long lSizeOffset = m_memory.GetSize(lOffset, sizeof(T));
	MemoryItem* pData = NULL;

	if (m_memory.GetData(hData, &pData))
		return NULL;

	T* data = (T*)pData->Data();

	if (lOffset > 0)
		data += lOffset;

	long hHandle = 0;
	
	if (lErr = m_memoryPointers.Allocate(pData->DeviceID(), data, lSize, &hHandle))
		return lErr;

	// Move the handle into the range [MAX_HANDLES, MAX_HANDLES*2]
	// to indicate that it is a memory pointer reference.
	hHandle += MAX_ITEMS;
	*phHandle = hHandle;

	return 0;
}

template <class T>
long Memory<T>::FreeMemoryPointer(long hData)
{
	return m_memoryPointers.Free(hData - MAX_ITEMS);
}


template <class T>
inline long Memory<T>::AllocMemory(int nDeviceID, long lCount, T* pSrc, long hStream, long* phHandle)
{
	cudaStream_t pStream = NULL;

	if (hStream > 0)
		pStream = (cudaStream_t)m_streams.GetData(hStream);

	long lSize = m_memory.GetSize(lCount, sizeof(T));

	return m_memory.Allocate(nDeviceID, lSize, pSrc, pStream, phHandle);
}

template <class T>
inline long Memory<T>::FreeMemory(long hHandle)
{
	return m_memory.Free(hHandle);
}

template <class T>
inline long Memory<T>::GetMemory(long hHandle, MemoryItem** ppItem)
{
	return m_memory.GetData(hHandle, ppItem);
}

template <class T>
inline T* Memory<T>::GetMemoryToHost(long hHandle, long* plCount)
{
	MemoryItem *pItem;

	if (m_memory.GetData(hHandle, &pItem))
		return NULL;

	T* pHost = NULL;
	long lCount = pItem->Size() / sizeof(T);

	if (AllocHost(lCount, &pHost, (T*)pItem->Data(), TRUE))
		return NULL;

	if (plCount != NULL)
		*plCount = lCount;

	return pHost;
}

template <class T>
inline long Memory<T>::SetMemoryToHost(long hHandle, T* pDst)
{
	LONG lErr;
	MemoryItem *pItem;

	if (lErr = m_memory.GetData(hHandle, &pItem))
		return lErr;

	long lSize = pItem->Size();

	return pItem->GetData(lSize, pDst);
}

template <class T>
inline long Memory<T>::SetMemory(long hHandle, T* pSrc, long lCount, long hStream)
{
	cudaStream_t pStream = NULL;

	if (hStream > 0)
		pStream = (cudaStream_t)m_streams.GetData(hStream);

	return m_memory.SetData(hHandle, lCount * sizeof(T), pSrc, pStream);
}

template <class T>
inline long Memory<T>::SetMemoryAt(long hHandle, T* pSrc, long lCount, int nOffset)
{
	return m_memory.SetDataAt(hHandle, lCount * sizeof(T), pSrc, nOffset * sizeof(T));
}


template <class T>
inline long Memory<T>::FreeHost(T* pDst)
{
	if (pDst == NULL)
		return 0;

#ifdef USE_PINNED_HOST_MEM
	return cudaFreeHost(pDst);
#else
	free(pDst);
	return 0;
#endif
}

template <class T>
inline long Memory<T>::FreeHost(LPTSTR pDst)
{
	if (pDst == NULL)
		return 0;

#ifdef USE_PINNED_HOST_MEM
	return cudaFreeHost(pDst);
#else
	free(pDst);
	return 0;
#endif
}

template <class T>
inline HostBuffer<T>* Memory<T>::GetHostBuffer(long hHandle)
{
	return (HostBuffer<T>*)m_hostbuffers.GetData(hHandle);
}

template <class T>
inline long Memory<T>::SetHostBuffer(long hHandle, long lCount, T* pData)
{
	HostBuffer<T>* p = (HostBuffer<T>*)m_hostbuffers.GetData(hHandle);
	if (p == NULL)
		return ERROR_MEMORY_OUT;

	return cudaMemcpy(p->Data(), pData, lCount * sizeof(T), cudaMemcpyHostToHost);
}


//-----------------------------------------------------------------------------
//	Streams
//-----------------------------------------------------------------------------

template <class T>
inline long Memory<T>::FreeStream(long hHandle)
{
	cudaStream_t h = (cudaStream_t)m_streams.Free(hHandle);
	
	if (h != NULL)
		cudaStreamDestroy(h);

	return 0;
}

template <class T>
inline long Memory<T>::SynchronizeStream(long hHandle)
{
	cudaStream_t h = cudaStreamDefault;

	if (hHandle > 0)
		h = (cudaStream_t)m_streams.GetData(hHandle);

	return cudaStreamSynchronize(h);
}

template <class T>
inline cudaStream_t Memory<T>::GetStream(long hHandle)
{
	return (cudaStream_t)m_streams.GetData(hHandle);
}

template <class T>
inline long Memory<T>::FreeCuDNN(long hHandle)
{
	cudnnHandle_t h = (cudnnHandle_t)m_cudnn.Free(hHandle);

	if (h != NULL)
		cudnnDestroy(h);

	return 0;
}

template <class T>
inline cudnnHandle_t Memory<T>::GetCuDNN(long hHandle)
{
	return (cudnnHandle_t)m_cudnn.GetData(hHandle);
}

template <class T>
inline long Memory<T>::FreeTensorDesc(long hHandle)
{
	cudnnTensorDescriptor_t desc = (cudnnTensorDescriptor_t)m_tensorDesc.Free(hHandle);

	if (desc != NULL)
		cudnnDestroyTensorDescriptor(desc);

	return 0;
}

template <class T>
inline cudnnTensorDescriptor_t Memory<T>::GetTensorDesc(long hHandle)
{
	return (cudnnTensorDescriptor_t)m_tensorDesc.GetData(hHandle);
}

template <class T>
inline long Memory<T>::SetTensorDesc(long hHandle, int n, int c, int h, int w, int stride_n, int stride_c, int stride_h, int stride_w)
{
	LONG lErr;
	cudnnTensorDescriptor_t desc = (cudnnTensorDescriptor_t)m_tensorDesc.GetData(hHandle);
	cudnnDataType_t type = (sizeof(T) == 4) ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;
	if (lErr = cudnnSetTensor4dDescriptorEx(desc, type, n, c, h, w, stride_n, stride_c, stride_h, stride_w))
		return lErr | ERROR_CUDNN_OFFSET;

	return CUDNN_STATUS_SUCCESS;
}

template <class T>
inline long Memory<T>::FreeFilterDesc(long hHandle)
{
	cudnnFilterDescriptor_t desc = (cudnnFilterDescriptor_t)m_filterDesc.Free(hHandle);

	if (desc != NULL)
		cudnnDestroyFilterDescriptor(desc);

	return 0;
}

template <class T>
inline cudnnFilterDescriptor_t Memory<T>::GetFilterDesc(long hHandle)
{
	return (cudnnFilterDescriptor_t)m_filterDesc.GetData(hHandle);
}

template <class T>
inline long Memory<T>::SetFilterDesc(long hHandle, int n, int c, int h, int w)
{
	LONG lErr;
	cudnnFilterDescriptor_t desc = (cudnnFilterDescriptor_t)m_filterDesc.GetData(hHandle);
	cudnnDataType_t type = (sizeof(T) == 4) ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;
#ifdef CUDNN_5
	if (lErr = cudnnSetFilter4dDescriptor(desc, type, CUDNN_TENSOR_NCHW, n, c, h, w))
		return lErr | ERROR_CUDNN_OFFSET;
#else
	if (lErr = cudnnSetFilter4dDescriptor(desc, type, n, c, h, w))
		return lErr | ERROR_CUDNN_OFFSET;
#endif
	return CUDNN_STATUS_SUCCESS;
}

template <class T>
inline long Memory<T>::FreeConvolutionDesc(long hHandle)
{
	cudnnConvolutionDescriptor_t desc = (cudnnConvolutionDescriptor_t)m_convDesc.Free(hHandle);

	if (desc != NULL)
		cudnnDestroyConvolutionDescriptor(desc);

	return 0;
}

template <class T>
inline cudnnConvolutionDescriptor_t Memory<T>::GetConvolutionDesc(long hHandle)
{
	return (cudnnConvolutionDescriptor_t)m_convDesc.GetData(hHandle);
}

template <class T>
inline long Memory<T>::SetConvolutionDesc(long hHandle, int pad_h, int pad_w, int stride_h, int stride_w)
{
	LONG lErr;
	cudnnConvolutionDescriptor_t desc = (cudnnConvolutionDescriptor_t)m_convDesc.GetData(hHandle);
#ifdef CUDNN_6
	cudnnDataType_t computeType = (sizeof(T) == sizeof(double)) ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;
	if (lErr = cudnnSetConvolution2dDescriptor(desc, pad_h, pad_w, stride_h, stride_w, 1, 1, CUDNN_CROSS_CORRELATION, computeType))
		return lErr | ERROR_CUDNN_OFFSET;
#else
	if (lErr = cudnnSetConvolution2dDescriptor(desc, pad_h, pad_w, stride_h, stride_w, 1, 1, CUDNN_CROSS_CORRELATION))
		return lErr | ERROR_CUDNN_OFFSET;
#endif
	return CUDNN_STATUS_SUCCESS;
}


template <class T>
inline long Memory<T>::FreePoolingDesc(long hHandle)
{
	cudnnPoolingDescriptor_t desc = (cudnnPoolingDescriptor_t)m_poolDesc.Free(hHandle);

	if (desc != NULL)
		cudnnDestroyPoolingDescriptor(desc);

	return 0;
}

template <class T>
inline cudnnPoolingDescriptor_t Memory<T>::GetPoolingDesc(long hHandle)
{
	return (cudnnPoolingDescriptor_t)m_poolDesc.GetData(hHandle);
}

template <class T>
inline long Memory<T>::SetPoolingDesc(long hHandle, PoolingMethod method, int h, int w, int pad_h, int pad_w, int stride_h, int stride_w)
{
	LONG lErr;
	cudnnPoolingDescriptor_t desc = (cudnnPoolingDescriptor_t)m_poolDesc.GetData(hHandle);
#ifdef CUDNN_5
	if (lErr = cudnnSetPooling2dDescriptor(desc, (cudnnPoolingMode_t)method, CUDNN_NOT_PROPAGATE_NAN, h, w, pad_h, pad_w, stride_h, stride_w))
		return lErr | ERROR_CUDNN_OFFSET;
#else
	if (lErr = cudnnSetPooling2dDescriptor(desc, (cudnnPoolingMode_t)method, h, w, pad_h, pad_w, stride_h, stride_w))
		return lErr | ERROR_CUDNN_OFFSET;
#endif
	return CUDNN_STATUS_SUCCESS;
}


template <class T>
inline long Memory<T>::FreeLRNDesc(long hHandle)
{
	cudnnLRNDescriptor_t desc = (cudnnLRNDescriptor_t)m_lrnDesc.Free(hHandle);

	if (desc != NULL)
		cudnnDestroyLRNDescriptor(desc);

	return 0;
}

template <class T>
inline cudnnLRNDescriptor_t Memory<T>::GetLRNDesc(long hHandle)
{
	return (cudnnLRNDescriptor_t)m_lrnDesc.GetData(hHandle);
}

template <class T>
inline long Memory<T>::SetLRNDesc(long hHandle, unsigned int nSize, T fAlpha, T fBeta, T fK)
{
	LONG lErr;
	cudnnLRNDescriptor_t desc = (cudnnLRNDescriptor_t)m_lrnDesc.GetData(hHandle);
	if (lErr = cudnnSetLRNDescriptor(desc, nSize, fAlpha, fBeta, fK))
		return lErr | ERROR_CUDNN_OFFSET;

	return CUDNN_STATUS_SUCCESS;
}

#ifdef CUDNN_5

template <class T>
inline long Memory<T>::CreateActivationDesc(long* phHandle)
{
	LONG lErr;
	cudnnActivationDescriptor_t desc = NULL;

	if (phHandle == NULL)
		return ERROR_PARAM_NULL;

	if (lErr = cudnnCreateActivationDescriptor(&desc))
		return lErr;

	long hHandle = m_activationDesc.Allocate(desc);
	if (hHandle < 0)
	{
		cudnnDestroyActivationDescriptor(desc);
		return ERROR_MEMORY_OUT;
	}

	*phHandle = hHandle;
	return 0;
}

template <class T>
inline long Memory<T>::FreeActivationDesc(long hHandle)
{
	cudnnActivationDescriptor_t desc = (cudnnActivationDescriptor_t)m_activationDesc.Free(hHandle);

	if (desc != NULL)
		cudnnDestroyActivationDescriptor(desc);

	return 0;
}

template <class T>
inline cudnnActivationDescriptor_t Memory<T>::GetActivationDesc(long hHandle)
{
	return (cudnnActivationDescriptor_t)m_activationDesc.GetData(hHandle);
}

template <class T>
inline long Memory<T>::SetActivationDesc(long hHandle, ActivationMethod method)
{
	LONG lErr;
	cudnnActivationDescriptor_t desc = GetActivationDesc(hHandle);
	if (lErr = cudnnSetActivationDescriptor(desc, (cudnnActivationMode_t)method, CUDNN_NOT_PROPAGATE_NAN, 0))
		return lErr | ERROR_CUDNN_OFFSET;

	return CUDNN_STATUS_SUCCESS;
}


template <class T>
inline long Memory<T>::FreeDropoutDesc(long hHandle)
{
	cudnnDropoutDescriptor_t desc = (cudnnDropoutDescriptor_t)m_dropoutDesc.Free(hHandle);

	if (desc != NULL)
		cudnnDestroyDropoutDescriptor(desc);

	return 0;
}

template <class T>
inline cudnnDropoutDescriptor_t Memory<T>::GetDropoutDesc(long hHandle)
{
	return (cudnnDropoutDescriptor_t)m_dropoutDesc.GetData(hHandle);
}

#endif // CUDNN_5


template <class T>
inline long Memory<T>::CreatePCA(int nMaxIterations, int nM, int nN, int nK, long hData, long hScoresResult, long hLoadsResult, long hResiduals, long hEigenvalues, Math<T>* pMath, long* phHandle)
{
	LONG lErr;
	pcaHandle<T>* pca = NULL;

	if (phHandle == NULL)
		return ERROR_PARAM_NULL;

	if ((pca = new pcaHandle<T>(nMaxIterations, nM, nN, nK, hData, hScoresResult, hLoadsResult, hResiduals, hEigenvalues)) == NULL)
		return ERROR_MEMORY_OUT;

	if (lErr = pca->Initialize(this, pMath))
	{
		delete pca;
		return lErr;
	}

	long hHandle = m_pca.Allocate(pca);
	if (hHandle < 0)
	{
		delete pca;
		return ERROR_MEMORY_OUT;
	}

	*phHandle = hHandle;
	return 0;
}

template <class T>
inline long Memory<T>::FreePCA(long hHandle)
{
	pcaHandle<T>* pca = (pcaHandle<T>*)m_pca.Free(hHandle);

	if (pca != NULL)
	{
		pca->CleanUp();
		delete pca;
	}

	return 0;
}

template <class T>
inline pcaHandle<T>* Memory<T>::GetPCA(long hHandle)
{
	return (pcaHandle<T>*)m_pca.GetData(hHandle);
}

template <class T>
inline long Memory<T>::RunPCA(long hHandle, int nSteps, bool* pbDone, int* pnCurrentIteration, int* pnCurrentK)
{
	pcaHandle<T>* pca = GetPCA(hHandle);

	if (pca == NULL)
		return ERROR_PARAM_NULL;

	return pca->Run(nSteps, pbDone, pnCurrentIteration, pnCurrentK);
}


template <class T>
inline long Memory<T>::CreateTsneGaussianPerplexity(unsigned int nM, unsigned int nN, unsigned int nK, long hX, long hCurP, long hValP, long hRowPonhost, long hColPonhost, T fPerplexity, Math<T>* pMath, long* phHandle)
{
	LONG lErr;
	tsnegpHandle<T>* tsne = NULL;

	if (phHandle == NULL)
		return ERROR_PARAM_NULL;

	if ((tsne = new tsnegpHandle<T>(nM, nN, nK, hX, hCurP, hValP, hRowPonhost, hColPonhost, fPerplexity)) == NULL)
		return ERROR_MEMORY_OUT;

	if (lErr = tsne->Initialize(this, pMath))
	{
		delete tsne;
		return lErr;
	}

	long hHandle = m_tsnegp.Allocate(tsne);
	if (hHandle < 0)
	{
		delete tsne;
		return ERROR_MEMORY_OUT;
	}

	*phHandle = hHandle;
	return 0;
}

template <class T>
inline long Memory<T>::FreeTsneGaussianPerplexity(long hHandle)
{
	tsnegpHandle<T>* tsne = (tsnegpHandle<T>*)m_tsnegp.Free(hHandle);

	if (tsne != NULL)
	{
		tsne->CleanUp();
		delete tsne;
	}

	return 0;
}

template <class T>
inline tsnegpHandle<T>* Memory<T>::GetTsneGaussianPerplexity(long hHandle)
{
	return (tsnegpHandle<T>*)m_tsnegp.GetData(hHandle);
}

template <class T>
inline long Memory<T>::FindTsneGaussianPerplexity(long hHandle, bool* pbDone, int* pnCurrentIteration, int* pnMaxIteration)
{
	tsnegpHandle<T>* tsne = GetTsneGaussianPerplexity(hHandle);

	if (tsne == NULL)
		return ERROR_PARAM_NULL;

	return tsne->Run(pbDone, pnCurrentIteration, pnMaxIteration);
}


template <class T>
inline long Memory<T>::CreateTsne(unsigned int nN, unsigned int nD, long hY, long hValP, long hRowP, long hColP, long hdC, T fTheta, Math<T>* pMath, long* phHandle)
{
	LONG lErr;
	tsnegHandle<T>* tsne = NULL;

	if (phHandle == NULL)
		return ERROR_PARAM_NULL;

	if ((tsne = new tsnegHandle<T>(nN, nD, hY, hValP, hRowP, hColP, hdC, fTheta)) == NULL)
		return ERROR_MEMORY_OUT;

	if (lErr = tsne->Initialize(this, pMath))
	{
		delete tsne;
		return lErr;
	}

	long hHandle = m_tsneg.Allocate(tsne);
	if (hHandle < 0)
	{
		delete tsne;
		return ERROR_MEMORY_OUT;
	}

	*phHandle = hHandle;
	return 0;
}

template <class T>
inline long Memory<T>::FreeTsne(long hHandle)
{
	tsnegHandle<T>* tsne = (tsnegHandle<T>*)m_tsneg.Free(hHandle);

	if (tsne != NULL)
	{
		tsne->CleanUp();
		delete tsne;
	}

	return 0;
}

template <class T>
inline tsnegHandle<T>* Memory<T>::GetTsne(long hHandle)
{
	return (tsnegHandle<T>*)m_tsneg.GetData(hHandle);
}

template <class T>
inline long Memory<T>::ComputeTsneGradient(long hHandle, bool bValPUpdated)
{
	tsnegHandle<T>* tsne = GetTsne(hHandle);

	if (tsne == NULL)
		return ERROR_PARAM_NULL;

	return tsne->ComputeGradient(bValPUpdated);
}

template <class T>
inline long Memory<T>::EvaluateTsneError(long hHandle, T* pfErr)
{
	tsnegHandle<T>* tsne = GetTsne(hHandle);

	if (tsne == NULL)
		return ERROR_PARAM_NULL;

	return tsne->EvaluateError(pfErr);
}


template <class T>
inline long Memory<T>::CreateMemoryTest(T fPctToAllocate, long* phHandle, size_t* pszTotalNumBlocks, T* pfMemAllocated, T* pfMemStartAddr, T* pfMemBlockSize)
{
	LONG lErr;
	memtestHandle<T>* memtest = NULL;

	if (phHandle == NULL)
		return ERROR_PARAM_NULL;

	if ((memtest = new memtestHandle<T>()) == NULL)
		return ERROR_MEMORY_OUT;

	if (lErr = memtest->Initialize(this, fPctToAllocate, pszTotalNumBlocks, pfMemAllocated, pfMemStartAddr, pfMemBlockSize))
	{
		delete memtest;
		return lErr;
	}

	long hHandle = m_memtest.Allocate(memtest);
	if (hHandle < 0)
	{
		delete memtest;
		return ERROR_MEMORY_OUT;
	}

	*phHandle = hHandle;
	return 0;
}

template <class T>
inline long Memory<T>::FreeMemoryTest(long hHandle)
{
	memtestHandle<T>* memtest = (memtestHandle<T>*)m_memtest.Free(hHandle);

	if (memtest != NULL)
	{
		memtest->CleanUp();
		delete memtest;
	}

	return 0;
}

template <class T>
inline memtestHandle<T>* Memory<T>::GetMemoryTest(long hHandle)
{
	return (memtestHandle<T>*)m_memtest.GetData(hHandle);
}

template <class T>
inline long Memory<T>::RunMemoryTest(long hHandle, MEMTEST_TYPE memtestType, size_t szStartOffset, size_t szCount, long* plCount, T** ppfData, bool bVerbose, bool bWrite, bool bReadWrite, bool bRead)
{
	memtestHandle<T>* memtest = GetMemoryTest(hHandle);

	if (memtest == NULL)
		return ERROR_PARAM_NULL;

	return memtest->Run(memtestType, szStartOffset, szCount, plCount, ppfData, bVerbose, bWrite, bReadWrite, bRead);
}


template <class T>
inline long Memory<T>::CreateNCCL(int nGpuID, int nCount, int nRank, char* szId, Math<T>* pMath, long* phHandle)
{
	LONG lErr;
	ncclHandle<T>* nccl = NULL;

	if (phHandle == NULL)
		return ERROR_PARAM_NULL;

	if ((nccl = new ncclHandle<T>()) == NULL)
		return ERROR_MEMORY_OUT;

	if (lErr = nccl->Initialize(this, pMath, nGpuID, nCount, nRank, szId))
	{
		delete nccl;
		return lErr;
	}

	long hHandle = m_nccl.Allocate(nccl);
	if (hHandle < 0)
	{
		delete nccl;
		return ERROR_MEMORY_OUT;
	}

	*phHandle = hHandle;
	return 0;
}

template <class T>
inline long Memory<T>::FreeNCCL(long hHandle)
{
	ncclHandle<T>* nccl = (ncclHandle<T>*)m_nccl.Free(hHandle);

	if (nccl != NULL && nccl->IsOwner())
	{
		nccl->CleanUp();

		if (nccl->RefCount() == 0)
			delete nccl;
	}

	return 0;
}

template <class T>
inline ncclHandle<T>* Memory<T>::GetNCCL(long hHandle)
{
	return (ncclHandle<T>*)m_nccl.GetData(hHandle);
}

template <class T>
inline long Memory<T>::SetNCCL(ncclHandle<T>* pNccl, long* phHandle)
{
	if (pNccl == NULL || phHandle == NULL)
		return ERROR_PARAM_NULL;

	long hHandle = m_nccl.Allocate(pNccl);
	if (hHandle < 0)
		return ERROR_MEMORY_OUT;

	*phHandle = hHandle;
	return 0;
}

template <class T>
long Memory<T>::NcclInitSingleProcess(long lBufferCount, long rgNcclHandle[], int nCount)
{
	ncclHandle<T>** rgNccl = new ncclHandle<T>*[nCount];
	if (rgNccl == NULL)
		return ERROR_MEMORY_OUT;

	for (int i = 0; i < nCount; i++)
	{
		rgNccl[i] = GetNCCL(rgNcclHandle[i]);
	}

	LONG lErr = rgNccl[0]->InitSingleProcess(lBufferCount, nCount, rgNccl);
	delete rgNccl;

	return lErr;
}

template <class T>
long Memory<T>::NcclInitMultiProcess(long lBufferCount, long hNccl)
{
	return GetNCCL(hNccl)->InitMultiProcess(lBufferCount);
}

template <class T>
inline long Memory<T>::NcclBroadcast(long hNccl, long hStream, long hX, int nCount)
{
	return GetNCCL(hNccl)->Broadcast(hStream, hX, nCount);
}

template <class T>
inline long Memory<T>::NcclAllReduce(long hNccl, long hStream, long hX, int nCount, NCCL_OP op, T fScale)
{
	return GetNCCL(hNccl)->AllReduce(hStream, hX, nCount, op, fScale);
}

#endif // __MEMORY_CU__