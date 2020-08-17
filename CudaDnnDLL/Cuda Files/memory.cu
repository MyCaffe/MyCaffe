//=============================================================================
//	FILE:	memory.cu
//
//	DESC:	This file the basic memory management for the given device
//=============================================================================

#include "memory.h"


//=============================================================================
//	Class Methods
//=============================================================================

template <class T>
Memory<T>::Memory() : m_memory(), m_memoryPointers(), m_hostbuffers(), m_streams(), m_tensorDesc(), m_filterDesc(), m_convDesc(), m_poolDesc(), m_rnnDesc(), m_rnnDataDesc2(), m_lrnDesc(), m_cudnn(), m_pca(), m_tsnegp(), m_tsneg(), m_memtest(), m_nccl(), m_ssd(), m_memoryMap()
{
	m_memory.SetMemoryPointers(&m_memoryPointers);

	m_tOne = (T)1;
	m_tZero = (T)0;

#ifdef CUDNN_5
	CreateActivationDesc(&m_hGlobalActivationSigmoid);
	SetActivationDesc(m_hGlobalActivationSigmoid, SIGMOID);
	CreateActivationDesc(&m_hGlobalActivationRelu);
	SetActivationDesc(m_hGlobalActivationRelu, RELU);
	CreateActivationDesc(&m_hGlobalActivationTanh);
	SetActivationDesc(m_hGlobalActivationTanh, TANH);
	CreateActivationDesc(&m_hGlobalActivationElu);
	SetActivationDesc(m_hGlobalActivationElu, ELU);
#endif
}

template Memory<double>::Memory();
template Memory<float>::Memory();


template <class T>
Memory<T>::~Memory()
{
	for (int i=0; i<m_hostbuffers.GetCount(); i++)
	{
		FreeHostBuffer(i);
	}

	for (int i=0; i<m_streams.GetCount(); i++)
	{
		FreeStream(i);
	}

	for (int i=0; i<m_tensorDesc.GetCount(); i++)
	{
		FreeTensorDesc(i);
	}

	for (int i=0; i<m_filterDesc.GetCount(); i++)
	{
		FreeFilterDesc(i);
	}

	for (int i=0; i<m_convDesc.GetCount(); i++)
	{
		FreeConvolutionDesc(i);
	}

	for (int i=0; i<m_poolDesc.GetCount(); i++)
	{
		FreePoolingDesc(i);
	}

	for (int i = 0; i < m_rnnDesc.GetCount(); i++)
	{
		FreeRnnDesc(i);
	}

	for (int i = 0; i < m_rnnDataDesc1.GetCount(); i++)
	{
		FreeRnnDataDesc1(i);
	}

	for (int i = 0; i < m_rnnDataDesc2.GetCount(); i++)
	{
		FreeRnnDataDesc2(i);
	}

	for (int i=0; i<m_lrnDesc.GetCount(); i++)
	{
		FreeLRNDesc(i);
	}

	for (int i=0; i<m_cudnn.GetCount(); i++)
	{
		FreeCuDNN(i);
	}

#ifdef CUDNN_5
	for (int i=0; i<m_activationDesc.GetCount(); i++)
	{
		FreeActivationDesc(i);
	}

	m_hGlobalActivationSigmoid = 0;
	m_hGlobalActivationRelu = 0;
	m_hGlobalActivationTanh = 0;
	m_hGlobalActivationElu = 0;

	for (int i = 0; i < m_dropoutDesc.GetCount(); i++)
	{
		FreeDropoutDesc(i);
	}
#endif

	for (int i=0; i<m_pca.GetCount(); i++)
	{
		FreePCA(i);
	}

	for (int i=0; i<m_tsnegp.GetCount(); i++)
	{
		FreeTsneGaussianPerplexity(i);
	}

	for (int i = 0; i < m_memtest.GetCount(); i++)
	{
		FreeMemoryTest(i);
	}

	for (int i = 0; i < m_imgop.GetCount(); i++)
	{
		FreeImageOp(i);
	}

	for (int i = 0; i < m_nccl.GetCount(); i++)
	{
		FreeNCCL(i);
	}

	for (int i = 0; i < m_ssd.GetCount(); i++)
	{
		FreeSSD(i);
	}

	m_memoryMap.clear();
	m_cudnnRef.clear();
	m_cudnnH2Dev.clear();
	m_cudnnDev2H.clear();

	m_streamRef.clear();
	m_streamH2Dev.clear();
	m_streamH2Idx.clear();
	m_streamH2CudnnH.clear();
	m_streamCudnnRef.clear();
	m_streamDev2Idx2H.clear();
}

template Memory<double>::~Memory();
template Memory<float>::~Memory();


template <class T>
long Memory<T>::GetDeviceMemory(int nDeviceID, T* pfTotal, T* pfFree, T* pfUsed, bool* pbEstimate)
{
	LONG lErr;
	size_t lFree = 0;
	size_t lTotal = 0;
	size_t lUsed = 0;
	int nOriginalDeviceID = -1;

	if (nDeviceID >= 0)
	{
		if (lErr = cudaGetDevice(&nOriginalDeviceID))
			return lErr;

		if (nDeviceID != nOriginalDeviceID)
		{
			if (lErr = cudaSetDevice(nDeviceID))
				return lErr;
		}
	}

	if (nDeviceID == -1)
	{
		cudaDeviceProp prop;

		memset(&prop, 0, sizeof(cudaDeviceProp));
		if (lErr = cudaGetDeviceProperties(&prop, nDeviceID))
			return lErr;

		lTotal = prop.totalGlobalMem;
		lUsed = (size_t)m_memory.GetTotalUsed();
		lFree = lTotal - lUsed;
		*pbEstimate = true;
	}
	else
	{
		if (lErr = cudaMemGetInfo(&lFree, &lTotal))
			return lErr;

		lUsed = lTotal - lFree;
		*pbEstimate = false;
	}

	*pfTotal = (T)((double)lTotal / (double)1000000000.0);
	*pfFree = (T)((double)lFree / (double)1000000000.0);
	*pfUsed = (T)((double)lUsed / (double)1000000000.0);

	if (nOriginalDeviceID >= 0 && nOriginalDeviceID != nDeviceID)
	{
		if (lErr = cudaSetDevice(nOriginalDeviceID))
			return lErr;
	}

	return cudaStreamSynchronize(0);
}

template long Memory<double>::GetDeviceMemory(int nDeviceID, double* pdfTotal, double* pdfFree, double* pdfUsed, bool* pbEstimate);
template long Memory<float>::GetDeviceMemory(int nDeviceID, float* pfTotal, float* pfFree, float* pfUsed, bool* pbEstimate);


template <class T>
long Memory<T>::AllocHost(LPTSTR* ppDst, LPTSTR pSrc)
{
	int nLen = (int)_tcslen(pSrc);

	if (nLen == 0)
		return ERROR_PARAM_OUT_OF_RANGE;

	nLen++;	// make room for NULL;

	LPTSTR pDst = NULL;
	LONG lSize = nLen * sizeof(TCHAR);
	LONG lErr = 0;

	if (lErr = alloc_host((void**)&pDst, lSize, false))
		return lErr;

	pDst[nLen] = (TCHAR)NULL;
	_tcsncpy(pDst, pSrc, nLen);

	*ppDst = pDst;

	return lErr;
}

template long Memory<double>::AllocHost(LPTSTR* ppDst, LPTSTR pSrc);
template long Memory<float>::AllocHost(LPTSTR* ppDst, LPTSTR pSrc);


template <class T>
long Memory<T>::AllocHost(size_t lCount, T** ppDst, void* pSrc, bool bSrcOnDevice, bool bHalf, bool bPinned)
{
	if (lCount == 0)
		return ERROR_PARAM_OUT_OF_RANGE;

	if (ppDst == NULL)
		return ERROR_PARAM_NULL;

	long long lSize = lCount * sizeof(T);
	if (lSize > SIZE_MAX)
		return ERROR_MEMORY_RANGE_EXCEEDED;

	T* pDst = NULL;
	LONG lErr;

	if (lErr = alloc_host((void**)&pDst, (size_t)lSize, bPinned))
		return lErr;

	if (pSrc != NULL)
	{
		cudaMemcpyKind kind = (bSrcOnDevice) ? cudaMemcpyDeviceToHost : cudaMemcpyHostToHost;

		if (bHalf)
		{
			if (lErr = convertHalf2BaseType(lCount, pSrc, pDst, kind))
			{
				FreeHost(pDst);
				return lErr;
			}
		}
		else
		{
			if (lErr = cudaMemcpy(pDst, pSrc, (size_t)lSize, kind))
			{
				FreeHost(pDst);
				return lErr;
			}
		}
	}
	else
	{
		memset(pDst, 0, lSize);
	}

	*ppDst = pDst;
	return cudaGetLastError();
}

template long Memory<double>::AllocHost(size_t lCount, double** ppDst, void* pSrc, bool bSrcOnDevice, bool bHalf, bool bPinned);
template long Memory<float>::AllocHost(size_t lCount, float** ppDst, void* pSrc, bool bSrcOnDevice, bool bHalf, bool bPinned);


template <class T>
long Memory<T>::CopyToHost(size_t lCount, T* pDst, void* pSrc, bool bSrcOnDevice, bool bHalf)
{
	if (lCount == 0)
		return ERROR_PARAM_OUT_OF_RANGE;

	if (pDst == NULL || pSrc == NULL)
		return ERROR_PARAM_NULL;

	cudaMemcpyKind kind = (bSrcOnDevice) ? cudaMemcpyDeviceToHost : cudaMemcpyHostToHost;

	long long lSize = lCount * sizeof(T);
	if (lSize > SIZE_MAX)
		return ERROR_MEMORY_RANGE_EXCEEDED;

	if (bHalf)
		return convertHalf2BaseType(lCount, pSrc, pDst, kind);
	else
		return cudaMemcpy(pDst, pSrc, (size_t)lSize, kind);
}

template long Memory<double>::CopyToHost(size_t lCount, double* pDst, void* pSrc, bool bSrcOnDevice, bool bHalf);
template long Memory<float>::CopyToHost(size_t lCount, float* pDst, void* pSrc, bool bSrcOnDevice, bool bHalf);


template <class T>
long Memory<T>::CopyGpuToHost(long lCount, long hGpuSrc, long hHostDst)
{
	LONG lErr;
	MemoryItem* pSrcD;

	if (lErr = m_memory.GetData(hGpuSrc, &pSrcD))
		return lErr;

	HostBuffer<T>* pDstH = GetHostBuffer(hHostDst);
	if (pDstH == NULL)
		return ERROR_MEMORY_NOT_FOUND;

	T* pSrc = (T*)pSrcD->Data();
	T* pDst = (T*)pDstH->Data();

	return CopyToHost(lCount, pDst, pSrc, true, false);
}

template long Memory<double>::CopyGpuToHost(long lCount, long hGpuSrc, long hHostDst);
template long Memory<float>::CopyGpuToHost(long lCount, long hGpuSrc, long hHostDst);


template <class T>
long Memory<T>::CopyHostToGpu(long lCount, long hHostSrc, long hGpuDst)
{
	LONG lErr;
	MemoryItem* pDstD;

	if (lErr = m_memory.GetData(hGpuDst, &pDstD))
		return lErr;

	HostBuffer<T>* pSrcH = GetHostBuffer(hHostSrc);
	if (pSrcH == NULL)
		return ERROR_MEMORY_NOT_FOUND;

	T* pDst = (T*)pDstD->Data();
	T* pSrc = (T*)pSrcH->Data();

	long long lSize = lCount * sizeof(T);
	if (lSize > SIZE_MAX)
		return ERROR_MEMORY_RANGE_EXCEEDED;

	return cudaMemcpy(pDst, pSrc, lSize, cudaMemcpyHostToDevice);
}

template long Memory<double>::CopyHostToGpu(long lCount, long hHostSrc, long hGpuDst);
template long Memory<float>::CopyHostToGpu(long lCount, long hHostSrc, long hGpuDst);


template <class T>
long Memory<T>::AllocHostBuffer(size_t lCount, long* phHandle)
{
	LONG lErr = 0;

	if (lCount % 2 != 0)
		lCount++;

	T* pMem = NULL;
	
	if (lErr = AllocHost(lCount, &pMem, NULL, FALSE, FALSE))
		return lErr;

	HostBuffer<T>* pHostBuf = new HostBuffer<T>(pMem, lCount);
	if (pHostBuf == NULL)
	{
		FreeHost(pMem);
		return ERROR_MEMORY_OUT;
	}

	long hHandle = m_hostbuffers.Allocate(pHostBuf);
	if (hHandle < 0)
	{
		delete pHostBuf;
		FreeHost(pMem);
		return ERROR_MEMORY_OUT;
	}

	m_rgActiveHostBuffers[pMem] = pHostBuf;

	*phHandle = hHandle;

	return 0;
}

template long Memory<double>::AllocHostBuffer(size_t lCount, long* phHandle);
template long Memory<float>::AllocHostBuffer(size_t lCount, long* phHandle);


template <class T>
long Memory<T>::FreeHostBuffer(long hHandle)
{
	HostBuffer<T>* pHostBuf = (HostBuffer<T>*)m_hostbuffers.Free(hHandle);
	
	if (pHostBuf != NULL)
	{
		if (pHostBuf->Data() != NULL)
		{
			m_rgActiveHostBuffers.erase(pHostBuf->Data());
			FreeHost(pHostBuf->Data());
		}

		delete pHostBuf;
	}

	return 0;
}

template long Memory<double>::FreeHostBuffer(long hHandle);
template long Memory<float>::FreeHostBuffer(long hHandle);


template <class T>
bool Memory<T>::IsHostBuffer(T* pf)
{
	if (m_rgActiveHostBuffers.find(pf) != m_rgActiveHostBuffers.end())
		return true;

	return false;
}

template bool Memory<double>::IsHostBuffer(double* pf);
template bool Memory<float>::IsHostBuffer(float* pf);


template <class T>
long Memory<T>::CreateStream(long* phHandle, bool bNonBlocking, int nIndex)
{
	std::unique_lock<std::mutex> lock(m_sync);

	LONG lErr;
	cudaStream_t stream = NULL;
	int nDeviceID = 0;

	if (phHandle == NULL)
		return ERROR_PARAM_NULL;

	if (nIndex >= 0)
	{
		if (lErr = cudaGetDevice(&nDeviceID))
			return lErr;

		if (m_streamDev2Idx2H.find(nDeviceID) != m_streamDev2Idx2H.end())
		{
			if (m_streamDev2Idx2H[nDeviceID].find(nIndex) != m_streamDev2Idx2H[nDeviceID].end())
			{
				stream = m_streamDev2Idx2H[nDeviceID][nIndex];
				m_streamRef[stream]++;
			}
		}
	}

	if (stream == NULL)
	{
		if (bNonBlocking)
		{
			if (lErr = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking))
				return lErr;
		}
		else
		{
			if (lErr = cudaStreamCreate(&stream))
				return lErr;
		}

		if (nIndex >= 0)
		{
			m_streamDev2Idx2H[nDeviceID][nIndex] = stream;
			m_streamRef[stream] = 1;
			m_streamH2Dev[stream] = nDeviceID;
			m_streamH2Idx[stream] = nIndex;
		}
	}

	long hHandle = m_streams.Allocate(stream);
	if (hHandle < 0)
	{
		cudaStreamDestroy(stream);
		return ERROR_MEMORY_OUT;
	}

	*phHandle = hHandle;
	return cudaStreamSynchronize(0);
}

template long Memory<double>::CreateStream(long* phHandle, bool bNonBlocking, int nIndex);
template long Memory<float>::CreateStream(long* phHandle, bool bNonBlocking, int nIndex);


template <typename T>
__global__ void synchronize_thread_kernel()
{
}

template <class T>
long Memory<T>::SynchronizeThread()
{
	synchronize_thread_kernel<T><<<1, 1>>>();
	return cudaGetLastError();
}

template long Memory<double>::SynchronizeThread();
template long Memory<float>::SynchronizeThread();



template <class T>
long Memory<T>::CreateCuDNN(long hStream, long* phHandle)
{
	std::unique_lock<std::mutex> lock(m_sync);

	LONG lErr;
	cudnnHandle_t cudnn = NULL;

	if (phHandle == NULL)
		return ERROR_PARAM_NULL;

	int nDeviceID;
	if (lErr = cudaGetDevice(&nDeviceID))
		return lErr;

	if (hStream == 0)
	{
		// If the cudnn is already created for the device, share it
		// and increase the ref count.
		if (m_cudnnDev2H.find(nDeviceID) != m_cudnnDev2H.end())
		{
			cudnn = m_cudnnDev2H[nDeviceID];
			m_cudnnRef[cudnn]++;
		}
		// Otherwise create a new cudnn for the device and add it to 
		// the maps to share with a ref count of 1.
		else
		{
			if (lErr = cudnnCreate(&cudnn))
				return lErr | ERROR_CUDNN_OFFSET;

			m_cudnnRef[cudnn] = 1;
			m_cudnnDev2H[nDeviceID] = cudnn;
			m_cudnnH2Dev[cudnn] = nDeviceID;
		}
	}
	else
	{
		cudaStream_t stream = GetStream(hStream);
		// If the stream is a shared stream, share see if the cudnn
		// is already shared with the shared stream and use the shared
		// cudnn if it exists, making sure to increase its ref count.
		if (m_streamRef.find(stream) != m_streamRef.end())
		{
			if (m_streamH2CudnnH.find(stream) != m_streamH2CudnnH.end())
			{
				cudnn = m_streamH2CudnnH[stream];
				m_streamCudnnRef[cudnn]++;
			}
		}

		if (cudnn == NULL)
		{
			if (lErr = cudnnCreate(&cudnn))
				return lErr | ERROR_CUDNN_OFFSET;

			if (lErr = cudnnSetStream(cudnn, stream))
			{
				cudnnDestroy(cudnn);

				if (m_streamH2CudnnH.find(stream) != m_streamH2CudnnH.end())
					m_streamH2CudnnH.erase(stream);

				return lErr | ERROR_CUDNN_OFFSET;
			}

			// If the stream is a shared stream, add the cudnn to
			// it and set the ref count to 1.
			if (m_streamRef.find(stream) != m_streamRef.end())
			{
				m_streamH2CudnnH[stream] = cudnn;
				m_streamCudnnRef[cudnn] = 1;
			}
		}
	}

	long hHandle = m_cudnn.Allocate(cudnn);
	if (hHandle < 0)
	{
		free_cudnn(cudnn);
		return ERROR_MEMORY_OUT;
	}

	*phHandle = hHandle;
	return cudaStreamSynchronize(0);
}

template long Memory<double>::CreateCuDNN(long hStream, long* phHandle);
template long Memory<float>::CreateCuDNN(long hStream, long* phHandle);


template <class T>
long Memory<T>::CreateTensorDesc(long* phHandle)
{
	LONG lErr;
	cudnnTensorDescriptor_t desc = NULL;

	if (phHandle == NULL)
		return ERROR_PARAM_NULL;

	if (lErr = cudnnCreateTensorDescriptor(&desc))
		return lErr | ERROR_CUDNN_OFFSET;

	long hHandle = m_tensorDesc.Allocate(desc);
	if (hHandle < 0)
	{
		cudnnDestroyTensorDescriptor(desc);
		return ERROR_MEMORY_OUT;
	}

	*phHandle = hHandle;
	return cudaStreamSynchronize(0);
}

template long Memory<double>::CreateTensorDesc(long* phHandle);
template long Memory<float>::CreateTensorDesc(long* phHandle);


template <class T>
long Memory<T>::AddTensor(long hHandle, T fAlpha, long hSrcDesc, long hSrc, int nSrcOffset, T fBeta, long hDstDesc, long hDst, int nDstOffset)
{
	LONG lErr;
	cudnnHandle_t cudnn = GetCuDNN(hHandle);
	cudnnTensorDescriptor_t srcdesc = GetTensorDesc(hSrcDesc);
	cudnnTensorDescriptor_t dstdesc = GetTensorDesc(hDstDesc);
	MemoryItem* pSrc;
	MemoryItem* pDst;

	if (lErr = m_memory.GetData(hSrc, &pSrc))
		return lErr;

	if (lErr = m_memory.GetData(hDst, &pDst))
		return lErr;

	if (cudnn == NULL || srcdesc == NULL || dstdesc == NULL)
		return ERROR_PARAM_NULL;

	T* src = (T*)pSrc->Data();
	T* dst = (T*)pDst->Data();

	if (nSrcOffset > 0)
		src += nSrcOffset;

	if (nDstOffset > 0)
		dst += nDstOffset;

#ifdef CUDNN_4
	if (lErr = cudnnAddTensor(cudnn, &fAlpha, srcdesc, src, &fBeta, dstdesc, dst))
		return lErr | ERROR_CUDNN_OFFSET;
#else
	if (lErr = cudnnAddTensor(cudnn, CUDNN_ADD_SAME_C, &fAlpha, srcdesc, src, &fBeta, dstdesc, dst))
		return lErr | ERROR_CUDNN_OFFSET;
#endif
	return cudaStreamSynchronize(0);
}

template long Memory<double>::AddTensor(long hHandle, double dfAlpha, long hSrcDesc, long hSrc, int nSrcOffset, double dfBeta, long hDstDesc, long hDst, int nDstOffset);
template long Memory<float>::AddTensor(long hHandle, float fAlpha, long hSrcDesc, long hSrc, int nSrcOffset, float fBeta, long hDstDesc, long hDst, int nDstOffset);


template <class T>
long Memory<T>::CreateFilterDesc(long* phHandle)
{
	LONG lErr;
	cudnnFilterDescriptor_t desc = NULL;

	if (phHandle == NULL)
		return ERROR_PARAM_NULL;

	if (lErr = cudnnCreateFilterDescriptor(&desc))
		return lErr | ERROR_CUDNN_OFFSET;

	long hHandle = m_filterDesc.Allocate(desc);
	if (hHandle < 0)
	{
		cudnnDestroyFilterDescriptor(desc);
		return ERROR_MEMORY_OUT;
	}

	*phHandle = hHandle;
	return cudaStreamSynchronize(0);
}

template long Memory<double>::CreateFilterDesc(long* phHandle);
template long Memory<float>::CreateFilterDesc(long* phHandle);


template <class T>
long Memory<T>::CreateConvolutionDesc(long* phHandle)
{
	LONG lErr;
	cudnnConvolutionDescriptor_t desc = NULL;

	if (phHandle == NULL)
		return ERROR_PARAM_NULL;

	if (lErr = cudnnCreateConvolutionDescriptor(&desc))
		return lErr | ERROR_CUDNN_OFFSET;

	long hHandle = m_convDesc.Allocate(desc);
	if (hHandle < 0)
	{
		cudnnDestroyConvolutionDescriptor(desc);
		return ERROR_MEMORY_OUT;
	}

	*phHandle = hHandle;
	return cudaStreamSynchronize(0);
}

template long Memory<double>::CreateConvolutionDesc(long* phHandle);
template long Memory<float>::CreateConvolutionDesc(long* phHandle);


template <class T>
long Memory<T>::GetConvolutionInfo(long hHandle, long hBottomDesc, long hFilterDesc, long hConvDesc, long hTopDesc, size_t lWsLimitInBytes, bool bUseTensorCores, long* palgoFwd, size_t* plWsSizeFwd, long* palgoBwdFilter, size_t* plWsSizeBwdFilter, long* palgoBwdData, size_t* plWsSizeBwdData, int nPreferredFwdAlgo)
{
	cudnnStatus_t lErr;	
	cudnnHandle_t cudnn = GetCuDNN(hHandle);
	cudnnTensorDescriptor_t bottom = GetTensorDesc(hBottomDesc);
	cudnnFilterDescriptor_t filter = GetFilterDesc(hFilterDesc);
	cudnnConvolutionDescriptor_t conv = GetConvolutionDesc(hConvDesc);
	cudnnTensorDescriptor_t top = GetTensorDesc(hTopDesc);

	// Choose forward algorithm for convolution.
	cudnnConvolutionFwdAlgo_t algoFwd;
#ifdef CUDA11_0
	int nAlgCount;
	cudnnConvolutionFwdAlgoPerf_t fwdPref[10];
	if (lErr = cudnnGetConvolutionForwardAlgorithm_v7(cudnn, bottom, filter, conv, top, 10, &nAlgCount, fwdPref))
		return lErr;

	if (lWsLimitInBytes == (((size_t)-1) / 2 + 1) || lWsLimitInBytes >= (SIZE_MAX - 10))
	{
		algoFwd = fwdPref[0].algo;
	}
	else
	{
		int nIdx = 0;
		while (fwdPref[nIdx].status == CUDNN_STATUS_SUCCESS && fwdPref[nIdx].memory > lWsLimitInBytes && nIdx < nAlgCount && nIdx < 10)
		{
			nIdx++;
		}

		if (nIdx == nAlgCount)
			return ERROR_PARAM_OUT_OF_RANGE;

		algoFwd = fwdPref[nIdx].algo;
	}
#else
	// Setup the algorithm preference.
	cudnnConvolutionFwdPreference_t fwdPref = CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT;
	cudnnConvolutionBwdFilterPreference_t bwdFltPref = CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT;
	cudnnConvolutionBwdDataPreference_t bwdDataPref = CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT;

	if (lWsLimitInBytes == (((size_t)-1)/2 + 1) || lWsLimitInBytes >= (SIZE_MAX - 10))
	{
		lWsLimitInBytes = 0;
		fwdPref = CUDNN_CONVOLUTION_FWD_PREFER_FASTEST;
		bwdFltPref = CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST;
		bwdDataPref = CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST;
	}
	else if (lWsLimitInBytes == 0)
	{
		lWsLimitInBytes = 0;
		fwdPref = CUDNN_CONVOLUTION_FWD_NO_WORKSPACE;
		bwdFltPref = CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE;
		bwdDataPref = CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE;
	}

	if (lErr = cudnnGetConvolutionForwardAlgorithm(cudnn, bottom, filter, conv, top, fwdPref, lWsLimitInBytes, &algoFwd))
		return lErr | ERROR_CUDNN_OFFSET;
#endif

	// Get workspace size for forward algorithm.
	size_t szFwd = 0;
	if (lErr = cudnnGetConvolutionForwardWorkspaceSize(cudnn, bottom, filter, conv, top, algoFwd, &szFwd))
		return lErr | ERROR_CUDNN_OFFSET;

	// CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD has been found by the native Caffe team to work better than 
	// CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM on deconvolution (which acts a bit buggy in this
	// situation.  For this reason, when using cuDnn deconvolution, the C# side sets the preferred
	// fwd algo to CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD which is used only when the workspace is less
	// than or equat to the default workspace size and no errors occur when attempting to get the
	// workspace size for WINOGRAD.  By default, the nPrefferredFwdAlgo paraeter is ignored.
	if (nPreferredFwdAlgo >= 0 && 
		algoFwd == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM && 
		(int)algoFwd != nPreferredFwdAlgo)
	{
		size_t lWinogradWorkspaceSize = 0;
		lErr = cudnnGetConvolutionForwardWorkspaceSize(cudnn, bottom, filter, conv, top, (cudnnConvolutionFwdAlgo_t)nPreferredFwdAlgo, &lWinogradWorkspaceSize);
		if (lErr == CUDNN_STATUS_SUCCESS)
		{
			if (lWinogradWorkspaceSize <= szFwd)
			{
				algoFwd = (cudnnConvolutionFwdAlgo_t)nPreferredFwdAlgo;
				szFwd = lWinogradWorkspaceSize;
			}
		}
	}

	// Choose backward filter algorithm.
	cudnnConvolutionBwdFilterAlgo_t algoBwdFilter;
#ifdef CUDA11_0
	cudnnConvolutionBwdFilterAlgoPerf_t bwdFltPref[10];
	if (lErr = cudnnGetConvolutionBackwardFilterAlgorithm_v7(cudnn, bottom, top, conv, filter, 10, &nAlgCount, bwdFltPref))
		return lErr;

	if (lWsLimitInBytes == (((size_t)-1) / 2 + 1) || lWsLimitInBytes >= (SIZE_MAX - 10))
	{
		algoBwdFilter = bwdFltPref[0].algo;
	}
	else
	{
		int nIdx = 0;
		while (bwdFltPref[nIdx].status == CUDNN_STATUS_SUCCESS && bwdFltPref[nIdx].memory > lWsLimitInBytes && nIdx < nAlgCount && nIdx < 10)
		{
			nIdx++;
		}

		if (nIdx == nAlgCount)
			return ERROR_PARAM_OUT_OF_RANGE;

		algoBwdFilter = bwdFltPref[nIdx].algo;
	}
#else
	if (lErr = cudnnGetConvolutionBackwardFilterAlgorithm(cudnn, bottom, top, conv, filter, bwdFltPref, lWsLimitInBytes, &algoBwdFilter))
		return lErr | ERROR_CUDNN_OFFSET;
#endif

	// Get workspace size for backward filter algorithm.
	size_t szBwdFilter = 0;
	if (lErr = cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn, bottom, top, conv, filter, algoBwdFilter, &szBwdFilter))
		return lErr | ERROR_CUDNN_OFFSET;

	// Choose backward data algorithm.
	cudnnConvolutionBwdDataAlgo_t algoBwdData;
#ifdef CUDA11_0
	cudnnConvolutionBwdDataAlgoPerf_t bwdDataPref[10];
	if (lErr = cudnnGetConvolutionBackwardDataAlgorithm_v7(cudnn, filter, top, conv, bottom, 5, &nAlgCount, bwdDataPref))
		return lErr;

	if (lWsLimitInBytes == (((size_t)-1) / 2 + 1) || lWsLimitInBytes >= (SIZE_MAX - 10))
	{
		algoBwdData = bwdDataPref[0].algo;
	}
	else
	{
		int nIdx = 0;
		while (bwdDataPref[nIdx].status == CUDNN_STATUS_SUCCESS && bwdDataPref[nIdx].memory > lWsLimitInBytes && nIdx < nAlgCount && nIdx < 10)
		{
			nIdx++;
		}

		if (nIdx == nAlgCount)
			return ERROR_PARAM_OUT_OF_RANGE;

		algoBwdData = bwdDataPref[nIdx].algo;
	}
#else
	if (lErr = cudnnGetConvolutionBackwardDataAlgorithm(cudnn, filter, top, conv, bottom, bwdDataPref, lWsLimitInBytes, &algoBwdData))
		return lErr | ERROR_CUDNN_OFFSET;
#endif

	// Get workspace size for backward data algorithm.
	size_t szBwdData = 0;
	if (lErr = cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn, filter, top, conv, bottom, algoBwdData, &szBwdData))
		return lErr | ERROR_CUDNN_OFFSET;

	*palgoFwd = (long)algoFwd;
	*plWsSizeFwd = szFwd;
	*palgoBwdFilter = (long)algoBwdFilter;
	*plWsSizeBwdFilter = szBwdFilter;
	*palgoBwdData = (long)algoBwdData;
	*plWsSizeBwdData = szBwdData;

	return cudaStreamSynchronize(0);
}

template long Memory<double>::GetConvolutionInfo(long hHandle, long hBottomDesc, long hFilterDesc, long hConvDesc, long hTopDesc, size_t lWsLimitInBytes, bool bUseTensorCores, long* palgoFwd, size_t* plWsSizeFwd, long* palgoBwdFilter, size_t* plWsSizeBwdFilter, long* palgoBwdData, size_t* plWsSizeBwdData, int nPreferredFwdAlgo);
template long Memory<float>::GetConvolutionInfo(long hHandle, long hBottomDesc, long hFilterDesc, long hConvDesc, long hTopDesc, size_t lWsLimitInBytes, bool bUseTensorCores, long* palgoFwd, size_t* plWsSizeFwd, long* palgoBwdFilter, size_t* plWsSizeBwdFilter, long* palgoBwdData, size_t* plWsSizeBwdData, int nPreferredFwdAlgo);


template <class T>
long Memory<T>::ConvolutionForward(long hHandle, T fAlpha, long hBottomDesc, long hBottomData, int nBottomOffset, long hFilterDesc, long hWeight, int nWeightOffset, long hConvDesc, long algo, long hWorkspace, int nWorkspaceOffset, size_t lWorkspaceSize, T fBeta, long hTopDesc, long hTopData, int nTopOffset, bool bSyncStream)
{
	LONG lErr;
	cudnnHandle_t cudnn = GetCuDNN(hHandle);
	cudnnTensorDescriptor_t btmdesc = GetTensorDesc(hBottomDesc);
	cudnnFilterDescriptor_t filterdesc = GetFilterDesc(hFilterDesc);
	cudnnConvolutionDescriptor_t convdesc = GetConvolutionDesc(hConvDesc);
	cudnnTensorDescriptor_t topdesc = GetTensorDesc(hTopDesc);
	MemoryItem* pBtmData;
	MemoryItem* pTopData;
	MemoryItem* pWeight;
	MemoryItem* pWorkspace = NULL;

	if (lErr = m_memory.GetData(hBottomData, &pBtmData))
		return lErr;

	if (lErr = m_memory.GetData(hTopData, &pTopData))
		return lErr;

	if (lErr = m_memory.GetData(hWeight, &pWeight))
		return lErr;

	T* btmdata = (T*)pBtmData->Data();
	T* topdata = (T*)pTopData->Data();
	T* weight = (T*)pWeight->Data();
	T* wksp = NULL;

	if (hWorkspace != 0)
	{
		if (lErr = m_memory.GetData(hWorkspace, &pWorkspace))
			return lErr;

		wksp = (T*)pWorkspace->Data();
	}
	else if (lWorkspaceSize != 0)
	{
		return ERROR_PARAM_OUT_OF_RANGE;
	}

	if (nBottomOffset > 0)
		btmdata += nBottomOffset;

	if (nTopOffset > 0)
		topdata += nTopOffset;

	if (nWeightOffset > 0)
		weight += nWeightOffset;

	if (wksp != NULL && nWorkspaceOffset > 0)
		wksp += nWorkspaceOffset;

	if (lErr = cudnnConvolutionForward(cudnn, &fAlpha, btmdesc, btmdata, filterdesc, weight, convdesc, (cudnnConvolutionFwdAlgo_t)algo, wksp, lWorkspaceSize, &fBeta, topdesc, topdata))
		return lErr | ERROR_CUDNN_OFFSET;

	if (bSyncStream)
		return cudaStreamSynchronize(0);

	return CUDNN_STATUS_SUCCESS;
}

template long Memory<double>::ConvolutionForward(long hHandle, double dfAlpha, long hBottomDesc, long hBottomData, int nBottomOffset, long hFilterDesc, long hWeight, int nWeightOffset, long hConvDesc, long algo, long hWorkspace, int nWorkspaceOffset, size_t lWorkspaceSize, double dfBeta, long hTopDesc, long hTopData, int nTopOffset, bool bSyncStream);
template long Memory<float>::ConvolutionForward(long hHandle, float fAlpha, long hBottomDesc, long hBottomData, int nBottomOffset, long hFilterDesc, long hWeight, int nWeightOffset, long hConvDesc, long algo, long hWorkspace, int nWorkspaceOffset, size_t lWorkspaceSize, float fBeta, long hTopDesc, long hTopData, int nTopOffset, bool bSyncStream);


template <class T>
long Memory<T>::ConvolutionBackwardBias(long hHandle, T fAlpha, long hTopDesc, long hTopDiff, int nTopOffset, T fBeta, long hBiasDesc, long hBiasDiff, int nBiasOffset, bool bSyncStream)
{
	LONG lErr;
	cudnnHandle_t cudnn = GetCuDNN(hHandle);
	cudnnTensorDescriptor_t topdesc = GetTensorDesc(hTopDesc);
	cudnnTensorDescriptor_t biasdesc = GetTensorDesc(hBiasDesc);
	MemoryItem* pTopDiff;
	MemoryItem* pBiasDiff;

	if (lErr = m_memory.GetData(hTopDiff, &pTopDiff))
		return lErr;

	if (lErr = m_memory.GetData(hBiasDiff, &pBiasDiff))
		return lErr;

	T* topdiff = (T*)pTopDiff->Data();
	T* biasdiff = (T*)pBiasDiff->Data();

	if (nTopOffset > 0)
		topdiff += nTopOffset;

	if (nBiasOffset > 0)
		biasdiff += nBiasOffset;

	if (lErr = cudnnConvolutionBackwardBias(cudnn, &fAlpha, topdesc, topdiff, &fBeta, biasdesc, biasdiff))
		return lErr | ERROR_CUDNN_OFFSET;

	if (bSyncStream)
		return cudaStreamSynchronize(0);

	return CUDNN_STATUS_SUCCESS;
}

template long Memory<double>::ConvolutionBackwardBias(long hHandle, double dfAlpha, long hTopDesc, long hTopDiff, int nTopOffset, double dfBeta, long hBiasDesc, long hBiasDiff, int nBiasOffset, bool bSyncStream);
template long Memory<float>::ConvolutionBackwardBias(long hHandle, float fAlpha, long hTopDesc, long hTopDiff, int nTopOffset, float fBeta, long hBiasDesc, long hBiasDiff, int nBiasOffset, bool bSyncStream);


template <class T>
long Memory<T>::ConvolutionBackwardFilter(long hHandle, T fAlpha, long hBottomDesc, long hBottomData, int nBottomOffset, long hTopDesc, long hTopDiff, int nTopOffset, long hConvDesc, long algo, long hWorkspace, int nWorkspaceOffset, size_t lWorkspaceSize, T fBeta, long hFilterDesc, long hWeightDiff, int nWeightOffset, bool bSyncStream)
{
	LONG lErr;
	cudnnHandle_t cudnn = GetCuDNN(hHandle);
	cudnnTensorDescriptor_t btmdesc = GetTensorDesc(hBottomDesc);
	cudnnTensorDescriptor_t topdesc = GetTensorDesc(hTopDesc);
	cudnnConvolutionDescriptor_t convdesc = GetConvolutionDesc(hConvDesc);
	cudnnFilterDescriptor_t filterdesc = GetFilterDesc(hFilterDesc);
	MemoryItem* pBtmData;
	MemoryItem* pTopDiff;
	MemoryItem* pWeightDiff;
	MemoryItem* pWorkspace = NULL;

	if (lErr = m_memory.GetData(hBottomData, &pBtmData))
		return lErr;

	if (lErr = m_memory.GetData(hTopDiff, &pTopDiff))
		return lErr;

	if (lErr = m_memory.GetData(hWeightDiff, &pWeightDiff))
		return lErr;

	T* btmdata = (T*)pBtmData->Data();
	T* topdiff = (T*)pTopDiff->Data();
	T* weightdiff = (T*)pWeightDiff->Data();
	T* wksp = NULL;
	
	if (hWorkspace != 0)
	{
		if (lErr = m_memory.GetData(hWorkspace, &pWorkspace))
			return lErr;

		wksp = (T*)pWorkspace->Data();
	}
	else if (lWorkspaceSize != 0)
	{
		return ERROR_PARAM_OUT_OF_RANGE;
	}

	if (nBottomOffset > 0)
		btmdata += nBottomOffset;

	if (nTopOffset > 0)
		topdiff += nTopOffset;

	if (nWeightOffset > 0)
		weightdiff += nWeightOffset;

	if (wksp != NULL && nWorkspaceOffset > 0)
		wksp += nWorkspaceOffset;
	
#ifdef CUDNN_5
	if (lErr = cudnnConvolutionBackwardFilter(cudnn, &fAlpha, btmdesc, btmdata, topdesc, topdiff, convdesc, (cudnnConvolutionBwdFilterAlgo_t)algo, wksp, lWorkspaceSize, &fBeta, filterdesc, weightdiff))
		return lErr | ERROR_CUDNN_OFFSET;
#else
	if (lErr = cudnnConvolutionBackwardFilter_v3(cudnn, &fAlpha, btmdesc, btmdata, topdesc, topdiff, convdesc, (cudnnConvolutionBwdFilterAlgo_t)algo, wksp, lWorkspaceSize, &fBeta, filterdesc, weightdiff))
		return lErr | ERROR_CUDNN_OFFSET;
#endif

	if (bSyncStream)
		return cudaStreamSynchronize(0);

	return CUDNN_STATUS_SUCCESS;
}

template long Memory<double>::ConvolutionBackwardFilter(long hHandle, double dfAlpha, long hBottomDesc, long hBottomData, int nBottomOffset, long hTopDesc, long hTopDiff, int nTopOffset, long hConvDesc, long algo, long hWorkspace, int nWorkspaceOffset, size_t lWorkspaceSize, double dfBeta, long hFilterDesc, long hWeightDiff, int nWeightOffset, bool bSyncStream);
template long Memory<float>::ConvolutionBackwardFilter(long hHandle, float fAlpha, long hBottomDesc, long hBottomData, int nBottomOffset, long hTopDesc, long hTopDiff, int nTopOffset, long hConvDesc, long algo, long hWorkspace, int nWorkspaceOffset, size_t lWorkspaceSize, float fBeta, long hFilterDesc, long hWeightDiff, int nWeightOffset, bool bSyncStream);


template <class T>
long Memory<T>::ConvolutionBackwardData(long hHandle, T fAlpha, long hFilterDesc, long hWeight, int nWeightOffset, long hTopDesc, long hTopDiff, int nTopOffset, long hConvDesc, long algo, long hWorkspace, int nWorkspaceOffset, size_t lWorkspaceSize, T fBeta, long hBottomDesc, long hBottomDiff, int nBottomOffset, bool bSyncStream)
{
	LONG lErr;
	cudnnHandle_t cudnn = GetCuDNN(hHandle);
	cudnnFilterDescriptor_t filterdesc = GetFilterDesc(hFilterDesc);
	cudnnTensorDescriptor_t topdesc = GetTensorDesc(hTopDesc);
	cudnnConvolutionDescriptor_t convdesc = GetConvolutionDesc(hConvDesc);
	cudnnTensorDescriptor_t btmdesc = GetTensorDesc(hBottomDesc);
	MemoryItem* pWeight;
	MemoryItem* pTopDiff;
	MemoryItem* pBtmDiff;
	MemoryItem* pWorkspace = NULL;

	if (lErr = m_memory.GetData(hWeight, &pWeight))
		return lErr;

	if (lErr = m_memory.GetData(hTopDiff, &pTopDiff))
		return lErr;

	if (lErr = m_memory.GetData(hBottomDiff, &pBtmDiff))
		return lErr;

	T* weight = (T*)pWeight->Data();
	T* topdiff = (T*)pTopDiff->Data();
	T* btmdiff = (T*)pBtmDiff->Data();
	T* wksp = NULL;

	if (hWorkspace != 0)
	{
		if (lErr = m_memory.GetData(hWorkspace, &pWorkspace))
			return lErr;

		wksp = (T*)pWorkspace->Data();
	}
	else if (lWorkspaceSize != 0)
	{
		return ERROR_PARAM_OUT_OF_RANGE;
	}

	if (nWeightOffset > 0)
		weight += nWeightOffset;

	if (nTopOffset > 0)
		topdiff += nTopOffset;

	if (nBottomOffset > 0)
		btmdiff += nBottomOffset;

	if (wksp != NULL && nWorkspaceOffset > 0)
		wksp += nWorkspaceOffset;

#ifdef CUDNN_5
	if (lErr = cudnnConvolutionBackwardData(cudnn, &fAlpha, filterdesc, weight, topdesc, topdiff, convdesc, (cudnnConvolutionBwdDataAlgo_t)algo, wksp, lWorkspaceSize, &fBeta, btmdesc, btmdiff))
		return lErr | ERROR_CUDNN_OFFSET;
#else
	if (lErr = cudnnConvolutionBackwardData_v3(cudnn, &fAlpha, filterdesc, weight, topdesc, topdiff, convdesc, (cudnnConvolutionBwdDataAlgo_t)algo, wksp, lWorkspaceSize, &fBeta, btmdesc, btmdiff))
		return lErr | ERROR_CUDNN_OFFSET;
#endif

	if (bSyncStream)
		return cudaStreamSynchronize(0);

	return CUDNN_STATUS_SUCCESS;
}

template long Memory<double>::ConvolutionBackwardData(long hHandle, double dfAlpha, long hBottomDesc, long hBottomData, int nBottomOffset, long hTopDesc, long hTopDiff, int nTopOffset, long hConvDesc, long algo, long hWorkspace, int nWorkspaceOffset, size_t lWorkspaceSize, double dfBeta, long hFilterDesc, long hWeightDiff, int nWeightOffset, bool bSyncStream);
template long Memory<float>::ConvolutionBackwardData(long hHandle, float fAlpha, long hBottomDesc, long hBottomData, int nBottomOffset, long hTopDesc, long hTopDiff, int nTopOffset, long hConvDesc, long algo, long hWorkspace, int nWorkspaceOffset, size_t lWorkspaceSize, float fBeta, long hFilterDesc, long hWeightDiff, int nWeightOffset, bool bSyncStream);


template <class T>
long Memory<T>::CreatePoolingDesc(long* phHandle)
{
	LONG lErr;
	cudnnPoolingDescriptor_t desc = NULL;

	if (phHandle == NULL)
		return ERROR_PARAM_NULL;

	if (lErr = cudnnCreatePoolingDescriptor(&desc))
		return lErr | ERROR_CUDNN_OFFSET;

	long hHandle = m_poolDesc.Allocate(desc);
	if (hHandle < 0)
	{
		cudnnDestroyPoolingDescriptor(desc);
		return ERROR_MEMORY_OUT;
	}

	*phHandle = hHandle;
	return 0;
}

template long Memory<double>::CreatePoolingDesc(long* phHandle);
template long Memory<float>::CreatePoolingDesc(long* phHandle);


template <class T>
long Memory<T>::PoolingForward(long hHandle, long hPoolingDesc, T fAlpha, long hBottomDesc, long hBottomData, T fBeta, long hTopDesc, long hTopData)
{
	LONG lErr;
	cudnnHandle_t cudnn = GetCuDNN(hHandle);
	cudnnPoolingDescriptor_t pooldesc = GetPoolingDesc(hPoolingDesc);
	cudnnTensorDescriptor_t topdesc = GetTensorDesc(hTopDesc);
	cudnnTensorDescriptor_t btmdesc = GetTensorDesc(hBottomDesc);
	MemoryItem* pTopData;
	MemoryItem* pBtmData;

	if (lErr = m_memory.GetData(hTopData, &pTopData))
		return lErr;

	if (lErr = m_memory.GetData(hBottomData, &pBtmData))
		return lErr;

	T* topdata = (T*)pTopData->Data();
	T* btmdata = (T*)pBtmData->Data();

	if (lErr = cudnnPoolingForward(cudnn, pooldesc, &fAlpha, btmdesc, btmdata, &fBeta, topdesc, topdata))
		return lErr | ERROR_CUDNN_OFFSET;

	return cudaStreamSynchronize(0);
}

template long Memory<double>::PoolingForward(long hHandle, long hPoolingDesc, double dfAlpha, long hBottomDesc, long hBottomData, double dfBeta, long hTopDesc, long hTopData);
template long Memory<float>::PoolingForward(long hHandle, long hPoolingDesc, float fAlpha, long hBottomDesc, long hBottomData, float fBeta, long hTopDesc, long hTopData);


template <class T>
long Memory<T>::PoolingBackward(long hHandle, long hPoolingDesc, T fAlpha, long hTopDataDesc, long hTopData, long hTopDiffDesc, long hTopDiff, long hBottomDataDesc, long hBottomData, T fBeta, long hBottomDiffDesc, long hBottomDiff)
{
	LONG lErr;
	cudnnHandle_t cudnn = GetCuDNN(hHandle);
	cudnnPoolingDescriptor_t pooldesc = GetPoolingDesc(hPoolingDesc);
	cudnnTensorDescriptor_t topdatadesc = GetTensorDesc(hTopDataDesc);
	cudnnTensorDescriptor_t btmdatadesc = GetTensorDesc(hBottomDataDesc);
	cudnnTensorDescriptor_t topdiffdesc = (hTopDataDesc == hTopDiffDesc) ? topdatadesc : GetTensorDesc(hTopDiffDesc);
	cudnnTensorDescriptor_t btmdiffdesc = (hBottomDataDesc == hBottomDiffDesc) ? btmdatadesc : GetTensorDesc(hBottomDiffDesc);
	MemoryItem* pTopData;
	MemoryItem* pBtmData;
	MemoryItem* pTopDiff;
	MemoryItem* pBtmDiff;

	if (lErr = m_memory.GetData(hTopData, &pTopData))
		return lErr;

	if (lErr = m_memory.GetData(hBottomData, &pBtmData))
		return lErr;

	if (lErr = m_memory.GetData(hTopDiff, &pTopDiff))
		return lErr;

	if (lErr = m_memory.GetData(hBottomDiff, &pBtmDiff))
		return lErr;

	T* topdata = (T*)pTopData->Data();
	T* btmdata = (T*)pBtmData->Data();
	T* topdiff = (T*)pTopDiff->Data();
	T* btmdiff = (T*)pBtmDiff->Data();

	if (lErr = cudnnPoolingBackward(cudnn, pooldesc, &fAlpha, topdatadesc, topdata, topdiffdesc, topdiff, btmdatadesc, btmdata, &fBeta, btmdiffdesc, btmdiff))
		return lErr | ERROR_CUDNN_OFFSET;

	return cudaStreamSynchronize(0);
}

template long Memory<double>::PoolingBackward(long hHandle, long hPoolingDesc, double dfAlpha, long hTopDataDesc, long hTopData, long hTopDiffDesc, long hTopDiff, long hBottomDataDesc, long hBottomData, double dfBeta, long hBottomDiffDesc, long hBottomDiff);
template long Memory<float>::PoolingBackward(long hHandle, long hPoolingDesc, float fAlpha, long hTopDataDesc, long hTopData, long hTopDiffDesc, long hTopDiff, long hBottomDataDesc, long hBottomData, float fBeta, long hBottomDiffDesc, long hBottomDiff);

template <class T>
long Memory<T>::DeriveBatchNormDesc(long hFwdScaleBiasMeanVarDesc, long hFwdBottomDesc, long hBwdScaleBiasMeanVarDesc, long hBwdBottomDesc, int mode)
{
	LONG lErr;
	cudnnTensorDescriptor_t fwdscalemeanvardesc = GetTensorDesc(hFwdScaleBiasMeanVarDesc);
	cudnnTensorDescriptor_t fwdbtmdesc = GetTensorDesc(hFwdBottomDesc);
	cudnnTensorDescriptor_t bwdscalemeanvardesc = GetTensorDesc(hBwdScaleBiasMeanVarDesc);
	cudnnTensorDescriptor_t bwdbtmdesc = GetTensorDesc(hBwdBottomDesc);

	if (lErr = cudnnDeriveBNTensorDescriptor(fwdscalemeanvardesc, fwdbtmdesc, (cudnnBatchNormMode_t)mode))
		return lErr;

	if (lErr = cudnnDeriveBNTensorDescriptor(bwdscalemeanvardesc, bwdbtmdesc, (cudnnBatchNormMode_t)mode))
		return lErr;

	return CUDNN_STATUS_SUCCESS;
}

template long Memory<double>::DeriveBatchNormDesc(long hFwdScaleBiasMeanVarDesc, long hFwdBottomDesc, long hBwdScaleBiasMeanVarDesc, long hBwdBottomDesc, int mode);
template long Memory<float>::DeriveBatchNormDesc(long hFwdScaleBiasMeanVarDesc, long hFwdBottomDesc, long hBwdScaleBiasMeanVarDesc, long hBwdBottomDesc, int mode);


template <class T>
long Memory<T>::BatchNormForward(long hHandle, int mode, T fAlpha, T fBeta, long hFwdBottomDesc, long hBottomData, long hFwdTopDesc, long hTopData, long hFwdScaleBiasMeanVarDesc, long hScaleData, long hBiasData, T fFactor, long hGlobalMean, long hGlobalVar, T fEps, long hSaveMean, long hSaveVar, bool bTraining)
{
	LONG lErr;
	cudnnHandle_t cudnn = GetCuDNN(hHandle);
	cudnnTensorDescriptor_t fwdbtmdesc = GetTensorDesc(hFwdBottomDesc);
	cudnnTensorDescriptor_t fwdtopdesc = GetTensorDesc(hFwdTopDesc);
	cudnnTensorDescriptor_t fwdscalemeanvardesc = GetTensorDesc(hFwdScaleBiasMeanVarDesc);
	MemoryItem* pBtmData;
	MemoryItem* pTopData;
	MemoryItem* pScaleData;
	MemoryItem* pBiasData;
	MemoryItem* pGlobalMean;
	MemoryItem* pGlobalVar;
	MemoryItem* pSaveMean;
	MemoryItem* pSaveVar;

	if (lErr = m_memory.GetData(hBottomData, &pBtmData))
		return lErr;

	if (lErr = m_memory.GetData(hTopData, &pTopData))
		return lErr;

	if (lErr = m_memory.GetData(hScaleData, &pScaleData))
		return lErr;

	if (lErr = m_memory.GetData(hBiasData, &pBiasData))
		return lErr;

	if (lErr = m_memory.GetData(hGlobalMean, &pGlobalMean))
		return lErr;

	if (lErr = m_memory.GetData(hGlobalVar, &pGlobalVar))
		return lErr;

	T* btmdata = (T*)pBtmData->Data();
	T* topdata = (T*)pTopData->Data();
	T* scaledata = (T*)pScaleData->Data();
	T* biasdata = (T*)pBiasData->Data();
	T* globalmean = (T*)pGlobalMean->Data();
	T* globalvar = (T*)pGlobalVar->Data();

	if (sizeof(T) == 4)
	{
		if ((float)fEps < CUDNN_BN_MIN_EPSILON)
			fEps = 0.0001f;
	}

	if (bTraining)
	{
		if (lErr = m_memory.GetData(hSaveMean, &pSaveMean))
			return lErr;

		if (lErr = m_memory.GetData(hSaveVar, &pSaveVar))
			return lErr;

		T* savemean = (T*)pSaveMean->Data();
		T* savevar = (T*)pSaveVar->Data();

		if (lErr = cudnnBatchNormalizationForwardTraining(cudnn, (cudnnBatchNormMode_t)mode, &fAlpha, &fBeta, fwdbtmdesc, btmdata, fwdtopdesc, topdata, fwdscalemeanvardesc, scaledata, biasdata, fFactor, globalmean, globalvar, fEps, savemean, savevar))
			return lErr;
	}
	else
	{
		if (lErr = cudnnBatchNormalizationForwardInference(cudnn, (cudnnBatchNormMode_t)mode, &fAlpha, &fBeta, fwdbtmdesc, btmdata, fwdtopdesc, topdata, fwdscalemeanvardesc, scaledata, biasdata, globalmean, globalvar, fEps))
			return lErr;
	}

	return cudaStreamSynchronize(0);
}

template long Memory<double>::BatchNormForward(long hHandle, int mode, double dfAlpha, double dfBeta, long hFwdBottomDesc, long hBottomData, long hFwdTopDesc, long hTopData, long hFwdScaleBiasMeanVarDesc, long hScaleData, long hBiasData, double fFactor, long hGlobalMean, long hGlobalVar, double fEps, long hSaveMean, long hSaveVar, bool bTraining);
template long Memory<float>::BatchNormForward(long hHandle, int mode, float fAlpha, float fBeta, long hFwdBottomDesc, long hBottomData, long hFwdTopDesc, long hTopData, long hFwdScaleBiasMeanVarDesc, long hScaleData, long hBiasData, float fFactor, long hGlobalMean, long hGlobalVar, float fEps, long hSaveMean, long hSaveVar, bool bTraining);

template <class T>
long Memory<T>::BatchNormBackward(long hHandle, int mode, T fAlphaDiff, T fBetaDiff, T fAlphaParamDiff, T fBetaParamDiff, long hBwdBottomDesc, long hBottomData, long hTopDiffDesc, long hTopDiff, long hBottomDiffDesc, long hBottomDiff, long hBwdScaleBiasMeanVarDesc, long hScaleData, long hScaleDiff, long hBiasDiff, T fEps, long hSaveMean, long hSaveVar)
{
	LONG lErr;
	cudnnHandle_t cudnn = GetCuDNN(hHandle);
	cudnnTensorDescriptor_t bwdbtmdesc = GetTensorDesc(hBwdBottomDesc);
	cudnnTensorDescriptor_t topdiffdesc = GetTensorDesc(hTopDiffDesc);
	cudnnTensorDescriptor_t btmdiffdesc = GetTensorDesc(hBottomDiffDesc);
	cudnnTensorDescriptor_t bwdscalemeanvardesc = GetTensorDesc(hBwdScaleBiasMeanVarDesc);
	MemoryItem* pBtmData;
	MemoryItem* pTopDiff;
	MemoryItem* pBtmDiff;
	MemoryItem* pScaleData;
	MemoryItem* pScaleDiff;
	MemoryItem* pBiasDiff;
	MemoryItem* pSaveMean = NULL;
	MemoryItem* pSaveVar = NULL;

	if (lErr = m_memory.GetData(hBottomData, &pBtmData))
		return lErr;

	if (lErr = m_memory.GetData(hTopDiff, &pTopDiff))
		return lErr;

	if (lErr = m_memory.GetData(hBottomDiff, &pBtmDiff))
		return lErr;

	if (lErr = m_memory.GetData(hScaleData, &pScaleData))
		return lErr;

	if (lErr = m_memory.GetData(hScaleDiff, &pScaleDiff))
		return lErr;

	if (lErr = m_memory.GetData(hBiasDiff, &pBiasDiff))
		return lErr;

	T* btmdata = (T*)pBtmData->Data();
	T* topdiff = (T*)pTopDiff->Data();
	T* btmdiff = (T*)pBtmDiff->Data();
	T* scaledata = (T*)pScaleData->Data();
	T* scalediff = (T*)pScaleDiff->Data();
	T* biasdiff = (T*)pBiasDiff->Data();
	T* savemean = NULL;
	T* savevar = NULL;

	if (hSaveMean != 0 && hSaveVar != 0)
	{
		if (lErr = m_memory.GetData(hSaveMean, &pSaveMean))
			return lErr;

		if (lErr = m_memory.GetData(hSaveVar, &pSaveVar))
			return lErr;

		savemean = (T*)pSaveMean->Data();
		savevar = (T*)pSaveVar->Data();
	}

	if (sizeof(T) == 4)
	{
		if ((float)fEps < CUDNN_BN_MIN_EPSILON)
			fEps = 0.0001f;
	}

	if (lErr = cudnnBatchNormalizationBackward(cudnn, (cudnnBatchNormMode_t)mode, &fAlphaDiff, &fBetaDiff, &fAlphaParamDiff, &fBetaParamDiff, bwdbtmdesc, btmdata, topdiffdesc, topdiff, btmdiffdesc, btmdiff, bwdscalemeanvardesc, scaledata, scalediff, biasdiff, fEps, savemean, savevar))
		return lErr;

	return cudaStreamSynchronize(0);
}

template long Memory<double>::BatchNormBackward(long hHandle, int mode, double dfAlphaDiff, double dfBetaDiff, double dfAlphaParamDiff, double dfBetaParamDiff, long hBtmBottomDesc, long hBottomData, long hTopDiffDesc, long hTopDiff, long hBottomDiffDesc, long hBottomDiff, long hBwdScaleBiasMeanVarDesc, long hScaleData, long hScaleDiff, long hBiasDiff, double fEps, long hSaveMean, long hSaveVar);
template long Memory<float>::BatchNormBackward(long hHandle, int mode, float fAlphaDiff, float fBetaDiff, float fAlphaParamDiff, float fBetaParamDiff, long hBtmBottomDesc, long hBottomData, long hTopDiffDesc, long hTopDiff, long hBottomDiffDesc, long hBottomDiff, long hBwdScaleBiasMeanVarDesc, long hScaleData, long hScaleDiff, long hBiasDiff, float fEps, long hSaveMean, long hSaveVar);


template <class T>
long Memory<T>::CreateDropoutDesc(long* phHandle)
{
	LONG lErr;
	cudnnDropoutDescriptor_t desc = NULL;

	if (phHandle == NULL)
		return ERROR_PARAM_NULL;

	if (lErr = cudnnCreateDropoutDescriptor(&desc))
		return lErr | ERROR_CUDNN_OFFSET;

	long hHandle = m_dropoutDesc.Allocate(desc);
	if (hHandle < 0)
	{
		cudnnDestroyDropoutDescriptor(desc);
		return ERROR_MEMORY_OUT;
	}

	*phHandle = hHandle;
	return 0;
}

template long Memory<double>::CreateDropoutDesc(long* phHandle);
template long Memory<float>::CreateDropoutDesc(long* phHandle);


template <class T>
long Memory<T>::SetDropoutDesc(long hHandle, long hDropoutDesc, T fDropout, long hStates, long lSeed)
{
	LONG lErr;
	cudnnHandle_t cudnn = GetCuDNN(hHandle);
	cudnnDropoutDescriptor_t desc = GetDropoutDesc(hDropoutDesc);
	MemoryItem* pStates;
	
	if (lErr = m_memory.GetData(hStates, &pStates))
		return lErr;

	T* states = (T*)pStates->Data();
	size_t szStates = (size_t)pStates->Size();

	if (lErr = cudnnSetDropoutDescriptor(desc, cudnn, (float)fDropout, states, szStates, (unsigned long long)lSeed))
		return lErr | ERROR_CUDNN_OFFSET;

	return CUDNN_STATUS_SUCCESS;
}

template long Memory<double>::SetDropoutDesc(long hHandle, long hDropoutDesc, double fDropout, long hStates, long lSeed);
template long Memory<float>::SetDropoutDesc(long hHandle, long hDropoutDesc, float fDropout, long hStates, long lSeed);


template <class T>
long Memory<T>::GetDropoutInfo(long hHandle, long hBottomDesc, unsigned long* plState, unsigned long* plReserved)
{
	LONG lErr;
	cudnnHandle_t cudnn = GetCuDNN(hHandle);
	cudnnTensorDescriptor_t bottomDesc = NULL;
	size_t szStates = 0;
	size_t szReserved = 0;

	if (hBottomDesc > 0)
		bottomDesc = GetTensorDesc(hBottomDesc);

	if (plState == NULL || plReserved == NULL)
		return ERROR_PARAM_NULL;

	if (lErr = cudnnDropoutGetStatesSize(cudnn, &szStates))
		return lErr | ERROR_CUDNN_OFFSET;

	if (bottomDesc != NULL)
	{
		if (lErr = cudnnDropoutGetReserveSpaceSize(bottomDesc, &szReserved))
			return lErr | ERROR_CUDNN_OFFSET;
	}

	*plState = (unsigned long)szStates;
	*plReserved = (unsigned long)szReserved;

	return 0;
}

template long Memory<double>::GetDropoutInfo(long hHandle, long hBottomDesc, unsigned long* plState, unsigned long* plReserved);
template long Memory<float>::GetDropoutInfo(long hHandle, long hBottomDesc, unsigned long* plState, unsigned long* plReserved);


template <class T>
long Memory<T>::DropoutForward(long hHandle, long hDropoutDesc, long hBottomDesc, long hBottom, long hTopDesc, long hTop, long hReservedSpace)
{
	LONG lErr;
	cudnnHandle_t cudnn = GetCuDNN(hHandle);
	cudnnDropoutDescriptor_t desc = GetDropoutDesc(hDropoutDesc);
	cudnnTensorDescriptor_t bottomDesc = GetTensorDesc(hBottomDesc);
	cudnnTensorDescriptor_t topDesc = GetTensorDesc(hTopDesc);
	MemoryItem* pBottom;
	MemoryItem* pTop;
	MemoryItem* pReserved;

	if (lErr = m_memory.GetData(hBottom, &pBottom))
		return lErr;

	if (lErr = m_memory.GetData(hTop, &pTop))
		return lErr;

	if (lErr = m_memory.GetData(hReservedSpace, &pReserved))
		return lErr;

	T* bottom = (T*)pBottom->Data();
	T* top = (T*)pTop->Data();
	T* reserved = (T*)pReserved->Data();
	size_t szReserved = (size_t)pReserved->Size();

	if (lErr = cudnnDropoutForward(cudnn, desc, bottomDesc, bottom, topDesc, top, reserved, szReserved))
		return lErr | ERROR_CUDNN_OFFSET;

	return cudaStreamSynchronize(0);
}

template long Memory<double>::DropoutForward(long hHandle, long hDropoutDesc, long hBottomDesc, long hBottom, long hTopDesc, long hTop, long hReservedSpace);
template long Memory<float>::DropoutForward(long hHandle, long hDropoutDesc, long hBottomDesc, long hBottom, long hTopDesc, long hTop, long hReservedSpace);


template <class T>
long Memory<T>::DropoutBackward(long hHandle, long hDropoutDesc, long hTopDesc, long hTop, long hBottomDesc, long hBottom, long hReservedSpace)
{
	LONG lErr;
	cudnnHandle_t cudnn = GetCuDNN(hHandle);
	cudnnDropoutDescriptor_t desc = GetDropoutDesc(hDropoutDesc);
	cudnnTensorDescriptor_t topDesc = GetTensorDesc(hTopDesc);
	cudnnTensorDescriptor_t bottomDesc = GetTensorDesc(hBottomDesc);
	MemoryItem* pTop;
	MemoryItem* pBottom;
	MemoryItem* pReserved;

	if (lErr = m_memory.GetData(hTop, &pTop))
		return lErr;

	if (lErr = m_memory.GetData(hBottom, &pBottom))
		return lErr;

	if (lErr = m_memory.GetData(hReservedSpace, &pReserved))
		return lErr;

	T* top = (T*)pTop->Data();
	T* bottom = (T*)pBottom->Data();
	T* reserved = (T*)pReserved->Data();
	size_t szReserved = (size_t)pReserved->Size();

	if (lErr = cudnnDropoutBackward(cudnn, desc, topDesc, top, bottomDesc, bottom, reserved, szReserved))
		return lErr | ERROR_CUDNN_OFFSET;

	return cudaStreamSynchronize(0);
}

template long Memory<double>::DropoutBackward(long hHandle, long hDropoutDesc, long hTopDesc, long hTop, long hBottomDesc, long hBottom, long hReservedSpace);
template long Memory<float>::DropoutBackward(long hHandle, long hDropoutDesc, long hTopDesc, long hTop, long hBottomDesc, long hBottom, long hReservedSpace);


template <class T>
long Memory<T>::CreateLRNDesc(long* phHandle)
{
	LONG lErr;
	cudnnLRNDescriptor_t desc = NULL;

	if (phHandle == NULL)
		return ERROR_PARAM_NULL;

	if (lErr = cudnnCreateLRNDescriptor(&desc))
		return lErr | ERROR_CUDNN_OFFSET;

	long hHandle = m_lrnDesc.Allocate(desc);
	if (hHandle < 0)
	{
		cudnnDestroyLRNDescriptor(desc);
		return ERROR_MEMORY_OUT;
	}

	*phHandle = hHandle;
	return 0;
}

template long Memory<double>::CreateLRNDesc(long* phHandle);
template long Memory<float>::CreateLRNDesc(long* phHandle);


template <class T> 
long Memory<T>::LRNForwardCC(long hHandle, long hNormDesc, T fAlpha, long hBottomDataDesc, long hBottomData, T fBeta, long hTopDataDesc, long hTopData)
{
	LONG lErr;
	cudnnHandle_t cudnn = GetCuDNN(hHandle);
	cudnnLRNDescriptor_t normdesc = GetLRNDesc(hNormDesc);
	cudnnTensorDescriptor_t topdatadesc = GetTensorDesc(hTopDataDesc);
	cudnnTensorDescriptor_t btmdatadesc = GetTensorDesc(hBottomDataDesc);
	MemoryItem* pBottomData;
	MemoryItem* pTopData;

	if (lErr = m_memory.GetData(hTopData, &pTopData))
		return lErr;

	if (lErr = m_memory.GetData(hBottomData, &pBottomData))
		return lErr;

	T* topdata = (T*)pTopData->Data();
	T* btmdata = (T*)pBottomData->Data();

	if (lErr = cudnnLRNCrossChannelForward(cudnn, normdesc, CUDNN_LRN_CROSS_CHANNEL_DIM1, &fAlpha, btmdatadesc, btmdata, &fBeta, topdatadesc, topdata))
		return lErr | ERROR_CUDNN_OFFSET;

	return cudaStreamSynchronize(0);
}

template long Memory<double>::LRNForwardCC(long hHandle, long hNormDesc, double fAlpha, long hBottomDesc, long hBottomData, double fBeta, long hTopDesc, long hTopData);
template long Memory<float>::LRNForwardCC(long hHandle, long hNormDesc, float fAlpha, long hBottomDesc, long hBottomData, float fBeta, long hTopDesc, long hTopData);


template <class T> 
long Memory<T>::LRNBackwardCC(long hHandle, long hNormDesc, T fAlpha, long hTopDataDesc, long hTopData, long hTopDiffDesc, long hTopDiff, long hBottomDataDesc, long hBottomData, T fBeta, long hBottomDiffDesc, long hBottomDiff)
{
	LONG lErr;
	cudnnHandle_t cudnn = GetCuDNN(hHandle);
	cudnnLRNDescriptor_t normdesc = GetLRNDesc(hNormDesc);
	cudnnTensorDescriptor_t topdatadesc = GetTensorDesc(hTopDataDesc);
	cudnnTensorDescriptor_t btmdatadesc = GetTensorDesc(hBottomDataDesc);
	cudnnTensorDescriptor_t topdiffdesc = (hTopDataDesc == hTopDiffDesc) ? topdatadesc : GetTensorDesc(hTopDiffDesc);
	cudnnTensorDescriptor_t btmdiffdesc = (hBottomDataDesc == hBottomDiffDesc) ? btmdatadesc : GetTensorDesc(hBottomDiffDesc);
	MemoryItem* pTopData;
	MemoryItem* pBtmData;
	MemoryItem* pTopDiff;
	MemoryItem* pBtmDiff;

	if (lErr = m_memory.GetData(hTopData, &pTopData))
		return lErr;

	if (lErr = m_memory.GetData(hBottomData, &pBtmData))
		return lErr;

	if (lErr = m_memory.GetData(hTopDiff, &pTopDiff))
		return lErr;

	if (lErr = m_memory.GetData(hBottomDiff, &pBtmDiff))
		return lErr;

	T* topdata = (T*)pTopData->Data();
	T* btmdata = (T*)pBtmData->Data();
	T* topdiff = (T*)pTopDiff->Data();
	T* btmdiff = (T*)pBtmDiff->Data();

	if (lErr = cudnnLRNCrossChannelBackward(cudnn, normdesc, CUDNN_LRN_CROSS_CHANNEL_DIM1, &fAlpha, topdatadesc, topdata, topdiffdesc, topdiff, btmdatadesc, btmdata, &fBeta, btmdiffdesc, btmdiff))
		return lErr | ERROR_CUDNN_OFFSET;

	return cudaStreamSynchronize(0);
}

template long Memory<double>::LRNBackwardCC(long hHandle, long hNormDesc, double fAlpha, long hTopDataDesc, long hTopData, long hTopDiffDesc, long hTopDiff, long hBottomDataDesc, long hBottomDadta, double fBeta, long hBottomDiffDesc, long hBottomDiff);
template long Memory<float>::LRNBackwardCC(long hHandle, long hNormDesc, float fAlpha, long hTopDataDesc, long hTopData, long hTopDiffDesc, long hTopDiff, long hBottomDataDesc, long hBottomDadta, float fBeta, long hBottomDiffDesc, long hBottomDiff);


template <class T> 
long Memory<T>::LCNForwardCC(long hHandle, long hNormDesc, T fAlpha, long hBottomDataDesc, long hBottomData, long hTemp1, long hTemp2, T fBeta, long hTopDataDesc, long hTopData)
{
	LONG lErr;
	cudnnHandle_t cudnn = GetCuDNN(hHandle);
	cudnnLRNDescriptor_t normdesc = GetLRNDesc(hNormDesc);
	cudnnTensorDescriptor_t topdatadesc = GetTensorDesc(hTopDataDesc);
	cudnnTensorDescriptor_t btmdatadesc = GetTensorDesc(hBottomDataDesc);
	MemoryItem* pBottomData;
	MemoryItem* pTopData;
	MemoryItem* pTemp1;
	MemoryItem* pTemp2;

	if (lErr = m_memory.GetData(hTopData, &pTopData))
		return lErr;

	if (lErr = m_memory.GetData(hBottomData, &pBottomData))
		return lErr;

	if (lErr = m_memory.GetData(hTemp1, &pTemp1))
		return lErr;

	if (lErr = m_memory.GetData(hTemp2, &pTemp2))
		return lErr;

	T* topdata = (T*)pTopData->Data();
	T* btmdata = (T*)pBottomData->Data();
	T* temp1 = (T*)pTemp1->Data();
	T* temp2 = (T*)pTemp2->Data();

	if (lErr = cudnnDivisiveNormalizationForward(cudnn, normdesc, CUDNN_DIVNORM_PRECOMPUTED_MEANS, &fAlpha, btmdatadesc, btmdata, NULL, temp1, temp2, &fBeta, topdatadesc, topdata))
		return lErr | ERROR_CUDNN_OFFSET;

	return cudaStreamSynchronize(0);
}

template long Memory<double>::LCNForwardCC(long hHandle, long hNormDesc, double fAlpha, long hBottomDesc, long hBottomData, long hTemp1, long hTemp2, double fBeta, long hTopDesc, long hTopData);
template long Memory<float>::LCNForwardCC(long hHandle, long hNormDesc, float fAlpha, long hBottomDesc, long hBottomData, long hTemp1, long hTemp2, float fBeta, long hTopDesc, long hTopData);


template <class T> 
long Memory<T>::LCNBackwardCC(long hHandle, long hNormDesc, T fAlpha, long hBottomDataDesc, long hBottomData, long hTopDiff, long hTemp1, long hTemp2, T fBeta, long hBottomDiffDesc, long hBottomDiff)
{
	LONG lErr;
	cudnnHandle_t cudnn = GetCuDNN(hHandle);
	cudnnLRNDescriptor_t normdesc = GetLRNDesc(hNormDesc);
	cudnnTensorDescriptor_t btmdatadesc = GetTensorDesc(hBottomDataDesc);
	cudnnTensorDescriptor_t btmdiffdesc = (hBottomDataDesc == hBottomDiffDesc) ? btmdatadesc : GetTensorDesc(hBottomDiffDesc);
	MemoryItem* pBtmData;
	MemoryItem* pTopDiff;
	MemoryItem* pBtmDiff;
	MemoryItem* pTemp1;
	MemoryItem* pTemp2;

	if (lErr = m_memory.GetData(hBottomData, &pBtmData))
		return lErr;

	if (lErr = m_memory.GetData(hTopDiff, &pTopDiff))
		return lErr;

	if (lErr = m_memory.GetData(hBottomDiff, &pBtmDiff))
		return lErr;

	if (lErr = m_memory.GetData(hTemp1, &pTemp1))
		return lErr;

	if (lErr = m_memory.GetData(hTemp2, &pTemp2))
		return lErr;

	T* btmdata = (T*)pBtmData->Data();
	T* topdiff = (T*)pTopDiff->Data();
	T* btmdiff = (T*)pBtmDiff->Data();
	T* temp1 = (T*)pTemp1->Data();
	T* temp2 = (T*)pTemp2->Data();

	if (lErr = cudnnDivisiveNormalizationBackward(cudnn, normdesc, CUDNN_DIVNORM_PRECOMPUTED_MEANS, &fAlpha, btmdatadesc, btmdata, NULL, topdiff, temp1, temp2, &fBeta, btmdiffdesc, btmdiff, NULL))
		return lErr | ERROR_CUDNN_OFFSET;

	return cudaStreamSynchronize(0);
}

template long Memory<double>::LCNBackwardCC(long hHandle, long hNormDesc, double fAlpha, long hBottomDataDesc, long hBottomData, long hTopDiff, long hTemp1, long hTemp2, double fBeta, long hBottomDiffDesc, long hBottomDiff);
template long Memory<float>::LCNBackwardCC(long hHandle, long hNormDesc, float fAlpha, long hBottomDataDesc, long hBottomData, long hTopDiff, long hTemp1, long hTemp2, float fBeta, long hBottomDiffDesc, long hBottomDiff);


template <class T>
long Memory<T>::TanhForward(long hHandle, T fAlpha, long hBottomDesc, long hBottomData, T fBeta, long hTopDesc, long hTopData)
{
	LONG lErr;
	cudnnHandle_t cudnn = GetCuDNN(hHandle);
	cudnnTensorDescriptor_t topdesc = GetTensorDesc(hTopDesc);
	cudnnTensorDescriptor_t btmdesc = GetTensorDesc(hBottomDesc);
	MemoryItem* pTopData;
	MemoryItem* pBtmData;

	if (lErr = m_memory.GetData(hTopData, &pTopData))
		return lErr;

	if (lErr = m_memory.GetData(hBottomData, &pBtmData))
		return lErr;

	T* topdata = (T*)pTopData->Data();
	T* btmdata = (T*)pBtmData->Data();

#ifdef CUDNN_5
	cudnnActivationDescriptor_t desc = GetActivationDesc(m_hGlobalActivationTanh);
	if (lErr = cudnnActivationForward(cudnn, desc, &fAlpha, btmdesc, btmdata, &fBeta, topdesc, topdata))
		return lErr | ERROR_CUDNN_OFFSET;
#else
	if (lErr = cudnnActivationForward(cudnn, CUDNN_ACTIVATION_TANH, &fAlpha, btmdesc, btmdata, &fBeta, topdesc, topdata))
		return lErr | ERROR_CUDNN_OFFSET;
#endif

	return cudaStreamSynchronize(0);
}

template long Memory<double>::TanhForward(long hHandle, double dfAlpha, long hBottomDesc, long hBottomData, double dfBeta, long hTopDesc, long hTopData);
template long Memory<float>::TanhForward(long hHandle, float fAlpha, long hBottomDesc, long hBottomData, float fBeta, long hTopDesc, long hTopData);


template <class T>
long Memory<T>::TanhBackward(long hHandle, T fAlpha, long hTopDataDesc, long hTopData, long hTopDiffDesc, long hTopDiff, long hBottomDataDesc, long hBottomData, T fBeta, long hBottomDiffDesc, long hBottomDiff)
{
	LONG lErr;
	cudnnHandle_t cudnn = GetCuDNN(hHandle);
	cudnnTensorDescriptor_t topdatadesc = GetTensorDesc(hTopDataDesc);
	cudnnTensorDescriptor_t btmdatadesc = GetTensorDesc(hBottomDataDesc);
	cudnnTensorDescriptor_t topdiffdesc = (hTopDataDesc == hTopDiffDesc) ? topdatadesc : GetTensorDesc(hTopDiffDesc);
	cudnnTensorDescriptor_t btmdiffdesc = (hBottomDataDesc == hBottomDiffDesc) ? btmdatadesc : GetTensorDesc(hBottomDiffDesc);
	MemoryItem* pTopData;
	MemoryItem* pBtmData;
	MemoryItem* pTopDiff;
	MemoryItem* pBtmDiff;

	if (lErr = m_memory.GetData(hTopData, &pTopData))
		return lErr;

	if (lErr = m_memory.GetData(hBottomData, &pBtmData))
		return lErr;

	if (lErr = m_memory.GetData(hTopDiff, &pTopDiff))
		return lErr;

	if (lErr = m_memory.GetData(hBottomDiff, &pBtmDiff))
		return lErr;

	T* topdata = (T*)pTopData->Data();
	T* btmdata = (T*)pBtmData->Data();
	T* topdiff = (T*)pTopDiff->Data();
	T* btmdiff = (T*)pBtmDiff->Data();

#ifdef CUDNN_5
	cudnnActivationDescriptor_t desc = GetActivationDesc(m_hGlobalActivationTanh);
	if (lErr = cudnnActivationBackward(cudnn, desc, &fAlpha, topdatadesc, topdata, topdiffdesc, topdiff, btmdatadesc, btmdata, &fBeta, btmdiffdesc, btmdiff))
		return lErr | ERROR_CUDNN_OFFSET;
#else
	if (lErr = cudnnActivationBackward(cudnn, CUDNN_ACTIVATION_TANH, &fAlpha, topdatadesc, topdata, topdiffdesc, topdiff, btmdatadesc, btmdata, &fBeta, btmdiffdesc, btmdiff))
		return lErr | ERROR_CUDNN_OFFSET;
#endif

	return cudaStreamSynchronize(0);
}

template long Memory<double>::TanhBackward(long hHandle, double dfAlpha, long hTopDataDesc, long hTopData, long hTopDiffDesc, long hTopDiff, long hBottomDataDesc, long hBottomData, double dfBeta, long hBottomDiffDesc, long hBottomDiff);
template long Memory<float>::TanhBackward(long hHandle, float fAlpha, long hTopDataDesc, long hTopData, long hTopDiffDesc, long hTopDiff, long hBottomDataDesc, long hBottomData, float fBeta, long hBottomDiffDesc, long hBottomDiff);


template <class T>
long Memory<T>::EluForward(long hHandle, T fAlpha, long hBottomDesc, long hBottomData, T fBeta, long hTopDesc, long hTopData)
{
	LONG lErr;
	cudnnHandle_t cudnn = GetCuDNN(hHandle);
	cudnnTensorDescriptor_t topdesc = GetTensorDesc(hTopDesc);
	cudnnTensorDescriptor_t btmdesc = GetTensorDesc(hBottomDesc);
	MemoryItem* pTopData;
	MemoryItem* pBtmData;

	if (lErr = m_memory.GetData(hTopData, &pTopData))
		return lErr;

	if (lErr = m_memory.GetData(hBottomData, &pBtmData))
		return lErr;

	T* topdata = (T*)pTopData->Data();
	T* btmdata = (T*)pBtmData->Data();

#ifdef CUDNN_5
	cudnnActivationDescriptor_t desc = GetActivationDesc(m_hGlobalActivationElu);
	if (lErr = cudnnActivationForward(cudnn, desc, &fAlpha, btmdesc, btmdata, &fBeta, topdesc, topdata))
		return lErr | ERROR_CUDNN_OFFSET;
#else
	if (lErr = cudnnActivationForward(cudnn, CUDNN_ACTIVATION_ELU, &fAlpha, btmdesc, btmdata, &fBeta, topdesc, topdata))
		return lErr | ERROR_CUDNN_OFFSET;
#endif

	return cudaStreamSynchronize(0);
}

template long Memory<double>::EluForward(long hHandle, double dfAlpha, long hBottomDesc, long hBottomData, double dfBeta, long hTopDesc, long hTopData);
template long Memory<float>::EluForward(long hHandle, float fAlpha, long hBottomDesc, long hBottomData, float fBeta, long hTopDesc, long hTopData);


template <class T>
long Memory<T>::EluBackward(long hHandle, T fAlpha, long hTopDataDesc, long hTopData, long hTopDiffDesc, long hTopDiff, long hBottomDataDesc, long hBottomData, T fBeta, long hBottomDiffDesc, long hBottomDiff)
{
	LONG lErr;
	cudnnHandle_t cudnn = GetCuDNN(hHandle);
	cudnnTensorDescriptor_t topdatadesc = GetTensorDesc(hTopDataDesc);
	cudnnTensorDescriptor_t btmdatadesc = GetTensorDesc(hBottomDataDesc);
	cudnnTensorDescriptor_t topdiffdesc = (hTopDataDesc == hTopDiffDesc) ? topdatadesc : GetTensorDesc(hTopDiffDesc);
	cudnnTensorDescriptor_t btmdiffdesc = (hBottomDataDesc == hBottomDiffDesc) ? btmdatadesc : GetTensorDesc(hBottomDiffDesc);
	MemoryItem* pTopData;
	MemoryItem* pBtmData;
	MemoryItem* pTopDiff;
	MemoryItem* pBtmDiff;

	if (lErr = m_memory.GetData(hTopData, &pTopData))
		return lErr;

	if (lErr = m_memory.GetData(hBottomData, &pBtmData))
		return lErr;

	if (lErr = m_memory.GetData(hTopDiff, &pTopDiff))
		return lErr;

	if (lErr = m_memory.GetData(hBottomDiff, &pBtmDiff))
		return lErr;

	T* topdata = (T*)pTopData->Data();
	T* btmdata = (T*)pBtmData->Data();
	T* topdiff = (T*)pTopDiff->Data();
	T* btmdiff = (T*)pBtmDiff->Data();

#ifdef CUDNN_5
	cudnnActivationDescriptor_t desc = GetActivationDesc(m_hGlobalActivationElu);
	if (lErr = cudnnActivationBackward(cudnn, desc, &fAlpha, topdatadesc, topdata, topdiffdesc, topdiff, btmdatadesc, btmdata, &fBeta, btmdiffdesc, btmdiff))
		return lErr | ERROR_CUDNN_OFFSET;
#else
	if (lErr = cudnnActivationBackward(cudnn, CUDNN_ACTIVATION_ELU, &fAlpha, topdatadesc, topdata, topdiffdesc, topdiff, btmdatadesc, btmdata, &fBeta, btmdiffdesc, btmdiff))
		return lErr | ERROR_CUDNN_OFFSET;
#endif

	return cudaStreamSynchronize(0);
}

template long Memory<double>::EluBackward(long hHandle, double dfAlpha, long hTopDataDesc, long hTopData, long hTopDiffDesc, long hTopDiff, long hBottomDataDesc, long hBottomData, double dfBeta, long hBottomDiffDesc, long hBottomDiff);
template long Memory<float>::EluBackward(long hHandle, float fAlpha, long hTopDataDesc, long hTopData, long hTopDiffDesc, long hTopDiff, long hBottomDataDesc, long hBottomData, float fBeta, long hBottomDiffDesc, long hBottomDiff);


template <class T>
long Memory<T>::SigmoidForward(long hHandle, T fAlpha, long hBottomDesc, long hBottomData, T fBeta, long hTopDesc, long hTopData)
{
	LONG lErr;
	cudnnHandle_t cudnn = GetCuDNN(hHandle);
	cudnnTensorDescriptor_t topdesc = GetTensorDesc(hTopDesc);
	cudnnTensorDescriptor_t btmdesc = GetTensorDesc(hBottomDesc);
	MemoryItem* pTopData;
	MemoryItem* pBtmData;

	if (lErr = m_memory.GetData(hTopData, &pTopData))
		return lErr;

	if (lErr = m_memory.GetData(hBottomData, &pBtmData))
		return lErr;

	T* topdata = (T*)pTopData->Data();
	T* btmdata = (T*)pBtmData->Data();

#ifdef CUDNN_5
	cudnnActivationDescriptor_t desc = GetActivationDesc(m_hGlobalActivationSigmoid);
	if (lErr = cudnnActivationForward(cudnn, desc, &fAlpha, btmdesc, btmdata, &fBeta, topdesc, topdata))
		return lErr | ERROR_CUDNN_OFFSET;
#else
	if (lErr = cudnnActivationForward(cudnn, CUDNN_ACTIVATION_SIGMOID, &fAlpha, btmdesc, btmdata, &fBeta, topdesc, topdata))
		return lErr | ERROR_CUDNN_OFFSET;
#endif

	return cudaStreamSynchronize(0);
}

template long Memory<double>::SigmoidForward(long hHandle, double dfAlpha, long hBottomDesc, long hBottomData, double dfBeta, long hTopDesc, long hTopData);
template long Memory<float>::SigmoidForward(long hHandle, float fAlpha, long hBottomDesc, long hBottomData, float fBeta, long hTopDesc, long hTopData);


template <class T>
long Memory<T>::SigmoidBackward(long hHandle, T fAlpha, long hTopDataDesc, long hTopData, long hTopDiffDesc, long hTopDiff, long hBottomDataDesc, long hBottomData, T fBeta, long hBottomDiffDesc, long hBottomDiff)
{
	LONG lErr;
	cudnnHandle_t cudnn = GetCuDNN(hHandle);
	cudnnTensorDescriptor_t topdatadesc = GetTensorDesc(hTopDataDesc);
	cudnnTensorDescriptor_t btmdatadesc = GetTensorDesc(hBottomDataDesc);
	cudnnTensorDescriptor_t topdiffdesc = (hTopDataDesc == hTopDiffDesc) ? topdatadesc : GetTensorDesc(hTopDiffDesc);
	cudnnTensorDescriptor_t btmdiffdesc = (hBottomDataDesc == hBottomDiffDesc) ? btmdatadesc : GetTensorDesc(hBottomDiffDesc);
	MemoryItem* pTopData;
	MemoryItem* pBtmData;
	MemoryItem* pTopDiff;
	MemoryItem* pBtmDiff;

	if (lErr = m_memory.GetData(hTopData, &pTopData))
		return lErr;

	if (lErr = m_memory.GetData(hBottomData, &pBtmData))
		return lErr;

	if (lErr = m_memory.GetData(hTopDiff, &pTopDiff))
		return lErr;

	if (lErr = m_memory.GetData(hBottomDiff, &pBtmDiff))
		return lErr;

	T* topdata = (T*)pTopData->Data();
	T* btmdata = (T*)pBtmData->Data();
	T* topdiff = (T*)pTopDiff->Data();
	T* btmdiff = (T*)pBtmDiff->Data();

#ifdef CUDNN_5
	cudnnActivationDescriptor_t desc = GetActivationDesc(m_hGlobalActivationSigmoid);
	if (lErr = cudnnActivationBackward(cudnn, desc, &fAlpha, topdatadesc, topdata, topdiffdesc, topdiff, btmdatadesc, btmdata, &fBeta, btmdiffdesc, btmdiff))
		return lErr | ERROR_CUDNN_OFFSET;
#else
	if (lErr = cudnnActivationBackward(cudnn, CUDNN_ACTIVATION_SIGMOID, &fAlpha, topdatadesc, topdata, topdiffdesc, topdiff, btmdatadesc, btmdata, &fBeta, btmdiffdesc, btmdiff))
		return lErr | ERROR_CUDNN_OFFSET;
#endif

	return cudaStreamSynchronize(0);
}

template long Memory<double>::SigmoidBackward(long hHandle, double dfAlpha, long hTopDataDesc, long hTopData, long hTopDiffDesc, long hTopDiff, long hBottomDataDesc, long hBottomData, double dfBeta, long hBottomDiffDesc, long hBottomDiff);
template long Memory<float>::SigmoidBackward(long hHandle, float fAlpha, long hTopDataDesc, long hTopData, long hTopDiffDesc, long hTopDiff, long hBottomDataDesc, long hBottomData, float fBeta, long hBottomDiffDesc, long hBottomDiff);


template <class T>
long Memory<T>::ReLUForward(long hHandle, T fAlpha, long hBottomDesc, long hBottomData, T fBeta, long hTopDesc, long hTopData)
{
	LONG lErr;
	cudnnHandle_t cudnn = GetCuDNN(hHandle);
	cudnnTensorDescriptor_t topdesc = GetTensorDesc(hTopDesc);
	cudnnTensorDescriptor_t btmdesc = GetTensorDesc(hBottomDesc);
	MemoryItem* pTopData;
	MemoryItem* pBtmData;

	if (lErr = m_memory.GetData(hTopData, &pTopData))
		return lErr;

	if (lErr = m_memory.GetData(hBottomData, &pBtmData))
		return lErr;

	T* topdata = (T*)pTopData->Data();
	T* btmdata = (T*)pBtmData->Data();

#ifdef CUDNN_5
	cudnnActivationDescriptor_t desc = GetActivationDesc(m_hGlobalActivationRelu);
	if (lErr = cudnnActivationForward(cudnn, desc, &fAlpha, btmdesc, btmdata, &fBeta, topdesc, topdata))
		return lErr | ERROR_CUDNN_OFFSET;
#else
	if (lErr = cudnnActivationForward(cudnn, CUDNN_ACTIVATION_RELU, &fAlpha, btmdesc, btmdata, &fBeta, topdesc, topdata))
		return lErr | ERROR_CUDNN_OFFSET;
#endif

	return cudaStreamSynchronize(0);
}

template long Memory<double>::ReLUForward(long hHandle, double dfAlpha, long hBottomDesc, long hBottomData, double dfBeta, long hTopDesc, long hTopData);
template long Memory<float>::ReLUForward(long hHandle, float fAlpha, long hBottomDesc, long hBottomData, float fBeta, long hTopDesc, long hTopData);


template <class T>
long Memory<T>::ReLUBackward(long hHandle, T fAlpha, long hTopDataDesc, long hTopData, long hTopDiffDesc, long hTopDiff, long hBottomDataDesc, long hBottomData, T fBeta, long hBottomDiffDesc, long hBottomDiff)
{
	LONG lErr;
	cudnnHandle_t cudnn = GetCuDNN(hHandle);
	cudnnTensorDescriptor_t topdatadesc = GetTensorDesc(hTopDataDesc);
	cudnnTensorDescriptor_t btmdatadesc = GetTensorDesc(hBottomDataDesc);
	cudnnTensorDescriptor_t topdiffdesc = (hTopDataDesc == hTopDiffDesc) ? topdatadesc : GetTensorDesc(hTopDiffDesc);
	cudnnTensorDescriptor_t btmdiffdesc = (hBottomDataDesc == hBottomDiffDesc) ? btmdatadesc : GetTensorDesc(hBottomDiffDesc);
	MemoryItem* pTopData;
	MemoryItem* pBtmData;
	MemoryItem* pTopDiff;
	MemoryItem* pBtmDiff;

	if (lErr = m_memory.GetData(hTopData, &pTopData))
		return lErr;

	if (lErr = m_memory.GetData(hBottomData, &pBtmData))
		return lErr;

	if (lErr = m_memory.GetData(hTopDiff, &pTopDiff))
		return lErr;

	if (lErr = m_memory.GetData(hBottomDiff, &pBtmDiff))
		return lErr;

	T* topdata = (T*)pTopData->Data();
	T* btmdata = (T*)pBtmData->Data();
	T* topdiff = (T*)pTopDiff->Data();
	T* btmdiff = (T*)pBtmDiff->Data();

#ifdef CUDNN_5
	cudnnActivationDescriptor_t desc = GetActivationDesc(m_hGlobalActivationRelu);
	if (lErr = cudnnActivationBackward(cudnn, desc, &fAlpha, topdatadesc, topdata, topdiffdesc, topdiff, btmdatadesc, btmdata, &fBeta, btmdiffdesc, btmdiff))
		return lErr | ERROR_CUDNN_OFFSET;
#else
	if (lErr = cudnnActivationBackward(cudnn, CUDNN_ACTIVATION_RELU, &fAlpha, topdatadesc, topdata, topdiffdesc, topdiff, btmdatadesc, btmdata, &fBeta, btmdiffdesc, btmdiff))
		return lErr | ERROR_CUDNN_OFFSET;
#endif

	return cudaStreamSynchronize(0);
}

template long Memory<double>::ReLUBackward(long hHandle, double dfAlpha, long hTopDataDesc, long hTopData, long hTopDiffDesc, long hTopDiff, long hBottomDataDesc, long hBottomData, double dfBeta, long hBottomDiffDesc, long hBottomDiff);
template long Memory<float>::ReLUBackward(long hHandle, float fAlpha, long hTopDataDesc, long hTopData, long hTopDiffDesc, long hTopDiff, long hBottomDataDesc, long hBottomData, float fBeta, long hBottomDiffDesc, long hBottomDiff);



template <class T>
long Memory<T>::SoftmaxForward(long hHandle, T fAlpha, long hBottomDesc, long hBottomData, T fBeta, long hTopDesc, long hTopData)
{
	LONG lErr;
	cudnnHandle_t cudnn = GetCuDNN(hHandle);
	cudnnTensorDescriptor_t topdesc = GetTensorDesc(hTopDesc);
	cudnnTensorDescriptor_t btmdesc = GetTensorDesc(hBottomDesc);
	MemoryItem* pTopData;
	MemoryItem* pBtmData;

	if (lErr = m_memory.GetData(hTopData, &pTopData))
		return lErr;

	if (lErr = m_memory.GetData(hBottomData, &pBtmData))
		return lErr;

	T* topdata = (T*)pTopData->Data();
	T* btmdata = (T*)pBtmData->Data();

	if (lErr = cudnnSoftmaxForward(cudnn, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &fAlpha, btmdesc, btmdata, &fBeta, topdesc, topdata))
		return lErr | ERROR_CUDNN_OFFSET;

	return cudaStreamSynchronize(0);
}

template long Memory<double>::SoftmaxForward(long hHandle, double dfAlpha, long hBottomDesc, long hBottomData, double dfBeta, long hTopDesc, long hTopData);
template long Memory<float>::SoftmaxForward(long hHandle, float fAlpha, long hBottomDesc, long hBottomData, float fBeta, long hTopDesc, long hTopData);


template <class T>
long Memory<T>::SoftmaxBackward(long hHandle, T fAlpha, long hTopDataDesc, long hTopData, long hTopDiffDesc, long hTopDiff, T fBeta, long hBottomDiffDesc, long hBottomDiff)
{
	LONG lErr;
	cudnnHandle_t cudnn = GetCuDNN(hHandle);
	cudnnTensorDescriptor_t topdatadesc = GetTensorDesc(hTopDataDesc);
	cudnnTensorDescriptor_t topdiffdesc = (hTopDataDesc == hTopDiffDesc) ? topdatadesc : GetTensorDesc(hTopDiffDesc);
	cudnnTensorDescriptor_t btmdiffdesc = GetTensorDesc(hBottomDiffDesc);
	MemoryItem* pTopData;
	MemoryItem* pTopDiff;
	MemoryItem* pBtmDiff;

	if (lErr = m_memory.GetData(hTopData, &pTopData))
		return lErr;

	if (lErr = m_memory.GetData(hTopDiff, &pTopDiff))
		return lErr;

	if (lErr = m_memory.GetData(hBottomDiff, &pBtmDiff))
		return lErr;

	T* topdata = (T*)pTopData->Data();
	T* topdiff = (T*)pTopDiff->Data();
	T* btmdiff = (T*)pBtmDiff->Data();

	if (lErr = cudnnSoftmaxBackward(cudnn, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &fAlpha, topdatadesc, topdata, topdiffdesc, topdiff, &fBeta, btmdiffdesc, btmdiff))
		return lErr | ERROR_CUDNN_OFFSET;

	return cudaStreamSynchronize(0);
}

template long Memory<double>::SoftmaxBackward(long hHandle, double dfAlpha, long hTopDataDesc, long hTopData, long hTopDiffDesc, long hTopDiff, double dfBeta, long hBottomDiffDesc, long hBottomDiff);
template long Memory<float>::SoftmaxBackward(long hHandle, float fAlpha, long hTopDataDesc, long hTopData, long hTopDiffDesc, long hTopDiff, float fBeta, long hBottomDiffDesc, long hBottomDiff);


template <class T>
long Memory<T>::CreateRnnDataDesc1(long* phHandle)
{
	LONG lErr;
	rnnDataHandle<T>* desc = NULL;

	if (phHandle == NULL)
		return ERROR_PARAM_NULL;

	if ((desc = new rnnDataHandle<T>()) == NULL)
		return ERROR_MEMORY_OUT;

	if (lErr = desc->Initialize(this))
		return lErr;

	long hHandle = m_rnnDataDesc1.Allocate(desc);
	if (hHandle < 0)
	{
		desc->CleanUp();
		delete desc;
		return ERROR_MEMORY_OUT;
	}

	*phHandle = hHandle;
	return 0;
}

template long Memory<double>::CreateRnnDataDesc1(long* phHandle);
template long Memory<float>::CreateRnnDataDesc1(long* phHandle);


template <class T>
long Memory<T>::CreateRnnDataDesc2(long* phHandle)
{
	LONG lErr;
	cudnnRNNDataDescriptor_t desc = NULL;

	if (phHandle == NULL)
		return ERROR_PARAM_NULL;

	if (lErr = cudnnCreateRNNDataDescriptor(&desc))
		return lErr | ERROR_CUDNN_OFFSET;

	long hHandle = m_rnnDataDesc2.Allocate(desc);
	if (hHandle < 0)
	{
		cudnnDestroyRNNDataDescriptor(desc);
		return ERROR_MEMORY_OUT;
	}

	*phHandle = hHandle;
	return 0;
}

template long Memory<double>::CreateRnnDataDesc2(long* phHandle);
template long Memory<float>::CreateRnnDataDesc2(long* phHandle);


template <class T>
long Memory<T>::CreateRnnDesc(long* phHandle)
{
	LONG lErr;
	cudnnRNNDescriptor_t desc = NULL;

	if (phHandle == NULL)
		return ERROR_PARAM_NULL;

	if (lErr = cudnnCreateRNNDescriptor(&desc))
		return lErr | ERROR_CUDNN_OFFSET;

	long hHandle = m_rnnDesc.Allocate(desc);
	if (hHandle < 0)
	{
		cudnnDestroyRNNDescriptor(desc);
		return ERROR_MEMORY_OUT;
	}

	*phHandle = hHandle;
	return 0;
}

template long Memory<double>::CreateRnnDesc(long* phHandle);
template long Memory<float>::CreateRnnDesc(long* phHandle);


template <class T>
long Memory<T>::GetRnnParamCount(long hHandle, long hRnnDesc, long hXDesc, int* pnCount)
{
	LONG lErr;
	cudnnHandle_t cudnn = GetCuDNN(hHandle);
	cudnnRNNDescriptor_t desc = (cudnnRNNDescriptor_t)m_rnnDesc.GetData(hRnnDesc);
	rnnDataHandle<T>* descX = (rnnDataHandle<T>*)m_rnnDataDesc1.GetData(hXDesc);
	cudnnDataType_t type = (sizeof(T) == 4) ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;

	if (descX == NULL)
		return ERROR_PARAM_NULL;

	if (pnCount == NULL)
		return ERROR_PARAM_NULL;

	size_t sizeInBytes;
	if (lErr = cudnnGetRNNParamsSize(cudnn, desc, descX->GetFirstTensor(), &sizeInBytes, type))
		return lErr;

	int nCount = (int)(sizeInBytes / sizeof(T));
	*pnCount = nCount;

	return 0;
}

template long Memory<double>::GetRnnParamCount(long hHandle, long hRnnDesc, long hXDesc, int* pnCount);
template long Memory<float>::GetRnnParamCount(long hHandle, long hRnnDesc, long hXDesc, int* pnCount);


template <class T>
long Memory<T>::GetRnnParamCountEx(long hHandle, long hRnnDesc, long hXDesc, int* pnCount)
{
	LONG lErr;
	cudnnDataType_t type = (sizeof(T) == 4) ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;
	cudnnHandle_t cudnn = GetCuDNN(hHandle);
	cudnnRNNDescriptor_t desc = (cudnnRNNDescriptor_t)m_rnnDesc.GetData(hRnnDesc);	
	cudnnRNNDataDescriptor_t descX = (cudnnRNNDataDescriptor_t)m_rnnDataDesc2.GetData(hXDesc);

	if (descX == NULL)
		return ERROR_PARAM_NULL;

	if (pnCount == NULL)
		return ERROR_PARAM_NULL;

	cudnnDataType_t type0;
	cudnnRNNDataLayout_t layout;
	int nMaxSeqLen = 0;
	int nBatchSize = 0;
	int nVectorSize = 0;
	T fFill;
	int* rgSeqLen = (int*)malloc(sizeof(int) * 1);

	if (rgSeqLen == NULL)
		return ERROR_MEMORY_OUT;

	lErr = cudnnGetRNNDataDescriptor(descX, &type0, &layout, &nMaxSeqLen, &nBatchSize, &nVectorSize, 1, rgSeqLen, (void*)&fFill);
	free(rgSeqLen);

	if (lErr)
		return lErr;

	cudnnTensorDescriptor_t tensorX;

	if (lErr = cudnnCreateTensorDescriptor(&tensorX))
		return lErr;

	int rgDim[3];
	rgDim[0] = (layout == CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED) ? nMaxSeqLen : nBatchSize;
	rgDim[1] = nVectorSize;
	rgDim[2] = 1;

	int rgStride[3];
	rgStride[0] = rgDim[2] * rgDim[1];
	rgStride[1] = rgDim[2];
	rgStride[2] = 1;

	if (lErr = cudnnSetTensorNdDescriptor(tensorX, type, 3, rgDim, rgStride))
	{
		cudnnDestroyTensorDescriptor(tensorX);
		return lErr;
	}

	size_t sizeInBytes;
	lErr = cudnnGetRNNParamsSize(cudnn, desc, tensorX, &sizeInBytes, type);
	cudnnDestroyTensorDescriptor(tensorX);

	if (lErr)
		return lErr;

	int nCount = (int)(sizeInBytes / sizeof(T));
	*pnCount = nCount;

	return 0;
}

template long Memory<double>::GetRnnParamCountEx(long hHandle, long hRnnDesc, long hXDesc, int* pnCount);
template long Memory<float>::GetRnnParamCountEx(long hHandle, long hRnnDesc, long hXDesc, int* pnCount);


template <class T>
long Memory<T>::GetRnnWorkspaceCount(long hHandle, long hRnnDesc, long hXDesc, int* pnWsCount, int* pnResCount)
{
	LONG lErr;
	cudnnHandle_t cudnn = GetCuDNN(hHandle);
	cudnnRNNDescriptor_t desc = (cudnnRNNDescriptor_t)m_rnnDesc.GetData(hRnnDesc);
	rnnDataHandle<T>* descX = (rnnDataHandle<T>*)m_rnnDataDesc1.GetData(hXDesc);

	if (pnWsCount == NULL || pnResCount == NULL)
		return ERROR_PARAM_NULL;

	size_t sizeInBytes;
	if (lErr = cudnnGetRNNWorkspaceSize(cudnn, desc, descX->MaxSeqLen(), descX->SeqTensors(), &sizeInBytes))
		return lErr;

	int nWsCount = (int)(sizeInBytes / sizeof(T));

	if (lErr = cudnnGetRNNTrainingReserveSize(cudnn, desc, descX->MaxSeqLen(), descX->SeqTensors(), &sizeInBytes))
		return lErr;

	int nResCount = (int)(sizeInBytes / sizeof(T));

	*pnWsCount = nWsCount;
	*pnResCount = nResCount;

	return 0;
}

template long Memory<double>::GetRnnWorkspaceCount(long hHandle, long hRnnDesc, long hXDesc, int* pnWsCount, int* pnResCount);
template long Memory<float>::GetRnnWorkspaceCount(long hHandle, long hRnnDesc, long hXDesc, int* pnWsCount, int* pnResCount);


template <class T>
long Memory<T>::GetRnnWorkspaceCountEx(long hHandle, long hRnnDesc, long hXDesc, int* pnWsCount, int* pnResCount)
{
	LONG lErr;
	cudnnDataType_t type = (sizeof(T) == 4) ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;
	cudnnHandle_t cudnn = GetCuDNN(hHandle);
	cudnnRNNDescriptor_t desc = (cudnnRNNDescriptor_t)m_rnnDesc.GetData(hRnnDesc);
	cudnnRNNDataDescriptor_t descX = (cudnnRNNDataDescriptor_t)m_rnnDataDesc2.GetData(hXDesc);

	if (descX == NULL)
		return ERROR_PARAM_NULL;

	if (pnWsCount == NULL || pnResCount == NULL)
		return ERROR_PARAM_NULL;

	cudnnDataType_t type0;
	cudnnRNNDataLayout_t layout;
	int nMaxSeqLen = 0;
	int nBatchSize = 0;
	int nVectorSize = 0;
	T fFill;
	int* rgSeqLen = (int*)malloc(sizeof(int) * 1);

	if (rgSeqLen == NULL)
		return ERROR_MEMORY_OUT;

	lErr = cudnnGetRNNDataDescriptor(descX, &type0, &layout, &nMaxSeqLen, &nBatchSize, &nVectorSize, 1, rgSeqLen, (void*)&fFill);
	free(rgSeqLen);

	if (lErr)
		return lErr;

	cudnnTensorDescriptor_t* rgDescX = (cudnnTensorDescriptor_t*)malloc(sizeof(cudnnTensorDescriptor_t) * nMaxSeqLen);
	if (rgDescX == NULL)
		return ERROR_OUTOFMEMORY;

	memset(rgDescX, NULL, sizeof(cudnnTensorDescriptor_t) * nMaxSeqLen);

	for (int i = 0; i < nMaxSeqLen; i++)
	{
		if (lErr = cudnnCreateTensorDescriptor(&rgDescX[i]))
			break;

		int rgDim[3];
		rgDim[0] = (layout == CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED) ? nMaxSeqLen : nBatchSize;
		rgDim[1] = nVectorSize;
		rgDim[2] = 1;

		int rgStride[3];
		rgStride[0] = rgDim[2] * rgDim[1];
		rgStride[1] = rgDim[2];
		rgStride[2] = 1;

		if (lErr = cudnnSetTensorNdDescriptor(rgDescX[i], type, 3, rgDim, rgStride))
			break;
	}

	size_t sizeInBytes;
	int nWsCount = 0;

	if (!lErr)
	{
		lErr = cudnnGetRNNWorkspaceSize(cudnn, desc, nMaxSeqLen, rgDescX, &sizeInBytes);

		if (!lErr)
		{
			nWsCount = (int)(sizeInBytes / sizeof(T)) + 1;

			lErr = cudnnGetRNNTrainingReserveSize(cudnn, desc, nMaxSeqLen, rgDescX, &sizeInBytes);
		}
	}

	for (int i = 0; i < nMaxSeqLen; i++)
	{
		if (rgDescX[i] != NULL)
			cudnnDestroyTensorDescriptor(rgDescX[i]);
	}

	free(rgDescX);

	if (lErr)
		return lErr;

	int nResCount = (int)(sizeInBytes / sizeof(T)) + 1;

	*pnWsCount = nWsCount;
	*pnResCount = nResCount;

	return 0;
}

template long Memory<double>::GetRnnWorkspaceCountEx(long hHandle, long hRnnDesc, long hXDesc, int* pnWsCount, int* pnResCount);
template long Memory<float>::GetRnnWorkspaceCountEx(long hHandle, long hRnnDesc, long hXDesc, int* pnWsCount, int* pnResCount);


template <class T>
long Memory<T>::GetRnnLinLayerParams(long hHandle, long hRnnDesc, int nLayer, long hXDesc, long hWtDesc, long hWtData, int nLinLayer, int* pnWtCount, long* phWt, int* pnBiasCount, long* phBias)
{
	LONG lErr;
	cudnnHandle_t cudnn = GetCuDNN(hHandle);
	cudnnRNNDescriptor_t desc = (cudnnRNNDescriptor_t)m_rnnDesc.GetData(hRnnDesc);
	rnnDataHandle<T>* descX = (rnnDataHandle<T>*)m_rnnDataDesc1.GetData(hXDesc);
	cudnnFilterDescriptor_t descWt = (cudnnFilterDescriptor_t)m_filterDesc.GetData(hWtDesc);
	MemoryItem* pWtData;

	if (descX == NULL)
		return ERROR_PARAM_OUT_OF_RANGE;

	if (lErr = m_memory.GetData(hWtData, &pWtData))
		return lErr;

	if (pnWtCount == NULL || phWt == NULL || pnBiasCount == NULL || phBias == NULL)
		return ERROR_PARAM_NULL;

	// Get the Weight Counts
	cudnnFilterDescriptor_t filterWts;
	if (lErr = cudnnCreateFilterDescriptor(&filterWts))
		return lErr;

	void* pWtDevMem;
	if (lErr = cudnnGetRNNLinLayerMatrixParams(cudnn, desc, nLayer, descX->GetFirstTensor(), descWt, pWtData->Data(), nLinLayer, filterWts, &pWtDevMem))
	{
		cudnnDestroyFilterDescriptor(filterWts);
		return lErr;
	}

	cudnnDataType_t type;
	cudnnTensorFormat_t fmt;
	int nbDims;
	int rgDimA[3];

	if (lErr = cudnnGetFilterNdDescriptor(filterWts, 3, &type, &fmt, &nbDims, rgDimA))
	{
		cudnnDestroyFilterDescriptor(filterWts);
		return lErr;
	}

	int nWtCount = rgDimA[0] * rgDimA[1] * rgDimA[2];

	cudnnDestroyFilterDescriptor(filterWts);


	// Get the Bias Counts
	cudnnFilterDescriptor_t filterBias;
	if (lErr = cudnnCreateFilterDescriptor(&filterBias))
		return lErr;

	void* pBiasDevMem;
	if (lErr = cudnnGetRNNLinLayerBiasParams(cudnn, desc, nLayer, descX->GetFirstTensor(), descWt, pWtData->Data(), nLinLayer, filterBias, &pBiasDevMem))
	{
		cudnnDestroyFilterDescriptor(filterBias);
		return lErr;
	}

	if (lErr = cudnnGetFilterNdDescriptor(filterBias, 3, &type, &fmt, &nbDims, rgDimA))
	{
		cudnnDestroyFilterDescriptor(filterBias);
		return lErr;
	}

	int nBiasCount = rgDimA[0] * rgDimA[1] * rgDimA[2];

	cudnnDestroyFilterDescriptor(filterBias);


	// Create the memory pointer handles.
	long hWtMemPtr;
	long long lWtSize = nWtCount * sizeof(T);
	if (lWtSize > SIZE_MAX)
		return ERROR_MEMORY_RANGE_EXCEEDED;

	if (lErr = CreateMemoryPointer(pWtData->DeviceID(), pWtData->IsHalf(), (T*)pWtDevMem, (size_t)lWtSize, &hWtMemPtr))
		return lErr;

	long hBiasMemPtr;
	long long lBiasSize = nBiasCount * sizeof(T);
	if (lBiasSize > SIZE_MAX)
		return ERROR_MEMORY_RANGE_EXCEEDED;

	if (lErr = CreateMemoryPointer(pWtData->DeviceID(), pWtData->IsHalf(), (T*)pBiasDevMem, (size_t)lBiasSize, &hBiasMemPtr))
		return lErr;

	*pnWtCount = nWtCount;
	*phWt = hWtMemPtr;
	*pnBiasCount = nBiasCount;
	*phBias = hBiasMemPtr;

	return 0;
}

template long Memory<double>::GetRnnLinLayerParams(long hHandle, long hRnnDesc, int nLayer, long hXDesc, long hWtDesc, long hWtData, int nLinLayer, int* pnWtCount, long* phWt, int* pnBiasCount, long* phBias);
template long Memory<float>::GetRnnLinLayerParams(long hHandle, long hRnnDesc, int nLayer, long hXDesc, long hWtDesc, long hWtData, int nLinLayer, int* pnWtCount, long* phWt, int* pnBiasCount, long* phBias);


template <class T>
long Memory<T>::GetRnnLinLayerParamsEx(long hHandle, long hRnnDesc, int nLayer, long hXDesc, long hWtDesc, long hWtData, int nLinLayer, int* pnWtCount, long* phWt, int* pnBiasCount, long* phBias)
{
	LONG lErr;
	cudnnDataType_t type = (sizeof(T) == 4) ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;
	cudnnHandle_t cudnn = GetCuDNN(hHandle);
	cudnnRNNDescriptor_t desc = (cudnnRNNDescriptor_t)m_rnnDesc.GetData(hRnnDesc);
	cudnnRNNDataDescriptor_t descX = (cudnnRNNDataDescriptor_t)m_rnnDataDesc2.GetData(hXDesc);
	cudnnFilterDescriptor_t descWt = (cudnnFilterDescriptor_t)m_filterDesc.GetData(hWtDesc);
	MemoryItem* pWtData;

	if (lErr = m_memory.GetData(hWtData, &pWtData))
		return lErr;

	if (descX == NULL)
		return ERROR_PARAM_NULL;

	if (pnWtCount == NULL || phWt == NULL || pnBiasCount == NULL || phBias == NULL)
		return ERROR_PARAM_NULL;

	cudnnDataType_t type0;
	cudnnRNNDataLayout_t layout;
	int nMaxSeqLen = 0;
	int nBatchSize = 0;
	int nVectorSize = 0;
	T fFill;
	int* rgSeqLen = (int*)malloc(sizeof(int) * 1);

	if (rgSeqLen == NULL)
		return ERROR_MEMORY_OUT;

	lErr = cudnnGetRNNDataDescriptor(descX, &type0, &layout, &nMaxSeqLen, &nBatchSize, &nVectorSize, 1, rgSeqLen, (void*)&fFill);
	free(rgSeqLen);

	if (lErr)
		return lErr;

	cudnnTensorDescriptor_t tensorX;

	if (lErr = cudnnCreateTensorDescriptor(&tensorX))
		return lErr;

	int rgDim[3];
	rgDim[0] = (layout == CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED) ? nMaxSeqLen : nBatchSize;
	rgDim[1] = nVectorSize;
	rgDim[2] = 1;

	int rgStride[3];
	rgStride[0] = rgDim[2] * rgDim[1];
	rgStride[1] = rgDim[2];
	rgStride[2] = 1;

	if (lErr = cudnnSetTensorNdDescriptor(tensorX, type, 3, rgDim, rgStride))
	{
		cudnnDestroyTensorDescriptor(tensorX);
		return lErr;
	}

	// Get the Weight Counts
	cudnnFilterDescriptor_t filterWts;
	if (lErr = cudnnCreateFilterDescriptor(&filterWts))
	{
		cudnnDestroyTensorDescriptor(tensorX);
		return lErr;
	}

	void* pWtDevMem;
	if (lErr = cudnnGetRNNLinLayerMatrixParams(cudnn, desc, nLayer, tensorX, descWt, pWtData->Data(), nLinLayer, filterWts, &pWtDevMem))
	{
		cudnnDestroyTensorDescriptor(tensorX);
		cudnnDestroyFilterDescriptor(filterWts);
		return lErr;
	}

	cudnnTensorFormat_t fmt;
	int nbDims;
	int rgDimA[3];

	if (lErr = cudnnGetFilterNdDescriptor(filterWts, 3, &type0, &fmt, &nbDims, rgDimA))
	{
		cudnnDestroyTensorDescriptor(tensorX);
		cudnnDestroyFilterDescriptor(filterWts);
		return lErr;
	}

	int nWtCount = rgDimA[0] * rgDimA[1] * rgDimA[2];

	cudnnDestroyFilterDescriptor(filterWts);


	// Get the Bias Counts
	cudnnFilterDescriptor_t filterBias;
	if (lErr = cudnnCreateFilterDescriptor(&filterBias))
	{
		cudnnDestroyTensorDescriptor(tensorX);
		return lErr;
	}

	void* pBiasDevMem;
	if (lErr = cudnnGetRNNLinLayerBiasParams(cudnn, desc, nLayer, tensorX, descWt, pWtData->Data(), nLinLayer, filterBias, &pBiasDevMem))
	{
		cudnnDestroyTensorDescriptor(tensorX);
		cudnnDestroyFilterDescriptor(filterBias);
		return lErr;
	}

	if (lErr = cudnnGetFilterNdDescriptor(filterBias, 3, &type, &fmt, &nbDims, rgDimA))
	{
		cudnnDestroyTensorDescriptor(tensorX);
		cudnnDestroyFilterDescriptor(filterBias);
		return lErr;
	}

	int nBiasCount = rgDimA[0] * rgDimA[1] * rgDimA[2];

	cudnnDestroyFilterDescriptor(filterBias);
	cudnnDestroyTensorDescriptor(tensorX);

	
	// Create the memory pointer handles.
	long hWtMemPtr;
	long long lWtSize = nWtCount * sizeof(T);
	if (lWtSize > SIZE_MAX)
		return ERROR_MEMORY_RANGE_EXCEEDED;

	if (lErr = CreateMemoryPointer(pWtData->DeviceID(), pWtData->IsHalf(), (T*)pWtDevMem, (size_t)lWtSize, &hWtMemPtr))
		return lErr;

	long hBiasMemPtr;
	long long lBiasSize = nBiasCount * sizeof(T);
	if (lBiasSize > SIZE_MAX)
		return ERROR_MEMORY_RANGE_EXCEEDED;

	if (lErr = CreateMemoryPointer(pWtData->DeviceID(), pWtData->IsHalf(), (T*)pBiasDevMem, (size_t)lBiasSize, &hBiasMemPtr))
		return lErr;

	*pnWtCount = nWtCount;
	*phWt = hWtMemPtr;
	*pnBiasCount = nBiasCount;
	*phBias = hBiasMemPtr;

	return 0;
}

template long Memory<double>::GetRnnLinLayerParamsEx(long hHandle, long hRnnDesc, int nLayer, long hXDesc, long hWtDesc, long hWtData, int nLinLayer, int* pnWtCount, long* phWt, int* pnBiasCount, long* phBias);
template long Memory<float>::GetRnnLinLayerParamsEx(long hHandle, long hRnnDesc, int nLayer, long hXDesc, long hWtDesc, long hWtData, int nLinLayer, int* pnWtCount, long* phWt, int* pnBiasCount, long* phBias);


template <class T>
long Memory<T>::RnnForward(long hHandle, long hRnnDesc, long hXDesc, long hXData, long hHxDesc, long hHxData, long hCxDesc, long hCxData, long hWtDesc, long hWtData, long hYDesc, long hYData, long hHyDesc, long hHyData, long hCyDesc, long hCyData, long hWorkspaceData, int nWsCount, long hReservedData, int nResCount, bool bTraining)
{
	LONG lErr;
	cudnnHandle_t cudnn = GetCuDNN(hHandle);
	cudnnRNNDescriptor_t desc = (cudnnRNNDescriptor_t)m_rnnDesc.GetData(hRnnDesc);
	rnnDataHandle<T>* descX = (rnnDataHandle<T>*)m_rnnDataDesc1.GetData(hXDesc);
	rnnDataHandle<T>* descY = (rnnDataHandle<T>*)m_rnnDataDesc1.GetData(hYDesc);
	cudnnTensorDescriptor_t descHx = (cudnnTensorDescriptor_t)m_tensorDesc.GetData(hHxDesc);
	cudnnTensorDescriptor_t descCx = (cudnnTensorDescriptor_t)m_tensorDesc.GetData(hCxDesc);
	cudnnFilterDescriptor_t descWt = (cudnnFilterDescriptor_t)m_filterDesc.GetData(hWtDesc);
	cudnnTensorDescriptor_t descHy = (cudnnTensorDescriptor_t)m_tensorDesc.GetData(hHyDesc);
	cudnnTensorDescriptor_t descCy = (cudnnTensorDescriptor_t)m_tensorDesc.GetData(hCyDesc);
	MemoryItem* pXData;
	MemoryItem* pHxData;
	MemoryItem* pCxData;
	MemoryItem* pWtData;
	MemoryItem* pYData;
	MemoryItem* pHyData;
	MemoryItem* pCyData;
	MemoryItem* pWorkspaceData;
	MemoryItem* pReservedData;

	if (descX == NULL || descY == NULL)
		return ERROR_PARAM_OUT_OF_RANGE;

	if (lErr = m_memory.GetData(hXData, &pXData))
		return lErr;

	if (lErr = m_memory.GetData(hHxData, &pHxData))
		return lErr;

	if (lErr = m_memory.GetData(hCxData, &pCxData))
		return lErr;

	if (lErr = m_memory.GetData(hWtData, &pWtData))
		return lErr;

	if (lErr = m_memory.GetData(hYData, &pYData))
		return lErr;

	if (lErr = m_memory.GetData(hHyData, &pHyData))
		return lErr;

	if (lErr = m_memory.GetData(hCyData, &pCyData))
		return lErr;

	if (lErr = m_memory.GetData(hWorkspaceData, &pWorkspaceData))
		return lErr;

	if (bTraining)
	{
		if (lErr = m_memory.GetData(hReservedData, &pReservedData))
			return lErr;
	}

	if (!bTraining)
	{
		lErr = cudnnRNNForwardInference(cudnn,			
			desc,
			descX->MaxSeqLen(),
			descX->SeqTensors(),
			pXData->Data(),
			descHx,
			pHxData->Data(),
			descCx,
			pCxData->Data(),
			descWt,
			pWtData->Data(),
			descY->SeqTensors(),
			pYData->Data(),
			descHy,
			pHyData->Data(),
			descCy,
			pCyData->Data(),
			pWorkspaceData->Data(),
			pWorkspaceData->Size());
	}
	else
	{
		lErr = cudnnRNNForwardTraining(cudnn,
			desc,
			descX->MaxSeqLen(),
			descX->SeqTensors(),
			pXData->Data(),
			descHx,
			pHxData->Data(),
			descCx,
			pCxData->Data(),
			descWt,
			pWtData->Data(),
			descY->SeqTensors(),
			pYData->Data(),
			descHy,
			pHyData->Data(),
			descCy,
			pCyData->Data(),
			pWorkspaceData->Data(),
			pWorkspaceData->Size(),
			pReservedData->Data(),
			pReservedData->Size());

	}

	return lErr;
}

template long Memory<double>::RnnForward(long hHandle, long hRnnDesc, long hXDesc, long hXData, long hHxDesc, long hHxData, long hCxDesc, long hCxData, long hWtDesc, long hWtData, long hYDesc, long hYData, long hHyDesc, long hHyData, long hCyDesc, long hCyData, long hWorkspace, int nWsCount, long hReserved, int nResCount, bool bTraining);
template long Memory<float>::RnnForward(long hHandle, long hRnnDesc, long hXDesc, long hXData, long hHxDesc, long hHxData, long hCxDesc, long hCxData, long hWtDesc, long hWtData, long hYDesc, long hYData, long hHyDesc, long hHyData, long hCyDesc, long hCyData, long hWorkspace, int nWsCount, long hReserved, int nResCount, bool bTraining);

template <class T>
long Memory<T>::RnnForwardEx(long hHandle, long hRnnDesc, long hXDesc, long hXData, long hHxDesc, long hHxData, long hCxDesc, long hCxData, long hWtDesc, long hWtData, long hYDesc, long hYData, long hHyDesc, long hHyData, long hCyDesc, long hCyData, long hWorkspaceData, int nWsCount, long hReservedData, int nResCount, bool bTraining)
{
	LONG lErr;
	cudnnHandle_t cudnn = GetCuDNN(hHandle);
	cudnnRNNDescriptor_t desc = (cudnnRNNDescriptor_t)m_rnnDesc.GetData(hRnnDesc);
	cudnnRNNDataDescriptor_t descX = (cudnnRNNDataDescriptor_t)m_rnnDataDesc2.GetData(hXDesc);
	cudnnRNNDataDescriptor_t descY = (cudnnRNNDataDescriptor_t)m_rnnDataDesc2.GetData(hYDesc);
	cudnnTensorDescriptor_t descHx = (cudnnTensorDescriptor_t)m_tensorDesc.GetData(hHxDesc);
	cudnnTensorDescriptor_t descCx = (cudnnTensorDescriptor_t)m_tensorDesc.GetData(hCxDesc);
	cudnnFilterDescriptor_t descWt = (cudnnFilterDescriptor_t)m_filterDesc.GetData(hWtDesc);
	cudnnTensorDescriptor_t descHy = (cudnnTensorDescriptor_t)m_tensorDesc.GetData(hHyDesc);
	cudnnTensorDescriptor_t descCy = (cudnnTensorDescriptor_t)m_tensorDesc.GetData(hCyDesc);
	MemoryItem* pXData;
	MemoryItem* pHxData;
	MemoryItem* pCxData;
	MemoryItem* pWtData;
	MemoryItem* pYData;
	MemoryItem* pHyData;
	MemoryItem* pCyData;
	MemoryItem* pWorkspaceData;
	MemoryItem* pReservedData;

	if (lErr = m_memory.GetData(hXData, &pXData))
		return lErr;

	if (lErr = m_memory.GetData(hHxData, &pHxData))
		return lErr;

	if (lErr = m_memory.GetData(hCxData, &pCxData))
		return lErr;

	if (lErr = m_memory.GetData(hWtData, &pWtData))
		return lErr;

	if (lErr = m_memory.GetData(hYData, &pYData))
		return lErr;

	if (lErr = m_memory.GetData(hHyData, &pHyData))
		return lErr;

	if (lErr = m_memory.GetData(hCyData, &pCyData))
		return lErr;

	if (lErr = m_memory.GetData(hWorkspaceData, &pWorkspaceData))
		return lErr;

	if (bTraining)
	{
		if (lErr = m_memory.GetData(hReservedData, &pReservedData))
			return lErr;
	}

	if (!bTraining)
	{
		lErr = cudnnRNNForwardInferenceEx(cudnn,
										desc,
										descX,
										pXData->Data(),
										descHx,
										pHxData->Data(),
										descCx,
										pCxData->Data(),
										descWt,
										pWtData->Data(),
										descY,
										pYData->Data(),
										descHy,
										pHyData->Data(),
										descCy,
										pCyData->Data(),
			                            NULL,
			                            NULL,
			                            NULL,
			                            NULL,
			                            NULL,
			                            NULL,
			                            NULL,
			                            NULL,
										pWorkspaceData->Data(),
										pWorkspaceData->Size());
	}
	else
	{
		lErr = cudnnRNNForwardTrainingEx(cudnn,
										desc,
										descX,
										pXData->Data(),
										descHx,
										pHxData->Data(),
										descCx,
										pCxData->Data(),
										descWt,
										pWtData->Data(),
										descY,
										pYData->Data(),
										descHy,
										pHyData->Data(),
										descCy,
										pCyData->Data(),
										NULL,
										NULL,
										NULL,
										NULL,
										NULL,
										NULL,
										NULL,
										NULL,
										pWorkspaceData->Data(),
										pWorkspaceData->Size(),
										pReservedData->Data(),
										pReservedData->Size());

	}

	return lErr;
}

template long Memory<double>::RnnForwardEx(long hHandle, long hRnnDesc, long hXDesc, long hXData, long hHxDesc, long hHxData, long hCxDesc, long hCxData, long hWtDesc, long hWtData, long hYDesc, long hYData, long hHyDesc, long hHyData, long hCyDesc, long hCyData, long hWorkspace, int nWsCount, long hReserved, int nResCount, bool bTraining);
template long Memory<float>::RnnForwardEx(long hHandle, long hRnnDesc, long hXDesc, long hXData, long hHxDesc, long hHxData, long hCxDesc, long hCxData, long hWtDesc, long hWtData, long hYDesc, long hYData, long hHyDesc, long hHyData, long hCyDesc, long hCyData, long hWorkspace, int nWsCount, long hReserved, int nResCount, bool bTraining);


template <class T>
long Memory<T>::RnnBackwardData(long hHandle, long hRnnDesc, long hYDesc, long hYData, long hYDiff, long hHyDesc, long hHyDiff, long hCyDesc, long hCyDiff, long hWtDesc, long hWtData, long hHxDesc, long hHxData, long hCxDesc, long hCxData, long hXDesc, long hXDiff, long hdHxDesc, long hHxDiff, long hdCxDesc, long hCxDiff, long hWorkspaceData, int nWsCount, long hReservedData, int nResCount)
{
	LONG lErr;
	cudnnHandle_t cudnn = GetCuDNN(hHandle);
	cudnnRNNDescriptor_t desc = (cudnnRNNDescriptor_t)m_rnnDesc.GetData(hRnnDesc);
	rnnDataHandle<T>* descX = (rnnDataHandle<T>*)m_rnnDataDesc1.GetData(hXDesc);
	rnnDataHandle<T>* descY = (rnnDataHandle<T>*)m_rnnDataDesc1.GetData(hYDesc);
	cudnnTensorDescriptor_t descHy = (cudnnTensorDescriptor_t)m_tensorDesc.GetData(hHyDesc);
	cudnnTensorDescriptor_t descCy = (cudnnTensorDescriptor_t)m_tensorDesc.GetData(hCyDesc);
	cudnnFilterDescriptor_t descWt = (cudnnFilterDescriptor_t)m_filterDesc.GetData(hWtDesc);
	cudnnTensorDescriptor_t descHx = (cudnnTensorDescriptor_t)m_tensorDesc.GetData(hHxDesc);
	cudnnTensorDescriptor_t descCx = (cudnnTensorDescriptor_t)m_tensorDesc.GetData(hCxDesc);
	cudnnTensorDescriptor_t descHxd = (cudnnTensorDescriptor_t)m_tensorDesc.GetData(hdHxDesc);
	cudnnTensorDescriptor_t descCxd = (cudnnTensorDescriptor_t)m_tensorDesc.GetData(hdCxDesc);
	MemoryItem* pYData;
	MemoryItem* pYDiff;
	MemoryItem* pHyDiff;
	MemoryItem* pCyDiff;
	MemoryItem* pWtData;
	MemoryItem* pHxData;
	MemoryItem* pCxData;
	MemoryItem* pXDiff;
	MemoryItem* pHxDiff;
	MemoryItem* pCxDiff;
	MemoryItem* pWorkspaceData;
	MemoryItem* pReservedData;

	if (descX == NULL || descY == NULL)
		return ERROR_PARAM_OUT_OF_RANGE;

	if (lErr = m_memory.GetData(hYData, &pYData))
		return lErr;

	if (lErr = m_memory.GetData(hYDiff, &pYDiff))
		return lErr;

	if (lErr = m_memory.GetData(hHyDiff, &pHyDiff))
		return lErr;

	if (lErr = m_memory.GetData(hCyDiff, &pCyDiff))
		return lErr;

	if (lErr = m_memory.GetData(hWtData, &pWtData))
		return lErr;

	if (lErr = m_memory.GetData(hHxData, &pHxData))
		return lErr;

	if (lErr = m_memory.GetData(hCxData, &pCxData))
		return lErr;

	if (lErr = m_memory.GetData(hXDiff, &pXDiff))
		return lErr;

	if (lErr = m_memory.GetData(hHxDiff, &pHxDiff))
		return lErr;

	if (lErr = m_memory.GetData(hCxDiff, &pCxDiff))
		return lErr;

	if (lErr = m_memory.GetData(hWorkspaceData, &pWorkspaceData))
		return lErr;

	if (lErr = m_memory.GetData(hReservedData, &pReservedData))
		return lErr;

	lErr = cudnnRNNBackwardData(cudnn,
		desc,
		descY->MaxSeqLen(),
		descY->SeqTensors(),
		pYData->Data(),
		descY->SeqTensors(),
		pYDiff->Data(),
		descHy,
		pHyDiff->Data(),
		descCy,
		pCyDiff->Data(),
		descWt,
		pWtData->Data(),
		descHx,
		pHxData->Data(),
		descCx,
		pCxData->Data(),
		descX->SeqTensors(),
		pXDiff->Data(),
		descHxd,
		pHxDiff->Data(),
		descCxd,
		pCxDiff->Data(),
		pWorkspaceData->Data(),
		pWorkspaceData->Size(),
		pReservedData->Data(),
		pReservedData->Size());

	return lErr;
}

template long Memory<double>::RnnBackwardData(long hHandle, long hRnnDesc, long hYDesc, long hYData, long hYDiff, long hHyDesc, long hHyDiff, long hCyDesc, long hCyDiff, long hWtDesc, long hWtData, long hHxDesc, long hHxData, long hCxDesc, long hCxData, long hXDesc, long hXDiff, long hdHxDesc, long hHxDiff, long hdCxDesc, long hCxDiff, long hWorkspace, int nWsCount, long hReserved, int nResCount);
template long Memory<float>::RnnBackwardData(long hHandle, long hRnnDesc, long hYDesc, long hYData, long hYDiff, long hHyDesc, long hHyDiff, long hCyDesc, long hCyDiff, long hWtDesc, long hWtData, long hHxDesc, long hHxData, long hCxDesc, long hCxData, long hXDesc, long hXDiff, long hdHxDesc, long hHxDiff, long hdCxDesc, long hCxDiff, long hWorkspace, int nWsCount, long hReserved, int nResCount);

template <class T>
long Memory<T>::RnnBackwardDataEx(long hHandle, long hRnnDesc, long hYDesc, long hYData, long hYDiff, long hHyDesc, long hHyDiff, long hCyDesc, long hCyDiff, long hWtDesc, long hWtData, long hHxDesc, long hHxData, long hCxDesc, long hCxData, long hXDesc, long hXDiff, long hdHxDesc, long hHxDiff, long hdCxDesc, long hCxDiff, long hWorkspaceData, int nWsCount, long hReservedData, int nResCount)
{
	LONG lErr;
	cudnnHandle_t cudnn = GetCuDNN(hHandle);
	cudnnRNNDescriptor_t desc = (cudnnRNNDescriptor_t)m_rnnDesc.GetData(hRnnDesc);
	cudnnRNNDataDescriptor_t descX = (cudnnRNNDataDescriptor_t)m_rnnDataDesc2.GetData(hXDesc);
	cudnnRNNDataDescriptor_t descY = (cudnnRNNDataDescriptor_t)m_rnnDataDesc2.GetData(hYDesc);
	cudnnTensorDescriptor_t descHy = (cudnnTensorDescriptor_t)m_tensorDesc.GetData(hHyDesc);
	cudnnTensorDescriptor_t descCy = (cudnnTensorDescriptor_t)m_tensorDesc.GetData(hCyDesc);
	cudnnFilterDescriptor_t descWt = (cudnnFilterDescriptor_t)m_filterDesc.GetData(hWtDesc);
	cudnnTensorDescriptor_t descHx = (cudnnTensorDescriptor_t)m_tensorDesc.GetData(hHxDesc);
	cudnnTensorDescriptor_t descCx = (cudnnTensorDescriptor_t)m_tensorDesc.GetData(hCxDesc);
	cudnnTensorDescriptor_t descHxd = (cudnnTensorDescriptor_t)m_tensorDesc.GetData(hdHxDesc);
	cudnnTensorDescriptor_t descCxd = (cudnnTensorDescriptor_t)m_tensorDesc.GetData(hdCxDesc);
	MemoryItem* pYData;
	MemoryItem* pYDiff;
	MemoryItem* pHyDiff;
	MemoryItem* pCyDiff;
	MemoryItem* pWtData;
	MemoryItem* pHxData;
	MemoryItem* pCxData;
	MemoryItem* pXDiff;
	MemoryItem* pHxDiff;
	MemoryItem* pCxDiff;
	MemoryItem* pWorkspaceData;
	MemoryItem* pReservedData;

	if (lErr = m_memory.GetData(hYData, &pYData))
		return lErr;

	if (lErr = m_memory.GetData(hYDiff, &pYDiff))
		return lErr;

	if (lErr = m_memory.GetData(hHyDiff, &pHyDiff))
		return lErr;

	if (lErr = m_memory.GetData(hCyDiff, &pCyDiff))
		return lErr;

	if (lErr = m_memory.GetData(hWtData, &pWtData))
		return lErr;

	if (lErr = m_memory.GetData(hHxData, &pHxData))
		return lErr;

	if (lErr = m_memory.GetData(hCxData, &pCxData))
		return lErr;

	if (lErr = m_memory.GetData(hXDiff, &pXDiff))
		return lErr;

	if (lErr = m_memory.GetData(hHxDiff, &pHxDiff))
		return lErr;

	if (lErr = m_memory.GetData(hCxDiff, &pCxDiff))
		return lErr;

	if (lErr = m_memory.GetData(hWorkspaceData, &pWorkspaceData))
		return lErr;

	if (lErr = m_memory.GetData(hReservedData, &pReservedData))
		return lErr;

	lErr = cudnnRNNBackwardDataEx(cudnn,
								desc,
								descY,
								pYData->Data(),
								descY,
								pYDiff->Data(),
		                        NULL,
		                        NULL,
								descHy,
								pHyDiff->Data(),
								descCy,
								pCyDiff->Data(),
								descWt,
								pWtData->Data(),
								descHx,
								pHxData->Data(),
								descCx,
								pCxData->Data(),
								descX,
								pXDiff->Data(),
								descHxd,
								pHxDiff->Data(),
								descCxd,
								pCxDiff->Data(),
		                        NULL,
		                        NULL,
								pWorkspaceData->Data(),
								pWorkspaceData->Size(),
								pReservedData->Data(),
								pReservedData->Size());

	return lErr;
}

template long Memory<double>::RnnBackwardDataEx(long hHandle, long hRnnDesc, long hYDesc, long hYData, long hYDiff, long hHyDesc, long hHyDiff, long hCyDesc, long hCyDiff, long hWtDesc, long hWtData, long hHxDesc, long hHxData, long hCxDesc, long hCxData, long hXDesc, long hXDiff, long hdHxDesc, long hHxDiff, long hdCxDesc, long hCxDiff, long hWorkspace, int nWsCount, long hReserved, int nResCount);
template long Memory<float>::RnnBackwardDataEx(long hHandle, long hRnnDesc, long hYDesc, long hYData, long hYDiff, long hHyDesc, long hHyDiff, long hCyDesc, long hCyDiff, long hWtDesc, long hWtData, long hHxDesc, long hHxData, long hCxDesc, long hCxData, long hXDesc, long hXDiff, long hdHxDesc, long hHxDiff, long hdCxDesc, long hCxDiff, long hWorkspace, int nWsCount, long hReserved, int nResCount);


template <class T>
long Memory<T>::RnnBackwardWeights(long hHandle, long hRnnDesc, long hXDesc, long hXData, long hHxDesc, long hHxData, long hYDesc, long hYData, long hWorkspaceData, int nWsCount, long hWtDesc, long hWtDiff, long hReservedData, int nResCount)
{
	LONG lErr;
	cudnnHandle_t cudnn = GetCuDNN(hHandle);
	cudnnRNNDescriptor_t desc = (cudnnRNNDescriptor_t)m_rnnDesc.GetData(hRnnDesc);
	rnnDataHandle<T>* descX = (rnnDataHandle<T>*)m_rnnDataDesc1.GetData(hXDesc);
	rnnDataHandle<T>* descY = (rnnDataHandle<T>*)m_rnnDataDesc1.GetData(hYDesc);
	cudnnTensorDescriptor_t descHx = (cudnnTensorDescriptor_t)m_tensorDesc.GetData(hHxDesc);
	cudnnFilterDescriptor_t descWt = (cudnnFilterDescriptor_t)m_filterDesc.GetData(hWtDesc);
	MemoryItem* pXData;
	MemoryItem* pHxData;
	MemoryItem* pYData;
	MemoryItem* pWtDiff;
	MemoryItem* pWorkspaceData;
	MemoryItem* pReservedData;

	if (descX == NULL || descY == NULL)
		return ERROR_PARAM_OUT_OF_RANGE;

	if (lErr = m_memory.GetData(hXData, &pXData))
		return lErr;

	if (lErr = m_memory.GetData(hHxData, &pHxData))
		return lErr;

	if (lErr = m_memory.GetData(hYData, &pYData))
		return lErr;

	if (lErr = m_memory.GetData(hWtDiff, &pWtDiff))
		return lErr;

	if (lErr = m_memory.GetData(hWorkspaceData, &pWorkspaceData))
		return lErr;

	if (lErr = m_memory.GetData(hReservedData, &pReservedData))
		return lErr;

	lErr = cudnnRNNBackwardWeights(cudnn,
		desc,
		descX->MaxSeqLen(),
		descX->SeqTensors(),
		pXData->Data(),
		descHx,
		pHxData->Data(),
		descY->SeqTensors(),
		pYData->Data(),
		pWorkspaceData->Data(),
		pWorkspaceData->Size(),
		descWt,
		pWtDiff->Data(),
		pReservedData->Data(),
		pReservedData->Size());

	return lErr;
}

template long Memory<double>::RnnBackwardWeights(long hHandle, long hRnnDesc, long hXDesc, long hXData, long hHxDesc, long hHxData, long hYDesc, long hYData, long hWorkspace, int nWsCount, long hWtDesc, long hWtDiff, long hReserved, int nResCount);
template long Memory<float>::RnnBackwardWeights(long hHandle, long hRnnDesc, long hXDesc, long hXData, long hHxDesc, long hHxData, long hYDesc, long hYData, long hWorkspace, int nWsCount, long hWtDesc, long hWtDiff, long hReserved, int nResCount);

template <class T>
long Memory<T>::RnnBackwardWeightsEx(long hHandle, long hRnnDesc, long hXDesc, long hXData, long hHxDesc, long hHxData, long hYDesc, long hYData, long hWorkspaceData, int nWsCount, long hWtDesc, long hWtDiff, long hReservedData, int nResCount)
{
	LONG lErr;
	cudnnHandle_t cudnn = GetCuDNN(hHandle);
	cudnnRNNDescriptor_t desc = (cudnnRNNDescriptor_t)m_rnnDesc.GetData(hRnnDesc);
	cudnnRNNDataDescriptor_t descX = (cudnnRNNDataDescriptor_t)m_rnnDataDesc2.GetData(hXDesc);
	cudnnRNNDataDescriptor_t descY = (cudnnRNNDataDescriptor_t)m_rnnDataDesc2.GetData(hYDesc);
	cudnnTensorDescriptor_t descHx = (cudnnTensorDescriptor_t)m_tensorDesc.GetData(hHxDesc);
	cudnnFilterDescriptor_t descWt = (cudnnFilterDescriptor_t)m_filterDesc.GetData(hWtDesc);
	MemoryItem* pXData;
	MemoryItem* pHxData;
	MemoryItem* pYData;
	MemoryItem* pWtDiff;
	MemoryItem* pWorkspaceData;
	MemoryItem* pReservedData;

	if (lErr = m_memory.GetData(hXData, &pXData))
		return lErr;

	if (lErr = m_memory.GetData(hHxData, &pHxData))
		return lErr;

	if (lErr = m_memory.GetData(hYData, &pYData))
		return lErr;

	if (lErr = m_memory.GetData(hWtDiff, &pWtDiff))
		return lErr;

	if (lErr = m_memory.GetData(hWorkspaceData, &pWorkspaceData))
		return lErr;

	if (lErr = m_memory.GetData(hReservedData, &pReservedData))
		return lErr;

	lErr = cudnnRNNBackwardWeightsEx(cudnn,
									desc,
									descX,
									pXData->Data(),
									descHx,
									pHxData->Data(),
									descY,
									pYData->Data(),
									pWorkspaceData->Data(),
									pWorkspaceData->Size(),
									descWt,
									pWtDiff->Data(),
									pReservedData->Data(),
									pReservedData->Size());

	return lErr;
}

template long Memory<double>::RnnBackwardWeightsEx(long hHandle, long hRnnDesc, long hXDesc, long hXData, long hHxDesc, long hHxData, long hYDesc, long hYData, long hWorkspace, int nWsCount, long hWtDesc, long hWtDiff, long hReserved, int nResCount);
template long Memory<float>::RnnBackwardWeightsEx(long hHandle, long hRnnDesc, long hXDesc, long hXData, long hHxDesc, long hHxData, long hYDesc, long hYData, long hWorkspace, int nWsCount, long hWtDesc, long hWtDiff, long hReserved, int nResCount);

//end memory.cu