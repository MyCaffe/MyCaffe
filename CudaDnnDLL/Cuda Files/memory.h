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
#include "imgop.h"
#include "pca.h"
#include "rnnData.h"
#include "tsne_gp.h"
#include "tsne_g.h"
#include "nccl.h"
#include "ssd.h"
#include "extension.h"
#include <vector>
#include <algorithm>
#include <map>
#include <mutex>
#include "..\inc\FunctionIDs.h"
#include <cuda_fp16.h>

//=============================================================================
//	Flags
//=============================================================================

// Uncomment to use cudaMallocHost/cudaFreeHost (when commented out malloc/free are used)
//#define USE_PINNED_HOST_MEM 1

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
	TANH = CUDNN_ACTIVATION_TANH,
	ELU = CUDNN_ACTIVATION_ELU
};

enum RnnMode
{
	RNN_RELU = CUDNN_RNN_RELU,
	RNN_TANH = CUDNN_RNN_TANH,
	LSTM = CUDNN_LSTM
};

enum RnnDataLayout
{
	RNN_DATALAYOUT_SEQ_MAJOR = CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED,
	RNN_DATALAYOUT_BATCH_MAJOR = CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED
};


//=============================================================================
//	Classes
//=============================================================================

template <class T>
class HostBuffer
{
	private:
		size_t m_lCount;
		T* m_pMemory;

	public:
		HostBuffer(T* pMem, size_t lCount)
		{
			m_pMemory = pMem;
			m_lCount = lCount;
		}

		T* Data()
		{
			return m_pMemory;
		}

		size_t Count()
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
		map<T*, HostBuffer<T>*> m_rgActiveHostBuffers;
		MemoryCollection m_memory;
		MemoryCollection m_memoryPointers;
		HandleCollection<MAX_HANDLES_MEM> m_hostbuffers;
		HandleCollection<MAX_HANDLES> m_streams;
		HandleCollection<MAX_HANDLES> m_tensorDesc;
		HandleCollection<MAX_HANDLES> m_filterDesc;
		HandleCollection<MAX_HANDLES> m_convDesc;
		HandleCollection<MID_HANDLES> m_poolDesc;
		HandleCollection<MAX_HANDLES> m_lrnDesc;
		HandleCollection<MID_HANDLES> m_rnnDesc;
		HandleCollection<MID_HANDLES> m_rnnDataDesc1;
		HandleCollection<MID_HANDLES> m_rnnDataDesc2;
		HandleCollection<MAX_HANDLES> m_cudnn; 
		HandleCollection<MIN_HANDLES> m_pca;
		HandleCollection<MIN_HANDLES> m_tsnegp;
		HandleCollection<MIN_HANDLES> m_tsneg;
		HandleCollection<MIN_HANDLES> m_memtest;
		HandleCollection<MIN_HANDLES> m_imgop;
		HandleCollection<MIN_HANDLES> m_nccl;
		HandleCollection<MIN_HANDLES> m_ssd;
		HandleCollection<MIN_HANDLES> m_extensions;
		T m_tOne;
		T m_tZero;
#ifdef CUDNN_5
		HandleCollection<MID_HANDLES> m_dropoutDesc;
		HandleCollection<MAX_HANDLES> m_activationDesc;
		long m_hGlobalActivationSigmoid;
		long m_hGlobalActivationRelu;
		long m_hGlobalActivationTanh;
		long m_hGlobalActivationElu;
#endif
		map<void*, bool> m_memoryMap;
		map<cudnnHandle_t, int> m_cudnnRef;
		map<cudnnHandle_t, int> m_cudnnH2Dev;
		map<int, cudnnHandle_t> m_cudnnDev2H;

		map<cudaStream_t, int> m_streamRef;
		map<cudaStream_t, int> m_streamH2Dev;
		map<cudaStream_t, int> m_streamH2Idx;
		map<cudaStream_t, cudnnHandle_t> m_streamH2CudnnH;
		map<cudnnHandle_t, int> m_streamCudnnRef;
		map<int, map<int, cudaStream_t>> m_streamDev2Idx2H;
		std::mutex m_sync;
		

		void free_cudnn(cudnnHandle_t h)
		{
			std::unique_lock<std::mutex> lock(m_sync);

			// If the cudnn is NOT a shared cudnn (e.g. created with stream > 0)
			if (m_cudnnRef.find(h) == m_cudnnRef.end())
			{
				// And it is not associated with a shared stream, then destroy it.
				if (m_streamCudnnRef.find(h) == m_streamCudnnRef.end())
				{
					cudnnDestroy(h);
					return;
				}

				// Otherwise decrement the shared ref count.
				m_streamCudnnRef[h]--;
				// And if there are no more references, then destroy the cudnn shared
				// with the stream.
				if (m_streamCudnnRef[h] <= 0)
				{
					cudnnDestroy(h);
					m_streamCudnnRef.erase(h);
				}

				return;
			}

			// Otherwise, if the cudnn is a shared cudnn (e.g. created with stream == 0), then decrement the ref count.
			m_cudnnRef[h]--;
			// And if no more references exist, destroy the cudnn and remove it from the maps.
			if (m_cudnnRef[h] <= 0)
			{
				int nDeviceID = m_cudnnH2Dev[h];
				m_cudnnRef.erase(h);
				m_cudnnH2Dev.erase(h);
				m_cudnnDev2H.erase(nDeviceID);
				cudnnDestroy(h);

				cudaStream_t stream = NULL;
				for (map<cudaStream_t, cudnnHandle_t>::iterator it = m_streamH2CudnnH.begin(); it != m_streamH2CudnnH.end(); it++)
				{
					if (it->second == h)
					{
						stream = it->first;
						break;
					}
				}

				if (stream != NULL)
					m_streamH2CudnnH.erase(stream);
			}
		}

		void free_stream(cudaStream_t h)
		{
			std::unique_lock<std::mutex> lock(m_sync);

			// If the stream is not an indexed, shared stream, destroy it.
			if (m_streamRef.find(h) == m_streamRef.end())
			{
				cudaStreamDestroy(h);
				return;
			}

			// Otherwise, decrement the reference count for the stream.
			m_streamRef[h]--;
			// And if no more references are used, destroy it and remove it from the maps.
			if (m_streamRef[h] <= 0)
			{
				int nDeviceID = m_streamH2Dev[h];
				int nIndex = m_streamH2Idx[h];
				m_streamRef.erase(h);
				m_streamH2Dev.erase(h);
				m_streamH2Idx.erase(h);				
				m_streamDev2Idx2H[nDeviceID].erase(nIndex);

				if (m_streamDev2Idx2H[nDeviceID].size() == 0)
					m_streamDev2Idx2H.erase(nDeviceID);

				if (m_streamH2CudnnH.find(h) != m_streamH2CudnnH.end())
					m_streamH2CudnnH.erase(h);
			}
		}

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

		long alloc_host(void** ppDst, size_t lSize, bool bPinned = true);
		long free_host(void* pData);
		long convertBaseType2Half(size_t lCount, T* pSrc, size_t* plSize, __half** ppDst);
		long convertHalf2BaseType(size_t lCount, void* pSrc, T* pDst, cudaMemcpyKind kind);

		long GetPointer(HANDLE_TYPE ht, long hHandle, void** ppPtr);

		long CheckMemoryAttributes(long hSrc, int nSrcDeviceID, long hDst, int nDstDeviceID, bool* pbResult);
		long GetDeviceMemory(int nDeviceID, T* plTotal, T* plFree, T* plUsed, bool* pbEstimate);

		long AllocMemory(int nDeviceID, bool bHalf, size_t lCount, T* pSrc, long hStream, long* phHandle);
		long FreeMemory(long hHandle);
		long GetMemory(long hHandle, MemoryItem** ppItem);
		long SetMemory(long hHandle, T* pSrc, size_t lCount, long hStream);
		long SetMemoryAt(long hHandle, T* pSrc, size_t lCount, size_t nOffset);

		T* GetMemoryToHost(long hHandle, size_t* plCount = NULL);
		long SetMemoryToHost(long hHandle, T* pDst);

		long AllocHostBuffer(size_t lCount, long* phHandle);
		long FreeHostBuffer(long hHandle);
		HostBuffer<T>* GetHostBuffer(long hHandle);
		long SetHostBuffer(long hHandle, size_t lCount, T* pData);
		bool IsHostBuffer(T* pf);

		long CopyGpuToHost(long lCount, long hGpuSrc, long hHostDst);
		long CopyHostToGpu(long lCount, long hHostSrc, long hGpuDst);

		long CopyToHost(size_t lCount, T* pDst, void* pSrc, bool bSrcOnDevice, bool bHalf);
		long AllocHost(size_t lCount, T** ppDst, void* pSrc, bool bSrcOnDevice, bool bHalf, bool bPinned = true);
		long FreeHost(T* pDst);

		long AllocHost(LPTSTR* ppDst, LPTSTR pSrc);
		long FreeHost(LPTSTR pDst);

		long CreateMemoryPointer(int nDeviceID, bool bHalf, T* pData, size_t lSize, long* phHandle);
		long CreateMemoryPointer(long hData, long lOffset, size_t lCount, long* phHandle);
		long FreeMemoryPointer(long hData);

		long CreateStream(long* phHandle, bool bNonBlocking = false, int nIndex = -1);
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
		long SetTensorDesc(long hHandle, int n, int c, int h, int w, int stride_n, int stride_c, int stride_h, int stride_w, bool bHalf);
		long SetTensorDesc(long hHandle, int* rgDim, int* rgStride, int nCount, bool bHalf);
		long AddTensor(long hHandle, T fAlpha, long hSrcDesc, long hSrc, int nSrcOffset, T fBeta, long hDstDesc, long hDst, int nDstOffset);

		long CreateFilterDesc(long* phHandle);
		long FreeFilterDesc(long hHandle);
		cudnnFilterDescriptor_t GetFilterDesc(long hHandle);
		long SetFilterDesc(long hHandle, int n, int c, int h, int w, bool bHalf);
		long SetFilterDesc(long hHandle, int* rgDim, int nCount, bool bHalf);

		long CreateConvolutionDesc(long* phHandle);
		long FreeConvolutionDesc(long hHandle);
		cudnnConvolutionDescriptor_t GetConvolutionDesc(long hHandle);
		long SetConvolutionDesc(long hHandle, int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h, int dilation_w, bool bHalf, bool bUseTensorCores);
	    long GetConvolutionInfo(long hHandle, long hBottomDesc, long hFilterDesc, long hConvDesc, long hTopDesc, size_t lWsLimitInBytes, bool bUseTensorCores, long* palgoFwd, size_t* plWsSizeFwd, long* palgoBwdFilter, size_t* plWsSizeBwdFilter, long* palgoBwdData, size_t* plWsSizeBwdData, int nPreferredFwdAlgo = -1);
		long ConvolutionForward(long hHandle, T fAlpha, long hBottomDesc, long hBottomData, int nBottomOffset, long hFilterDesc, long hWeight, int nWeightOffset, long hConvDesc, long algo, long hWorkspace, int nWorkspaceOffset, size_t lWorkspaceSize, T fBeta, long hTopDesc, long hTopData, int nTopOffset, bool bSyncStream);
		long ConvolutionBackwardBias(long hHandle, T fAlpha, long hTopDesc, long hTopDiff, int nTopOffset, T fBeta, long hBiasDesc, long hBiasDiff, int nBiasOffset, bool bSyncStream);
		long ConvolutionBackwardFilter(long hHandle, T fAlpha, long hBottomDesc, long hBottomData, int nBottomOffset, long hTopDesc, long hTopDiff, int nTopOffset, long hConvDesc, long algo, long hWorkspace, int nWorkspaceOffset, size_t lWorkspaceSize, T fBeta, long hFilterDesc, long hWeightDiff, int nWeightOffsete, bool bSyncStream);
		long ConvolutionBackwardData(long hHandle, T fAlpha, long hFilterDesc, long hWeight, int nWeightOffset, long hTopDesc, long hTopDiff, int nTopOffset, long hConvDesc, long algo, long hWorkspace, int nWorkspaceOffset, size_t lWorkspaceSize, T fBeta, long hBottomDesc, long hBottomDiff, int nBottomOffset, bool bSyncStream);

		long CreatePoolingDesc(long* phHandle);
		long FreePoolingDesc(long hHandle);
		cudnnPoolingDescriptor_t GetPoolingDesc(long hHandle);
		long SetPoolingDesc(long hHandle, PoolingMethod method, int h, int w, int pad_h, int pad_w, int stride_h, int stride_w);
		long PoolingForward(long hHandle, long hPoolingDesc, T fAlpha, long hBottomDesc, long hBottomData, T fBeta, long hTopDesc, long hTopData);
		long PoolingBackward(long hHandle, long hPoolingDesc, T fAlpha, long hTopDataDesc, long hTopData, long hTopDiffDesc, long hTopDiff, long hBottomDataDesc, long hBottomData, T fBeta, long hBottomDiffDesc, long hBottomDiff);

		long DeriveBatchNormDesc(long hFwdScaleBiasMeanVarDesc, long hFwdBottomDesc, long hBwdScaleBiasMeanVarDesc, long hBwdBottomDesc, int mode);
		long BatchNormForward(long hHandle, int mode, T fAlpha, T fBeta, long hFwdBottomDesc, long hBottomData, long hFwdTopDesc, long hTopData, long hFwdScaleBiasMeanVarDesc, long hScaleData, long hBiasData, T fFactor, long hGlobalMean, long hGlobalVar, T fEps, long hSaveMean, long hSaveVar, bool bTraining);
		long BatchNormBackward(long hHandle, int mode, T fAlphaDiff, T fBetaDiff, T fAlphaParamDiff, T fBetaParamDiff, long hBtmBottomDesc, long hBottomData, long hTopDiffDesc, long hTopDiff, long hBottomDiffDesc, long hBottomDiff, long hBwdScaleBiasMeanVarDesc, long hScaleData, long hScaleDiff, long hBiasDiff, T fEps, long hSaveMean, long hSaveVar);

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

		long EluForward(long hHandle, T fAlpha, long hBottomDesc, long hBottomData, T fBeta, long hTopDesc, long hTopData);
		long EluBackward(long hHandle, T fAlpha, long hTopDataDesc, long hTopData, long hTopDiffDesc, long hTopDiff, long hBottomDataDesc, long hBottomData, T fBeta, long hBottomDiffDesc, long hBottomDiff);

		long SigmoidForward(long hHandle, T fAlpha, long hBottomDesc, long hBottomData, T fBeta, long hTopDesc, long hTopData);
		long SigmoidBackward(long hHandle, T fAlpha, long hTopDataDesc, long hTopData, long hTopDiffDesc, long hTopDiff, long hBottomDataDesc, long hBottomData, T fBeta, long hBottomDiffDesc, long hBottomDiff);

		long ReLUForward(long hHandle, T fAlpha, long hBottomDesc, long hBottomData, T fBeta, long hTopDesc, long hTopData);
		long ReLUBackward(long hHandle, T fAlpha, long hTopDataDesc, long hTopData, long hTopDiffDesc, long hTopDiff, long hBottomDataDesc, long hBottomData, T fBeta, long hBottomDiffDesc, long hBottomDiff);

		long SoftmaxForward(long hHandle, T fAlpha, long hBottomDesc, long hBottomData, T fBeta, long hTopDesc, long hTopData);
		long SoftmaxBackward(long hHandle, T fAlpha, long hTopDataDesc, long hTopData, long hTopDiffDesc, long hTopDiff, T fBeta, long hBottomDiffDesc, long hBottomDiff);

		long CreateRnnDataDesc1(long* phHandle);
		long FreeRnnDataDesc1(long hHandle);
		rnnDataHandle<T>* GetRnnDataDesc1(long hHandle);
		long SetRnnDataDesc1(long hRnnDataDesc, RnnDataLayout layout, int nMaxSeqLen, int nBatchSize, int nVectorSize, int* rgSeqLen);

		long CreateRnnDataDesc2(long* phHandle);
		long FreeRnnDataDesc2(long hHandle);
		cudnnRNNDataDescriptor_t GetRnnDataDesc2(long hHandle);
		long SetRnnDataDesc2(long hRnnDataDesc, RnnDataLayout layout, int nMaxSeqLen, int nBatchSize, int nVectorSize, int* rgSeqLen);

		long CreateRnnDesc(long* phHandle);
		long FreeRnnDesc(long hHandle);
		cudnnRNNDescriptor_t GetRnnDesc(long hHandle);
		long SetRnnDesc(long hHandle, long hRnnDesc, int nHiddenCount, int nNumLayers, long hDropoutDesc, RnnMode nMode, bool bUseTensorCores);
		long GetRnnParamCount(long hHandle, long hRnnDesc, long hXDesc, int* pnCount);
		long GetRnnWorkspaceCount(long hHandle, long hRnnDesc, long hXDesc, int* pnWsCount, int* pnResCount);
		long GetRnnParamCountEx(long hHandle, long hRnnDesc, long hXDesc, int* pnCount);
		long GetRnnWorkspaceCountEx(long hHandle, long hRnnDesc, long hXDesc, int* pnWsCount, int* pnResCount);
		long GetRnnLinLayerParams(long hHandle, long hRnnDesc, int nLayer, long hXDesc, long hWtDesc, long hWtData, int nLinLayer, int* pnWtCount, long* phWt, int* pnBiasCount, long* phBias);
		long GetRnnLinLayerParamsEx(long hHandle, long hRnnDesc, int nLayer, long hXDesc, long hWtDesc, long hWtData, int nLinLayer, int* pnWtCount, long* phWt, int* pnBiasCount, long* phBias);
		long RnnForward(long hHandle, long hRnnDesc, long hXDesc, long hXData, long hHxDesc, long hHxData, long hCxDesc, long hCxData, long hWtDesc, long hWtData, long hYDesc, long hYData, long hHyDesc, long hHyData, long hCyDesc, long hCyData, long hWorkspace, int nWsCount, long hReserved, int nResCount, bool bTraining);
		long RnnBackwardData(long hHandle, long hRnnDesc, long hYDesc, long hYData, long hYDiff, long hHyDesc, long hHyDiff, long hCyDesc, long hCyDiff, long hWtDesc, long hWtData, long hHxDesc, long hHxData, long hCxDesc, long hCxData, long hXDesc, long hXDiff, long hdHxDesc, long hHxDiff, long hdCxDesc, long hCxDiff, long hWorkspace, int nWsCount, long hReserved, int nResCount);
		long RnnBackwardWeights(long hHandle, long hRnnDesc, long hXDesc, long hXData, long hHxDesc, long hHxData, long hYDesc, long hYData, long hWorkspace, int nWsCount, long hWtDesc, long hWtDiff, long hReserved, int nResCount);
		long RnnForwardEx(long hHandle, long hRnnDesc, long hXDesc, long hXData, long hHxDesc, long hHxData, long hCxDesc, long hCxData, long hWtDesc, long hWtData, long hYDesc, long hYData, long hHyDesc, long hHyData, long hCyDesc, long hCyData, long hWorkspace, int nWsCount, long hReserved, int nResCount, bool bTraining);
		long RnnBackwardDataEx(long hHandle, long hRnnDesc, long hYDesc, long hYData, long hYDiff, long hHyDesc, long hHyDiff, long hCyDesc, long hCyDiff, long hWtDesc, long hWtData, long hHxDesc, long hHxData, long hCxDesc, long hCxData, long hXDesc, long hXDiff, long hdHxDesc, long hHxDiff, long hdCxDesc, long hCxDiff, long hWorkspace, int nWsCount, long hReserved, int nResCount);
		long RnnBackwardWeightsEx(long hHandle, long hRnnDesc, long hXDesc, long hXData, long hHxDesc, long hHxData, long hYDesc, long hYData, long hWorkspace, int nWsCount, long hWtDesc, long hWtDiff, long hReserved, int nResCount);

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

		long CreateImageOp(int nNum, T fBrightnessProb, T fBrightnessDelta, T fContrastProb, T fContrastLower, T fContrastUpper, T fSaturationProb, T fSaturationLower, T fSaturationUpper, long lRandomSeed, long* phHandle);
		long FreeImageOp(long hHandle);
		imgopHandle<T>* GetImageOp(long hHandle);
		long DistortImage(long hHandle, int nCount, int nNum, int nDim, long hX, long hY);

		long CreateNCCL(int nGpuID, int nCount, int nRank, char* szId, Math<T>* pMath, long* phHandle);
		long FreeNCCL(long hHandle);
		ncclHandle<T>* GetNCCL(long hHandle);
		long Memory<T>::SetNCCL(ncclHandle<T>* pNccl, long* hHandle);
		long NcclInitSingleProcess(long lBufferCount, long rgNcclHandle[], int nCount);
		long NcclInitMultiProcess(long lBufferCount, long hNccl);
		long NcclBroadcast(long hNccl, long hStream, long hX, int nCount);
		long NcclAllReduce(long hNccl, long hStream, long hX, int nCount, NCCL_OP op, T fScale);

		long CreateSSD(int nGpuID, int nNumClasses, bool bShareLocation, int nLocClasses, int nBackgroundLabelId, bool bUseDifficultGt, SsdMiningType miningType, SsdMatchingType matchingType, T fOverlapThreshold, bool bUsePriorForMatching, SsdCodeType codeType, bool bEncodeVariantInTgt, bool bBpInside, bool bIgnoreCrossBoundaryBbox, bool bUsePriorForNms, SsdConfLossType confLossType, SsdLocLossType locLossType, T fNegPosRatio, T fNegOverlap, int nSampleSize, bool bMapObjectToAgnostic, bool bNmsActive, T fNmsThreshold, int nTopK, T fEta, Math<T>* pMath, long* phHandle);
		long FreeSSD(long hHandle);
		ssdHandle<T>* GetSSD(long hHandle);
		long SetupSSD(long hSsd, int nNum, int nNumPriors, int nNumGt);
		long SsdMultiboxLossForward(long hSsd, int nLocDataCount, long hLocData, int nConfDataCount, long hConfData, int nPriorDataCount, long hPriorData, int nGtDataCount, long hGtData, int* pnNumMatches, int* pnNumNegs);
		long SsdEncodeLocPrediction(long hSsd, int nLocPredCount, long hLocPred, int nLocGtCount, long hLocGt);
		long SsdEncodeConfPrediction(long hSsd, int nConfPredCount, long hConfPred, int nConfGtCount, long hConfGt);
		long SsdGetAllMatchIndices(long hSsd, vector<map<int, vector<int>>>* pall_match_indices);
		long SsdGetAllNegIndices(LONG hSsd, vector<vector<int>>* pall_neg_indices);

		long CreateExtensionFloat(HMODULE hParent, LONG lKernelIdx, LPTSTR pszDllPath, long *phHandle);
		long CreateExtensionDouble(HMODULE hParent, LONG lKernelIdx, LPTSTR pszDllPath, long *phHandle);
		long FreeExtension(long hHandle);
		extensionHandle<T>* GetExtension(long hHandle);
		long ExtensionRun(long hHandle, long lfnIdx, T* pfInput, long lCount, T** ppfOutput, long* plCount, LPTSTR pszErr, LONG lErrMax);
};


//=============================================================================
//	Inline Methods
//=============================================================================

template <class T>
inline long Memory<T>::alloc_host(void** ppDst, size_t lSize, bool bPinned)
{
	LONG lErr;

#ifdef USE_PINNED_HOST_MEM
	if (lErr = cudaMallocHost(ppDst, (size_t)lSize))
		return lErr;
#else
	if (bPinned)
	{
		if (lErr = cudaMallocHost(ppDst, (size_t)lSize))
			return lErr;
	}
	else
	{
		*ppDst = malloc((size_t)lSize);
		if (*ppDst == NULL)
			return ERROR_MEMORY_OUT;
	}

	m_memoryMap[*ppDst] = bPinned;
#endif

	return 0;
}

template <class T>
inline long Memory<T>::free_host(void* p)
{
	if (p == NULL)
		return 0;

#ifdef USE_PINNED_HOST_MEM
	return cudaFreeHost(p);
#else
	map<void*, bool>::iterator it = m_memoryMap.find(p);
	if (it == m_memoryMap.end())
		return ERROR_MEMORY_NOT_FOUND;

	if (it->second)
		cudaFreeHost(p);
	else
		free(p);

	m_memoryMap.erase(p);
	return 0;
#endif
}

template <class T>
inline long Memory<T>::convertBaseType2Half(size_t lCount, T* pSrc, size_t* plSize, __half** ppDst)
{
	LONG lErr;

	long long llSize = lCount * sizeof(__half);
	if (llSize > SIZE_MAX)
		return ERROR_MEMORY_RANGE_EXCEEDED;

	if (pSrc != NULL)
	{
		__half* pData = NULL;

		if (lErr = alloc_host((void**)&pData, (size_t)llSize))
			return lErr;

		for (int i = 0; i < lCount; i++)
		{
			pData[i] = __float2half((float)pSrc[i]);
		}

		*ppDst = pData;
	}

	*plSize = (size_t)llSize;

	return 0;
}

template <class T>
inline long Memory<T>::convertHalf2BaseType(size_t lCount, void* pSrc, T* pDst, cudaMemcpyKind kind)
{
	LONG lErr;

	__half* pSrc1 = NULL;
	size_t lSize1 = lCount * sizeof(__half);

	if (lErr = alloc_host((void**)&pSrc1, lSize1))
		return lErr;

	if (lErr = cudaMemcpy(pSrc1, pSrc, lSize1, kind))
	{
		free_host(pSrc1);
		return lErr;
	}

	for (int i = 0; i < lCount; i++)
	{
		pDst[i] = (T)__half2float(pSrc1[i]);
	}

	free_host(pSrc1);
	pSrc1 = NULL;

	return 0;
}



//-----------------------------------------------------------------------------
//	Memory
//-----------------------------------------------------------------------------

template <class T>
inline long Memory<T>::GetPointer(HANDLE_TYPE ht, long hHandle, void** ppPtr)
{
	switch (ht)
	{
		case HT_MEMORY:
			MemoryItem* pmi;
			m_memory.GetData(hHandle, &pmi);
			*ppPtr = pmi->Data();
			break;

		default:
			return ERROR_NOT_SUPPORTED;
	}

	return 0;
}

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
inline long Memory<T>::CreateMemoryPointer(long hData, long lOffset, size_t lCount, long* phHandle)
{
	long lErr;
	size_t lSize = m_memory.GetSize(lCount, sizeof(T));
	MemoryItem* pData = NULL;

	if (lErr = m_memory.GetData(hData, &pData))
		return lErr;

	T* data = (T*)pData->Data();

	if (lOffset > 0)
		data += lOffset;

	return CreateMemoryPointer(pData->DeviceID(), pData->IsHalf(), data, lSize, phHandle);
}

template <class T>
inline long Memory<T>::CreateMemoryPointer(int nDeviceID, bool bHalf, T* data, size_t lSize, long* phHandle)
{
	long lErr;
	long hHandle = 0;

	if (phHandle == NULL)
		return ERROR_PARAM_NULL;

	if (lErr = m_memoryPointers.Allocate(nDeviceID, bHalf, data, lSize, &hHandle))
		return lErr;

	// Move the handle into the range [MAX_HANDLES, MAX_HANDLES*2]
	// to indicate that it is a memory pointer reference.
	hHandle += MAX_ITEMS;
	*phHandle = hHandle;

	return 0;
}

template <class T>
inline long Memory<T>::FreeMemoryPointer(long hData)
{
	return m_memoryPointers.Free(hData - MAX_ITEMS);
}


template <class T>
inline long Memory<T>::AllocMemory(int nDeviceID, bool bHalf, size_t lCount, T* pSrc, long hStream, long* phHandle)
{
	LONG lErr;
	cudaStream_t pStream = NULL;

	if (hStream > 0)
		pStream = (cudaStream_t)m_streams.GetData(hStream);

	long long llSize = m_memory.GetSize(lCount, sizeof(T));
	if (llSize > SIZE_MAX)
		return ERROR_MEMORY_RANGE_EXCEEDED;

	void* pSrc1 = pSrc;

	if (bHalf)
	{
		lCount = m_memory.GetCount(lCount);

		if (lErr = convertBaseType2Half(lCount, pSrc, (size_t*)&llSize, (__half**)&pSrc1))
			return lErr;
	}

	lErr = m_memory.Allocate(nDeviceID, bHalf, (size_t)llSize, pSrc1, pStream, phHandle);

	if (bHalf && pSrc1 != NULL)
		free_host(pSrc1);

	return lErr;
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
inline T* Memory<T>::GetMemoryToHost(long hHandle, size_t* plCount)
{
	MemoryItem *pItem;

	if (m_memory.GetData(hHandle, &pItem))
		return NULL;

	T* pHost = NULL;
	size_t lCount = pItem->Size() / sizeof(T);

	if (AllocHost(lCount, &pHost, (T*)pItem->Data(), TRUE, pItem->IsHalf()))
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

	size_t lSize = pItem->Size();
	void* pDst1 = pDst;

	if (pItem->IsHalf())
	{
		if (lErr = alloc_host(&pDst1, lSize))
			return lErr;
	}

	if (lErr = pItem->GetData(lSize, pDst1))
	{
		if (pItem->IsHalf())
			free_host(pDst1);

		return lErr;
	}

	if (pItem->IsHalf())
	{
		size_t lCount = lSize / sizeof(__half);
		lErr = convertHalf2BaseType(lCount, pDst1, pDst, cudaMemcpyDeviceToHost);
	}

	return lErr;
}

template <class T>
inline long Memory<T>::SetMemory(long hHandle, T* pSrc, size_t lCount, long hStream)
{
	cudaStream_t pStream = NULL;

	if (hStream > 0)
		pStream = (cudaStream_t)m_streams.GetData(hStream);

	long long lSize = lCount;
	if (lCount < SIZE_MAX - 10)
		lSize = lCount * sizeof(T);

	if (lSize > SIZE_MAX)
		return ERROR_MEMORY_RANGE_EXCEEDED;

	LONG lErr;
	size_t lSize1 = (size_t)lSize;
	void* pSrc1 = (void*)pSrc;
	MemoryItem* pItem;

	if (lErr = m_memory.GetData(hHandle, &pItem))
		return lErr;

	if (pItem->IsHalf())
	{
		if (lErr = convertBaseType2Half(lCount, pSrc, &lSize1, (__half**)&pSrc1))
			return lErr;
	}

	lErr = m_memory.SetData(pItem, pItem->IsHalf(), lSize1, pSrc1, pStream);

	if (pItem->IsHalf())
		free_host(pSrc1);

	return lErr;
}

template <class T>
inline long Memory<T>::SetMemoryAt(long hHandle, T* pSrc, size_t lCount, size_t nOffset)
{
	long long lSize = lCount * sizeof(T);
	if (lSize > SIZE_MAX)
		return ERROR_MEMORY_RANGE_EXCEEDED;

	long long lOffset = nOffset * sizeof(T);
	if (lOffset > SIZE_MAX)
		return ERROR_MEMORY_RANGE_EXCEEDED;

	LONG lErr;
	size_t lSize1 = (size_t)lSize;
	size_t lOffset1 = (size_t)lOffset;
	void* pSrc1 = (void*)pSrc;
	MemoryItem* pItem;

	if (lErr = m_memory.GetData(hHandle, &pItem))
		return lErr;

	if (pItem->IsHalf())
	{
		if (lErr = convertBaseType2Half(lCount, pSrc, &lSize1, (__half**)&pSrc1))
			return lErr;

		lOffset1 = nOffset * sizeof(__half);
	}

	lErr = m_memory.SetDataAt(pItem, pItem->IsHalf(), (size_t)lSize1, pSrc1, (size_t)lOffset1);

	if (pItem->IsHalf())
		free_host(pSrc1);

	return lErr;
}


template <class T>
inline long Memory<T>::FreeHost(T* pDst)
{
	return free_host(pDst);
}

template <class T>
inline long Memory<T>::FreeHost(LPTSTR pDst)
{
	return free_host(pDst);
}

template <class T>
inline HostBuffer<T>* Memory<T>::GetHostBuffer(long hHandle)
{
	return (HostBuffer<T>*)m_hostbuffers.GetData(hHandle);
}

template <class T>
inline long Memory<T>::SetHostBuffer(long hHandle, size_t lCount, T* pData)
{
	HostBuffer<T>* p = (HostBuffer<T>*)m_hostbuffers.GetData(hHandle);
	if (p == NULL)
		return ERROR_MEMORY_OUT;

	long long lSize = lCount * sizeof(T);
	if (lSize > SIZE_MAX)
		return ERROR_MEMORY_RANGE_EXCEEDED;

	return cudaMemcpy(p->Data(), pData, (size_t)lSize, cudaMemcpyHostToHost);
}


//-----------------------------------------------------------------------------
//	Streams
//-----------------------------------------------------------------------------

template <class T>
inline long Memory<T>::FreeStream(long hHandle)
{
	cudaStream_t h = (cudaStream_t)m_streams.Free(hHandle);
	
	if (h != NULL)
		free_stream(h);

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
		free_cudnn(h);

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
inline long Memory<T>::SetTensorDesc(long hHandle, int n, int c, int h, int w, int stride_n, int stride_c, int stride_h, int stride_w, bool bHalf)
{
	LONG lErr;
	cudnnTensorDescriptor_t desc = (cudnnTensorDescriptor_t)m_tensorDesc.GetData(hHandle);
	cudnnDataType_t type = (sizeof(T) == 4) ? (bHalf) ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;
	if (lErr = cudnnSetTensor4dDescriptorEx(desc, type, n, c, h, w, stride_n, stride_c, stride_h, stride_w))
		return lErr | ERROR_CUDNN_OFFSET;

	return CUDNN_STATUS_SUCCESS;
}

template <class T>
inline long Memory<T>::SetTensorDesc(long hHandle, int* rgDim, int* rgStride, int nCount, bool bHalf)
{
	LONG lErr;
	cudnnTensorDescriptor_t desc = (cudnnTensorDescriptor_t)m_tensorDesc.GetData(hHandle);
	cudnnDataType_t type = (sizeof(T) == 4) ? (bHalf) ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;
	if (lErr = cudnnSetTensorNdDescriptor(desc, type, nCount, rgDim, rgStride))
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
inline long Memory<T>::SetFilterDesc(long hHandle, int n, int c, int h, int w, bool bHalf)
{
	LONG lErr;
	cudnnFilterDescriptor_t desc = (cudnnFilterDescriptor_t)m_filterDesc.GetData(hHandle);
	cudnnDataType_t type = (sizeof(T) == 4) ? (bHalf) ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;
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
inline long Memory<T>::SetFilterDesc(long hHandle, int* rgDim, int nCount, bool bHalf)
{
	LONG lErr;
	cudnnFilterDescriptor_t desc = (cudnnFilterDescriptor_t)m_filterDesc.GetData(hHandle);
	cudnnDataType_t type = (sizeof(T) == 4) ? (bHalf) ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;
	if (lErr = cudnnSetFilterNdDescriptor(desc, type, CUDNN_TENSOR_NCHW, nCount, rgDim))
		return lErr | ERROR_CUDNN_OFFSET;

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
inline long Memory<T>::SetConvolutionDesc(long hHandle, int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h, int dilation_w, bool bHalf, bool bUseTensorCores)
{
	LONG lErr;
	cudnnConvolutionDescriptor_t desc = (cudnnConvolutionDescriptor_t)m_convDesc.GetData(hHandle);
#ifdef CUDNN_6
	cudnnDataType_t type = (sizeof(T) == 4) ? (bHalf) ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;
	if (lErr = cudnnSetConvolution2dDescriptor(desc, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, CUDNN_CROSS_CORRELATION, type))
		return lErr | ERROR_CUDNN_OFFSET;
#else
	if (lErr = cudnnSetConvolution2dDescriptor(desc, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, CUDNN_CROSS_CORRELATION))
		return lErr | ERROR_CUDNN_OFFSET;
#endif

	if (bUseTensorCores)
	{
		if (lErr = cudnnSetConvolutionMathType(desc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION))
			return lErr | ERROR_CUDNN_OFFSET;
	}

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
inline long Memory<T>::FreeRnnDesc(long hHandle)
{
	cudnnRNNDescriptor_t desc = (cudnnRNNDescriptor_t)m_rnnDesc.Free(hHandle);

	if (desc != NULL)
		cudnnDestroyRNNDescriptor(desc);

	return 0;
}

template <class T>
inline cudnnRNNDescriptor_t Memory<T>::GetRnnDesc(long hHandle)
{
	return (cudnnRNNDescriptor_t)m_rnnDesc.GetData(hHandle);
}

template <class T>
inline long Memory<T>::SetRnnDesc(long hHandle, long hRnnDesc, int nHiddenCount, int nNumLayers, long hDropoutDesc, RnnMode mode, bool bUseTensorCores)
{
	LONG lErr;
	cudnnHandle_t cudnn = GetCuDNN(hHandle);
	cudnnRNNDescriptor_t desc = (cudnnRNNDescriptor_t)m_rnnDesc.GetData(hRnnDesc);
	cudnnDropoutDescriptor_t descDropout = NULL;
	cudnnDataType_t computeType = (sizeof(T) == sizeof(double)) ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;

	if (hDropoutDesc != 0)
		descDropout = (cudnnDropoutDescriptor_t)m_dropoutDesc.GetData(hDropoutDesc);
	
#ifdef CUDA11_0
	if (lErr = cudnnSetRNNDescriptor_v6(cudnn, desc, nHiddenCount, nNumLayers, descDropout, CUDNN_LINEAR_INPUT, CUDNN_UNIDIRECTIONAL, (cudnnRNNMode_t)mode, CUDNN_RNN_ALGO_STANDARD, computeType))
		return lErr | ERROR_CUDNN_OFFSET;
#else
	if (lErr = cudnnSetRNNDescriptor(cudnn, desc, nHiddenCount, nNumLayers, descDropout, CUDNN_LINEAR_INPUT, CUDNN_UNIDIRECTIONAL, (cudnnRNNMode_t)mode, CUDNN_RNN_ALGO_STANDARD, computeType))
		return lErr | ERROR_CUDNN_OFFSET;
#endif

	if (bUseTensorCores)
	{
		if (lErr = cudnnSetRNNMatrixMathType(desc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION))
			return lErr | ERROR_CUDNN_OFFSET;
	}

	return CUDNN_STATUS_SUCCESS;
}


template <class T>
inline long Memory<T>::FreeRnnDataDesc1(long hHandle)
{
	rnnDataHandle<T>* desc = (rnnDataHandle<T>*)m_rnnDataDesc1.Free(hHandle);

	if (desc != NULL)
	{
		desc->CleanUp();
		delete desc;
	}

	return 0;
}

template <class T>
inline rnnDataHandle<T>* Memory<T>::GetRnnDataDesc1(long hHandle)
{
	return (rnnDataHandle<T>*)m_rnnDataDesc1.GetData(hHandle);
}

template <class T>
inline long Memory<T>::SetRnnDataDesc1(long hRnnDataDesc, RnnDataLayout layout, int nMaxSeqLen, int nBatchSize, int nVectorSize, int* rgSeqLen)
{
	LONG lErr;
	rnnDataHandle<T>* desc = (rnnDataHandle<T>*)m_rnnDataDesc1.GetData(hRnnDataDesc);
	cudnnDataType_t computeType = (sizeof(T) == sizeof(double)) ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;

	if (lErr = desc->Set(computeType, (cudnnRNNDataLayout_t)layout, nMaxSeqLen, nBatchSize, nVectorSize, rgSeqLen))
		return lErr | ERROR_CUDNN_OFFSET;

	return CUDNN_STATUS_SUCCESS;
}

template <class T>
inline long Memory<T>::FreeRnnDataDesc2(long hHandle)
{
	cudnnRNNDataDescriptor_t desc = (cudnnRNNDataDescriptor_t)m_rnnDataDesc2.Free(hHandle);

	if (desc != NULL)
		cudnnDestroyRNNDataDescriptor(desc);

	return 0;
}

template <class T>
inline cudnnRNNDataDescriptor_t Memory<T>::GetRnnDataDesc2(long hHandle)
{
	return (cudnnRNNDataDescriptor_t)m_rnnDataDesc2.GetData(hHandle);
}

template <class T>
inline long Memory<T>::SetRnnDataDesc2(long hRnnDataDesc, RnnDataLayout layout, int nMaxSeqLen, int nBatchSize, int nVectorSize, int* rgSeqLen)
{
	LONG lErr;
	cudnnRNNDataDescriptor_t desc = (cudnnRNNDataDescriptor_t)m_rnnDataDesc2.GetData(hRnnDataDesc);
	cudnnDataType_t computeType = (sizeof(T) == sizeof(double)) ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;

	if (lErr = cudnnSetRNNDataDescriptor(desc, computeType, (cudnnRNNDataLayout_t)layout, nMaxSeqLen, nBatchSize, nVectorSize, rgSeqLen, NULL))
		return lErr | ERROR_CUDNN_OFFSET;

	return CUDNN_STATUS_SUCCESS;
}


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
inline long Memory<T>::CreateImageOp(int nNum, T fBrightnessProb, T fBrightnessDelta, T fContrastProb, T fContrastLower, T fContrastUpper, T fSaturationProb, T fSaturationLower, T fSaturationUpper, long lRandomSeed, long* phHandle)
{
	LONG lErr;
	imgopHandle<T>* imgop = NULL;

	if (phHandle == NULL)
		return ERROR_PARAM_NULL;

	if ((imgop = new imgopHandle<T>()) == NULL)
		return ERROR_MEMORY_OUT;

	if (lErr = imgop->Initialize(this, nNum, fBrightnessProb, fBrightnessDelta, fContrastProb, fContrastLower, fContrastUpper, fSaturationProb, fSaturationLower, fSaturationUpper, lRandomSeed))
	{
		delete imgop;
		return lErr;
	}

	long hHandle = m_imgop.Allocate(imgop);
	if (hHandle < 0)
	{
		delete imgop;
		return ERROR_MEMORY_OUT;
	}

	*phHandle = hHandle;
	return 0;
}

template <class T>
inline long Memory<T>::FreeImageOp(long hHandle)
{
	imgopHandle<T>* imgop = (imgopHandle<T>*)m_imgop.Free(hHandle);

	if (imgop != NULL)
	{
		imgop->CleanUp();
		delete imgop;
	}

	return 0;
}

template <class T>
inline imgopHandle<T>* Memory<T>::GetImageOp(long hHandle)
{
	return (imgopHandle<T>*)m_imgop.GetData(hHandle);
}

template <class T>
inline long Memory<T>::DistortImage(long hHandle, int nCount, int nNum, int nDim, long hX, long hY)
{
	imgopHandle<T>* imgop = GetImageOp(hHandle);

	if (imgop == NULL)
		return ERROR_PARAM_NULL;

	return imgop->DistortImage(nCount, nNum, nDim, hX, hY);
}


template <class T>
inline long Memory<T>::CreateSSD(int nGpuID, int nNumClasses, bool bShareLocation, int nLocClasses, int nBackgroundLabelId, bool bUseDifficultGt, SsdMiningType miningType, SsdMatchingType matchingType, T fOverlapThreshold, bool bUsePriorForMatching, SsdCodeType codeType, bool bEncodeVariantInTgt, bool bBpInside, bool bIgnoreCrossBoundaryBbox, bool bUsePriorForNms, SsdConfLossType confLossType, SsdLocLossType locLossType, T fNegPosRatio, T fNegOverlap, int nSampleSize, bool bMapObjectToAgnostic, bool bNmsActive, T fNmsThreshold, int nTopK, T fEta, Math<T>* pMath, long* phHandle)
{
	LONG lErr;
	ssdHandle<T>* ssd = NULL;

	if (phHandle == NULL)
		return ERROR_PARAM_NULL;

	if ((ssd = new ssdHandle<T>()) == NULL)
		return ERROR_MEMORY_OUT;

	if (lErr = ssd->Update(this, pMath))
	{
		delete ssd;
		return lErr;
	}

	if (lErr = ssd->Initialize(nGpuID, nNumClasses, bShareLocation, nLocClasses, nBackgroundLabelId, bUseDifficultGt, miningType, matchingType, fOverlapThreshold, bUsePriorForMatching, codeType, bEncodeVariantInTgt, bBpInside, bIgnoreCrossBoundaryBbox, bUsePriorForNms, confLossType, locLossType, fNegPosRatio, fNegOverlap, nSampleSize, bMapObjectToAgnostic, bNmsActive, fNmsThreshold, nTopK, fEta))
	{
		delete ssd;
		return lErr;
	}

	long hHandle = m_ssd.Allocate(ssd);
	if (hHandle < 0)
	{
		delete ssd;
		return ERROR_MEMORY_OUT;
	}

	*phHandle = hHandle;
	return 0;
}

template <class T>
inline long Memory<T>::FreeSSD(long hHandle)
{
	ssdHandle<T>* ssd = (ssdHandle<T>*)m_ssd.Free(hHandle);

	if (ssd != NULL && ssd->IsOwner())
	{
		ssd->CleanUp();

		if (ssd->RefCount() == 0)
			delete ssd;
	}

	return 0;
}

template <class T>
inline ssdHandle<T>* Memory<T>::GetSSD(long hHandle)
{
	return (ssdHandle<T>*)m_ssd.GetData(hHandle);
}

template <class T>
inline long Memory<T>::SetupSSD(long hSsd, int nNum, int nNumPriors, int nNumGt)
{
	ssdHandle<T>* pSsd = (ssdHandle<T>*)m_ssd.GetData(hSsd);
	if (pSsd == NULL)
		return ERROR_SSD_NOT_INITIALIZED;

	return pSsd->Setup(nNum, nNumPriors, nNumGt);
}

template <class T>
inline long Memory<T>::SsdMultiboxLossForward(long hSsd, int nLocDataCount, long hLocData, int nConfDataCount, long hConfData, int nPriorDataCount, long hPriorData, int nGtDataCount, long hGtData, int* pnNumMatches, int* pnNumNegs)
{
	ssdHandle<T>* pSsd = (ssdHandle<T>*)m_ssd.GetData(hSsd);
	if (pSsd == NULL)
		return ERROR_SSD_NOT_INITIALIZED;

	return pSsd->MultiboxLossForward(nLocDataCount, hLocData, nConfDataCount, hConfData, nPriorDataCount, hPriorData, nGtDataCount, hGtData, pnNumMatches, pnNumNegs);
}

template <class T>
inline long Memory<T>::SsdEncodeLocPrediction(long hSsd, int nLocPredCount, long hLocPred, int nLocGtCount, long hLocGt)
{
	ssdHandle<T>* pSsd = (ssdHandle<T>*)m_ssd.GetData(hSsd);
	if (pSsd == NULL)
		return ERROR_SSD_NOT_INITIALIZED;

	return pSsd->EncodeLocPrediction(nLocPredCount, hLocPred, nLocGtCount, hLocGt);
}

template <class T>
inline long Memory<T>::SsdEncodeConfPrediction(long hSsd, int nConfPredCount, long hConfPred, int nConfGtCount, long hConfGt)
{
	ssdHandle<T>* pSsd = (ssdHandle<T>*)m_ssd.GetData(hSsd);
	if (pSsd == NULL)
		return ERROR_SSD_NOT_INITIALIZED;

	return pSsd->EncodeLocPrediction(nConfPredCount, hConfPred, nConfGtCount, hConfGt);
}

template <class T>
inline long Memory<T>::SsdGetAllMatchIndices(long hSsd, vector<map<int, vector<int>>>* pall_match_indices)
{
	ssdHandle<T>* pSsd = (ssdHandle<T>*)m_ssd.GetData(hSsd);
	if (pSsd == NULL)
		return ERROR_SSD_NOT_INITIALIZED;

	return pSsd->GetAllMatchIndices(pall_match_indices);
}

template <class T>
inline long Memory<T>::SsdGetAllNegIndices(LONG hSsd, vector<vector<int>>* pall_neg_indices)
{
	ssdHandle<T>* pSsd = (ssdHandle<T>*)m_ssd.GetData(hSsd);
	if (pSsd == NULL)
		return ERROR_SSD_NOT_INITIALIZED;

	return pSsd->GetAllNegIndices(pall_neg_indices);
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

template <class T>
inline long Memory<T>::CreateExtensionFloat(HMODULE hParent, LONG lKernelIdx, LPTSTR pszDllPath, long *phHandle)
{
	LONG lErr;
	extensionHandle<T>* extension = NULL;

	if (phHandle == NULL)
		return ERROR_PARAM_NULL;

	if ((extension = new extensionHandle<T>()) == NULL)
		return ERROR_MEMORY_OUT;

	if (lErr = extension->InitializeFloat(hParent, lKernelIdx, pszDllPath))
	{
		delete extension;
		return lErr;
	}

	long hHandle = m_extensions.Allocate(extension);
	if (hHandle < 0)
	{
		delete extension;
		return ERROR_MEMORY_OUT;
	}

	*phHandle = hHandle;
	return 0;
}

template <class T>
inline long Memory<T>::CreateExtensionDouble(HMODULE hParent, LONG lKernelIdx, LPTSTR pszDllPath, long *phHandle)
{
	LONG lErr;
	extensionHandle<T>* extension = NULL;

	if (phHandle == NULL)
		return ERROR_PARAM_NULL;

	if ((extension = new extensionHandle<T>()) == NULL)
		return ERROR_MEMORY_OUT;

	if (lErr = extension->InitializeDouble(hParent, lKernelIdx, pszDllPath))
	{
		delete extension;
		return lErr;
	}

	long hHandle = m_extensions.Allocate(extension);
	if (hHandle < 0)
	{
		delete extension;
		return ERROR_MEMORY_OUT;
	}

	*phHandle = hHandle;
	return 0;
}

template <class T>
inline long Memory<T>::FreeExtension(long hHandle)
{
	extensionHandle<T>* extension = (extensionHandle<T>*)m_extensions.Free(hHandle);

	if (extension != NULL)
	{
		extension->CleanUp();
		delete extension;
	}

	return 0;
}

template <class T>
inline extensionHandle<T>* Memory<T>::GetExtension(long hHandle)
{
	return (extensionHandle<T>*)m_extensions.GetData(hHandle);
}

template <class T>
inline long Memory<T>::ExtensionRun(long hHandle, long lfnIdx, T* pfInput, long lCount, T** ppfOutput, long* plCount, LPTSTR pszErr, LONG lErrMax)
{
	return GetExtension(hHandle)->Run(lfnIdx, pfInput, lCount, ppfOutput, plCount, pszErr, lErrMax);
}


#endif // __MEMORY_CU__