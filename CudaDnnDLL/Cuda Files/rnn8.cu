//=============================================================================
//	FILE:	rnn8.cu
//
//	DESC:	This file implements the base class used to manage the the rnn8
//			functionality.
//=============================================================================

#include "util.h"
#include "memory.h"
#include "rnn8.h"

//=============================================================================
//	Class Methods
//=============================================================================

template <class T>
long rnn8Handle<T>::setupSeqArray(int nBatchSize, int nSequenceLength)
{
	LONG lErr;
	size_t lSize = sizeof(int) * nBatchSize;

	m_pnHostSeqArray = (int*)malloc(lSize);
	if (m_pnHostSeqArray == NULL)
		return ERROR_OUTOFMEMORY;

	for (int i = 0; i < nBatchSize; i++)
	{
		m_pnHostSeqArray[i] = nSequenceLength;
	}

	if (lErr = cudaMalloc(&m_pnDevSeqArray, lSize))
		return lErr;

	if (lErr = cudaMemcpy(m_pnDevSeqArray, m_pnHostSeqArray, lSize, cudaMemcpyHostToDevice))
		return lErr;

	return ERROR_SUCCESS;
}

template long rnn8Handle<double>::setupSeqArray(int nBatchSize, int nSequenceLength);
template long rnn8Handle<float>::setupSeqArray(int nBatchSize, int nSequenceLength);

template <class T>
long rnn8Handle<T>::Set(long hCuda, cudnnForwardMode_t fwdmode, cudnnDataType_t type, cudnnRNNDataLayout_t layout, cudnnRNNMode_t cellmode, cudnnRNNBiasMode_t biasMode, int nSequenceLength, int nBatchSize, int nInputSize, int nHiddenSize, int nOutputSize, int nProjectionSize, int nNumLayers, float fDropout, unsigned long long lSeed, bool bBidirectional)
{
	LONG lErr;

	CleanUp();

	if (m_pMem == NULL)
		return ERROR_RNN8_NOT_INITIALIZED;

	cudnnHandle_t cuda = m_pMem->GetCuDNN(hCuda);

	m_fwdmode = fwdmode;
	m_dataType = type;
	m_mathPrecision = type;
	m_mathType = CUDNN_DEFAULT_MATH;
	m_layout = layout;
	m_cellMode = cellmode;
	m_biasMode = biasMode;
	m_directionMode = (bBidirectional) ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL;
	m_nSequenceLength = nSequenceLength;
	m_nMaxSeqLen = nSequenceLength;
	m_nBatchSize = nBatchSize;
	m_nInputSize = nInputSize;
	m_nHiddenSize = nHiddenSize;
	m_nOutputSize = nOutputSize;
	m_nProjectionSize = nProjectionSize;
	m_nNumLayers = nNumLayers;

	m_nNumLinLayers = 0;
	switch (cellmode)
	{
		case CUDNN_RNN_RELU:
		case CUDNN_RNN_TANH:
			m_nNumLinLayers = 2;
			break;

		case CUDNN_LSTM:
			m_nNumLinLayers = 8;
			break;

		case CUDNN_GRU:
			m_nNumLinLayers = 6;
			break;
	}

	m_dfDropout = fDropout;
	m_lSeed = lSeed;
	m_nBidirectionalScale = (bBidirectional) ? 2 : 1;

	if (lErr = setupSeqArray(nBatchSize, nSequenceLength))
	{
		CleanUp();
		return lErr;
	}

	//-----------------------------------------------------
	//	Setup x and y descriptors
	//-----------------------------------------------------

	if (lErr = cudnnCreateRNNDataDescriptor(&m_xDesc))
	{
		CleanUp();
		return lErr;
	}

	if (lErr = cudnnCreateRNNDataDescriptor(&m_yDesc))
	{
		CleanUp();
		return lErr;
	}

	double paddingFill = 0;
	if (lErr = cudnnSetRNNDataDescriptor(m_xDesc, type, layout, nSequenceLength, nBatchSize, nInputSize, m_pnHostSeqArray, &paddingFill))
	{
		CleanUp();
		return lErr;
	}

	if (lErr = cudnnSetRNNDataDescriptor(m_yDesc, type, layout, nSequenceLength, nBatchSize, nHiddenSize * m_nBidirectionalScale, m_pnHostSeqArray, &paddingFill))
	{
		CleanUp();
		return lErr;
	}

	//-----------------------------------------------------
	//	Setup h and c descriptors
	//-----------------------------------------------------

	if (lErr = cudnnCreateTensorDescriptor(&m_hDesc))
	{
		CleanUp();
		return lErr;
	}

	if (lErr = cudnnCreateTensorDescriptor(&m_cDesc))
	{
		CleanUp();
		return lErr;
	}

	int dimHidden[3];
	dimHidden[0] = nNumLayers * m_nBidirectionalScale;
	dimHidden[1] = nBatchSize;
	dimHidden[2] = nHiddenSize;

	int strideHidden[3];
	strideHidden[0] = dimHidden[2] * dimHidden[1];
	strideHidden[1] = dimHidden[2];
	strideHidden[2] = 1;

	if (lErr = cudnnSetTensorNdDescriptor(m_hDesc, type, 3, dimHidden, strideHidden))
	{
		CleanUp();
		return lErr;
	}

	if (lErr = cudnnSetTensorNdDescriptor(m_cDesc, type, 3, dimHidden, strideHidden))
	{
		CleanUp();
		return lErr;
	}

	//-----------------------------------------------------
	//	Setup Dropout
	//-----------------------------------------------------

	if (lErr = cudnnDropoutGetStatesSize(cuda, &m_stateSize))
	{
		CleanUp();
		return lErr;
	}

	if (lErr = cudaMalloc(&m_states, m_stateSize))
	{
		CleanUp();
		return lErr;
	}

	if (lErr = cudnnCreateDropoutDescriptor(&m_dropoutDesc))
	{
		CleanUp();
		return lErr;
	}

	if (lErr = cudnnSetDropoutDescriptor(m_dropoutDesc, cuda, (float)m_dfDropout, m_states, m_stateSize, m_lSeed))
	{
		CleanUp();
		return lErr;
	}

	//-----------------------------------------------------
	//	Setup RNN
	//-----------------------------------------------------

	if (lErr = cudnnCreateRNNDescriptor(&m_rnnDesc))
	{
		CleanUp();
		return lErr;
	}

	if (lErr = cudnnSetRNNDescriptor_v8(m_rnnDesc, m_algo, m_cellMode, m_biasMode, m_directionMode, m_inputMode, m_dataType, m_mathPrecision, m_mathType, m_nInputSize, m_nHiddenSize, m_nProjectionSize, m_nNumLayers, m_dropoutDesc, 0))
	{
		CleanUp();
		return lErr;
	}

	if (lErr = cudnnGetRNNWeightSpaceSize(cuda, m_rnnDesc, &m_szWeightSizeInBytes))
		return lErr;

	if (lErr = cudnnGetRNNTempSpaceSizes(cuda, m_rnnDesc, m_fwdmode, m_xDesc, &m_szWorkSizeInBytes, &m_szReserveSizeInBytes))
		return lErr;



	return cudaStreamSynchronize(0);
}

template long rnn8Handle<double>::Set(long hCuda, cudnnForwardMode_t fwdmode, cudnnDataType_t type, cudnnRNNDataLayout_t layout, cudnnRNNMode_t cellmode, cudnnRNNBiasMode_t biasMode, int nSequenceLength, int nBatchSize, int nInputSize, int nHiddenSize, int nOutputSize, int nProjectionSize, int nNumLayers, float fDropout, unsigned long long lSeed, bool bBidirectional);
template long rnn8Handle<float>::Set(long hCuda, cudnnForwardMode_t fwdmode, cudnnDataType_t type, cudnnRNNDataLayout_t layout, cudnnRNNMode_t cellmode, cudnnRNNBiasMode_t biasMode, int nSequenceLength, int nBatchSize, int nInputSize, int nHiddenSize, int nOutputSize, int nProjectionSize, int nNumLayers, float fDropout, unsigned long long lSeed, bool bBidirectional);

template <class T>
long rnn8Handle<T>::CleanUp()
{
	if (m_pnHostSeqArray != NULL)
	{
		free(m_pnHostSeqArray);
		m_pnHostSeqArray = NULL;
	}

	if (m_pnDevSeqArray != NULL)
	{
		cudaFree(m_pnDevSeqArray);
		m_pnDevSeqArray = NULL;
	}

	if (m_states != NULL)
	{
		cudaFree(m_states);
		m_states = NULL;
	}

	if (m_dropoutDesc != NULL)
	{
		cudnnDestroyDropoutDescriptor(m_dropoutDesc);
		m_dropoutDesc = NULL;
	}

	if (m_cDesc != NULL)
	{
		cudnnDestroyTensorDescriptor(m_cDesc);
		m_cDesc = NULL;
	}

	if (m_hDesc != NULL)
	{
		cudnnDestroyTensorDescriptor(m_hDesc);
		m_hDesc = NULL;
	}

	if (m_yDesc != NULL)
	{
		cudnnDestroyRNNDataDescriptor(m_yDesc);
		m_yDesc = NULL;
	}

	if (m_xDesc != NULL)
	{
		cudnnDestroyRNNDataDescriptor(m_xDesc);
		m_xDesc = NULL;
	}

	if (m_rnnDesc != NULL)
	{
		cudnnDestroyRNNDescriptor(m_rnnDesc);
		m_rnnDesc = NULL;
	}

	return 0;
}

template long rnn8Handle<double>::CleanUp();
template long rnn8Handle<float>::CleanUp();

template <class T>
long rnn8Handle<T>::fill(FILLER_TYPE ft, T fVal, T fVal2, T* pMem, int nLen)
{
	switch (ft)
	{
		case FILLER_TYPE_CONSTANT:
			return m_pMath->set(nLen, pMem, fVal);

		case FILLER_TYPE_XAVIER:
		{
			double dfScale = sqrt(3.0 / nLen);
			double fPosScale = dfScale;
			double fNegScale = -dfScale;
			return m_pMath->rng_uniform(nLen, (T)fNegScale, (T)fPosScale, pMem);
		}

		case FILLER_TYPE_GAUSSIAN:
			return m_pMath->rng_gaussian(nLen, fVal, fVal2, pMem, nLen * sizeof(T));

		default:
			return ERROR_RNN8_INVALID_FILLER;
	}

	return 0;
}

template long rnn8Handle<double>::fill(FILLER_TYPE ft, double fVal, double fVal2, double* pMem, int nLen);
template long rnn8Handle<float>::fill(FILLER_TYPE ft, float fVal, float fVal2, float* pMem, int nLen);

template <class T>
long rnn8Handle<T>::InitializeWeights(long hCuda, long hWts, FILLER_TYPE ftWt, T fWtVal, T fWtVal2, FILLER_TYPE ftBias, T fBiasVal, T fBiasVal2)
{
	LONG lErr;
	MemoryItem* pWt;

	if (m_pMem == NULL)
		return ERROR_RNN8_NOT_INITIALIZED;

	cudnnHandle_t cuda = m_pMem->GetCuDNN(hCuda);

	if (lErr = m_pMem->GetMemory(hWts, &pWt))
		return lErr;

	T* wt = (T*)pWt->Data();
	size_t szWt = pWt->Size();

	cudnnTensorDescriptor_t wDesc = NULL;
	cudnnTensorDescriptor_t bDesc = NULL;

	if (lErr = cudnnCreateTensorDescriptor(&wDesc))
		return lErr;

	if (lErr = cudnnCreateTensorDescriptor(&bDesc))
	{
		cudnnDestroyTensorDescriptor(wDesc);
		return lErr;
	}

	for (int nLayer = 0; nLayer < m_nNumLayers * m_nBidirectionalScale; nLayer++)
	{
		for (int nLinLayer = 0; nLinLayer < m_nNumLinLayers; nLinLayer++)
		{
			cudnnDataType_t dataTypeTemp;
			int nbDims = 0;
			int dim[3] = { 0, 0, 0 };
			int stride[3] = { 0, 0, 0 };
			T* pLinLayerWt = NULL;
			T* pLinLayerBias = NULL;

			if (lErr = cudnnGetRNNWeightParams(cuda, m_rnnDesc, nLayer, m_szWeightSizeInBytes, wt, nLinLayer, wDesc, (void**)&pLinLayerWt, bDesc, (void**)&pLinLayerBias))
			{
				cudnnDestroyTensorDescriptor(wDesc);
				cudnnDestroyTensorDescriptor(bDesc);
				return lErr;
			}

			if (pLinLayerWt != NULL)
			{
				if (lErr = cudnnGetTensorNdDescriptor(wDesc, 3, &dataTypeTemp, &nbDims, dim, stride))
				{
					cudnnDestroyTensorDescriptor(wDesc);
					cudnnDestroyTensorDescriptor(bDesc);
					return lErr;
				}

				if (lErr = fill(ftWt, fWtVal, fWtVal2, pLinLayerWt, dim[0] * dim[1] * dim[2]))
				{
					cudnnDestroyTensorDescriptor(wDesc);
					cudnnDestroyTensorDescriptor(bDesc);
					return lErr;
				}
			}

			if (pLinLayerBias != NULL)
			{
				if (lErr = cudnnGetTensorNdDescriptor(bDesc, 3, &dataTypeTemp, &nbDims, dim, stride))
				{
					cudnnDestroyTensorDescriptor(wDesc);
					cudnnDestroyTensorDescriptor(bDesc);
					return lErr;
				}

				if (lErr = fill(ftBias, fBiasVal, fBiasVal2, pLinLayerBias, dim[0] * dim[1] * dim[2]))
				{
					cudnnDestroyTensorDescriptor(wDesc);
					cudnnDestroyTensorDescriptor(bDesc);
					return lErr;
				}
			}
		}
	}

	cudnnDestroyTensorDescriptor(wDesc);
	cudnnDestroyTensorDescriptor(bDesc);

	return cudaStreamSynchronize(0);
}

template long rnn8Handle<double>::InitializeWeights(long hCuda, long hWts, FILLER_TYPE ftWt, double fWtVal, double fWtVal2, FILLER_TYPE ftBias, double fBiasVal, double fBiasVal2);
template long rnn8Handle<float>::InitializeWeights(long hCuda, long hWts, FILLER_TYPE ftWt, float fWtVal, float fWtVal2, FILLER_TYPE ftBias, float fBiasVal, float fBiasVal2);

template <class T>
long rnn8Handle<T>::Forward(long hCuda, long hX, long hY, long hhX, long hhY, long hcX, long hcY, long hWt, long hWork, long hReserved)
{
	LONG lErr;
	MemoryItem* pX;
	MemoryItem* pY;
	MemoryItem* phX;
	MemoryItem* phY;
	MemoryItem* pcX;
	MemoryItem* pcY;
	MemoryItem* pWt;
	MemoryItem* pWork;
	MemoryItem* pReserved = NULL;

	if (m_pMem == NULL)
		return ERROR_RNN8_NOT_INITIALIZED;

	cudnnHandle_t cuda = m_pMem->GetCuDNN(hCuda);

	if (lErr = m_pMem->GetMemory(hX, &pX))
		return lErr;

	if (lErr = m_pMem->GetMemory(hY, &pY))
		return lErr;

	if (lErr = m_pMem->GetMemory(hhX, &phX))
		return lErr;

	if (lErr = m_pMem->GetMemory(hhY, &phY))
		return lErr;

	if (lErr = m_pMem->GetMemory(hcX, &pcX))
		return lErr;

	if (lErr = m_pMem->GetMemory(hcY, &pcY))
		return lErr;

	if (lErr = m_pMem->GetMemory(hWt, &pWt))
		return lErr;

	if (lErr = m_pMem->GetMemory(hWork, &pWork))
		return lErr;

	if (hReserved != 0)
	{
		if (lErr = m_pMem->GetMemory(hReserved, &pReserved))
			return lErr;
	}

	T* x = (T*)pX->Data();
	T* y = (T*)pY->Data();
	T* hx = (T*)phX->Data();
	T* hy = (T*)phY->Data();
	T* cx = (T*)pcX->Data();
	T* cy = (T*)pcY->Data();
	T* wt = (T*)pWt->Data();
	size_t szWt = pWt->Size();
	T* work = (T*)pWork->Data();
	size_t szWork = pWork->Size();
	T* reserved = (pReserved != NULL) ? (T*)pReserved->Data() : NULL;
	size_t szReserved = (pReserved != NULL) ? pReserved->Size() : 0;

	if (lErr = cudnnRNNForward(cuda, m_rnnDesc, m_fwdmode, m_pnDevSeqArray, m_xDesc, x, m_yDesc, y, m_hDesc, hx, hy, m_cDesc, cx, cy, szWt, wt, szWork, work, szReserved, reserved))
		return lErr;

	return cudaStreamSynchronize(0);
}

template long rnn8Handle<double>::Forward(long hCuda, long hX, long hY, long hhX, long hhY, long hcX, long hcY, long hWt, long hWork, long hReserved);
template long rnn8Handle<float>::Forward(long hCuda, long hX, long hY, long hhX, long hhY, long hcX, long hcY, long hWt, long hWork, long hReserved);

template <class T>
long rnn8Handle<T>::Backward(long hCuda, long hY, long hdY, long hX, long hdX, long hhX, long hdhY, long hdhX, long hcX, long hdcY, long hdcX, long hWt, long hdWt, long hWork, long hReserved)
{
	LONG lErr;
	MemoryItem* pY;
	MemoryItem* pdY;
	MemoryItem* pX;
	MemoryItem* pdX;
	MemoryItem* phX;
	MemoryItem* pdhY = NULL;
	MemoryItem* pdhX;
	MemoryItem* pcX;
	MemoryItem* pdcY = NULL;
	MemoryItem* pdcX;
	MemoryItem* pWt;
	MemoryItem* pdWt;
	MemoryItem* pWork;
	MemoryItem* pReserved = NULL;

	if (m_pMem == NULL)
		return ERROR_RNN8_NOT_INITIALIZED;

	cudnnHandle_t cuda = m_pMem->GetCuDNN(hCuda);

	if (lErr = m_pMem->GetMemory(hY, &pY))
		return lErr;

	if (lErr = m_pMem->GetMemory(hdY, &pdY))
		return lErr;

	if (lErr = m_pMem->GetMemory(hX, &pX))
		return lErr;

	if (lErr = m_pMem->GetMemory(hdX, &pdX))
		return lErr;

	if (lErr = m_pMem->GetMemory(hhX, &phX))
		return lErr;

	if (hdhY != 0)
	{
		if (lErr = m_pMem->GetMemory(hdhY, &pdhY))
			return lErr;
	}

	if (lErr = m_pMem->GetMemory(hdhX, &pdhX))
		return lErr;

	if (lErr = m_pMem->GetMemory(hcX, &pcX))
		return lErr;

	if (hdcY != 0)
	{
		if (lErr = m_pMem->GetMemory(hdcY, &pdcY))
			return lErr;
	}

	if (lErr = m_pMem->GetMemory(hdcX, &pdcX))
		return lErr;

	if (lErr = m_pMem->GetMemory(hWt, &pWt))
		return lErr;

	if (lErr = m_pMem->GetMemory(hdWt, &pdWt))
		return lErr;

	if (lErr = m_pMem->GetMemory(hWork, &pWork))
		return lErr;

	if (hReserved != 0)
	{
		if (lErr = m_pMem->GetMemory(hReserved, &pReserved))
			return lErr;
	}

	T* y = (T*)pY->Data();
	T* dy = (T*)pdY->Data();
	T* x = (T*)pX->Data();
	T* dx = (T*)pdX->Data();
	T* hx = (T*)phX->Data();
	T* dhy = (pdhY != NULL) ? (T*)pdhY->Data() : NULL;
	T* dhx = (T*)pdhX->Data();
	T* cx = (T*)pcX->Data();
	T* dcx = (T*)pdcX->Data();
	T* dcy = (pdcY != NULL) ? (T*)pdcY->Data() : NULL;
	T* wt = (T*)pWt->Data();
	T* dwt = (T*)pdWt->Data();
	size_t szWt = pWt->Size();
	T* work = (T*)pWork->Data();
	size_t szWork = pWork->Size();
	T* reserved = (pReserved != NULL) ? (T*)pReserved->Data() : NULL;
	size_t szReserved = (pReserved != NULL) ? pReserved->Size() : 0;

	if (lErr = cudnnRNNBackwardData_v8(cuda, m_rnnDesc, m_pnDevSeqArray, m_yDesc, y, dy, m_xDesc, dx, m_hDesc, hx, dhy, dhx, m_cDesc, cx, dcy, dcx, szWt, wt, szWork, work, szReserved, reserved))
		return lErr;

	cudaMemset(dwt, 0, szWt);
	if (lErr = cudnnRNNBackwardWeights_v8(cuda, m_rnnDesc, CUDNN_WGRAD_MODE_ADD, m_pnDevSeqArray, m_xDesc, x, m_hDesc, hx, m_yDesc, y, szWt, dwt, szWork, work, szReserved, reserved))
		return lErr;

	return cudaStreamSynchronize(0);
}

template long rnn8Handle<double>::Backward(long hCuda, long hY, long hdY, long hX, long hdX, long hhX, long hdhY, long hdhX, long hcX, long hdcY, long hdcX, long hWt, long hdWt, long hWork, long hReserved);
template long rnn8Handle<float>::Backward(long hCuda, long hY, long hdY, long hX, long hdX, long hhX, long hdhY, long hdhX, long hcX, long hdcY, long hdcX, long hWt, long hdWt, long hWork, long hReserved);

// end