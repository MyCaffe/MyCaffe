//=============================================================================
//	FILE:	rnn8.h
//
//	DESC:	This file manages the RNN8 functinality (requires cuDnn 8.0+)
//=============================================================================
#ifndef __RNN8_CU__
#define __RNN8_CU__

#include "util.h"
#include "math.h"

//=============================================================================
//	Flags
//=============================================================================

//=============================================================================
//	Classes
//=============================================================================

template <class T>
class Memory;


//-----------------------------------------------------------------------------
//	RNN8 Handle Class
//
//	This class stores the RNN8 description information.
//-----------------------------------------------------------------------------
template <class T>
class rnn8Handle
{
	Memory<T>* m_pMem;
	Math<T>* m_pMath;
	cudnnDataType_t m_dataType;
	cudnnDataType_t m_mathPrecision;
	cudnnMathType_t m_mathType;
	cudnnRNNDataLayout_t m_layout;
	cudnnRNNAlgo_t m_algo;
	cudnnRNNMode_t m_cellMode;
	cudnnRNNBiasMode_t m_biasMode;
	cudnnDirectionMode_t m_directionMode;
	cudnnRNNInputMode_t m_inputMode;
	int m_nGpuID;
	int m_nSequenceLength;
	int m_nMaxSeqLen;
	int m_nBatchSize;
	int m_nInputSize;
	int m_nHiddenSize;
	int m_nOutputSize;
	int m_nProjectionSize;
	int m_nNumLayers;
	int m_nNumLinLayers;
	int m_nBidirectionalScale;

	size_t m_szWeightSizeInBytes;
	size_t m_szWorkSizeInBytes;
	size_t m_szReserveSizeInBytes;

	double m_dfDropout;
	unsigned long long m_lSeed;
	size_t m_stateSize;
	void* m_states;

	cudnnRNNDescriptor_t m_rnnDesc;
	cudnnRNNDataDescriptor_t m_xDesc;
	cudnnRNNDataDescriptor_t m_yDesc;
	cudnnTensorDescriptor_t m_hDesc;
	cudnnTensorDescriptor_t m_cDesc;
	cudnnDropoutDescriptor_t m_dropoutDesc;
	cudnnForwardMode_t m_fwdmode;
	int* m_pnHostSeqArray;
	int* m_pnDevSeqArray;

public:
	
	rnn8Handle()
	{
		m_pMem = NULL;
		m_pMath = NULL;
		m_szWeightSizeInBytes = 0;
		m_szWorkSizeInBytes = 0;
		m_szReserveSizeInBytes = 0;
		m_dataType = CUDNN_DATA_FLOAT;
		m_mathPrecision = CUDNN_DATA_FLOAT;
		m_mathType = CUDNN_DEFAULT_MATH;
		m_layout = CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED;
		m_algo = CUDNN_RNN_ALGO_STANDARD;
		m_cellMode = CUDNN_LSTM;
		m_biasMode = CUDNN_RNN_NO_BIAS;
		m_directionMode = CUDNN_UNIDIRECTIONAL;
		m_inputMode = CUDNN_LINEAR_INPUT;
		m_fwdmode = CUDNN_FWD_MODE_TRAINING;
		m_nSequenceLength = 1;
		m_nMaxSeqLen = 1;
		m_nBatchSize = 1;
		m_nInputSize = 1;
		m_nHiddenSize = 1;
		m_nOutputSize = 1;
		m_nProjectionSize = 1;
		m_dfDropout = 0.0;
		m_lSeed = 0;
		m_stateSize = 0;
		m_states = NULL;
		m_pnHostSeqArray = NULL;
		m_pnDevSeqArray = NULL;
		m_nNumLayers = 1;
		m_nNumLinLayers = 1;
		m_nBidirectionalScale = 1;
		m_rnnDesc = NULL;
		m_xDesc = NULL;
		m_yDesc = NULL;
		m_hDesc = NULL;
		m_cDesc = NULL;
		m_dropoutDesc = NULL;
	}

	long Initialize(Memory<T>* pMem, Math<T>* pMath)
	{
#ifndef CUDNN_8
		return ERROR_RNN8_INCOMPATIBLE_CUDNN_VER;
#endif
		m_pMem = pMem;
		m_pMath = pMath;
		return 0;
	}

	long CleanUp();

	long setupSeqArray(int nBatchSize, int nSequenceLength);

	long Set(long hCuda, cudnnForwardMode_t fwdmode, cudnnDataType_t type, cudnnRNNDataLayout_t layout, cudnnRNNMode_t cellmode, cudnnRNNBiasMode_t biasMode, int nSequenceLength, int nBatchSize, int nInputSize, int nHiddenSize, int nOutputSize, int nProjectionSize, int nNumLayers, float fDropout, unsigned long long lSeed, bool bBidirectional = false);

	long GetMemorySizes(long hCuda, size_t* pWeightCount, size_t* pWorkSize, size_t* pReserveSize)
	{
		if (m_pMem == NULL)
			return ERROR_RNN8_NOT_INITIALIZED;

		*pWeightCount = m_szWeightSizeInBytes / sizeof(T);
		*pWorkSize = m_szWorkSizeInBytes;
		*pReserveSize = m_szReserveSizeInBytes;

		return 0;
	}

	long fill(FILLER_TYPE ft, T fVal, T fVal2, T* pMem, int nLen);

	long InitializeWeights(long hCuda, long hWts, FILLER_TYPE ftWt, T fWtVal, T fWtVal2, FILLER_TYPE ftBias, T fBiasVal, T fBiasVal2);

	long Forward(long hCuda, long hX, long hY, long hhX, long hhY, long hcX, long hcY, long hWt, long hWork, long hReserved);

	long Backward(long hCuda, long hY, long hdY, long hX, long hdX, long hhX, long hdhY, long hdhX, long hcX, long hdcY, long hdcX, long hWt, long hdWt, long hWork, long hReserved);
};


//=============================================================================
//	Inline Methods
//=============================================================================


#endif