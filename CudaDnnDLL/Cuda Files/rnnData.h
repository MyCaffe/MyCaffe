//=============================================================================
//	FILE:	rnnData.h
//
//	DESC:	This file manages the RNNDATA algorithm
//=============================================================================
#ifndef __RNNDATA_CU__
#define __RNNDATA_CU__

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
//	RNNDATA Handle Class
//
//	This class stores the RNNDATA description information.
//-----------------------------------------------------------------------------
template <class T>
class rnnDataHandle
{
	Memory<T>* m_pMem;
	cudnnDataType_t m_type;
	cudnnRNNDataLayout_t m_layout;
	cudnnTensorDescriptor_t* m_rgSeqTensors;
	int m_nMaxSeqLen;
	int m_nBatchSize;
	int m_nVectorSize;

public:
	
	rnnDataHandle()
	{
		m_pMem = NULL;
		m_type = CUDNN_DATA_FLOAT;
		m_layout = CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED;
		m_rgSeqTensors = NULL;
		m_nBatchSize = 0;
		m_nMaxSeqLen = 0;
		m_nVectorSize = 0;
	}

	long Initialize(Memory<T>* pMem)
	{
		m_pMem = pMem;
		return 0;
	}

	long rnnDataHandle<T>::CleanUp()
	{
		if (m_rgSeqTensors != NULL)
		{
			for (int i = 0; i < m_nMaxSeqLen; i++)
			{
				if (m_rgSeqTensors[i] != NULL)
					cudnnDestroyTensorDescriptor(m_rgSeqTensors[i]);
			}

			free(m_rgSeqTensors);
			m_rgSeqTensors = NULL;
		}

		return 0;
	}

	long rnnDataHandle<T>::Set(cudnnDataType_t type, cudnnRNNDataLayout_t layout, int nMaxSeqLen, int nBatchSize, int nInputSize, int* rgSeqLen /*not used*/, bool bBidirectional = false)
	{
		LONG lErr;

		CleanUp();

		m_type = type;
		m_layout = layout;
		m_nMaxSeqLen = nMaxSeqLen;
		m_nBatchSize = nBatchSize;
		m_nVectorSize = nInputSize;

		m_rgSeqTensors = (cudnnTensorDescriptor_t*)malloc(sizeof(cudnnTensorDescriptor_t) * nMaxSeqLen);
		if (m_rgSeqTensors == NULL)
		{
			CleanUp();
			return ERROR_MEMORY_OUT;
		}

		memset(m_rgSeqTensors, NULL, sizeof(cudnnTensorDescriptor_t) * nMaxSeqLen);

		int rgDimA[3];
		int rgStrideA[3];

		for (int i = 0; i < nMaxSeqLen; i++)
		{
			if (lErr = cudnnCreateTensorDescriptor(&m_rgSeqTensors[i]))
			{
				CleanUp();
				return lErr;
			}

			rgDimA[0] = nBatchSize;
			rgDimA[1] = (bBidirectional) ? nInputSize * 2 : nInputSize;
			rgDimA[2] = 1;

			rgStrideA[0] = rgDimA[2] * rgDimA[1];
			rgStrideA[1] = rgDimA[2];
			rgStrideA[2] = 1;

			if (lErr = cudnnSetTensorNdDescriptor(m_rgSeqTensors[i], type, 3, rgDimA, rgStrideA))
			{
				CleanUp();
				return lErr;
			}
		}

		return 0;
	}

	cudnnTensorDescriptor_t GetFirstTensor()
	{
		if (m_rgSeqTensors == NULL || m_nMaxSeqLen == 0)
			return NULL;

		return m_rgSeqTensors[0];
	}

	cudnnDataType_t DataType()
	{
		return m_type;
	}

	cudnnRNNDataLayout_t Layout()
	{
		return m_nlayout;
	}

	int MaxSeqLen()
	{
		return m_nMaxSeqLen;
	}

	int BatchSize()
	{
		return m_nBatchSize;
	}

	int VectorSize()
	{
		return m_nVectorSize;
	}

	cudnnTensorDescriptor_t* SeqTensors()
	{
		return m_rgSeqTensors;
	}
};


//=============================================================================
//	Inline Methods
//=============================================================================


#endif