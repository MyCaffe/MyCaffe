//=============================================================================
//	FILE:	ssd_core.cu
//
//	DESC:	This file implements the single-shot multi-box detection 
//			(ssd_core) algorithm GPU implentations.
//=============================================================================

#include "util.h"
#include "ssd_core.h"
#include "math.h"
#include "memory.h"
#include "memorycol.h"
#include "handlecol.h"
#include <iterator>


//=============================================================================
//	Function Definitions
//=============================================================================

template <typename T>
__device__ T Max(const T x, const T y)
{
	return x < y ? x : y;
}

template <typename T>
__global__ void compute_conf_loss_kernel(const int n, const T* conf_data, const int nNumPredsPerClass, const int nNumClasses, const SsdConfLossType loss_type, const T* match_data, T* conf_loss_data)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
	{
		int nLabel = match_data[i];
		int nNum = i / nNumPredsPerClass;
		int p = i % nNumPredsPerClass;
		int nStartIdx = (nNum * nNumPredsPerClass + p) * nNumClasses;
		T fLoss = 0;

		// Compute softmax probability
		if (loss_type == SSD_CONF_LOSS_TYPE_SOFTMAX)
		{
			T fProb = conf_data[nStartIdx + nLabel];
			fLoss = -log(Max(fProb, T(FLT_MIN)));
		}
		else if (loss_type == SSD_CONF_LOSS_TYPE_LOGISTIC)
		{
			int nTarget = 0;
			for (int c = 0; c < nNumClasses; c++)
			{
				nTarget = (c == nLabel) ? 1 : 0;
				
				T fInput = conf_data[nStartIdx + c];
				fLoss -= fInput * (nTarget - (fInput >= 0)) - log(1 + exp(fInput - 2 * fInput * (fInput >= 0)));
			}
		}

		conf_loss_data[i] = fLoss;
	}
}


//=============================================================================
//	Private Classes
//=============================================================================

//=============================================================================
//	Class Methods - SsdMemory
//=============================================================================

//=============================================================================
//	Class Methods - SsdData
//=============================================================================

template <class T>
long SsdData<T>::computeConfLoss(int nNum, int nNumPriors, int nNumClasses, vector<map<int, vector<int>>>& all_match_indices, map<int, vector<BBOX>>& rgAllGt, vector<vector<T>>*pall_conf_loss)
{
	LONG lErr;

	if (m_nBackgroundLabelId >= nNumClasses)
		return ERROR_SSD_BACKGROUND_LABEL_OUT_OF_RANGE;

	for (int i = 0; i < nNum; i++)
	{
		const map<int, vector<int>>& match_indices = all_match_indices[i];

		for (int p = 0; p < nNumPriors; p++)
		{
			// Get the label index.
			int nLabel = m_nBackgroundLabelId;

			for (map<int, vector<int>>::const_iterator it = match_indices.begin(); it != match_indices.end(); it++)
			{
				const vector<int>& match_index = it->second;

				if (match_index.size() != nNumPriors)
					return ERROR_SSD_COMPUTE_CONF_LOSS_MATCH_INDEX_INCORRECT;

				if (match_index[p] > -1)
				{
					if (rgAllGt.find(i) == rgAllGt.end())
						return ERROR_SSD_COMPUTE_CONF_LOSS_GT_MISSING_ITEM;

					vector<BBOX> rgGt = rgAllGt.find(i)->second;
					if (match_index[p] >= (int)rgGt.size())
						return ERROR_SSD_COMPUTE_CONF_LOSS_MATCH_INDEX_OUT_OF_RANGE;

					int nIdx = match_index[p];
					nLabel = getLabel(rgGt[nIdx]);
					if (nLabel < 0 || nLabel > nNumClasses || nLabel == m_nBackgroundLabelId)
						return ERROR_SSD_COMPUTE_CONF_LOSS_INVALID_LABEL;

					// A prior can only be matched ot one gt bbox.
					break;
				}
			}

			m_pMatch->m_host[i * nNumPriors + p] = T(nLabel);
		}
	}

	// Get the probability data.
	SsdMemory<T>* pConf = m_rgBbox[MEM_CONF];
	if (m_confLossType == SSD_CONF_LOSS_TYPE_SOFTMAX)
	{
		if (lErr = softmax(m_rgBbox[MEM_CONF], nNum * nNumPriors, m_nNumClasses, 1, m_pProb))
			return lErr;

		pConf = m_pProb;
	}

	// Compute the loss.
	const int n = nNum * nNumPriors;
	compute_conf_loss_kernel<T> << <CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS >> > (n, pConf->gpu_data(), nNumPriors, nNumClasses, m_confLossType, m_pMatch->gpu_data(), m_pConfLoss->gpu_data());
	if (lErr = cudaStreamSynchronize(0))
		return lErr;

	// Save the loss.
	pall_conf_loss->clear();
	m_pConfLoss->CopyGpuToCpu();
	load_conf_loss(nNum, nNumPriors, m_pConfLoss, pall_conf_loss);

	return 0;
}

template long SsdData<double>::computeConfLoss(int nNum, int nNumPriors, int nNumClasses, vector<map<int, vector<int>>>& all_match_indices, map<int, vector<BBOX>>& rgAllGt, vector<vector<double>>*pall_conf_loss);
template long SsdData<float>::computeConfLoss(int nNum, int nNumPriors, int nNumClasses, vector<map<int, vector<int>>>& all_match_indices, map<int, vector<BBOX>>& rgAllGt, vector<vector<float>>*pall_conf_loss);

// end