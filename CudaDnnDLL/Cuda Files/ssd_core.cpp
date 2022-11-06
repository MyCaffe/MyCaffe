//=============================================================================
//	FILE:	ssd_core.cu
//
//	DESC:	This file implements the single-shot multi-box detection (ssd_core) algorithm
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
bool sortScorePairDescend(const tuple<float, T>& t1, const tuple<float, T>& t2)
{
	if (std::get<0>(t1) > std::get<0>(t2))
		return true;
	else
		return false;
}


//=============================================================================
//	Private Classes
//=============================================================================

//=============================================================================
//	Class Methods - SsdMemory
//=============================================================================

template <class T>
SsdMemory<T>::SsdMemory(Memory<T>* pMem, int nGpuID)
{
	m_pMem = pMem;
	m_pMemCol = pMem->GetMemoryCollection();
	m_nGpuID = nGpuID;
	m_nMax = 0;
	m_host = NULL;
	m_device = NULL;
	m_handle = 0;
}

template SsdMemory<double>::SsdMemory(Memory<double>* pMem, int nGpuID);
template SsdMemory<float>::SsdMemory(Memory<float>* pMem, int nGpuID);


template <class T>
LONG SsdMemory<T>::Initialize(int nCount, long hHandle)
{
	LONG lErr;
	T* pSrc = NULL;

	m_nCount = nCount;

	if (m_nMax < nCount)
	{
		if (m_host != NULL)
		{
			m_pMem->FreeHost(m_host);
			m_host = NULL;
		}

		if (m_handle != 0)
		{
			if (m_bOwnHandle)
				m_pMem->FreeMemory(m_handle);

			m_handle = 0;
			m_device = NULL;
		}

		m_nMax = 0;
	}

	if (hHandle != 0)
	{
		MemoryItem* pItem;

		if (lErr = m_pMemCol->GetData(hHandle, &pItem))
			return lErr;

		m_device = (T*)pItem->Data();
		pSrc = m_device;

		if (m_handle != 0)
		{
			if (m_bOwnHandle)
				m_pMem->FreeMemory(m_handle);

			m_handle = 0;
		}

		m_handle = hHandle;
		m_bOwnHandle = false;
	}
	else
	{
		if (m_handle == 0)
		{
			if (lErr = m_pMem->AllocMemory(m_nGpuID, false, nCount, NULL, 0, &m_handle))
				return lErr;

			m_bOwnHandle = true;
		}

		MemoryItem* pItem;
		if (lErr = m_pMemCol->GetData(m_handle, &pItem))
			return lErr;

		m_device = (T*)pItem->Data();
	}

	if (m_host == NULL)
	{
		if (lErr = m_pMem->AllocHost(nCount, &m_host, pSrc, true, false))
			return lErr;

		if (!m_bOwnHandle)
		{
			if (lErr = CopyGpuToCpu())
				return lErr;
		}
	}
	else if (pSrc != NULL)
	{
		if (lErr = m_pMem->CopyToHost(nCount, m_host, pSrc, true, false))
			return lErr;
	}

	if (m_nMax == 0)
		m_nMax = nCount;

	return 0;
}

template LONG SsdMemory<double>::Initialize(int nCount, long hHandle);
template LONG SsdMemory<float>::Initialize(int nCount, long hHandle);


template <class T>
void SsdMemory<T>::CleanUp()
{
	if (m_host != NULL)
	{
		m_pMem->FreeHost(m_host);
		m_host = NULL;
	}

	if (m_handle != NULL)
	{
		if (m_bOwnHandle)
			m_pMem->FreeMemory(m_handle);
		m_handle = 0;
	}

	m_device = NULL;
	m_nMax = 0;
}

template void SsdMemory<double>::CleanUp();
template void SsdMemory<float>::CleanUp();


//=============================================================================
//	Class Methods - SsdData
//=============================================================================

template <class T>
SsdData<T>::SsdData(Memory<T>* pMem, Math<T>* pMath) : m_rgBbox(MEM_COUNT, NULL)
{
	m_pMem = pMem;
	m_pMath = pMath;
	m_nNum = 0;
	m_nNumPriors = 0;
	m_nNumGt = 0;
	m_nNumClasses = 0;
	m_bShareLocation = true;
	m_nLocClasses = 0;
	m_nBackgroundLabelId = -1;
	m_bUseDifficultGt = false;
	m_miningType = SSD_MINING_TYPE_NONE;
	m_matchingType = SSD_MATCHING_TYPE_BIPARTITE;
	m_fOverlapThreshold = 0;
	m_bUsePriorForMatching = true;
	m_codeType = SSD_CODE_TYPE_CORNER_SIZE;
	m_bEncodeVariantInTgt = false;
	m_bBpInside = false;
	m_bIgnoreCrossBoundaryBbox = true;
	m_bUsePriorForNms = true;
	m_confLossType = SSD_CONF_LOSS_TYPE_SOFTMAX;
	m_locLossType = SSD_LOC_LOSS_TYPE_L2;
	m_fNegOverlap = 0;
	m_fNegPosRatio = 0;
	m_nSampleSize = 0;
	m_bMapObjectToAgnostic = false;
	m_fNmsThreshold = 0;
	m_nTopK = -1;
	m_fEta = 1;

	m_pMatch = NULL;
	m_pProb = NULL;
	m_pConfLoss = NULL;
	m_pScale = NULL;
}

template SsdData<double>::SsdData(Memory<double>* pMem, Math<double>* pMath);
template SsdData<float>::SsdData(Memory<float>* pMem, Math<float>* pMath);


template <class T>
SsdData<T>::~SsdData()
{
	for (int i = 0; i < m_rgBbox.size(); i++)
	{
		if (m_rgBbox[i] != NULL)
			delete m_rgBbox[i];
	}

	if (m_pConf != NULL)
		delete m_pConf;

	if (m_pMatch != NULL)
		delete m_pMatch;

	if (m_pProb != NULL)
		delete m_pProb;

	if (m_pConfLoss != NULL)
		delete m_pConfLoss;

	if (m_pScale != NULL)
		delete m_pScale;
}

template SsdData<double>::~SsdData();
template SsdData<float>::~SsdData();


template <class T>
long SsdData<T>::match(vector<BBOX>& rgGt, vector<BBOX>& rgPredBBox, int nLabel, vector<int>* match_indices, vector<float>* match_overlaps)
{
	int nNumPred = (int)rgPredBBox.size();

	match_indices->clear();
	match_indices->resize(nNumPred, -1);
	match_overlaps->clear();
	match_overlaps->resize(nNumPred, 0.0);

	int nNumGt = 0;
	vector<int> rgGtIndices;

	// Label -1 means that we are comparing against all ground truths.
	if (nLabel == -1)
	{
		nNumGt = (int)rgGt.size();
		for (int i = 0; i < nNumGt; i++)
		{
			rgGtIndices.push_back(i);
		}
	}
	// Count number of ground truth boxes which have the desired label.
	else
	{
		for (int i = 0; i < (int)rgGt.size(); i++)
		{
			int nGtLabel = getLabel(rgGt[i]);
			if (nGtLabel == nLabel)
			{
				nNumGt++;
				rgGtIndices.push_back(i);
			}
		}
	}

	if (nNumGt == 0)
		return 0;

	// Store the positive overlap between predictions and ground truth.
	map<int, map<int, float>> overlaps;
	for (int j = 0; j < nNumGt; j++)
	{
		T fxminGt;
		T fyminGt;
		T fxmaxGt;
		T fymaxGt;
		getBounds(rgGt[rgGtIndices[j]], &fxminGt, &fyminGt, &fxmaxGt, &fymaxGt);

		for (int i = 0; i < nNumPred; i++)
		{
			if (m_bIgnoreCrossBoundaryBbox && isCrossBoundaryBbox(rgPredBBox[i]))
			{
				(*match_indices)[i] = -2;
				continue;
			}

			float fOverlap = jaccardOverlap(rgPredBBox[i], fxminGt, fyminGt, fxmaxGt, fymaxGt);
			if (fOverlap > T(1e-6))
			{
				(*match_overlaps)[i] = (std::max)((*match_overlaps)[i], fOverlap);
				overlaps[i][j] = fOverlap;
			}
		}
	}

	// Bipartite matching
	vector<int> rgGtPool;
	for (int i = 0; i < nNumGt; i++)
	{
		rgGtPool.push_back(i);
	}

	// Find the most overlapped gt and corresponding predictions.
	while (rgGtPool.size() > 0)
	{
		int nMaxIdx = -1;
		int nMaxGtIdx = -1;
		float fMaxOverlap = -1;

		for (map<int, map<int, float>>::iterator it = overlaps.begin(); it != overlaps.end(); it++)
		{
			int i = it->first;

			// The prediction already has matched ground truth or is ignored.
			if ((*match_indices)[i] != -1)
				continue;

			for (int p = 0; p < (int)rgGtPool.size(); p++)
			{
				int j = rgGtPool[p];

				// No overlap between i-th prediction and j-th ground truth.
				if (it->second.find(j) == it->second.end())
					continue;

				// Find the maximum overlapped pair.
				if (it->second[j] > fMaxOverlap)
				{
					// If the prediction has not been matched to any ground truth,
					// and the overlap is larger than the maximum overlap, update.
					nMaxIdx = i;
					nMaxGtIdx = j;
					fMaxOverlap = it->second[j];
				}
			}
		}

		// Cannot find a good match.
		if (nMaxIdx == -1)
		{
			break;
		}
		else
		{
			if ((*match_indices)[nMaxIdx] != -1)
				return ERROR_SSD_BAD_MATCH;

			(*match_indices)[nMaxIdx] = rgGtIndices[nMaxGtIdx];
			(*match_overlaps)[nMaxIdx] = fMaxOverlap;

			// Erase the ground truth.
			rgGtPool.erase(std::find(rgGtPool.begin(), rgGtPool.end(), nMaxGtIdx));
		}
	}

	// Do the matching
	switch (m_matchingType)
	{
		case SSD_MATCHING_TYPE_BIPARTITE:
			// Already done.
			break;

		case SSD_MATCHING_TYPE_PER_PREDICTION:
			// Get most overlapped for the rest of the prediction bboxes.
			for (map<int, map<int, float>>::iterator it = overlaps.begin(); it != overlaps.end(); it++)
			{
				int i = it->first;

				// The prediction already has matched ground truth or is ignored.
				if ((*match_indices)[i] != -1)
					continue;

				int nMaxGtIdx = -1;
				float fMaxOverlap = -1;

				for (int j = 0; j < nNumGt; j++)
				{
					// No overlap between the i-th prediction and the j-th ground truth.
					if (it->second.find(j) == it->second.end())
						continue;

					// Find the maximum overlapped pair.
					float fOverlap = it->second[j];
					if (fOverlap >= m_fOverlapThreshold && fOverlap > fMaxOverlap)
					{
						// If the prediction has not been matched to any ground truth,
						// and the overlap is larger than the maximum overlap, update.
						nMaxGtIdx = j;
						fMaxOverlap = fOverlap;
					}
				}

				if (nMaxGtIdx != -1)
				{
					// Found a matched ground truth.
					if ((*match_indices)[i] != -1)
						return ERROR_SSD_BAD_MATCH;

					(*match_indices)[i] = rgGtIndices[nMaxGtIdx];
					(*match_overlaps)[i] = fMaxOverlap;
				}
			}
			break;
	}

	return 0;
}

template long SsdData<double>::match(vector<BBOX>& rgGt, vector<BBOX>& rgPredBBox, int nLabel, vector<int>* match_indices, vector<float>* match_overlaps);
template long SsdData<float>::match(vector<BBOX>& rgGt, vector<BBOX>& rgPredBBox, int nLabel, vector<int>* match_indices, vector<float>* match_overlaps);


template <class T>
long SsdData<T>::findMatches(vector<map<int, vector<BBOX>>>& rgAllLocPreds, map<int, vector<BBOX>>& rgAllGt, vector<BBOX>& rgPriorBbox, vector<BBOX>& rgPriorVariances, vector<map<int, vector<float>>>& all_match_overlaps, vector<map<int, vector<int>>>& all_match_indices)
{
	LONG lErr;

	all_match_indices.clear();
	all_match_overlaps.clear();

	// Find the matches.
	for (int i = 0; i < m_nNum; i++)
	{
		map<int, vector<int>> match_indices;
		map<int, vector<float>> match_overlaps;

		// Check if there is a ground truth for the current image.
		map<int, vector<BBOX>>::iterator pGt = rgAllGt.find(i);
		if (pGt == rgAllGt.end())
		{
			// Three is no gt for the current image, so all predictiosn are negative.
			all_match_indices.push_back(match_indices);
			all_match_overlaps.push_back(match_overlaps);
			continue;
		}

		// Find match between predictions and ground truth.
		vector<BBOX> rgGtBbox = pGt->second;
		if (!m_bUsePriorForMatching)
		{
			for (int c = 0; c < m_nLocClasses; c++)
			{
				int nLabel = (m_bShareLocation) ? -1 : c;

				// Ignore background loc predictions.
				if (!m_bShareLocation && nLabel == m_nBackgroundLabelId)
					continue;

				// Decode the predictions into bbox first.
				vector<BBOX> rgLocBox;
				bool clip_bbox = false;
				if (lErr = decode(rgPriorBbox, rgPriorVariances, clip_bbox, rgAllLocPreds[i].find(nLabel)->second, rgLocBox))
					return lErr;

				if (lErr = match(rgGtBbox, rgLocBox, nLabel, &match_indices[nLabel], &match_overlaps[nLabel]))
					return lErr;
			}
		}
		// Use prior bboxes to match against all ground truth.
		else
		{
			vector<int> temp_match_indices;
			vector<float> temp_match_overlaps;
			const int nLabel = -1;

			if (lErr = match(rgGtBbox, rgPriorBbox, nLabel, &temp_match_indices, &temp_match_overlaps))
				return lErr;

			if (m_bShareLocation)
			{
				match_indices[nLabel] = temp_match_indices;
				match_overlaps[nLabel] = temp_match_overlaps;
			}
			else
			{
				// Get ground truth label for each ground truth bbox.
				vector<int> rgGtLabels;
				for (int g = 0; g < (int)rgGtBbox.size(); g++)
				{
					int nLabel1 = m_rgBbox[MEM_GT]->label(rgGtBbox[g]);
					rgGtLabels.push_back(nLabel1);
				}

				// Distribute the matching results to different loc_class.
				for (int c = 0; c < m_nLocClasses; c++)
				{
					// Ignore background loc predictions.
					if (c == m_nBackgroundLabelId)
						continue;

					match_indices[c].resize(temp_match_indices.size(), -1);
					match_overlaps[c] = temp_match_overlaps;

					for (int m = 0; m < (int)temp_match_indices.size(); m++)
					{
						if (temp_match_indices[m] > -1)
						{
							const int nGtIdx = temp_match_indices[m];
							if (nGtIdx >= (int)rgGtLabels.size())
								return ERROR_SSD_GT_LABEL_OUT_OF_RANGE;

							if (c == rgGtLabels[nGtIdx])
								match_indices[c][m] = nGtIdx;
						}
					}
				}
			}

			all_match_indices.push_back(match_indices);
			all_match_overlaps.push_back(match_overlaps);
		}
	}

	return 0;
}

template long SsdData<double>::findMatches(vector<map<int, vector<BBOX>>>& rgAllLocPreds, map<int, vector<BBOX>>& rgAllGt, vector<BBOX>& rgPriorBbox, vector<BBOX>& rgPriorVariances, vector<map<int, vector<float>>>& all_match_overlaps, vector<map<int, vector<int>>>& all_match_indices);
template long SsdData<float>::findMatches(vector<map<int, vector<BBOX>>>& rgAllLocPreds, map<int, vector<BBOX>>& rgAllGt, vector<BBOX>& rgPriorBbox, vector<BBOX>& rgPriorVariances, vector<map<int, vector<float>>>& all_match_overlaps, vector<map<int, vector<int>>>& all_match_indices);


template <class T>
long SsdData<T>::softmax(SsdMemory<T>* pData, const int nOuterNum, const int nChannels, const int nInnerNum, SsdMemory<T>* pProb)
{
	LONG lErr;
	int nCount = nOuterNum * nChannels * nInnerNum;

	if (pData->m_nMax < nCount)
		return ERROR_MEMORY_RANGE_EXCEEDED;

	// We need to subtract the max to avoid nuerical issues, compute the exp, 
	// and then normalize.

	// Compute Max
	if (lErr = m_pMath->channel_max(nOuterNum * nInnerNum, nOuterNum, nChannels, nInnerNum, pData->gpu_handle(), m_pScale->gpu_handle()))
		return lErr;

	// Subtract
	if (lErr = m_pMath->channel_sub(nCount, nOuterNum, nChannels, nInnerNum, pData->gpu_handle(), m_pScale->gpu_handle(), pProb->gpu_handle()))
		return lErr;

	// Exponetiate
	if (lErr = m_pMath->exp(nCount, pProb->gpu_handle(), pProb->gpu_handle()))
		return lErr;

	// Sum after exponentiate
	if (lErr = m_pMath->channel_sum(nOuterNum * nInnerNum, nOuterNum, nChannels, nInnerNum, pProb->gpu_handle(), m_pScale->gpu_handle(), true))
		return lErr;

	// Divide
	if (lErr = m_pMath->channel_div(nCount, nOuterNum, nChannels, nInnerNum, m_pScale->gpu_handle(), pProb->gpu_handle()))
		return lErr;

	return 0;
}

template long SsdData<double>::softmax(SsdMemory<double>* pData, const int nOuterNum, const int nChannels, const int nInnerNum, SsdMemory<double>* pProb);
template long SsdData<float>::softmax(SsdMemory<float>* pData, const int nOuterNum, const int nChannels, const int nInnerNum, SsdMemory<float>* pProb);


template <class T>
long SsdData<T>::encodeLocPrediction(vector<map<int, vector<BBOX>>>& rgAllLocPreds, map<int, vector<BBOX>>& rgAllGt, vector<map<int, vector<int>>>& all_match_indices, vector<BBOX>& rgPriorBboxes, vector<BBOX>& rgPriorVariances, SsdMemory<T>* pLocPred, SsdMemory<T>* pLocGt)
{
	LONG lErr;

	int nNum = (int)rgAllLocPreds.size();
	int nCount = 0;

	for (int i = 0; i < nNum; i++)
	{
		for (map<int, vector<int>>::const_iterator it = all_match_indices[i].begin(); it != all_match_indices[i].end(); it++)
		{
			const int nLabel = it->first;
			const vector<int>& match_index = it->second;

			map<int, vector<BBOX>>::iterator pLabelLocPred = rgAllLocPreds[i].find(nLabel);
			if (pLabelLocPred == rgAllLocPreds[i].end())
				return ERROR_SSD_LOC_PRED_LABEL_NOT_FOUND;

			const vector<BBOX>& loc_pred = pLabelLocPred->second;

			for (int j = 0; j < match_index.size(); j++)
			{
				if (match_index[j] <= -1)
					continue;

				// Store encoded ground truth.
				const int nGtIdx = match_index[j];
				map<int, vector<BBOX>>::iterator pGt = rgAllGt.find(i);
				if (pGt == rgAllGt.end() || nGtIdx >= pGt->second.size())
					return ERROR_SSD_GT_LABEL_OUT_OF_RANGE;

				BBOX gtbbox = pGt->second[nGtIdx];

				if (j >= rgPriorBboxes.size())
					return ERROR_PARAM_OUT_OF_RANGE;

				T xmin_gtencode;
				T ymin_gtencode;
				T xmax_gtencode;
				T ymax_gtencode;
				if (lErr = encode(rgPriorBboxes[j], rgPriorVariances[j], gtbbox, &xmin_gtencode, &ymin_gtencode, &xmax_gtencode, &ymax_gtencode))
					return lErr;

				m_rgBbox[MEM_LOCGT]->setBounds(nCount, xmin_gtencode, ymin_gtencode, xmax_gtencode, ymax_gtencode);

				// Store the location prediction.
				if (j >= loc_pred.size())
					return ERROR_PARAM_OUT_OF_RANGE;

				MEM locPredType = std::get<1>(loc_pred[j]);
				int nOffset = std::get<0>(loc_pred[j]);

				if (m_bBpInside)
				{
					T xmin_match = m_rgBbox[MEM_PRIOR]->xmin(j);
					T ymin_match = m_rgBbox[MEM_PRIOR]->ymin(j);
					T xmax_match = m_rgBbox[MEM_PRIOR]->xmax(j);
					T ymax_match = m_rgBbox[MEM_PRIOR]->ymax(j);
					T size_match;

					if (m_bUsePriorForMatching)
					{
						bool clip_box = false;
						if (lErr = decode(rgPriorBboxes[j], rgPriorVariances[j], clip_box, loc_pred[j], &xmin_match, &ymin_match, &xmax_match, &ymax_match, &size_match))
							return lErr;
					}

					// When a dimension of match_bbox is outside of an image region, use
					// gt_encode to simulate zero gradient.
					T xmin = (xmin_match < 0 || xmin_match > 1) ? xmin_gtencode : m_rgBbox[locPredType]->xmin(nOffset);
					T ymin = (ymin_match < 0 || ymin_match > 1) ? ymin_gtencode : m_rgBbox[locPredType]->ymin(nOffset);
					T xmax = (xmax_match < 0 || xmax_match > 1) ? xmax_gtencode : m_rgBbox[locPredType]->xmax(nOffset);
					T ymax = (ymax_match < 0 || ymax_match > 1) ? ymax_gtencode : m_rgBbox[locPredType]->ymax(nOffset);
					m_rgBbox[MEM_LOCPRED]->setBounds(nCount, xmin, ymin, xmax, ymax);
				}
				else
				{				
					T xmin = m_rgBbox[locPredType]->xmin(nOffset);
					T ymin = m_rgBbox[locPredType]->ymin(nOffset);
					T xmax = m_rgBbox[locPredType]->xmax(nOffset);
					T ymax = m_rgBbox[locPredType]->ymax(nOffset);
					m_rgBbox[MEM_LOCPRED]->setBounds(nCount, xmin, ymin, xmax, ymax);
				}

				if (m_bEncodeVariantInTgt)
				{
					T xmin_var;
					T ymin_var;
					T xmax_var;
					T ymax_var;
					m_rgBbox[MEM_PRIOR]->getBounds(rgPriorVariances[j], &xmin_var, &ymin_var, &xmax_var, &ymax_var);
					m_rgBbox[MEM_LOCPRED]->divBounds(nCount, xmin_var, ymin_var, xmax_var, ymax_var);
					m_rgBbox[MEM_LOCGT]->divBounds(nCount, xmin_var, ymin_var, xmax_var, ymax_var);
				}

				nCount++;
			}
		}
	}

	return 0;
}

template long SsdData<double>::encodeLocPrediction(vector<map<int, vector<BBOX>>>& rgAllLocPreds, map<int, vector<BBOX>>& rgAllGt, vector<map<int, vector<int>>>& all_match_indices, vector<BBOX>& rgPriorBboxes, vector<BBOX>& rgPriorVariances, SsdMemory<double>* pLocPred, SsdMemory<double>* pLocGt);
template long SsdData<float>::encodeLocPrediction(vector<map<int, vector<BBOX>>>& rgAllLocPreds, map<int, vector<BBOX>>& rgAllGt, vector<map<int, vector<int>>>& all_match_indices, vector<BBOX>& rgPriorBboxes, vector<BBOX>& rgPriorVariances, SsdMemory<float>* pLocPred, SsdMemory<float>* pLocGt);



template <class T>
long SsdData<T>::encodeConfPrediction(SsdMemory<T>* pConf, vector<map<int, vector<int>>>& all_match_indices, vector<vector<int>>& all_neg_indices, map<int, vector<BBOX>>& rgAllGt, SsdMemory<T>* pConfPred, SsdMemory<T>* pConfGt)
{
	LONG lErr;

	if (m_bMapObjectToAgnostic)
	{
		if (m_nBackgroundLabelId >= 0)
		{
			if (m_nNumClasses != 2)
				return ERROR_SSD_INVALID_NUM_CLASSES;
		}
		else
		{
			if (m_nNumClasses != 1)
				return ERROR_SSD_INVALID_NUM_CLASSES;
		}
	}

	T* conf_data = pConf->cpu_data();
	T* conf_gt_data = pConfGt->cpu_data();
	T* conf_pred_data = pConfPred->cpu_data();
	bool bDoNegMining = (m_miningType != SSD_MINING_TYPE_NONE);
	int nCount = 0;

	for (int i = 0; i < m_nNum; i++)
	{
		map<int, vector<BBOX>>::iterator pGt = rgAllGt.find(i);
		if (pGt != rgAllGt.end())
		{
			// Save matched (positive) bboxes scores and labels
			const map<int, vector<int>>& match_indices = all_match_indices[i];
			for (map<int, vector<int>>::const_iterator it = match_indices.begin(); it != match_indices.end(); it++)
			{
				const vector<int>& match_index = it->second;
				if (match_index.size() != m_nNumPriors)
					return ERROR_SSD_COMPUTE_CONF_LOSS_MATCH_INDEX_INCORRECT;

				for (int j = 0; j < m_nNumPriors; j++)
				{
					if (match_index[j] <= -1)
						continue;

					const int gt_label = (m_bMapObjectToAgnostic) ? m_nBackgroundLabelId + 1 : getLabel(pGt->second[match_index[j]]);
					int nIdx = (bDoNegMining) ? nCount : j;

					switch (m_confLossType)
					{
					case SSD_CONF_LOSS_TYPE_SOFTMAX:
						conf_gt_data[nIdx] = T(gt_label);
						break;

					case SSD_CONF_LOSS_TYPE_LOGISTIC:
						conf_gt_data[nIdx * m_nNumClasses + gt_label] = T(1);
						break;

					default:
						return ERROR_SSD_INVALID_CONF_LOSS_TYPE;
					}

					if (bDoNegMining)
					{
						// Copy scores for matched bboxes.
						if (lErr = m_pMem->CopyToHost(m_nNumClasses, conf_pred_data + nCount, conf_data + j * m_nNumClasses, false, false))
							return lErr;
						nCount++;
					}
				}
			}

			// Go to next image.
			if (bDoNegMining)
			{
				// Save negative bboxes scores and labels
				for (int n = 0; n < all_neg_indices[i].size(); n++)
				{
					int j = all_neg_indices[i][n];
					if (j >= m_nNumPriors)
						return ERROR_PARAM_OUT_OF_RANGE;

					if (lErr = m_pMem->CopyToHost(m_nNumClasses, conf_pred_data + nCount, conf_data + j * m_nNumClasses, false, false))
						return lErr;

					switch (m_confLossType)
					{
					case SSD_CONF_LOSS_TYPE_SOFTMAX:
						conf_gt_data[nCount] = T(m_nBackgroundLabelId);
						break;

					case SSD_CONF_LOSS_TYPE_LOGISTIC:
						if (m_nBackgroundLabelId >= 0 && m_nBackgroundLabelId < m_nNumClasses)
							conf_gt_data[nCount * m_nNumClasses + m_nBackgroundLabelId] = T(1);
						break;

					default:
						return ERROR_SSD_INVALID_CONF_LOSS_TYPE;
					}

					nCount++;
				}
			}
		}

		if (bDoNegMining)
			conf_data += m_nNumPriors * m_nNumClasses;
		else
			conf_gt_data += m_nNumPriors;
	}

	return 0;
}

template long SsdData<double>::encodeConfPrediction(SsdMemory<double>* pConf, vector<map<int, vector<int>>>& all_match_indices, vector<vector<int>>& all_neg_indices, map<int, vector<BBOX>>& rgAllGt, SsdMemory<double>* pConfPred, SsdMemory<double>* pConfGt);
template long SsdData<float>::encodeConfPrediction(SsdMemory<float>* pConf, vector<map<int, vector<int>>>& all_match_indices, vector<vector<int>>& all_neg_indices, map<int, vector<BBOX>>& rgAllGt, SsdMemory<float>* pConfPred, SsdMemory<float>* pConfGt);


template <class T>
long SsdData<T>::computeLocLoss(SsdMemory<T>* pLocPred, SsdMemory<T>* pLocPredDiff, SsdMemory<T>* pLocGt, vector<map<int, vector<int>>>& all_match_indices, int nNum, int nNumPriors, SsdLocLossType loss_type, vector<vector<T>>* pall_loc_loss)
{
	LONG lErr;
	int nLocCount = pLocPred->count();
	int nGtCount = pLocGt->count();

	if (nLocCount != nGtCount || nLocCount == 0)
		return ERROR_SSD_INVALID_LOCCOUNT_GTCOUNT;

	if (lErr = m_pMath->sub(nLocCount, pLocPred->gpu_handle(), pLocGt->gpu_handle(), pLocPredDiff->gpu_handle()))
		return lErr;

	if (lErr = pLocPredDiff->CopyGpuToCpu())
		return lErr;

	T* diff_data = pLocPredDiff->cpu_data();

	int nCount = 0;
	for (int i = 0; i < nNum; i++)
	{
		vector<T> loc_loss(nNumPriors, T(0.0));
		for (map<int, vector<int>>::const_iterator it = all_match_indices[i].begin(); it != all_match_indices[i].end(); it++)
		{
			const vector<int>& match_index = it->second;

			if (match_index.size() != nNumPriors)
				return ERROR_SSD_INVALID_LOC_LOSS_MATCH_COUNT;

			for (int j = 0; j < match_index.size(); j++)
			{
				if (match_index[j] <= -1)
					continue;

				T fLoss = 0;
				for (int k = 0; k < 4; k++)
				{
					T fVal = diff_data[nCount * 4 + k];

					if (m_locLossType == SSD_LOC_LOSS_TYPE_SMOOTH_L1)
					{
						T fAbsVal = T(fabs(fVal));
						if (fAbsVal < T(1.0))
							fLoss += T(0.5) * fVal * fVal;
						else
							fLoss += fAbsVal - T(0.5);
					}
					else if (m_locLossType == SSD_LOC_LOSS_TYPE_L2)
					{
						fLoss += T(0.5) * fVal * fVal;
					}
					else
					{
						return ERROR_SSD_INVALID_LOC_LOSS_TYPE;
					}
				}

				loc_loss[j] = fLoss;
				nCount++;
			}
		}

		pall_loc_loss->push_back(loc_loss);
	}

	return 0;
}

template long SsdData<double>::computeLocLoss(SsdMemory<double>* pLocPred, SsdMemory<double>* pLocPredDiff, SsdMemory<double>* pLogGt, vector<map<int, vector<int>>>& all_match_indices, int nNum, int nNumPriors, SsdLocLossType loss_type, vector<vector<double>>* pall_loc_loss);
template long SsdData<float>::computeLocLoss(SsdMemory<float>* pLocPred, SsdMemory<float>* pLocPredDiff, SsdMemory<float>* pLogGt, vector<map<int, vector<int>>>& all_match_indices, int nNum, int nNumPriors, SsdLocLossType loss_type, vector<vector<float>>* pall_loc_loss);


template <class T>
long SsdData<T>::getTopKScoreIndex(vector<T>& scores, vector<int>& indices, int nTopK, vector<tuple<float, int>>* pscore_index)
{
	if (scores.size() != indices.size())
		return ERROR_PARAM_OUT_OF_RANGE;

	// Generate index score pairs.
	for (int i = 0; i < scores.size(); i++)
	{
		pscore_index->push_back(std::make_tuple((float)scores[i], indices[i]));
	}

	// Sort the score pair according to the scores in descending order
	std::stable_sort(pscore_index->begin(), pscore_index->end(), sortScorePairDescend<int>);

	// Keep nTopK scores if needed
	if (nTopK > -1 && nTopK < pscore_index->size())
		pscore_index->resize(nTopK);

	return 0;
}

template long SsdData<double>::getTopKScoreIndex(vector<double>& scores, vector<int>& indices, int nTopK, vector<tuple<float, int>>* pscore_index);
template long SsdData<float>::getTopKScoreIndex(vector<float>& scores, vector<int>& indices, int nTopK, vector<tuple<float, int>>* pscore_index);


template <class T>
long SsdData<T>::applyNMS(vector<BBOX>& bboxes, vector<T>& scores, T fThreshold, int nTopK, bool bReuseOverlaps, map<int, map<int, T>>* poverlaps, vector<int>* pindices)
{
	if (bboxes.size() != scores.size())
		return ERROR_PARAM_OUT_OF_RANGE;

	// Get top_k scores with (corresponding indices).
	int nScoreCount = (int)scores.size();
//	vector<int> idx(boost::counting_iterator<int>(0), boost::counting_iterator<int>(nScoreCount));
	vector<int> idx(nScoreCount, 0);
	for (int i = 0; i < nScoreCount; i++)
	{
		idx[i] = i;
	}

	vector<tuple<float, int>> score_index_vec;
	getTopKScoreIndex(scores, idx, nTopK, &score_index_vec);

	// Do nms.
	pindices->clear();
	while (score_index_vec.size() != 0)
	{
		// Get the current highest score box.
		int nBestIdx = std::get<1>(score_index_vec.front());
		BBOX best_bbox = bboxes[nBestIdx];

		// Erase small bbox.
		if (getSize(best_bbox) < T(1e-5))
		{
			score_index_vec.erase(score_index_vec.begin());
			continue;
		}

		T fxminBestBbox;
		T fyminBestBbox;
		T fxmaxBestBbox;
		T fymaxBestBbox;
		getBounds(best_bbox, &fxminBestBbox, &fyminBestBbox, &fxmaxBestBbox, &fymaxBestBbox);

		pindices->push_back(nBestIdx);
		// Erase the best box.
		score_index_vec.erase(score_index_vec.begin());

		// Stop if finding enough bboxes for nms.
		if (nTopK > -1 && pindices->size() >= nTopK)
			break;

		// Compute overlap between best bbox and other remaining bboxes.
		// Remove a bbox if the overlap with the best bbox is larger than the nms_threshold.
		for (vector<tuple<float, int>>::iterator it = score_index_vec.begin(); it != score_index_vec.end();)
		{
			int nCurIdx = std::get<1>(*it);
			const BBOX cur_bbox = bboxes[nCurIdx];

			// Erase small bbox
			if (getSize(cur_bbox) < T(1e-5))
			{
				it = score_index_vec.erase(it);
				continue;
			}

			float fCurOverlap = T(0.0);
			if (bReuseOverlaps)
			{
				if (poverlaps->find(nBestIdx) != poverlaps->end() &&
					poverlaps->find(nBestIdx)->second.find(nCurIdx) != (*poverlaps)[nBestIdx].end())
				{
					// Use the computed overlap.
					fCurOverlap = (float)(*poverlaps)[nBestIdx][nCurIdx];
				}
				else if (poverlaps->find(nCurIdx) != poverlaps->end() &&
					poverlaps->find(nCurIdx)->second.find(nBestIdx) != (*poverlaps)[nCurIdx].end())
				{
					// Use the computed overlap.
					fCurOverlap = (float)(*poverlaps)[nCurIdx][nBestIdx];
				}
				else
				{
					fCurOverlap = jaccardOverlap(fxminBestBbox, fyminBestBbox, fxmaxBestBbox, fymaxBestBbox, cur_bbox);
					// Store the overlap for future use.
					(*poverlaps)[nBestIdx][nCurIdx] = fCurOverlap;
				}
			}
			else
			{
				fCurOverlap = jaccardOverlap(fxminBestBbox, fyminBestBbox, fxmaxBestBbox, fymaxBestBbox, cur_bbox);
			}

			// Remove it if necessary.
			if (fCurOverlap > (float)fThreshold)
				it = score_index_vec.erase(it);
			else
				it++;
		}
	}

	return 0;
}

template long SsdData<double>::applyNMS(vector<BBOX>& bboxes, vector<double>& scores, double fThreshold, int nTopK, bool bReuseOverlaps, map<int, map<int, double>>* poverlaps, vector<int>* pindices);
template long SsdData<float>::applyNMS(vector<BBOX>& bboxes, vector<float>& scores, float fThreshold, int nTopK, bool bReuseOverlaps, map<int, map<int, float>>* poverlaps, vector<int>* pindices);


template <class T>
long SsdData<T>::mineHardExamples(vector<map<int, vector<BBOX>>>& rgAllLocPreds, map<int, vector<BBOX>>& rgAllGt, vector<BBOX>& rgPriorBboxes, vector<BBOX>& rgPriorVariances, vector<map<int, vector<float>>>& all_match_overlaps, vector<map<int, vector<int>>>& all_match_indices, vector<vector<int>>& all_neg_indices, int* pnNumMatches, int* pnNumNegs)
{
	LONG lErr;
	int nNum = (int)rgAllLocPreds.size();
	*pnNumMatches = countNumMatches(all_match_indices, nNum);
	*pnNumNegs = 0;

	if (*pnNumMatches == 0)
		return 0;

	int nNumPriors = (int)rgPriorBboxes.size();

	if (nNumPriors != (int)rgPriorVariances.size())
		return ERROR_SSD_INVALID_PRIOR_VARIANCE_COUNT;

	if (m_miningType == SSD_MINING_TYPE_NONE)
		return 0;

	// Compute confidence losses based on matching results.
	vector<vector<T>> all_conf_loss;
	if (lErr = computeConfLoss(nNum, nNumPriors, m_nNumClasses, all_match_indices, rgAllGt, &all_conf_loss))
		return lErr;

	// Compute localization losses based on matching results
	vector<vector<T>> all_loc_loss;
	if (m_miningType == SSD_MINING_TYPE_HARD_EXAMPLE)
	{
		if (lErr = m_rgBbox[MEM_LOCGT]->Initialize(*pnNumMatches * 4))
			return lErr;

		if (lErr = m_rgBbox[MEM_LOCPRED]->Initialize(*pnNumMatches * 4))
			return lErr;

		if (lErr = m_rgBbox[MEM_LOCPRED_DIFF]->Initialize(*pnNumMatches * 4))
			return lErr;

		if (lErr = encodeLocPrediction(rgAllLocPreds, rgAllGt, all_match_indices, rgPriorBboxes, rgPriorVariances, m_rgBbox[MEM_LOCPRED], m_rgBbox[MEM_LOCGT]))
			return lErr;

		if (lErr = computeLocLoss(m_rgBbox[MEM_LOCPRED], m_rgBbox[MEM_LOCPRED_DIFF], m_rgBbox[MEM_LOCGT], all_match_indices, nNum, nNumPriors, m_locLossType, &all_loc_loss))
			return lErr;
	}

	// No localization loss.
	else
	{
		for (int i = 0; i < nNum; i++)
		{
			vector<T> loc_loss(nNumPriors, T(0.0));
			all_loc_loss.push_back(loc_loss);
		}
	}

	for (int i = 0; i < nNum; i++)
	{
		map<int, vector<int>>& match_indices = all_match_indices[i];
		const map<int, vector<float>>& match_overlaps = all_match_overlaps[i];

		// loc + conv loss.
		const vector<T>& conf_loss = all_conf_loss[i];
		const vector<T>& loc_loss = all_loc_loss[i];
		vector<T> rgLoss;

		std::transform(conf_loss.begin(), conf_loss.end(), loc_loss.begin(), std::back_inserter(rgLoss), std::plus<T>());

		// Pick negatives or hard examples based on loss.
		set<int> sel_indices;
		vector<int> neg_indices;
		for (map<int, vector<int>>::iterator it = match_indices.begin(); it != match_indices.end(); it++)
		{
			const int nLabel = it->first;
			int nNumSel = 0;

			// Get potential indices and loss pairs.
			vector<tuple<float, int>> loss_indices;
			for (int m = 0; m < match_indices[nLabel].size(); m++)
			{
				if (isEligibleMining(m_miningType, match_indices[nLabel][m], match_overlaps.find(nLabel)->second[m], (float)m_fNegOverlap))
				{
					loss_indices.push_back(std::make_tuple((float)rgLoss[m], m));
					nNumSel++;
				}
			}

			if (m_miningType == SSD_MINING_TYPE_MAX_NEGATIVE)
			{
				int nNumPos = 0;

				for (int m = 0; m < match_indices[nLabel].size(); m++)
				{
					if (match_indices[nLabel][m] > -1)
						nNumPos++;
				}

				nNumSel = (std::min)(static_cast<int>(nNumPos * m_fNegPosRatio), nNumSel);
			}
			else if (m_miningType == SSD_MINING_TYPE_HARD_EXAMPLE)
			{
				if (m_nSampleSize <= 0)
					return ERROR_SSD_SAMPLE_SIZE_TOO_SMALL;

				nNumSel = (std::min)(m_nSampleSize, nNumSel);
			}

			// Select samples.
			if (m_bNmsActive)
			{
				// Do nms before selecting samples.
				vector<T> sel_loss;
				vector<BBOX> sel_bboxes;

				if (m_bUsePriorForNms)
				{
					for (int m = 0; m < match_indices[nLabel].size(); m++)
					{
						if (isEligibleMining(m_miningType, match_indices[nLabel][m], match_overlaps.find(nLabel)->second[m], (float)m_fNegOverlap))
						{
							sel_loss.push_back(rgLoss[m]);
							sel_bboxes.push_back(rgPriorBboxes[m]);
						}
					}
				}
				else
				{
					// Decode the prediction into bbox first.
					vector<BBOX> rgLocBboxes;
					bool clip_bbox = false;
					if (lErr = decode(rgPriorBboxes, rgPriorVariances, clip_bbox, rgAllLocPreds[i].find(nLabel)->second, rgLocBboxes))
						return lErr;

					for (int m = 0; m < match_indices[nLabel].size(); m++)
					{
						if (isEligibleMining(m_miningType, match_indices[nLabel][m], match_overlaps.find(nLabel)->second[m], (float)m_fNegOverlap))
						{
							sel_loss.push_back(rgLoss[m]);
							sel_bboxes.push_back(rgLocBboxes[m]);
						}
					}
				}

				// Do non-maximum suppresion based on the loss.
				vector<int> nms_indices;
				if (lErr = applyNMS(sel_bboxes, sel_loss, m_fNmsThreshold, m_nTopK, &nms_indices))
					return lErr;

				if (nms_indices.size() < nNumSel)
					OutputDebugString(L"Not enough sample after nms!\r\n");

				// Pick top example indices after nms.
				nNumSel = (std::min)(static_cast<int>(nms_indices.size()), nNumSel);
				for (int n = 0; n < nNumSel; n++)
				{
					sel_indices.insert(std::get<1>(loss_indices[nms_indices[n]]));
				}
			}
			else
			{
				// Pick top example indices based on loss.
				std::sort(loss_indices.begin(), loss_indices.end(), sortScorePairDescend<int>);

				for (int n = 0; n < nNumSel; n++)
				{
					sel_indices.insert(std::get<1>(loss_indices[n]));
				}
			}

			// Update the match indices and select neg_indices.
			for (int m = 0; m < match_indices[nLabel].size(); m++)
			{
				if (match_indices[nLabel][m] > -1)
				{
					if (m_miningType == SSD_MINING_TYPE_HARD_EXAMPLE && sel_indices.find(m) == sel_indices.end())
					{
						match_indices[nLabel][m] = -1;
						*pnNumMatches -= 1;
					}
				}
				else if (match_indices[nLabel][m] == -1)
				{
					if (sel_indices.find(m) != sel_indices.end())
					{
						neg_indices.push_back(m);
						*pnNumNegs += 1;
					}
				}
			}
		}

		all_neg_indices.push_back(neg_indices);
	}

	return 0;
}

template long SsdData<double>::mineHardExamples(vector<map<int, vector<BBOX>>>& rgAllLocPreds, map<int, vector<BBOX>>& rgAllGt, vector<BBOX>& rgPriorBboxes, vector<BBOX>& rgPriorVariances, vector<map<int, vector<float>>>& all_match_overlaps, vector<map<int, vector<int>>>& all_match_indices, vector<vector<int>>& all_neg_indices, int* pnNumMatches, int* pnNumNegs);
template long SsdData<float>::mineHardExamples(vector<map<int, vector<BBOX>>>& rgAllLocPreds, map<int, vector<BBOX>>& rgAllGt, vector<BBOX>& rgPriorBboxes, vector<BBOX>& rgPriorVariances, vector<map<int, vector<float>>>& all_match_overlaps, vector<map<int, vector<int>>>& all_match_indices, vector<vector<int>>& all_neg_indices, int* pnNumMatches, int* pnNumNegs);


// end