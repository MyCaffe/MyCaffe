//=============================================================================
//	FILE:	ssd_core.h
//
//	DESC:	This file manages the single-shot multi-box detection (ssd_core) algorithm
//=============================================================================
#ifndef __SSD_CORE_CU__
#define __SSD_CORE_CU__

#include "util.h"
#include <algorithm>
#include <vector>
#include <map>
#include <tuple>
#include <set>


//=============================================================================
//	Types
//=============================================================================

enum SsdMiningType
{
	SSD_MINING_TYPE_NONE = 0,
	SSD_MINING_TYPE_MAX_NEGATIVE = 1,
	SSD_MINING_TYPE_HARD_EXAMPLE = 2
};

enum SsdMatchingType
{
	SSD_MATCHING_TYPE_BIPARTITE = 0,
	SSD_MATCHING_TYPE_PER_PREDICTION = 1
};

enum SsdCodeType
{
	SSD_CODE_TYPE_CORNER = 1,
	SSD_CODE_TYPE_CENTER_SIZE = 2,
	SSD_CODE_TYPE_CORNER_SIZE = 3
};

enum SsdConfLossType
{
	SSD_CONF_LOSS_TYPE_SOFTMAX = 0,
	SSD_CONF_LOSS_TYPE_LOGISTIC = 1
};

enum SsdLocLossType
{
	SSD_LOC_LOSS_TYPE_L2 = 0,
	SSD_LOC_LOSS_TYPE_SMOOTH_L1 = 1
};

enum MEM
{
	MEM_LOC,
	MEM_CONF,
	MEM_PRIOR,
	MEM_GT,
	MEM_DECODE,
	MEM_LOCGT,
	MEM_LOCPRED,
	MEM_CONFGT,
	MEM_CONFPRED
};

//=============================================================================
//	Classes
//=============================================================================

template <class T>
class SsdData;

template <class T>
class Memory;

class MemoryCollection;

template <class T>
class Math;

using std::vector;
using std::map;
using std::tuple;
using std::set;

typedef tuple<int, MEM> BBOX;

template <class T>
class SsdMemory
{
	Memory<T>* m_pMem;
	MemoryCollection* m_pMemCol;
	int m_nGpuID;

public:
	int m_nMax;
	T* m_host;
	long m_handle;
	T* m_device;

	SsdMemory(Memory<T>* pMem, int nGpuID);
	~SsdMemory()
	{
		CleanUp();
	}

	LONG Initialize(int nCount, long hHandle = 0);
	void CleanUp();

	T* gpu_data()
	{
		return m_device;
	}

	T* cpu_data()
	{
		return m_host;
	}

	long gpu_handle()
	{
		return m_handle;
	}

	long CopyGpuToCpu()
	{
		return cudaMemcpy(m_host, m_device, m_nMax * sizeof(T), cudaMemcpyDeviceToHost);
	}
};

// Bbox ordering - full size
// 0 - nImageId
// 1 - label
// 2
// 3 - xmin 
// 4 - ymin 
// 5 - xmax
// 6 - ymax
// 7 - difficult
// Bbox ordering - half size
// 0 - xmin 
// 1 - ymin 
// 2 - xmax
// 3 - ymax
template <class T>
class SsdBbox : public SsdMemory<T>
{
	bool m_bFull = false;
	int m_nOffset = 0;
	int m_nTotal = 4;

public:
	SsdBbox(Memory<T>* pMem, int nGpuID, bool bFull = false) : SsdMemory(pMem, nGpuID)
	{
		m_bFull = bFull;

		if (bFull)
		{
			m_nOffset = 3;
			m_nTotal = 8;
		}
	}

	int offset(int nIdx)
	{
		return nIdx * m_nTotal;
	}

	int itemId(int nIdx)
	{
		if (!m_bFull)
			throw ERROR_SSD_NOT_SUPPORTED_IN_HALF_BBOX;

		int nOffset = nIdx * m_nTotal;
		return (int)m_host[nOffset];
	}

	int label(BBOX idx)
	{
		return label(std::get<0>(idx));
	}

	int label(int nIdx)
	{
		if (!m_bFull)
			throw ERROR_SSD_NOT_SUPPORTED_IN_HALF_BBOX;

		return labelAtOffset(nIdx * m_nTotal);
	}

	int labelAtOffset(int nOffset)
	{
		return (int)m_host[nOffset + 1];
	}

	T xmin(int nIdx)
	{
		int nOffset = offset(nIdx);
		return xminAtOffset(nOffset);
	}

	T xminAtOffset(int nOffset)
	{
		return m_host[nOffset + m_nOffset + 0];
	}

	T ymin(int nIdx)
	{
		int nOffset = offset(nIdx);
		return yminAtOffset(nOffset);
	}

	T yminAtOffset(int nOffset)
	{
		return m_host[nOffset + m_nOffset + 1];
	}

	T xmax(int nIdx)
	{
		int nOffset = offset(nIdx);
		return xmaxAtOffset(nOffset);
	}

	T xmaxAtOffset(int nOffset)
	{
		return m_host[nOffset + m_nOffset + 2];
	}

	T ymax(int nIdx)
	{
		int nOffset = offset(nIdx);
		return ymaxAtOffset(nOffset);
	}

	T ymaxAtOffset(int nOffset)
	{
		return m_host[nOffset + m_nOffset + 3];
	}

	long getBounds(int nIdx, T* pfxmin, T* pfymin, T* pfxmax, T* pfymax)
	{
		int nOffset = offset(nIdx);
		return getBoundsAtOffset(nOffset, pfxmin, pfymin, pfxmax, pfymax);
	}

	long getBoundsAtOffset(int nOffset, T* pfxmin, T* pfymin, T* pfxmax, T* pfymax)
	{
		*pfxmin = xminAtOffset(nOffset);
		*pfymin = yminAtOffset(nOffset);
		*pfxmax = xmaxAtOffset(nOffset);
		*pfymax = ymaxAtOffset(nOffset);
		return 0;
	}

	long setBounds(int nIdx, T fxmin, T fymin, T fxmax, T fymax)
	{
		int nOffset = offset(nIdx);
		return setBoundsAtOffset(nOffset, fxmin, fymin, fxmax, fymax);
	}

	long setBoundsAtOffset(int nOffset, T fxmin, T fymin, T fxmax, T fymax)
	{
		m_host[nOffset + m_nOffset + 0] = fxmin;
		m_host[nOffset + m_nOffset + 1] = fymin;
		m_host[nOffset + m_nOffset + 2] = fxmax;
		m_host[nOffset + m_nOffset + 3] = fymax;
		return 0;
	}

	long divBounds(int nIdx, T fxmin, T fymin, T fxmax, T fymax)
	{
		int nOffset = offset(nIdx);
		return divBoundsAtOffset(nOffset, fxmin, fymin, fxmax, fymax);
	}

	long divBoundsAtOffset(int nOffset, T fxmin, T fymin, T fxmax, T fymax)
	{
		m_host[nOffset + m_nOffset + 0] /= fxmin;
		m_host[nOffset + m_nOffset + 1] /= fymin;
		m_host[nOffset + m_nOffset + 2] /= fxmax;
		m_host[nOffset + m_nOffset + 3] /= fymax;
		return 0;
	}

	static T getSize(T xmin, T ymin, T xmax, T ymax, bool bNormalized = true)
	{
		// If bbox is invalid (e.g. xmax < xmin or ymax < ymin), return 0.
		if (xmax < xmin || ymax < ymin)
			return 0;

		T width = xmax - xmin;
		T height = ymax - ymin;

		if (bNormalized)
			return width * height;

		// If bbox is not within range [0, 1]
		return (width + 1) * (height + 1);
	}

	T getSize(BBOX idx, bool bNormalized = true)
	{
		return getSize(std::get<0>(idx), bNormalized);
	}

	T getSize(int nIdx, bool bNormalized = true)
	{
		int nOffset = offset(nIdx);

		T xmin;
		T ymin;
		T xmax;
		T ymax;

		getBounds(nOffset, &xmin, &ymin, &xmax, &ymax);
		return getSize(xmin, ymin, xmax, ymax, bNormalized);
	}

	static void clip(T* pfxmin, T* pfymin, T* pfxmax, T* pfymax)
	{
		*pfxmin = std::max(std::min(*pfxmin, T(1)), T(0));
		*pfymin = std::max(std::min(*pfymin, T(1)), T(0));
		*pfxmax = std::max(std::min(*pfxmax, T(1)), T(0));
		*pfymax = std::max(std::min(*pfymax, T(1)), T(0));
	}

	void clip(int nIdx, T* pfSize)
	{
		int nOffset = offset(nIdx);

		T xmin;
		T ymin;
		T xmax;
		T ymax;

		getBounds(nOffset, &xmin, &ymin, &xmax, &ymax);
		clip(&xmin, &ymin, &xmax, &ymax);
		setBounds(nOffset, xmin, ymin, xmax, ymax);

		*pfSize = getSize(MEM_DECODE, nIdx * 4);
	}

	bool isCrossBoundaryBbox(BBOX idx)
	{
		return isCrossBoundaryBbox(std::get<0>(idx));
	}

	bool isCrossBoundaryBbox(int nIdx)
	{
		T fxmin;
		T fymin;
		T fxmax;
		T fymax;

		getBounds(nIdx, &fxmin, &fymin, &fxmax, &fymax);

		return (fxmin < 0 || fxmin > 1 ||
			fymin < 0 || fymin > 1 ||
			fxmax < 0 || fxmax > 1 ||
			fymax < 0 || fymax > 1);
	}
};

template <class T>
class SsdData
{
	Memory<T>* m_pMem;
	Math<T>* m_pMath;

public:
	int m_nGpuID;
	int m_nNum;
	int m_nNumPriors;
	int m_nNumGt;
	int m_nNumClasses;
	bool m_bShareLocation;
	int m_nLocClasses;
	int m_nBackgroundLabelId;
	bool m_bUseDifficultGt;
	SsdMiningType m_miningType;
	SsdMatchingType m_matchingType;
	T m_fOverlapThreshold;
	bool m_bUsePriorForMatching;
	SsdCodeType m_codeType;
	bool m_bEncodeVariantInTgt;
	bool m_bBpInside;
	bool m_bIgnoreCrossBoundaryBbox;
	bool m_bUsePriorForNms;
	SsdConfLossType m_confLossType;
	SsdLocLossType m_locLossType;
	T m_fNegPosRatio;
	T m_fNegOverlap;
	int m_nSampleSize;
	bool m_bMapObjectToAgnostic;
	T m_fNmsThreshold;
	int m_nTopK;
	T m_fEta;
	vector<map<int, vector<int>>> m_all_match_indices;
	vector<vector<int>> m_all_neg_indices;

	vector<SsdBbox<T>*> m_rgBbox;
	SsdMemory<T>* m_pMatch;
	SsdMemory<T>* m_pProb;
	SsdMemory<T>* m_pConfLoss;
	SsdMemory<T>* m_pScale;

	SsdData(Memory<T>* pMem, Math<T>* pMath);
	~SsdData();

	LONG Initialize(int nGpuID, int nNumClasses, bool bShareLocation, int nLocClasses, int nBackgroundLabelId, bool bUseDifficultGt, SsdMiningType miningType, SsdMatchingType matchingType, T fOverlapThreshold, bool bUsePriorForMatching, SsdCodeType codeType, bool bEncodeVariantInTgt, bool bBpInside, bool bIgnoreCrossBoundaryBbox, bool bUsePriorForNms, SsdConfLossType confLossType, SsdLocLossType locLossType, T fNegPosRatio, T fNegOverlap, int nSampleSize, bool bMapObjectToAgnostic, T fNmsThreshold, int nTopK, T fEta)
	{
		m_nNumClasses = nNumClasses;
		m_bShareLocation = bShareLocation;
		m_nLocClasses = nLocClasses;
		m_nBackgroundLabelId = nBackgroundLabelId;
		m_bUseDifficultGt = bUseDifficultGt;
		m_miningType = miningType;
		m_matchingType = matchingType;
		m_fOverlapThreshold = fOverlapThreshold;
		m_bUsePriorForMatching = bUsePriorForMatching;
		m_codeType = codeType;
		m_bEncodeVariantInTgt = bEncodeVariantInTgt;
		m_bBpInside = bBpInside;
		m_bIgnoreCrossBoundaryBbox = bIgnoreCrossBoundaryBbox;
		m_bUsePriorForNms = bUsePriorForNms;
		m_confLossType = confLossType;
		m_locLossType = locLossType;
		m_fNegOverlap = fNegOverlap;
		m_fNegPosRatio = fNegPosRatio;
		m_nSampleSize = nSampleSize;
		m_bMapObjectToAgnostic = bMapObjectToAgnostic;
		m_fNmsThreshold = fNmsThreshold;
		m_nTopK = nTopK;
		m_fEta = fEta;

		if ((m_rgBbox[MEM_LOC] = new SsdBbox<T>(m_pMem, nGpuID)) == NULL)
			return ERROR_MEMORY_OUT;

		if ((m_rgBbox[MEM_CONF] = new SsdBbox<T>(m_pMem, nGpuID)) == NULL)
			return ERROR_MEMORY_OUT;

		if ((m_rgBbox[MEM_PRIOR] = new SsdBbox<T>(m_pMem, nGpuID)) == NULL)
			return ERROR_MEMORY_OUT;

		if ((m_rgBbox[MEM_GT] = new SsdBbox<T>(m_pMem, nGpuID, true)) == NULL)
			return ERROR_MEMORY_OUT;

		if ((m_rgBbox[MEM_DECODE] = new SsdBbox<T>(m_pMem, nGpuID, true)) == NULL)
			return ERROR_MEMORY_OUT;

		if ((m_rgBbox[MEM_LOCGT] = new SsdBbox<T>(m_pMem, nGpuID, true)) == NULL)
			return ERROR_MEMORY_OUT;

		if ((m_rgBbox[MEM_LOCPRED] = new SsdBbox<T>(m_pMem, nGpuID, true)) == NULL)
			return ERROR_MEMORY_OUT;

		if ((m_rgBbox[MEM_CONFGT] = new SsdBbox<T>(m_pMem, nGpuID, true)) == NULL)
			return ERROR_MEMORY_OUT;

		if ((m_rgBbox[MEM_CONFPRED] = new SsdBbox<T>(m_pMem, nGpuID, true)) == NULL)
			return ERROR_MEMORY_OUT;

		if ((m_pMatch = new SsdMemory<T>(m_pMem, nGpuID)) == NULL)
			return ERROR_MEMORY_OUT;

		if ((m_pProb = new SsdMemory<T>(m_pMem, nGpuID)) == NULL)
			return ERROR_MEMORY_OUT;

		if ((m_pConfLoss = new SsdMemory<T>(m_pMem, nGpuID)) == NULL)
			return ERROR_MEMORY_OUT;

		if ((m_pScale = new SsdMemory<T>(m_pMem, nGpuID)) == NULL)
			return ERROR_MEMORY_OUT;

		return 0;
	}

	LONG SetMemory(int nLocDataCount, long hLocData, int nConfDataCount, long hConfData, int nPriorDataCount, long hPriorData, int nGtDataCount, long hGtData)
	{
		LONG lErr;

		if (lErr = m_rgBbox[MEM_LOC]->Initialize(nLocDataCount, hLocData))
			return lErr;

		if (lErr = m_rgBbox[MEM_CONF]->Initialize(nConfDataCount, hConfData))
			return lErr;

		if (lErr = m_rgBbox[MEM_PRIOR]->Initialize(nPriorDataCount, hPriorData))
			return lErr;

		if (lErr = m_rgBbox[MEM_GT]->Initialize(nGtDataCount, hGtData))
			return lErr;

		if (lErr = m_rgBbox[MEM_DECODE]->Initialize(nPriorDataCount / 2))
			return lErr;

		if (lErr = m_pMatch->Initialize(m_nNum * m_nNumPriors))
			return lErr;

		if (lErr = m_pProb->Initialize(nConfDataCount))
			return lErr;

		if (lErr = m_pConfLoss->Initialize((m_nNum * m_nNumPriors) * 1 * 1))
			return lErr;

		if (lErr = m_pScale->Initialize((m_nNum * m_nNumPriors) * m_nNumClasses * 1))
			return lErr;

		return 0;
	}

	LONG SetMemoryLocPred(int nLocPredCount, long hLocPred, int nLocGtCount, long hLocGt)
	{
		LONG lErr;

		if (lErr = m_rgBbox[MEM_LOCPRED]->Initialize(nLocPredCount, hLocPred))
			return lErr;

		if (lErr = m_rgBbox[MEM_LOCGT]->Initialize(nLocGtCount, hLocGt))
			return lErr;

		return 0;
	}

	LONG SetMemoryConfPred(int nConfPredCount, long hConfPred, int nConfGtCount, long hConfGt)
	{
		LONG lErr;

		if (lErr = m_rgBbox[MEM_CONFPRED]->Initialize(nConfPredCount, hConfPred))
			return lErr;

		if (lErr = m_rgBbox[MEM_CONFGT]->Initialize(nConfGtCount, hConfGt))
			return lErr;

		return 0;
	}

	LONG Setup(int nNum, int nNumPriors, int nNumGt)
	{
		m_nNum = nNum;
		m_nNumPriors = nNumPriors;
		m_nNumGt = nNumGt;

		return 0;
	}

	int getPrior(vector<BBOX>& rgPriorBbox, vector<int>& rgPriorVariances)
	{
		rgPriorBbox.clear();
		rgPriorVariances.clear();

		for (int i = 0; i < m_nNumPriors; i++)
		{
			rgPriorBbox.push_back(BBOX(i, MEM_PRIOR));
		}

		for (int i = 0; i < m_nNumPriors; i++)
		{
			rgPriorVariances.push_back(m_nNumPriors + i);
		}

		return (int)rgPriorBbox.size();
	}

	int getGt(map<int, vector<BBOX>>& rgGt)
	{
		rgGt.clear();

		for (int i = 0; i < m_nNumGt; i++)
		{
			int nItemId = m_rgBbox[MEM_GT]->itemId(i);
			if (nItemId == -1)
				continue;

			rgGt[nItemId].push_back(BBOX(i, MEM_GT));
		}

		return (int)rgGt.size();
	}

	int getLocPrediction(vector<map<int, vector<BBOX>>>& rgLocPreds)
	{
		int nNumPredsPerClass = m_nNumPriors;
		int nNumLocClasses = m_nLocClasses;
		int nIdx = 0;

		rgLocPreds.clear();
		rgLocPreds.resize(m_nNum);

		for (int i = 0; i < m_nNum; i++)
		{
			map<int, vector<BBOX>> labelbox = rgLocPreds[i];

			for (int p = 0; p < nNumPredsPerClass; p++)
			{
				int nStartIdx = p * nNumLocClasses * 4;

				for (int c = 0; c < nNumLocClasses; c++)
				{
					int nLabel = (m_bShareLocation) ? -1 : c;

					if (labelbox.find(nLabel) == labelbox.end())
						labelbox[nLabel].resize(m_nNumPriors);

					labelbox[nLabel][p] = BBOX(nIdx + nStartIdx + c, MEM_LOC);
				}
			}

			nIdx += nNumPredsPerClass * nNumLocClasses;
		}

		return (int)rgLocPreds.size();
	}

	long decode(BBOX priorBbox, int nPriorVar, bool bClip, BBOX locPred, T* pfxmin, T* pfymin, T* pfxmax, T* pfymax, T* pfsize);

	long decode(int i, BBOX priorBbox, int nPriorVar, bool bClip, BBOX locPred, T* pfDecodeSize)
	{
		LONG lErr;

		T fxmin_decode;
		T fymin_decode;
		T fxmax_decode;
		T fymax_decode;
		T fsize_decode;

		if (lErr = decode(priorBbox, nPriorVar, bClip, locPred, &fxmin_decode, &fymin_decode, &fxmax_decode, &fymax_decode, &fsize_decode))
			return lErr;

		m_rgBbox[MEM_DECODE]->setBounds(i, fxmin_decode, fymin_decode, fxmax_decode, fymax_decode);

		return 0;
	}

	long decode(vector<BBOX>& rgPriorBbox, vector<int>& rgPriorVariances, bool bClip, vector<BBOX>& rgLocPreds, vector<BBOX>& rgDecodeBbox)
	{
		LONG lErr;

		int nNumBoxes = m_nNumPriors;
		rgDecodeBbox.clear();

		for (int i = 0; i < nNumBoxes; i++)
		{
			T fDecodeSize;
			if (lErr = decode(i, rgPriorBbox[i], rgPriorVariances[i], bClip, rgLocPreds[i], &fDecodeSize))
				return lErr;

			rgDecodeBbox.push_back(BBOX(i, MEM_DECODE));
		}

		return 0;
	}

	long encode(BBOX priorBbox, int nPriorVar, BBOX gtBbox, T* pfxmin, T* pfymin, T* pfxmax, T* pfymax);

	void intersect(T fxmin1, T fymin1, T fxmax1, T fymax1, T fxmin2, T fymin2, T fxmax2, T fymax2, T* pfxmin, T* pfymin, T* pfxmax, T* pfymax)
	{
		if (fxmin2 > fxmax1 || fxmax2 < fxmin1 ||
			fymin2 > fymax1 || fymax2 < fymin2)
		{
			*pfxmin = 0;
			*pfymin = 0;
			*pfxmax = 0;
			*pfymax = 0;
		}
		else
		{
			*pfxmin = std::max(fxmin1, fxmin2);
			*pfymin = std::max(fymin1, fymin2);
			*pfxmax = std::min(fxmax1, fxmax2);
			*pfymax = std::min(fymax1, fymax2);
		}
	}

	int getLabel(BBOX bbox)
	{
		int nOffset = std::get<0>(bbox);
		MEM type = std::get<1>(bbox);
		return m_rgBbox[type]->label(nOffset);
	}

	T getSize(BBOX bbox)
	{
		int nOffset = std::get<0>(bbox);
		MEM type = std::get<1>(bbox);
		return m_rgBbox[type]->getSize(nOffset);
	}

	bool isCrossBoundaryBbox(BBOX bbox)
	{
		int nOffset = std::get<0>(bbox);
		MEM type = std::get<1>(bbox);
		return m_rgBbox[type]->isCrossBoundaryBbox(nOffset);
	}

	long jaccardOverlap(BBOX bbox1, BBOX bbox2, float* pfJaccardOverlap)
	{
		int nOffset = std::get<0>(bbox1);
		MEM type1 = std::get<1>(bbox1);
		T fxmin1;
		T fymin1;
		T fxmax1;
		T fymax1;
		m_rgBbox[type1]->getBounds(nOffset, &fxmin1, &fymin1, &fxmax1, &fymax1);

		nOffset = std::get<0>(bbox2);
		MEM type2 = std::get<1>(bbox2);
		T fxmin2;
		T fymin2;
		T fxmax2;
		T fymax2;
		m_rgBbox[type2]->getBounds(nOffset, &fxmin2, &fymin2, &fxmax2, &fymax2);

		T fxmin_intersect;
		T fymin_intersect;
		T fxmax_intersect;
		T fymax_intersect;
		intersect(fxmin1, fymin1, fxmax1, fymax2, fxmin2, fymin2, fxmax2, fymax2, &fxmin_intersect, &fymin_intersect, &fxmax_intersect, &fymax_intersect);

		T finter_width = fxmax_intersect - fxmin_intersect;
		T finter_height = fymax_intersect - fymin_intersect;
		T finter_size = finter_width * finter_height;

		T fsize1 = m_rgBbox[type1]->getSize(bbox1);
		T fsize2 = m_rgBbox[type2]->getSize(bbox2);

		*pfJaccardOverlap = (float)(finter_size / (fsize1 + fsize2 - finter_size));

		return 0;
	}

	long match(vector<BBOX>& rgGt, vector<BBOX>& rgPredBBox, int nLabel, vector<int>* match_indices, vector<float>* match_overlaps);

	long findMatches(vector<map<int, vector<BBOX>>>& rgAllLocPreds, map<int, vector<BBOX>>& rgAllGt, vector<BBOX>& rgPriorBbox, vector<int>& rgPriorVariances, vector<map<int, vector<float>>>& all_match_overlaps, vector<map<int, vector<int>>>& all_match_indices);

	int countNumMatches(vector<map<int, vector<int>>>& all_match_indices, int nNum)
	{
		int nNumMatches = 0;

		for (int i = 0; i < nNum; i++)
		{
			const map<int, vector<int>>& match_indices = all_match_indices[i];

			for (map<int, vector<int>>::const_iterator it = match_indices.begin(); it != match_indices.end(); it++)
			{
				const vector<int>& match_index = it->second;

				for (int m = 0; m < (int)match_index.size(); m++)
				{
					if (match_index[m] > -1)
						nNumMatches++;
				}
			}
		}

		return nNumMatches;
	}

	long softmax(SsdMemory<T>* pData, const int nOuterNum, const int nChannels, const int nInnerNum, SsdMemory<T>* pProb);

	long computeConfLoss(int nNum, int nNumPriors, int nNumClasses, vector<map<int, vector<int>>>& all_match_indices, map<int, vector<BBOX>>& rgAllGt, vector<vector<T>>*pall_conf_loss);

	long load_conf_loss(int nNum, int nNumPriors, SsdMemory<T>* pConfLoss, vector<vector<T>>* pall_conf_loss);

	long encodeLocPrediction(vector<map<int, vector<BBOX>>>& rgAllLocPreds, map<int, vector<BBOX>>& rgAllGt, vector<map<int, vector<int>>>& all_match_indices, vector<BBOX>& rgPriorBboxes, vector<int>& rgPriorVariances, SsdMemory<T>* pLocPred, SsdMemory<T>* pLocGt);

	long encodeConfPrediction(SsdBbox<T>* pConf, vector<map<int, vector<int>>>& all_match_indices, vector<vector<int>>& all_neg_indices, map<int, vector<BBOX>>& rgAllGt, SsdMemory<T>* pConfPred, SsdMemory<T>* pConfGt);

	long computeLocLoss(SsdMemory<T>* pLocPred, SsdMemory<T>* pLogGt, vector<map<int, vector<int>>>& all_match_indices, int nNum, int nNumPriors, SsdLocLossType loss_type, vector<vector<T>>* pall_loc_loss);

	bool isEligibleMining(const SsdMiningType miningType, const int nMatchIdx, const float fMatchOverlap, const float fNegOverlap)
	{
		if (miningType == SSD_MINING_TYPE_MAX_NEGATIVE)
			return nMatchIdx == -1 && fMatchOverlap < fNegOverlap;
		else if (miningType == SSD_MINING_TYPE_HARD_EXAMPLE)
			return true;
		else
			return false;
	}

	long getTopKScoreIndex(vector<T>& scores, vector<int>& indices, int nTopK, vector<tuple<float, int>>* pscore_index);

	long applyNMS(vector<BBOX>& bboxes, vector<T>& scores, T fThreshold, int nTopK, vector<int>* pindices);

	long mineHardExamples(vector<map<int, vector<BBOX>>>& rgAllLocPreds, map<int, vector<BBOX>>& rgAllGt, vector<BBOX>& rgPriorBboxes, vector<int>& rgPriorVariances, vector<map<int, vector<float>>>& all_match_overlaps, vector<map<int, vector<int>>>& all_match_indices, vector<vector<int>>& all_neg_indices, int* pnNumMatches, int* pnNumNegs);
};

//=============================================================================
//	Inline Methods
//=============================================================================

template <>
inline long SsdData<double>::load_conf_loss(int nNum, int nNumPriors, SsdMemory<double>* pConfLoss, vector<vector<double>>* pall_conf_loss)
{
	const double* loss_data = m_pConfLoss->cpu_data();
	for (int i = 0; i < nNum; i++)
	{
		vector<double> conf_loss(loss_data, loss_data + nNumPriors);
		pall_conf_loss->push_back(conf_loss);
		loss_data += nNumPriors;
	}

	return 0;
}

template <>
inline long SsdData<float>::load_conf_loss(int nNum, int nNumPriors, SsdMemory<float>* pConfLoss, vector<vector<float>>* pall_conf_loss)
{
	const float* loss_data = m_pConfLoss->cpu_data();
	for (int i = 0; i < nNum; i++)
	{
		vector<float> conf_loss(loss_data, loss_data + nNumPriors);
		pall_conf_loss->push_back(conf_loss);
		loss_data += nNumPriors;
	}

	return 0;
}

#endif