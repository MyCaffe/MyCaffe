//=============================================================================
//	FILE:	ssd.cu
//
//	DESC:	This file implements the single-shot multi-box detection (ssd) algorithm
//=============================================================================

#include "util.h"
#include "ssd.h"
#include "memory.h"

#include "boost/iterator/counting_iterator.hpp"

//=============================================================================
//	Function Definitions
//=============================================================================

template <typename T>
bool sortScorePairDescend(const tuple<T, T>& t1, const tuple<T, T>& t2)
{
	if (std::get<0>(t1) > std::get<0>(t2))
		return true;
	else
		return false;
}

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

	SsdMemory(Memory<T>* pMem, int nGpuID)
	{
		m_pMem = pMem;
		m_pMemCol = pMem->GetMemoryCollection();
		m_nGpuID = nGpuID;
		m_nMax = 0;
		m_host = NULL;
		m_device = NULL;
		m_handle = 0;
	}

	~SsdMemory()
	{
		CleanUp();
	}

	LONG Initialize(int nCount, long hHandle = 0)
	{
		LONG lErr;
		T* pSrc = NULL;

		if (m_nMax < nCount)
		{
			if (m_host != NULL)
			{
				m_pMem->FreeHost(m_host);
				m_host = NULL;
			}

			if (m_handle != 0)
			{
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
				m_pMem->FreeMemory(m_handle);
				m_handle = 0;
			}
		}
		else
		{
			if (m_handle == 0)
			{
				if (lErr = m_pMem->AllocMemory(m_nGpuID, false, nCount, NULL, 0, &m_handle))
					return lErr;
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

	void CleanUp()
	{
		if (m_host != NULL)
		{
			m_pMem->FreeHost(m_host);
			m_host = NULL;
		}

		if (m_handle != NULL)
		{
			m_pMem->FreeMemory(m_handle);
			m_handle = 0;
		}

		m_device = NULL;
		m_nMax = 0;
	}

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

	SsdData(Memory<T>* pMem, Math<T>* pMath) : m_rgBbox(9, NULL)
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

	~SsdData()
	{
		for (int i = 0; i < m_rgBbox.size(); i++)
		{
			if (m_rgBbox[i] != NULL)
				delete m_rgBbox[i];
		}

		if (m_pMatch != NULL)
			delete m_pMatch;

		if (m_pProb != NULL)
			delete m_pProb;

		if (m_pConfLoss != NULL)
			delete m_pConfLoss;

		if (m_pScale != NULL)
			delete m_pScale;
	}

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

	long decode(BBOX priorBbox, int nPriorVar, bool bClip, BBOX locPred, T* pfxmin, T* pfymin, T* pfxmax, T* pfymax, T* pfsize)
	{
		int nOffset = std::get<0>(priorBbox);
		MEM memPrior = std::get<1>(priorBbox);
		T fxmin_prior;
		T fymin_prior;
		T fxmax_prior;
		T fymax_prior;
		m_rgBbox[memPrior]->getBounds(nOffset, &fxmin_prior, &fymin_prior, &fxmax_prior, &fymax_prior);

		nOffset = nPriorVar;
		T fxmin_prior_var;
		T fymin_prior_var;
		T fxmax_prior_var;
		T fymax_prior_var;
		m_rgBbox[memPrior]->getBounds(nOffset, &fxmin_prior_var, &fymin_prior_var, &fxmax_prior_var, &fymax_prior_var);

		nOffset = std::get<0>(locPred);
		MEM memLoc = std::get<1>(locPred);
		T fxmin_bbox;
		T fymin_bbox;
		T fxmax_bbox;
		T fymax_bbox;
		m_rgBbox[memLoc]->getBounds(nOffset, &fxmin_bbox, &fymin_bbox, &fxmax_bbox, &fymax_bbox);

		if (m_codeType == SSD_CODE_TYPE_CORNER)
		{
			// Variance is encoded in target, we simply need to add the offset predictions.
			if (m_bEncodeVariantInTgt)
			{
				*pfxmin = fxmin_prior + fxmin_bbox;
				*pfymin = fymin_prior + fymin_bbox;
				*pfxmax = fxmax_prior + fxmax_bbox;
				*pfymax = fymax_prior + fymax_bbox;
			}
			// Variance is encoded in bbox, we need to scale the offset accordingly.
			else
			{
				*pfxmin = fxmin_prior + fxmin_prior_var * fxmin_bbox;
				*pfymin = fymin_prior + fymin_prior_var * fymin_bbox;
				*pfxmax = fxmax_prior + fxmax_prior_var * fxmax_bbox;
				*pfymax = fymax_prior + fymax_prior_var * fymax_bbox;
			}
		}
		else if (m_codeType == SSD_CODE_TYPE_CENTER_SIZE)
		{
			T fprior_width = fxmax_prior - fxmin_prior;
			if (fprior_width < 0)
				return ERROR_SSD_INVALID_BBOX_DIMENSION;
			T fprior_height = fymax_prior - fymin_prior;
			if (fprior_height < 0)
				return ERROR_SSD_INVALID_BBOX_DIMENSION;
			T fprior_center_x = (fxmin_prior + fxmax_prior) / T(2.0);
			T fprior_center_y = (fymin_prior + fymax_prior) / T(2.0);
			T fdecode_center_x;
			T fdecode_center_y;
			T fdecode_width;
			T fdecode_height;

			// Variance is encoded in target, we simply need to add the offset predictions.
			if (m_bEncodeVariantInTgt)
			{
				fdecode_center_x = fxmin_bbox * fprior_width + fprior_center_x;
				fdecode_center_y = fymin_bbox * fprior_height + fprior_center_y;
				fdecode_width = exp(fxmax_bbox) * fprior_width;
				fdecode_height = exp(fymax_bbox) * fprior_height;
			}
			// Variance is encoded in bbox, we need to scale the offset accordingly.
			else
			{
				fdecode_center_x = fxmin_prior_var * fxmin_bbox * fprior_width + fprior_center_x;
				fdecode_center_y = fymin_prior_var * fymin_bbox * fprior_height + fprior_center_y;
				fdecode_width = exp(fxmax_prior_var * fxmax_bbox) * fprior_width;
				fdecode_height = exp(fymax_prior_var * fymax_bbox) * fprior_height;
			}

			*pfxmin = fdecode_center_x - fdecode_width / T(2.0);
			*pfymin = fdecode_center_y - fdecode_height / T(2.0);
			*pfxmax = fdecode_center_x + fdecode_width / T(2.0);
			*pfymax = fdecode_center_y + fdecode_height / T(2.0);
		}
		else if (m_codeType == SSD_CODE_TYPE_CORNER_SIZE)
		{
			T fprior_width = fxmax_prior - fxmin_prior;
			if (fprior_width < 0)
				return ERROR_SSD_INVALID_BBOX_DIMENSION;
			T fprior_height = fymax_prior - fymin_prior;
			if (fprior_height < 0)
				return ERROR_SSD_INVALID_BBOX_DIMENSION;

			// Variance is encoded in target, we simply need to add the offset predictions.
			if (m_bEncodeVariantInTgt)
			{
				*pfxmin = fxmin_prior + fprior_width * fxmin_bbox;
				*pfymin = fymin_prior + fprior_height * fymin_bbox;
				*pfxmax = fxmax_prior + fprior_width * fxmax_bbox;
				*pfymax = fymax_prior + fprior_height * fymax_bbox;
			}
			// Variance is encoded in bbox, we need to scale the offset accordingly.
			else
			{
				*pfxmin = fxmin_prior + fxmin_prior_var * fprior_width * fxmin_bbox;
				*pfymin = fymin_prior + fymin_prior_var * fprior_height * fymin_bbox;
				*pfxmax = fxmax_prior + fxmax_prior_var * fprior_width * fxmax_bbox;
				*pfymax = fymax_prior + fymax_prior_var * fprior_height * fymax_bbox;
			}
		}
		else
		{
			return ERROR_SSD_INVALID_CODE_TYPE;
		}

		if (bClip)
			SsdBbox<T>::clip(pfxmin, pfymin, pfxmax, pfymax);

		*pfsize = SsdBbox<T>::getSize(*pfxmin, *pfymin, *pfxmax, *pfymax);
		return 0;
	}

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

	long encode(BBOX priorBbox, int nPriorVar, BBOX gtBbox, T* pfxmin, T* pfymin, T* pfxmax, T* pfymax)
	{
		int nOffset = std::get<0>(priorBbox);
		MEM memPrior = std::get<1>(priorBbox);
		T fxmin_prior;
		T fymin_prior;
		T fxmax_prior;
		T fymax_prior;
		m_rgBbox[memPrior]->getBounds(nOffset, &fxmin_prior, &fymin_prior, &fxmax_prior, &fymax_prior);

		nOffset = nPriorVar;
		T fxmin_prior_var;
		T fymin_prior_var;
		T fxmax_prior_var;
		T fymax_prior_var;
		m_rgBbox[memPrior]->getBounds(nOffset, &fxmin_prior_var, &fymin_prior_var, &fxmax_prior_var, &fymax_prior_var);

		nOffset = std::get<0>(gtBbox);
		MEM memGt = std::get<1>(gtBbox);
		T fxmin_bbox;
		T fymin_bbox;
		T fxmax_bbox;
		T fymax_bbox;
		m_rgBbox[memGt]->getBounds(nOffset, &fxmin_bbox, &fymin_bbox, &fxmax_bbox, &fymax_bbox);

		if (m_codeType == SSD_CODE_TYPE_CORNER)
		{
			if (m_bEncodeVariantInTgt)
			{
				*pfxmin = fxmin_bbox - fxmin_prior;
				*pfymin = fymin_bbox - fymin_prior;
				*pfxmax = fxmax_bbox - fxmax_prior;
				*pfymax = fymax_bbox - fymax_prior;
			}
			// Encode variance in bbox
			else
			{
				if (fxmin_prior_var == 0 || fymin_prior_var == 0 || fxmax_prior_var == 0 || fymax_prior_var == 0)
					return ERROR_PARAM_OUT_OF_RANGE;

				*pfxmin = (fxmin_bbox - fxmin_prior) / fxmin_prior_var;
				*pfymin = (fymin_bbox - fymin_prior) / fymin_prior_var;
				*pfxmax = (fxmax_bbox - fxmax_prior) / fxmax_prior_var;
				*pfymax = (fymax_bbox - fymax_prior) / fymax_prior_var;
			}
		}
		else if (m_codeType == SSD_CODE_TYPE_CENTER_SIZE)
		{
			T fprior_width = fxmax_prior - fxmin_prior;
			if (fprior_width < 0)
				return ERROR_SSD_INVALID_BBOX_DIMENSION;
			T fprior_height = fymax_prior - fymin_prior;
			if (fprior_height < 0)
				return ERROR_SSD_INVALID_BBOX_DIMENSION;
			T fprior_center_x = (fxmin_prior + fxmax_prior) / T(2.0);
			T fprior_center_y = (fymin_prior + fymax_prior) / T(2.0);

			T fbbox_width = fxmax_bbox - fxmin_bbox;
			if (fbbox_width < 0)
				return ERROR_SSD_INVALID_BBOX_DIMENSION;
			T fbbox_height = fymax_bbox - fymin_bbox;
			if (fbbox_height < 0)
				return ERROR_SSD_INVALID_BBOX_DIMENSION;
			T fbbox_center_x = (fxmin_bbox + fxmax_bbox) / T(2.0);
			T fbbox_center_y = (fymin_bbox + fymax_bbox) / T(2.0);

			if (m_bEncodeVariantInTgt)
			{
				*pfxmin = (fbbox_center_x - fprior_center_x) / fprior_width;
				*pfymin = (fbbox_center_y - fprior_center_y) / fprior_height;
				*pfxmax = log(fbbox_width / fprior_width);
				*pfymax = log(fbbox_height / fprior_height);
			}
			// Encode variance in bbox.
			else
			{
				*pfxmin = (fbbox_center_x - fprior_center_x) / fprior_width / fxmin_prior_var;
				*pfymin = (fbbox_center_y - fprior_center_y) / fprior_height / fymin_prior_var;
				*pfxmax = log(fbbox_width / fprior_width) / fxmax_prior_var;
				*pfymax = log(fbbox_height / fprior_height) / fymax_prior_var;
			}
		}
		else if (m_codeType == SSD_CODE_TYPE_CORNER_SIZE)
		{
			T fprior_width = fxmax_prior - fxmin_prior;
			if (fprior_width < 0)
				return ERROR_SSD_INVALID_BBOX_DIMENSION;
			T fprior_height = fymax_prior - fymin_prior;
			if (fprior_height < 0)
				return ERROR_SSD_INVALID_BBOX_DIMENSION;

			if (m_bEncodeVariantInTgt)
			{
				*pfxmin = (fxmin_bbox - fxmin_prior) / fprior_width;
				*pfymin = (fymin_bbox - fymin_prior) / fprior_height;
				*pfxmax = (fxmax_bbox - fxmax_prior) / fprior_width;
				*pfymax = (fymax_bbox - fymax_prior) / fprior_height;
			}
			// Encode variance in bbox.
			else
			{
				if (fxmin_prior_var == 0 || fymin_prior_var == 0 || fxmax_prior_var == 0 || fymax_prior_var == 0)
					return ERROR_PARAM_OUT_OF_RANGE;

				*pfxmin = (fxmin_bbox - fxmin_prior) / fprior_width / fxmin_prior_var;
				*pfymin = (fymin_bbox - fymin_prior) / fprior_height / fymin_prior_var;
				*pfxmax = (fxmax_bbox - fxmax_prior) / fprior_width / fxmax_prior_var;
				*pfymax = (fymax_bbox - fymax_prior) / fprior_height / fymax_prior_var;
			}
		}
		else
		{
			return ERROR_SSD_INVALID_CODE_TYPE;
		}

		return 0;
	}

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

	long match(vector<BBOX>& rgGt, vector<BBOX>& rgPredBBox, int nLabel, vector<int>* match_indices, vector<float>* match_overlaps)
	{
		LONG lErr;
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
				MEM mem = std::get<1>(rgGt[i]);
				int nGtLabel = m_rgBbox[mem]->label(rgGt[i]);
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
		for (int i = 0; i < nNumPred; i++)
		{
			if (m_bIgnoreCrossBoundaryBbox && isCrossBoundaryBbox(rgPredBBox[i]))
			{
				(*match_indices)[i] = -2;
				continue;
			}

			for (int j = 0; j < nNumGt; j++)
			{
				float fOverlap;
				if (lErr = jaccardOverlap(rgPredBBox[i], rgGt[rgGtIndices[j]], &fOverlap))
					return lErr;

				if (fOverlap > T(1e-6))
				{
					(*match_overlaps)[i] = std::max((*match_overlaps)[i], fOverlap);
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

			// Perform the per prediction matching (bipartite is aready done)
			if (m_matchingType == SSD_MATCHING_TYPE_PER_PREDICTION)
			{
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
						if ((*match_indices)[nMaxIdx] != -1)
							return ERROR_SSD_BAD_MATCH;

						(*match_indices)[i] = rgGtIndices[nMaxGtIdx];
						(*match_overlaps)[i] = fMaxOverlap;
					}
				}
			}
		}

		return 0;
	}

	long findMatches(vector<map<int, vector<BBOX>>>& rgAllLocPreds, map<int, vector<BBOX>>& rgAllGt, vector<BBOX>& rgPriorBbox, vector<int>& rgPriorVariances, vector<map<int, vector<float>>>& all_match_overlaps, vector<map<int, vector<int>>>& all_match_indices)
	{
		LONG lErr;

		// Find the matches.
		for (int i = 0; i < m_nNum; i++)
		{
			map<int, vector<int>> match_indices;
			map<int, vector<float>> match_overlaps;

			// Check if there is a ground truth for the current image.
			if (rgAllGt.find(i) == rgAllGt.end())
			{
				// Three is no gt for the current image, so all predictiosn are negative.
				all_match_indices.push_back(match_indices);
				all_match_overlaps.push_back(match_overlaps);
				continue;
			}

			// Find match between predictions and ground truth.
			vector<BBOX> rgGtBbox = rgAllGt.find(i)->second;
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

	long softmax(SsdMemory<T>* pData, const int nOuterNum, const int nChannels, const int nInnerNum, SsdMemory<T>* pProb)
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
		if (lErr = m_pMath->channel_sum(nOuterNum * nInnerNum, nOuterNum, nChannels, nInnerNum, pProb->gpu_handle(), m_pScale->gpu_handle()))
			return lErr;

		// Diide
		if (lErr = m_pMath->channel_div(nCount, nOuterNum, nChannels, nInnerNum, m_pScale->gpu_handle(), pProb->gpu_handle()))
			return lErr;

		return 0;
	}

	long computeConfLoss(int nNum, int nNumPriors, int nNumClasses, vector<map<int, vector<int>>>& all_match_indices, map<int, vector<BBOX>>& rgAllGt, vector<vector<T>>*pall_conf_loss)
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
		compute_conf_loss_kernel<T><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n, pConf->gpu_data(), nNumPriors, nNumClasses, m_confLossType, m_pMatch->gpu_data(), m_pConfLoss->gpu_data());
		if (lErr = cudaStreamSynchronize(0))
			return lErr;

		// Save the loss.
		pall_conf_loss->clear();
		m_pConfLoss->CopyGpuToCpu();
		load_conf_loss(nNum, nNumPriors, m_pConfLoss, pall_conf_loss);

		return 0;
	}

	long load_conf_loss(int nNum, int nNumPriors, SsdMemory<T>* pConfLoss, vector<vector<T>>* pall_conf_loss);

	long encodeLocPrediction(vector<map<int, vector<BBOX>>>& rgAllLocPreds, map<int, vector<BBOX>>& rgAllGt, vector<map<int, vector<int>>>& all_match_indices, vector<BBOX>& rgPriorBboxes, vector<int>& rgPriorVariances, SsdMemory<T>* pLocPred, SsdMemory<T>* pLocGt)
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

				if (rgAllLocPreds[i].find(nLabel) == rgAllLocPreds[i].end())
					return ERROR_SSD_LOC_PRED_LABEL_NOT_FOUND;

				const vector<BBOX>& loc_pred = rgAllLocPreds[i].find(nLabel)->second;

				for (int j = 0; j < match_index.size(); j++)
				{
					if (match_index[j] <= -1)
						continue;

					// Store encoded ground truth.
					const int nGtIdx = match_index[j];
					if (rgAllGt.find(i) == rgAllGt.end() || nGtIdx >= rgAllGt.find(i)->second.size())
						return ERROR_SSD_GT_LABEL_OUT_OF_RANGE;

					BBOX gtbbox = rgAllGt.find(i)->second[nGtIdx];

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
						m_rgBbox[MEM_LOCPRED]->setBounds(nCount,
							(xmin_match < 0 || xmin_match > 1) ? xmin_gtencode : m_rgBbox[MEM_LOC]->xmin(j),
							(ymin_match < 0 || ymin_match > 1) ? ymin_gtencode : m_rgBbox[MEM_LOC]->ymin(j),
							(xmax_match < 0 || xmax_match > 1) ? xmax_gtencode : m_rgBbox[MEM_LOC]->xmax(j),
							(ymax_match < 0 || ymax_match > 1) ? ymax_gtencode : m_rgBbox[MEM_LOC]->ymax(j));
					}
					else
					{
						m_rgBbox[MEM_LOCPRED]->setBounds(nCount,
							m_rgBbox[MEM_LOC]->xmin(j),
							m_rgBbox[MEM_LOC]->ymin(j),
							m_rgBbox[MEM_LOC]->xmax(j),
							m_rgBbox[MEM_LOC]->ymax(j));
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

	long encodeConfPrediction(SsdBbox<T>* pConf, vector<map<int, vector<int>>>& all_match_indices, vector<vector<int>>& all_neg_indices, map<int, vector<BBOX>>& rgAllGt, SsdMemory<T>* pConfPred, SsdMemory<T>* pConfGt)
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
			if (rgAllGt.find(i) != rgAllGt.end())
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

						const int gt_label = (m_bMapObjectToAgnostic) ? m_nBackgroundLabelId + 1 : getLabel(rgAllGt.find(i)->second[match_index[j]]);
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
								return ERROR_SSD_CONF_LOSS_TYPE_UNKNOWN;
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
								return ERROR_SSD_CONF_LOSS_TYPE_UNKNOWN;
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

	long computeLocLoss(SsdMemory<T>* pLocPred, SsdMemory<T>* pLogGt, vector<map<int, vector<int>>>& all_match_indices, int nNum, int nNumPriors, SsdLocLossType loss_type, vector<vector<T>>* pall_loc_loss)
	{
		return 0;
	}

	bool isEligibleMining(const SsdMiningType miningType, const int nMatchIdx, const float fMatchOverlap, const float fNegOverlap)
	{
		if (miningType == SSD_MINING_TYPE_MAX_NEGATIVE)
			return nMatchIdx == -1 && fMatchOverlap < fNegOverlap;
		else if (miningType == SSD_MINING_TYPE_HARD_EXAMPLE)
			return true;
		else
			return false;
	}

	long getTopKScoreIndex(vector<T>& scores, vector<int>& indices, int nTopK, vector<tuple<float, int>>* pscore_index)
	{
		if (scores.size() != indices.size())
			return ERROR_PARAM_OUT_OF_RANGE;

		// Generate index score pairs.
		for (int i = 0; i < scores.size(); i++)
		{
			pscore_index->push_back(std::make_tuple(scores[i], indices[i]));
		}

		// Sort the score pair according to the scores in descending order
		std::stable_sort(pscore_index->begin(), pscore_index->end(), sortScorePairDescend<int>);

		// Keep nTopK scores if needed
		if (nTopK > -1 && nTopK < pscore_index->size())
			pscore_index->resize(nTopK);

		return 0;
	}

	long applyNMS(vector<BBOX>& bboxes, vector<T>& scores, T fThreshold, int nTopK, vector<int>* pindices)
	{
		LONG lErr;
		bool bReuseOverlaps = false;
		map<int, map<int, float>> overlaps;

		if (bboxes.size() != scores.size())
			return ERROR_PARAM_OUT_OF_RANGE;

		// Get top_k scores with (corresponding indices).
		vector<int> idx(boost::counting_iterator<int>(0), boost::counting_iterator<int>(scores.size()));
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
					if (overlaps.find(nBestIdx) != overlaps.end() &&
						overlaps.find(nBestIdx)->second.find(nCurIdx) != overlaps[nBestIdx].end())
					{
						// Use the computed overlap.
						fCurOverlap = overlaps[nBestIdx][nCurIdx];
					}
					else if (overlaps.find(nCurIdx) != overlaps.end() &&
						overlaps.find(nCurIdx)->second.find(nBestIdx) != overlaps[nCurIdx].end())
					{
						// Use the computed overlap.
						fCurOverlap = overlaps[nCurIdx][nBestIdx];
					}
					else
					{
						if (lErr = jaccardOverlap(best_bbox, cur_bbox, &fCurOverlap))
							return lErr;

						// Store the overlap for future use.
						overlaps[nBestIdx][nCurIdx] = fCurOverlap;
					}
				}
				else
				{
					if (lErr = jaccardOverlap(best_bbox, cur_bbox, &fCurOverlap))
						return lErr;
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

	long mineHardExamples(vector<map<int, vector<BBOX>>>& rgAllLocPreds, map<int, vector<BBOX>>& rgAllGt, vector<BBOX>& rgPriorBboxes, vector<int>& rgPriorVariances, vector<map<int, vector<float>>>& all_match_overlaps, vector<map<int, vector<int>>>& all_match_indices, vector<vector<int>>& all_neg_indices, int* pnNumMatches, int* pnNumNegs)
	{
		LONG lErr;
		int nNum = (int)rgAllLocPreds.size();
		*pnNumMatches = countNumMatches(all_match_indices, nNum);
		*pnNumNegs = 0;
		int nNumPriors = (int)rgPriorBboxes.size();

		if (nNumPriors != (int)rgPriorVariances.size())
			return ERROR_SSD_PRIOR_VARIANCE_COUNT;

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
			if (!pnNumMatches != 0)
			{
				if (lErr = m_rgBbox[MEM_LOCGT]->Initialize(*pnNumMatches * 4))
					return lErr;

				if (lErr = m_rgBbox[MEM_LOCPRED]->Initialize(*pnNumMatches * 4))
					return lErr;

				if (lErr = encodeLocPrediction(rgAllLocPreds, rgAllGt, all_match_indices, rgPriorBboxes, rgPriorVariances, m_rgBbox[MEM_LOCPRED], m_rgBbox[MEM_LOCGT]))
					return lErr;
			}

			if (lErr = computeLocLoss(m_rgBbox[MEM_LOCPRED], m_rgBbox[MEM_LOCGT], all_match_indices, nNum, nNumPriors, m_locLossType, &all_loc_loss))
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
				vector<tuple<T, int>> loss_indices;
				for (int m = 0; m < match_indices[nLabel].size(); m++)
				{
					if (isEligibleMining(m_miningType, match_indices[nLabel][m], match_overlaps.find(nLabel)->second[m], (float)m_fNegOverlap))
					{
						loss_indices.push_back(std::make_tuple(rgLoss[m], m));
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

					nNumSel = std::min(static_cast<int>(nNumPos * m_fNegPosRatio), nNumSel);
				}
				else if (m_miningType == SSD_MINING_TYPE_HARD_EXAMPLE)
				{
					if (m_nSampleSize <= 0)
						return ERROR_SSD_SAMPLE_SIZE_TOO_SMALL;
					
					nNumSel = std::min(m_nSampleSize, nNumSel);
				}

				// Select samples.
				if (m_fNmsThreshold > 0)
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
						OutputDebugString(L"Not enough sample after nms!");

					// Pick top example indices after nms.
					nNumSel = std::min(static_cast<int>(nms_indices.size()), nNumSel);
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
};

template <>
long SsdData<double>::load_conf_loss(int nNum, int nNumPriors, SsdMemory<double>* pConfLoss, vector<vector<double>>* pall_conf_loss)
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
long SsdData<float>::load_conf_loss(int nNum, int nNumPriors, SsdMemory<float>* pConfLoss, vector<vector<float>>* pall_conf_loss)
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


//=============================================================================
//	Class Methods
//=============================================================================

template <class T>
long ssdHandle<T>::Initialize(int nGpuID, int nNumClasses, bool bShareLocation, int nLocClasses, int nBackgroundLabelId, bool bUseDifficultGt, SsdMiningType miningType, SsdMatchingType matchingType, T fOverlapThreshold, bool bUsePriorForMatching, SsdCodeType codeType, bool bEncodeVariantInTgt, bool bBpInside, bool bIgnoreCrossBoundaryBbox, bool bUsePriorForNms, SsdConfLossType confLossType, SsdLocLossType locLossType, T fNegPosRatio, T fNegOverlap, int nSampleSize, bool bMapObjectToAgnostic, T fNmsThreshold, int nTopK, T fEta)
{
	long lErr;

	if (lErr = m_pData->Initialize(nGpuID, nNumClasses, bShareLocation, nLocClasses, nBackgroundLabelId, bUseDifficultGt, miningType, matchingType, fOverlapThreshold, bUsePriorForMatching, codeType, bEncodeVariantInTgt, bBpInside, bIgnoreCrossBoundaryBbox, bUsePriorForNms, confLossType, locLossType, fNegPosRatio, fNegOverlap, nSampleSize, bMapObjectToAgnostic, fNmsThreshold, nTopK, fEta))
		return lErr;

	return 0;
}

template long ssdHandle<double>::Initialize(int nGpuID, int nNumClasses, bool bShareLocation, int nLocClasses, int nBackgroundLabelId, bool bUseDifficultGt, SsdMiningType miningType, SsdMatchingType matchingType, double fOverlapThreshold, bool bUsePriorForMatching, SsdCodeType codeType, bool bEncodeVariantInTgt, bool bBpInside, bool bIgnoreCrossBoundaryBbox, bool bUsePriorForNms, SsdConfLossType confLossType, SsdLocLossType locLossType, double fNegPosRatio, double fNegOverlap, int nSampleSize, bool bMapObjectToAgnostic, double fNmsThreshold, int nTopK, double fEta);
template long ssdHandle<float>::Initialize(int nGpuID, int nNumClasses, bool bShareLocation, int nLocClasses, int nBackgroundLabelId, bool bUseDifficultGt, SsdMiningType miningType, SsdMatchingType matchingType, float fOverlapThreshold, bool bUsePriorForMatching, SsdCodeType codeType, bool bEncodeVariantInTgt, bool bBpInside, bool bIgnoreCrossBoundaryBbox, bool bUsePriorForNms, SsdConfLossType confLossType, SsdLocLossType locLossType, float fNegPosRatio, float fNegOverlap, int nSampleSize, bool bMapObjectToAgnostic, float fNmsThreshold, int nTopK, float fEta);


template <class T>
long ssdHandle<T>::Update(Memory<T>* pMem, Math<T>* pMath)
{
	m_pMem = pMem;
	m_pMath = pMath;
	m_nRefCount++;

	m_pData = new SsdData<T>(pMem, pMath);
	if (m_pData == NULL)
		return ERROR_MEMORY_OUT;

	return 0;
}

template long ssdHandle<double>::Update(Memory<double>* pMem, Math<double>* pMath);
template long ssdHandle<float>::Update(Memory<float>* pMem, Math<float>* pMath);


template <class T>
long ssdHandle<T>::CleanUp()
{
	m_nRefCount--;

	if (m_nRefCount == 0)
	{
		if (m_pData != NULL)
		{
			delete m_pData;
			m_pData = NULL;
		}
	}

	return 0;
}

template long ssdHandle<double>::CleanUp();
template long ssdHandle<float>::CleanUp();

template <class T>
long ssdHandle<T>::Setup(int nNum, int nNumPriors, int nNumGt)
{
	LONG lErr;

	if (m_pData == NULL)
		return ERROR_SSD_NOT_INITIALIZED;

	if (lErr = m_pData->Setup(nNum, nNumPriors, nNumGt))
		return lErr;

	return 0;
}

template long ssdHandle<double>::Setup(int nNum, int nNumPriors, int nNumGt);
template long ssdHandle<float>::Setup(int nNum, int nNumPriors, int nNumGt);


template <class T>
long ssdHandle<T>::MultiboxLossForward(int nLocDataCount, long hLocData, int nConfDataCount, long hConfData, int nPriorDataCount, long hPriorData, int nGtDataCount, long hGtData, int* pnNumMatches, int* pnNumNegs)
{
	LONG lErr;

	if (lErr = m_pData->SetMemory(nLocDataCount, hLocData, nConfDataCount, hConfData, nPriorDataCount, hPriorData, nGtDataCount, hGtData))
		return lErr;


	//--------------------------------------------------------
	//  Load all offsets.
	//--------------------------------------------------------

	map<int, vector<BBOX>> rgAllGt;
	m_pData->getGt(rgAllGt);

	vector<BBOX> rgPriorBbox;
	vector<int> rgPriorVariances;
	m_pData->getPrior(rgPriorBbox, rgPriorVariances);

	vector<map<int, vector<BBOX>>> rgAllLocPreds;
	m_pData->getLocPrediction(rgAllLocPreds);


	//--------------------------------------------------------
	//  Find all matches.
	//--------------------------------------------------------
	vector<map<int, vector<float>>> all_match_overlaps;
	lErr = m_pData->findMatches(rgAllLocPreds, rgAllGt, rgPriorBbox, rgPriorVariances, all_match_overlaps, m_pData->m_all_match_indices);

	if (lErr == 0)
		lErr = m_pData->mineHardExamples(rgAllLocPreds, rgAllGt, rgPriorBbox, rgPriorVariances, all_match_overlaps, m_pData->m_all_match_indices, m_pData->m_all_neg_indices, pnNumMatches, pnNumNegs);

	return lErr;
}

template long ssdHandle<double>::MultiboxLossForward(int nLocDataCount, long hLocData, int nConfDataCount, long hConfData, int nPriorDataCount, long hPriorData, int nGtDataCount, long hGtData, int* pnNumMatches, int* pnNumNegs);
template long ssdHandle<float>::MultiboxLossForward(int nLocDataCount, long hLocData, int nConfDataCount, long hConfData, int nPriorDataCount, long hPriorData, int nGtDataCount, long hGtData, int* pnNumMatches, int* pnNumNegs);


template <class T>
long ssdHandle<T>::EncodeLocPrediction(int nLocPredCount, long hLocPred, int nLocGtCount, long hLocGt)
{
	LONG lErr;

	if (lErr = m_pData->SetMemoryLocPred(nLocPredCount, hLocPred, nLocGtCount, hLocGt))
		return lErr;

	vector<map<int, vector<BBOX>>> rgAllLocPreds;
	m_pData->getLocPrediction(rgAllLocPreds);

	vector<BBOX> rgPriorBbox;
	vector<int> rgPriorVariances;
	m_pData->getPrior(rgPriorBbox, rgPriorVariances);

	map<int, vector<BBOX>> rgAllGt;
	m_pData->getGt(rgAllGt);

	return m_pData->encodeLocPrediction(rgAllLocPreds, rgAllGt, m_pData->m_all_match_indices, rgPriorBbox, rgPriorVariances, m_pData->m_rgBbox[MEM_LOCPRED], m_pData->m_rgBbox[MEM_LOCGT]);
}

template long ssdHandle<double>::EncodeLocPrediction(int nLocPredCount, long hLocPred, int nLocGtCount, long hLocGt);
template long ssdHandle<float>::EncodeLocPrediction(int nLocPredCount, long hLocPred, int nLocGtCount, long hLocGt);


template <class T>
long ssdHandle<T>::EncodeConfPrediction(int nConfPredCount, long hConfPred, int nConfGtCount, long hConfGt)
{
	LONG lErr;

	if (lErr = m_pData->SetMemoryConfPred(nConfPredCount, hConfPred, nConfGtCount, hConfGt))
		return lErr;

	map<int, vector<BBOX>> rgAllGt;
	m_pData->getGt(rgAllGt);

	return m_pData->encodeConfPrediction(m_pData->m_rgBbox[MEM_CONF], m_pData->m_all_match_indices, m_pData->m_all_neg_indices, rgAllGt, m_pData->m_rgBbox[MEM_CONFPRED], m_pData->m_rgBbox[MEM_CONFGT]);
}

template long ssdHandle<double>::EncodeConfPrediction(int nConfPredCount, long hConfPred, int nConfGtCount, long hConfGt);
template long ssdHandle<float>::EncodeConfPrediction(int nConfPredCount, long hConfPred, int nConfGtCount, long hConfGt);

// end