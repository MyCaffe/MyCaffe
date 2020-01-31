//=============================================================================
//	FILE:	ssd.h
//
//	DESC:	This file manages the single-shot multi-box detection (ssd) algorithm
//=============================================================================
#ifndef __SSD_CU__
#define __SSD_CU__

#include "util.h"
#include "math.h"
#include "memorycol.h"
#include "handlecol.h"
#include "ssd_core.h"
#include <vector>
#include <map>
#include <tuple>
#include <set>


//=============================================================================
//	Types
//=============================================================================

//=============================================================================
//	Classes
//=============================================================================

template <class T>
class SsdData;

template <class T>
class Memory;

using std::vector;
using std::map;
using std::tuple;
using std::set;

//-----------------------------------------------------------------------------
//	SSD Handle Class
//
//	This class stores the SSD description information.
//-----------------------------------------------------------------------------
template <class T>
class ssdHandle
{
	int m_nRefCount = 0;
	Memory<T>* m_pMem;
	MemoryCollection* m_pMemCol;
	Math<T>* m_pMath;
	SsdData<T>* m_pData;
	bool m_bOwner;

public:
	
	ssdHandle()
	{
		m_pData = NULL;
		m_bOwner = true;
	}

	int RefCount()
	{
		return m_nRefCount;
	}

	bool IsOwner()
	{
		return m_bOwner;
	}

	void SetOwner(bool bOwner)
	{
		m_bOwner = bOwner;
	}

	long Update(Memory<T>* pMem, Math<T>* pMath);

	long Initialize(int nGpuID, int nNumClasses, bool bShareLocation, int nLocClasses, int nBackgroundLabelId, bool bUseDifficultGt, SsdMiningType miningType, SsdMatchingType matchingType, T fOverlapThreshold, bool bUsePriorForMatching, SsdCodeType codeType, bool bEncodeVariantInTgt, bool bBpInside, bool bIgnoreCrossBoundaryBbox, bool bUsePriorForNms, SsdConfLossType confLossType, SsdLocLossType locLossType, T fNegPosRatio, T fNegOverlap, int nSampleSize, bool bMapObjectToAgnostic, bool bNmsActive, T fNmsThreshold, int nTopK, T fEta)
	{
		long lErr;

		if (lErr = m_pData->Initialize(nGpuID, nNumClasses, bShareLocation, nLocClasses, nBackgroundLabelId, bUseDifficultGt, miningType, matchingType, fOverlapThreshold, bUsePriorForMatching, codeType, bEncodeVariantInTgt, bBpInside, bIgnoreCrossBoundaryBbox, bUsePriorForNms, confLossType, locLossType, fNegPosRatio, fNegOverlap, nSampleSize, bMapObjectToAgnostic, bNmsActive, fNmsThreshold, nTopK, fEta))
			return lErr;

		return 0;
	}

	long CleanUp()
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

	long Setup(int nNum, int nNumPriors, int nNumGt)
	{
		LONG lErr;

		if (m_pData == NULL)
			return ERROR_SSD_NOT_INITIALIZED;

		if (lErr = m_pData->Setup(nNum, nNumPriors, nNumGt))
			return lErr;

		return 0;
	}

	long MultiboxLossForward(int nLocDataCount, long hLocData, int nConfDataCount, long hConfData, int nPriorDataCount, long hPriorData, int nGtDataCount, long hGtData, int* pnNumMatches, int* pnNumNegs);
	long EncodeLocPrediction(int nLocPredCount, long hLocPred, int nLocGtCount, long hLocGt);
	long EncodeConfPrediction(int nConfPredCount, long hConfPred, int nConfGtCount, long hConfGt);

	long GetAllMatchIndices(vector<map<int, vector<int>>>* pall_match_indices)
	{
		if (m_pData == NULL)
			return ERROR_SSD_NOT_INITIALIZED;
		*pall_match_indices = m_pData->GetAllMatchIndices();
		return 0;
	}

	long GetAllNegIndices(vector<vector<int>>* pall_neg_indices)
	{
		if (m_pData == NULL)
			return ERROR_SSD_NOT_INITIALIZED;
		*pall_neg_indices = m_pData->GetAllNegIndices();
		return 0;
	}
};


//=============================================================================
//	Inline Methods
//=============================================================================


#endif