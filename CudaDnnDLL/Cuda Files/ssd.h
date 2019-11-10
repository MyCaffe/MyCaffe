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
	long Initialize(int nGpuID, int nNumClasses, bool bShareLocation, int nLocClasses, int nBackgroundLabelId, bool bUseDifficultGt, SsdMiningType miningType, SsdMatchingType matchingType, T fOverlapThreshold, bool bUsePriorForMatching, SsdCodeType codeType, bool bEncodeVariantInTgt, bool bBpInside, bool bIgnoreCrossBoundaryBbox, bool bUsePriorForNms, SsdConfLossType confLossType, SsdLocLossType locLossType, T fNegPosRatio, T fNegOverlap, int nSampleSize, bool bMapObjectToAgnostic, T fNmsThreshold, int nTopK, T fEta);
	long CleanUp();

	long Setup(int nNum, int nNumPriors, int nNumGt);
	long MultiboxLossForward(int nLocDataCount, long hLocData, int nConfDataCount, long hConfData, int nPriorDataCount, long hPriorData, int nGtDataCount, long hGtData, int* pnNumMatches, int* pnNumNegs);
	long EncodeLocPrediction(int nLocPredCount, long hLocPred, int nLocGtCount, long hLocGt);
	long EncodeConfPrediction(int nConfPredCount, long hConfPred, int nConfGtCount, long hConfGt);
};


//=============================================================================
//	Inline Methods
//=============================================================================


#endif