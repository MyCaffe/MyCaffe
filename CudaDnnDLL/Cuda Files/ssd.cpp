//=============================================================================
//	FILE:	ssd.cu
//
//	DESC:	This file implements the single-shot multi-box detection (ssd) algorithm
//=============================================================================

#include "util.h"
#include "ssd.h"
#include "memory.h"

//=============================================================================
//	Function Definitions
//=============================================================================

//=============================================================================
//	Private Classes
//=============================================================================

//=============================================================================
//	Class Methods
//=============================================================================

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
long ssdHandle<T>::MultiboxLossForward(int nLocDataCount, long hLocData, int nConfDataCount, long hConfData, int nPriorDataCount, long hPriorData, int nGtDataCount, long hGtData, int* pnNumMatches, int* pnNumNegs)
{
	LONG lErr;

	if (lErr = m_pData->SetMemory(nLocDataCount, hLocData, nConfDataCount, hConfData, nPriorDataCount, hPriorData, nGtDataCount, hGtData))
		return lErr;


	//--------------------------------------------------------
	//  Load all offsets.
	//--------------------------------------------------------

	map<int, vector<BBOX>> rgAllGt;
	if (lErr = m_pData->getGt(rgAllGt))
		return lErr;

	vector<BBOX> rgPriorBbox;
	vector<BBOX> rgPriorVariances;
	if (lErr = m_pData->getPrior(rgPriorBbox, rgPriorVariances))
		return lErr;

	vector<map<int, vector<BBOX>>> rgAllLocPreds;
	if (lErr = m_pData->getLocPrediction(rgAllLocPreds))
		return lErr;


	//--------------------------------------------------------
	//  Find all matches.
	//--------------------------------------------------------
	vector<map<int, vector<float>>> all_match_overlaps;
	if (lErr = m_pData->findMatches(rgAllLocPreds, rgAllGt, rgPriorBbox, rgPriorVariances, all_match_overlaps, m_pData->m_all_match_indices))
		return lErr;

	if (lErr = m_pData->mineHardExamples(rgAllLocPreds, rgAllGt, rgPriorBbox, rgPriorVariances, all_match_overlaps, m_pData->m_all_match_indices, m_pData->m_all_neg_indices, pnNumMatches, pnNumNegs))
		return lErr;

	return 0;
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
	vector<BBOX> rgPriorVariances;
	m_pData->getPrior(rgPriorBbox, rgPriorVariances);

	map<int, vector<BBOX>> rgAllGt;
	if (lErr = m_pData->getGt(rgAllGt))
		return lErr;

	if (lErr = m_pData->encodeLocPrediction(rgAllLocPreds, rgAllGt, m_pData->m_all_match_indices, rgPriorBbox, rgPriorVariances, m_pData->m_rgBbox[MEM_LOCPRED], m_pData->m_rgBbox[MEM_LOCGT]))
		return lErr;

	if (lErr = m_pData->m_rgBbox[MEM_LOCPRED]->CopyCpuToGpu())
		return lErr;

	if (lErr = m_pData->m_rgBbox[MEM_LOCGT]->CopyCpuToGpu())
		return lErr;

	return 0;
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
	if (lErr = m_pData->getGt(rgAllGt))
		return lErr;

	if (lErr = m_pData->encodeConfPrediction(m_pData->m_pConf, m_pData->m_all_match_indices, m_pData->m_all_neg_indices, rgAllGt, m_pData->m_rgBbox[MEM_CONFPRED], m_pData->m_rgBbox[MEM_CONFGT]))
		return lErr;

	if (lErr = m_pData->m_rgBbox[MEM_CONFPRED]->CopyCpuToGpu())
		return lErr;

	if (lErr = m_pData->m_rgBbox[MEM_CONFGT]->CopyCpuToGpu())
		return lErr;

	return 0;
}

template long ssdHandle<double>::EncodeConfPrediction(int nConfPredCount, long hConfPred, int nConfGtCount, long hConfGt);
template long ssdHandle<float>::EncodeConfPrediction(int nConfPredCount, long hConfPred, int nConfGtCount, long hConfGt);

// end