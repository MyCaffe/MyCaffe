//=============================================================================
//	FILE:	memorycol.cu
//
//	DESC:	This file the basic memory management for the given device
//=============================================================================

#include "memorycol.h"
#include <mutex>


//=============================================================================
//	Class Methods
//=============================================================================

std::mutex g_mutex;

long MemoryItem::Free()
{
	std::lock_guard<std::mutex> lock(g_mutex);

	if (m_pData == NULL)
		return 0;

	if (m_bOwner)
		cudaFree(m_pData);

	m_pData = NULL;
	m_lSize = 0;

	return 0;
}


MemoryCollection::~MemoryCollection()
{
	for (int i=0; i<MAX_ITEMS; i++)
	{
		m_rgHandles[i].Free();
	}
}

long MemoryCollection::Allocate(int nDeviceID, bool bHalf, size_t lSize, void* pSrc, cudaStream_t pStream, long* phHandle)
{
	LONG lErr = 0;
	long nFirstIdx = m_nLastIdx;

	for (int i=m_nLastIdx; i<MAX_ITEMS; i++)
	{
		if (m_rgHandles[i].IsFree())
		{
			m_nLastIdx = i;			
			
			if (lErr = m_rgHandles[i].Allocate(nDeviceID, bHalf, lSize, pSrc, pStream))
				return lErr;

			m_lTotalMem += (unsigned long)m_rgHandles[i].Size();
			*phHandle = i;
			return 0;
		}
	}

	m_nLastIdx = 1;

	for (int i=m_nLastIdx; i<nFirstIdx; i++)
	{
		if (m_rgHandles[i].IsFree())
		{
			m_nLastIdx = i;			
			
			if (lErr = m_rgHandles[i].Allocate(nDeviceID, bHalf, lSize, pSrc, pStream))
				return lErr;

			m_lTotalMem += (unsigned long)m_rgHandles[i].Size();
			*phHandle = i;
			return 0;
		}
	}

	return ERROR_MEMORY_OUT;
}

long MemoryCollection::Allocate(int nDeviceID, bool bHalf, void* pData, size_t lSize, long* phHandle)
{
	LONG lErr = 0;
	long nFirstIdx = m_nLastIdx;

	for (int i = m_nLastIdx; i<MAX_ITEMS; i++)
	{
		if (m_rgHandles[i].IsFree())
		{
			m_nLastIdx = i;

			if (lErr = m_rgHandles[i].Allocate(nDeviceID, bHalf, pData, lSize))
				return lErr;

			m_lTotalMem += (unsigned long)m_rgHandles[i].Size();
			*phHandle = i;
			return 0;
		}
	}

	m_nLastIdx = 1;

	for (int i = m_nLastIdx; i<nFirstIdx; i++)
	{
		if (m_rgHandles[i].IsFree())
		{
			m_nLastIdx = i;

			if (lErr = m_rgHandles[i].Allocate(nDeviceID, bHalf, pData, lSize))
				return lErr;

			m_lTotalMem += (unsigned long)m_rgHandles[i].Size();
			*phHandle = i;
			return 0;
		}
	}

	return ERROR_MEMORY_OUT;
}

long MemoryCollection::Free(long hHandle)
{
	if (hHandle < 1 || hHandle >= MAX_ITEMS)
		return ERROR_PARAM_OUT_OF_RANGE;

	m_lTotalMem -= (unsigned long)m_rgHandles[hHandle].Size();

	return m_rgHandles[hHandle].Free();
}


//end memorycol.cu