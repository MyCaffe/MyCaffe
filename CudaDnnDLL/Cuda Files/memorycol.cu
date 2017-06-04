//=============================================================================
//	FILE:	memorycol.cu
//
//	DESC:	This file the basic memory management for the given device
//=============================================================================

#include "memorycol.h"

//=============================================================================
//	Class Methods
//=============================================================================

MemoryCollection::~MemoryCollection()
{
	for (int i=0; i<MAX_ITEMS; i++)
	{
		m_rgHandles[i].Free();
	}
}

long MemoryCollection::Allocate(int nDeviceID, long lSize, void* pSrc, cudaStream_t pStream, long* phHandle)
{
	LONG lErr = 0;
	long nFirstIdx = m_nLastIdx;

	for (int i=m_nLastIdx; i<MAX_ITEMS; i++)
	{
		if (m_rgHandles[i].IsFree())
		{
			m_nLastIdx = i;			
			
			if (lErr = m_rgHandles[i].Allocate(nDeviceID, lSize, pSrc, pStream))
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
			
			if (lErr = m_rgHandles[i].Allocate(nDeviceID, lSize, pSrc, pStream))
				return lErr;

			m_lTotalMem += (unsigned long)m_rgHandles[i].Size();
			*phHandle = i;
			return 0;
		}
	}

	return ERROR_MEMORY_OUT;
}

long MemoryCollection::Allocate(int nDeviceID, void* pData, long lSize, long* phHandle)
{
	LONG lErr = 0;
	long nFirstIdx = m_nLastIdx;

	for (int i = m_nLastIdx; i<MAX_ITEMS; i++)
	{
		if (m_rgHandles[i].IsFree())
		{
			m_nLastIdx = i;

			if (lErr = m_rgHandles[i].Allocate(nDeviceID, pData, lSize))
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

			if (lErr = m_rgHandles[i].Allocate(nDeviceID, pData, lSize))
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