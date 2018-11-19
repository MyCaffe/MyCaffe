//=============================================================================
//	FILE:	handlecol.h
//
//	DESC:	This file implements the handle collection used to manage collections
//			of streams, tensors and descriptors.
//=============================================================================
#ifndef __HANDLECOL_CU__
#define __HANDLECOL_CU__

#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include "util.h"


//=============================================================================
//	Flags
//=============================================================================

const int MAX_HANDLES_MEM = 12288;
const int MAX_HANDLES = 4096;
const int MID_HANDLES = 1024;
const int MIN_HANDLES = 128;

//-----------------------------------------------------------------------------
//	HandleCollection Class
//
//	The HandleCollection class manages a set of generic handles.
//-----------------------------------------------------------------------------
template <size_t N>
class HandleCollection
{
	protected:
		void* m_rgHandles[N];
		long m_nLastIdx;

		void* free(void* plHandles[], long lCount, long hHandle);
		void* get(void* plHandles[], long lCount, long hHandle);

	public:
		HandleCollection();
		~HandleCollection();

		long Allocate(void* pData);
		void* Free(long hHandle);
		void* GetData(long hHandle);
		long GetCount();
};


//=============================================================================
//	Inline Methods
//=============================================================================

template <size_t N>
inline HandleCollection<N>::HandleCollection()
{
	memset(m_rgHandles, 0, sizeof(void*) * N);
	// Skip 0 index, so that this handle can be treated as NULL.
	m_nLastIdx = 1;
}

template <size_t N>
inline HandleCollection<N>::~HandleCollection()
{
}

template <size_t N>
inline long HandleCollection<N>::GetCount()
{
	return N;
}

template <size_t N>
inline long HandleCollection<N>::Allocate(void* pData)
{
	long hHandle = 0;
	long nFirstIdx = m_nLastIdx;

	while (m_nLastIdx < N)
	{
		if (m_rgHandles[m_nLastIdx] == NULL)
		{
			m_rgHandles[m_nLastIdx] = pData;
			hHandle = m_nLastIdx;
		}

		m_nLastIdx++;

		if (hHandle != 0)
			return hHandle;
	}

	// Skip 0 index, so that this handle can be treated as NULL.
	m_nLastIdx = 1;

	while (m_nLastIdx < nFirstIdx)
	{
		if (m_rgHandles[m_nLastIdx] == 0)
			hHandle = m_nLastIdx;

		m_nLastIdx++;

		if (hHandle != 0)
			return hHandle;
	}

	return -1;
}

template <size_t N>
inline void* HandleCollection<N>::free(void* plHandles[], long lCount, long hHandle)
{
	if (hHandle < 1 || hHandle >= lCount)
		return NULL;

	void* pData = plHandles[hHandle];
	plHandles[hHandle] = 0;

	return pData;
}

template <size_t N>
inline void* HandleCollection<N>::Free(long hHandle)
{
	return free(m_rgHandles, N, hHandle);
}

template <size_t N>
inline void* HandleCollection<N>::get(void* plHandles[], long lCount, long hHandle)
{
	if (hHandle < 1 || hHandle >= lCount)
		return NULL;

	return plHandles[hHandle];
}

template <size_t N>
inline void* HandleCollection<N>::GetData(long hHandle)
{
	return get(m_rgHandles, N, hHandle);
}

#endif // __HANDLECOL_CU__