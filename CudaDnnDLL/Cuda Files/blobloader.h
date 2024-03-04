//=============================================================================
//	FILE:	blobloader.h
//
//	DESC:	This file manages the BLOBLOADER functinality used to load weights
//          from very large files.
//=============================================================================
#ifndef __BLOBLOADER_CU__
#define __BLOBLOADER_CU__

#include "util.h"
#include "math.h"

//=============================================================================
//	Flags
//=============================================================================

//=============================================================================
//	Classes
//=============================================================================

template <class T>
class Memory;


//-----------------------------------------------------------------------------
//	BlobLoader Handle Class
//
//	This class stores the BlobLoader description information.
//-----------------------------------------------------------------------------
template <class T>
class blobloaderHandle
{
	Memory<T>* m_pMem;
	int m_nFd;
	size_t m_lFileSize;
	T* m_pfData;
	T* m_pfDataHost;
	long m_lDataHostSize;
	size_t m_lOffsetInBytes;
	size_t m_lOffsetInItems;

public:
	
	blobloaderHandle() 
	{
		m_pMem = NULL;
		m_pfData = NULL;
		m_pfDataHost = NULL;
		m_lDataHostSize = 0;
		m_nFd = -1;
		m_lFileSize = 0;
		m_lOffsetInBytes = 0;
		m_lOffsetInItems = 0;
	}

	long Initialize(Memory<T>* pMem, LPTSTR pszFile, int nHeaderSize);
	long CleanUp();
	long Load(long nCount, long hData, long lLocalItemOffset);

	long ResetOffset(unsigned long lOffsetInBytes)
	{
		m_lOffsetInBytes = lOffsetInBytes;
		m_lOffsetInItems = 0;
		return 0;
	}

	long AddToOffset(unsigned long lOffsetInItems)
	{
		m_lOffsetInItems += lOffsetInItems;
		return 0;
	}
};


//=============================================================================
//	Inline Methods
//=============================================================================


#endif