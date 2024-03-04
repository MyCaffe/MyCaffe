//=============================================================================
//	FILE:	blobloader.cu
//
//	DESC:	This file implements the class used to load blobs from very large
//          files.
//=============================================================================

#include "util.h"
#include "memory.h"
#include "blobloader.h"

#include <io.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/stat.h>
#include "win.h"

//=============================================================================
//	Class Methods
//=============================================================================

template <class T>
long blobloaderHandle<T>::Initialize(Memory<T>* pMem, LPTSTR pszFile, int nHeaderSize)
{
	USES_CONVERSION;
	std::string strFile = T2A(pszFile);
	m_pMem = pMem;

	WIN32_FILE_ATTRIBUTE_DATA fad;
	if (!GetFileAttributesEx(pszFile, GetFileExInfoStandard, &fad))
		return ERROR_FILE_NOT_FOUND;

	LARGE_INTEGER size;
	size.HighPart = fad.nFileSizeHigh;
	size.LowPart = fad.nFileSizeLow;
	m_lFileSize = (size_t)size.QuadPart;

	m_nFd = open(strFile.c_str(), O_RDONLY);
	if (m_nFd < 0)
		return ERROR_FILE_NOT_FOUND;

	m_pfData = (T*)mmap(NULL, m_lFileSize, PROT_READ, MAP_PRIVATE, m_nFd, 0);
	if (m_pfData == MAP_FAILED)
		return ERROR_FILE_NOT_FOUND;

	m_pfData = (T*)((char*)m_pfData + nHeaderSize);

	return 0;
}

template long blobloaderHandle<double>::Initialize(Memory<double>* pMem, LPTSTR pszFile, int nHeaderSize);
template long blobloaderHandle<float>::Initialize(Memory<float>* pMem, LPTSTR pszFile, int nHeaderSize);


template <class T>
long blobloaderHandle<T>::CleanUp()
{
	if (m_pfData != MAP_FAILED)
	{
		munmap(m_pfData, m_lFileSize);
		m_pfData = NULL;
		m_lFileSize = 0;
	}

	if (m_nFd >= 0)
	{
		_close(m_nFd);
		m_nFd = -1;
	}

	if (m_pfDataHost != NULL)
	{
		m_pMem->FreeHost(m_pfDataHost);
		m_pfDataHost = NULL;
		m_lDataHostSize = 0;	
	}

	return 0;
}

template long blobloaderHandle<double>::CleanUp();
template long blobloaderHandle<float>::CleanUp();

template <class T>
long blobloaderHandle<T>::Load(long nCount, long hData, long lLocalItemOffset)
{
	LONG lErr;

	if (m_pMem == NULL || m_pfData == NULL || m_nFd <= 0)
		return ERROR_BLOBLOADER_NOT_INITIALIZED;

	if (lLocalItemOffset < 0)
		return ERROR_INVALID_PARAMETER;

	MemoryItem* pData;
	MemoryCollection* pMem = m_pMem->GetMemoryCollection();

	if (lErr = pMem->GetData(hData, &pData))
		return lErr;

	T* data = (T*)pData->Data();
	long lSize = nCount * sizeof(T);

	if (lSize > pData->Size())
		return ERROR_INSUFFICIENT_BUFFER;

	if (lSize > m_lDataHostSize)
	{
		if (m_pfDataHost != NULL)
		{
			m_pMem->free_host(m_pfDataHost);
			m_lDataHostSize = 0;
		}

		if (lErr = m_pMem->alloc_host((void**)&m_pfDataHost, lSize))
			return lErr;

		m_lDataHostSize = lSize;
	}

	// Copy to pinned host memory.
	T* pfData = (T*)(((byte*)m_pfData) + m_lOffsetInBytes);
	memcpy(m_pfDataHost, pfData + m_lOffsetInItems + lLocalItemOffset, lSize);

	if (lErr = cudaMemcpy(data, m_pfDataHost, lSize, cudaMemcpyHostToDevice))
		return lErr;

	return 0;
}

template long blobloaderHandle<double>::Load(long nCount, long hData, long lLocalItemOffset);
template long blobloaderHandle<float>::Load(long nCount, long hData, long lLocalItemOffset);

// end