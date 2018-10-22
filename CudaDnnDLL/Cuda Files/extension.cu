//=============================================================================
//	FILE:	extension.cu
//
//	DESC:	This file implements DLL extension functionality of a single
//			extension.
//=============================================================================

#include "util.h"
#include "extension.h"
#include "memory.h"
#include "device.h"

//=============================================================================
//	Types
//=============================================================================

#define SZDLL_INITFLOATCUSTOM "DLL_InitFloatCustomExtension"
#define SZDLL_INVOKEFLOATCUSTOM "DLL_InvokeFloatCustomExtension"

#define SZDLL_INITDOUBLECUSTOM "DLL_InitDoubleCustomExtension"
#define SZDLL_INVOKEDOUBLECUSTOM "DLL_InvokeDoubleCustomExtension"

typedef LONG(WINAPI *LPFNDLLINITCUSTOM)(HMODULE hParent, long lKernelIdx);


//=============================================================================
//	Class Methods
//=============================================================================

long extensionHandle<float>::InitializeFloat(HMODULE hParent, LONG lKernelIdx, LPTSTR pszDllPath)
{
	long lErr;

	m_hLib = LoadLibrary(pszDllPath);
	if (m_hLib == NULL)
		return GetLastError();

	LPFNDLLINITCUSTOM pfnInit = (LPFNDLLINITCUSTOM)GetProcAddress(m_hLib, SZDLL_INITFLOATCUSTOM);
	if (pfnInit == NULL)
	{
		FreeLibrary(m_hLib);
		m_hLib = NULL;
		return ERROR_NOT_SUPPORTED;
	}

	if (lErr = (*pfnInit)(hParent, lKernelIdx))
	{
		FreeLibrary(m_hLib);
		m_hLib = NULL;
		return lErr;
	}

	m_pfn = GetProcAddress(m_hLib, SZDLL_INVOKEFLOATCUSTOM);
	if (m_pfn == NULL)
	{
		FreeLibrary(m_hLib);
		m_hLib = NULL;
		return ERROR_NOT_SUPPORTED;
	}

	return 0;
}

long extensionHandle<double>::InitializeDouble(HMODULE hParent, LONG lKernelIdx, LPTSTR pszDllPath)
{
	long lErr;

	m_hLib = LoadLibrary(pszDllPath);
	if (m_hLib == NULL)
		return GetLastError();

	LPFNDLLINITCUSTOM pfnInit = (LPFNDLLINITCUSTOM)GetProcAddress(m_hLib, SZDLL_INITDOUBLECUSTOM);
	if (pfnInit == NULL)
	{
		FreeLibrary(m_hLib);
		m_hLib = NULL;
		return ERROR_NOT_SUPPORTED;
	}

	if (lErr = (*pfnInit)(hParent, lKernelIdx))
	{
		FreeLibrary(m_hLib);
		m_hLib = NULL;
		return lErr;
	}

	m_pfn = GetProcAddress(m_hLib, SZDLL_INVOKEDOUBLECUSTOM);
	if (m_pfn == NULL)
	{
		FreeLibrary(m_hLib);
		m_hLib = NULL;
		return ERROR_NOT_SUPPORTED;
	}

	return 0;
}


template <class T>
long extensionHandle<T>::CleanUp()
{
	if (m_hLib != NULL)
	{
		FreeLibrary(m_hLib);
		m_hLib = NULL;
	}

	m_pfn = NULL;
	return 0;
}

template long extensionHandle<double>::CleanUp();
template long extensionHandle<float>::CleanUp();


// end