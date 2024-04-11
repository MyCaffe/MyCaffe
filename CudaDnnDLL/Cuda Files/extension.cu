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
#define SZDLL_INVOKEFLOATCUSTOMEX "DLL_InvokeFloatCustomExtensionEx"

#define SZDLL_INITDOUBLECUSTOM "DLL_InitDoubleCustomExtension"
#define SZDLL_INVOKEDOUBLECUSTOM "DLL_InvokeDoubleCustomExtension"
#define SZDLL_INVOKEDOUBLECUSTOMEX "DLL_InvokeDoubleCustomExtensionEx"

typedef LONG(WINAPI *LPFNDLLINITCUSTOM)(HMODULE hParent, long lKernelIdx);


//=============================================================================
//	Class Methods
//=============================================================================

long extensionHandle<float>::InitializeFloat(HMODULE hParent, LONG lKernelIdx, LPTSTR pszDllPath)
{
	long lErr;

	m_pszBuffer = (LPTSTR)malloc(BUFFER_MAX * sizeof(TCHAR));
	if (m_pszBuffer == NULL)
		return ERROR_OUTOFMEMORY;

	memset(m_pszBuffer, 0, BUFFER_MAX * sizeof(TCHAR));

	m_hLib = LoadLibrary(pszDllPath);
	if (m_hLib == NULL)
	{
		free(m_pszBuffer);
		return GetLastError();
	}

	LPFNDLLINITCUSTOM pfnInit = (LPFNDLLINITCUSTOM)GetProcAddress(m_hLib, SZDLL_INITFLOATCUSTOM);
	if (pfnInit == NULL)
	{
		free(m_pszBuffer);
		FreeLibrary(m_hLib);
		m_hLib = NULL;
		return ERROR_NOT_SUPPORTED;
	}

	if (lErr = (*pfnInit)(hParent, lKernelIdx))
	{
		free(m_pszBuffer);
		FreeLibrary(m_hLib);
		m_hLib = NULL;
		return lErr;
	}

	m_pfn = GetProcAddress(m_hLib, SZDLL_INVOKEFLOATCUSTOM);
	if (m_pfn == NULL)
	{
		free(m_pszBuffer);
		FreeLibrary(m_hLib);
		m_hLib = NULL;
		return ERROR_NOT_SUPPORTED;
	}

	m_pfn2 = GetProcAddress(m_hLib, SZDLL_INVOKEFLOATCUSTOMEX);

	return 0;
}

long extensionHandle<double>::InitializeDouble(HMODULE hParent, LONG lKernelIdx, LPTSTR pszDllPath)
{
	long lErr;

	m_pszBuffer = (LPTSTR)malloc(BUFFER_MAX * sizeof(TCHAR));
	if (m_pszBuffer == NULL)
		return ERROR_OUTOFMEMORY;

	memset(m_pszBuffer, 0, BUFFER_MAX * sizeof(TCHAR));

	m_hLib = LoadLibrary(pszDllPath);
	if (m_hLib == NULL)
	{
		free(m_pszBuffer);
		return GetLastError();
	}

	LPFNDLLINITCUSTOM pfnInit = (LPFNDLLINITCUSTOM)GetProcAddress(m_hLib, SZDLL_INITDOUBLECUSTOM);
	if (pfnInit == NULL)
	{
		free(m_pszBuffer);
		FreeLibrary(m_hLib);
		m_hLib = NULL;
		return ERROR_NOT_SUPPORTED;
	}

	if (lErr = (*pfnInit)(hParent, lKernelIdx))
	{
		free(m_pszBuffer);
		FreeLibrary(m_hLib);
		m_hLib = NULL;
		return lErr;
	}

	m_pfn = GetProcAddress(m_hLib, SZDLL_INVOKEDOUBLECUSTOM);
	if (m_pfn == NULL)
	{
		free(m_pszBuffer);
		FreeLibrary(m_hLib);
		m_hLib = NULL;
		return ERROR_NOT_SUPPORTED;
	}

	m_pfn2 = GetProcAddress(m_hLib, SZDLL_INVOKEDOUBLECUSTOMEX);

	return 0;
}


template <class T>
long extensionHandle<T>::CleanUp()
{
	if (m_pszBuffer != NULL)
	{
		free(m_pszBuffer);
		m_pszBuffer = NULL;
	}

	if (m_hLib != NULL)
	{
		FreeLibrary(m_hLib);
		m_hLib = NULL;
	}

	m_pfn = NULL;
	m_pfn2 = NULL;
	return 0;
}

template long extensionHandle<double>::CleanUp();
template long extensionHandle<float>::CleanUp();


long extensionHandle<float>::Query(long lfnIdx, LONG* pInput, long lCount, LPTSTR szOutput, long lOutputMax, LPTSTR pszErr, long lErrMax)
{
	LONG lErr;

	if (m_pfn2 == NULL)
		return ERROR_NOT_IMPLEMENTED;

	if (lCount > 5)
		return ERROR_PARAM_OUT_OF_RANGE;

	float rgArg[5];
	for (int i = 0; i < lCount && i < 5; i++)
	{
		rgArg[i] = (float)pInput[i];
	}

	float* pOutput = NULL;
	long lOutput;

	if (lErr = (*((LPFNDLLINVOKEFLOATEX)m_pfn2))(lfnIdx, rgArg, lCount, &pOutput, &lOutput, NULL, szOutput, lOutputMax, pszErr, lErrMax))
		return lErr;

	if (pOutput != NULL)
		_tcsncat(szOutput, _T("\n[END]"), lOutputMax);

	return 0;
}

long extensionHandle<double>::Query(long lfnIdx, LONG* pInput, long lCount, LPTSTR szOutput, long lOutputMax, LPTSTR pszErr, long lErrMax)
{
	LONG lErr;

	if (m_pfn2 == NULL)
		return ERROR_NOT_IMPLEMENTED;

	if (lCount > 5)
		return ERROR_PARAM_OUT_OF_RANGE;

	double rgArg[5];
	for (int i = 0; i < lCount && i < 5; i++)
	{
		rgArg[i] = (float)pInput[i];
	}

	double* pOutput = NULL;
	long lOutput;

	if (lErr = (*((LPFNDLLINVOKEDOUBLEEX)m_pfn2))(lfnIdx, rgArg, lCount, &pOutput, &lOutput, NULL, szOutput, lOutputMax, pszErr, lErrMax))
		return lErr;

	if (pOutput != NULL)
		_tcsncat(szOutput, _T("\n[END]"), lOutputMax);

	return 0;
}
// end