//=============================================================================
//	FILE:	extension.h
//
//	DESC:	This file manages the extension DLL's
//=============================================================================
#ifndef __EXTENSION_CU__
#define __EXTENSION_CU__

#include "util.h"
#include "math.h"
#include "memorycol.h"
#include "handlecol.h"
#include "..\inc\FunctionIDs.h"


//=============================================================================
//	Types
//=============================================================================

// Implemented by the Extension DLL
typedef LONG(WINAPI *LPFNDLLINVOKEFLOAT)(LONG lFunctionIdx,
	float* pInput, LONG lInput,
	float** ppOutput, LONG* plOutput,
	LPTSTR szErr, LONG lszErrMax);
typedef LONG(WINAPI *LPFNDLLINVOKEDOUBLE)(LONG lFunctionIdx,
	double* pInput, LONG lInput,
	double** ppOutput, LONG* plOutput,
	LPTSTR szErr, LONG lszErrMax);
typedef LONG(WINAPI* LPFNDLLINVOKEFLOATEX)(LONG lFunctionIdx,
	float* pInput, LONG lInput,
	float** ppOutput, LONG* plOutput,
	LPTSTR szInput, LPTSTR szOutput, LONG lMaxOutput,
	LPTSTR szErr, LONG lszErrMax);
typedef LONG(WINAPI* LPFNDLLINVOKEDOUBLEEX)(LONG lFunctionIdx,
	double* pInput, LONG lInput,
	double** ppOutput, LONG* plOutput,
	LPTSTR szInput, LPTSTR szOutput, LONG lMaxOutput,
	LPTSTR szErr, LONG lszErrMax);

#define BUFFER_MAX 4096

//=============================================================================
//	Classes
//=============================================================================

template <class T>
class Device;

template <class T>
class Memory;


//-----------------------------------------------------------------------------
//	EXTENSION Handle Class
//
//	This class stores the EXTENSION description information.
//-----------------------------------------------------------------------------
template <class T>
class extensionHandle
{
	LPTSTR m_pszBuffer;
	Memory<T>* m_pMem;
	HMODULE m_hLib;
	FARPROC m_pfn;
	FARPROC m_pfn2;

public:
	
	extensionHandle(Memory<T>* pMem)
	{
		m_pszBuffer = NULL;
		m_pMem = pMem;
		m_hLib = NULL;
		m_pfn = NULL;
		m_pfn2 = NULL;
	}

	long InitializeFloat(HMODULE hParent, LONG lKernelIdx, LPTSTR pszDllPath);
	long InitializeDouble(HMODULE hParent, LONG lKernelIdx, LPTSTR pszDllPath);
	long CleanUp();

	long Run(long lfnIdx, T* pfInput, long lCount, T** ppfOutput, long* plCount, LPTSTR pszErr, long lErrMax);
	long Run(long lfnIdx, T* pfInput, long lCount, LPTSTR pszInput, LPTSTR pszErr, long lErrMax);
	long Query(long lfnIdx, LONG* pInput, long lCount, LPTSTR pszOutput, long lOutputMax, LPTSTR pszErr, long lErrMax);
};


//=============================================================================
//	Inline Methods
//=============================================================================

inline long extensionHandle<float>::Run(long lfnIdx, float* pfInput, long lCount, float** ppfOutput, long* plCount, LPTSTR pszErr, long lErrMax)
{
	if (m_pfn == NULL)
		return ERROR_INVALID_FUNCTION;

	return (*((LPFNDLLINVOKEFLOAT)m_pfn))(lfnIdx, pfInput, lCount, ppfOutput, plCount, pszErr, lErrMax);
}

inline long extensionHandle<double>::Run(long lfnIdx, double* pfInput, long lCount, double** ppfOutput, long* plCount, LPTSTR pszErr, long lErrMax)
{
	if (m_pfn == NULL)
		return ERROR_INVALID_FUNCTION;

	return (*((LPFNDLLINVOKEDOUBLE)m_pfn))(lfnIdx, pfInput, lCount, ppfOutput, plCount, pszErr, lErrMax);
}

inline long extensionHandle<float>::Run(long lfnIdx, float* pfInput, long lCount, LPTSTR pszInput, LPTSTR pszErr, long lErrMax)
{
	if (m_pfn2 == NULL)
		return ERROR_NOT_IMPLEMENTED;

	return (*((LPFNDLLINVOKEFLOATEX)m_pfn2))(lfnIdx, pfInput, lCount, NULL, NULL, pszInput, NULL, 0, pszErr, lErrMax);
}

inline long extensionHandle<double>::Run(long lfnIdx, double* pfInput, long lCount, LPTSTR pszInput, LPTSTR pszErr, long lErrMax)
{
	if (m_pfn2 == NULL)
		return ERROR_NOT_IMPLEMENTED;

	return (*((LPFNDLLINVOKEDOUBLEEX)m_pfn2))(lfnIdx, pfInput, lCount, NULL, NULL, pszInput, NULL, 0, pszErr, lErrMax);
}

#endif