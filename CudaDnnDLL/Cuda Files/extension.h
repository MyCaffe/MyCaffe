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

//=============================================================================
//	Classes
//=============================================================================

template <class T>
class Device;


//-----------------------------------------------------------------------------
//	EXTENSION Handle Class
//
//	This class stores the EXTENSION description information.
//-----------------------------------------------------------------------------
template <class T>
class extensionHandle
{
	HMODULE m_hLib;
	FARPROC m_pfn;

public:
	
	extensionHandle()
	{
		m_hLib = NULL;
		m_pfn = NULL;
	}

	long InitializeFloat(HMODULE hParent, LONG lKernelIdx, LPTSTR pszDllPath);
	long InitializeDouble(HMODULE hParent, LONG lKernelIdx, LPTSTR pszDllPath);
	long CleanUp();

	long Run(long lfnIdx, T* pfInput, long lCount, T** ppfOutput, long* plCount, LPTSTR pszErr, LONG lErrMax);
};


//=============================================================================
//	Inline Methods
//=============================================================================

inline long extensionHandle<float>::Run(long lfnIdx, float* pfInput, long lCount, float** ppfOutput, long* plCount, LPTSTR pszErr, LONG lErrMax)
{
	if (m_pfn == NULL)
		return ERROR_INVALID_FUNCTION;

	return (*((LPFNDLLINVOKEFLOAT)m_pfn))(lfnIdx, pfInput, lCount, ppfOutput, plCount, pszErr, lErrMax);
}

inline long extensionHandle<double>::Run(long lfnIdx, double* pfInput, long lCount, double** ppfOutput, long* plCount, LPTSTR pszErr, LONG lErrMax)
{
	if (m_pfn == NULL)
		return ERROR_INVALID_FUNCTION;

	return (*((LPFNDLLINVOKEDOUBLE)m_pfn))(lfnIdx, pfInput, lCount, ppfOutput, plCount, pszErr, lErrMax);
}

#endif