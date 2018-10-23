//=============================================================================
//	parent.h
//
//	The parent manages all communications back to the CudaDnnDll
//=============================================================================
#ifndef __PARENT_H_
#define __PARENT_H_

//=============================================================================
//	Includes
//=============================================================================

#include "FunctionIDs.h"

//=============================================================================
//	Defines
//=============================================================================

//=============================================================================
//	Typedefs
//=============================================================================

//=============================================================================
//	Parent Classses
//=============================================================================

template <class T>
class Parent
{
	LONG m_lKernelIdx;
	HMODULE m_hParent;
	LPFNINTERNAL_INVOKEFLOAT m_pfnInvokeF;
	LPFNINTERNAL_INVOKEDOUBLE m_pfnInvokeD;
	LPFNINTERNAL_ALLOCHOSTFLOAT m_pfnAllocHostF;
	LPFNINTERNAL_ALLOCHOSTDOUBLE m_pfnAllocHostD;
	LPFNINTERNAL_GETPTRFLOAT m_pfnGetPtrF;
	LPFNINTERNAL_GETPTRDOUBLE m_pfnGetPtrD;

public:
	Parent()
	{
		m_lKernelIdx = 0;
		m_hParent = NULL;
		m_pfnInvokeF = NULL;
		m_pfnInvokeD = NULL;
		m_pfnAllocHostF = NULL;
		m_pfnAllocHostD = NULL;
		m_pfnGetPtrF = NULL;
		m_pfnGetPtrD = NULL;
	}

	~Parent()
	{
		CleanUp();
	}

	long Initialize(HMODULE hParent, long lKernelIdx);

	void CleanUp()
	{
		m_lKernelIdx = 0;
		m_hParent = NULL;
		m_pfnInvokeF = NULL;
		m_pfnInvokeD = NULL;
		m_pfnAllocHostF = NULL;
		m_pfnAllocHostD = NULL;
		m_pfnGetPtrF = NULL;
		m_pfnGetPtrD = NULL;
	}

	long GetMemoryPointer(long hHandle, void** ppPtr);

	long cuda_copy(int nCount, long hSrc, long hDst, int nSrcOffset, int nDstOffset);
	long cuda_add_scalar(int n, T fAlpha, long hY, int nYOff);
	long cuda_mul(int n, long hA, long hB, long hY, int nAOff, int nBOff, int nYOff);
	long cuda_mul_scalar(int n, T fAlpha, long hY, int nYOff);
	long cuda_powx(int n, long hA, T fAlpha, long hY, int nAOff, int nYOff);
};

//=============================================================================
//	Inline Methods
//=============================================================================

inline long Parent<double>::Initialize(HMODULE hParent, long lKernelIdx)
{
	m_lKernelIdx = lKernelIdx;
	m_hParent = hParent;

	if ((m_pfnInvokeD = (LPFNINTERNAL_INVOKEDOUBLE)GetProcAddress(hParent, SZFN_INTERNAL_INVOKEDOUBLE)) == NULL)
		return ERROR_INVALID_PARAMETER;

	if ((m_pfnAllocHostD = (LPFNINTERNAL_ALLOCHOSTDOUBLE)GetProcAddress(hParent, SZFN_INTERNAL_ALLOCHOSTDBL)) == NULL)
		return ERROR_INVALID_PARAMETER;

	if ((m_pfnGetPtrD = (LPFNINTERNAL_GETPTRDOUBLE)GetProcAddress(hParent, SZFN_INTERNAL_GETPOINTERDBL)) == NULL)
		return ERROR_INVALID_PARAMETER;

	return 0;
}

inline long Parent<float>::Initialize(HMODULE hParent, long lKernelIdx)
{
	m_lKernelIdx = lKernelIdx;
	m_hParent = hParent;

	if ((m_pfnInvokeF = (LPFNINTERNAL_INVOKEFLOAT)GetProcAddress(hParent, SZFN_INTERNAL_INVOKEFLOAT)) == NULL)
		return ERROR_INVALID_PARAMETER;

	if ((m_pfnAllocHostF = (LPFNINTERNAL_ALLOCHOSTFLOAT)GetProcAddress(hParent, SZFN_INTERNAL_ALLOCHOSTFLT)) == NULL)
		return ERROR_INVALID_PARAMETER;

	if ((m_pfnGetPtrF = (LPFNINTERNAL_GETPTRFLOAT)GetProcAddress(hParent, SZFN_INTERNAL_GETPOINTERFLT)) == NULL)
		return ERROR_INVALID_PARAMETER;

	return 0;
}

inline long Parent<double>::GetMemoryPointer(long hHandle, void** ppPtr)
{	
	return (*m_pfnGetPtrD)(HT_MEMORY, m_lKernelIdx, hHandle, ppPtr);
}

inline long Parent<float>::GetMemoryPointer(long hHandle, void** ppPtr)
{
	return (*m_pfnGetPtrF)(HT_MEMORY, m_lKernelIdx, hHandle, ppPtr);
}

inline long Parent<double>::cuda_copy(int nCount, long hSrc, long hDst, int nSrcOffset, int nDstOffset)
{
	double rgParam[5] = { (double)nCount, (double)hSrc, (double)hDst, (double)nSrcOffset, (double)nDstOffset };
	return (*m_pfnInvokeD)(m_lKernelIdx, CUDA_FN_COPY, rgParam, 5, NULL, NULL, NULL, 0);
}

inline long Parent<float>::cuda_copy(int nCount, long hSrc, long hDst, int nSrcOffset, int nDstOffset)
{
	float rgParam[5] = { (float)nCount, (float)hSrc, (float)hDst, (float)nSrcOffset, (float)nDstOffset };
	return (*m_pfnInvokeF)(m_lKernelIdx, CUDA_FN_COPY, rgParam, 5, NULL, NULL, NULL, 0);
}

inline long Parent<double>::cuda_add_scalar(int n, double fAlpha, long hY, int nYOff)
{
	double rgParam[4] = { (double)n, (double)fAlpha, (double)hY, (double)nYOff };
	return (*m_pfnInvokeD)(m_lKernelIdx, CUDA_FN_ADD_SCALAR, rgParam, 4, NULL, NULL, NULL, 0);
}

inline long Parent<float>::cuda_add_scalar(int n, float fAlpha, long hY, int nYOff)
{
	float rgParam[4] = { (float)n, (float)fAlpha, (float)hY, (float)nYOff };
	return (*m_pfnInvokeF)(m_lKernelIdx, CUDA_FN_ADD_SCALAR, rgParam, 4, NULL, NULL, NULL, 0);
}

inline long Parent<double>::cuda_mul(int n, long hA, long hB, long hY, int nAOff, int nBOff, int nYOff)
{
	double rgParam[7] = { (double)n, (double)hA, (double)hB, (double)hY, (double)nAOff, (double)nBOff, (double)nYOff };
	return (*m_pfnInvokeD)(m_lKernelIdx, CUDA_FN_MUL, rgParam, 7, NULL, NULL, NULL, 0);
}

inline long Parent<float>::cuda_mul(int n, long hA, long hB, long hY, int nAOff, int nBOff, int nYOff)
{
	float rgParam[7] = { (float)n, (float)hA, (float)hB, (float)hY, (float)nAOff, (float)nBOff, (float)nYOff };
	return (*m_pfnInvokeF)(m_lKernelIdx, CUDA_FN_MUL, rgParam, 7, NULL, NULL, NULL, 0);
}

inline long Parent<double>::cuda_mul_scalar(int n, double fAlpha, long hY, int nYOff)
{
	double rgParam[4] = { (double)n, (double)fAlpha, (double)hY, (double)nYOff };
	return (*m_pfnInvokeD)(m_lKernelIdx, CUDA_FN_MUL_SCALAR, rgParam, 4, NULL, NULL, NULL, 0);
}

inline long Parent<float>::cuda_mul_scalar(int n, float fAlpha, long hY, int nYOff)
{
	float rgParam[4] = { (float)n, (float)fAlpha, (float)hY, (float)nYOff };
	return (*m_pfnInvokeF)(m_lKernelIdx, CUDA_FN_MUL_SCALAR, rgParam, 4, NULL, NULL, NULL, 0);
}

inline long Parent<double>::cuda_powx(int n, long hA, double fAlpha, long hY, int nAOff, int nYOff)
{
	double rgParam[6] = { (double)n, (double)hA, fAlpha, (double)hY, (double)nAOff, (double)nYOff };
	return (*m_pfnInvokeD)(m_lKernelIdx, CUDA_FN_POWX, rgParam, 6, NULL, NULL, NULL, 0);
}

inline long Parent<float>::cuda_powx(int n, long hA, float fAlpha, long hY, int nAOff, int nYOff)
{
	float rgParam[6] = { (float)n, (float)hA, fAlpha, (float)hY, (float)nAOff, (float)nYOff };
	return (*m_pfnInvokeF)(m_lKernelIdx, CUDA_FN_POWX, rgParam, 6, NULL, NULL, NULL, 0);
}


#endif // #ifndef __PARENT_H_
