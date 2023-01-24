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
	LPFNINTERNAL_INVOKEFLOATEX2 m_pfnInvokeF;
	LPFNINTERNAL_INVOKEDOUBLEEX2 m_pfnInvokeD;
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

	if ((m_pfnInvokeD = (LPFNINTERNAL_INVOKEDOUBLEEX2)GetProcAddress(hParent, SZFN_INTERNAL_INVOKEDOUBLEEX2)) == NULL)
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

	if ((m_pfnInvokeF = (LPFNINTERNAL_INVOKEFLOATEX2)GetProcAddress(hParent, SZFN_INTERNAL_INVOKEFLOATEX2)) == NULL)
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
	LONGLONG rgParam[5] = { nCount, hSrc, hDst, nSrcOffset, nDstOffset };
	return (*m_pfnInvokeD)(m_lKernelIdx, CUDA_FN_COPY, NULL, 0, rgParam, 5, NULL, NULL, NULL, 0);
}

inline long Parent<float>::cuda_copy(int nCount, long hSrc, long hDst, int nSrcOffset, int nDstOffset)
{
	LONGLONG rgParam[5] = { nCount, hSrc, hDst, nSrcOffset, nDstOffset };
	return (*m_pfnInvokeF)(m_lKernelIdx, CUDA_FN_COPY, NULL, 0, rgParam, 5, NULL, NULL, NULL, 0);
}

inline long Parent<double>::cuda_add_scalar(int n, double fAlpha, long hY, int nYOff)
{
	double rgParam[4] = { fAlpha };
	LONGLONG rgParamL[4] = { n, 0, hY, nYOff };
	return (*m_pfnInvokeD)(m_lKernelIdx, CUDA_FN_ADD_SCALAR, rgParam, 1, rgParamL, 4, NULL, NULL, NULL, 0);
}

inline long Parent<float>::cuda_add_scalar(int n, float fAlpha, long hY, int nYOff)
{
	float rgParam[4] = { (float)fAlpha };
	LONGLONG rgParamL[4] = { n, 0, hY, nYOff };
	return (*m_pfnInvokeF)(m_lKernelIdx, CUDA_FN_ADD_SCALAR, rgParam, 1, rgParamL, 4, NULL, NULL, NULL, 0);
}

inline long Parent<double>::cuda_mul(int n, long hA, long hB, long hY, int nAOff, int nBOff, int nYOff)
{
	LONGLONG rgParamL[7] = { n, hA, hB, hY, nAOff, nBOff, nYOff };
	return (*m_pfnInvokeD)(m_lKernelIdx, CUDA_FN_MUL, NULL, 0, rgParamL, 7, NULL, NULL, NULL, 0);
}

inline long Parent<float>::cuda_mul(int n, long hA, long hB, long hY, int nAOff, int nBOff, int nYOff)
{
	LONGLONG rgParamL[7] = { n, hA, hB, hY, nAOff, nBOff, nYOff };
	return (*m_pfnInvokeF)(m_lKernelIdx, CUDA_FN_MUL, NULL, 0, rgParamL, 7, NULL, NULL, NULL, 0);
}

inline long Parent<double>::cuda_mul_scalar(int n, double fAlpha, long hY, int nYOff)
{
	double rgParam[4] = { (double)fAlpha };
	LONGLONG rgParamL[4] = { n, 0, hY, nYOff };
	return (*m_pfnInvokeD)(m_lKernelIdx, CUDA_FN_MUL_SCALAR, rgParam, 1, rgParamL, 4, NULL, NULL, NULL, 0);
}

inline long Parent<float>::cuda_mul_scalar(int n, float fAlpha, long hY, int nYOff)
{
	float rgParam[4] = { (float)fAlpha };
	LONGLONG rgParamL[4] = { n, 0, hY, nYOff };
	return (*m_pfnInvokeF)(m_lKernelIdx, CUDA_FN_MUL_SCALAR, rgParam, 1, rgParamL, 4, NULL, NULL, NULL, 0);
}

inline long Parent<double>::cuda_powx(int n, long hA, double fAlpha, long hY, int nAOff, int nYOff)
{
	double rgParam[6] = { fAlpha };
	LONGLONG rgParamL[6] = { n, hA, 0, hY, nAOff, nYOff };
	return (*m_pfnInvokeD)(m_lKernelIdx, CUDA_FN_POWX, rgParam, 1, rgParamL, 6, NULL, NULL, NULL, 0);
}

inline long Parent<float>::cuda_powx(int n, long hA, float fAlpha, long hY, int nAOff, int nYOff)
{
	float rgParam[6] = { fAlpha };
	LONGLONG rgParamL[6] = { n, hA, 0, hY, nAOff, nYOff };
	return (*m_pfnInvokeF)(m_lKernelIdx, CUDA_FN_POWX, rgParam, 1, rgParamL, 6, NULL, NULL, NULL, 0);
}


#endif // #ifndef __PARENT_H_
