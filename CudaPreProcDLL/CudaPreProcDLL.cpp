// CudaPreProcDLL.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include "Cuda Files\main.h"


//=============================================================================
//	Cuda Functions
//=============================================================================

const LONG MAX_BUFFER = 1024;
const LONG MAX_ERROR = 1024;


//=============================================================================
//	Cuda Exports
//=============================================================================

extern HMODULE g_hModule;
extern Kernel<double>* g_pKernelD;
extern Kernel<float>* g_pKernelF;


//=============================================================================
//	Cuda Objects
//=============================================================================

//=============================================================================
//	Local Function Prototypes
//=============================================================================

//=============================================================================
//	Main DLL Functions
//=============================================================================

extern "C" LONG WINAPI DLL_InitFloatCustomExtension(HMODULE hParent, long lKernelIdx)
{
	LONG lErr;

	if (lErr = g_pKernelF->Initialize(hParent, lKernelIdx))
	{
		delete g_pKernelF;
		g_pKernelF = NULL;
	}

	return 0;
}

extern "C" LONG WINAPI DLL_InvokeFloatCustomExtension(LONG lfnIdx, float* pInput, LONG lInput, float** ppOutput, LONG* plOutput, LPTSTR szErr, LONG lErrMax)
{
	if (g_pKernelF == NULL)
		return ERROR_INTERNAL_ERROR;

	return g_pKernelF->Run(lfnIdx, pInput, lInput, ppOutput, plOutput);
}

extern "C" LONG WINAPI DLL_InitDoubleCustomExtension(HMODULE hParent, long lKernelIdx)
{
	LONG lErr;

	if (lErr = g_pKernelD->Initialize(hParent, lKernelIdx))
	{
		delete g_pKernelD;
		g_pKernelD = NULL;
	}

	return 0;
}

extern "C" LONG WINAPI DLL_InvokeDoubleCustomExtension(LONG lfnIdx, double* pInput, LONG lInput, double** ppOutput, LONG* plOutput, LPTSTR szErr, LONG lErrMax)
{
	if (g_pKernelD == NULL)
		return ERROR_INTERNAL_ERROR;

	return g_pKernelD->Run(lfnIdx, pInput, lInput, ppOutput, plOutput);
}

//end CudaPreProcDLL.cpp

