// CudaDnnDLL.cpp : Defines the exported functions for the DLL application.
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
extern Kernel<double>* g_rgdwDoubleKernelTable[];
extern Kernel<float>* g_rgdwFloatKernelTable[];
extern DWORD g_dwMaxKernelCount;
extern DWORD g_dwLastKernelDoubleIndex;
extern DWORD g_dwLastKernelFloatIndex;
extern CRITICAL_SECTION g_DoubleKernelTableLock;
extern bool m_bDoubleKernelTableLockInit;
extern CRITICAL_SECTION g_FloatKernelTableLock;
extern bool m_bFloatKernelTableLockInit;

//=============================================================================
//	Cuda Objects
//=============================================================================

//=============================================================================
//	Local Function Prototypes
//=============================================================================

void setCurrentDirectory();

LONG getKernelDoubleIndex(LONG* plIdx);
LONG getKernelFloatIndex(LONG* plIdx);

LONG addKernelToKernel(Kernel<float>* pKernel, LONG lInput, float* pInput);
LONG copyMemKernelToKernel(Kernel<float>* pKernel, LONG lInput, float* pInput);

LONG addKernelToKernel(Kernel<double>* pKernel, LONG lInput, double* pInput);
LONG copyMemKernelToKernel(Kernel<double>* pKernel, LONG lInput, double* pInput);

void getError(long lKernel, long lErr, LPTSTR szErr, LONG lszErrMax);


//=============================================================================
//	Internal DLL Functions used by extensions.
//=============================================================================

extern "C" LONG WINAPI Internal_GetPointerFloat(HANDLE_TYPE ht, LONG lKernelIdx, LONG hHandle, void** ppPtr)
{
	Kernel<float>* pKernel = NULL;
	LONG lErr = 0;


	if (lKernelIdx < 0 || lKernelIdx >= (LONG)g_dwMaxKernelCount)
		return ERROR_PARAM_OUT_OF_RANGE;

	if ((pKernel = g_rgdwFloatKernelTable[lKernelIdx]) == NULL)
	{
		lErr = ERROR_PARAM_NULL;
		return lErr;
	}

	return pKernel->GetPointer(ht, hHandle, ppPtr);
}

extern "C" LONG WINAPI Internal_GetPointerDouble(HANDLE_TYPE ht, LONG lKernelIdx, LONG hHandle, void** ppPtr)
{
	Kernel<double>* pKernel = NULL;
	LONG lErr = 0;


	if (lKernelIdx < 0 || lKernelIdx >= (LONG)g_dwMaxKernelCount)
		return ERROR_PARAM_OUT_OF_RANGE;

	if ((pKernel = g_rgdwDoubleKernelTable[lKernelIdx]) == NULL)
	{
		lErr = ERROR_PARAM_NULL;
		return lErr;
	}

	return pKernel->GetPointer(ht, hHandle, ppPtr);
}

extern "C" LONG WINAPI Internal_AllocHostFloat(LONG lKernelIdx, LONG lCount, float** ppfDst, float* pfSrc, bool bSrcOnDevice)
{
	Kernel<float>* pKernel = NULL;
	LONG lErr = 0;


	if (lKernelIdx < 0 || lKernelIdx >= (LONG)g_dwMaxKernelCount)
		return ERROR_PARAM_OUT_OF_RANGE;

	if ((pKernel = g_rgdwFloatKernelTable[lKernelIdx]) == NULL)
	{
		lErr = ERROR_PARAM_NULL;
		return lErr;
	}

	return pKernel->AllocHost(lCount, ppfDst, pfSrc, bSrcOnDevice);
}

extern "C" LONG WINAPI Internal_AllocHostDouble(LONG lKernelIdx, LONG lCount, double** ppfDst, double* pfSrc, bool bSrcOnDevice)
{
	Kernel<double>* pKernel = NULL;
	LONG lErr = 0;


	if (lKernelIdx < 0 || lKernelIdx >= (LONG)g_dwMaxKernelCount)
		return ERROR_PARAM_OUT_OF_RANGE;

	if ((pKernel = g_rgdwDoubleKernelTable[lKernelIdx]) == NULL)
	{
		lErr = ERROR_PARAM_NULL;
		return lErr;
	}

	return pKernel->AllocHost(lCount, ppfDst, pfSrc, bSrcOnDevice);
}


//=============================================================================
//	Main DLL Functions
//=============================================================================

extern "C" LONG WINAPI DLL_InvokeFloat(LONG lKernelIdx,
									   LONG lFunctionIdx,
		   							   float* pInput, LONG lInput,
							           float** ppOutput, LONG* plOutput,
								       LPTSTR szErr, LONG lszErrMax)
{
	Kernel<float>* pKernel = NULL;
	LONG lErr = 0;


	if (lKernelIdx < 0 || lKernelIdx >= (LONG)g_dwMaxKernelCount)
		return ERROR_PARAM_OUT_OF_RANGE;


	//-------------------------------------------
	//	Process the requested function.
	//-------------------------------------------

	switch (lFunctionIdx)
	{
		case CUDA_DLL_INITIALIZE:
			setCurrentDirectory();

			if (lErr = getKernelFloatIndex(&lKernelIdx))
			{
				getError(lKernelIdx, lErr, szErr, lszErrMax);
				return lErr;
			}

			pKernel = new Kernel<float>();

			if (lErr = pKernel->Initialize(pInput, lInput))
			{
				delete pKernel;
				getError(lKernelIdx, lErr, szErr, lszErrMax);
				return lErr;
			}

			g_rgdwFloatKernelTable[lKernelIdx] = pKernel;

			(*ppOutput)[0] = (float)lKernelIdx;
			*plOutput = 1;
			break;

		case CUDA_DLL_CLEANUP:
			pKernel = g_rgdwFloatKernelTable[lKernelIdx];
			if (pKernel != NULL)
			{
				pKernel->CleanUp();
				delete pKernel;
				g_rgdwFloatKernelTable[lKernelIdx] = NULL;
			}
			break;

		case CUDA_DLL_FREEMEM:
			if ((pKernel = g_rgdwFloatKernelTable[lKernelIdx]) == NULL)
			{
				lErr = ERROR_PARAM_NULL;
				getError(lKernelIdx, lErr, szErr, lszErrMax);
				return lErr;
			}

			if (lErr = pKernel->FreeHost(pInput))
			{
				getError(lKernelIdx, lErr, szErr, lszErrMax);
				return lErr;
			}
			break;

		case CUDA_DLL_KERNEL_MEMCPY:
			if ((pKernel = g_rgdwFloatKernelTable[lKernelIdx]) == NULL)
			{
				lErr = ERROR_PARAM_NULL;
				getError(lKernelIdx, lErr, szErr, lszErrMax);
				return lErr;
			}

			if (lErr = copyMemKernelToKernel(pKernel, lInput, pInput))
			{
				getError(lKernelIdx, lErr, szErr, lszErrMax);
				return lErr;
			}
			break;

		case CUDA_DLL_KERNEL_ADD:
			if ((pKernel = g_rgdwFloatKernelTable[lKernelIdx]) == NULL)
			{
				lErr = ERROR_PARAM_NULL;
				getError(lKernelIdx, lErr, szErr, lszErrMax);
				return lErr;
			}

			if (lErr = addKernelToKernel(pKernel, lInput, pInput))
			{
				getError(lKernelIdx, lErr, szErr, lszErrMax);
				return lErr;
			}
			break;

		case CUDA_DLL_KERNEL_COPY_NCCL:
			{
				if ((pKernel = g_rgdwFloatKernelTable[lKernelIdx]) == NULL)
				{
					lErr = ERROR_PARAM_NULL;
					getError(lKernelIdx, lErr, szErr, lszErrMax);
					return lErr;
				}

				if (lInput != 2)
				{
					return ERROR_PARAM_OUT_OF_RANGE;
					getError(lKernelIdx, lErr, szErr, lszErrMax);
					return lErr;
				}

				long hKernelSrc = (long)pInput[0];
				long hNcclSrc = (long)pInput[1];

				if (hKernelSrc == 0 || hNcclSrc == 0)
				{
					return ERROR_PARAM_OUT_OF_RANGE;
					getError(lKernelIdx, lErr, szErr, lszErrMax);
					return lErr;
				}

				Kernel<float>* pKernelSrc = NULL;
				if ((pKernelSrc = g_rgdwFloatKernelTable[hKernelSrc]) == NULL)
				{
					lErr = ERROR_PARAM_NULL;
					getError(lKernelIdx, lErr, szErr, lszErrMax);
					return lErr;
				}

				ncclHandle<float>* pNccl = NULL;
				if ((pNccl = pKernelSrc->GetNCCL(hNcclSrc)) == NULL)
				{
					lErr = ERROR_PARAM_NULL;
					getError(lKernelIdx, lErr, szErr, lszErrMax);
					return lErr;
				}

				if (lErr = pKernel->SetNCCL(pNccl, ppOutput, plOutput))
				{
					getError(lKernelIdx, lErr, szErr, lszErrMax);
					return lErr;
				}
			}
			break;

		case CUDA_FN_EXTENSION_RUN:
			if ((pKernel = g_rgdwFloatKernelTable[lKernelIdx]) == NULL)
			{
				lErr = ERROR_PARAM_NULL;
				getError(lKernelIdx, lErr, szErr, lszErrMax);
				return lErr;
			}

			if (lErr = pKernel->Run(lFunctionIdx, pInput, lInput, ppOutput, plOutput, szErr, lszErrMax))
				return lErr;
			break;

		default:
			if ((pKernel = g_rgdwFloatKernelTable[lKernelIdx]) == NULL)
			{
				lErr = ERROR_PARAM_NULL;
				getError(lKernelIdx, lErr, szErr, lszErrMax);
				return lErr;
			}

			if (lErr = pKernel->Run(lFunctionIdx, pInput, lInput, NULL, 0, ppOutput, plOutput))
			{
				getError(lKernelIdx, lErr, szErr, lszErrMax);
				return lErr;
			}
			break;
	}

	return lErr;
}

extern "C" LONG WINAPI DLL_InvokeFloatEx2(LONG lKernelIdx,
	LONG lFunctionIdx,
	float* pInput, LONG lInput,
	LONGLONG* plInput, LONG llInput,
	float** ppOutput, LONG * plOutput,
	LPTSTR szErr, LONG lszErrMax)
{
	Kernel<float>* pKernel = NULL;
	LONG lErr = 0;


	if (lKernelIdx < 0 || lKernelIdx >= (LONG)g_dwMaxKernelCount)
		return ERROR_PARAM_OUT_OF_RANGE;


	//-------------------------------------------
	//	Process the requested function.
	//-------------------------------------------

	switch (lFunctionIdx)
	{
	case CUDA_DLL_INITIALIZE:
		setCurrentDirectory();

		if (lErr = getKernelFloatIndex(&lKernelIdx))
		{
			getError(lKernelIdx, lErr, szErr, lszErrMax);
			return lErr;
		}

		pKernel = new Kernel<float>();

		if (lErr = pKernel->Initialize(pInput, lInput))
		{
			delete pKernel;
			getError(lKernelIdx, lErr, szErr, lszErrMax);
			return lErr;
		}

		g_rgdwFloatKernelTable[lKernelIdx] = pKernel;

		(*ppOutput)[0] = (float)lKernelIdx;
		*plOutput = 1;
		break;

	case CUDA_DLL_CLEANUP:
		pKernel = g_rgdwFloatKernelTable[lKernelIdx];
		if (pKernel != NULL)
		{
			pKernel->CleanUp();
			delete pKernel;
			g_rgdwFloatKernelTable[lKernelIdx] = NULL;
		}
		break;

	case CUDA_DLL_FREEMEM:
		if ((pKernel = g_rgdwFloatKernelTable[lKernelIdx]) == NULL)
		{
			lErr = ERROR_PARAM_NULL;
			getError(lKernelIdx, lErr, szErr, lszErrMax);
			return lErr;
		}

		if (lErr = pKernel->FreeHost(pInput))
		{
			getError(lKernelIdx, lErr, szErr, lszErrMax);
			return lErr;
		}
		break;

	case CUDA_DLL_KERNEL_MEMCPY:
		if ((pKernel = g_rgdwFloatKernelTable[lKernelIdx]) == NULL)
		{
			lErr = ERROR_PARAM_NULL;
			getError(lKernelIdx, lErr, szErr, lszErrMax);
			return lErr;
		}

		if (lErr = copyMemKernelToKernel(pKernel, lInput, pInput))
		{
			getError(lKernelIdx, lErr, szErr, lszErrMax);
			return lErr;
		}
		break;

	case CUDA_DLL_KERNEL_ADD:
		if ((pKernel = g_rgdwFloatKernelTable[lKernelIdx]) == NULL)
		{
			lErr = ERROR_PARAM_NULL;
			getError(lKernelIdx, lErr, szErr, lszErrMax);
			return lErr;
		}

		if (lErr = addKernelToKernel(pKernel, lInput, pInput))
		{
			getError(lKernelIdx, lErr, szErr, lszErrMax);
			return lErr;
		}
		break;

	case CUDA_DLL_KERNEL_COPY_NCCL:
	{
		if ((pKernel = g_rgdwFloatKernelTable[lKernelIdx]) == NULL)
		{
			lErr = ERROR_PARAM_NULL;
			getError(lKernelIdx, lErr, szErr, lszErrMax);
			return lErr;
		}

		if (lInput != 2)
		{
			return ERROR_PARAM_OUT_OF_RANGE;
			getError(lKernelIdx, lErr, szErr, lszErrMax);
			return lErr;
		}

		long hKernelSrc = (long)pInput[0];
		long hNcclSrc = (long)pInput[1];

		if (hKernelSrc == 0 || hNcclSrc == 0)
		{
			return ERROR_PARAM_OUT_OF_RANGE;
			getError(lKernelIdx, lErr, szErr, lszErrMax);
			return lErr;
		}

		Kernel<float>* pKernelSrc = NULL;
		if ((pKernelSrc = g_rgdwFloatKernelTable[hKernelSrc]) == NULL)
		{
			lErr = ERROR_PARAM_NULL;
			getError(lKernelIdx, lErr, szErr, lszErrMax);
			return lErr;
		}

		ncclHandle<float>* pNccl = NULL;
		if ((pNccl = pKernelSrc->GetNCCL(hNcclSrc)) == NULL)
		{
			lErr = ERROR_PARAM_NULL;
			getError(lKernelIdx, lErr, szErr, lszErrMax);
			return lErr;
		}

		if (lErr = pKernel->SetNCCL(pNccl, ppOutput, plOutput))
		{
			getError(lKernelIdx, lErr, szErr, lszErrMax);
			return lErr;
		}
	}
	break;

	case CUDA_FN_EXTENSION_RUN:
		if ((pKernel = g_rgdwFloatKernelTable[lKernelIdx]) == NULL)
		{
			lErr = ERROR_PARAM_NULL;
			getError(lKernelIdx, lErr, szErr, lszErrMax);
			return lErr;
		}

		if (lErr = pKernel->Run(lFunctionIdx, pInput, lInput, ppOutput, plOutput, szErr, lszErrMax))
			return lErr;
		break;

	default:
		if ((pKernel = g_rgdwFloatKernelTable[lKernelIdx]) == NULL)
		{
			lErr = ERROR_PARAM_NULL;
			getError(lKernelIdx, lErr, szErr, lszErrMax);
			return lErr;
		}

		if (lErr = pKernel->Run(lFunctionIdx, pInput, lInput, plInput, llInput, ppOutput, plOutput))
		{
			getError(lKernelIdx, lErr, szErr, lszErrMax);
			return lErr;
		}
		break;
	}

	return lErr;
}

extern "C" LONG WINAPI DLL_InvokeDouble(LONG lKernelIdx,
										LONG lFunctionIdx,
		   							    double* pInput, LONG lInput,
							            double** ppOutput, LONG* plOutput,
								        LPTSTR szErr, LONG lszErrMax)
{
	Kernel<double>* pKernel = NULL;
	LONG lErr = 0;

	if (lKernelIdx < 0 || lKernelIdx >= (LONG)g_dwMaxKernelCount)
		return ERROR_PARAM_OUT_OF_RANGE;


	//-------------------------------------------
	//	Process the requested function.
	//-------------------------------------------

	switch (lFunctionIdx)
	{
		case CUDA_DLL_INITIALIZE:
			setCurrentDirectory();

			if (lErr = getKernelDoubleIndex(&lKernelIdx))
			{
				getError(lKernelIdx, lErr, szErr, lszErrMax);
				return lErr;
			}

			pKernel = new Kernel<double>();

			if (lErr = pKernel->Initialize(pInput, lInput))
			{
				delete pKernel;
				getError(lKernelIdx, lErr, szErr, lszErrMax);
				return lErr;
			}

			g_rgdwDoubleKernelTable[lKernelIdx] = pKernel;

			(*ppOutput)[0] = (double)lKernelIdx;
			*plOutput = 1;
			break;

		case CUDA_DLL_CLEANUP:
			pKernel = g_rgdwDoubleKernelTable[lKernelIdx];
			if (pKernel != NULL)
			{
				pKernel->CleanUp();
				delete pKernel;
				g_rgdwDoubleKernelTable[lKernelIdx] = NULL;
			}
			break;

		case CUDA_DLL_FREEMEM:
			if ((pKernel = g_rgdwDoubleKernelTable[lKernelIdx]) == NULL)
			{
				lErr = ERROR_PARAM_NULL;
				getError(lKernelIdx, lErr, szErr, lszErrMax);
				return lErr;
			}

			if (lErr = pKernel->FreeHost(pInput))
			{
				getError(lKernelIdx, lErr, szErr, lszErrMax);
				return lErr;
			}
			break;

		case CUDA_DLL_KERNEL_MEMCPY:
			if ((pKernel = g_rgdwDoubleKernelTable[lKernelIdx]) == NULL)
			{
				lErr = ERROR_PARAM_NULL;
				getError(lKernelIdx, lErr, szErr, lszErrMax);
				return lErr;
			}

			if (lErr = copyMemKernelToKernel(pKernel, lInput, pInput))
			{
				getError(lKernelIdx, lErr, szErr, lszErrMax);
				return lErr;
			}
			break;

		case CUDA_DLL_KERNEL_ADD:
			if ((pKernel = g_rgdwDoubleKernelTable[lKernelIdx]) == NULL)
			{
				lErr = ERROR_PARAM_NULL;
				getError(lKernelIdx, lErr, szErr, lszErrMax);
				return lErr;
			}

			if (lErr = addKernelToKernel(pKernel, lInput, pInput))
			{
				getError(lKernelIdx, lErr, szErr, lszErrMax);
				return lErr;
			}
			break;

		case CUDA_DLL_KERNEL_COPY_NCCL:
			{
				if ((pKernel = g_rgdwDoubleKernelTable[lKernelIdx]) == NULL)
				{
					lErr = ERROR_PARAM_NULL;
					getError(lKernelIdx, lErr, szErr, lszErrMax);
					return lErr;
				}

				if (lInput != 2)
				{
					return ERROR_PARAM_OUT_OF_RANGE;
					getError(lKernelIdx, lErr, szErr, lszErrMax);
					return lErr;
				}

				long hKernelSrc = (long)pInput[0];
				long hNcclSrc = (long)pInput[1];

				if (hKernelSrc == 0 || hNcclSrc == 0)
				{
					return ERROR_PARAM_OUT_OF_RANGE;
					getError(lKernelIdx, lErr, szErr, lszErrMax);
					return lErr;
				}

				Kernel<double>* pKernelSrc = NULL;
				if ((pKernelSrc = g_rgdwDoubleKernelTable[hKernelSrc]) == NULL)
				{
					lErr = ERROR_PARAM_NULL;
					getError(lKernelIdx, lErr, szErr, lszErrMax);
					return lErr;
				}

				ncclHandle<double>* pNccl = NULL;
				if ((pNccl = pKernelSrc->GetNCCL(hNcclSrc)) == NULL)
				{
					lErr = ERROR_PARAM_NULL;
					getError(lKernelIdx, lErr, szErr, lszErrMax);
					return lErr;
				}

				if (lErr = pKernel->SetNCCL(pNccl, ppOutput, plOutput))
				{
					getError(lKernelIdx, lErr, szErr, lszErrMax);
					return lErr;
				}
			}
			break;

		case CUDA_FN_EXTENSION_RUN:
			if ((pKernel = g_rgdwDoubleKernelTable[lKernelIdx]) == NULL)
			{
				lErr = ERROR_PARAM_NULL;
				getError(lKernelIdx, lErr, szErr, lszErrMax);
				return lErr;
			}

			if (lErr = pKernel->Run(lFunctionIdx, pInput, lInput, ppOutput, plOutput, szErr, lszErrMax))
				return lErr;
			break;

		default:
			if ((pKernel = g_rgdwDoubleKernelTable[lKernelIdx]) == NULL)
			{
				lErr = ERROR_PARAM_NULL;
				getError(lKernelIdx, lErr, szErr, lszErrMax);
				return lErr;
			}

			if (lErr = pKernel->Run(lFunctionIdx, pInput, lInput, NULL, 0, ppOutput, plOutput))
			{
				getError(lKernelIdx, lErr, szErr, lszErrMax);
				return lErr;
			}
			break;
	}

	return lErr;
}

extern "C" LONG WINAPI DLL_InvokeDoubleEx2(LONG lKernelIdx,
	LONG lFunctionIdx,
	double* pInput, LONG lInput,
	LONGLONG* plInput, LONG llInput,
	double** ppOutput, LONG * plOutput,
	LPTSTR szErr, LONG lszErrMax)
{
	Kernel<double>* pKernel = NULL;
	LONG lErr = 0;

	if (lKernelIdx < 0 || lKernelIdx >= (LONG)g_dwMaxKernelCount)
		return ERROR_PARAM_OUT_OF_RANGE;


	//-------------------------------------------
	//	Process the requested function.
	//-------------------------------------------

	switch (lFunctionIdx)
	{
	case CUDA_DLL_INITIALIZE:
		setCurrentDirectory();

		if (lErr = getKernelDoubleIndex(&lKernelIdx))
		{
			getError(lKernelIdx, lErr, szErr, lszErrMax);
			return lErr;
		}

		pKernel = new Kernel<double>();

		if (lErr = pKernel->Initialize(pInput, lInput))
		{
			delete pKernel;
			getError(lKernelIdx, lErr, szErr, lszErrMax);
			return lErr;
		}

		g_rgdwDoubleKernelTable[lKernelIdx] = pKernel;

		(*ppOutput)[0] = (double)lKernelIdx;
		*plOutput = 1;
		break;

	case CUDA_DLL_CLEANUP:
		pKernel = g_rgdwDoubleKernelTable[lKernelIdx];
		if (pKernel != NULL)
		{
			pKernel->CleanUp();
			delete pKernel;
			g_rgdwDoubleKernelTable[lKernelIdx] = NULL;
		}
		break;

	case CUDA_DLL_FREEMEM:
		if ((pKernel = g_rgdwDoubleKernelTable[lKernelIdx]) == NULL)
		{
			lErr = ERROR_PARAM_NULL;
			getError(lKernelIdx, lErr, szErr, lszErrMax);
			return lErr;
		}

		if (lErr = pKernel->FreeHost(pInput))
		{
			getError(lKernelIdx, lErr, szErr, lszErrMax);
			return lErr;
		}
		break;

	case CUDA_DLL_KERNEL_MEMCPY:
		if ((pKernel = g_rgdwDoubleKernelTable[lKernelIdx]) == NULL)
		{
			lErr = ERROR_PARAM_NULL;
			getError(lKernelIdx, lErr, szErr, lszErrMax);
			return lErr;
		}

		if (lErr = copyMemKernelToKernel(pKernel, lInput, pInput))
		{
			getError(lKernelIdx, lErr, szErr, lszErrMax);
			return lErr;
		}
		break;

	case CUDA_DLL_KERNEL_ADD:
		if ((pKernel = g_rgdwDoubleKernelTable[lKernelIdx]) == NULL)
		{
			lErr = ERROR_PARAM_NULL;
			getError(lKernelIdx, lErr, szErr, lszErrMax);
			return lErr;
		}

		if (lErr = addKernelToKernel(pKernel, lInput, pInput))
		{
			getError(lKernelIdx, lErr, szErr, lszErrMax);
			return lErr;
		}
		break;

	case CUDA_DLL_KERNEL_COPY_NCCL:
	{
		if ((pKernel = g_rgdwDoubleKernelTable[lKernelIdx]) == NULL)
		{
			lErr = ERROR_PARAM_NULL;
			getError(lKernelIdx, lErr, szErr, lszErrMax);
			return lErr;
		}

		if (lInput != 2)
		{
			return ERROR_PARAM_OUT_OF_RANGE;
			getError(lKernelIdx, lErr, szErr, lszErrMax);
			return lErr;
		}

		long hKernelSrc = (long)pInput[0];
		long hNcclSrc = (long)pInput[1];

		if (hKernelSrc == 0 || hNcclSrc == 0)
		{
			return ERROR_PARAM_OUT_OF_RANGE;
			getError(lKernelIdx, lErr, szErr, lszErrMax);
			return lErr;
		}

		Kernel<double>* pKernelSrc = NULL;
		if ((pKernelSrc = g_rgdwDoubleKernelTable[hKernelSrc]) == NULL)
		{
			lErr = ERROR_PARAM_NULL;
			getError(lKernelIdx, lErr, szErr, lszErrMax);
			return lErr;
		}

		ncclHandle<double>* pNccl = NULL;
		if ((pNccl = pKernelSrc->GetNCCL(hNcclSrc)) == NULL)
		{
			lErr = ERROR_PARAM_NULL;
			getError(lKernelIdx, lErr, szErr, lszErrMax);
			return lErr;
		}

		if (lErr = pKernel->SetNCCL(pNccl, ppOutput, plOutput))
		{
			getError(lKernelIdx, lErr, szErr, lszErrMax);
			return lErr;
		}
	}
	break;

	case CUDA_FN_EXTENSION_RUN:
		if ((pKernel = g_rgdwDoubleKernelTable[lKernelIdx]) == NULL)
		{
			lErr = ERROR_PARAM_NULL;
			getError(lKernelIdx, lErr, szErr, lszErrMax);
			return lErr;
		}

		if (lErr = pKernel->Run(lFunctionIdx, pInput, lInput, ppOutput, plOutput, szErr, lszErrMax))
			return lErr;
		break;

	default:
		if ((pKernel = g_rgdwDoubleKernelTable[lKernelIdx]) == NULL)
		{
			lErr = ERROR_PARAM_NULL;
			getError(lKernelIdx, lErr, szErr, lszErrMax);
			return lErr;
		}

		if (lErr = pKernel->Run(lFunctionIdx, pInput, lInput, plInput, llInput, ppOutput, plOutput))
		{
			getError(lKernelIdx, lErr, szErr, lszErrMax);
			return lErr;
		}
		break;
	}

	return lErr;
}


extern "C" LONG WINAPI DLL_QueryString(LONG lKernelIdx,
										LONG lFunctionIdx,
		   							    LONG* pInput, LONG lInput,
							            LPTSTR* ppOutput,
								        LPTSTR szErr, LONG lszErrMax)
{
	LONG lErr = 0;

	if (lKernelIdx < 0 || lKernelIdx >= (LONG)g_dwMaxKernelCount)
		return ERROR_PARAM_OUT_OF_RANGE;


	//-------------------------------------------
	//	Process the requested function.
	//-------------------------------------------

	Kernel<double>* pKernelD = NULL;
	if ((pKernelD = g_rgdwDoubleKernelTable[lKernelIdx]) != NULL)
	{
		switch (lFunctionIdx)
		{
			case CUDA_DLL_FREEMEM:
				if (ppOutput != NULL && *ppOutput != NULL)
				{
					free(*ppOutput);
					*ppOutput = NULL;
				}
				break;

			default:
				if (ppOutput != NULL)
					*ppOutput = NULL;
				lErr = pKernelD->Query(lFunctionIdx, pInput, lInput, ppOutput);
				break;
		}

		if (lErr)
		{
			getError(lKernelIdx, lErr, szErr, lszErrMax);
			return lErr;
		}

		return 0;
	}

	Kernel<float>* pKernelF = NULL;
	if ((pKernelF = g_rgdwFloatKernelTable[lKernelIdx]) != NULL)
	{
		switch (lFunctionIdx)
		{
			case CUDA_DLL_FREEMEM:
				if (ppOutput != NULL && *ppOutput != NULL)
				{
					free(*ppOutput);
					*ppOutput = NULL;
				}
				break;

			default:
				if (ppOutput != NULL)
					*ppOutput = NULL;
				lErr = pKernelF->Query(lFunctionIdx, pInput, lInput, ppOutput);
				break;
		}

		if (lErr)
		{
			getError(lKernelIdx, lErr, szErr, lszErrMax);
			return lErr;
		}

		return 0;
	}

	lErr = ERROR_PARAM_OUT_OF_RANGE;
	getError(lKernelIdx, lErr, szErr, lszErrMax);

	return lErr;
}

extern "C" LONG WINAPI DLL_InvokeFloatEx(LONG lKernelIdx,
										LONG lFunctionIdx,
										float* pInput, LONG lInput,
										LPTSTR pszInput,
										float** ppOutput, LONG* plOutput,
										LPTSTR szErr, LONG lszErrMax)
{
	Kernel<float>* pKernel = NULL;
	LONG lErr = 0;


	if (lKernelIdx < 0 || lKernelIdx >= (LONG)g_dwMaxKernelCount)
		return ERROR_PARAM_OUT_OF_RANGE;


	//-------------------------------------------
	//	Process the requested function.
	//-------------------------------------------

	switch (lFunctionIdx)
	{
		case CUDA_FN_CREATE_EXTENSION:
			if ((pKernel = g_rgdwFloatKernelTable[lKernelIdx]) == NULL)
			{
				lErr = ERROR_PARAM_NULL;
				getError(lKernelIdx, lErr, szErr, lszErrMax);
				return lErr;
			}

			if (lErr = pKernel->CreateExtensionFloat(g_hModule, lKernelIdx, pszInput, ppOutput, plOutput))
			{
				getError(lKernelIdx, lErr, szErr, lszErrMax);
				return lErr;
			}
			break;

		default:
			return ERROR_NOT_SUPPORTED;
	}

	return lErr;
}

extern "C" LONG WINAPI DLL_InvokeDoubleEx(LONG lKernelIdx,
	LONG lFunctionIdx,
	double* pInput, LONG lInput,
	LPTSTR pszInput,
	double** ppOutput, LONG* plOutput,
	LPTSTR szErr, LONG lszErrMax)
{
	Kernel<double>* pKernel = NULL;
	LONG lErr = 0;


	if (lKernelIdx < 0 || lKernelIdx >= (LONG)g_dwMaxKernelCount)
		return ERROR_PARAM_OUT_OF_RANGE;


	//-------------------------------------------
	//	Process the requested function.
	//-------------------------------------------

	switch (lFunctionIdx)
	{
		case CUDA_FN_CREATE_EXTENSION:
			if ((pKernel = g_rgdwDoubleKernelTable[lKernelIdx]) == NULL)
			{
				lErr = ERROR_PARAM_NULL;
				getError(lKernelIdx, lErr, szErr, lszErrMax);
				return lErr;
			}

			if (lErr = pKernel->CreateExtensionDouble(g_hModule, lKernelIdx, pszInput, ppOutput, plOutput))
			{
				getError(lKernelIdx, lErr, szErr, lszErrMax);
				return lErr;
			}
			break;

		default:
			return ERROR_NOT_SUPPORTED;
	}

	return lErr;
}

LONG addKernelToKernel(Kernel<float>* pSrcKernel, LONG lInput, float* pInput)
{
	LONG lErr = 0;

	if (lInput != 5)
		return ERROR_PARAM_OUT_OF_RANGE;

	int nCount = (int)pInput[0];
	long hA = (LONG)pInput[1];
	long lDstKernelIdx = (LONG)pInput[2];
	long hB = (LONG)pInput[3];
	long hC = (LONG)pInput[4];

	Kernel<float>* pDstKernel; 
	if ((pDstKernel = g_rgdwFloatKernelTable[lDstKernelIdx]) == NULL)
		return ERROR_PARAM_NULL;

	int nDevice1 = pSrcKernel->GetDevice();
	int nDevice2 = pDstKernel->GetDevice();

	if (nDevice1 != nDevice2)
	{
		if (lErr = cudaDeviceEnablePeerAccess(nDevice2, 0))
		{
			if (lErr != cudaErrorPeerAccessAlreadyEnabled)
				return lErr;
			else
				cudaGetLastError();	// clear the error.
		}
	}

	MemoryItem* pA;
	MemoryItem* pB;
	MemoryItem* pC;

	if (lErr = pSrcKernel->GetMemory(hA, &pA))
		return lErr;

	if (lErr = pDstKernel->GetMemory(hB, &pB))
		return lErr;

	pC = pB;

	if (hB != hC)
	{
		if (lErr = pDstKernel->GetMemory(hC, &pC))
			return lErr;
	}

	float* a = (float*)pA->Data();
	float* b = (float*)pB->Data();
	float* c = (float*)pC->Data();

	Math<float> math;

	return math.add(nCount, a, b, c);
}


LONG copyMemKernelToKernel(Kernel<float>* pSrcKernel, LONG lInput, float* pInput)
{
	LONG lErr = 0;

	if (lInput != 9)
		return ERROR_PARAM_OUT_OF_RANGE;

	int nCount = (int)pInput[0];
	long hSrc = (LONG)pInput[1];
	int nSrcOffset = (int)pInput[2];
	long lDstKernelIdx = (LONG)pInput[3];
	long hDst = (LONG)pInput[4];
	int nDstOffset = (int)pInput[5];
	long hHost = (long)pInput[6];
	long lHostKernelIdx = (long)pInput[7];
	long hStream = (long)pInput[8];

	if (lHostKernelIdx >= MAX_KERNELS)
		return ERROR_PARAM_OUT_OF_RANGE;

	if (lDstKernelIdx >= MAX_KERNELS)
		return ERROR_PARAM_OUT_OF_RANGE;

	Kernel<float>* pDstKernel; 
	if ((pDstKernel = g_rgdwFloatKernelTable[lDstKernelIdx]) == NULL)
		return ERROR_PARAM_NULL;

	MemoryItem* pSrc;
	MemoryItem* pDst;

	if (lErr = pSrcKernel->GetMemory(hSrc, &pSrc))
		return lErr;

	if (lErr = pDstKernel->GetMemory(hDst, &pDst))
		return lErr;


	int nSrcSize = (nSrcOffset + nCount) * sizeof(float);
	if (nSrcSize > pSrc->Size())
		return ERROR_PARAM_OUT_OF_RANGE;

	int nDstSize = (nDstOffset + nCount) * sizeof(float);
	if (nDstSize > pDst->Size())
		return ERROR_PARAM_OUT_OF_RANGE;

	float* src = (float*)(pSrc->Data());
	float* dst = (float*)(pDst->Data());

	src += nSrcOffset;
	dst += nDstOffset;
		
	int nSrcDeviceID = pSrcKernel->GetDevice();
	int nDstDeviceID = pDstKernel->GetDevice();
	int nSize = nCount * sizeof(float);

	if (nSrcDeviceID == nDstDeviceID)
	{
		if (hStream < 0)
		{
			lErr = cudaMemcpy(dst, src, nSize, cudaMemcpyDeviceToDevice);
			if (lErr)
				return lErr;

			return cudaStreamSynchronize(0);
		}
		else if (hStream == 0)
		{
			lErr = cudaMemcpyAsync(dst, src, nSize, cudaMemcpyDeviceToDevice, cudaStreamDefault);
			if (lErr)
				return lErr;

			return cudaStreamSynchronize(cudaStreamDefault);
		}
		else
		{
			cudaStream_t strm = pSrcKernel->GetStream(hStream);
			lErr = cudaMemcpyAsync(dst, src, nSize, cudaMemcpyDeviceToDevice, strm);
			if (lErr)
				return lErr;

			return cudaStreamSynchronize(strm);
		}
	}
	else
	{
		int canAccessPeer = 0;

		if (lErr = cudaDeviceCanAccessPeer(&canAccessPeer, nSrcDeviceID, nDstDeviceID))
			return lErr;

		if (canAccessPeer != 0)
		{
			int nCurrentDeviceID = -1;

			if (lErr = cudaGetDevice(&nCurrentDeviceID))
				return lErr;

			if (nCurrentDeviceID != nSrcDeviceID)
			{
				if (lErr = cudaSetDevice(nSrcDeviceID))
					return lErr;
			}

			if (lErr = cudaDeviceEnablePeerAccess(nDstDeviceID, 0))
			{
				if (lErr != cudaErrorPeerAccessAlreadyEnabled)
					return lErr;
				else
					cudaGetLastError(); // clear the error.
			}

			if (hStream < 0)
			{
				lErr = cudaMemcpyPeer(dst, nDstDeviceID, src, nSrcDeviceID, nSize);
				if (lErr)
					return lErr;

				return cudaStreamSynchronize(0);
			}
			else if (hStream == 0)
			{
				lErr = cudaMemcpyPeerAsync(dst, nDstDeviceID, src, nSrcDeviceID, nSize, cudaStreamDefault);
				if (lErr)
					return lErr;

				return cudaStreamSynchronize(cudaStreamDefault);
			}
			else
			{
				cudaStream_t strm = pSrcKernel->GetStream(hStream);
				lErr = cudaMemcpyPeerAsync(dst, nDstDeviceID, src, nSrcDeviceID, nSize, strm);
				if (lErr)
					return lErr;

				return cudaStreamSynchronize(strm);
			}

			if (nCurrentDeviceID != nSrcDeviceID)
			{
				if (lErr = cudaSetDevice(nCurrentDeviceID))
					return lErr;
			}
		}
		else
		{
			Kernel<float>* pHostKernel = pSrcKernel;
			if (lHostKernelIdx > 0)
			{
				if ((pHostKernel = g_rgdwFloatKernelTable[lHostKernelIdx]) == NULL)
					return ERROR_PARAM_NULL;
			}

			float* pHost = pHostKernel->GetHostBuffer(hHost)->Data();

			if (lErr = cudaMemcpy(pHost, src, nSize, cudaMemcpyDeviceToHost))
				return lErr;

			if (lErr = cudaStreamSynchronize(0))
				return lErr;

			if (hStream < 0)
			{
				lErr = cudaMemcpy(dst, pHost, nSize, cudaMemcpyHostToDevice);
				if (lErr)
					return lErr;

				return cudaStreamSynchronize(0);
			}
			else if (hStream == 0)
			{
				lErr = cudaMemcpyAsync(dst, pHost, nSize, cudaMemcpyHostToDevice, cudaStreamDefault);
				if (lErr)
					return lErr;

				return cudaStreamSynchronize(cudaStreamDefault);
			}
			else
			{
				cudaStream_t strm = pSrcKernel->GetStream(hStream);
				lErr = cudaMemcpyAsync(dst, pHost, nSize, cudaMemcpyHostToDevice, strm);
				if (lErr)
					return lErr;

				return cudaStreamSynchronize(strm);
			}
		}
	}

	return lErr;
}


LONG addKernelToKernel(Kernel<double>* pSrcKernel, LONG lInput, double* pInput)
{
	LONG lErr = 0;

	if (lInput != 5)
		return ERROR_PARAM_OUT_OF_RANGE;

	int nCount = (int)pInput[0];
	long hA = (LONG)pInput[1];
	long lDstKernelIdx = (LONG)pInput[2];
	long hB = (LONG)pInput[3];
	long hC = (LONG)pInput[4];

	Kernel<double>* pDstKernel; 
	if ((pDstKernel = g_rgdwDoubleKernelTable[lDstKernelIdx]) == NULL)
		return ERROR_PARAM_NULL;

	int nDevice1 = pSrcKernel->GetDevice();
	int nDevice2 = pDstKernel->GetDevice();

	if (nDevice1 != nDevice2)
	{
		if (lErr = cudaDeviceEnablePeerAccess(nDevice2, 0))
		{
			if (lErr != cudaErrorPeerAccessAlreadyEnabled)
				return lErr;
			else
				cudaGetLastError(); // clear the error.
		}
	}

	MemoryItem* pA;
	MemoryItem* pB;
	MemoryItem* pC;

	if (lErr = pSrcKernel->GetMemory(hA, &pA))
		return lErr;

	if (lErr = pDstKernel->GetMemory(hB, &pB))
		return lErr;

	pC = pB;

	if (hB != hC)
	{
		if (lErr = pDstKernel->GetMemory(hC, &pC))
			return lErr;
	}

	double* a = (double*)pA->Data();
	double* b = (double*)pB->Data();
	double* c = (double*)pC->Data();

	Math<double> math;

	return math.add(nCount, a, b, c);
}

LONG copyMemKernelToKernel(Kernel<double>* pSrcKernel, LONG lInput, double* pInput)
{
	LONG lErr = 0;

	if (lInput != 9)
		return ERROR_PARAM_OUT_OF_RANGE;

	int nCount = (int)pInput[0];
	long hSrc = (LONG)pInput[1];
	int nSrcOffset = (int)pInput[2];
	long lDstKernelIdx = (LONG)pInput[3];
	long hDst = (LONG)pInput[4];
	int nDstOffset = (int)pInput[5];
	long hHost = (long)pInput[6];
	long lHostKernelIdx = (long)pInput[7];
	long hStream = (long)pInput[8];

	if (lHostKernelIdx >= MAX_KERNELS)
		return ERROR_PARAM_OUT_OF_RANGE;

	if (lDstKernelIdx >= MAX_KERNELS)
		return ERROR_PARAM_OUT_OF_RANGE;

	Kernel<double>* pDstKernel;
	if ((pDstKernel = g_rgdwDoubleKernelTable[lDstKernelIdx]) == NULL)
		return ERROR_PARAM_NULL;

	MemoryItem* pSrc;
	MemoryItem* pDst;

	if (lErr = pSrcKernel->GetMemory(hSrc, &pSrc))
		return lErr;

	if (lErr = pDstKernel->GetMemory(hDst, &pDst))
		return lErr;

	int nSrcSize = (nSrcOffset + nCount) * sizeof(double);
	if (nSrcSize > pSrc->Size())
		return ERROR_PARAM_OUT_OF_RANGE;

	int nDstSize = (nDstOffset + nCount) * sizeof(double);
	if (nDstSize > pDst->Size())
		return ERROR_PARAM_OUT_OF_RANGE;

	double* src = (double*)(pSrc->Data());
	double* dst = (double*)(pDst->Data());

	src += nSrcOffset;
	dst += nDstOffset;

	int nSrcDeviceID = pSrcKernel->GetDevice();
	int nDstDeviceID = pDstKernel->GetDevice();
	int nSize = nCount * sizeof(double);

	if (nSrcDeviceID == nDstDeviceID)
	{
		if (hStream < 0)
		{
			lErr = cudaMemcpy(dst, src, nSize, cudaMemcpyDeviceToDevice);
			if (lErr)
				return lErr;

			return cudaStreamSynchronize(0);
		}
		else if (hStream == 0)
		{
			lErr = cudaMemcpyAsync(dst, src, nSize, cudaMemcpyDeviceToDevice, cudaStreamDefault);
			if (lErr)
				return lErr;

			return cudaStreamSynchronize(cudaStreamDefault);
		}
		else
		{
			cudaStream_t strm = pSrcKernel->GetStream(hStream);
			lErr = cudaMemcpyAsync(dst, src, nSize, cudaMemcpyDeviceToDevice, strm);
			if (lErr)
				return lErr;

			return cudaStreamSynchronize(strm);
		}
	}
	else
	{
		int canAccessPeer = 0;

		if (lErr = cudaDeviceCanAccessPeer(&canAccessPeer, nSrcDeviceID, nDstDeviceID))
			return lErr;

		if (canAccessPeer != 0)
		{
			int nCurrentDeviceID = -1;

			if (lErr = cudaGetDevice(&nCurrentDeviceID))
				return lErr;

			if (nCurrentDeviceID != nSrcDeviceID)
			{
				if (lErr = cudaSetDevice(nSrcDeviceID))
					return lErr;
			}

			if (lErr = cudaDeviceEnablePeerAccess(nDstDeviceID, 0))
			{
				if (lErr != cudaErrorPeerAccessAlreadyEnabled)
					return lErr;
				else
					cudaGetLastError(); // clear the error.
			}

			if (hStream < 0)
			{
				lErr = cudaMemcpyPeer(dst, nDstDeviceID, src, nSrcDeviceID, nSize);
				if (lErr)
					return lErr;

				return cudaStreamSynchronize(0);
			}
			else if (hStream == 0)
			{
				lErr = cudaMemcpyPeerAsync(dst, nDstDeviceID, src, nSrcDeviceID, nSize, cudaStreamDefault);
				if (lErr)
					return lErr;

				return cudaStreamSynchronize(cudaStreamDefault);
			}
			else
			{
				cudaStream_t strm = pSrcKernel->GetStream(hStream);
				lErr = cudaMemcpyPeerAsync(dst, nDstDeviceID, src, nSrcDeviceID, nSize, strm);
				if (lErr)
					return lErr;

				return cudaStreamSynchronize(strm);
			}

			if (nCurrentDeviceID != nSrcDeviceID)
			{
				if (lErr = cudaSetDevice(nCurrentDeviceID))
					return lErr;
			}
		}
		else
		{
			Kernel<double>* pHostKernel = pSrcKernel;
			if (lHostKernelIdx > 0)
			{
				if ((pHostKernel = g_rgdwDoubleKernelTable[lHostKernelIdx]) == NULL)
					return ERROR_PARAM_NULL;
			}

			double* pHost = pHostKernel->GetHostBuffer(hHost)->Data();

			if (lErr = cudaMemcpy(pHost, src, nSize, cudaMemcpyDeviceToHost))
				return lErr;

			if (lErr = cudaStreamSynchronize(0))
				return lErr;

			if (hStream < 0)
			{
				lErr = cudaMemcpy(dst, pHost, nSize, cudaMemcpyHostToDevice);
				if (lErr)
					return lErr;

				return cudaStreamSynchronize(0);
			}
			else if (hStream == 0)
			{
				lErr = cudaMemcpyAsync(dst, pHost, nSize, cudaMemcpyHostToDevice, cudaStreamDefault);
				if (lErr)
					return lErr;

				return cudaStreamSynchronize(cudaStreamDefault);
			}
			else
			{
				cudaStream_t strm = pSrcKernel->GetStream(hStream);
				lErr = cudaMemcpyAsync(dst, pHost, nSize, cudaMemcpyHostToDevice, strm);
				if (lErr)
					return lErr;

				return cudaStreamSynchronize(strm);
			}
		}
	}

	return lErr;
}

void setCurrentDirectory()
{
	TCHAR szName[MAX_BUFFER+1];
	LPTSTR psz;

	GetModuleFileName(g_hModule, szName, MAX_BUFFER);
	szName[MAX_BUFFER] = (TCHAR)NULL;
	psz = (LPTSTR)_tcsrchr(szName, '\\');

	if (psz != NULL)
		*psz = (TCHAR)NULL;

	SetCurrentDirectory(szName);
}

void getError(long lKernel, long lErr, LPTSTR szErr, LONG lszErrMax)
{
	USES_CONVERSION;
	char szErr1[MAX_ERROR + 1];

	if (!GetCudaErrorString(lKernel, lErr, szErr1, MAX_ERROR) &&
		!GetErrorString(lKernel, lErr, szErr1, MAX_ERROR))
	{
		snprintf(szErr1, MAX_ERROR, "Unknown error #%ld, Kernel = %ld", lErr, lKernel);
	}
	
	szErr1[MAX_ERROR] = (TCHAR)NULL;

	if (strlen(szErr1) > 0)
	{
		LPTSTR psz = A2T(szErr1);
		_tcsncpy(szErr, psz, lszErrMax);
	}
}

LONG getKernelDoubleIndex(LONG* plIdx)
{
	if (!m_bDoubleKernelTableLockInit)
		return ERROR_DLL_NOT_INIT;

	Lock lock(&g_DoubleKernelTableLock);

	while (g_rgdwDoubleKernelTable[g_dwLastKernelDoubleIndex] != NULL && g_dwLastKernelDoubleIndex < g_dwMaxKernelCount)
	{
		g_dwLastKernelDoubleIndex++;
	}

	if (g_dwLastKernelDoubleIndex == g_dwMaxKernelCount)
	{
		g_dwLastKernelDoubleIndex = 1;  // 0 is reserved for the global kernel.

		while (g_rgdwDoubleKernelTable[g_dwLastKernelDoubleIndex] != NULL && g_dwLastKernelDoubleIndex < g_dwMaxKernelCount)
		{
			g_dwLastKernelDoubleIndex++;
		}

		if (g_dwLastKernelDoubleIndex == g_dwMaxKernelCount)
			return ERROR_MEMORY_OUT;
	}

	*plIdx = (LONG)g_dwLastKernelDoubleIndex;

	return 0;
}

LONG getKernelFloatIndex(LONG* plIdx)
{
	if (!m_bFloatKernelTableLockInit)
		return ERROR_DLL_NOT_INIT;

	Lock lock(&g_FloatKernelTableLock);

	while (g_rgdwFloatKernelTable[g_dwLastKernelFloatIndex] != NULL && g_dwLastKernelFloatIndex < g_dwMaxKernelCount)
	{
		g_dwLastKernelFloatIndex++;
	}

	if (g_dwLastKernelFloatIndex == g_dwMaxKernelCount)
	{
		g_dwLastKernelFloatIndex = 1;  // 0 is reserved for the global kernel.

		while (g_rgdwFloatKernelTable[g_dwLastKernelFloatIndex] != NULL && g_dwLastKernelFloatIndex < g_dwMaxKernelCount)
		{
			g_dwLastKernelFloatIndex++;
		}

		if (g_dwLastKernelFloatIndex == g_dwMaxKernelCount)
			return ERROR_MEMORY_OUT;
	}

	*plIdx = (LONG)g_dwLastKernelFloatIndex;

	return 0;
}

//end CudaDLL.cpp

