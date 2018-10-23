//=============================================================================
//	main.mu
//
//	The kernel manages the interface to the DLL.
//=============================================================================

//=============================================================================
//	Includes
//=============================================================================

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys\timeb.h>
#include <Windows.h>

// includes, project
#include "main.h"

#ifdef _DEBUG
#ifdef _TRACEAPI
static char s_msgbuf[256];

char* GetApiName(long lfnIdx);
#endif
#endif

//=============================================================================
//	Methods
//=============================================================================

template <class T>
long Kernel<T>::Run(long lfnIdx, T* pfInput, long lCount, T** ppfOutput, long* plCount)
{
#ifdef _DEBUG
#ifdef _TRACEAPI
	snprintf(s_msgbuf, 256, "calling CudaPreProcDLL FunctionID (%ld) %s\n", lfnIdx, GetApiName(lfnIdx));
	OutputDebugStringA(s_msgbuf);
#endif
#endif

	switch (lfnIdx)
	{
		case CUDAPP_FN_INITIALIZE:
			return m_device.Initialize(lCount, pfInput, plCount, ppfOutput);

		case CUDAPP_FN_CLEANUP:
			return m_device.CleanUp(lCount, pfInput, plCount, ppfOutput);

		case CUDAPP_FN_SETMEMORY:
			return m_device.SetMemory(lCount, pfInput, plCount, ppfOutput);

		case CUDAPP_FN_ADDDATA:
			return m_device.AddData(lCount, pfInput, plCount, ppfOutput);

		case CUDAPP_FN_PROCESSDATA:
			return m_device.ProcessData(lCount, pfInput, plCount, ppfOutput);

		case CUDAPP_FN_GETVISUALIZATION:
			return m_device.GetVisualization(lCount, pfInput, plCount, ppfOutput);

		case CUDAPP_FN_CLEAR:
			return m_device.Clear(lCount, pfInput, plCount, ppfOutput);

		default:
			return ERROR_INVALID_PARAMETER;
	}
}

template long Kernel<float>::Run(long lfnIdx, float* pfInput, long lCount, float** ppfOutput, long* plCount);
template long Kernel<double>::Run(long lfnIdx, double* pfInput, long lCount, double** ppfOutput, long* plCount);


#ifdef _DEBUG
#ifdef _TRACEAPI
char* GetApiName(long lfnIdx)
{
	switch (lfnIdx)
	{
		case CUDAPP_FN_INITIALIZE:
			return "CUDAPP_FN_INITIALIZE";

		case CUDAPP_FN_CLEANUP:
			return "CUDAPP_FN_CLEANUP";

		case CUDAPP_FN_SETMEMORY:
			return "CUDAPP_FN_SETMEMORY";

		case CUDAPP_FN_ADDDATA:
			return "CUDAPP_FN_ADDDATA";

		case CUDAPP_FN_PROCESSDATA:
			return "CUDAPP_FN_PROCESSDATA";

		case CUDAPP_FN_GETVISUALIZATION:
			return "CUDAPP_FN_GETVISUALIZATION";

		case CUDAPP_FN_CLEAR:
			return "CUDAPP_FN_CLEAR";

		default:
			return "UNKNOWN";
	}
}
#endif
#endif

//end main.cu