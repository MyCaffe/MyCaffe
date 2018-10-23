//=============================================================================
//	FILE:	device.cu
//
//	DESC:	This file implements the base class used to manage the underlying
//			GPU device.
//=============================================================================

#include "device.h"
#include <nvapi.h>
#include <string>


//=============================================================================
//	local constants
//=============================================================================

//=============================================================================
//	Class Methods
//=============================================================================

template <class T>
long Device<T>::CanAccessPeer(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	long lErr;

	if (lErr = verifyInput(lInput, pfInput, 2, 2))
		return lErr;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	int nSrcDeviceID = (int)pfInput[0];
	int nDstDeviceID = (int)pfInput[1];
	int nAccess;

	if (lErr = cudaDeviceCanAccessPeer(&nAccess, nSrcDeviceID, nDstDeviceID))
		return lErr;

	T fVal = (T)nAccess;

	return setOutput(fVal, plOutput, ppfOutput);
}

template long Device<double>::CanAccessPeer(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::CanAccessPeer(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::EnablePeerAccess(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	long lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;

	int nSrcDeviceID = (int)pfInput[0];

	return cudaDeviceEnablePeerAccess(nSrcDeviceID, 0);
}

template long Device<double>::EnablePeerAccess(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::EnablePeerAccess(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::DisablePeerAccess(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	long lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;

	int nSrcDeviceID = (int)pfInput[0];

	return cudaDeviceDisablePeerAccess(nSrcDeviceID);
}

template long Device<double>::DisablePeerAccess(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::DisablePeerAccess(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::GetDeviceName(int nDevice, LPTSTR* pszDevice)
{
	USES_CONVERSION;
	LONG lErr;
	cudaDeviceProp prop;

	if (lErr = cudaGetDeviceProperties(&prop, nDevice))
		return lErr;

	bool b64Bit = (sizeof(void*) == 8) ? true : false;
	bool bTcc = (prop.tccDriver == 1) ? true : false;
	bool bVer = (prop.major >= 2) ? true : false;

	double dfGB = (double)prop.totalGlobalMem / 1000000000.00;

	LPTSTR pDst = (LPTSTR)malloc(sizeof(TCHAR) * 512);
	if (pDst == NULL)
		return ERROR_OUTOFMEMORY;

	memset(pDst, 0, sizeof(TCHAR) * 512);
	_sntprintf(pDst, 511, _T("%s (%0.2lf GB - %s)"), A2T(prop.name), dfGB, (b64Bit && bTcc && bVer) ? _T("P2P on") : _T("P2P off"));
	*pszDevice = pDst;

	return lErr;
}

template long Device<double>::GetDeviceName(int nDevice, LPTSTR* pszDevice);
template long Device<float>::GetDeviceName(int nDevice, LPTSTR* pszDevice);


template <class T>
long Device<T>::GetDeviceName(long lInput, LONG* pInput, LPTSTR* ppOutput)
{
	LONG lErr;

	if (pInput == NULL)
		return ERROR_PARAM_NULL;
		
	if (lInput < 1)
		return ERROR_PARAM_OUT_OF_RANGE;

	int nDevice = (int)pInput[0];
	LPTSTR pszDevice;

	if (lErr = GetDeviceName(nDevice, &pszDevice))
		return lErr;

	*ppOutput = pszDevice;

	return 0;
}

template long Device<double>::GetDeviceName(long lInput, LONG* pInput, LPTSTR* ppOutput);
template long Device<float>::GetDeviceName(long lInput, LONG* pInput, LPTSTR* ppOutput);

int getSPcores(cudaDeviceProp prop)
{
	int mp = prop.multiProcessorCount;

	switch (prop.major)
	{
		case 2:	// Fermi
			if (prop.minor == 1)
				return mp * 48;
			else
				return mp * 32;

		case 3: // Kepler
			return mp * 192;

		case 5: // Maxwell
			return mp * 128;

		case 6: // Pascal
			if (prop.minor == 1)
				return mp * 128;
			if (prop.minor == 0)
				return mp * 64;
			break;
	}

	return -1;
}

template <class T>
long Device<T>::GetDeviceP2PInfo(int nDevice, LPTSTR* pszDevice)
{
	USES_CONVERSION;
	LONG lErr;
	cudaDeviceProp prop;

	if (lErr = cudaGetDeviceProperties(&prop, nDevice))
		return lErr;

	bool bCapable = false;
	bool b64bit = (sizeof(void*) == 8) ? true : false;

	if (prop.tccDriver && prop.major >= 2 && b64bit)
		bCapable = true;

	LPTSTR pDst = (LPTSTR)malloc(sizeof(TCHAR) * 2048);
	if (pDst == NULL)
		return ERROR_OUTOFMEMORY;

	memset(pDst, 0, sizeof(TCHAR) * 2048);
	_sntprintf(pDst, 2047, _T("%s (Device: %d) -> TCC Driver = %s, 64bit = %s, Major = %d, Minor = %d, ComputeMode = %d, P2P Capable = %s, Cores = %d"), A2T(prop.name), nDevice, (prop.tccDriver) ? _T("YES") : _T("NO"), (b64bit) ? _T("YES") : _T("NO"), prop.major, prop.minor, prop.computeMode, (bCapable) ? _T("YES") : _T("NO"), getSPcores(prop));
	*pszDevice = pDst;

	return lErr;
}

template long Device<double>::GetDeviceP2PInfo(int nDevice, LPTSTR* pszDevice);
template long Device<float>::GetDeviceP2PInfo(int nDevice, LPTSTR* pszDevice);


template <class T>
long Device<T>::GetDeviceP2PInfo(long lInput, LONG* pInput, LPTSTR* ppOutput)
{
	LONG lErr;

	if (pInput == NULL)
		return ERROR_PARAM_NULL;
		
	if (lInput < 1)
		return ERROR_PARAM_OUT_OF_RANGE;

	int nDevice = (int)pInput[0];
	LPTSTR pszDevice;

	if (lErr = GetDeviceP2PInfo(nDevice, &pszDevice))
		return lErr;

	*ppOutput = pszDevice;

	return 0;
}

template long Device<double>::GetDeviceP2PInfo(long lInput, LONG* pInput, LPTSTR* ppOutput);
template long Device<float>::GetDeviceP2PInfo(long lInput, LONG* pInput, LPTSTR* ppOutput);


template <class T>
long Device<T>::GetDeviceInfo(int nDevice, LPTSTR* pszDevice, bool bVerbose)
{
	USES_CONVERSION;
	LONG lErr;

	char rgID[256];
	if (lErr = cudaDeviceGetPCIBusId(rgID, 255, nDevice))
		return lErr;

	char* psz = strtok(rgID, ":");
	if (psz == NULL)
	{
		LPCSTR pszErr = "Could not find the Cuda PCI Bus Id.";
		ReportEventA(m_hEventSrc, EVENTLOG_ERROR_TYPE, 0, ERROR_NOT_IMPLEMENTED, NULL, 1, 0, &pszErr, NULL);
		return ERROR_NOT_IMPLEMENTED;
	}

	psz = strtok(NULL, ":");
	if (psz == NULL)
	{
		LPCSTR pszErr = "Could not find the Cuda PCI Bus Id.";
		ReportEventA(m_hEventSrc, EVENTLOG_ERROR_TYPE, 0, ERROR_NOT_IMPLEMENTED, NULL, 1, 0, &pszErr, NULL);
		return ERROR_NOT_IMPLEMENTED;
	}

	std::string str = "0x";
	str += psz;
	int nCudaBusID = std::stoul(str, nullptr, 16);

	NvAPI_Status status;

	if ((status = NvAPI_Initialize()) != NVAPI_OK)
	{
		LPCSTR pszErr = "NvAPI Initializing...";
		ReportEventA(m_hEventSrc, EVENTLOG_ERROR_TYPE, 0, ERROR_NOT_IMPLEMENTED, NULL, 1, 0, &pszErr, NULL);
		NvAPI_ShortString szErr;
		NvAPI_GetErrorMessage(status, szErr);
		ReportEventA(m_hEventSrc, EVENTLOG_ERROR_TYPE, 0, ERROR_NOT_IMPLEMENTED, NULL, 1, 0, &pszErr, NULL);
		return status;
	}

	NvPhysicalGpuHandle gpuHandles[256];
	NvPhysicalGpuHandle gpuTccHandles[256];
	NvU32 numOfGPUs;
	NvU32 numOfTccGPUs;

	if ((status = NvAPI_EnumPhysicalGPUs(gpuHandles, &numOfGPUs)) != NVAPI_OK)
	{
		LPCSTR pszErr = "NvAPI Enumerating Physical GPUs...";
		ReportEventA(m_hEventSrc, EVENTLOG_ERROR_TYPE, 0, ERROR_NOT_IMPLEMENTED, NULL, 1, 0, &pszErr, NULL);
		NvAPI_ShortString szErr;
		NvAPI_GetErrorMessage(status, szErr);
		ReportEventA(m_hEventSrc, EVENTLOG_ERROR_TYPE, 0, ERROR_NOT_IMPLEMENTED, NULL, 1, 0, &pszErr, NULL);
		return status;
	}

	if ((status = NvAPI_EnumTCCPhysicalGPUs(gpuTccHandles, &numOfTccGPUs)) != NVAPI_OK)
	{
		LPCSTR pszErr = "NvAPI Enumerating Physical GPUs...";
		ReportEventA(m_hEventSrc, EVENTLOG_ERROR_TYPE, 0, ERROR_NOT_IMPLEMENTED, NULL, 1, 0, &pszErr, NULL);
		NvAPI_ShortString szErr;
		NvAPI_GetErrorMessage(status, szErr);
		ReportEventA(m_hEventSrc, EVENTLOG_ERROR_TYPE, 0, ERROR_NOT_IMPLEMENTED, NULL, 1, 0, &pszErr, NULL);
		return status;
	}

	int nIdx = -1;
	int nIdxTcc = -1;

	for (int i = 0; i < (int)numOfGPUs; i++)
	{
		NvU32 busID = 0;

		if ((status = NvAPI_GPU_GetBusId(gpuHandles[i], &busID)) != NVAPI_OK)
		{
			LPCSTR pszErr = "NvAPI Getting BusId...";
			ReportEventA(m_hEventSrc, EVENTLOG_ERROR_TYPE, 0, ERROR_NOT_IMPLEMENTED, NULL, 1, 0, &pszErr, NULL);
			NvAPI_ShortString szErr;
			NvAPI_GetErrorMessage(status, szErr);
			ReportEventA(m_hEventSrc, EVENTLOG_ERROR_TYPE, 0, ERROR_NOT_IMPLEMENTED, NULL, 1, 0, &pszErr, NULL);
			return status;
		}

		if (nCudaBusID == (int)busID)
		{
			nIdx = i;
			break;
		}
	}

	if (nIdx == -1)
	{
		for (int i = 0; i < (int)numOfTccGPUs; i++)
		{
			NvU32 busID = 0;

			if ((status = NvAPI_GPU_GetBusId(gpuTccHandles[i], &busID)) != NVAPI_OK)
			{
				LPCSTR pszErr = "NvAPI Getting BusId...";
				ReportEventA(m_hEventSrc, EVENTLOG_ERROR_TYPE, 0, ERROR_NOT_IMPLEMENTED, NULL, 1, 0, &pszErr, NULL);
				NvAPI_ShortString szErr;
				NvAPI_GetErrorMessage(status, szErr);
				ReportEventA(m_hEventSrc, EVENTLOG_ERROR_TYPE, 0, ERROR_NOT_IMPLEMENTED, NULL, 1, 0, &pszErr, NULL);
				return status;
			}

			if (nCudaBusID == (int)busID)
			{
				nIdxTcc = i;
				break;
			}
		}
	}

	if (nIdx == -1 && nIdxTcc == -1)
		return ERROR_NOT_IMPLEMENTED;

	NvU32 connectedDisplays = 0;

	if (nIdx >= 0)
	{
		if ((status = NvAPI_GPU_GetConnectedDisplayIds(gpuHandles[nIdx], NULL, &connectedDisplays, NULL)) != NVAPI_OK)
		{
			LPCSTR pszErr = "NvAPI Getting Connected Display Ids...";
			ReportEventA(m_hEventSrc, EVENTLOG_ERROR_TYPE, 0, ERROR_NOT_IMPLEMENTED, NULL, 1, 0, &pszErr, NULL);
			NvAPI_ShortString szErr;
			NvAPI_GetErrorMessage(status, szErr);
			ReportEventA(m_hEventSrc, EVENTLOG_ERROR_TYPE, 0, ERROR_NOT_IMPLEMENTED, NULL, 1, 0, &pszErr, NULL);
			return status;
		}
	}

	NV_GPU_THERMAL_SETTINGS thermal;
	thermal.version = NV_GPU_THERMAL_SETTINGS_VER;

	if (nIdx >= 0)
		status = NvAPI_GPU_GetThermalSettings(gpuHandles[nIdx], 0, &thermal);
	else
		status = NvAPI_GPU_GetThermalSettings(gpuTccHandles[nIdxTcc], 0, &thermal);

	if (status != NVAPI_OK)
	{
		LPCSTR pszErr = "NvAPI Getting Thermal Settings...";
		ReportEventA(m_hEventSrc, EVENTLOG_ERROR_TYPE, 0, ERROR_NOT_IMPLEMENTED, NULL, 1, 0, &pszErr, NULL);
		NvAPI_ShortString szErr;
		NvAPI_GetErrorMessage(status, szErr);
		ReportEventA(m_hEventSrc, EVENTLOG_ERROR_TYPE, 0, ERROR_NOT_IMPLEMENTED, NULL, 1, 0, &pszErr, NULL);
		return status;
	}

	NV_GPU_DYNAMIC_PSTATES_INFO_EX states;
	states.version = NV_GPU_DYNAMIC_PSTATES_INFO_EX_VER;

	if (nIdx >= 0)
		status = NvAPI_GPU_GetDynamicPstatesInfoEx(gpuHandles[nIdx], &states);
	else
		status = NvAPI_GPU_GetDynamicPstatesInfoEx(gpuTccHandles[nIdxTcc], &states);

	if (status != NVAPI_OK)
	{
		LPCSTR pszErr = "NvAPI Getting Dynamic Pstates...";
		ReportEventA(m_hEventSrc, EVENTLOG_ERROR_TYPE, 0, ERROR_NOT_IMPLEMENTED, NULL, 1, 0, &pszErr, NULL);
		NvAPI_ShortString szErr;
		NvAPI_GetErrorMessage(status, szErr);
		ReportEventA(m_hEventSrc, EVENTLOG_ERROR_TYPE, 0, ERROR_NOT_IMPLEMENTED, NULL, 1, 0, &pszErr, NULL);
		return status;
	}

	double dfUtilization = 0;
	int nUtilization = 0;

	if (states.utilization[0].bIsPresent)
	{
		dfUtilization = (double)states.utilization[0].percentage;
		nUtilization = (int)dfUtilization;
	}

	double dfC = (double)thermal.sensor[0].currentTemp;
	int nTemp = (int)dfC;

	LPTSTR pDst = (LPTSTR)malloc(sizeof(TCHAR) * 2048);
	if (pDst == NULL)
		return ERROR_OUTOFMEMORY;

	memset(pDst, 0, sizeof(TCHAR) * 2048);
	char szTmp[16];
	_snprintf(szTmp, 16, "%c", (char)176);
	_sntprintf(pDst, 2047, _T(" GPU = %d, MonitorOn = %s, GPU_Temp = %d C%s, GPU_Use = %d%%"), nDevice, (connectedDisplays == 0) ? _T("NO") : _T("YES"), nTemp, A2T(szTmp), nUtilization);
	*pszDevice = pDst;

	if (bVerbose)
	{
		cudaDeviceProp prop;
		if (lErr = cudaGetDeviceProperties(&prop, nDevice))
			return lErr;

		char szBuffer[1024];
		_snprintf(szBuffer, 1023, "\r\n Major: %d, Minor: %d, Compute Mode: %d\r\n Max Grid: { %d, %d, %d }, Max Thread Dim: { %d, %d, %d }\r\n Shared Memory/Block: %zd\r\n", prop.major, prop.minor, prop.computeMode, prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2], prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2], prop.sharedMemPerBlock);
		_tcsncat(pDst, A2T(szBuffer), 2047);
	}

	*pszDevice = pDst;

	return lErr;
}

template long Device<double>::GetDeviceInfo(int nDevice, LPTSTR* pszDevice, bool bVerbose);
template long Device<float>::GetDeviceInfo(int nDevice, LPTSTR* pszDevice, bool bVerbose);


template <class T>
long Device<T>::GetDeviceInfo(long lInput, LONG* pInput, LPTSTR* ppOutput)
{
	LONG lErr;

	if (pInput == NULL)
		return ERROR_PARAM_NULL;

	if (lInput < 1)
		return ERROR_PARAM_OUT_OF_RANGE;

	int nDevice = (int)pInput[0];
	bool bVerbose = false;
	LPTSTR pszDevice;

	if (lInput > 1)
	{
		if (pInput[1] != 0)
			bVerbose = true;
	}

	if (lErr = GetDeviceInfo(nDevice, &pszDevice, bVerbose))
		return lErr;

	*ppOutput = pszDevice;

	return 0;
}

template long Device<double>::GetDeviceInfo(long lInput, LONG* pInput, LPTSTR* ppOutput);
template long Device<float>::GetDeviceInfo(long lInput, LONG* pInput, LPTSTR* ppOutput);

template <class T>
long Device<T>::SetDevice(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;
	
	if (lErr = verifyInput(lInput, pfInput, 2, 3))
		return lErr;

	int nDevice = (int)pfInput[0];
	int nFlags = (int)pfInput[1];
	long lSeed = 0;

	if (lInput > 2)
	{
		lSeed = (long)pfInput[2];
		nFlags |= DEVINIT_SETSEED;
	}

	return SetDevice(nDevice, nFlags, lSeed);
}

template long Device<double>::SetDevice(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::SetDevice(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::SetDevice(int nDeviceID, int nInitFlags, long lSeed)
{
	LONG lErr;

	if ((nInitFlags & DEVINIT_RESETDEVICE) == DEVINIT_RESETDEVICE)
	{
		if (lErr = cudaDeviceReset())
			return lErr;
	}

	if (lErr = cudaSetDevice(nDeviceID))
		return lErr;

	m_nDevice = nDeviceID;

	if ((nInitFlags & DEVINIT_CUBLAS) == DEVINIT_CUBLAS)
	{
		if (m_cublas != NULL)
		{
			cublasDestroy(m_cublas);
			m_cublas = NULL;
		}

		if (lErr = cublasCreate(&m_cublas))
			return lErr;
	}

	if ((nInitFlags & DEVINIT_CURAND) == DEVINIT_CURAND)
	{
		if (m_curand != NULL)
		{
			curandDestroyGenerator(m_curand);
			m_curand = NULL;
		}

		if (lErr = curandCreateGenerator(&m_curand, CURAND_RNG_PSEUDO_DEFAULT))
			return lErr;
	}

	m_math.SetHandles(nDeviceID, m_cublas, m_curand);

	if ((nInitFlags & DEVINIT_SETSEED) == DEVINIT_SETSEED)
	{
		if (lErr = m_math.rng_setseed(lSeed))
			return lErr;
	}

	return 0;
}

template long Device<double>::SetDevice(int nDeviceID, int nInitFlags, long lSeed);
template long Device<float>::SetDevice(int nDeviceID, int nInitFlags, long lSeed);


template <class T>
long Device<T>::AllocMemory(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, INT_MAX))
		return lErr;

	long hHandle = 0;
	long hStream = 0;
	long lCount = (long)pfInput[0];
	T* pSrc = NULL;

	if (lInput > 1)
	{
		if (lInput == lCount + 1)
		{
			pSrc = &pfInput[1];
		}
		else if (lInput == lCount + 2)
		{
			hStream = (long)pfInput[1];
			pSrc = &pfInput[2];
		}
		else
		{
			return ERROR_PARAM_OUT_OF_RANGE;
		}
	}

	if (lErr = m_memory.AllocMemory(GetDevice(), lCount, pSrc, hStream, &hHandle))
		return lErr;

	if (lErr = setOutput(hHandle, plOutput, ppfOutput))
		return lErr;

	return 0;
}

template long Device<double>::AllocMemory(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::AllocMemory(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::FreeMemory(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;

	long hHandle = (long)pfInput[0];

	return m_memory.FreeMemory(hHandle);
}

template long Device<double>::FreeMemory(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::FreeMemory(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::GetMemory(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 2))
		return lErr;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	long hHandle = (long)pfInput[0];
	long lCount = 0;
	MemoryItem* pItem;

	if (lInput > 1)
		lCount = (long)pfInput[1];

	if (lErr = m_memory.GetMemory(hHandle, &pItem))
		return lErr;

	long lAllocatedCount = pItem->Size() / sizeof(T);

	if (lCount < 0)
		lCount = lAllocatedCount;
	else if (lCount > lAllocatedCount)
		return ERROR_PARAM_OUT_OF_RANGE;

	if (lCount <= *plOutput)
	{
		T* pfOutput = *ppfOutput;

		if (lErr = m_memory.CopyToHost(lCount, pfOutput, (T*)pItem->Data(), true))
			return lErr;
	}
	else
	{
		T* pfOutput = NULL;

		if (lErr = m_memory.AllocHost(lCount, &pfOutput, (T*)pItem->Data(), true))
			return lErr;

		*ppfOutput = pfOutput;
	}

	*plOutput = lCount;
	return 0;
}

template long Device<double>::GetMemory(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::GetMemory(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::SetMemory(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 2, INT_MAX))
		return lErr;

	long hHandle = (long)pfInput[0];
	long lCount = (int)pfInput[1];
	long hStream = 0;
	T* pData = NULL;

	if (lCount == lInput - 2)
	{
		pData = &pfInput[2];
	}
	else if (lCount == lInput - 3)
	{
		hStream = (long)pfInput[2];
		pData = &pfInput[3];
	}
	else
		return ERROR_PARAM_OUT_OF_RANGE;

	if (lErr = m_memory.SetMemory(hHandle, pData, lCount, hStream))
		return lErr;

	return 0;
}

template long Device<double>::SetMemory(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::SetMemory(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::SetMemoryAt(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 3, INT_MAX))
		return lErr;

	long hHandle = (long)pfInput[0];
	long lCount = (int)pfInput[1];
	int nOffset = (int)pfInput[2];
	T* pData = &pfInput[3];

	if (lErr = m_memory.SetMemoryAt(hHandle, pData, lCount, nOffset))
		return lErr;

	return 0;
}

template long Device<double>::SetMemoryAt(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::SetMemoryAt(long lInput, float* pfInput, long* plOutput, float** ppfOutput);

template <class T>
long Device<T>::AllocHostBuffer(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;

	long hHandle = 0;
	long lCount = (long)pfInput[0];

	if (lErr = m_memory.AllocHostBuffer(lCount, &hHandle))
		return lErr;

	if (lErr = setOutput(hHandle, plOutput, ppfOutput))
		return lErr;

	return 0;
}

template long Device<double>::AllocHostBuffer(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::AllocHostBuffer(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::FreeHostBuffer(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;

	long hHandle = (long)pfInput[0];

	return m_memory.FreeHostBuffer(hHandle);
}

template long Device<double>::FreeHostBuffer(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::FreeHostBuffer(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::GetHostBufferCapacity(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;

	long hHandle = (long)pfInput[0];

	HostBuffer<T>* pbuf = m_memory.GetHostBuffer(hHandle);
	if (pbuf == NULL)
		return ERROR_PARAM_OUT_OF_RANGE;

	if (lErr = setOutput((T)pbuf->Count(), plOutput, ppfOutput))
		return lErr;

	return 0;
}

template long Device<double>::GetHostBufferCapacity(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::GetHostBufferCapacity(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::GetHostMemory(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	long hHandle = (long)pfInput[0];

	HostBuffer<T>* pHostBuf = m_memory.GetHostBuffer(hHandle);

	if (pHostBuf != NULL)
	{
		*plOutput = pHostBuf->Count();
		*ppfOutput = pHostBuf->Data();
	}
	else
	{
		*plOutput = 0;
		*ppfOutput = NULL;
	}

	return 0;
}

template long Device<double>::GetHostMemory(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::GetHostMemory(long lInput, float* pfInput, long* plOutput, float** ppfOutput);



template <class T>
long Device<T>::SetHostMemory(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 3, INT_MAX))
		return lErr;

	long hHandle = (long)pfInput[0];
	long lCount = (long)pfInput[1];
	T* pData = &pfInput[2];

	return m_memory.SetHostBuffer(hHandle, lCount, pData);
}

template long Device<double>::SetHostMemory(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::SetHostMemory(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::RunMemoryTest(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 8, 8))
		return lErr;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	long hHandle = (long)pfInput[0];
	MEMTEST_TYPE memTestType = (MEMTEST_TYPE)(int)pfInput[1];
	size_t szStartOffset = (size_t)pfInput[2];
	size_t szCount = (size_t)pfInput[3];
	bool bVerbose = (pfInput[4] == 0) ? false : true;
	bool bWrite = (pfInput[5] == 0) ? false : true;
	bool bReadWrite = (pfInput[6] == 0) ? false : true;
	bool bRead = (pfInput[7] == 0) ? false : true;

	return m_memory.RunMemoryTest(hHandle, memTestType, szStartOffset, szCount, plOutput, ppfOutput, bVerbose, bWrite, bReadWrite, bRead);
}

template long Device<double>::RunMemoryTest(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::RunMemoryTest(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::SetTensorDesc(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 5, 9, true))
		return lErr;

	long hHandle = (long)pfInput[0];
	int n = (int)pfInput[1];
	int c = (int)pfInput[2];
	int h = (int)pfInput[3];
	int w = (int)pfInput[4];
	int nStride;
	int cStride;
	int hStride;
	int wStride;

	if (lInput == 5)
	{
		wStride = 1;
		hStride = w * wStride;
		cStride = h * hStride;
		nStride = c * cStride;
	}
	else
	{
		nStride = (int)pfInput[5];
		cStride = (int)pfInput[6];
		hStride = (int)pfInput[7];
		wStride = (int)pfInput[8];
	}

	if (lErr = m_memory.SetTensorDesc(hHandle, n, c, h, w, nStride, cStride, hStride, wStride))
		return lErr;

	return 0;
}

template long Device<double>::SetTensorDesc(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::SetTensorDesc(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::GetDropoutInfo(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;	
	unsigned long lStates = 0;
	unsigned long lReserved = 0;

	if (lErr = verifyInput(lInput, pfInput, 2, 2))
		return lErr;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	long hCuda = (long)pfInput[0];
	long hBottomDesc = (long)pfInput[1];

	if (lErr = m_memory.GetDropoutInfo(hCuda, hBottomDesc, &lStates, &lReserved))
		return lErr;

	T* pfOutput = NULL;

	if (lErr = m_memory.AllocHost(2, &pfOutput, NULL, false))
		return lErr;

	pfOutput[0] = (T)lStates;
	pfOutput[1] = (T)lReserved;

	*ppfOutput = pfOutput;
	*plOutput = 2;

	return 0;
}

template long Device<double>::GetDropoutInfo(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::GetDropoutInfo(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_get(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 2, 3))
		return lErr;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	int nCount = (int)pfInput[0];
	long hHandle = (long)pfInput[1];
	int nIdx = -1;
	int nItems = nCount;

	if (lInput > 2)
	{
		nIdx = (int)pfInput[2];

		if (nIdx >= 0)
			nItems = 1;
	}

	T* pfOutput = NULL;
	
	if (lErr = m_memory.AllocHost(nItems, &pfOutput, NULL, false))
		return lErr;

	if (lErr = m_math.get(nCount, hHandle, nIdx, pfOutput))
	{
		m_memory.FreeHost(pfOutput);
		return lErr;
	}

	*plOutput = nItems;
	*ppfOutput = pfOutput;
	return 0;
}

template long Device<double>::cuda_get(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_get(long lInput, float* pfInput, long* plOutput, float** ppfOutput);

template <class T>
long Device<T>::cuda_gemm(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 10, 13))
		return lErr;

	bool bTransA = (pfInput[0] == 0.0) ? false : true;
	bool bTransB = (pfInput[1] == 0.0) ? false : true;
	int m = (int)pfInput[2];
	int n = (int)pfInput[3];
	int k = (int)pfInput[4];
	T fAlpha = pfInput[5];
	long hA = (long)pfInput[6];
	long hB = (long)pfInput[7];
	T fBeta = pfInput[8];
	long hC = (long)pfInput[9];
	int nAOff = 0;
	int nBOff = 0;
	int nCOff = 0;

	if (lInput > 10)
		nAOff = (int)pfInput[10];

	if (lInput > 11)
		nBOff = (int)pfInput[11];

	if (lInput > 12)
		nCOff = (int)pfInput[12];

	return m_math.gemm(bTransA, bTransB, m, n, k, fAlpha, hA, hB, fBeta, hC, nAOff, nBOff, nCOff);
}

template long Device<double>::cuda_gemm(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_gemm(long lInput, float* pfInput, long* plOutput, float** ppfOutput);

template <class T>
long Device<T>::cuda_gemm2(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 13, 13))
		return lErr;

	bool bTransA = (pfInput[0] == 0.0) ? false : true;
	bool bTransB = (pfInput[1] == 0.0) ? false : true;
	int m = (int)pfInput[2];
	int n = (int)pfInput[3];
	int k = (int)pfInput[4];
	T fAlpha = pfInput[5];
	long hA = (long)pfInput[6];
	long hB = (long)pfInput[7];
	T fBeta = pfInput[8];
	long hC = (long)pfInput[9];
	int lda = (int)pfInput[10];
	int ldb = (int)pfInput[11];
	int ldc = (int)pfInput[12];

	return m_math.gemm2(bTransA, bTransB, m, n, k, fAlpha, hA, hB, fBeta, hC, lda, ldb, ldc);
}

template long Device<double>::cuda_gemm2(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_gemm2(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_gemv(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 8, 11))
		return lErr;

	bool bTransA = (pfInput[0] == 0.0) ? false : true;
	int n = (int)pfInput[1];
	int m = (int)pfInput[2];
	T fAlpha = pfInput[3];
	long hA = (long)pfInput[4];
	long hX = (long)pfInput[5];
	T fBeta = pfInput[6];
	long hY = (long)pfInput[7];
	int nAOffset = 0;
	int nXOffset = 0;
	int nYOffset = 0;

	if (lInput > 8)
		nAOffset = (int)pfInput[8];

	if (lInput > 9)
		nXOffset = (int)pfInput[9];

	if (lInput > 10)
		nYOffset = (int)pfInput[10];

	return m_math.gemv(bTransA, n, m, fAlpha, hA, hX, fBeta, hY, nAOffset, nXOffset, nYOffset);
}

template long Device<double>::cuda_gemv(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_gemv(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_axpy(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 4, 6))
		return lErr;

	int n = (int)pfInput[0];
	T fAlpha = pfInput[1];
	long hX = (long)pfInput[2];
	long hY = (long)pfInput[3];
	int nXOff = 0;
	int nYOff = 0;

	if (lInput > 4)
		nXOff = (int)pfInput[4];

	if (lInput > 5)
		nYOff = (int)pfInput[5];

	return m_math.axpy(n, fAlpha, hX, hY, nXOff, nYOff);
}

template long Device<double>::cuda_axpy(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_axpy(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_axpby(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 5, 5))
		return lErr;

	int n = (int)pfInput[0];
	T fAlpha = pfInput[1];
	long hX = (long)pfInput[2];
	T fBeta = pfInput[3];
	long hY = (long)pfInput[4];

	return m_math.axpby(n, fAlpha, hX, fBeta, hY);
}

template long Device<double>::cuda_axpby(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_axpby(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_scal(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 3, 4))
		return lErr;

	int n = (int)pfInput[0];
	T fAlpha = pfInput[1];
	long hX = (long)pfInput[2];
	int nXOff = 0;

	if (lInput > 3)
		nXOff = (int)pfInput[3];

	return m_math.scal(n, fAlpha, hX, nXOff);
}

template long Device<double>::cuda_scal(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_scal(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_dot(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 3, 5))
		return lErr;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	int n = (int)pfInput[0];
	long hX = (long)pfInput[1];
	long hY = (long)pfInput[2];
	int nXOff = 0;
	int nYOff = 0;
	T fOutput = 0;

	if (lInput > 3)
		nXOff = (int)pfInput[3];

	if (lInput > 4)
		nYOff = (int)pfInput[4];

	if (lErr = m_math.dot(n, hX, hY, &fOutput, nXOff, nYOff))
		return lErr;

	return setOutput(fOutput, plOutput, ppfOutput);
}

template long Device<double>::cuda_dot(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_dot(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_asum(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 2, 3))
		return lErr;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	int n = (int)pfInput[0];
	long hX = (long)pfInput[1];
	int nXOff = 0;

	if (lInput > 2)
		nXOff = (int)pfInput[2];

	T fOutput = 0;

	if (lErr = m_math.asum(n, hX, &fOutput, nXOff))
		return lErr;

	return setOutput(fOutput, plOutput, ppfOutput);
}

template long Device<double>::cuda_asum(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_asum(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_scale(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 4, 6))
		return lErr;

	int n = (int)pfInput[0];
	T fAlpha = pfInput[1];
	long hX = (long)pfInput[2];
	long hY = (long)pfInput[3];
	int nXOff = 0;
	int nYOff = 0;

	if (lInput > 4)
		nXOff = (int)pfInput[4];

	if (lInput > 5)
		nYOff = (int)pfInput[5];

	return m_math.scale(n, fAlpha, hX, hY, nXOff, nYOff);
}

template long Device<double>::cuda_scale(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_scale(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_add_scalar(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 3, 4))
		return lErr;

	int n = (int)pfInput[0];
	T fAlpha = pfInput[1];
	long hY = (long)pfInput[2];
	int nYOff = 0;

	if (lInput > 3)
		nYOff = (int)pfInput[3];

	return m_math.add_scalar(n, fAlpha, hY, nYOff);
}

template long Device<double>::cuda_add_scalar(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_add_scalar(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_add(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 4, 5))
		return lErr;

	int n = (int)pfInput[0];
	long hA = (long)pfInput[1];
	long hB = (long)pfInput[2];
	long hY = (long)pfInput[3];
	T fAlpha = 1.0;

	if (lInput > 4)
		fAlpha = pfInput[4];

	return m_math.add(n, hA, hB, hY, fAlpha);
}

template long Device<double>::cuda_add(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_add(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_add2(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 6, 9))
		return lErr;

	int n = (int)pfInput[0];
	long hA = (long)pfInput[1];
	long hB = (long)pfInput[2];
	long hY = (long)pfInput[3];
	T fAlphaA = pfInput[4];
	T fAlphaB = pfInput[5];
	int nAOff = 0;
	int nBOff = 0;
	int nYOff = 0;

	if (lInput > 6)
		nAOff = (int)pfInput[6];

	if (lInput > 7)
		nBOff = (int)pfInput[7];

	if (lInput > 8)
		nYOff = (int)pfInput[8];

	return m_math.add2(n, hA, hB, hY, fAlphaA, fAlphaB, nAOff, nBOff, nYOff);
}

template long Device<double>::cuda_add2(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_add2(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_sub(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 4, 7))
		return lErr;

	int n = (int)pfInput[0];
	long hA = (long)pfInput[1];
	long hB = (long)pfInput[2];
	long hY = (long)pfInput[3];
	int nAOff = 0;
	int nBOff = 0;
	int nYOff = 0;

	if (lInput > 4)
		nAOff = (int)pfInput[4];

	if (lInput > 5)
		nBOff = (int)pfInput[5];

	if (lInput > 6)
		nYOff = (int)pfInput[6];

	return m_math.sub(n, hA, hB, hY, nAOff, nBOff, nYOff);
}

template long Device<double>::cuda_sub(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_sub(long lInput, float* pfInput, long* plOutput, float** ppfOutput);



template <class T>
long Device<T>::cuda_sub_and_dot(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 6, 9))
		return lErr;

	int n = (int)pfInput[0];
	int nN = (int)pfInput[1];
	int nLen = (int)pfInput[2];
	long hA = (long)pfInput[3];
	long hB = (long)pfInput[4];
	long hY = (long)pfInput[5];
	int nAOff = 0;
	int nBOff = 0;
	int nYOff = 0;

	if (lInput > 6)
		nAOff = (int)pfInput[6];

	if (lInput > 7)
		nBOff = (int)pfInput[7];

	if (lInput > 8)
		nYOff = (int)pfInput[8];

	return m_math.sub_and_dot(n, nN, nLen, hA, hB, hY, nAOff, nBOff, nYOff);
}

template long Device<double>::cuda_sub_and_dot(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_sub_and_dot(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_mul_scalar(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 3, 4))
		return lErr;

	int n = (int)pfInput[0];
	T fAlpha = pfInput[1];
	long hY = (long)pfInput[2];
	int nYOff = 0;

	if (lInput > 3)
		nYOff = (int)pfInput[3];

	return m_math.mul_scalar(n, fAlpha, hY, nYOff);
}

template long Device<double>::cuda_mul_scalar(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_mul_scalar(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_mul(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 4, 7))
		return lErr;

	int n = (int)pfInput[0];
	long hA = (long)pfInput[1];
	long hB = (long)pfInput[2];
	long hY = (long)pfInput[3];
	int nAOff = 0;
	int nBOff = 0;
	int nYOff = 0;

	if (lInput > 4)
		nAOff = (int)pfInput[4];

	if (lInput > 5)
		nBOff = (int)pfInput[5];

	if (lInput > 6)
		nYOff = (int)pfInput[6];

	return m_math.mul(n, hA, hB, hY, nAOff, nBOff, nYOff);
}

template long Device<double>::cuda_mul(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_mul(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_div(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 4, 4))
		return lErr;

	int n = (int)pfInput[0];
	long hA = (long)pfInput[1];
	long hB = (long)pfInput[2];
	long hY = (long)pfInput[3];

	return m_math.div(n, hA, hB, hY);
}

template long Device<double>::cuda_div(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_div(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_abs(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 3, 3))
		return lErr;

	int n = (int)pfInput[0];
	long hA = (long)pfInput[1];
	long hY = (long)pfInput[2];

	return m_math.abs(n, hA, hY);
}

template long Device<double>::cuda_abs(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_abs(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_exp(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 3, 6))
		return lErr;

	int n = (int)pfInput[0];
	long hA = (long)pfInput[1];
	long hY = (long)pfInput[2];
	int nAOff = 0;
	int nYOff = 0;
	T fBeta = 1.0;

	if (lInput > 3)
		nAOff = (int)pfInput[3];

	if (lInput > 4)
		nYOff = (int)pfInput[4];

	if (lInput > 5)
		fBeta = pfInput[5];

	return m_math.exp(n, hA, hY, nAOff, nYOff, fBeta);
}

template long Device<double>::cuda_exp(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_exp(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_log(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 3, 5))
		return lErr;

	int n = (int)pfInput[0];
	long hA = (long)pfInput[1];
	long hY = (long)pfInput[2];
	T fBeta = 1;
	T fAlpha = 0;

	if (lInput > 3)
	{
		fBeta = pfInput[3];

		if (lInput > 4)
			fAlpha = pfInput[4];
	}

	return m_math.log(n, hA, hY, fBeta, fAlpha);
}

template long Device<double>::cuda_log(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_log(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_powx(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 4, 6))
		return lErr;

	int n = (int)pfInput[0];
	long hA = (long)pfInput[1];
	T fAlpha = pfInput[2];
	long hY = (long)pfInput[3];
	int nAOff = 0;
	int nYOff = 0;

	if (lInput > 4)
		nAOff = (int)pfInput[4];

	if (lInput > 5)
		nYOff = (int)pfInput[5];

	return m_math.powx(n, hA, fAlpha, hY, nAOff, nYOff);
}

template long Device<double>::cuda_powx(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_powx(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_sign(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 3, 5))
		return lErr;

	int n = (int)pfInput[0];
	long hX = (long)pfInput[1];
	long hY = (long)pfInput[2];
	int nXOff = 0;
	int nYOff = 0;

	if (lInput > 3)
		nXOff = (int)pfInput[3];

	if (lInput > 4)
		nYOff = (int)pfInput[4];

	return m_math.sign(n, hX, hY, nXOff, nYOff);
}

template long Device<double>::cuda_sign(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_sign(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_sqrt(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 3, 3))
		return lErr;

	int n = (int)pfInput[0];
	long hX = (long)pfInput[1];
	long hY = (long)pfInput[2];

	return m_math.sqrt(n, hX, hY);
}

template long Device<double>::cuda_sqrt(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_sqrt(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_reciprocol(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 3, 3))
		return lErr;

	int n = (int)pfInput[0];
	long hX = (long)pfInput[1];
	long hY = (long)pfInput[2];

	return m_math.reciprocol(n, hX, hY);
}

template long Device<double>::cuda_reciprocol(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_reciprocol(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_student(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 3, 3))
		return lErr;

	int n = (int)pfInput[0];
	long hX = (long)pfInput[1];
	long hY = (long)pfInput[2];

	return m_math.student(n, hX, hY);
}

template long Device<double>::cuda_student(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_student(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_logistic1(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 3, 3))
		return lErr;

	int n = (int)pfInput[0];
	long hX = (long)pfInput[1];
	long hY = (long)pfInput[2];

	return m_math.logistic1(n, hX, hY);
}

template long Device<double>::cuda_logistic1(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_logistic1(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_logistic2(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 3, 3))
		return lErr;

	int n = (int)pfInput[0];
	long hX = (long)pfInput[1];
	long hY = (long)pfInput[2];

	return m_math.logistic2(n, hX, hY);
}

template long Device<double>::cuda_logistic2(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_logistic2(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_compare_signs(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 4, 4))
		return lErr;

	int n = (int)pfInput[0];
	long hA = (long)pfInput[1];
	long hB = (long)pfInput[2];
	long hY = (long)pfInput[3];

	return m_math.compare_signs(n, hA, hB, hY);
}

template long Device<double>::cuda_compare_signs(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_compare_signs(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_maxval(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 2, 3))
		return lErr;

	int n = (int)pfInput[0];
	long hA = (long)pfInput[1];
	int nAOff = 0;
	T fOutput = 0;

	if (lInput > 2)
		nAOff = (int)pfInput[2];

	if (lErr = m_math.maxval(n, hA, &fOutput, nAOff))
		return lErr;

	return setOutput(fOutput, plOutput, ppfOutput);
}

template long Device<double>::cuda_maxval(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_maxval(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_minval(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 2, 3))
		return lErr;

	int n = (int)pfInput[0];
	long hA = (long)pfInput[1];
	int nAOff = 0;
	T fOutput = 0;

	if (lInput > 2)
		nAOff = (int)pfInput[2];

	if (lErr = m_math.minval(n, hA, &fOutput, nAOff))
		return lErr;

	return setOutput(fOutput, plOutput, ppfOutput);
}

template long Device<double>::cuda_minval(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_minval(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_minmaxval(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 4, 6))
		return lErr;

	int n = (int)pfInput[0];
	long hA = (long)pfInput[1];
	long hWork1 = (long)pfInput[2];
	long hWork2 = (long)pfInput[3];
	bool bDetectNans = false;
	int nAOff = 0;
	T fMin;
	T fMax;
	T fNan = 0;
	T fInf = 0;

	if (lInput > 4)
		bDetectNans = (pfInput[4] == 0) ? false : true;

	if (lInput > 5)
		nAOff = (int)pfInput[5];

	if (lErr = m_math.minmaxval(n, hA, hWork1, hWork2, &fMin, &fMax, nAOff))
		return lErr;

	if (bDetectNans)
	{
		if (lErr = m_math.naninfval(n, hA, hWork1, hWork2, &fNan, &fInf, nAOff))
			return lErr;
	}

	T* pfOutput = NULL;

	if (lErr = m_memory.AllocHost(4, &pfOutput, NULL, false))
		return lErr;

	pfOutput[0] = fMin;
	pfOutput[1] = fMax;
	pfOutput[2] = fNan;
	pfOutput[3] = fInf;

	*ppfOutput = pfOutput;
	*plOutput = 4;

	return 0;
}

template long Device<double>::cuda_minmaxval(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_minmaxval(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_sumsq(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 3, 4))
		return lErr;

	int n = (int)pfInput[0];
	long hW = (long)pfInput[1];
	long hA = (long)pfInput[2];
	int nAOff = 0;

	if (lInput > 3)
		nAOff = (int)pfInput[3];

	T fOutput = 0;

	if (lErr = m_math.sumsq(n, hW, hA, nAOff, &fOutput))
		return lErr;

	return setOutput(fOutput, plOutput, ppfOutput);
}

template long Device<double>::cuda_sumsq(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_sumsq(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_sumsqdiff(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 4, 6))
		return lErr;

	int n = (int)pfInput[0];
	long hW = (long)pfInput[1];
	long hA = (long)pfInput[2];
	long hB = (long)pfInput[3];
	int nAOff = 0;
	int nBOff = 0;

	if (lInput > 4)
		nAOff = (int)pfInput[4];

	if (lInput > 5)
		nBOff = (int)pfInput[5];

	T fOutput = 0;

	if (lErr = m_math.sumsqdiff(n, hW, hA, hB, nAOff, nBOff, &fOutput))
		return lErr;

	return setOutput(fOutput, plOutput, ppfOutput);
}

template long Device<double>::cuda_sumsqdiff(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_sumsqdiff(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_width(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 6, 6))
		return lErr;

	int n = (int)pfInput[0];
	long hMean = (long)pfInput[1];
	long hMin = (long)pfInput[2];
	long hMax = (long)pfInput[3];
	T fAlpha = pfInput[4];
	long hWidth = (long)pfInput[5];

	return m_math.width(n, hMean, hMin, hMax, fAlpha, hWidth);
}

template long Device<double>::cuda_width(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_width(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_contains_point(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 5, 6))
		return lErr;

	int n = (int)pfInput[0];
	long hMean = (long)pfInput[1];
	long hWidth = (long)pfInput[2];
	long hX = (long)pfInput[3];
	long hWork = (long)pfInput[4];
	int nXOff = 0;

	if (lInput > 5)
		nXOff = (int)pfInput[5];

	T fOutput = 0;

	if (lErr = m_math.contains_point(n, hMean, hWidth, hX, hWork, &fOutput, nXOff))
		return lErr;

	return setOutput(fOutput, plOutput, ppfOutput);
}

template long Device<double>::cuda_contains_point(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_contains_point(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_denan(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 3, 3))
		return lErr;

	int n = (int)pfInput[0];
	long hX = (long)pfInput[1];
	T fReplacement = pfInput[2];

	return m_math.denan(n, hX, fReplacement);
}

template long Device<double>::cuda_denan(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_denan(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_channel_max(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 6, 6))
		return lErr;

	int n = (int)pfInput[0];
	int nOutNum = (int)pfInput[1];
	int nChannels = (int)pfInput[2];
	int nInNum = (int)pfInput[3];
	long hX = (long)pfInput[4];
	long hY = (long)pfInput[5];

	return m_math.channel_max(n, nOutNum, nChannels, nInNum, hX, hY);
}

template long Device<double>::cuda_channel_max(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_channel_max(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_channel_sub(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 6, 6))
		return lErr;

	int n = (int)pfInput[0];
	int nOutNum = (int)pfInput[1];
	int nChannels = (int)pfInput[2];
	int nInNum = (int)pfInput[3];
	long hX = (long)pfInput[4];
	long hY = (long)pfInput[5];

	return m_math.channel_sub(n, nOutNum, nChannels, nInNum, hX, hY);
}

template long Device<double>::cuda_channel_sub(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_channel_sub(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_channel_sum(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 6, 6))
		return lErr;

	int n = (int)pfInput[0];
	int nOutNum = (int)pfInput[1];
	int nChannels = (int)pfInput[2];
	int nInNum = (int)pfInput[3];
	long hX = (long)pfInput[4];
	long hY = (long)pfInput[5];

	return m_math.channel_sum(n, nOutNum, nChannels, nInNum, hX, hY);
}

template long Device<double>::cuda_channel_sum(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_channel_sum(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_channel_div(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 6, 7))
		return lErr;

	int n = (int)pfInput[0];
	int nOutNum = (int)pfInput[1];
	int nChannels = (int)pfInput[2];
	int nInNum = (int)pfInput[3];
	long hX = (long)pfInput[4];
	long hY = (long)pfInput[5];
	int nMethod = 1;

	if (lInput > 6)
		nMethod = (int)pfInput[6];

	return m_math.channel_div(n, nOutNum, nChannels, nInNum, hX, hY, nMethod);
}

template long Device<double>::cuda_channel_div(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_channel_div(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_channel_mul(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 6, 7))
		return lErr;

	int n = (int)pfInput[0];
	int nOutNum = (int)pfInput[1];
	int nChannels = (int)pfInput[2];
	int nInNum = (int)pfInput[3];
	long hX = (long)pfInput[4];
	long hY = (long)pfInput[5];
	int nMethod = 1;

	if (lInput > 6)
		nMethod = (int)pfInput[6];

	return m_math.channel_mul(n, nOutNum, nChannels, nInNum, hX, hY, nMethod);
}

template long Device<double>::cuda_channel_mul(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_channel_mul(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_channel_dot(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 7, 7))
		return lErr;

	int n = (int)pfInput[0];
	int nOutNum = (int)pfInput[1];
	int nChannels = (int)pfInput[2];
	int nInNum = (int)pfInput[3];
	long hX = (long)pfInput[4];
	long hA = (long)pfInput[5];
	long hY = (long)pfInput[6];

	return m_math.channel_dot(n, nOutNum, nChannels, nInNum, hX, hA, hY);
}

template long Device<double>::cuda_channel_dot(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_channel_dot(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_im2col(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 15, 15))
		return lErr;

	long hDataIm = (long)pfInput[0];
	int nDataImOffset = (int)pfInput[1];
	int nChannels = (int)pfInput[2];
	int nHeight = (int)pfInput[3];
	int nWidth = (int)pfInput[4];
	int nKernelH = (int)pfInput[5];
	int nKernelW = (int)pfInput[6];
	int nPadH = (int)pfInput[7];
	int nPadW = (int)pfInput[8];
	int nStrideH = (int)pfInput[9];
	int nStrideW = (int)pfInput[10];
	int nDilationH = (int)pfInput[11];
	int nDilationW = (int)pfInput[12];
	long hDataCol = (long)pfInput[13];
	int nDataColOffset = (int)pfInput[14];

	return m_math.im2col(hDataIm, nDataImOffset, nChannels, nHeight, nWidth, nKernelH, nKernelW, nPadH, nPadW, nStrideH, nStrideW, nDilationH, nDilationW, hDataCol, nDataColOffset);
}

template long Device<double>::cuda_im2col(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_im2col(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_im2col_nd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 13, 13))
		return lErr;

	long hDataIm = (long)pfInput[0];
	int nDataImOffset = (int)pfInput[1];
	int nNumSpatialAxes = (int)pfInput[2];
	int nImCount = (int)pfInput[3];
	int nChannelAxis = (int)pfInput[4];
	long hImShape = (long)pfInput[5];
	long hColShape = (long)pfInput[6];
	long hKernelShape = (long)pfInput[7];
	long hPad = (long)pfInput[8];
	long hStride = (long)pfInput[9];
	long hDilation = (long)pfInput[10];
	long hDataCol = (long)pfInput[11];
	int nDataColOffset = (int)pfInput[12];

	return m_math.im2col_nd(hDataIm, nDataImOffset, nNumSpatialAxes, nImCount, nChannelAxis, hImShape, hColShape, hKernelShape, hPad, hStride, hDilation, hDataCol, nDataColOffset);
}

template long Device<double>::cuda_im2col_nd(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_im2col_nd(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_col2im(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 15, 15))
		return lErr;

	long hDataCol = (long)pfInput[0];
	int nDataColOffset = (int)pfInput[1];
	int nChannels = (int)pfInput[2];
	int nHeight = (int)pfInput[3];
	int nWidth = (int)pfInput[4];
	int nKernelH = (int)pfInput[5];
	int nKernelW = (int)pfInput[6];
	int nPadH = (int)pfInput[7];
	int nPadW = (int)pfInput[8];
	int nStrideH = (int)pfInput[9];
	int nStrideW = (int)pfInput[10];
	int nDilationH = (int)pfInput[11];
	int nDilationW = (int)pfInput[12];
	long hDataIm = (long)pfInput[13];
	int nDataImOffset = (int)pfInput[14];

	return m_math.col2im(hDataCol, nDataColOffset, nChannels, nHeight, nWidth, nKernelH, nKernelW, nPadH, nPadW, nStrideH, nStrideW, nDilationH, nDilationW, hDataIm, nDataImOffset);
}

template long Device<double>::cuda_col2im(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_col2im(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_col2im_nd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 13, 13))
		return lErr;

	long hDataCol = (long)pfInput[0];
	int nDataColOffset = (int)pfInput[1];
	int nNumSpatialAxes = (int)pfInput[2];
	int nColCount = (int)pfInput[3];
	int nChannelAxis = (int)pfInput[4];
	long hImShape = (long)pfInput[5];
	long hColShape = (long)pfInput[6];
	long hKernelShape = (long)pfInput[7];
	long hPad = (long)pfInput[8];
	long hStride = (long)pfInput[9];
	long hDilation = (long)pfInput[10];
	long hDataIm = (long)pfInput[11];
	int nDataImOffset = (int)pfInput[12];

	return m_math.col2im_nd(hDataCol, nDataColOffset, nNumSpatialAxes, nColCount, nChannelAxis, hImShape, hColShape, hKernelShape, hPad, hStride, hDilation, hDataIm, nDataImOffset);
}

template long Device<double>::cuda_col2im_nd(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_col2im_nd(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_rng_setseed(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;

	long lSeed = (long)pfInput[0];

	return m_math.rng_setseed(lSeed);
}

template long Device<double>::cuda_rng_setseed(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_rng_setseed(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_rng_uniform(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 4, 4))
		return lErr;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	int n = (int)pfInput[0];
	T fMin = pfInput[1];
	T fMax = pfInput[2];
	long hY = (long)pfInput[3];

	return m_math.rng_uniform(n, fMin, fMax, hY);
}

template long Device<double>::cuda_rng_uniform(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_rng_uniform(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_rng_gaussian(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 4, 4))
		return lErr;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	int n = (int)pfInput[0];
	T fMu = pfInput[1];
	T fSigma = pfInput[2];
	long hY = (long)pfInput[3];

	return m_math.rng_gaussian(n, fMu, fSigma, hY);
}

template long Device<double>::cuda_rng_gaussian(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_rng_gaussian(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_rng_bernoulli(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 3, 3))
		return lErr;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	int n = (int)pfInput[0];
	T fNonZeroProb = pfInput[1];
	long hY = (long)pfInput[2];

	return m_math.rng_bernoulli(n, fNonZeroProb, hY);
}

template long Device<double>::cuda_rng_bernoulli(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_rng_bernoulli(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_sgd_update(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 5, 5))
		return lErr;

	int n = (int)pfInput[0];
	long hNetParamDiff = (long)pfInput[1];
	long hHistoryData = (long)pfInput[2];
	T fMomentum = pfInput[3];
	T fLocalRate = pfInput[4];

	return m_math.sgd_update(n, hNetParamDiff, hHistoryData, fMomentum, fLocalRate);
}

template long Device<double>::cuda_sgd_update(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_sgd_update(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_nesterov_update(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 5, 5))
		return lErr;

	int n = (int)pfInput[0];
	long hNetParamDiff = (long)pfInput[1];
	long hHistoryData = (long)pfInput[2];
	T fMomentum = pfInput[3];
	T fLocalRate = pfInput[4];

	return m_math.nesterov_update(n, hNetParamDiff, hHistoryData, fMomentum, fLocalRate);
}

template long Device<double>::cuda_nesterov_update(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_nesterov_update(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_adagrad_update(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 5, 5))
		return lErr;

	int n = (int)pfInput[0];
	long hNetParamDiff = (long)pfInput[1];
	long hHistoryData = (long)pfInput[2];
	T fDelta = pfInput[3];
	T fLocalRate = pfInput[4];

	return m_math.adagrad_update(n, hNetParamDiff, hHistoryData, fDelta, fLocalRate);
}

template long Device<double>::cuda_adagrad_update(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_adagrad_update(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_adadelta_update(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 7, 7))
		return lErr;

	int n = (int)pfInput[0];
	long hNetParamDiff = (long)pfInput[1];
	long hHistoryData1 = (long)pfInput[2];
	long hHistoryData2 = (long)pfInput[3];
	T fMomentum = pfInput[4];
	T fDelta = pfInput[5];
	T fLocalRate = pfInput[6];

	return m_math.adadelta_update(n, hNetParamDiff, hHistoryData1, hHistoryData2, fMomentum, fDelta, fLocalRate);
}

template long Device<double>::cuda_adadelta_update(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_adadelta_update(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_adam_update(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 8, 8))
		return lErr;

	int n = (int)pfInput[0];
	long hNetParamDiff = (long)pfInput[1];
	long hValM = (long)pfInput[2];
	long hValV = (long)pfInput[3];
	T fBeta1 = pfInput[4];
	T fBeta2 = pfInput[5];
	T fEpsHat = pfInput[6];
	T fCorrectedLocalRate = pfInput[7];

	return m_math.adam_update(n, hNetParamDiff, hValM, hValV, fBeta1, fBeta2, fEpsHat, fCorrectedLocalRate);
}

template long Device<double>::cuda_adam_update(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_adam_update(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_rmsprop_update(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 6, 6))
		return lErr;

	int n = (int)pfInput[0];
	long hNetParamDiff = (long)pfInput[1];
	long hHistoryData = (long)pfInput[2];
	T fRmsDecay = pfInput[3];
	T fDelta = pfInput[4];
	T fLocalRate = pfInput[5];

	return m_math.rmsprop_update(n, hNetParamDiff, hHistoryData, fRmsDecay, fDelta, fLocalRate);
}

template long Device<double>::cuda_rmsprop_update(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_rmsprop_update(long lInput, float* pfInput, long* plOutput, float** ppfOutput);



template <class T>
long Device<T>::cuda_combine_data(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 7, 7))
		return lErr;

	int nCount = (int)pfInput[0];
	long hOriginal = (long)pfInput[1];
	long hUpdated = (long)pfInput[2];
	T fUpdatedPct = pfInput[3];
	long hServer = (long)pfInput[4];
	T fServerPct = pfInput[5];
	long hNewData = (long)pfInput[6];

	return m_math.combine_data(nCount, hOriginal, hUpdated, fUpdatedPct, hServer, fServerPct, hNewData);
}

template long Device<double>::cuda_combine_data(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_combine_data(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_mtx_set_diagonal(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 4, 4))
		return lErr;

	int nCount = (int)pfInput[0];
	int nRows = (int)pfInput[1];
	T fVal = pfInput[2];
	long hData = (long)pfInput[3];

	return m_math.mtx_set_diagonal(nCount, nRows, fVal, hData);
}

template long Device<double>::cuda_mtx_set_diagonal(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_mtx_set_diagonal(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_mtx_set_diagonal2(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 6, 6))
		return lErr;

	int nCount = (int)pfInput[0];
	int nRows = (int)pfInput[1];
	long hDiagonal = (long)pfInput[2];
	T fScaleA = pfInput[3];
	T fScaleB = pfInput[4];
	long hData = (long)pfInput[5];

	return m_math.mtx_set_diagonal(nCount, nRows, hDiagonal, fScaleA, fScaleB, hData);
}

template long Device<double>::cuda_mtx_set_diagonal2(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_mtx_set_diagonal2(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_mtx_add_vector(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 7, 7))
		return lErr;

	int nOrientation = (int)pfInput[0];
	int nWidth = (int)pfInput[1];
	int nHeight = (int)pfInput[2];
	T fScale = pfInput[3];
	long hA = (long)pfInput[4];
	long hB = (long)pfInput[5];
	long hY = (long)pfInput[6];

	return m_math.mtx_add_vector(nOrientation, nWidth, nHeight, fScale, hA, hB, hY);
}

template long Device<double>::cuda_mtx_add_vector(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_mtx_add_vector(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_mtx_transpose_op(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 6, 8))
		return lErr;

	int nOp = (int)pfInput[0];
	int nWidth = (int)pfInput[1];
	int nHeight = (int)pfInput[2];
	long hA = (long)pfInput[3];
	long hB = (long)pfInput[4];
	long hY = (long)pfInput[5];
	T fScaleA = 1.0;
	T fScaleB = 1.0;

	if (lInput > 6)
		fScaleA = pfInput[6];

	if (lInput > 7)
		fScaleB = pfInput[7];

	return m_math.mtx_transpose_op(nOp, nWidth, nHeight, hA, hB, hY, fScaleA, fScaleB);
}

template long Device<double>::cuda_mtx_transpose_op(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_mtx_transpose_op(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_mtx_aggregate_cols(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 5, 5))
		return lErr;

	int nOp = (int)pfInput[0];
	int nWidth = (int)pfInput[1];
	int nHeight = (int)pfInput[2];
	long hA = (long)pfInput[3];
	long hY = (long)pfInput[4];

	return m_math.mtx_aggregate_cols(nOp, nWidth, nHeight, hA, hY);
}

template long Device<double>::cuda_mtx_aggregate_cols(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_mtx_aggregate_cols(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_mtx_aggregate_rows(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 6, 6))
		return lErr;

	int nOp = (int)pfInput[0];
	int nWidth = (int)pfInput[1];
	int nHeight = (int)pfInput[2];
	long hA = (long)pfInput[3];
	long hOnes = (long)pfInput[4];
	long hY = (long)pfInput[5];

	return m_math.mtx_aggregate_rows(nOp, nWidth, nHeight, hA, hOnes, hY);
}

template long Device<double>::cuda_mtx_aggregate_rows(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_mtx_aggregate_rows(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_mtx_transpose(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 4, 4))
		return lErr;

	int nWidth = (int)pfInput[0];
	int nHeight = (int)pfInput[1];
	long hA = (long)pfInput[2];
	long hY = (long)pfInput[3];

	return m_math.mtx_transpose(nWidth, nHeight, hA, hY);
}

template long Device<double>::cuda_mtx_transpose(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_mtx_transpose(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_mtx_meancenter_by_column(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 5, 6))
		return lErr;

	int nWidth = (int)pfInput[0];
	int nHeight = (int)pfInput[1];
	long hA = (long)pfInput[2];
	long hB = (long)pfInput[3];
	long hY = (long)pfInput[4];
	bool bNormalize = false;

	if (lInput > 5)
		bNormalize = (pfInput[5] == 0) ? false : true;

	return m_math.mtx_meancenter_by_column(nWidth, nHeight, hA, hB, hY, bNormalize);
}

template long Device<double>::cuda_mtx_meancenter_by_column(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_mtx_meancenter_by_column(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_mtx_euclidean_dist(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 7, 7))
		return lErr;

	long hX = (long)pfInput[0];
	long hY = (long)pfInput[1];
	long hOut = (long)pfInput[2];
	int n = (int)pfInput[3];
	int d = (int)pfInput[4];
	int nStart = (int)pfInput[5];
	int nEnd = (int)pfInput[6];

	return m_math.mtx_euclidean_dist(hX, hY, hOut, n, d, nStart, nEnd);
}

template long Device<double>::cuda_mtx_euclidean_dist(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_mtx_euclidean_dist(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_mtx_dot(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 6, 6))
		return lErr;

	// C(m,k) = A(m,n) * B(n,k)

	int m = (int)pfInput[0];	// rows in A, C
	int n = (int)pfInput[1];	// cols in A, rows in B
	int k = (int)pfInput[2];	// cost in C, B
	long hA = (long)pfInput[3];	// m x n matrix (m rows, n cols)
	long hB = (long)pfInput[4]; // n x k matrix (n rows, k cols)
	long hC = (long)pfInput[5]; // k x m matrix (k rows, m cols)

	return m_math.mtx_dot(m, n, k, hA, hB, hC);
}

template long Device<double>::cuda_mtx_dot(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_mtx_dot(long lInput, float* pfInput, long* plOutput, float** ppfOutput);

template <class T>
long Device<T>::cuda_tsne_update(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 9, 9))
		return lErr;

	unsigned int n = (unsigned int)pfInput[0];
	T fMomentum = pfInput[1];
	T fLearningRate = pfInput[2];
	long hdY = (long)pfInput[3];
	long huY = (long)pfInput[4];
	long hGains = (long)pfInput[5];
	long hY = (long)pfInput[6];
	T fGainFactor1 = pfInput[7];
	T fGainFactor2 = pfInput[8];

	return m_math.tsne_update(n, fMomentum, fLearningRate, hdY, huY, hGains, hY, fGainFactor1, fGainFactor2);
}

template long Device<double>::cuda_tsne_update(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_tsne_update(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_tsne_update_grad(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 5, 5))
		return lErr;

	unsigned int n = (unsigned int)pfInput[0];
	long hPosF = (long)pfInput[1];
	long hNegF = (long)pfInput[2];
	T fSumQ = pfInput[3];
	long hdC = (long)pfInput[4];

	return m_math.tsne_update_grad(n, hPosF, hNegF, fSumQ, hdC);
}

template long Device<double>::cuda_tsne_update_grad(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_tsne_update_grad(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_tsne_compute_exact_error(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 4, 4))
		return lErr;

	unsigned int n = (unsigned int)pfInput[0];
	long hP = (long)pfInput[1];
	long hQ = (long)pfInput[2];
	long hY = (long)pfInput[3];

	return m_math.tsne_compute_exact_error(n, hP, hQ, hY);
}

template long Device<double>::cuda_tsne_compute_exact_error(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_tsne_compute_exact_error(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_tsne_compute_squared_euclidean_distance(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 5, 5))
		return lErr;

	unsigned int n = (unsigned int)pfInput[0];
	unsigned int d = (unsigned int)pfInput[1];
//	long hW = (long)pfInput[2];  // currently not used.
	long hX = (long)pfInput[3];
	long hDD = (long)pfInput[4];

	HostBuffer<T>* pDD = m_memory.GetHostBuffer(hDD);
	T* pX_on_host = m_memory.GetMemoryToHost(hX);

	lErr = m_math.tsne_compute_squared_euclidean_distance(n, d, pX_on_host, pDD->Data());

	if (pX_on_host != NULL)
		m_memory.FreeHost(pX_on_host);

	return lErr;
}

template long Device<double>::cuda_tsne_compute_squared_euclidean_distance(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_tsne_compute_squared_euclidean_distance(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_tsne_compute_q_matrix(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 4, 4))
		return lErr;

	unsigned int n = (unsigned int)pfInput[0];
	long hDD = (long)pfInput[1];
	long hQ = (long)pfInput[2];
	bool bQisHostMem = (pfInput[3] == 1.0) ? true : false;

	HostBuffer<T>* pDD = m_memory.GetHostBuffer(hDD);
	T* pQ_on_host = NULL;
	T fSumQ = T(0.000001);

	if (bQisHostMem)
		pQ_on_host = m_memory.GetHostBuffer(hQ)->Data();
	else
		pQ_on_host = m_memory.GetMemoryToHost(hQ);

	lErr = m_math.tsne_compute_q_matrix(n, pDD->Data(), pQ_on_host, &fSumQ);

	if (!lErr)
		lErr = m_memory.SetMemory(hQ, pQ_on_host, n * n, -1);

	if (!bQisHostMem && pQ_on_host != NULL)
		m_memory.FreeHost(pQ_on_host);

	return setOutput(fSumQ, plOutput, ppfOutput);
}

template long Device<double>::cuda_tsne_compute_q_matrix(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_tsne_compute_q_matrix(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_tsne_compute_exact_gradient(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 8, 8))
		return lErr;

	unsigned int n = (unsigned int)pfInput[0];
	unsigned int d = (unsigned int)pfInput[1];
	long hY = (long)pfInput[2];
	long hP = (long)pfInput[3];
	long hQ = (long)pfInput[4];
	bool bQisHostMem = (pfInput[5] == 1.0) ? true : false;
	long hdC = (long)pfInput[6];
	T fSumQ = pfInput[7];

	T* pY_on_host = m_memory.GetMemoryToHost(hY);
	T* pP_on_host = m_memory.GetMemoryToHost(hP);
	T* pdC_on_host = m_memory.GetMemoryToHost(hdC);
	T* pQ_on_host = NULL;

	if (bQisHostMem)
		pQ_on_host = m_memory.GetHostBuffer(hQ)->Data();
	else
		pQ_on_host = m_memory.GetMemoryToHost(hQ);

	if (pY_on_host == NULL || pP_on_host == NULL || pdC_on_host == NULL || pQ_on_host == NULL)
		lErr = ERROR_MEMORY_OUT;

	if (!lErr)
		lErr = m_math.tsne_compute_exact_gradient(n, d, pY_on_host, pP_on_host, pQ_on_host, pdC_on_host, fSumQ);

	if (!lErr)
		lErr = m_memory.SetMemory(hdC, pdC_on_host, -1, -1);

	if (!bQisHostMem && pQ_on_host != NULL)
		m_memory.FreeHost(pQ_on_host);

	if (pY_on_host != NULL)
		m_memory.FreeHost(pY_on_host);

	if (pP_on_host != NULL)
		m_memory.FreeHost(pP_on_host);

	if (pdC_on_host != NULL)
		m_memory.FreeHost(pdC_on_host);

	return 0;
}

template long Device<double>::cuda_tsne_compute_exact_gradient(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_tsne_compute_exact_gradient(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_tsne_symmetrize_matrix(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 4, 4))
		return lErr;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	unsigned int n = (unsigned int)pfInput[0];
	long hRowP = (long)pfInput[1];
	long hColP = (long)pfInput[2];
	long hValP = (long)pfInput[3];
	unsigned int nRowCount = 0;

	if (lErr = m_math.tsne_symmetrize_matrix(n, hRowP, hColP, hValP, &nRowCount))
		return lErr;

	return setOutput(T(nRowCount), plOutput, ppfOutput);
}

template long Device<double>::cuda_tsne_symmetrize_matrix(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_tsne_symmetrize_matrix(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_tsne_compute_knn_bounds(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 3, 3))
		return lErr;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	unsigned int n = (unsigned int)pfInput[0];
	long hData = (long)pfInput[1];
	T fPctInCircle = pfInput[2];
	T fMinX;
	T fMinY;
	T fMaxX;
	T fMaxY;

	if (lErr = m_math.tsne_compute_knn_bounds(n, hData, fPctInCircle, &fMinX, &fMinY, &fMaxX, &fMaxY))
		return lErr;

	T* pfOutput = NULL;

	if (lErr = m_memory.AllocHost(4, &pfOutput, NULL, false))
		return lErr;

	pfOutput[0] = fMinX;
	pfOutput[1] = fMinY;
	pfOutput[2] = fMaxX;
	pfOutput[3] = fMaxY;

	*ppfOutput = pfOutput;
	*plOutput = 4;

	return 0;
}

template long Device<double>::cuda_tsne_compute_knn_bounds(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_tsne_compute_knn_bounds(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_guassian_blur(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 7, 7))
		return lErr;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	int n = (int)pfInput[0];
	int c = (int)pfInput[1];
	int h = (int)pfInput[2];
	int w = (int)pfInput[3];
	T fSigma = pfInput[4];
	long hX = (long)pfInput[5];
	long hY = (long)pfInput[6];

	return m_math.gaussian_blur(n, c, h, w, fSigma, hX, hY);
}

template long Device<double>::cuda_guassian_blur(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_guassian_blur(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_hamming_diff(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 5, 8))
		return lErr;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	int n = (int)pfInput[0];
	T fThreshold = pfInput[1];
	long hA = (long)pfInput[2];
	long hB = (long)pfInput[3];
	long hY = (long)pfInput[4];
	int nOffA = 0;
	int nOffB = 0;
	int nOffY = 0;

	if (lInput > 5)
		nOffA = (int)pfInput[5];

	if (lInput > 6)
		nOffB = (int)pfInput[6];

	if (lInput > 7)
		nOffY = (int)pfInput[7];

	return m_math.hamming_diff(n, fThreshold, hA, hB, hY, nOffA, nOffB, nOffY);
}

template long Device<double>::cuda_hamming_diff(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_hamming_diff(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_calc_batch_dist(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 9, SHRT_MAX))
		return lErr;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	int nDistMethod = (int)pfInput[0];
	T fThreshold = pfInput[1];
	int nItemDim = (int)pfInput[2];
	long hSrc = (long)pfInput[3];
	long hTargets = (long)pfInput[4];
	long hWork = (long)pfInput[5];
	int nDim0 = (int)pfInput[6];
	int nDim1 = (int)pfInput[7];

	if (nDim1 != 2)
		return ERROR_PARAM_OUT_OF_RANGE;

	if (nDim0 < 0)
		return ERROR_PARAM_OUT_OF_RANGE;

	T* pfOutput = NULL;
	if (lErr = m_memory.AllocHost(nDim0, &pfOutput, NULL, false))
		return lErr;

	lErr = m_math.calc_batch_dist(nDistMethod, fThreshold, nItemDim, hSrc, hTargets, hWork, nDim0, nDim1, &pfInput[8], pfOutput);

	if (!lErr)
	{
		*ppfOutput = pfOutput;
		*plOutput = nDim0;
	}
	else
	{
		m_memory.FreeHost(pfOutput);
	}

	return lErr;
}

template long Device<double>::cuda_calc_batch_dist(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_calc_batch_dist(long lInput, float* pfInput, long* plOutput, float** ppfOutput);

//end device.cu