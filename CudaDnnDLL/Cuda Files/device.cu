//=============================================================================
//	FILE:	device.cu
//
//	DESC:	This file implements the base class used to manage the underlying
//			GPU device.
//=============================================================================

#include "device.h"
#include <nvapi.h>
#include <nvml.h>
#include <string>

#define USES_CONVERSION_SIMPLE int _convert; UINT _acp = ATL::_AtlGetConversionACP() /*CP_THREAD_ACP*/; LPCSTR _lpa;

//=============================================================================
//	Class Methods - HwInfo
//=============================================================================

template <class T>
long HwInfo<T>::Initialize(HANDLE hEvtSrc)
{
	m_hEventSrc = hEvtSrc;

	if (!m_bInitializedNvml)
	{
		nvmlReturn_t res;
		if ((res = nvmlInit_v2()) != NVML_SUCCESS)
		{
			LPCSTR pszErr = "NVML Initializing...";
			ReportEventA(m_hEventSrc, EVENTLOG_ERROR_TYPE, 0, ERROR_NOT_IMPLEMENTED, NULL, 1, 0, &pszErr, NULL);
			return res;
		}

		m_bInitializedNvml = TRUE;
	}

	if (!m_bInitializedNvApi)
	{
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

		if (m_gpuWdmHandles == NULL)
		{
			if ((m_gpuWdmHandles = malloc(sizeof(NvPhysicalGpuHandle) * 256)) == NULL)
				return ERROR_MEMORY_OUT;
		}

		if (m_gpuTccHandles == NULL)
		{
			if ((m_gpuTccHandles = malloc(sizeof(NvPhysicalGpuHandle) * 256)) == NULL)
				return ERROR_MEMORY_OUT;
		}

		NvU32 numOfWdmGPUs;
		if ((status = NvAPI_EnumPhysicalGPUs((NvPhysicalGpuHandle*)m_gpuWdmHandles, &numOfWdmGPUs)) != NVAPI_OK)
		{
			LPCSTR pszErr = "NvAPI Enumerating Physical GPUs...";
			ReportEventA(m_hEventSrc, EVENTLOG_ERROR_TYPE, 0, ERROR_NOT_IMPLEMENTED, NULL, 1, 0, &pszErr, NULL);
			NvAPI_ShortString szErr;
			NvAPI_GetErrorMessage(status, szErr);
			ReportEventA(m_hEventSrc, EVENTLOG_ERROR_TYPE, 0, ERROR_NOT_IMPLEMENTED, NULL, 1, 0, &pszErr, NULL);
			return status;
		}

		m_nNumWdmGpus = (int)numOfWdmGPUs;

		NvU32 numOfTccGPUs;
		if ((status = NvAPI_EnumTCCPhysicalGPUs((NvPhysicalGpuHandle*)m_gpuTccHandles, &numOfTccGPUs)) != NVAPI_OK)
		{
			LPCSTR pszErr = "NvAPI Enumerating Physical GPUs...";
			ReportEventA(m_hEventSrc, EVENTLOG_ERROR_TYPE, 0, ERROR_NOT_IMPLEMENTED, NULL, 1, 0, &pszErr, NULL);
			NvAPI_ShortString szErr;
			NvAPI_GetErrorMessage(status, szErr);
			ReportEventA(m_hEventSrc, EVENTLOG_ERROR_TYPE, 0, ERROR_NOT_IMPLEMENTED, NULL, 1, 0, &pszErr, NULL);
			return status;
		}

		m_nNumTccGpus = (int)numOfTccGPUs;

		m_bInitializedNvApi = TRUE;
	}

	return 0;
}

template <class T>
long HwInfo<T>::CleanUp()
{
	if (m_bInitializedNvml)
	{
		nvmlShutdown();
		m_bInitializedNvml = FALSE;
	}

	if (m_bInitializedNvApi)
	{
		NvAPI_Unload();
		m_bInitializedNvApi = FALSE;
	}

	return 0;
}

template <class T>
long HwInfo<T>::FindDevice(int nDevice)
{
	USES_CONVERSION_SIMPLE;
	int nIdxWdm = -1;
	int nIdxTcc = -1;
	NvAPI_Status status;
	LONG lErr;
	char rgPciID[256];

	if (lErr = cudaDeviceGetPCIBusId(rgPciID, 255, nDevice))
		return lErr;

	if (m_bInitializedNvml)
	{
		nvmlReturn_t res;

		if ((res = nvmlDeviceGetHandleByPciBusId_v2(rgPciID, (nvmlDevice_t*)&m_device)) != NVML_SUCCESS)
		{
			LPCSTR pszErr = "NVML Get Device Handle by PCI Bus ID...";
			ReportEventA(m_hEventSrc, EVENTLOG_ERROR_TYPE, 0, ERROR_NOT_IMPLEMENTED, NULL, 1, 0, &pszErr, NULL);
			return res;
		}
	}

	if (m_bInitializedNvApi)
	{
		char* psz = strtok(rgPciID, ":");
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

		for (int i = 0; i < m_nNumWdmGpus; i++)
		{
			NvU32 busID = 0;

			if ((status = NvAPI_GPU_GetBusId(((NvPhysicalGpuHandle*)m_gpuWdmHandles)[i], &busID)) != NVAPI_OK)
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
				nIdxWdm = i;
				break;
			}
		}

		if (nIdxWdm == -1)
		{
			for (int i = 0; i < m_nNumTccGpus; i++)
			{
				NvU32 busID = 0;

				if ((status = NvAPI_GPU_GetBusId(((NvPhysicalGpuHandle*)m_gpuTccHandles)[i], &busID)) != NVAPI_OK)
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
	}

	m_nIdxWdm = nIdxWdm;
	m_nIdxTcc = nIdxTcc;

	return 0;
}

template <class T>
long HwInfo<T>::GetConnectedDisplays(int* pnDisplayCount)
{
	NvU32 connectedDisplays = 0;
	NvAPI_Status status;	
	bool bHandled = false;

	if (m_bInitializedNvApi)
	{
		if (m_gpuWdmHandles == NULL)
			return NVAPI_API_NOT_INITIALIZED;

		if (m_nIdxWdm >= 0)
		{
			if ((status = NvAPI_GPU_GetConnectedDisplayIds(((NvPhysicalGpuHandle*)m_gpuWdmHandles)[m_nIdxWdm], NULL, &connectedDisplays, NULL)) != NVAPI_OK)
			{
				LPCSTR pszErr = "NvAPI Getting Connected Display Ids...";
				ReportEventA(m_hEventSrc, EVENTLOG_ERROR_TYPE, 0, ERROR_NOT_IMPLEMENTED, NULL, 1, 0, &pszErr, NULL);
				NvAPI_ShortString szErr;
				NvAPI_GetErrorMessage(status, szErr);
				ReportEventA(m_hEventSrc, EVENTLOG_ERROR_TYPE, 0, ERROR_NOT_IMPLEMENTED, NULL, 1, 0, &pszErr, NULL);
				return status;
			}

			bHandled = true;
		}
		else if (m_nIdxTcc >= 0)
		{
			bHandled = true;
		}
	}

	if (!bHandled)
	{
		if (m_bInitializedNvml && m_device != NULL && connectedDisplays == 0)
		{
			nvmlReturn_t res;
			nvmlEnableState_t active;

			if ((res = nvmlDeviceGetDisplayMode((nvmlDevice_t)m_device, &active)) == NVML_SUCCESS)
			{
				if (active == NVML_FEATURE_ENABLED)
					connectedDisplays = 1;
			}
		}
	}

	*pnDisplayCount = (int)connectedDisplays;
	return 0;
}

template <class T>
long HwInfo<T>::GetDeviceTemperature(int* pnTemp)
{
	NvAPI_Status status = NVAPI_API_NOT_INITIALIZED;
	int nTemp = -1;

	if (m_bInitializedNvApi)
	{
		if (m_gpuWdmHandles == NULL)
			return NVAPI_API_NOT_INITIALIZED;

		if (m_gpuTccHandles == NULL)
			return NVAPI_API_NOT_INITIALIZED;

		NV_GPU_THERMAL_SETTINGS thermal;
		thermal.version = NV_GPU_THERMAL_SETTINGS_VER;

		status = NVAPI_ACCESS_DENIED;
		if (m_nIdxWdm >= 0)
			status = NvAPI_GPU_GetThermalSettings(((NvPhysicalGpuHandle*)m_gpuWdmHandles)[m_nIdxWdm], 0, &thermal);
		else if (m_nIdxTcc >= 0)
			status = NvAPI_GPU_GetThermalSettings(((NvPhysicalGpuHandle*)m_gpuTccHandles)[m_nIdxTcc], 0, &thermal);

		if (status == NVAPI_OK)
		{
			if (thermal.count > 0)
				nTemp = (int)thermal.sensor[0].currentTemp;
		}
	}

	if (m_bInitializedNvml && m_device != NULL && status != NVAPI_OK)
	{
		nvmlReturn_t res;

		if ((res = nvmlDeviceGetTemperature((nvmlDevice_t)m_device, NVML_TEMPERATURE_GPU, (unsigned int*)&nTemp)) != NVML_SUCCESS)
		{
			LPCSTR pszErr = "NvAPI Getting Thermal Settings...";
			ReportEventA(m_hEventSrc, EVENTLOG_ERROR_TYPE, 0, ERROR_NOT_IMPLEMENTED, NULL, 1, 0, &pszErr, NULL);
			return res;
		}
	}

	*pnTemp = nTemp;

	return 0;
}

template <class T>
long HwInfo<T>::GetDeviceUtilization(int* pnUtilization)
{
	NvAPI_Status status = NVAPI_API_NOT_INITIALIZED;
	int nUtilization = -1;

	if (m_bInitializedNvApi)
	{
		if (m_gpuWdmHandles == NULL)
			return NVAPI_API_NOT_INITIALIZED;

		if (m_gpuTccHandles == NULL)
			return NVAPI_API_NOT_INITIALIZED;

		NV_GPU_DYNAMIC_PSTATES_INFO_EX states;
		states.version = NV_GPU_DYNAMIC_PSTATES_INFO_EX_VER;

		status = NVAPI_ACCESS_DENIED;
		if (m_nIdxWdm >= 0)
			status = NvAPI_GPU_GetDynamicPstatesInfoEx(((NvPhysicalGpuHandle*)m_gpuWdmHandles)[m_nIdxWdm], &states);
		else if (m_nIdxTcc >= 0)
			status = NvAPI_GPU_GetDynamicPstatesInfoEx(((NvPhysicalGpuHandle*)m_gpuTccHandles)[m_nIdxTcc], &states);

		if (status == NVAPI_OK)
		{
			if (states.utilization[0].bIsPresent)
			{
				double dfUtilization = (double)states.utilization[0].percentage;
				nUtilization = (int)dfUtilization;
			}
		}
	}

	if (m_bInitializedNvml && m_device != NULL && status != NVAPI_OK)
	{
		nvmlReturn_t res;

		unsigned int nPower;
		if ((res = nvmlDeviceGetPowerUsage((nvmlDevice_t)m_device, &nPower)) != NVML_SUCCESS)
		{
			LPCSTR pszErr = "NvAPI Getting Dynamic PStates Info...";
			ReportEventA(m_hEventSrc, EVENTLOG_ERROR_TYPE, 0, ERROR_NOT_IMPLEMENTED, NULL, 1, 0, &pszErr, NULL);
			return res;
		}

		unsigned int nLimit;
		if ((res = nvmlDeviceGetEnforcedPowerLimit((nvmlDevice_t)m_device, &nLimit)) != NVML_SUCCESS)
		{
			LPCSTR pszErr = "NvAPI Getting Dynamic PStates Info...";
			ReportEventA(m_hEventSrc, EVENTLOG_ERROR_TYPE, 0, ERROR_NOT_IMPLEMENTED, NULL, 1, 0, &pszErr, NULL);
			return res;
		}

		double dfPct = (double)nPower / (double)nLimit;
		nUtilization = (unsigned int)(dfPct * 100);
	}

	*pnUtilization = nUtilization;
	return 0;
}


//=============================================================================
//	Class Methods - Device
//=============================================================================

template <class T>
long Device<T>::Initialize()
{
	LONG lErr;
	if ((lErr = m_hwInfo.Initialize(m_hEventSrc)) != 0)
		return lErr;

	m_bInitialized = TRUE;

	return NOERROR;
}

template long Device<double>::Initialize();
template long Device<float>::Initialize();


template <class T>
long Device<T>::CleanUp()
{
	m_hwInfo.CleanUp();
	m_bInitialized = FALSE;
	return NOERROR;
}

template long Device<double>::CleanUp();
template long Device<float>::CleanUp();

template <class T>
long Device<T>::CanAccessPeer(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
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

template long Device<double>::CanAccessPeer(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::CanAccessPeer(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::EnablePeerAccess(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	long lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;

	int nSrcDeviceID = (int)pfInput[0];

	return cudaDeviceEnablePeerAccess(nSrcDeviceID, 0);
}

template long Device<double>::EnablePeerAccess(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::EnablePeerAccess(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::DisablePeerAccess(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	long lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;

	int nSrcDeviceID = (int)pfInput[0];

	return cudaDeviceDisablePeerAccess(nSrcDeviceID);
}

template long Device<double>::DisablePeerAccess(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::DisablePeerAccess(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::GetDeviceName(int nDevice, LPTSTR* pszDevice)
{
	USES_CONVERSION_SIMPLE;
	LONG lErr;
	cudaDeviceProp prop;

	if (lErr = cudaGetDeviceProperties(&prop, nDevice))
		return lErr;

	m_nMajor = prop.major;
	m_nMinor = prop.minor;

	bool b64Bit = (sizeof(void*) == 8) ? true : false;
	bool bTcc = (prop.tccDriver == 1) ? true : false;
	bool bVer = (prop.major >= 2) ? true : false;
	double dfGB = (double)prop.totalGlobalMem / 1000000000.00;

	LPTSTR pDst = (LPTSTR)malloc(sizeof(TCHAR) * 512);
	if (pDst == NULL)
		return ERROR_OUTOFMEMORY;

	memset(pDst, 0, sizeof(TCHAR) * 512);
	_sntprintf(pDst, 511, _T("%s (%d.%02d GB - %s; compute %d.%d)"), A2T(prop.name), (int)dfGB, (int)(dfGB * 100) % 100, (b64Bit && bTcc && bVer) ? _T("P2P on") : _T("P2P off"), m_nMajor, m_nMinor);
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
			if (prop.minor >= 1)
				return mp * 128;
			if (prop.minor == 0)
				return mp * 64;
			break;

		case 7: // Turing
			return mp * 64;
	}

	return -1;
}

template <class T>
long Device<T>::GetDeviceP2PInfo(int nDevice, LPTSTR* pszDevice)
{
	USES_CONVERSION_SIMPLE;
	LONG lErr;
	cudaDeviceProp prop;

	if (lErr = cudaGetDeviceProperties(&prop, nDevice))
		return lErr;

	m_nMajor = prop.major;
	m_nMinor = prop.minor;

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
	USES_CONVERSION_SIMPLE;
	LONG lErr;

	if (lErr = m_hwInfo.FindDevice(nDevice))
		return lErr;

	int nConnectedDisplays;
	if (lErr = m_hwInfo.GetConnectedDisplays(&nConnectedDisplays))
		return lErr;

	int nTemp;
	if (lErr = m_hwInfo.GetDeviceTemperature(&nTemp))
		return lErr;

	int nUtilization;
	if (lErr = m_hwInfo.GetDeviceUtilization(&nUtilization))
		return lErr;

	LPTSTR pDst = (LPTSTR)malloc(sizeof(TCHAR) * 2048);
	if (pDst == NULL)
		return ERROR_OUTOFMEMORY;

	memset(pDst, 0, sizeof(TCHAR) * 2048);
	char szTmp[16];
	_snprintf(szTmp, 16, "%c", (char)176);
	_sntprintf(pDst, 2047, _T(" GPU = %d, MonitorOn = %s, GPU_Temp = %d C%s, GPU_Use = %d%%"), nDevice, (nConnectedDisplays == 0) ? _T("NO") : _T("YES"), nTemp, A2T(szTmp), nUtilization);
	*pszDevice = pDst;

	if (bVerbose)
	{
		cudaDeviceProp prop;
		if (lErr = cudaGetDeviceProperties(&prop, nDevice))
			return lErr;

		m_nMajor = prop.major;
		m_nMinor = prop.minor;

		float fDriverVer = 0;
		NvU32 v;
		NvAPI_ShortString bVer;
		NvAPI_Status status;
		status = NvAPI_SYS_GetDriverAndBranchVersion(&v, bVer);
		if (status == NVAPI_OK)
			fDriverVer = (float)v / 100.0f;

		char szBuffer[1024];
		_snprintf(szBuffer, 1023, ",\r\n Major: %d, Minor: %d, Compute Mode: %d,\r\n Max Grid: { %d, %d, %d }, Max Thread Dim: { %d, %d, %d },\r\n Shared Memory/Block: %zd,\r\n Driver Version: %.2f", prop.major, prop.minor, prop.computeMode, prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2], prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2], prop.sharedMemPerBlock, fDriverVer);
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
long Device<T>::SetDevice(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
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

template long Device<double>::SetDevice(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::SetDevice(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


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
long Device<T>::AllocMemory(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (pfInput != NULL && lInput > 1)
	{
		if (lErr = verifyInput(lInput, pfInput, 1, INT_MAX))
			return lErr;
	}
	if (lErr = verifyInput(llInput, plInput, 1, 2))
		return lErr;

	long hHandle = 0;
	long hStream = 0;
	size_t lCount = (size_t)plInput[0];
	T* pSrc = NULL;

	if (lInput > 1)
	{
		if (lInput == lCount + 1)
		{
			pSrc = &pfInput[1];
		}
		else if (lInput == lCount + 2)
		{
			hStream = (long)plInput[1];
			pSrc = &pfInput[2];
		}
		else
		{
			return ERROR_PARAM_OUT_OF_RANGE;
		}
	}

	if (lErr = m_memory.AllocMemory(GetDevice(), false, lCount, pSrc, hStream, &hHandle))
		return lErr;

	if (lErr = setOutput(hHandle, plOutput, ppfOutput))
		return lErr;

	return 0;
}

template long Device<double>::AllocMemory(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::AllocMemory(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::AllocMemoryHalf(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (pfInput != NULL && lInput > 1)
	{
		if (lErr = verifyInput(lInput, pfInput, 1, INT_MAX))
			return lErr;
	}
	if (lErr = verifyInput(llInput, plInput, 1, 2))
		return lErr;

	long hHandle = 0;
	long hStream = 0;
	size_t lCount = (size_t)plInput[0];
	T* pSrc = NULL;

	if (lInput > 1)
	{
		if (lInput == lCount + 1)
		{
			pSrc = &pfInput[1];
		}
		else if (lInput == lCount + 2)
		{
			hStream = (long)plInput[1];
			pSrc = &pfInput[2];
		}
		else
		{
			return ERROR_PARAM_OUT_OF_RANGE;
		}
	}

	if (lErr = m_memory.AllocMemory(GetDevice(), true, lCount, pSrc, hStream, &hHandle))
		return lErr;

	if (lErr = setOutput(hHandle, plOutput, ppfOutput))
		return lErr;

	return 0;
}

template long Device<double>::AllocMemoryHalf(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::AllocMemoryHalf(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::FreeMemory(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 1, 1))
		return lErr;

	long hHandle = (long)plInput[0];

	return m_memory.FreeMemory(hHandle);
}

template long Device<double>::FreeMemory(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::FreeMemory(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::GetMemory(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 1, 2))
		return lErr;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	long hHandle = (long)plInput[0];
	LONGLONG lCount = 0;
	MemoryItem* pItem;

	if (llInput > 1)
		lCount = plInput[1];

	if (lErr = m_memory.GetMemory(hHandle, &pItem))
		return lErr;

	size_t lAllocatedCount = pItem->Size() / (pItem->IsHalf() ? sizeof(__half) : sizeof(T));

	if (lCount < 0)
		lCount = lAllocatedCount;
	else if (lCount > lAllocatedCount)
		return ERROR_PARAM_OUT_OF_RANGE;

	if (lCount <= *plOutput)
	{
		T* pfOutput = *ppfOutput;

		if (lErr = m_memory.CopyToHost(lCount, pfOutput, (T*)pItem->Data(), true, pItem->IsHalf()))
			return lErr;
	}
	else
	{
		T* pfOutput = NULL;

		if (lErr = m_memory.AllocHost(lCount, &pfOutput, (T*)pItem->Data(), true, pItem->IsHalf(), true))
			return lErr;

		*ppfOutput = pfOutput;
	}

	if ((long)lCount < 0)
		return ERROR_PARAM_OUT_OF_RANGE;

	*plOutput = (long)lCount;
	return 0;
}

template long Device<double>::GetMemory(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::GetMemory(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::SetMemory(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 2, INT_MAX))
		return lErr;
	if (lErr = verifyInput(llInput, plInput, 2, 2))
		return lErr;

	long hHandle = (long)plInput[0];
	size_t lCount = (size_t)plInput[1];
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

	// Lock in critical section from this point to end of function.
	if (!m_bMemHostLockInit)
		return ERROR_DLL_NOT_INIT;

	Lock lock(&m_MemHostLock);

	if (m_hSetMemHost == 0)
	{
		if (lErr = m_memory.AllocHostBuffer(INITIAL_SET_MEM_BUFFER, &m_hSetMemHost))
			return lErr;
	}

	HostBuffer<T>* pHost = m_memory.GetHostBuffer(m_hSetMemHost);
	if (pHost == NULL)
		return ERROR_MEMORY_NOT_FOUND;

	if (pHost->Count() < lCount)
	{
		size_t lNewSize = ((lCount / INITIAL_SET_MEM_BUFFER) + 2) * INITIAL_SET_MEM_BUFFER;
		m_memory.FreeHostBuffer(m_hSetMemHost);
		m_hSetMemHost = 0;

		if (lErr = m_memory.AllocHostBuffer(lNewSize, &m_hSetMemHost))
			return lErr;

		if ((pHost = m_memory.GetHostBuffer(m_hSetMemHost)) == NULL)
			return ERROR_MEMORY_NOT_FOUND;
	}

	if (lErr = m_memory.CopyToHost(lCount, pHost->Data(), pData, false, false))
		return lErr;

	if (lErr = m_memory.SetMemory(hHandle, pHost->Data(), lCount, hStream))
		return lErr;

	return 0;
}

template long Device<double>::SetMemory(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::SetMemory(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::SetMemoryAt(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 3, INT_MAX))
		return lErr;
	if (lErr = verifyInput(llInput, plInput, 3, 3))
		return lErr;

	long hHandle = (long)plInput[0];
	size_t lCount = (size_t)plInput[1];
	size_t nOffset = (size_t)plInput[2];
	T* pData = &pfInput[3];

	if (lErr = m_memory.SetMemoryAt(hHandle, pData, lCount, nOffset))
		return lErr;

	return 0;
}

template long Device<double>::SetMemoryAt(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::SetMemoryAt(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::SetPixel(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 5, INT_MAX))
		return lErr;

	long hHandle = (long)pfInput[0];
	size_t lCount = (size_t)pfInput[1];
	bool bReturnOriginal = (pfInput[2] == 0) ? false : true;
	int nOffset = (int)pfInput[3];
	size_t lPixels = (size_t)pfInput[4];

	T* pPixels = &pfInput[5];

	if (bReturnOriginal)
	{
		if (lErr = verifyOutput(plOutput, ppfOutput))
			return lErr;

		if (lErr = GetPixel(lInput, pfInput, llInput, plInput, plOutput, ppfOutput))
			return lErr;
	}

	for (int i = 0; i < lPixels; i++)
	{
		int nPixelIdx = i * 2;
		int nIdx = (int)pPixels[nPixelIdx];
		T fVal = pPixels[nPixelIdx + 1];

		if (lErr = m_memory.SetMemoryAt(hHandle, &fVal, 1, nIdx + nOffset))
			return lErr;
	}

	return 0;
}

template long Device<double>::SetPixel(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::SetPixel(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::GetPixel(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 5, INT_MAX))
		return lErr;

	long hHandle = (long)pfInput[0];
	size_t lCount = (size_t)pfInput[1];
	bool bReturnOriginal = (pfInput[2] == 0) ? false : true;
	int nOffset = (int)pfInput[3];
	size_t lPixels = (size_t)pfInput[4];
	if ((long)lPixels < 0)
		return ERROR_PARAM_OUT_OF_RANGE;

	T* pPixels = &pfInput[5];
	T* pfOutput = *ppfOutput;

	if (lPixels > *plOutput)
	{
		if (lErr = m_memory.AllocHost(lPixels, &pfOutput, NULL, false, false, true))
			return lErr;
	}

	for (int i = 0; i < lPixels; i++)
	{
		int nPixelIdx = i * 2;
		int nIdx = (int)pPixels[nPixelIdx];

		if (lErr = m_memory.GetMemoryAt(hHandle, &pfOutput[i], 1, nIdx + nOffset))
			return lErr;
	}

	*plOutput = (long)lPixels;
	return 0;
}

template long Device<double>::GetPixel(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::GetPixel(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::CopyGpuToHostBuffer(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 3, 3))
		return lErr;

	long lCount = (long)plInput[0];
	long hGpuSrc = (long)plInput[1];
	long hHostDst = (long)plInput[2];

	if (lErr = m_memory.CopyGpuToHost(lCount, hGpuSrc, hHostDst))
		return lErr;

	return 0;
}

template long Device<double>::CopyGpuToHostBuffer(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::CopyGpuToHostBuffer(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::CopyHostBufferToGpu(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 3, 3))
		return lErr;

	long lCount = (long)plInput[0];
	long hHostSrc = (long)plInput[1];
	long hGpuDst = (long)plInput[2];

	if (lErr = m_memory.CopyHostToGpu(lCount, hHostSrc, hGpuDst))
		return lErr;

	return 0;
}

template long Device<double>::CopyHostBufferToGpu(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::CopyHostBufferToGpu(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::AllocHostBuffer(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 1, 1))
		return lErr;

	long hHandle = 0;
	long lCount = (long)plInput[0];

	if (lErr = m_memory.AllocHostBuffer(lCount, &hHandle))
		return lErr;

	if (lErr = setOutput(hHandle, plOutput, ppfOutput))
		return lErr;

	return 0;
}

template long Device<double>::AllocHostBuffer(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::AllocHostBuffer(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::FreeHostBuffer(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 1, 1))
		return lErr;

	long hHandle = (long)plInput[0];

	return m_memory.FreeHostBuffer(hHandle);
}

template long Device<double>::FreeHostBuffer(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::FreeHostBuffer(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::GetHostBufferCapacity(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 1, 1))
		return lErr;

	long hHandle = (long)plInput[0];

	HostBuffer<T>* pbuf = m_memory.GetHostBuffer(hHandle);
	if (pbuf == NULL)
		return ERROR_PARAM_OUT_OF_RANGE;

	if (lErr = setOutput((T)pbuf->Count(), plOutput, ppfOutput))
		return lErr;

	return 0;
}

template long Device<double>::GetHostBufferCapacity(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::GetHostBufferCapacity(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::GetHostMemory(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 1, 1))
		return lErr;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	long hHandle = (long)plInput[0];

	HostBuffer<T>* pHostBuf = m_memory.GetHostBuffer(hHandle);

	if (pHostBuf != NULL)
	{
		long lCount = (long)pHostBuf->Count();
		if (lCount < 0)
			return ERROR_PARAM_OUT_OF_RANGE;

		*plOutput = lCount;
		*ppfOutput = pHostBuf->Data();
	}
	else
	{
		*plOutput = 0;
		*ppfOutput = NULL;
	}

	return 0;
}

template long Device<double>::GetHostMemory(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::GetHostMemory(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);



template <class T>
long Device<T>::SetHostMemory(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 3, INT_MAX))
		return lErr;
	if (lErr = verifyInput(llInput, plInput, 2, INT_MAX))
		return lErr;

	long hHandle = (long)plInput[0];
	long lCount = (long)plInput[1];
	T* pData = &pfInput[2];

	return m_memory.SetHostBuffer(hHandle, lCount, pData);
}

template long Device<double>::SetHostMemory(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::SetHostMemory(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::RunMemoryTest(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 8, 8))
		return lErr;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	long hHandle = (long)plInput[0];
	MEMTEST_TYPE memTestType = (MEMTEST_TYPE)(int)plInput[1];
	size_t szStartOffset = (size_t)plInput[2];
	size_t szCount = (size_t)plInput[3];
	bool bVerbose = (plInput[4] == 0) ? false : true;
	bool bWrite = (plInput[5] == 0) ? false : true;
	bool bReadWrite = (plInput[6] == 0) ? false : true;
	bool bRead = (plInput[7] == 0) ? false : true;

	return m_memory.RunMemoryTest(hHandle, memTestType, szStartOffset, szCount, plOutput, ppfOutput, bVerbose, bWrite, bReadWrite, bRead);
}

template long Device<double>::RunMemoryTest(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::RunMemoryTest(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);

template <class T>
long Device<T>::DistortImage(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 6, 6))
		return lErr;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	long hHandle = (long)plInput[0];
	int nCount = (int)plInput[1];
	int nNum = (int)plInput[2];
	int nDim = (int)plInput[3];
	long hX = (long)plInput[4];
	long hY = (long)plInput[5];

	return m_memory.DistortImage(hHandle, nCount, nNum, nDim, hX, hY);
}

template long Device<double>::DistortImage(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::DistortImage(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::SetTensorDesc(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 6, 10, true))
		return lErr;

	long hHandle = (long)plInput[0];
	bool bHalf = (bool)(plInput[1] == 1) ? true : false;
	int n = (int)plInput[2];
	int c = (int)plInput[3];
	int h = (int)plInput[4];
	int w = (int)plInput[5];
	int nStride;
	int cStride;
	int hStride;
	int wStride;

	if (llInput == 6)
	{
		wStride = 1;
		hStride = w * wStride;
		cStride = h * hStride;
		nStride = c * cStride;
	}
	else
	{
		nStride = (int)plInput[6];
		cStride = (int)plInput[7];
		hStride = (int)plInput[8];
		wStride = (int)plInput[9];
	}

	if (lErr = m_memory.SetTensorDesc(hHandle, n, c, h, w, nStride, cStride, hStride, wStride, bHalf))
		return lErr;

	return 0;
}

template long Device<double>::SetTensorDesc(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::SetTensorDesc(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::SetTensorNdDesc(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 4, MAX_ARG))
		return lErr;

	long hHandle = (long)plInput[0];
	bool bHalf = (bool)(plInput[1] == 1) ? true : false;
	int nCount = (int)plInput[2];

	if (nCount > MAX_DIM || nCount <= 0 || nCount > (llInput - 2)/2)
		return ERROR_PARAM_OUT_OF_RANGE;

	int* rgDim = (int*)malloc(sizeof(int) * nCount);
	if (rgDim == NULL)
		return ERROR_OUTOFMEMORY;

	int* rgStride = (int*)malloc(sizeof(int) * nCount);
	if (rgStride == NULL)
	{
		free(rgDim);
		return ERROR_OUTOFMEMORY;
	}

	int nIdx = 3;
	for (int i = 0; i < nCount; i++)
	{
		rgDim[i] = (int)plInput[nIdx];
		nIdx++;
	}

	for (int i = 0; i < nCount; i++)
	{
		rgStride[i] = (int)plInput[nIdx];
		nIdx++;
	}

	lErr = m_memory.SetTensorDesc(hHandle, rgDim, rgStride, nCount, bHalf);

	free(rgDim);
	free(rgStride);

	return lErr;
}

template long Device<double>::SetTensorNdDesc(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::SetTensorNdDesc(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::SetFilterNdDesc(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 3, MAX_ARG))
		return lErr;

	long hHandle = (long)plInput[0];
	bool bHalf = (bool)(plInput[1] == 1) ? true : false;
	int nCount = (int)plInput[2];

	if (nCount > MAX_DIM || nCount <= 0 || nCount > (llInput - 2))
		return ERROR_PARAM_OUT_OF_RANGE;

	int* rgDim = (int*)malloc(sizeof(int) * nCount);
	if (rgDim == NULL)
		return ERROR_OUTOFMEMORY;

	int nIdx = 3;
	for (int i = 0; i < nCount; i++)
	{
		rgDim[i] = (int)plInput[nIdx];
		nIdx++;
	}

	lErr = m_memory.SetFilterDesc(hHandle, rgDim, nCount, bHalf);

	free(rgDim);

	return lErr;
}

template long Device<double>::SetFilterNdDesc(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::SetFilterNdDesc(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::GetDropoutInfo(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;	
	unsigned long lStates = 0;
	unsigned long lReserved = 0;

	if (lErr = verifyInput(llInput, plInput, 2, 2))
		return lErr;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	long hCuda = (long)plInput[0];
	long hBottomDesc = (long)plInput[1];

	if (lErr = m_memory.GetDropoutInfo(hCuda, hBottomDesc, &lStates, &lReserved))
		return lErr;

	// ppfOutput has up to MAX_OUTPUT(16) pre-allocated items
	T* pfOutput = *ppfOutput;

	pfOutput[0] = (T)lStates;
	pfOutput[1] = (T)lReserved;

	*ppfOutput = pfOutput;
	*plOutput = 2;

	return 0;
}

template long Device<double>::GetDropoutInfo(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::GetDropoutInfo(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_get(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 2, 3))
		return lErr;
	
	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	int nCount = (int)plInput[0];
	long hHandle = (long)plInput[1];
	int nIdx = -1;
	int nItems = nCount;

	if (llInput > 2)
	{
		nIdx = (int)plInput[2];

		if (nIdx >= 0)
			nItems = 1;
	}

	T* pfOutput = NULL;
	
	if (lErr = m_memory.AllocHost(nItems, &pfOutput, NULL, false, false, false))
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

template long Device<double>::cuda_get(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_get(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);

template <class T>
long Device<T>::cuda_gemm(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 2, 2))
		return lErr;
	if (lErr = verifyInput(llInput, plInput, 10, 17))
		return lErr;

	bool bTransA = (plInput[0] == 0.0) ? false : true;
	bool bTransB = (plInput[1] == 0.0) ? false : true;
	int m = (int)plInput[2];
	int n = (int)plInput[3];
	int k = (int)plInput[4];
	T fAlpha = pfInput[0];
	long hA = (long)plInput[6];
	long hB = (long)plInput[7];
	T fBeta = pfInput[1];
	long hC = (long)plInput[9];
	int nAOff = 0;
	int nBOff = 0;
	int nCOff = 0;
	int nGroups = 1;
	int nGroupAOff = 0;
	int nGroupBOff = 0;
	int nGroupCOff = 0;

	if (llInput > 10)
		nAOff = (int)plInput[10];

	if (llInput > 11)
		nBOff = (int)plInput[11];

	if (llInput > 12)
		nCOff = (int)plInput[12];

	if (llInput > 13)
		nGroups = (int)plInput[13];

	if (llInput > 14)
		nGroupAOff = (int)plInput[14];

	if (llInput > 15)
		nGroupBOff = (int)plInput[15];

	if (llInput > 16)
		nGroupCOff = (int)plInput[16];

	return m_math.gemm(bTransA, bTransB, m, n, k, fAlpha, hA, hB, fBeta, hC, nAOff, nBOff, nCOff, nGroups, nGroupAOff, nGroupBOff, nGroupCOff);
}

template long Device<double>::cuda_gemm(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_gemm(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);

template <class T>
long Device<T>::cuda_gemm2(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 2, 2))
		return lErr;
	if (lErr = verifyInput(llInput, plInput, 13, 17))
		return lErr;

	if (llInput != 13 && llInput != 17)
		return ERROR_PARAM_OUT_OF_RANGE;

	bool bTransA = (plInput[0] == 0.0) ? false : true;
	bool bTransB = (plInput[1] == 0.0) ? false : true;
	int m = (int)plInput[2];
	int n = (int)plInput[3];
	int k = (int)plInput[4];
	T fAlpha = pfInput[0];
	long hA = (long)plInput[6];
	long hB = (long)plInput[7];
	T fBeta = pfInput[1];
	long hC = (long)plInput[9];
	int lda = (int)plInput[10];
	int ldb = (int)plInput[11];
	int ldc = (int)plInput[12];

	if (llInput == 13)
		return m_math.gemm2(bTransA, bTransB, m, n, k, fAlpha, hA, hB, fBeta, hC, lda, ldb, ldc);

	int stridea = (int)plInput[13];
	int strideb = (int)plInput[14];
	int stridec = (int)plInput[15];
	int batch_count = (int)plInput[16];
	
	return m_math.gemm2(bTransA, bTransB, m, n, k, fAlpha, hA, hB, fBeta, hC, lda, ldb, ldc, stridea, strideb, stridec, batch_count);
}

template long Device<double>::cuda_gemm2(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_gemm2(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_gemv(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 2, 2))
		return lErr;
	if (lErr = verifyInput(llInput, plInput, 8, 11))
		return lErr;

	bool bTransA = (plInput[0] == 0.0) ? false : true;
	int n = (int)plInput[1];
	int m = (int)plInput[2];
	T fAlpha = pfInput[0];
	long hA = (long)plInput[4];
	long hX = (long)plInput[5];
	T fBeta = pfInput[1];
	long hY = (long)plInput[7];
	int nAOffset = 0;
	int nXOffset = 0;
	int nYOffset = 0;

	if (llInput > 8)
		nAOffset = (int)plInput[8];

	if (llInput > 9)
		nXOffset = (int)plInput[9];

	if (llInput > 10)
		nYOffset = (int)plInput[10];

	return m_math.gemv(bTransA, n, m, fAlpha, hA, hX, fBeta, hY, nAOffset, nXOffset, nYOffset);
}

template long Device<double>::cuda_gemv(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_gemv(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_geam(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 2, 2))
		return lErr;
	if (lErr = verifyInput(llInput, plInput, 9, 12))
		return lErr;

	bool bTransA = (plInput[0] == 0.0) ? false : true;
	bool bTransB = (plInput[1] == 0.0) ? false : true;
	int m = (int)plInput[2];
	int n = (int)plInput[3];
	T fAlpha = pfInput[0];
	long hA = (long)plInput[5];
	long hB = (long)plInput[6];
	T fBeta = pfInput[1];
	long hC = (long)plInput[8];
	int nAOff = 0;
	int nBOff = 0;
	int nCOff = 0;

	if (llInput > 9)
		nAOff = (int)plInput[9];

	if (llInput > 10)
		nBOff = (int)plInput[10];

	if (llInput > 11)
		nCOff = (int)plInput[11];

	return m_math.geam(bTransA, bTransB, m, n, fAlpha, hA, hB, fBeta, hC, nAOff, nBOff, nCOff);
}

template long Device<double>::cuda_geam(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_geam(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_ger(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;
	if (lErr = verifyInput(llInput, plInput, 6, 6))
		return lErr;

	int n = (int)plInput[0];
	int m = (int)plInput[1];
	T fAlpha = pfInput[0];
	long hA = (long)plInput[3];
	long hB = (long)plInput[4];
	long hC = (long)plInput[5];

	return m_math.ger(m, n, fAlpha, hA, hB, hC);
}

template long Device<double>::cuda_ger(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_ger(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_axpy(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;
	if (lErr = verifyInput(llInput, plInput, 4, 6))
		return lErr;

	int n = (int)plInput[0];
	T fAlpha = pfInput[0];
	long hX = (long)plInput[2];
	long hY = (long)plInput[3];
	int nXOff = 0;
	int nYOff = 0;

	if (llInput > 4)
		nXOff = (int)plInput[4];

	if (llInput > 5)
		nYOff = (int)plInput[5];

	return m_math.axpy(n, fAlpha, hX, hY, nXOff, nYOff);
}

template long Device<double>::cuda_axpy(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_axpy(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_axpby(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 2, 2))
		return lErr;
	if (lErr = verifyInput(llInput, plInput, 5, 5))
		return lErr;

	int n = (int)plInput[0];
	T fAlpha = pfInput[0];
	long hX = (long)plInput[2];
	T fBeta = pfInput[1];
	long hY = (long)plInput[4];

	return m_math.axpby(n, fAlpha, hX, fBeta, hY);
}

template long Device<double>::cuda_axpby(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_axpby(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_scal(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;
	if (lErr = verifyInput(llInput, plInput, 3, 4))
		return lErr;

	int n = (int)plInput[0];
	T fAlpha = pfInput[0];
	long hX = (long)plInput[2];
	int nXOff = 0;

	if (llInput > 3)
		nXOff = (int)plInput[3];

	return m_math.scal(n, fAlpha, hX, nXOff);
}

template long Device<double>::cuda_scal(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_scal(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_dot(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 3, 5))
		return lErr;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	int n = (int)plInput[0];
	long hX = (long)plInput[1];
	long hY = (long)plInput[2];
	int nXOff = 0;
	int nYOff = 0;
	T fOutput = 0;

	if (llInput > 3)
		nXOff = (int)plInput[3];

	if (llInput > 4)
		nYOff = (int)plInput[4];

	if (lErr = m_math.dot(n, hX, hY, &fOutput, nXOff, nYOff))
		return lErr;

	return setOutput(fOutput, plOutput, ppfOutput);
}

template long Device<double>::cuda_dot(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_dot(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_asum(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 2, 3))
		return lErr;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	int n = (int)plInput[0];
	long hX = (long)plInput[1];
	int nXOff = 0;

	if (llInput > 2)
		nXOff = (int)plInput[2];

	T fOutput = 0;

	if (lErr = m_math.asum(n, hX, &fOutput, nXOff))
		return lErr;

	return setOutput(fOutput, plOutput, ppfOutput);
}

template long Device<double>::cuda_asum(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_asum(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_mulbsx(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 10, 10))
		return lErr;

	int n = (int)plInput[0];
	long hA = (long)plInput[1];
	int nAOff = (int)plInput[2];
	long hX = (long)plInput[3];
	int nXOff = (int)plInput[4];
	int nC = (int)plInput[5];
	int nSpatialDim = (int)plInput[6];
	bool bTranspose = (plInput[7] == 0) ? false : true;
	long hB = (long)plInput[8];
	int nBOff = (int)plInput[9];

	return m_math.mulbsx(n, hA, nAOff, hX, nXOff, nC, nSpatialDim, bTranspose, hB, nBOff);
}

template long Device<double>::cuda_mulbsx(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_mulbsx(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_divbsx(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 10, 10))
		return lErr;

	int n = (int)plInput[0];
	long hA = (long)plInput[1];
	int nAOff = (int)plInput[2];
	long hX = (long)plInput[3];
	int nXOff = (int)plInput[4];
	int nC = (int)plInput[5];
	int nSpatialDim = (int)plInput[6];
	bool bTranspose = (plInput[7] == 0) ? false : true;
	long hB = (long)plInput[8];
	int nBOff = (int)plInput[9];

	return m_math.divbsx(n, hA, nAOff, hX, nXOff, nC, nSpatialDim, bTranspose, hB, nBOff);
}

template long Device<double>::cuda_divbsx(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_divbsx(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_scale(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;
	if (lErr = verifyInput(llInput, plInput, 4, 6))
		return lErr;

	int n = (int)plInput[0];
	T fAlpha = pfInput[0];
	long hX = (long)plInput[2];
	long hY = (long)plInput[3];
	int nXOff = 0;
	int nYOff = 0;

	if (llInput > 4)
		nXOff = (int)plInput[4];

	if (llInput > 5)
		nYOff = (int)plInput[5];

	return m_math.scale(n, fAlpha, hX, hY, nXOff, nYOff);
}

template long Device<double>::cuda_scale(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_scale(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_scale_to_range(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 2, 2))
		return lErr;
	if (lErr = verifyInput(llInput, plInput, 5, 5))
		return lErr;

	int n = (int)plInput[0];
	long hX = (long)plInput[1];
	long hY = (long)plInput[2];
	T fMin = pfInput[0];
	T fMax = pfInput[1];

	return m_math.scale_to_range(n, hX, hY, fMin, fMax);
}

template long Device<double>::cuda_scale_to_range(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_scale_to_range(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);

template <class T>
long Device<T>::cuda_erf(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;

	T fVal = pfInput[0];
	T fResult;

	if (lErr = m_math.erf(fVal, &fResult))
		return lErr;

	return setOutput(fResult, plOutput, ppfOutput);
}

template long Device<double>::cuda_erf(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_erf(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_mask(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 2, 2))
		return lErr;
	if (lErr = verifyInput(llInput, plInput, 7, 7))
		return lErr;

	int n = (int)plInput[0];
	int nMaskDim = (int)plInput[1];
	T fSearch = pfInput[0];
	T fReplace = pfInput[1];
	long hX = (long)plInput[4];
	long hMask = (long)plInput[5];
	long hY = (long)plInput[6];

	return m_math.mask(n, nMaskDim, fSearch, fReplace, hX, hMask, hY);
}

template long Device<double>::cuda_mask(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_mask(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_mask_batch(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 2, 2))
		return lErr;
	if (lErr = verifyInput(llInput, plInput, 8, 8))
		return lErr;

	int n = (int)plInput[0];
	int nBatch = (int)plInput[1];
	int nMaskDim = (int)plInput[2];
	T fSearch = pfInput[0];
	T fReplace = pfInput[1];
	long hX = (long)plInput[5];
	long hMask = (long)plInput[6];
	long hY = (long)plInput[7];

	return m_math.mask_batch(n, nBatch, nMaskDim, fSearch, fReplace, hX, hMask, hY);
}

template long Device<double>::cuda_mask_batch(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_mask_batch(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);

template <class T>
long Device<T>::cuda_interp2(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 16, 16))
		return lErr;

	int nC = (int)plInput[0];
	long hX = (long)plInput[1];
	int nX1 = (int)plInput[2];
	int nY1 = (int)plInput[3];
	int nH1 = (int)plInput[4];
	int nW1 = (int)plInput[5];
	int nH1A = (int)plInput[6];
	int nW1A = (int)plInput[7];
	long hY = (long)plInput[8];
	int nX2 = (int)plInput[9];
	int nY2 = (int)plInput[10];
	int nH2 = (int)plInput[11];
	int nW2 = (int)plInput[12];
	int nH2B = (int)plInput[13];
	int nW2B = (int)plInput[14];
	bool bBwd = (plInput[15] == 0) ? false : true;

	return m_math.interp2(nC, hX, nX1, nY1, nH1, nW1, nH1A, nW1A, hY, nX2, nY2, nH2, nW2, nH2B, nW2B, bBwd);
}

template long Device<double>::cuda_interp2(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_interp2(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_add_scalar(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;
	if (lErr = verifyInput(llInput, plInput, 3, 4))
		return lErr;

	int n = (int)plInput[0];
	T fAlpha = pfInput[0];
	long hY = (long)plInput[2];
	int nYOff = 0;

	if (llInput > 3)
		nYOff = (int)plInput[3];

	return m_math.add_scalar(n, fAlpha, hY, nYOff);
}

template long Device<double>::cuda_add_scalar(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_add_scalar(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_add(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lInput > 0 && pfInput != NULL)
	{
		if (lErr = verifyInput(lInput, pfInput, 1, 1))
			return lErr;
	}
	if (lErr = verifyInput(llInput, plInput, 4, 5))
		return lErr;

	int n = (int)plInput[0];
	long hA = (long)plInput[1];
	long hB = (long)plInput[2];
	long hY = (long)plInput[3];
	T fAlpha = 1.0;

	if (lInput > 0 && pfInput != NULL)
		fAlpha = pfInput[0];

	return m_math.add(n, hA, hB, hY, fAlpha);
}

template long Device<double>::cuda_add(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_add(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_add2(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 2, 2))
		return lErr;
	if (lErr = verifyInput(llInput, plInput, 6, 9))
		return lErr;

	int n = (int)plInput[0];
	long hA = (long)plInput[1];
	long hB = (long)plInput[2];
	long hY = (long)plInput[3];
	T fAlphaA = pfInput[0];
	T fAlphaB = pfInput[1];
	int nAOff = 0;
	int nBOff = 0;
	int nYOff = 0;

	if (llInput > 6)
		nAOff = (int)plInput[6];

	if (llInput > 7)
		nBOff = (int)plInput[7];

	if (llInput > 8)
		nYOff = (int)plInput[8];

	return m_math.add2(n, hA, hB, hY, fAlphaA, fAlphaB, nAOff, nBOff, nYOff);
}

template long Device<double>::cuda_add2(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_add2(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_add3(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 5, 5))
		return lErr;

	int n = (int)plInput[0];
	long hA = (long)plInput[1];
	long hB = (long)plInput[2];
	long hC = (long)plInput[3];
	long hY = (long)plInput[4];

	return m_math.add3(n, hA, hB, hC, hY);
}

template long Device<double>::cuda_add3(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_add3(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_sub(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 4, 8))
		return lErr;

	int n = (int)plInput[0];
	long hA = (long)plInput[1];
	long hB = (long)plInput[2];
	long hY = (long)plInput[3];
	int nAOff = 0;
	int nBOff = 0;
	int nYOff = 0;
	int nB = 0;

	if (llInput > 4)
		nAOff = (int)plInput[4];

	if (llInput > 5)
		nBOff = (int)plInput[5];

	if (llInput > 6)
		nYOff = (int)plInput[6];

	if (llInput > 7)
	{
		nB = (int)plInput[7];

		if (nB > 0)
		{
			if (n % nB != 0)
				return ERROR_PARAM_OUT_OF_RANGE;
		}
	}

	return m_math.sub(n, hA, hB, hY, nAOff, nBOff, nYOff, nB);
}

template long Device<double>::cuda_sub(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_sub(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);



template <class T>
long Device<T>::cuda_sub_and_dot(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 6, 9))
		return lErr;

	int n = (int)plInput[0];
	int nN = (int)plInput[1];
	int nLen = (int)plInput[2];
	long hA = (long)plInput[3];
	long hB = (long)plInput[4];
	long hY = (long)plInput[5];
	int nAOff = 0;
	int nBOff = 0;
	int nYOff = 0;

	if (llInput > 6)
		nAOff = (int)plInput[6];

	if (llInput > 7)
		nBOff = (int)plInput[7];

	if (llInput > 8)
		nYOff = (int)plInput[8];

	return m_math.sub_and_dot(n, nN, nLen, hA, hB, hY, nAOff, nBOff, nYOff);
}

template long Device<double>::cuda_sub_and_dot(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_sub_and_dot(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_mul_scalar(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;
	if (lErr = verifyInput(llInput, plInput, 3, 4))
		return lErr;

	int n = (int)plInput[0];
	T fAlpha = pfInput[0];
	long hY = (long)plInput[2];
	int nYOff = 0;

	if (llInput > 3)
		nYOff = (int)plInput[3];

	return m_math.mul_scalar(n, fAlpha, hY, nYOff);
}

template long Device<double>::cuda_mul_scalar(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_mul_scalar(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_mul(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 4, 7))
		return lErr;

	int n = (int)plInput[0];
	long hA = (long)plInput[1];
	long hB = (long)plInput[2];
	long hY = (long)plInput[3];
	int nAOff = 0;
	int nBOff = 0;
	int nYOff = 0;

	if (llInput > 4)
		nAOff = (int)plInput[4];

	if (llInput > 5)
		nBOff = (int)plInput[5];

	if (llInput > 6)
		nYOff = (int)plInput[6];

	return m_math.mul(n, hA, hB, hY, nAOff, nBOff, nYOff);
}

template long Device<double>::cuda_mul(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_mul(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_div(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 4, 4))
		return lErr;

	int n = (int)plInput[0];
	long hA = (long)plInput[1];
	long hB = (long)plInput[2];
	long hY = (long)plInput[3];

	return m_math.div(n, hA, hB, hY);
}

template long Device<double>::cuda_div(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_div(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_abs(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 3, 3))
		return lErr;

	int n = (int)plInput[0];
	long hA = (long)plInput[1];
	long hY = (long)plInput[2];

	return m_math.abs(n, hA, hY);
}

template long Device<double>::cuda_abs(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_abs(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_exp(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 0, 1))
		return lErr;
	if (lErr = verifyInput(llInput, plInput, 3, 6))
		return lErr;

	int n = (int)plInput[0];
	long hA = (long)plInput[1];
	long hY = (long)plInput[2];
	int nAOff = 0;
	int nYOff = 0;
	T fBeta = 1.0;

	if (llInput > 3)
		nAOff = (int)plInput[3];

	if (llInput > 4)
		nYOff = (int)plInput[4];

	if (lInput > 0)
		fBeta = pfInput[0];

	return m_math.exp(n, hA, hY, nAOff, nYOff, fBeta);
}

template long Device<double>::cuda_exp(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_exp(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_log(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 0, 2))
		return lErr;
	if (lErr = verifyInput(llInput, plInput, 3, 5))
		return lErr;

	int n = (int)plInput[0];
	long hA = (long)plInput[1];
	long hY = (long)plInput[2];
	T fBeta = 1;
	T fAlpha = 0;

	if (lInput > 0)
	{
		fBeta = pfInput[0];

		if (lInput > 1)
			fAlpha = pfInput[1];
	}

	return m_math.log(n, hA, hY, fBeta, fAlpha);
}

template long Device<double>::cuda_log(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_log(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_powx(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;
	if (lErr = verifyInput(llInput, plInput, 4, 6))
		return lErr;

	int n = (int)plInput[0];
	long hA = (long)plInput[1];
	T fAlpha = pfInput[0];
	long hY = (long)plInput[3];
	int nAOff = 0;
	int nYOff = 0;

	if (llInput > 4)
		nAOff = (int)plInput[4];

	if (llInput > 5)
		nYOff = (int)plInput[5];

	return m_math.powx(n, hA, fAlpha, hY, nAOff, nYOff);
}

template long Device<double>::cuda_powx(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_powx(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_sign(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 3, 5))
		return lErr;

	int n = (int)plInput[0];
	long hX = (long)plInput[1];
	long hY = (long)plInput[2];
	int nXOff = 0;
	int nYOff = 0;

	if (llInput > 3)
		nXOff = (int)plInput[3];

	if (llInput > 4)
		nYOff = (int)plInput[4];

	return m_math.sign(n, hX, hY, nXOff, nYOff);
}

template long Device<double>::cuda_sign(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_sign(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_sqrt(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 3, 3))
		return lErr;

	int n = (int)plInput[0];
	long hX = (long)plInput[1];
	long hY = (long)plInput[2];

	return m_math.sqrt(n, hX, hY);
}

template long Device<double>::cuda_sqrt(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_sqrt(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_reciprocol(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 3, 3))
		return lErr;

	int n = (int)plInput[0];
	long hX = (long)plInput[1];
	long hY = (long)plInput[2];

	return m_math.reciprocol(n, hX, hY);
}

template long Device<double>::cuda_reciprocol(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_reciprocol(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_student(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 3, 3))
		return lErr;

	int n = (int)plInput[0];
	long hX = (long)plInput[1];
	long hY = (long)plInput[2];

	return m_math.student(n, hX, hY);
}

template long Device<double>::cuda_student(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_student(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_logistic1(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 3, 3))
		return lErr;

	int n = (int)plInput[0];
	long hX = (long)plInput[1];
	long hY = (long)plInput[2];

	return m_math.logistic1(n, hX, hY);
}

template long Device<double>::cuda_logistic1(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_logistic1(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_logistic2(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 3, 3))
		return lErr;

	int n = (int)plInput[0];
	long hX = (long)plInput[1];
	long hY = (long)plInput[2];

	return m_math.logistic2(n, hX, hY);
}

template long Device<double>::cuda_logistic2(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_logistic2(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_compare_signs(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 4, 4))
		return lErr;

	int n = (int)plInput[0];
	long hA = (long)plInput[1];
	long hB = (long)plInput[2];
	long hY = (long)plInput[3];

	return m_math.compare_signs(n, hA, hB, hY);
}

template long Device<double>::cuda_compare_signs(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_compare_signs(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_maxval(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 2, 4))
		return lErr;

	int n = (int)plInput[0];
	long hA = (long)plInput[1];
	int nAOff = 0;
	T fOutput = 0;
	long hWork = 0;

	if (llInput > 2)
		nAOff = (int)plInput[2];

	if (llInput > 3)
		hWork = (long)plInput[3];

	long lPos = -1;	
	if (hWork != 0)
	{
		if (lErr = m_math.maxvalEx(n, hA, hWork, &fOutput, nAOff))
			return lErr;
	}
	else
	{
		if (lErr = m_math.maxval(n, hA, &fOutput, nAOff, &lPos))
			return lErr;
	}

	// ppfOutput has up to MAX_OUTPUT(16) pre-allocated items
	T* pfOutput = *ppfOutput;

	pfOutput[0] = fOutput;
	pfOutput[1] = (T)lPos;

	*ppfOutput = pfOutput;
	*plOutput = 2;

	return 0;
}

template long Device<double>::cuda_maxval(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_maxval(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_minval(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 2, 4))
		return lErr;

	int n = (int)plInput[0];
	long hA = (long)plInput[1];
	int nAOff = 0;
	T fOutput = 0;
	long hWork = 0;
	
	if (llInput > 2)
		nAOff = (int)plInput[2];

	if (llInput > 3)
		hWork = (long)plInput[3];

	long lPos = -1;
	if (hWork != 0)
	{
		if (lErr = m_math.minvalEx(n, hA, hWork, &fOutput, nAOff))
			return lErr;
	}
	else
	{
		if (lErr = m_math.minval(n, hA, &fOutput, nAOff, &lPos))
			return lErr;
	}

	// ppfOutput has up to MAX_OUTPUT(16) pre-allocated items
	T* pfOutput = *ppfOutput;

	pfOutput[0] = fOutput;
	pfOutput[1] = (T)lPos;

	*ppfOutput = pfOutput;
	*plOutput = 2;

	return 0;
}

template long Device<double>::cuda_minval(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_minval(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_minmaxval(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 4, 6))
		return lErr;

	int n = (int)plInput[0];
	long hA = (long)plInput[1];
	long hWork1 = (long)plInput[2];
	long hWork2 = (long)plInput[3];
	bool bDetectNans = false;
	int nAOff = 0;
	T fMin;
	T fMax;
	T fNan = 0;
	T fInf = 0;

	if (llInput > 4)
		bDetectNans = (plInput[4] == 0) ? false : true;

	if (llInput > 5)
		nAOff = (int)plInput[5];

	if (lErr = m_math.minmaxval(n, hA, hWork1, hWork2, &fMin, &fMax, nAOff))
		return lErr;

	if (bDetectNans)
	{
		if (lErr = m_math.naninfval(n, hA, hWork1, hWork2, &fNan, &fInf, nAOff))
			return lErr;
	}

	// ppfOutput has up to MAX_OUTPUT(16) pre-allocated items
	T* pfOutput = *ppfOutput;

	pfOutput[0] = fMin;
	pfOutput[1] = fMax;
	pfOutput[2] = fNan;
	pfOutput[3] = fInf;

	*ppfOutput = pfOutput;
	*plOutput = 4;

	return 0;
}

template long Device<double>::cuda_minmaxval(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_minmaxval(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_minmaxvec(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 8, 8))
		return lErr;

	int n = (int)plInput[0];
	long hA = (long)plInput[1];
	long hWork1 = (long)plInput[2];
	long hWork2 = (long)plInput[3];
	int nK = (int)plInput[4];
	long hMin = (long)plInput[5];
	long hMax = (long)plInput[6];
	bool bNonZero = (bool)(plInput[7] == 0) ? true : false;

	return m_math.minmaxvec(n, hA, hWork1, hWork2, nK, hMin, hMax, bNonZero);
}

template long Device<double>::cuda_minmaxvec(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_minmaxvec(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_transpose(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 8, 8))
		return lErr;

	int n = (int)plInput[0];
	long hX = (long)plInput[1];
	long hY = (long)plInput[2];
	long hXCounts = (long)plInput[3];
	long hYCounts = (long)plInput[4];
	long hMapping = (long)plInput[5];
	int nNumAxes = (int)plInput[6];
	long hBuffer = (long)plInput[7];

	return m_math.transpose(n, hX, hY, hXCounts, hYCounts, hMapping, nNumAxes, hBuffer);
}

template long Device<double>::cuda_transpose(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_transpose(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_transpose_hw(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 6, 6))
		return lErr;

	int n = (int)plInput[0];
	int c = (int)plInput[1];
	int h = (int)plInput[2];
	int w = (int)plInput[3];
	long hSrc = (long)plInput[4];
	long hDst = (long)plInput[5];

	return m_math.transpose_hw(n, c, h, w, hSrc, hDst);
}

template long Device<double>::cuda_transpose_hw(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_transpose_hw(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_sumsq(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 3, 4))
		return lErr;

	int n = (int)plInput[0];
	long hW = (long)plInput[1];
	long hA = (long)plInput[2];
	int nAOff = 0;

	if (llInput > 3)
		nAOff = (int)plInput[3];

	T fOutput = 0;

	if (lErr = m_math.sumsq(n, hW, hA, nAOff, &fOutput))
		return lErr;

	return setOutput(fOutput, plOutput, ppfOutput);
}

template long Device<double>::cuda_sumsq(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_sumsq(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_sumsqdiff(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 4, 6))
		return lErr;

	int n = (int)plInput[0];
	long hW = (long)plInput[1];
	long hA = (long)plInput[2];
	long hB = (long)plInput[3];
	int nAOff = 0;
	int nBOff = 0;

	if (llInput > 4)
		nAOff = (int)plInput[4];

	if (llInput > 5)
		nBOff = (int)plInput[5];

	T fOutput = 0;

	if (lErr = m_math.sumsqdiff(n, hW, hA, hB, nAOff, nBOff, &fOutput))
		return lErr;

	return setOutput(fOutput, plOutput, ppfOutput);
}

template long Device<double>::cuda_sumsqdiff(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_sumsqdiff(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_sum(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 5, 5))
		return lErr;

	int n = (int)plInput[0];
	int nOutNum = (int)plInput[1];
	int nInNum = (int)plInput[2];
	long hX = (long)plInput[3];
	long hY = (long)plInput[4];

	return m_math.sum(n, nOutNum, nInNum, hX, hY);
}

template long Device<double>::cuda_sum(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_sum(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_sqrt_scale(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 3, 3))
		return lErr;

	int n = (int)plInput[0];
	long hX = (long)plInput[1];
	long hY = (long)plInput[2];

	return m_math.sqrt_scale(n, hX, hY);
}

template long Device<double>::cuda_sqrt_scale(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_sqrt_scale(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_width(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;
	if (lErr = verifyInput(llInput, plInput, 6, 6))
		return lErr;

	int n = (int)plInput[0];
	long hMean = (long)plInput[1];
	long hMin = (long)plInput[2];
	long hMax = (long)plInput[3];
	T fAlpha = pfInput[0];
	long hWidth = (long)plInput[5];

	return m_math.width(n, hMean, hMin, hMax, fAlpha, hWidth);
}

template long Device<double>::cuda_width(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_width(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_contains_point(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 5, 6))
		return lErr;

	int n = (int)plInput[0];
	long hMean = (long)plInput[1];
	long hWidth = (long)plInput[2];
	long hX = (long)plInput[3];
	long hWork = (long)plInput[4];
	int nXOff = 0;

	if (llInput > 5)
		nXOff = (int)plInput[5];

	T fOutput = 0;

	if (lErr = m_math.contains_point(n, hMean, hWidth, hX, hWork, &fOutput, nXOff))
		return lErr;

	return setOutput(fOutput, plOutput, ppfOutput);
}

template long Device<double>::cuda_contains_point(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_contains_point(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_denan(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;
	if (lErr = verifyInput(llInput, plInput, 3, 3))
		return lErr;

	int n = (int)plInput[0];
	long hX = (long)plInput[1];
	T fReplacement = pfInput[0];

	return m_math.denan(n, hX, fReplacement);
}

template long Device<double>::cuda_denan(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_denan(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_set_bounds(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 2, 2))
		return lErr;
	if (lErr = verifyInput(llInput, plInput, 4, 4))
		return lErr;

	int n = (int)plInput[0];
	T fMin = pfInput[0];
	T fMax = pfInput[1];
	long hX = (long)plInput[3];

	return m_math.set_bounds(n, fMin, fMax, hX);
}

template long Device<double>::cuda_set_bounds(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_set_bounds(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_channel_min(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 7, 7))
		return lErr;

	int n = (int)plInput[0];
	int nOutNum = (int)plInput[1];
	int nChannels = (int)plInput[2];
	int nInNum = (int)plInput[3];
	long hX = (long)plInput[4];
	long hY = (long)plInput[5];
	bool bReturnIdx = (plInput[6] == 0) ? false : true;

	return m_math.channel_min(n, nOutNum, nChannels, nInNum, hX, hY, bReturnIdx);
}

template long Device<double>::cuda_channel_min(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_channel_min(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_channel_max(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 7, 7))
		return lErr;

	int n = (int)plInput[0];
	int nOutNum = (int)plInput[1];
	int nChannels = (int)plInput[2];
	int nInNum = (int)plInput[3];
	long hX = (long)plInput[4];
	long hY = (long)plInput[5];
	bool bReturnIdx = (plInput[6] == 0) ? false : true;

	return m_math.channel_max(n, nOutNum, nChannels, nInNum, hX, hY, bReturnIdx);
}

template long Device<double>::cuda_channel_max(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_channel_max(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_channel_mean(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 6, 6))
		return lErr;

	int n = (int)plInput[0];
	int nOutNum = (int)plInput[1];
	int nChannels = (int)plInput[2];
	int nInNum = (int)plInput[3];
	long hX = (long)plInput[4];
	long hY = (long)plInput[5];

	return m_math.channel_mean(n, nOutNum, nChannels, nInNum, hX, hY);
}

template long Device<double>::cuda_channel_mean(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_channel_mean(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_channel_sub(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 6, 7))
		return lErr;

	int n = (int)plInput[0];
	int nOutNum = (int)plInput[1];
	int nChannels = (int)plInput[2];
	int nInNum = (int)plInput[3];
	long hX = (long)plInput[4];
	long hY = (long)plInput[5];
	long hA = hY;
	
	if (llInput == 7)
		hA = (long)plInput[6];

	return m_math.channel_sub(n, nOutNum, nChannels, nInNum, hA, hX, hY);
}

template long Device<double>::cuda_channel_sub(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_channel_sub(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_channel_sum(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 7, 7))
		return lErr;

	int n = (int)plInput[0];
	int nOutNum = (int)plInput[1];
	int nChannels = (int)plInput[2];
	int nInNum = (int)plInput[3];
	long hX = (long)plInput[4];
	long hY = (long)plInput[5];
	bool bSumAcrossChannels = (plInput[6] == 0) ? false : true;

	return m_math.channel_sum(n, nOutNum, nChannels, nInNum, hX, hY, bSumAcrossChannels);
}

template long Device<double>::cuda_channel_sum(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_channel_sum(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_channel_div(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 6, 7))
		return lErr;

	int n = (int)plInput[0];
	int nOutNum = (int)plInput[1];
	int nChannels = (int)plInput[2];
	int nInNum = (int)plInput[3];
	long hX = (long)plInput[4];
	long hY = (long)plInput[5];
	int nMethod = 1;

	if (llInput > 6)
		nMethod = (int)plInput[6];

	return m_math.channel_div(n, nOutNum, nChannels, nInNum, hX, hY, nMethod);
}

template long Device<double>::cuda_channel_div(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_channel_div(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);

template <class T>
long Device<T>::cuda_channel_mul(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 6, 7))
		return lErr;

	int n = (int)plInput[0];
	int nOutNum = (int)plInput[1];
	int nChannels = (int)plInput[2];
	int nInNum = (int)plInput[3];
	long hX = (long)plInput[4];
	long hY = (long)plInput[5];
	int nMethod = 1;

	if (llInput > 6)
		nMethod = (int)plInput[6];

	return m_math.channel_mul(n, nOutNum, nChannels, nInNum, hX, hY, nMethod);
}

template long Device<double>::cuda_channel_mul(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_channel_mul(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_channel_mulv(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 7, 7))
		return lErr;

	int n = (int)plInput[0];
	int nOutNum = (int)plInput[1];
	int nChannels = (int)plInput[2];
	int nInNum = (int)plInput[3];
	long hA = (long)plInput[4];
	long hX = (long)plInput[5];
	long hC = (long)plInput[6];

	return m_math.channel_mulv(n, nOutNum, nChannels, nInNum, hA, hX, hC);
}

template long Device<double>::cuda_channel_mulv(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_channel_mulv(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_channel_scale(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 7, 7))
		return lErr;

	int n = (int)plInput[0];
	int nOutNum = (int)plInput[1];
	int nChannels = (int)plInput[2];
	int nInNum = (int)plInput[3];
	long hX = (long)plInput[4];
	long hA = (long)plInput[5];
	long hY = (long)plInput[6];

	return m_math.channel_scale(n, nOutNum, nChannels, nInNum, hX, hA, hY);
}

template long Device<double>::cuda_channel_scale(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_channel_scale(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_channel_dot(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 7, 7))
		return lErr;

	int n = (int)plInput[0];
	int nOutNum = (int)plInput[1];
	int nChannels = (int)plInput[2];
	int nInNum = (int)plInput[3];
	long hX = (long)plInput[4];
	long hA = (long)plInput[5];
	long hY = (long)plInput[6];

	return m_math.channel_dot(n, nOutNum, nChannels, nInNum, hX, hA, hY);
}

template long Device<double>::cuda_channel_dot(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_channel_dot(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_channel_compare(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 6, 6))
		return lErr;

	int n = (int)plInput[0];
	int nOutNum = (int)plInput[1];
	int nChannels = (int)plInput[2];
	int nInNum = (int)plInput[3];
	long hX = (long)plInput[4];
	long hY = (long)plInput[5];

	return m_math.channel_compare(n, nOutNum, nChannels, nInNum, hX, hY);
}

template long Device<double>::cuda_channel_compare(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_channel_compare(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_channel_fill(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 8, 8))
		return lErr;

	int n = (int)plInput[0];
	int nOutNum = (int)plInput[1];
	int nChannels = (int)plInput[2];
	int nInNum = (int)plInput[3];
	long hX = (long)plInput[4];
	int nLabelDim = (int)plInput[5];
	long hLabels = (long)plInput[6];
	long hY = (long)plInput[7];

	return m_math.channel_fill(n, nOutNum, nChannels, nInNum, hX, nLabelDim, hLabels, hY);
}

template long Device<double>::cuda_channel_fill(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_channel_fill(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_channel_fillfrom(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 7, 7))
		return lErr;

	int n = (int)plInput[0];
	int nOutNum = (int)plInput[1];
	int nChannels = (int)plInput[2];
	int nInNum = (int)plInput[3];
	long hX = (long)plInput[4];
	long hY = (long)plInput[5];
	int nDir = (int)plInput[6];

	return m_math.channel_fillfrom(n, nOutNum, nChannels, nInNum, hX, hY, nDir);
}

template long Device<double>::cuda_channel_fillfrom(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_channel_fillfrom(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_channel_copy(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 9, 9))
		return lErr;

	int n = (int)plInput[0];
	int nOutNum = (int)plInput[1];
	int nChannels = (int)plInput[2];
	int nBlocks = (int)plInput[3];
	int nInNum = (int)plInput[4];
	int nOffset = (int)plInput[5];
	long hX = (long)plInput[6];
	long hY = (long)plInput[7];
	int nDir = (int)plInput[8];

	return m_math.channel_copy(n, nOutNum, nChannels, nBlocks, nInNum, nOffset, hX, hY, nDir);
}

template long Device<double>::cuda_channel_copy(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_channel_copy(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_channel_copyall(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 6, 6))
		return lErr;

	int n = (int)plInput[0];
	int nOutNum = (int)plInput[1];
	int nChannels = (int)plInput[2];
	int nInNum = (int)plInput[3];
	long hX = (long)plInput[4];
	long hY = (long)plInput[5];

	return m_math.channel_copyall(n, nOutNum, nChannels, nInNum, hX, hY);
}

template long Device<double>::cuda_channel_copyall(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_channel_copyall(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_channel_duplicate(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 6, 6))
		return lErr;

	int n = (int)plInput[0];
	int nOutNum = (int)plInput[1];
	int nChannels = (int)plInput[2];
	int nInNum = (int)plInput[3];
	long hX = (long)plInput[4];
	long hY = (long)plInput[5];

	return m_math.channel_duplicate(n, nOutNum, nChannels, nInNum, hX, hY);
}

template long Device<double>::cuda_channel_duplicate(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_channel_duplicate(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_im2col(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 15, 15))
		return lErr;

	long hDataIm = (long)plInput[0];
	int nDataImOffset = (int)plInput[1];
	int nChannels = (int)plInput[2];
	int nHeight = (int)plInput[3];
	int nWidth = (int)plInput[4];
	int nKernelH = (int)plInput[5];
	int nKernelW = (int)plInput[6];
	int nPadH = (int)plInput[7];
	int nPadW = (int)plInput[8];
	int nStrideH = (int)plInput[9];
	int nStrideW = (int)plInput[10];
	int nDilationH = (int)plInput[11];
	int nDilationW = (int)plInput[12];
	long hDataCol = (long)plInput[13];
	int nDataColOffset = (int)plInput[14];

	return m_math.im2col(hDataIm, nDataImOffset, nChannels, nHeight, nWidth, nKernelH, nKernelW, nPadH, nPadW, nStrideH, nStrideW, nDilationH, nDilationW, hDataCol, nDataColOffset);
}

template long Device<double>::cuda_im2col(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_im2col(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_im2col_nd(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 13, 13))
		return lErr;

	long hDataIm = (long)plInput[0];
	int nDataImOffset = (int)plInput[1];
	int nNumSpatialAxes = (int)plInput[2];
	int nImCount = (int)plInput[3];
	int nChannelAxis = (int)plInput[4];
	long hImShape = (long)plInput[5];
	long hColShape = (long)plInput[6];
	long hKernelShape = (long)plInput[7];
	long hPad = (long)plInput[8];
	long hStride = (long)plInput[9];
	long hDilation = (long)plInput[10];
	long hDataCol = (long)plInput[11];
	int nDataColOffset = (int)plInput[12];

	return m_math.im2col_nd(hDataIm, nDataImOffset, nNumSpatialAxes, nImCount, nChannelAxis, hImShape, hColShape, hKernelShape, hPad, hStride, hDilation, hDataCol, nDataColOffset);
}

template long Device<double>::cuda_im2col_nd(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_im2col_nd(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_col2im(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 15, 15))
		return lErr;

	long hDataCol = (long)plInput[0];
	int nDataColOffset = (int)plInput[1];
	int nChannels = (int)plInput[2];
	int nHeight = (int)plInput[3];
	int nWidth = (int)plInput[4];
	int nKernelH = (int)plInput[5];
	int nKernelW = (int)plInput[6];
	int nPadH = (int)plInput[7];
	int nPadW = (int)plInput[8];
	int nStrideH = (int)plInput[9];
	int nStrideW = (int)plInput[10];
	int nDilationH = (int)plInput[11];
	int nDilationW = (int)plInput[12];
	long hDataIm = (long)plInput[13];
	int nDataImOffset = (int)plInput[14];

	return m_math.col2im(hDataCol, nDataColOffset, nChannels, nHeight, nWidth, nKernelH, nKernelW, nPadH, nPadW, nStrideH, nStrideW, nDilationH, nDilationW, hDataIm, nDataImOffset);
}

template long Device<double>::cuda_col2im(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_col2im(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_col2im_nd(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 13, 13))
		return lErr;

	long hDataCol = (long)plInput[0];
	int nDataColOffset = (int)plInput[1];
	int nNumSpatialAxes = (int)plInput[2];
	int nColCount = (int)plInput[3];
	int nChannelAxis = (int)plInput[4];
	long hImShape = (long)plInput[5];
	long hColShape = (long)plInput[6];
	long hKernelShape = (long)plInput[7];
	long hPad = (long)plInput[8];
	long hStride = (long)plInput[9];
	long hDilation = (long)plInput[10];
	long hDataIm = (long)plInput[11];
	int nDataImOffset = (int)plInput[12];

	return m_math.col2im_nd(hDataCol, nDataColOffset, nNumSpatialAxes, nColCount, nChannelAxis, hImShape, hColShape, hKernelShape, hPad, hStride, hDilation, hDataIm, nDataImOffset);
}

template long Device<double>::cuda_col2im_nd(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_col2im_nd(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_rng_setseed(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;

	long lSeed = (long)pfInput[0];

	return m_math.rng_setseed(lSeed);
}

template long Device<double>::cuda_rng_setseed(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_rng_setseed(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_rng_uniform(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 2, 2))
		return lErr;
	if (lErr = verifyInput(llInput, plInput, 4, 4))
		return lErr;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	int n = (int)plInput[0];
	T fMin = pfInput[0];
	T fMax = pfInput[1];
	long hY = (long)plInput[3];

	return m_math.rng_uniform(n, fMin, fMax, hY);
}

template long Device<double>::cuda_rng_uniform(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_rng_uniform(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_rng_gaussian(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 2, 2))
		return lErr;
	if (lErr = verifyInput(llInput, plInput, 4, 4))
		return lErr;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	int n = (int)plInput[0];
	T fMu = pfInput[0];
	T fSigma = pfInput[1];
	long hY = (long)plInput[3];

	return m_math.rng_gaussian(n, fMu, fSigma, hY);
}

template long Device<double>::cuda_rng_gaussian(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_rng_gaussian(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_rng_bernoulli(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
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

template long Device<double>::cuda_rng_bernoulli(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_rng_bernoulli(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_sgd_update(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 2, 2))
		return lErr;
	if (lErr = verifyInput(llInput, plInput, 5, 5))
		return lErr;

	int n = (int)plInput[0];
	long hNetParamDiff = (long)plInput[1];
	long hHistoryData = (long)plInput[2];
	T fMomentum = pfInput[0];
	T fLocalRate = pfInput[1];

	return m_math.sgd_update(n, hNetParamDiff, hHistoryData, fMomentum, fLocalRate);
}

template long Device<double>::cuda_sgd_update(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_sgd_update(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_nesterov_update(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 2, 2))
		return lErr;
	if (lErr = verifyInput(llInput, plInput, 5, 5))
		return lErr;

	int n = (int)plInput[0];
	long hNetParamDiff = (long)plInput[1];
	long hHistoryData = (long)plInput[2];
	T fMomentum = pfInput[0];
	T fLocalRate = pfInput[1];

	return m_math.nesterov_update(n, hNetParamDiff, hHistoryData, fMomentum, fLocalRate);
}

template long Device<double>::cuda_nesterov_update(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_nesterov_update(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_adagrad_update(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 2, 2))
		return lErr;
	if (lErr = verifyInput(llInput, plInput, 5, 5))
		return lErr;

	int n = (int)plInput[0];
	long hNetParamDiff = (long)plInput[1];
	long hHistoryData = (long)plInput[2];
	T fDelta = pfInput[0];
	T fLocalRate = pfInput[1];

	return m_math.adagrad_update(n, hNetParamDiff, hHistoryData, fDelta, fLocalRate);
}

template long Device<double>::cuda_adagrad_update(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_adagrad_update(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_adadelta_update(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 3, 3))
		return lErr;
	if (lErr = verifyInput(llInput, plInput, 7, 7))
		return lErr;

	int n = (int)plInput[0];
	long hNetParamDiff = (long)plInput[1];
	long hHistoryData1 = (long)plInput[2];
	long hHistoryData2 = (long)plInput[3];
	T fMomentum = pfInput[0];
	T fDelta = pfInput[1];
	T fLocalRate = pfInput[2];

	return m_math.adadelta_update(n, hNetParamDiff, hHistoryData1, hHistoryData2, fMomentum, fDelta, fLocalRate);
}

template long Device<double>::cuda_adadelta_update(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_adadelta_update(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_adam_update(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 5, 5))
		return lErr;
	if (lErr = verifyInput(llInput, plInput, 9, 9))
		return lErr;

	int n = (int)plInput[0];
	long hNetParamDiff = (long)plInput[1];
	long hValM = (long)plInput[2];
	long hValV = (long)plInput[3];
	T fBeta1 = pfInput[0];
	T fBeta2 = pfInput[1];
	T fEpsHat = pfInput[2];
	T fLearningRate = pfInput[3];
	T fCorrection = pfInput[4];
	
	return m_math.adam_update(n, hNetParamDiff, hValM, hValV, fBeta1, fBeta2, fEpsHat, fLearningRate, fCorrection);
}

template long Device<double>::cuda_adam_update(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_adam_update(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_adamw_update(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 5, 5))
		return lErr;
	if (lErr = verifyInput(llInput, plInput, 11, 11))
		return lErr;

	int n = (int)plInput[0];
	long hNetParamDiff = (long)plInput[1];
	long hValM = (long)plInput[2];
	long hValV = (long)plInput[3];
	T fBeta1 = pfInput[0];
	T fBeta2 = pfInput[1];
	T fEpsHat = pfInput[2];
	T fLearningRate = pfInput[3];
	T fDecayRate = pfInput[4];
	long hNetParamData = (long)plInput[9];
	int nStep = (int)plInput[10];

	return m_math.adamw_update(n, hNetParamDiff, hValM, hValV, fBeta1, fBeta2, fEpsHat, fLearningRate, fDecayRate, hNetParamData, nStep);
}

template long Device<double>::cuda_adamw_update(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_adamw_update(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_rmsprop_update(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 3, 3))
		return lErr;
	if (lErr = verifyInput(llInput, plInput, 6, 6))
		return lErr;

	int n = (int)plInput[0];
	long hNetParamDiff = (long)plInput[1];
	long hHistoryData = (long)plInput[2];
	T fRmsDecay = pfInput[0];
	T fDelta = pfInput[1];
	T fLocalRate = pfInput[2];

	return m_math.rmsprop_update(n, hNetParamDiff, hHistoryData, fRmsDecay, fDelta, fLocalRate);
}

template long Device<double>::cuda_rmsprop_update(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_rmsprop_update(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);



template <class T>
long Device<T>::cuda_combine_data(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 2, 2))
		return lErr;
	if (lErr = verifyInput(llInput, plInput, 7, 7))
		return lErr;

	int nCount = (int)plInput[0];
	long hOriginal = (long)plInput[1];
	long hUpdated = (long)plInput[2];
	T fUpdatedPct = pfInput[0];
	long hServer = (long)plInput[4];
	T fServerPct = pfInput[1];
	long hNewData = (long)plInput[6];

	return m_math.combine_data(nCount, hOriginal, hUpdated, fUpdatedPct, hServer, fServerPct, hNewData);
}

template long Device<double>::cuda_combine_data(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_combine_data(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_mtx_set_diagonal(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;
	if (lErr = verifyInput(llInput, plInput, 4, 4))
		return lErr;

	int nCount = (int)plInput[0];
	int nRows = (int)plInput[1];
	T fVal = pfInput[0];
	long hData = (long)plInput[3];

	return m_math.mtx_set_diagonal(nCount, nRows, fVal, hData);
}

template long Device<double>::cuda_mtx_set_diagonal(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_mtx_set_diagonal(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_mtx_set_diagonal2(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 2, 2))
		return lErr;
	if (lErr = verifyInput(llInput, plInput, 6, 6))
		return lErr;

	int nCount = (int)plInput[0];
	int nRows = (int)plInput[1];
	long hDiagonal = (long)plInput[2];
	T fScaleA = pfInput[0];
	T fScaleB = pfInput[1];
	long hData = (long)plInput[5];

	return m_math.mtx_set_diagonal(nCount, nRows, hDiagonal, fScaleA, fScaleB, hData);
}

template long Device<double>::cuda_mtx_set_diagonal2(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_mtx_set_diagonal2(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_mtx_add_vector(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;
	if (lErr = verifyInput(llInput, plInput, 7, 7))
		return lErr;

	int nOrientation = (int)plInput[0];
	int nWidth = (int)plInput[1];
	int nHeight = (int)plInput[2];
	T fScale = pfInput[0];
	long hA = (long)plInput[4];
	long hB = (long)plInput[5];
	long hY = (long)plInput[6];

	return m_math.mtx_add_vector(nOrientation, nWidth, nHeight, fScale, hA, hB, hY);
}

template long Device<double>::cuda_mtx_add_vector(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_mtx_add_vector(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_mtx_transpose_op(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 0, 2))
		return lErr;
	if (lErr = verifyInput(llInput, plInput, 6, 8))
		return lErr;

	int nOp = (int)plInput[0];
	int nWidth = (int)plInput[1];
	int nHeight = (int)plInput[2];
	long hA = (long)plInput[3];
	long hB = (long)plInput[4];
	long hY = (long)plInput[5];
	T fScaleA = 1.0;
	T fScaleB = 1.0;

	if (lInput > 0)
		fScaleA = pfInput[0];

	if (lInput > 1)
		fScaleB = pfInput[1];

	return m_math.mtx_transpose_op(nOp, nWidth, nHeight, hA, hB, hY, fScaleA, fScaleB);
}

template long Device<double>::cuda_mtx_transpose_op(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_mtx_transpose_op(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_mtx_aggregate_cols(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 5, 5))
		return lErr;

	int nOp = (int)plInput[0];
	int nWidth = (int)plInput[1];
	int nHeight = (int)plInput[2];
	long hA = (long)plInput[3];
	long hY = (long)plInput[4];

	return m_math.mtx_aggregate_cols(nOp, nWidth, nHeight, hA, hY);
}

template long Device<double>::cuda_mtx_aggregate_cols(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_mtx_aggregate_cols(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_mtx_aggregate_rows(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 6, 6))
		return lErr;

	int nOp = (int)plInput[0];
	int nWidth = (int)plInput[1];
	int nHeight = (int)plInput[2];
	long hA = (long)plInput[3];
	long hOnes = (long)plInput[4];
	long hY = (long)plInput[5];

	return m_math.mtx_aggregate_rows(nOp, nWidth, nHeight, hA, hOnes, hY);
}

template long Device<double>::cuda_mtx_aggregate_rows(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_mtx_aggregate_rows(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_mtx_transpose(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 4, 4))
		return lErr;

	int nWidth = (int)plInput[0];
	int nHeight = (int)plInput[1];
	long hA = (long)plInput[2];
	long hY = (long)plInput[3];

	return m_math.mtx_transpose(nWidth, nHeight, hA, hY);
}

template long Device<double>::cuda_mtx_transpose(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_mtx_transpose(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_mtx_meancenter_by_column(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 5, 6))
		return lErr;

	int nWidth = (int)plInput[0];
	int nHeight = (int)plInput[1];
	long hA = (long)plInput[2];
	long hB = (long)plInput[3];
	long hY = (long)plInput[4];
	bool bNormalize = false;

	if (llInput > 5)
		bNormalize = (plInput[5] == 0) ? false : true;

	return m_math.mtx_meancenter_by_column(nWidth, nHeight, hA, hB, hY, bNormalize);
}

template long Device<double>::cuda_mtx_meancenter_by_column(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_mtx_meancenter_by_column(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_mtx_euclidean_dist(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 7, 7))
		return lErr;

	long hX = (long)plInput[0];
	long hY = (long)plInput[1];
	long hOut = (long)plInput[2];
	int n = (int)plInput[3];
	int d = (int)plInput[4];
	int nStart = (int)plInput[5];
	int nEnd = (int)plInput[6];

	return m_math.mtx_euclidean_dist(hX, hY, hOut, n, d, nStart, nEnd);
}

template long Device<double>::cuda_mtx_euclidean_dist(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_mtx_euclidean_dist(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_mtx_dot(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 6, 6))
		return lErr;

	// C(m,k) = A(m,n) * B(n,k)

	int m = (int)plInput[0];	// rows in A, C
	int n = (int)plInput[1];	// cols in A, rows in B
	int k = (int)plInput[2];	// cost in C, B
	long hA = (long)plInput[3];	// m x n matrix (m rows, n cols)
	long hB = (long)plInput[4]; // n x k matrix (n rows, k cols)
	long hC = (long)plInput[5]; // k x m matrix (k rows, m cols)

	return m_math.mtx_dot(m, n, k, hA, hB, hC);
}

template long Device<double>::cuda_mtx_dot(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_mtx_dot(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_mtx_mean(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;
	if (lErr = verifyInput(llInput, plInput, 6, 6))
		return lErr;

	int nWid = (int)plInput[0];	// cols in A and Y
	int nHt = (int)plInput[1];	// rows in A and Y
	long hA = (long)plInput[2];	// 
	long hOnes = (long)plInput[3]; // nWid in length, contains 1 on each column to include in the mean calculation.
	T fAlpha = pfInput[0];
	long hY = (long)plInput[5]; // reduction leaves results in first nHt items of hY

	return m_math.mtx_mean(nWid, nHt, hA, hOnes, fAlpha, hY);
}

template long Device<double>::cuda_mtx_mean(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_mtx_mean(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_mtx_stdev(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 7, 7))
		return lErr;

	int nWid = (int)plInput[0];	// cols in A and Y
	int nHt = (int)plInput[1];	// rows in A and Y
	long hA = (long)plInput[2];	// 
	long hOnes = (long)plInput[3]; // nWid in length, contains 1 on each column to include in the mean calculation.
	long hMean = (long)plInput[4]; // nHt items containing the mean of each row.
	long hWork = (long)plInput[5]; // same size as A.
	long hY = (long)plInput[6]; // reduction leaves results in first nHt items of hY

	return m_math.mtx_stdev(nWid, nHt, hA, hOnes, hMean, hWork, hY);
}

template long Device<double>::cuda_mtx_stdev(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_mtx_stdev(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_mtx_correlation(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 8, 8))
		return lErr;

	int nWid = (int)plInput[0];	// cols in A and Y
	int nHt = (int)plInput[1];	// rows in A and Y
	long hA = (long)plInput[2];	// 
	long hOnes = (long)plInput[3]; // nWid in length, contains 1 on each column to include in the mean calculation.
	long hMean = (long)plInput[4]; // nHt items containing the mean of each row.
	long hStdev = (long)plInput[5]; // nHt items containing the stdev of each row.
	long hWork = (long)plInput[6]; // same size as A.
	long hY = (long)plInput[7]; // reduction leaves results in first nHt items of hY

	return m_math.mtx_correlation(nWid, nHt, hA, hOnes, hMean, hStdev, hWork, hY);
}

template long Device<double>::cuda_mtx_correlation(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_mtx_correlation(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_tsne_update(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 4, 4))
		return lErr;
	if (lErr = verifyInput(llInput, plInput, 9, 9))
		return lErr;

	unsigned int n = (unsigned int)plInput[0];
	T fMomentum = pfInput[0];
	T fLearningRate = pfInput[1];
	long hdY = (long)plInput[3];
	long huY = (long)plInput[4];
	long hGains = (long)plInput[5];
	long hY = (long)plInput[6];
	T fGainFactor1 = pfInput[2];
	T fGainFactor2 = pfInput[3];

	return m_math.tsne_update(n, fMomentum, fLearningRate, hdY, huY, hGains, hY, fGainFactor1, fGainFactor2);
}

template long Device<double>::cuda_tsne_update(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_tsne_update(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_tsne_update_grad(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;
	if (lErr = verifyInput(llInput, plInput, 5, 5))
		return lErr;

	unsigned int n = (unsigned int)plInput[0];
	long hPosF = (long)plInput[1];
	long hNegF = (long)plInput[2];
	T fSumQ = pfInput[0];
	long hdC = (long)plInput[4];

	return m_math.tsne_update_grad(n, hPosF, hNegF, fSumQ, hdC);
}

template long Device<double>::cuda_tsne_update_grad(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_tsne_update_grad(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_tsne_compute_exact_error(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 4, 4))
		return lErr;

	unsigned int n = (unsigned int)plInput[0];
	long hP = (long)plInput[1];
	long hQ = (long)plInput[2];
	long hY = (long)plInput[3];

	return m_math.tsne_compute_exact_error(n, hP, hQ, hY);
}

template long Device<double>::cuda_tsne_compute_exact_error(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_tsne_compute_exact_error(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_tsne_compute_squared_euclidean_distance(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 5, 5))
		return lErr;

	unsigned int n = (unsigned int)plInput[0];
	unsigned int d = (unsigned int)plInput[1];
//	long hW = (long)plInput[2];  // currently not used.
	long hX = (long)plInput[3];
	long hDD = (long)plInput[4];

	HostBuffer<T>* pDD = m_memory.GetHostBuffer(hDD);
	T* pX_on_host = m_memory.GetMemoryToHost(hX);

	lErr = m_math.tsne_compute_squared_euclidean_distance(n, d, pX_on_host, pDD->Data());

	if (pX_on_host != NULL)
		m_memory.FreeHost(pX_on_host);

	return lErr;
}

template long Device<double>::cuda_tsne_compute_squared_euclidean_distance(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_tsne_compute_squared_euclidean_distance(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_tsne_compute_q_matrix(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 4, 4))
		return lErr;

	unsigned int n = (unsigned int)plInput[0];
	long hDD = (long)plInput[1];
	long hQ = (long)plInput[2];
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

template long Device<double>::cuda_tsne_compute_q_matrix(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_tsne_compute_q_matrix(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_tsne_compute_exact_gradient(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;
	if (lErr = verifyInput(llInput, plInput, 8, 8))
		return lErr;

	unsigned int n = (unsigned int)plInput[0];
	unsigned int d = (unsigned int)plInput[1];
	long hY = (long)plInput[2];
	long hP = (long)plInput[3];
	long hQ = (long)plInput[4];
	bool bQisHostMem = (plInput[5] == 1.0) ? true : false;
	long hdC = (long)plInput[6];
	T fSumQ = pfInput[0];

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
		lErr = m_memory.SetMemory(hdC, pdC_on_host, SIZE_MAX, -1);

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

template long Device<double>::cuda_tsne_compute_exact_gradient(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_tsne_compute_exact_gradient(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_tsne_symmetrize_matrix(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 4, 4))
		return lErr;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	unsigned int n = (unsigned int)plInput[0];
	long hRowP = (long)plInput[1];
	long hColP = (long)plInput[2];
	long hValP = (long)plInput[3];
	unsigned int nRowCount = 0;

	if (lErr = m_math.tsne_symmetrize_matrix(n, hRowP, hColP, hValP, &nRowCount))
		return lErr;

	return setOutput(T(nRowCount), plOutput, ppfOutput);
}

template long Device<double>::cuda_tsne_symmetrize_matrix(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_tsne_symmetrize_matrix(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_tsne_compute_knn_bounds(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;
	if (lErr = verifyInput(llInput, plInput, 3, 3))
		return lErr;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	unsigned int n = (unsigned int)plInput[0];
	long hData = (long)plInput[1];
	T fPctInCircle = pfInput[0];
	T fMinX;
	T fMinY;
	T fMaxX;
	T fMaxY;

	if (lErr = m_math.tsne_compute_knn_bounds(n, hData, fPctInCircle, &fMinX, &fMinY, &fMaxX, &fMaxY))
		return lErr;

	// ppfOutput has up to MAX_OUTPUT(16) pre-allocated items
	T* pfOutput = *ppfOutput;

	pfOutput[0] = fMinX;
	pfOutput[1] = fMinY;
	pfOutput[2] = fMaxX;
	pfOutput[3] = fMaxY;

	*ppfOutput = pfOutput;
	*plOutput = 4;

	return 0;
}

template long Device<double>::cuda_tsne_compute_knn_bounds(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_tsne_compute_knn_bounds(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_guassian_blur(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;
	if (lErr = verifyInput(llInput, plInput, 7, 7))
		return lErr;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	int n = (int)plInput[0];
	int c = (int)plInput[1];
	int h = (int)plInput[2];
	int w = (int)plInput[3];
	T fSigma = pfInput[0];
	long hX = (long)plInput[5];
	long hY = (long)plInput[6];

	return m_math.gaussian_blur(n, c, h, w, fSigma, hX, hY);
}

template long Device<double>::cuda_guassian_blur(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_guassian_blur(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_calc_dft(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(llInput, plInput, 4, 4))
		return lErr;

	int n = (int)plInput[0];
	int hX = (long)plInput[1];
	int m = (int)plInput[2];
	int hY = (long)plInput[3];

	return m_math.calc_dft(n, hX, m, hY);
}

template long Device<double>::cuda_calc_dft(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_calc_dft(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);



template <class T>
long Device<T>::cuda_hamming_diff(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;
	if (lErr = verifyInput(llInput, plInput, 5, 8))
		return lErr;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	int n = (int)plInput[0];
	T fThreshold = pfInput[0];
	long hA = (long)plInput[2];
	long hB = (long)plInput[3];
	long hY = (long)plInput[4];
	int nOffA = 0;
	int nOffB = 0;
	int nOffY = 0;

	if (llInput > 5)
		nOffA = (int)plInput[5];

	if (llInput > 6)
		nOffB = (int)plInput[6];

	if (llInput > 7)
		nOffY = (int)plInput[7];

	return m_math.hamming_diff(n, fThreshold, hA, hB, hY, nOffA, nOffB, nOffY);
}

template long Device<double>::cuda_hamming_diff(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_hamming_diff(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::cuda_calc_batch_dist(long lInput, T* pfInput, long llInput, LONGLONG* plInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;
	if (lErr = verifyInput(llInput, plInput, 9, SHRT_MAX))
		return lErr;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	int nDistMethod = (int)plInput[0];
	T fThreshold = pfInput[0];
	int nItemDim = (int)plInput[2];
	long hSrc = (long)plInput[3];
	long hTargets = (long)plInput[4];
	long hWork = (long)plInput[5];
	int nDim0 = (int)plInput[6];
	int nDim1 = (int)plInput[7];

	if (nDim1 != 2)
		return ERROR_PARAM_OUT_OF_RANGE;

	if (nDim0 < 0)
		return ERROR_PARAM_OUT_OF_RANGE;

	T* pfOutput = NULL;
	if (lErr = m_memory.AllocHost(nDim0, &pfOutput, NULL, false, false, false))
		return lErr;

	lErr = m_math.calc_batch_dist(nDistMethod, fThreshold, nItemDim, hSrc, hTargets, hWork, nDim0, nDim1, &plInput[8], pfOutput);

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

template long Device<double>::cuda_calc_batch_dist(long lInput, double* pfInput, long llInput, LONGLONG* plInput, long* plOutput, double** ppfOutput);
template long Device<float>::cuda_calc_batch_dist(long lInput, float* pfInput, long llInput, LONGLONG* plInput, long* plOutput, float** ppfOutput);

//end device.cu