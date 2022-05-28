//=============================================================================
//	FILE:	nccl.cu
//
//	DESC:	This file implements the mutli-gpu communication functionality
//
//	NOTES:  Uses the 'Nickel' NCCL library located at: https://github.com/NVIDIA/nccl
//=============================================================================

#include "util.h"
#include "nccl.h"
#include "memory.h"
#include "..\_nccl\nccl.h"
#include <nvapi.h>
#include <nvml.h>


//=============================================================================
//	Function Definitions
//=============================================================================

typedef ncclResult_t (*LPNCCLCOMMINITRANK)(ncclComm_t* comm, int ndev, ncclUniqueId commId, int rank);
typedef ncclResult_t (*LPNCCLCOMMINITALL)(ncclComm_t* comm, int ndev, const int* devlist);
typedef ncclResult_t (*LPNCCLCOMMDESTROY)(ncclComm_t comm);
typedef const char* (*LPNCCLGETERRORSTRING)(ncclResult_t result);
typedef ncclResult_t (*LPNCCLALLREDUCE)(const void* sendbuff, void* recvbuff, int count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);
typedef ncclResult_t (*LPNCCLBCAST)(void* buff, int count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream);

extern HMODULE g_hModule;

//=============================================================================
//	Private Classes
//=============================================================================

class Data
{
public: 
	ncclUniqueId m_id;
	ncclComm_t m_comm;
	int m_nCount;
	int m_nRank;
	HINSTANCE m_hDLL;
	LPNCCLCOMMINITALL m_pCommInitAll;
	LPNCCLCOMMINITRANK m_pCommInitRank;
	LPNCCLCOMMDESTROY m_pCommDestroy;
	LPNCCLALLREDUCE m_pAllReduce;
	LPNCCLBCAST m_pBcast;
	LPNCCLGETERRORSTRING m_pGetErrorString;

	Data(int nCount, int nRank, char* szId)
	{
		m_hDLL = NULL;
		m_comm = NULL;
		m_nCount = nCount;
		m_nRank = nRank;
		strncpy(m_id.internal, szId, NCCL_UNIQUE_ID_BYTES);
	}

	LONG Initialize()
	{
		TCHAR szPath[1024] = { 0 };
		TCHAR szNcclPath[1024] = { 0 };
		TCHAR* pszVer = NULL;
		LONG lErr = GetModuleFileName(g_hModule, szPath, sizeof(szPath));
		if (lErr == 0 || lErr == sizeof(szPath))
			return ERROR_PARAM_NULL;

		int nLen = (int)_tcslen(szPath);
		for (int i = nLen - 1; i >= 0; i--)
		{
			if (szPath[i] == _T('\\') && i < nLen-1)
			{
				for (int j = i; j < nLen; j++)
				{
					if (szPath[j] == _T('.'))
					{
						pszVer = &szPath[j];
						break;
					}
				}

				_tcsncpy(szNcclPath, szPath, i + 1);
				break;
			}
		}

		_tcscat(szNcclPath, _T("nccl64_134"));

		if (pszVer != NULL)
			_tcscat(szNcclPath, pszVer);
		else
			_tcscat(szNcclPath, _T(".dll"));

		m_hDLL = LoadLibrary(szNcclPath);
		if (m_hDLL == NULL)
			return ERROR_CUDA_MISSING_NCCL64DLL;

		m_pCommInitAll = (LPNCCLCOMMINITALL)GetProcAddress(m_hDLL, "ncclCommInitAll");
		if (m_pCommInitAll == NULL)
			return ERROR_PARAM_NULL;

		m_pCommInitRank = (LPNCCLCOMMINITRANK)GetProcAddress(m_hDLL, "ncclCommInitRank");
		if (m_pCommInitRank == NULL)
			return ERROR_PARAM_NULL;

		m_pCommDestroy = (LPNCCLCOMMDESTROY)GetProcAddress(m_hDLL, "ncclCommDestroy");
		if (m_pCommDestroy == NULL)
			return ERROR_PARAM_NULL;

		m_pAllReduce = (LPNCCLALLREDUCE)GetProcAddress(m_hDLL, "ncclAllReduce");
		if (m_pAllReduce == NULL)
			return ERROR_PARAM_NULL;

		m_pBcast = (LPNCCLBCAST)GetProcAddress(m_hDLL, "ncclBcast");
		if (m_pBcast == NULL)
			return ERROR_PARAM_NULL;

		m_pGetErrorString = (LPNCCLGETERRORSTRING)GetProcAddress(m_hDLL, "ncclGetErrorString");
		if (m_pGetErrorString == NULL)
			return ERROR_PARAM_NULL;

		return 0;
	}

	ncclResult_t NcclCommInitRank(ncclComm_t* comm, int ndev, ncclUniqueId commId, int rank)
	{
		return (*m_pCommInitRank)(comm, ndev, commId, rank);
	}

	ncclResult_t NcclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist)
	{
		return (*m_pCommInitAll)(comm, ndev, devlist);
	}

	ncclResult_t NcclCommDestroy(ncclComm_t comm)
	{
		return (*m_pCommDestroy)(comm);
	}

	const char* NcclGetErrorString(ncclResult_t result)
	{
		return (*m_pGetErrorString)(result);
	}

	ncclResult_t NcclAllReduce(const void* sendbuff, void* recvbuff, int count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream)
	{
		return (*m_pAllReduce)(sendbuff, recvbuff, count, datatype, op, comm, stream);
	}

	ncclResult_t NcclBcast(void* buff, int count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream)
	{
		return (*m_pBcast)(buff, count, datatype, root, comm, stream);
	}

	~Data()
	{
		if (m_comm != NULL)
		{
			NcclCommDestroy(m_comm);
			m_comm = NULL;
		}

		if (m_hDLL != NULL)
		{
			FreeLibrary(m_hDLL);
			m_hDLL = NULL;
			m_pCommDestroy = NULL;
			m_pCommInitAll = NULL;
			m_pCommInitRank = NULL;
			m_pAllReduce = NULL;
			m_pBcast = NULL;
			m_pGetErrorString = NULL;
		}
	}
};


//=============================================================================
//	Class Methods
//=============================================================================

template <class T>
long ncclHandle<T>::isDisplayConnectedToGpu(int nGpuID, bool* pbIsDisplayOn)
{
	LONG lErr;
	nvmlReturn_t res;
	void* device;

	*pbIsDisplayOn = false;

	char rgPciID[256];
	if (lErr = cudaDeviceGetPCIBusId(rgPciID, 255, nGpuID))
		return lErr;

	if ((res = nvmlDeviceGetHandleByPciBusId_v2(rgPciID, (nvmlDevice_t*)&device)) != NVML_SUCCESS)
		return (int)res;

	nvmlEnableState_t active;
	if ((res = nvmlDeviceGetDisplayMode((nvmlDevice_t)device, &active)) == NVML_SUCCESS)
	{
		if (active == NVML_FEATURE_ENABLED)
			*pbIsDisplayOn = true;
	}

	return 0;
}


template <class T>
void ncclHandle<T>::setBufferSize(long lBufferCount)
{
	if (lBufferCount > 0)
	{
		lBufferCount *= sizeof(T);
		lBufferCount /= 64;
		lBufferCount = (lBufferCount + 1) * 64;

		char szBuffer[256];
		snprintf(szBuffer, 255, "NCCL_BUFFSIZE=%ld", lBufferCount);
		putenv(szBuffer);
	}
}

template <class T>
long ncclHandle<T>::Initialize(Memory<T>* pMem, Math<T>* pMath, int nGpuID, int nCount, int nRank, char* szId)
{
	long lErr;
	int nDevCount;

	m_nGpuID = nGpuID;
	Update(pMem, pMath);

	if (lErr = cudaGetDeviceCount(&nDevCount))
		return lErr;

	if (nGpuID < 0 || nGpuID >= nDevCount)
		return ERROR_PARAM_OUT_OF_RANGE;

	if (!m_bNvmlInit)
	{
		nvmlReturn_t res;
		if ((res = nvmlInit_v2()) != NVML_SUCCESS)
			return (int)res;

		m_bNvmlInit = true;
	}

	bool bDisplayOn = false;
	if (lErr = isDisplayConnectedToGpu(nGpuID, &bDisplayOn))
		return lErr;

	if (bDisplayOn)
		return ERROR_CUDA_NOTSUPPORED_ON_DISPLAYGPU;

	m_pData = new Data(nCount, nRank, szId);
	if (m_pData == NULL)
		return ERROR_MEMORY_OUT;

	if (lErr = m_pData->Initialize())
		return lErr;

	return 0;
}

template long ncclHandle<double>::Initialize(Memory<double>* pMem, Math<double>* pMath, int nGpuID, int nCount, int nRank, char* szId);
template long ncclHandle<float>::Initialize(Memory<float>* pMem, Math<float>* pMath, int nGpuID, int nCount, int nRank, char* szId);


template <class T>
long ncclHandle<T>::Update(Memory<T>* pMem, Math<T>* pMath)
{
	m_pMem = pMem;
	m_pMath = pMath;
	m_pMemCol = pMem->GetMemoryCollection();
	m_nRefCount++;
	return 0;
}

template long ncclHandle<double>::Update(Memory<double>* pMem, Math<double>* pMath);
template long ncclHandle<float>::Update(Memory<float>* pMem, Math<float>* pMath);


template <class T>
long ncclHandle<T>::CleanUp()
{
	m_nRefCount--;

	if (m_nRefCount == 0)
	{
		if (m_pData != NULL)
		{
			delete m_pData;
			m_pData = NULL;
		}
	}

	if (m_bNvmlInit)
	{
		nvmlShutdown();
		m_bNvmlInit = false;
	}

	return 0;
}

template long ncclHandle<double>::CleanUp();
template long ncclHandle<float>::CleanUp();


template <class T>
LPCSTR ncclHandle<T>::GetErrorString(long lErr)
{
	return m_pData->NcclGetErrorString((ncclResult_t)lErr);
}

template LPCSTR ncclHandle<double>::GetErrorString(long lErr);
template LPCSTR ncclHandle<float>::GetErrorString(long lErr);


template <class T>
long ncclHandle<T>::InitSingleProcess(long lBufferCount, int nCount, ncclHandle<T>* rgHandles[])
{
	LONG lErr;
	ncclComm_t* rgComm = (ncclComm_t*)malloc(sizeof(ncclComm_t) * nCount);
	if (rgComm == NULL)
		return ERROR_MEMORY_OUT;

	int* rgGpu = (int*)malloc(sizeof(int) * nCount);
	if (rgGpu == NULL)
	{
		free(rgComm);
		return ERROR_MEMORY_OUT;
	}

	for (int i = 0; i < nCount; i++)
	{
		rgGpu[i] = rgHandles[i]->m_nGpuID;
	}

	setBufferSize(lBufferCount);
	lErr = m_pData->NcclCommInitAll(rgComm, nCount, rgGpu);
	if (!lErr)
	{
		for (int i = 0; i < nCount; i++)
		{
			rgHandles[i]->m_pData->m_comm = rgComm[i];
		}
	}

	free(rgComm);
	free(rgGpu);

	return lErr;
}

template long ncclHandle<double>::InitSingleProcess(long lBufferCount, int nCount, ncclHandle<double>* rgHandles[]);
template long ncclHandle<float>::InitSingleProcess(long lBufferCount, int nCount, ncclHandle<float>* rgHandles[]);


template <class T>
long ncclHandle<T>::InitMultiProcess(long lBufferCount)
{
	setBufferSize(lBufferCount);
	return m_pData->NcclCommInitRank(&m_pData->m_comm, m_pData->m_nCount, m_pData->m_id, m_pData->m_nRank);
}

template long ncclHandle<double>::InitMultiProcess(long lBufferCount);
template long ncclHandle<float>::InitMultiProcess(long lBufferCount);


template <class T>
long ncclHandle<T>::Broadcast(long hStream, long hX, int nCount)
{
	ncclDataType_t type = (sizeof(T) == sizeof(double)) ? ncclDouble : ncclFloat;
	MemoryItem* pX;
	LONG lErr;

	if (lErr = m_pMemCol->GetData(hX, &pX))
		return lErr;

	T* x = (T*)pX->Data();

	if (lErr = cudaSetDevice(m_nGpuID))
		return lErr;

	cudaStream_t stream = cudaStreamDefault;
	if (hStream != 0)
		stream = m_pMem->GetStream(hStream);

	if (lErr = m_pData->NcclBcast(x, nCount, type, 0, m_pData->m_comm, stream))
		return lErr;

	return 0;
}

template long ncclHandle<double>::Broadcast(long hStream, long hData, int nCount);
template long ncclHandle<float>::Broadcast(long hStream, long hData, int nCount);


template <class T>
long ncclHandle<T>::AllReduce(long hStream, long hX, int nCount, NCCL_OP op, T fScale)
{
	long lErr;
	ncclRedOp_t ncclop = ncclSum;

	if (op == NCCL_PROD)
		ncclop = ncclProd;
	else if (op == NCCL_MIN)
		ncclop = ncclMin;
	else if (op == NCCL_MAX)
		ncclop = ncclMax;

	MemoryItem* pX;

	if (lErr = m_pMemCol->GetData(hX, &pX))
		return lErr;

	T* x = (T*)pX->Data();

	if (lErr = cudaSetDevice(m_nGpuID))
		return lErr;

	cudaStream_t stream = cudaStreamDefault;
	if (hStream != 0)
		stream = m_pMem->GetStream(hStream);

	ncclDataType_t type = (sizeof(T) == sizeof(double)) ? ncclDouble : ncclFloat;
	if (lErr = m_pData->NcclAllReduce(x, x, nCount, type, ncclop, m_pData->m_comm, stream))
		return lErr;

	if (fScale != T(1.0))
		return m_pMath->scal(nCount, fScale, hX, 0, hStream);

	return 0;
}

template long ncclHandle<double>::AllReduce(long hStream, long hData, int nCount, NCCL_OP op, double dfScale);
template long ncclHandle<float>::AllReduce(long hStream, long hData, int nCount, NCCL_OP op, float fScale);

// end