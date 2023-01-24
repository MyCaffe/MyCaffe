//=============================================================================
//	main.h
//
//	The kernel manages the interface to the DLL.
//=============================================================================
#ifndef __MAIN_H_
#define __MAIN_H_

//=============================================================================
//	Includes
//=============================================================================
#include "util.h"
#include "device.h"

#include "..\inc\FunctionIDs.h"

//=============================================================================
//	Defines
//=============================================================================

const int MAX_KERNELS = 2048;

const int CUDA_DLL_KERNEL_COPY_NCCL = -10;
const int CUDA_DLL_CREATEKERNEL		= -9;
const int CUDA_DLL_DESTROYKERNEL	= -8;
const int CUDA_DLL_SETKERNEL		= -7;
const int CUDA_DLL_KERNEL_ADD		= -5;
const int CUDA_DLL_KERNEL_MEMCPY	= -4;

const int CUDA_DLL_FREEMEM			= -1;
const int CUDA_DLL_INITIALIZE		= -2;
const int CUDA_DLL_CLEANUP			= -3;

const int DEVICE_PROP_NAME = 1;
const int DEVICE_PROP_MULTIGPUBOARDGROUPID = 2;

const int CUDA_FN_GET_DEVICE_NAME	= 1000;
const int CUDA_FN_GET_P2P_INFO		= 1001;
const int CUDA_FN_GET_DEVICE_INFO   = 1002;


//=============================================================================
//	Typedefs
//=============================================================================

//=============================================================================
//	Kernel Classses
//=============================================================================

template <class T>
class Kernel
{
	Device<T> m_device;

public:
	Kernel() : m_device()
	{
	}

	~Kernel()
	{
		CleanUp();
	}

	long Initialize(T* pfInput, long lCount)
	{
		LONG lErr;
		
		if (lErr = m_device.Initialize())
			return lErr;

		return Run(CUDA_FN_SETDEVICE, pfInput, lCount, (LONGLONG*)NULL, 0, NULL, NULL);
	}

	void CleanUp()
	{
		m_device.CleanUp();
	}

	int GetDevice()
	{
		return m_device.GetDevice();
	}

	long GetPointer(HANDLE_TYPE ht, long hHandle, void** ppPtr)
	{
		return m_device.GetPointer(ht, hHandle, ppPtr);
	}

	HostBuffer<T>* GetHostBuffer(long hHandle)
	{
		return m_device.GetHostBuffer(hHandle);
	}

	cudaStream_t GetStream(long hStream)
	{
		return m_device.GetStream(hStream);
	}

	cudnnHandle_t GetCuDNN(long h)
	{
		return m_device.GetCuDNN(h);
	}

	cudnnTensorDescriptor_t GetTensorDesc(long hDesc)
	{
		return m_device.GetTensorDesc(hDesc);
	}

	cudnnFilterDescriptor_t GetFilterDesc(long hDesc)
	{
		return m_device.GetFilterDesc(hDesc);
	}

	cudnnConvolutionDescriptor_t GetConvolutionDesc(long hDesc)
	{
		return m_device.GetConvolutionDesc(hDesc);
	}

	long GetMemory(long hHandle, MemoryItem** ppItem)
	{
		return m_device.GetMemory(hHandle, ppItem);
	}

	long AllocHost(long lCount, T** ppfOutput, T* pSrc, bool bSrcOnDevice = false)
	{
		return m_device.AllocHost(lCount, ppfOutput, pSrc, bSrcOnDevice);
	}

	long FreeHost(T* pfInput)
	{
		return m_device.FreeHost(pfInput);
	}

	long FreeHost(LPTSTR pfInput)
	{
		return m_device.FreeHost(pfInput);
	}

	ncclHandle<T>* GetNCCL(long hNccl)
	{
		return m_device.GetNccl(hNccl);
	}

	long SetNCCL(ncclHandle<T>* pNccl, T** ppfOutput, long* plCount)
	{
		return m_device.SetNccl(pNccl, plCount, ppfOutput);
	}

	long Run(long lfnIdx, T* pfInput, long lCount, LONGLONG* plInput, long llCount, T** ppfOutput, long* plCount);

	long Run(long lfnIdx, T* pfInput, long lCount, LPTSTR pszInput, T** ppfOutput, long* plCount);

	long Run(long lfnIdx, T* pfInput, long lCount, T** ppfOutput, long* plCount, LPTSTR szErr, long lErrMax);

	long CreateExtensionFloat(HMODULE hParent, LONG lKernelIdx, LPTSTR pszInput, T** ppfOutput, long* plCount)
	{
		return m_device.CreateExtensionFloat(hParent, lKernelIdx, plCount, ppfOutput, pszInput);
	}

	long CreateExtensionDouble(HMODULE hParent, LONG lKernelIdx, LPTSTR pszInput, T** ppfOutput, long* plCount)
	{
		return m_device.CreateExtensionDouble(hParent, lKernelIdx, plCount, ppfOutput, pszInput);
	}

	long Query(long lfnIdx, LONG* pfInput, long lCount, LPTSTR* ppfOutput);
};


#endif // #ifndef __MAIN_H_
