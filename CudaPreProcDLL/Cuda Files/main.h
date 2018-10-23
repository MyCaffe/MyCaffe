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

#include "parent.h"
#include "device.h"

//=============================================================================
//	Defines
//=============================================================================

const int CUDAPP_FN_INITIALIZE = 1;
const int CUDAPP_FN_CLEANUP = 2;
const int CUDAPP_FN_SETMEMORY = 3;
const int CUDAPP_FN_ADDDATA = 4;
const int CUDAPP_FN_PROCESSDATA = 5;
const int CUDAPP_FN_GETVISUALIZATION = 6;
const int CUDAPP_FN_CLEAR = 7;

//=============================================================================
//	Typedefs
//=============================================================================

//=============================================================================
//	Kernel Classses
//=============================================================================

template <class T>
class Kernel
{
	long m_lKernelIdx;
	Parent<T> m_parent;
	Device<T> m_device;

public:
	Kernel() : m_parent(), m_device()
	{
		m_lKernelIdx = 0;
	}

	~Kernel()
	{
		CleanUp();
	}

	long Initialize(HMODULE hParent, long lKernelIdx)
	{
		LONG lErr;

		m_lKernelIdx = lKernelIdx;

		if (lErr = m_parent.Initialize(hParent, lKernelIdx))
			return lErr;

		m_device.Connect(&m_parent);

		return 0;
	}

	void CleanUp()
	{
		m_lKernelIdx = 0;
		m_parent.CleanUp();
	}

	long Run(long lfnIdx, T* pfInput, long lCount, T** ppfOutput, long* plCount);
};


#endif // #ifndef __MAIN_H_
