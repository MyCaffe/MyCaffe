//=============================================================================
//	FILE:	device.h
//
//	DESC:	This file implements the class used to manage the underlying
//			device.
//=============================================================================
#ifndef __DEVICE_CU__
#define __DEVICE_CU__

//=============================================================================
//	Includes
//=============================================================================

#include "parent.h"

//=============================================================================
//	Classes
//=============================================================================
//-----------------------------------------------------------------------------
//	Device Class
//
//	The device class implements manages underying GPU device.
//-----------------------------------------------------------------------------
template <class T>
class Device
{
	Parent<T>* m_pParent;
	T* m_pInputMem;
	T* m_pOutputMem;
	T* m_pOutputWork;
	long m_hInputMem;
	long m_hInputWork;
	int m_nInputCount;
	long m_hOutputMem;
	long m_hOutputWork;
	int m_nOutputCount;
	int m_nInputFields;
	int m_nOutputFields;
	int m_nDepth;
	bool m_bCallProcessDataAfterAdd;

	protected:
		long verifyInput(long lInput, T* pfInput, long lMin, long lMax, bool bExact = false);
		long verifyOutput(long* plOutput, T** ppfOutput);
		long setOutput(long hHandle, long* plOutput, T** ppfOutput);
		long setOutput(T fVal, long* plOutput, T** ppfOutput);

	public:
		Device();
		~Device();

		void Connect(Parent<T>* pParent);

		long Initialize(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long CleanUp(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long SetMemory(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long AddData(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long ProcessData(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long GetVisualization(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long Clear(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
};


//=============================================================================
//	Inline Methods
//=============================================================================

template <class T>
inline long Device<T>::verifyInput(long lInput, T* pfInput, long lMin, long lMax, bool bExact)
{
	if (lInput < lMin || lInput > lMax)
		return ERROR_INVALID_PARAMETER;

	if (lMin == 0 && lMax == 0)
		return 0;

	if (bExact && lInput != lMin && lInput != lMax)
		return ERROR_INVALID_PARAMETER;

	if (pfInput == NULL)
		return ERROR_INVALID_PARAMETER;

	return 0;
}

template <class T>
inline long Device<T>::verifyOutput(long* plOutput, T** ppfOutput)
{
	if (plOutput == NULL)
		return ERROR_INVALID_PARAMETER;

	if (ppfOutput == NULL)
		return ERROR_INVALID_PARAMETER;

	return 0;
}

template <class T>
inline long Device<T>::setOutput(long hHandle, long* plOutput, T** ppfOutput)
{
	*plOutput = 1;
	(*ppfOutput)[0] = (T)hHandle;

	return 0;
}

template <class T>
inline long Device<T>::setOutput(T fVal, long* plOutput, T** ppfOutput)
{
	*plOutput = 1;
	(*ppfOutput)[0] = fVal;

	return 0;
}


//=============================================================================
//	Device Methods
//=============================================================================

template <class T>
inline Device<T>::Device()
{
	m_pInputMem = NULL;
	m_pOutputMem = NULL;
	m_pOutputWork = NULL;
	m_nInputCount = 0;
	m_nOutputCount = 0;
	m_hInputMem = 0;
	m_hInputWork = 0;
	m_hOutputMem = 0;
	m_hOutputWork = 0;
	m_bCallProcessDataAfterAdd = false;
}

template <class T>
inline Device<T>::~Device()
{
}

template <class T>
inline void Device<T>::Connect(Parent<T>* pParent)
{
	m_pParent = pParent;
}

#endif // __DEVICE_CU__