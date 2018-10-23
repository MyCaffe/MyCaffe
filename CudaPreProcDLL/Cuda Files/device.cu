//=============================================================================
//	FILE:	device.cu
//
//	DESC:	This file implements the base class used to manage the underlying
//			GPU device.
//=============================================================================

#include <limits.h>
#include <Windows.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cudnn.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include "device.h"

//=============================================================================
//	local constants
//=============================================================================

//=============================================================================
//	Class Methods
//=============================================================================

template <class T>
long Device<T>::Initialize(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	long lErr;

	if (lErr = verifyInput(lInput, pfInput, 2, 3))
		return lErr;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	int nFields = (int)pfInput[0];
	int nDepth = (int)pfInput[1];
	bool bCallProcessAfterAdd = false;

	if (lInput > 2)
		bCallProcessAfterAdd = (pfInput[0] != 0) ? true : false;

	m_nInputFields = nFields;
	m_nOutputFields = nFields;
	m_nDepth = nDepth;
	m_bCallProcessDataAfterAdd = bCallProcessAfterAdd;

	T fVal = (T)nFields;
	return setOutput(fVal, plOutput, ppfOutput);
}

template long Device<double>::Initialize(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::Initialize(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::CleanUp(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	return 0;
}

template long Device<double>::CleanUp(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::CleanUp(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::SetMemory(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	long lErr;

	if (lErr = verifyInput(lInput, pfInput, 8, 8))
		return lErr;

	int nInputCount = (int)pfInput[0];
	long hInputMem = (long)pfInput[1];
	int nInputWorkCount = (int)pfInput[2];
	long hInputWork = (long)pfInput[3];
	int nOutputCount = (int)pfInput[4];
	long hOutputMem = (long)pfInput[5];
	int nOutputWorkCount = (int)pfInput[6];
	long hOutputWork = (long)pfInput[7];

	if (nInputCount != nInputWorkCount)
		return ERROR_INVALID_PARAMETER;

	if (nOutputCount != nOutputWorkCount)
		return ERROR_INVALID_PARAMETER;

	if (lErr = m_pParent->GetMemoryPointer(hInputMem, (void**)&m_pInputMem))
		return lErr;

	m_hInputMem = hInputMem;
	m_hInputWork = hInputWork;
	m_nInputCount = nInputCount;

	if (lErr = m_pParent->GetMemoryPointer(hOutputMem, (void**)&m_pOutputMem))
		return lErr;

	if (lErr = m_pParent->GetMemoryPointer(hOutputWork, (void**)&m_pOutputWork))
		return lErr;

	m_hOutputMem = hOutputMem;
	m_hOutputWork = hOutputWork;
	m_nOutputCount = nOutputCount;

	return 0;
}

template long Device<double>::SetMemory(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::SetMemory(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::AddData(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	long lErr;

	cudaGetLastError();

	if (lErr = verifyInput(lInput, pfInput, 1, INT_MAX))
		return lErr;

	if ((lInput % m_nInputFields) != 0)
		return ERROR_INVALID_PARAMETER;

	if (m_pInputMem == NULL || m_pOutputMem == NULL)
		return ERROR_INTERNAL_ERROR;

	int nDepth = lInput / m_nInputFields;

	for (int i = 0; i < m_nInputFields; i++)
	{
		int nIdx = i * m_nDepth;

		if (m_nDepth > nDepth)
		{
			// Shift the existing data back (on the input work mem)
			if (lErr = m_pParent->cuda_copy(m_nDepth - nDepth, m_hInputMem, m_hInputWork, nIdx + nDepth, nIdx))
				return lErr;

			if (lErr = m_pParent->cuda_copy(m_nDepth - nDepth, m_hInputWork, m_hInputMem, nIdx, nIdx))
				return lErr;
		}

		// Copy the new Data to the GPU at the end of the input data.
		int nInputSrc = i * nDepth;
		int nInputDst = nIdx + (m_nDepth - nDepth);
		int nInputSize = nDepth * sizeof(T);

		if (lErr = cudaMemcpy(&m_pInputMem[nInputDst], &pfInput[nInputSrc], nInputSize, cudaMemcpyHostToDevice))
			return lErr;
	}

	if (m_bCallProcessDataAfterAdd)
	{
		if (lErr = ProcessData(0, NULL, NULL, NULL))
			return lErr;
	}

	return 0;
}

template long Device<double>::AddData(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::AddData(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::ProcessData(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	long lErr;

	cudaGetLastError();

	if (m_pInputMem == NULL || m_pOutputMem == NULL)
		return ERROR_INTERNAL_ERROR;

	for (int i = 0; i < m_nInputFields; i++)
	{
		int nIdx = i * m_nDepth;

		// Copy the input data to the output data.
		if (lErr = m_pParent->cuda_copy(m_nDepth, m_hInputMem, m_hOutputMem, nIdx, nIdx))
			return lErr;

		// Center the output by subtracting the mean.
		thrust::device_ptr<T> work1(m_pOutputMem);
		T fSum = thrust::reduce(work1 + nIdx, work1 + nIdx + m_nDepth);
		T fMean = fSum / (T)m_nDepth;

		if (lErr = m_pParent->cuda_add_scalar(m_nDepth, -fMean, m_hOutputMem, nIdx))
			return lErr;

		// Normalize by dividing out the std deviation.
		if (lErr = m_pParent->cuda_powx(m_nDepth, m_hOutputMem, (T)2.0, m_hOutputWork, nIdx, nIdx))
			return lErr;

		thrust::device_ptr<T> work2(m_pOutputWork);
		T fSumSq = thrust::reduce(work2 + nIdx, work2 + nIdx + m_nDepth);
		T fStdDev = (fSumSq == 0) ? 0 : sqrt(fSumSq / (T)m_nDepth);

		if (lErr = m_pParent->cuda_mul_scalar(m_nDepth, ((T)1.0) / fStdDev, m_hOutputMem, nIdx))
			return lErr;
	}

	return 0;
}

template long Device<double>::ProcessData(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::ProcessData(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::GetVisualization(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	long lErr;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	return 0;
}

template long Device<double>::GetVisualization(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::GetVisualization(long lInput, float* pfInput, long* plOutput, float** ppfOutput);


template <class T>
long Device<T>::Clear(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	return 0;
}

template long Device<double>::Clear(long lInput, double* pfInput, long* plOutput, double** ppfOutput);
template long Device<float>::Clear(long lInput, float* pfInput, long* plOutput, float** ppfOutput);

//end device.cu