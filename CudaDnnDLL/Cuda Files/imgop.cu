//=============================================================================
//	FILE:	imgop.cu
//
//	DESC:	This file implements the image operations taking place on the GPU.
//=============================================================================

#include "util.h"
#include "memory.h"
#include "imgop.h"

//=============================================================================
//	Helper functions.
//=============================================================================

template <typename T>
inline __device__ T truncate(const T val);

template<>
inline __device__ float truncate(const float val)
{
	if (val < 0.0)
		return 0.0;
	else if (val > 255.0)
		return 255.0;
	else
		return val;
}

template<>
inline __device__ double truncate(const double val)
{
	if (val < 0.0)
		return 0.0;
	else if (val > 255.0)
		return 255.0;
	else
		return val;
}

//=============================================================================
//	Class Methods
//=============================================================================

template <class T>
long imgopHandle<T>::Initialize(Memory<T>* pMem, int nNum, T fBrightnessProb, T fBrightnessDelta, T fContrastProb, T fContrastLower, T fContrastUpper, T fSaturationProb, T fSaturationLower, T fSaturationUpper, long lRandomSeed)
{
	LONG lErr;

	m_pMem = pMem;
	m_nNum = nNum;

	try
	{
		if (lErr = cudaMalloc(&m_pOrdering, nNum * sizeof(T)))
			throw lErr;

		if (lErr = cudaMalloc(&m_pBrightness, nNum * sizeof(T)))
			throw lErr;

		if (lErr = cudaMalloc(&m_pContrast, nNum * sizeof(T)))
			throw lErr;

		if (lErr = cudaMalloc(&m_pSaturation, nNum * sizeof(T)))
			throw lErr;

		m_fBrightnessProb = fBrightnessProb;
		m_fBrightnessDelta = fBrightnessDelta;
		m_fContrastProb = fContrastProb;
		m_fContrastLower = fContrastLower;
		m_fContrastUpper = fContrastUpper;
		m_fSaturationProb = fSaturationProb;
		m_fSaturationLower = fSaturationLower;
		m_fSaturationUpper = fSaturationUpper;
		m_lRandomSeed = lRandomSeed;
	}
	catch (LONG lErrEx)
	{
		CleanUp();
		return lErrEx;
	}

	return 0;
}

template long imgopHandle<double>::Initialize(Memory<double>* pMem, int nNum, double fBrightnessProb, double fBrightnessDelta, double fContrastProb, double fContrastLower, double fContrastUpper, double fSaturationProb, double fSaturationLower, double fSaturationUpper, long lRandomSeed);
template long imgopHandle<float>::Initialize(Memory<float>* pMem, int nNum, float fBrightnessProb, float fBrightnessDelta, float fContrastProb, float fContrastLower, float fContrastUpper, float fSaturationProb, float fSaturationLower, float fSaturationUpper, long lRandomSeed);


template <class T>
long imgopHandle<T>::CleanUp()
{
	if (m_pOrdering != NULL)
	{
		cudaFree(m_pOrdering);
		m_pOrdering = NULL;
	}

	if (m_pBrightness != NULL)
	{
		cudaFree(m_pBrightness);
		m_pBrightness = NULL;
	}

	if (m_pContrast != NULL)
	{
		cudaFree(m_pContrast);
		m_pContrast = NULL;
	}

	if (m_pSaturation != NULL)
	{
		cudaFree(m_pSaturation);
		m_pSaturation = NULL;
	}

	return 0;
}

template long imgopHandle<double>::CleanUp();
template long imgopHandle<float>::CleanUp();


template <typename T>
__global__ void distort_image_kernel(const int nCount, const int nNum, const T* order, const T* brightness, const T* contrast, const T* saturation, T* x, T* y)
{
	for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < nCount; i += blockDim.x * gridDim.x)
	{
		const int n = i % nNum;

		// Brightness.
		y[i] = truncate(x[i] + brightness[n]);

		if (order[n] == 1)
		{
			// Contrast.
			y[i] = truncate(contrast[n] * (y[i] - T(128.0)) + T(128.0));

			// Gamma (Saturation)
			y[i] = (int)(T(255.0) * pow(y[i] / T(255.0), saturation[n]));
		}
		else
		{
			// Gamma (Saturation)
			y[i] = (int)(T(255.0) * pow(y[i] / T(255.0), saturation[n]));

			// Contrast.
			y[i] = truncate(contrast[n] * (y[i] - T(128.0)) + T(128.0));
		}
	}
}

template <class T>
long imgopHandle<T>::DistortImage(int nCount, int nNum, int nDim, long hX, long hY)
{
	LONG lErr = 0;
	MemoryItem* pX;
	MemoryItem* pY;

	if (nNum != m_nNum)
		return ERROR_PARAM_OUT_OF_RANGE;

	if (nCount / nNum != nDim)
		return ERROR_PARAM_OUT_OF_RANGE;

	if (lErr = m_pMem->GetMemoryCollection()->GetData(hX, &pX))
		return lErr;

	if (lErr = m_pMem->GetMemoryCollection()->GetData(hY, &pY))
		return lErr;

	T fRand;

	if (m_lRandomSeed > 0)
		srand((unsigned int)m_lRandomSeed);

	for (int i = 0; i < nNum; i++)
	{
		T fOrder = T(0.0);
		fRand = (T)(rand() / (T)RAND_MAX);
		if (fRand <= T(0.5))
			fOrder = T(1.0);

		if (lErr = cudaMemcpy(&(m_pOrdering[i]), &fOrder, sizeof(T), cudaMemcpyHostToDevice))
			throw lErr;

		T fBrightness = T(0.0);
		fRand = (T)(rand() / (T)RAND_MAX);
		if (fRand < m_fBrightnessProb)
			fBrightness = get_brightness(m_fBrightnessDelta);

		if (lErr = cudaMemcpy(&(m_pBrightness[i]), &fBrightness, sizeof(T), cudaMemcpyHostToDevice))
			throw lErr;

		T fContrast = T(1.0);
		fRand = (T)(rand() / (T)RAND_MAX);
		if (fRand < m_fContrastProb)
			fContrast = get_contrast(m_fContrastLower, m_fContrastUpper);

		if (lErr = cudaMemcpy(&(m_pContrast[i]), &fContrast, sizeof(T), cudaMemcpyHostToDevice))
			throw lErr;

		T fSaturation = T(1.0);
		fRand = (T)(rand() / (T)RAND_MAX);
		if (fRand < m_fSaturationProb)
			fSaturation = get_saturation(m_fSaturationLower, m_fSaturationUpper);

		if (lErr = cudaMemcpy(&(m_pSaturation[i]), &fSaturation, sizeof(T), cudaMemcpyHostToDevice))
			throw lErr;
	}

	T* x = (T*)pX->Data();
	T* y = (T*)pY->Data();

	distort_image_kernel<T> << <CAFFE_GET_BLOCKS(nCount), CAFFE_CUDA_NUM_THREADS >> > (nCount, nNum, m_pOrdering, m_pBrightness, m_pContrast, m_pSaturation, x, y);

	return cudaStreamSynchronize(0);
}

template long imgopHandle<double>::DistortImage(int nCount, int nNum, int nDim, long hX, long hY);
template long imgopHandle<float>::DistortImage(int nCount, int nNum, int nDim, long hX, long hY);

// end