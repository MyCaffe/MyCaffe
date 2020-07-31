//=============================================================================
//	FILE:	imgop.h
//
//	DESC:	This file is used to manage the memory testing.
//=============================================================================
#ifndef __IMGOP_CU__
#define __IMGOP_CU__

#include "util.h"

//=============================================================================
//	Types
//=============================================================================

//=============================================================================
//	Classes
//=============================================================================

template <class T>
class Memory;


//-----------------------------------------------------------------------------
//	MemoryTest Handle Class
//
//	This class stores the Memory Test description information.
//-----------------------------------------------------------------------------
template <class T>
class imgopHandle
{
	Memory<T>* m_pMem;
	int m_nNum;
	T* m_pOrdering;
	T* m_pContrast;
	T* m_pBrightness;
	T* m_pSaturation;
	T m_fBrightnessProb;
	T m_fBrightnessDelta;
	T m_fContrastProb;
	T m_fContrastLower;
	T m_fContrastUpper;
	T m_fSaturationProb;
	T m_fSaturationLower;
	T m_fSaturationUpper;
	long m_lRandomSeed;

	T get_random(T min, T max)
	{
		T fRange = max - min;
		T fRand = T(rand() / T(RAND_MAX));
		T fVal = (min + (fRange * fRand));

		return T(1.0) + T(fVal);
	}

	T get_brightness(T fBrightnessDelta)
	{
		return get_random(-fBrightnessDelta, fBrightnessDelta);
	}

	T get_contrast(T fContrastLower, T fContrastUpper)
	{
		T fContrast = get_random(fContrastLower, fContrastUpper);
		fContrast -= T(1.0);
		fContrast *= T(255.0);

		T fFactor = (T(259.0) * (fContrast + T(255.0))) / (T(255.0) * (T(259.0) - fContrast));

		return fFactor;
	}

	T get_saturation(T fSaturationLower, T fSaturationUpper)
	{
		T fGamma = get_random(fSaturationLower, fSaturationUpper);
		return T(1.0) / fGamma;
	}


public:	
	imgopHandle()
	{
		m_nNum = 0;
		m_pOrdering = NULL;
		m_pContrast = NULL;
		m_pBrightness = NULL;
		m_pSaturation = NULL;
	}

	long Initialize(Memory<T>* pMem, int nNum, T fBrightnessProb, T fBrightnessDelta, T fContrastProb, T fContrastLower, T fContrastUpper, T fSaturationProb, T fSaturationLower, T fSaturationUpper, long lRandomSeed);
	long DistortImage(int nCount, int nNum, int nDim, long hX, long hY);
	long CleanUp();
};


//=============================================================================
//	Inline Methods
//=============================================================================

#endif