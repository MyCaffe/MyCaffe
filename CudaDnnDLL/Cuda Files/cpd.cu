//=============================================================================
//	FILE:	cpd.cu
//
//	DESC:	This file implements the base class used to manage the the cpd
//			(change point detection) primitives.
// 
// @see [A Contrastive Approach to Online Change Point Detection](https://arxiv.org/abs/2206.10143) by 
// Artur Goldman, Nikita Puchkin, Valeriia Shcherbakova, and Uliana Vinogradova, 2022, arXiv
// @see [Numerical experiments on the WISDM data set described in the paper 
// "A Contrastive Approach to Online Change Point Detection"](https://github.com/npuchkin/contrastive_change_point_detection/blob/main/WISDM_experiments.ipynb) 
// by npuchkin, GitHub 2023
//=============================================================================

#include "util.h"
#include "memory.h"
#include "cpd.h"

//=============================================================================
//	Class Methods
//=============================================================================

template <class T>
long cpdHandle<T>::CleanUp()
{
	if (m_hT != 0)
	{
		m_pMem->FreeMemory(m_hT);
		m_hT = 0;
	}

	if (m_hD != 0)
	{
		m_pMem->FreeMemory(m_hD);
		m_hD = 0;
	}

	if (m_hWork != 0)
	{
		m_pMem->FreeMemory(m_hWork);
		m_hWork = 0;
	}

	m_nN = 0;
	m_nB = 0;

	return 0;
}

template long cpdHandle<double>::CleanUp();
template long cpdHandle<float>::CleanUp();

template <class T>
long cpdHandle<T>::Set(int nN, int nB)
{
	LONG lErr;

	CleanUp();

	try
	{
		int nDeviceID;

		if (lErr = cudaGetDevice(&nDeviceID))
			return lErr;

		m_nN = nN;
		m_nB = nB;

		if (m_hT == 0)
		{
			if (lErr = m_pMem->AllocMemory(nDeviceID, false, nN * nN, NULL, -1, &m_hT))
				throw lErr;
		}
		if (m_hD == 0)
		{
			if (lErr = m_pMem->AllocMemory(nDeviceID, false, nN, NULL, -1, &m_hD))
				throw lErr;
		}
		if (m_hWork == 0)
		{
			if (lErr = m_pMem->AllocMemory(nDeviceID, false, nN * nN, NULL, -1, &m_hWork))
				throw lErr;
		}

		if (lErr = m_pMath->set(nN * nN, m_hT, 0, -1))
			throw lErr;
	}
	catch (LONG lErrEx)
	{
		CleanUp();
		return lErrEx;
	}

	return cudaStreamSynchronize(0);
}

template long cpdHandle<double>::Set(int nN, int nB);
template long cpdHandle<float>::Set(int nN, int nB);


template <class T>
long cpdHandle<T>::ComputeTvalueAt(int nT, int nTau, int nZ, long hZ, T* pfTVal)
{
	LONG lErr;

	if (m_hT == 0 || m_hD == 0 || m_hWork == 0)
		return ERROR_CPD_NOT_INITIALIZED;

	if (lErr = m_pMath->set(m_nN, m_hD, 0, -1))
		return lErr;

	if (lErr = m_pMath->set(m_nN * m_nN, m_hWork, 0, -1))
		return lErr;

	if (lErr = m_pMath->clip_fwd(nZ, hZ, m_hD, -m_nB, m_nB))
		return lErr;

	// Compute D[:tau] = 2 / (1 + exp(-Z[:tau]))
	if (lErr = m_pMath->scal(nTau, -1, m_hD))
		return lErr;

	if (lErr = m_pMath->exp(nTau, m_hD, m_hD))
		return lErr;

	if (lErr = m_pMath->add_scalar(nTau, 1, m_hD))
		return lErr;

	if (lErr = m_pMath->invert(nTau, m_hD, m_hD))
		return lErr;

	if (lErr = m_pMath->scal(nTau, 2, m_hD))
		return lErr;

	// Compute D[tau:] = 2 / (1 + exp(Z[tau:]))
	if (lErr = m_pMath->exp(nT - nTau, m_hD, m_hD, nTau, nTau))
		return lErr;

	if (lErr = m_pMath->add_scalar(nT - nTau, 1, m_hD, nTau))
		return lErr;

	if (lErr = m_pMath->invert(nT - nTau, m_hD, m_hD, nTau, nTau))
		return lErr;

	if (lErr = m_pMath->scal(nT - nTau, 2, m_hD, nTau))
		return lErr;

	// Compute D = np.log(D)
	if (lErr = m_pMath->log(nT, m_hD, m_hD))
		return lErr;

	// Compute statistics for a specific t and the change point candidate tau.
	if (lErr = m_pMath->channel_mean(nTau, 1, 1, nTau, m_hD, m_hWork, 0))
		return lErr;

	T fMean1;
	if (lErr = m_pMath->get(1, m_hWork, 0, &fMean1))
		return lErr;

	if (lErr = m_pMath->channel_mean(nT - nTau, 1, 1, nT - nTau, m_hD, m_hWork, nTau))
		return lErr;

	T fMean2;
	if (lErr = m_pMath->get(1, m_hWork, 0, &fMean2))
		return lErr;

	T fMean = fMean1 + fMean2;
	T fTauVal = (T)nTau * (T)(nT - nTau) / (T)nT * fMean;
	int nIdx = nTau * m_nN + nT;

	*pfTVal = fTauVal;

	if (lErr = m_pMath->set(1, m_hT, fTauVal, nIdx))
		return lErr;

	return 0;
}

template long cpdHandle<double>::ComputeTvalueAt(int nT, int nTau, int nZ, long hZ, double* pfTVal);
template long cpdHandle<float>::ComputeTvalueAt(int nT, int nTau, int nZ, long hZ, float* pfTVal);


template <class T>
long cpdHandle<T>::ComputeSvalues(int nS, long hS)
{
	LONG lErr;

	if (m_hT == 0 || m_hD == 0 || m_hWork == 0)
		return ERROR_CPD_NOT_INITIALIZED;

	if (nS < m_nN)
		return ERROR_PARAM_OUT_OF_RANGE;

	if (lErr = m_pMath->channel_max(m_nN * m_nN, 1, m_nN, m_nN, m_hT, hS, false, true))
		return lErr;

	return 0;
}

template long cpdHandle<double>::ComputeSvalues(int nS, long hS);
template long cpdHandle<float>::ComputeSvalues(int nS, long hS);

// end