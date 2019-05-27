//=============================================================================
//	FILE:	pca.cu
//
//	DESC:	This file implements the base class used to manage the underlying
//			GPU device.
//
//	NOTES:  For more information on the Iterative PCA Algorithm, see:
//			M. Andrecut, "Parallel GPU Implementation of Iterative PCA Algorithms", 2008
//=============================================================================

#include "util.h"
#include "memory.h"
#include "pca.h"

//=============================================================================
//	Class Methods
//=============================================================================

template <class T>
long pcaHandle<T>::Initialize(Memory<T>* pMem, Math<T>* pMath)
{
	LONG lErr;
	int nDeviceID;

	if (lErr = cudaGetDevice(&nDeviceID))
		return lErr;

	m_pMem = pMem;
	m_pMath = pMath;
	m_nCurrentIteration = 0;
	m_nCurrentK = 0;

	try
	{
		if (m_hResiduals == 0)
		{
			if (lErr = m_pMem->AllocMemory(nDeviceID, false, m_nN * m_nD, NULL, -1, &m_hResiduals))
				throw lErr;

			m_bOwnResiduals = true;
		}
		else
		{
			m_bOwnResiduals = false;
		}

		if (m_hEigenvalues == 0)
		{
			if (lErr = m_pMem->AllocHostBuffer(m_nK * 1, &m_hEigenvalues))
				throw lErr;

			m_bOwnEigenvalues = true;
		}
		else
		{
			m_bOwnEigenvalues = false;
		}

		m_pfEigenvalues = m_pMem->GetHostBuffer(m_hEigenvalues)->Data();

		if (lErr = m_pMem->AllocMemory(nDeviceID, false, m_nN * 1, NULL, -1, &m_hMeanCenter))
			throw lErr;


		//------------------------------------------------
		//	Get the device memory pointers.
		//------------------------------------------------

		MemoryItem* pResiduals;
		MemoryItem* pScores;
		MemoryItem* pLoads;
		MemoryItem* pSums;

		if (lErr = m_pMem->GetMemory(m_hResiduals, &pResiduals))
			return lErr;

		if (lErr = m_pMem->GetMemory(m_hScores, &pScores))
			return lErr;

		if (lErr = m_pMem->GetMemory(m_hLoads, &pLoads))
			return lErr;

		if (lErr = m_pMem->GetMemory(m_hMeanCenter, &pSums))
			return lErr;

		m_pfR = (T*)pResiduals->Data();
		m_pfT = (T*)pScores->Data();
		m_pfP = (T*)pLoads->Data();
		m_pfU = (T*)pSums->Data();

		m_fA = 0;
	}
	catch (LONG lErrEx)
	{
		CleanUp();
		return lErrEx;
	}

	return 0;
}

template long pcaHandle<double>::Initialize(Memory<double>* pMem, Math<double>* pMath);
template long pcaHandle<float>::Initialize(Memory<float>* pMem, Math<float>* pMath);


template <class T>
long pcaHandle<T>::CleanUp()
{
	if (m_bOwnResiduals)
	{
		if (m_hResiduals > 0)
		{
			m_pMem->FreeMemory(m_hResiduals);
			m_hResiduals = 0;
		}
	}

	if (m_bOwnEigenvalues)
	{
		if (m_hEigenvalues > 0)
		{
			m_pMem->FreeHostBuffer(m_hEigenvalues);
			m_hEigenvalues = 0;
		}
	}

	if (m_hMeanCenter > 0)
	{
		m_pMem->FreeMemory(m_hMeanCenter);
		m_hMeanCenter = 0;
	}

	m_pfR = NULL;
	m_pfT = NULL;
	m_pfP = NULL;
	m_pfU = NULL;

	return 0;
}

template long pcaHandle<double>::CleanUp();
template long pcaHandle<float>::CleanUp();


template <>
long pcaHandle<double>::Run(int nSteps, bool *pbDone, int* pnCurrentIteration, int* pnCurrentK)
{
	LONG lErr = 0;
	double dfMaxErr = 1.0e-7;
	cublasHandle_t cublas = m_pMath->GetCublasHandle();


	//-------------------------------------------------
	//	Run on the first iteration for the current
	//  component.
	//-------------------------------------------------

	int k = m_nCurrentK;

	if (m_nCurrentIteration == 0)
	{
		if (m_nCurrentK == 0)
		{
			//-----------------------------------------
			//	Load the data.
			//-----------------------------------------

			if (lErr = m_pMath->mtx_transpose(m_nD, m_nN, m_hData, m_hResiduals))
				throw lErr;


			//-----------------------------------------
			//	Mean center the data.
			//-----------------------------------------

			if (lErr = m_pMath->mtx_meancenter_by_column(m_nN, m_nD, m_hResiduals, m_hMeanCenter, m_hResiduals, false))
				return lErr;
		}

		m_fA = 0;

		if (lErr = cublasDcopy(cublas, m_nN, m_pfR + (k * m_nN), 1, m_pfT + (k * m_nN), 1))
			return lErr;
	}


	//-------------------------------------------------
	//	Run step iterations on the current component.
	//-------------------------------------------------

	double dfMOne = -1.0;
	double dfOne = 1.0;
	double dfZero = 0.0;
	double dfAlpha;
	double dfNrm;

	for (int i=0; i<nSteps; i++)
	{
		int j = m_nCurrentIteration + i;
		if (j >= m_nNaxIterations)
			break;	


		if (lErr = cublasDgemv(cublas, CUBLAS_OP_T, m_nN, m_nD, &dfOne, m_pfR, m_nN, m_pfT + (k * m_nN), 1, &dfZero, m_pfP + (k * m_nD), 1))
			return lErr;

		if (k > 0)
		{
			if (lErr = cublasDgemv(cublas, CUBLAS_OP_T, m_nD, k, &dfOne, m_pfP, m_nD, m_pfP + (k * m_nD), 1, &dfZero, m_pfU, 1))
				return lErr;

			if (lErr = cublasDgemv(cublas, CUBLAS_OP_N, m_nD, k, &dfMOne, m_pfP, m_nD, m_pfU, 1, &dfOne, m_pfP + (k * m_nD), 1))
				return lErr;
		}

		if (lErr = cublasDnrm2(cublas, m_nD, m_pfP + (k * m_nD), 1, &dfNrm))
			return lErr;

		if (dfNrm != 0)
		{
			dfAlpha = 1.0 / dfNrm;

			if (lErr = cublasDscal(cublas, m_nD, &dfAlpha, m_pfP + (k * m_nD), 1))
				return lErr;
		}

		if (lErr = cublasDgemv(cublas, CUBLAS_OP_N, m_nN, m_nD, &dfOne, m_pfR, m_nN, m_pfP + (k * m_nD), 1, &dfZero, m_pfT + (k * m_nN), 1))
			return lErr;

		if (k > 0)
		{
			if (lErr = cublasDgemv(cublas, CUBLAS_OP_T, m_nN, k, &dfOne, m_pfT, m_nN, m_pfT + (k * m_nN), 1, &dfZero, m_pfU, 1))
				return lErr;

			if (lErr = cublasDgemv(cublas, CUBLAS_OP_N, m_nN, k, &dfMOne, m_pfT, m_nN, m_pfU, 1, &dfOne, m_pfT + (k * m_nN), 1))
				return lErr;
		}

		if (lErr = cublasDnrm2(cublas, m_nN, m_pfT + (k * m_nN), 1, &dfNrm))
			return lErr;

		m_pfEigenvalues[k] = dfNrm;

		if (dfNrm != 0)
		{
			dfAlpha = 1.0 / dfNrm;

			if (lErr = cublasDscal(cublas, m_nN, &dfAlpha, m_pfT + (k * m_nN), 1))
				return lErr;
		}

		if (fabs(m_fA - m_pfEigenvalues[k]) < dfMaxErr * m_pfEigenvalues[k])
		{
			m_nCurrentIteration = m_nNaxIterations;
			break;
		}

		m_fA = m_pfEigenvalues[k];
	}


	//-------------------------------------------------
	//	Check to see if all iterations have been run
	//	on all 'K' components and if so, we're done.
	//-------------------------------------------------

	*pbDone = FALSE;
	m_nCurrentIteration += nSteps;

	if (m_nCurrentIteration >= m_nNaxIterations)
	{
		dfAlpha = -m_pfEigenvalues[k];

		if (lErr = cublasDger(cublas, m_nN, m_nD, &dfAlpha, m_pfT + (k * m_nN), 1, m_pfP + (k * m_nD), 1, m_pfR, m_nN))
			return lErr;

		m_nCurrentIteration = 0;
		m_nCurrentK++;

		if (m_nCurrentK >= m_nK)
		{
			for (int i=0; i<m_nK; i++)
			{
				dfAlpha = m_pfEigenvalues[k];

				if (lErr = cublasDscal(cublas, m_nN, &dfAlpha, m_pfT + (k * m_nN), 1))
					return lErr;
			}

			*pbDone = TRUE;
		}
	}
	
	if (pnCurrentIteration != NULL)
		*pnCurrentIteration = m_nCurrentIteration;

	if (pnCurrentK != NULL)
		*pnCurrentK = m_nCurrentK;

	return 0;
}

template <>
long pcaHandle<float>::Run(int nSteps, bool *pbDone, int* pnCurrentIteration, int* pnCurrentK)
{
	LONG lErr = 0;
	float fMaxErr = (float)1.0e-7;
	cublasHandle_t cublas = m_pMath->GetCublasHandle();


	//-------------------------------------------------
	//	Run on the first iteration for the current
	//  component.
	//-------------------------------------------------

	int k = m_nCurrentK;

	if (m_nCurrentIteration == 0)
	{
		if (m_nCurrentK == 0)
		{
			//-----------------------------------------
			//	Load the data.
			//-----------------------------------------

			if (lErr = m_pMath->mtx_transpose(m_nD, m_nN, m_hData, m_hResiduals))
				throw lErr;


			//-----------------------------------------
			//	Mean center the data.
			//-----------------------------------------

			if (lErr = m_pMath->mtx_meancenter_by_column(m_nN, m_nD, m_hResiduals, m_hMeanCenter, m_hResiduals, false))
				return lErr;
		}

		m_fA = 0;

		if (lErr = cublasScopy(cublas, m_nN, m_pfR + (k * m_nN), 1, m_pfT + (k * m_nN), 1))
			return lErr;
	}


	//-------------------------------------------------
	//	Run step iterations on the current component.
	//-------------------------------------------------

	float fMOne = -1.0;
	float fOne = 1.0;
	float fZero = 0.0;
	float fAlpha;
	float fNrm;

	for (int i=0; i<nSteps; i++)
	{
		int j = m_nCurrentIteration + i;
		if (j >= m_nNaxIterations)
			break;	


		if (lErr = cublasSgemv(cublas, CUBLAS_OP_T, m_nN, m_nD, &fOne, m_pfR, m_nN, m_pfT + (k * m_nN), 1, &fZero, m_pfP + (k * m_nD), 1))
			return lErr;

		if (k > 0)
		{
			if (lErr = cublasSgemv(cublas, CUBLAS_OP_T, m_nD, k, &fOne, m_pfP, m_nD, m_pfP + (k * m_nD), 1, &fZero, m_pfU, 1))
				return lErr;

			if (lErr = cublasSgemv(cublas, CUBLAS_OP_N, m_nD, k, &fMOne, m_pfP, m_nD, m_pfU, 1, &fOne, m_pfP + (k * m_nD), 1))
				return lErr;
		}

		if (lErr = cublasSnrm2(cublas, m_nD, m_pfP + (k * m_nD), 1, &fNrm))
			return lErr;

		fAlpha = 1.0f / fNrm;

		if (lErr = cublasSscal(cublas, m_nD, &fAlpha, m_pfP + (k * m_nD), 1))
			return lErr;

		if (lErr = cublasSgemv(cublas, CUBLAS_OP_N, m_nN, m_nD, &fOne, m_pfR, m_nN, m_pfP + (k * m_nD), 1, &fZero, m_pfT + (k * m_nN), 1))
			return lErr;

		if (k > 0)
		{
			if (lErr = cublasSgemv(cublas, CUBLAS_OP_T, m_nN, k, &fOne, m_pfT, m_nN, m_pfT + (k * m_nN), 1, &fZero, m_pfU, 1))
				return lErr;

			if (lErr = cublasSgemv(cublas, CUBLAS_OP_N, m_nN, k, &fMOne, m_pfT, m_nN, m_pfU, 1, &fOne, m_pfT + (k * m_nN), 1))
				return lErr;
		}

		if (lErr = cublasSnrm2(cublas, m_nN, m_pfT + (k * m_nN), 1, &fNrm))
			return lErr;

		m_pfEigenvalues[k] = fNrm;

		fAlpha = 1.0f / fNrm;

		if (lErr = cublasSscal(cublas, m_nN, &fAlpha, m_pfT + (k * m_nN), 1))
			return lErr;

		if (fabs(m_fA - m_pfEigenvalues[k]) < fMaxErr * m_pfEigenvalues[k])
		{
			m_nCurrentIteration = m_nNaxIterations;
			break;
		}

		m_fA = m_pfEigenvalues[k];
	}


	//-------------------------------------------------
	//	Check to see if all iterations have been run
	//	on all 'K' components and if so, we're done.
	//-------------------------------------------------

	*pbDone = FALSE;
	m_nCurrentIteration += nSteps;

	if (m_nCurrentIteration >= m_nNaxIterations)
	{
		fAlpha = -m_pfEigenvalues[k];

		if (lErr = cublasSger(cublas, m_nN, m_nD, &fAlpha, m_pfT + (k * m_nN), 1, m_pfP + (k * m_nD), 1, m_pfR, m_nN))
			return lErr;

		m_nCurrentIteration = 0;
		m_nCurrentK++;

		if (m_nCurrentK >= m_nK)
		{
			for (int i=0; i<m_nK; i++)
			{
				fAlpha = m_pfEigenvalues[k];

				if (lErr = cublasSscal(cublas, m_nN, &fAlpha, m_pfT + (k * m_nN), 1))
					return lErr;
			}

			*pbDone = TRUE;
		}
	}
	
	if (pnCurrentIteration != NULL)
		*pnCurrentIteration = m_nCurrentIteration;

	if (pnCurrentK != NULL)
		*pnCurrentK = m_nCurrentK;

	return 0;
}

// end