//=============================================================================
//	FILE:	LLM.cpp
//
//	DESC:	This file implements the LLM inferencing.
//=============================================================================

#include "..\Cuda Files\util.h"
#include "LLM.h"


//=============================================================================
//	LLM Helper Classes
//=============================================================================

template <class T>
class LLMData
{
	int m_nResponseCount;

public:
	LLMData()
	{
		m_nResponseCount = 0;
	}

	~LLMData()
	{
	}

	long Load(LPTSTR szInput)
	{
		LONG lErr;

		return 0;
	}

	long QueryStatus(float* pPct, LONG* plStatus)
	{
		LONG lErr;

		*pPct = 1;
		*plStatus = 1;

		return 0;
	}

	long Generate(LPTSTR szInput)
	{
		LONG lErr;

		return 0;
	}

	long QueryResults(LPTSTR szOutput, LONG lMax, LONG* lEnd)
	{
		LONG lErr;

		if (m_nResponseCount == 0)
		{
			_tcsncpy(szOutput, _T("This is a test"), lMax);
		}
		else
		{
			szOutput[0] = (TCHAR)0;
		}

		*lEnd = 1;
		m_nResponseCount++;

		return 0;
	}
};


//=============================================================================
//	LLM Functions
//=============================================================================

template <class T>
long LLM<T>::cleanup()
{
	if (m_pObj != NULL)
	{
		delete ((LLMData<T>*)m_pObj);
		m_pObj = NULL;
	}

	return 0;
}

template long LLM<double>::cleanup();
template long LLM<float>::cleanup();


//=============================================================================
//	Function Definitions
//=============================================================================

template <class T>
long LLM<T>::Load(LPTSTR szInput)
{
	if ((m_pObj = new LLMData<T>()) == NULL)
		return ERROR_MEMORY_OUT;

	return ((LLMData<T>*)m_pObj)->Load(szInput);
}

template long LLM<double>::Load(LPTSTR szInput);
template long LLM<float>::Load(LPTSTR szInput);


template <class T>
long LLM<T>::QueryStatus(float* pPct, LONG* plStatus)
{
	if (m_pObj == NULL)
		return ERROR_INVALID_STATE;

	return ((LLMData<T>*)m_pObj)->QueryStatus(pPct, plStatus);
}

template long LLM<double>::QueryStatus(float* pPct, LONG* plStatus);
template long LLM<float>::QueryStatus(float* pPct, LONG* plStatus);


template <class T>
long LLM<T>::Generate(LPTSTR szInput)
{
	if (m_pObj == NULL)
		return ERROR_INVALID_STATE;

	return ((LLMData<T>*)m_pObj)->Generate(szInput);
}

template long LLM<double>::Generate(LPTSTR szInput);
template long LLM<float>::Generate(LPTSTR szInput);


template <class T>
long LLM<T>::QueryResponse(LPTSTR szOutput, LONG lMax, LONG* plEnd)
{
	if (m_pObj == NULL)
		return ERROR_INVALID_STATE;

	return ((LLMData<T>*)m_pObj)->QueryResults(szOutput, lMax, plEnd);
}

template long LLM<double>::QueryResponse(LPTSTR szOutput, LONG lMax, LONG* plEnd);
template long LLM<float>::QueryResponse(LPTSTR szOutput, LONG lMax, LONG* plEnd);

// end