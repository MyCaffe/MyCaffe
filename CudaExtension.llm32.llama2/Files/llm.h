//=============================================================================
//	FILE:	LLM.h
//
//	DESC:	This file the LLM inferencing processing
//=============================================================================
#ifndef __LLM_CU__
#define __LLM_CU__

#include <atomic>

//=============================================================================
//	Types
//=============================================================================

const long ERROR_LLM_RENDERED_PROMPT_TO_LONG = 10000;
const long ERROR_LLM_LOAD_MODEL = 10001;
const long ERROR_LLM_LOAD_MODEL_MISSING_MODEL_FILE = 10002;
const long ERROR_LLM_LOAD_MODEL_MISSING_TOKENIZER_FILE = 10003;
const long ERROR_LLM_GENERATE = 10004;


//=============================================================================
//	Classes
//=============================================================================

//-----------------------------------------------------------------------------
//	LLM Class
//
//	This class us used to manage the LLM inferencing processing.
//-----------------------------------------------------------------------------
template <class T>
class LLM
{
	std::atomic<bool> m_bCancel;
	std::atomic<bool> m_bBusy;
	void* m_pObj;

	long cleanup();

public:
	
	LLM(T fTemperature, T fTopp, long lSeed);
	
	~LLM()
	{
		cleanup();
	}

	void Reset()
	{
		m_bCancel.store(false);
	}

	void Cancel()
	{
		m_bCancel.store(true);
	}

	void SetBusy(bool bBusy)
	{
		m_bBusy.store(bBusy);
	}

	bool IsBusy()
	{
		return m_bBusy.load();
	}

	long Load(LPTSTR szInput);
	long QueryStatus(float* pPct, LONG* plStatus);
	long Generate(LPTSTR szInput);
	long QueryResponse(LPTSTR szOutput, LONG lMax, LONG* plEnd);
};


//=============================================================================
//	Inline Methods
//=============================================================================


#endif