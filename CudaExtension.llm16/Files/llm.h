//=============================================================================
//	FILE:	LLM.h
//
//	DESC:	This file the LLM inferencing processing
//=============================================================================
#ifndef __LLM_CU__
#define __LLM_CU__

//=============================================================================
//	Types
//=============================================================================

const long ERROR_LLM_RENDERED_PROMPT_TO_LONG = 10000;

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
	void* m_pObj;

	long cleanup();

public:
	
	LLM(T fTemperature, T fTopp, long lSeed);
	
	~LLM()
	{
		cleanup();
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