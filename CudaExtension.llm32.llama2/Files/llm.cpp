//=============================================================================
//	FILE:	LLM.cpp
//
//	DESC:	This file implements the LLM inferencing.
//=============================================================================

#include "..\Cuda Files\util.h"
#include "LLM.h"

#include <string>
#include <mutex>
#include <chrono>

#include "llama2_gpu.h"
#include "llama2_tokenizer.h"
#include "llama2_sampler.h"


//=============================================================================
//	LLM Helper Classes
//=============================================================================

template <class T>
class LLMData
{
	T m_fTemperature;
	T m_fTopp;
	long m_lSeed;
	Transformer* m_transformer;
	Tokenizer m_tokenizer;
	Sampler m_sampler;

	int m_nSteps;
	int m_nMaxInputPrompt;
	int m_nMaxRenderedPrompt;
	char* m_pszRenderedPrompt;
	int m_nMaxPromptTokens;
	int* m_pnPromptTokens;
	std::string m_strResponse;
	std::atomic<bool>* m_pbCancel;
	std::atomic<bool> m_bLoaded;
	std::atomic<bool> m_bEnd;
	std::mutex m_mtxResponse;

public:
	LLMData(T fTemperature, T fTopp, long lSeed, std::atomic<bool>* pbCancel) : m_tokenizer(), m_sampler(), m_strResponse(), m_mtxResponse(), m_bLoaded(false), m_bEnd(false)
	{
		m_fTemperature = fTemperature;
		m_fTopp = fTopp;
		m_lSeed = lSeed;
		m_transformer = NULL;

		m_nSteps = 2048;
		m_nMaxRenderedPrompt = 4096;
		m_nMaxPromptTokens = 4096;
		m_pnPromptTokens = (int*)malloc(m_nMaxPromptTokens * sizeof(int));
		m_pbCancel = pbCancel;
	}

	~LLMData()
	{
		if (m_transformer != NULL)
		{
			delete m_transformer;
			m_transformer = NULL;
		}

		if (m_pnPromptTokens != NULL)
		{
			free(m_pnPromptTokens);
			m_pnPromptTokens = NULL;
		}

		free_sampler(&m_sampler);
		free_tokenizer(&m_tokenizer);
	}

	void split(LPTSTR szInput, std::string& strModelPath, std::string& strTokenizerPath)
	{
		USES_CONVERSION;

		std::string strInput;
		size_t nPos;

		strInput = T2A(szInput);
		nPos = strInput.find(";");

		if (nPos == std::string::npos)
		{
			strModelPath = strInput;
			strTokenizerPath = "";
		}
		else
		{
			strModelPath = strInput.substr(0, nPos);
			strTokenizerPath = strInput.substr(nPos + 1);
		}
	}

	long Load(LPTSTR szInput);
	long QueryStatus(float* pPct, LONG* plStatus);
	long Generate(LPTSTR szInput);
	long QueryResults(LPTSTR szOutput, LONG lMax, LONG* lEnd);
};

template <class T>
long LLMData<T>::Load(LPTSTR szInput)
{
	LONG lErr = 0;

	if (m_pnPromptTokens == NULL)
		return ERROR_MEMORY_OUT;

	try
	{
		if (m_transformer != NULL)
		{
			delete m_transformer;
			m_transformer = NULL;
		}

		std::string strModelPath;
		std::string strTokenizerPath;
		split(szInput, strModelPath, strTokenizerPath);

		FILE* file = fopen(strModelPath.c_str(), "r");
		if (file == NULL)
			return ERROR_LLM_LOAD_MODEL_MISSING_MODEL_FILE;
		fclose(file);

		file = fopen(strTokenizerPath.c_str(), "r");
		if (file == NULL)
			return ERROR_LLM_LOAD_MODEL_MISSING_TOKENIZER_FILE;
		fclose(file);

		m_transformer = new TransformerGpu();
		m_transformer->build(strModelPath.c_str());

		build_tokenizer(&m_tokenizer, (char*)strTokenizerPath.c_str(), m_transformer->m_config.vocab_size);
		build_sampler(&m_sampler, m_transformer->m_config.vocab_size, (float)m_fTemperature, (float)m_fTopp, m_lSeed);

		m_bLoaded.store(true);
	}
	catch (std::exception& e)
	{
		lErr = ERROR_LLM_LOAD_MODEL;
	}
	catch (...)
	{
		lErr = ERROR_LLM_LOAD_MODEL;
	}

	return lErr;
}

template long LLMData<float>::Load(LPTSTR szInput);
template long LLMData<double>::Load(LPTSTR szInput);


template <class T>
long LLMData<T>::QueryStatus(float* pPct, LONG* plStatus)
{
	LONG lErr;

	if (!m_bLoaded.load())
	{
		*pPct = 0;
		*plStatus = 0;
	}
	else
	{
		*pPct = 1;
		*plStatus = 1;
	}

	return 0;
}

template long LLMData<float>::QueryStatus(float* pPct, LONG* plStatus);
template long LLMData<double>::QueryStatus(float* pPct, LONG* plStatus);


template <class T>
long LLMData<T>::Generate(LPTSTR szRenderedPrompt)
{
	USES_CONVERSION;
	LONG lErr = 0;
	int num_prompt_tokens = 0;
	int user_idx;

	// start the main loop
	int user_turn = 1;	// user starts.
	int next = 0;		// will store the next token in the sequence.
	int token;			// stores current token to feed into the transformer.
	int prev_token = 0;	// previous token
	int pos = 0;		// position in sequence

	m_bEnd.store(false);

	try
	{
		if (_tcslen(szRenderedPrompt) > m_nMaxRenderedPrompt)
			return ERROR_LLM_RENDERED_PROMPT_TO_LONG;

		std::string strRenderedPrompt = T2A(szRenderedPrompt);
		m_pbCancel->store(false);

		while (pos < m_nSteps)
		{
			// when it is the user's turn, we need to get the next token from the user.
			if (user_turn)
			{
				encode(&m_tokenizer, (char*)strRenderedPrompt.c_str(), 1, 0, m_pnPromptTokens, &num_prompt_tokens);
				user_idx = 0; // reset the user index.
				user_turn = 0;
			}

			// determine the token to pass into the transformer next
			if (user_idx < num_prompt_tokens)
			{
				// if we are still processing the input prompt, force the next prompt token
				token = m_pnPromptTokens[user_idx++];
			}
			else
			{
				// otherwise use the next token sampled form the previous turn.
				token = next;
			}

			// forward the transformer to get logits for the next token
			float* logits = m_transformer->forward(token, pos);
			next = sample(&m_sampler, logits);
			pos++;

			if (user_idx >= num_prompt_tokens && next != 2)
			{
				// The assistant is responding, so we need to add the token to the response.
				char* piece = decode(&m_tokenizer, token, next);

				// lock the response mutex to prevent multiple threads from writing to the response at the same time.
				std::lock_guard<std::mutex> lock(m_mtxResponse);
				m_strResponse += piece;
			}

			// EOS (=2) token is the end of the sequence.
			if (token == 2 || next == 2 || m_pbCancel->load())
				break;
		}

		if (m_pbCancel->load())
			m_strResponse.clear();
	}
	catch (std::exception& e)
	{
		lErr = ERROR_LLM_GENERATE;
	}
	catch (...)
	{
		lErr = ERROR_LLM_GENERATE;
	}

	m_bEnd.store(true);

	return lErr;
}

template long LLMData<float>::Generate(LPTSTR szInput);
template long LLMData<double>::Generate(LPTSTR szInput);


template <class T>
long LLMData<T>::QueryResults(LPTSTR szOutput, LONG lMax, LONG* lEnd)
{
	USES_CONVERSION;
	LONG lErr;

	// lock the response mutex to prevent multiple threads from writing to the response at the same time.
	std::lock_guard<std::mutex> lock(m_mtxResponse);

	if (m_strResponse.length() < lMax)
	{
		_tcscpy(szOutput, A2T(m_strResponse.c_str()));
		m_strResponse.clear();
	}
	else
	{
		_tcscpy(szOutput, A2T(m_strResponse.substr(0, lMax).c_str()));
		m_strResponse = m_strResponse.substr(lMax);
	}

	*lEnd = (m_bEnd.load() == true) ? 1 : 0;
	m_bEnd.store(false);

	return 0;
}

template long LLMData<float>::QueryResults(LPTSTR szOutput, LONG lMax, LONG* lEnd);
template long LLMData<double>::QueryResults(LPTSTR szOutput, LONG lMax, LONG* lEnd);


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
LLM<T>::LLM(T fTemperature, T fTopp, long lSeed) : m_bCancel(false), m_bBusy(false)
{
	m_pObj = new LLMData<T>(fTemperature, fTopp, lSeed, &m_bCancel);
}

template LLM<float>::LLM(float fTemperature, float fTopp, long lSeed);
template LLM<double>::LLM(double fTemperature, double fTopp, long lSeed);


template <class T>
long LLM<T>::Load(LPTSTR szInput)
{
	if (m_pObj == NULL)
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