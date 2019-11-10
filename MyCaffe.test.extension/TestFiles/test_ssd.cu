//=============================================================================
//	FILE:	test_ssd.cu
//
//	DESC:	This file implements the single-shot multi-box detection testing code.
//=============================================================================

#include "..\Cuda Files\util.h"
#include "test_ssd.h"

#include "..\Cuda Files\memory.h"
#include "..\Cuda Files\math.h"
#include "..\Cuda Files\ssd.h"


//=============================================================================
//	Test Enum.
//=============================================================================

enum TEST
{
	CREATE = 1,
	APPLYNMS = 2
};


//=============================================================================
//	Test Helper Classes
//=============================================================================

template <class T>
class TestData
{
	Memory<T> m_memory;
	Math<T> m_math;

public:
	SsdData<T> m_ssd;

	TestData() : m_memory(), m_math(), m_ssd(&m_memory, &m_math)
	{
	}

	long TestCreate(int nConfig)
	{
		LONG lErr;

		if (lErr = m_ssd.Initialize(0, 2, true, 2, 0, false, SSD_MINING_TYPE_NONE, SSD_MATCHING_TYPE_BIPARTITE, 0.3, true, SSD_CODE_TYPE_CORNER, true, false, true, true, SSD_CONF_LOSS_TYPE_SOFTMAX, SSD_LOC_LOSS_TYPE_L2, 0, 0, 10, false, 0.1, 10, 0.1))
			return lErr;

		return 0;
	}

	long TestApplyNMS()
	{
		LONG lErr;
		vector<BBOX> bboxes;
		vector<T> scores;
		T fThreshold = T(0);
		int nTopK = 3;
		vector<int> indices;

		if (lErr = m_ssd.applyNMS(bboxes, scores, fThreshold, nTopK, &indices))
			return lErr;

		return 0;
	}
};


//=============================================================================
//	Test Functions
//=============================================================================

template <class T>
long TestSsd<T>::cleanup()
{
	if (m_pObj != NULL)
	{
		delete ((TestData<T>*)m_pObj);
		m_pObj = NULL;
	}

	return 0;
}

template long TestSsd<double>::cleanup();
template long TestSsd<float>::cleanup();


template <class T>
long TestSsd<T>::test_create(int nConfig)
{
	LONG lErr;

	if ((m_pObj = new TestData<T>()) == NULL)
		return ERROR_MEMORY_OUT;

	return ((TestData<T>*)m_pObj)->TestCreate(nConfig);
}

template long TestSsd<double>::test_create(int nConfig);
template long TestSsd<float>::test_create(int nConfig);


//=============================================================================
//	Function Definitions
//=============================================================================

template <class T>
long TestSsd<T>::RunTest(LONG lInput, T* pfInput)
{
	TEST tst = (TEST)(int)pfInput[0];
	int nConfig = 0;

	if (lInput > 1)
		nConfig = (int)pfInput[1];

	try
	{
		switch (tst)
		{
			case CREATE:
				test_create(nConfig);
				break;

			case APPLYNMS:
				test_create(nConfig);
				((TestData<T>*)m_pObj)->TestApplyNMS();
				break;

			default:
				return ERROR_PARAM_OUT_OF_RANGE;
		}

		cleanup();
	}
	catch (...)
	{
		cleanup();
		return ERROR_SSD;
	}

	return 0;
}

template long TestSsd<double>::RunTest(LONG lInput, double* pfInput);
template long TestSsd<float>::RunTest(LONG lInput, float* pfInput);

// end