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

	BBOX_SIZE = 2,
	BBOX_BOUNDS = 3,
	BBOX_DIVBOUNDS = 4,
	BBOX_CLIP = 5,

	BBOX_DECODE = 6,
	BBOX_ENCODE = 7,
	BBOX_INTERSECT = 8,
	BBOX_JACCARDOVERLAP = 9,
	BBOX_MATCH = 10,

	FINDMATCHES = 11,
	COUNTMATCHES = 12,
	SOFTMAX = 13,
	COMPUTE_CONF_LOSS = 14,
	COMPUTE_LOC_LOSS = 15,
	GET_TOPK_SCORES = 16,
	APPLYNMS = 17,
	MINE_HARD_EXAMPLES = 18
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

	long TestBBOX_Size(int nConfig)
	{
		return ERROR_NOT_IMPLEMENTED;
	}

	long TestBBOX_Bounds(int nConfig)
	{
		return ERROR_NOT_IMPLEMENTED;
	}

	long TestBBOX_DivBounds(int nConfig)
	{
		return ERROR_NOT_IMPLEMENTED;
	}

	long TestBBOX_Clip(int nConfig)
	{
		return ERROR_NOT_IMPLEMENTED;
	}

	long TestBBOX_Decode(int nConfig)
	{
		return ERROR_NOT_IMPLEMENTED;
	}

	long TestBBOX_Encode(int nConfig)
	{
		return ERROR_NOT_IMPLEMENTED;
	}

	long TestBBOX_Intersect(int nConfig)
	{
		return ERROR_NOT_IMPLEMENTED;
	}

	long TestBBOX_JaccardOverlap(int nConfig)
	{
		return ERROR_NOT_IMPLEMENTED;
	}

	long TestBBOX_Match(int nConfig)
	{
		return ERROR_NOT_IMPLEMENTED;
	}

	long TestFindMatches(int nConfig)
	{
		return ERROR_NOT_IMPLEMENTED;
	}

	long TestCountMatches(int nConfig)
	{
		return ERROR_NOT_IMPLEMENTED;
	}

	long TestSoftMax(int nConfig)
	{
		return ERROR_NOT_IMPLEMENTED;
	}

	long TestComputeConfLoss(int nConfig)
	{
		return ERROR_NOT_IMPLEMENTED;
	}

	long TestComputeLocLoss(int nConfig)
	{
		return ERROR_NOT_IMPLEMENTED;
	}

	long TestGetTopKScores(int nConfig)
	{
		return ERROR_NOT_IMPLEMENTED;
	}

	long TestApplyNMS(int nConfig)
	{
		LONG lErr;
		vector<BBOX> bboxes;
		vector<T> scores;
		T fThreshold = T(0);
		int nTopK = 3;
		vector<int> indices;

		if (nConfig > 0)
		{
			return ERROR_NOT_IMPLEMENTED;
		}

		if (lErr = m_ssd.applyNMS(bboxes, scores, fThreshold, nTopK, &indices))
			return lErr;

		return 0;
	}

	long TestMineHardExamples(int nConfig)
	{
		return ERROR_NOT_IMPLEMENTED;
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

			case BBOX_SIZE:
				test_create(nConfig);
				((TestData<T>*)m_pObj)->TestBBOX_Size(nConfig);
				break;

			case BBOX_BOUNDS:
				test_create(nConfig);
				((TestData<T>*)m_pObj)->TestBBOX_Bounds(nConfig);
				break;

			case BBOX_DIVBOUNDS:
				test_create(nConfig);
				((TestData<T>*)m_pObj)->TestBBOX_DivBounds(nConfig);
				break;

			case BBOX_CLIP:
				test_create(nConfig);
				((TestData<T>*)m_pObj)->TestBBOX_Clip(nConfig);
				break;

			case BBOX_DECODE:
				test_create(nConfig);
				((TestData<T>*)m_pObj)->TestBBOX_Decode(nConfig);
				break;

			case BBOX_ENCODE:
				test_create(nConfig);
				((TestData<T>*)m_pObj)->TestBBOX_Encode(nConfig);
				break;

			case BBOX_INTERSECT:
				test_create(nConfig);
				((TestData<T>*)m_pObj)->TestBBOX_Intersect(nConfig);
				break;

			case BBOX_JACCARDOVERLAP:
				test_create(nConfig);
				((TestData<T>*)m_pObj)->TestBBOX_JaccardOverlap(nConfig);
				break;

			case BBOX_MATCH:
				test_create(nConfig);
				((TestData<T>*)m_pObj)->TestBBOX_Match(nConfig);
				break;

			case FINDMATCHES:
				test_create(nConfig);
				((TestData<T>*)m_pObj)->TestFindMatches(nConfig);
				break;

			case COUNTMATCHES:
				test_create(nConfig);
				((TestData<T>*)m_pObj)->TestCountMatches(nConfig);
				break;

			case SOFTMAX:
				test_create(nConfig);
				((TestData<T>*)m_pObj)->TestSoftMax(nConfig);
				break;

			case COMPUTE_CONF_LOSS:
				test_create(nConfig);
				((TestData<T>*)m_pObj)->TestComputeConfLoss(nConfig);
				break;

			case COMPUTE_LOC_LOSS:
				test_create(nConfig);
				((TestData<T>*)m_pObj)->TestComputeLocLoss(nConfig);
				break;

			case GET_TOPK_SCORES:
				test_create(nConfig);
				((TestData<T>*)m_pObj)->TestGetTopKScores(nConfig);
				break;

			case APPLYNMS:
				test_create(nConfig);
				((TestData<T>*)m_pObj)->TestApplyNMS(nConfig);
				break;

			case MINE_HARD_EXAMPLES:
				test_create(nConfig);
				((TestData<T>*)m_pObj)->TestMineHardExamples(nConfig);
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