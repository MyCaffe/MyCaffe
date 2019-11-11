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
	T m_fEps;

public:
	SsdData<T> m_ssd;

	TestData() : m_memory(), m_math(), m_ssd(&m_memory, &m_math)
	{
		m_fEps = (T)1e-6;
	}

	void EXPECT_NEAR(T t1, T t2, T fErr = 0)
	{
		if (fErr == 0)
			fErr = m_fEps;

		T fDiff = (T)fabs(t1 - t2);
		if (fDiff > fErr)
			throw ERROR_PARAM_OUT_OF_RANGE;
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
		T fSize;

		// Valid box.
		T xmin = T(0.2);
		T ymin = T(0.3);
		T xmax = T(0.3);
		T ymax = T(0.5);
		fSize = SsdBbox<T>::getSize(xmin, ymin, xmax, ymax, true);
		EXPECT_NEAR(fSize, T(0.02));

		// A line.
		xmin = T(0.2);
		ymin = T(0.3);
		xmax = T(0.2);
		ymax = T(0.5);
		fSize = SsdBbox<T>::getSize(xmin, ymin, xmax, ymax, true);
		EXPECT_NEAR(fSize, T(0.0));

		// Invalid box.
		xmin = T(0.2);
		ymin = T(0.3);
		xmax = T(0.1);
		ymax = T(0.5);
		fSize = SsdBbox<T>::getSize(xmin, ymin, xmax, ymax, true);
		EXPECT_NEAR(fSize, T(0.0));

		return 0;
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
		T xmin = T(0.2);
		T ymin = T(0.3);
		T xmax = T(0.3);
		T ymax = T(0.5);

		T xmin2 = T(0.2);
		T ymin2 = T(0.3);
		T xmax2 = T(0.3);
		T ymax2 = T(0.5);

		SsdBbox<T>::clip(&xmin, &ymin, &xmax, &ymax);
		T fSize = SsdBbox<T>::getSize(xmin, ymin, xmax, ymax);
		EXPECT_NEAR(xmin, xmin2);
		EXPECT_NEAR(ymin, ymin2);
		EXPECT_NEAR(xmax, xmax2);
		EXPECT_NEAR(ymax, ymax2);
		EXPECT_NEAR(fSize, T(0.02));

		xmin = T(-0.2);
		ymin = T(-0.3);
		xmax = T(1.3);
		ymax = T(1.5);
		SsdBbox<T>::clip(&xmin, &ymin, &xmax, &ymax);
		fSize = SsdBbox<T>::getSize(xmin, ymin, xmax, ymax);
		EXPECT_NEAR(xmin, T(0.0));
		EXPECT_NEAR(ymin, T(0.0));
		EXPECT_NEAR(xmax, T(1.0));
		EXPECT_NEAR(ymax, T(1.0));
		EXPECT_NEAR(fSize, T(1.0));

		fSize = SsdBbox<T>::getSize(xmin, ymin, xmax, ymax);
		EXPECT_NEAR(fSize, T(1.0));

		return 0;
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
		T xmin;
		T ymin;
		T xmax;
		T ymax;

		T xmin_ref = T(0.2);
		T ymin_ref = T(0.3);
		T xmax_ref = T(0.3);
		T ymax_ref = T(0.5);

		// Partially overlapped.
		T xmin_test = T(0.1);
		T ymin_test = T(0.1);
		T xmax_test = T(0.3);
		T ymax_test = T(0.4);

		SsdBbox<T>::intersect(xmin_ref, ymin_ref, xmax_ref, ymax_ref, xmin_test, ymin_test, xmax_test, ymax_test, &xmin, &ymin, &xmax, &ymax);
		EXPECT_NEAR(xmin, T(0.2));
		EXPECT_NEAR(ymin, T(0.3));
		EXPECT_NEAR(xmax, T(0.3));
		EXPECT_NEAR(ymax, T(0.4));

		// Fully contain.
		xmin_test = T(0.1);
		ymin_test = T(0.1);
		xmax_test = T(0.4);
		ymax_test = T(0.6);

		SsdBbox<T>::intersect(xmin_ref, ymin_ref, xmax_ref, ymax_ref, xmin_test, ymin_test, xmax_test, ymax_test, &xmin, &ymin, &xmax, &ymax);
		EXPECT_NEAR(xmin, T(0.2));
		EXPECT_NEAR(ymin, T(0.3));
		EXPECT_NEAR(xmax, T(0.3));
		EXPECT_NEAR(ymax, T(0.5));

		// Outside.
		xmin_test = T(0.0);
		ymin_test = T(0.0);
		xmax_test = T(0.1);
		ymax_test = T(0.1);

		SsdBbox<T>::intersect(xmin_ref, ymin_ref, xmax_ref, ymax_ref, xmin_test, ymin_test, xmax_test, ymax_test, &xmin, &ymin, &xmax, &ymax);
		EXPECT_NEAR(xmin, T(0.0));
		EXPECT_NEAR(ymin, T(0.0));
		EXPECT_NEAR(xmax, T(0.0));
		EXPECT_NEAR(ymax, T(0.0));

		return 0;
	}

	long TestBBOX_JaccardOverlap(int nConfig)
	{
		T fxmin1 = T(0.2);
		T fymin1 = T(0.3);
		T fxmax1 = T(0.3);
		T fymax1 = T(0.5);

		// Partially overlapped
		T fxmin2 = T(0.1);
		T fymin2 = T(0.1);
		T fxmax2 = T(0.3);
		T fymax2 = T(0.4);
		T fOverlap = SsdBbox<T>::jaccardOverlap(fxmin1, fymin1, fxmax1, fymax1, fxmin2, fymin2, fxmax2, fymax2);
		T fExpected = T(1.0 / 7);
		EXPECT_NEAR(fOverlap, fExpected);

		// Fully contain
		fxmin2 = T(0.1);
		fymin2 = T(0.1);
		fxmax2 = T(0.4);
		fymax2 = T(0.6);
		fOverlap = SsdBbox<T>::jaccardOverlap(fxmin1, fymin1, fxmax1, fymax1, fxmin2, fymin2, fxmax2, fymax2);
		fExpected = T(2.0 / 15);
		EXPECT_NEAR(fOverlap, fExpected);

		// Outside
		fxmin2 = T(0.0);
		fymin2 = T(0.0);
		fxmax2 = T(0.1);
		fymax2 = T(0.1);
		fOverlap = SsdBbox<T>::jaccardOverlap(fxmin1, fymin1, fxmax1, fymax1, fxmin2, fymin2, fxmax2, fymax2);
		EXPECT_NEAR(fOverlap, T(0));

		return 0;
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
		LONG lErr;

		if (lErr = test_create(nConfig))
			throw lErr;

		switch (tst)
		{
			case CREATE:
				break;

			case BBOX_SIZE:
				if (lErr = ((TestData<T>*)m_pObj)->TestBBOX_Size(nConfig))
					throw lErr;
				break;

			case BBOX_BOUNDS:
				if (lErr = ((TestData<T>*)m_pObj)->TestBBOX_Bounds(nConfig))
					throw lErr;
				break;

			case BBOX_DIVBOUNDS:
				if (lErr = ((TestData<T>*)m_pObj)->TestBBOX_DivBounds(nConfig))
					throw lErr;
				break;

			case BBOX_CLIP:
				if (lErr = ((TestData<T>*)m_pObj)->TestBBOX_Clip(nConfig))
					throw lErr;
				break;

			case BBOX_DECODE:
				if (lErr = ((TestData<T>*)m_pObj)->TestBBOX_Decode(nConfig))
					throw lErr;
				break;

			case BBOX_ENCODE:
				if (lErr = ((TestData<T>*)m_pObj)->TestBBOX_Encode(nConfig))
					throw lErr;
				break;

			case BBOX_INTERSECT:
				if (lErr = ((TestData<T>*)m_pObj)->TestBBOX_Intersect(nConfig))
					throw lErr;
				break;

			case BBOX_JACCARDOVERLAP:
				if (lErr = ((TestData<T>*)m_pObj)->TestBBOX_JaccardOverlap(nConfig))
					throw lErr;
				break;

			case BBOX_MATCH:
				if (lErr = ((TestData<T>*)m_pObj)->TestBBOX_Match(nConfig))
					throw lErr;
				break;

			case FINDMATCHES:
				if (lErr = ((TestData<T>*)m_pObj)->TestFindMatches(nConfig))
					throw lErr;
				break;

			case COUNTMATCHES:
				if (lErr = ((TestData<T>*)m_pObj)->TestCountMatches(nConfig))
					throw lErr;
				break;

			case SOFTMAX:
				if (lErr = ((TestData<T>*)m_pObj)->TestSoftMax(nConfig))
					throw lErr;
				break;

			case COMPUTE_CONF_LOSS:
				if (lErr = ((TestData<T>*)m_pObj)->TestComputeConfLoss(nConfig))
					throw lErr;
				break;

			case COMPUTE_LOC_LOSS:
				if (lErr = ((TestData<T>*)m_pObj)->TestComputeLocLoss(nConfig))
					throw lErr;
				break;

			case GET_TOPK_SCORES:
				if (lErr = ((TestData<T>*)m_pObj)->TestGetTopKScores(nConfig))
					throw lErr;
				break;

			case APPLYNMS:
				if (lErr = ((TestData<T>*)m_pObj)->TestApplyNMS(nConfig))
					throw lErr;
				break;

			case MINE_HARD_EXAMPLES:
				if (lErr = ((TestData<T>*)m_pObj)->TestMineHardExamples(nConfig))
					throw lErr;
				break;

			default:
				return ERROR_PARAM_OUT_OF_RANGE;
		}

		cleanup();
	}
	catch (long lErrEx)
	{
		cleanup();
		return lErrEx;
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