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

	BBOX_DECODE1_CORNER = 6,
	BBOX_DECODE1_CENTER_SIZE = 7,
	BBOX_DECODEN_CORNER = 8,
	BBOX_DECODEN_CENTER_SIZE = 9,
	BBOX_ENCODE_CORNER = 10,
	BBOX_ENCODE_CENTER_SIZE = 11,
	BBOX_INTERSECT = 12,
	BBOX_JACCARDOVERLAP = 13,
	BBOX_MATCH_ONEBIPARTITE = 14,
	BBOX_MATCH_ALLBIPARTITE = 15,
	BBOX_MATCH_ONEPERPREDICTION = 16,
	BBOX_MATCH_ALLPERPREDICTION = 17,
	BBOX_MATCH_ALLPERPREDICTIONEX = 18,

	GET_GT = 19,
	GET_LOCPRED_SHARED = 21,
	GET_LOCPRED_UNSHARED = 22,
	GET_CONF_SCORES = 23,

	FINDMATCHES = 24,
	COUNTMATCHES = 25,
	SOFTMAX = 26,
	COMPUTE_CONF_LOSS = 27,
	COMPUTE_LOC_LOSS = 28,
	GET_TOPK_SCORES = 29,
	APPLYNMS = 30,
	MINE_HARD_EXAMPLES = 31
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
	long m_hLocData;
	long m_hConfData;
	long m_hPriorData;
	long m_hGtData;

public:
	SsdData<T> m_ssd;

	TestData() : m_memory(), m_math(), m_ssd(&m_memory, &m_math)
	{
		m_fEps = (T)1e-6;
	}

	void free(long& h)
	{
		if (h != 0)
		{
			m_memory.FreeMemory(h);
			h = 0;
		}
	}

	~TestData()
	{
		free(m_hLocData);
		free(m_hConfData);
		free(m_hPriorData);
		free(m_hGtData);
	}

	void EXPECT_NEAR(T t1, T t2, T fErr = 0)
	{
		if (fErr == 0)
			fErr = m_fEps;

		T fDiff = (T)fabs(t1 - t2);
		if (fDiff > fErr)
			throw ERROR_PARAM_OUT_OF_RANGE;
	}

	void EXPECT_NEAR(T xmin1, T ymin1, T xmax1, T ymax1, T xmin2, T ymin2, T xmax2, T ymax2, T fErr = 0)
	{
		EXPECT_NEAR(xmin1, xmin2, fErr);
		EXPECT_NEAR(ymin1, ymin2, fErr);
		EXPECT_NEAR(xmax1, xmax2, fErr);
		EXPECT_NEAR(ymax1, ymax2, fErr);
	}

	void CHECK_EQ(T t1, T t2)
	{
		if (t1 != t2)
			throw ERROR_PARAM_OUT_OF_RANGE;
	}

	void CHECK_EQ(int t1, int t2)
	{
		if (t1 != t2)
			throw ERROR_PARAM_OUT_OF_RANGE;
	}

	void CHECK_EQ(unsigned long t1, unsigned long t2)
	{
		if (t1 != t2)
			throw ERROR_PARAM_OUT_OF_RANGE;
	}

	void CHECK_EQ(T xmin1, T ymin1, T xmax1, T ymax1, T xmin2, T ymin2, T xmax2, T ymax2)
	{
		CHECK_EQ(xmin1, xmin2);
		CHECK_EQ(ymin1, ymin2);
		CHECK_EQ(xmax1, xmax2);
		CHECK_EQ(ymax1, ymax2);
	}

	void CHECK_BBOX(BBOX b, int nLabel, T fxmin, T fymin, T fxmax, T fymax, bool bDifficult, T fsize)
	{
		int nLabel1 = m_ssd.getLabel(b);

		T fxmin1;
		T fymin1;
		T fxmax1;
		T fymax1;
		m_ssd.getBounds(b, &fxmin1, &fymin1, &fxmax1, &fymax1);
		bool bDifficult1 = m_ssd.getDifficult(b);
		T fsize1 = m_ssd.getSize(b);

		if (nLabel1 != nLabel)
			throw ERROR_PARAM_OUT_OF_RANGE;

		if (bDifficult != bDifficult)
			throw ERROR_PARAM_OUT_OF_RANGE;

		EXPECT_NEAR(fsize1, fsize);
		CHECK_EQ(fxmin1, fymin1, fxmax1, fymax1, fxmin, fymin, fxmax, fymax);
	}

	long TestCreate(int nConfig)
	{
		LONG lErr;

		int nGpuID = 0;
		int nNumClasses = 2;
		bool bShareLocation = true;
		int nLocClasses = 2;
		int nBackgroundLabelId = 0;
		SsdMiningType miningType = SSD_MINING_TYPE_NONE;
		SsdMatchingType matchingType = SSD_MATCHING_TYPE_BIPARTITE;
		bool bUseDifficultGt = false;
		T fOverlapThreshold = T(0.3);
		bool bUsePriorForMatching = true;
		SsdCodeType codeType = SSD_CODE_TYPE_CORNER;
		bool bEncodeVariantInTgt = true;
		bool bBpInsize = false;
		bool bIgnoreCrossBoundary = true;
		bool bUsePriorForNms = true;
		SsdConfLossType confLossType = SSD_CONF_LOSS_TYPE_SOFTMAX;
		SsdLocLossType locLossType = SSD_LOC_LOSS_TYPE_L2;
		T fNegPosRatio = 0;
		T fNegOverlap = 0;
		int nSampleSize = 10;
		bool bMapObjectToAgnostic = false;
		T fNmsThreshold = T(0.1);
		int nNmsTopK = 10;
		T fNmsEta = T(0.0);

		if (lErr = m_ssd.Initialize(nGpuID, nNumClasses, bShareLocation, nLocClasses, nBackgroundLabelId, bUseDifficultGt, miningType, matchingType, fOverlapThreshold, bUsePriorForMatching, codeType, bEncodeVariantInTgt, bBpInsize, bIgnoreCrossBoundary, bUsePriorForNms, confLossType, locLossType, fNegPosRatio, fNegOverlap, nSampleSize, bMapObjectToAgnostic, fNmsThreshold, nNmsTopK, fNmsEta))
			return lErr;

		int nNum = 8;
		int nNumPriors = 6;
		int nNumGt = 4;
		int nNumConf = nNum * nNumPriors * nNumClasses;

		if (lErr = m_ssd.Setup(nNum, nNumPriors, nNumGt))
			return lErr;

		if (lErr = m_memory.AllocMemory(0, false, m_ssd.m_nNum * 4, NULL, 0, &m_hLocData))
			return lErr;

		if (lErr = m_memory.AllocMemory(0, false, nNumConf, NULL, 0, &m_hConfData))
			return lErr;

		if (lErr = m_memory.AllocMemory(0, false, m_ssd.m_nNumPriors * 4 * 2, NULL, 0, &m_hPriorData))
			return lErr;

		if (lErr = m_memory.AllocMemory(0, false, m_ssd.m_nNumGt * 8, NULL, 0, &m_hGtData))
			return lErr;

		if (lErr = m_ssd.SetMemory(m_ssd.m_nNum * 4, m_hLocData, nNumConf, m_hConfData, m_ssd.m_nNumPriors * 4, m_hPriorData, m_ssd.m_nNumGt * 8, m_hGtData))
			return lErr;

		return 0;
	}

	long FillBBoxes(vector<BBOX>& gt_bboxes, vector<BBOX>& pred_bboxes)
	{
		gt_bboxes.clear();
		pred_bboxes.clear();

		// Fill in ground truth bboxes
		BBOX gt1(0, MEM_GT);
		m_ssd.setBounds(gt1, T(0.1), T(0.1), T(0.3), T(0.3));
		m_ssd.setLabel(gt1, 1);
		gt_bboxes.push_back(gt1);

		BBOX gt2(1, MEM_GT);
		m_ssd.setBounds(gt2, T(0.3), T(0.3), T(0.6), T(0.5));
		m_ssd.setLabel(gt2, 2);
		gt_bboxes.push_back(gt2);

		// Fill in the prediction bboxes (use PRIOR mem)
		// 4/9 with label 1
		// 0 with label 2
		MEM type = MEM_PRIOR;
		BBOX bbox1(0, type);
		m_ssd.setBounds(bbox1, T(0.1), T(0.0), T(0.4), T(0.3));
		pred_bboxes.push_back(bbox1);

		// 2/6 with label 1
		// 0 with label 2
		BBOX bbox2(1, type);
		m_ssd.setBounds(bbox2, T(0.0), T(0.1), T(0.2), T(0.3));
		pred_bboxes.push_back(bbox2);

		// 2/8 with label 1
		// 1/11 with label 2
		BBOX bbox3(2, type);
		m_ssd.setBounds(bbox3, T(0.2), T(0.1), T(0.4), T(0.4));
		pred_bboxes.push_back(bbox3);

		// 0 with label 1
		// 4/8 with label 2
		BBOX bbox4(3, type);
		m_ssd.setBounds(bbox4, T(0.4), T(0.3), T(0.7), T(0.5));
		pred_bboxes.push_back(bbox4);

		// 0 with label 1
		// 1/11 with label 2
		BBOX bbox5(4, type);
		m_ssd.setBounds(bbox5, T(0.5), T(0.4), T(0.7), T(0.7));
		pred_bboxes.push_back(bbox5);

		// 0 with label 1
		// 0 with label 2
		BBOX bbox6(5, type);
		m_ssd.setBounds(bbox6, T(0.7), T(0.7), T(0.8), T(0.8));
		pred_bboxes.push_back(bbox6);

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
		LONG lErr;
		T xmin1 = T(0.1);
		T ymin1 = T(0.2);
		T xmax1 = T(0.3);
		T ymax1 = T(0.4);
		T xmin2 = T(0.0);
		T ymin2 = T(0.0);
		T xmax2 = T(0.0);
		T ymax2 = T(0.0);

		for (int i = 0; i < m_ssd.m_rgBbox.size() && i < 4; i++)
		{
			if (lErr = m_ssd.m_rgBbox[i]->setBounds(0, xmin1, ymin1, xmax1, ymax1))
				return lErr;

			if (lErr = m_ssd.m_rgBbox[i]->getBounds(0, &xmin2, &ymin2, &xmax2, &ymax2))
				return lErr;

			CHECK_EQ(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2);

			xmin1 += T(0.01);
			ymin1 += T(0.01);
			xmax1 += T(0.01);
			ymax1 += T(0.01);
		}

		return 0;
	}

	long TestBBOX_DivBounds(int nConfig)
	{
		LONG lErr;
		T xmin1 = T(0.1);
		T ymin1 = T(0.2);
		T xmax1 = T(0.3);
		T ymax1 = T(0.4);
		T xmin2 = T(0.1);
		T ymin2 = T(0.01);
		T xmax2 = T(0.001);
		T ymax2 = T(0.0001);

		for (int i = 0; i < m_ssd.m_rgBbox.size() && i < 4; i++)
		{
			if (lErr = m_ssd.m_rgBbox[i]->setBounds(0, xmin1, ymin1, xmax1, ymax1))
				return lErr;
		}

		for (int i = 0; i < m_ssd.m_rgBbox.size() && i < 4; i++)
		{
			if (lErr = m_ssd.m_rgBbox[i]->divBounds(0, xmin2, ymin2, xmax2, ymax2))
				return lErr;
		}

		for (int i = 0; i < m_ssd.m_rgBbox.size() && i < 4; i++)
		{
			if (lErr = m_ssd.m_rgBbox[i]->getBounds(0, &xmin2, &ymin2, &xmax2, &ymax2))
				return lErr;

			CHECK_EQ(xmin2, ymin2, xmax2, ymax2, xmin1 / T(0.1), ymin1 / T(0.01), xmax1 / T(0.001), ymax1 / T(0.0001));
		}

		return 0;
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

	long TestBBOX_Encode_Corner(int nConfig)
	{
		LONG lErr;

		BBOX prior_bbox(0, MEM_PRIOR);
		if (lErr = m_ssd.setBounds(prior_bbox, T(0.1), T(0.1), T(0.3), T(0.3)))
			return lErr;

		// Set variance which immediately follows the prior bbox in the prior mem.
		BBOX prior_bbox_var(m_ssd.m_nNumPriors, MEM_PRIOR);
		if (lErr = m_ssd.setBounds(prior_bbox_var, T(0.1), T(0.1), T(0.1), T(0.1)))
			return lErr;

		BBOX bbox(0, MEM_GT);
		if (lErr = m_ssd.setBounds(bbox, T(0.0), T(0.2), T(0.4), T(0.5)))
			return lErr;

		T xmin;
		T ymin;
		T xmax;
		T ymax;

		m_ssd.m_codeType = SSD_CODE_TYPE_CORNER;
		m_ssd.m_bEncodeVariantInTgt = true;
		if (lErr = m_ssd.encode(prior_bbox, m_ssd.m_nNumPriors, bbox, &xmin, &ymin, &xmax, &ymax))
			return lErr;

		EXPECT_NEAR(xmin, ymin, xmax, ymax, T(-0.1), T(0.1), T(0.1), T(0.2));

		m_ssd.m_bEncodeVariantInTgt = false;
		if (lErr = m_ssd.encode(prior_bbox, m_ssd.m_nNumPriors, bbox, &xmin, &ymin, &xmax, &ymax))
			return lErr;

		EXPECT_NEAR(xmin, ymin, xmax, ymax, T(-1), T(1), T(1), T(2));

		return 0;
	}

	long TestBBOX_Encode_CenterSize(int nConfig)
	{
		LONG lErr;

		BBOX prior_bbox(0, MEM_PRIOR);
		if (lErr = m_ssd.setBounds(prior_bbox, T(0.1), T(0.1), T(0.3), T(0.3)))
			return lErr;

		// Set variance which immediately follows the prior bbox in the prior mem.
		BBOX prior_bbox_var(m_ssd.m_nNumPriors, MEM_PRIOR);
		if (lErr = m_ssd.setBounds(prior_bbox_var, T(0.1), T(0.1), T(0.2), T(0.2)))
			return lErr;

		BBOX bbox(0, MEM_GT);
		if (lErr = m_ssd.setBounds(bbox, T(0.0), T(0.2), T(0.4), T(0.5)))
			return lErr;

		T xmin;
		T ymin;
		T xmax;
		T ymax;

		m_ssd.m_codeType = SSD_CODE_TYPE_CENTER_SIZE;
		m_ssd.m_bEncodeVariantInTgt = true;
		if (lErr = m_ssd.encode(prior_bbox, m_ssd.m_nNumPriors, bbox, &xmin, &ymin, &xmax, &ymax))
			return lErr;

		EXPECT_NEAR(xmin, ymin, xmax, ymax, T(0.0), T(0.75), log(T(2.0)), log(T(3.0/2)));

		m_ssd.m_bEncodeVariantInTgt = false;
		if (lErr = m_ssd.encode(prior_bbox, m_ssd.m_nNumPriors, bbox, &xmin, &ymin, &xmax, &ymax))
			return lErr;

		EXPECT_NEAR(xmin, ymin, xmax, ymax, T(0.0 / 0.1), T(0.75 / 0.1), log(T(2.0))/T(0.2), log(T(3.0/2))/T(0.2), T(1e-5));

		return 0;
	}

	long TestBBOX_Decode1_Corner(int nConfig)
	{
		LONG lErr;

		BBOX prior_bbox(0, MEM_PRIOR);
		if (lErr = m_ssd.setBounds(prior_bbox, T(0.1), T(0.1), T(0.3), T(0.3)))
			return lErr;

		// Set variance which immediately follows the prior bbox in the prior mem.
		BBOX prior_bbox_var(m_ssd.m_nNumPriors, MEM_PRIOR);
		if (lErr = m_ssd.setBounds(prior_bbox_var, T(0.1), T(0.1), T(0.1), T(0.1)))
			return lErr;

		BBOX bbox(0, MEM_GT);
		if (lErr = m_ssd.setBounds(bbox, T(-1.0), T(1.0), T(1.0), T(2.0)))
			return lErr;

		T xmin;
		T ymin;
		T xmax;
		T ymax;
		T fsize;

		m_ssd.m_codeType = SSD_CODE_TYPE_CORNER;
		m_ssd.m_bEncodeVariantInTgt = false;
		if (lErr = m_ssd.decode(prior_bbox, m_ssd.m_nNumPriors, false, bbox, &xmin, &ymin, &xmax, &ymax, &fsize))
			return lErr;

		EXPECT_NEAR(xmin, ymin, xmax, ymax, T(0.0), T(0.2), T(0.4), T(0.5));

		m_ssd.m_bEncodeVariantInTgt = true;
		if (lErr = m_ssd.decode(prior_bbox, m_ssd.m_nNumPriors, false, bbox, &xmin, &ymin, &xmax, &ymax, &fsize))
			return lErr;

		EXPECT_NEAR(xmin, ymin, xmax, ymax, T(-0.9), T(1.1), T(1.3), T(2.3));

		return 0;
	}

	long TestBBOX_Decode1_CenterSize(int nConfig)
	{
		LONG lErr;

		BBOX prior_bbox(0, MEM_PRIOR);
		if (lErr = m_ssd.setBounds(prior_bbox, T(0.1), T(0.1), T(0.3), T(0.3)))
			return lErr;

		// Set variance which immediately follows the prior bbox in the prior mem.
		BBOX prior_bbox_var(m_ssd.m_nNumPriors, MEM_PRIOR);
		if (lErr = m_ssd.setBounds(prior_bbox_var, T(0.1), T(0.1), T(0.2), T(0.2)))
			return lErr;

		BBOX bbox(0, MEM_GT);
		if (lErr = m_ssd.setBounds(bbox, T(0.0), T(0.75), log(T(2.0)), log(T(3.0/2))))
			return lErr;

		T xmin;
		T ymin;
		T xmax;
		T ymax;
		T fsize;

		m_ssd.m_codeType = SSD_CODE_TYPE_CENTER_SIZE;
		m_ssd.m_bEncodeVariantInTgt = true;
		if (lErr = m_ssd.decode(prior_bbox, m_ssd.m_nNumPriors, false, bbox, &xmin, &ymin, &xmax, &ymax, &fsize))
			return lErr;

		EXPECT_NEAR(xmin, ymin, xmax, ymax, T(0.0), T(0.2), T(0.4), T(0.5));

		if (lErr = m_ssd.setBounds(bbox, T(0.0), T(7.5), log(T(2.0)) * 5, log(T(3.0 / 2)) * 5))
			return lErr;

		m_ssd.m_bEncodeVariantInTgt = false;
		if (lErr = m_ssd.decode(prior_bbox, m_ssd.m_nNumPriors, false, bbox, &xmin, &ymin, &xmax, &ymax, &fsize))
			return lErr;

		EXPECT_NEAR(xmin, ymin, xmax, ymax, T(0.0), T(0.2), T(0.4), T(0.5));

		return 0;
	}

	long TestBBOX_DecodeN_Corner(int nConfig)
	{
		LONG lErr;

		vector<BBOX> prior_bboxes;
		vector<int> prior_variances;
		vector<BBOX> bboxes;
		vector<BBOX> decodeBboxes;

		for (int i = 1; i < 5; i++)
		{
			BBOX prior_bbox((i - 1), MEM_PRIOR);
			if (lErr = m_ssd.setBounds(prior_bbox, T(0.1 * i), T(0.1 * i), T(0.1 * i + 0.2), T(0.1 * i + 0.2)))
				return lErr;

			prior_bboxes.push_back(prior_bbox);

			// Set variance which immediately follows the prior bbox in the prior mem.
			BBOX prior_bbox_var((i - 1) + m_ssd.m_nNumPriors, MEM_PRIOR);
			if (lErr = m_ssd.setBounds(prior_bbox_var, T(0.1), T(0.1), T(0.1), T(0.1)))
				return lErr;

			prior_variances.push_back(std::get<0>(prior_bbox_var));

			BBOX bbox((i - 1), MEM_GT);
			if (lErr = m_ssd.setBounds(bbox, T(-1.0 * (i%2)), T((i + 1) % 2), T((i + 1) % 2), T(i % 2)))
				return lErr;

			bboxes.push_back(bbox);
		}

		m_ssd.m_codeType = SSD_CODE_TYPE_CORNER;
		m_ssd.m_bEncodeVariantInTgt = false;
		if (lErr = m_ssd.decode(prior_bboxes, prior_variances, false, bboxes, decodeBboxes))
			return lErr;

		CHECK_EQ(decodeBboxes.size(), 4);

		T xmin;
		T ymin;
		T xmax;
		T ymax;

		for (int i = 1; i < 5; i++)
		{
			if (lErr = m_ssd.getBounds(decodeBboxes[i-1], &xmin, &ymin, &xmax, &ymax))
				return lErr;

			EXPECT_NEAR(xmin, ymin, xmax, ymax,
				T(0.1*i + i % 2 * -0.1),
				T(0.1*i + (i + 1) % 2 * 0.1),
				T(0.1*i + 0.2 + (i + 1) % 2 * 0.1),
				T(0.1*i + 0.2 + i % 2 * 0.1));
		}

		m_ssd.m_bEncodeVariantInTgt = true;
		if (lErr = m_ssd.decode(prior_bboxes, prior_variances, false, bboxes, decodeBboxes))
			return lErr;

		for (int i = 1; i < 5; i++)
		{
			if (lErr = m_ssd.getBounds(decodeBboxes[i - 1], &xmin, &ymin, &xmax, &ymax))
				return lErr;

			EXPECT_NEAR(xmin, ymin, xmax, ymax,
				T(0.1*i + i % 2 * -1),
				T(0.1*i + (i + 1) % 2 * 1),
				T(0.1*i + 0.2 + (i + 1) % 2 * 1),
				T(0.1*i + 0.2 + i % 2 * 1));
		}

		return 0;
	}

	long TestBBOX_DecodeN_CenterSize(int nConfig)
	{
		LONG lErr;

		vector<BBOX> prior_bboxes;
		vector<int> prior_variances;
		vector<BBOX> bboxes;
		vector<BBOX> decodeBboxes;

		for (int i = 1; i < 5; i++)
		{
			BBOX prior_bbox((i - 1), MEM_PRIOR);
			if (lErr = m_ssd.setBounds(prior_bbox, T(0.1 * i), T(0.1 * i), T(0.1 * i + 0.2), T(0.1 * i + 0.2)))
				return lErr;

			prior_bboxes.push_back(prior_bbox);

			// Set variance which immediately follows the prior bbox in the prior mem.
			BBOX prior_bbox_var((i - 1) + m_ssd.m_nNumPriors, MEM_PRIOR);
			if (lErr = m_ssd.setBounds(prior_bbox_var, T(0.1), T(0.1), T(0.2), T(0.2)))
				return lErr;

			prior_variances.push_back(std::get<0>(prior_bbox_var));

			BBOX bbox((i - 1), MEM_GT);
			if (lErr = m_ssd.setBounds(bbox, T(0.0), T(0.75), log(T(2.0)), log(T(3.0/2))))
				return lErr;

			bboxes.push_back(bbox);
		}

		m_ssd.m_codeType = SSD_CODE_TYPE_CENTER_SIZE;
		m_ssd.m_bEncodeVariantInTgt = true;
		if (lErr = m_ssd.decode(prior_bboxes, prior_variances, false, bboxes, decodeBboxes))
			return lErr;

		CHECK_EQ(decodeBboxes.size(), 4);

		T fEps = T(1e-5);
		T xmin;
		T ymin;
		T xmax;
		T ymax;

		for (int i = 1; i < 5; i++)
		{
			if (lErr = m_ssd.getBounds(decodeBboxes[i - 1], &xmin, &ymin, &xmax, &ymax))
				return lErr;

			EXPECT_NEAR(xmin, ymin, xmax, ymax,
				T(0 + (i - 1) * 0.1),
				T(0.2 + (i - 1) * 0.1),
				T(0.4 + (i - 1) * 0.1),
				T(0.5 + (i - 1) * 0.1), fEps);
		}

		for (int i = 0; i < bboxes.size(); i++)
		{
			if (lErr = m_ssd.setBounds(bboxes[i], T(0.0), T(7.5), log(T(2.0)) * 5, log(T(3.0 / 2)) * 5))
				return lErr;
		}

		m_ssd.m_bEncodeVariantInTgt = false;
		if (lErr = m_ssd.decode(prior_bboxes, prior_variances, false, bboxes, decodeBboxes))
			return lErr;

		for (int i = 1; i < 5; i++)
		{
			if (lErr = m_ssd.getBounds(decodeBboxes[i - 1], &xmin, &ymin, &xmax, &ymax))
				return lErr;

			EXPECT_NEAR(xmin, ymin, xmax, ymax,
				T(0 + (i - 1) * 0.1),
				T(0.2 + (i - 1) * 0.1),
				T(0.4 + (i - 1) * 0.1),
				T(0.5 + (i - 1) * 0.1), fEps);
		}

		return 0;
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

	long TestBBOX_Match_OneBipartite(int nConfig)
	{
		LONG lErr;
		vector<BBOX> gt_bboxes;
		vector<BBOX> pred_bboxes;

		FillBBoxes(gt_bboxes, pred_bboxes);

		int nLabel = 1;
		m_ssd.m_matchingType = SSD_MATCHING_TYPE_BIPARTITE;
		m_ssd.m_fOverlapThreshold = -1;
		m_ssd.m_bIgnoreCrossBoundaryBbox = true;
		
		vector<int> match_indices;
		vector<float> match_overlaps;

		if (lErr = m_ssd.match(gt_bboxes, pred_bboxes, nLabel, &match_indices, &match_overlaps))
			return lErr;

		CHECK_EQ(match_indices.size(), 6);
		CHECK_EQ(match_overlaps.size(), 6);

		CHECK_EQ(match_indices[0], 0);
		CHECK_EQ(match_indices[1], -1);
		CHECK_EQ(match_indices[2], -1);
		EXPECT_NEAR(match_overlaps[0], T(4.0 / 9));
		EXPECT_NEAR(match_overlaps[1], T(2.0 / 6));
		EXPECT_NEAR(match_overlaps[2], T(2.0 / 8));

		for (int i = 3; i < 6; i++)
		{
			CHECK_EQ(match_indices[i], -1);
			EXPECT_NEAR(match_overlaps[i], 0);
		}

		return 0;
	}

	long TestBBOX_Match_AllBipartite(int nConfig)
	{
		LONG lErr;
		vector<BBOX> gt_bboxes;
		vector<BBOX> pred_bboxes;

		FillBBoxes(gt_bboxes, pred_bboxes);

		int nLabel = -1;
		m_ssd.m_matchingType = SSD_MATCHING_TYPE_BIPARTITE;
		m_ssd.m_fOverlapThreshold = -1;
		m_ssd.m_bIgnoreCrossBoundaryBbox = true;

		vector<int> match_indices;
		vector<float> match_overlaps;

		if (lErr = m_ssd.match(gt_bboxes, pred_bboxes, nLabel, &match_indices, &match_overlaps))
			return lErr;

		CHECK_EQ(match_indices.size(), 6);
		CHECK_EQ(match_overlaps.size(), 6);

		CHECK_EQ(match_indices[0], 0);
		CHECK_EQ(match_indices[3], 1);
		EXPECT_NEAR(match_overlaps[0], T(4.0 / 9));
		EXPECT_NEAR(match_overlaps[1], T(2.0 / 6));
		EXPECT_NEAR(match_overlaps[2], T(2.0 / 8));
		EXPECT_NEAR(match_overlaps[3], T(4.0 / 8));
		EXPECT_NEAR(match_overlaps[4], T(1.0 / 11));
		EXPECT_NEAR(match_overlaps[5], T(0));

		for (int i = 0; i < 6; i++)
		{
			if (i == 0 || i == 3)
				continue;

			CHECK_EQ(match_indices[i], -1);
		}

		return 0;
	}

	long TestBBOX_Match_OnePerPrediction(int nConfig)
	{
		LONG lErr;
		vector<BBOX> gt_bboxes;
		vector<BBOX> pred_bboxes;

		FillBBoxes(gt_bboxes, pred_bboxes);

		int nLabel = 1;
		m_ssd.m_matchingType = SSD_MATCHING_TYPE_PER_PREDICTION;
		m_ssd.m_fOverlapThreshold = T(0.3);
		m_ssd.m_bIgnoreCrossBoundaryBbox = true;

		vector<int> match_indices;
		vector<float> match_overlaps;

		if (lErr = m_ssd.match(gt_bboxes, pred_bboxes, nLabel, &match_indices, &match_overlaps))
			return lErr;

		CHECK_EQ(match_indices.size(), 6);
		CHECK_EQ(match_overlaps.size(), 6);

		CHECK_EQ(match_indices[0], 0);
		CHECK_EQ(match_indices[1], 0);
		CHECK_EQ(match_indices[2], -1);
		EXPECT_NEAR(match_overlaps[0], T(4.0 / 9));
		EXPECT_NEAR(match_overlaps[1], T(2.0 / 6));
		EXPECT_NEAR(match_overlaps[2], T(2.0 / 8));

		for (int i = 3; i < 6; i++)
		{
			CHECK_EQ(match_indices[i], -1);
			EXPECT_NEAR(match_overlaps[i], 0);
		}

		return 0;
	}

	long TestBBOX_Match_AllPerPrediction(int nConfig)
	{
		LONG lErr;
		vector<BBOX> gt_bboxes;
		vector<BBOX> pred_bboxes;

		FillBBoxes(gt_bboxes, pred_bboxes);

		int nLabel = -1;
		m_ssd.m_matchingType = SSD_MATCHING_TYPE_PER_PREDICTION;
		m_ssd.m_fOverlapThreshold = T(0.3);
		m_ssd.m_bIgnoreCrossBoundaryBbox = true;

		vector<int> match_indices;
		vector<float> match_overlaps;

		if (lErr = m_ssd.match(gt_bboxes, pred_bboxes, nLabel, &match_indices, &match_overlaps))
			return lErr;

		CHECK_EQ(match_indices.size(), 6);
		CHECK_EQ(match_overlaps.size(), 6);

		CHECK_EQ(match_indices[0], 0);
		CHECK_EQ(match_indices[1], 0);
		CHECK_EQ(match_indices[2], -1);
		CHECK_EQ(match_indices[3], 1);
		CHECK_EQ(match_indices[4], -1);
		CHECK_EQ(match_indices[5], -1);
		EXPECT_NEAR(match_overlaps[0], T(4.0 / 9));
		EXPECT_NEAR(match_overlaps[1], T(2.0 / 6));
		EXPECT_NEAR(match_overlaps[2], T(2.0 / 8));
		EXPECT_NEAR(match_overlaps[3], T(4.0 / 8));
		EXPECT_NEAR(match_overlaps[4], T(1.0 / 11));
		EXPECT_NEAR(match_overlaps[5], T(0));

		return 0;
	}

	long TestBBOX_Match_AllPerPredictionEx(int nConfig)
	{
		LONG lErr;
		vector<BBOX> gt_bboxes;
		vector<BBOX> pred_bboxes;

		FillBBoxes(gt_bboxes, pred_bboxes);

		int nLabel = -1;
		m_ssd.m_matchingType = SSD_MATCHING_TYPE_PER_PREDICTION;
		m_ssd.m_fOverlapThreshold = T(0.001);
		m_ssd.m_bIgnoreCrossBoundaryBbox = true;

		vector<int> match_indices;
		vector<float> match_overlaps;

		if (lErr = m_ssd.match(gt_bboxes, pred_bboxes, nLabel, &match_indices, &match_overlaps))
			return lErr;

		CHECK_EQ(match_indices.size(), 6);
		CHECK_EQ(match_overlaps.size(), 6);

		CHECK_EQ(match_indices[0], 0);
		CHECK_EQ(match_indices[1], 0);
		CHECK_EQ(match_indices[2], 0);
		CHECK_EQ(match_indices[3], 1);
		CHECK_EQ(match_indices[4], 1);
		CHECK_EQ(match_indices[5], -1);
		EXPECT_NEAR(match_overlaps[0], T(4.0 / 9));
		EXPECT_NEAR(match_overlaps[1], T(2.0 / 6));
		EXPECT_NEAR(match_overlaps[2], T(2.0 / 8));
		EXPECT_NEAR(match_overlaps[3], T(4.0 / 8));
		EXPECT_NEAR(match_overlaps[4], T(1.0 / 11));
		EXPECT_NEAR(match_overlaps[5], T(0));

		return 0;
	}

	long TestGetGt(int nConfig)
	{
		LONG lErr;
		int nNumGt = 4;

		for (int i = 0; i < nNumGt; i++)
		{
			BBOX gt(i, MEM_GT);

			int nImageId = ceil(i / (T)2.0);
			int nLabel = i;
			bool bDifficult = (i % 2);

			m_ssd.setBbox(gt, nImageId, nLabel, T(0.1), T(0.1), T(0.3), T(0.3), bDifficult);
		}

		m_ssd.m_nNumGt = nNumGt;
		m_ssd.m_nBackgroundLabelId = -1;
		m_ssd.m_bUseDifficultGt = true;

		map<int, vector<BBOX>> all_gt_bboxes;
		if (lErr = m_ssd.getGt(all_gt_bboxes))
			return lErr;

		CHECK_EQ(all_gt_bboxes.size(), 3);

		CHECK_EQ(all_gt_bboxes[0].size(), 1);
		CHECK_BBOX(all_gt_bboxes[0][0], 0, T(0.1), T(0.1), T(0.3), T(0.3), false, T(0.04));

		CHECK_EQ(all_gt_bboxes[1].size(), 2);
		for (int i = 1; i < 3; i++)
		{
			CHECK_BBOX(all_gt_bboxes[1][i-1], i, T(0.1), T(0.1), T(0.3), T(0.3), i%2, T(0.04));
		}

		CHECK_EQ(all_gt_bboxes[2].size(), 1);
		CHECK_BBOX(all_gt_bboxes[2][0], 3, T(0.1), T(0.1), T(0.3), T(0.3), true, T(0.04));

		// Skip difficult ground truth.
		m_ssd.m_nBackgroundLabelId = -1;
		m_ssd.m_bUseDifficultGt = false;

		if (lErr = m_ssd.getGt(all_gt_bboxes))
			return lErr;

		CHECK_EQ(all_gt_bboxes.size(), 2);

		CHECK_EQ(all_gt_bboxes[0].size(), 1);
		CHECK_BBOX(all_gt_bboxes[0][0], 0, T(0.1), T(0.1), T(0.3), T(0.3), false, T(0.04));

		CHECK_EQ(all_gt_bboxes[1].size(), 1);
		CHECK_BBOX(all_gt_bboxes[1][0], 2, T(0.1), T(0.1), T(0.3), T(0.3), false, T(0.04));

		return 0;
	}

	long TestGetLocPredShared(int nConfig)
	{
		int nNum = 2;
		int nNumPredsPerClass = 2;
		int nNumLocClasses = 1;
		bool bShareLocation = true;
		int nIdx = 0;

		for (int i = 0; i < nNum; i++)
		{
			for (int j = 0; j < nNumPredsPerClass; j++)
			{
				BBOX bbox(nIdx, MEM_LOC);
				T xmin = T(i * nNumPredsPerClass * 0.1 + j * 0.1);
				T ymin = T(i * nNumPredsPerClass * 0.1 + j * 0.1);
				T xmax = T(i * nNumPredsPerClass * 0.1 + j * 0.1 + 0.2);
				T ymax = T(i * nNumPredsPerClass * 0.1 + j * 0.1 + 0.2);
				m_ssd.setBounds(bbox, xmin, ymin, xmax, ymax);
				nIdx++;
			}
		}

		m_ssd.Setup(nNum, nNumPredsPerClass, 2);
		m_ssd.m_bShareLocation = bShareLocation;
		m_ssd.m_nLocClasses = nNumLocClasses;

		vector<map<int, vector<BBOX>>> all_loc_bboxes;
		m_ssd.getLocPrediction(all_loc_bboxes);

		CHECK_EQ(all_loc_bboxes.size(), nNum);

		for (int i = 0; i < nNum; i++)
		{
			CHECK_EQ(all_loc_bboxes[i].size(), 1);

			map<int, vector<BBOX>>::iterator it = all_loc_bboxes[i].begin();
			CHECK_EQ(it->first, -1);

			vector<BBOX> bboxes = it->second;
			CHECK_EQ(bboxes.size(), nNumPredsPerClass);

			T fStartVal = T(i * nNumPredsPerClass * 0.1);

			for (int j = 0; j < nNumPredsPerClass; j++)
			{
				T xmin1 = fStartVal + j * T(0.1);
				T ymin1 = fStartVal + j * T(0.1);
				T xmax1 = fStartVal + j * T(0.1) + T(0.2);
				T ymax1 = fStartVal + j * T(0.1) + T(0.2);
				T xmin;
				T ymin;
				T xmax;
				T ymax;
				
				m_ssd.getBounds(bboxes[j], &xmin, &ymin, &xmax, &ymax);
				CHECK_EQ(xmin1, ymin1, xmax1, ymax1, xmin, ymin, xmax, ymax);
			}
		}

		return 0;
	}

	long TestGetLocPredUnShared(int nConfig)
	{
		int nNum = 2;
		int nNumPredsPerClass = 2;
		int nNumLocClasses = 2;
		bool bShareLocation = false;
		int nIdx = 0;

		for (int i = 0; i < nNum; i++)
		{
			for (int j = 0; j < nNumPredsPerClass; j++)
			{
				T fStartValue = T((i * nNumPredsPerClass + j) * nNumLocClasses * 0.1);

				for (int c = 0; c < nNumLocClasses; c++)
				{
					nIdx = ((i * nNumPredsPerClass + j) * nNumLocClasses + c);
					BBOX bbox(nIdx, MEM_LOC);
					T xmin = T(fStartValue + c * 0.1);
					T ymin = T(fStartValue + c * 0.1);
					T xmax = T(fStartValue + c * 0.1 + 0.2);
					T ymax = T(fStartValue + c * 0.1 + 0.2);
					m_ssd.setBounds(bbox, xmin, ymin, xmax, ymax);
				}
			}
		}

		m_ssd.Setup(nNum, nNumPredsPerClass, 2);
		m_ssd.m_bShareLocation = bShareLocation;
		m_ssd.m_nLocClasses = nNumLocClasses;

		vector<map<int, vector<BBOX>>> all_loc_bboxes;
		m_ssd.getLocPrediction(all_loc_bboxes);

		CHECK_EQ(all_loc_bboxes.size(), nNum);

		for (int i = 0; i < nNum; i++)
		{
			CHECK_EQ(all_loc_bboxes[i].size(), nNumLocClasses);

			for (int c = 0; c < nNumLocClasses; c++)
			{
				map<int, vector<BBOX>>::iterator it = all_loc_bboxes[i].find(c);
				CHECK_EQ(it->first, c);

				vector<BBOX> bboxes = it->second;
				CHECK_EQ(bboxes.size(), nNumPredsPerClass);

				for (int j = 0; j < nNumPredsPerClass; j++)
				{
					T fStartVal = T(i * nNumPredsPerClass + j) * T(nNumLocClasses * 0.1);
					T xmin1 = fStartVal + c * T(0.1);
					T ymin1 = fStartVal + c * T(0.1);
					T xmax1 = fStartVal + c * T(0.1) + T(0.2);
					T ymax1 = fStartVal + c * T(0.1) + T(0.2);
					T xmin;
					T ymin;
					T xmax;
					T ymax;

					m_ssd.getBounds(bboxes[j], &xmin, &ymin, &xmax, &ymax);
					CHECK_EQ(xmin1, ymin1, xmax1, ymax1, xmin, ymin, xmax, ymax);
				}
			}
		}

		return 0;
	}

	long TestGetConfScores(int nConfig);

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

template <>
long TestData<float>::TestGetConfScores(int nConfig)
{
	LONG lErr;
	int nNum = 2;
	int nNumPredsPerClass = 2;
	int nNumClasses = 2;
	int nIdx;
	float* conf_data = m_ssd.m_pConf->cpu_data();

	for (int i = 0; i < nNum; i++)
	{
		for (int j = 0; j < nNumPredsPerClass; j++)
		{
			for (int c = 0; c < nNumClasses; c++)
			{
				int nIdx = (i * nNumPredsPerClass + j) * nNumClasses + c;
				conf_data[nIdx] = nIdx * float(0.1);
			}
		}
	}

	m_ssd.Setup(nNum, nNumPredsPerClass, 2);
	m_ssd.m_nNumClasses = nNumClasses;

	vector<map<int, vector<float>>> all_conf_preds;
	if (lErr = m_ssd.getConfidenceScores(false, all_conf_preds))
		return lErr;

	for (int i = 0; i < nNum; i++)
	{
		CHECK_EQ(all_conf_preds[i].size(), nNumClasses);

		for (int c = 0; c < nNumClasses; c++)
		{
			map<int, vector<float>>::iterator it = all_conf_preds[i].find(c);

			CHECK_EQ(it->first, c);
			const vector<float>& confidences = it->second;

			for (int j = 0; j < nNumPredsPerClass; j++)
			{
				int nIdx = (i * nNumPredsPerClass + j) * nNumClasses + c;
				EXPECT_NEAR(confidences[j], nIdx * float(0.1));
			}
		}
	}

	return 0;
}

template <>
long TestData<double>::TestGetConfScores(int nConfig)
{
	LONG lErr;
	int nNum = 2;
	int nNumPredsPerClass = 2;
	int nNumClasses = 2;
	int nIdx;
	double* conf_data = m_ssd.m_pConf->cpu_data();

	for (int i = 0; i < nNum; i++)
	{
		for (int j = 0; j < nNumPredsPerClass; j++)
		{
			for (int c = 0; c < nNumClasses; c++)
			{
				int nIdx = (i * nNumPredsPerClass + j) * nNumClasses + c;
				conf_data[nIdx] = nIdx * double(0.1);
			}
		}
	}

	m_ssd.Setup(nNum, nNumPredsPerClass, 2);
	m_ssd.m_nNumClasses = nNumClasses;

	vector<map<int, vector<double>>> all_conf_preds;
	if (lErr = m_ssd.getConfidenceScores(false, all_conf_preds))
		return lErr;

	for (int i = 0; i < nNum; i++)
	{
		CHECK_EQ(all_conf_preds[i].size(), nNumClasses);

		for (int c = 0; c < nNumClasses; c++)
		{
			map<int, vector<double>>::iterator it = all_conf_preds[i].find(c);

			CHECK_EQ(it->first, c);
			const vector<double>& confidences = it->second;

			for (int j = 0; j < nNumPredsPerClass; j++)
			{
				int nIdx = (i * nNumPredsPerClass + j) * nNumClasses + c;
				EXPECT_NEAR(confidences[j], nIdx * double(0.1));
			}
		}
	}

	return 0;
}


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

			case BBOX_DECODE1_CORNER:
				if (lErr = ((TestData<T>*)m_pObj)->TestBBOX_Decode1_Corner(nConfig))
					throw lErr;
				break;

			case BBOX_DECODE1_CENTER_SIZE:
				if (lErr = ((TestData<T>*)m_pObj)->TestBBOX_Decode1_CenterSize(nConfig))
					throw lErr;
				break;

			case BBOX_DECODEN_CORNER:
				if (lErr = ((TestData<T>*)m_pObj)->TestBBOX_DecodeN_Corner(nConfig))
					throw lErr;
				break;

			case BBOX_DECODEN_CENTER_SIZE:
				if (lErr = ((TestData<T>*)m_pObj)->TestBBOX_DecodeN_CenterSize(nConfig))
					throw lErr;
				break;

			case BBOX_ENCODE_CORNER:
				if (lErr = ((TestData<T>*)m_pObj)->TestBBOX_Encode_Corner(nConfig))
					throw lErr;
				break;

			case BBOX_ENCODE_CENTER_SIZE:
				if (lErr = ((TestData<T>*)m_pObj)->TestBBOX_Encode_CenterSize(nConfig))
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

			case BBOX_MATCH_ONEBIPARTITE:
				if (lErr = ((TestData<T>*)m_pObj)->TestBBOX_Match_OneBipartite(nConfig))
					throw lErr;
				break;

			case BBOX_MATCH_ALLBIPARTITE:
				if (lErr = ((TestData<T>*)m_pObj)->TestBBOX_Match_AllBipartite(nConfig))
					throw lErr;
				break;

			case BBOX_MATCH_ONEPERPREDICTION:
				if (lErr = ((TestData<T>*)m_pObj)->TestBBOX_Match_OnePerPrediction(nConfig))
					throw lErr;
				break;

			case BBOX_MATCH_ALLPERPREDICTION:
				if (lErr = ((TestData<T>*)m_pObj)->TestBBOX_Match_AllPerPrediction(nConfig))
					throw lErr;
				break;

			case BBOX_MATCH_ALLPERPREDICTIONEX:
				if (lErr = ((TestData<T>*)m_pObj)->TestBBOX_Match_AllPerPredictionEx(nConfig))
					throw lErr;
				break;

			case GET_GT:
				if (lErr = ((TestData<T>*)m_pObj)->TestGetGt(nConfig))
					throw lErr;
				break;

			case GET_LOCPRED_SHARED:
				if (lErr = ((TestData<T>*)m_pObj)->TestGetLocPredShared(nConfig))
					throw lErr;
				break;

			case GET_LOCPRED_UNSHARED:
				if (lErr = ((TestData<T>*)m_pObj)->TestGetLocPredUnShared(nConfig))
					throw lErr;
				break;

			case GET_CONF_SCORES:
				if (lErr = ((TestData<T>*)m_pObj)->TestGetConfScores(nConfig))
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