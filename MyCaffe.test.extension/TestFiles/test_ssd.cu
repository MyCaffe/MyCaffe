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
	BBOX_MATCH = 14,

	FINDMATCHES = 15,
	COUNTMATCHES = 16,
	SOFTMAX = 17,
	COMPUTE_CONF_LOSS = 18,
	COMPUTE_LOC_LOSS = 19,
	GET_TOPK_SCORES = 20,
	APPLYNMS = 21,
	MINE_HARD_EXAMPLES = 22
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

	void CHECK_EQ(T xmin1, T ymin1, T xmax1, T ymax1, T xmin2, T ymin2, T xmax2, T ymax2)
	{
		CHECK_EQ(xmin1, xmin2);
		CHECK_EQ(ymin1, ymin2);
		CHECK_EQ(xmax1, xmax2);
		CHECK_EQ(ymax1, ymax2);
	}

	long TestCreate(int nConfig)
	{
		LONG lErr;

		if (lErr = m_ssd.Initialize(0, 2, true, 2, 0, false, SSD_MINING_TYPE_NONE, SSD_MATCHING_TYPE_BIPARTITE, T(0.3), true, SSD_CODE_TYPE_CORNER, true, false, true, true, SSD_CONF_LOSS_TYPE_SOFTMAX, SSD_LOC_LOSS_TYPE_L2, 0, 0, 10, false, T(0.1), 10, T(0.1)))
			return lErr;

		if (lErr = m_ssd.Setup(2, 6, 2))
			return lErr;

		if (lErr = m_memory.AllocMemory(0, false, m_ssd.m_nNum * 4, NULL, 0, &m_hLocData))
			return lErr;

		if (lErr = m_memory.AllocMemory(0, false, m_ssd.m_nNum * 4, NULL, 0, &m_hConfData))
			return lErr;

		if (lErr = m_memory.AllocMemory(0, false, m_ssd.m_nNumPriors * 4 * 2, NULL, 0, &m_hPriorData))
			return lErr;

		if (lErr = m_memory.AllocMemory(0, false, m_ssd.m_nNumGt * 8, NULL, 0, &m_hGtData))
			return lErr;

		if (lErr = m_ssd.SetMemory(m_ssd.m_nNum * 4, m_hLocData, m_ssd.m_nNum * 4, m_hConfData, m_ssd.m_nNumPriors * 4, m_hPriorData, m_ssd.m_nNumGt * 8, m_hGtData))
			return lErr;

		return 0;
	}

	long FillBBoxes()
	{
		// Fill in ground truth bboxes
		m_ssd.m_rgBbox[MEM_GT]->setBounds(0, T(0.1), T(0.1), T(0.3), T(0.3));
		m_ssd.m_rgBbox[MEM_GT]->setLabel(0, 1);

		m_ssd.m_rgBbox[MEM_GT]->setBounds(1, T(0.3), T(0.3), T(0.6), T(0.5));
		m_ssd.m_rgBbox[MEM_GT]->setLabel(1, 2);

		// Fill in the prediction bboxes (use PRIOR mem)
		// 4/9 with label 1
		// 0 with label 2
		m_ssd.m_rgBbox[MEM_PRIOR]->setBounds(0, T(0.1), T(0.0), T(0.4), T(0.3));

		// 2/6 with label 1
		// 0 with label 2
		m_ssd.m_rgBbox[MEM_PRIOR]->setBounds(1, T(0.0), T(0.1), T(0.2), T(0.3));

		// 2/8 with label 1
		// 1/11 with label 2
		m_ssd.m_rgBbox[MEM_PRIOR]->setBounds(2, T(0.2), T(0.1), T(0.4), T(0.4));

		// 0 with label 1
		// 4/8 with label 2
		m_ssd.m_rgBbox[MEM_PRIOR]->setBounds(3, T(0.4), T(0.3), T(0.7), T(0.5));

		// 0 with label 1
		// 1/11 with label 2
		m_ssd.m_rgBbox[MEM_PRIOR]->setBounds(4, T(0.5), T(0.4), T(0.7), T(0.7));

		// 0 with label 1
		// 0 with label 2
		m_ssd.m_rgBbox[MEM_PRIOR]->setBounds(5, T(0.7), T(0.7), T(0.8), T(0.8));
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