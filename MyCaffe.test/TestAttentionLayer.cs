using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.param;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.fillers;
using MyCaffe.layers;
using System.Diagnostics;

namespace MyCaffe.test
{
    [TestClass]
    public class TestAttentionLayer
    {
        [TestMethod]
        public void TestForward()
        {
            AttentionLayerTest test = new AttentionLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IAttentionLayerTest t in test.Tests)
                {
                    t.TestForward();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradient()
        {
            AttentionLayerTest test = new AttentionLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IAttentionLayerTest t in test.Tests)
                {
                    t.TestGradient();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradient2()
        {
            AttentionLayerTest test = new AttentionLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IAttentionLayerTest t in test.Tests)
                {
                    t.TestGradient2();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface IAttentionLayerTest : ITest
    {
        void TestForward();
        void TestGradient();
        void TestGradient2();
    }

    class AttentionLayerTest : TestBase
    {
        public AttentionLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Attention Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new AttentionLayerTest2<double>(strName, nDeviceID, engine);
            else
                return new AttentionLayerTest2<float>(strName, nDeviceID, engine);
        }
    }

    class AttentionLayerTest2<T> : TestEx<T>, IAttentionLayerTest
    {
        Blob<T> m_blobA = null;
        Blob<T> m_blobB = null;
        Blob<T> m_blobC = null;
        Blob<T> m_blobState = null;
        Blob<T> m_blobHy = null;
        Blob<T> m_blobClip = null;
        float[] m_rgUa;
        float[] m_rgWa;
        float[] m_rgV;
        float[] m_rgWc;
        float[] m_rgAa;
        float[] m_rgSoftmax;

        public AttentionLayerTest2(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 3, 2, 4, 1 }, nDeviceID)
        {
            m_engine = engine;
            m_blobA = new Blob<T>(m_cuda, m_log);
            m_blobB = new Blob<T>(m_cuda, m_log);
            m_blobC = new Blob<T>(m_cuda, m_log);
            m_blobHy = new Blob<T>(m_cuda, m_log);
            m_blobState = new Blob<T>(m_cuda, m_log);
            m_blobClip = new Blob<T>(m_cuda, m_log);
        }

        protected override FillerParameter getFillerParam()
        {
            return base.getFillerParam();
        }

        private void dispose1(ref Blob<T> b)
        {
            if (b != null)
            {
                b.Dispose();
                b = null;
            }
        }

        protected override void dispose()
        {
            dispose1(ref m_blobA);
            dispose1(ref m_blobB);
            dispose1(ref m_blobC);
            dispose1(ref m_blobHy);
            dispose1(ref m_blobState);
            base.dispose();
        }

        public float[] Fill(AttentionParameter p)
        {
            // timesteps = 3, batch = 2, input = 4
            float[] rgData = convertF(m_blob_bottom.mutable_cpu_data); // shape (3, 2, 4, 1)
            // timestep 1, batch 1
            rgData[0] = 1.11f;
            rgData[1] = 1.12f;
            rgData[2] = 1.13f;
            rgData[3] = 1.14f;
            // timestep 1, batch 2
            rgData[4] = 1.21f;
            rgData[5] = 1.22f;
            rgData[6] = 1.23f;
            rgData[7] = 1.24f;

            // timestep 2, batch 1
            rgData[8] = 2.11f;
            rgData[9] = 2.12f;
            rgData[10] = 2.13f;
            rgData[11] = 2.14f;
            // timestep 2, batch 2
            rgData[12] = 2.21f;
            rgData[13] = 2.22f;
            rgData[14] = 2.23f;
            rgData[15] = 2.24f;

            // timestep 3, batch 1
            rgData[16] = 3.11f;
            rgData[17] = 3.12f;
            rgData[18] = 3.13f;
            rgData[19] = 3.14f;
            // timestep 3, batch 2
            rgData[20] = 3.21f;
            rgData[21] = 3.22f;
            rgData[22] = 3.23f;
            rgData[23] = 3.24f;

            m_blob_bottom.mutable_cpu_data = convert(rgData);

            m_blobState.Reshape(1, 2, (int)p.dim, 1);
            m_blobHy.ReshapeLike(m_blobState);
            m_blobHy.SetData(0);

            List<int> rgShape = Utility.Clone<int>(m_blob_bottom.shape());
            while (rgShape.Count > 2)
            {
                rgShape.RemoveAt(rgShape.Count - 1);
            }
            m_blobClip.Reshape(rgShape);
            m_blobClip.SetData(1);

            BottomVec.Clear();
            BottomVec.Add(m_blob_bottom);
            BottomVec.Add(m_blobHy);
            BottomVec.Add(m_blobState);
            BottomVec.Add(m_blobClip);

            return rgData;
        }

        public float[] FillSmall(AttentionParameter p)
        {
            m_blob_bottom.Reshape(2, 1, 1, 1);

            // timesteps = 2, batch = 1, input = 1
            float[] rgData = convertF(m_blob_bottom.mutable_cpu_data); // shape (2, 1, 1, 1)
            // timestep 1, batch 1
            rgData[0] = 1.11f;
            rgData[1] = 2.11f;

            m_blob_bottom.mutable_cpu_data = convert(rgData);
            m_blobState.Reshape(1, 1, (int)p.dim, 1);
            m_blobHy.ReshapeLike(m_blobState);
            m_blobHy.SetData(0);

            List<int> rgShape = Utility.Clone<int>(m_blob_bottom.shape());
            while (rgShape.Count > 2)
            {
                rgShape.RemoveAt(rgShape.Count - 1);
            }
            m_blobClip.Reshape(rgShape);
            m_blobClip.SetData(1);

            BottomVec.Clear();
            BottomVec.Add(m_blob_bottom);
            BottomVec.Add(m_blobHy);
            BottomVec.Add(m_blobState);
            BottomVec.Add(m_blobClip);

            return rgData;
        }

        private float[] gemm(bool bTransposeA, bool bTransposeB, int nM, int nN, int nK, float fAlpha, float[] rgA, float[] rgB, float fBeta)
        {
            m_blobA.Reshape(nM, nK, 1, 1);
            m_blobB.Reshape(nN, nK, 1, 1);
            m_blobC.Reshape(nM, nN, 1, 1);

            m_log.CHECK_EQ(rgA.Length, nM * nK, "The vector A has incorrect size, expected " + nM.ToString() + " x " + nK.ToString());
            m_log.CHECK_EQ(rgB.Length, nN * nK, "The vector B has incorrect size, expected " + nN.ToString() + " x " + nK.ToString());

            m_blobA.mutable_cpu_data = convert(rgA);
            m_blobB.mutable_cpu_data = convert(rgB);

            m_cuda.gemm(bTransposeA, bTransposeB, nM, nN, nK, fAlpha, m_blobA.gpu_data, m_blobB.gpu_data, fBeta, m_blobC.mutable_gpu_data);

            return convertF(m_blobC.mutable_cpu_data);
        }

        private float[] expand(float[] rg, int n, float fScale = 1.0f)
        {
            float[] rgDst = new float[rg.Length * n];

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < rg.Length; j++)
                {
                    int nDstIdx = (i * rg.Length) + j;
                    rgDst[nDstIdx] = rg[j] * fScale;
                }
            }

            return rgDst;
        }

        private float[] add(float[] rgA, float[] rgB)
        {
            float[] rgC = new float[rgA.Length];

            for (int i = 0; i < rgA.Length; i++)
            {
                rgC[i] = rgA[i] + rgB[i];
            }

            return rgC;
        }

        private float[] tanh(float[] rgA)
        {
            float[] rgB = new float[rgA.Length];

            for (int i = 0; i < rgA.Length; i++)
            {
                rgB[i] = (float)Math.Tanh(rgA[i]);
            }

            return rgB;
        }

        private void create_weights(ref float[] rg, int n)
        {
            if (rg == null)
                rg = new float[n];

            for (int i = 0; i < n; i++)
            {
                rg[i] = 1;
            }
        }

        private float[] softmax(float[] rg, int nB, int nT)
        {
            float[] rgRes = new float[nB * nT];
            List<float> rgMax = new List<float>();
            List<float> rgSum = new List<float>();

            // Get the maximum on each channel (e.g. set of timesteps)
            for (int i = 0; i < nB; i++)
            {
                float fMax = 0;

                for (int j = 0; j < nT; j++)
                {
                    int nIdx = i * nT + j;
                    float fVal = rg[nIdx];
                    fMax = Math.Max(fMax, fVal);
                }

                rgMax.Add(fMax);
            }

            // Subtract the max, exponetiate and sum over T.
            for (int i = 0; i < nB; i++)
            {
                float fSum = 0;
                float fMax = rgMax[i];

                for (int j = 0; j < nT; j++)
                {
                    int nIdx = i * nT + j;
                    float fVal = rg[nIdx];

                    fVal -= fMax;
                    fVal = (float)Math.Exp(fVal);

                    rgRes[nIdx] = fVal;
                    fSum += fVal;
                }

                rgSum.Add(fSum);
            }

            // Divide by the sums.
            for (int i = 0; i < nB; i++)
            {
                float fSum = rgSum[i];

                for (int j = 0; j < nT; j++)
                {
                    int nIdx = i * nT + j;
                    rgRes[nIdx] /= fSum;
                }
            }

            return rgRes;
        }

        private float[] softmaxGrad(float[] rgData, float[] rgDiff, int nB, int nT)
        {
            float[] rgGrad = new float[nB * nT];
            float[] rgScale = new float[nB * nT];

            for (int i = 0; i < rgDiff.Length; i++)
            {
                rgGrad[i] = rgDiff[i];
            }

            for (int i = 0; i < nB; i++)
            {
                for (int j = 0; j < nT; j++)
                {
                    int nIdx = i * nT + j;
                    rgScale[nIdx] = rgDiff[nIdx] * rgData[nIdx];
                    rgGrad[nIdx] -= rgScale[nIdx];
                    rgGrad[nIdx] *= rgData[nIdx];
                }
            }

            return rgGrad;
        }

        private float[] mul(float[] rgM, float[] rgV, int nB, int nT, int nI)
        {
            float[] rgRes = new float[rgM.Length];

            for (int i = 0; i < nB; i++)
            {
                for (int j = 0; j < nT; j++)
                {
                    int nIdxV = (i * nT) + j;

                    for (int k = 0; k < nI; k++)
                    {
                        int nIdxM = (i * nT * nI) + (j * nI) + k;

                        rgRes[nIdxM] = rgM[nIdxM] * rgV[nIdxV];
                    }
                }
            }

            return rgRes;
        }

        private float[] div(float[] rgM, float[] rgV, int nB, int nT, int nI)
        {
            float[] rgRes = new float[rgM.Length];

            for (int i = 0; i < nB; i++)
            {
                for (int j = 0; j < nT; j++)
                {
                    int nIdxV = (i * nT) + j;

                    for (int k = 0; k < nI; k++)
                    {
                        int nIdxM = (i * nT * nI) + (j * nI) + k;

                        rgRes[nIdxM] = (rgV[nIdxV] == 0) ? 0 : rgM[nIdxM] / rgV[nIdxV];
                    }
                }
            }

            return rgRes;
        }

        private float[] sum(float[] rgM, int nB, int nT, int nI)
        {
            float[] rgRes = new float[nB * nI];

            for (int i = 0; i < nB; i++)
            {
                for (int j = 0; j < nT; j++)
                {
                    for (int k = 0; k < nI; k++)
                    {
                        int nDstIdx = (i * nI) + k;
                        int nSrcIdx = (i * nT * nI) + (j * nI) + k;

                        rgRes[nDstIdx] += rgM[nSrcIdx];
                    }
                }
            }

            return rgRes;
        }

        private float sumsq(float[] rg)
        {
            float fSum = 0;

            for (int i = 0; i < rg.Length; i++)
            {
                fSum += rg[i] * rg[i];
            }

            return fSum;
        }

        private float[] calculateAttention(AttentionParameter p, float[] rgData, float[] rgState, int nT, int nB, int nI, int nS)
        {
            int nM = nT * nB;
            int nN = (int)p.dim;
            int nK = nI;

            float[] rgDataT = SimpleDatum.Transpose(rgData, nT, nB, nI);
            float[] rgStateT = SimpleDatum.Transpose(rgState, 1, nB, nS);

            // IP input data with rgUa wts.
            create_weights(ref m_rgUa, nN * nK);
            float[] rgUh = gemm(false, false, nM, nN, nK, 1.0f, rgDataT, m_rgUa, 0.0f);

            // IP rgFullState with rgWa wts.
            nM = nB;
            nK = nT;
            create_weights(ref m_rgWa, nN * nK);
            float[] rgWc = gemm(false, false, nM, nN, nK, 1.0f, rgState, m_rgWa, 0.0f);

            // Copy rgWc across all T.
            float[] rgFullWc = expand(rgWc, nT);

            // Add uh + wc
            float[] rgUhWc = add(rgUh, rgFullWc);

            // rgGg = Tanh(un + wc);
            float[] rgGg = tanh(rgUhWc);

            // rgAa = IP rgGg with rgV wts.
            nM = nT * nB;
            nN = 1;
            nK = nS;
            create_weights(ref m_rgV, nN * nK);
            m_rgAa = gemm(false, false, nM, nN, nK, 1.0f, rgGg, m_rgV, 0.0f);

            // Softmax over time steps T.
            m_rgSoftmax = softmax(m_rgAa, nB, nT);

            // Multiply softmax vector with input data.
            float[] rgFocusInput = mul(rgDataT, m_rgSoftmax, nB, nT, nI);

            // Sum across all T.
            float[] rgContext = sum(rgFocusInput, nB, nT, nI);

            // IP context with Wc
            nM = nB;
            nN = (int)p.dim;
            nK = nI;
            create_weights(ref m_rgWc, nN * nK);
            float[] rgNewState = gemm(false, false, nM, nN, nK, 1.0f, rgContext, m_rgWc, 0.0f);

            return rgNewState;
        }


        public void TestForward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.ATTENTION);
            p.attention_param.axis = 2;
            p.attention_param.dim = 3;
            p.attention_param.weight_filler = new FillerParameter("constant", 1);
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            m_log.CHECK(layer.type == LayerParameter.LayerType.ATTENTION, "The layer type is incorrect!");

            float[] rgData = Fill(p.attention_param);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            // Now, check values
            float[] rgExpected = calculateAttention(p.attention_param, rgData, convertF(m_blobState.mutable_cpu_data), m_blob_bottom.num, m_blob_bottom.channels, m_blob_bottom.count(2), m_blobState.count(2));
            float[] rgActual = convertF(m_blob_top.mutable_cpu_data);

            for (int i = 0; i < rgExpected.Length; i++)
            {
                float fExpected = rgExpected[i];
                float fActual = rgActual[i];
                float fErr = 0.002f;

                m_log.EXPECT_NEAR_FLOAT(fExpected, fActual, fErr, "The values are not as expected!");
            }
        }

        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.ATTENTION);
            p.attention_param.axis = 2;
            p.attention_param.dim = 3;
            p.attention_param.weight_filler = new FillerParameter("constant", 1);
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            m_log.CHECK(layer.type == LayerParameter.LayerType.ATTENTION, "The layer type is incorrect!");

            Fill(p.attention_param);

            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 0.01, 0.01);
            checker.CheckGradient(layer, BottomVec, TopVec);
        }

        public void TestGradient2()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.ATTENTION);
            p.attention_param.axis = 2;
            p.attention_param.dim = 1;
            p.attention_param.weight_filler = new FillerParameter("constant", 1);
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            m_log.CHECK(layer.type == LayerParameter.LayerType.ATTENTION, "The layer type is incorrect!");

            float[] rgData = FillSmall(p.attention_param);

            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 0.01, 0.01);
            checker.CheckGradient(layer, BottomVec, TopVec);
        }
    }
}
