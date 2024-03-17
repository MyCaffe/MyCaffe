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
using MyCaffe.layers.beta;

namespace MyCaffe.test
{
    [TestClass]
    public class TestFusedComputations
    {
        [TestMethod]
        public void TestMatMul()
        {
            FusedComputationTest test = new FusedComputationTest();

            try
            {
                foreach (IFusedComputationTest t in test.Tests)
                {
                    t.TestMatMul();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface IFusedComputationTest : ITest
    {
        void TestMatMul();
    }

    class FusedComputationTest : TestBase
    {
        public FusedComputationTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("FusedComputation Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new FusedComputationTest<double>(strName, nDeviceID, engine);
            else
                return new FusedComputationTest<float>(strName, nDeviceID, engine);
        }
    }

    class FusedComputationTest<T> : TestEx<T>, IFusedComputationTest
    {
        CryptoRandom m_random = new CryptoRandom();

        public FusedComputationTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
        }

        public void TestMatMul()
        {
            Blob<T> A = null;
            Blob<T> B = null;
            Blob<T> C = null;
            Blob<T> D = null;
            Blob<T> E = null;
            Blob<T> F = null;
            Blob<T> blobWork = null;
            long hMatMul = 0;
            long hCuda = 0;
            long hWorkspace = 0;
            long lWorkspaceSize = 0;

            try
            {
                int nB = 16;
                int nM = 32;  
                int nN = 64;  
                int nK = 128; 
                A = new Blob<T>(m_cuda, m_log, nB, 1, nM, nK);
                B = new Blob<T>(m_cuda, m_log, nB, 1, nK, nN);
                C = new Blob<T>(m_cuda, m_log);

                m_filler.Fill(A);
                m_filler.Fill(B);

                C.MatMul(A, B, true);

                FusedComputation<T> fc = new FusedComputation<T>(m_cuda, m_log);
                D = new Blob<T>(m_cuda, m_log);
                E = new Blob<T>(m_cuda, m_log);
                F = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                D.CopyFrom(A, false, true);
                E.CopyFrom(B, false, true);
                F.CopyFrom(C, false, true);

                hCuda = m_cuda.CreateCuDNN();
                long hD, hE, hF;
                hMatMul = fc.CreateMatMulOp(hCuda, D, E, F, out hD, out hE, out hF, out lWorkspaceSize);

                if (lWorkspaceSize > 0)
                {
                    long lBaseSize = (typeof(T) == typeof(float)) ? 4 : 8;
                    long lSize = lWorkspaceSize / lBaseSize;
                    hWorkspace = m_cuda.AllocMemory(lSize);
                }

                fc.RunOp(hMatMul, hD, hE, hF, D, E, F, hWorkspace);
                fc.FreeOp(hMatMul);

                m_log.CHECK(C.Compare(F, blobWork, false, 0.1), "The blobs are different!");

            }
            finally
            {
                dispose(ref A);
                dispose(ref B);
                dispose(ref C);
                dispose(ref D);
                dispose(ref E);
                dispose(ref F);
                dispose(ref blobWork);

                if (hWorkspace != 0)
                    m_cuda.FreeMemory(hWorkspace);

                if (hMatMul != 0)
                    m_cuda.FreeFusedCompute(hMatMul);

                if (hCuda != 0)
                    m_cuda.FreeCuDNN(hCuda);
            }
        }
    }
}
