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
using MyCaffe.fused_ops;

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

        [TestMethod]
        public void TestMatMul2()
        {
            FusedComputationTest test = new FusedComputationTest();

            try
            {
                foreach (IFusedComputationTest t in test.Tests)
                {
                    t.TestMatMul2();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMatMul3()
        {
            FusedComputationTest test = new FusedComputationTest();

            try
            {
                foreach (IFusedComputationTest t in test.Tests)
                {
                    t.TestMatMul3();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMatMul4()
        {
            FusedComputationTest test = new FusedComputationTest();

            try
            {
                foreach (IFusedComputationTest t in test.Tests)
                {
                    t.TestMatMul4();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMatMulGradPyTorchCompatible()
        {
            FusedComputationTest test = new FusedComputationTest();

            try
            {
                foreach (IFusedComputationTest t in test.Tests)
                {
                    t.TestMatMulGrad(false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMatMulGradNonPyTorchCompatible()
        {
            FusedComputationTest test = new FusedComputationTest();

            try
            {
                foreach (IFusedComputationTest t in test.Tests)
                {
                    t.TestMatMulGrad(true);
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
        void TestMatMul2();
        void TestMatMul3();
        void TestMatMul4();
        void TestMatMulGrad(bool bForceFallbackUse);
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

        private string getDataPath()
        {
            //return Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\llama\\test\\instr_llama\\";
            return "C:\\temp\\projects\\llama2\\llama2\\llama2_instruct\\test\\";
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
            long hWorkspace = 0;
            long lWorkspaceSize = 0;
            FusedComputation<T> fc = null;

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

                fc = new FusedComputation<T>(m_cuda, m_log);
                D = new Blob<T>(m_cuda, m_log);
                E = new Blob<T>(m_cuda, m_log);
                F = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                D.CopyFrom(A, false, true);
                E.CopyFrom(B, false, true);
                F.CopyFrom(C, false, true);

                long hD, hE, hF, hAws, hBws;
                hMatMul = fc.CreateMatMulOp(false, false, D, E, F, out hD, out hE, out hF, out lWorkspaceSize, out hAws, out hBws);

                if (lWorkspaceSize > 0)
                {
                    long lBaseSize = (typeof(T) == typeof(float)) ? 4 : 8;
                    long lSize = lWorkspaceSize / lBaseSize;
                    hWorkspace = m_cuda.AllocMemory(lSize);
                }

                fc.RunOp(hMatMul, hD, hE, hF, D.gpu_data, E.gpu_data, F.mutable_gpu_data, hWorkspace, hAws, hBws);
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

                if (fc != null)
                    fc.Dispose();
            }
        }

        public void TestMatMul2()
        {
            Blob<T> A = null;
            Blob<T> B = null;
            Blob<T> C = null;
            Blob<T> D = null;
            Blob<T> E = null;
            Blob<T> F = null;
            Blob<T> blobWork = null;
            long hMatMul = 0;
            long hWorkspace = 0;
            long lWorkspaceSize = 0;
            FusedComputation<T> fc = null;

            try
            {
                int nB = 1;
                int nM = 288;
                int nN = 288;
                int nK = 2;
                A = new Blob<T>(m_cuda, m_log, nB, 1, nM, nK);
                B = new Blob<T>(m_cuda, m_log, nB, 1, nK, nN);
                C = new Blob<T>(m_cuda, m_log);

                m_filler.Fill(A);
                m_filler.Fill(B);

                C.MatMul(A, B, true);

                fc = new FusedComputation<T>(m_cuda, m_log);
                D = new Blob<T>(m_cuda, m_log);
                E = new Blob<T>(m_cuda, m_log);
                F = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                D.CopyFrom(A, false, true);
                E.CopyFrom(B, false, true);
                F.CopyFrom(C, false, true);

                long hD, hE, hF, hAws, hBws;
                hMatMul = fc.CreateMatMulOp(false, false, D, E, F, out hD, out hE, out hF, out lWorkspaceSize, out hAws, out hBws);

                if (lWorkspaceSize > 0)
                {
                    long lBaseSize = (typeof(T) == typeof(float)) ? 4 : 8;
                    long lSize = lWorkspaceSize / lBaseSize;
                    hWorkspace = m_cuda.AllocMemory(lSize);
                }

                fc.RunOp(hMatMul, hD, hE, hF, D.gpu_data, E.gpu_data, F.mutable_gpu_data, hWorkspace, hAws, hBws);
                fc.FreeOp(hMatMul);

                m_log.CHECK(C.Compare(F, blobWork, false, 0.006), "The blobs are different!");
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

                if (fc != null)
                    fc.Dispose();
            }
        }

        public void TestMatMul3()
        {
            Blob<T> A = null;
            Blob<T> B = null;
            Blob<T> C = null;
            Blob<T> D = null;
            Blob<T> E = null;
            Blob<T> F = null;
            Blob<T> blobWork = null;
            Blob<T> blobVal = null;
            long hMatMul = 0;
            long hWorkspace = 0;
            long lWorkspaceSize = 0;
            FusedComputation<T> fc = null;
            string strPath = getDataPath();

            try
            {
                int nB = 64;
                int nC = 350;
                int nH = 1;
                int nW = 288;
                List<int> rgShape = new List<int>() { nB, nC, nH, nW };
                A = new Blob<T>(m_cuda, m_log, rgShape);
                rgShape = new List<int>() { nW, nW };
                B = new Blob<T>(m_cuda, m_log, rgShape);
                C = new Blob<T>(m_cuda, m_log);

                blobVal = new Blob<T>(m_cuda, m_log);

                m_filler.Fill(A);
                m_filler.Fill(B);

                A.LoadFromNumpy(strPath + "x.npy");
                blobVal.LoadFromNumpy(strPath + "wq.npy");
                B.CopyFromAndTransposeHeightWidth(blobVal);

                rgShape = new List<int>() { 1, 1, nB * nC, nW };
                A.Reshape(rgShape);
                rgShape = new List<int>() { 1, 1, nW, nW };
                B.Reshape(rgShape);

                C.MatMul(A, B, true);

                fc = new FusedComputation<T>(m_cuda, m_log);
                D = new Blob<T>(m_cuda, m_log);
                E = new Blob<T>(m_cuda, m_log);
                F = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                blobVal.LoadFromNumpy(strPath + "wq.npy");
                B.CopyFrom(blobVal, false, true);

                rgShape = new List<int>() { 1, 1, nW, nW };
                B.Reshape(rgShape);

                D.CopyFrom(A, false, true);
                E.CopyFrom(B, false, true);
                F.CopyFrom(C, false, true);

                long hD, hE, hF, hAws, hBws;
                hMatMul = fc.CreateMatMulOp(false, true, D, E, F, out hD, out hE, out hF, out lWorkspaceSize, out hAws, out hBws);

                if (lWorkspaceSize > 0)
                {
                    long lBaseSize = (typeof(T) == typeof(float)) ? 4 : 8;
                    long lSize = lWorkspaceSize / lBaseSize;
                    hWorkspace = m_cuda.AllocMemory(lSize);
                }

                fc.RunOp(hMatMul, hD, hE, hF, D.gpu_data, E.gpu_data, F.mutable_gpu_data, hWorkspace, hAws, hBws);
                fc.FreeOp(hMatMul);

                m_log.CHECK(C.Compare(F, blobWork, false, 0.01), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + "xq.npy");
                m_log.CHECK(blobVal.Compare(F, blobWork, false, (typeof(T) == typeof(float)) ? 1e-08 : 0.01), "The blobs are different!");
                m_log.CHECK(blobVal.Compare(C, blobWork, false, 0.01), "The blobs are different!");
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
                dispose(ref blobVal);

                if (hWorkspace != 0)
                    m_cuda.FreeMemory(hWorkspace);

                if (hMatMul != 0)
                    m_cuda.FreeFusedCompute(hMatMul);

                if (fc != null)
                    fc.Dispose();
            }
        }

        public void TestMatMul4()
        {
            Blob<T> A = null;
            Blob<T> B = null;
            Blob<T> C = null;
            Blob<T> D = null;
            Blob<T> E = null;
            Blob<T> F = null;
            Blob<T> blobWork = null;
            Blob<T> blobVal = null;
            MatMulOp<T> matmul = null;
            string strPath = getDataPath();

            try
            {
                int nB = 64;
                int nC = 350;
                int nH = 1;
                int nW = 288;
                List<int> rgShape = new List<int>() { nB, nC, nH, nW };
                A = new Blob<T>(m_cuda, m_log, rgShape);
                rgShape = new List<int>() { nW, nW };
                B = new Blob<T>(m_cuda, m_log, rgShape);
                C = new Blob<T>(m_cuda, m_log);

                blobVal = new Blob<T>(m_cuda, m_log);

                m_filler.Fill(A);
                m_filler.Fill(B);

                A.LoadFromNumpy(strPath + "x.npy");
                blobVal.LoadFromNumpy(strPath + "wq.npy");
                B.CopyFromAndTransposeHeightWidth(blobVal);

                rgShape = new List<int>() { 1, 1, nB * nC, nW };
                A.Reshape(rgShape);
                rgShape = new List<int>() { 1, 1, nW, nW };
                B.Reshape(rgShape);

                C.MatMul(A, B, true);

                D = new Blob<T>(m_cuda, m_log);
                E = new Blob<T>(m_cuda, m_log);
                F = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                blobVal.LoadFromNumpy(strPath + "wq.npy");
                B.CopyFrom(blobVal, false, true);

                rgShape = new List<int>() { 1, 1, nW, nW };
                B.Reshape(rgShape);

                D.CopyFrom(A, false, true);
                E.CopyFrom(B, false, true);

                matmul = new MatMulOp<T>(m_cuda, m_log, 2);
                matmul.Create(D, E, F, false, true);
                matmul.Run(D.gpu_data, E.gpu_data, F.mutable_gpu_data);

                m_log.CHECK(C.Compare(F, blobWork, false, 0.01), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + "xq.npy");
                m_log.CHECK(blobVal.Compare(F, blobWork, false, (typeof(T) == typeof(float)) ? 1e-08 : 0.01), "The blobs are different!");
                m_log.CHECK(blobVal.Compare(C, blobWork, false, 0.01), "The blobs are different!");
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
                dispose(ref blobVal);

                if (matmul != null)
                    matmul.Dispose();
            }
        }

        public void TestMatMulGrad(bool bForceFallbackUse)
        {
            Blob<T> x = null;
            Blob<T> x1 = null;
            Blob<T> x2 = null;
            Blob<T> x3 = null;
            Blob<T> wq = null;
            Blob<T> wk = null;
            Blob<T> wv = null;
            Blob<T> xq = null;
            Blob<T> xk = null;
            Blob<T> xv = null;
            Blob<T> blobWork = null;
            Blob<T> blobVal = null;
            MatMulOp<T> matmulQ = null;
            MatMulOp<T> matmulK = null;
            MatMulOp<T> matmulV = null;
            MatMulGradOp<T> matmulGrad = null;
            string strPath = getDataPath();

            try
            {
                blobWork = new Blob<T>(m_cuda, m_log);
                blobVal = new Blob<T>(m_cuda, m_log);

                int nB = 64;
                int nC = 350;
                int nH = 1;
                int nW = 288;
                List<int> rgShape = new List<int>() { nB, nC, nH, nW };
                x = new Blob<T>(m_cuda, m_log, rgShape);
                x1 = new Blob<T>(m_cuda, m_log, rgShape);
                x2 = new Blob<T>(m_cuda, m_log, rgShape);
                x3 = new Blob<T>(m_cuda, m_log, rgShape);

                rgShape = new List<int>() { nW, nW };
                wq = new Blob<T>(m_cuda, m_log, rgShape);
                wk = new Blob<T>(m_cuda, m_log, rgShape);
                wv = new Blob<T>(m_cuda, m_log, rgShape);

                xq = new Blob<T>(m_cuda, m_log);
                xk = new Blob<T>(m_cuda, m_log);
                xv = new Blob<T>(m_cuda, m_log);

                x.LoadFromNumpy(strPath + "att.x.npy");

                x1.LoadFromNumpy(strPath + "att.x1.npy");
                x2.LoadFromNumpy(strPath + "att.x2.npy");
                x3.LoadFromNumpy(strPath + "att.x3.npy");

                wq.LoadFromNumpy(strPath + "wq.npy");
                wk.LoadFromNumpy(strPath + "wk.npy");
                wv.LoadFromNumpy(strPath + "wv.npy");

                int nAxis = 2;

                x1.Unsqueeze(nAxis);
                x2.Unsqueeze(nAxis);
                x3.Unsqueeze(nAxis);

                matmulQ = new MatMulOp<T>(m_cuda, m_log, nAxis, bForceFallbackUse);
                matmulQ.Create(x1, wq, xq, false, true);
                matmulQ.Run(x.gpu_data, wq.gpu_data, xq.mutable_gpu_data);

                matmulK = new MatMulOp<T>(m_cuda, m_log, nAxis, bForceFallbackUse);
                matmulK.Create(x2, wk, xk, false, true);
                matmulK.Run(x.gpu_data, wk.gpu_data, xk.mutable_gpu_data);

                matmulV = new MatMulOp<T>(m_cuda, m_log, nAxis, bForceFallbackUse);
                matmulV.Create(x3, wv, xv, false, true);
                matmulV.Run(x.gpu_data, wv.gpu_data, xv.mutable_gpu_data);

                xq.LoadFromNumpy(strPath + "xq.grad.npy", true);
                xk.LoadFromNumpy(strPath + "xk.grad.npy", true);
                xv.LoadFromNumpy(strPath + "xv.grad.npy", true);

                xq.Unsqueeze(nAxis);
                xk.Unsqueeze(nAxis);
                xv.Unsqueeze(nAxis);

                matmulGrad = new MatMulGradOp<T>(m_cuda, m_log, nAxis, bForceFallbackUse);
                matmulGrad.Create(x1, wq, xq, false, (matmulQ.BwsHandle == 0) ? true : false);
                matmulGrad.Run(x1, wq, xq, 0, matmulQ.BwsHandle);
                matmulGrad.Run(x2, wk, xk, 0, matmulK.BwsHandle);
                matmulGrad.Run(x3, wv, xv, 0, matmulV.BwsHandle);

                x.CopyFrom(x1, true, true);
                m_cuda.add(x.count(), x.gpu_diff, x2.gpu_diff, x.mutable_gpu_diff);
                m_cuda.add(x.count(), x.gpu_diff, x3.gpu_diff, x.mutable_gpu_diff);

                xq.Squeeze(nAxis);
                xk.Squeeze(nAxis);
                xv.Squeeze(nAxis);

                blobVal.LoadFromNumpy(strPath + "att.x1.grad.npy", true);
                m_log.CHECK(blobVal.Compare(x1, blobWork, true, (typeof(T) == typeof(double) || bForceFallbackUse) ? 4e-07 : 1e-10), "The blobs are different!");
                blobVal.LoadFromNumpy(strPath + "att.x2.grad.npy", true);
                m_log.CHECK(blobVal.Compare(x2, blobWork, true, (typeof(T) == typeof(double) || bForceFallbackUse) ? 4e-07 : 1e-10), "The blobs are different!");
                blobVal.LoadFromNumpy(strPath + "att.x3.grad.npy", true);
                m_log.CHECK(blobVal.Compare(x3, blobWork, true, (typeof(T) == typeof(double) || bForceFallbackUse) ? 4e-07 : 1e-10), "The blobs are different!");
                blobVal.LoadFromNumpy(strPath + "att.x.grad.npy", true);
                m_log.CHECK(blobVal.Compare(x, blobWork, true, (typeof(T) == typeof(double) || bForceFallbackUse) ? 4e-07 : 1e-10), "The blobs are different!");
            }
            finally
            {
                dispose(ref x);
                dispose(ref x1);
                dispose(ref x2);
                dispose(ref x3);
                dispose(ref wq);
                dispose(ref wk);
                dispose(ref wv);
                dispose(ref xq);
                dispose(ref xk);
                dispose(ref xv);
                dispose(ref blobWork);
                dispose(ref blobVal);

                if (matmulQ != null)
                    matmulQ.Dispose();
                if (matmulK != null)
                    matmulK.Dispose();
                if (matmulV != null)
                    matmulV.Dispose();

                if (matmulGrad != null)
                    matmulGrad.Dispose();
            }
        }
    }
}
