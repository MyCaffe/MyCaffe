using MyCaffe.basecode;
using MyCaffe.common;
using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.fused_ops
{
    /// <summary>
    /// The FusedComputation class is used to perform fused computations using the CudaDnn class.
    /// </summary>
    /// <typeparam name="T">Specifies the base type of 'double' or 'float'.</typeparam>
    public class FusedComputation<T> : IDisposable
    {
        int m_nM;
        int m_nN;
        int m_nK;
        int m_nOuter;
        CudaDnn<T> m_cuda;
        long m_hCuda = 0;
        Log m_log;
        List<long> m_rgTensors = new List<long>(32);
        List<long> m_rgData = new List<long>(32);
        List<long> m_rgDataWs = new List<long>(32);
        Dictionary<long, Tuple<FUSEDCOMPUTE_PREBUILT_OP, bool, bool>> m_rgUseFallback = new Dictionary<long, Tuple<FUSEDCOMPUTE_PREBUILT_OP, bool, bool>>();
        long m_lBaseSize;
        List<int> m_rgShape = new List<int>(4);
        int m_nLocalID = -1;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="cuda">Specifies the connection to the low-level CUDA implementations.</param>
        /// <param name="log">Specifies the output log.</param>
        /// <param name="nLocalID">Optionally, specifies the local ID used for graph caching.</param>
        public FusedComputation(CudaDnn<T> cuda, Log log, int nLocalID = -1)
        {
            m_lBaseSize = (typeof(T) == typeof(double)) ? 8 : 4;    
            m_cuda = cuda;
            m_log = log;
            m_nLocalID = nLocalID;
        }

        /// <summary>
        /// Release all resources used.
        /// </summary>
        public void Dispose()
        {
            if (m_hCuda != 0)
            {
                m_cuda.FreeCuDNN(m_hCuda);
                m_hCuda = 0;
            }
        }

        /// <summary>
        /// Returns the M dimension.
        /// </summary>
        public int M
        {
            get { return m_nM; }
        }

        /// <summary>
        /// Returns the N dimension.
        /// </summary>
        public int N
        {
            get { return m_nN; }
        }

        /// <summary>
        /// Returns the K dimension.
        /// </summary>
        public int K
        {
            get { return m_nK; }
        }

        /// <summary>
        /// Returns the Outer dimension.
        /// </summary>
        public int Outer
        {
            get { return m_nOuter; }
        }

        /// <summary>
        /// Create a fused computation using the specified pre-built operation.
        /// </summary>
        /// <param name="bTransA">Specifies to transpose A first.</param>
        /// <param name="bTransB">Specifies to transpose B first.</param>
        /// <param name="A">Specifies the A matrix.</param>
        /// <param name="B">Specifies the B matrix.</param>
        /// <param name="C">Specifies the resulting C matrix which is reshaped to the shape N,C,Ha,Wb.</param>
        /// <param name="hA">Returns the handle to the A matrix tensor.</param>
        /// <param name="hB">Returns the handle to the B matrix tensor.</param>
        /// <param name="hC">Returns the handle to the C matrix tensor.</param>
        /// <param name="lWorkspaceSizeInItems">Returns the size of the general workspace needed when executing.</param>
        /// <param name="hAws">Returns a handle to the tensor A workspace GPU memory (or 0 if not used).</param>
        /// <param name="hBws">Returns a handle to the tensor B workspace GPU memory (or 0 if not used).</param>
        /// <param name="nSharedIndex">Optionally, specifies the shared index used for the fused computation.</param>
        /// <param name="bForceFallbackUse">Optionally, force using the fall-back.</param>
        /// <returns>The handle to the operation is returned.</returns>
        public long CreateMatMulOp(bool bTransA, bool bTransB, Blob<T> A, Blob<T> B, Blob<T> C, out long hA, out long hB, out long hC, out long lWorkspaceSizeInItems, out long hAws, out long hBws, long nSharedIndex = -1, bool bForceFallbackUse = false)
        {
            lWorkspaceSizeInItems = 0;
            hAws = 0;
            hBws = 0;

            // Fused computations are not supported for double precision.
            if (typeof(T) == typeof(double))
                bForceFallbackUse = true;

            if (m_hCuda == 0)
                m_hCuda = m_cuda.CreateCuDNN();

            if (A.num_axes != 2 && A.num_axes != 4)
                throw new Exception("A must have 2 or 4 axes.");
            if (B.num_axes != 2 && B.num_axes != 4)
                throw new Exception("B must have 2 or 4 axes.");

            m_rgShape.Clear();

            int nB = 1;
            int nC = 1;

            if (A.num_axes == 4)
            {
                nB = A.num;
                nC = A.channels;
                m_rgShape.Add(nB);
                m_rgShape.Add(nC);
            }

            int nHa = (A.num_axes == 2) ? A.num : A.height;
            int nWa = (A.num_axes == 2) ? A.channels : A.width;
            int nHb = (B.num_axes == 2) ? B.num : B.height;
            int nWb = (B.num_axes == 2) ? (bTransB) ? B.num : B.channels : (bTransB) ? B.height : B.width;

            if (bTransA && nHa != nHb)
                throw new Exception("The 'height' in A must be the same as the 'height' in B.");
            else if (bTransB && nWa != nWb)
                throw new Exception("The 'width' in A must be the same as the 'width' in B.");
            else if (!bTransA && !bTransB && nWa != nHb)
                throw new Exception("The 'width' in A must be the same as the 'height' in B.");

            m_nOuter = nB * nC;
            m_nM = (bTransA) ? nWa : nHa;
            m_nN = (bTransB) ? nHb : nWb;
            m_nK = (bTransA) ? nHa : nWa;

            m_rgShape.Add(m_nM);
            m_rgShape.Add(m_nN);
            if (!C.CompareShape(m_rgShape))
                C.Reshape(m_rgShape);

            FUSEDCOMPUTE_DATA_TYPE dt = (typeof(T) == typeof(double)) ? FUSEDCOMPUTE_DATA_TYPE.DOUBLE : FUSEDCOMPUTE_DATA_TYPE.FLOAT;
            long hFc = m_cuda.CreateFusedCompute(nSharedIndex, m_hCuda, m_cuda.GetDeviceID(), dt, dt, dt, FUSEDCOMPUTE_PREBUILT_OP.NONE);
            hAws = 0;
            hBws = 0;
            lWorkspaceSizeInItems = 0;
            hA = 0;
            hB = 0;
            hC = 0;

            if (!bForceFallbackUse)
            {
                try
                {
                    hA = m_cuda.FusedCompAddTensor(hFc, dt, nB, nC, m_nM, m_nK, bTransA, out hAws);
                    hB = m_cuda.FusedCompAddTensor(hFc, dt, nB, nC, m_nK, m_nN, bTransB, out hBws);
                    FUSEDCOMPUTE_OP op = FUSEDCOMPUTE_OP.MATMUL;
                    hC = m_cuda.FusedCompAddOp(hFc, op, dt, 0, hA, hB);

                    long lWorkspaceSizeInBytes = m_cuda.FusedCompBuild(hFc, FUSEDCOMP_HEUR_MODE.A, FUSEDCOMP_HEUR_MODE.FALLBACK, m_nLocalID);
                    lWorkspaceSizeInItems = (lWorkspaceSizeInBytes == 0) ? 0 : (lWorkspaceSizeInBytes / m_lBaseSize) + 1;
                }
                catch (Exception)
                {
                    bForceFallbackUse = true;
                }
            }

            if (bForceFallbackUse)
            {
                m_rgUseFallback.Add(hFc, new Tuple<FUSEDCOMPUTE_PREBUILT_OP, bool, bool>(FUSEDCOMPUTE_PREBUILT_OP.MATMUL, bTransA, bTransB));

                if (hAws != 0)
                {
                    m_cuda.FreeMemory(hAws);
                    hAws = 0;
                }

                if (hBws != 0)
                {
                    m_cuda.FreeMemory(hBws);
                    hBws = 0;
                }
            }

            return hFc;
        }

        /// <summary>
        /// Reshape a fused computation using the specified pre-built operation.
        /// </summary>
        /// <param name="hFc">Specifies the previously created Fc.</param>
        /// <param name="bTransA">Specifies to transpose A first.</param>
        /// <param name="bTransB">Specifies to transpose B first.</param>
        /// <param name="A">Specifies the A matrix.</param>
        /// <param name="B">Specifies the B matrix.</param>
        /// <param name="C">Specifies the resulting C matrix which is reshaped to the shape N,C,Ha,Wb.</param>
        /// <param name="hA">Returns the handle to the A matrix tensor.</param>
        /// <param name="hB">Returns the handle to the B matrix tensor.</param>
        /// <param name="hC">Returns the handle to the C matrix tensor.</param>
        /// <param name="lWorkspaceSizeInItems">Returns the size of the general workspace needed when executing.</param>
        /// <param name="hAws">Returns a handle to the tensor A workspace GPU memory (or 0 if not used).</param>
        /// <param name="hBws">Returns a handle to the tensor B workspace GPU memory (or 0 if not used).</param>
        /// <param name="nSharedIndex">Optionally, specifies the shared index used for the fused computation.</param>
        /// <param name="bForceFallbackUse">Optionally, force using the fall-back.</param>
        /// <returns>The handle to the operation is returned, or -1 if no Reshape occurs.</returns>
        public long ReshapeMatMulOp(long hFc, bool bTransA, bool bTransB, Blob<T> A, Blob<T> B, Blob<T> C, ref long hA, ref long hB, ref long hC, out long lWorkspaceSizeInItems, ref long hAws, ref long hBws, long nSharedIndex = -1, bool bForceFallbackUse = false)
        {
            lWorkspaceSizeInItems = 0;

            if (bForceFallbackUse)
                return -1;

            if (m_hCuda == 0)
                m_hCuda = m_cuda.CreateCuDNN();

            int nB = (A.num_axes == 2) ? 1 : A.num;
            int nC = (A.num_axes == 2) ? 1 : A.channels;
            int nHa = (A.num_axes == 2) ? A.num : A.height;
            int nWa = (A.num_axes == 2) ? A.channels : A.width;
            int nHb = (B.num_axes == 2) ? B.num : B.height;
            int nWb = (B.num_axes == 2) ? (bTransB) ? B.num : B.channels : (bTransB) ? B.height : B.width;

            if (bTransA && nHa != nHb)
                throw new Exception("The 'height' in A must be the same as the 'height' in B.");
            else if (bTransB && nWa != nWb)
                throw new Exception("The 'height' in A must be the same as the 'height' in B.");
            else if (!bTransA && !bTransB && nWa != nHb)
                throw new Exception("The 'width' in A must be the same as the 'height' in B.");

            int nOuter = nB * nC;
            int nM = (bTransA) ? nWa : nHa;
            int nN = (bTransB) ? nHb : nWb;
            int nK = (bTransA) ? nHa : nWa;

            if (m_nOuter == nOuter && m_nM == nM && m_nN == nN && m_nK == nK)
                return -1;

            // Reshape the output C to the new shape.
            FreeOp(hFc);
            hFc = 0;

            if (hAws != 0)
            {
                m_cuda.FreeMemory(hAws);
                hAws = 0;
            }

            if (hBws != 0)
            {
                m_cuda.FreeMemory(hBws);
                hBws = 0;
            }

            return CreateMatMulOp(bTransA, bTransB, A, B, C, out hA, out hB, out hC, out lWorkspaceSizeInItems, out hAws, out hBws, nSharedIndex, bForceFallbackUse);
        }

        /// <summary>
        /// Free the fused computation created with on of the OP creation methods.
        /// </summary>
        /// <param name="hFc">Specifies a handle to the fused computation.</param>
        public void FreeOp(long hFc)
        {
            if (hFc != 0)
                m_cuda.FreeFusedCompute(hFc);
        }

        /// <summary>
        /// Run the fused computation created with on of the OP creation methods.
        /// </summary>
        /// <param name="hFc">Specifies the fused computation to use.</param>
        /// <param name="hA">Specifies a handle to the A matrix tensor.</param>
        /// <param name="hB">Specifies a handle to the B matrix tensor.</param>
        /// <param name="hC">Specifies a handle to the C matrix tensor.</param>
        /// <param name="hAd">Specifie the handle to the GPU data for the A matrix data.</param>
        /// <param name="hBd">Specifie the handle to the GPU data for the B matrix data.</param>
        /// <param name="hCd">Specifie the handle to the GPU data for the C matrix data.</param>
        /// <param name="hWorkspace">Specifies the workspace data retunred by the OP creation method.</param>
        /// <param name="hAws">Specifies the workspace for the A tensor.</param>
        /// <param name="hBws">Specifies the workspace for the B tensor.</param>
        public void RunOp(long hFc, long hA, long hB, long hC, long hAd, long hBd, long hCd, long hWorkspace, long hAws = 0, long hBws = 0)
        {
            if (m_rgUseFallback.ContainsKey(hFc))
            {
                Tuple<FUSEDCOMPUTE_PREBUILT_OP, bool, bool> op = m_rgUseFallback[hFc];

                switch (op.Item1)
                {
                    case FUSEDCOMPUTE_PREBUILT_OP.MATMUL:
                        m_cuda.matmul((uint)m_nOuter, m_nM, m_nN, m_nK, hAd, hBd, hCd, 1.0, op.Item2, op.Item3);
                        return;
                }

                return;
            }   

            m_rgTensors.Clear();
            m_rgTensors.Add(hA);
            m_rgTensors.Add(hB);
            m_rgTensors.Add(hC);

            m_rgData.Clear();
            m_rgData.Add(hAd);
            m_rgData.Add(hBd);
            m_rgData.Add(hCd);

            m_rgDataWs.Clear();
            m_rgDataWs.Add(hAws);
            m_rgDataWs.Add(hBws);
            m_rgDataWs.Add(0);

            m_cuda.FusedCompExecute(hFc, hWorkspace, m_rgTensors, m_rgData, m_rgDataWs, m_nLocalID);
        }
    }
}
