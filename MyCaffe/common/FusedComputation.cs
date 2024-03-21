using MyCaffe.basecode;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.common
{
    /// <summary>
    /// The MatMulGradOp class is used to perform the gradient of the matrix multiplication operation.
    /// </summary>
    /// <typeparam name="T">Specifies the base data type of 'double' or 'float'</typeparam>
    public class MatMulGradOp<T> : IDisposable
    {
        MatMulOp<T> m_op1 = null;
        MatMulOp<T> m_op2 = null;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="cuda">Specifiees the connection to Cuda.</param>
        /// <param name="log">Specifies the output log.</param>
        /// <param name="bForceFallbackUse">Optionally, specifies to force using the fallback.</param>
        public MatMulGradOp(CudaDnn<T> cuda, Log log, bool bForceFallbackUse = false)
        {
            m_op1 = new MatMulOp<T>(cuda, log, bForceFallbackUse);
            m_op2 = new MatMulOp<T>(cuda, log, bForceFallbackUse); 
        }

        /// <summary>
        /// Release all resources used.
        /// </summary>
        public void Dispose()
        {
            if (m_op1 != null)
            {
                m_op1.Dispose();
                m_op1 = null;
            }

            if (m_op2 != null)
            {
                m_op2.Dispose();
                m_op2 = null;
            }
        }

        /// <summary>
        /// Create the gradient of the matrix multiplication operation.
        /// </summary>
        /// <param name="A">Specifies the A blob that will have its diff updated.</param>
        /// <param name="B">Specifies the B blob that will have its </param>
        /// <param name="C">Specifies the C blob containing the originating diff to be propagated.</param>
        /// <param name="bTransA">Specifies that the A blob was transposed during the forward pass.</param>
        /// <param name="bTransB">Specifies that the B blob was transposed during the forward pass.</param>
        /// <param name="nSharedIndex1">Specifies the shared index for the first MatMul op, or -1 to ignore.</param>
        /// <param name="nSharedIndex2">Specifies the shared index for the second MatMul op, or -1 to ignore.</param>
        public void Create(Blob<T> A, Blob<T> B, Blob<T> C, bool bTransA, bool bTransB,  int nSharedIndex1 = -1, int nSharedIndex2 = -1)
        {
            // No need to transpose B since we will use the data from the already tansposed B from the forward pass, passed in as hBws.
            m_op1.Create(C, B, A, false, !bTransB, nSharedIndex1);
            m_op2.Create(A, C, B, !bTransA, false, nSharedIndex2);
        }

        /// <summary>
        /// Run the gradient of the matrix multiplication operation.
        /// </summary>
        /// <param name="A">Specifies the A blob that will have its diff updated.</param>
        /// <param name="B">Specifies the B blob that will have its </param>
        /// <param name="C">Specifies the C blob containing the originating diff to be propagated.</param>
        /// <param name="hAws">Optionally, specifies the A workspace data that contains the transposed A from a forward pass MatMul.  A value of 0 ignores this parameter.</param>
        /// <param name="hBws">Optionally, specifies the B workspace data that contains the transposed B from a forward pass MatMul.  A value of 0 ignores this parameter.</param>
        public void Run(Blob<T> A, Blob<T> B, Blob<T> C, long hAws = 0, long hBws = 0)
        {
            if (hBws == 0)
                hBws = B.gpu_data;

            m_op1.Run(C.gpu_diff, hBws, A.mutable_gpu_diff);

            if (hAws == 0)
                hAws = A.gpu_data;

            m_op2.Run(hAws, C.gpu_diff, B.mutable_gpu_diff);
        }
    }

    /// <summary>
    /// The MatMulOp class is used to perform the matrix multiplication operation.
    /// </summary>
    /// <typeparam name="T">Specifies the base data type of 'double' or 'float'</typeparam>
    public class MatMulOp<T> : IDisposable
    {
        CudaDnn<T> m_cuda = null;
        FusedComputation<T> m_fc = null;
        long m_hMatMul = 0;
        long m_hA = 0;
        long m_hB = 0;
        long m_hC = 0;
        long m_hWorkspace = 0;
        long m_hAws = 0;
        long m_hBws = 0;
        bool m_bForceFallbackUse = false;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="cuda">Specifies the connection to Cuda.</param>
        /// <param name="log">Specifies the output log.</param>
        /// <param name="bForceFallbackUse">Optionally, force using the fallback.</param>
        public MatMulOp(CudaDnn<T> cuda, Log log, bool bForceFallbackUse = false)
        {
            m_bForceFallbackUse = bForceFallbackUse;
            m_cuda = cuda;
            m_fc = new FusedComputation<T>(cuda, log);
        }

        /// <summary>
        /// Release all resources used.
        /// </summary>
        public void Dispose()
        {
            CleanUp();
        }

        /// <summary>
        /// returns the handle to the A tensor workspace data which contains the transposed A tensor when used.
        /// </summary>
        public long AwsHandle
        {
            get { return m_hAws; }
        }

        /// <summary>
        /// returns the handle to the B tensor workspace data which contains the transposed B tensor when used.
        /// </summary>
        public long BwsHandle
        {
            get { return m_hBws; }
        }

        /// <summary>
        /// Clean up the resources used.
        /// </summary>
        public void CleanUp()
        {
            if (m_hWorkspace != 0)
            {
                m_cuda.FreeMemory(m_hWorkspace);
                m_hWorkspace = 0;
            }

            if (m_hAws != 0)
            {
                m_cuda.FreeMemory(m_hAws);
                m_hAws = 0;
            }

            if (m_hBws != 0)
            {
                m_cuda.FreeMemory(m_hBws);
                m_hBws = 0;
            }

            if (m_hMatMul != 0)
            {
                m_fc.FreeOp(m_hMatMul);
                m_hMatMul = 0;
            }

            if (m_fc != null)
            {
                m_fc.Dispose();
                m_fc = null;
            }
        }

        /// <summary>
        /// Create the matrix multiplication operation.
        /// </summary>
        /// <param name="A">Specifies the input A tensor.</param>
        /// <param name="B">Specifies the input B tensor.</param>
        /// <param name="C">Specifies the output C tensor.</param>
        /// <param name="bTransA"></param>
        /// <param name="bTransB"></param>
        /// <param name="nSharedIndex"></param>
        /// <exception cref="Exception"></exception>
        public void Create(Blob<T> A, Blob<T> B, Blob<T> C, bool bTransA, bool bTransB, int nSharedIndex = -1)
        {
            if (bTransA && bTransB)
                throw new Exception("Either A or B must be transposed, not both.");

            long lWorkspaceSizeInBytes;
            long hAws;
            long hBws;

            try
            {
                m_hMatMul = m_fc.CreateMatMulOp(bTransA, bTransB, A, B, C, out m_hA, out m_hB, out m_hC, out lWorkspaceSizeInBytes, out hAws, out hBws, nSharedIndex, m_bForceFallbackUse);
                
                if (lWorkspaceSizeInBytes > 0)
                    m_hWorkspace = m_cuda.AllocMemory(lWorkspaceSizeInBytes);
                else
                    m_hWorkspace = 0;

                m_hAws = hAws;
                m_hBws = hBws;
            }
            catch (Exception excpt)
            {
                CleanUp();
                throw new Exception("The fused computation failed to create the MatMulOp.", excpt);
            }
        }

        /// <summary>
        /// Run the matrix multiplication operation.
        /// </summary>
        /// <param name="hAd">Specifies the GPU handle to the A matrix data.</param>
        /// <param name="hBd">Specifies the GPU handle to the B matrix data.</param>
        /// <param name="hCd">Specifies the GPU handle to the C matrix data.</param>
        /// <exception cref="Exception">An exception is thrown on failure.</exception>
        public void Run(long hAd, long hBd, long hCd)
        {
            try
            {
                m_fc.RunOp(m_hMatMul, m_hA, m_hB, m_hC, hAd, hBd, hCd, m_hWorkspace, m_hAws, m_hBws);
            }
            catch (Exception excpt)
            {
                CleanUp();
                throw new Exception("The fused computation failed to run the MatMulOp.", excpt);
            }
        }
    }

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

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="cuda">Specifies the connection to the low-level CUDA implementations.</param>
        /// <param name="log">Specifies the output log.</param>
        public FusedComputation(CudaDnn<T> cuda, Log log)
        {
            m_cuda = cuda;
            m_log = log;
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
        /// <param name="lWorkspaceSizeInBytes">Returns the size of the general workspace needed when executing.</param>
        /// <param name="hAws">Returns a handle to the tensor A workspace GPU memory (or 0 if not used).</param>
        /// <param name="hBws">Returns a handle to the tensor B workspace GPU memory (or 0 if not used).</param>
        /// <param name="nSharedIndex">Optionally, specifies the shared index used for the fused computation.</param>
        /// <param name="bForceFallbackUse">Optionally, force using the fall-back.</param>
        /// <returns>The handle to the operation is returned.</returns>
        public long CreateMatMulOp(bool bTransA, bool bTransB, Blob<T> A, Blob<T> B, Blob<T> C, out long hA, out long hB, out long hC, out long lWorkspaceSizeInBytes, out long hAws, out long hBws, long nSharedIndex = -1, bool bForceFallbackUse = false)
        {
            lWorkspaceSizeInBytes = 0;
            hAws = 0;
            hBws = 0;

            if (m_hCuda == 0)
                m_hCuda = m_cuda.CreateCuDNN();

            if (A.num != B.num)
                throw new Exception("The 'num' in A and B must be the same.");
            if (A.channels != B.channels)
                throw new Exception("The 'channels' in A and B must be the same.");

            if (bTransA && A.height != B.height)
                throw new Exception("The 'height' in A must be the same as the 'height' in B.");
            else if (bTransB && A.width != B.width)
                throw new Exception("The 'height' in A must be the same as the 'height' in B.");
            else if (!bTransA && !bTransB && A.width != B.height)
                throw new Exception("The 'width' in A must be the same as the 'height' in B.");

            m_nOuter = A.num * A.channels;
            m_nM = (bTransA) ? A.width : A.height;
            m_nN = (bTransB) ? B.height : B.width;
            m_nK = (bTransA) ? A.height : A.width;

            C.Reshape(A.num, A.channels, m_nM, m_nN);

            FUSEDCOMPUTE_DATA_TYPE dt = (typeof(T) == typeof(double)) ? FUSEDCOMPUTE_DATA_TYPE.DOUBLE : FUSEDCOMPUTE_DATA_TYPE.FLOAT;
            long hFc = m_cuda.CreateFusedCompute(nSharedIndex, m_hCuda, m_cuda.GetDeviceID(), dt, dt, dt, FUSEDCOMPUTE_PREBUILT_OP.NONE);
            hA = m_cuda.FusedCompAddTensor(hFc, dt, A.num, A.channels, m_nM, m_nK, bTransA, out hAws);
            hB = m_cuda.FusedCompAddTensor(hFc, dt, B.num, B.channels, m_nK, m_nN, bTransB, out hBws);
            FUSEDCOMPUTE_OP op = FUSEDCOMPUTE_OP.MATMUL;
            hC = m_cuda.FusedCompAddOp(hFc, op, dt, 0, hA, hB);

            try
            {
                if (!bForceFallbackUse)
                    lWorkspaceSizeInBytes = m_cuda.FusedCompBuild(hFc, FUSEDCOMP_HEUR_MODE.A, FUSEDCOMP_HEUR_MODE.FALLBACK);
            }
            catch (Exception)
            {
                bForceFallbackUse = true;
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

            m_cuda.FusedCompExecute(hFc, hWorkspace, m_rgTensors, m_rgData, m_rgDataWs);
        }
    }
}
