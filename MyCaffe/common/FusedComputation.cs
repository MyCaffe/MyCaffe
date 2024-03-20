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
    /// The FusedComputation class is used to perform fused computations using the CudaDnn class.
    /// </summary>
    /// <typeparam name="T">Specifies the base type of 'double' or 'float'.</typeparam>
    public class FusedComputation<T> : IDisposable
    {
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
        /// <param name="C">Specifies the resulting C matrix.</param>
        /// <param name="hA">Returns the handle to the A matrix tensor.</param>
        /// <param name="hB">Returns the handle to the B matrix tensor.</param>
        /// <param name="hC">Returns the handle to the C matrix tensor.</param>
        /// <param name="lWorkspaceSizeInBytes">Returns the size of the general workspace needed when executing.</param>
        /// <param name="hAworkspace">Returns a handle to the tensor A workspace GPU memory (or 0 if not used).</param>
        /// <param name="hBworkspace">Returns a handle to the tensor B workspace GPU memory (or 0 if not used).</param>
        /// <param name="nSharedIndex">Optionally, specifies the shared index used for the fused computation.</param>
        /// <returns>The handle to the operation is returned.</returns>
        public long CreateMatMulOp(bool bTransA, bool bTransB, Blob<T> A, Blob<T> B, Blob<T> C, out long hA, out long hB, out long hC, out long lWorkspaceSizeInBytes, out long hAworkspace, out long hBworkspace, long nSharedIndex = -1)
        {
            if (m_hCuda == 0)
                m_hCuda = m_cuda.CreateCuDNN();

            FUSEDCOMPUTE_DATA_TYPE dt = (typeof(T) == typeof(double)) ? FUSEDCOMPUTE_DATA_TYPE.DOUBLE : FUSEDCOMPUTE_DATA_TYPE.FLOAT;
            long hFc = m_cuda.CreateFusedCompute(nSharedIndex, m_hCuda, m_cuda.GetDeviceID(), dt, dt, dt, FUSEDCOMPUTE_PREBUILT_OP.NONE);
            hA = m_cuda.FusedCompAddTensor(hFc, dt, A.num, A.channels, A.height, A.width, bTransA, out hAworkspace);
            hB = m_cuda.FusedCompAddTensor(hFc, dt, B.num, B.channels, B.height, B.width, bTransB, out hBworkspace);
            FUSEDCOMPUTE_OP op = FUSEDCOMPUTE_OP.MATMUL;
            hC = m_cuda.FusedCompAddOp(hFc, op, dt, 0, hA, hB);
            lWorkspaceSizeInBytes = 0;

            try
            {
                lWorkspaceSizeInBytes = m_cuda.FusedCompBuild(hFc, FUSEDCOMP_HEUR_MODE.A, FUSEDCOMP_HEUR_MODE.FALLBACK);
            }
            catch (Exception)
            {
                m_rgUseFallback.Add(hFc, new Tuple<FUSEDCOMPUTE_PREBUILT_OP,bool,bool>(FUSEDCOMPUTE_PREBUILT_OP.MATMUL, bTransA, bTransB));
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
        /// <param name="A">Specifie the Blob with the A matrix data.</param>
        /// <param name="B">Specifie the Blob with the B matrix data.</param>
        /// <param name="C">Specifie the Blob with the C matrix data.</param>
        /// <param name="hWorkspace">Specifies the workspace data retunred by the OP creation method.</param>
        /// <param name="bADiff">Specifies whether to use the A diff or data.</param>
        /// <param name="bBDiff">Specifies whether to use the B diff or data.</param>
        /// <param name="bCDiff">Specifies whether to use the C diff or data.</param>
        public void RunOp(long hFc, long hA, long hB, long hC, Blob<T> A, Blob<T> B, Blob<T> C, long hWorkspace, long hAws, long hBws, bool bADiff = false, bool bBDiff = false, bool bCDiff = false)
        {
            if (m_rgUseFallback.ContainsKey(hFc))
            {
                Tuple<FUSEDCOMPUTE_PREBUILT_OP, bool, bool> op = m_rgUseFallback[hFc];

                switch (op.Item1)
                {
                    case FUSEDCOMPUTE_PREBUILT_OP.MATMUL:
                        C.MatMul(A, B, true, op.Item2, op.Item3);
                        return;
                }

                return;
            }   

            m_rgTensors.Clear();
            m_rgTensors.Add(hA);
            m_rgTensors.Add(hB);
            m_rgTensors.Add(hC);

            m_rgData.Clear();
            m_rgData.Add((bADiff) ? A.gpu_diff : A.gpu_data);
            m_rgData.Add((bBDiff) ? B.gpu_diff : B.gpu_data);
            m_rgData.Add((bCDiff) ? C.mutable_gpu_diff : C.mutable_gpu_data);

            m_rgDataWs.Clear();
            m_rgDataWs.Add(hAws);
            m_rgDataWs.Add(hBws);
            m_rgDataWs.Add(0);

            m_cuda.FusedCompExecute(hFc, hWorkspace, m_rgTensors, m_rgData, m_rgDataWs);
        }
    }
}
