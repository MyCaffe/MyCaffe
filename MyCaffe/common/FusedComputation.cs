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
    public class FusedComputation<T>
    {
        CudaDnn<T> m_cuda;
        Log m_log;
        List<long> m_rgTensors = new List<long>(32);
        List<long> m_rgData = new List<long>(32);
        Dictionary<long, FUSEDCOMPUTE_PREBUILT_OP> m_rgUseFallback = new Dictionary<long, FUSEDCOMPUTE_PREBUILT_OP>();

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
        /// Create a fused computation using the specified pre-built operation.
        /// </summary>
        /// <param name="hCuda">Specifies a handle to the CuDnn created with CreateCuDnn.</param>
        /// <param name="A">Specifies the A matrix.</param>
        /// <param name="B">Specifies the B matrix.</param>
        /// <param name="C">Specifies the resulting C matrix.</param>
        /// <param name="hA">Returns the handle to the A matrix tensor.</param>
        /// <param name="hB">Returns the handle to the B matrix tensor.</param>
        /// <param name="hC">Returns the handle to the C matrix tensor.</param>
        /// <param name="lWorkspaceSizeInBytes">Returns the size of the workspace needed when executing.</param>
        /// <returns>The handle to the operation is returned.</returns>
        public long CreateMatMulOp(long hCuda, Blob<T> A, Blob<T> B, Blob<T> C, out long hA, out long hB, out long hC, out long lWorkspaceSizeInBytes)
        {
            FUSEDCOMPUTE_DATA_TYPE dt = (typeof(T) == typeof(double)) ? FUSEDCOMPUTE_DATA_TYPE.DOUBLE : FUSEDCOMPUTE_DATA_TYPE.FLOAT;
            long hFc = m_cuda.CreateFusedCompute(-1, hCuda, dt, dt, dt, FUSEDCOMPUTE_PREBUILT_OP.NONE);
            hA = m_cuda.FusedCompAddTensor(hFc, dt, A.num, A.channels, A.height, A.width);
            hB = m_cuda.FusedCompAddTensor(hFc, dt, B.num, B.channels, B.height, B.width);
            FUSEDCOMPUTE_OP op = FUSEDCOMPUTE_OP.MATMUL;
            hC = m_cuda.FusedCompAddOp(hFc, op, dt, 0, hA, hB);
            lWorkspaceSizeInBytes = 0;

            try
            {
                lWorkspaceSizeInBytes = m_cuda.FusedCompBuild(hFc, FUSEDCOMP_HEUR_MODE.A, FUSEDCOMP_HEUR_MODE.FALLBACK);
            }
            catch (Exception)
            {
                m_rgUseFallback.Add(hFc, FUSEDCOMPUTE_PREBUILT_OP.MATMUL);
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
        public void RunOp(long hFc, long hA, long hB, long hC, Blob<T> A, Blob<T> B, Blob<T> C, long hWorkspace, bool bADiff = false, bool bBDiff = false, bool bCDiff = false)
        {
            if (m_rgUseFallback.ContainsKey(hFc))
            {
                FUSEDCOMPUTE_PREBUILT_OP op = m_rgUseFallback[hFc];

                switch (op)
                {
                    case FUSEDCOMPUTE_PREBUILT_OP.MATMUL:
                        C.MatMul(A, B, true);
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

            m_cuda.FusedCompExecute(hFc, hWorkspace, m_rgTensors, m_rgData);
        }
    }
}
