using MyCaffe.basecode;
using MyCaffe.common;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.fused_ops
{
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
        long m_lWorkspaceInItems = 0;
        Dictionary<long, List<int>> m_rgOriginalShapes = new Dictionary<long, List<int>>();
        List<int> m_rgShape = new List<int>(4);
        int m_nAxis;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="cuda">Specifies the connection to Cuda.</param>
        /// <param name="log">Specifies the output log.</param>
        /// <param name="nAxis">Specifies the axis of the operation (default = 2).</param>
        /// <param name="bForceFallbackUse">Optionally, force using the fallback.</param>
        /// <param name="nLocalID">Optionally, specifies the localID used for graph caching.</param>
        public MatMulOp(CudaDnn<T> cuda, Log log, int nAxis = 2, bool bForceFallbackUse = false, int nLocalID = -1)
        {
            m_nAxis = nAxis;
            m_bForceFallbackUse = bForceFallbackUse;
            m_cuda = cuda;
            m_fc = new FusedComputation<T>(cuda, log, nLocalID);
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

        private void prep(Blob<T> b)
        {
            if (b.num_axes == 2 || b.num_axes == 0)
                return;

            if (!m_rgOriginalShapes.ContainsKey(b.gpu_data))
            {
                List<int> rgShape = Utility.Clone<int>(b.shape());
                m_rgOriginalShapes.Add(b.gpu_data, rgShape);
            }

            if (b.shape(0) == 1 && b.shape(1) == 1)
            {
                m_rgShape.Clear();
                m_rgShape.Add(b.shape(2));
                m_rgShape.Add(b.shape(3));
                b.Reshape(m_rgShape);
                return;
            }

            m_rgShape.Clear();
            m_rgShape.Add(1);
            m_rgShape.Add(1);

            for (int i = 0; i < m_nAxis; i++)
            {
                m_rgShape[0] *= b.shape(i);
            }

            for (int i = m_nAxis; i < b.num_axes; i++)
            {
                m_rgShape[1] *= b.shape(i);
            }

            b.Reshape(m_rgShape);
        }   

        private void unprep(Blob<T> blob)
        {
            if (m_rgOriginalShapes.ContainsKey(blob.gpu_data))
            {
                blob.Reshape(m_rgOriginalShapes[blob.gpu_data]);
                m_rgOriginalShapes.Remove(blob.gpu_data);
            }
        }

        private void prep(Blob<T> A, Blob<T> B, Blob<T> C, bool bTransA, bool bTransB)
        {
            if (A.num_axes == 3)
                A.Unsqueeze(m_nAxis);

            if (B.num_axes == 3)
                B.Unsqueeze(m_nAxis);

            if (A.num_axes != 2 && A.num_axes != 4)
                throw new Exception("The tensor A must have either 2 or 4 axes.");

            if (B.num_axes != 2 && B.num_axes != 4)
                throw new Exception("The tensor B must have either 2 or 4 axes.");

            if (C.num_axes == 0)
            {
                m_rgShape.Clear();

                int nIdxA = 0;
                if (A.num_axes == 4)
                {
                    m_rgShape.Add(A.shape(nIdxA));
                    nIdxA++;
                    m_rgShape.Add(A.shape(nIdxA));
                    nIdxA++;
                }

                int nIdxB = B.num_axes - 2;
                int nM = (bTransA) ? A.shape(nIdxA + 1) : A.shape(nIdxA);
                int nN = (bTransB) ? B.shape(nIdxB) : B.shape(nIdxB + 1);

                m_rgShape.Add(nM);
                m_rgShape.Add(nN);
                C.Reshape(m_rgShape);
            }

            prep(A);
            prep(B);
            prep(C);
        }

        private void unprep(Blob<T> A, Blob<T> B, Blob<T> C)
        {
            unprep(A);
            unprep(B);
            unprep(C);
        }

        /// <summary>
        /// Create the matrix multiplication operation.
        /// </summary>
        /// <param name="A">Specifies the input A tensor.</param>
        /// <param name="B">Specifies the input B tensor.</param>
        /// <param name="C">Specifies the output C tensor.</param>
        /// <param name="bTransA">Specifies to transpose the A tensor.</param>
        /// <param name="bTransB">Specifies to transpose the B tensor.</param>
        /// <param name="nSharedIndex">Specifies a shared index to share instances.</param>
        /// <exception cref="Exception">An exception is thrown when transposing more than one tensor.</exception>
        public void Create(Blob<T> A, Blob<T> B, Blob<T> C, bool bTransA, bool bTransB, int nSharedIndex = -1)
        {
            if (bTransA && bTransB)
                throw new Exception("Either A or B must be transposed, not both.");

            try
            {
                prep(A, B, C, bTransA, bTransB);

                long hAws;
                long hBws;
                m_hMatMul = m_fc.CreateMatMulOp(bTransA, bTransB, A, B, C, out m_hA, out m_hB, out m_hC, out m_lWorkspaceInItems, out hAws, out hBws, nSharedIndex, m_bForceFallbackUse);

                if (m_lWorkspaceInItems > 0)
                    m_hWorkspace = m_cuda.AllocMemory(m_lWorkspaceInItems);
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
            finally
            {
                unprep(A, B, C);
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
        public void Reshape(Blob<T> A, Blob<T> B, Blob<T> C, bool bTransA, bool bTransB, int nSharedIndex = -1)
        {
            if (bTransA && bTransB)
                throw new Exception("Either A or B must be transposed, not both.");

            long lWorkspaceInItems;
            long hAws = 0;
            long hBws = 0;

            try
            {
                prep(A, B, C, bTransA, bTransB);

                long hMatMul = m_fc.ReshapeMatMulOp(m_hMatMul, bTransA, bTransB, A, B, C, ref m_hA, ref m_hB, ref m_hC, out lWorkspaceInItems, ref hAws, ref hBws, nSharedIndex, m_bForceFallbackUse);

                if (hMatMul > 0)
                {
                    m_hMatMul = hMatMul;
                    if (m_hWorkspace != 0 && m_lWorkspaceInItems < lWorkspaceInItems)
                    {
                        m_cuda.FreeMemory(m_hWorkspace);
                        m_hWorkspace = 0;
                    }

                    if (lWorkspaceInItems > 0)
                    {
                        m_hWorkspace = m_cuda.AllocMemory(lWorkspaceInItems);
                        m_lWorkspaceInItems = lWorkspaceInItems;
                    }
                    else
                    {
                        m_hWorkspace = 0;
                        m_lWorkspaceInItems = 0;
                    }

                    m_hAws = hAws;
                    m_hBws = hBws;
                }
            }
            catch (Exception excpt)
            {
                CleanUp();
                throw new Exception("The fused computation failed to create the MatMulOp.", excpt);
            }
            finally
            {
                unprep(A, B, C);
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
        /// <param name="nAxis">Specifies the axis to run on (default = 2).</param>
        /// <param name="bForceFallbackUse">Optionally, specifies to force using the fallback.</param>
        /// <param name="nLocalID">Optionally, specifies the localID used for graph caching - NOTE, ID's must be factors of 2.</param>
        public MatMulGradOp(CudaDnn<T> cuda, Log log, int nAxis = 2, bool bForceFallbackUse = false, int nLocalID = -1)
        {
            if (nLocalID >= 0 && nLocalID % 2 != 0)
                throw new Exception("The localID must be a factor of 2.");

            int nLocalID1 = (nLocalID < 0) ? -1 : nLocalID;
            int nLocalID2 = (nLocalID < 0) ? -1 : nLocalID + 1;

            m_op1 = new MatMulOp<T>(cuda, log, nAxis, bForceFallbackUse, nLocalID1);
            m_op2 = new MatMulOp<T>(cuda, log, nAxis, bForceFallbackUse, nLocalID2);
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
        public void Create(Blob<T> A, Blob<T> B, Blob<T> C, bool bTransA, bool bTransB, int nSharedIndex1 = -1, int nSharedIndex2 = -1)
        {
            m_op1.Create(C, B, A, false, !bTransB, nSharedIndex1);
            m_op2.Create(A, C, B, !bTransA, false, nSharedIndex2);
        }

        /// <summary>
        /// Reshape the gradient of the matrix multiplication operation.
        /// </summary>
        /// <param name="A">Specifies the A blob that will have its diff updated.</param>
        /// <param name="B">Specifies the B blob that will have its </param>
        /// <param name="C">Specifies the C blob containing the originating diff to be propagated.</param>
        /// <param name="bTransA">Specifies that the A blob was transposed during the forward pass.</param>
        /// <param name="bTransB">Specifies that the B blob was transposed during the forward pass.</param>
        public void Reshape(Blob<T> A, Blob<T> B, Blob<T> C, bool bTransA, bool bTransB)
        {
            m_op1.Reshape(C, B, A, false, !bTransB);
            m_op2.Reshape(A, C, B, !bTransA, false);
        }

        /// <summary>
        /// Run the gradient of the matrix multiplication operation.
        /// </summary>
        /// <param name="A">Specifies the A blob that will have its diff updated.</param>
        /// <param name="B">Specifies the B blob that will have its </param>
        /// <param name="C">Specifies the C blob containing the originating diff to be propagated.</param>
        /// <param name="hAws">Optionally, specifies the A workspace data that contains the transposed A from a forward pass MatMul.  A value of 0 ignores this parameter.</param>
        /// <param name="hBws">Optionally, specifies the B workspace data that contains the transposed B from a forward pass MatMul.  A value of 0 ignores this parameter.</param>
        /// <param name="bFreezeLearning">Optionally, specifies that learning is frozen so gradients on B are not to be calculated.</param>
        public void Run(Blob<T> A, Blob<T> B, Blob<T> C, long hAws = 0, long hBws = 0, bool bFreezeLearning = false)
        {
            if (hBws == 0)
                hBws = B.gpu_data;

            m_op1.Run(C.gpu_diff, hBws, A.mutable_gpu_diff);

            if (!bFreezeLearning)
            {
                if (hAws == 0)
                    hAws = A.gpu_data;

                m_op2.Run(hAws, C.gpu_diff, B.mutable_gpu_diff);
            }
        }
    }
}
