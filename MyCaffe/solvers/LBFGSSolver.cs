using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.db.image;
using MyCaffe.param;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MyCaffe.solvers
{
    /// <summary>
    /// Optimizes the parameters of a Net using L-BFGS.  This implementation
    /// is based on minFunc, by Marc Schmidt.
    /// </summary>
    /// <remarks>
    /// @see [minFunc](https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html) by Marc Schmidt, 2005
    /// @see [ftokarev/caffe-neural-style Github](https://github.com/ftokarev/caffe-neural-style) by ftokarev, 2017. 
    /// @see [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge, 2015 
    /// </remarks>
    /// <typeparam name="T">Specifies the base type of <i>double</i> or <i>float</i>.</typeparam>
    public class LBFGSSolver<T> : Solver<T>
    {
        Blob<T> m_blobGradientsPrev;
        Blob<T> m_blobGradients;
        Blob<T> m_blobDirection;
        BlobCollection<T> m_colBlobHistoryS = new BlobCollection<T>();
        BlobCollection<T> m_colBlobHistoryY = new BlobCollection<T>();
        List<double> m_rgRhoHistory = new List<double>();
        int m_nStart;
        int m_nEnd;
        int m_nN;
        double m_dfH0;
        double m_dfStep;
        T m_tZero;
        T m_tOne;
        T m_tMinusOne;

        /// <summary>
        /// The LBFGSSolver constructor.
        /// </summary>
        /// <param name="cuda">Specifies the instance of CudaDnn to use.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies teh SolverParameter.</param>
        /// <param name="evtCancel">Specifies a CancelEvent used to cancel the current operation (e.g. training, testing) for which the Solver is performing.</param>
        /// <param name="evtForceSnapshot">Specifies an automatic reset event that causes the Solver to perform a Snapshot when set.</param>
        /// <param name="evtForceTest">Specifies an automatic reset event that causes teh Solver to run a testing cycle when set.</param>
        /// <param name="imgDb">Specifies the MyCaffeImageDatabase.</param>
        /// <param name="persist">Specifies the peristence used for loading and saving weights.</param>
        /// <param name="nSolverCount">Specifies the number of Solvers participating in a multi-GPU session.</param>
        /// <param name="nSolverRank">Specifies the rank of this Solver in a multi-GPU session.</param>
        /// <param name="shareNet">Optionally, specifies the net to share when creating the training network (default = null, meaning no share net is used).</param>
        /// <param name="getws">Optionally, specifies the handler for getting the workspace.</param>
        /// <param name="setws">Optionally, specifies the handler for setting the workspace.</param>
        public LBFGSSolver(CudaDnn<T> cuda, Log log, SolverParameter p, CancelEvent evtCancel, AutoResetEvent evtForceSnapshot, AutoResetEvent evtForceTest, IXImageDatabaseBase imgDb, IXPersist<T> persist, int nSolverCount = 1, int nSolverRank = 0, Net<T> shareNet = null, onGetWorkspace getws = null, onSetWorkspace setws = null)
            : base(cuda, log, p, evtCancel, evtForceSnapshot, evtForceTest, imgDb, persist, nSolverCount, nSolverRank, shareNet, getws, setws)
        {
            m_tZero = (T)Convert.ChangeType(0, typeof(T));
            m_tOne = (T)Convert.ChangeType(1, typeof(T));
            m_tMinusOne = (T)Convert.ChangeType(-1, typeof(T));
            PreSolve();
        }

        /// <summary>
        /// Releases all resources (GPU and Host) used by the Solver.
        /// </summary>
        protected override void dispose()
        {
            if (m_blobGradients != null)
            {
                m_blobGradients.Dispose();
                m_blobGradients = null;
            }

            if (m_blobGradientsPrev != null)
            {
                m_blobGradientsPrev.Dispose();
                m_blobGradientsPrev = null;
            }

            if (m_blobDirection != null)
            {
                m_blobDirection.Dispose();
                m_blobDirection = null;
            }

            if (m_colBlobHistoryY != null)
            {
                m_colBlobHistoryY.Dispose();
                m_colBlobHistoryY = null;
            }

            if (m_colBlobHistoryS != null)
            {
                m_colBlobHistoryS.Dispose();
                m_colBlobHistoryS = null;
            }

            base.dispose();
        }

        /// <summary>
        /// Runs the pre-solve which parpares the Solver to start Solving.
        /// </summary>
        public void PreSolve()
        {
            try
            {
                BlobCollection<T> net_params = m_net.learnable_parameters;

                m_nN = 0;

                for (int i = 0; i < net_params.Count; i++)
                {
                    if (m_net.params_lr[i] != 0)
                        m_nN += net_params[i].count();
                }

                // Nothing to do, all learnable parameters have lr_mult = 0
                if (m_nN == 0)
                    return;

                List<int> rgShape = new List<int>() { m_nN };
                m_colBlobHistoryS.Clear(true);
                m_colBlobHistoryY.Clear(true);
                m_rgRhoHistory.Clear();
                m_nStart = 0;
                m_nEnd = -1;

                m_blobGradients = new Blob<T>(m_cuda, m_log, rgShape, false);
                m_blobGradients.Name = "gradients";
                m_blobGradientsPrev = new Blob<T>(m_cuda, m_log, rgShape, false);
                m_blobGradientsPrev.Name = "gradients prev";
                m_blobDirection = new Blob<T>(m_cuda, m_log, rgShape, false);
                m_blobDirection.Name = "direction";

                for (int i = 0; i < m_param.lbgfs_corrections; i++)
                {
                    m_colBlobHistoryS.Add(new Blob<T>(m_cuda, m_log, rgShape, false));
                    m_colBlobHistoryY.Add(new Blob<T>(m_cuda, m_log, rgShape, false));
                    m_rgRhoHistory.Add(0);
                }
            }
            catch (Exception excpt)
            {
                m_colBlobHistoryS.Clear(true);
                m_colBlobHistoryY.Clear(true);
                m_rgRhoHistory.Clear();

                if (m_blobGradients != null)
                {
                    m_blobGradients.Dispose();
                    m_blobGradients = null;
                }

                if (m_blobGradientsPrev != null)
                {
                    m_blobGradientsPrev.Dispose();
                    m_blobGradientsPrev = null;
                }

                if (m_blobDirection != null)
                {
                    m_blobDirection.Dispose();
                    m_blobDirection = null;
                }

                throw excpt;
            }
            finally
            {
            }
        }

        /// <summary>
        /// Apply the gradients to the network.
        /// </summary>
        /// <param name="nIterationOverride">Optionally, specifies an iteration override (default = -1, which ignores the override).</param>
        /// <returns></returns>
        public override double ApplyUpdate(int nIterationOverride = -1)
        {
            if (m_nN == 0)
            {
                for (int i = 0; i < m_net.learnable_parameters.Count; i++)
                {
                    m_net.learnable_parameters[i].SetDiff(0);
                }

                return 0;
            }

            m_log.CHECK(is_root_solver, "You can only apply the LBFGS Solver updates on the root solver.");

            CollectGradients();
            UpdateHistory();
            ComputeInitialHessianApprox();
            ComputeDirection();
            ComputeStep();
            UpdateNet();

            // Increment the internal iter_ counter -- its value should always indicate
            // the number of times the weights have been updated.
            m_nIter++;

            return 0;
        }

        /// <summary>
        /// Collect the gradients from the network learnable parameters.
        /// </summary>
        public virtual void CollectGradients()
        {
            BlobCollection<T> net_params = m_net.learnable_parameters;

            if (m_nIter != 0)
                m_cuda.copy(m_nN, m_blobGradients.gpu_data, m_blobGradientsPrev.mutable_gpu_data);

            int nDstOffset = 0;
            for (int i = 0; i < net_params.Count; i++)
            {
                if (m_net.params_lr[i] != 0)
                {
                    m_cuda.copy(net_params[i].count(), net_params[i].gpu_diff, m_blobGradients.mutable_gpu_data, 0, nDstOffset);
                    nDstOffset += net_params[i].count();
                }
            }
        }

        /// <summary>
        /// Update the history values with the gradients and direction.
        /// </summary>
        public virtual void UpdateHistory()
        {
            if (m_nIter == 0)
                return;

            m_cuda.scal(m_nN, m_tMinusOne, m_blobDirection.mutable_gpu_data); // s
            m_cuda.axpby(m_nN, m_tOne, m_blobGradients.gpu_data, m_tMinusOne, m_blobGradientsPrev.mutable_gpu_data); // y
            T fYs = m_cuda.dot(m_nN, m_blobDirection.gpu_data, m_blobGradientsPrev.gpu_data);
            double dfYs = Utility.ConvertVal<T>(fYs);

            if (dfYs < 1e-10)
            {
                m_log.WriteLine("WARNING: Skipping L-BFGS update.");
                if (m_nEnd < 0)
                    m_nEnd = 0;

                return;
            }

            m_nEnd += 1;

            if (m_nEnd < m_param.lbgfs_corrections)
            {
                if (m_nStart != 0)
                {
                    m_nStart += 1;

                    if (m_nStart == m_param.lbgfs_corrections)
                        m_nStart = 0;
                }
            }
            else
            {
                m_nStart = 1;
                m_nEnd = 0;
            }

            m_cuda.copy(m_nN, m_blobDirection.gpu_data, m_colBlobHistoryS[m_nEnd].mutable_gpu_data);
            m_cuda.copy(m_nN, m_blobGradientsPrev.gpu_data, m_colBlobHistoryY[m_nEnd].mutable_gpu_data);
            m_rgRhoHistory[m_nEnd] = 1.0 / dfYs;
        }

        /// <summary>
        /// Compute the initial Hessian approximation.
        /// </summary>
        public virtual void ComputeInitialHessianApprox()
        {
            if (m_nIter == 0)
                return;

            T fh0 = m_cuda.dot(m_nN, m_colBlobHistoryY[m_nEnd].gpu_data, m_colBlobHistoryY[m_nEnd].gpu_data);
            double dfH0 = Utility.ConvertVal<T>(fh0);

            m_dfH0 = 1.0 / m_rgRhoHistory[m_nEnd] / dfH0;
        }

        private List<int> lbfgs_history_indices(int nStart, int nEnd, int nMax)
        {
            List<int> rgIndices = Utility.Create<int>((nStart == 0) ? nEnd + 1 : nMax, 0);

            if (nStart == 0)
            {
                for (int i = nStart; i <= nEnd; i++)
                {
                    rgIndices[i] = i;
                }
            }
            else
            {
                int j = 0;

                for (int i = nStart; i < rgIndices.Count; i++)
                {
                    rgIndices[j++] = i;
                }

                for (int i = 0; i <= nEnd; i++)
                {
                    rgIndices[j++] = i;
                }
            }

            return rgIndices;
        }

        /// <summary>
        /// Compute the direction.
        /// </summary>
        public virtual void ComputeDirection()
        {
            m_cuda.copy(m_nN, m_blobGradients.gpu_data, m_blobDirection.mutable_gpu_data);

            if (m_nIter == 0)
                return;

            List<int> rgIndices = lbfgs_history_indices(m_nStart, m_nEnd, m_param.lbgfs_corrections);
            List<double> rgAlpha = Utility.Create<double>(rgIndices.Count, 0);
            double dfBeta = 0;

            for (int i = rgIndices.Count - 1; i >= 0; i--)
            {
                int nIdx = rgIndices[i];

                T fAlpha = m_cuda.dot(m_nN, m_colBlobHistoryS[nIdx].gpu_data, m_blobDirection.gpu_data);
                rgAlpha[nIdx] = (double)Utility.ConvertVal<T>(fAlpha);
                rgAlpha[nIdx] *= m_rgRhoHistory[nIdx];

                m_cuda.axpy(m_nN, -rgAlpha[nIdx], m_colBlobHistoryY[nIdx].gpu_data, m_blobDirection.mutable_gpu_data);
            }

            m_cuda.scal(m_nN, m_dfH0, m_blobDirection.mutable_gpu_data);

            for (int i = 0; i < rgIndices.Count; i++)
            {
                int nIdx = rgIndices[i];

                T fBeta = m_cuda.dot(m_nN, m_colBlobHistoryY[nIdx].gpu_data, m_blobDirection.gpu_data);
                dfBeta = (double)Utility.ConvertVal<T>(fBeta);
                dfBeta *= m_rgRhoHistory[nIdx];

                m_cuda.axpy(m_nN, rgAlpha[nIdx] - dfBeta, m_colBlobHistoryS[nIdx].gpu_data, m_blobDirection.mutable_gpu_data);
            }
        }

        /// <summary>
        /// Compute the step.
        /// </summary>
        public virtual void ComputeStep()
        {
            m_dfStep = 1.0;
        }

        /// <summary>
        /// Update the network.
        /// </summary>
        public virtual void UpdateNet()
        {
            m_cuda.scal(m_nN, m_dfStep, m_blobDirection.mutable_gpu_data);

            BlobCollection<T> net_params = m_net.learnable_parameters;

            int nOffset = 0;
            for (int i = 0; i < net_params.Count; i++)
            {
                int nCount = net_params[i].count();

                if (m_net.params_lr[i] != 0)
                {
                    double dfLr = m_net.params_lr[i].GetValueOrDefault(1.0) * m_param.base_lr;

                    if (dfLr != 1.0)
                    {
                        T fLr = (T)Convert.ChangeType(m_net.params_lr[i], typeof(T));
                        m_cuda.scale(nCount, fLr, m_blobDirection.gpu_data, net_params[i].mutable_gpu_diff, nOffset, 0);
                    }

                    nOffset += nCount;
                }
                else
                {
                    net_params[i].SetDiff(0);
                }
            }

            m_net.Update();
        }

        /// <summary>
        /// Restore a previously saved solver state.
        /// </summary>
        /// <param name="rgState">Specifies the solver state to restore.</param>
        protected override void RestoreSolverState(byte[] rgState)
        {
            SolverState state = m_persist.LoadSolverState(rgState, m_param.type);

            m_nIter = state.iter;
            m_nCurrentStep = state.current_step;
            m_nStart = state.start;
            m_nEnd = state.end;

            List<int> rgIndices = lbfgs_history_indices(m_nStart, m_nEnd, m_param.lbgfs_corrections);

            for (int i = 0; i < rgIndices.Count; i++)
            {
                int nIdx = rgIndices[i];

                m_colBlobHistoryS[i].FromProto(state.history[nIdx]);
                m_colBlobHistoryY[i].FromProto(state.s_history[nIdx]);
                m_rgRhoHistory[i] = state.rho_history[i];
            }

            m_blobGradients.FromProto(state.gradients);
            m_blobDirection.FromProto(state.direction);
        }

        /// <summary>
        /// Save the solver state.
        /// </summary>
        /// <returns>They byte stream of the solver state is returned.</returns>
        protected override byte[] SnapshotSolverState()
        {
            SolverState state = new SolverState();

            state.iter = m_nIter;
            state.current_step = m_nCurrentStep;
            state.start = m_nStart;
            state.end = m_nEnd;

            List<int> rgIndices = lbfgs_history_indices(m_nStart, m_nEnd, m_param.lbgfs_corrections);

            for (int i = 0; i < rgIndices.Count; i++)
            {
                int nIdx = rgIndices[i];

                state.s_history.Add(m_colBlobHistoryS[nIdx].ToProto());
                state.history.Add(m_colBlobHistoryY[nIdx].ToProto());
                state.rho_history.Add(m_rgRhoHistory[nIdx]);
            }

            state.gradients = m_blobGradients.ToProto();
            state.direction = m_blobDirection.ToProto();

            return m_persist.SaveSolverState(state, m_param.type);
        }
    }
}
