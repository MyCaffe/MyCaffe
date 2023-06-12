using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.IO;
using MyCaffe.basecode;
using MyCaffe.db.image;
using MyCaffe.common;
using MyCaffe.param;

namespace MyCaffe.solvers
{
    /// <summary>
    /// Use AdaGrad Solver based optimization like SGD that tries to find rarely seen features.
    /// </summary>
    /// <remarks>
    /// @see [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf) by John Duchi, Elad Hazan, and Yoram Singer, 2011.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class AdaGradSolver<T> : SGDSolver<T>
    {
        /// <summary>
        /// The NesterovSolver constructor.
        /// </summary>
        /// <param name="cuda">Specifies the instance of CudaDnn to use.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies teh SolverParameter.</param>
        /// <param name="evtCancel">Specifies a CancelEvent used to cancel the current operation (e.g. training, testing) for which the Solver is performing.</param>
        /// <param name="evtForceSnapshot">Specifies an automatic reset event that causes the Solver to perform a Snapshot when set.</param>
        /// <param name="evtForceTest">Specifies an automatic reset event that causes teh Solver to run a testing cycle when set.</param>
        /// <param name="db">Specifies the in-memory MyCaffeDatabase.</param>
        /// <param name="persist">Specifies the peristence used for loading and saving weights.</param>
        /// <param name="nSolverCount">Specifies the number of Solvers participating in a multi-GPU session.</param>
        /// <param name="nSolverRank">Specifies the rank of this Solver in a multi-GPU session.</param>
        /// <param name="shareNet">Optionally, specifies the net to share when creating the training network (default = null, meaning no share net is used).</param>
        /// <param name="getws">Optionally, specifies the handler for getting the workspace.</param>
        /// <param name="setws">Optionally, specifies the handler for setting the workspace.</param>
        public AdaGradSolver(CudaDnn<T> cuda, Log log, SolverParameter p, CancelEvent evtCancel, AutoResetEvent evtForceSnapshot, AutoResetEvent evtForceTest, IXDatabaseBase db, IXPersist<T> persist, int nSolverCount = 1, int nSolverRank = 0, Net<T> shareNet = null, onGetWorkspace getws = null, onSetWorkspace setws = null)
            : base(cuda, log, p, evtCancel, evtForceSnapshot, evtForceTest, db, persist, nSolverCount, nSolverRank, shareNet, getws, setws)
        {
            m_log.CHECK_EQ(0, m_param.momentum, "Momentum cannot be used with AdaGrad.");
        }

        /// <summary>
        /// Compute the AdaGrad update value that will be applied to a learnable blobs in the training Net.
        /// </summary>
        /// <param name="param_id">Specifies the id of the Blob.</param>
        /// <param name="dfRate">Specifies the learning rate.</param>
        /// <param name="nIterationOverride">Optionally, specifies an iteration override, or -1 which is ignored.</param>
        public override void ComputeUpdateValue(int param_id, double dfRate, int nIterationOverride = -1)
        {
            BlobCollection<T> colNetParams = m_net.learnable_parameters;

            if (!colNetParams[param_id].DiffExists)
                return;

            List<double?> net_params_lr = m_net.params_lr;
            T fDelta = Utility.ConvertVal<T>(m_param.delta);
            T fLocalRate = Utility.ConvertVal<T>(dfRate * net_params_lr[param_id].GetValueOrDefault(0));

            // Compute the update to history, then copy it to the parameter diff.
            m_cuda.adagrad_update(colNetParams[param_id].count(), 
                                  colNetParams[param_id].mutable_gpu_diff, 
                                  m_colHistory[param_id].mutable_gpu_data, 
                                  fDelta, 
                                  fLocalRate);
        }
    }
}
