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
    /// Use AdamW Solver which uses gradient based optimization like Adam with a decoupled weight decay.
    /// </summary>
    /// <remarks>
    /// @see [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) by Loshchilov, I. and Hutter, F., 2019.
    /// @see [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980v9) by Diederik P. Kingma, and Jimmy Ba, 2014.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class AdamWSolver<T> : SGDSolver<T>
    {
        double m_dfDetachedWeightDecayRate = 0.0f;

        /// <summary>
        /// The AdamSolver constructor.
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
        public AdamWSolver(CudaDnn<T> cuda, Log log, SolverParameter p, CancelEvent evtCancel, AutoResetEvent evtForceSnapshot, AutoResetEvent evtForceTest, IXImageDatabaseBase imgDb, IXPersist<T> persist, int nSolverCount = 1, int nSolverRank = 0, Net<T> shareNet = null, onGetWorkspace getws = null, onSetWorkspace setws = null)
            : base(cuda, log, p, evtCancel, evtForceSnapshot, evtForceTest, imgDb, persist, nSolverCount, nSolverRank, shareNet, getws, setws)
        {
            m_dfDetachedWeightDecayRate = p.adamw_decay;
            AdamPreSolve();
        }

        /// <summary>
        /// Runs the AdamSolver pre-solve which parpares the Solver to start Solving.
        /// </summary>
        public virtual void AdamPreSolve()
        {
            // Add the extra history entries for AdaDelta after those from
            // SGDSolver::PreSolve
            BlobCollection<T> colNetParams = m_net.learnable_parameters;

            for (int i = 0; i < colNetParams.Count; i++)
            {
                List<int> rgShape = colNetParams[i].shape();
                Blob<T> blob = new Blob<T>(m_cuda, m_log, rgShape);
                m_colHistory.Add(blob);
            }
        }

        /// <summary>
        /// Compute the AdamWSolver update value that will be applied to a learnable blobs in the training Net.
        /// </summary>
        /// <param name="param_id">Specifies the id of the Blob.</param>
        /// <param name="dfRate">Specifies the learning rate.</param>
        /// <param name="nIterationOverride">Optionally, specifies an iteration override, or -1 which is ignored.</param>
        public override void ComputeUpdateValue(int param_id, double dfRate, int nIterationOverride = -1)
        {
            BlobCollection<T> colNetParams = m_net.learnable_parameters;

            if (!colNetParams[param_id].DiffExists)
                return;

            if (nIterationOverride == -1)
                nIterationOverride = m_nIter;

            List<double?> net_params_lr = m_net.params_lr;
            double dfLocalRate = dfRate * net_params_lr[param_id].GetValueOrDefault(0);
            List<double?> net_params_decay = net.params_weight_decay;
            double dfLocalDecay = m_dfDetachedWeightDecayRate * net_params_decay[param_id].GetValueOrDefault(0);
            double dfBeta1 = m_param.momentum;
            T fBeta1 = Utility.ConvertVal<T>(dfBeta1);
            double dfBeta2 = m_param.momentum2;
            T fBeta2 = Utility.ConvertVal<T>(dfBeta2);

            // we create aliases for convienience
            int nUpdateHistoryOffset = colNetParams.Count;
            Blob<T> val_m = m_colHistory[param_id];
            Blob<T> val_v = m_colHistory[param_id + nUpdateHistoryOffset];

            int nT = nIterationOverride + 1;
            // Set the schedule multiplier
            int nN = colNetParams[param_id].count();
            double dfEpsHat = m_param.delta;

            // Compute the update to history, then copy it to the parameter diff.
            m_cuda.adamw_update(nN,
                               colNetParams[param_id].mutable_gpu_diff,
                               val_m.mutable_gpu_data,
                               val_v.mutable_gpu_data,
                               fBeta1,
                               fBeta2,
                               Utility.ConvertVal<T>(dfEpsHat),
                               Utility.ConvertVal<T>(dfLocalRate),
                               Utility.ConvertVal<T>(dfLocalDecay),
                               colNetParams[param_id].mutable_gpu_data,
                               nT);
        }
    }
}
