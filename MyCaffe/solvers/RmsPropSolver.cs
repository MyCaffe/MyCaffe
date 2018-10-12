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
    /// Use RmsProp Solver which uses gradient based optimization like SGD.
    /// </summary>
    /// <remarks>
    /// @see [Lecture 6e	rmsprop: Divide the gradient by a running average of its recent magnitude](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) by Tieleman and Hinton, 2012,
    /// @see [RMSProp and equilibrated adaptive learning rates for non-convex optimization](https://arxiv.org/abs/1502.04390v1) by Yann N. Dauphin, Harm de Vries, Junyoung Chung, and Yoshua Bengio, 2015.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class RmsPropSolver<T> : SGDSolver<T>
    {
        /// <summary>
        /// The RmsPropSolver constructor.
        /// </summary>
        /// <param name="cuda">Specifies the instance of CudaDnn to use.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies teh SolverParameter.</param>
        /// <param name="evtCancel">Specifies a CancelEvent used to cancel the current operation (e.g. training, testing) for which the Solver is performing.</param>
        /// <param name="evtForceSnapshot">Specifies an automatic reset event that causes the Solver to perform a Snapshot when set.</param>
        /// <param name="evtForceTest">Specifies an automatic reset event that causes teh Solver to run a testing cycle when set.</param>
        /// <param name="imgDb">Specifies the CaffeImageDatabase.</param>
        /// <param name="persist">Specifies the peristence used for loading and saving weights.</param>
        /// <param name="nSolverCount">Specifies the number of Solvers participating in a multi-GPU session.</param>
        /// <param name="nSolverRank">Specifies the rank of this Solver in a multi-GPU session.</param>
        public RmsPropSolver(CudaDnn<T> cuda, Log log, SolverParameter p, CancelEvent evtCancel, AutoResetEvent evtForceSnapshot, AutoResetEvent evtForceTest, IXImageDatabase imgDb, IXPersist<T> persist, int nSolverCount = 1, int nSolverRank = 0)
            : base(cuda, log, p, evtCancel, evtForceSnapshot, evtForceTest, imgDb, persist, nSolverCount, nSolverRank)
        {
            m_log.CHECK_EQ(0, m_param.momentum, "Momentum cannot be used with RmsProp.");
            m_log.CHECK_GE(m_param.rms_decay, 0, "rms_decay should lie between 0 and 1.");
            m_log.CHECK_LT(m_param.rms_decay, 1, "rms_decay should lie between 0 and 1.");
        }

        /// <summary>
        /// Compute the RmsProp update value that will be applied to a learnable blobs in the training Net.
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
            T fRmsDecay = Utility.ConvertVal<T>(m_param.rms_decay);
            T fLocalRate = Utility.ConvertVal<T>(dfRate * net_params_lr[param_id].GetValueOrDefault(0));

            // Compute the update to history, then copy it to the parameter diff.
            m_cuda.rmsprop_update(colNetParams[param_id].count(), 
                                  colNetParams[param_id].mutable_gpu_diff, 
                                  m_colHistory[param_id].mutable_gpu_data, 
                                  fRmsDecay,
                                  fDelta, 
                                  fLocalRate);
        }
    }
}
