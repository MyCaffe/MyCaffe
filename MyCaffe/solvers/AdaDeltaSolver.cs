using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.IO;
using MyCaffe.basecode;
using MyCaffe.imagedb;
using MyCaffe.common;
using MyCaffe.param;

namespace MyCaffe.solvers
{
    /// <summary>
    /// Use AdaDelta Solver which has gradient based optimization like SGD.
    /// </summary>
    /// <remarks>
    /// See [ADADELTA: An Adaptive Learning Rate Method](https://arxiv.org/abs/1212.5701) by Matthew D. Zeiler, 2012.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class AdaDeltaSolver<T> : SGDSolver<T>
    {
        /// <summary>
        /// The SGDSolver constructor.
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
        public AdaDeltaSolver(CudaDnn<T> cuda, Log log, SolverParameter p, CancelEvent evtCancel, AutoResetEvent evtForceSnapshot, AutoResetEvent evtForceTest, IXImageDatabase imgDb, IXPersist<T> persist, int nSolverCount = 1, int nSolverRank = 0)
            : base(cuda, log, p, evtCancel, evtForceSnapshot, evtForceTest, imgDb, persist, nSolverCount, nSolverRank)
        {
            AdaDeltaPreSolve();
        }

        /// <summary>
        /// Runs the AdaDleta pre-solve which parpares the Solver to start Solving.
        /// </summary>
        public void AdaDeltaPreSolve()
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
        /// Compute the AdaDelta update value that will be applied to a learnable blobs in the training Net.
        /// </summary>
        /// <param name="param_id">Specifies the id of the Blob.</param>
        /// <param name="dfRate">Specifies the learning rate.</param>
        public override void ComputeUpdateValue(int param_id, double dfRate)
        {
            BlobCollection<T> colNetParams = m_net.learnable_parameters;
            List<double?> net_params_lr = m_net.params_lr;
            T fDelta = Utility.ConvertVal<T>(m_param.delta);
            T fMomentum = Utility.ConvertVal<T>(m_param.momentum);
            T fLocalRate = Utility.ConvertVal<T>(dfRate * net_params_lr[param_id].GetValueOrDefault(0));
            int nUpdateHistoryOffset = colNetParams.Count;

            // Compute the update to history, then copy it to the parameter diff.
            m_cuda.adadelta_update(colNetParams[param_id].count(), 
                                   colNetParams[param_id].mutable_gpu_diff, 
                                   m_colHistory[param_id].mutable_gpu_data, 
                                   m_colHistory[nUpdateHistoryOffset + param_id].mutable_gpu_data,
                                   fMomentum, 
                                   fDelta, 
                                   fLocalRate);
        }
    }
}
