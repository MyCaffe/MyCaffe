﻿using System;
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
    /// Use Nesterov's accelerated gradient Solver, which is similar to SGD, but the error gradient is computed on the weights with added momentum.
    /// </summary>
    /// <remarks>
    /// @see [Lecture 6c The momentum method](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) by Geoffrey Hinton, Nitish Srivastava, and Kevin Swersky, 2012.
    /// @see [Nesterov's Accelerated Gradient and Momentum as approximations to Regularised Update Descent](https://arxiv.org/abs/1607.01981) by Alexandar Botev, Guy Lever, and David Barber, 2016.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class NesterovSolver<T> : SGDSolver<T>
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
        /// <param name="imgDb">Specifies the CaffeImageDatabase.</param>
        /// <param name="persist">Specifies the peristence used for loading and saving weights.</param>
        /// <param name="nSolverCount">Specifies the number of Solvers participating in a multi-GPU session.</param>
        /// <param name="nSolverRank">Specifies the rank of this Solver in a multi-GPU session.</param>
        /// <param name="shareNet">Optionally, specifies the net to share when creating the training network (default = null, meaning no share net is used).</param>
        public NesterovSolver(CudaDnn<T> cuda, Log log, SolverParameter p, CancelEvent evtCancel, AutoResetEvent evtForceSnapshot, AutoResetEvent evtForceTest, IXImageDatabase imgDb, IXPersist<T> persist, int nSolverCount = 1, int nSolverRank = 0, Net<T> shareNet = null)
            : base(cuda, log, p, evtCancel, evtForceSnapshot, evtForceTest, imgDb, persist, nSolverCount, nSolverRank, shareNet)
        {
        }

        /// <summary>
        /// Compute the Nesterov update value that will be applied to a learnable blobs in the training Net.
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
            T fMomentum = Utility.ConvertVal<T>(m_param.momentum);
            T fLocalRate = Utility.ConvertVal<T>(dfRate * net_params_lr[param_id].GetValueOrDefault(0));

            // Compute the update to history, then copy it to the parameter diff.
            m_cuda.nesterov_update(colNetParams[param_id].count(), colNetParams[param_id].mutable_gpu_diff, m_colHistory[param_id].mutable_gpu_data, fMomentum, fLocalRate);
        }
    }
}
