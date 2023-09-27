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
    /// Stochastic Gradient Descent solver with momentum updates weights by a linear combination of the negative gradient and the previous weight update.
    /// </summary>
    /// <remarks>
    /// @see [Stochastic Gradient Descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) Wikipedia.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class SGDSolver<T> : Solver<T>
    {
        /// <summary>
        /// History maintains the historical momentum data.
        /// </summary>
        protected BlobCollection<T> m_colHistory = new BlobCollection<T>();

        /// <summary>
        /// Update maintains update related data and is not needed in snapshots.
        /// </summary>
        //  protected BlobCollection<T> m_colUpdate = new BlobCollection<T>();  // not used in GPU version

        /// <summary>
        /// Temp maintains other information that might be needed in computation
        /// of gradients/updates and is not needed in snapshots.
        /// </summary>
        protected BlobCollection<T> m_colTemp = new BlobCollection<T>();

        /// <summary>
        /// The SGDSolver constructor.
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
        public SGDSolver(CudaDnn<T> cuda, Log log, SolverParameter p, CancelEvent evtCancel, AutoResetEvent evtForceSnapshot, AutoResetEvent evtForceTest, IXDatabaseBase db, IXPersist<T> persist, int nSolverCount = 1, int nSolverRank = 0, Net<T> shareNet = null, onGetWorkspace getws = null, onSetWorkspace setws = null)
            : base(cuda, log, p, evtCancel, evtForceSnapshot, evtForceTest, db, persist, nSolverCount, nSolverRank, shareNet, getws, setws)
        {
            PreSolve();
        }

        /// <summary>
        /// Releases all resources (GPU and Host) used by the Solver.
        /// </summary>
        protected override void dispose()
        {
            if (m_colHistory != null)
            {
                m_colHistory.Dispose();
                m_colHistory = null;
            }

            if (m_colTemp != null)
            {
                m_colTemp.Dispose();
                m_colTemp = null;
            }

            base.dispose();
        }

        /// <summary>
        /// Returns the history BlobCollection containing historical momentum data.
        /// </summary>
        public BlobCollection<T> history
        {
            get { return m_colHistory; }
        }

        /// <summary>
        /// Runs the pre-solve which prepares the Solver to start Solving.
        /// </summary>
        public void PreSolve()
        {
            BlobCollection<T> colNetParams = m_net.learnable_parameters;
            m_colHistory.Clear(true);
//            m_colUpdate.Clear(true);
            m_colTemp.Clear(true);

            for (int i = 0; i < colNetParams.Count; i++)
            {
                List<int> rgShape = colNetParams[i].shape();

                m_colHistory.Add(new Blob<T>(m_cuda, m_log, rgShape, false));   // diff never used
//                m_colUpdate.Add(new Blob<T>(m_cuda, m_log, rgShape, false));
                m_colTemp.Add(new Blob<T>(m_cuda, m_log, rgShape, false));      // diff never used
            }
        }

        /// <summary>
        /// Return the current learning rate. 
        /// </summary>
        /// <remarks>
        /// The currently implemented learning rate policies are as follows:
        ///    - fixed: always return @f$ base_lr @f$.
        ///    - step: return @f$ base_lr * gamma ^ {floor{iter / step}} @f$
        ///    - exp: return @f$ base_lr * gamma ^ iter @f$
        ///    - inv: return @f$ base_lr * {1 + gamma * iter} ^ {-power} @f$
        ///    - multistep: similar to step but it allows non-uniform steps defined by stepvalue.
        ///    - poly: the effective learning rate follows a polynomial decay, to be
        ///            zero by the max_iter.  return @f$ base_lr * {1 - iter/max_iter} ^ {power} @f$
        ///    - sigmoid: the effective learning rate follows a sigmoid decay.
        ///            return @f$ base_lr * {1/{1 + exp{-gamma * {iter - stepsize}}}} @f$
        ///            
        /// where base_lr, max_iter, gamma, step, stepvalue and power are defined int the
        /// solver protocol buffer, and iter is the current iteration.
        /// </remarks>
        /// <param name="nIterationOverride">Optionally, specifies an iteration override, or -1 which is ignored.</param>
        /// <returns>The learning rate value.</returns>
        public double GetLearningRate(int nIterationOverride = -1)
        {
            double dfRate = 0;

            if (nIterationOverride == -1)
                nIterationOverride = m_nIter;

            switch (m_param.lr_policy)
            {
                case "fixed":
                    dfRate = m_param.base_lr;
                    break;

                case "step":
                    m_log.CHECK_GT(m_param.stepsize, 0, "The stepsize must be greater than 0.");
                    m_nCurrentStep = nIterationOverride / m_param.stepsize;
                    m_log.CHECK_GE(m_param.gamma, 0, "The gamma must be greater than or equal to 0.");
                    dfRate = m_param.base_lr * Math.Pow(m_param.gamma, m_nCurrentStep);
                    break;

                case "exp":
                    m_log.CHECK_GE(m_param.gamma, 0, "The gamma must be greater than or equal to 0.");
                    dfRate = m_param.base_lr * Math.Pow(m_param.gamma, nIterationOverride);
                    break;

                case "inv":
                    m_log.CHECK_GE(m_param.gamma, 0, "The gamma must be greater than or equal to 0.");
                    dfRate = m_param.base_lr * Math.Pow(1.0 + m_param.gamma * nIterationOverride, -1.0 * m_param.power);
                    break;

                case "multistep":
                    if (m_nCurrentStep < m_param.stepvalue.Count && nIterationOverride >= m_param.stepvalue[m_nCurrentStep])
                    {
                        m_nCurrentStep++;
                        m_log.WriteLine("MultiStep Status: Iteration " + nIterationOverride.ToString() + ", step = " + m_nCurrentStep.ToString());
                    }
                    m_log.CHECK_GE(m_param.gamma, 0, "The gamma must be greater than or equal to 0.");
                    dfRate = m_param.base_lr * Math.Pow(m_param.gamma, m_nCurrentStep);
                    break;

                case "poly":
                    dfRate = m_param.base_lr * Math.Pow(1.0 - ((double)nIterationOverride / (double)m_param.max_iter), m_param.power);
                    break;

                case "sigmoid":
                    m_log.CHECK_GE(m_param.gamma, 0, "The gamma must be greater than or equal to 0.");
                    m_log.CHECK_GT(m_param.stepsize, 0, "The stepsize must be greater than 0.");
                    dfRate = m_param.base_lr * (1.0 / (1.0 + Math.Exp(-1.0 * m_param.gamma * (nIterationOverride - m_param.stepsize))));
                    break;

                default:
                    m_log.FAIL("Unknown learning rate policy: " + m_param.lr_policy);
                    break;
            }

            return dfRate;
        }

        /// <summary>
        /// Compute the update values and apply them to the training Net.
        /// </summary>
        /// <param name="nIterationOverride">Optionally, specifies an iteration override, or -1 which is ignored.</param>
        /// <returns>The learning rate used is returned.</returns>
        public override double ApplyUpdate(int nIterationOverride = -1)
        {
            double dfRate = GetLearningRate(nIterationOverride);

            if (LearningRateOverride > 0)
                dfRate = LearningRateOverride;

            if (m_param.display > 0 && (m_nIter % m_param.display) == 0)
            {
                string strOut = "Iteration " + m_nIter.ToString() + ", lr = " + dfRate.ToString() + ", Loss = " + m_dfSmoothedLoss.ToString();
                if (m_dfIterAccuracy.HasValue)
                    strOut += ", Iter Accuracy = " + m_dfIterAccuracy.Value.ToString() + " (" + m_dfIterAccuracy.Value.ToString("P3") + ")";
                
                m_log.WriteLine(strOut);
            }

            ClipGradients();

            for (int i = 0; i < m_net.learnable_parameters.Count; i++)
            {
                Normalize(i);
                Regularize(i);
                ComputeUpdateValue(i, dfRate, nIterationOverride);
            }

            m_net.Update();

            // Increment the internal iter_ counter -- its value should always indicate
            // the number of times the weights have been updated.
            m_nIter++;

            return dfRate;
        }

        /// <summary>
        /// Restore the state of the Solver.
        /// </summary>
        /// <param name="rgState">Specifies the state of the Solver.</param>
        protected override void RestoreSolverState(byte[] rgState)
        {
            SolverState state = m_persist.LoadSolverState(rgState);

            m_nIter = state.iter;
            m_nCurrentStep = state.current_step;

            m_log.CHECK_EQ(state.history.Count, m_colHistory.Count, "Incorrect length of state history blobs.");
            m_log.WriteLine("SGDSolver: restoring state history.");

            for (int i = 0; i < m_colHistory.Count; i++)
            {
                m_colHistory[i].FromProto(state.history[i]);
            }
        }

        /// <summary>
        /// Take a snapshot of the Solver state.
        /// </summary>
        /// <returns>The Solver state snapshot is returned.</returns>
        protected override byte[] SnapshotSolverState()
        {
            SolverState state = new SolverState();
            state.iter = m_nIter;
            state.current_step = m_nCurrentStep;

            foreach (Blob<T> blob in m_colHistory)
            {
                state.history.Add(blob.ToProto());
            }

            return m_persist.SaveSolverState(state);
        }

        /// <summary>
        /// Normalize a learnable Blob of the training Net.
        /// </summary>
        /// <param name="param_id">Specifies the id of the Blob.</param>
        public virtual void Normalize(int param_id)
        {
            if (m_param.iter_size == 1)
                return;

            // Scale gradient to counterbalance accumulation.
            BlobCollection<T> colNetParams = m_net.learnable_parameters;

            if (!colNetParams[param_id].DiffExists)
                return;

            double dfAccumNormalization = 1.0 / m_param.iter_size;
            m_cuda.scal(colNetParams[param_id].count(), dfAccumNormalization, colNetParams[param_id].mutable_gpu_diff);
        }

        /// <summary>
        /// Regularize a learnable Blob of the training net.
        /// </summary>
        /// <param name="param_id">Specifies the id of the Blob.</param>
        public virtual void Regularize(int param_id)
        {
            BlobCollection<T> colNetParams = m_net.learnable_parameters;

            if (!colNetParams[param_id].DiffExists)
                return;

            List<double?> rgNetParamWeightDecay = m_net.params_weight_decay;
            double dfWeightDecay = m_param.weight_decay;
            double dfLocalDecay = dfWeightDecay * rgNetParamWeightDecay[param_id].GetValueOrDefault(0);

            if (dfLocalDecay > 0)
            {
                switch (m_param.regularization_type)
                {
                    case "L2":
                        // add weight decay
                        m_cuda.axpy(colNetParams[param_id].count(), dfLocalDecay, colNetParams[param_id].gpu_data, colNetParams[param_id].mutable_gpu_diff);
                        break;

                    case "L1":
                        m_cuda.sign(colNetParams[param_id].count(), colNetParams[param_id].gpu_data, m_colTemp[param_id].mutable_gpu_data);
                        m_cuda.axpy(colNetParams[param_id].count(), dfLocalDecay, m_colTemp[param_id].gpu_data, colNetParams[param_id].mutable_gpu_diff);
                        break;
                }
            }
        }

        /// <summary>
        /// Compute the SGD update value that will be applied to a learnable blobs in the training Net.
        /// </summary>
        /// <param name="param_id">Specifies the id of the Blob.</param>
        /// <param name="dfRate">Specifies the learning rate.</param>
        /// <param name="nIterationOverride">Optionally, specifies an iteration override, or -1 which is ignored.</param>
        public virtual void ComputeUpdateValue(int param_id, double dfRate, int nIterationOverride = -1)
        {
            BlobCollection<T> colNetParams = m_net.learnable_parameters;

            if (!colNetParams[param_id].DiffExists)
                return;

            List<double?> net_params_lr = m_net.params_lr;
            T fMomentum = Utility.ConvertVal<T>(m_param.momentum);
            T fLocalRate = Utility.ConvertVal<T>(dfRate * net_params_lr[param_id].GetValueOrDefault(0));

            // Compute the update to history, then copy it to the parameter diff.
            if (m_colHistory != null)
                m_cuda.sgd_update(colNetParams[param_id].count(), colNetParams[param_id].mutable_gpu_diff, m_colHistory[param_id].mutable_gpu_data, fMomentum, fLocalRate);
        }

        /// <summary>
        /// Clip the gradients of all learnable blobs in the training Net.
        /// </summary>
        public virtual void ClipGradients()
        {
            double dfClipGradients = m_param.clip_gradients;

            if (dfClipGradients < 0)
                return;

            BlobCollection<T> colNetParams = m_net.learnable_parameters;
            double dfSumsqDiff = 0;

            for (int i = 0; i < colNetParams.Count; i++)
            {
                if (colNetParams[i].DiffExists)
                    dfSumsqDiff += Utility.ConvertVal<T>(colNetParams[i].sumsq_diff());
            }

            double dfL2NormDiff = Math.Sqrt(dfSumsqDiff);

            if (dfL2NormDiff > dfClipGradients)
            {
                double dfScaleFactor = dfClipGradients / dfL2NormDiff;

                if (m_param.enable_clip_gradient_status)
                    m_log.WriteLine("Gradient clipping: scaling down gradients (L2 norm " + dfL2NormDiff.ToString() + " > " + dfClipGradients.ToString() + ") by scale factor " + dfScaleFactor.ToString());

                for (int i = 0; i < colNetParams.Count; i++)
                {
                    if (colNetParams[i].DiffExists)
                        colNetParams[i].scale_diff(Utility.ConvertVal<T>(dfScaleFactor));
                }
            }
        }
    }
}
