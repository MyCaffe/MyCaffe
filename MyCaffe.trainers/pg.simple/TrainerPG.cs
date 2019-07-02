using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.fillers;
using MyCaffe.layers;
using MyCaffe.param;
using MyCaffe.solvers;

namespace MyCaffe.trainers.pg.simple
{
    /// <summary>
    /// The TrainerPG implements a simple Policy Gradient trainer inspired by Andrej Karpathy's blog posed referenced. 
    /// </summary>
    /// @see 1. [Deep Reinforcement Learning: Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/), by Andrej Karpathy, 2016, Github.io
    /// @see 2. [GitHub: karpathy/pg-pong.py](https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5), by Andrej Karpathy, 2016, Github
    /// @see 3. [CS231n Convolution Neural Networks for Visual Recognition](http://cs231n.github.io/neural-networks-2/#losses) by Karpathy, Stanford
    /// @see 4. [MyCaffe: A Complete C# Re-Write of Caffe with Reinforcement Learning](https://arxiv.org/abs/1810.02272) by D. Brown, 2018, arXiv
    /// <remarks></remarks>
    public class TrainerPG<T> : IxTrainerRL, IDisposable
    {
        IxTrainerCallback m_icallback;
        CryptoRandom m_random = new CryptoRandom(true);
        MyCaffeControl<T> m_mycaffe;
        PropertySet m_properties;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="mycaffe">Specifies the MyCaffeControl to use for learning and prediction.</param>
        /// <param name="properties">Specifies the property set containing the key/value pairs of property settings.</param>
        /// <param name="random">Specifies a Random number generator used for random selection.</param>
        /// <param name="icallback">Specifies the callback for parent notifications and queries.</param>
        public TrainerPG(MyCaffeControl<T> mycaffe, PropertySet properties, CryptoRandom random, IxTrainerCallback icallback)
        {
            m_icallback = icallback;
            m_mycaffe = mycaffe;
            m_properties = properties;
            m_random = random;
        }

        /// <summary>
        /// Releases all resources used.
        /// </summary>
        public void Dispose()
        {
        }

        /// <summary>
        /// Initialize the trainer.
        /// </summary>
        /// <returns>Returns <i>true</i>.</returns>
        public bool Initialize()
        {
            m_mycaffe.CancelEvent.Reset();
            m_icallback.OnInitialize(new InitializeArgs(m_mycaffe));
            return true;
        }

        private void wait(int nWait)
        {
            int nWaitInc = 250;
            int nTotalWait = 0;

            while (nTotalWait < nWait)
            {
                m_icallback.OnWait(new WaitArgs(nWaitInc));
                nTotalWait += nWaitInc;
            }
        }

        /// <summary>
        /// Shutdown the trainer.
        /// </summary>
        /// <param name="nWait">Specifies a wait in ms. for the shutdown to complete.</param>
        /// <returns>Returns <i>true</i>.</returns>
        public bool Shutdown(int nWait)
        {
            if (m_mycaffe != null)
            {
                m_mycaffe.CancelEvent.Set();
                wait(nWait);
            }

            m_icallback.OnShutdown();

            return true;
        }

        /// <summary>
        /// <summary>
        /// Run a single cycle on the environment after the delay.
        /// </summary>
        /// <param name="nDelay">Specifies a delay to wait before running.</param>
        /// <returns>The results of the run containing the action are returned.</returns>
        public ResultCollection RunOne(int nDelay = 1000)
        {
            m_mycaffe.CancelEvent.Reset();
            Agent<T> agent = new Agent<T>(m_icallback, m_mycaffe, m_properties, m_random, Phase.TRAIN);
            agent.Run(Phase.TEST, 1, ITERATOR_TYPE.ITERATION);
            agent.Dispose();
            return null;
        }

        /// <summary>
        /// Run a set of iterations and return the resuts.
        /// </summary>
        /// <param name="nN">Specifies the number of samples to run.</param>
        /// <param name="strRunProperties">Optionally specifies properties to use when running.</param>
        /// <param name="type">Returns the data type contained in the byte stream.</param>
        /// <returns>The results of the run containing the action are returned as a byte stream.</returns>
        public byte[] Run(int nN, string strRunProperties, out string type)
        {
            m_mycaffe.CancelEvent.Reset();
            Agent<T> agent = new Agent<T>(m_icallback, m_mycaffe, m_properties, m_random, Phase.RUN);
            byte[] rgResults = agent.Run(nN, out type);
            agent.Dispose();

            return rgResults;
        }

        /// <summary>
        /// Run the test cycle - currently this is not implemented.
        /// </summary>
        /// <param name="nN">Specifies the number of iterations (based on the ITERATION_TYPE) to run, or -1 to ignore.</param>
        /// <param name="type">Specifies the iteration type (default = ITERATION).</param>
        /// <returns>A value of <i>true</i> is returned when handled, <i>false</i> otherwise.</returns>
        public bool Test(int nN, ITERATOR_TYPE type)
        {
            int nDelay = 1000;
            string strProp = m_properties.ToString();

            // Turn off the num-skip to run at normal speed.
            strProp += "EnableNumSkip=False;";
            PropertySet properties = new PropertySet(strProp);

            m_mycaffe.CancelEvent.Reset();
            Agent<T> agent = new Agent<T>(m_icallback, m_mycaffe, properties, m_random, Phase.TRAIN);
            agent.Run(Phase.TEST, nN, type);

            agent.Dispose();
            Shutdown(nDelay);

            return true;
        }

        /// <summary>
        /// Train the network using a modified PG training algorithm optimized for GPU use.
        /// </summary>
        /// <param name="nN">Specifies the number of iterations (based on the ITERATION_TYPE) to run, or -1 to ignore.</param>
        /// <param name="type">Specifies the iteration type (default = ITERATION).</param>
        /// <param name="step">Specifies the stepping mode to use (when debugging).</param>
        /// <returns>A value of <i>true</i> is returned when handled, <i>false</i> otherwise.</returns>
        public bool Train(int nN, ITERATOR_TYPE type, TRAIN_STEP step)
        {
            if (step != TRAIN_STEP.NONE)
                throw new Exception("The simple traininer does not support stepping - use the 'PG.MT' trainer instead.");

            m_mycaffe.CancelEvent.Reset();
            Agent<T> agent = new Agent<T>(m_icallback, m_mycaffe, m_properties, m_random, Phase.TRAIN);
            agent.Run(Phase.TRAIN, nN, type);
            agent.Dispose();

            return false;
        }
    }

    class Agent<T> : IDisposable /** @private */
    {
        IxTrainerCallback m_icallback;
        Brain<T> m_brain;
        PropertySet m_properties;
        CryptoRandom m_random;
        float m_fGamma;
        bool m_bAllowDiscountReset = false;
        bool m_bUseRawInput = false;

        public Agent(IxTrainerCallback icallback, MyCaffeControl<T> mycaffe, PropertySet properties, CryptoRandom random, Phase phase)
        {
            m_icallback = icallback;
            m_brain = new Brain<T>(mycaffe, properties, random, phase);
            m_properties = properties;
            m_random = random;

            m_fGamma = (float)properties.GetPropertyAsDouble("Gamma", 0.99);
            m_bAllowDiscountReset = properties.GetPropertyAsBool("AllowDiscountReset", false);
            m_bUseRawInput = properties.GetPropertyAsBool("UseRawInput", false);
        }

        public void Dispose()
        {
            if (m_brain != null)
            {
                m_brain.Dispose();
                m_brain = null;
            }
        }

        private StateBase getData(Phase phase, int nAction)
        {
            GetDataArgs args = m_brain.getDataArgs(phase, nAction);
            m_icallback.OnGetData(args);
            return args.State;
        }

        private void updateStatus(int nIteration, int nEpisodeCount, double dfRewardSum, double dfRunningReward)
        {
            GetStatusArgs args = new GetStatusArgs(0, nIteration, nEpisodeCount, 1000000, dfRunningReward, dfRewardSum, 0, 0, 0, 0);
            m_icallback.OnUpdateStatus(args);
        }

        public byte[] Run(int nIterations, out string type)
        {
            IxTrainerCallbackRNN icallback = m_icallback as IxTrainerCallbackRNN;
            if (icallback == null)
                throw new Exception("The Run method requires an IxTrainerCallbackRNN interface to convert the results into the native format!");

            StateBase s = getData(Phase.RUN, -1);
            int nIteration = 0;
            List<float> rgResults = new List<float>();

            while (!m_brain.Cancel.WaitOne(0) && (nIterations == -1 || nIteration < nIterations))
            {
                // Preprocess the observation.
                SimpleDatum x = m_brain.Preprocess(s, m_bUseRawInput);

                // Forward the policy network and sample an action.
                float fAprob;
                int action = m_brain.act(x, out fAprob);

                rgResults.Add(s.Data.TimeStamp.ToFileTime());
                rgResults.Add((float)s.Data.RealData[0]);
                rgResults.Add(action);

                // Take the next step using the action
                StateBase s_ = getData(Phase.RUN, action);
                nIteration++;
            }

            ConvertOutputArgs args = new ConvertOutputArgs(nIterations, rgResults.ToArray());
            icallback.OnConvertOutput(args);

            type = args.RawType;
            return args.RawOutput;
        }

        private bool isAtIteration(int nN, ITERATOR_TYPE type, int nIteration, int nEpisode)
        {
            if (nN == -1)
                return false;

            if (type == ITERATOR_TYPE.EPISODE)
            {
                if (nEpisode < nN)
                    return false;

                return true;
            }
            else
            {
                if (nIteration < nN)
                    return false;

                return true;
            }
        }

        /// <summary>
        /// The Run method provides the main 'actor' loop that performs the following steps:
        /// 1.) get state
        /// 2.) build experience
        /// 3.) create policy gradients
        /// 4.) train on experiences
        /// </summary>
        /// <param name="phase">Specifies the phae.</param>
        /// <param name="nN">Specifies the number of iterations (based on the ITERATION_TYPE) to run, or -1 to ignore.</param>
        /// <param name="type">Specifies the iteration type (default = ITERATION).</param>
        public void Run(Phase phase, int nN, ITERATOR_TYPE type)
        {
            MemoryCollection m_rgMemory = new MemoryCollection();
            double? dfRunningReward = null;
            double dfEpisodeReward = 0;
            int nEpisode = 0;
            int nIteration = 0;

            StateBase s = getData(phase, -1);

            if (s.Clip != null)
                throw new Exception("The PG.SIMPLE trainer does not support recurrent layers or clip data, use the 'PG.ST' or 'PG.MT' trainer instead.");

            while (!m_brain.Cancel.WaitOne(0) && !isAtIteration(nN, type, nIteration, nEpisode))
            {
                // Preprocess the observation.
                SimpleDatum x = m_brain.Preprocess(s, m_bUseRawInput);

                // Forward the policy network and sample an action.
                float fAprob;
                int action = m_brain.act(x, out fAprob);

                // Take the next step using the action
                StateBase s_ = getData(phase, action);
                dfEpisodeReward += s_.Reward;

                if (phase == Phase.TRAIN)
                {
                    // Build up episode memory, using reward for taking the action.
                    m_rgMemory.Add(new MemoryItem(s, x, action, fAprob, (float)s_.Reward));

                    // An episode has finished.
                    if (s_.Done)
                    {
                        nEpisode++;
                        nIteration++;

                        m_brain.Reshape(m_rgMemory);

                        // Compute the discounted reward (backwards through time)
                        float[] rgDiscountedR = m_rgMemory.GetDiscountedRewards(m_fGamma, m_bAllowDiscountReset);
                        // Rewards are standardized when set to be unit normal (helps control the gradient estimator variance)
                        m_brain.SetDiscountedR(rgDiscountedR);

                        // Modulate the gradient with the advantage (PG magic happens right here.)
                        float[] rgDlogp = m_rgMemory.GetPolicyGradients();
                        // discounted R applied to policy agradient within loss function, just before the backward pass.
                        m_brain.SetPolicyGradients(rgDlogp);

                        // Train for one iteration, which triggers the loss function.
                        List<Datum> rgData = m_rgMemory.GetData();
                        m_brain.SetData(rgData);
                        m_brain.Train(nIteration);

                        // Update reward running
                        if (!dfRunningReward.HasValue)
                            dfRunningReward = dfEpisodeReward;
                        else
                            dfRunningReward = dfRunningReward * 0.99 + dfEpisodeReward * 0.01;

                        updateStatus(nIteration, nEpisode, dfEpisodeReward, dfRunningReward.Value);
                        dfEpisodeReward = 0;

                        s = getData(phase, -1);
                        m_rgMemory.Clear();
                    }
                    else
                    {
                        s = s_;
                    }
                }
                else
                {
                    if (s_.Done)
                    {
                        nEpisode++;

                        // Update reward running
                        if (!dfRunningReward.HasValue)
                            dfRunningReward = dfEpisodeReward;
                        else
                            dfRunningReward = dfRunningReward * 0.99 + dfEpisodeReward * 0.01;

                        updateStatus(nIteration, nEpisode, dfEpisodeReward, dfRunningReward.Value);
                        dfEpisodeReward = 0;

                        s = getData(phase, -1);
                    }
                    else
                    {
                        s = s_;
                    }

                    nIteration++;
                }
            }
        }
    }

    class Brain<T> : IDisposable /** @private */
    {
        MyCaffeControl<T> m_mycaffe;
        Net<T> m_net;
        Solver<T> m_solver;
        MemoryDataLayer<T> m_memData;
        MemoryLossLayer<T> m_memLoss;
        PropertySet m_properties;
        CryptoRandom m_random;
        Blob<T> m_blobDiscountedR;
        Blob<T> m_blobPolicyGradient;
        bool m_bSkipLoss;
        int m_nMiniBatch = 10;
        SimpleDatum m_sdLast = null;
 
        public Brain(MyCaffeControl<T> mycaffe, PropertySet properties, CryptoRandom random, Phase phase)
        {
            m_mycaffe = mycaffe;
            m_net = mycaffe.GetInternalNet(phase);
            m_solver = mycaffe.GetInternalSolver();
            m_properties = properties;
            m_random = random;

            m_memData = m_net.FindLayer(LayerParameter.LayerType.MEMORYDATA, null) as MemoryDataLayer<T>;
            m_memLoss = m_net.FindLayer(LayerParameter.LayerType.MEMORY_LOSS, null) as MemoryLossLayer<T>;
            SoftmaxLayer<T> softmax = m_net.FindLayer(LayerParameter.LayerType.SOFTMAX, null) as SoftmaxLayer<T>;

            if (softmax != null)
                throw new Exception("The PG.SIMPLE trainer does not support the Softmax layer, use the 'PG.ST' or 'PG.MT' trainer instead.");

            if (m_memData == null)
                throw new Exception("Could not find the MemoryData Layer!");

            if (m_memLoss == null)
                throw new Exception("Could not find the MemoryLoss Layer!");

            m_memLoss.OnGetLoss += memLoss_OnGetLoss;

            m_blobDiscountedR = new Blob<T>(mycaffe.Cuda, mycaffe.Log);
            m_blobPolicyGradient = new Blob<T>(mycaffe.Cuda, mycaffe.Log);

            int nMiniBatch = mycaffe.CurrentProject.GetBatchSize(phase);
            if (nMiniBatch != 0)
                m_nMiniBatch = nMiniBatch;

            m_nMiniBatch = m_properties.GetPropertyAsInt("MiniBatch", m_nMiniBatch);
        }

        private void dispose(ref Blob<T> b)
        {
            if (b != null)
            {
                b.Dispose();
                b = null;
            }
        }

        public void Dispose()
        {
            m_memLoss.OnGetLoss -= memLoss_OnGetLoss;
            dispose(ref m_blobDiscountedR);
            dispose(ref m_blobPolicyGradient);
        }

        public void Reshape(MemoryCollection col)
        {
            int nNum = col.Count;
            int nChannels = col[0].Data.Channels;
            int nHeight = col[0].Data.Height;
            int nWidth = col[0].Data.Height;

            m_blobDiscountedR.Reshape(nNum, 1, 1, 1);
            m_blobPolicyGradient.Reshape(nNum, 1, 1, 1);
        }

        public void SetDiscountedR(float[] rg)
        {
            double dfMean = m_blobDiscountedR.mean(rg);
            double dfStd = m_blobDiscountedR.std(dfMean, rg);
            m_blobDiscountedR.SetData(Utility.ConvertVec<T>(rg));
            m_blobDiscountedR.NormalizeData(dfMean, dfStd);
        }

        public void SetPolicyGradients(float[] rg)
        {
            m_blobPolicyGradient.SetData(Utility.ConvertVec<T>(rg));
        }

        public void SetData(List<Datum> rgData)
        {
            m_memData.AddDatumVector(rgData, null, 1, true, true);
        }

        public GetDataArgs getDataArgs(Phase phase, int nAction)
        {
            bool bReset = (nAction == -1) ? true : false;
            return new GetDataArgs(phase, 0, m_mycaffe, m_mycaffe.Log, m_mycaffe.CancelEvent, bReset, nAction, false);
        }

        public Log Log
        {
            get { return m_mycaffe.Log; }
        }

        public CancelEvent Cancel
        {
            get { return m_mycaffe.CancelEvent; }
        }

        public SimpleDatum Preprocess(StateBase s, bool bUseRawInput)
        {
            SimpleDatum sd = new SimpleDatum(s.Data, true);

            if (bUseRawInput)
                return sd;

            if (m_sdLast == null)
                sd.Zero();
            else
                sd.Sub(m_sdLast);

            m_sdLast = s.Data;

            return sd;
        }

        public int act(SimpleDatum sd, out float fAprob)
        {
            List<Datum> rgData = new List<Datum>();
            rgData.Add(new Datum(sd));
            double dfLoss;

            m_memData.AddDatumVector(rgData, null, 1, true, true);
            m_bSkipLoss = true;
            BlobCollection<T> res = m_net.Forward(out dfLoss);
            m_bSkipLoss = false;
            float[] rgfAprob = null;

            for (int i = 0; i < res.Count; i++)
            {
                if (res[i].type != Blob<T>.BLOB_TYPE.LOSS)
                {
                    rgfAprob = Utility.ConvertVecF<T>(res[i].update_cpu_data());
                    break;
                }
            }

            if (rgfAprob == null)
                throw new Exception("Could not find a non-loss output!  Your model should output the loss and the action probabilities.");

            if (rgfAprob.Length != 1)
                throw new Exception("The simple policy gradient only supports a single data output!");

            fAprob = rgfAprob[0];

            // Roll the dice!
            if (m_random.NextDouble() < (double)fAprob)
                return 0;
            else
                return 1;
        }

        public void Train(int nIteration)
        {
            m_mycaffe.Log.Enable = false;
            m_solver.Step(1, TRAIN_STEP.NONE, false, false, true, true);  // accumulate grad over batch

            if (nIteration % m_nMiniBatch == 0)
            {
                m_solver.ApplyUpdate(nIteration);
                m_net.ClearParamDiffs();
            }

            m_mycaffe.Log.Enable = true;
        }

        private void memLoss_OnGetLoss(object sender, MemoryLossLayerGetLossArgs<T> e)
        {
            if (m_bSkipLoss)
                return;

            int nCount = m_blobPolicyGradient.count();
            long hPolicyGrad = m_blobPolicyGradient.mutable_gpu_data;
            long hBottomDiff = e.Bottom[0].mutable_gpu_diff;
            long hDiscountedR = m_blobDiscountedR.gpu_data;

            // Calculate the actual loss.
            double dfSumSq = Utility.ConvertVal<T>(m_blobPolicyGradient.sumsq_data());
            double dfMean = dfSumSq;

            e.Loss = dfMean;
            e.EnableLossUpdate = false; // apply gradients to bottom directly.

            // Modulate the gradient with the advantage (PG magic happens right here.)
            m_mycaffe.Cuda.mul(nCount, hPolicyGrad, hDiscountedR, hPolicyGrad);
            m_mycaffe.Cuda.copy(nCount, hPolicyGrad, hBottomDiff);
            m_mycaffe.Cuda.mul_scalar(nCount, -1.0, hBottomDiff);
        }
    }

    class MemoryCollection : GenericList<MemoryItem> /** @private */
    {
        public MemoryCollection()
        {
        }

        public float[] GetDiscountedRewards(float fGamma, bool bAllowReset)
        {
            float fRunningAdd = 0;
            float[] rgR = m_rgItems.Select(p => p.Reward).ToArray();
            float[] rgDiscountedR = new float[rgR.Length];

            for (int t = Count - 1; t >= 0; t--)
            {
                if (bAllowReset && rgR[t] != 0)
                    fRunningAdd = 0;

                fRunningAdd = fRunningAdd * fGamma + rgR[t];
                rgDiscountedR[t] = fRunningAdd;
            }

            return rgDiscountedR;
        }

        public float[] GetPolicyGradients()
        {
            return m_rgItems.Select(p => p.dlogps).ToArray();
        }

        public List<Datum> GetData()
        {
            List<Datum> rgData = new List<Datum>();

            for (int i = 0; i < m_rgItems.Count; i++)
            {
                rgData.Add(new Datum(m_rgItems[i].Data));
            }

            return rgData;
        }

        public List<Datum> GetClip()
        {
            return null;
        }
    }

    class MemoryItem /** @private */
    {
        StateBase m_state;
        SimpleDatum m_x;
        int m_nAction;
        float m_fAprob;
        float m_fReward;

        public MemoryItem(StateBase s, SimpleDatum x, int nAction, float fAprob, float fReward)
        {
            m_state = s;
            m_x = x;
            m_nAction = nAction;
            m_fAprob = fAprob;
            m_fReward = fReward;
        }

        public StateBase State
        {
            get { return m_state; }
        }

        public SimpleDatum Data
        {
            get { return m_x; }
        }

        public int Action
        {
            get { return m_nAction; }
        }

        public float Reward
        {
            get { return m_fReward; }
        }

        /// <summary>
        /// Gradient that encourages the action that was taken to be taken.
        /// </summary>
        /// <remarks>
        /// @see [CS231n Convolution Neural Networks for Visual Recognition](http://cs231n.github.io/neural-networks-2/#losses) by Karpathy, Stanford
        /// </remarks>
        public float dlogps
        {
            get
            {
                float fY = 0;

                if (m_nAction == 0)
                    fY = 1;

                return fY - m_fAprob;
            }
        }

        public override string ToString()
        {
            return "action = " + m_nAction.ToString() + " reward = " + m_fReward.ToString("N2") + " aprob = " + m_fAprob.ToString("N5") + " dlogps = " + dlogps.ToString("N5");
        }
    }
}
