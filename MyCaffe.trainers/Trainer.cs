using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.layers;
using MyCaffe.param;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.trainers
{
    /// <summary>
    /// The TrainerA2C implements the Advantage, Actor-Critic Reinforcement Learning algorithm.
    /// </summary>
    /// <remarks>
    /// @see 1. [Massively Parallel Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1507.04296) by A. Nair, P. Srinivasan, S. Blackwell, C. Alcicek, R. Fearon, A. De Maria, V. Panneershelvam, M. Suleyman, C. Beattie, S. Petersen, S. Legg, V. Mnih, K. Kavukcuoglu and D. Silver, 2015, arXiv:1507.04296
    /// @see 2. [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) by V. Mnih, A. P. Badia, M. Mirza, A. Graves, T. P. Lillicrap, D. Silver and K. Kavukcuoglu, 2016, arXiv:1602.01783
    /// @see 3. [GitHub: (A3C) universe-starter-agent](https://github.com/openai/universe-starter-agent) by OpenAi, 2016, GitHub
    /// @see 4. [Simple Reinforcement Learning with Tensorflow Part 8: Asynchronous Actor-Critic Agents (A3C)](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2) by Arthur Juliani, 2016, Medium 
    /// @see 5. [GitHub: Simple implementation of Reinforcement Learning (A3C) using Pytorch](https://github.com/MorvanZhou/pytorch-A3C) by MorvanZhou, 2018, GitHub
    /// @see 6. [Deep Reinforcement Learning: Playing CartPole through Asynchronous Advantage Actor Critic (A3C) with tf.keras and eager execution](https://medium.com/tensorflow/deep-reinforcement-learning-playing-cartpole-through-asynchronous-advantage-actor-critic-a3c-7eab2eea5296) by R.Yuan, 2018, Medium 
    /// </remarks>
    public class Trainer<T> : IxTrainer, IDisposable 
    {
        int m_nIndex = 0;
        /// <summary>
        /// Specifies the output log used for general text based output.
        /// </summary>
        protected Log m_log;
        /// <summary>
        /// Specifies the cancellation event.
        /// </summary>
        protected CancelEvent m_evtCancel;
        /// <summary>
        /// Specifies the unserlying MyCaffeControl with the open project.
        /// </summary>
        protected MyCaffeControl<T> m_caffe;
        /// <summary>
        /// Specifies the mini batch size to use for the maximum number of items in the experience memory - this is defined by the MemoryDataParameter.
        /// </summary>
        protected int m_nMiniBatchSize = 1;
        /// <summary>
        /// Specifies the maximum number of episodes within each experience.
        /// </summary>
        protected int m_nMaxEpisodeSteps = 1;
        /// <summary>
        /// Specifies the general properties initialized from the key-value pair within the string sent to Initialize.
        /// </summary>
        protected PropertySet m_properties;
        /// <summary>
        /// Specifies the operating mode A2C (single-trainer) or A3C (multi-trainer)
        /// </summary>
        protected TRAINING_MODE m_mode = TRAINING_MODE.A2C;
        private Random m_random;
        private double m_dfGamma; // discount factor
        private double m_dfBeta; // percent of entropy to use.
        private Memory<T> m_memory = new Memory<T>();
        private Blob<T> m_blobInput;
        private Blob<T> m_blobAdvantage;
        private Blob<T> m_blobValueLoss;
        private Blob<T> m_blobEntropy;
        private Blob<T> m_blobPolicy;
        private Blob<T> m_blobActionOneHot;
        private Blob<T> m_blobPolicyLoss;
        private Blob<T> m_blobLoss;
        private Layer<T> m_softmax;
        private Layer<T> m_crossentropy;
        private CudaDnn<T> m_cuda;
        private MyCaffeControl<T> m_local;
        private MyCaffeControl<T> m_global;
        double m_dfLocalLearningRate = 0;
        int m_nLastBatchSize = 0;
        bool m_bSoftMaxSetup = false;
        bool m_bCrossEntropySetup = false;
        double m_dfExplorationPct = 0.2;
        int m_nGlobalEpExplorationStep = 100;
        double m_dfExplorationStepDownFactor = 0.75;

        /// <summary>
        /// The OnIntialize event fires when initializing the trainer.
        /// </summary>
        public event EventHandler<InitializeArgs> OnInitialize;
        /// <summary>
        /// The OnGetData event fires from within the Train method and is used to get a new observation data.
        /// </summary>
        public event EventHandler<GetDataArgs> OnGetData;
        /// <summary>
        /// The OnGetGlobalEpisodeCount event fires when the episode processing loop needs to retrieve the global episode count.
        /// </summary>
        public event EventHandler<GlobalEpisodeCountArgs> OnGetGlobalEpisodeCount;
        /// <summary>
        /// The OnUpdateGlobalRewards event fires to add local rewards to the global rewards.
        /// </summary>
        public event EventHandler<UpdateGlobalRewardArgs> OnUpdateGlobalRewards;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="mycaffe">Specifies the MyCaffeControl with an open project.</param>
        /// <param name="log">Specifies the output log.</param>
        /// <param name="evtCancel">Specifies the cancellation event.</param>
        /// <param name="properties">Specifies the set of key-value properties to use.</param>
        /// <param name="mode">Optionally, specifies either the A2C (single trainer) or A3C (multi-trainer) mode.</param>
        /// <param name="nGpuID">Optionally, specifies the GPU to use (default = 0)</param>
        /// <param name="nIndex">Optionally, specifies the index of this trainer.</param>
        public Trainer(MyCaffeControl<T> mycaffe, Log log, CancelEvent evtCancel, PropertySet properties, TRAINING_MODE mode = TRAINING_MODE.A2C, int nGpuID = 0, int nIndex = 0)
        {
            m_nIndex = nIndex;
            m_random = new Random();
            m_properties = properties;
            m_caffe = mycaffe;
            m_log = log;
            m_evtCancel = evtCancel;
            m_nMiniBatchSize = m_caffe.CurrentProject.GetBatchSize(Phase.TRAIN);
            m_nLastBatchSize = m_nMiniBatchSize;
            m_dfExplorationPct = m_properties.GetPropertyAsDouble("ExplorationPercent", 0.2);
            m_nMaxEpisodeSteps = m_properties.GetPropertyAsInt("MaxEpisodeSteps", 200);
            m_dfGamma = m_properties.GetPropertyAsDouble("Gamma", 0.99);
            m_dfBeta = m_properties.GetPropertyAsDouble("Beta", 0.01);
            m_nGlobalEpExplorationStep = m_properties.GetPropertyAsInt("GlobalExplorationStep", 100);
            m_dfExplorationStepDownFactor = m_properties.GetPropertyAsDouble("ExplorationStepDownFactor", 0.75);

            int? nTestIter = m_caffe.CurrentProject.GetSolverSettingAsInt("test_iter");
            m_log.CHECK(!nTestIter.HasValue, "There should be not 'test_iter' to turn off testing.");

            int? nVal = m_caffe.CurrentProject.GetSolverSettingAsInt("test_interval");
            if (nVal.HasValue)
                m_log.CHECK_EQ(nVal.Value, 0, "The solver 'test_interval' must be 0 to turn off testing.");

            bool? bVal = m_caffe.CurrentProject.GetSolverSettingAsBool("test_initialization");
            if (bVal.HasValue)
                m_log.CHECK(!bVal.Value, "The solver 'test_initialization' must be False to turn off testing.");

            m_caffe.EnableTesting = false;

            m_log.CHECK_GT(m_nMaxEpisodeSteps, 0, "The maximum episode steps (" + m_nMaxEpisodeSteps.ToString() + ") must be greater than zero.");
            m_log.CHECK_LT(m_nMiniBatchSize, m_nMaxEpisodeSteps, "The mini-batch size (" + m_nMiniBatchSize.ToString() + ") must be less than or equal to the MaxEpisodeSteps(" + m_nMaxEpisodeSteps.ToString() + ")");
            m_log.CHECK_GE(m_dfGamma, 0, "The gamma value of " + m_dfGamma.ToString() + " (discount factor) must be equal to zero or greater.");
            m_log.CHECK_LE(m_dfGamma, 1.0, "The gamma value of " + m_dfGamma.ToString() + " (discount factor) must be less than or equal to 1.0");
            m_log.CHECK_GE(m_dfBeta, 0, "The beta value of " + m_dfBeta.ToString() + " (percent of entropy to use) must be equal to zero or greater.");
            m_log.CHECK_LE(m_dfBeta, 1.0, "The beta value of " + m_dfBeta.ToString() + " (percent of entropy to use) must be less than or equal to 1.0");

            getNets(nGpuID, mode, out m_local, out m_global, out m_dfLocalLearningRate);
            m_cuda = m_local.Cuda;

            m_blobInput = new Blob<T>(m_cuda, m_log, false);
            m_blobAdvantage = new Blob<T>(m_cuda, m_log, false);
            m_blobValueLoss = new Blob<T>(m_cuda, m_log, false);
            m_blobPolicy = new Blob<T>(m_cuda, m_log, true);
            m_blobEntropy = new Blob<T>(m_cuda, m_log, false);
            m_blobActionOneHot = new Blob<T>(m_cuda, m_log, true);
            m_blobPolicyLoss = new Blob<T>(m_cuda, m_log, true);
            m_blobLoss = new Blob<T>(m_cuda, m_log, true);

            LayerParameter p1 = new LayerParameter(LayerParameter.LayerType.SOFTMAX);
            p1.softmax_param.axis = 1;
            m_softmax = Layer<T>.Create(m_cuda, m_log, p1, m_evtCancel);

            LayerParameter p2 = new LayerParameter(LayerParameter.LayerType.SOFTMAXCROSSENTROPY_LOSS);
            p2.softmax_param.axis = 1;
            p2.loss_weight.Add(1);
            p2.loss_weight.Add(1);
            p2.loss_param.normalization = LossParameter.NormalizationMode.BATCH_SIZE;
            m_crossentropy = Layer<T>.Create(m_cuda, m_log, p2, m_evtCancel);
        }

        /// <summary>
        /// Releases all resources used.
        /// </summary>
        public void Dispose()
        {
            if (m_blobInput != null)
            {
                m_blobInput.Dispose();
                m_blobInput = null;
            }

            if (m_blobAdvantage != null)
            {
                m_blobAdvantage.Dispose();
                m_blobAdvantage = null;
            }

            if (m_blobValueLoss != null)
            {
                m_blobValueLoss.Dispose();
                m_blobValueLoss = null;
            }

            if (m_blobPolicy != null)
            {
                m_blobPolicy.Dispose();
                m_blobPolicy = null;
            }

            if (m_blobEntropy != null)
            {
                m_blobEntropy.Dispose();
                m_blobEntropy = null;
            }

            if (m_blobActionOneHot != null)
            {
                m_blobActionOneHot.Dispose();
                m_blobActionOneHot = null;
            }

            if (m_blobPolicyLoss != null)
            {
                m_blobPolicyLoss.Dispose();
                m_blobPolicyLoss = null;
            }

            if (m_blobLoss != null)
            {
                m_blobLoss.Dispose();
                m_blobLoss = null;
            }

            if (m_softmax != null)
            {
                m_softmax.Dispose();
                m_softmax = null;
            }

            if (m_crossentropy != null)
            {
                m_crossentropy.Dispose();
                m_crossentropy = null;
            }

            if (m_mode == TRAINING_MODE.A3C)
            {
                m_local.Dispose();
                m_local = null;
            }
        }

        /// <summary>
        /// Returns the index of this trainer.
        /// </summary>
        public int Index
        {
            get { return m_nIndex; }
        }

        private void getNets(int nGpuID, TRAINING_MODE mode, out MyCaffeControl<T> local, out MyCaffeControl<T> global, out double dfLocalLr)
        {
            m_mode = mode;
            dfLocalLr = -1;
            local = m_caffe;
            global = null;

            if (mode == TRAINING_MODE.A3C)
            {
                global = m_caffe;
                local = global.Clone(nGpuID);
            }

            Net<T> net = local.GetInternalNet(Phase.TRAIN);
            MemoryLossLayer<T> memLoss = getLayer(net, LayerParameter.LayerType.MEMORY_LOSS) as MemoryLossLayer<T>;

            if (memLoss == null)
                throw new Exception("Coult not find a MemoryLossLayer in the training net!");

            memLoss.OnGetLoss += MemLoss_OnGetLoss;
        }

        /// <summary>
        /// Initialize the trainer.
        /// </summary>
        /// <returns>On success <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool Initialize()
        {
            if (OnInitialize != null)
            {
                InitializeArgs e = new InitializeArgs(m_caffe, m_nIndex);
                OnInitialize(this, e);
            }

            return true;
        }

        public bool Test(int nIterations)
        {
            return false;
        }

        public bool Train(int nIterations)
        {
            verifyEvents();

            GlobalEpisodeCountArgs globalEpCount = new GlobalEpisodeCountArgs();
            OnGetGlobalEpisodeCount(this, globalEpCount);
            int nGlobalEp = globalEpCount.GlobalEpisodeCount;
            int nGlobalEpMax = globalEpCount.MaximumGlobalEpisodeCount;

            if (nGlobalEpMax < nIterations)
                nGlobalEpMax = nIterations;

            Stopwatch sw = new Stopwatch();
            sw.Start();

            int nEpisodes = 0;
            int nMaxEpisode = 0;
            int nTotalStep = 1;

            //-------------------------------------------------------
            //  The Episode Processing Loop
            //-------------------------------------------------------
            while (nGlobalEp < nGlobalEpMax)
            {
                if (m_evtCancel.WaitOne(0))
                    return false;

                if (sw.Elapsed.TotalMilliseconds > 1000)
                {
                    double dfPct = ((double)nGlobalEp / (double)nGlobalEpMax);
                    m_log.Progress = dfPct;
                    m_log.WriteLine("Episode processing loop at " + dfPct.ToString("P") + "...");
                    sw.Restart();
                }

                if (nGlobalEp % m_nGlobalEpExplorationStep == 0 && nGlobalEp > 0 && m_dfExplorationPct != 0)
                {
                    m_dfExplorationPct *= m_dfExplorationStepDownFactor;
                    Trace.WriteLine("Exploration rate = " + m_dfExplorationPct.ToString("P"));

                    if (m_dfExplorationPct < 0.001)
                        m_dfExplorationPct = 0;
                }

                m_memory = new Memory<T>();
                double dfEpR = 0;
                bool bDone = false;

                // Get the initial state.
                GetDataArgs dataArg = new GetDataArgs(m_local, m_nIndex, true);
                OnGetData(this, dataArg);
                StateBase state = dataArg.State;

                //---------------------------------------------------
                //  The Episode Building Loop
                //---------------------------------------------------
                for (int t=0; t<m_nMaxEpisodeSteps; t++)
                {
                    int nAction = getAction(m_local, state);

                    dataArg = new GetDataArgs(m_local, m_nIndex, false, nAction);
                    OnGetData(this, dataArg);
                    StateBase newState = dataArg.State;
                    bDone = newState.Done;

                    if (t == m_nMaxEpisodeSteps - 1)
                        bDone = true;

                    if (bDone)
                        newState.Reward = -1;

                    dfEpR += newState.Reward;
                    m_memory.Add(new MemoryItem(state, nAction, newState.Reward));

                    if (nTotalStep % m_nMiniBatchSize == 0 || bDone)
                    {
                        double dfVs = (bDone) ? 0 : getValue(m_local, newState);

                        // Calculate the discounted rewards (in reverse)
                        for (int i = m_memory.Count - 1; i >= 0; i--)
                        {
                            dfVs = m_memory[i].Reward + m_dfGamma * dfVs;
                            m_memory[i].Target = dfVs;
                        }

                        // train one iteration on the memory data items.
                        if (addInputData(m_local, m_memory, m_nMiniBatchSize))
                        {
                            m_log.Enable = false;
                            m_local.Train(1, 0, TRAIN_STEP.NONE, m_dfLocalLearningRate);                           
                            setBatchSize(m_local, Phase.TRAIN, m_nMiniBatchSize);
                            m_log.Enable = true;
                        }

                        m_memory = new Memory<T>();
                        updateGlobalNet(m_global, m_local, nGlobalEp);

                        if (bDone)
                        {
                            OnUpdateGlobalRewards(this, new UpdateGlobalRewardArgs(dfEpR));
                            break;
                        }

                        nEpisodes = t;
                    }

                    state = newState;
                    nTotalStep++;
                }

                if (nMaxEpisode < nEpisodes)
                {
                    nMaxEpisode = nEpisodes;
                    Trace.WriteLine("** Max Episode = " + nMaxEpisode.ToString() + " **");
                }

                OnGetGlobalEpisodeCount(this, globalEpCount);
                nGlobalEp = globalEpCount.GlobalEpisodeCount;
            }

            return true;
        }

        /// <summary>
        /// This event fires during the training of the local net.
        /// </summary>
        /// <remarks>
        /// This loss algorithm is mainly based on [3].
        /// </remarks>
        /// <param name="sender">Specifies the instance of the MemoryLossLayer.</param>
        /// <param name="e">Specifies the arguments.</param>
        private void MemLoss_OnGetLoss(object sender, MemoryLossLayerGetLossArgs<T> e)
        {
            Blob<T> blobValues = null;
            Blob<T> blobLogits = null;

            if (e.Bottom.Count > 2)
                throw new Exception("Expected only two bottom values: logits(action_size), values(1)");

            for (int i = 0; i < e.Bottom.Count; i++)
            {
                if (e.Bottom[i].count(1) == 1)
                    blobValues = e.Bottom[i];
                else
                    blobLogits = e.Bottom[i];
            }

            if (blobValues == null)
                throw new Exception("Could not find the values blob collection!");

            m_blobAdvantage.ReshapeLike(blobValues);
            m_blobValueLoss.ReshapeLike(blobValues);
            m_blobEntropy.ReshapeLike(blobValues);
            m_blobPolicyLoss.ReshapeLike(blobValues);
            m_blobLoss.ReshapeLike(blobValues);

            m_blobPolicy.ReshapeLike(blobLogits);
            m_blobActionOneHot.ReshapeLike(blobLogits);

            int nValueCount = blobValues.count();
            long hValues = blobValues.gpu_data;
            long hAdvantage = m_blobAdvantage.mutable_gpu_data;
            long hValLoss = m_blobValueLoss.mutable_gpu_data;
            long hEntropy = m_blobEntropy.mutable_gpu_data;
            long hPolicyLoss = m_blobPolicyLoss.mutable_gpu_data;

            int nLogitCount = blobLogits.count();
            long hPolicy = m_blobPolicy.mutable_gpu_data;

            int nActionCount = nLogitCount / nValueCount;
            BlobCollection<T> colBottom = new BlobCollection<T>();
            BlobCollection<T> colTop = new BlobCollection<T>();

            // Calculate the Advantage 'td'
            m_blobAdvantage.SetData(m_memory.GetTargets(nValueCount));
            m_cuda.sub(nValueCount, hAdvantage, hValues, hAdvantage);

            // Calculate the value loss 'c_loss'
            m_cuda.powx(nValueCount, hAdvantage, 2.0, hValLoss);

            // Calculate the policy loss.
            // -- get the policy 'probs' --
            colBottom.Add(blobLogits);
            colTop.Add(m_blobPolicy);

            if (!m_bSoftMaxSetup)
            {
                m_softmax.Setup(colBottom, colTop);
                m_bSoftMaxSetup = true;
            }

            m_softmax.Forward(colBottom, colTop);

            // -- get the entropy 'exp_v' --
            T[] rgActions = m_memory.GetActions(nValueCount);
            T[] rgLogProb = log_prob(m_blobPolicy, rgActions, nValueCount, nActionCount);
            m_blobEntropy.SetData(rgLogProb);
            m_cuda.mul(nValueCount, hEntropy, hAdvantage, hEntropy);

            // -- calculate the total loss --
            m_cuda.sub(nValueCount, hValLoss, hEntropy, hPolicyLoss);
            double dfAsum = Utility.ConvertVal<T>(m_blobPolicyLoss.asum_data());
            double dfTotalLoss = dfAsum / nValueCount; // mean

            e.application = MemoryLossLayerGetLossArgs<T>.APPLICATION.AS_LOSS_DIRECTLY;
            e.Loss = dfTotalLoss;
        }

        private T[] log_prob(Blob<T> blobDist, T[] rgActions, int nValCount, int nActionCount)
        {
            float[] rgDist = Utility.ConvertVecF<T>(blobDist.update_cpu_data());
            float[] rgfActions = Utility.ConvertVecF<T>(rgActions);
            List<float> rgLogProb = new List<float>();

            for (int i = 0; i < nValCount; i++)
            {
                int nAction = (int)rgfActions[i];
                float fSum = 0;

                for (int j = 0; j <= nAction; j++)
                {                    
                    int nIdx = (i * nActionCount) + j;
                    fSum += rgDist[nIdx];
                }

                rgLogProb.Add((float)Math.Log(fSum));
            }

            return Utility.ConvertVec<T>(rgLogProb.ToArray());
        }

        /// <summary>
        /// This event fires during the training of the local net.
        /// </summary>
        /// <remarks>
        /// This loss algorithm is mainly based on [6], however this loss calculation
        /// tended to blow-up for us after just a few runs.
        /// </remarks>
        /// <param name="sender">Specifies the instance of the MemoryLossLayer.</param>
        /// <param name="e">Specifies the arguments.</param>
        private void MemLoss_OnGetLoss_6(object sender, MemoryLossLayerGetLossArgs<T> e)
        {
            Blob<T> blobValues = null;
            Blob<T> blobLogits = null;

            if (e.Bottom.Count > 2)
                throw new Exception("Expected only two bottom values: logits(action_size), values(1)");

            for (int i = 0; i < e.Bottom.Count; i++)
            {
                if (e.Bottom[i].count(1) == 1)
                    blobValues = e.Bottom[i];
                else
                    blobLogits = e.Bottom[i];
            }

            if (blobValues == null)
                throw new Exception("Could not find the values blob collection!");

            m_blobAdvantage.ReshapeLike(blobValues);
            m_blobValueLoss.ReshapeLike(blobValues);
            m_blobEntropy.ReshapeLike(blobValues);
            m_blobPolicyLoss.ReshapeLike(blobValues);
            m_blobLoss.ReshapeLike(blobValues);

            m_blobPolicy.ReshapeLike(blobLogits);
            m_blobActionOneHot.ReshapeLike(blobLogits);

            int nValueCount = blobValues.count();
            long hValues = blobValues.gpu_data;
            long hAdvantage = m_blobAdvantage.mutable_gpu_data;
            long hValLoss = m_blobValueLoss.mutable_gpu_data;
            long hEntropy = m_blobEntropy.mutable_gpu_data;
            long hPolicyLoss = m_blobPolicyLoss.mutable_gpu_data;

            int nLogitCount = blobLogits.count();
            long hPolicy = m_blobPolicy.mutable_gpu_data;
            long hPolicyLog = m_blobPolicy.mutable_gpu_diff;

            int nActionCount = nLogitCount / nValueCount;
            BlobCollection<T> colBottom = new BlobCollection<T>();
            BlobCollection<T> colTop = new BlobCollection<T>();

            // Calculate the Advantage
            m_blobAdvantage.SetData(m_memory.GetTargets(nValueCount));
            m_cuda.sub(nValueCount, hAdvantage, hValues, hAdvantage);

            // Calculate the value loss
            m_cuda.mul(nValueCount, hAdvantage, hAdvantage, hValLoss);

            // Calculate the policy loss.
            // -- get the policy --
            colBottom.Add(blobLogits);
            colTop.Add(m_blobPolicy);

            if (!m_bSoftMaxSetup)
            {
                m_softmax.Setup(colBottom, colTop);
                m_bSoftMaxSetup = true;
            }

            m_softmax.Forward(colBottom, colTop);

            // -- get action one-hot vector
            T[] rgActionOneHot = m_memory.GetActionOneHot(nValueCount, nActionCount);
            m_blobActionOneHot.SetData(rgActionOneHot);

            // -- get the entropy --
            m_cuda.log(nLogitCount, hPolicy, hPolicyLog, 1.0, 1e-20);
            m_cuda.mul(nLogitCount, hPolicy, hPolicyLog, hPolicy);

            T[] rgEntropy = asum(nLogitCount, nValueCount, m_blobPolicy);
            m_blobEntropy.SetData(rgEntropy);

            // -- get the policy loss
            colBottom.Clear();
            colBottom.Add(blobLogits);          // input  (bottom[0])
            colBottom.Add(m_blobActionOneHot);  // target (bottom[1])
            colTop.Clear();
            colTop.Add(m_blobLoss);
            colTop.Add(m_blobPolicy);

            if (!m_bCrossEntropySetup)
            {
                m_crossentropy.Setup(colBottom, colTop);
                m_bCrossEntropySetup = true;
            }

            m_crossentropy.Forward(colBottom, colTop);

            T[] rgPolicyLoss = asum(nLogitCount, nValueCount, m_blobPolicy);
            m_blobPolicyLoss.SetData(rgPolicyLoss);

            // -- add the advantage to the policy loss --
            m_cuda.mul(nValueCount, hPolicyLoss, hAdvantage, hPolicyLoss);

            // -- subtract the entropy from the policy loss --
            m_cuda.mul_scalar(nValueCount, m_dfBeta, hEntropy);
            m_cuda.sub(nValueCount, hPolicyLoss, hEntropy, hPolicyLoss);

            // -- calculate the total loss --
            m_cuda.mul_scalar(nValueCount, 0.5, hValLoss);
            m_cuda.add(nValueCount, hValLoss, hPolicyLoss, hPolicyLoss);

            double dfAsum = Utility.ConvertVal<T>(m_blobPolicyLoss.asum_data());
            double dfTotalLoss = dfAsum / nValueCount; // mean

            e.application = MemoryLossLayerGetLossArgs<T>.APPLICATION.AS_LOSS_DIRECTLY;
            e.Loss = dfTotalLoss;
        }

        private T[] asum(int nCount, int nItems, Blob<T> b)
        {
            int nSubItems = nCount / nItems;
            double[] rgDf = Utility.ConvertVec<T>(b.update_cpu_data());
            List<double> rgSum = new List<double>();

            for (int i = 0; i < nItems; i++)
            {
                double dfTotal = 0;

                for (int j = 0; j < nSubItems; j++)
                {
                    int nIdx = i * nSubItems + j;
                    dfTotal += rgDf[nIdx];
                }

                rgSum.Add(dfTotal);
            }

            return Utility.ConvertVec<T>(rgSum.ToArray());
        }

        private void updateGlobalNet(MyCaffeControl<T> global, MyCaffeControl<T> local, int nIteration)
        {
            if (m_mode == TRAINING_MODE.A2C)
                return;

            global.CopyGradientsFrom(local);
            global.ApplyUpdate(nIteration);
            local.CopyWeightsFrom(global);
        }

        private Layer<T> getLayer(Net<T> net, LayerParameter.LayerType type)
        {
            foreach (Layer<T> layer in net.layers)
            {
                if (layer.layer_param.type == type)
                    return layer;
            }

            return null;
        }

        private bool addInputData(MyCaffeControl<T> mycaffe, Memory<T> mem, int nBatchSize)
        {
            if (mem.Count < nBatchSize)
                nBatchSize = mem.Count;

            if (nBatchSize == 0)
                return false;

            if (mem.Count == 1 && mem[0].State.Done)
                return false;

            MemoryDataLayer<T> memData = setBatchSize(mycaffe, Phase.TRAIN, nBatchSize, true);
            List<Datum> rgData = new List<Datum>();

            foreach (MemoryItem item in mem)
            {
                rgData.Add(new Datum(item.State.Data));
            }

            memData.AddDatumVector(rgData, 1, true);

            return true;
        }

        private MemoryDataLayer<T> setBatchSize(MyCaffeControl<T> mycaffe, Phase phase, int nBatchSize, bool bUpdateLast = false)
        {
            Net<T> net = mycaffe.GetInternalNet(phase);
            MemoryDataLayer<T> memData = getLayer(net, LayerParameter.LayerType.MEMORYDATA) as MemoryDataLayer<T>;

            if (memData == null)
                throw new Exception("Could not find the memory data layer!");

            if (bUpdateLast && nBatchSize != m_nLastBatchSize)
            {
                memData.batch_size = nBatchSize;
                m_nLastBatchSize = nBatchSize;
            }

            return memData;
        }

        private double getValue(MyCaffeControl<T> mycaffe, StateBase state)
        {
            m_blobInput.SetData(state.Data, true, true);
            mycaffe.DataTransformer.SetRange(m_blobInput);

            BlobCollection<T> colBottom = new BlobCollection<T>() { m_blobInput };
            double dfLoss;

            Net<T> net = mycaffe.GetInternalNet(Phase.RUN);
            BlobCollection<T> colOutput = net.Forward(colBottom, out dfLoss);
            Blob<T> blobValues = null;

            foreach (Blob<T> blob in colOutput)
            {
                if (blob.count(1) == 1)
                {
                    blobValues = blob;
                    break;
                }
            }

            if (blobValues == null)
                throw new Exception("Could not find the value blob of count() = 1!  This blob should be a bottom blob of the MemoryLossLayer.");

            double dfVal = Utility.ConvertVal<T>(blobValues.GetData(0));

            return dfVal;
        }

        private int getAction(MyCaffeControl<T> mycaffe, StateBase state)
        {
            double dfRandomSelection = m_random.NextDouble();

            // -- Exploration --
            if (dfRandomSelection < m_dfExplorationPct)
                return m_random.Next(state.ActionCount);

            // -- Learning --
            m_blobInput.SetData(state.Data, true, true);
            mycaffe.DataTransformer.SetRange(m_blobInput);

            BlobCollection<T> colBottom = new BlobCollection<T>() { m_blobInput };
            double dfLoss;

            Net<T> net = mycaffe.GetInternalNet(Phase.RUN);
            net.Forward(colBottom, out dfLoss);

            SoftmaxLayer<T> softmax = getLayer(net, LayerParameter.LayerType.SOFTMAX) as SoftmaxLayer<T>;
            if (softmax == null)
                throw new Exception("Could not find the softmax layer!");

            BlobCollection<T> colTop = net.FindTopBlobsOfLayer(softmax.layer_param.name);
            Blob<T> blobProb = colTop[0];
            float[] rgProb = Utility.ConvertVecF<T>(blobProb.update_cpu_data());

            double dfVal = m_random.NextDouble();
            double dfSum = (double)rgProb[0];

            if (dfVal <= dfSum)
                return 0;

            for (int i = 1; i < rgProb.Length; i++)
            {
                dfSum += rgProb[i];

                if (dfVal <= dfSum)
                    return i;
            }

            // Should never get here
            throw new Exception("Could not find the action!");
        }

        private void verifyEvents()
        {
            if (OnGetGlobalEpisodeCount == null)
                throw new Exception("You must connect the OnGetGlobalEpisodeCount event.");

            if (OnUpdateGlobalRewards == null)
                throw new Exception("You must connect the OnUpdateGlobalRewards event.");

            if (OnGetData == null)
                throw new Exception("You must connect the OnGetData event.");
        }
    }


    /// <summary>
    /// The InitializeArgs is passed to the OnInitialize event.
    /// </summary>
    public class InitializeArgs : EventArgs
    {
        int m_nIndex = 0;
        int m_nOriginalDsId = 0;
        int m_nDsID = 0;
        Component m_caffe;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="mycaffe">Specifies the MyCaffeControl used.</param>
        /// <param name="nIndex">Specifies the index of the trainer.</param>
        public InitializeArgs(Component mycaffe, int nIndex)
        {
            m_nIndex = nIndex;
            m_caffe = mycaffe;

            if (mycaffe is MyCaffeControl<double>)
            {
                MyCaffeControl<double> mycaffe1 = mycaffe as MyCaffeControl<double>;
                m_nOriginalDsId = mycaffe1.CurrentProject.Dataset.ID;
            }
            else
            {
                MyCaffeControl<float> mycaffe1 = mycaffe as MyCaffeControl<float>;
                m_nOriginalDsId = mycaffe1.CurrentProject.Dataset.ID;
            }
        }

        /// <summary>
        /// Returns the index of the trainer that fires the event.
        /// </summary>
        public int Index
        {
            get { return m_nIndex; }
        }

        /// <summary>
        /// Returns the MyCaffeControl used.
        /// </summary>
        public Component MyCaffe
        {
            get { return m_caffe; }
        }

        /// <summary>
        /// Returns the original Dataset ID of the open project held by the MyCaffeControl.
        /// </summary>
        public int OriginalDatasetID
        {
            get { return m_nOriginalDsId; }
        }

        /// <summary>
        /// Get/set a new Dataset ID which is actually used. 
        /// </summary>
        public int DatasetID
        {
            get { return m_nDsID; }
            set { m_nDsID = value; }
        }
    }

    /// <summary>
    /// The GetObservationArgs is passed to the OnGetObservations event.
    /// </summary>
    public class GetDataArgs : EventArgs
    {
        int m_nIndex;
        int m_nAction;
        bool m_bReset;
        Component m_caffe;
        StateBase m_state = null;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="mycaffe">Specifies the MyCaffeControl used.</param>
        /// <param name="nIndex">Specifies the index of the trainer.</param>
        /// <param name="bReset">Specifies to reset the environment.</param>
        /// <param name="nAction">Specifies the action to run.  If less than zero this parameter is ignored.</param>
        public GetDataArgs(Component mycaffe, int nIndex, bool bReset, int nAction = -1)
        {
            m_nIndex = nIndex;
            m_nAction = nAction;
            m_caffe = mycaffe;
            m_bReset = bReset;
        }

        /// <summary>
        /// Returns the index of the trainer that fires the event.
        /// </summary>
        public int Index
        {
            get { return m_nIndex; }
        }

        /// <summary>
        /// Specifies the state data of the observations.
        /// </summary>
        public StateBase State
        {
            get { return m_state; }
            set { m_state = value; }
        }

        /// <summary>
        /// Returns the action to run.  If less than zero, this parameter is ignored.
        /// </summary>
        public int Action
        {
            get { return m_nAction; }
        }

        /// <summary>
        /// Returns the MyCaffeControl used.
        /// </summary>
        public Component MyCaffe
        {
            get { return m_caffe; }
        }

        /// <summary>
        /// Returns whether or not to reset the observation environment or not.
        /// </summary>
        public bool Reset
        {
            get { return m_bReset; }
        }
    }

    /// <summary>
    /// Specifies the argumetns for the OnGetGlobalEpisodeCount and OnSetGlobalEpisodeCount events.
    /// </summary>
    public class GlobalEpisodeCountArgs : EventArgs
    {
        int m_nGlobalEpisodeCount = 0;
        int m_nMaximumGlobalEpisodeCount = 0;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="nCount">Specifies the new count, if any - used with OnSetGlobalEpisodeCount event.</param>
        public GlobalEpisodeCountArgs(int nCount = 0)
        {
            m_nGlobalEpisodeCount = nCount;
        }

        /// <summary>
        /// Get/set the GlobalEpisodeCount value.
        /// </summary>
        public int GlobalEpisodeCount
        {
            get { return m_nGlobalEpisodeCount; }
            set { m_nGlobalEpisodeCount = value; }
        }

        /// <summary>
        /// Get/set the maximum global episode count allowed.
        /// </summary>
        public int MaximumGlobalEpisodeCount
        {
            get { return m_nMaximumGlobalEpisodeCount; }
            set { m_nMaximumGlobalEpisodeCount = value; }
        }
    }

    /// <summary>
    /// Specifies the arguments used with the UpdateGlobalRewards event.
    /// </summary>
    public class UpdateGlobalRewardArgs : EventArgs
    {
        double m_dfReward;

        /// <summary>
        /// Specifies the 
        /// </summary>
        /// <param name="dfReward"></param>
        public UpdateGlobalRewardArgs(double dfReward = 0)
        {
            m_dfReward = dfReward;
        }

        /// <summary>
        /// Specifies the local reward.
        /// </summary>
        public double Reward
        {
            get { return m_dfReward; }
            set { m_dfReward = value; }
        }
    }
}
