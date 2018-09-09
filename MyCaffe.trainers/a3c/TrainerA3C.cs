using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.layers;
using MyCaffe.param;

namespace MyCaffe.trainers.a3c
{
    /// <summary>
    /// The TrainerA3C implements the Advantage, Actor-Critic Reinforcement Learning algorithm.
    /// </summary>
    /// <remarks>
    /// IMPORTANT NOT COMPLETE - work in progress.
    /// @see 1. [Let’s make an A3C: Implementation](https://jaromiru.com/2017/03/26/lets-make-an-a3c-implementation/) by Jaromír Janisch, 2017, ヤロミル
    /// @see 2. [AI-blog/CartPole-A3C.py](https://github.com/jaara/AI-blog/blob/master/CartPole-A3C.py) by jaara, 2017, GitHub
    /// @see 3. [Reinforcement Learning through Asynchronous Advantage Actor-Critic on a GPU](https://arxiv.org/abs/1611.06256) by M. Babaeizadeh, I. Frosio, S. Tyree, J. Clemons, J. Kautz, 2017, arXiv:1611.06256
    /// @see 4. [Massively Parallel Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1507.04296) by A. Nair, P. Srinivasan, S. Blackwell, C. Alcicek, R. Fearon, A. De Maria, V. Panneershelvam, M. Suleyman, C. Beattie, S. Petersen, S. Legg, V. Mnih, K. Kavukcuoglu and D. Silver, 2015, arXiv:1507.04296
    /// @see 5. [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) by V. Mnih, A. P. Badia, M. Mirza, A. Graves, T. P. Lillicrap, D. Silver and K. Kavukcuoglu, 2016, arXiv:1602.01783
    /// </remarks>
    public class TrainerA3C<T> : IxTrainer, IDisposable
    {
        IxTrainerCallback m_icallback;
        Random m_random = new Random();
        Brain<T> m_brain;
        MyCaffeControl<T> m_mycaffe;
        PropertySet m_properties;
        int m_nThreads;
        int m_nOptimizers;
        int m_nIterations;
        int m_nGlobalEpisodes = 0;
        bool m_bWindowOpen = false;


        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="mycaffe">Specifies the MyCaffeControl to use for learning and prediction.</param>
        /// <param name="properties">Specifies the property set containing the key/value pairs of property settings.</param>
        /// <param name="random">Specifies a Random number generator used for random selection.</param>
        /// <param name="icallback">Specifies the callback used to communicate with the parent.</param>
        public TrainerA3C(MyCaffeControl<T> mycaffe, PropertySet properties, Random random, IxTrainerCallback icallback)
        {
            m_icallback = icallback;
            m_mycaffe = mycaffe;
            m_properties = properties;
            m_nThreads = properties.GetPropertyAsInt("Threads", 8);
            m_nOptimizers = properties.GetPropertyAsInt("Optimizers", 2);
            m_nIterations = mycaffe.CurrentProject.GetSolverSettingAsInt("max_iter").GetValueOrDefault(10000);

#warning "AC3 Trainer - work in progress."
            mycaffe.Log.WriteLine("WARNING: The A3C trainer is a work in progress, use the PG trainer instead.");
        }

        /// <summary>
        /// Releases all resources uses.
        /// </summary>
        public void Dispose()
        {
            if (m_brain != null)
            {
                m_brain.Dispose();
                m_brain = null;
            }
        }

        /// <summary>
        /// Initialize the trainer.
        /// </summary>
        /// <returns>A value of <i>true</i> is returned when handled, <i>false</i> otherwise.</returns>
        public bool Initialize()
        {
            m_mycaffe.CancelEvent.Reset();
            m_brain = new Brain<T>(m_mycaffe, m_properties, m_random);
            m_icallback.OnInitialize(new InitializeArgs(m_mycaffe));

            return true;
        }

        private void wait(int nWait)
        {
            int nWaitInc = 250;
            int nTotalWait = 0;

            while (m_brain.ReferenceCount > 0 && nTotalWait < nWait)
            {
                m_icallback.OnWait(new WaitArgs(nWaitInc));
                nTotalWait += nWaitInc;
            }
        }

        /// <summary>
        /// Shutdown the trainer.
        /// </summary>
        /// <returns></returns>
        public bool Shutdown(int nWait)
        {
            if (m_brain != null)
            {
                if (m_mycaffe != null)
                {
                    m_mycaffe.CancelEvent.Set();
                    wait(nWait);
                }

                m_brain.Dispose();
                m_brain = null;
            }

            m_icallback.OnShutdown();

            return true;
        }

        /// <summary>
        /// Run a single cycle on the environment after the delay.
        /// </summary>
        /// <param name="nDelay">Specifies a delay to wait before running.</param>
        /// <returns>The results of the run containing the action are returned.</returns>
        public ResultCollection Run(int nDelay = 1000)
        {
            m_brain.MyCaffeControl.CancelEvent.Reset();
            Environment<T> env = new Environment<T>(m_brain, m_properties, m_random, 1, Phase.RUN, 0);
            env.OnGetData += Env_OnGetData;
            Tuple<int, int> res = env.Run(nDelay);

            List<KeyValuePair<int, double>> rgActions = new List<KeyValuePair<int, double>>();
            for (int i = 0; i < res.Item2; i++)
            {
                if (res.Item1 == i)
                    rgActions.Add(new KeyValuePair<int, double>(i, 1.0));
                else
                    rgActions.Add(new KeyValuePair<int, double>(i, 0.0));
            }

            return new ResultCollection(rgActions);
        }

        /// <summary>
        /// Run the test cycle - currently this is not implemented.
        /// </summary>
        /// <param name="nIterations">Specifies the number of iterations to run.</param>
        /// <returns>A value of <i>true</i> is returned when handled, <i>false</i> otherwise.</returns>
        public bool Test(int nIterations)
        {
            Environment<T> env = new Environment<T>(m_brain, m_properties, m_random, nIterations, Phase.TEST, 0);
            env.OnGetData += Env_OnGetData;
            env.OnGetStatus += Env_OnGetStatus;
            env.Start(10);

            m_nGlobalEpisodes = 0;
            m_brain.MyCaffeControl.CancelEvent.Reset();

            while (!m_brain.MyCaffeControl.CancelEvent.WaitOne(250))
            {
                Thread.Sleep(1);
            }

            env.Stop(1000);
            env.Dispose();

            Shutdown(3000);

            return true;
        }

        /// <summary>
        /// Train the network using a modified A3C training algorithm optimized for GPU use.
        /// </summary>
        /// <param name="nIterations">Specifies the number of iterations to run.</param>
        /// <param name="step">Specifies the stepping mode to use (when debugging).</param>
        /// <returns>A value of <i>true</i> is returned when handled, <i>false</i> otherwise.</returns>
        public bool Train(int nIterations, TRAIN_STEP step)
        {
            List<Worker> rgEnvironments = new List<Worker>();
            List<Worker> rgOptimizers = new List<Worker>();

            m_nGlobalEpisodes = 0;
            m_brain.MyCaffeControl.CancelEvent.Reset();

            for (int i = 0; i < m_nOptimizers; i++)
            {
                Optimizer<T> opt = new Optimizer<T>(m_brain, i);
                opt.Start(1);
                rgOptimizers.Add(opt);
            }

            for (int i = 0; i < m_nThreads; i++)
            {
                Environment<T> env = new Environment<T>(m_brain, m_properties, m_random, m_nIterations, Phase.TRAIN, i);
                env.OnGetData += Env_OnGetData;
                env.OnGetStatus += Env_OnGetStatus;
                env.Start(1);
                rgEnvironments.Add(env);
            }

            while (!m_brain.MyCaffeControl.CancelEvent.WaitOne(250))
            {
                if (m_nGlobalEpisodes >= m_nIterations)
                    break;
            }

            foreach (Optimizer<T> opt in rgOptimizers)
            {
                opt.Stop(1000);
                opt.Dispose();
            }

            foreach (Environment<T> env in rgEnvironments)
            {
                env.Stop(1000);
                env.Dispose();
            }

            Shutdown(3000);

            return true;
        }

        /// <summary>
        /// Get/set whether or not an informational window is open or not.
        /// </summary>
        /// <remarks>
        /// When an informational window is open, the training process is delayed so as to show the state of the training visually.
        /// For example, when a gym window is open, the training is slowed down to show the gym simulation visually.
        /// </remarks>
        public bool WindowOpen
        {
            get { return m_bWindowOpen; }
            set { m_bWindowOpen = true; }
        }

        private void Env_OnGetStatus(object sender, GetStatusArgs e)
        {
            m_nGlobalEpisodes = Math.Max(m_nGlobalEpisodes, e.Frames);
            m_icallback.OnUpdateStatus(e);
        }

        private void Env_OnGetData(object sender, GetDataArgs e)
        {
            m_icallback.OnGetData(e);
        }
    }

    class Worker /** @private */
    {
        protected int m_nIndex = -1;
        protected AutoResetEvent m_evtCancel = new AutoResetEvent(false);
        protected ManualResetEvent m_evtDone = new ManualResetEvent(false);
        protected Task m_workTask = null;

        public Worker(int nIdx)
        {
            m_nIndex = nIdx;
        }

        protected virtual void doWork(object arg)
        {
        }

        public void Start(int nCycleDelay)
        {
            if (m_workTask == null)
                m_workTask = Task.Factory.StartNew(new Action<object>(doWork), nCycleDelay);
        }

        public void Stop(int nWait)
        {
            m_evtCancel.Set();
            m_workTask = null;
            m_evtDone.WaitOne(nWait);
        }
    }

    class Environment<T> : Worker, IDisposable /** @private */
    {
        Brain<T> m_brain;
        Agent<T> m_agent;
        int m_nIterations = 0;
        Phase m_phase;

        public event EventHandler<GetDataArgs> OnGetData;
        public event EventHandler<GetStatusArgs> OnGetStatus;


        public Environment(Brain<T> brain, PropertySet properties, Random random, int nIterations, Phase phase, int nIdx)
            : base(nIdx)
        {
            m_phase = phase;
            m_brain = brain;
            m_agent = new Agent<T>(m_brain, properties, random);
            m_nIterations = nIterations;
        }

        public void Dispose()
        {
            Stop(2000);
        }

        protected override void doWork(object arg)
        {
            int nCycleDelay = (int)arg;
            Stopwatch sw = new Stopwatch();

            m_evtDone.Reset();
            m_evtCancel.Reset();
            m_brain.ReferenceCount++;

            sw.Start();

            // Main training loop
            while (!m_evtCancel.WaitOne(nCycleDelay))
            {
                if (!runEpisode())
                    break;
            }

            m_evtDone.Set();
            m_brain.ReferenceCount--;
        }

        private StateBase getData(int nAction)
        {
            int nIndex = (m_phase == Phase.TRAIN) ? m_nIndex : 0;
            bool bAllowUi = (m_phase == Phase.TRAIN) ? false : true;
            bool bReset = (nAction == -1) ? true : false;

            GetDataArgs dataArg = new GetDataArgs(nIndex, m_brain.MyCaffeControl, m_brain.MyCaffeControl.Log, m_brain.MyCaffeControl.CancelEvent, bReset, nAction, bAllowUi);
            OnGetData(this, dataArg);

            return dataArg.State;
        }

        private void updateStatus(double dfR)
        {
            if (OnGetStatus != null)
                OnGetStatus(this, new GetStatusArgs(Agent<T>.Frames, m_nIterations, dfR, m_agent.epsilon));
        }

        private bool runEpisode()
        {
            // Reset the environment and get the initial state.
            StateBase s = getData(-1);
            if (s == null)
                return true;

            bool bDone = false;
            double dfR = 0;

            while (!m_evtCancel.WaitOne(1) && !bDone)
            {
                int a = m_agent.act(m_phase, s, true);
                StateBase s_ = getData(a);
                double dfReward = s_.Reward;
                bDone = s_.Done;

                if (m_brain.MyCaffeControl.CancelEvent.WaitOne(1))
                    return false;

                if (bDone)
                    s_ = null;

                if (m_phase == Phase.TRAIN)
                    m_agent.train(s, a, dfReward, s_);

                s = s_;
                dfR += dfReward;

                updateStatus(dfR);
            }

            return true;
        }

        public Tuple<int, int> Run(int nDelay = 1000)
        {
            // Reset the environment and get the initial state.
            GetDataArgs dataArg = new GetDataArgs(0, m_brain.MyCaffeControl, m_brain.MyCaffeControl.Log, m_brain.MyCaffeControl.CancelEvent, true, -1, false);
            OnGetData(this, dataArg);

            Thread.Sleep(nDelay);

            dataArg = new GetDataArgs(0, m_brain.MyCaffeControl, m_brain.MyCaffeControl.Log, m_brain.MyCaffeControl.CancelEvent, false, -1, false);
            OnGetData(this, dataArg);

            int a = m_agent.act(Phase.RUN, dataArg.State, true);

            return new Tuple<int, int>(a, dataArg.State.ActionCount);
        }
    }

    class Optimizer<T> : Worker, IDisposable /** @private */
    {
        Brain<T> m_brain;
        Blob<T> m_blobProb;
        Blob<T> m_blobVal;

        public Optimizer(Brain<T> brain, int nIdx)
            : base(nIdx)
        {
            m_brain = brain;
            m_blobProb = brain.CreateBlob();
            m_blobVal = brain.CreateBlob();
        }

        public void Dispose()
        {
            Stop(2000);

            if (m_blobVal != null)
            {
                m_blobVal.Dispose();
                m_blobVal = null;
            }

            if (m_blobProb != null)
            {
                m_blobProb.Dispose();
                m_blobProb = null;
            }
        }

        protected override void doWork(object arg)
        {
            int nCycleDelay = (int)arg;

            m_evtDone.Reset();
            m_brain.ReferenceCount++;

            while (!m_evtCancel.WaitOne(nCycleDelay))
            {
                m_brain.optimize(m_blobProb, m_blobVal);
            }

            m_evtDone.Set();
            m_brain.ReferenceCount--;
        }
    }

    class Agent<T> : IDisposable /** @private */
    {
        static int m_nFrames = 0;   // global
        Brain<T> m_brain;
        Random m_random;
        int m_nEpsSteps = 0;
        double m_dfEpsStart = 0;
        double m_dfEpsEnd = 0;
        double m_dfR = 0;
        Memory<T> m_memory;

        public Agent(Brain<T> brain, PropertySet properties, Random random)
        {
            m_brain = brain;
            m_random = random;

            m_nEpsSteps = properties.GetPropertyAsInt("EpsSteps", 0);
            m_dfEpsStart = properties.GetPropertyAsDouble("EpsStart", 0);
            m_dfEpsEnd = properties.GetPropertyAsDouble("EpsEnd", 0);
            m_memory = new Memory<T>(brain.RewardScale, m_nFrames);
        }

        public void Dispose()
        {
        }

        public static int Frames
        {
            get { return m_nFrames; }
        }

        public double epsilon
        {
            get
            {
                if (m_nFrames >= m_nEpsSteps)
                    return m_dfEpsEnd;

                return m_dfEpsStart + m_nFrames * (m_dfEpsEnd - m_dfEpsStart) / m_nEpsSteps;
            }
        }

        public int act(Phase phase, StateBase s, bool bStochastic)
        {
            double dfEpsilon = epsilon;

            m_nFrames++;

            // -- Exploration --
            if (phase == Phase.TRAIN && m_random.NextDouble() < dfEpsilon)
                return m_random.Next(s.ActionCount);

            // -- Prediction --
            Tuple<float[], float> res = m_brain.predict(s);
            float[] rgProb = res.Item1;

            // Stochastic Selection
            if (bStochastic)
            {
                float fSum = 0;
                float fVal = (float)m_random.NextDouble();

                for (int i = 0; i < rgProb.Length; i++)
                {
                    fSum += rgProb[i];

                    if (fVal <= fSum)
                        return i;
                }

                return rgProb.Length - 1;
            }

            // Deterministic Selection
            else
            {
                float fMax = -float.MaxValue;
                int nMax = 0;

                for (int i = 0; i < rgProb.Length; i++)
                {
                    if (fMax < rgProb[i])
                    {
                        fMax = rgProb[i];
                        nMax = i;
                    }
                }

                return nMax;
            }
        }

        public MemoryItem get_sample(int n)
        {
            return new MemoryItem(m_memory[0].State0, m_memory[0].Action, m_dfR, m_memory[n - 1].State1, m_memory[0].FrameIndex);
        }

        public void train(StateBase s, int nAction, double dfReward, StateBase s_)
        {
            m_memory.Add(new MemoryItem(s, nAction, dfReward, s_, m_nFrames));
            m_dfR = (m_dfR + dfReward * m_brain.GammaN) / m_brain.Gamma;

            if (s_ == null) // we are done
            {
                while (m_memory.Count > 0)
                {
                    MemoryItem item = get_sample(m_memory.Count);
                    m_brain.train_push(item);
                    m_dfR = (m_dfR - m_memory[0].Reward) / m_brain.Gamma;
                    m_memory.RemoveAt(0);
                }

                m_dfR = 0;
            }

            if (m_memory.Count >= m_brain.NStepReturn)
            {
                MemoryItem item = get_sample(m_brain.NStepReturn);
                m_brain.train_push(item);
                m_dfR = m_dfR - m_memory[0].Reward;
                m_memory.RemoveAt(0);
            }
        }
    }

    class Brain<T> : IDisposable /** @private */
    {
        MyCaffeControl<T> m_caffe;
        Net<T> m_net;
        MemoryDataLayer<T> m_memData;
        MemoryLossLayer<T> m_memLoss;
        SoftmaxLayer<T> m_softmax;
        MemoryCollection<T> m_rgTrainingQueues = new MemoryCollection<T>(6, 12);
        Memory<T> m_trainingQueue;
        int m_nMinBatch = 0;
        object m_syncQueue = new object();
        object m_syncTrain = new object();
        Blob<T> m_blobInput;
        Blob<T> m_blobR;
        Blob<T> m_blobSmask;
        Blob<T> m_blobActionOneHot;
        Blob<T> m_blobLogProb;
        Blob<T> m_blobAdvantage;
        Blob<T> m_blobLossPolicy;
        Blob<T> m_blobLossValue;
        Blob<T> m_blobEntropy;
        Blob<T> m_blobWork;
        Blob<T> m_blobLoss;
        Blob<T> m_blobOutVal;
        Blob<T> m_blobOutProb;
        int m_nNStepReturn = 8;
        double m_dfGamma = 0.99;
        double m_dfNGamma = 0;
        double m_dfLossCoefficient = 0.5;
        double m_dfEntropyCoefficient = 0.01;
        double m_dfOptimalEpisodeCoefficient = 0.5;
        int m_nOptimalEpisodeStart = 5000;
        bool m_bSkipLoss = false;
        int m_nRefCount = 0;
        Random m_random;
        double m_dfRewardScale = 1.0;
        SoftmaxCrossEntropyLossLayer<T> m_softmaxCe;
        bool m_bSoftmaxCeSetup = false;

        public Brain(MyCaffeControl<T> mycaffe, PropertySet properties, Random random)
        {
            m_caffe = mycaffe;
            m_random = random;
            setupNet(m_caffe);

            m_nNStepReturn = properties.GetPropertyAsInt("NStepReturn", 8);
            m_dfGamma = properties.GetPropertyAsDouble("Gamma", 0.99);
            m_dfNGamma = Math.Pow(m_dfGamma, m_nNStepReturn);
            m_dfLossCoefficient = properties.GetPropertyAsDouble("LossCoefficient", 0.5);
            m_dfEntropyCoefficient = properties.GetPropertyAsDouble("EntropyCoefficient", 0.01);
            m_dfOptimalEpisodeCoefficient = properties.GetPropertyAsDouble("OptimalEpisodeCoefficient", 0.5);
            m_nOptimalEpisodeStart = properties.GetPropertyAsInt("OptimalEpisodeStart", 5000);
            m_dfRewardScale = properties.GetPropertyAsDouble("RewardScale", 1.0);

            m_blobInput = new Blob<T>(mycaffe.Cuda, mycaffe.Log, false);
            m_blobR = new Blob<T>(mycaffe.Cuda, mycaffe.Log, true);
            m_blobSmask = new Blob<T>(mycaffe.Cuda, mycaffe.Log, false);
            m_blobActionOneHot = new Blob<T>(mycaffe.Cuda, mycaffe.Log, true);
            m_blobLogProb = new Blob<T>(mycaffe.Cuda, mycaffe.Log, false);
            m_blobAdvantage = new Blob<T>(mycaffe.Cuda, mycaffe.Log, false);
            m_blobLossPolicy = new Blob<T>(mycaffe.Cuda, mycaffe.Log, false);
            m_blobLossValue = new Blob<T>(mycaffe.Cuda, mycaffe.Log, false);
            m_blobLoss = new Blob<T>(mycaffe.Cuda, mycaffe.Log, true);
            m_blobEntropy = new Blob<T>(mycaffe.Cuda, mycaffe.Log, false);
            m_blobWork = new Blob<T>(mycaffe.Cuda, mycaffe.Log, true);
            m_blobOutProb = new Blob<T>(mycaffe.Cuda, mycaffe.Log, false);
            m_blobOutVal = new Blob<T>(mycaffe.Cuda, mycaffe.Log, false);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SOFTMAXCROSSENTROPY_LOSS);
            p.loss_param.normalization = LossParameter.NormalizationMode.NONE;
            m_softmaxCe = new SoftmaxCrossEntropyLossLayer<T>(m_caffe.Cuda, m_caffe.Log, p);
        }

        public void Dispose()
        {
            cleanupNet(m_caffe);

            if (m_blobInput != null)
            {
                m_blobInput.Dispose();
                m_blobInput = null;
            }

            if (m_blobR != null)
            {
                m_blobR.Dispose();
                m_blobR = null;
            }

            if (m_blobSmask != null)
            {
                m_blobSmask.Dispose();
                m_blobSmask = null;
            }

            if (m_blobActionOneHot != null)
            {
                m_blobActionOneHot.Dispose();
                m_blobActionOneHot = null;
            }

            if (m_blobLogProb != null)
            {
                m_blobLogProb.Dispose();
                m_blobLogProb = null;
            }

            if (m_blobAdvantage != null)
            {
                m_blobAdvantage.Dispose();
                m_blobAdvantage = null;
            }

            if (m_blobLossPolicy != null)
            {
                m_blobLossPolicy.Dispose();
                m_blobLossPolicy = null;
            }

            if (m_blobLossValue != null)
            {
                m_blobLossValue.Dispose();
                m_blobLossValue = null;
            }

            if (m_blobLoss != null)
            {
                m_blobLoss.Dispose();
                m_blobLoss = null;
            }

            if (m_blobEntropy != null)
            {
                m_blobEntropy.Dispose();
                m_blobEntropy = null;
            }

            if (m_blobWork != null)
            {
                m_blobWork.Dispose();
                m_blobWork = null;
            }

            if (m_blobOutProb != null)
            {
                m_blobOutProb.Dispose();
                m_blobOutProb = null;
            }

            if (m_blobOutVal != null)
            {
                m_blobOutVal.Dispose();
                m_blobOutVal = null;
            }

            if (m_softmaxCe != null)
            {
                m_softmaxCe.Dispose();
                m_softmaxCe = null;
            }
        }

        public int ReferenceCount
        {
            get { return m_nRefCount; }
            set { m_nRefCount = value; }
        }

        public MyCaffeControl<T> MyCaffeControl
        {
            get { return m_caffe; }
        }

        public double Gamma
        {
            get { return m_dfGamma; }
        }

        public double GammaN
        {
            get { return m_dfNGamma; }
        }

        public int NStepReturn
        {
            get { return m_nNStepReturn; }
        }

        public double RewardScale
        {
            get { return m_dfRewardScale; }
        }

        private void setupNet(MyCaffeControl<T> mycaffe)
        {
            m_net = mycaffe.GetInternalNet(Phase.TRAIN);
            m_memData = m_net.FindLayer(LayerParameter.LayerType.MEMORYDATA, null) as MemoryDataLayer<T>;
            m_memLoss = m_net.FindLayer(LayerParameter.LayerType.MEMORY_LOSS, null) as MemoryLossLayer<T>;
            m_softmax = m_net.FindLayer(LayerParameter.LayerType.SOFTMAX, null) as SoftmaxLayer<T>;

            if (m_memData == null)
                throw new Exception("Could not find the MemoryDataLayer in the training net!");

            if (m_memLoss == null)
                throw new Exception("Could not find the MemoryLossLayer in the training net!");

            if (m_softmax == null)
                throw new Exception("Could not find the SoftmaxLayer in the training net!");

            m_memLoss.OnGetLoss += MemLoss_OnGetLoss;

            m_nMinBatch = (int)m_memData.layer_param.memory_data_param.batch_size;
            m_trainingQueue = new Memory<T>(m_dfRewardScale);
        }

        private void cleanupNet(MyCaffeControl<T> mycaffe)
        {
            m_memLoss.OnGetLoss -= MemLoss_OnGetLoss;
        }
       
        public void train_push(MemoryItem item)
        {
            lock (m_syncQueue)
            {
                m_trainingQueue.Add(item);
            }
        }

        public Blob<T> CreateBlob()
        {
            return new Blob<T>(m_caffe.Cuda, m_caffe.Log);
        }

        private Memory<T> get_batch()
        {          
            Memory<T> mem = null;

            lock (m_syncQueue)
            {
                if (m_trainingQueue.Count < m_nMinBatch)
                    return null;

                mem = m_trainingQueue.Clone(true);
            }

            m_rgTrainingQueues.Add(mem);

            if (mem.FrameIndex >= m_nOptimalEpisodeStart)
            {
                if (m_random.NextDouble() < m_dfOptimalEpisodeCoefficient)
                    return m_rgTrainingQueues.GetMemory(m_random);
            }

            return mem;
        }

        private int trainingQueueSize
        {
            get
            {
                lock (m_syncQueue)
                {
                    return m_trainingQueue.Count;
                }
            }
        }

        public Tuple<Blob<T>, Blob<T>> getBlobs(bool bGetProb, bool bGetVal)
        {
            Blob<T> blobProb = null;
            Blob<T> blobVal = null;

            if (bGetProb)
            {
                BlobCollection<T> colTop = m_net.FindTopBlobsOfLayer(m_softmax.layer_param.name);
                if (colTop == null || colTop.Count == 0)
                    throw new Exception("Could not find the tops of the softmax layer!");

                foreach (Blob<T> b in colTop)
                {
                    if (b.num_axes > 1 && b.count(1) > 1)
                    {
                        blobProb = b;
                        break;
                    }
                }

                if (blobProb == null)
                    throw new Exception("Could not find the action probability blob!");
            }

            if (bGetVal)
            {
                BlobCollection<T> colBottom = m_net.FindBottomBlobsOfLayer(m_memLoss.layer_param.name);
                if (colBottom == null || colBottom.Count == 0)
                    throw new Exception("Could not find the bottoms of the memory loss layer!");

                foreach (Blob<T> b in colBottom)
                {
                    if (b.num_axes > 1 && b.count(1) == 1)
                    {
                        blobVal = b;
                        break;
                    }
                }

                if (blobVal == null)
                    throw new Exception("Could not find the value blob!");
            }

            return new Tuple<Blob<T>, Blob<T>>(blobProb, blobVal);
        }

        public Tuple<float[], float> predict(StateBase s)
        {
            List<Datum> rgData = new List<Datum>();

            rgData.Add(new Datum(s.Data));


            lock (m_syncTrain)
            {
                Tuple<Blob<T>, Blob<T>> res = predict(rgData);

                float[] rgProb = Utility.ConvertVecF<T>(res.Item1.update_cpu_data());
                float[] rgVal = Utility.ConvertVecF<T>(res.Item2.update_cpu_data());

                return new Tuple<float[], float>(rgProb, rgVal[0]);
            }
        }

        public Tuple<Blob<T>, Blob<T>> predict(List<Datum> rgData)
        {
            double dfLoss;

            // Run the forward but skip the loss for we are not learning on this pass.
            m_caffe.Log.Enable = false;
            m_memData.AddDatumVector(rgData, 1, true, true);
            m_bSkipLoss = true;
            m_net.Forward(out dfLoss);
            m_bSkipLoss = false;
            m_caffe.Log.Enable = true;

            return getBlobs(true, true);
        }

        public bool optimize(Blob<T> blobProb, Blob<T> blobVal)
        {
            if (trainingQueueSize < m_nMinBatch)
            {
                Thread.Sleep(0);
                return false;
            }

            Memory<T> batch = get_batch();
            if (batch == null)
                return false;

            lock (m_syncTrain)
            {
                try
                {
                    // predict_v(s_)
                    List<Datum> rgNewStates = batch.GetStates(Memory<T>.STATE_TYPE.NEW);
                    Tuple<Blob<T>, Blob<T>> res = predict(rgNewStates);
                    blobProb.CopyFrom(res.Item1, false, true);
                    blobVal.CopyFrom(res.Item2, false, true);

                    // Get more batch data.
                    T[] rgSMask = batch.GetStateMask();
                    m_blobSmask.ReshapeLike(blobVal);
                    m_blobSmask.SetData(rgSMask);

                    T[] rgR = batch.GetRewards();
                    m_blobR.ReshapeLike(blobVal);
                    m_blobR.SetData(rgR);

                    T[] rgActionOneHot = batch.GetActions(Memory<T>.VALUE_TYPE.ONEHOT);
                    m_blobActionOneHot.ReshapeLike(blobProb);
                    m_blobActionOneHot.SetData(rgActionOneHot);

                    // r = r + GAMMA_N * v * s_mask (v set to 0 where s_ is terminal, by AddDatumVector)
                    // blobR.diff = v * s_mask
                    m_caffe.Cuda.mul(m_blobR.count(), blobVal.gpu_data, m_blobSmask.gpu_data, m_blobR.mutable_gpu_diff);
//                    Trace.WriteLine("v * s_mask = " + m_blobR.ToString(10, true));

                    // blobR.diff = GAMMA_N * (v * s_mask)
                    m_caffe.Cuda.mul_scalar(m_blobR.count(), m_dfNGamma, m_blobR.mutable_gpu_diff);
//                    Trace.WriteLine("GAMMA_N * (v * s_mask) = " + m_blobR.ToString(10, true));

                    // blobR.data = blobR.data + blobR.diff
                    m_caffe.Cuda.add(m_blobR.count(), m_blobR.gpu_data, m_blobR.gpu_diff, m_blobR.mutable_gpu_data);
//                    Trace.WriteLine("R = " + m_blobR.ToString(10));

                    // Train (action one hot used in the loss function fired during the MemoryLossLayer forward)
                    m_caffe.Log.Enable = false;
                    List<Datum> rgStates = batch.GetStates(Memory<T>.STATE_TYPE.OLD);
                    m_memData.AddDatumVector(rgStates, 1, true, true);
                    m_caffe.Train(1, 0, TRAIN_STEP.NONE, 0, true);
                    m_caffe.Log.Enable = true;
                }
                catch (Exception excpt)
                {
                    throw excpt;
                }
                finally
                {
                }
            }

            return true;
        }

        /// <summary>
        /// The MemLoss_OnGetLoss event is called from the Forward pass of the MemoryLossLayer.
        /// </summary>
        /// <param name="sender">Specifies the event sender which is the MemoryLossLayer.</param>
        /// <param name="e">Specifies the event arguments.</param>
        private void MemLoss_OnGetLoss(object sender, MemoryLossLayerGetLossArgs<T> e)
        {
            if (m_bSkipLoss)
                return;

            Blob<T> blobValues = null;
            Blob<T> blobLogits = null;
            Blob<T> blobProb = null;
            int nIdxValues = 0;
            int nIdxLogits = 0;

            if (e.Bottom.Count != 2)
                throw new Exception("Expected only two bottom values: logits(action_size), values(1)");

            Tuple<Blob<T>, Blob<T>> blobs = getBlobs(true, false);

            blobProb = blobs.Item1;

            for (int i = 0; i < m_net.output_blobs.Count; i++)
            {
                if (m_net.output_blobs[i].count(1) != 1)
                {
                    blobProb = m_net.output_blobs[i];
                    break;
                }
            }

            for (int i = 0; i < e.Bottom.Count; i++)
            {
                if (e.Bottom[i].count(1) == 1)
                {
                    nIdxValues = i;
                    blobValues = e.Bottom[i];
                }
                else
                {
                    nIdxLogits = i;
                    blobLogits = e.Bottom[i];
                }
            }

            if (blobProb == null)
                throw new Exception("Could not find the action probability output blob!");

            if (blobValues == null)
                throw new Exception("Could not find the values blob!");

            if (blobLogits == null)
                throw new Exception("Could not find the logits blob!");

            m_blobLoss.Reshape(new List<int>());

            int nValueCount = blobValues.count();
            m_blobLogProb.ReshapeLike(blobValues);
            m_blobAdvantage.ReshapeLike(blobValues);
            m_blobLossPolicy.ReshapeLike(blobValues);
            m_blobLossValue.ReshapeLike(blobValues);
            m_blobEntropy.ReshapeLike(blobValues);

            int nLogitCount = blobLogits.count();
            m_blobWork.ReshapeLike(blobLogits);

            // Calculate the softmax of the logits.
            int nActionCount = nLogitCount / nValueCount;

            m_blobR.SetDiff(0);
            m_blobActionOneHot.SetDiff(0);
            m_blobLoss.SetDiff(0);
            m_blobWork.SetData(0);
            m_blobWork.SetDiff(0);

            //Trace.WriteLine("Prob = " + blobProb.ToString(10));
            //Trace.WriteLine("Logits = " + blobLogits.ToString(10));
            //Trace.WriteLine("Action = " + m_blobActionOneHot.ToString(10));
            //Trace.WriteLine("Values = " + blobValues.ToString(10));

            // calculate 'log_prob'
            m_blobWork.ReshapeLike(blobLogits);
            m_caffe.Cuda.mul(nLogitCount, blobProb.gpu_data, m_blobActionOneHot.gpu_data, m_blobWork.mutable_gpu_data);
            T[] rgSum = asum(nLogitCount, nValueCount, m_blobWork);
            m_blobLogProb.SetData(rgSum);
            m_caffe.Cuda.log(nValueCount, m_blobLogProb.gpu_data, m_blobLogProb.mutable_gpu_data, 1.0, 1e-10);

//            Trace.WriteLine("LogProb = " + m_blobLogProb.ToString(10));

            // calculate 'advantage'
            m_caffe.Cuda.sub(nValueCount, m_blobR.gpu_data, blobValues.gpu_data, m_blobAdvantage.mutable_gpu_data);

//            Trace.WriteLine("Advantage = " + m_blobAdvantage.ToString(10));

            // calculate 'loss_policy'
            m_caffe.Cuda.mul(nValueCount, m_blobLogProb.gpu_data, m_blobAdvantage.gpu_data, m_blobLossPolicy.mutable_gpu_data);
            m_caffe.Cuda.mul_scalar(nValueCount, -1.0, m_blobLossPolicy.mutable_gpu_data);

//            Trace.WriteLine("Loss Policy = " + m_blobLossPolicy.ToString(10));

            // calculate 'loss value'
            m_caffe.Cuda.powx(nValueCount, m_blobAdvantage.gpu_data, 2.0f, m_blobLossValue.mutable_gpu_data);
            m_caffe.Cuda.mul_scalar(nValueCount, m_dfLossCoefficient, m_blobLossValue.mutable_gpu_data);

//            Trace.WriteLine("Loss Value = " + m_blobLossValue.ToString(10));

            // calculate 'entropy'
            m_caffe.Cuda.log(nLogitCount, blobProb.gpu_data, m_blobWork.mutable_gpu_diff, 1.0, 1e-10);
            m_caffe.Cuda.mul(nLogitCount, blobProb.gpu_data, m_blobWork.gpu_diff, m_blobWork.mutable_gpu_data);
            rgSum = asum(nLogitCount, nValueCount, m_blobWork);
            m_blobEntropy.SetData(rgSum);
            m_caffe.Cuda.mul_scalar(nValueCount, m_dfEntropyCoefficient, m_blobEntropy.mutable_gpu_data);

//            Trace.WriteLine("Entropy = " + m_blobEntropy.ToString(10));

            // calculate 'total_loss'
            m_blobWork.ReshapeLike(blobValues);
            m_caffe.Cuda.add(nValueCount, m_blobLossPolicy.gpu_data, m_blobLossValue.gpu_data, m_blobWork.mutable_gpu_diff);
            m_caffe.Cuda.add(nValueCount, m_blobEntropy.gpu_data, m_blobWork.gpu_diff, m_blobWork.mutable_gpu_diff);

//            Trace.WriteLine("Total Loss = " + m_blobWork.ToString(10, true));

            double dfAsum = Utility.ConvertVal<T>(m_blobWork.asum_diff());
            double dfTotalLoss = dfAsum / nValueCount;

//            Trace.WriteLine("Total Loss = " + dfTotalLoss.ToString());

            e.Loss = dfTotalLoss;

            // Apply the loss to the value diffs.
            blobValues.SetDiff(1);

            // Apply the loss to the logits diffs.
            BlobCollection<T> colBottom = new BlobCollection<T>();
            BlobCollection<T> colTop = new BlobCollection<T>();

            colBottom.Add(blobLogits);          // logits
            colBottom.Add(m_blobActionOneHot);  // targets
            colTop.Add(m_blobLoss);

            if (!m_bSoftmaxCeSetup)
            {
                m_softmaxCe.Setup(colBottom, colTop);
                m_bSoftmaxCeSetup = true;
            }

            m_softmaxCe.Forward(colBottom, colTop);
            m_caffe.Cuda.copy(m_blobLoss.count(), m_blobLoss.gpu_data, m_blobLoss.mutable_gpu_diff);
            m_softmaxCe.Backward(colTop, new List<bool>() { true, false }, colBottom);

//            Trace.WriteLine("Val Diff = " + blobValues.ToString(10, true));
//            Trace.WriteLine("Logit Diff = " + colBottom[0].ToString(10, true));
        }

        private T[] asum(int nCount, int nItems, Blob<T> b, bool bDiff = false)
        {
            int nSubItems = nCount / nItems;
            double[] rgDf = Utility.ConvertVec<T>((bDiff) ? b.update_cpu_diff() : b.update_cpu_data());
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
    }
}
