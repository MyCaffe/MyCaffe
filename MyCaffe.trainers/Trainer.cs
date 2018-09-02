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

namespace MyCaffe.trainers
{
    /// <summary>
    /// The Trainer implements the Advantage, Actor-Critic Reinforcement Learning algorithm.
    /// </summary>
    /// <remarks>
    /// @see 1. [Let’s make an A3C: Implementation](https://jaromiru.com/2017/03/26/lets-make-an-a3c-implementation/) by Jaromír Janisch, 2017, ヤロミル
    /// @see 2. [AI-blog/CartPole-A3C.py](https://github.com/jaara/AI-blog/blob/master/CartPole-A3C.py) by jaara, 2017, GitHub
    /// @see 3. [Reinforcement Learning through Asynchronous Advantage Actor-Critic on a GPU](https://arxiv.org/abs/1611.06256) by M. Babaeizadeh, I. Frosio, S. Tyree, J. Clemons, J. Kautz, 2017, arXiv:1611.06256
    /// @see 4. [Massively Parallel Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1507.04296) by A. Nair, P. Srinivasan, S. Blackwell, C. Alcicek, R. Fearon, A. De Maria, V. Panneershelvam, M. Suleyman, C. Beattie, S. Petersen, S. Legg, V. Mnih, K. Kavukcuoglu and D. Silver, 2015, arXiv:1507.04296
    /// @see 5. [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) by V. Mnih, A. P. Badia, M. Mirza, A. Graves, T. P. Lillicrap, D. Silver and K. Kavukcuoglu, 2016, arXiv:1602.01783
    /// </remarks>
    public class Trainer<T> : IxTrainer, IDisposable
    {
        Random m_random = new Random();
        Brain<T> m_brain;
        MyCaffeControl<T> m_mycaffe;
        PropertySet m_properties;
        int m_nThreads;
        int m_nOptimizers;
        int m_nIterations;
        int m_nGlobalEpisodes = 0;

        /// <summary>
        /// The OnIntialize event fires when initializing the trainer.
        /// </summary>
        public event EventHandler<InitializeArgs> OnInitialize;
        /// <summary>
        /// The OnShutdown event fires when shutting down the trainer.
        /// </summary>
        public event EventHandler OnShutdown;
        /// <summary>
        /// The OnGetData event fires from within the Train method and is used to get a new observation data.
        /// </summary>
        public event EventHandler<GetDataArgs> OnGetData;
        /// <summary>
        /// The OnGetStatus event fires on each iteration within the Train method.
        /// </summary>
        public event EventHandler<GetStatusArgs> OnGetStatus;
        /// <summary>
        /// The OnWait event fires when waiting for a shutdown.
        /// </summary>
        public event EventHandler<WaitArgs> OnWait;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="mycaffe">Specifies the MyCaffeControl to use for learning and prediction.</param>
        /// <param name="properties">Specifies the property set containing the key/value pairs of property settings.</param>
        /// <param name="random">Specifies a Random number generator used for random selection.</param>
        public Trainer(MyCaffeControl<T> mycaffe, PropertySet properties, Random random)
        {
            m_mycaffe = mycaffe;
            m_properties = properties;
            m_nThreads = properties.GetPropertyAsInt("Threads", 8);
            m_nOptimizers = properties.GetPropertyAsInt("Optimizers", 2);
            m_nIterations = mycaffe.CurrentProject.GetSolverSettingAsInt("max_iter").GetValueOrDefault(10000);
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

            if (OnInitialize != null)
            {
                InitializeArgs e = new InitializeArgs(m_brain.MyCaffeControl);
                OnInitialize(this, e);
            }

            return true;
        }

        private void wait(int nWait)
        {
            int nWaitInc = 250;
            int nTotalWait = 0;

            while (m_brain.ReferenceCount > 0 && nTotalWait < nWait)
            {
                if (OnWait != null)
                    OnWait(this, new WaitArgs(nWaitInc));
                else
                    Thread.Sleep(nWaitInc);

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

            if (OnShutdown != null)
                OnShutdown(this, new EventArgs());

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
            Environment<T> env = new Environment<T>(m_brain, m_properties, m_random, 1);
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
            Environment<T> env = new Environment<T>(m_brain, m_properties, m_random, nIterations);
            env.OnGetData += Env_OnGetData;
            env.OnGetStatus += Env_OnGetStatus;
            env.Start(10);

            while (!m_brain.MyCaffeControl.CancelEvent.WaitOne(250))
            {
                if (m_nGlobalEpisodes >= nIterations)
                    break;
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

            m_brain.MyCaffeControl.CancelEvent.Reset();

            for (int i = 0; i < m_nOptimizers; i++)
            {
                Optimizer<T> opt = new Optimizer<T>(m_brain);
                opt.Start(1);
                rgOptimizers.Add(opt);
            }

            for (int i = 0; i < m_nThreads; i++)
            {
                Environment<T> env = new Environment<T>(m_brain, m_properties, m_random, m_nIterations);
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

        private void Env_OnGetStatus(object sender, GetStatusArgs e)
        {
            m_nGlobalEpisodes = Math.Max(m_nGlobalEpisodes, e.Frames);

            if (OnGetStatus != null)
                OnGetStatus(sender, e);
        }

        private void Env_OnGetData(object sender, GetDataArgs e)
        {
            if (OnGetData != null)
                OnGetData(sender, e);
        }
    }

    class Worker /** @private */
    {
        protected int m_nIndex = -1;
        protected AutoResetEvent m_evtCancel = new AutoResetEvent(false);
        protected ManualResetEvent m_evtDone = new ManualResetEvent(false);
        protected Task m_workTask = null;

        public Worker()
        {
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

        public event EventHandler<GetDataArgs> OnGetData;
        public event EventHandler<GetStatusArgs> OnGetStatus;


        public Environment(Brain<T> brain, PropertySet properties, Random random, int nIterations)
        {
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

        private bool runEpisode()
        {
            // Reset the environment and get the initial state.
            GetDataArgs dataArg = new GetDataArgs(m_brain.MyCaffeControl, m_brain.MyCaffeControl.Log, m_brain.MyCaffeControl.CancelEvent, true, m_nIndex);
            OnGetData(this, dataArg);

            if (!dataArg.Success)
                return true;

            StateBase s = dataArg.State;
            bool bDone = false;
            double dfR = 0;

            m_nIndex = dataArg.Index;

            while (!m_evtCancel.WaitOne(1) && !bDone)
            {
                int a = m_agent.act(s);

                dataArg = new GetDataArgs(m_brain.MyCaffeControl, m_brain.MyCaffeControl.Log, m_brain.MyCaffeControl.CancelEvent, false, m_nIndex, a);
                OnGetData(this, dataArg);

                if (!dataArg.Success)
                    return true;

                if (m_brain.MyCaffeControl.CancelEvent.WaitOne(1))
                    return false;

                StateBase s_ = dataArg.State;
                double dfReward = s_.Reward;

                bDone = s_.Done;
                if (bDone)
                    s_ = null;

                m_agent.train(s, a, dfReward, s_);

                s = s_;
                dfR += dfReward;

                if (OnGetStatus != null)
                    OnGetStatus(this, new GetStatusArgs(Agent<T>.Frames, m_nIterations, dfR, m_agent.epsilon));
            }

            return true;
        }

        public Tuple<int, int> Run(int nDelay = 1000)
        {
            // Reset the environment and get the initial state.
            GetDataArgs dataArg = new GetDataArgs(m_brain.MyCaffeControl, m_brain.MyCaffeControl.Log, m_brain.MyCaffeControl.CancelEvent, true, -1, -1, false);
            OnGetData(this, dataArg);

            Thread.Sleep(nDelay);

            dataArg = new GetDataArgs(m_brain.MyCaffeControl, m_brain.MyCaffeControl.Log, m_brain.MyCaffeControl.CancelEvent, false, dataArg.Index, -1, false);
            OnGetData(this, dataArg);

            int a = m_agent.act(dataArg.State);

            return new Tuple<int, int>(a, dataArg.State.ActionCount);
        }
    }

    class Optimizer<T> : Worker, IDisposable /** @private */
    {
        Brain<T> m_brain;
        Blob<T> m_blobProb;
        Blob<T> m_blobVal;

        public Optimizer(Brain<T> brain)
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
            m_memory = new Memory<T>(brain.RewardScale);
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

        public int act(StateBase s)
        {
            double dfEpsilon = epsilon;

            m_nFrames++;

            // -- Exploration --
            if (m_random.NextDouble() < dfEpsilon)
                return m_random.Next(s.ActionCount);

            // -- Prediction --
            Tuple<float[], float> res = m_brain.predict(s);

            float[] rgProb = res.Item1;
            float fMax = -float.MaxValue;
            int nMaxIdx = 0;
            for (int i = 0; i < rgProb.Length; i++)
            {
                if (rgProb[i] > fMax)
                {
                    fMax = rgProb[i];
                    nMaxIdx = i;
                }
            }

            return nMaxIdx;
        }

        public MemoryItem get_sample(int n)
        {
            return new MemoryItem(m_memory[0].State0, m_memory[0].Action, m_dfR, m_memory[n - 1].State1);
        }

        public void train(StateBase s, int nAction, double dfReward, StateBase s_)
        {
            m_memory.Add(new MemoryItem(s, nAction, dfReward, s_));
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
        bool m_bSkipLoss = false;
        int m_nRefCount = 0;
        Random m_random;
        double m_dfRewardScale = 1.0;
        SoftmaxCrossEntropyLossLayer<T> m_softmaxCe;
        bool m_bSoftmaxSetup = false;

        public Brain(MyCaffeControl<T> mycaffe, PropertySet properties, Random random)
        {
            m_caffe = mycaffe;
            m_random = random;
            setupNet(m_caffe);

            m_nNStepReturn = properties.GetPropertyAsInt("NStepReturn", 8);
            m_dfGamma = properties.GetPropertyAsDouble("Gamma", 0.99);
            m_dfNGamma = Math.Pow(m_dfGamma,m_nNStepReturn);
            m_dfLossCoefficient = properties.GetPropertyAsDouble("LossCoefficient", 0.5);
            m_dfEntropyCoefficient = properties.GetPropertyAsDouble("EntropyCoefficient", 0.01);
            m_dfOptimalEpisodeCoefficient = properties.GetPropertyAsDouble("OptimalEpisodeCoefficient", 0.5);
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
            Net<T> net = mycaffe.GetInternalNet(Phase.TRAIN);
            MemoryLossLayer<T> memLoss = getLayer(net, LayerParameter.LayerType.MEMORY_LOSS) as MemoryLossLayer<T>;

            if (memLoss == null)
                throw new Exception("Could not find a MemoryLossLayer in the training net!");

            memLoss.OnGetLoss += MemLoss_OnGetLoss;

            MemoryDataLayer<T> memData = getLayer(net, LayerParameter.LayerType.MEMORYDATA) as MemoryDataLayer<T>;

            if (memData == null)
                throw new Exception("Could not find the MemoryDataLayer in the training net!");

            m_nMinBatch = (int)memData.layer_param.memory_data_param.batch_size;
            m_trainingQueue = new Memory<T>(m_dfRewardScale);
        }

        private void cleanupNet(MyCaffeControl<T> mycaffe)
        {
            Net<T> net = mycaffe.GetInternalNet(Phase.TRAIN);
            MemoryLossLayer<T> memLoss = getLayer(net, LayerParameter.LayerType.MEMORY_LOSS) as MemoryLossLayer<T>;

            if (memLoss == null)
                throw new Exception("Could not find a MemoryLossLayer in the training net!");

            memLoss.OnGetLoss -= MemLoss_OnGetLoss;
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
                {
                    Thread.Sleep(0);
                    return null;
                }

                mem = m_trainingQueue.Clone(true);
            }

            m_rgTrainingQueues.Add(mem);

            if (m_random.NextDouble() < m_dfOptimalEpisodeCoefficient)
                return m_rgTrainingQueues.GetMemory(m_random);

            return mem;
        }

        public Tuple<float[], float> predict(StateBase s)
        {
            List<Datum> rgData = new List<Datum>();

            rgData.Add(new Datum(s.Data));

            Net<T> net = m_caffe.GetInternalNet(Phase.TRAIN);
            MemoryDataLayer<T> memData = getLayer(net, LayerParameter.LayerType.MEMORYDATA) as MemoryDataLayer<T>;
            if (memData == null)
                throw new Exception("Could not find the MemoryDataLayer in the training net!");

            lock (m_syncTrain)
            {
                Tuple<Blob<T>, Blob<T>> res = predict(net, memData, rgData);

                float[] rgProb = Utility.ConvertVecF<T>(res.Item1.update_cpu_data());
                float[] rgVal = Utility.ConvertVecF<T>(res.Item2.update_cpu_data());

                return new Tuple<float[], float>(rgProb, rgVal[0]);
            }
        }

        public Tuple<Blob<T>, Blob<T>> predict(Net<T> net, MemoryDataLayer<T> memData, List<Datum> rgData)
        {
            BlobCollection<T> colOut;
            double dfLoss;
            Blob<T> blobProb = null;
            Blob<T> blobVal = null;
            List<Datum> rgState = new List<Datum>();

            m_caffe.Log.Enable = false;
            memData.AddDatumVector(rgData, 1, true, true);

            // Run the forward but skip the loss for we are not learning on this pass.
            m_bSkipLoss = true;
            colOut = net.Forward(out dfLoss);
            m_bSkipLoss = false;
            m_caffe.Log.Enable = true;

            // Find the softmax output for the action probabilities.
            for (int i = 0; i < colOut.Count; i++)
            {
                if (colOut[i].num_axes > 1 && colOut[i].count(1) > 1)
                {
                    m_blobOutProb.CopyFrom(colOut[i], false, true);
                    break;
                }
            }

            blobProb = m_blobOutProb;

            // Find the value bottom to the memory loss layer.
            MemoryLossLayer<T> memLoss = getLayer(net, LayerParameter.LayerType.MEMORY_LOSS) as MemoryLossLayer<T>;
            if (memLoss == null)
                throw new Exception("Could not find the MemoryLossLayer in the training net!");

            BlobCollection<T> colBottom = net.FindBottomBlobsOfLayer(memLoss.layer_param.name);
            for (int i = 0; i < colBottom.Count; i++)
            {
                if (colBottom[i].num_axes > 1 && colBottom[i].count(1) == 1)
                {
                    m_blobOutVal.CopyFrom(colBottom[i], false, true);
                    break;
                }
            }

            blobVal = m_blobOutVal;

            return new Tuple<Blob<T>, Blob<T>>(blobProb, blobVal);
        }

        public bool optimize(Blob<T> blobProb, Blob<T> blobVal)
        {
            Memory<T> batch = get_batch();
            if (batch == null)
                return false;

            Net<T> net = m_caffe.GetInternalNet(Phase.TRAIN);          
            MemoryDataLayer<T> memData = getLayer(net, LayerParameter.LayerType.MEMORYDATA) as MemoryDataLayer<T>;
            if (memData == null)
                throw new Exception("Could not find the MemoryDataLayer in the training net!");

            lock (m_syncTrain)
            {
                // predict_v(s_)
                List<Datum> rgNewStates = batch.GetStates(Memory<T>.STATE_TYPE.NEW);
                Tuple<Blob<T>, Blob<T>> res = predict(net, memData, rgNewStates);
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
                m_caffe.Cuda.mul(blobVal.count(), blobVal.gpu_data, m_blobSmask.gpu_data, m_blobR.mutable_gpu_diff);
                // blobR.diff = GAMMA_N * (v * s_mask)
                m_caffe.Cuda.mul_scalar(m_blobR.count(), m_dfNGamma, m_blobR.mutable_gpu_diff);
                // blobR.data = blobR.data + blobR.diff
                m_caffe.Cuda.add(m_blobR.count(), m_blobR.gpu_data, m_blobR.gpu_diff, m_blobR.mutable_gpu_data);

                // Train (action one hot used in the loss function fired during the MemoryLossLayer forward)
                m_caffe.Log.Enable = false;
                List<Datum> rgStates = batch.GetStates(Memory<T>.STATE_TYPE.OLD);
                memData.AddDatumVector(rgStates, 1, true, true);
                m_caffe.Train(1, 0, TRAIN_STEP.NONE, 0, true);
                m_caffe.Log.Enable = true;
            }

            return true;
        }

        private void MemLoss_OnGetLoss(object sender, MemoryLossLayerGetLossArgs<T> e)
        {
            if (m_bSkipLoss)
                return;

            Blob<T> blobValues = null;
            Blob<T> blobLogits = null;
            Blob<T> blobProb = null;
            int nIdxValues = 0;
            int nIdxLogits = 0;

            Net<T> net = m_caffe.GetInternalNet(Phase.TRAIN);
            SoftmaxLayer<T> softmax = getLayer(net, LayerParameter.LayerType.SOFTMAX) as SoftmaxLayer<T>;
            if (softmax == null)
                throw new Exception("Expected a softmax layer for action probabilities!");

            BlobCollection<T> col = net.FindTopBlobsOfLayer(softmax.layer_param.name);
            blobProb = col[0];

            if (e.Bottom.Count != 2)
                throw new Exception("Expected only two bottom values: logits(action_size), values(1)");

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

            if (blobValues == null)
                throw new Exception("Could not find the values blob collection!");

            if (blobLogits == null)
                throw new Exception("Could not find the values blob collection!");

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

            // calculate 'log_prob'
            m_blobWork.ReshapeLike(blobLogits);
            m_caffe.Cuda.mul(nLogitCount, blobProb.gpu_data, m_blobActionOneHot.gpu_data, m_blobWork.mutable_gpu_data);
            T[] rgSum = asum(nLogitCount, nValueCount, m_blobWork);
            m_blobLogProb.SetData(rgSum);
            m_caffe.Cuda.log(nValueCount, m_blobLogProb.gpu_data, m_blobLogProb.mutable_gpu_data, 1.0, 1e-10);

            // calculate 'advantage'
            m_caffe.Cuda.sub(nValueCount, m_blobR.gpu_data, blobValues.gpu_data, m_blobAdvantage.mutable_gpu_data);

            // calculate 'loss_policy'
            m_caffe.Cuda.mul(nValueCount, m_blobLogProb.gpu_data, m_blobAdvantage.gpu_data, m_blobLossPolicy.mutable_gpu_data);
            m_caffe.Cuda.mul_scalar(nValueCount, -1.0, m_blobLossPolicy.mutable_gpu_data);

            // calculate 'loss value'
            m_caffe.Cuda.powx(nValueCount, m_blobAdvantage.gpu_data, 2.0f, m_blobLossValue.mutable_gpu_data);
            m_caffe.Cuda.mul_scalar(nValueCount, m_dfLossCoefficient, m_blobLossValue.mutable_gpu_data);

            // calculate 'entropy'
            m_caffe.Cuda.log(nLogitCount, blobProb.gpu_data, m_blobWork.mutable_gpu_diff, 1.0, 1e-10);
            m_caffe.Cuda.mul(nLogitCount, blobProb.gpu_data, m_blobWork.gpu_diff, m_blobWork.mutable_gpu_data);
            rgSum = asum(nLogitCount, nValueCount, m_blobWork);
            m_blobEntropy.SetData(rgSum);
            m_caffe.Cuda.mul_scalar(nValueCount, m_dfEntropyCoefficient, m_blobEntropy.mutable_gpu_data);

            // calculate 'total_loss'
            m_blobWork.ReshapeLike(blobValues);
            m_caffe.Cuda.add(nValueCount, m_blobLossPolicy.gpu_data, m_blobLossValue.gpu_data, m_blobWork.mutable_gpu_diff);
            m_caffe.Cuda.add(nValueCount, m_blobEntropy.gpu_data, m_blobWork.gpu_diff, m_blobWork.mutable_gpu_diff);      
            double dfAsum = Utility.ConvertVal<T>(m_blobWork.asum_diff());
            double dfTotalLoss = dfAsum / m_blobWork.count(); // mean

            dfTotalLoss /= e.Normalizer;
            e.Loss = dfTotalLoss;

            // Apply the loss to the value diffs.
            blobValues.SetDiff(dfTotalLoss);

            // Apply the loss to the logits diffs.
            BlobCollection<T> colBottom = new BlobCollection<T>();
            BlobCollection<T> colTop = new BlobCollection<T>();

            colBottom.Add(blobLogits);          // logits
            colBottom.Add(m_blobActionOneHot);  // targets
            colTop.Add(m_blobLoss);

            if (!m_bSoftmaxSetup)
            {
                m_softmaxCe.Setup(colBottom, colTop);
                m_bSoftmaxSetup = true;
            }

            m_softmaxCe.Forward(colBottom, colTop);
            m_blobLoss.SetDiff(dfTotalLoss);
            m_softmaxCe.Backward(colTop, new List<bool>() { true, false }, colBottom);
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
