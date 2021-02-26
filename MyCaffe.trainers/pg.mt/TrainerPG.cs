using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
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

namespace MyCaffe.trainers.pg.mt
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
        CryptoRandom m_random = new CryptoRandom();
        MyCaffeControl<T> m_mycaffe;
        PropertySet m_properties;
        int m_nThreads = 1;
        List<int> m_rgGpuID = new List<int>();
        Optimizer<T> m_optimizer = null;

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

            m_nThreads = m_properties.GetPropertyAsInt("Threads", 1);
            m_rgGpuID.Add(m_mycaffe.Cuda.GetDeviceID());

            string strGpuID = m_properties.GetProperty("GPUIDs", false);
            if (strGpuID != null && m_nThreads > 1)
            {
                int nDeviceCount = m_mycaffe.Cuda.GetDeviceCount();

                m_rgGpuID.Clear();
                string[] rgstrGpuIDs = strGpuID.Split(',');
                foreach (string strID in rgstrGpuIDs)
                {
                    int nDevId = int.Parse(strID);

                    if (nDevId < 0 || nDevId >= nDeviceCount)
                        throw new Exception("Invalid device ID - value must be within the range [0," + (nDeviceCount - 1).ToString() + "].");

                    m_rgGpuID.Add(nDevId);
                }
            }
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
        /// Run a single cycle on the environment after the delay.
        /// </summary>
        /// <param name="nDelay">Specifies a delay to wait before running.</param>
        /// <returns>The results of the run containing the action are returned.</returns>
        public ResultCollection RunOne(int nDelay = 1000)
        {
            m_mycaffe.CancelEvent.Reset();
            Agent<T> agent = new Agent<T>(0, m_icallback, m_mycaffe, m_properties, m_random, Phase.TRAIN, 0, 1);
            Tuple<int,int> res = agent.Run(nDelay);

            List<Result> rgActions = new List<Result>();
            for (int i = 0; i < res.Item2; i++)
            {
                if (res.Item1 == i)
                    rgActions.Add(new Result(i, 1.0));
                else
                    rgActions.Add(new Result(i, 0.0));
            }

            agent.Dispose();

            return new ResultCollection(rgActions, LayerParameter.LayerType.SOFTMAX);
        }

        /// <summary>
        /// Run a set of iterations and return the resuts.
        /// </summary>
        /// <param name="nN">Specifies the number of samples to run.</param>
        /// <param name="runProp">Optionally specifies properties to use when running.</param>
        /// <param name="type">Returns the data type contained in the byte stream.</param>
        /// <returns>The results of the run containing the action are returned as a byte stream.</returns>
        public byte[] Run(int nN, PropertySet runProp, out string type)
        {
            m_mycaffe.CancelEvent.Reset();
            Agent<T> agent = new Agent<T>(0, m_icallback, m_mycaffe, m_properties, m_random, Phase.RUN, 0, 1);
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
            Agent<T> agent = new Agent<T>(0, m_icallback, m_mycaffe, properties, m_random, Phase.TRAIN, 0, 1);           
            agent.Run(Phase.TEST, nN, type, TRAIN_STEP.NONE);

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
            List<Agent<T>> rgAgents = new List<Agent<T>>();
            int nGpuIdx = 0;

            m_mycaffe.CancelEvent.Reset();

            if (m_nThreads > 1)
                m_optimizer = new Optimizer<T>(m_mycaffe);

            for (int i = 0; i < m_nThreads; i++)
            {
                int nGpuID = m_rgGpuID[nGpuIdx];

                Agent<T> agent = new Agent<T>(i, m_icallback, m_mycaffe, m_properties, m_random, Phase.TRAIN, nGpuID, m_nThreads);
                agent.OnApplyUpdates += Agent_OnApplyUpdates;
                rgAgents.Add(agent);

                nGpuIdx++;
                if (nGpuIdx == m_rgGpuID.Count)
                    nGpuIdx = 0;
            }

            if (m_optimizer != null)
                m_optimizer.Start(new WorkerStartArgs(0, Phase.TRAIN, nN, type, step));

            WorkerStartArgs args = new WorkerStartArgs(1, Phase.TRAIN, nN, type, step);            
            foreach (Agent<T> agent in rgAgents)
            {
                agent.Start(args);
            }

            while (!m_mycaffe.CancelEvent.WaitOne(250))
            {
            }

            foreach (Agent<T> agent in rgAgents)
            {
                agent.Stop(1000);
                agent.Dispose();
            }

            if (m_optimizer != null)
            {
                m_optimizer.Stop(1000);
                m_optimizer.Dispose();
                m_optimizer = null;
            }

            Shutdown(3000);

            return false;
        }

        private void Agent_OnApplyUpdates(object sender, ApplyUpdateArgs<T> e)
        {
            if (m_optimizer != null)
                m_optimizer.ApplyUpdates(e.MyCaffeWorker, e.Iteration);
        }
    }

    /// <summary>
    /// The WorkerStartArgs provides the arguments used when starting the agent thread.
    /// </summary>
    class WorkerStartArgs
    {
        int m_nCycleDelay;
        Phase m_phase;
        int m_nN;
        ITERATOR_TYPE m_type;
        TRAIN_STEP m_step = TRAIN_STEP.NONE;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="nCycleDelay">Specifies the cycle delay specifies the amount of time to wait for a cancel on each training loop.</param>
        /// <param name="phase">Specifies the phase on which to run.</param>
        /// <param name="nN">Specifies the number of iterations (based on the ITERATION_TYPE) to run, or -1 to ignore.</param>
        /// <param name="type">Specifies the iteration type (default = ITERATION).</param>
        /// <param name="step">Specifies a training step, if any - this is used during debugging.</param>
        public WorkerStartArgs(int nCycleDelay, Phase phase, int nN, ITERATOR_TYPE type, TRAIN_STEP step)
        {
            m_nCycleDelay = nCycleDelay;
            m_phase = phase;
            m_nN = nN;
            m_type = type;
            m_step = step;
        }

        /// <summary>
        /// Returns the training step to take (if any).  This is used for debugging.
        /// </summary>
        public TRAIN_STEP Step
        {
            get { return m_step; }
        }

        /// <summary>
        /// Returns the cycle delay which specifies the amount of time to wait for a cancel.
        /// </summary>
        public int CycleDelay
        {
            get { return m_nCycleDelay; }
        }

        /// <summary>
        /// Return the phase on which to run.
        /// </summary>
        public Phase Phase
        {
            get { return m_phase; }
        }

        /// <summary>
        /// Returns the maximum number of episodes to run.
        /// </summary>
        public int N
        {
            get { return m_nN; }
        }

        /// <summary>
        /// Returns the iteration type.
        /// </summary>
        public ITERATOR_TYPE IterationType
        {
            get { return m_type; }
        }
    }

    /// <summary>
    /// The Worker class provides the base class for both the Environment and Optimizer and provides the basic threading functionality used by both.
    /// </summary>
    class Worker
    {
        /// <summary>
        /// Specifies the index of this worker.
        /// </summary>
        protected int m_nIndex = -1;
        /// <summary>
        /// Specfies the cancel event used to cancel this worker.
        /// </summary>
        protected AutoResetEvent m_evtCancel = new AutoResetEvent(false);
        /// <summary>
        /// Specfies the done event set when this worker completes.
        /// </summary>
        protected ManualResetEvent m_evtDone = new ManualResetEvent(false);
        /// <summary>
        /// Specifies the worker task that runs the thread function.
        /// </summary>
        protected Task m_workTask = null;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="nIdx">Specifies the index of this worker.</param>
        public Worker(int nIdx)
        {
            m_nIndex = nIdx;
        }

        /// <summary>
        /// This is the actual thread function that is overriden by each derivative class.
        /// </summary>
        /// <param name="arg">Specifies the arguments to the thread function.</param>
        protected virtual void doWork(object arg)
        {
        }

        /// <summary>
        /// Start running the thread.
        /// </summary>
        /// <param name="args">Specifies the start arguments.</param>
        public void Start(WorkerStartArgs args)
        {
            if (m_workTask == null)
                m_workTask = Task.Factory.StartNew(new Action<object>(doWork), args);
        }

        /// <summary>
        /// Stop running the thread.
        /// </summary>
        /// <param name="nWait">Specifies an amount of time to wait for the thread to terminate.</param>
        public void Stop(int nWait)
        {
            m_evtCancel.Set();
            m_workTask = null;
            m_evtDone.WaitOne(nWait);
        }
    }

    /// <summary>
    /// The Optimizer manages a single thread used to apply updates to the primary instance of MyCaffe.  Once applied,
    /// the new weights are then copied back to the worker who just applied its gradients to the primary MyCaffe.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    class Optimizer<T> : Worker, IDisposable
    {
        MyCaffeControl<T> m_mycaffePrimary;
        MyCaffeControl<T> m_mycaffeWorker;
        int m_nIteration;
        double m_dfLearningRate;
        AutoResetEvent m_evtApplyUpdates = new AutoResetEvent(false);
        ManualResetEvent m_evtDoneApplying = new ManualResetEvent(false);
        object m_syncObj = new object();

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="mycaffePrimary">Specifies the primary MyCaffe instance that holds the open project to be trained.</param>
        public Optimizer(MyCaffeControl<T> mycaffePrimary)
            : base(0)
        {
            m_mycaffePrimary = mycaffePrimary;
        }

        /// <summary>
        /// Release all resources used.
        /// </summary>
        public void Dispose()
        {
        }

        /// <summary>
        /// This override is the thread used to apply all updates, its CUDA DeviceID is set to the same device ID used by the primary
        /// instance of MyCaffe.
        /// </summary>
        /// <param name="arg">Specifies the argument to the thread.</param>
        protected override void doWork(object arg)
        {
            WorkerStartArgs args = arg as WorkerStartArgs;

            m_mycaffePrimary.Cuda.SetDeviceID();

            List<WaitHandle> rgWait = new List<WaitHandle>();
            rgWait.Add(m_evtApplyUpdates);
            rgWait.AddRange(m_mycaffePrimary.CancelEvent.Handles);

            int nWait = WaitHandle.WaitAny(rgWait.ToArray());

            while (nWait == 0)
            {
                if (args.Step != TRAIN_STEP.FORWARD)
                {
                    m_mycaffePrimary.CopyGradientsFrom(m_mycaffeWorker);
                    m_mycaffePrimary.Log.Enable = false;
                    m_dfLearningRate = m_mycaffePrimary.ApplyUpdate(m_nIteration);
                    m_mycaffePrimary.Log.Enable = true;
                    m_mycaffeWorker.CopyWeightsFrom(m_mycaffePrimary);
                }

                m_evtDoneApplying.Set();

                nWait = WaitHandle.WaitAny(rgWait.ToArray());

                if (args.Step != TRAIN_STEP.NONE)
                    break;
            }
        }

        /// <summary>
        /// The ApplyUpdates function sets the parameters, signals the Apply Updates thread, blocks for the operation to complete and
        /// returns the learning rate used.
        /// </summary>
        /// <param name="mycaffeWorker">Specifies the worker instance of MyCaffe whos gradients are to be applied to the primary instance.</param>
        /// <param name="nIteration">Specifies the iteration of the gradients.</param>
        /// <returns>The learning rate used is returned.</returns>
        public double ApplyUpdates(MyCaffeControl<T> mycaffeWorker, int nIteration)
        {
            lock (m_syncObj)
            {
                m_mycaffeWorker = mycaffeWorker;
                m_nIteration = nIteration;

                m_evtDoneApplying.Reset();
                m_evtApplyUpdates.Set();

                List<WaitHandle> rgWait = new List<WaitHandle>();
                rgWait.Add(m_evtDoneApplying);
                rgWait.AddRange(m_mycaffePrimary.CancelEvent.Handles);

                int nWait = WaitHandle.WaitAny(rgWait.ToArray());
                if (nWait != 0)
                    return 0;

                return m_dfLearningRate;
            }
        }
    }

    /// <summary>
    /// The Agent both builds episodes from the envrionment and trains on them using the Brain.
    /// </summary>
    /// <typeparam name="T">Specifies the base type, which should be the same base type used for MyCaffe.  This type is either <i>double</i> or <i>float</i>.</typeparam>
    class Agent<T> : Worker, IDisposable 
    {
        IxTrainerCallback m_icallback;
        Brain<T> m_brain;
        PropertySet m_properties;
        CryptoRandom m_random;
        float m_fGamma;
        bool m_bAllowDiscountReset = false;
        bool m_bUseRawInput = false;
        int m_nEpsSteps = 0;
        double m_dfEpsStart = 0;
        double m_dfEpsEnd = 0;
        double m_dfExplorationRate = 0;
        int m_nEpisodeBatchSize = 1;
        double m_dfEpisodeElitePercentile = 1;
        static object m_syncObj = new object();
        bool m_bShowActionProb = false;
        bool m_bVerbose = false;

        /// <summary>
        /// The OnApplyUpdates event fires each time the Agent needs to apply its updates to the primary instance of MyCaffe.
        /// </summary>
        public event EventHandler<ApplyUpdateArgs<T>> OnApplyUpdates;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="nIdx">Specifies the index of this agent.</param>
        /// <param name="icallback">Specifies the callback used for update notifications sent to the parent.</param>
        /// <param name="mycaffe">Specifies the instance of MyCaffe with the open project.</param>
        /// <param name="properties">Specifies the properties passed into the trainer.</param>
        /// <param name="random">Specifies the random number generator used.</param>
        /// <param name="phase">Specifies the phase of the internal network to use.</param>
        /// <param name="nGpuID">Specifies the GPUID on which to run this brain.</param>
        /// <param name="nThreadCount">Specifies the total number of agents used.</param>
        public Agent(int nIdx, IxTrainerCallback icallback, MyCaffeControl<T> mycaffe, PropertySet properties, CryptoRandom random, Phase phase, int nGpuID, int nThreadCount)
            : base(nIdx)
        {
            m_icallback = icallback;
            m_brain = new Brain<T>(mycaffe, properties, random, phase, nGpuID, nThreadCount);
            m_brain.OnApplyUpdate += brain_OnApplyUpdate;
            m_properties = properties;
            m_random = random;

            m_fGamma = (float)properties.GetPropertyAsDouble("Gamma", 0.99);
            m_bAllowDiscountReset = properties.GetPropertyAsBool("AllowDiscountReset", false);
            m_bUseRawInput = properties.GetPropertyAsBool("UseRawInput", false);
            m_nEpsSteps = properties.GetPropertyAsInt("EpsSteps", 0);
            m_dfEpsStart = properties.GetPropertyAsDouble("EpsStart", 0);
            m_dfEpsEnd = properties.GetPropertyAsDouble("EpsEnd", 0);
            m_nEpisodeBatchSize = m_properties.GetPropertyAsInt("EpisodeBatchSize", 1);
            m_dfEpisodeElitePercentile = properties.GetPropertyAsDouble("EpisodeElitePercent", 1.0);
            m_bShowActionProb = properties.GetPropertyAsBool("ShowActionProb", false);
            m_bVerbose = properties.GetPropertyAsBool("Verbose", false);

            if (m_dfEpsStart < 0 || m_dfEpsStart > 1)
                throw new Exception("The 'EpsStart' is out of range - please specify a real number in the range [0,1]");

            if (m_dfEpsEnd < 0 || m_dfEpsEnd > 1)
                throw new Exception("The 'EpsEnd' is out of range - please specify a real number in the range [0,1]");

            if (m_dfEpsEnd > m_dfEpsStart)
                throw new Exception("The 'EpsEnd' must be less than the 'EpsStart' value.");
        }

        private void brain_OnApplyUpdate(object sender, ApplyUpdateArgs<T> e)
        {
            if (OnApplyUpdates != null)
                OnApplyUpdates(sender, e);
        }

        /// <summary>
        /// Release all resources used.
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
        /// This is the main agent thread that runs the agent.
        /// </summary>
        /// <param name="arg">Specifies the agent thread start arguments.</param>
        protected override void doWork(object arg)
        {
            try
            {
                WorkerStartArgs args = arg as WorkerStartArgs;

                lock (m_syncObj)
                {
                    m_brain.Create();
                }

                m_evtDone.Reset();
                m_evtCancel.Reset();
                Run(args.Phase, args.N, args.IterationType, args.Step);
                m_evtDone.Set();
            }
            catch (Exception excpt)
            {
                m_brain.OutputLog.WriteError(excpt);
            }

            m_brain.Cancel.Set();
        }

        private double getEpsilon(int nEpisode)
        {
            if (m_nEpsSteps == 0)
                return 0;

            if (nEpisode >= m_nEpsSteps)
                return m_dfEpsEnd;

            return m_dfEpsStart + (double)(nEpisode * (m_dfEpsEnd - m_dfEpsStart)/m_nEpsSteps);
        }

        private StateBase getData(Phase phase, int nIdx, int nAction, bool? bResetOverride = null)
        {
            GetDataArgs args = m_brain.getDataArgs(phase, nIdx, nAction, bResetOverride);
            m_icallback.OnGetData(args);
            return args.State;
        }

        private int getAction(int nEpisode, SimpleDatum sd, SimpleDatum sdClip, int nActionCount, TRAIN_STEP step, out float[] rgfAprob)
        {
            if (step == TRAIN_STEP.NONE)
            {
                m_dfExplorationRate = getEpsilon(nEpisode);

                if (m_dfExplorationRate > 0 && m_random.NextDouble() < m_dfExplorationRate)
                {
                    rgfAprob = new float[nActionCount];
                    int nAction = m_random.Next(nActionCount);
                    rgfAprob[nAction] = 1.0f;
                    return nAction;
                }
            }

            return m_brain.act(sd, sdClip, out rgfAprob);
        }

        private int updateStatus(int nIteration, int nEpisodeCount, double dfRunningReward, double dfRewardSum, double dfLoss, double dfLearningRate)
        {
            GetStatusArgs args = new GetStatusArgs(m_nIndex, nIteration, nEpisodeCount, 1000000, dfRunningReward, dfRewardSum, m_dfExplorationRate, 0, dfLoss, dfLearningRate);
            m_icallback.OnUpdateStatus(args);
            return args.NewFrameCount;
        }

        /// <summary>
        /// Run a single action on the model.
        /// </summary>
        /// <param name="nDelay">Specifies the delay between the reset and the data grab.</param>
        /// <returns>A tuple containing the action and action count is returned.</returns>
        public Tuple<int, int> Run(int nDelay = 1000)
        {
            // Reset the environment and get the initial state.
            getData(Phase.RUN, m_nIndex, -1);
            Thread.Sleep(nDelay);

            StateBase state = getData(Phase.RUN, m_nIndex, -1, false);
            float[] rgfAprob;

            m_brain.Create();

            int a = m_brain.act(state.Data, state.Clip, out rgfAprob);

            return new Tuple<int, int>(a, state.ActionCount);
        }

        /// <summary>
        /// Run the action on a set number of iterations and return the results with no training.
        /// </summary>
        /// <param name="nIterations">Specifies the iterations to run.</param>
        /// <param name="type">Specifies the type of data returned in the byte stream.</param>
        /// <returns>A byte stream of the results is returned.</returns>
        public byte[] Run(int nIterations, out string type)
        {
            IxTrainerCallbackRNN icallback = m_icallback as IxTrainerCallbackRNN;
            if (icallback == null)
                throw new Exception("The Run method requires an IxTrainerCallbackRNN interface to convert the results into the native format!");

            m_brain.Create();

            StateBase s = getData(Phase.RUN, m_nIndex, -1);
            int nIteration = 0;
            List<float> rgResults = new List<float>();
            int nLookahead = m_properties.GetPropertyAsInt("Lookahead", 0);

            while (!m_brain.Cancel.WaitOne(0) && (nIterations == -1 || nIteration < nIterations))
            {
                // Preprocess the observation.
                SimpleDatum x = m_brain.Preprocess(s, m_bUseRawInput);

                // Forward the policy network and sample an action.
                float[] rgfAprob;
                int nAction = m_brain.act(x, s.Clip, out rgfAprob);

                if (m_bShowActionProb && m_bVerbose)
                {
                    string strOut = "Action Prob: " + Utility.ToString<float>(rgfAprob.ToList(), 4) + " -> " + nAction.ToString();
                    m_brain.OutputLog.WriteLine(strOut);
                }

                int nSeqLen = m_brain.RecurrentSequenceLength;
                int nItemLen = s.Data.ItemCount / nSeqLen;
                int nData1Idx = s.Data.ItemCount - (nItemLen * (nLookahead + 1));

                rgResults.Add(s.Data.TimeStamp.ToFileTime());
                rgResults.Add((float)s.Data.GetDataAtF(nData1Idx));
                rgResults.Add(nAction);

                // Take the next step using the action
                s = getData(Phase.RUN, m_nIndex, nAction);
                nIteration++;

                m_brain.OutputLog.Progress = ((double)nIteration / (double)nIterations);
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
        /// <param name="step">Specifies the training step to take, if any.  This is only used when debugging.</param>
        public void Run(Phase phase, int nN, ITERATOR_TYPE type, TRAIN_STEP step)
        {
            MemoryCache rgMemoryCache = new MemoryCache(m_nEpisodeBatchSize);
            Memory rgMemory = new Memory();
            double? dfRunningReward = null;
            double dfEpisodeReward = 0;
            int nEpisode = 0;
            int nIteration = 0;

            m_brain.Create();

            StateBase s = getData(phase, m_nIndex, -1);

            while (!m_brain.Cancel.WaitOne(0) && !isAtIteration(nN, type, nIteration, nEpisode))
            {
                // Preprocess the observation.
                SimpleDatum x = m_brain.Preprocess(s, m_bUseRawInput);

                // Forward the policy network and sample an action.
                float[] rgfAprob;
                int action = getAction(nIteration, x, s.Clip, s.ActionCount, step, out rgfAprob);

                if (m_bShowActionProb && m_bVerbose)
                {
                    string strOut = "Action Prob: " + Utility.ToString<float>(rgfAprob.ToList(), 4) + " -> " + action.ToString();
                    m_brain.OutputLog.WriteLine(strOut);
                }

                if (step == TRAIN_STEP.FORWARD)
                    return;

                // Take the next step using the action
                StateBase s_ = getData(phase, m_nIndex, action);
                dfEpisodeReward += s_.Reward;

                if (phase == Phase.TRAIN)
                {
                    // Build up episode memory, using reward for taking the action.
                    rgMemory.Add(new MemoryItem(s, x, action, rgfAprob, (float)s_.Reward));

                    // An episode has finished.
                    if (s_.Done)
                    {
                        nEpisode++;
                        nIteration++;

                        if (rgMemoryCache.Add(rgMemory))
                        {
                            if (m_bShowActionProb)
                                m_brain.OutputLog.WriteLine("---learning---");

                            rgMemoryCache.PurgeNonElite(m_dfEpisodeElitePercentile);

                            for (int i=0; i<rgMemoryCache.Count; i++)
                            {
                                Memory rgMemory1 = rgMemoryCache[i];

                                m_brain.Reshape(rgMemory1);

                                // Compute the discounted reward (backwards through time)
                                float[] rgDiscountedR = rgMemory1.GetDiscountedRewards(m_fGamma, m_bAllowDiscountReset);
                                // Rewards are normalized when set to be unit normal (helps control the gradient estimator variance)
                                m_brain.SetDiscountedR(rgDiscountedR);

                                // Sigmoid models, set the probabilities up font.
                                if (!m_brain.UsesSoftMax)
                                {
                                    // Get the action probabilities.
                                    float[] rgfAprobSet = rgMemory1.GetActionProbabilities();
                                    // The action probabilities are used to calculate the initial gradient within the loss function.
                                    m_brain.SetActionProbabilities(rgfAprobSet);
                                }

                                // Get the action one-hot vectors.  When using Softmax, this contains the one-hot vector containing
                                // each action set (e.g. 3 actions with action 0 set would return a vector <1,0,0>).  
                                // When using a binary probability (e.g. with Sigmoid), the each action set only contains a
                                // single element which is set to the action value itself (e.g. 0 for action '0' and 1 for action '1')
                                float[] rgfAonehotSet = rgMemory1.GetActionOneHotVectors();
                                m_brain.SetActionOneHotVectors(rgfAonehotSet);

                                // Train for one iteration, which triggers the loss function.
                                List<Datum> rgData = rgMemory1.GetData();
                                List<Datum> rgClip = rgMemory1.GetClip();

                                m_brain.SetData(rgData, rgClip);

                                bool bApplyGradients = (i == rgMemoryCache.Count - 1) ? true : false;
                                m_brain.Train(nIteration, step, bApplyGradients);

                                // Update reward running
                                if (!dfRunningReward.HasValue)
                                    dfRunningReward = dfEpisodeReward;
                                else
                                    dfRunningReward = dfRunningReward * 0.99 + dfEpisodeReward * 0.01;

                                nEpisode = updateStatus(nIteration, nEpisode, dfRunningReward.Value, dfEpisodeReward, m_brain.LastLoss, m_brain.LearningRate);
                                dfEpisodeReward = 0;
                            }

                            rgMemoryCache.Clear();
                        }

                        s = getData(phase, m_nIndex, -1);
                        rgMemory = new Memory();

                        if (step != TRAIN_STEP.NONE)
                            return;
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

                        nEpisode = updateStatus(nIteration, nEpisode, dfRunningReward.Value, dfEpisodeReward, m_brain.LastLoss, m_brain.LearningRate);
                        dfEpisodeReward = 0;

                        s = getData(phase, m_nIndex, -1);
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

    /// <summary>
    /// The Brain uses the instance of MyCaffe (e.g. the open project) to run new actions and train the network.
    /// </summary>
    /// <typeparam name="T">Specifies the base type, which should be the same base type used for MyCaffe.  This type is either <i>double</i> or <i>float</i>.</typeparam>
    class Brain<T> : IDisposable
    {
        MyCaffeControl<T> m_mycaffePrimary;
        MyCaffeControl<T> m_mycaffeWorker;
        Net<T> m_net;
        Solver<T> m_solver;
        MemoryDataLayer<T> m_memData;
        MemoryLossLayer<T> m_memLoss;
        SoftmaxLayer<T> m_softmax = null;
        SoftmaxCrossEntropyLossLayer<T> m_softmaxCe = null;
        bool m_bSoftmaxCeSetup = false;
        PropertySet m_properties;
        CryptoRandom m_random;
        BlobCollection<T> m_colAccumulatedGradients = new BlobCollection<T>();
        Blob<T> m_blobDiscountedR;
        Blob<T> m_blobPolicyGradient;
        Blob<T> m_blobActionOneHot;
        Blob<T> m_blobDiscountedR1;
        Blob<T> m_blobPolicyGradient1;
        Blob<T> m_blobActionOneHot1;
        Blob<T> m_blobLoss;
        Blob<T> m_blobAprobLogit;
        bool m_bSkipLoss;
        int m_nMiniBatch = 10;
        SimpleDatum m_sdLast = null;
        double m_dfLastLoss = 0;
        double m_dfLearningRate = 0;
        Phase m_phase;
        int m_nGpuID = 0;
        int m_nThreadCount = 1;
        bool m_bCreated = false;
        bool m_bUseAcceleratedTraining = false;
        int m_nRecurrentSequenceLength = 0;
        List<Datum> m_rgData = null;
        List<Datum> m_rgClip = null;

        /// <summary>
        /// The OnApplyUpdate event fires when the Brain needs to apply its gradients to the primary instance of MyCaffe.
        /// </summary>
        public event EventHandler<ApplyUpdateArgs<T>> OnApplyUpdate;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="mycaffe">Specifies the instance of MyCaffe assoiated with the open project - when using more than one Brain, this is the master project.</param>
        /// <param name="properties">Specifies the properties passed into the trainer.</param>
        /// <param name="random">Specifies the random number generator used.</param>
        /// <param name="phase">Specifies the phase under which to run.</param>
        /// <param name="nGpuID">Specifies the GPUID on which to run this brain.</param>
        /// <param name="nThreadCount">Specifies the total number of threads used.</param>
        public Brain(MyCaffeControl<T> mycaffe, PropertySet properties, CryptoRandom random, Phase phase, int nGpuID, int nThreadCount)
        {
            m_properties = properties;
            m_random = random;
            m_phase = phase;
            m_nGpuID = nGpuID;
            m_nThreadCount = nThreadCount;
            m_mycaffePrimary = mycaffe;

            int nMiniBatch = mycaffe.CurrentProject.GetBatchSize(phase);
            if (nMiniBatch != 0)
                m_nMiniBatch = nMiniBatch;

            m_nMiniBatch = m_properties.GetPropertyAsInt("MiniBatch", m_nMiniBatch);

            double? dfRate = mycaffe.CurrentProject.GetSolverSettingAsNumeric("base_lr");
            if (dfRate.HasValue)
                m_dfLearningRate = dfRate.Value;

            m_bUseAcceleratedTraining = properties.GetPropertyAsBool("UseAcceleratedTraining", false);
        }

        /// <summary>
        /// Create the Brain CUDA objects - this is called on the thread from which the Brain runs.
        /// </summary>
        public void Create()
        {
            if (m_bCreated)
                return;

            m_mycaffePrimary.Log.Enable = false;

            if (m_nThreadCount == 1)
            {
                m_mycaffeWorker = m_mycaffePrimary;
                m_mycaffePrimary.Cuda.SetDeviceID();
            }
            else
            {
                m_mycaffeWorker = m_mycaffePrimary.Clone(m_nGpuID);
            }

            m_mycaffePrimary.Log.Enable = true;

            m_mycaffeWorker.Cuda.SetDeviceID();

            m_net = m_mycaffeWorker.GetInternalNet(m_phase);
            m_solver = m_mycaffeWorker.GetInternalSolver();

            m_memData = m_net.FindLayer(LayerParameter.LayerType.MEMORYDATA, null) as MemoryDataLayer<T>;
            m_memLoss = m_net.FindLayer(LayerParameter.LayerType.MEMORY_LOSS, null) as MemoryLossLayer<T>;
            m_softmax = m_net.FindLayer(LayerParameter.LayerType.SOFTMAX, null) as SoftmaxLayer<T>;

            if (m_memData == null)
                throw new Exception("Could not find the MemoryData Layer!");

            if (m_memLoss == null && m_phase != Phase.RUN)
                throw new Exception("Could not find the MemoryLoss Layer!");

            m_memData.OnDataPack += memData_OnDataPack;

            if (m_memLoss != null)
                m_memLoss.OnGetLoss += memLoss_OnGetLoss;

            m_blobDiscountedR = new Blob<T>(m_mycaffeWorker.Cuda, m_mycaffeWorker.Log);
            m_blobPolicyGradient = new Blob<T>(m_mycaffeWorker.Cuda, m_mycaffeWorker.Log);
            m_blobActionOneHot = new Blob<T>(m_mycaffeWorker.Cuda, m_mycaffeWorker.Log);
            m_blobDiscountedR1 = new Blob<T>(m_mycaffeWorker.Cuda, m_mycaffeWorker.Log);
            m_blobPolicyGradient1 = new Blob<T>(m_mycaffeWorker.Cuda, m_mycaffeWorker.Log);
            m_blobActionOneHot1 = new Blob<T>(m_mycaffeWorker.Cuda, m_mycaffeWorker.Log);
            m_blobLoss = new Blob<T>(m_mycaffeWorker.Cuda, m_mycaffeWorker.Log);
            m_blobAprobLogit = new Blob<T>(m_mycaffeWorker.Cuda, m_mycaffeWorker.Log);

            if (m_softmax != null)
            {
                LayerParameter p = new LayerParameter(LayerParameter.LayerType.SOFTMAXCROSSENTROPY_LOSS);
                p.loss_weight.Add(1);
                p.loss_weight.Add(0);
                p.loss_param.normalization = LossParameter.NormalizationMode.NONE;
                m_softmaxCe = new SoftmaxCrossEntropyLossLayer<T>(m_mycaffeWorker.Cuda, m_mycaffeWorker.Log, p);
            }

            m_colAccumulatedGradients = m_net.learnable_parameters.Clone();
            m_colAccumulatedGradients.SetDiff(0);

            m_bCreated = true;
        }

        private void dispose(ref Blob<T> b)
        {
            if (b != null)
            {
                b.Dispose();
                b = null;
            }
        }

        /// <summary>
        /// Release all resources used by the Brain.
        /// </summary>
        public void Dispose()
        {
            if (m_memLoss != null)
                m_memLoss.OnGetLoss -= memLoss_OnGetLoss;

            if (m_memData != null)
                m_memData.OnDataPack -= memData_OnDataPack;

            dispose(ref m_blobDiscountedR);
            dispose(ref m_blobPolicyGradient);
            dispose(ref m_blobActionOneHot);
            dispose(ref m_blobDiscountedR1);
            dispose(ref m_blobPolicyGradient1);
            dispose(ref m_blobActionOneHot1);
            dispose(ref m_blobLoss);
            dispose(ref m_blobAprobLogit);

            if (m_colAccumulatedGradients != null)
            {
                m_colAccumulatedGradients.Dispose();
                m_colAccumulatedGradients = null;
            }

            if (m_mycaffeWorker != m_mycaffePrimary && m_mycaffeWorker != null)
                m_mycaffeWorker.Dispose();

            m_mycaffeWorker = null;
        }

        /// <summary>
        /// Returns the recurrent sequence length detected when training a recurrent network, otherwise 0 is returned.
        /// </summary>
        public int RecurrentSequenceLength
        {
            get { return m_nRecurrentSequenceLength; }
        }

        /// <summary>
        /// Returns the primary MyCaffe output log for writing output information.
        /// </summary>
        public Log OutputLog
        {
            get { return m_mycaffePrimary.Log; }
        }

        /// <summary>
        /// Returns <i>true</i> if the current model uses a SoftMax, <i>false</i> otherwise.
        /// </summary>
        public bool UsesSoftMax
        {
            get { return (m_softmax == null) ? false : true; }
        }

        /// <summary>
        /// Reshape all Blobs used based on the Memory specified.
        /// </summary>
        /// <param name="mem">Specifies the Memory to reshape on.</param>
        /// <returns>The number of ActionProbs is returned.</returns>
        public int Reshape(Memory mem)
        {
            int nNum = mem.Count;
            int nChannels = mem[0].Data.Channels;
            int nHeight = mem[0].Data.Height;
            int nWidth = mem[0].Data.Height;
            int nActionProbs = 1;
            int nFound = 0;

            for (int i = 0; i < m_net.output_blobs.Count; i++)
            {
                if (m_net.output_blobs[i].type != BLOB_TYPE.LOSS)
                {
                    int nCh = m_net.output_blobs[i].channels;
                    nActionProbs = Math.Max(nCh, nActionProbs);
                    nFound++;
                }
            }

            if (nFound == 0)
                throw new Exception("Could not find a non-loss output!  Your model should output the loss and the action probabilities.");

            m_blobDiscountedR.Reshape(nNum, nActionProbs, 1, 1);
            m_blobPolicyGradient.Reshape(nNum, nActionProbs, 1, 1);
            m_blobActionOneHot.Reshape(nNum, nActionProbs, 1, 1);
            m_blobDiscountedR1.Reshape(nNum, nActionProbs, 1, 1);
            m_blobPolicyGradient1.Reshape(nNum, nActionProbs, 1, 1);
            m_blobActionOneHot1.Reshape(nNum, nActionProbs, 1, 1);
            m_blobLoss.Reshape(1, 1, 1, 1);

            return nActionProbs;
        }

        /// <summary>
        /// Sets the discounted returns in the Discounted Returns Blob.
        /// </summary>
        /// <param name="rg">Specifies the discounted return values.</param>
        public void SetDiscountedR(float[] rg)
        {
            double dfMean = m_blobDiscountedR.mean(rg);
            double dfStd = m_blobDiscountedR.std(dfMean, rg);
            int nC = m_blobDiscountedR.channels;

            // Fill all items in each channel with the same discount value.
            if (nC > 1)
            {
                List<float> rgR = new List<float>();

                for (int i = 0; i < rg.Length; i++)
                {
                    for (int j = 0; j < nC; j++)
                    {
                        rgR.Add(rg[i]);
                    }
                }

                rg = rgR.ToArray();
            }

            m_blobDiscountedR.SetData(Utility.ConvertVec<T>(rg));
            m_blobDiscountedR.NormalizeData(dfMean, dfStd);
        }

        /// <summary>
        /// Set the action probabilities in the Policy Gradient Blob.
        /// </summary>
        /// <param name="rg">Specifies the action probabilities (Aprob) values.</param>
        public void SetActionProbabilities(float[] rg)
        {
            m_blobPolicyGradient.SetData(Utility.ConvertVec<T>(rg));           
        }

        /// <summary>
        /// Set the action one-hot vectors in the Action OneHot Vector Blob.
        /// </summary>
        /// <param name="rg">Specifies the action one-hot vector values.</param>
        public void SetActionOneHotVectors(float[] rg)
        {
            m_blobActionOneHot.SetData(Utility.ConvertVec<T>(rg));
        }

        /// <summary>
        /// Add the data to the model by adding it to the MemoryData layer.
        /// </summary>
        /// <param name="rgData">Specifies the data to add.</param>
        /// <param name="rgClip">Specifies the clip data to add if any exists.</param>
        public void SetData(List<Datum> rgData, List<Datum> rgClip)
        {
            if (m_nRecurrentSequenceLength != 1 && rgData.Count > 1 && rgClip != null)
            {
                m_rgData = rgData;
                m_rgClip = rgClip;
            }
            else
            {
                m_memData.AddDatumVector(rgData, rgClip, 1, true, true);
                m_rgData = null;
                m_rgClip = null;
            }
        }

        /// <summary>
        /// Returns the GetDataArgs used to retrieve new data from the envrionment implemented by derived parent trainer.
        /// </summary>
        /// <param name="phase">Specifies the phase under which to get the data.</param>
        /// <param name="nIdx">Specifies the envrionment index.</param>
        /// <param name="nAction">Specifies the action to run, or -1 to reset the environment.</param>
        /// <param name="bResetOverride">Optionally, specifies to reset the environment when <i>true</i> (default = <i>null</i>).</param>
        /// <returns>A new GetDataArgs is returned.</returns>
        public GetDataArgs getDataArgs(Phase phase, int nIdx, int nAction, bool? bResetOverride = null)
        {
            bool bReset = (nAction == -1) ? true : false;
            return new GetDataArgs(phase, nIdx, m_mycaffePrimary, m_mycaffePrimary.Log, m_mycaffePrimary.CancelEvent, bReset, nAction, false, true);
        }

        /// <summary>
        /// Return the last loss received.
        /// </summary>
        public double LastLoss
        {
            get { return m_dfLastLoss; }
        }

        /// <summary>
        /// Return the learning rate used.
        /// </summary>
        public double LearningRate
        {
            get { return m_dfLearningRate; }
        }

        /// <summary>
        /// Returns the output log.
        /// </summary>
        public Log Log
        {
            get { return m_mycaffePrimary.Log; }
        }

        /// <summary>
        /// Returns the Cancel event used to cancel  all MyCaffe tasks.
        /// </summary>
        public CancelEvent Cancel
        {
            get { return m_mycaffePrimary.CancelEvent; }
        }

        /// <summary>
        /// Preprocesses the data.
        /// </summary>
        /// <param name="s">Specifies the state and data to use.</param>
        /// <param name="bUseRawInput">Specifies whether or not to use the raw data <i>true</i>, or a difference of the current and previous data <i>false</i> (default = <i>false</i>).</param>
        /// <returns></returns>
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

        /// <summary>
        /// Returns the action from running the model.  The action returned is either randomly selected (when using Exploration),
        /// or calculated via a forward pass (when using Exploitation).
        /// </summary>
        /// <param name="sd">Specifies the data to run the model on.</param>
        /// <param name="sdClip">Specifies the clip data (if any exits).</param>
        /// <param name="rgfAprob">Returns the Aprob values calculated (NOTE: this is only used in non-Softmax models).</param>
        /// <returns>The action value is returned.</returns>
        public int act(SimpleDatum sd, SimpleDatum sdClip, out float[] rgfAprob)
        {
            List<Datum> rgData = new List<Datum>();
            rgData.Add(new Datum(sd));
            List<Datum> rgClip = null;

            if (sdClip != null)
            {
                rgClip = new List<Datum>();
                rgClip.Add(new Datum(sdClip));
            }

            double dfLoss;
            float fRandom = (float)m_random.NextDouble(); // Roll the dice.

            m_memData.AddDatumVector(rgData, rgClip, 1, true, true);
            m_bSkipLoss = true;
            BlobCollection<T> res = m_net.Forward(out dfLoss);
            m_bSkipLoss = false;

            rgfAprob = null;

            for (int i = 0; i < res.Count; i++)
            {
                if (res[i].type != BLOB_TYPE.LOSS)
                {
                    int nStart = 0;
                    // When using recurrent learning, only act on the last outputs.
                    if (m_nRecurrentSequenceLength > 1 && res[i].num > 1)
                    {
                        int nCount = res[i].count();
                        int nOutput = nCount / res[i].num;
                        nStart = nCount - nOutput;

                        if (nStart < 0)
                            throw new Exception("The start must be zero or greater!");
                    }

                    rgfAprob = Utility.ConvertVecF<T>(res[i].update_cpu_data(), nStart);
                    break;
                }
            }

            if (rgfAprob == null)
                throw new Exception("Could not find a non-loss output!  Your model should output the loss and the action probabilities.");

            // Select the action from the probability distribution.
            float fSum = 0;
            for (int i = 0; i < rgfAprob.Length; i++)
            {
                fSum += rgfAprob[i];

                if (fRandom < fSum)
                    return i;
            }

            if (rgfAprob.Length == 1)
                return 1;

            return rgfAprob.Length - 1;
        }

        private void prepareBlob(Blob<T> b1, Blob<T> b)
        {
            b1.CopyFrom(b, 0, 0, b1.count(), true, true);
            b.Reshape(1, b.channels, b.height, b.width);
        }

        private void copyBlob(int nIdx, Blob<T> src, Blob<T> dst)
        {
            int nCount = dst.count();
            dst.CopyFrom(src, nIdx * nCount, 0, nCount, true, false);
        }

        /// <summary>
        /// Train the model at the current iteration.
        /// </summary>
        /// <param name="nIteration">Specifies the current iterations.  NOTE: at each 'MiniBatch' (specified as the <i>batch_size</i> in the model), the accumulated gradients are applied.</param>
        /// <param name="step">Specifies the training step to use (if any).  This is only used for debugging.</param>
        /// <param name="bApplyGradients">Apply the gradients.</param>
        public void Train(int nIteration, TRAIN_STEP step, bool bApplyGradients = true)
        {
            // Run data/clip groups > 1 in non batch mode.
            if (m_nRecurrentSequenceLength != 1 && m_rgData != null && m_rgData.Count > 1 && m_rgClip != null)
            {
                prepareBlob(m_blobActionOneHot1, m_blobActionOneHot);
                prepareBlob(m_blobDiscountedR1, m_blobDiscountedR);
                prepareBlob(m_blobPolicyGradient1, m_blobPolicyGradient);

                for (int i = 0; i < m_rgData.Count; i++)
                {
                    copyBlob(i, m_blobActionOneHot1, m_blobActionOneHot);
                    copyBlob(i, m_blobDiscountedR1, m_blobDiscountedR);
                    copyBlob(i, m_blobPolicyGradient1, m_blobPolicyGradient);

                    List<Datum> rgData1 = new List<Datum>() { m_rgData[i] };
                    List<Datum> rgClip1 = new List<Datum>() { m_rgClip[i] };

                    m_memData.AddDatumVector(rgData1, rgClip1, 1, true, true);
                   
                    m_solver.Step(1, step, true, m_bUseAcceleratedTraining, true, true);
                    m_colAccumulatedGradients.Accumulate(m_mycaffeWorker.Cuda, m_net.learnable_parameters, true);
                }

                m_blobActionOneHot.ReshapeLike(m_blobActionOneHot1);
                m_blobDiscountedR.ReshapeLike(m_blobDiscountedR1);
                m_blobPolicyGradient.ReshapeLike(m_blobPolicyGradient1);

                m_rgData = null;
                m_rgClip = null;
            }
            else
            {
                m_solver.Step(1, step, true, m_bUseAcceleratedTraining, true, true);
                m_colAccumulatedGradients.Accumulate(m_mycaffeWorker.Cuda, m_net.learnable_parameters, true);
            }

            if (nIteration % m_nMiniBatch == 0 || bApplyGradients || step == TRAIN_STEP.BACKWARD || step == TRAIN_STEP.BOTH)
            {
                m_net.learnable_parameters.CopyFrom(m_colAccumulatedGradients, true);
                m_colAccumulatedGradients.SetDiff(0);

                if (m_mycaffePrimary == m_mycaffeWorker)
                {
                    m_dfLearningRate = m_solver.ApplyUpdate(nIteration);
                }
                else
                {
                    ApplyUpdateArgs<T> args = new ApplyUpdateArgs<T>(nIteration, m_mycaffeWorker);
                    OnApplyUpdate(this, args);
                    m_dfLearningRate = args.LearningRate;
                }

                m_net.ClearParamDiffs();
            }
        }

        private T[] unpackLabel(Datum d)
        {
            if (d.DataCriteria == null)
                return null;

            if (d.DataCriteriaFormat == SimpleDatum.DATA_FORMAT.LIST_FLOAT)
            {
                List<float> rgf = BinaryData.UnPackFloatList(d.DataCriteria, SimpleDatum.DATA_FORMAT.LIST_FLOAT);
                return Utility.ConvertVec<T>(rgf.ToArray());
            }
            else if (d.DataCriteriaFormat == SimpleDatum.DATA_FORMAT.LIST_DOUBLE)
            {
                List<double> rgf = BinaryData.UnPackDoubleList(d.DataCriteria, SimpleDatum.DATA_FORMAT.LIST_DOUBLE);
                return Utility.ConvertVec<T>(rgf.ToArray());
            }

            return null;
        }

        /// <summary>
        /// Pack the data in an ordering expected by the RNN layer used.  This method is only called when using Data/Clip inputs.
        /// </summary>
        /// <param name="sender">Specifies the sender of the event, which is the MemoryDataLayer.</param>
        /// <param name="e">Specifies the event parameters.</param>
        /// <remarks>
        /// This method is responsible for packing the datum received into the data blob within the event arguments and do so in the
        /// expected LSTM ordering.
        /// </remarks>
        private void memData_OnDataPack(object sender, MemoryDataLayerPackDataArgs<T> e)
        {            
            List<int> rgDataShape = e.Data.shape();
            List<int> rgClipShape = e.Clip.shape();
            List<int> rgLabelShape = e.Label.shape();
            int nBatch = e.DataItems.Count;
            int nSeqLen = rgDataShape[0];

            e.Data.Log.CHECK_GT(nSeqLen, 0, "The sequence lenth must be greater than zero!");
            e.Data.Log.CHECK_EQ(nBatch, e.ClipItems.Count, "The data and clip should have the same number of items.");
            e.Data.Log.CHECK_EQ(nSeqLen, rgClipShape[0], "The data and clip should have the same sequence count.");

            rgDataShape[1] = nBatch;  // LSTM uses sizing: seq, batch, data1, data2
            rgClipShape[1] = nBatch;
            rgLabelShape[1] = nBatch;

            e.Data.Reshape(rgDataShape);
            e.Clip.Reshape(rgClipShape);
            e.Label.Reshape(rgLabelShape);

            T[] rgRawData = new T[e.Data.count()];
            T[] rgRawClip = new T[e.Clip.count()];
            T[] rgRawLabel = new T[e.Label.count()];

            int nDataSize = e.Data.count(2);
            T[] rgDataItem = new T[nDataSize];
            T dfClip;
            int nIdx;

            for (int i = 0; i < nBatch; i++)
            {
                Datum data = e.DataItems[i];
                Datum clip = e.ClipItems[i];

                T[] rgLabel = unpackLabel(data);

                for (int j = 0; j < nSeqLen; j++)
                {
                    dfClip = clip.GetDataAt<T>(j);

                    for (int k = 0; k < nDataSize; k++)
                    {
                        rgDataItem[k] = data.GetDataAt<T>(j * nDataSize + k);
                    }

                    // LSTM: Create input data, the data must be in the order
                    // seq1_val1, seq2_val1, ..., seqBatch_Size_val1, seq1_val2, seq2_val2, ..., seqBatch_Size_valSequence_Length
                    if (e.LstmType == LayerParameter.LayerType.LSTM)
                        nIdx = nBatch * j + i;  

                    // LSTM_SIMPLE: Create input data, the data must be in the order
                    // seq1_val1, seq1_val2, ..., seq1_valBatchSize, seq2_val1, seq2_val2, ..., seqSequenceLength_valBatchSize
                    else
                        nIdx = i * nBatch + j;  

                    Array.Copy(rgDataItem, 0, rgRawData, nIdx * nDataSize, nDataSize);
                    rgRawClip[nIdx] = dfClip;

                    if (rgLabel != null)
                    {
                        if (rgLabel.Length == nSeqLen)
                            rgRawLabel[nIdx] = rgLabel[j];
                        else if (rgLabel.Length == 1)
                        {
                            if (j == nSeqLen - 1)
                                rgRawLabel[0] = rgLabel[0];
                        }
                        else
                        {
                            throw new Exception("The Solver SequenceLength parameter does not match the actual sequence length!  The label length '" + rgLabel.Length.ToString() + "' must be either '1' for SINGLE labels, or the sequence length of '" + nSeqLen.ToString() + "' for MULTI labels.  Stopping training.");
                        }
                    }
                }
            }

            e.Data.mutable_cpu_data = rgRawData;
            e.Clip.mutable_cpu_data = rgRawClip;
            e.Label.mutable_cpu_data = rgRawLabel;
            m_nRecurrentSequenceLength = nSeqLen;
        }


        /// <summary>
        /// Calcualte the loss and initial gradients.  This event function fires, when training, during the forward pass of the MemoryLoss layer.
        /// </summary>
        /// <param name="sender">Specifies the MemoryLoss layer firing the event.</param>
        /// <param name="e">Specifies the arguments with the Bottom(s) flowing into the MemoryLoss layer and the loss value to be filled out.</param>
        /// <remarks>
        /// The initial gradient is calculated such that it encourages the action that was taken to be taken.
        /// 
        /// When using a Sigmoid, the gradient = (action=0) ? 1 - Aprob : 0 - Aprob.
        /// When using a Softmax, the gradient = the SoftmaxCrossEntropyLoss backward.
        /// 
        /// @see [CS231n Convolution Neural Networks for Visual Recognition](http://cs231n.github.io/neural-networks-2/#losses) by Karpathy, Stanford University
        /// 
        /// Regardless of the gradient used, the gradient is then modulated by multiplying it with the discounted rewards.
        /// </remarks>
        private void memLoss_OnGetLoss(object sender, MemoryLossLayerGetLossArgs<T> e)
        {
            if (m_bSkipLoss)
                return;

            int nCount = m_blobActionOneHot.count();
            long hActionOneHot = m_blobActionOneHot.gpu_data;
            long hPolicyGrad = 0;
            long hDiscountedR = m_blobDiscountedR.gpu_data;
            double dfLoss;
            int nDataSize = e.Bottom[0].count(1);
            bool bUsingEndData = false;

            // When using a recurrent model and receiving data with more than one sequence,
            // copy and only use the last sequence data.
            if (m_nRecurrentSequenceLength > 1)
            {
                if (e.Bottom[0].num > 1)
                {
                    m_blobAprobLogit.CopyFrom(e.Bottom[0], false, true);
                    m_blobAprobLogit.CopyFrom(e.Bottom[0], true);

                    List<int> rgShape = e.Bottom[0].shape();
                    rgShape[0] = 1;
                    e.Bottom[0].Reshape(rgShape);
                    e.Bottom[0].CopyFrom(m_blobAprobLogit, (m_blobAprobLogit.num - 1) * nDataSize, 0, nDataSize, true, true);
                    bUsingEndData = true;
                }
            }

            long hBottomDiff = e.Bottom[0].mutable_gpu_diff;

            // Calculate the initial gradients (policy grad initially just contains the action probabilities)
            if (m_softmax != null)
            {
                BlobCollection<T> colBottom = new BlobCollection<T>();
                BlobCollection<T> colTop = new BlobCollection<T>();

                colBottom.Add(e.Bottom[0]);             // aprob logit
                colBottom.Add(m_blobActionOneHot);      // action one-hot vectors
                colTop.Add(m_blobLoss);
                colTop.Add(m_blobPolicyGradient);

                if (!m_bSoftmaxCeSetup)
                {
                    m_softmaxCe.Setup(colBottom, colTop);
                    m_bSoftmaxCeSetup = true;
                }

                dfLoss = m_softmaxCe.Forward(colBottom, colTop);
                m_softmaxCe.Backward(colTop, new List<bool>() { true, false }, colBottom);
                hPolicyGrad = colBottom[0].gpu_diff;
            }
            else
            {
                hPolicyGrad = m_blobPolicyGradient.mutable_gpu_data;

                // Calculate (a=0) ? 1-aprob : 0-aprob
                m_mycaffeWorker.Cuda.add_scalar(nCount, -1.0, hActionOneHot); // invert one hot
                m_mycaffeWorker.Cuda.abs(nCount, hActionOneHot, hActionOneHot); 
                m_mycaffeWorker.Cuda.mul_scalar(nCount, -1.0, hPolicyGrad);   // negate Aprob
                m_mycaffeWorker.Cuda.add(nCount, hActionOneHot, hPolicyGrad, hPolicyGrad);  // gradient = ((a=0)?1:0) - Aprob
                dfLoss = Utility.ConvertVal<T>(m_blobPolicyGradient.sumsq_data());

                m_mycaffeWorker.Cuda.mul_scalar(nCount, -1.0, hPolicyGrad); // invert for ApplyUpdate subtracts the gradients
            }

            // Modulate the gradient with the advantage (PG magic happens right here.)
            m_mycaffeWorker.Cuda.mul(nCount, hPolicyGrad, hDiscountedR, hPolicyGrad);

            e.Loss = dfLoss;
            e.EnableLossUpdate = false; // dont apply loss to loss weight.

            if (hPolicyGrad != hBottomDiff)
                m_mycaffeWorker.Cuda.copy(nCount, hPolicyGrad, hBottomDiff);

            // When using recurrent model with more than one sequence of data, only
            // copy the diff to the last in the sequence and zero out the rest in the sequence.
            if (m_nRecurrentSequenceLength > 1 && bUsingEndData)
            {
                m_blobAprobLogit.SetDiff(0);
                m_blobAprobLogit.CopyFrom(e.Bottom[0], 0, (m_blobAprobLogit.num - 1) * nDataSize, nDataSize, false, true);
                e.Bottom[0].CopyFrom(m_blobAprobLogit, false, true);
                e.Bottom[0].CopyFrom(m_blobAprobLogit, true);
            }

            m_dfLastLoss = e.Loss;
        }
    }

    /// <summary>
    /// Contains the best memory episodes (best by highest total rewards)
    /// </summary>
    class MemoryCache : IEnumerable<Memory>
    {
        int m_nMax;
        List<Memory> m_rgMemory = new List<Memory>();

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="nMax">Specifies the memory cache size.</param>
        public MemoryCache(int nMax)
        {
            m_nMax = nMax;
        }

        /// <summary>
        /// Returns the number of items in the cache.
        /// </summary>
        public int Count
        {
            get { return m_rgMemory.Count; }
        }

        /// <summary>
        /// Returns an item at the specified index.
        /// </summary>
        /// <param name="nIdx">Specifies the index of the item to get.</param>
        /// <returns>The item at the index is returned.</returns>
        public Memory this[int nIdx]
        {
            get { return m_rgMemory[nIdx]; }
        }

        /// <summary>
        /// Add a new episode to the memory cache.
        /// </summary>
        /// <param name="mem"></param>
        /// <returns></returns>
        public bool Add(Memory mem)
        {
            m_rgMemory.Add(mem);

            if (m_rgMemory.Count == m_nMax)
                return true;

            return false;
        }

        /// <summary>
        /// Clear all items from the memory cache.
        /// </summary>
        public void Clear()
        {
            m_rgMemory.Clear();
        }

        /// <summary>
        /// Purge all non elite episodes.
        /// </summary>
        /// <param name="dfElitePercent">Specifies the pecentile of elite to keep (e.g. 0.7 specifies to keep the top 70% by reward sum).</param>
        public void PurgeNonElite(double dfElitePercent)
        {
            if (dfElitePercent <= 0.0 || dfElitePercent >= 1.0)
                return;

            double dfMin = m_rgMemory.Min(p => p.RewardSum);
            double dfMax = m_rgMemory.Max(p => p.RewardSum);
            double dfRange = dfMax - dfMin;
            double dfCutoff = dfMin + ((1.0 - dfElitePercent) * dfRange);
            List<Memory> rgMem = m_rgMemory.OrderByDescending(p => p.RewardSum).ToList();
            List<Memory> rgElite = new List<Memory>();

            for (int i = 0; i < rgMem.Count; i++)
            {
                double dfSum = rgMem[i].RewardSum;

                if (dfSum >= dfCutoff)
                    rgElite.Add(rgMem[i]);
                else
                    break;
            }

            m_rgMemory = rgElite;
        }

        /// <summary>
        /// Returns the enumerator.
        /// </summary>
        /// <returns>The enumerator is returned.</returns>
        public IEnumerator<Memory> GetEnumerator()
        {
            return m_rgMemory.GetEnumerator();
        }

        /// <summary>
        /// Returns the enumerator.
        /// </summary>
        /// <returns>The enumerator is returned.</returns>
        IEnumerator IEnumerable.GetEnumerator()
        {
            return m_rgMemory.GetEnumerator();
        }
    }

    /// <summary>
    /// Specifies a single Memory (e.g. an episode).
    /// </summary>
    class Memory
    {
        List<MemoryItem> m_rgItems = new List<MemoryItem>();
        int m_nEpisodeNumber = 0;
        double m_dfRewardSum = 0;

        /// <summary>
        /// The constructor.
        /// </summary>
        public Memory()
        {
        }

        /// <summary>
        /// Returns the number of memory items in the memory.
        /// </summary>
        public int Count
        {
            get { return m_rgItems.Count; }
        }

        /// <summary>
        /// Add a new item to the memory.
        /// </summary>
        /// <remarks>
        /// If the MaxItems is exceeded, the list of items is sorted and the bottom item(s) are dropped.
        /// </remarks>
        /// <param name="item">Specifies the item to add.</param>
        public void Add(MemoryItem item)
        {
            m_dfRewardSum += item.Reward;
            m_rgItems.Add(item);
        }

        /// <summary>
        /// Remove all items in the list.
        /// </summary>
        public void Clear()
        {
            m_dfRewardSum = 0;
            m_rgItems.Clear();
        }

        /// <summary>
        /// Get/set an item within the memory at a given index.
        /// </summary>
        /// <param name="nIdx">Specifies the index of the item to access.</param>
        /// <returns>The item at the index is returned.</returns>
        public MemoryItem this[int nIdx]
        {
            get { return m_rgItems[nIdx]; }
            set { m_rgItems[nIdx] = value; }
        }

        /// <summary>
        /// Get/set the episode number of this memory.
        /// </summary>
        public int EpisodeNumber
        {
            get { return m_nEpisodeNumber; }
            set { m_nEpisodeNumber = value; }
        }

        /// <summary>
        /// Get/set the reward sum of this memory.
        /// </summary>
        public double RewardSum
        {
            get { return m_dfRewardSum; }
            set { m_dfRewardSum = value; }
        }

        /// <summary>
        /// Retrieve the discounted rewards for this episode.
        /// </summary>
        /// <param name="fGamma">Specifies the discounting factor.</param>
        /// <param name="bAllowReset">Specifies whether or not to allow resetting the running discount on non zero values.</param>
        /// <returns>The discounted rewards is returned (one value for each step in the episode).</returns>
        public float[] GetDiscountedRewards(float fGamma, bool bAllowReset)
        {
            float[] rgR = m_rgItems.Select(p => p.Reward).ToArray();
            float fRunningAdd = 0;
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

        /// <summary>
        /// Retrieve the action probabilities of the episode.
        /// </summary>
        /// <remarks>
        /// NOTE: These values are only used in non-Softmax models.
        /// </remarks>
        /// <returns>The action probabilities (Aprob) values are returned.</returns>
        public float[] GetActionProbabilities()
        {
            List<float> rgfAprob = new List<float>();

            for (int i = 0; i < m_rgItems.Count; i++)
            {
                rgfAprob.AddRange(m_rgItems[i].Aprob);
            }

            return rgfAprob.ToArray();
        }

        /// <summary>
        /// Retrieve the action one-hot vectors for the episode.
        /// </summary>
        /// <returns>The action one-hot vector values are returned.</returns>
        public float[] GetActionOneHotVectors()
        {
            List<float> rgfAonehot = new List<float>();

            for (int i = 0; i < m_rgItems.Count; i++)
            {
                float[] rgfOneHot = new float[m_rgItems[0].Aprob.Length];

                if (rgfOneHot.Length == 1)
                    rgfOneHot[0] = m_rgItems[i].Action;
                else
                    rgfOneHot[m_rgItems[i].Action] = 1;

                rgfAonehot.AddRange(rgfOneHot);
            }

            return rgfAonehot.ToArray();
        }

        /// <summary>
        /// Retrieve the data of each step in the episode.
        /// </summary>
        /// <returns>The data of each step is returned.</returns>
        public List<Datum> GetData()
        {
            List<Datum> rgData = new List<Datum>();

            for (int i = 0; i < m_rgItems.Count; i++)
            {
                rgData.Add(new Datum(m_rgItems[i].Data));
            }

            return rgData;
        }

        /// <summary>
        /// Returns the clip data if it exists, or <i>null</i>.
        /// </summary>
        /// <returns>The clip data is returned, or <i>null</i>.</returns>
        public List<Datum> GetClip()
        {
            if (m_rgItems.Count == 0)
                return null;

            if (m_rgItems[0].State.Clip == null)
                return null;

            List<Datum> rgData = new List<Datum>();

            for (int i = 0; i < m_rgItems.Count; i++)
            {
                if (m_rgItems[i].State.Clip == null)
                    return null;

                rgData.Add(new Datum(m_rgItems[i].State.Clip));
            }

            return rgData;
        }
    }

    /// <summary>
    /// The MemoryItem stores the information for one step in an episode.
    /// </summary>
    class MemoryItem
    {
        StateBase m_state;
        SimpleDatum m_x;
        int m_nAction;
        float[] m_rgfAprob;
        float m_fReward;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="s">Contains the state and data of the step.</param>
        /// <param name="x">Specifies the pre-processed data of the step.</param>
        /// <param name="nAction">Specifies the action taken at the step.</param>
        /// <param name="rgfAprob">Specifies the action probability values (Aprob) of the step. NOTE: These values are only used in non-Softmax models.</param>
        /// <param name="fReward">Specifies the reward for taking the action.</param>
        public MemoryItem(StateBase s, SimpleDatum x, int nAction, float[] rgfAprob, float fReward)
        {
            m_state = s;
            m_x = x;
            m_nAction = nAction;
            m_rgfAprob = rgfAprob;
            m_fReward = fReward;
        }

        /// <summary>
        /// Returns the state and data of this episode step.
        /// </summary>
        public StateBase State
        {
            get { return m_state; }
        }

        /// <summary>
        /// Returns the pre-processed data (run through the model) of this episode step.
        /// </summary>
        public SimpleDatum Data
        {
            get { return m_x; }
        }

        /// <summary>
        /// Returns the action of this episode step.
        /// </summary>
        public int Action
        {
            get { return m_nAction; }
        }

        /// <summary>
        /// Returns the reward for taking the action in this episode step.
        /// </summary>
        public float Reward
        {
            get { return m_fReward; }
        }

        /// <summary>
        /// Returns the action probabilities which are only used with non-Softmax models.
        /// </summary>
        public float[] Aprob
        {
            get { return m_rgfAprob; }
        }

        /// <summary>
        /// Returns the string representation of this episode step.
        /// </summary>
        /// <returns>The string representation is returned.</returns>
        public override string ToString()
        {
            return "action = " + m_nAction.ToString() + " reward = " + m_fReward.ToString("N2") + " aprob = " + tostring(m_rgfAprob);
        }

        private string tostring(float[] rg)
        {
            string str = "{";

            for (int i = 0; i < rg.Length; i++)
            {
                str += rg[i].ToString("N5");
                str += ",";
            }

            str = str.TrimEnd(',');
            str += "}";

            return str;
        }
    }
}
