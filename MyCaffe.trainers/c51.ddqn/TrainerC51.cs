using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.data;
using MyCaffe.layers;
using MyCaffe.param;
using MyCaffe.solvers;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.trainers.c51.ddqn
{
    /// <summary>
    /// The TrainerC51 implements the C51-DDQN algorithm as described by Bellemare et al., and 'flyyufelix'
    /// </summary>
    /// <remarks>
    /// @see [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887), Marc G. Bellemare, Will Dabney, Remi Munos, 2017, arXiv:1707.06887
    /// @see [GitHub:flyyufelix/C51-DDQN-Keras](https://github.com/flyyufelix/C51-DDQN-Keras/blob/master/c51_ddqn.py) 2017, MIT License
    /// </remarks>
    /// <typeparam name="T"></typeparam>
    public class TrainerC51<T> : IxTrainerRL, IDisposable
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
        public TrainerC51(MyCaffeControl<T> mycaffe, PropertySet properties, CryptoRandom random, IxTrainerCallback icallback)
        {
            m_icallback = icallback;
            m_mycaffe = mycaffe;
            m_properties = properties;
            m_random = random;
        }

        /// <summary>
        /// Release all resources used.
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
        /// Run a single cycle on the environment after the delay.
        /// </summary>
        /// <param name="nDelay">Specifies a delay to wait before running.</param>
        /// <returns>The results of the run containing the action are returned.</returns>
        public ResultCollection RunOne(int nDelay = 1000)
        {
            m_mycaffe.CancelEvent.Reset();
            Agent<T> agent = new Agent<T>(m_icallback, m_mycaffe, m_properties, m_random, Phase.TRAIN);
            agent.Run(Phase.TEST, 1, ITERATOR_TYPE.ITERATION, TRAIN_STEP.NONE);
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
            m_mycaffe.CancelEvent.Reset();
            Agent<T> agent = new Agent<T>(m_icallback, m_mycaffe, m_properties, m_random, Phase.TRAIN);
            agent.Run(Phase.TRAIN, nN, type, step);
            agent.Dispose();

            return false;
        }
    }


    /// <summary>
    /// The Agent both builds episodes from the envrionment and trains on them using the Brain.
    /// </summary>
    /// <typeparam name="T">Specifies the base type, which should be the same base type used for MyCaffe.  This type is either <i>double</i> or <i>float</i>.</typeparam>
    class Agent<T> : IDisposable 
    {
        IxTrainerCallback m_icallback;
        Brain<T> m_brain;
        PropertySet m_properties;
        CryptoRandom m_random;
        float m_fGamma = 0.95f;
        bool m_bUseRawInput = true;
        int m_nMaxMemory = 50000;
        int m_nTrainingUpdateFreq = 5000;
        int m_nExplorationNum = 50000;
        int m_nEpsSteps = 0;
        double m_dfEpsStart = 0;
        double m_dfEpsEnd = 0;
        double m_dfExplorationRate = 0;
        STATE m_state = STATE.EXPLORING;

        enum STATE
        {
            EXPLORING,
            TRAINING
        }


        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="icallback">Specifies the callback used for update notifications sent to the parent.</param>
        /// <param name="mycaffe">Specifies the instance of MyCaffe with the open project.</param>
        /// <param name="properties">Specifies the properties passed into the trainer.</param>
        /// <param name="random">Specifies the random number generator used.</param>
        /// <param name="phase">Specifies the phase of the internal network to use.</param>
        public Agent(IxTrainerCallback icallback, MyCaffeControl<T> mycaffe, PropertySet properties, CryptoRandom random, Phase phase)
        {
            m_icallback = icallback;
            m_brain = new Brain<T>(mycaffe, properties, random, phase);
            m_properties = properties;
            m_random = random;

            m_fGamma = (float)properties.GetPropertyAsDouble("Gamma", m_fGamma);
            m_bUseRawInput = properties.GetPropertyAsBool("UseRawInput", m_bUseRawInput);
            m_nMaxMemory = properties.GetPropertyAsInt("MaxMemory", m_nMaxMemory);
            m_nTrainingUpdateFreq = properties.GetPropertyAsInt("TrainingUpdateFreq", m_nTrainingUpdateFreq);
            m_nExplorationNum = properties.GetPropertyAsInt("ExplorationNum", m_nExplorationNum);
            m_nEpsSteps = properties.GetPropertyAsInt("EpsSteps", m_nEpsSteps);
            m_dfEpsStart = properties.GetPropertyAsDouble("EpsStart", m_dfEpsStart);
            m_dfEpsEnd = properties.GetPropertyAsDouble("EpsEnd", m_dfEpsEnd);

            if (m_dfEpsStart < 0 || m_dfEpsStart > 1)
                throw new Exception("The 'EpsStart' is out of range - please specify a real number in the range [0,1]");

            if (m_dfEpsEnd < 0 || m_dfEpsEnd > 1)
                throw new Exception("The 'EpsEnd' is out of range - please specify a real number in the range [0,1]");

            if (m_dfEpsEnd > m_dfEpsStart)
                throw new Exception("The 'EpsEnd' must be less than the 'EpsStart' value.");
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

        private double getEpsilon(int nIteration)
        {
            if (m_state == STATE.EXPLORING)
                return 1;

            if (m_nEpsSteps == 0)
                return 0;

            if (nIteration >= m_nEpsSteps)
                return m_dfEpsEnd;

            return m_dfEpsStart + (double)(nIteration * (m_dfEpsEnd - m_dfEpsStart) / m_nEpsSteps);
        }

        private StateBase getData(Phase phase, int nAction, int nIdx)
        {
            GetDataArgs args = m_brain.getDataArgs(phase, nAction);
            m_icallback.OnGetData(args);
            args.State.Data.Index = nIdx;
            return args.State;
        }


        private int getAction(int nIteration, SimpleDatum sd, SimpleDatum sdClip, int nActionCount, TRAIN_STEP step)
        {
            if (step == TRAIN_STEP.NONE)
            {
                m_dfExplorationRate = getEpsilon(nIteration);

                if (m_dfExplorationRate > 0 && m_random.NextDouble() < m_dfExplorationRate)
                {
                    int nAction = m_random.Next(nActionCount);
                    return nAction;
                }
            }

            return m_brain.act(sd, sdClip, nActionCount);
        }

        private void updateStatus(int nIteration, int nEpisodeCount, double dfRewardSum, double dfRunningReward, double dfLoss, double dfLearningRate, bool bModelUpdated)
        {
            GetStatusArgs args = new GetStatusArgs(0, nIteration, nEpisodeCount, 1000000, dfRunningReward, dfRewardSum, m_dfExplorationRate, 0, dfLoss, dfLearningRate, bModelUpdated);
            m_icallback.OnUpdateStatus(args);
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

            StateBase s = getData(Phase.RUN, -1, 0);
            int nIteration = 0;
            List<float> rgResults = new List<float>();
            bool bDifferent;

            while (!m_brain.Cancel.WaitOne(0) && (nIterations == -1 || nIteration < nIterations))
            {
                // Preprocess the observation.
                SimpleDatum x = m_brain.Preprocess(s, m_bUseRawInput, out bDifferent);

                // Forward the policy network and sample an action.
                int action = m_brain.act(x, s.Clip, s.ActionCount);

                rgResults.Add(s.Data.TimeStamp.ToFileTime());
                rgResults.Add((float)s.Data.RealData[0]);
                rgResults.Add(action);

                nIteration++;

                // Take the next step using the action
                s = getData(Phase.RUN, action, nIteration);
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
        /// The Run method provides the main loop that performs the following steps:
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
            MemoryCollection rgMemory = new MemoryCollection(m_nMaxMemory);
            int nIteration = 0;
            double? dfRunningReward = null;
            double dfRewardSum = 0;
            int nEpisode = 0;
            bool bDifferent = false;

            StateBase s = getData(phase, -1, -1);
            // Preprocess the observation.
            SimpleDatum x = m_brain.Preprocess(s, m_bUseRawInput, out bDifferent, true);

            while (!m_brain.Cancel.WaitOne(0) && !isAtIteration(nN, type, nIteration, nEpisode))
            {
                if (nIteration > m_nExplorationNum && rgMemory.Count > m_brain.BatchSize)
                    m_state = STATE.TRAINING;

                // Forward the policy network and sample an action.
                int action = getAction(nIteration, x, s.Clip, s.ActionCount, step);

                // Take the next step using the action
                StateBase s_ = getData(phase, action, nIteration);

                // Preprocess the next observation.
                SimpleDatum x_ = m_brain.Preprocess(s_, m_bUseRawInput, out bDifferent);
                if (!bDifferent)
                    m_brain.Log.WriteLine("WARNING: The current state is the same as the previous state!");

                dfRewardSum += s_.Reward;

                // Build up episode memory, using reward for taking the action.
                rgMemory.Add(new MemoryItem(s, x, action, s_, x_, s_.Reward, s_.Done, nIteration, nEpisode));

                // Do the training
                if (m_state == STATE.TRAINING)
                {
                    MemoryCollection rgRandomSamples = rgMemory.GetRandomSamples(m_random, m_brain.BatchSize);
                    m_brain.Train(rgRandomSamples, s.ActionCount);

                    if (nIteration % m_nTrainingUpdateFreq == 0)
                        m_brain.UpdateTargetModel();
                }

                if (s_.Done)
                {
                    // Update reward running
                    if (!dfRunningReward.HasValue)
                        dfRunningReward = dfRewardSum;
                    else
                        dfRunningReward = dfRunningReward.Value * 0.99 + dfRewardSum * 0.01;

                    nEpisode++;
                    updateStatus(nIteration, nEpisode, dfRewardSum, dfRunningReward.Value, 0, 0, m_brain.GetModelUpdated());

                    s = getData(phase, -1, -1);
                    x = m_brain.Preprocess(s, m_bUseRawInput, out bDifferent, true);
                    dfRewardSum = 0;
                }
                else
                {
                    s = s_;
                    x = x_;
                }

                nIteration++;
            }
        }
    }

    /// <summary>
    /// The Brain uses the instance of MyCaffe (e.g. the open project) to run new actions and train the network.
    /// </summary>
    /// <typeparam name="T">Specifies the base type, which should be the same base type used for MyCaffe.  This type is either <i>double</i> or <i>float</i>.</typeparam>
    class Brain<T> : IDisposable, IxTrainerGetDataCallback
    {
        MyCaffeControl<T> m_mycaffe;
        Solver<T> m_solver;
        Net<T> m_net;
        Net<T> m_netTarget;
        PropertySet m_properties;
        CryptoRandom m_random;
        SimpleDatum m_sdLast = null;
        DataTransformer<T> m_transformer;
        MemoryLossLayer<T> m_memLoss;
        SoftmaxCrossEntropyLossLayer<T> m_softmaxLoss = null;
        SoftmaxLayer<T> m_softmax;
        Blob<T> m_blobZ = null;
        Blob<T> m_blobZ1 = null;
        Blob<T> m_blobQ = null;
        Blob<T> m_blobMLoss = null;
        Blob<T> m_blobPLoss = null;
        Blob<T> m_blobLoss = null;
        Blob<T> m_blobActionBinaryLoss = null;
        float m_fDeltaZ = 0;
        float[] m_rgfZ = null;
        float m_fGamma = 0.99f;
        int m_nAtoms = 51;
        double m_dfVMax = 10;       // Max possible score for Pong per action is 1
        double m_dfVMin = -10;      // Min possible score for Pong per action is -1
        int m_nFramesPerX = 4;
        int m_nStackPerX = 4;
        int m_nBatchSize = 32;
        MemoryCollection m_rgSamples;
        int m_nActionCount = 2;
        bool m_bModelUpdated = false;
        Font m_font = null;
        Dictionary<Color, Tuple<Brush, Brush, Pen, Brush>> m_rgStyle = new Dictionary<Color, Tuple<Brush, Brush, Pen, Brush>>();
        List<SimpleDatum> m_rgX = new List<SimpleDatum>();
        bool m_bNormalizeOverlay = true;


        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="mycaffe">Specifies the instance of MyCaffe assoiated with the open project - when using more than one Brain, this is the master project.</param>
        /// <param name="properties">Specifies the properties passed into the trainer.</param>
        /// <param name="random">Specifies the random number generator used.</param>
        /// <param name="phase">Specifies the phase under which to run.</param>
        public Brain(MyCaffeControl<T> mycaffe, PropertySet properties, CryptoRandom random, Phase phase)
        {
            m_mycaffe = mycaffe;
            m_solver = mycaffe.GetInternalSolver();
            m_net = mycaffe.GetInternalNet(phase);
            m_netTarget = new Net<T>(m_mycaffe.Cuda, m_mycaffe.Log, m_net.net_param, m_mycaffe.CancelEvent, null, phase);
            m_properties = properties;
            m_random = random;

            m_transformer = m_mycaffe.DataTransformer;
            m_transformer.param.mean_value.Add(255 / 2); // center
            m_transformer.param.mean_value.Add(255 / 2);
            m_transformer.param.mean_value.Add(255 / 2);
            m_transformer.param.scale = 1.0 / 255;       // normalize

            m_fGamma = (float)properties.GetPropertyAsDouble("Gamma", m_fGamma);
            m_nAtoms = properties.GetPropertyAsInt("Atoms", m_nAtoms);
            m_dfVMin = properties.GetPropertyAsDouble("VMin", m_dfVMin);
            m_dfVMax = properties.GetPropertyAsDouble("VMax", m_dfVMax);

            m_blobZ = new Blob<T>(m_mycaffe.Cuda, m_mycaffe.Log, false);
            m_blobZ1 = new Blob<T>(m_mycaffe.Cuda, m_mycaffe.Log, false);
            m_blobQ = new Blob<T>(m_mycaffe.Cuda, m_mycaffe.Log, true);
            m_blobMLoss = new Blob<T>(m_mycaffe.Cuda, m_mycaffe.Log, true);
            m_blobPLoss = new Blob<T>(m_mycaffe.Cuda, m_mycaffe.Log, true);
            m_blobLoss = new Blob<T>(m_mycaffe.Cuda, m_mycaffe.Log, true);
            m_blobActionBinaryLoss = new Blob<T>(m_mycaffe.Cuda, m_mycaffe.Log, false);

            m_memLoss = m_net.FindLastLayer(LayerParameter.LayerType.MEMORY_LOSS) as MemoryLossLayer<T>;
            if (m_memLoss == null)
                m_mycaffe.Log.FAIL("Missing the expected MEMORY_LOSS layer!");

            m_memLoss.OnGetLoss += m_memLoss_OnGetLoss;

            Blob<T> data = m_net.blob_by_name("data");
            if (data == null)
                m_mycaffe.Log.FAIL("Missing the expected input 'data' blob!");

            m_nFramesPerX = data.channels;
            m_nBatchSize = data.num;

            m_solver.parameter.delta = 0.01 / (double)m_nBatchSize;
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
            dispose(ref m_blobZ);
            dispose(ref m_blobZ1);
            dispose(ref m_blobQ);
            dispose(ref m_blobMLoss);
            dispose(ref m_blobPLoss);
            dispose(ref m_blobActionBinaryLoss);

            if (m_softmax != null)
            {
                m_softmax.Dispose();
                m_softmax = null;
            }

            if (m_netTarget != null)
            {
                m_netTarget.Dispose();
                m_netTarget = null;
            }

            if (m_font != null)
            {
                m_font.Dispose();
                m_font = null;
            }

            foreach (KeyValuePair<Color, Tuple<Brush, Brush, Pen, Brush>> kv in m_rgStyle)
            {
                kv.Value.Item1.Dispose();
                kv.Value.Item2.Dispose();
                kv.Value.Item3.Dispose();
                kv.Value.Item4.Dispose();
            }

            m_rgStyle.Clear();
        }

        /// <summary>
        /// Returns the GetDataArgs used to retrieve new data from the envrionment implemented by derived parent trainer.
        /// </summary>
        /// <param name="phase">Specifies the phase under which to get the data.</param>
        /// <param name="nAction">Specifies the action to run, or -1 to reset the environment.</param>
        /// <returns>A new GetDataArgs is returned.</returns>
        public GetDataArgs getDataArgs(Phase phase, int nAction)
        {
            bool bReset = (nAction == -1) ? true : false;
            return new GetDataArgs(phase, 0, m_mycaffe, m_mycaffe.Log, m_mycaffe.CancelEvent, bReset, nAction, true, false, false, this);
        }

        /// <summary>
        /// Specifies the number of frames per X value.
        /// </summary>
        public int FrameStack
        {
            get { return m_nFramesPerX; }
        }

        /// <summary>
        /// Returns the batch size defined by the model.
        /// </summary>
        public int BatchSize
        {
            get { return m_nBatchSize; }
        }

        /// <summary>
        /// Returns the output log.
        /// </summary>
        public Log Log
        {
            get { return m_mycaffe.Log; }
        }

        /// <summary>
        /// Returns the Cancel event used to cancel  all MyCaffe tasks.
        /// </summary>
        public CancelEvent Cancel
        {
            get { return m_mycaffe.CancelEvent; }
        }

        /// <summary>
        /// Preprocesses the data.
        /// </summary>
        /// <param name="s">Specifies the state and data to use.</param>
        /// <param name="bUseRawInput">Specifies whether or not to use the raw data <i>true</i>, or a difference of the current and previous data <i>false</i> (default = <i>false</i>).</param>
        /// <param name="bDifferent">Returns whether or not the current state data is different from the previous - note this is only set when NOT using raw input, otherwise <i>true</i> is always returned.</param>
        /// <param name="bReset">Optionally, specifies to reset the last sd to null.</param>
        /// <returns>The preprocessed data is returned.</returns>
        public SimpleDatum Preprocess(StateBase s, bool bUseRawInput, out bool bDifferent, bool bReset = false)
        {
            bDifferent = false;

            SimpleDatum sd = new SimpleDatum(s.Data, true);

            if (!bUseRawInput)
            {
                if (bReset)
                    m_sdLast = null;

                if (m_sdLast == null)
                    sd.Zero();
                else
                    bDifferent = sd.Sub(m_sdLast);

                m_sdLast = new SimpleDatum(s.Data, true);
            }
            else
            {
                bDifferent = true;
            }

            sd.Tag = bReset;

            if (bReset)
            {
                m_rgX = new List<SimpleDatum>();

                for (int i = 0; i < m_nFramesPerX * m_nStackPerX; i++)
                {
                    m_rgX.Add(sd);
                }
            }
            else
            {
                m_rgX.Add(sd);
                m_rgX.RemoveAt(0);
            }

            SimpleDatum[] rgSd = new SimpleDatum[m_nStackPerX];

            for (int i=0; i<m_nStackPerX; i++)
            {
                int nIdx = ((m_nStackPerX - i) * m_nFramesPerX) - 1;
                rgSd[i] = m_rgX[nIdx];
            }

            return new SimpleDatum(rgSd.ToList());
        }

        private float[] createZArray(double dfVMin, double dfVMax, int nAtoms, out float fDeltaZ)
        {
            float[] rgZ = new float[nAtoms];
            float fZ = (float)dfVMin;
            fDeltaZ = (float)((dfVMax - dfVMin) / (nAtoms - 1));

            for (int i = 0; i < nAtoms; i++)
            {
                rgZ[i] = fZ;
                fZ += fDeltaZ;
            }

            return rgZ;
        }

        private void createZ(int nNumSamples, int nActions, int nAtoms)
        {
            int nOffset = 0;

            if (m_rgfZ == null)
            {
                m_blobZ1.Reshape(1, nAtoms, 1, 1);

                m_rgfZ = createZArray(m_dfVMin, m_dfVMax, m_nAtoms, out m_fDeltaZ);
                T[] rgfZ0 = Utility.ConvertVec<T>(m_rgfZ);

                m_blobZ1.mutable_cpu_data = rgfZ0;
                nOffset = 0;

                m_blobZ.Reshape(nActions, m_nBatchSize, nAtoms, 1);

                for (int i = 0; i < nActions; i++)
                {
                    for (int j = 0; j < m_nBatchSize; j++)
                    {
                        m_mycaffe.Cuda.copy(m_blobZ1.count(), m_blobZ1.gpu_data, m_blobZ.mutable_gpu_data, 0, nOffset);
                        nOffset += m_blobZ1.count();
                    }
                }
            }

            m_blobZ.Reshape(nActions, nNumSamples, nAtoms, 1);
        }

        /// <summary>
        /// Returns the action from running the model.  The action returned is either randomly selected (when using Exploration),
        /// or calculated via a forward pass (when using Exploitation).
        /// </summary>
        /// <param name="sd">Specifies the data to run the model on.</param>
        /// <param name="sdClip">Specifies the clip data (if any exits).</param>
        /// <param name="nActionCount">Returns the number of actions in the action set.</param>
        /// <returns>The action value is returned.</returns>
        public int act(SimpleDatum sd, SimpleDatum sdClip, int nActionCount)
        {
            Blob<T> actions = m_net.blob_by_name("actions");
            if (actions == null)
                throw new Exception("Missing expected 'actions' blob!");

            setData(m_net, sd, sdClip);
            m_net.ForwardFromTo(0, m_net.layers.Count - 2);

            createZ(1, nActionCount, m_nAtoms);
            m_blobQ.ReshapeLike(actions);

            m_mycaffe.Cuda.mul(actions.count(), actions.gpu_data, m_blobZ.gpu_data, m_blobQ.mutable_gpu_data);
            reduce_sum_axis2(m_blobQ);

            return argmax(Utility.ConvertVecF<T>(m_blobQ.mutable_cpu_diff), nActionCount, 0);
        }

        /// <summary>
        /// Get whether or not the model has been udpated or not.
        /// </summary>
        /// <returns>If the model has been updated from the last call to this function, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool GetModelUpdated()
        {
            bool bModelUpdated = m_bModelUpdated;
            m_bModelUpdated = false;
            return bModelUpdated;
        }

        /// <summary>
        /// The UpdateTargetModel transfers the trained layers from the active Net to the target Net.
        /// </summary>
        public void UpdateTargetModel()
        {
            m_mycaffe.Log.Enable = false;
            m_net.CopyTrainedLayersTo(m_netTarget);
            m_mycaffe.Log.Enable = true;
            m_bModelUpdated = true;
        }

        /// <summary>
        /// Train the model at the current iteration.
        /// </summary>
        /// <param name="rgSamples">Contains the samples to train the model with.</param>
        /// <param name="nActionCount">Specifies the number of actions in the action set.</param>
        /// <returns>The smoothed loss is returned.</returns>
        public double Train(MemoryCollection rgSamples, int nActionCount)
        {
            double dfLoss;

            m_rgSamples = rgSamples;
            m_nActionCount = nActionCount;

            setData1(m_netTarget, rgSamples);
            m_netTarget.ForwardFromTo(0, m_netTarget.layers.Count - 2);

            setData0(m_net, rgSamples);
            dfLoss = m_net.ForwardFromTo();

            m_solver.Step(1, TRAIN_STEP.BACKWARD, true, true, true, true, dfLoss);

            return dfLoss;
        }

        private void m_memLoss_OnGetLoss(object sender, MemoryLossLayerGetLossArgs<T> e)
        {
            Blob<T> actions = m_net.blob_by_name("actions");
            if (actions == null)
                throw new Exception("Missing expected 'actions' blob!");

            Blob<T> p_actions = m_netTarget.blob_by_name("actions");
            if (p_actions == null)
                throw new Exception("Missing expected 'actions' blob!");

            int nNumSamples = m_rgSamples.Count;

            createZ(nNumSamples, m_nActionCount, m_nAtoms);

            m_blobQ.ReshapeLike(actions);

            m_mycaffe.Log.CHECK_EQ(m_blobQ.shape(0), nNumSamples, "The result should have shape(0) = NumSamples which is " + nNumSamples.ToString());
            m_mycaffe.Log.CHECK_EQ(m_blobQ.shape(1), m_nActionCount, "The result should have shape(1) = Actions which is " + m_nActionCount.ToString());
            m_mycaffe.Log.CHECK_EQ(m_blobQ.shape(2), m_nAtoms, "The result should have shape(2) = Atoms which is " + m_nAtoms.ToString());

            // Get Optimal Actions for the next states (for distribution z)
            m_mycaffe.Cuda.mul(actions.count(), actions.gpu_data, m_blobZ.gpu_data, m_blobQ.mutable_gpu_data);
            m_mycaffe.Cuda.sum(m_blobQ.count(), m_blobQ.count(0, 2), m_blobQ.count(2), m_blobQ.gpu_data, m_blobQ.mutable_gpu_diff);
            m_blobQ.Reshape(nNumSamples, m_nActionCount, 1, 1);

            float[] rgQbatch = Utility.ConvertVecF<T>(m_blobQ.mutable_cpu_diff);
            float[] rgPbatch = Utility.ConvertVecF<T>(p_actions.mutable_cpu_data);
            float[] rgMBatch = new float[nNumSamples * m_nAtoms];

            for (int i = 0; i < nNumSamples; i++)
            {
                int nActionMax = argmax(rgQbatch, m_nActionCount, i);

                if (m_rgSamples[i].IsTerminated)
                {
                    double dfTz = m_rgSamples[i].Reward;

                    // Bounding Tz
                    dfTz = setBounds(dfTz, m_dfVMin, m_dfVMax);

                    double dfB = (dfTz - m_dfVMin) / m_fDeltaZ;
                    int nL = (int)Math.Floor(dfB);
                    int nU = (int)Math.Ceiling(dfB);
                    int nIdx = i * m_nAtoms;

                    rgMBatch[nIdx + nL] += (float)(nU - dfB);
                    rgMBatch[nIdx + nU] += (float)(dfB - nL);
                }
                else
                {
                    for (int j = 0; j < m_nAtoms; j++)
                    {
                        double dfTz = m_rgSamples[i].Reward + m_fGamma * m_rgfZ[j];

                        // Bounding Tz
                        dfTz = setBounds(dfTz, m_dfVMin, m_dfVMax);

                        double dfB = (dfTz - m_dfVMin) / m_fDeltaZ;
                        int nL = (int)Math.Floor(dfB);
                        int nU = (int)Math.Ceiling(dfB);
                        int nIdx = i * m_nAtoms;
                        int nIdxT = (i * m_nActionCount * m_nAtoms) + (nActionMax * m_nAtoms);

                        rgMBatch[nIdx + nL] += rgPbatch[nIdxT + j] * (float)(nU - dfB);
                        rgMBatch[nIdx + nU] += rgPbatch[nIdxT + j] * (float)(dfB - nL);
                    }
                }

                // Normalize the atom values to range [0,1]
                float fSum = 0;
                for (int j = 0; j < m_nAtoms; j++)
                {
                    fSum += rgMBatch[(i * m_nAtoms) + j];
                }

                if (fSum == 0)
                    throw new Exception("The sum of atom values cannot be zero!");

                for (int j = 0; j < m_nAtoms; j++)
                {
                    int nIdx = i * m_nAtoms;
                    rgMBatch[nIdx + j] /= fSum;
                }
            }

            m_blobMLoss.Reshape(nNumSamples, m_nAtoms, 1, 1);
            m_blobMLoss.mutable_cpu_data = Utility.ConvertVec<T>(rgMBatch);

            m_blobActionBinaryLoss.Reshape(nNumSamples, m_nActionCount, m_nAtoms, 1);
            m_blobActionBinaryLoss.SetData(0.0);

            for (int i = 0; i < m_rgSamples.Count; i++)
            {
                int nAction = m_rgSamples[i].Action;
                int nIdx = (i * m_nActionCount * m_nAtoms) + (nAction * m_nAtoms);

                m_blobActionBinaryLoss.SetData(1.0, nIdx, m_nAtoms);
            }


            //-------------------------------------------------------
            //  Loss function
            //-------------------------------------------------------

            Blob<T> logits = m_net.blob_by_name("logits");
            if (logits == null)
                throw new Exception("Could not find the expected 'logits' blob!");

            m_blobPLoss.ReshapeLike(logits);

            m_mycaffe.Cuda.mul(logits.count(), logits.gpu_data, m_blobActionBinaryLoss.mutable_gpu_data, m_blobPLoss.mutable_gpu_data); // Logits valid
            m_blobPLoss.Reshape(m_blobPLoss.num, m_nActionCount, m_nAtoms, 1);
            reduce_sum_axis1(m_blobPLoss);

            e.Loss = softmaxLoss_forward(m_blobPLoss, m_blobMLoss, m_blobLoss);

            e.EnableLossUpdate = false;
            m_blobLoss.SetDiff(1);

            softmaxLoss_backward(m_blobPLoss, m_blobMLoss, m_blobLoss);

            int nDstOffset = 0;
            int nSrcOffset = 0;

            for (int i = 0; i < nNumSamples; i++)
            {
                for (int j = 0; j < m_nActionCount; j++)
                {
                    m_mycaffe.Cuda.mul(m_nAtoms, m_blobPLoss.gpu_diff, m_blobActionBinaryLoss.gpu_data, e.Bottom[0].mutable_gpu_diff, nSrcOffset, nDstOffset, nDstOffset);
                    nDstOffset += m_nAtoms;
                }

                nSrcOffset += m_nAtoms;
            }
        }

        private void reduce_sum_axis1(Blob<T> b)
        {
            int nNum = b.shape(0);
            int nActions = b.shape(1);
            int nAtoms = b.shape(2);
            float[] rg = Utility.ConvertVecF<T>(b.mutable_cpu_data);
            float[] rgSum = new float[nNum * nAtoms];

            for (int i = 0; i < nNum; i++)
            {
                for (int j = 0; j < nAtoms; j++)
                {
                    float fSum = 0;

                    for (int k = 0; k < nActions; k++)
                    {
                        int nIdx = (i * nActions * nAtoms) + (k * nAtoms);
                        fSum += rg[nIdx + j];
                    }

                    int nIdxR = i * nAtoms;
                    rgSum[nIdxR + j] = fSum;
                }
            }

            b.Reshape(nNum, nAtoms, 1, 1);
            b.mutable_cpu_data = Utility.ConvertVec<T>(rgSum);
        }

        private void reduce_sum_axis2(Blob<T> b)
        {
            int nNum = b.shape(0);
            int nActions = b.shape(1);
            int nAtoms = b.shape(2);
            float[] rg = Utility.ConvertVecF<T>(b.mutable_cpu_data);
            float[] rgSum = new float[nNum * nActions];

            for (int i = 0; i < nNum; i++)
            {
                for (int j = 0; j < nActions; j++)
                {
                    int nIdx = (i * nActions * nAtoms) + (j * nAtoms);
                    float fSum = 0;

                    for (int k = 0; k < nAtoms; k++)
                    {
                        fSum += rg[nIdx + k];
                    }

                    int nIdxR = i * nActions;
                    rgSum[nIdxR + j] = fSum;
                }
            }

            b.Reshape(nNum, nAtoms, 1, 1);
            b.mutable_cpu_data = Utility.ConvertVec<T>(rgSum);
        }

        private double softmaxLoss_forward(Blob<T> actual, Blob<T> target, Blob<T> loss)
        {
            BlobCollection<T> colBottom = new BlobCollection<T>();
            colBottom.Add(actual);
            colBottom.Add(target);

            BlobCollection<T> colTop = new BlobCollection<T>();
            colTop.Add(loss);

            if (m_softmaxLoss == null)
            {
                LayerParameter p = new LayerParameter(LayerParameter.LayerType.SOFTMAXCROSSENTROPY_LOSS);
                p.softmax_param.axis = 1;
                m_softmaxLoss = new SoftmaxCrossEntropyLossLayer<T>(m_mycaffe.Cuda, m_mycaffe.Log, p);
                m_softmaxLoss.Setup(colBottom, colTop);
            }

            return m_softmaxLoss.Forward(colBottom, colTop);
        }

        private void softmaxLoss_backward(Blob<T> actual, Blob<T> target, Blob<T> loss)
        {
            BlobCollection<T> colBottom = new BlobCollection<T>();
            colBottom.Add(actual);
            colBottom.Add(target);

            BlobCollection<T> colTop = new BlobCollection<T>();
            colTop.Add(loss);

            m_softmaxLoss.Backward(colTop, new List<bool>() { true, false }, colBottom);
        }

        private void softmax_forward(Blob<T> b)
        {
            BlobCollection<T> colBottom = new BlobCollection<T>();
            colBottom.Add(b);

            BlobCollection<T> colTop = new BlobCollection<T>();
            colTop.Add(b);

            if (m_softmax == null)
            {
                LayerParameter p = new LayerParameter(LayerParameter.LayerType.SOFTMAX);
                p.softmax_param.axis = 1;
                m_softmax = new SoftmaxLayer<T>(m_mycaffe.Cuda, m_mycaffe.Log, p);
                m_softmax.Setup(colBottom, colTop);
            }

            m_softmax.Reshape(colBottom, colTop);
            m_softmax.Forward(colBottom, colTop);
        }


        private double setBounds(double z, double dfMin, double dfMax)
        {
            if (z > dfMax)
                return dfMax;

            if (z < dfMin)
                return dfMin;

            return z;
        }

        private int argmax(float[] rgProb, int nActionCount, int nSampleIdx)
        {
            float[] rgfProb = new float[nActionCount];

            for (int j = 0; j < nActionCount; j++)
            {
                int nIdx = (nSampleIdx * nActionCount) + j;
                rgfProb[j] = rgProb[nIdx];
            }

            return argmax(rgfProb);
        }

        private int argmax(float[] rgfAprob)
        {
            float fMax = -float.MaxValue;
            int nIdx = 0;

            for (int i = 0; i < rgfAprob.Length; i++)
            {
                if (rgfAprob[i] == fMax)
                {
                    if (m_random.NextDouble() > 0.5)
                        nIdx = i;
                }
                else if (fMax < rgfAprob[i])
                {
                    fMax = rgfAprob[i];
                    nIdx = i;
                }
            }

            return nIdx;
        }

        private void setData(Net<T> net, SimpleDatum sdData, SimpleDatum sdClip)
        {
            SimpleDatum[] rgData = new SimpleDatum[] { sdData };
            SimpleDatum[] rgClip = null;

            if (sdClip != null)
                rgClip = new SimpleDatum[] { sdClip };

            setData(net, rgData, rgClip);
        }

        private void setData0(Net<T> net, MemoryCollection rgSamples)
        {
            List<SimpleDatum> rgData0 = rgSamples.GetData0();
            List<SimpleDatum> rgClip0 = rgSamples.GetClip0();

            SimpleDatum[] rgData = rgData0.ToArray();
            SimpleDatum[] rgClip = (rgClip0 != null) ? rgClip0.ToArray() : null;

            setData(net, rgData, rgClip);
        }

        private void setData1(Net<T> net, MemoryCollection rgSamples)
        {
            List<SimpleDatum> rgData1 = rgSamples.GetData1();
            List<SimpleDatum> rgClip1 = rgSamples.GetClip1();

            SimpleDatum[] rgData = rgData1.ToArray();
            SimpleDatum[] rgClip = (rgClip1 != null) ? rgClip1.ToArray() : null;

            setData(net, rgData, rgClip);
        }

        private void setData(Net<T> net, SimpleDatum[] rgData, SimpleDatum[] rgClip)
        {
            Blob<T> data = net.blob_by_name("data");

            data.Reshape(rgData.Length, data.channels, data.height, data.width);
            m_transformer.Transform(rgData, data, m_mycaffe.Cuda, m_mycaffe.Log);

            if (rgClip != null)
            {
                Blob<T> clip = net.blob_by_name("clip");

                if (clip != null)
                {
                    clip.Reshape(rgClip.Length, rgClip[0].Channels, rgClip[0].Height, rgClip[0].Width);
                    m_transformer.Transform(rgClip, clip, m_mycaffe.Cuda, m_mycaffe.Log, true);
                }
            }
        }

        /// <summary>
        /// The OnOverlay callback is called just before displaying the gym image, thus allowing for an overlay to be applied to the image.
        /// </summary>
        /// <param name="e">Specifies the arguments to the callback which contains the original display image.</param>
        public void OnOverlay(OverlayArgs e)
        {
            Blob<T> actions = m_net.blob_by_name("actions");
            if (actions == null)
                return;

            float[] rgActions = Utility.ConvertVecF<T>(actions.mutable_cpu_data);

            List<List<float>> rgData = new List<List<float>>();
            for (int i = 0; i < m_nActionCount; i++)
            {
                List<float> rgProb = new List<float>();

                for (int j = 0; j < m_nAtoms; j++)
                {
                    int nIdx = (i * m_nAtoms) + j;
                    rgProb.Add(rgActions[nIdx]);
                }

                rgData.Add(rgProb);
            }

            using (Graphics g = Graphics.FromImage(e.DisplayImage))
            {
                int nBorder = 30;
                int nWid = e.DisplayImage.Width - (nBorder * 2);
                int nWid1 = nWid / rgData.Count;
                int nHt1 = (int)(e.DisplayImage.Height * 0.3);
                int nX = nBorder;
                int nY = e.DisplayImage.Height - nHt1;
                ColorMapper clrMap = new ColorMapper(0, rgData.Count + 1, Color.Black, Color.Red);            
                float[] rgfMin = new float[rgData.Count];
                float[] rgfMax = new float[rgData.Count];
                float fMax = -float.MaxValue;
                float fMaxMax = -float.MaxValue;
                int nMaxIdx = 0;

                for (int i=0; i<rgData.Count; i++)
                {
                    rgfMin[i] = rgData[i].Min(p => p);
                    rgfMax[i] = rgData[i].Max(p => p);

                    if (rgfMax[i] > fMax)
                    {
                        fMax = rgfMax[i];
                        nMaxIdx = i;
                    }

                    fMaxMax = Math.Max(fMax, fMaxMax);
                }

                if (fMaxMax > 0.2f)
                    m_bNormalizeOverlay = false;

                for (int i = 0; i < rgData.Count; i++)
                {
                    drawProbabilities(g, nX, nY, nWid1, nHt1, i, rgData[i], clrMap.GetColor(i + 1), rgfMin[i], rgfMax[i], (i == nMaxIdx) ? true : false, m_bNormalizeOverlay);
                    nX += nWid1;
                }
            }
        }

        private void drawProbabilities(Graphics g, int nX, int nY, int nWid, int nHt, int nAction, List<float> rgProb, Color clr, float fMin, float fMax, bool bMax, bool bNormalize)
        {
            string str = "";

            if (m_font == null)
                m_font = new Font("Century Gothic", 9.0f);

            if (!m_rgStyle.ContainsKey(clr))
            {
                Color clr1 = Color.FromArgb(128, clr);
                Brush br1 = new SolidBrush(clr1);
                Color clr2 = Color.FromArgb(64, clr);
                Pen pen = new Pen(clr2, 1.0f);
                Brush br2 = new SolidBrush(clr2);
                Brush brBright = new SolidBrush(clr);
                m_rgStyle.Add(clr, new Tuple<Brush, Brush, Pen, Brush>(br1, br2, pen, brBright));
            }

            Brush brBack = m_rgStyle[clr].Item1;
            Brush brFront = m_rgStyle[clr].Item2;
            Brush brTop = m_rgStyle[clr].Item4;
            Pen penLine = m_rgStyle[clr].Item3;

            if (fMin != 0 || fMax != 0)
            {
                str = "Action " + nAction.ToString() + " (" + (fMax - fMin).ToString("N7") + ")";
            }
            else
            {
                str = "Action " + nAction.ToString() + " - No Probabilities";
            }

            SizeF sz = g.MeasureString(str, m_font);

            int nY1 = (int)(nY + (nHt - sz.Height));
            int nX1 = (int)(nX + (nWid / 2) - (sz.Width / 2));
            g.DrawString(str, m_font, (bMax) ? brTop : brFront, new Point(nX1, nY1));

            if (fMin != 0 || fMax != 0)
            {
                float fX = nX;
                float fWid = nWid / (float)rgProb.Count;
                nHt -= (int)sz.Height;                

                for (int i = 0; i < rgProb.Count; i++)
                {
                    float fProb = rgProb[i];

                    if (bNormalize)
                        fProb = (fProb - fMin) / (fMax - fMin);

                    float fHt = nHt * fProb;
                    float fHt1 = nHt - fHt;
                    RectangleF rc1 = new RectangleF(fX, nY + fHt1, fWid, fHt);
                    g.FillRectangle(brBack, rc1);
                    g.DrawRectangle(penLine, rc1.X, rc1.Y, rc1.Width, rc1.Height);
                    fX += fWid;
                }
            }
        }
    }

    class MemoryCollection : IEnumerable<MemoryItem> /** @private */
    {
        int m_nMax;
        List<MemoryItem> m_rgItems = new List<MemoryItem>();

        public MemoryCollection(int nMax)
        {
            m_nMax = nMax;
        }

        public int Count
        {
            get { return m_rgItems.Count; }
        }

        public MemoryItem this[int nIdx]
        {
            get { return m_rgItems[nIdx]; }
        }

        public void Add(MemoryItem item)
        {
            m_rgItems.Add(item);

            if (m_rgItems.Count > m_nMax)
                m_rgItems.RemoveAt(0);
        }

        public void Clear()
        {
            m_rgItems.Clear();
        }

        public MemoryCollection GetRandomSamples(CryptoRandom random, int nCount)
        {
            MemoryCollection col = new MemoryCollection(m_nMax);
            List<int> rgIdx = new List<int>();

            while (col.Count < nCount)
            {
                int nIdx = random.Next(m_rgItems.Count);
                if (!rgIdx.Contains(nIdx))
                {
                    col.Add(m_rgItems[nIdx]);
                    rgIdx.Add(nIdx);
                }
            }

            return col;
        }

        public List<StateBase> GetState1()
        {
            return m_rgItems.Select(p => p.State1).ToList();
        }

        public List<SimpleDatum> GetData1()
        {
            return m_rgItems.Select(p => p.Data1).ToList();
        }

        public List<SimpleDatum> GetClip1()
        {
            if (m_rgItems[0].State1.Clip != null)
                return m_rgItems.Select(p => p.State1.Clip).ToList();

            return null;
        }

        public List<SimpleDatum> GetData0()
        {
            return m_rgItems.Select(p => p.Data0).ToList();
        }

        public List<SimpleDatum> GetClip0()
        {
            if (m_rgItems[0].State0.Clip != null)
                return m_rgItems.Select(p => p.State0.Clip).ToList();

            return null;
        }

        public IEnumerator<MemoryItem> GetEnumerator()
        {
            return m_rgItems.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return m_rgItems.GetEnumerator();
        }
    }

    class MemoryItem /** @private */
    {
        StateBase m_state0;
        StateBase m_state1;
        SimpleDatum m_x0;
        SimpleDatum m_x1;
        int m_nAction;
        int m_nIteration;
        int m_nEpisode;
        bool m_bTerminated;
        double m_dfReward;

        public MemoryItem(StateBase s, SimpleDatum x, int nAction, StateBase s_, SimpleDatum x_, double dfReward, bool bTerminated, int nIteration, int nEpisode)
        {
            m_state0 = s;
            m_state1 = s_;
            m_x0 = x;
            m_x1 = x_;
            m_nAction = nAction;
            m_bTerminated = bTerminated;
            m_dfReward = dfReward;
            m_nIteration = nIteration;
            m_nEpisode = nEpisode;
        }

        public bool IsTerminated
        {
            get { return m_bTerminated; }
        }

        public double Reward
        {
            get { return m_dfReward; }
            set { m_dfReward = value; }
        }

        public StateBase State0
        {
            get { return m_state0; }
        }

        public StateBase State1
        {
            get { return m_state1; }
        }

        public SimpleDatum Data0
        {
            get { return m_x0; }
        }

        public SimpleDatum Data1
        {
            get { return m_x1; }
        }

        public int Action
        {
            get { return m_nAction; }
        }

        public int Iteration
        {
            get { return m_nIteration; }
        }

        public int Episode
        {
            get { return m_nEpisode; }
        }

        public override string ToString()
        {
            return "episode = " + m_nEpisode.ToString() + " action = " + m_nAction.ToString() + " reward = " + m_dfReward.ToString("N2");
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
