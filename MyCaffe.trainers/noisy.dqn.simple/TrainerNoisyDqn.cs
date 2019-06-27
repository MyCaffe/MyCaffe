using MyCaffe;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.data;
using MyCaffe.layers;
using MyCaffe.param;
using MyCaffe.solvers;
using MyCaffe.trainers;
using MyCaffe.trainers.common;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.trainers.noisy.dqn.simple
{
    /// <summary>
    /// The TrainerNoisyDqn implements the Noisy-DQN algorithm as described by Gheshlagi et al. and inspired by 'higgsfield'
    /// </summary>
    /// <remarks>
    /// @see [Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295), Meire Fortunato, Mohammad Gheshlaghi Azar, Bilal Piot, Jacob Menick, Ian Osband, Alex Graves, Vlad Mnih, Remi Munos, Demis Hassabis, Olivier Pietquin, Charles Blundell, Shane Legg, arXiv:1706.10295
    /// @see [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952), Tom Schaul, John Quan, Ioannis Antonoglou, David Silver, 2016
    /// @see [Github:higgsfield/RL-Adventure](https://github.com/higgsfield/RL-Adventure), higgsfield, 2018
    /// @see [Github:openai/baselines](https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py), OpenAI, 2018
    /// </remarks>
    /// <typeparam name="T"></typeparam>
    public class TrainerNoisyDqn<T> : IxTrainerRL, IDisposable
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
        public TrainerNoisyDqn(MyCaffeControl<T> mycaffe, PropertySet properties, CryptoRandom random, IxTrainerCallback icallback)
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
        float m_fGamma = 0.99f;
        bool m_bUseRawInput = true;
        double m_dfBetaStart = 0.4;
        int m_nBetaFrames = 1000;
        int m_nMemorySize = 10000;
        float m_fPriorityAlpha = 0.6f;
        int m_nUpdateTargetFreq = 1000;
        MEMTYPE m_memType = MEMTYPE.PRIORITY;


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

        private double beta_by_frame(int nFrameIdx)
        {
            return Math.Min(1.0, m_dfBetaStart + nFrameIdx * (1.0 - m_dfBetaStart) / m_nBetaFrames);
        }

        private StateBase getData(Phase phase, int nAction, int nIdx)
        {
            GetDataArgs args = m_brain.getDataArgs(phase, nAction);
            m_icallback.OnGetData(args);
            args.State.Data.Index = nIdx;
            return args.State;
        }

        private void updateStatus(int nIteration, int nEpisodeCount, double dfRewardSum, double dfRunningReward, double dfLoss, double dfLearningRate, bool bModelUpdated)
        {
            GetStatusArgs args = new GetStatusArgs(0, nIteration, nEpisodeCount, 1000000, dfRunningReward, dfRewardSum, 0, 0, dfLoss, dfLearningRate, bModelUpdated);
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
            IMemoryCollection iMemory = MemoryCollectionFactory.CreateMemory(m_memType, m_nMemorySize, m_fPriorityAlpha);
            int nIteration = 0;
            double dfRunningReward = 0;
            double dfEpisodeReward = 0;
            int nEpisode = 0;
            bool bDifferent = false;

            StateBase state = getData(phase, -1, -1);
            // Preprocess the observation.
            SimpleDatum x = m_brain.Preprocess(state, m_bUseRawInput, out bDifferent, true);

            // Set the initial target model to the current model.
            m_brain.UpdateTargetModel();

            while (!m_brain.Cancel.WaitOne(0) && !isAtIteration(nN, type, nIteration, nEpisode))
            {
                // Forward the policy network and sample an action.
                int action = m_brain.act(x, state.Clip, state.ActionCount);

                // Take the next step using the action
                StateBase state_next = getData(phase, action, nIteration);

                // Preprocess the next observation.
                SimpleDatum x_next = m_brain.Preprocess(state_next, m_bUseRawInput, out bDifferent);
                if (!bDifferent)
                    m_brain.Log.WriteLine("WARNING: The current state is the same as the previous state!");

                // Build up episode memory, using reward for taking the action.
                iMemory.Add(new MemoryItem(state, x, action, state_next, x_next, state_next.Reward, state_next.Done, nIteration, nEpisode));
                dfEpisodeReward += state_next.Reward;

                // Do the training
                if (iMemory.Count > m_brain.BatchSize)
                {
                    double dfBeta = beta_by_frame(nIteration);
                    MemoryCollection rgSamples = iMemory.GetSamples(m_random, m_brain.BatchSize, dfBeta);
                    m_brain.Train(nIteration, rgSamples, state.ActionCount);
                    iMemory.Update(rgSamples);

                    if (nIteration % m_nUpdateTargetFreq == 0)
                        m_brain.UpdateTargetModel();
                }
           
                if (state_next.Done)
                {
                    // Update reward running
                    dfRunningReward = dfRunningReward * 0.99 + dfEpisodeReward * 0.01;

                    nEpisode++;
                    updateStatus(nIteration, nEpisode, dfEpisodeReward, dfRunningReward, 0, 0, m_brain.GetModelUpdated());

                    state = getData(phase, -1, -1);
                    x = m_brain.Preprocess(state, m_bUseRawInput, out bDifferent, true);
                    dfEpisodeReward = 0;
                }
                else
                {
                    state = state_next;
                    x = x_next;
                }
               
                nIteration++;
            }

            iMemory.CleanUp();
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
        Blob<T> m_blobActions = null;
        Blob<T> m_blobQValue = null;
        Blob<T> m_blobNextQValue = null;
        Blob<T> m_blobExpectedQValue = null;
        Blob<T> m_blobDone = null;
        Blob<T> m_blobLoss = null;
        Blob<T> m_blobWeights = null;
        BlobCollection<T> m_colAccumulatedGradients = new BlobCollection<T>();
        bool m_bUseAcceleratedTraining = false;
        double m_dfLearningRate;
        int m_nMiniBatch = 1;  
        float m_fGamma = 0.99f;
        int m_nBatchSize = 32;
        MemoryCollection m_rgSamples;
        int m_nActionCount = 2;
        bool m_bModelUpdated = false;
        Font m_font = null;
        Dictionary<Color, Tuple<Brush, Brush, Pen, Brush>> m_rgStyle = new Dictionary<Color, Tuple<Brush, Brush, Pen, Brush>>();
        List<SimpleDatum> m_rgX = new List<SimpleDatum>();
        float[] m_rgOverlay = null;


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

            Blob<T> data = m_net.blob_by_name("data");
            if (data == null)
                m_mycaffe.Log.FAIL("Missing the expected input 'data' blob!");

            m_nBatchSize = data.num;

            Blob<T> logits = m_net.blob_by_name("logits");
            if (logits == null)
                m_mycaffe.Log.FAIL("Missing the expected input 'logits' blob!");

            m_nActionCount = logits.channels;

            m_transformer = m_mycaffe.DataTransformer;
            m_blobActions = new Blob<T>(m_mycaffe.Cuda, m_mycaffe.Log, false);
            m_blobQValue = new Blob<T>(m_mycaffe.Cuda, m_mycaffe.Log);
            m_blobNextQValue = new Blob<T>(m_mycaffe.Cuda, m_mycaffe.Log);
            m_blobExpectedQValue = new Blob<T>(m_mycaffe.Cuda, m_mycaffe.Log);
            m_blobDone = new Blob<T>(m_mycaffe.Cuda, m_mycaffe.Log, false);
            m_blobLoss = new Blob<T>(m_mycaffe.Cuda, m_mycaffe.Log);
            m_blobWeights = new Blob<T>(m_mycaffe.Cuda, m_mycaffe.Log, false);

            m_fGamma = (float)properties.GetPropertyAsDouble("Gamma", m_fGamma);

            m_memLoss = m_net.FindLastLayer(LayerParameter.LayerType.MEMORY_LOSS) as MemoryLossLayer<T>;
            if (m_memLoss == null)
                m_mycaffe.Log.FAIL("Missing the expected MEMORY_LOSS layer!");

            double? dfRate = mycaffe.CurrentProject.GetSolverSettingAsNumeric("base_lr");
            if (dfRate.HasValue)
                m_dfLearningRate = dfRate.Value;

            m_nMiniBatch = m_properties.GetPropertyAsInt("MiniBatch", m_nMiniBatch);
            m_bUseAcceleratedTraining = properties.GetPropertyAsBool("UseAcceleratedTraining", false);

            if (m_nMiniBatch > 1)
            {
                m_colAccumulatedGradients = m_net.learnable_parameters.Clone();
                m_colAccumulatedGradients.SetDiff(0);
            }
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
            dispose(ref m_blobActions);
            dispose(ref m_blobQValue);
            dispose(ref m_blobNextQValue);
            dispose(ref m_blobExpectedQValue);
            dispose(ref m_blobDone);
            dispose(ref m_blobLoss);
            dispose(ref m_blobWeights);

            if (m_colAccumulatedGradients != null)
            {
                m_colAccumulatedGradients.Dispose();
                m_colAccumulatedGradients = null;
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

            return sd;
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
            setData(m_net, sd, sdClip);
            m_net.ForwardFromTo(0, m_net.layers.Count - 2);

            Blob<T> output = m_net.blob_by_name("logits");
            if (output == null)
                throw new Exception("Missing expected 'logits' blob!");

            // Choose greedy action
            return argmax(Utility.ConvertVecF<T>(output.mutable_cpu_data));
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
        /// <param name="nIteration">Specifies the current iteration.</param>
        /// <param name="rgSamples">Contains the samples to train the model with along with the priorities associated with the samples.</param>
        /// <param name="nActionCount">Specifies the number of actions in the action set.</param>
        public void Train(int nIteration, MemoryCollection rgSamples, int nActionCount)
        {
            m_rgSamples = rgSamples;

            if (m_nActionCount != nActionCount)
                throw new Exception("The logit output of '" + m_nActionCount.ToString() + "' does not match the action count of '" + nActionCount.ToString() + "'!");

            // Get next_q_values
            m_mycaffe.Log.Enable = false;
            setNextStateData(m_netTarget, rgSamples);
            m_netTarget.ForwardFromTo(0, m_netTarget.layers.Count - 2);

            setCurrentStateData(m_net, rgSamples);
            m_memLoss.OnGetLoss += m_memLoss_ComputeTdLoss;

            if (m_nMiniBatch == 1)
            {
                m_solver.Step(1);
            }
            else
            {
                m_solver.Step(1, TRAIN_STEP.NONE, true, m_bUseAcceleratedTraining, true, true);
                m_colAccumulatedGradients.Accumulate(m_mycaffe.Cuda, m_net.learnable_parameters, true);

                if (nIteration % m_nMiniBatch == 0)
                {
                    m_net.learnable_parameters.CopyFrom(m_colAccumulatedGradients, true);
                    m_colAccumulatedGradients.SetDiff(0);
                    m_dfLearningRate = m_solver.ApplyUpdate(nIteration);
                    m_net.ClearParamDiffs();
                }
            }

            m_memLoss.OnGetLoss -= m_memLoss_ComputeTdLoss;
            m_mycaffe.Log.Enable = true;

            resetNoise(m_net);
            resetNoise(m_netTarget);
        }

        /// <summary>
        /// Calculate the gradients between the target m_loss and actual p_loss.
        /// </summary>
        /// <param name="sender">Specifies the sender.</param>
        /// <param name="e">Specifies the arguments.</param>
        private void m_memLoss_ComputeTdLoss(object sender, MemoryLossLayerGetLossArgs<T> e)
        {
            MemoryCollection rgMem = m_rgSamples;

            Blob<T> q_values = m_net.blob_by_name("logits");
            Blob<T> next_q_values = m_netTarget.blob_by_name("logits");

            float[] rgActions = rgMem.GetActionsAsOneHotVector(m_nActionCount);
            m_blobActions.ReshapeLike(q_values);
            m_blobActions.mutable_cpu_data = Utility.ConvertVec<T>(rgActions);
            m_blobQValue.ReshapeLike(q_values);

            // q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
            m_mycaffe.Cuda.mul(m_blobActions.count(), m_blobActions.gpu_data, q_values.gpu_data, m_blobQValue.mutable_gpu_data);
            reduce_sum_axis1(m_blobQValue);

            // next_q_value = next_q_values.max(1)[0]
            m_blobNextQValue.CopyFrom(next_q_values, false, true);
            reduce_argmax_axis1(m_blobNextQValue);

            // expected_q_values
            float[] rgRewards = rgMem.GetRewards();
            m_blobExpectedQValue.ReshapeLike(m_blobQValue);
            m_blobExpectedQValue.mutable_cpu_data = Utility.ConvertVec<T>(rgRewards);

            float[] rgDone = rgMem.GetInvertedDoneAsOneHotVector();
            m_blobDone.ReshapeLike(m_blobQValue);
            m_blobDone.mutable_cpu_data = Utility.ConvertVec<T>(rgDone);

            m_mycaffe.Cuda.mul(m_blobNextQValue.count(), m_blobNextQValue.gpu_data, m_blobDone.gpu_data, m_blobExpectedQValue.mutable_gpu_diff);           // next_q_val * (1- done)
            m_mycaffe.Cuda.mul_scalar(m_blobExpectedQValue.count(), m_fGamma, m_blobExpectedQValue.mutable_gpu_diff);                                      // gamma *  ^
            m_mycaffe.Cuda.add(m_blobExpectedQValue.count(), m_blobExpectedQValue.gpu_diff, m_blobExpectedQValue.gpu_data, m_blobExpectedQValue.gpu_data); // reward + ^

            // loss = (q_value - expected_q_value.detach()).pow(2) 
            m_blobLoss.ReshapeLike(m_blobQValue);
            m_mycaffe.Cuda.sub(m_blobQValue.count(), m_blobQValue.gpu_data, m_blobExpectedQValue.gpu_data, m_blobQValue.mutable_gpu_diff); // q_value - expected_q_value
            m_mycaffe.Cuda.powx(m_blobLoss.count(), m_blobQValue.gpu_diff, 2.0, m_blobLoss.mutable_gpu_data);                              // (q_value - expected_q_value)^2

            // loss = (q_value - expected_q_value.detach()).pow(2) * weights
            m_blobWeights.ReshapeLike(m_blobQValue);
            m_blobWeights.mutable_cpu_data = Utility.ConvertVec<T>(m_rgSamples.Priorities); // weights
            m_mycaffe.Cuda.mul(m_blobLoss.count(), m_blobLoss.gpu_data, m_blobWeights.gpu_data, m_blobLoss.mutable_gpu_data);               //    ^ * weights

            // prios = loss + 1e-5
            m_mycaffe.Cuda.copy(m_blobLoss.count(), m_blobLoss.gpu_data, m_blobLoss.mutable_gpu_diff);
            m_mycaffe.Cuda.add_scalar(m_blobLoss.count(), 1e-5, m_blobLoss.mutable_gpu_diff);
            double[] rgPrios = Utility.ConvertVec<T>(m_blobLoss.mutable_cpu_diff);

            for (int i = 0; i < rgPrios.Length; i++)
            {
                m_rgSamples.Priorities[i] = rgPrios[i];
            }


            //-------------------------------------------------------
            //  Calculate the gradient - unroll the operations
            //  (autograd - psha! how about manualgrad :-D)
            //-------------------------------------------------------

            // initial gradient
            double dfGradient = 1.0;
            if (m_memLoss.layer_param.loss_weight.Count > 0)
                dfGradient *= m_memLoss.layer_param.loss_weight[0];

            // mean gradient - expand and divide by batch count
            dfGradient /= m_blobLoss.count();
            m_blobLoss.SetDiff(dfGradient);

            // multiplication gradient - multiply by the other side.
            m_mycaffe.Cuda.mul(m_blobLoss.count(), m_blobLoss.gpu_diff, m_blobWeights.gpu_data, m_blobLoss.mutable_gpu_diff);

            // power gradient - multiply by the exponent.
            m_mycaffe.Cuda.mul_scalar(m_blobLoss.count(), 2.0, m_blobLoss.mutable_gpu_diff);

            // q_value - expected_q_value gradient
            m_mycaffe.Cuda.mul(m_blobLoss.count(), m_blobLoss.gpu_diff, m_blobQValue.gpu_diff, m_blobLoss.mutable_gpu_diff);

            // squeeze/gather gradient
            mul(m_blobLoss, m_blobActions, e.Bottom[0]);

            e.Loss = reduce_mean(m_blobLoss, false);
            e.EnableLossUpdate = false;
        }

        private void resetNoise(Net<T> net)
        {
            foreach (Layer<T> layer in net.layers)
            {
                if (layer.type == LayerParameter.LayerType.INNERPRODUCT)
                {
                    if (layer.layer_param.inner_product_param.enable_noise)
                        ((InnerProductLayer<T>)layer).ResetNoise();
                }
            }
        }

        private void mul(Blob<T> val, Blob<T> actions, Blob<T> result)
        {
            float[] rgVal = Utility.ConvertVecF<T>(val.mutable_cpu_diff);
            float[] rgActions = Utility.ConvertVecF<T>(actions.mutable_cpu_data);
            float[] rgResult = new float[rgActions.Length];

            for (int i = 0; i < actions.num; i++)
            {
                float fPred = rgVal[i];

                for (int j = 0; j < actions.channels; j++)
                {
                    int nIdx = (i * actions.channels) + j;
                    rgResult[nIdx] = rgActions[nIdx] * fPred; 
                }
            }

            result.mutable_cpu_diff = Utility.ConvertVec<T>(rgResult);
        }

        private float reduce_mean(Blob<T> b, bool bDiff)
        {
            float[] rg = Utility.ConvertVecF<T>((bDiff) ? b.mutable_cpu_diff : b.mutable_cpu_data);
            float fSum = rg.Sum(p => p);
            return fSum / rg.Length;
        }

        private void reduce_sum_axis1(Blob<T> b)
        {
            int nNum = b.shape(0);
            int nActions = b.shape(1);
            int nInnerCount = b.count(2);
            float[] rg = Utility.ConvertVecF<T>(b.mutable_cpu_data);
            float[] rgSum = new float[nNum * nInnerCount];

            for (int i = 0; i < nNum; i++)
            {
                for (int j = 0; j < nInnerCount; j++)
                {
                    float fSum = 0;

                    for (int k = 0; k < nActions; k++)
                    {
                        int nIdx = (i * nActions * nInnerCount) + (k * nInnerCount);
                        fSum += rg[nIdx + j];
                    }

                    int nIdxR = i * nInnerCount;
                    rgSum[nIdxR + j] = fSum;
                }
            }

            b.Reshape(nNum, nInnerCount, 1, 1);
            b.mutable_cpu_data = Utility.ConvertVec<T>(rgSum);
        }

        private void reduce_argmax_axis1(Blob<T> b)
        {
            int nNum = b.shape(0);
            int nActions = b.shape(1);
            int nInnerCount = b.count(2);
            float[] rg = Utility.ConvertVecF<T>(b.mutable_cpu_data);
            float[] rgMax = new float[nNum * nInnerCount];

            for (int i = 0; i < nNum; i++)
            {
                for (int j = 0; j < nInnerCount; j++)
                {
                    float fMax = -float.MaxValue;

                    for (int k = 0; k < nActions; k++)
                    {
                        int nIdx = (i * nActions * nInnerCount) + (k * nInnerCount);
                        fMax = Math.Max(fMax, rg[nIdx + j]);
                    }

                    int nIdxR = i * nInnerCount;
                    rgMax[nIdxR + j] = fMax;
                }
            }

            b.Reshape(nNum, nInnerCount, 1, 1);
            b.mutable_cpu_data = Utility.ConvertVec<T>(rgMax);
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
            double fMax = -float.MaxValue;
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

        private void setCurrentStateData(Net<T> net, MemoryCollection rgSamples)
        {
            List<SimpleDatum> rgData0 = rgSamples.GetCurrentStateData();
            List<SimpleDatum> rgClip0 = rgSamples.GetCurrentStateClip();

            SimpleDatum[] rgData = rgData0.ToArray();
            SimpleDatum[] rgClip = (rgClip0 != null) ? rgClip0.ToArray() : null;

            setData(net, rgData, rgClip);
        }

        private void setNextStateData(Net<T> net, MemoryCollection rgSamples)
        {
            List<SimpleDatum> rgData1 = rgSamples.GetNextStateData();
            List<SimpleDatum> rgClip1 = rgSamples.GetNextStateClip();

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
            Blob<T> logits = m_net.blob_by_name("logits");
            if (logits == null)
                return;

            if (logits.num == 1)
                m_rgOverlay = Utility.ConvertVecF<T>(logits.mutable_cpu_data);

            if (m_rgOverlay == null)
                return;

            using (Graphics g = Graphics.FromImage(e.DisplayImage))
            {
                int nBorder = 30;
                int nWid = e.DisplayImage.Width - (nBorder * 2);
                int nWid1 = nWid / m_rgOverlay.Length;
                int nHt1 = (int)(e.DisplayImage.Height * 0.3);
                int nX = nBorder;
                int nY = e.DisplayImage.Height - nHt1;
                ColorMapper clrMap = new ColorMapper(0, m_rgOverlay.Length + 1, Color.Black, Color.Red);            
                float fMax = -float.MaxValue;
                int nMaxIdx = 0;
                float fMin1 = m_rgOverlay.Min(p => p);
                float fMax1 = m_rgOverlay.Max(p => p);

                for (int i=0; i<m_rgOverlay.Length; i++)
                {
                    if (fMin1 < 0 || fMax1 > 1)
                       m_rgOverlay[i] = (m_rgOverlay[i] - fMin1) / (fMax1 - fMin1);

                    if (m_rgOverlay[i] > fMax)
                    {
                        fMax = m_rgOverlay[i];
                        nMaxIdx = i;
                    }
                }

                for (int i = 0; i < m_rgOverlay.Length; i++)
                {
                    drawProbabilities(g, nX, nY, nWid1, nHt1, i, m_rgOverlay[i], fMin1, fMax1, clrMap.GetColor(i + 1), (i == nMaxIdx) ? true : false);
                    nX += nWid1;
                }
            }
        }

        private void drawProbabilities(Graphics g, int nX, int nY, int nWid, int nHt, int nAction, float fProb, float fMin, float fMax, Color clr, bool bMax)
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
                str = "Action " + nAction.ToString() + " (" + fProb.ToString("N7") + ")";
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
                float fWid = nWid ;
                nHt -= (int)sz.Height;                

                float fHt = nHt * fProb;
                float fHt1 = nHt - fHt;
                RectangleF rc1 = new RectangleF(fX, nY + fHt1, fWid, fHt);
                g.FillRectangle(brBack, rc1);
                g.DrawRectangle(penLine, rc1.X, rc1.Y, rc1.Width, rc1.Height);
            }
        }

        /// <summary>
        /// Save the weight and bias values to file.
        /// </summary>
        /// <param name="strFile">Specifies the file to save the weights to.</param>
        public void SaveWeights(string strFile)
        {
            if (File.Exists(strFile))
                File.Delete(strFile);

            using (StreamWriter sw = new StreamWriter(strFile))
            {
                save(sw, m_net.layer_by_name("linear") as InnerProductLayer<T>);
                save(sw, m_net.layer_by_name("noisy1") as InnerProductLayer<T>);
                save(sw, m_net.layer_by_name("noisy2") as InnerProductLayer<T>);
                save(sw, m_netTarget.layer_by_name("linear") as InnerProductLayer<T>);
                save(sw, m_netTarget.layer_by_name("noisy1") as InnerProductLayer<T>);
                save(sw, m_netTarget.layer_by_name("noisy2") as InnerProductLayer<T>);
            }
        }

        private void save(StreamWriter sw, InnerProductLayer<T> layer)
        {
            float[] rgfwt = Utility.ConvertVecF<T>(layer.blobs[0].mutable_cpu_data);
            float[] rgfb = Utility.ConvertVecF<T>(layer.blobs[1].mutable_cpu_data);
            string strLine = "";

            for (int i = 0; i < rgfwt.Length; i++)
            {
                strLine += rgfwt[i].ToString() + ",";
            }

            sw.WriteLine(strLine.TrimEnd(','));
            strLine = "";

            for (int i = 0; i < rgfb.Length; i++)
            {
                strLine += rgfb[i].ToString() + ",";
            }

            sw.WriteLine(strLine.TrimEnd(','));
        }

        /// <summary>
        /// Load the weight and bias values from file.
        /// </summary>
        /// <param name="strFile">Specifies the file to load the weights from.</param>
        public void LoadWeights(string strFile)
        {
            if (!File.Exists(strFile))
                return;

            using (StreamReader sr = new StreamReader(strFile))
            {
                load(sr, m_net.layer_by_name("linear") as InnerProductLayer<T>);
                load(sr, m_net.layer_by_name("noisy1") as InnerProductLayer<T>);
                load(sr, m_net.layer_by_name("noisy2") as InnerProductLayer<T>);
                load(sr, m_netTarget.layer_by_name("linear") as InnerProductLayer<T>);
                load(sr, m_netTarget.layer_by_name("noisy1") as InnerProductLayer<T>);
                load(sr, m_netTarget.layer_by_name("noisy2") as InnerProductLayer<T>);
            }
        }

        private void load(StreamReader sr, InnerProductLayer<T> layer)
        {
            List<float> rgfwt = new List<float>();
            List<float> rgfb = new List<float>();
            string strLine;

            strLine = sr.ReadLine();
            string[] rgstr = strLine.Split(',');

            for (int i = 0; i < rgstr.Length; i++)
            {
                rgfwt.Add(float.Parse(rgstr[i]));
            }

            strLine = sr.ReadLine();
            rgstr = strLine.Split(',');

            for (int i = 0; i < rgstr.Length; i++)
            {
                rgfb.Add(float.Parse(rgstr[i]));
            }

            layer.blobs[0].mutable_cpu_data = Utility.ConvertVec<T>(rgfwt.ToArray());
            layer.blobs[1].mutable_cpu_data = Utility.ConvertVec<T>(rgfb.ToArray());
        }
    }
}
