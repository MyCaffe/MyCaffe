using MyCaffe;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.data;
using MyCaffe.layers;
using MyCaffe.param;
using MyCaffe.solvers;
using MyCaffe.trainers;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.trainers.noisy.dqn.simple
{
    /// <summary>
    /// The TrainerNoisyDqn implements the Noisy-DQN algorithm as described by Gheshlagi et al., and 'Kyushik'
    /// </summary>
    /// <remarks>
    /// @see [Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295), Meire Fortunato, Mohammad Gheshlaghi Azar, Bilal Piot, Jacob Menick, Ian Osband, Alex Graves, Vlad Mnih, Remi Munos, Demis Hassabis, Olivier Pietquin, Charles Blundell, Shane Legg, arXiv:1706.10295
    /// @see [Github:higgsfield/RL-Adventure](https://github.com/higgsfield/RL-Adventure) 2018
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
            PrioritizedMemoryCollection rgMemory = new PrioritizedMemoryCollection(10000, 0.6f);
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
                // Forward the policy network and sample an action.
                int action = m_brain.act(x, s.Clip, s.ActionCount);

                // Take the next step using the action
                StateBase s_ = getData(phase, action, nIteration);

                // Preprocess the next observation.
                SimpleDatum x_ = m_brain.Preprocess(s_, m_bUseRawInput, out bDifferent);
                if (!bDifferent)
                    m_brain.Log.WriteLine("WARNING: The current state is the same as the previous state!");

                // Build up episode memory, using reward for taking the action.
                rgMemory.Add(new MemoryItem(s, x, action, s_, x_, s_.Reward, s_.Done, nIteration, nEpisode));
                dfRewardSum += s_.Reward;

                if (rgMemory.Count > m_brain.BatchSize)
                {
                    double dfBeta = beta_by_frame(nIteration);
                    Tuple<MemoryCollection, int[], float[]> rgRandomSamples = rgMemory.GetSamples(m_random, m_brain.BatchSize, dfBeta);
                    m_brain.Train(rgRandomSamples, s.ActionCount);
                    rgMemory.UpdatePriorities(rgRandomSamples.Item2, rgRandomSamples.Item3);

                    if (nIteration % 1000 == 0)
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
        ArgMaxLayer<T> m_argmax = null;
        Blob<T> m_blobActions = null;
        Blob<T> m_blobQValue = null;
        Blob<T> m_blobNextQValue = null;
        Blob<T> m_blobExpectedQValue = null;
        Blob<T> m_blobDone = null;
        Blob<T> m_blobLoss = null;
        Blob<T> m_blobWeights = null;
        float m_fGamma = 0.99f;
        int m_nBatchSize = 32;
        Tuple<MemoryCollection, int[], float[]> m_rgSamples;
        int m_nActionCount = 3;
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

            m_transformer = m_mycaffe.DataTransformer;
            m_transformer.param.mean_value.Add(255 / 2); // center
            m_transformer.param.mean_value.Add(255 / 2);
            m_transformer.param.mean_value.Add(255 / 2);
            m_transformer.param.mean_value.Add(255 / 2);
            m_transformer.param.scale = 1.0 / 255;       // normalize
            m_transformer.Update();

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

            Blob<T> data = m_net.blob_by_name("data");
            if (data == null)
                m_mycaffe.Log.FAIL("Missing the expected input 'data' blob!");

            m_nBatchSize = data.num;
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

            if (m_argmax != null)
            {
                m_argmax.Dispose();
                m_argmax = null;
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
        /// <param name="rgSamples">Contains the samples to train the model with along with the priorities associated with the samples.</param>
        /// <param name="nActionCount">Specifies the number of actions in the action set.</param>
        public void Train(Tuple<MemoryCollection, int[], float[]> rgSamples1, int nActionCount)
        {
            MemoryCollection rgSamples = rgSamples1.Item1;

            m_rgSamples = rgSamples1;
            m_nActionCount = nActionCount;

            // Get next_q_values
            m_mycaffe.Log.Enable = false;
            setData1(m_netTarget, rgSamples);
            m_netTarget.ForwardFromTo(0, m_netTarget.layers.Count - 2);

            setData0(m_net, rgSamples);
            m_memLoss.OnGetLoss += m_memLoss_ComputeTdLoss;
            m_solver.Step(1);
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
            Blob<T> q_values = m_net.blob_by_name("logits");
            Blob<T> next_q_values = m_netTarget.blob_by_name("logits");

            float[] rgActions = m_rgSamples.Item1.GetActionsAsOneHotVector(m_nActionCount);
            m_blobActions.ReshapeLike(q_values);
            m_blobActions.mutable_cpu_data = Utility.ConvertVec<T>(rgActions);
            m_blobQValue.ReshapeLike(q_values);

            // q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
            m_mycaffe.Cuda.mul(m_blobActions.count(), m_blobActions.gpu_data, q_values.gpu_data, m_blobQValue.mutable_gpu_data);
            reduce_sum_axis1(m_blobQValue);

            // next_q_value = next_q_values.max(1)[0]
            argmax_forward(next_q_values, m_blobNextQValue);

            // expected_q_values
            float[] rgRewards = m_rgSamples.Item1.GetRewards();
            m_blobExpectedQValue.ReshapeLike(m_blobQValue);
            m_blobExpectedQValue.mutable_cpu_data = Utility.ConvertVec<T>(rgRewards);

            float[] rgDone = m_rgSamples.Item1.GetInvertedDoneAsOneHotVector();
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
            m_blobWeights.mutable_cpu_data = Utility.ConvertVec<T>(m_rgSamples.Item3); // weights
            m_mycaffe.Cuda.mul(m_blobLoss.count(), m_blobLoss.gpu_data, m_blobWeights.gpu_data, m_blobLoss.mutable_gpu_data);               //    ^ * weights

            // prios = loss + 1e-5
            m_mycaffe.Cuda.copy(m_blobLoss.count(), m_blobLoss.gpu_data, m_blobLoss.mutable_gpu_diff);
            m_mycaffe.Cuda.add_scalar(m_blobLoss.count(), 1e-5, m_blobLoss.mutable_gpu_diff);
            float[] rgWeights = Utility.ConvertVecF<T>(m_blobLoss.mutable_cpu_diff);

            for (int i = 0; i < rgWeights.Length; i++)
            {
                m_rgSamples.Item3[i] = rgWeights[i];
            }


            //-------------------------------------------------------
            //  Calculate the gradient - unroll the operations
            //  (autograd - psha! how about manualgrad :-D)
            //-------------------------------------------------------

            // initial gradient
            double dfGradient = 1.0;
            if (m_memLoss.layer_param.loss_weight.Count > 0)
                dfGradient *= m_memLoss.layer_param.loss_weight[0];

            // mean gradient - expand and divide by count
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

        private void argmax_forward(Blob<T> blobBottom, Blob<T> blobTop)
        {
            BlobCollection<T> colBottom = new BlobCollection<T>();
            colBottom.Add(blobBottom);
            BlobCollection<T> colTop = new BlobCollection<T>();
            colTop.Add(blobTop);

            if (m_argmax == null)
            {
                LayerParameter p = new LayerParameter(LayerParameter.LayerType.ARGMAX);
                p.argmax_param.axis = 1;
                m_argmax = Layer<T>.Create(m_mycaffe.Cuda, m_mycaffe.Log, p, null) as ArgMaxLayer<T>;
                m_argmax.Setup(colBottom, colTop);
            }

            m_argmax.Forward(colBottom, colTop);
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
    }

    /// <summary>
    /// The PrioritizedMemoryCollection provides a sampling based on prioritizations.
    /// </summary>
    class PrioritizedMemoryCollection : MemoryCollection
    {
        float m_fAlpha;
        float m_fMaxPriority = 1.0f;
        int m_nItCapacity = 1;
        SumSegmentTree m_ItSum;
        MinSegmentTree m_ItMin;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="nMax">Specifies the maximum number of items in the collection.</param>
        /// <param name="fAlpha">Specifies how much prioritization is used (0 = no prioritization, 1 = full prioritization).</param>
        public PrioritizedMemoryCollection(int nMax, float fAlpha)
            : base(nMax)
        {
            m_fAlpha = fAlpha;

            while (m_nItCapacity < nMax)
            {
                m_nItCapacity *= 2;
            }

            m_ItSum = new SumSegmentTree(m_nItCapacity);
            m_ItMin = new MinSegmentTree(m_nItCapacity);
        }

        /// <summary>
        /// Add a new item to the collection.
        /// </summary>
        /// <param name="m"></param>
        public override void Add(MemoryItem m)
        {
            int nIdx = m_nNextIdx;
            base.Add(m);

            int nVal = (int)Math.Pow(m_fMaxPriority, m_fAlpha);
            m_ItSum[nIdx] = nVal;
            m_ItMin[nIdx] = nVal;
        }

        private int[] getSamplesProportional(CryptoRandom random, int nCount)
        {
            int[] rgIdx = new int[nCount];

            for (int i = 0; i < nCount; i++)
            {
                double dfRand = random.NextDouble();
                double dfMass = dfRand * m_ItSum.sum(0, Count - 1);
                int nIdx = m_ItSum.find_prefixsum_idx((float)dfMass);
                rgIdx[i] = nIdx;
            }

            return rgIdx;
        }

        /// <summary>
        /// Return a batch of items.
        /// </summary>
        /// <param name="random">Specifies the random number generator.</param>
        /// <param name="nCount">Specifies the number of items to sample.</param>
        /// <param name="dfBeta">Specifies the degree to use importance weights (0 = no corrections, 1 = full corrections).</param>
        /// <returns>The prioritized array of items is returned along with the weights and indexes.</returns>
        public Tuple<MemoryCollection, int[], float[]> GetSamples(CryptoRandom random, int nCount, double dfBeta)
        {
            int[] rgIdx = getSamplesProportional(random, nCount);
            float[] rgfWeights = new float[nCount];
            float fSum = m_ItSum.sum();
            float fMin = m_ItMin.min();
            float fPMin = fMin / fSum;
            float fMaxWeight = (float)Math.Pow(fPMin * Count, -dfBeta);
            MemoryCollection col = new MemoryCollection(nCount);

            for (int i = 0; i < rgIdx.Length; i++)
            {
                int nIdx = rgIdx[i];
                float fItSum = m_ItSum[nIdx];
                float fPSample = fItSum / fSum;
                float fWeight = (float)Math.Pow(fPSample * Count, -dfBeta);
                rgfWeights[i] = fWeight / fMaxWeight;

                col.Add(m_rgItems[nIdx]);
            }

            return new Tuple<MemoryCollection, int[], float[]>(col, rgIdx, rgfWeights);
        }

        /// <summary>
        /// Update the priorities of sampled transitions.
        /// </summary>
        /// <remarks>
        /// Sets priority of transitions at index rgIdx[i] in buffer to priorities[i].
        /// </remarks>
        /// <param name="rgIdx">Specifies the list of indexed sampled transitions.</param>
        /// <param name="rgfPriorities">Specifies the list of updated priorities corresponding to transitions at the sampled indexes donated by variable 'rgIdx'.</param>
        public void UpdatePriorities(int[] rgIdx, float[] rgfPriorities)
        {
            if (rgIdx.Length != rgfPriorities.Length)
                throw new Exception("The index and priority arrays must have the same length.");

            for (int i = 0; i < rgIdx.Length; i++)
            {
                int nIdx = rgIdx[i];
                float fPriority = rgfPriorities[i];

                if (fPriority <= 0)
                    throw new Exception("The priority at index '" + i.ToString() + "' is zero!");

                if (nIdx < 0 || nIdx >= m_rgItems.Length)
                    throw new Exception("The index at index '" + i.ToString() + "' is out of range!");

                float fNewPriority = (float)Math.Pow(fPriority, m_fAlpha);
                m_ItSum[nIdx] = fNewPriority;
                m_ItMin[nIdx] = fNewPriority;
                m_fMaxPriority = Math.Max(m_fMaxPriority, fPriority);
            }
        }
    }

    /// <summary>
    /// Segment tree data structure
    /// </summary>
    /// <remarks>
    /// The segment tree can be used as a regular array, but with two important differences:
    /// 
    ///   a.) Setting an item's value is slightly slower: O(lg capacity) instead of O(1).
    ///   b.) User has access to an efficient 'reduce' operation which reduces the 'operation' 
    ///       over a contiguous subsequence of items in the array.
    /// 
    /// @see [Wikipedia: Segment tree](https://en.wikipedia.org/wiki/Segment_tree)
    /// @see [GitHub: openai/baselines](https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py) 2018
    /// @see [GitHub: higgsfield/RL-Adventure](https://github.com/higgsfield/RL-Adventure/blob/master/common/replay_buffer.py) 2018
    /// </remarks>
    class SegmentTree
    {
        protected int m_nCapacity;
        protected OPERATION m_op;
        protected float[] m_rgfValues;

        /// <summary>
        /// Specifies the operations used during the reduction.
        /// </summary>
        public enum OPERATION
        {
            /// <summary>
            /// Sum the two elements together.
            /// </summary>
            SUM,
            /// <summary>
            /// Return the minimum of the two elements.
            /// </summary>
            MIN
        }

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="nCapacity">Specifies the total size of the array - must be a power of two.</param>
        /// <param name="oper">Specifies the operation for combining elements (e.g. sum, min)</param>
        /// <param name="fNeutralElement">Specifies the nautral element for the operation above (e.g. float.MaxValue for min and 0 for sum).</param>
        public SegmentTree(int nCapacity, OPERATION oper, float fNeutralElement)
        {
            if (nCapacity <= 0 || (nCapacity % 2) != 0)
                throw new Exception("The capacity must be positive and a power of 2.");

            m_nCapacity = nCapacity;
            m_op = oper;
            m_rgfValues = new float[2 * nCapacity];

            for (int i = 0; i < m_rgfValues.Length; i++)
            {
                m_rgfValues[i] = fNeutralElement;
            }
        }

        private float operation(float f1, float f2)
        {
            switch (m_op)
            {
                case OPERATION.MIN:
                    return Math.Min(f1, f2);

                case OPERATION.SUM:
                    return f1 + f2;

                default:
                    throw new Exception("Unknown operation '" + m_op.ToString() + "'!");
            }
        }

        private float reduce_helper(int nStart, int nEnd, int nNode, int nNodeStart, int nNodeEnd)
        {
            if (nStart == nNodeStart && nEnd == nNodeEnd)
                return m_rgfValues[nNode];

            int nMid = (int)Math.Floor((nNodeStart + nNodeEnd) / 2.0);

            if (nEnd <= nMid)
            {
                return reduce_helper(nStart, nEnd, 2 * nNode, nNodeStart, nMid);
            }
            else
            {
                if (nMid + 1 < nStart)
                {
                    return reduce_helper(nStart, nMid, 2 * nNode + 1, nMid + 1, nNodeEnd);
                }
                else
                {
                    float f1 = reduce_helper(nStart, nMid, 2 * nNode, nNodeStart, nMid);
                    float f2 = reduce_helper(nMid + 1, nEnd, 2 * nNode + 1, nMid + 1, nNodeEnd);
                    return operation(f1, f2);
                }
            }
        }

        /// <summary>
        /// Returns result of applying self.operation to a contiguous subsequence of the array.
        /// operation(arr[start], operation(ar[start+1], operation(..., arr[end])))
        /// </summary>
        /// <param name="nStart">Beginning of the subsequence.</param>
        /// <param name="nEnd">End of the subsequence</param>
        /// <returns></returns>
        public float reduce(int nStart, int? nEnd1 = null)
        {
            int nEnd = nEnd1.GetValueOrDefault(m_nCapacity);

            if (nEnd < 0)
                nEnd += m_nCapacity;

            nEnd -= 1;

            return reduce_helper(nStart, nEnd, 1, 0, m_nCapacity - 1);
        }

        /// <summary>
        /// Element accessor to get and set items.
        /// </summary>
        /// <param name="nIdx">Specifies the index of the item to access.</param>
        /// <returns>The item at the specified index is returned.</returns>
        public float this[int nIdx]
        {
            get
            {
                if (nIdx < 0 || nIdx >= m_nCapacity)
                    throw new Exception("The index is out of range!");

                return m_rgfValues[m_nCapacity + nIdx];
            }
            set
            {
                nIdx += m_nCapacity;
                m_rgfValues[nIdx] = value;

                nIdx = (int)Math.Floor(nIdx / 2.0);

                while (nIdx >= 1)
                {
                    m_rgfValues[nIdx] = operation(m_rgfValues[2 * nIdx], m_rgfValues[2 * nIdx + 1]);
                    nIdx = (int)Math.Floor(nIdx / 2.0);
                }
            }
        }
    }

    /// <summary>
    /// The SumSegmentTree provides a sum reduction of the items within the array.
    /// </summary>
    class SumSegmentTree : SegmentTree
    {
        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="nCapacity">Specifies the total size of the array - must be a power of two.</param>
        public SumSegmentTree(int nCapacity)
            : base(nCapacity, OPERATION.SUM, 0.0f)
        {
        }

        /// <summary>
        /// Returns arr[start] + ... + arr[end]
        /// </summary>
        /// <param name="nStart">Beginning of the subsequence.</param>
        /// <param name="nEnd">End of the subsequence</param>
        /// <returns>Returns the sum of all items in the array.</returns>
        public float sum(int nStart=0, int? nEnd1 = null)
        {
            return reduce(nStart, nEnd1);
        }

        /// <summary>
        /// Finds the highest indes 'i' in the array such that sum(arr[0] + arr[1] + ... + arr[i-1]) less than or equal to the 'fPrefixSum'
        /// </summary>
        /// <remarks>
        /// If array values are probabilities, this function allows to sample indexes according to the discrete probability efficiently.
        /// </remarks>
        /// <param name="fPrefixSum">Specifies the upper bound on the sum of array prefix.</param>
        /// <returns>The highest index satisfying the prefixsum constraint is returned.</returns>
        public int find_prefixsum_idx(float fPrefixSum)
        {
            if (fPrefixSum < 0)
                throw new Exception("The prefix sum must be greater than zero.");

            float fSum = sum() + (float)1e-5;
            if (fPrefixSum > fSum)
                throw new Exception("The prefix sum cannot exceed the overall sum of '" + fSum.ToString() + "'!");

            int nIdx = 1;
            while (nIdx < m_nCapacity) // while non-leaf
            {
                if (m_rgfValues[2 * nIdx] > fPrefixSum)
                {
                    nIdx = 2 * nIdx;
                }
                else
                {
                    fPrefixSum -= m_rgfValues[2 * nIdx];
                    nIdx = 2 * nIdx + 1;
                }
            }

            return nIdx - m_nCapacity;
        }
    }

    /// <summary>
    /// The MinSegmentTree performs a reduction over the array and returns the minimum value.
    /// </summary>
    class MinSegmentTree : SegmentTree
    {
        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="nCapacity">Specifies the total size of the array - must be a power of two.</param>
        public MinSegmentTree(int nCapacity)
            : base(nCapacity, OPERATION.MIN, float.MaxValue)
        {
        }

        /// <summary>
        /// Returns the minimum element in the array.
        /// </summary>
        /// <param name="nStart">Beginning of the subsequence.</param>
        /// <param name="nEnd">End of the subsequence</param>
        /// <returns>The minimum item in the sequence is returned.</returns>
        public float min(int nStart = 0, int? nEnd1 = null)
        {
            return reduce(nStart, nEnd1);
        }
    }

    class MemoryCollection /** @private */
    {
        protected MemoryItem[] m_rgItems;
        protected int m_nNextIdx = 0;

        public MemoryCollection(int nMax)
        {
            m_rgItems = new MemoryItem[nMax];
        }

        public int Count
        {
            get { return m_nNextIdx; }
        }

        public MemoryItem this[int nIdx]
        {
            get { return m_rgItems[nIdx]; }
        }

        public virtual void Add(MemoryItem item)
        {
            m_rgItems[m_nNextIdx] = item;
            m_nNextIdx++;

            if (m_nNextIdx == m_rgItems.Length)
                m_nNextIdx = 0;
        }

        public MemoryCollection GetRandomSamples(CryptoRandom random, int nCount)
        {
            MemoryCollection col = new MemoryCollection(nCount);
            List<int> rgIdx = new List<int>();

            while (col.Count < nCount)
            {
                int nIdx = random.Next(m_rgItems.Length);
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

        public float[] GetActionsAsOneHotVector(int nActionCount)
        {
            float[] rg = new float[m_rgItems.Length * nActionCount];

            for (int i = 0; i < m_rgItems.Length; i++)
            {
                int nAction = m_rgItems[i].Action;

                for (int j = 0; j < nActionCount; j++)
                {
                    rg[(i * nActionCount) + j] = (j == nAction) ? 1 : 0;
                }
            }

            return rg;
        }

        public float[] GetInvertedDoneAsOneHotVector()
        {
            float[] rgDoneInv = new float[m_rgItems.Length];

            for (int i = 0; i < m_rgItems.Length; i++)
            {
                if (m_rgItems[i].IsTerminated)
                    rgDoneInv[i] = 0;
                else
                    rgDoneInv[i] = 1;
            }

            return rgDoneInv;
        }

        public float[] GetRewards()
        {
            return m_rgItems.Select(p => (float)p.Reward).ToArray();
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
