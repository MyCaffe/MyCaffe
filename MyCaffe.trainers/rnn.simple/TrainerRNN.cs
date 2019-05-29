using System;
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

namespace MyCaffe.trainers.rnn.simple
{
    /// <summary>
    /// The TrainerRNN implements a simple RNN trainer inspired by adepierre's GitHub site  referenced. 
    /// </summary>
    /// @see 1. [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/), by Andrej Karpathy, 2015, Github.io
    /// @see 2. [GitHub: adepierre/caffe-char-rnn](https://github.com/adepierre/caffe-char-rnn), by adepierre, 2017, Github
    /// @see 4. [MyCaffe: A Complete C# Re-Write of Caffe with Reinforcement Learning](https://arxiv.org/abs/1810.02272) by D. Brown, 2018, arXiv
    /// <remarks></remarks>
    public class TrainerRNN<T> : IxTrainerRNN, IDisposable
    {
        IxTrainerCallback m_icallback;
        MyCaffeControl<T> m_mycaffe;
        PropertySet m_properties;
        CryptoRandom m_random;
        BucketCollection m_rgVocabulary = null;
        bool m_bUsePreloadData = true;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="mycaffe">Specifies the MyCaffeControl to use for learning and prediction.</param>
        /// <param name="properties">Specifies the property set containing the key/value pairs of property settings.</param>
        /// <param name="random">Specifies the random number generator to use.</param>
        /// <param name="icallback">Specifies the callback for parent notifications and queries.</param>
        /// <param name="rgVocabulary">Specifies the vocabulary to use.</param>
        /// <param name="bUsePreloadData">Specifies whether or not to use the preloaded data, and if not, to use dynamic data.</param>
        public TrainerRNN(MyCaffeControl<T> mycaffe, PropertySet properties, CryptoRandom random, IxTrainerCallback icallback, BucketCollection rgVocabulary, bool bUsePreloadData)
        {
            m_icallback = icallback;
            m_mycaffe = mycaffe;
            m_properties = properties;
            m_random = random;
            m_rgVocabulary = rgVocabulary;
            m_bUsePreloadData = bUsePreloadData;
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
        /// <param name="nN">specifies the number of samples to run.</param>
        /// <param name="strRunProperties">Optionally specifies properties to use when running.</param>
        /// <returns>The results of the run containing the action are returned.</returns>
        public float[] Run(int nN, string strRunProperties)
        {
            m_mycaffe.CancelEvent.Reset();
            Agent<T> agent = new Agent<T>(m_icallback, m_mycaffe, m_properties, m_random, Phase.RUN, m_rgVocabulary, m_bUsePreloadData, strRunProperties);
            float[] rgResults = agent.Run(nN);
            agent.Dispose();

            return rgResults;
        }

        /// <summary>
        /// <summary>
        /// Run a single cycle on the environment after the delay.
        /// </summary>
        /// <param name="nN">Specifies the number of samples to run.</param>
        /// <param name="strRunProperties">Optionally specifies properties to use when running.</param>
        /// <param name="type">Returns the data type contained in the byte stream.</param>
        /// <returns>The results of the run containing the action are returned as a byte stream.</returns>
        public byte[] Run(int nN, string strRunProperties, out string type)
        {
            m_mycaffe.CancelEvent.Reset();
            Agent<T> agent = new Agent<T>(m_icallback, m_mycaffe, m_properties, m_random, Phase.RUN, m_rgVocabulary, m_bUsePreloadData, strRunProperties);
            byte[] rgResults = agent.Run(nN, out type);
            agent.Dispose();

            return rgResults;
        }

        /// <summary>
        /// Run the test cycle - currently this is not implemented.
        /// </summary>
        /// <param name="nIterations">Specifies the number of iterations to run.</param>
        /// <returns>A value of <i>true</i> is returned when handled, <i>false</i> otherwise.</returns>
        public bool Test(int nIterations)
        {
            int nDelay = 1000;

            m_mycaffe.CancelEvent.Reset();
            Agent<T> agent = new Agent<T>(m_icallback, m_mycaffe, m_properties, m_random, Phase.TEST, m_rgVocabulary, m_bUsePreloadData);
            agent.Run(Phase.TEST, nIterations, TRAIN_STEP.NONE);

            agent.Dispose();
            Shutdown(nDelay);

            return true;
        }

        /// <summary>
        /// Train the network using a modified PG training algorithm optimized for GPU use.
        /// </summary>
        /// <param name="nIterations">Specifies the number of iterations to run.</param>
        /// <param name="step">Specifies the stepping mode to use (when debugging).</param>
        /// <returns>A value of <i>true</i> is returned when handled, <i>false</i> otherwise.</returns>
        public bool Train(int nIterations, TRAIN_STEP step)
        {
            m_mycaffe.CancelEvent.Reset();
            Agent<T> agent = new Agent<T>(m_icallback, m_mycaffe, m_properties, m_random, Phase.TRAIN, m_rgVocabulary, m_bUsePreloadData);
            agent.Run(Phase.TRAIN, nIterations, step);

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

        public Agent(IxTrainerCallback icallback, MyCaffeControl<T> mycaffe, PropertySet properties, CryptoRandom random, Phase phase, BucketCollection rgVocabulary, bool bUsePreloadData, string strRunProperties = null)
        {
            m_icallback = icallback;
            m_brain = new Brain<T>(mycaffe, properties, random, icallback as IxTrainerCallbackRNN, phase, rgVocabulary, bUsePreloadData, strRunProperties);
            m_properties = properties;
            m_random = random;
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
            GetDataArgs args = m_brain.getDataArgs(phase, 0, nAction, true);
            m_icallback.OnGetData(args);
            return args.State;
        }


        /// <summary>
        /// The Run method provides the main 'actor' loop that performs the following steps:
        /// 1.) Feed data into the network.
        /// 2.) either Test the network...
        /// 3.) ... or Train the network.
        /// </summary>
        /// <param name="phase">Specifies the phae.</param>
        /// <param name="nIterations">Specifies the number of iterations to run.</param>
        /// <param name="step">Specifies the training step (used only during debugging).</param>
        /// <returns>The vocabulary built up during training and testing is returned.</returns>
        public void Run(Phase phase, int nIterations, TRAIN_STEP step)
        {
            StateBase s = getData(phase, -1);

            while (!m_brain.Cancel.WaitOne(0) && !s.Done)
            {
                if (phase == Phase.TEST)
                    m_brain.Test(s, nIterations);
                else if (phase == Phase.TRAIN)
                    m_brain.Train(s, nIterations, step);

                s = getData(phase, 1);
            }
        }

        /// <summary>
        /// The Run method provides the main 'actor' that runs data through the trained network.
        /// </summary>
        /// <param name="nN">specifies the number of samples to run.</param>
        /// <returns>The results of the run are returned.</returns>
        public float[] Run(int nN)
        {
            return m_brain.Run(nN);
        }

        /// <summary>
        /// The Run method provides the main 'actor' that runs data through the trained network.
        /// </summary>
        /// <param name="nN">specifies the number of samples to run.</param>
        /// <returns>The results of the run are returned in the native format used by the CustomQuery.</returns>
        public byte[] Run(int nN, out string type)
        {
            float[] rgResults = m_brain.Run(nN);

            ConvertOutputArgs args = new ConvertOutputArgs(nN, rgResults);
            IxTrainerCallbackRNN icallback = m_icallback as IxTrainerCallbackRNN;
            if (icallback == null)
                throw new Exception("The Run method requires an IxTrainerCallbackRNN interface to convert the results into the native format!");

            icallback.OnConvertOutput(args);

            type = args.RawType;
            return args.RawOutput;
        }
    }

    class Brain<T> : IDisposable /** @private */
    {
        IxTrainerCallbackRNN m_icallback;
        MyCaffeControl<T> m_mycaffe;
        Net<T> m_net;
        Solver<T> m_solver;
        PropertySet m_properties;
        PropertySet m_runProperties = null;
        Blob<T> m_blobData;
        Blob<T> m_blobClip;
        Blob<T> m_blobLabel;
        Blob<T> m_blobOutput = null;
        int m_nSequenceLength;
        int m_nSequenceLengthLabel;
        int m_nBatchSize;
        int m_nVocabSize = 1;
        CryptoRandom m_random;
        T[] m_rgDataInput;
        T[] m_rgLabelInput;
        T m_tZero;
        T m_tOne;
        double m_dfTemperature = 0;
        byte[] m_rgTestData = null;
        byte[] m_rgTrainData = null;
        double[] m_rgdfTestData = null;
        double[] m_rgdfTrainData = null;
        bool m_bIsDataReal = false;
        Stopwatch m_sw = new Stopwatch();
        double m_dfLastLoss = 0;
        double m_dfLastLearningRate = 0;
        BucketCollection m_rgVocabulary = null;
        bool m_bUsePreloadData = true;
        bool m_bDisableVocabulary = false;
        Phase m_phaseOnRun = Phase.NONE;
        LayerParameter.LayerType m_lstmType = LayerParameter.LayerType.LSTM;
        int m_nSolverSequenceLength = -1;
        int m_nThreads = 1;
        DataCollectionPool m_dataPool = new DataCollectionPool();
        double m_dfScale = 1.0;

        public Brain(MyCaffeControl<T> mycaffe, PropertySet properties, CryptoRandom random, IxTrainerCallbackRNN icallback, Phase phase, BucketCollection rgVocabulary, bool bUsePreloadData, string strRunProperties = null)
        {
            string strOutputBlob = null;

            if (strRunProperties != null)
                m_runProperties = new PropertySet(strRunProperties);

            m_icallback = icallback;
            m_mycaffe = mycaffe;
            m_properties = properties;
            m_random = random;
            m_rgVocabulary = rgVocabulary;
            m_bUsePreloadData = bUsePreloadData;
            m_nSolverSequenceLength = m_properties.GetPropertyAsInt("SequenceLength", -1);
            m_bDisableVocabulary = m_properties.GetPropertyAsBool("DisableVocabulary", false);
            m_nThreads = m_properties.GetPropertyAsInt("Threads", 1);
            m_dfScale = m_properties.GetPropertyAsDouble("Scale", 1.0);

            if (m_nThreads > 1)
                m_dataPool.Initialize(m_nThreads, icallback);

            if (m_runProperties != null)
            {
                m_dfTemperature = Math.Abs(m_runProperties.GetPropertyAsDouble("Temperature", 0));
                if (m_dfTemperature > 1.0)
                    m_dfTemperature = 1.0;

                string strPhaseOnRun = m_runProperties.GetProperty("PhaseOnRun", false);
                switch (strPhaseOnRun)
                {
                    case "RUN":
                        m_phaseOnRun = Phase.RUN;
                        break;

                    case "TEST":
                        m_phaseOnRun = Phase.TEST;
                        break;

                    case "TRAIN":
                        m_phaseOnRun = Phase.TRAIN;
                        break;
                }

                if (phase == Phase.RUN && m_phaseOnRun != Phase.NONE)
                {
                    if (m_phaseOnRun != Phase.RUN)
                        m_mycaffe.Log.WriteLine("Warning: Running on the '" + m_phaseOnRun.ToString() + "' network.");

                    strOutputBlob = m_runProperties.GetProperty("OutputBlob", false);
                    if (strOutputBlob == null)
                        throw new Exception("You must specify the 'OutputBlob' when Running with a phase other than RUN.");

                    strOutputBlob = Utility.Replace(strOutputBlob, '~', ';');

                    phase = m_phaseOnRun;
                }
            }

            m_net = mycaffe.GetInternalNet(phase);
            if (m_net == null)
            {
                mycaffe.Log.WriteLine("WARNING: Test net does not exist, set test_iteration > 0.  Using TRAIN phase instead.");
                m_net = mycaffe.GetInternalNet(Phase.TRAIN);
            }

            // Find the first LSTM layer to determine how to load the data.
            // NOTE: Only LSTM has a special loading order, other layers use the standard N, C, H, W ordering.
            LSTMLayer<T> lstmLayer = null;
            LSTMSimpleLayer<T> lstmSimpleLayer = null;
            foreach (Layer<T> layer1 in m_net.layers)
            {
                if (layer1.layer_param.type == LayerParameter.LayerType.LSTM)
                {
                    lstmLayer = layer1 as LSTMLayer<T>;
                    m_lstmType = LayerParameter.LayerType.LSTM;
                    break;
                }
                else if (layer1.layer_param.type == LayerParameter.LayerType.LSTM_SIMPLE)
                {
                    lstmSimpleLayer = layer1 as LSTMSimpleLayer<T>;
                    m_lstmType = LayerParameter.LayerType.LSTM_SIMPLE;
                    break;
                }
            }

            if (lstmLayer == null && lstmSimpleLayer == null)
                throw new Exception("Could not find the required LSTM or LSTM_SIMPLE layer!");

            if (m_phaseOnRun != Phase.NONE && m_phaseOnRun != Phase.RUN && strOutputBlob != null)
            {
                if ((m_blobOutput = m_net.FindBlob(strOutputBlob)) == null)
                    throw new Exception("Could not find the 'Output' layer top named '" + strOutputBlob + "'!");
            }

            if ((m_blobData = m_net.FindBlob("data")) == null)
                throw new Exception("Could not find the 'Input' layer top named 'data'!");

            if ((m_blobClip = m_net.FindBlob("clip")) == null)
                throw new Exception("Could not find the 'Input' layer top named 'clip'!");

            Layer<T> layer = m_net.FindLastLayer(LayerParameter.LayerType.INNERPRODUCT);
            m_mycaffe.Log.CHECK(layer != null, "Could not find an ending INNERPRODUCT layer!");

            if (!m_bDisableVocabulary)
            {
                m_nVocabSize = (int)layer.layer_param.inner_product_param.num_output;
                if (rgVocabulary != null)
                    m_mycaffe.Log.CHECK_EQ(m_nVocabSize, rgVocabulary.Count, "The vocabulary count = '" + rgVocabulary.Count.ToString() + "' and last inner product output count = '" + m_nVocabSize.ToString() + "' - these do not match but they should!");
            }

            if (m_lstmType == LayerParameter.LayerType.LSTM)
            {
                m_nSequenceLength = m_blobData.shape(0);
                m_nBatchSize = m_blobData.shape(1);
            }
            else
            {
                m_nBatchSize = (int)lstmSimpleLayer.layer_param.lstm_simple_param.batch_size;
                m_nSequenceLength = m_blobData.shape(0) / m_nBatchSize;

                if (phase == Phase.RUN)
                {
                    m_nBatchSize = 1;

                    List<int> rgNewShape = new List<int>() { m_nSequenceLength, 1 };
                    m_blobData.Reshape(rgNewShape);
                    m_blobClip.Reshape(rgNewShape);
                    m_net.Reshape();
                }
            }

            m_mycaffe.Log.CHECK_EQ(m_nSequenceLength, m_blobData.num, "The data num must equal the sequence lengh of " + m_nSequenceLength.ToString());

            m_rgDataInput = new T[m_nSequenceLength * m_nBatchSize];

            T[] rgClipInput = new T[m_nSequenceLength * m_nBatchSize];
            m_mycaffe.Log.CHECK_EQ(rgClipInput.Length, m_blobClip.count(), "The clip count must equal the sequence length * batch size: " + rgClipInput.Length.ToString());
            m_tZero = (T)Convert.ChangeType(0, typeof(T));
            m_tOne = (T)Convert.ChangeType(1, typeof(T));

            for (int i = 0; i < rgClipInput.Length; i++)
            {
                if (m_lstmType == LayerParameter.LayerType.LSTM)
                    rgClipInput[i] = (i < m_nBatchSize) ? m_tZero : m_tOne;
                else
                    rgClipInput[i] = (i % m_nSequenceLength == 0) ? m_tZero : m_tOne;
            }

            m_blobClip.mutable_cpu_data = rgClipInput;

            if (phase != Phase.RUN)
            {
                m_solver = mycaffe.GetInternalSolver();
                m_solver.OnStart += m_solver_OnStart;
                m_solver.OnTestStart += m_solver_OnTestStart;
                m_solver.OnTestingIteration += m_solver_OnTestingIteration;
                m_solver.OnTrainingIteration += m_solver_OnTrainingIteration;

                if ((m_blobLabel = m_net.FindBlob("label")) == null)
                    throw new Exception("Could not find the 'Input' layer top named 'label'!");

                m_nSequenceLengthLabel = m_blobLabel.count(0, 2);
                m_rgLabelInput = new T[m_nSequenceLengthLabel];
                m_mycaffe.Log.CHECK_EQ(m_rgLabelInput.Length, m_blobLabel.count(), "The label count must equal the label sequence length * batch size: " + m_rgLabelInput.Length.ToString());
                m_mycaffe.Log.CHECK(m_nSequenceLengthLabel == m_nSequenceLength * m_nBatchSize || m_nSequenceLengthLabel == 1, "The label sqeuence length must be 1 or equal the length of the sequence: " + m_nSequenceLength.ToString());
            }
        }

        private void m_solver_OnTrainingIteration(object sender, TrainingIterationArgs<T> e)
        {
            if (m_sw.Elapsed.TotalMilliseconds > 1000)
            {
                m_dfLastLoss = e.SmoothedLoss;
                m_dfLastLearningRate = e.LearningRate;
                updateStatus(e.Iteration, m_solver.MaximumIteration, e.Accuracy, e.SmoothedLoss, e.LearningRate);
                m_sw.Restart();
            }
        }

        private void m_solver_OnTestingIteration(object sender, TestingIterationArgs<T> e)
        {
            if (m_sw.Elapsed.TotalMilliseconds > 1000)
            {
                updateStatus(e.Iteration, m_solver.MaximumIteration, e.Accuracy, m_dfLastLoss, m_dfLastLearningRate);
                m_sw.Restart();
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

        public void Dispose()
        {
            if (m_dataPool != null)
            {
                m_dataPool.Shutdown();
                m_dataPool = null;
            }
        }

        private void updateStatus(int nIteration, int nMaxIteration, double dfAccuracy, double dfLoss, double dfLearningRate)
        {
            GetStatusArgs args = new GetStatusArgs(0, nIteration, nMaxIteration, dfAccuracy, 0, 0, dfLoss, dfLearningRate);
            m_icallback.OnUpdateStatus(args);
        }

        public GetDataArgs getDataArgs(Phase phase, int nIdx, int nAction, bool bGetLabel = false, int nBatchSize = 1)
        {
            bool bReset = (nAction == -1) ? true : false;
            return new GetDataArgs(phase, nIdx, m_mycaffe, m_mycaffe.Log, m_mycaffe.CancelEvent, bReset, nAction, false, bGetLabel, (nBatchSize > 1) ? true : false);
        }

        public Log Log
        {
            get { return m_mycaffe.Log; }
        }

        public CancelEvent Cancel
        {
            get { return m_mycaffe.CancelEvent; }
        }

        private void getRawData(StateBase s)
        {
            int nTestLen = (int)(s.Data.ItemCount * 0.2);
            int nTrainLen = s.Data.ItemCount - nTestLen;

            if (s.Data.IsRealData)
            {
                m_bIsDataReal = true;
                m_rgdfTestData = new double[nTestLen];
                m_rgdfTrainData = new double[nTrainLen];

                Array.Copy(s.Data.RealData, 0, m_rgdfTrainData, 0, nTrainLen);
                Array.Copy(s.Data.RealData, nTrainLen, m_rgdfTestData, 0, nTestLen);
            }
            else
            {
                m_bIsDataReal = false;
                m_rgTestData = new byte[nTestLen];
                m_rgTrainData = new byte[nTrainLen];

                Array.Copy(s.Data.ByteData, 0, m_rgTrainData, 0, nTrainLen);
                Array.Copy(s.Data.ByteData, nTrainLen, m_rgTestData, 0, nTestLen);
            }
        }

        public void Test(StateBase s, int nIterations)
        {
            if (nIterations <= 0)
            {
                nIterations = 20;

                if (m_solver.parameter.test_iter.Count > 0)
                    nIterations = m_solver.parameter.test_iter[0];
            }

            getRawData(s);
            m_sw.Start();
            m_solver.TestAll(nIterations);
        }

        private void m_solver_OnTestStart(object sender, EventArgs e)
        {
            FeedNet(false);
        }

        public void Train(StateBase s, int nIterations, TRAIN_STEP step)
        {
            if (nIterations <= 0)
                nIterations = m_solver.parameter.max_iter;

            getRawData(s);
            m_sw.Start();
            m_solver.Solve(nIterations, null, null, step);
        }

        private void m_solver_OnStart(object sender, EventArgs e)
        {
            FeedNet(true);
        }

        public void FeedNet(bool bTrain)
        {
            bool bFound;
            int nIdx;
            Phase phase = (bTrain) ? Phase.TRAIN : Phase.TEST;

            // Real Data 
            if (m_bIsDataReal)
            {
                if (m_bUsePreloadData)
                {
                    double[] rgdfData = (bTrain) ? m_rgdfTrainData : m_rgdfTestData;

                    // Re-order the data according to caffe input specification for LSTM layer.
                    for (int i = 0; i < m_nBatchSize; i++)
                    {
                        int nCurrentValIdx = m_random.Next(rgdfData.Length - m_nSequenceLength - 1);

                        for (int j = 0; j < m_nSequenceLength; j++)
                        {
                            // Feed the net with input data and labels (clips are always the same)
                            double dfData = rgdfData[nCurrentValIdx + j];
                            // Labels are the same with an offset of +1
                            double dfLabel = rgdfData[nCurrentValIdx + j + 1]; // predict next value
                            float fDataIdx = findIndex(dfData, out bFound);
                            float fLabelIdx = findIndex(dfLabel, out bFound);

                            // LSTM: Create input data, the data must be in the order
                            // seq1_val1, seq2_val1, ..., seqBatch_Size_val1, seq1_val2, seq2_val2, ..., seqBatch_Size_valSequence_Length
                            if (m_lstmType == LayerParameter.LayerType.LSTM)
                                nIdx = m_nBatchSize * j + i;

                            // LSTM_SIMPLE: Create input data, the data must be in the order
                            // seq1_val1, seq1_val2, ..., seq1_valBatchSize, seq2_val1, seq2_val2, ..., seqSequenceLength_valBatchSize
                            else
                                nIdx = i * m_nBatchSize + j;

                            m_rgDataInput[nIdx] = (T)Convert.ChangeType(fDataIdx, typeof(T));

                            if (m_nSequenceLengthLabel == (m_nSequenceLength * m_nBatchSize) || j == m_nSequenceLength - 1)
                                m_rgLabelInput[nIdx] = (T)Convert.ChangeType(fLabelIdx, typeof(T));
                        }
                    }

                    m_blobData.mutable_cpu_data = m_rgDataInput;
                    m_blobLabel.mutable_cpu_data = m_rgLabelInput;
                }
                else
                {
                    m_mycaffe.Log.CHECK_EQ(m_nBatchSize, m_nThreads, "The 'Threads' setting of " + m_nThreads.ToString() + " must match the batch size = " + m_nBatchSize.ToString() + "!");

                    List<GetDataArgs> rgDataArgs = new List<GetDataArgs>();

                    if (m_nBatchSize == 1)
                    {
                        GetDataArgs e = getDataArgs(phase, 0, 0, true, m_nBatchSize);
                        m_icallback.OnGetData(e);
                        rgDataArgs.Add(e);
                    }
                    else
                    {
                        for (int i = 0; i < m_nBatchSize; i++)
                        {
                            rgDataArgs.Add(getDataArgs(phase, i, 0, true, m_nBatchSize));
                        }

                        if (!m_dataPool.Run(rgDataArgs))
                            m_mycaffe.Log.FAIL("Data Time Out - Failed to collect all data to build the RNN batch!");
                    }

                    double[] rgData = rgDataArgs[0].State.Data.RealData;
                    double[] rgLabel = rgDataArgs[0].State.Label.RealData;
                    double[] rgClip = rgDataArgs[0].State.Clip.RealData;

                    int nDataLen = rgData.Length;
                    int nLabelLen = rgLabel.Length;
                    int nClipLen = rgClip.Length;
                    int nDataItem = nDataLen / nLabelLen;

                    if (m_nBatchSize > 1)
                    {
                        rgData = new double[nDataLen * m_nBatchSize];
                        rgLabel = new double[nLabelLen * m_nBatchSize];
                        rgClip = new double[nClipLen * m_nBatchSize];

                        for (int i = 0; i < m_nBatchSize; i++)
                        {
                            for (int j = 0; j < m_nSequenceLength; j++)
                            {
                                // LSTM: Create input data, the data must be in the order
                                // seq1_val1, seq2_val1, ..., seqBatch_Size_val1, seq1_val2, seq2_val2, ..., seqBatch_Size_valSequence_Length
                                if (m_lstmType == LayerParameter.LayerType.LSTM)
                                    nIdx = m_nBatchSize * j + i;

                                // LSTM_SIMPLE: Create input data, the data must be in the order
                                // seq1_val1, seq1_val2, ..., seq1_valBatchSize, seq2_val1, seq2_val2, ..., seqSequenceLength_valBatchSize
                                else
                                    nIdx = i * m_nBatchSize + j;

                                Array.Copy(rgDataArgs[i].State.Data.RealData, 0, rgData, nIdx * nDataItem, nDataItem);
                                rgLabel[nIdx] = rgDataArgs[i].State.Label.RealData[j];
                                rgClip[nIdx] = rgDataArgs[i].State.Clip.RealData[j];
                            }
                        }
                    }

                    string strSolverErr = "";
                    if (m_nSolverSequenceLength >= 0 && m_nSolverSequenceLength != m_nSequenceLength)
                        strSolverErr = "The solver parameter 'SequenceLength' length of " + m_nSolverSequenceLength.ToString() + " must match the model sequence length of " + m_nSequenceLength.ToString() + ".  ";

                    int nExpectedCount = m_blobData.count();
                    m_mycaffe.Log.CHECK_EQ(nExpectedCount, rgData.Length, strSolverErr + "The size of the data received ('" + rgData.Length.ToString() + "') does mot match the expected data count of '" + nExpectedCount.ToString() + "'!");
                    m_blobData.mutable_cpu_data = Utility.ConvertVec<T>(rgData);

                    nExpectedCount = m_blobLabel.count();
                    m_mycaffe.Log.CHECK_EQ(nExpectedCount, rgLabel.Length, strSolverErr + "The size of the label received ('" + rgLabel.Length.ToString() + "') does not match the expected label count of '" + nExpectedCount.ToString() + "'!");
                    m_blobLabel.mutable_cpu_data = Utility.ConvertVec<T>(rgLabel);

                    nExpectedCount = m_blobClip.count();
                    m_mycaffe.Log.CHECK_EQ(nExpectedCount, rgClip.Length, strSolverErr + "The size of the clip received ('" + rgClip.Length.ToString() + "') does not match the expected clip count of '" + nExpectedCount.ToString() + "'!");
                    m_blobClip.mutable_cpu_data = Utility.ConvertVec<T>(rgClip);
                }
            }
            // Byte Data (uses a vocabulary if available)
            else
            {
                byte[] rgData = (bTrain) ? m_rgTrainData : m_rgTestData;
                // Create input data, the data must be in the order
                // seq1_char1, seq2_char1, ..., seqBatch_Size_char1, seq1_char2, seq2_char2, ..., seqBatch_Size_charSequence_Length
                // As seq1_charSequence_Length == seq2_charSequence_Length-1 == seq3_charSequence_Length-2 == ... we can perform block copy for efficientcy.
                // Labels are the same with an offset of +1

                // Re-order the data according to caffe input specification for LSTM layer.
                for (int i = 0; i < m_nBatchSize; i++)
                {
                    int nCurrentCharIdx = m_random.Next(rgData.Length - m_nSequenceLength - 2);

                    for (int j = 0; j < m_nSequenceLength; j++)
                    {
                        // Feed the net with input data and labels (clips are always the same)
                        byte bData = rgData[nCurrentCharIdx + j];
                        // Labels are the same with an offset of +1
                        byte bLabel = rgData[nCurrentCharIdx + j + 1]; // predict next character
                        float fDataIdx = findIndex(bData, out bFound);
                        float fLabelIdx = findIndex(bLabel, out bFound);

                        // LSTM: Create input data, the data must be in the order
                        // seq1_val1, seq2_val1, ..., seqBatch_Size_val1, seq1_val2, seq2_val2, ..., seqBatch_Size_valSequence_Length
                        if (m_lstmType == LayerParameter.LayerType.LSTM)
                            nIdx = m_nBatchSize * j + i;

                        // LSTM_SIMPLE: Create input data, the data must be in the order
                        // seq1_val1, seq1_val2, ..., seq1_valBatchSize, seq2_val1, seq2_val2, ..., seqSequenceLength_valBatchSize
                        else
                            nIdx = i * m_nBatchSize + j;

                        m_rgDataInput[nIdx] = (T)Convert.ChangeType(fDataIdx, typeof(T));

                        if (m_nSequenceLengthLabel == (m_nSequenceLength * m_nBatchSize) || j == m_nSequenceLength - 1)
                            m_rgLabelInput[nIdx] = (T)Convert.ChangeType(fLabelIdx, typeof(T));
                    }
                }

                m_blobData.mutable_cpu_data = m_rgDataInput;
                m_blobLabel.mutable_cpu_data = m_rgLabelInput;
            }
        }

        private float findIndex(byte b, out bool bFound)
        {
            bFound = false;

            if (m_rgVocabulary == null || m_bDisableVocabulary)
                return b;

            bFound = true;

            return m_rgVocabulary.FindIndex(b);
        }

        private float findIndex(double df, out bool bFound)
        {
            bFound = false;

            if (m_rgVocabulary == null || m_bDisableVocabulary)
                return (float)df;

            return m_rgVocabulary.FindIndex(df);
        }

        private List<T> getInitialInput(bool bIsReal)
        {
            List<T> rgInput = new List<T>();
            float[] rgCorrectLengthSequence = new float[m_nSequenceLength];

            for (int i = 0; i < m_nSequenceLength; i++)
            {
                rgCorrectLengthSequence[i] = (int)m_random.Next(m_nVocabSize);
            }

            // If a seed is specified, add it to the end of the sequence.
            if (!bIsReal && m_runProperties != null)
            {
                string strSeed = m_runProperties.GetProperty("Seed", false);
                if (!string.IsNullOrEmpty(strSeed))
                {
                    strSeed = Utility.Replace(strSeed, '~', ';');

                    int nStart = rgCorrectLengthSequence.Length - strSeed.Length;
                    if (nStart < 0)
                        nStart = 0;

                    for (int i = nStart; i < rgCorrectLengthSequence.Length; i++)
                    {
                        char ch = strSeed[i - nStart];
                        bool bFound;
                        int nIdx = (int)findIndex((byte)ch, out bFound);

                        if (bFound)
                            rgCorrectLengthSequence[i] = nIdx;
                    }
                }
            }

            for (int i = 0; i < rgCorrectLengthSequence.Length; i++)
            {
                rgInput.Add((T)Convert.ChangeType(rgCorrectLengthSequence[i], typeof(T)));
            }

            return rgInput;
        }

        public float[] Run(int nN)
        {
            try
            {
                Stopwatch sw = new Stopwatch();
                float[] rgPredictions = new float[nN];

                sw.Start();

                m_bIsDataReal = true;

                if (m_rgVocabulary != null)
                    m_bIsDataReal = m_rgVocabulary.IsDataReal;

                m_mycaffe.Log.Enable = false;

                if (m_bIsDataReal && !m_bUsePreloadData)
                {
                    string strSolverErr = "";
                    int nLookahead = 1;
                    if (m_nSolverSequenceLength >= 0 && m_nSolverSequenceLength < m_nSequenceLength)
                        nLookahead = m_nSequenceLength - m_nSolverSequenceLength;

                    rgPredictions = new float[nN * 2 * nLookahead];

                    for (int i = 0; i < nN; i++)
                    {
                        GetDataArgs e = getDataArgs(Phase.RUN, 0, 0, true);
                        m_icallback.OnGetData(e);

                        int nExpectedCount = m_blobData.count();
                        m_mycaffe.Log.CHECK_EQ(nExpectedCount, e.State.Data.ItemCount, strSolverErr + "The size of the data received ('" + e.State.Data.ItemCount.ToString() + "') does mot match the expected data count of '" + nExpectedCount.ToString() + "'!");
                        m_blobData.mutable_cpu_data = Utility.ConvertVec<T>(e.State.Data.RealData);

                        if (m_blobLabel != null)
                        {
                            nExpectedCount = m_blobLabel.count();
                            m_mycaffe.Log.CHECK_EQ(nExpectedCount, e.State.Label.ItemCount, strSolverErr + "The size of the label received ('" + e.State.Label.ItemCount.ToString() + "') does not match the expected label count of '" + nExpectedCount.ToString() + "'!");
                            m_blobLabel.mutable_cpu_data = Utility.ConvertVec<T>(e.State.Label.RealData);
                        }

                        double dfLoss;
                        BlobCollection<T> colResults = m_net.Forward(out dfLoss);
                        Blob<T> blobOutput = colResults[0];

                        if (m_blobOutput != null)
                            blobOutput = m_blobOutput;

                        float[] rgResults = Utility.ConvertVecF<T>(blobOutput.update_cpu_data());

                        for (int j = nLookahead; j > 0; j--)
                        {
                            float fPrediction = getLastPrediction(rgResults, m_rgVocabulary, j);
                            float fActual = (float)e.State.Label.RealData[e.State.Label.RealData.Length - j];

                            int nIdx0 = ((nLookahead - j) * nN * 2);
                            int nIdx1 = nIdx0 + nN;

                            if (m_dfScale != 1.0 && m_dfScale > 0)
                                fActual /= (float)m_dfScale;

                            if (m_rgVocabulary == null || m_bDisableVocabulary)
                            {
                                if (m_dfScale != 1.0 && m_dfScale > 0)
                                    fPrediction /= (float)m_dfScale;

                                rgPredictions[nIdx0 + i] = fPrediction;
                                rgPredictions[nIdx1 + i] = fActual;
                            }
                            else
                            {
                                rgPredictions[nIdx0 + i] = (float)m_rgVocabulary.GetValueAt((int)fPrediction, true);
                                rgPredictions[nIdx1 + i] = (float)m_rgVocabulary.GetValueAt((int)fActual, true);
                            }
                        }

                        if (sw.Elapsed.TotalMilliseconds > 1000)
                        {
                            double dfPct = (double)i / (double)nN;
                            m_mycaffe.Log.Enable = true;
                            m_mycaffe.Log.Progress = dfPct;
                            m_mycaffe.Log.WriteLine("Running at " + dfPct.ToString("P") + " complete...");
                            m_mycaffe.Log.Enable = false;
                            sw.Restart();
                        }

                        if (m_mycaffe.CancelEvent.WaitOne(0))
                            break;
                    }
                }
                else
                {
                    int nIdx = 0;
                    List<T> rgInput = rgInput = getInitialInput(m_bIsDataReal);

                    for (int i = 0; i < nN; i++)
                    {
                        T[] rgInputVector = new T[m_blobData.count()];
                        for (int j = 0; j < m_nSequenceLength; j++)
                        {
                            // The batch is filled with 0 except for the first sequence which is the one we want to use for prediction.
                            nIdx = j * m_nBatchSize;
                            rgInputVector[nIdx] = rgInput[j];
                        }

                        m_blobData.mutable_cpu_data = rgInputVector;

                        double dfLoss;
                        BlobCollection<T> colResults = m_net.Forward(out dfLoss);
                        Blob<T> blobOutput = colResults[0];

                        if (m_blobOutput != null)
                            blobOutput = m_blobOutput;

                        float[] rgResults = Utility.ConvertVecF<T>(blobOutput.update_cpu_data());
                        float fPrediction = getLastPrediction(rgResults, m_rgVocabulary, 1);

                        //Add the new prediction and discard the oldest one
                        rgInput.Add((T)Convert.ChangeType(fPrediction, typeof(T)));
                        rgInput.RemoveAt(0);

                        if (m_rgVocabulary == null || m_bDisableVocabulary)
                            rgPredictions[i] = fPrediction;
                        else
                            rgPredictions[i] = (float)m_rgVocabulary.GetValueAt((int)fPrediction);

                        if (sw.Elapsed.TotalMilliseconds > 1000)
                        {
                            double dfPct = (double)i / (double)nN;
                            m_mycaffe.Log.Enable = true;
                            m_mycaffe.Log.Progress = dfPct;
                            m_mycaffe.Log.WriteLine("Running at " + dfPct.ToString("P") + " complete...");
                            m_mycaffe.Log.Enable = false;
                            sw.Restart();
                        }

                        if (m_mycaffe.CancelEvent.WaitOne(0))
                            break;
                    }
                }

                return rgPredictions;
            }
            catch (Exception excpt)
            {
                throw excpt;
            }
            finally
            {
                m_mycaffe.Log.Enable = true;
            }
        }

        private float getLastPrediction(float[] rgDataRaw, BucketCollection rgVocabulary, int nLookahead)
        {
            // Get the probabilities for the last character of the first sequence in the batch
            int nOffset = (m_nSequenceLength - nLookahead) * m_nBatchSize * m_nVocabSize;

            if (m_bDisableVocabulary)
                return rgDataRaw[nOffset];

            float[] rgData = new float[m_nVocabSize];

            for (int i = 0; i < rgData.Length; i++)
            {
                rgData[i] = rgDataRaw[nOffset + i];
            }

            return getLastPrediction(rgData, rgVocabulary);
        }

        private int getLastPrediction(float[] rgData, BucketCollection rgVocabulary)
        {
            int nIdx = m_nVocabSize - 1;

            // If no temperature, return directly the character with the best score
            if (m_dfTemperature == 0)
            {
                nIdx = ArgMax(rgData, 0, m_nVocabSize);
            }
            else
            {
                // Otherwise, compute the probabilities with the temperature and select the character according to the probabilities.
                double[] rgAccumulatedProba = new double[m_nVocabSize];
                double[] rgProba = new double[m_nVocabSize];
                double dfExpoSum = 0;

                for (int i = 0; i < m_nVocabSize; i++)
                {
                    // The max value is subtracted for numerical stability
                    rgProba[i] = Math.Exp((rgData[i] - (m_nVocabSize - 1)) / m_dfTemperature);
                    dfExpoSum += rgProba[i];
                }

                rgProba[0] /= dfExpoSum;
                rgAccumulatedProba[0] = rgProba[0];

                double dfRandom = m_random.NextDouble();

                for (int i = 1; i < rgProba.Length; i++)
                {
                    // Return the first index for which the accumulated probability is bigger than the random number.
                    if (rgAccumulatedProba[i - 1] > dfRandom)
                        return i - 1;

                    rgProba[i] /= dfExpoSum;
                    rgAccumulatedProba[i] = rgAccumulatedProba[i - 1] + rgProba[i];
                }
            }

            if (nIdx < 0 || nIdx > rgVocabulary.Count)
                throw new Exception("Invalid index - out of the vocabulary range of [0," + rgVocabulary.Count.ToString() + "]");

            return nIdx;
        }

        private int ArgMax(float[] rg, int nOffset, int nCount)
        {
            if (nCount == 0)
                return -1;

            int nMaxIdx = nOffset;
            float fMax = rg[nOffset];

            for (int i = nOffset; i < nOffset + nCount; i++)
            {
                if (rg[i] > fMax)
                {
                    nMaxIdx = i;
                    fMax = rg[i];
                }
            }

            return nMaxIdx - nOffset;
        }
    }

    class DataCollectionPool /** @private */
    {
        List<DataCollector> m_rgCollectors = new List<DataCollector>();

        public DataCollectionPool()
        {
        }

        public void Initialize(int nThreads, IxTrainerCallback icallback)
        {
            for (int i = 0; i < nThreads; i++)
            {
                m_rgCollectors.Add(new DataCollector(icallback));
            }
        }

        public void Shutdown()
        {
            foreach (DataCollector col in m_rgCollectors)
            {
                col.CleanUp();
            }
        }

        public bool Run(List<GetDataArgs> rgStartup)
        {
            List<ManualResetEvent> rgWait = new List<ManualResetEvent>();

            if (rgStartup.Count != m_rgCollectors.Count)
                throw new Exception("The startup count does not match the collector count.");

            for (int i = 0; i < rgStartup.Count; i++)
            {
                rgWait.Add(rgStartup[i].DataReady);
                m_rgCollectors[i].Run(rgStartup[i]);
            }

            return WaitHandle.WaitAll(rgWait.ToArray(), 10000);
        }
    }

    class DataCollector /** @private */
    {
        ManualResetEvent m_evtAbort = new ManualResetEvent(false);
        AutoResetEvent m_evtRun = new AutoResetEvent(false);
        Thread m_thread;
        GetDataArgs m_args;
        IxTrainerCallback m_icallback;

        public DataCollector(IxTrainerCallback icallback)
        {
            m_icallback = icallback;
            m_thread = new Thread(new ThreadStart(doWork));
            m_thread.Start();
        }

        public void CleanUp()
        {
            m_evtAbort.Set();          
        }

        public void Run(GetDataArgs args)
        {
            m_args = args;
            m_evtRun.Set();
        }

        private void doWork()
        {
            bool bDone = false;
            List<WaitHandle> rgWait = new List<WaitHandle>();
            rgWait.Add(m_evtAbort);
            rgWait.Add(m_evtRun);

            while (!bDone)
            {
                int nWait = WaitHandle.WaitAny(rgWait.ToArray());
                if (nWait == 0)
                    return;

                m_icallback.OnGetData(m_args);
                m_args.DataReady.Set();
            }
        }
    }
}
