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
        List<int> m_rgVocabulary = null;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="mycaffe">Specifies the MyCaffeControl to use for learning and prediction.</param>
        /// <param name="properties">Specifies the property set containing the key/value pairs of property settings.</param>
        /// <param name="random">Specifies the random number generator to use.</param>
        /// <param name="icallback">Specifies the callback for parent notifications and queries.</param>
        /// <param name="rgVocabulary">Specifies the vocabulary to use.</param>
        public TrainerRNN(MyCaffeControl<T> mycaffe, PropertySet properties, CryptoRandom random, IxTrainerCallback icallback, List<int> rgVocabulary)
        {
            m_icallback = icallback;
            m_mycaffe = mycaffe;
            m_properties = properties;
            m_random = random;
            m_rgVocabulary = rgVocabulary;
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
            Agent<T> agent = new Agent<T>(m_icallback, m_mycaffe, m_properties, m_random, Phase.RUN, m_rgVocabulary, strRunProperties);
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
        public byte[] Run(int nN, string strRunProperties, out Type type)
        {
            m_mycaffe.CancelEvent.Reset();
            Agent<T> agent = new Agent<T>(m_icallback, m_mycaffe, m_properties, m_random, Phase.RUN, m_rgVocabulary, strRunProperties);
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
            Agent<T> agent = new Agent<T>(m_icallback, m_mycaffe, m_properties, m_random, Phase.TEST, m_rgVocabulary);
            agent.Run(Phase.TEST, nIterations);

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
            if (step != TRAIN_STEP.NONE)
                throw new Exception("The simple traininer does not support stepping.");

            m_mycaffe.CancelEvent.Reset();
            Agent<T> agent = new Agent<T>(m_icallback, m_mycaffe, m_properties, m_random, Phase.TRAIN, m_rgVocabulary);
            agent.Run(Phase.TRAIN, nIterations);

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

        public Agent(IxTrainerCallback icallback, MyCaffeControl<T> mycaffe, PropertySet properties, CryptoRandom random, Phase phase, List<int> rgVocabulary, string strRunProperties = null)
        {
            m_icallback = icallback;
            m_brain = new Brain<T>(mycaffe, properties, random, icallback as IxTrainerCallbackRNN, phase, rgVocabulary, strRunProperties);
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

        private StateBase getData(int nAction)
        {
            GetDataArgs args = m_brain.getDataArgs(nAction);
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
        /// <returns>The vocabulary built up during training and testing is returned.</returns>
        public void Run(Phase phase, int nIterations)
        {
            StateBase s = getData(-1);

            while (!m_brain.Cancel.WaitOne(0) && !s.Done)
            {
                if (phase == Phase.TEST)
                    m_brain.Test(s, nIterations);
                else if (phase == Phase.TRAIN)
                    m_brain.Train(s, nIterations, TRAIN_STEP.NONE);

                s = getData(1);
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
        public byte[] Run(int nN, out Type type)
        {
            float[] rgResults = m_brain.Run(nN);

            ConvertOutputArgs args = new ConvertOutputArgs(rgResults);
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
        int m_nBatchSize;
        int m_nVocabSize;
        CryptoRandom m_random;
        T[] m_rgDataInput;
        T[] m_rgLabelInput;
        T m_tZero;
        T m_tOne;
        double m_dfTemperature = 0;
        byte[] m_rgTestData;
        byte[] m_rgTrainData;
        Stopwatch m_sw = new Stopwatch();
        double m_dfLastLoss = 0;
        double m_dfLastLearningRate = 0;
        List<int> m_rgVocabulary = null;
        Phase m_phaseOnRun = Phase.NONE;

        public Brain(MyCaffeControl<T> mycaffe, PropertySet properties, CryptoRandom random, IxTrainerCallbackRNN icallback, Phase phase, List<int> rgVocabulary, string strRunProperties = null)
        {
            string strOutputBlob = null;

            if (strRunProperties != null)
                m_runProperties = new PropertySet(strRunProperties);

            m_icallback = icallback;
            m_mycaffe = mycaffe;
            m_properties = properties;
            m_random = random;
            m_rgVocabulary = rgVocabulary;

            if (m_runProperties != null)
            {
                m_dfTemperature = m_runProperties.GetPropertyAsDouble("Temperature", 0);
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

            m_nVocabSize = (int)layer.layer_param.inner_product_param.num_output;
            if (rgVocabulary != null && rgVocabulary.Count > 0)
                m_mycaffe.Log.CHECK_EQ(m_nVocabSize, rgVocabulary.Count, "The vocabulary count and last inner product output count should match!");

            m_nSequenceLength = m_blobData.shape(0);
            m_nBatchSize = m_blobData.shape(1);

            m_mycaffe.Log.CHECK_EQ(m_blobData.count(), m_blobClip.count(), "The data and clip blobs must have the same count!");

            m_rgDataInput = new T[m_nSequenceLength * m_nBatchSize];

            T[] rgClipInput = new T[m_nSequenceLength * m_nBatchSize];
            m_tZero = (T)Convert.ChangeType(0, typeof(T));
            m_tOne = (T)Convert.ChangeType(1, typeof(T));

            for (int i = 0; i < rgClipInput.Length; i++)
            {
                rgClipInput[i] = (i < m_nBatchSize) ? m_tZero : m_tOne;
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

                m_mycaffe.Log.CHECK_EQ(m_blobData.count(), m_blobLabel.count(), "The data and label blobs must have the same count!");

                m_rgLabelInput = new T[m_nSequenceLength * m_nBatchSize];
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
        }

        private void updateStatus(int nIteration, int nMaxIteration, double dfAccuracy, double dfLoss, double dfLearningRate)
        {
            GetStatusArgs args = new GetStatusArgs(0, nIteration, nMaxIteration, dfAccuracy, 0, 0, dfLoss, dfLearningRate);
            m_icallback.OnUpdateStatus(args);
        }

        public GetDataArgs getDataArgs(int nAction)
        {
            bool bReset = (nAction == -1) ? true : false;
            return new GetDataArgs(0, m_mycaffe, m_mycaffe.Log, m_mycaffe.CancelEvent, bReset, nAction, false);
        }

        public Log Log
        {
            get { return m_mycaffe.Log; }
        }

        public CancelEvent Cancel
        {
            get { return m_mycaffe.CancelEvent; }
        }

        public void Test(StateBase s, int nIterations)
        {
            if (nIterations <= 0)
            {
                nIterations = 20;

                if (m_solver.parameter.test_iter.Count > 0)
                    nIterations = m_solver.parameter.test_iter[0];
            }

            int nTestLen = (int)(s.Data.ItemCount * 0.2);
            int nTrainLen = s.Data.ItemCount - nTestLen;
            m_rgTestData = new byte[nTestLen];
            Array.Copy(s.Data.ByteData, nTrainLen, m_rgTestData, 0, nTestLen);

            m_sw.Start();
            m_solver.TestAll(nIterations);
        }

        private void m_solver_OnTestStart(object sender, EventArgs e)
        {
            FeedNet(false);
        }

        public void Train(StateBase s, int nIterations, TRAIN_STEP step)
        {
            int nTestLen = (int)(s.Data.ItemCount * 0.2);
            int nTrainLen = s.Data.ItemCount - nTestLen;
            m_rgTestData = new byte[nTestLen];
            m_rgTrainData = new byte[nTrainLen];

            Array.Copy(s.Data.ByteData, 0, m_rgTrainData, 0, nTrainLen);
            Array.Copy(s.Data.ByteData, nTrainLen, m_rgTestData, 0, nTestLen);

            m_sw.Start();
            m_solver.Solve(nIterations, null, null, step);
        }

        private void m_solver_OnStart(object sender, EventArgs e)
        {
            FeedNet(true);
        }

        public void FeedNet(bool bTrain)
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
                    byte bData = rgData[nCurrentCharIdx + j];
                    byte bLabel = rgData[nCurrentCharIdx + j + 1]; // predict next character
                    bool bFound;
                    float fDataIdx = findIndex(bData, out bFound);
                    float fLabelIdx = findIndex(bLabel, out bFound);

                    // Feed the net with input data and labels (clips are always the same)
                    int nIdx = m_nBatchSize * j + i;
                    m_rgDataInput[nIdx] = (T)Convert.ChangeType(fDataIdx, typeof(T));
                    m_rgLabelInput[nIdx] = (T)Convert.ChangeType(fLabelIdx, typeof(T));
                }
            }

            m_blobData.mutable_cpu_data = m_rgDataInput;
            m_blobLabel.mutable_cpu_data = m_rgLabelInput;
        }

        private float findIndex(byte b, out bool bFound)
        {
            bFound = false;

            if (m_rgVocabulary == null)
                return b;

            for (int i = 0; i < m_rgVocabulary.Count; i++)
            {
                if ((int)b == m_rgVocabulary[i])
                {
                    bFound = true;
                    return i;
                }
            }

            return m_rgVocabulary.Count - 1;
        }

        public float[] Run(int nN)
        {
            float[] rgPredictions = new float[nN];
            List<int> rgInput = new List<int>();
            Stopwatch sw = new Stopwatch();
            int[] rgCorrectLengthSequence = new int[m_nSequenceLength];

            // Set the initialization sequence to random characters.
            for (int i = 0; i < m_nSequenceLength; i++)
            {
                rgCorrectLengthSequence[i] = (int)m_random.Next(m_nVocabSize);
            }

            // If a seed is specified, add it to the end of the sequence.
            if (m_runProperties != null)
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
                rgInput.Add(rgCorrectLengthSequence[i]);
            }

            sw.Start();

            for (int i = 0; i < nN; i++)
            {
                // The batch is filled with 0 except for the first sequence which is the one we want to use for prediction.
                T[] rgInputVector = new T[m_nSequenceLength * m_nBatchSize];
                for (int j = 0; j < m_nSequenceLength; j++)
                {
                    rgInputVector[j * m_nBatchSize] = (T)Convert.ChangeType(rgInput[j], typeof(T));
                }

                m_blobData.mutable_cpu_data = rgInputVector;

                double dfLoss;
                BlobCollection<T> colResults = m_net.Forward(out dfLoss);
                int nPrediction = getLastPrediction(colResults[0], m_rgVocabulary);

                //Add the new prediction and discard the oldest one
                rgInput.Add(nPrediction);
                rgInput.RemoveAt(0);

                rgPredictions[i] = (m_rgVocabulary != null) ? m_rgVocabulary[nPrediction] : nPrediction;

                if (sw.Elapsed.TotalMilliseconds > 1000)
                {
                    double dfPct = (double)i / (double)nN;
                    m_mycaffe.Log.Progress = dfPct;
                    m_mycaffe.Log.WriteLine("Running at " + dfPct.ToString("P") + " complete...");
                    sw.Restart();
                }

                if (m_mycaffe.CancelEvent.WaitOne(0))
                    break;
            }

            return rgPredictions;
        }

        private int getLastPrediction(Blob<T> blobOutput, List<int> rgVocabulary)
        {
            if (m_blobOutput != null)
                blobOutput = m_blobOutput;

            float[] rgData = Utility.ConvertVecF<T>(blobOutput.update_cpu_data());

            // Get the probabilities for the last character of the first sequence in the batch
            int nOffset = (m_nSequenceLength - 1) * m_nBatchSize * m_nVocabSize;
            int nIdx = m_nVocabSize - 1;

            // If no temperature, return directly the character with the best score
            if (m_dfTemperature == 0)
                return ArgMax(rgData, nOffset, m_nVocabSize);

            // Otherwise, compute the probabilities with the temperature and select the character according to the probabilities.
            double[] rgAccumulatedProba = new double[m_nVocabSize];
            double[] rgProba = new double[m_nVocabSize];
            double dfExpoSum = 0;

            for (int i = 0; i < m_nVocabSize; i++)
            {
                // The max value is subtracted for numerical stability
                rgProba[i] = Math.Exp((rgData[nOffset + i] - (m_nVocabSize - 1)) / m_dfTemperature);
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

            // If we are here, its the last character.
            return nIdx;
        }

        private int ArgMax(float[] rg, int nOffset, int nCount)
        {
            if (nCount == 0)
                return -1;

            int nMaxIdx = 0;
            float fMax = rg[0];

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
}
