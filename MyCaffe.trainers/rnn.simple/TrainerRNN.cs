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

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="mycaffe">Specifies the MyCaffeControl to use for learning and prediction.</param>
        /// <param name="properties">Specifies the property set containing the key/value pairs of property settings.</param>
        /// <param name="random">Specifies the random number generator to use.</param>
        /// <param name="icallback">Specifies the callback for parent notifications and queries.</param>
        public TrainerRNN(MyCaffeControl<T> mycaffe, PropertySet properties, CryptoRandom random, IxTrainerCallback icallback)
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
        /// <param name="nN">specifies the number of samples to run.</param>
        /// <returns>The results of the run containing the action are returned.</returns>
        public float[] Run(int nN)
        {
            m_mycaffe.CancelEvent.Reset();
            Agent<T> agent = new Agent<T>(m_icallback, m_mycaffe, m_properties, m_random, Phase.RUN);
            float[] rgResults = agent.Run(nN);
            agent.Dispose();

            return rgResults;
        }

        /// <summary>
        /// <summary>
        /// Run a single cycle on the environment after the delay.
        /// </summary>
        /// <param name="nN">Specifies the number of samples to run.</param>
        /// <param name="type">Returns the data type contained in the byte stream.</param>
        /// <returns>The results of the run containing the action are returned as a byte stream.</returns>
        public byte[] Run(int nN, out Type type)
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
        /// <param name="nIterations">Specifies the number of iterations to run.</param>
        /// <returns>A value of <i>true</i> is returned when handled, <i>false</i> otherwise.</returns>
        public bool Test(int nIterations)
        {
            int nDelay = 1000;

            m_mycaffe.CancelEvent.Reset();
            Agent<T> agent = new Agent<T>(m_icallback, m_mycaffe, m_properties, m_random, Phase.TEST);
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
            Agent<T> agent = new Agent<T>(m_icallback, m_mycaffe, m_properties, m_random, Phase.TRAIN);
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

        public Agent(IxTrainerCallback icallback, MyCaffeControl<T> mycaffe, PropertySet properties, CryptoRandom random, Phase phase)
        {
            m_icallback = icallback;
            m_brain = new Brain<T>(mycaffe, properties, random, icallback as IxTrainerCallbackRNN, phase);
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
        Blob<T> m_blobData;
        Blob<T> m_blobClip;
        Blob<T> m_blobLabel;
        int m_nSequenceLength;
        int m_nBatchSize;
        int m_nVocabSize;
        CryptoRandom m_random;
        T[] m_rgDataInput;
        T[] m_rgLabelInput;
        T m_tZero;
        T m_tOne;
        double m_dfTemperature = 0;
        List<int> m_rgCorrectLengthSequence = null;
        byte[] m_rgTestData;
        byte[] m_rgTrainData;
        string m_strSeed = null;
        Stopwatch m_sw = new Stopwatch();
        double m_dfLastLoss = 0;
        double m_dfLastLearningRate = 0;

        public Brain(MyCaffeControl<T> mycaffe, PropertySet properties, CryptoRandom random, IxTrainerCallbackRNN icallback, Phase phase)
        {
            m_icallback = icallback;
            m_mycaffe = mycaffe;
            m_net = mycaffe.GetInternalNet(phase);         
            m_properties = properties;
            m_random = random;

            m_dfTemperature = m_properties.GetPropertyAsDouble("Temperature", 0);
            m_strSeed = Utility.Replace(m_properties.GetProperty("Seed", false), "[sp]", ' ');

            if ((m_blobData = m_net.FindBlob("data")) == null)
                throw new Exception("Could not find the 'Input' layer top named 'data'!");

            if ((m_blobClip = m_net.FindBlob("clip")) == null)
                throw new Exception("Could not find the 'Input' layer top named 'clip'!");

            Layer<T> layer = m_net.FindLastLayer(LayerParameter.LayerType.INNERPRODUCT);
            m_mycaffe.Log.CHECK(layer != null, "Could not find an ending INNERPRODUCT layer!");

            m_nVocabSize = (int)layer.layer_param.inner_product_param.num_output;
            m_nSequenceLength = m_blobData.shape(0);
            m_nBatchSize = m_blobData.shape(1);

            m_mycaffe.Log.CHECK(m_nVocabSize == 128 || m_nVocabSize == 256, "The vocab size (EMBED input_dim) must be 128 or 256 for the ASCII characters.");
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
            else
            {
                m_rgCorrectLengthSequence = new List<int>();

                for (int i = 0; i < m_nSequenceLength; i++)
                {
                    m_rgCorrectLengthSequence.Add((int)m_random.Next(m_nVocabSize));
                }
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

                    if (bData > m_nVocabSize || bLabel > m_nVocabSize)
                    {
                        bData = 0;
                        bLabel = 0;
                    }

                    // Feed the net with input data and labels (clips are always the same)
                    int nIdx = m_nBatchSize * j + i;
                    m_rgDataInput[nIdx] = (T)Convert.ChangeType(bData, typeof(T));
                    m_rgLabelInput[nIdx] = (T)Convert.ChangeType(bLabel, typeof(T));
                }
            }

            m_blobData.mutable_cpu_data = m_rgDataInput;
            m_blobLabel.mutable_cpu_data = m_rgLabelInput;
        }

        public float[] Run(int nN)
        {
            float[] rgPredictions = new float[nN];
            List<int> rgInput = new List<int>();
            Stopwatch sw = new Stopwatch();

            for (int i = 0; i < m_rgCorrectLengthSequence.Count; i++)
            {
                rgInput.Add(m_rgCorrectLengthSequence[i]);
            }

            if (m_strSeed != null)
            {
                int nStart = m_rgCorrectLengthSequence.Count - m_strSeed.Length;
                if (nStart < 0)
                    nStart = 0;

                for (int i = nStart; i < m_rgCorrectLengthSequence.Count; i++)
                {
                    m_rgCorrectLengthSequence[i] = m_strSeed[i - nStart];
                }
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
                int nPrediction = getLastPrediction(colResults[0]);

                //Add the new prediction and discard the oldest one
                rgInput.Add(nPrediction);
                rgInput.RemoveAt(0);

                rgPredictions[i] = nPrediction;

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

        private int getLastPrediction(Blob<T> blobOutput)
        {
            float[] rgData = Utility.ConvertVecF<T>(blobOutput.update_cpu_data());

            // Get the probabilities for the last character of the first sequence in the batch
            int nOffset = (m_nSequenceLength - 1) * m_nBatchSize * m_nVocabSize;

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
            return m_nVocabSize - 1;
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
