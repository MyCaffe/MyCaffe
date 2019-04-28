using System;
using System.Collections.Generic;
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

namespace MyCaffe.trainers.pg.st
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
        public TrainerPG(MyCaffeControl<T> mycaffe, PropertySet properties, CryptoRandom random, IxTrainerCallback icallback)
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
        /// <param name="nDelay">Specifies a delay to wait before running.</param>
        /// <returns>The results of the run containing the action are returned.</returns>
        public ResultCollection Run(int nDelay = 1000)
        {
            m_mycaffe.CancelEvent.Reset();
            Agent<T> agent = new Agent<T>(m_icallback, m_mycaffe, m_properties, m_random, Phase.TRAIN);
            agent.Run(Phase.TEST, 1);
            agent.Dispose();
            return null;
        }

        /// <summary>
        /// Run the test cycle - currently this is not implemented.
        /// </summary>
        /// <param name="nIterations">Specifies the number of iterations to run.</param>
        /// <returns>A value of <i>true</i> is returned when handled, <i>false</i> otherwise.</returns>
        public bool Test(int nIterations)
        {
            int nDelay = 1000;
            string strProp = m_properties.ToString();

            // Turn off the num-skip to run at normal speed.
            strProp += "EnableNumSkip=False;";
            PropertySet properties = new PropertySet(strProp);

            m_mycaffe.CancelEvent.Reset();
            Agent<T> agent = new Agent<T>(m_icallback, m_mycaffe, properties, m_random, Phase.TRAIN);
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
                throw new Exception("The single threaded traininer does not support stepping - use the 'PG.MT' trainer instead.");

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
        float m_fGamma;
        bool m_bAllowDiscountReset = false;
        bool m_bUseRawInput = false;

        public Agent(IxTrainerCallback icallback, MyCaffeControl<T> mycaffe, PropertySet properties, CryptoRandom random, Phase phase)
        {
            m_icallback = icallback;
            m_brain = new Brain<T>(mycaffe, properties, random, phase);
            m_properties = properties;
            m_random = random;

            m_fGamma = (float)properties.GetPropertyAsDouble("Gamma", 0.99);
            m_bAllowDiscountReset = properties.GetPropertyAsBool("AllowDiscountReset", false);
            m_bUseRawInput = properties.GetPropertyAsBool("UseRawInput", false);
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

        private void updateStatus(int nEpisodeCount, double dfRewardSum, double dfRunningReward)
        {
            GetStatusArgs args = new GetStatusArgs(0, nEpisodeCount, 1000000, dfRunningReward, 0, 0, 0, 0);
            m_icallback.OnUpdateStatus(args);
        }

        /// <summary>
        /// The Run method provides the main 'actor' loop that performs the following steps:
        /// 1.) get state
        /// 2.) build experience
        /// 3.) create policy gradients
        /// 4.) train on experiences
        /// </summary>
        /// <param name="phase">Specifies the phae.</param>
        /// <param name="nIterations">Specifies the number of iterations to run.</param>
        public void Run(Phase phase, int nIterations)
        {
            MemoryCollection m_rgMemory = new MemoryCollection();
            double? dfRunningReward = null;
            double dfRewardSum = 0;
            int nEpisodeNumber = 0;
            int nIteration = 0;

            StateBase s = getData(-1);
          
            while (!m_brain.Cancel.WaitOne(0) && (nIterations == -1 || nIteration < nIterations))
            {
                // Preprocess the observation.
                SimpleDatum x = m_brain.Preprocess(s, m_bUseRawInput);

                // Forward the policy network and sample an action.
                float[] rgfAprob;
                int action = m_brain.act(x, s.Clip, out rgfAprob);

                // Take the next step using the action
                StateBase s_ = getData(action);
                dfRewardSum += s_.Reward;

                if (phase == Phase.TRAIN)
                {
                    // Build up episode memory, using reward for taking the action.
                    m_rgMemory.Add(new MemoryItem(s, x, action, rgfAprob, (float)s_.Reward));

                    // An episode has finished.
                    if (s_.Done)
                    {
                        nEpisodeNumber++;
                        nIteration++;

                        m_brain.Reshape(m_rgMemory);

                        // Compute the discounted reward (backwards through time)
                        float[] rgDiscountedR = m_rgMemory.GetDiscountedRewards(m_fGamma, m_bAllowDiscountReset);
                        // Rewards are standardized when set to be unit normal (helps control the gradient estimator variance)
                        m_brain.SetDiscountedR(rgDiscountedR);

                        // Get the action probabilities.
                        float[] rgfAprobSet = m_rgMemory.GetActionProbabilities();
                        // The action probabilities are used to calculate the initial gradient within the loss function.
                        m_brain.SetActionProbabilities(rgfAprobSet);

                        // Get the action one-hot vectors.  When using Softmax, this contains the one-hot vector containing
                        // each action set (e.g. 3 actions with action 0 set would return a vector <1,0,0>).  
                        // When using a binary probability (e.g. with Sigmoid), the each action set only contains a
                        // single element which is set to the action value itself (e.g. 0 for action '0' and 1 for action '1')
                        float[] rgfAonehotSet = m_rgMemory.GetActionOneHotVectors();
                        m_brain.SetActionOneHotVectors(rgfAonehotSet);

                        // Train for one iteration, which triggers the loss function.
                        List<Datum> rgData = m_rgMemory.GetData();
                        List<Datum> rgClip = m_rgMemory.GetClip();
                        m_brain.SetData(rgData, rgClip);
                        m_brain.Train(nEpisodeNumber);

                        // Update reward running
                        if (!dfRunningReward.HasValue)
                            dfRunningReward = dfRewardSum;
                        else
                            dfRunningReward = dfRunningReward.Value * 0.99 + dfRewardSum * 0.01;

                        updateStatus(nEpisodeNumber, dfRewardSum, dfRunningReward.Value);
                        dfRewardSum = 0;

                        s = getData(-1);
                        m_rgMemory.Clear();
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
                        nEpisodeNumber++;

                        // Update reward running
                        if (!dfRunningReward.HasValue)
                            dfRunningReward = dfRewardSum;
                        else
                            dfRunningReward = dfRunningReward.Value * 0.99 + dfRewardSum * 0.01;

                        updateStatus(nEpisodeNumber, dfRewardSum, dfRunningReward.Value);
                        dfRewardSum = 0;

                        s = getData(-1);
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

    class Brain<T> : IDisposable /** @private */
    {
        MyCaffeControl<T> m_mycaffe;
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
        int m_nRecurrentSequenceLength = 0;
        List<Datum> m_rgData = null;
        List<Datum> m_rgClip = null;

        public Brain(MyCaffeControl<T> mycaffe, PropertySet properties, CryptoRandom random, Phase phase)
        {
            m_mycaffe = mycaffe;
            m_net = mycaffe.GetInternalNet(phase);
            m_solver = mycaffe.GetInternalSolver();
            m_properties = properties;
            m_random = random;

            m_memData = m_net.FindLayer(LayerParameter.LayerType.MEMORYDATA, null) as MemoryDataLayer<T>;
            m_memLoss = m_net.FindLayer(LayerParameter.LayerType.MEMORY_LOSS, null) as MemoryLossLayer<T>;
            m_softmax = m_net.FindLayer(LayerParameter.LayerType.SOFTMAX, null) as SoftmaxLayer<T>;

            if (m_memData == null)
                throw new Exception("Could not find the MemoryData Layer!");

            if (m_memLoss == null)
                throw new Exception("Could not find the MemoryLoss Layer!");

            m_memData.OnDataPack += memData_OnDataPack;
            m_memLoss.OnGetLoss += memLoss_OnGetLoss;

            m_blobDiscountedR = new Blob<T>(mycaffe.Cuda, mycaffe.Log);
            m_blobPolicyGradient = new Blob<T>(mycaffe.Cuda, mycaffe.Log);
            m_blobActionOneHot = new Blob<T>(mycaffe.Cuda, mycaffe.Log);
            m_blobDiscountedR1 = new Blob<T>(mycaffe.Cuda, mycaffe.Log);
            m_blobPolicyGradient1 = new Blob<T>(mycaffe.Cuda, mycaffe.Log);
            m_blobActionOneHot1 = new Blob<T>(mycaffe.Cuda, mycaffe.Log);
            m_blobLoss = new Blob<T>(mycaffe.Cuda, mycaffe.Log);
            m_blobAprobLogit = new Blob<T>(mycaffe.Cuda, mycaffe.Log);

            if (m_softmax != null)
            {
                LayerParameter p = new LayerParameter(LayerParameter.LayerType.SOFTMAXCROSSENTROPY_LOSS);
                p.loss_weight.Add(1);
                p.loss_weight.Add(0);
                p.loss_param.normalization = LossParameter.NormalizationMode.NONE;
                m_softmaxCe = new SoftmaxCrossEntropyLossLayer<T>(mycaffe.Cuda, mycaffe.Log, p);
            }

            m_colAccumulatedGradients = m_net.learnable_parameters.Clone();
            m_colAccumulatedGradients.SetDiff(0);

            m_nMiniBatch = mycaffe.CurrentProject.GetBatchSize(phase);
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
            m_memLoss.OnGetLoss -= memLoss_OnGetLoss;
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
        }

        public int RecurrentSequenceLength
        {
            get { return m_nRecurrentSequenceLength; }
        }

        public int Reshape(MemoryCollection col)
        {
            int nNum = col.Count;
            int nChannels = col[0].Data.Channels;
            int nHeight = col[0].Data.Height;
            int nWidth = col[0].Data.Height;
            int nActionProbs = 1;
            int nFound = 0;

            for (int i = 0; i < m_net.output_blobs.Count; i++)
            {
                if (m_net.output_blobs[i].type != Blob<T>.BLOB_TYPE.LOSS)
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
            m_blobDiscountedR1.ReshapeLike(m_blobDiscountedR);
            m_blobPolicyGradient1.ReshapeLike(m_blobPolicyGradient);
            m_blobActionOneHot1.ReshapeLike(m_blobActionOneHot);
            m_blobLoss.Reshape(1, 1, 1, 1);

            return nActionProbs;
        }

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

        public void SetActionProbabilities(float[] rg)
        {
            m_blobPolicyGradient.SetData(Utility.ConvertVec<T>(rg));           
        }

        public void SetActionOneHotVectors(float[] rg)
        {
            m_blobActionOneHot.SetData(Utility.ConvertVec<T>(rg));
        }

        public void SetData(List<Datum> rgData, List<Datum> rgClip)
        {
            if (m_nRecurrentSequenceLength > 1 && rgData.Count > 1 && rgClip != null)
            {
                m_rgData = rgData;
                m_rgClip = rgClip;
            }
            else
            {
                m_memData.AddDatumVector(rgData, rgClip, 1, true, true);
            }
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

        public int act(SimpleDatum sd, SimpleDatum sdClip, out float[] rgfAprob)
        {
            List<Datum> rgData = new List<Datum>();
            rgData.Add(new Datum(sd));
            double dfLoss;
            float fRandom = (float)m_random.NextDouble(); // Roll the dice.
            List<Datum> rgClip = null;

            if (sdClip != null)
            {
                rgClip = new List<Datum>();
                rgClip.Add(new Datum(sdClip));
            }

            m_memData.AddDatumVector(rgData, rgClip, 1, true, true);
            m_bSkipLoss = true;
            BlobCollection<T> res = m_net.Forward(out dfLoss);
            m_bSkipLoss = false;

            rgfAprob = null;

            for (int i = 0; i < res.Count; i++)
            {
                if (res[i].type != Blob<T>.BLOB_TYPE.LOSS)
                {
                    int nStart = 0;
                    // When using clip data with recurrent learning, only act on the last outputs.
                    if (rgClip != null)
                    {
                        int nCount = res[i].count();
                        int nOutput = nCount / rgClip[0].ItemCount;
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
            b1.CopyFrom(b, false, true);

            List<int> rgShape = b.shape();
            rgShape[0] = 1;
            b.Reshape(rgShape);
        }

        private void copyBlob(int nIdx, Blob<T> src, Blob<T> dst)
        {
            int nCount = dst.count();
            m_mycaffe.Cuda.copy(nCount, src.gpu_data, dst.mutable_gpu_data, nIdx * nCount, 0);
        }

        public void Train(int nIteration)
        {
            m_mycaffe.Log.Enable = false;

            // Run data/clip groups > 1 in non batch mode.
            if (m_nRecurrentSequenceLength > 1 && m_rgData != null && m_rgData.Count > 1 && m_rgClip != null)
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

                    m_solver.Step(1, TRAIN_STEP.NONE, true, false, true);
                }
            }
            else
            {
                m_solver.Step(1, TRAIN_STEP.NONE, true, false, true);
            }

            m_colAccumulatedGradients.Accumulate(m_mycaffe.Cuda, m_net.learnable_parameters, true);

            if (nIteration % m_nMiniBatch == 0)
            {
                m_net.learnable_parameters.CopyFrom(m_colAccumulatedGradients, true);
                m_colAccumulatedGradients.SetDiff(0);
                m_solver.ApplyUpdate(nIteration);
                m_net.ClearParamDiffs();
            }

            m_mycaffe.Log.Enable = true;
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
            int nBatch = e.DataItems.Count;
            int nSeqLen = rgDataShape[0];

            e.Data.Log.CHECK_EQ(nBatch, e.ClipItems.Count, "The data and clip should have the same number of items.");
            e.Data.Log.CHECK_EQ(nSeqLen, rgClipShape[0], "The data and clip should have the same sequence count.");

            rgDataShape[1] = nBatch;  // LSTM uses sizing: seq, batch, data1, data2
            rgClipShape[1] = nBatch;

            e.Data.Reshape(rgDataShape);
            e.Clip.Reshape(rgClipShape);

            T[] rgRawData = new T[e.Data.count()];
            T[] rgRawClip = new T[e.Clip.count()];

            int nDataSize = e.Data.count(2);
            T[] rgDataItem = new T[nDataSize];
            T dfClip;
            int nIdx;

            for (int i = 0; i < nBatch; i++)
            {
                Datum data = e.DataItems[i];
                Datum clip = e.ClipItems[i];

                for (int j = 0; j < nSeqLen; j++)
                {
                    dfClip = (T)Convert.ChangeType(clip.RealData[j], typeof(T));

                    for (int k = 0; k < nDataSize; k++)
                    {
                        rgDataItem[k] = (T)Convert.ChangeType(data.RealData[j * nDataSize + k], typeof(T));
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
                }
            }

            e.Data.mutable_cpu_data = rgRawData;
            e.Clip.mutable_cpu_data = rgRawClip;
            m_nRecurrentSequenceLength = nSeqLen;
        }

        /// <summary>
        /// Calcualte the loss and initial gradients.
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

            int nCount = m_blobPolicyGradient.count();
            long hActionOneHot = m_blobActionOneHot.gpu_data;
            long hPolicyGrad = m_blobPolicyGradient.mutable_gpu_data;
            long hDiscountedR = m_blobDiscountedR.gpu_data;
            double dfLoss;
            Blob<T> blobOriginalBottom = e.Bottom[0];
            int nDataSize = e.Bottom[0].count(1);

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
                // Calculate (a=0) ? 1-aprob : 0-aprob
                m_mycaffe.Cuda.add_scalar(nCount, -1.0, hActionOneHot); // invert one hot
                m_mycaffe.Cuda.abs(nCount, hActionOneHot, hActionOneHot); 
                m_mycaffe.Cuda.mul_scalar(nCount, -1.0, hPolicyGrad);   // negate Aprob
                m_mycaffe.Cuda.add(nCount, hActionOneHot, hPolicyGrad, hPolicyGrad);  // gradient = ((a=0)?1:0) - Aprob
                dfLoss = Utility.ConvertVal<T>(m_blobPolicyGradient.sumsq_data());

                m_mycaffe.Cuda.mul_scalar(nCount, -1.0, hPolicyGrad); // invert for we ApplyUpdate subtracts the gradients
            }

            // Modulate the gradient with the advantage (PG magic happens right here.)
            m_mycaffe.Cuda.mul(nCount, hPolicyGrad, hDiscountedR, hPolicyGrad);

            e.Loss = dfLoss;
            e.EnableLossUpdate = false; // apply gradients to bottom directly.

            if (hPolicyGrad != hBottomDiff)
                m_mycaffe.Cuda.copy(nCount, hPolicyGrad, hBottomDiff);

            // Copy the diff to the last in the sequence and 
            // zero out the rest in the sequence.
            if (m_nRecurrentSequenceLength > 1)
            {
                m_blobAprobLogit.SetDiff(0);
                m_blobAprobLogit.CopyFrom(e.Bottom[0], 0, (m_blobAprobLogit.num - 1) * nDataSize, nDataSize, false, true);
                e.Bottom[0].CopyFrom(m_blobAprobLogit, false, true);
                e.Bottom[0].CopyFrom(m_blobAprobLogit, true);
            }
        }
    }

    class MemoryCollection : GenericList<MemoryItem> /** @private */
    {
        public MemoryCollection()
        {
        }

        public float[] GetDiscountedRewards(float fGamma, bool bAllowReset)
        {
            float fRunningAdd = 0;
            float[] rgR = m_rgItems.Select(p => p.Reward).ToArray();
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

        public float[] GetActionProbabilities()
        {
            List<float> rgfAprob = new List<float>();

            for (int i = 0; i < m_rgItems.Count; i++)
            {
                rgfAprob.AddRange(m_rgItems[i].Aprob);
            }

            return rgfAprob.ToArray();
        }

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

        public List<Datum> GetData()
        {
            List<Datum> rgData = new List<Datum>();

            for (int i = 0; i < m_rgItems.Count; i++)
            {
                rgData.Add(new Datum(m_rgItems[i].Data));
            }

            return rgData;
        }

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

    class MemoryItem /** @private */
    {
        StateBase m_state;
        SimpleDatum m_x;
        int m_nAction;
        float[] m_rgfAprob;
        float m_fReward;

        public MemoryItem(StateBase s, SimpleDatum x, int nAction, float[] rgfAprob, float fReward)
        {
            m_state = s;
            m_x = x;
            m_nAction = nAction;
            m_rgfAprob = rgfAprob;
            m_fReward = fReward;
        }

        public StateBase State
        {
            get { return m_state; }
        }

        public SimpleDatum Data
        {
            get { return m_x; }
        }

        public int Action
        {
            get { return m_nAction; }
        }

        public float Reward
        {
            get { return m_fReward; }
        }

        /// <summary>
        /// Returns the action probabilities which are either a single Sigmoid output, or a set from a Softmax output.
        /// </summary>
        public float[] Aprob
        {
            get { return m_rgfAprob; }
        }

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
