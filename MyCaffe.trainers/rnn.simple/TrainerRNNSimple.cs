using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.fillers;
using MyCaffe.gym;
using MyCaffe.layers;
using MyCaffe.param;
using MyCaffe.solvers;

namespace MyCaffe.trainers.rnn.simple
{
    /// <summary>
    /// The TrainerRNNSimple implements a very simple RNN trainer inspired by adepierre's GitHub site referenced. 
    /// </summary>
    /// @see 1. [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/), by Andrej Karpathy, 2015, Github.io
    /// @see 2. [karpathy/char-rnn sample.lua](https://github.com/karpathy/char-rnn/blob/master/sample.lua)
    /// @see 3. [GitHub: adepierre/caffe-char-rnn](https://github.com/adepierre/caffe-char-rnn), by adepierre, 2017, Github
    /// @see 4. [MyCaffe: A Complete C# Re-Write of Caffe with Reinforcement Learning](https://arxiv.org/abs/1810.02272) by D. Brown, 2018, arXiv
    /// <remarks></remarks>
    public class TrainerRNNSimple<T> : IxTrainerRNN, IDisposable
    {
        IxTrainerCallback m_icallback;
        MyCaffeControl<T> m_mycaffe;
        PropertySet m_properties;
        CryptoRandom m_random;
        BucketCollection m_rgVocabulary = null;
        bool m_bUsePreloadData = true;
        GetDataArgs m_getDataTrainArgs = null;


        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="mycaffe">Specifies the MyCaffeControl to use for learning and prediction.</param>
        /// <param name="properties">Specifies the property set containing the key/value pairs of property settings.</param>
        /// <param name="random">Specifies the random number generator to use.</param>
        /// <param name="icallback">Specifies the callback for parent notifications and queries.</param>
        /// <param name="rgVocabulary">Specifies the vocabulary to use.</param>
        /// <remarks>
        /// The 'bUsePreloadData' parameter has been replaced with a property 'UsePreLoadData'. When this property does not exist,
        /// the 'UsePreLoadData' defaults to 'true'.
        /// </remarks>
        public TrainerRNNSimple(MyCaffeControl<T> mycaffe, PropertySet properties, CryptoRandom random, IxTrainerCallback icallback, BucketCollection rgVocabulary)
        {
            m_icallback = icallback;
            m_mycaffe = mycaffe;
            m_properties = properties;
            m_random = random;
            m_rgVocabulary = rgVocabulary;            
            m_bUsePreloadData = properties.GetPropertyAsBool("UsePreLoadData", true);
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

        private void updateStatus(int nIteration, int nMaxIteration, double dfAccuracy, double dfLoss, double dfLearningRate)
        {
            GetStatusArgs args = new GetStatusArgs(0, nIteration, nIteration, nMaxIteration, dfAccuracy, 0, 0, 0, dfLoss, dfLearningRate);
            m_icallback.OnUpdateStatus(args);
        }

        private float computeAccuracy(List<Tuple<float, float>> rg, float fThreshold)
        {
            int nMatch = 0;

            for (int i=0; i<rg.Count; i++)
            {
                float fDiff = Math.Abs(rg[i].Item1 - rg[i].Item2);

                if (fDiff < fThreshold)
                    nMatch++;
            }

            return (float)nMatch / (float)rg.Count;
        }

        /// <summary>
        /// Run a single cycle on the environment after the delay.
        /// </summary>
        /// <param name="nN">specifies the number of samples to run.</param>
        /// <param name="runProp">Optionally specifies properties to use when running.</param>
        /// <returns>The results of the run containing the action are returned.</returns>
        public float[] Run(int nN, PropertySet runProp)
        {
            m_mycaffe.CancelEvent.Reset();
            return null;
        }

        /// <summary>
        /// Run a single cycle on the environment after the delay.
        /// </summary>
        /// <param name="nN">Specifies the number of samples to run.</param>
        /// <param name="runProp">Optionally specifies properties to use when running.</param>
        /// <param name="type">Returns the data type contained in the byte stream.</param>
        /// <returns>The results of the run containing the action are returned as a byte stream.</returns>
        public byte[] Run(int nN, PropertySet runProp, out string type)
        {
            m_mycaffe.CancelEvent.Reset();
            type = "";
            return null;
        }

        /// <summary>
        /// Run the test cycle - currently this is not implemented.
        /// </summary>
        /// <param name="nN">Specifies the number of iterations (based on the ITERATION_TYPE) to run, or -1 to ignore.</param>
        /// <param name="type">Specifies the iteration type (default = ITERATION).</param>
        /// <returns>A value of <i>true</i> is returned when handled, <i>false</i> otherwise.</returns>
        public bool Test(int nN, ITERATOR_TYPE type)
        {
            return run(nN, type, TRAIN_STEP.NONE, Phase.TEST);
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
            return run(nN, type, step, Phase.TRAIN);
        }

        private bool run(int nN, ITERATOR_TYPE type, TRAIN_STEP step, Phase phase)
        { 
            PropertySet prop = new PropertySet();

            prop.SetProperty("TrainingStart", "0");

            m_mycaffe.CancelEvent.Reset();

            if (m_getDataTrainArgs == null)
            {
                m_getDataTrainArgs = new GetDataArgs(Phase.TRAIN, 0, m_mycaffe, m_mycaffe.Log, m_mycaffe.CancelEvent, true, -1);
                m_getDataTrainArgs.ExtraProperties = prop;
                m_icallback.OnGetData(m_getDataTrainArgs);
            }

            m_getDataTrainArgs.Action = 0;
            m_getDataTrainArgs.Reset = false;

            Net<T> net = m_mycaffe.GetInternalNet(Phase.TRAIN);
            Solver<T> solver = m_mycaffe.GetInternalSolver();

            InputLayer<T> input = net.layers[0] as InputLayer<T>;
            if (input == null)
                throw new Exception("Missing expected input layer!");

            int nBatchSize = input.layer_param.input_param.shape[0].dim[0];
            if (nBatchSize != 1)
                throw new Exception("Expected batch size of 1!");

            int nInputDim = input.layer_param.input_param.shape[0].dim[1];
            int nOutputDim = input.layer_param.input_param.shape[3].dim[1];

            string strVal = m_properties.GetProperty("BlobNames");
            string[] rgstrVal = strVal.Split('|');
            Dictionary<string, string> rgstrValMap = new Dictionary<string, string>();

            foreach (string strVal1 in rgstrVal)
            {
                string[] rgstrVal2 = strVal1.Split('~');
                if (rgstrVal2.Length != 2)
                    throw new Exception("Invalid BlobNames property, expected 'name=blobname'!");

                rgstrValMap.Add(rgstrVal2[0], rgstrVal2[1]);
            }

            Blob<T> blobX = null;
            if (rgstrValMap.ContainsKey("x"))
                blobX = net.blob_by_name(rgstrValMap["x"]);

            Blob<T> blobTt = null;
            if (rgstrValMap.ContainsKey("tt"))
                blobTt = net.blob_by_name(rgstrValMap["tt"]);

            Blob<T> blobMask = null;
            if (rgstrValMap.ContainsKey("mask"))
                blobMask = net.blob_by_name(rgstrValMap["mask"]);

            Blob<T> blobTarget = null;
            if (rgstrValMap.ContainsKey("target"))
                blobTarget = net.blob_by_name(rgstrValMap["target"]);

            Blob<T> blobXhat = null;
            if (rgstrValMap.ContainsKey("xhat"))
                blobXhat = net.blob_by_name(rgstrValMap["xhat"]);

            if (blobX == null)
                throw new Exception("The 'x' blob was not found in the 'BlobNames' property!");
            if (blobTt == null)
                throw new Exception("The 'tt' blob was not found in the 'BlobNames' property!");    
            if (blobMask == null)
                throw new Exception("The 'mask' blob was not found in the 'BlobNames' property!");
            if (blobTarget == null)
                throw new Exception("The 'target' blob was not found in the 'BlobNames' property!");
            if (blobXhat == null)
                throw new Exception("The 'xhat' blob was not found in the 'BlobNames' property!");

            if (blobX.count() != nInputDim)
                throw new Exception("The 'x' blob must have a count of '" + nInputDim.ToString() + "'!");
            if (blobTt.count() != nInputDim)
                throw new Exception("The 'tt' blob must have a count of '" + nInputDim.ToString() + "'!");
            if (blobMask.count() != nInputDim)
                throw new Exception("The 'mask' blob must have a count of '" + nInputDim.ToString() + "'!");
            if (blobTarget.count() != nOutputDim)
                throw new Exception("The 'target' blob must have a count of '" + nOutputDim.ToString() + "'!");
            if (blobXhat.count() != nOutputDim)
                throw new Exception("The 'xhat' blob must have a count of '" + nOutputDim.ToString() + "'!");

            float[] rgInput = new float[nInputDim];
            float[] rgTimeSteps = new float[nInputDim];
            float[] rgMask = new float[nInputDim];
            float[] rgTarget = new float[nOutputDim];

            List<Tuple<float, float>> rgAccHistory = new List<Tuple<float, float>>();

            for (int i = 0; i < nN; i++)
            {
                double dfLoss = 0;
                double dfAccuracy = 0;
                float fPredictedY = 0;
                float fTargetY = 0;

                m_icallback.OnGetData(m_getDataTrainArgs);

                if (m_getDataTrainArgs.CancelEvent.WaitOne(0))
                    break;  

                if (m_mycaffe.CancelEvent.WaitOne(0))
                    break;

                List<DataPoint> rgHistory = m_getDataTrainArgs.State.History;
                DataPoint dpLast = (rgHistory.Count > 0) ? rgHistory.Last() : null;

                if (dpLast != null)
                    fTargetY = dpLast.Target;
                else
                    fTargetY = -1;

                if (rgHistory.Count >= nInputDim)
                {
                    for (int j = 0; j < nInputDim; j++)
                    {
                        int nIdx = rgHistory.Count - nInputDim + j;
                        rgInput[j] = rgHistory[nIdx].Inputs[0];
                        rgTimeSteps[j] = rgHistory[nIdx].Time;
                        rgMask[j] = rgHistory[nIdx].Mask[0];
                        rgTarget[0] = rgHistory[nIdx].Target;
                    }

                    blobX.mutable_cpu_data = Utility.ConvertVec<T>(rgInput);
                    blobTt.mutable_cpu_data = Utility.ConvertVec<T>(rgTimeSteps);
                    blobMask.mutable_cpu_data = Utility.ConvertVec<T>(rgMask);
                    blobTarget.mutable_cpu_data = Utility.ConvertVec<T>(rgTarget);

                    net.Forward(out dfLoss);

                    if (phase == Phase.TRAIN)
                    {
                        net.Backward();
                        solver.Step(1);
                    }

                    float[] rgOutput = Utility.ConvertVecF<T>(blobXhat.mutable_cpu_data);
                    fPredictedY = rgOutput[0];

                    prop.SetProperty("override_prediction", fPredictedY.ToString());
                }
                else
                {
                    Thread.Sleep(50);
                }

                rgAccHistory.Add(new Tuple<float, float>(fTargetY, fPredictedY));
                if (rgAccHistory.Count > 100)
                    rgAccHistory.RemoveAt(0);

                dfAccuracy = computeAccuracy(rgAccHistory, 0.005f);

                updateStatus(i, nN, dfAccuracy, dfLoss, solver.parameter.base_lr);
            }

            return false;
        }
    }
}
