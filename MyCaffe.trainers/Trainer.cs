using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.imagedb;
using MyCaffe.layers;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyCaffe.basecode.descriptors;
using MyCaffe.param;
using System.ComponentModel;

namespace MyCaffe.trainers
{
    /// <summary>
    /// The IxTrainer interface is implemented by each Trainer.
    /// </summary>
    public interface IxTrainer
    {
        /// <summary>
        /// Initialize the trainer.
        /// </summary>
        /// <returns>Returns <i>true</i> on success, <i>false</i> on failure.</returns>
        bool Initialize();
        /// <summary>
        /// Train the network.
        /// </summary>
        /// <param name="nIterations">Specifies the number of iterations to run.</param>
        /// <returns>Returns <i>true</i> on success, <i>false</i> on failure.</returns>
        bool Train(int nIterations);
        /// <summary>
        /// Test the newtork.
        /// </summary>
        /// <param name="nIterations">Specifies the number of iterations to run.</param>
        /// <returns>Returns <i>true</i> on success, <i>false</i> on failure.</returns>
        bool Test(int nIterations);
    }

    /// <summary>
    /// The SimpleTrainer implements a simple reinforcement learning algorithm when observations are collected,
    /// the top observations are selected and then used to train the network.
    /// </summary>
    /// <typeparam name="T">Specifies the base data type of <i>float</i> or <i>double</i>.</typeparam>
    public class SimpleTrainer<T> : IxTrainer, IDisposable
    {
        /// <summary>
        /// Specifies the output log used for general text based output.
        /// </summary>
        protected Log m_log;
        /// <summary>
        /// Specifies the cancellation event.
        /// </summary>
        protected CancelEvent m_evtCancel;
        /// <summary>
        /// Specifies the unserlying MyCaffeControl with the open project.
        /// </summary>
        protected MyCaffeControl<T> m_caffe;
        /// <summary>
        /// Specifies the image database used to retrieve images based on the open project.
        /// </summary>
        protected IXImageDatabase m_imgDb;
        /// <summary>
        /// Specifies the mini batch size to use - this is defined by the MemoryDataParameter.
        /// </summary>
        protected int m_nMiniBatchSize = 1;
        /// <summary>
        /// Specifies the maximum number of items within each experience.
        /// </summary>
        protected int m_nExperienceMax = 1;
        /// <summary>
        /// Specifies the DataSet ID of the open project.
        /// </summary>
        protected int m_nDsId = 0;
        /// <summary>
        /// Specifies the general properties initialized from the key-value pair within the string sent to Initialize.
        /// </summary>
        protected PropertySet m_properties;
        Batch m_rgBatch;

        /// <summary>
        /// The OnIntialize event fires when initializing the trainer.
        /// </summary>
        public event EventHandler<InitializeArgs> OnInitialize;
        /// <summary>
        /// The OnGetObservations event fires from within the Train method and is used to get a new set of observation data.
        /// </summary>
        public event EventHandler<GetObservationArgs> OnGetObservations;
        /// <summary>
        /// The OnProcessObservations event first from within the Train method and is used to process each observation.
        /// </summary>
        public event EventHandler<ProcessObservationArgs> OnProcessObservations;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="mycaffe">Specifies the MyCaffeControl with an open project.</param>
        /// <param name="log">Specifies the output log.</param>
        /// <param name="evtCancel">Specifies the cancellation event.</param>
        /// <param name="properties">Specifies the set of key-value properties to use.</param>
        public SimpleTrainer(MyCaffeControl<T> mycaffe, Log log, CancelEvent evtCancel, PropertySet properties)
        {
            m_properties = properties;
            m_caffe = mycaffe;
            m_imgDb = m_caffe.ImageDatabase;
            m_log = log;
            m_evtCancel = evtCancel;
            m_nMiniBatchSize = m_caffe.CurrentProject.GetBatchSize(Phase.TRAIN);

            int nBatchSizeTest = m_caffe.CurrentProject.GetBatchSize(Phase.TEST);
            m_log.CHECK_EQ(m_nMiniBatchSize, nBatchSizeTest, "The training and testing phases must both have the same batch sizes!");

            m_nExperienceMax = m_properties.GetPropertyAsInt("ExperienceMax", 24);

            int nBatchBuffer = m_properties.GetPropertyAsInt("BatchBufferCount", 240);
            m_rgBatch = new Batch(nBatchBuffer);
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
        /// <returns>On success <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool Initialize()
        {
            if (OnInitialize != null)
            {
                InitializeArgs e = new InitializeArgs(m_caffe);
                OnInitialize(this, e);
                m_nDsId = e.DatasetID;
            }

            return true;
        }

        /// <summary>
        /// Train the network for the number of iterations specified.
        /// </summary>
        /// <param name="nIterations">Specifies the number of iterations to run.</param>
        /// <returns>On success <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool Train(int nIterations)
        {
            if (m_nDsId == 0)
                m_nDsId = m_caffe.CurrentProject.Dataset.ID;

            int nIteration = 0;
            Stopwatch sw = new Stopwatch();
            DatasetDescriptor ds = m_imgDb.GetDatasetById(m_nDsId);
            int nSrcId = ds.TrainingSource.ID;
            double dfTopPercent = m_properties.GetPropertyAsDouble("TopPercent", 0.33);

            if (OnGetObservations == null)
                throw new Exception("You need to connect the OnGetObservations event!");

            if (OnProcessObservations == null)
                throw new Exception("You need to connect the OnProcessObservations event!");

            if (nIterations <= 0)
            {
                int? nVal = m_caffe.CurrentProject.GetSolverSettingAsInt("max_iter");
                if (nVal.HasValue)
                    nIterations = nVal.Value;
            }

            //-----------------------------------------
            //  Fixup the Memory Data layers.
            //-----------------------------------------
            setupDataLayers(m_caffe, Phase.TRAIN);
            setupDataLayers(m_caffe, Phase.TEST);
            sw.Start();

            while (nIteration < nIterations)
            {
                //------------------------------------------
                //  Get a contiguous set of observations.
                //------------------------------------------
                GetObservationArgs getObsArgs = new GetObservationArgs(m_caffe, nSrcId, m_nExperienceMax);
                OnGetObservations(this, getObsArgs);
                List<SimpleDatum> rgSd = getObsArgs.Data;

                while (rgSd == null)
                {
                    if (m_evtCancel.WaitOne(0))
                        return false;

                    getObsArgs = new GetObservationArgs(m_caffe, nSrcId, m_nExperienceMax);
                    OnGetObservations(this, getObsArgs);
                    rgSd = getObsArgs.Data;
                }


                //-----------------------------------------
                //  Build a batch of observations from 
                //  the same start time.  The randomness
                //  of the action probability distribution
                //  should create varying actions - we 
                //  want to find the best ones.
                //-----------------------------------------
                Batch rgLocalBatch = new Batch();

                while (rgLocalBatch.Count < m_nMiniBatchSize)
                {
                    ProcessObservationArgs procObsArgs = new ProcessObservationArgs(m_caffe, rgSd);
                    OnProcessObservations(this, procObsArgs);
                    rgLocalBatch.Add(procObsArgs.Results);

                    if (sw.Elapsed.TotalMilliseconds > 1000)
                    {
                        double dfPct = (double)rgLocalBatch.Count / (double)m_nMiniBatchSize;
                        m_log.WriteLine("Iteration " + nIteration.ToString("N0") + " batch item " + rgLocalBatch.Count.ToString() + " (" + dfPct.ToString("P") + ")...");
                        sw.Restart();
                    }

                    if (m_evtCancel.WaitOne(0))
                        return false;
                }

                //-----------------------------------------
                //  Sort the batch by reward in decending
                //  order, and only keep the top third
                //  best outcomes.
                //-----------------------------------------
                rgLocalBatch.TrimToBest(dfTopPercent, m_nMiniBatchSize);
                m_rgBatch.Add(rgLocalBatch);


                //-----------------------------------------
                //  Train on the batch of the best third,
                //  which is automatically loaded in the
                //  MemoryDataLayer_OnGetData event.
                //-----------------------------------------
                int nTrainingIterations = m_rgBatch.GetObservationCount() / m_nMiniBatchSize;
                List<Datum> rgTrainingData = m_rgBatch.GetTrainingData();
                setTrainingData(m_caffe, Phase.TRAIN, rgTrainingData);
                copyTrainingData(m_caffe, Phase.TRAIN, Phase.TEST);

                m_caffe.Train(nTrainingIterations);
                nIteration++;
            }

            return true;
        }

        private bool setTrainingData(MyCaffeControl<T> caffe, Phase phase, List<Datum> rgData)
        {
            Net<T> net = caffe.GetInternalNet(phase);

            for (int i = 0; i < net.layers.Count; i++)
            {
                if (net.layers[i].type == MyCaffe.param.LayerParameter.LayerType.MEMORYDATA)
                {
                    ((MemoryDataLayer<T>)net.layers[i]).AddDatumVector(rgData);
                    break;
                }
            }

            return false;
        }

        private bool copyTrainingData(MyCaffeControl<T> caffe, Phase srcPhase, Phase dstPhase)
        {
            Net<T> netSrc = caffe.GetInternalNet(srcPhase);
            Net<T> netDst = caffe.GetInternalNet(dstPhase);
            MemoryDataLayer<T> layerSrc = null;
            MemoryDataLayer<T> layerDst = null;

            for (int i = 0; i < netSrc.layers.Count; i++)
            {
                if (netSrc.layers[i].type == LayerParameter.LayerType.MEMORYDATA)
                {
                    layerSrc = netSrc.layers[i] as MemoryDataLayer<T>;
                    break;
                }
            }

            for (int i = 0; i < netDst.layers.Count; i++)
            {
                if (netDst.layers[i].type == LayerParameter.LayerType.MEMORYDATA)
                {
                    layerDst = netDst.layers[i] as MemoryDataLayer<T>;
                    break;
                }
            }

            if (layerSrc == null)
                throw new Exception("The source phase " + srcPhase.ToString() + " is missing the expected MemoryDataLayer!");

            if (layerDst == null)
                throw new Exception("The destination phase " + dstPhase.ToString() + " is missing the expected MemoryDataLayer!");

            layerDst.Copy(layerSrc);

            return true;
        }

        private void setupDataLayers(MyCaffeControl<T> caffe, Phase phase)
        {
            Net<T> net = caffe.GetInternalNet(phase);
            bool bFound = false;

            for (int i = 0; i < net.layers.Count; i++)
            {
                if (net.layers[i].type == LayerParameter.LayerType.MEMORYDATA)
                {
                    MemoryDataLayer<T> layer = net.layers[i] as MemoryDataLayer<T>;
                    layer.ImageMean = m_imgDb.GetImageMean(m_caffe.CurrentProject.Dataset.TrainingSource.ID);
                    layer.Transformer.ImageMean = ((MemoryDataLayer<T>)net.layers[i]).ImageMean;
                    bFound = true;
                    break;
                }
            }

            if (!bFound)
                throw new Exception("The " + phase.ToString() + " network must use the MemoryDataLayer when performing reinforcement learning!");
        }

        /// <summary>
        /// Test the network for the number of iterations specified.
        /// </summary>
        /// <param name="nIterations">Specifies the number of iterations to run.</param>
        /// <returns>On success <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool Test(int nIterations)
        {
            return false;
        }
    }

    /// <summary>
    /// The InitializeArgs is passed to the OnInitialize event.
    /// </summary>
    public class InitializeArgs : EventArgs
    {
        int m_nOriginalDsId = 0;
        int m_nDsID = 0;
        Component m_caffe;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="mycaffe">Specifies the MyCaffeControl used.</param>
        public InitializeArgs(Component mycaffe)
        {
            m_caffe = mycaffe;

            if (mycaffe is MyCaffeControl<double>)
            {
                MyCaffeControl<double> mycaffe1 = mycaffe as MyCaffeControl<double>;
                m_nOriginalDsId = mycaffe1.CurrentProject.Dataset.ID;
            }
            else
            {
                MyCaffeControl<float> mycaffe1 = mycaffe as MyCaffeControl<float>;
                m_nOriginalDsId = mycaffe1.CurrentProject.Dataset.ID;
            }
        }

        /// <summary>
        /// Returns the MyCaffeControl used.
        /// </summary>
        public Component MyCaffe
        {
            get { return m_caffe; }
        }

        /// <summary>
        /// Returns the original Dataset ID of the open project held by the MyCaffeControl.
        /// </summary>
        public int OriginalDatasetID
        {
            get { return m_nOriginalDsId; }
        }

        /// <summary>
        /// Get/set a new Dataset ID which is actually used. 
        /// </summary>
        public int DatasetID
        {
            get { return m_nDsID; }
            set { m_nDsID = value; }
        }
    }

    /// <summary>
    /// The GetObservationArgs is passed to the OnGetObservations event.
    /// </summary>
    public class GetObservationArgs : EventArgs
    {
        int m_nSrcId;
        int m_nMax;
        List<SimpleDatum> m_rgData;
        Component m_caffe;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="mycaffe">Specifies the MyCaffeControl used.</param>
        /// <param name="nSrcId">Specifies the Source ID to use which is from the Dataset ID returned when initializing.</param>
        /// <param name="nMax">Specifies the maximum number of items for the set of observations returned.</param>
        public GetObservationArgs(Component mycaffe, int nSrcId, int nMax)
        {
            m_caffe = mycaffe;
            m_nSrcId = nSrcId;
            m_nMax = nMax;
            m_rgData = new List<SimpleDatum>();
        }

        /// <summary>
        /// Specifies the data items making up the data portion of the observations.
        /// </summary>
        public List<SimpleDatum> Data
        {
            get { return m_rgData; }
            set { m_rgData = value; }
        }

        /// <summary>
        /// Returns the MyCaffeControl used.
        /// </summary>
        public Component MyCaffe
        {
            get { return m_caffe; }
        }

        /// <summary>
        /// Returns the Source ID used.
        /// </summary>
        public int SourceID
        {
            get { return m_nSrcId; }
        }

        /// <summary>
        /// Returns the maximum number of items in the list of observation data items returned.
        /// </summary>
        public int Max
        {
            get { return m_nMax; }
        }
    }

    /// <summary>
    /// The ProcessObservationsArgs is passed to the OnProcessObservations event.
    /// </summary>
    public class ProcessObservationArgs : EventArgs
    {
        List<SimpleDatum> m_rgData;
        ObservationCollection m_results = new ObservationCollection();
        Component m_caffe;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="mycaffe">Specifies the MyCaffeControl used.</param>
        /// <param name="rgSd">Specifies a portion of the observation data previously collected.</param>
        public ProcessObservationArgs(Component mycaffe, List<SimpleDatum> rgSd)
        {
            m_caffe = mycaffe;
            m_rgData = rgSd;
        }

        /// <summary>
        /// Specifies the observation data to be processed.
        /// </summary>
        public List<SimpleDatum> Data
        {
            get { return m_rgData; }
        }

        /// <summary>
        /// Specifies the results of the processing which is a collection of observations containing
        /// the results of running the network on the data and state.
        /// </summary>
        public ObservationCollection Results
        {
            get { return m_results; }
            set { m_results = value; }
        }

        /// <summary>
        /// Returns the MyCaffeControl used.
        /// </summary>
        public Component MyCaffe
        {
            get { return m_caffe; }
        }
    }
}
