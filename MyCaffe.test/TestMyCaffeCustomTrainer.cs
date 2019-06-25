using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.param;
using MyCaffe.basecode;
using System.Threading;
using MyCaffe.common;
using System.Drawing;
using System.Diagnostics;
using MyCaffe.db.image;
using MyCaffe.basecode.descriptors;
using MyCaffe.gym;
using MyCaffe.trainers;
using System.ServiceModel;
using MyCaffe.db.stream;
using System.IO;
using System.Runtime.InteropServices;
using System.Reflection;
using System.Security.AccessControl;
using System.Security.Principal;

namespace MyCaffe.test
{
    [TestClass]
    public class TestMyCaffeCustomTrainer
    {
        [TestMethod]
        public void Train_PGSIMPLE_CartPoleWithOutUi()
        {
            MyCaffeCustomTrainerTest test = new MyCaffeCustomTrainerTest();

            try
            {
                foreach (IMyCaffeCustomTrainerTest t in test.Tests)
                {
                    t.TrainCartPolePG(false, false, "PG.SIMPLE", 100);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void Train_PGST_CartPoleWithOutUi()
        {
            MyCaffeCustomTrainerTest test = new MyCaffeCustomTrainerTest();

            try
            {
                foreach (IMyCaffeCustomTrainerTest t in test.Tests)
                {
                    t.TrainCartPolePG(false, false, "PG.ST", 100);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void Train_PGST_CartPoleWithOutUi_Dual()
        {
            MyCaffeCustomTrainerTest test = new MyCaffeCustomTrainerTest();

            try
            {
                foreach (IMyCaffeCustomTrainerTest t in test.Tests)
                {
                    t.TrainCartPolePG(true, false, "PG.ST", 100);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void Train_PGMT_CartPoleWithOutUi()
        {
            MyCaffeCustomTrainerTest test = new MyCaffeCustomTrainerTest();

            try
            {
                foreach (IMyCaffeCustomTrainerTest t in test.Tests)
                {
                    t.TrainCartPolePG(false, false, "PG.MT", 100);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void Train_PGMT_AtariWithOutUi()
        {
            MyCaffeCustomTrainerTest test = new MyCaffeCustomTrainerTest();

            try
            {
                foreach (IMyCaffeCustomTrainerTest t in test.Tests)
                {
                    t.TrainAtariPG(false, false, "PG.MT", 10);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void Train_PGST_AtariWithOutUi()
        {
            MyCaffeCustomTrainerTest test = new MyCaffeCustomTrainerTest();

            try
            {
                foreach (IMyCaffeCustomTrainerTest t in test.Tests)
                {
                    t.TrainAtariPG(false, false, "PG.ST", 10);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void Train_PGST_AtariWithOutUi_Dual()
        {
            MyCaffeCustomTrainerTest test = new MyCaffeCustomTrainerTest();

            try
            {
                foreach (IMyCaffeCustomTrainerTest t in test.Tests)
                {
                    t.TrainAtariPG(true, false, "PG.ST", 10);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void Train_PGSIMPLE_AtariWithOutUi()
        {
            MyCaffeCustomTrainerTest test = new MyCaffeCustomTrainerTest();

            try
            {
                foreach (IMyCaffeCustomTrainerTest t in test.Tests)
                {
                    t.TrainAtariPG(false, false, "PG.SIMPLE", 10);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void Train_C51ST_AtariWithOutUi_Dual()
        {
            MyCaffeCustomTrainerTest test = new MyCaffeCustomTrainerTest();

            try
            {
                foreach (IMyCaffeCustomTrainerTest t in test.Tests)
                {
                    t.TrainAtariC51Dual(true, "C51.ST", 10);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void Train_NoisyNetST_AtariWithOutUi_Dual()
        {
            MyCaffeCustomTrainerTest test = new MyCaffeCustomTrainerTest();

            try
            {
                foreach (IMyCaffeCustomTrainerTest t in test.Tests)
                {
                    t.TrainAtariNoisyNetDual(true, "NOISYDQN.ST", 10);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void Train_NoisyNetSimple_AtariWithOutUi_Dual()
        {
            MyCaffeCustomTrainerTest test = new MyCaffeCustomTrainerTest();

            try
            {
                foreach (IMyCaffeCustomTrainerTest t in test.Tests)
                {
                    t.TrainAtariNoisyNetDual(true, "NOISYDQN.SIMPLE", 10);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void Train_RNNSIMPLE_CharRNN_LSTM()
        {
            MyCaffeCustomTrainerTest test = new MyCaffeCustomTrainerTest();

            try
            {
                foreach (IMyCaffeCustomTrainerTest t in test.Tests)
                {
                    // NOTE: 1000 iterations is quite short and may not produce results,
                    // for real training 100,000+ is a more common iteration to use.
                    t.TrainCharRNN(false, false, "RNN.SIMPLE", LayerParameter.LayerType.LSTM, 1000);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void Train_RNNSIMPLE_CharRNN_LSTM_cuDnn()
        {
            MyCaffeCustomTrainerTest test = new MyCaffeCustomTrainerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (IMyCaffeCustomTrainerTest t in test.Tests)
                {
                    // NOTE: 1000 iterations is quite short and may not produce results,
                    // for real training 100,000+ is a more common iteration to use.
                    t.TrainCharRNN(false, false, "RNN.SIMPLE", LayerParameter.LayerType.LSTM, 1000);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void Train_RNNSIMPLE_CharRNN_LSTM_cuDnn_Dual()
        {
            MyCaffeCustomTrainerTest test = new MyCaffeCustomTrainerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (IMyCaffeCustomTrainerTest t in test.Tests)
                {
                    // NOTE: 1000 iterations is quite short and may not produce results,
                    // for real training 100,000+ is a more common iteration to use.
                    t.TrainCharRNN(true, false, "RNN.SIMPLE", LayerParameter.LayerType.LSTM, 1000);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void Train_RNNSIMPLE_CharRNN_LSTMSIMPLE()
        {
            MyCaffeCustomTrainerTest test = new MyCaffeCustomTrainerTest();

            try
            {
                foreach (IMyCaffeCustomTrainerTest t in test.Tests)
                {
                    // NOTE: 1000 iterations is quite short and may not produce results,
                    // for real training 100,000+ is a more common iteration to use.
                    t.TrainCharRNN(false, false, "RNN.SIMPLE", LayerParameter.LayerType.LSTM_SIMPLE, 1000);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void Train_RNNSIMPLE_WavRNN_LSTM()
        {
            MyCaffeCustomTrainerTest test = new MyCaffeCustomTrainerTest();

            try
            {
                foreach (IMyCaffeCustomTrainerTest t in test.Tests)
                {
                    // NOTE: 1000 iterations is quite short and may not produce results,
                    // for real training 100,000+ is a more common iteration to use.
                    t.TrainWavRNN(false, false, "RNN.SIMPLE", LayerParameter.LayerType.LSTM, 1000);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void Train_RNNSIMPLE_WavRNN_LSTMSIMPLE()
        {
            MyCaffeCustomTrainerTest test = new MyCaffeCustomTrainerTest();

            try
            {
                foreach (IMyCaffeCustomTrainerTest t in test.Tests)
                {
                    // NOTE: 1000 iterations is quite short and may not produce results,
                    // for real training 100,000+ is a more common iteration to use.
                    t.TrainWavRNN(false, false, "RNN.SIMPLE", LayerParameter.LayerType.LSTM_SIMPLE, 1000);
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface IMyCaffeCustomTrainerTest : ITest
    {
        void TrainCartPolePG(bool bDual, bool bShowUi, string strTrainerType, int nIterations = 1000, bool bUseAcceleratedTraining = false, bool bAllowDiscountReset = false);
        void TrainCartPoleNoisyNetDual(bool bShowUi, string strTrainerType, int nIterations = 1000, bool bUseAcceleratedTraining = false, bool bAllowDiscountReset = false);
        void TrainAtariPG(bool bDual, bool bShowUi, string strTrainerType, int nIterations = 1000, bool bUseAcceleratedTraining = false, bool bAllowDiscountReset = false, string strAtariRom = null, bool bAllowNegRewards = false, bool bTerminateOnRallyEnd = false);
        void TrainAtariC51Dual(bool bShowUi, string strTrainerType, int nIterations = 100, int nIteratorType = 0, string strAtariRom = null, bool bAllowNegRewards = false, bool bTerminateOnRallyEnd = false, bool bLoadWeightsIfExist = false, double dfVMin = -10, double dfVMax = 10);
        void TrainAtariNoisyNetDual(bool bShowUi, string strTrainerType, int nIterations = 100, int nIteratorType = 0, string strAtariRom = null, bool bAllowNegRewards = false, bool bTerminateOnRallyEnd = false, bool bLoadWeightsIfExist = false);

        void TrainCharRNN(bool bDual, bool bShowUi, string strTrainerType, LayerParameter.LayerType lstm, int nIterations = 1000, bool bUseAcceleratedTraining = false, bool bAllowDiscountReset = false);
        void TrainWavRNN(bool bDual, bool bShowUi, string strTrainerType, LayerParameter.LayerType lstm, int nIterations = 1000, bool bUseAcceleratedTraining = false, bool bAllowDiscountReset = false);
    }

    class MyCaffeCustomTrainerTest : TestBase
    {
        public MyCaffeCustomTrainerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("MyCaffe Custom Trainer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new MyCaffeCustomTrainerTest<double>(strName, nDeviceID, engine);
            else
                return new MyCaffeCustomTrainerTest<float>(strName, nDeviceID, engine);
        }
    }

    public class MyCaffeCustomTrainerTest<T> : TestEx<T>, IMyCaffeCustomTrainerTest, IXMyCaffeCustomTrainerCallbackRNN
    {
        SettingsCaffe m_settings = new SettingsCaffe();
        CancelEvent m_evtCancel = new CancelEvent();
        string m_strModelPath;
        TestingProgressSet m_progress = new TestingProgressSet();
        int m_nMaxIteration = 0;


        public MyCaffeCustomTrainerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_engine = engine;
            m_settings.ImageDbLoadMethod = IMAGEDB_LOAD_METHOD.LOAD_ALL;
            m_settings.GpuIds = nDeviceID.ToString();
        }

        protected override void dispose()
        {
            m_progress.Dispose();
            base.dispose();
        }

        public CancelEvent CancelEvent
        {
            get { return m_evtCancel; }
        }


        public void TrainCartPolePG(bool bDual, bool bShowUi, string strTrainerType, int nIterations = 1000, bool bUseAcceleratedTraining = false, bool bAllowDiscountReset = false)
        {
            if (bDual)
                TrainCartPolePGDual(bShowUi, strTrainerType, nIterations, bUseAcceleratedTraining, bAllowDiscountReset);
            else
                TrainCartPolePG(bShowUi, strTrainerType, nIterations, bUseAcceleratedTraining, bAllowDiscountReset);
        }

        public void TrainCartPolePG(bool bShowUi, string strTrainerType, int nIterations = 1000, bool bUseAcceleratedTraining = false, bool bAllowDiscountReset = false)
        {
            m_evtCancel.Reset();

            GymCollection col = new GymCollection();
            col.Load();
            IXMyCaffeGym igym = col.Find("Cart-Pole");
            string strAccelTrain = (bUseAcceleratedTraining) ? "ON" : "OFF";
            string strAllowReset = (bAllowDiscountReset) ? "YES" : "NO";

            if (strTrainerType != "PG.MT")
                strAccelTrain = "NOT SUPPORTED";

            m_log.WriteHeader("Test Training Cart-Pole for " + nIterations.ToString("N0") + " iterations.");
            m_log.WriteLine("Using trainer = " + strTrainerType + ", Accelerated Training = " + strAccelTrain + ", AllowDiscountReset = " + strAllowReset);
            MyCaffeControl<T> mycaffe = new MyCaffeControl<T>(m_settings, m_log, m_evtCancel);
            MyCaffeCartPoleTrainer trainer = new MyCaffeCartPoleTrainer();
            ProjectEx project = getReinforcementProject(igym, nIterations, DATA_TYPE.VALUES, strTrainerType.Contains("SIMPLE"));
            DatasetDescriptor ds = trainer.GetDatasetOverride(0);

            m_log.CHECK(ds != null, "The MyCaffeCartPoleTrainer should return its dataset override returned by the Gym that it uses.");

            // load the project to train (note the project must use the MemoryDataLayer for input).
            mycaffe.Load(Phase.TRAIN, project, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.NONE, false, null, false);

            // Train the network using the custom trainer
            //  - Iterations (maximum frames cumulative across all threads) = 1000 (normally this would be much higher such as 500,000)
            //  - Learning rate = 0.001 (defined in solver.prototxt)
            //  - Mini Batch Size = 10 (defined in train_val.prototxt for MemoryDataLayer)
            //
            //  - TraingerType = 'strTrainerType' ('PG.MT' = use multi-threaded Policy Gradient trainer, 'PG.ST' = single-threaded trainer, 'PG.SIMPLE' = basic trainer with Sigmoid output support only)
            //  - RewardType = MAX (display the maximum rewards received, a setting of VAL displays the actual reward received)
            //  - Gamma = 0.99 (discounting factor)
            //  - Init1 = default force of 10.
            //  - Init2 = do not use additive force.                    
            //  - Threads = 1 (only use 1 thread if multi-threading is supported)
            trainer.Initialize("TrainerType=" + strTrainerType + ";RewardType=VAL;UseAcceleratedTraining=" + bUseAcceleratedTraining.ToString() + ";AllowDiscountReset=" + bAllowDiscountReset.ToString() + ";Gamma=0.99;Init1=10;Init2=0;Threads=1", this);

            if (bShowUi)
                trainer.OpenUi();

            m_nMaxIteration = nIterations;
            trainer.Train(mycaffe, nIterations);
            trainer.CleanUp();

            // Release the mycaffe resources.
            mycaffe.Dispose();
        }

        public void TrainCartPolePGDual(bool bShowUi, string strTrainerType, int nIterations = 1000, bool bUseAcceleratedTraining = false, bool bAllowDiscountReset = false)
        {
            m_evtCancel.Reset();

            GymCollection col = new GymCollection();
            col.Load();
            IXMyCaffeGym igym = col.Find("Cart-Pole");
            string strAccelTrain = (bUseAcceleratedTraining) ? "ON" : "OFF";
            string strAllowReset = (bAllowDiscountReset) ? "YES" : "NO";

            if (strTrainerType != "PG.MT")
                strAccelTrain = "NOT SUPPORTED";

            m_log.WriteHeader("Test Training Cart-Pole for " + nIterations.ToString("N0") + " iterations.");
            m_log.WriteLine("Using trainer = " + strTrainerType + ", Accelerated Training = " + strAccelTrain + ", AllowDiscountReset = " + strAllowReset);
            MyCaffeControl<T> mycaffe = new MyCaffeControl<T>(m_settings, m_log, m_evtCancel);
            MyCaffeCartPoleTrainerDual trainerX = new MyCaffeCartPoleTrainerDual();

            IXMyCaffeCustomTrainerRL itrainer = trainerX as IXMyCaffeCustomTrainerRL;
            if (itrainer == null)
                throw new Exception("The trainer must implement the IXMyCaffeCustomTrainerRL interface!");

            ProjectEx project = getReinforcementProject(igym, nIterations, DATA_TYPE.VALUES, strTrainerType.Contains("SIMPLE"));
            DatasetDescriptor ds = itrainer.GetDatasetOverride(0);

            m_log.CHECK(ds != null, "The MyCaffeCartPoleTrainer should return its dataset override returned by the Gym that it uses.");

            // Train the network using the custom trainer
            //  - Iterations (maximum frames cumulative across all threads) = 1000 (normally this would be much higher such as 500,000)
            //  - Learning rate = 0.001 (defined in solver.prototxt)
            //  - Mini Batch Size = 10 (defined in train_val.prototxt for MemoryDataLayer)
            //
            //  - TraingerType = 'strTrainerType' ('PG.MT' = use multi-threaded Policy Gradient trainer, 'PG.ST' = single-threaded trainer, 'PG.SIMPLE' = basic trainer with Sigmoid output support only)
            //  - RewardType = MAX (display the maximum rewards received, a setting of VAL displays the actual reward received)
            //  - Gamma = 0.99 (discounting factor)
            //  - Init1 = default force of 10.
            //  - Init2 = do not use additive force.                    
            //  - Threads = 1 (only use 1 thread if multi-threading is supported)
            itrainer.Initialize("TrainerType=" + strTrainerType + ";RewardType=VAL;UseAcceleratedTraining=" + bUseAcceleratedTraining.ToString() + ";AllowDiscountReset=" + bAllowDiscountReset.ToString() + ";Gamma=0.99;Init1=10;Init2=0;Threads=1", this);

            // load the project to train (note the project must use the MemoryDataLayer for input).
            mycaffe.Load(Phase.TRAIN, project, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.NONE, false, null, false, true, itrainer.Stage.ToString());

            if (bShowUi)
                itrainer.OpenUi();

            m_nMaxIteration = nIterations;
            itrainer.Train(mycaffe, nIterations);
            itrainer.CleanUp();

            // Release the mycaffe resources.
            mycaffe.Dispose();
        }

        public void TrainCartPoleNoisyNetDual(bool bShowUi, string strTrainerType, int nIterations = 1000, bool bUseAcceleratedTraining = false, bool bAllowDiscountReset = false)
        {
            m_evtCancel.Reset();

            GymCollection col = new GymCollection();
            col.Load();
            IXMyCaffeGym igym = col.Find("Cart-Pole");
            string strAccelTrain = (bUseAcceleratedTraining) ? "ON" : "OFF";
            string strAllowReset = (bAllowDiscountReset) ? "YES" : "NO";

            if (strTrainerType != "NOISYDQN.SIMPLE")
                strAccelTrain = "NOT SUPPORTED";

            m_log.WriteHeader("Test Training Cart-Pole for " + nIterations.ToString("N0") + " iterations.");
            m_log.WriteLine("Using trainer = " + strTrainerType + ", Accelerated Training = " + strAccelTrain + ", AllowDiscountReset = " + strAllowReset);
            MyCaffeControl<T> mycaffe = new MyCaffeControl<T>(m_settings, m_log, m_evtCancel);
            MyCaffeCartPoleTrainerDual trainerX = new MyCaffeCartPoleTrainerDual();

            IXMyCaffeCustomTrainerRL itrainer = trainerX as IXMyCaffeCustomTrainerRL;
            if (itrainer == null)
                throw new Exception("The trainer must implement the IXMyCaffeCustomTrainerRL interface!");

            ProjectEx project = getReinforcementProjectNoisyNet(igym, nIterations, false, strTrainerType);
            DatasetDescriptor ds = itrainer.GetDatasetOverride(0);

            m_log.CHECK(ds != null, "The MyCaffeCartPoleTrainer should return its dataset override returned by the Gym that it uses.");

            // Train the network using the custom trainer
            //  - Iterations (maximum frames cumulative across all threads) = 1000 (normally this would be much higher such as 500,000)
            //  - Learning rate = 0.001 (defined in solver.prototxt)
            //  - Mini Batch Size = 10 (defined in train_val.prototxt for MemoryDataLayer)
            //
            //  - TraingerType = 'strTrainerType' ('PG.MT' = use multi-threaded Policy Gradient trainer, 'PG.ST' = single-threaded trainer, 'PG.SIMPLE' = basic trainer with Sigmoid output support only)
            //  - UseRawInput = do not preprocess by subtracting the current from the last for we are using a lot of RelU activations which set negative values to zero.
            //  - RewardType = MAX (display the maximum rewards received, a setting of VAL displays the actual reward received)
            //  - Gamma = 0.99 (discounting factor)
            //  - Init1 = default force of 10.
            //  - Init2 = do not use additive force.                    
            //  - Threads = 1 (only use 1 thread if multi-threading is supported)
            itrainer.Initialize("TrainerType=" + strTrainerType + ";UseRawInput=True;RewardType=VAL;UseAcceleratedTraining=" + bUseAcceleratedTraining.ToString() + ";AllowDiscountReset=" + bAllowDiscountReset.ToString() + ";Gamma=0.99;Init1=10;Init2=0;Threads=1", this);

            // load the project to train (note the project must use the MemoryDataLayer for input).
            mycaffe.Load(Phase.TRAIN, project, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.NONE, false, null, false, true, itrainer.Stage.ToString());

            if (bShowUi)
                itrainer.OpenUi();

            m_nMaxIteration = nIterations;
            itrainer.Train(mycaffe, nIterations);
            itrainer.CleanUp();

            // Release the mycaffe resources.
            mycaffe.Dispose();
        }


        public void TrainAtariPG(bool bDual, bool bShowUi, string strTrainerType, int nIterations = 1000, bool bUseAcceleratedTraining = false, bool bAllowDiscountReset = false, string strAtariRom = null, bool bAllowNegRewards = false, bool bTerminateOnRallyEnd = false)
        {
            if (bDual)
                TrainAtariPGDual(bShowUi, strTrainerType, nIterations, bUseAcceleratedTraining, bAllowDiscountReset, strAtariRom, bAllowNegRewards, bTerminateOnRallyEnd);
            else
                TrainAtariPG(bShowUi, strTrainerType, nIterations, bUseAcceleratedTraining, bAllowDiscountReset, strAtariRom, bAllowNegRewards, bTerminateOnRallyEnd);
        }

        public void TrainAtariPG(bool bShowUi, string strTrainerType, int nIterations = 100, bool bUseAcceleratedTraining = false, bool bAllowDiscountReset = false, string strAtariRom = null, bool bAllowNegRewards = false, bool bTerminateOnRallyEnd = false)
        {
            m_evtCancel.Reset();

            GymCollection col = new GymCollection();
            col.Load();
            IXMyCaffeGym igym = col.Find("ATARI");
            DATA_TYPE dt = DATA_TYPE.BLOB;
            string strAccelTrain = (bUseAcceleratedTraining) ? "ON" : "OFF";
            string strAllowReset = (bAllowDiscountReset) ? "YES" : "NO";

            if (strTrainerType != "PG.MT")
                strAccelTrain = "NOT SUPPORTED";

            m_log.WriteHeader("Test Training ATARI for " + nIterations.ToString("N0") + " iterations.");
            m_log.WriteLine("Using trainer = " + strTrainerType + ", Accelerated Training = " + strAccelTrain + ", AllowDiscountReset = " + strAllowReset);
            MyCaffeControl<T> mycaffe = new MyCaffeControl<T>(m_settings, m_log, m_evtCancel);
            MyCaffeAtariTrainer trainer = new MyCaffeAtariTrainer();
            ProjectEx project = getReinforcementProject(igym, nIterations, dt);
            DatasetDescriptor ds = trainer.GetDatasetOverride(0);

            if (strAtariRom == null)
                strAtariRom = "pong";

            string strRom = getRomPath(strAtariRom + ".bin");

            m_log.CHECK(ds != null, "The MyCaffeAtariTrainer should return its dataset override returned by the Gym that it uses.");

            // load the project to train (note the project must use the MemoryDataLayer for input).
            mycaffe.Load(Phase.TRAIN, project, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.NONE, false, null, false);

            // Train the network using the custom trainer
            //  - Iterations (maximum frames cumulative across all threads) = 1000 (normally this would be much higher such as 500,000)
            //  - Learning rate = 0.001 (defined in solver.prototxt)
            //  - Mini Batch Size = 10 (defined in train_val.prototxt for MemoryDataLayer)
            //
            //  - TrainerType = 'PG.MT' ('PG.MT' = use multi-threaded Policy Gradient trainer)
            //  - RewardType = MAX (display the maximum rewards received, a setting of VAL displays the actual reward received)
            //  - EnableBinaryActions = True (only use actions move left, move right for binary decisions)
            //  - Gamma = 0.99 (discounting factor)
            //  - Threads = 1 (only use 1 thread if multi-threading is supported)
            //  - AllowNegativeRewards = False (when enabled and the ball falls behind our player, a -1 reward is given).
            //  - TerminateOnRallyEnd = False (when enabled a termination state is given each time the ball falls behind our player).
            //  - GameROM = 'path to game ROM'
            string strParam = "TrainerType=" + strTrainerType + ";RewardType=VAL;";
            strParam += "EnableBinaryActions=True;";
            strParam += "UseAcceleratedTraining=" + bUseAcceleratedTraining + ";";
            strParam += "AllowDiscountReset=" + bAllowDiscountReset + ";";
            strParam += "Gamma=0.99;";
            strParam += "AllowNegativeRewards=" + bAllowNegRewards.ToString() + ";";
            strParam += "TerminateOnRallyEnd=" + bTerminateOnRallyEnd.ToString() + ";";
            strParam += "GameROM=" + strRom;
            trainer.Initialize(strParam, this);

            if (bShowUi)
                trainer.OpenUi();

            m_nMaxIteration = nIterations;
            trainer.Train(mycaffe, nIterations);
            trainer.CleanUp();

            // Release the mycaffe resources.
            mycaffe.Dispose();
        }

        public void TrainAtariPGDual(bool bShowUi, string strTrainerType, int nIterations = 100, bool bUseAcceleratedTraining = false, bool bAllowDiscountReset = false, string strAtariRom = null, bool bAllowNegRewards = false, bool bTerminateOnRallyEnd = false)
        {
            m_evtCancel.Reset();

            GymCollection col = new GymCollection();
            col.Load();
            IXMyCaffeGym igym = col.Find("ATARI");
            DATA_TYPE dt = DATA_TYPE.BLOB;
            string strAccelTrain = (bUseAcceleratedTraining) ? "ON" : "OFF";
            string strAllowReset = (bAllowDiscountReset) ? "YES" : "NO";

            if (strTrainerType != "PG.MT")
                strAccelTrain = "NOT SUPPORTED";

            m_log.WriteHeader("Test Training ATARI for " + nIterations.ToString("N0") + " iterations.");
            m_log.WriteLine("Using trainer = " + strTrainerType + ", Accelerated Training = " + strAccelTrain + ", AllowDiscountReset = " + strAllowReset);
            MyCaffeControl<T> mycaffe = new MyCaffeControl<T>(m_settings, m_log, m_evtCancel);
            MyCaffeAtariTrainerDual trainerX = new MyCaffeAtariTrainerDual();

            IXMyCaffeCustomTrainerRL itrainer = trainerX as IXMyCaffeCustomTrainerRL;
            if (itrainer == null)
                throw new Exception("The trainer must implement the IXMyCaffeCustomTrainerRL interface!");

            ProjectEx project = getReinforcementProject(igym, nIterations, dt);
            DatasetDescriptor ds = itrainer.GetDatasetOverride(0);

            if (strAtariRom == null)
                strAtariRom = "pong";

            string strRom = getRomPath(strAtariRom + ".bin");

            m_log.CHECK(ds != null, "The MyCaffeAtariTrainer should return its dataset override returned by the Gym that it uses.");

            // Train the network using the custom trainer
            //  - Iterations (maximum frames cumulative across all threads) = 1000 (normally this would be much higher such as 500,000)
            //  - Learning rate = 0.001 (defined in solver.prototxt)
            //  - Mini Batch Size = 10 (defined in train_val.prototxt for MemoryDataLayer)
            //
            //  - TrainerType = 'PG.MT' ('PG.MT' = use multi-threaded Policy Gradient trainer)
            //  - RewardType = MAX (display the maximum rewards received, a setting of VAL displays the actual reward received)
            //  - EnableBinaryActions = True (only use actions move left, move right for binary decisions)
            //  - Gamma = 0.99 (discounting factor)
            //  - Threads = 1 (only use 1 thread if multi-threading is supported)
            //  - AllowNegativeRewards = False (when enabled and the ball falls behind our player, a -1 reward is given).
            //  - TerminateOnRallyEnd = False (when enabled a termination state is given each time the ball falls behind our player).
            //  - GameROM = 'path to game ROM'
            string strParam = "TrainerType=" + strTrainerType + ";RewardType=VAL;";
            strParam += "EnableBinaryActions=True;";
            strParam += "UseAcceleratedTraining=" + bUseAcceleratedTraining + ";";
            strParam += "AllowDiscountReset=" + bAllowDiscountReset + ";";
            strParam += "Gamma=0.99;";
            strParam += "AllowNegativeRewards=" + bAllowNegRewards.ToString() + ";";
            strParam += "TerminateOnRallyEnd=" + bTerminateOnRallyEnd.ToString() + ";";
            strParam += "GameROM=" + strRom;
            itrainer.Initialize(strParam, this);

            // load the project to train (note the project must use the MemoryDataLayer for input).
            mycaffe.Load(Phase.TRAIN, project, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.NONE, false, null, false, true, itrainer.Stage.ToString());

            if (bShowUi)
                itrainer.OpenUi();

            m_nMaxIteration = nIterations;
            itrainer.Train(mycaffe, nIterations);
            itrainer.CleanUp();

            // Release the mycaffe resources.
            mycaffe.Dispose();
        }

        public void TrainAtariC51Dual(bool bShowUi, string strTrainerType, int nIterations = 100, int iteratorType = 0, string strAtariRom = null, bool bAllowNegRewards = false, bool bTerminateOnRallyEnd = false, bool bLoadWeightsIfExists = false, double dfVMin = -10, double dfVMax = 10)
        {
            m_evtCancel.Reset();

            if (strTrainerType != "C51.ST")
                throw new Exception("Currently only the C51.ST trainer supports C51 training.");

            GymCollection col = new GymCollection();
            col.Load();
            IXMyCaffeGym igym = col.Find("ATARI");
            string strAccelTrain = "OFF";
            string strAllowReset = "NO";

            m_log.WriteHeader("Test Training ATARI for " + nIterations.ToString("N0") + " iterations.");
            m_log.WriteLine("Using trainer = " + strTrainerType + ", Accelerated Training = " + strAccelTrain + ", AllowDiscountReset = " + strAllowReset);
            MyCaffeControl<T> mycaffe = new MyCaffeControl<T>(m_settings, m_log, m_evtCancel);
            mycaffe.OnSnapshot += Mycaffe_OnSnapshot;

            MyCaffeAtariTrainerDual trainerX = new MyCaffeAtariTrainerDual();
            IXMyCaffeCustomTrainerRL itrainer = trainerX as IXMyCaffeCustomTrainerRL;
            if (itrainer == null)
                throw new Exception("The trainer must implement the IXMyCaffeCustomTrainerRL interface!");

            ProjectEx project = getReinforcementProjectC51(igym, nIterations, bLoadWeightsIfExists, (strTrainerType == "C51b.ST") ? true : false);
            DatasetDescriptor ds = itrainer.GetDatasetOverride(0);

            if (strAtariRom == null)
                strAtariRom = "pong";

            string strRom = getRomPath(strAtariRom + ".bin");

            m_log.CHECK(ds != null, "The MyCaffeAtariTrainer should return its dataset override returned by the Gym that it uses.");

            // Train the network using the custom trainer
            //  - Iterations (maximum frames cumulative across all threads) = 1000 (normally this would be much higher such as 500,000)
            //  - Learning rate = 0.001 (defined in solver.prototxt)
            //  - Mini Batch Size = 10 (defined in train_val.prototxt for MemoryDataLayer)
            //
            //  - TrainerType = 'C51.ST' ('C51.ST' = use single-threaded C51 trainer)
            //  - RewardType = MAX (display the maximum rewards received, a setting of VAL displays the actual reward received)
            //  - Gamma = 0.99 (discounting factor)
            //  - Threads = 1 (only use 1 thread if multi-threading is supported)
            //  - AllowNegativeRewards = False (when enabled and the ball falls behind our player, a -1 reward is given).
            //  - TerminateOnRallyEnd = False (when enabled a termination state is given each time the ball falls behind our player).
            //  - GameROM = 'path to game ROM'
            string strParam = "TrainerType=" + strTrainerType + ";RewardType=VAL;Gamma=0.99;";
            strParam += "UseRawInput=True;";        // use the input values directly, do not take a difference of them with the previous.
            strParam += "Preprocess=False;";        // do not preprocess (turn inputs to 1 or 0).
            strParam += "ActionForceGray=True;";    // force inputs to gray with a single color channel.
            strParam += "FrameSkip=1;";             // process one frame of data at a time.
            strParam += "AllowNegativeRewards=" + bAllowNegRewards.ToString() + ";";    // receive -1 reward on rally's where ball not even hit.
            strParam += "TerminateOnRallyEnd=" + bTerminateOnRallyEnd.ToString() + ";"; // play only one rally at a time then restart the game.
            strParam += "EpsSteps=" + nIterations.ToString() + ";EpsStart=1.0;EpsEnd=0.01;";
            strParam += "VMin=" + dfVMin.ToString() + ";VMax=" + dfVMax.ToString() + ";";
            strParam += "GameROM=" + strRom;
            itrainer.Initialize(strParam, this);

            // load the project to train (note the project must use the MemoryDataLayer for input).
            mycaffe.Load(Phase.TRAIN, project, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.NONE, false, null, false, true, itrainer.Stage.ToString());

            if (bShowUi)
                itrainer.OpenUi();

            m_nMaxIteration = nIterations;
            itrainer.Train(mycaffe, nIterations, (ITERATOR_TYPE)iteratorType);
            itrainer.CleanUp();

            // Release the mycaffe resources.
            mycaffe.Dispose();
        }

        public void TrainAtariNoisyNetDual(bool bShowUi, string strTrainerType, int nIterations = 100, int iteratorType = 0, string strAtariRom = null, bool bAllowNegRewards = false, bool bTerminateOnRallyEnd = false, bool bLoadWeightsIfExists = false)
        {
            m_evtCancel.Reset();

            if (strTrainerType != "NOISYDQN.ST" && 
                strTrainerType != "NOISYDQN.SIMPLE")
                throw new Exception("Currently only the NOISYDQN.ST and NOISYDQN.SIMPLE trainers support NoisyNet training.");

            GymCollection col = new GymCollection();
            col.Load();
            IXMyCaffeGym igym = col.Find("ATARI");
            string strAccelTrain = "OFF";
            string strAllowReset = "NO";

            m_log.WriteHeader("Test Training ATARI for " + nIterations.ToString("N0") + " iterations.");
            m_log.WriteLine("Using trainer = " + strTrainerType + ", Accelerated Training = " + strAccelTrain + ", AllowDiscountReset = " + strAllowReset);
            MyCaffeControl<T> mycaffe = new MyCaffeControl<T>(m_settings, m_log, m_evtCancel);
            mycaffe.OnSnapshot += Mycaffe_OnSnapshot;

            MyCaffeAtariTrainerDual trainerX = new MyCaffeAtariTrainerDual();
            IXMyCaffeCustomTrainerRL itrainer = trainerX as IXMyCaffeCustomTrainerRL;
            if (itrainer == null)
                throw new Exception("The trainer must implement the IXMyCaffeCustomTrainerRL interface!");

            ProjectEx project = getReinforcementProjectNoisyNet(igym, nIterations, bLoadWeightsIfExists, strTrainerType);
            DatasetDescriptor ds = itrainer.GetDatasetOverride(0);

            if (strAtariRom == null)
                strAtariRom = "pong";

            string strRom = getRomPath(strAtariRom + ".bin");

            m_log.CHECK(ds != null, "The MyCaffeAtariTrainer should return its dataset override returned by the Gym that it uses.");

            // Train the network using the custom trainer
            //  - Iterations (maximum frames cumulative across all threads) = 1000 (normally this would be much higher such as 500,000)
            //  - Learning rate = 0.001 (defined in solver.prototxt)
            //  - Mini Batch Size = 10 (defined in train_val.prototxt for MemoryDataLayer)
            //
            //  - TrainerType = 'C51.ST' ('C51.ST' = use single-threaded C51 trainer)
            //  - RewardType = MAX (display the maximum rewards received, a setting of VAL displays the actual reward received)
            //  - Gamma = 0.99 (discounting factor)
            //  - Threads = 1 (only use 1 thread if multi-threading is supported)
            //  - AllowNegativeRewards = False (when enabled and the ball falls behind our player, a -1 reward is given).
            //  - TerminateOnRallyEnd = False (when enabled a termination state is given each time the ball falls behind our player).
            //  - GameROM = 'path to game ROM'
            string strParam = "TrainerType=" + strTrainerType + ";RewardType=VAL;Gamma=0.99;";
            strParam += "UseRawInput=True;";        // use the input values directly, do not take a difference of them with the previous.
            strParam += "Preprocess=False;";        // do not preprocess (turn inputs to 1 or 0).
            strParam += "ActionForceGray=True;";    // force inputs to gray with a single color channel.
            strParam += "FrameSkip=1;";             // process one frame of data at a time.
            strParam += "AllowNegativeRewards=" + bAllowNegRewards.ToString() + ";";    // receive -1 reward on rally's where ball not even hit.
            strParam += "TerminateOnRallyEnd=" + bTerminateOnRallyEnd.ToString() + ";"; // play only one rally at a time then restart the game.
            strParam += "EpsSteps=" + nIterations.ToString() + ";EpsStart=1.0;EpsEnd=0.01;";
            strParam += "GameROM=" + strRom;
            itrainer.Initialize(strParam, this);

            // load the project to train (note the project must use the MemoryDataLayer for input).
            mycaffe.Load(Phase.TRAIN, project, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.NONE, false, null, false, true, itrainer.Stage.ToString());

            if (bShowUi)
                itrainer.OpenUi();

            m_nMaxIteration = nIterations;
            itrainer.Train(mycaffe, nIterations, (ITERATOR_TYPE)iteratorType);
            itrainer.CleanUp();

            // Release the mycaffe resources.
            mycaffe.Dispose();
        }

        public void TrainCharRNN(bool bDual, bool bShowUi, string strTrainerType, LayerParameter.LayerType lstm, int nIterations = 100, bool bUseAcceleratedTraining = false, bool bAllowDiscountReset = false)
        {
            if (bDual)
                TrainCharRNNDual(bShowUi, strTrainerType, lstm, nIterations, bUseAcceleratedTraining, bAllowDiscountReset);
            else
                TrainCharRNN(bShowUi, strTrainerType, lstm, nIterations, bUseAcceleratedTraining, bAllowDiscountReset);
        }

        public void TrainCharRNN(bool bShowUi, string strTrainerType, LayerParameter.LayerType lstm, int nIterations = 100, bool bUseAcceleratedTraining = false, bool bAllowDiscountReset = false)
        {
            m_evtCancel.Reset();

            GymCollection col = new GymCollection();
            col.Load();
            IXMyCaffeGymData igym = col.Find("DataGeneral") as IXMyCaffeGymData;
            m_log.CHECK(igym != null, "The 'DataGeneral' gym should implement the IXMyCaffeGymData interface.");

            string strAccelTrain = (bUseAcceleratedTraining) ? "ON" : "OFF";

            if (strTrainerType != "RNN.SIMPLE")
                throw new Exception("Currently, only the RNN.SIMPLE is supported.");

            m_log.WriteHeader("Test Training CharRNN for " + nIterations.ToString("N0") + " iterations.");
            m_log.WriteLine("Using trainer = " + strTrainerType + ", Accelerated Training = " + strAccelTrain);
            MyCaffeControl<T> mycaffe = new MyCaffeControl<T>(m_settings, m_log, m_evtCancel);
            mycaffe.OnSnapshot += Mycaffe_OnSnapshot;

            string strModelPath = getTestPath("\\MyCaffe\\test_data\\models\\rnn\\char_rnn\\" + lstm.ToString().ToLower(), true);

            MyCaffeDataGeneralTrainer trainer = new MyCaffeDataGeneralTrainer();
            ProjectEx project = getCharRNNProject(igym, nIterations, strModelPath, m_engine);
            DatasetDescriptor ds = trainer.GetDatasetOverride(0);

            m_log.CHECK(ds != null, "The MyCaffeDataTrainer should return its dataset override returned by the Gym that it uses.");

            string strEngine = m_engine.ToString();
            string strWeights = strModelPath + "\\weights." + strEngine + ".mycaffemodel";
            if (File.Exists(strWeights))
            {
                using (FileStream fs = File.OpenRead(strWeights))
                using (BinaryReader bw = new BinaryReader(fs))
                {
                    if (fs.Length > 0)
                    {
                        byte[] rgWeights = new byte[fs.Length];
                        bw.Read(rgWeights, 0, (int)fs.Length);
                        project.WeightsState = rgWeights;
                    }
                }
            }
            m_strModelPath = strModelPath;

            // Train the network using the custom trainer
            //  - Iterations (maximum frames cumulative across all threads) = 1000 (normally this would be much higher such as 500,000)
            //  - Learning rate = 0.05 (defined in solver.prototxt)
            //  - Mini Batch Size = 25 for LSTM, 1 for LSTM_SIMPLE (defined in train_val.prototxt for InputLayer)
            //
            //  - TrainerType = 'RNN.SIMPLE' (currently only one supported)
            //  - UseAcceleratedTraining = False (disable accelerated training).
            //  - ConnectionCount=1 (using one query)
            //  - Connection0_CustomQueryName=StdTextFileQuery (using standard text file query to read the text files)
            //  - Connection0_CustomQueryParam=params (set the custom query parameters to the packed parameters containing the FilePath where the text files are to be loaded).
            string strSchema = "ConnectionCount=1;";
            string strDataPath = getTestPath("\\MyCaffe\\test_data\\data\\char-rnn", true);
            string strParam = "FilePath=" + strDataPath + ";";

            strParam = ParamPacker.Pack(strParam);
            strSchema += "Connection0_CustomQueryName=StdTextFileQuery;";
            strSchema += "Connection0_CustomQueryParam=" + strParam + ";";

            string strProp = "TrainerType=" + strTrainerType + ";UseAcceleratedTraining=" + bUseAcceleratedTraining.ToString() + ";" + strSchema;
            trainer.Initialize(strProp, this);

            BucketCollection rgVocabulary = trainer.PreloadData(m_log, m_evtCancel, 0);
            project.ModelDescription = trainer.ResizeModel(m_log, project.ModelDescription, rgVocabulary);

            // load the project to train (note the project must use the InputLayer for input).
            mycaffe.Load(Phase.TRAIN, project, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.NONE, false, null, false);

            if (bShowUi)
                trainer.OpenUi();

            m_nMaxIteration = nIterations;
            trainer.Train(mycaffe, nIterations);

            string type;
            int nN = 1000; // Note: see iterations used, for real training the iterations should be 100,000+
            byte[] rgOutput = trainer.Run(mycaffe, nN, out type); // For Run Parameters, see GetRunProperties() callback below.
            m_log.CHECK(type == "String", "The output type should be a string type!");
            string strOut;

            using (MemoryStream ms = new MemoryStream(rgOutput))
            {
                strOut = Encoding.ASCII.GetString(ms.ToArray());
            }

            m_log.WriteLine(strOut);

            string strOutputFile = strModelPath + "\\output" + ((typeof(T) == typeof(float)) ? "F" : "D") + ".txt";
            if (File.Exists(strOutputFile))
                File.Delete(strOutputFile);

            using (StreamWriter sw = new StreamWriter(strOutputFile))
            {
                sw.Write(strOut);
            }

            trainer.CleanUp();

            // Release the mycaffe resources.
            mycaffe.Dispose();
        }

        public void TrainCharRNNDual(bool bShowUi, string strTrainerType, LayerParameter.LayerType lstm, int nIterations = 100, bool bUseAcceleratedTraining = false, bool bAllowDiscountReset = false)
        {
            m_evtCancel.Reset();

            GymCollection col = new GymCollection();
            col.Load();
            IXMyCaffeGymData igym = col.Find("DataGeneral") as IXMyCaffeGymData;
            m_log.CHECK(igym != null, "The 'DataGeneral' gym should implement the IXMyCaffeGymData interface.");

            string strAccelTrain = (bUseAcceleratedTraining) ? "ON" : "OFF";

            if (strTrainerType != "RNN.SIMPLE")
                throw new Exception("Currently, only the RNN.SIMPLE is supported.");

            m_log.WriteHeader("Test Training CharRNN for " + nIterations.ToString("N0") + " iterations.");
            m_log.WriteLine("Using trainer = " + strTrainerType + ", Accelerated Training = " + strAccelTrain);
            MyCaffeControl<T> mycaffe = new MyCaffeControl<T>(m_settings, m_log, m_evtCancel);
            mycaffe.OnSnapshot += Mycaffe_OnSnapshot;

            string strModelPath = getTestPath("\\MyCaffe\\test_data\\models\\rnn\\char_rnn\\" + lstm.ToString().ToLower(), true);

            MyCaffeDataGeneralTrainerDual trainerX = new MyCaffeDataGeneralTrainerDual();
            ProjectEx project = getCharRNNProject(igym, nIterations, strModelPath, m_engine);

            IXMyCaffeCustomTrainerRNN itrainer = trainerX as IXMyCaffeCustomTrainerRNN;
            if (itrainer == null)
                throw new Exception("The trainer must implement the IXMyCaffeCustomTrainerRNN interface!");

            DatasetDescriptor ds = itrainer.GetDatasetOverride(0);

            m_log.CHECK(ds != null, "The MyCaffeDataTrainer should return its dataset override returned by the Gym that it uses.");

            string strEngine = m_engine.ToString();
            string strWeights = strModelPath + "\\weights." + strEngine + ".mycaffemodel";
            if (File.Exists(strWeights))
            {
                using (FileStream fs = File.OpenRead(strWeights))
                using (BinaryReader bw = new BinaryReader(fs))
                {
                    if (fs.Length > 0)
                    {
                        byte[] rgWeights = new byte[fs.Length];
                        bw.Read(rgWeights, 0, (int)fs.Length);
                        project.WeightsState = rgWeights;
                    }
                }
            }
            m_strModelPath = strModelPath;

            // Train the network using the custom trainer
            //  - Iterations (maximum frames cumulative across all threads) = 1000 (normally this would be much higher such as 500,000)
            //  - Learning rate = 0.05 (defined in solver.prototxt)
            //  - Mini Batch Size = 25 for LSTM, 1 for LSTM_SIMPLE (defined in train_val.prototxt for InputLayer)
            //
            //  - TrainerType = 'RNN.SIMPLE' (currently only one supported)
            //  - UseAcceleratedTraining = False (disable accelerated training).
            //  - ConnectionCount=1 (using one query)
            //  - Connection0_CustomQueryName=StdTextFileQuery (using standard text file query to read the text files)
            //  - Connection0_CustomQueryParam=params (set the custom query parameters to the packed parameters containing the FilePath where the text files are to be loaded).
            string strSchema = "ConnectionCount=1;";
            string strDataPath = getTestPath("\\MyCaffe\\test_data\\data\\char-rnn", true);
            string strParam = "FilePath=" + strDataPath + ";";

            strParam = ParamPacker.Pack(strParam);
            strSchema += "Connection0_CustomQueryName=StdTextFileQuery;";
            strSchema += "Connection0_CustomQueryParam=" + strParam + ";";

            string strProp = "TrainerType=" + strTrainerType + ";UseAcceleratedTraining=" + bUseAcceleratedTraining.ToString() + ";" + strSchema;
            itrainer.Initialize(strProp, this);

            BucketCollection rgVocabulary = itrainer.PreloadData(m_log, m_evtCancel, 0);
            project.ModelDescription = itrainer.ResizeModel(m_log, project.ModelDescription, rgVocabulary);

            // load the project to train (note the project must use the InputLayer for input).
            mycaffe.Load(Phase.TRAIN, project, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.NONE, false, null, false, true, itrainer.Stage.ToString());

            if (bShowUi)
                itrainer.OpenUi();

            m_nMaxIteration = nIterations;
            itrainer.Train(mycaffe, nIterations);

            string type;
            int nN = 1000; // Note: see iterations used, for real training the iterations should be 100,000+
            byte[] rgOutput = itrainer.Run(mycaffe, nN, out type); // For Run Parameters, see GetRunProperties() callback below.
            m_log.CHECK(type == "String", "The output type should be a string type!");
            string strOut;

            using (MemoryStream ms = new MemoryStream(rgOutput))
            {
                strOut = Encoding.ASCII.GetString(ms.ToArray());
            }

            m_log.WriteLine(strOut);

            string strOutputFile = strModelPath + "\\output" + ((typeof(T) == typeof(float)) ? "F" : "D") + ".txt";
            if (File.Exists(strOutputFile))
                File.Delete(strOutputFile);

            using (StreamWriter sw = new StreamWriter(strOutputFile))
            {
                sw.Write(strOut);
            }

            itrainer.CleanUp();

            // Release the mycaffe resources.
            mycaffe.Dispose();
        }


        public void TrainWavRNN(bool bDual, bool bShowUi, string strTrainerType, LayerParameter.LayerType lstm, int nIterations = 100, bool bUseAcceleratedTraining = false, bool bAllowDiscountReset = false)
        {
            if (bDual)
                TrainWavRNNDual(bShowUi, strTrainerType, lstm, nIterations, bUseAcceleratedTraining, bAllowDiscountReset);
            else
                TrainWavRNN(bShowUi, strTrainerType, lstm, nIterations, bUseAcceleratedTraining, bAllowDiscountReset);
        }

        public void TrainWavRNN(bool bShowUi, string strTrainerType, LayerParameter.LayerType lstm, int nIterations = 100, bool bUseAcceleratedTraining = false, bool bAllowDiscountReset = false)
        {
            m_evtCancel.Reset();

            GymCollection col = new GymCollection();
            col.Load();
            IXMyCaffeGymData igym = col.Find("DataGeneral") as IXMyCaffeGymData;
            m_log.CHECK(igym != null, "The 'DataGeneral' gym should implement the IXMyCaffeGymData interface.");

            string strAccelTrain = (bUseAcceleratedTraining) ? "ON" : "OFF";

            if (strTrainerType != "RNN.SIMPLE")
                throw new Exception("Currently, only the RNN.SIMPLE is supported.");

            m_log.WriteHeader("Test Training CharRNN for " + nIterations.ToString("N0") + " iterations.");
            m_log.WriteLine("Using trainer = " + strTrainerType + ", Accelerated Training = " + strAccelTrain);
            MyCaffeControl<T> mycaffe = new MyCaffeControl<T>(m_settings, m_log, m_evtCancel);
            mycaffe.OnSnapshot += Mycaffe_OnSnapshot;

            string strModelPath = getTestPath("\\MyCaffe\\test_data\\models\\rnn\\wav\\" + lstm.ToString().ToLower(), true);

            MyCaffeDataGeneralTrainer trainer = new MyCaffeDataGeneralTrainer();
            ProjectEx project = getCharRNNProject(igym, nIterations, strModelPath, EngineParameter.Engine.DEFAULT);
            DatasetDescriptor ds = trainer.GetDatasetOverride(0);

            m_log.CHECK(ds != null, "The MyCaffeDataTrainer should return its dataset override returned by the Gym that it uses.");

            string strEngine = m_engine.ToString();
            string strWeights = strModelPath + "\\weights." + strEngine + ".mycaffemodel";
            if (File.Exists(strWeights))
            {
                using (FileStream fs = File.OpenRead(strWeights))
                using (BinaryReader bw = new BinaryReader(fs))
                {
                    if (fs.Length > 0)
                    {
                        byte[] rgWeights = new byte[fs.Length];
                        bw.Read(rgWeights, 0, (int)fs.Length);
                        project.WeightsState = rgWeights;
                    }
                }
            }
            m_strModelPath = strModelPath;

            // Train the network using the custom trainer
            //  - Iterations (maximum frames cumulative across all threads) = 1000 (normally this would be much higher such as 500,000)
            //  - Learning rate = 0.05 (defined in solver.prototxt)
            //  - Mini Batch Size = 25 for LSTM, 1 for LSTM_SIMPLE (defined in train_val.prototxt for InputLayer)
            //
            //  - TrainerType = 'RNN.SIMPLE' (currently only one supported)
            //  - UseAcceleratedTraining = False (disable accelerated training).
            //  - ConnectionCount=1 (using one query)
            //  - Connection0_CustomQueryName=StdWAVFileQuery (using standard text file query to read the text files)
            //  - Connection0_CustomQueryParam=params (set the custom query parameters to the packed parameters containing the FilePath where the text files are to be loaded).
            string strSchema = "ConnectionCount=1;";
            string strDataPath = getTestPath("\\MyCaffe\\test_data\\data\\wav", true);
            string strParam = "FilePath=" + strDataPath + ";";

            strParam = ParamPacker.Pack(strParam);
            strSchema += "Connection0_CustomQueryName=StdWAVFileQuery;";
            strSchema += "Connection0_CustomQueryParam=" + strParam + ";";

            string strProp = "TrainerType=" + strTrainerType + ";UseAcceleratedTraining=" + bUseAcceleratedTraining.ToString() + ";Temperature=0.5;" + strSchema;
            trainer.Initialize(strProp, this);

            BucketCollection rgVocabulary = trainer.PreloadData(m_log, m_evtCancel, 0);
            project.ModelDescription = trainer.ResizeModel(m_log, project.ModelDescription, rgVocabulary);

            // load the project to train (note the project must use the InputLayer for input).
            mycaffe.Load(Phase.TRAIN, project, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.NONE, false, null, false);

            if (bShowUi)
                trainer.OpenUi();

            m_nMaxIteration = nIterations;
            trainer.Train(mycaffe, nIterations);

            string type;
            int nN = 1000; // Note: see iterations used, for real training the iterations should be 100,000+
            byte[] rgOutput = trainer.Run(mycaffe, nN, out type);  // For Run Parameters, see GetRunProperties() callback below.
            m_log.CHECK(type == "WAV", "The output type should be a WAV type!");

            WaveFormat fmt;
            List<double[]> rgrgSamples = StandardQueryWAVFile.UnPackBytes(rgOutput, out fmt);

            string strOutputFile = strModelPath + "\\output.wav";
            using (FileStream fs = File.OpenWrite(strOutputFile))
            using (WAVWriter wav = new WAVWriter(fs))
            {
                wav.Format = fmt;
                wav.Samples = rgrgSamples;
                wav.WriteAll();
            }

            trainer.CleanUp();

            // Release the mycaffe resources.
            mycaffe.Dispose();
        }

        public void TrainWavRNNDual(bool bShowUi, string strTrainerType, LayerParameter.LayerType lstm, int nIterations = 100, bool bUseAcceleratedTraining = false, bool bAllowDiscountReset = false)
        {
            m_evtCancel.Reset();

            GymCollection col = new GymCollection();
            col.Load();
            IXMyCaffeGymData igym = col.Find("DataGeneral") as IXMyCaffeGymData;
            m_log.CHECK(igym != null, "The 'DataGeneral' gym should implement the IXMyCaffeGymData interface.");

            string strAccelTrain = (bUseAcceleratedTraining) ? "ON" : "OFF";

            if (strTrainerType != "RNN.SIMPLE")
                throw new Exception("Currently, only the RNN.SIMPLE is supported.");

            m_log.WriteHeader("Test Training CharRNN for " + nIterations.ToString("N0") + " iterations.");
            m_log.WriteLine("Using trainer = " + strTrainerType + ", Accelerated Training = " + strAccelTrain);
            MyCaffeControl<T> mycaffe = new MyCaffeControl<T>(m_settings, m_log, m_evtCancel);
            mycaffe.OnSnapshot += Mycaffe_OnSnapshot;

            string strModelPath = getTestPath("\\MyCaffe\\test_data\\models\\rnn\\wav\\" + lstm.ToString().ToLower(), true);

            MyCaffeDataGeneralTrainerDual trainerX = new MyCaffeDataGeneralTrainerDual();
            ProjectEx project = getCharRNNProject(igym, nIterations, strModelPath, EngineParameter.Engine.DEFAULT);

            IXMyCaffeCustomTrainerRNN itrainer = trainerX as IXMyCaffeCustomTrainerRNN;
            if (itrainer == null)
                throw new Exception("The trainer must implement the IXMyCaffeCustomTrainerRNN interface!");

            DatasetDescriptor ds = itrainer.GetDatasetOverride(0);

            m_log.CHECK(ds != null, "The MyCaffeDataTrainer should return its dataset override returned by the Gym that it uses.");

            string strEngine = m_engine.ToString();
            string strWeights = strModelPath + "\\weights." + strEngine + ".mycaffemodel";
            if (File.Exists(strWeights))
            {
                using (FileStream fs = File.OpenRead(strWeights))
                using (BinaryReader bw = new BinaryReader(fs))
                {
                    if (fs.Length > 0)
                    {
                        byte[] rgWeights = new byte[fs.Length];
                        bw.Read(rgWeights, 0, (int)fs.Length);
                        project.WeightsState = rgWeights;
                    }
                }
            }
            m_strModelPath = strModelPath;

            // Train the network using the custom trainer
            //  - Iterations (maximum frames cumulative across all threads) = 1000 (normally this would be much higher such as 500,000)
            //  - Learning rate = 0.05 (defined in solver.prototxt)
            //  - Mini Batch Size = 25 for LSTM, 1 for LSTM_SIMPLE (defined in train_val.prototxt for InputLayer)
            //
            //  - TrainerType = 'RNN.SIMPLE' (currently only one supported)
            //  - UseAcceleratedTraining = False (disable accelerated training).
            //  - ConnectionCount=1 (using one query)
            //  - Connection0_CustomQueryName=StdWAVFileQuery (using standard text file query to read the text files)
            //  - Connection0_CustomQueryParam=params (set the custom query parameters to the packed parameters containing the FilePath where the text files are to be loaded).
            string strSchema = "ConnectionCount=1;";
            string strDataPath = getTestPath("\\MyCaffe\\test_data\\data\\wav", true);
            string strParam = "FilePath=" + strDataPath + ";";

            strParam = ParamPacker.Pack(strParam);
            strSchema += "Connection0_CustomQueryName=StdWAVFileQuery;";
            strSchema += "Connection0_CustomQueryParam=" + strParam + ";";

            string strProp = "TrainerType=" + strTrainerType + ";UseAcceleratedTraining=" + bUseAcceleratedTraining.ToString() + ";Temperature=0.5;" + strSchema;
            itrainer.Initialize(strProp, this);

            BucketCollection rgVocabulary = itrainer.PreloadData(m_log, m_evtCancel, 0);
            project.ModelDescription = itrainer.ResizeModel(m_log, project.ModelDescription, rgVocabulary);

            // load the project to train (note the project must use the InputLayer for input).
            mycaffe.Load(Phase.TRAIN, project, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.NONE, false, null, false, true, itrainer.Stage.ToString());

            if (bShowUi)
                itrainer.OpenUi();

            m_nMaxIteration = nIterations;
            itrainer.Train(mycaffe, nIterations);

            string type;
            int nN = 1000; // Note: see iterations used, for real training the iterations should be 100,000+
            byte[] rgOutput = itrainer.Run(mycaffe, nN, out type);  // For Run Parameters, see GetRunProperties() callback below.
            m_log.CHECK(type == "WAV", "The output type should be a WAV type!");

            WaveFormat fmt;
            List<double[]> rgrgSamples = StandardQueryWAVFile.UnPackBytes(rgOutput, out fmt);

            string strOutputFile = strModelPath + "\\output.wav";
            using (FileStream fs = File.OpenWrite(strOutputFile))
            using (WAVWriter wav = new WAVWriter(fs))
            {
                wav.Format = fmt;
                wav.Samples = rgrgSamples;
                wav.WriteAll();
            }

            itrainer.CleanUp();

            // Release the mycaffe resources.
            mycaffe.Dispose();
        }

        private void Mycaffe_OnSnapshot(object sender, SnapshotArgs e)
        {
            byte[] rgWeights = e.UpdateWeights();
            string strWeights = m_strModelPath + "\\weights." + m_engine.ToString() + ".mycaffemodel";

            using (new SingleGlobalInstance(0, false))
            {
                if (File.Exists(strWeights))
                    File.Delete(strWeights);

                using (FileStream fs = File.Open(strWeights, FileMode.OpenOrCreate))
                using (BinaryWriter bw = new BinaryWriter(fs))
                {
                    bw.Write(rgWeights);
                }
            }
        }

        private ProjectEx getReinforcementProject(IXMyCaffeGym igym, int nIterations, DATA_TYPE dt = DATA_TYPE.VALUES, bool bForceSimple = false)
        {
            ProjectEx p = new ProjectEx("test");

            string strModelFile = getTestPath("\\MyCaffe\\test_data\\models\\reinforcement\\cartpole\\train_val.prototxt");
            string strSolverFile = getTestPath("\\MyCaffe\\test_data\\models\\reinforcement\\cartpole\\solver.prototxt");

            if (bForceSimple)
                strModelFile = getTestPath("\\MyCaffe\\test_data\\models\\reinforcement\\cartpole\\train_val_sigmoid.prototxt");

            if (dt == DATA_TYPE.BLOB)
            {
                strModelFile = getTestPath("\\MyCaffe\\test_data\\models\\reinforcement\\atari\\train_val.prototxt");
                strSolverFile = getTestPath("\\MyCaffe\\test_data\\models\\reinforcement\\atari\\solver.prototxt");
            }

            RawProto protoM = RawProtoFile.LoadFromFile(strModelFile);
            p.ModelDescription = protoM.ToString();

            RawProto protoS = RawProtoFile.LoadFromFile(strSolverFile);
            RawProto iter = protoS.FindChild("max_iter");
            iter.Value = nIterations.ToString();

            p.SolverDescription = protoS.ToString();
            p.SetDataset(igym.GetDataset(dt));

            return p;
        }

        private ProjectEx getReinforcementProjectC51(IXMyCaffeGym igym, int nIterations, bool bLoadWeightsIfExist, bool bVersionB)
        {
            ProjectEx p = new ProjectEx("test");
            string strVer = (bVersionB) ? "b" : "";

            string strModelFile = getTestPath("\\MyCaffe\\test_data\\models\\reinforcement\\atari.c51\\train_val" + strVer + ".prototxt");
            string strSolverFile = getTestPath("\\MyCaffe\\test_data\\models\\reinforcement\\atari.c51\\solver.prototxt");

            RawProto protoM = RawProtoFile.LoadFromFile(strModelFile);
            p.ModelDescription = protoM.ToString();

            RawProto protoS = RawProtoFile.LoadFromFile(strSolverFile);
            RawProto iter = protoS.FindChild("max_iter");
            iter.Value = nIterations.ToString();

            p.SolverDescription = protoS.ToString();
            p.SetDataset(igym.GetDataset(DATA_TYPE.BLOB));

            m_strModelPath = Path.GetDirectoryName(strModelFile);

            if (bLoadWeightsIfExist)
            {
                string strWeights = m_strModelPath + "\\weights" + strVer + "." + m_engine.ToString() + ".mycaffemodel";

                if (File.Exists(strWeights))
                {
                    using (FileStream fs = new FileStream(strWeights, FileMode.Open))
                    using (BinaryReader br = new BinaryReader(fs))
                    {
                        p.WeightsState = br.ReadBytes((int)fs.Length);
                    }
                }
            }

            return p;
        }

        private ProjectEx getReinforcementProjectNoisyNet(IXMyCaffeGym igym, int nIterations, bool bLoadWeightsIfExist, string strTrainerType)
        {
            ProjectEx p = new ProjectEx("test");

            string strType = (strTrainerType.Contains("SIMPLE") ? "cartpole.noisy.dqn" : "atari.noisy.dqn");
            string strModelFile = getTestPath("\\MyCaffe\\test_data\\models\\reinforcement\\" + strType + "\\train_val.prototxt");
            string strSolverFile = getTestPath("\\MyCaffe\\test_data\\models\\reinforcement\\" + strType + "\\solver.prototxt");

            RawProto protoM = RawProtoFile.LoadFromFile(strModelFile);
            p.ModelDescription = protoM.ToString();

            RawProto protoS = RawProtoFile.LoadFromFile(strSolverFile);
            RawProto iter = protoS.FindChild("max_iter");
            iter.Value = nIterations.ToString();

            p.SolverDescription = protoS.ToString();
            p.SetDataset(igym.GetDataset(DATA_TYPE.BLOB));

            m_strModelPath = Path.GetDirectoryName(strModelFile);

            if (bLoadWeightsIfExist)
            {
                string strWeights = m_strModelPath + "\\weights." + m_engine.ToString() + ".mycaffemodel";

                if (File.Exists(strWeights))
                {
                    using (FileStream fs = new FileStream(strWeights, FileMode.Open))
                    using (BinaryReader br = new BinaryReader(fs))
                    {
                        p.WeightsState = br.ReadBytes((int)fs.Length);
                    }
                }
            }

            return p;
        }

        private string getRomPath(string strRom)
        {
            return getTestPath("\\MyCaffe\\test_data\\roms\\" + strRom);
        }

        private ProjectEx getCharRNNProject(IXMyCaffeGym igym, int nIterations, string strPath, EngineParameter.Engine engine)
        {
            ProjectEx p = new ProjectEx("test");

            string strModelFile = strPath + "\\train_val.prototxt";
            string strSolverFile = strPath + "\\solver.prototxt";

            RawProto protoM = RawProtoFile.LoadFromFile(strModelFile);

            if (engine == EngineParameter.Engine.CUDNN)
            {
                NetParameter net_param = NetParameter.FromProto(protoM);

                for (int i = 0; i < net_param.layer.Count; i++)
                {
                    if (net_param.layer[i].type == LayerParameter.LayerType.LSTM)
                    {
                        net_param.layer[i].recurrent_param.engine = engine;
                    }
                }

                protoM = net_param.ToProto("root");
            }

            p.ModelDescription = protoM.ToString();

            RawProto protoS = RawProtoFile.LoadFromFile(strSolverFile);
            RawProto iter = protoS.FindChild("max_iter");
            iter.Value = nIterations.ToString();

            p.SolverDescription = protoS.ToString();
            p.SetDataset(igym.GetDataset(DATA_TYPE.BLOB));

            return p;
        }

        /// <summary>
        /// Get the properties to use during each call to Run.
        /// </summary>
        /// <returns>The properties are returned as a set of key=value pairs.</returns>
        /// <remarks>
        /// For now, the TRAIN phase is used during the Run (e.g. the TRAIN network)
        /// due to a bug in sharing the weights between the TRAIN and TEST/RUN networks.
        /// </remarks>
        public string GetRunProperties()
        {
            return "";
        }

        public void Update(TRAINING_CATEGORY cat, Dictionary<string, double> rgValues)
        {
            if (rgValues.ContainsKey("GlobalIteration"))
            {
                int nIteration = (int)rgValues["GlobalIteration"];

                if (m_nMaxIteration > 0)
                {
                    double dfProgress = (int)nIteration / (double)m_nMaxIteration;
                    m_progress.SetProgress(dfProgress);
                }
            }
        }
    }

    class SingleGlobalInstance : IDisposable
    {
        public bool m_hasHandle = false;
        Mutex m_mutex;

        private void InitMutex()
        {
            string appGuid = ((GuidAttribute)Assembly.GetExecutingAssembly().GetCustomAttributes(typeof(GuidAttribute), false).GetValue(0)).Value;
            string mutexId = string.Format("Global\\{{{0}}}", appGuid);
            m_mutex = new Mutex(false, mutexId);

            var allowEveryoneRule = new MutexAccessRule(new SecurityIdentifier(WellKnownSidType.WorldSid, null), MutexRights.FullControl, AccessControlType.Allow);
            var securitySettings = new MutexSecurity();
            securitySettings.AddAccessRule(allowEveryoneRule);
            m_mutex.SetAccessControl(securitySettings);
        }

        public SingleGlobalInstance(int timeOut, bool bThrowException)
        {
            InitMutex();
            try
            {
                if (timeOut < 0)
                    m_hasHandle = m_mutex.WaitOne(Timeout.Infinite, false);
                else
                    m_hasHandle = m_mutex.WaitOne(timeOut, false);

                if (m_hasHandle == false && bThrowException)
                    throw new TimeoutException("Timeout waiting for exclusive access on SingleInstance");
            }
            catch (AbandonedMutexException)
            {
                m_hasHandle = true;
            }
        }


        public void Dispose()
        {
            if (m_mutex != null)
            {
                if (m_hasHandle)
                    m_mutex.ReleaseMutex();

                m_mutex.Close();
                m_mutex = null;
            }
        }
    }

    class MyCaffeCartPoleTrainer : MyCaffeTrainerRL, IXMyCaffeGymUiCallback
    {
        Stopwatch m_sw = new Stopwatch();        
        IXMyCaffeGym m_igym;
        Log m_log;
        bool m_bNormalizeInput = false;
        int m_nUiId = -1;
        MyCaffeGymUiProxy m_gymui = null;
        string m_strName = "Cart-Pole";
        GymCollection m_colGyms = new GymCollection();
        EventWaitHandle m_evtOpenUi = new EventWaitHandle(false, EventResetMode.AutoReset, "_MyCaffeTrainer_OpenUi_");

        public MyCaffeCartPoleTrainer() 
            : base()
        {
            m_colGyms.Load();
        }

        protected override void initialize(InitializeArgs e)
        {
            m_igym = m_colGyms.Find(m_strName);
            m_log = e.OutputLog;

            m_bNormalizeInput = m_properties.GetPropertyAsBool("NormalizeInput", false);
            m_igym.Initialize(m_log, m_properties);

            m_sw.Start();
        }

        protected override void shutdown()
        {
            if (m_igym != null)
            {
                m_igym.Close();
                m_igym = null;
            }
        }

        protected override void dispose()
        {
            base.dispose();
        }

        protected override string name
        {
            get { return "RL.Trainer"; }
        }

        protected override DatasetDescriptor get_dataset_override(int nProjectID)
        {
            if (m_igym == null)
                m_igym = m_colGyms.Find(m_strName);

            return m_igym.GetDataset(DATA_TYPE.VALUES);
        }

        protected override bool getData(GetDataArgs e)
        {
            Tuple<State, double, bool> state = null;

            if (e.Reset)
                state = m_igym.Reset();

            if (e.Action >= 0)
                state = m_igym.Step(e.Action);

            bool bIsOpen = (m_nUiId >= 0) ? true : false;
            Tuple<Bitmap, SimpleDatum> data = m_igym.Render(bIsOpen, 512, 512, true);
            int nDataLen = 0;
            SimpleDatum stateData = state.Item1.GetData(m_bNormalizeInput, out nDataLen);
            Observation obs = new Observation(null, ImageData.GetImage(data.Item2), m_igym.RequiresDisplayImage, stateData.RealData, state.Item2, state.Item3);

            e.State = new StateBase(m_igym.GetActionSpace().Count());
            e.State.Reward = obs.Reward;
            e.State.Data = new SimpleDatum(true, nDataLen, 1, 1, -1, DateTime.Now, null, stateData.RealData.ToList(), 0, false, 0);
            e.State.Done = obs.Done;
            e.State.IsValid = true;

            if (m_gymui != null && m_nUiId >= 0)
            {
                m_gymui.Render(m_nUiId, obs);
                Thread.Sleep(m_igym.UiDelay);
            }
            else
            {
                if (m_evtOpenUi.WaitOne(0))
                    openUi();
            }

            if (m_sw.Elapsed.TotalMilliseconds > 1000)
            {
                double dfPct = (GlobalEpisodeMax == 0) ? 0 : (double)GlobalEpisodeCount / (double)GlobalEpisodeMax;
                e.OutputLog.Progress = dfPct;
                e.OutputLog.WriteLine("(" + dfPct.ToString("P") + ") Global Episode #" + GlobalEpisodeCount.ToString() + "  Global Reward = " + GlobalRewards.ToString() + " Exploration Rate = " + ExplorationRate.ToString("P") + " Optimal Selection Rate = " + OptimalSelectionRate.ToString("P"));
                m_sw.Restart();
            }

            return true;
        }

        protected override void openUi()
        {
            m_gymui = new MyCaffeGymUiProxy(new InstanceContext(this));
            m_gymui.Open();

            string strName = m_strName;
            string strTrainer = m_properties.GetProperty("TrainerType");
            if (!string.IsNullOrEmpty(strTrainer))
                strName += ": " + strTrainer;

            m_nUiId = m_gymui.OpenUi(strName, m_nUiId);
        }

        public void Closing()
        {
            m_nUiId = -1;
            m_gymui.Close();
            m_gymui = null;
        }
    }

    class MyCaffeAtariTrainer : MyCaffeTrainerRL, IXMyCaffeGymUiCallback
    {
        Stopwatch m_sw = new Stopwatch();
        IXMyCaffeGym m_igym;
        Log m_log;
        int m_nUiId = -1;
        MyCaffeGymUiProxy m_gymui = null;
        string m_strName = "ATARI";
        GymCollection m_colGyms = new GymCollection();
        DatasetDescriptor m_ds;
        EventWaitHandle m_evtOpenUi = new EventWaitHandle(false, EventResetMode.AutoReset, "_MyCaffeTrainer_OpenUi_");

        public MyCaffeAtariTrainer()
            : base()
        {
            m_colGyms.Load();
        }

        protected override void initialize(InitializeArgs e)
        {
            m_igym = m_colGyms.Find(m_strName);
            m_log = e.OutputLog;

            m_igym.Initialize(m_log, m_properties);

            m_sw.Start();
        }

        protected override void shutdown()
        {
            if (m_igym != null)
            {
                m_igym.Close();
                m_igym = null;
            }
        }

        protected override void dispose()
        {
            base.dispose();
        }

        protected override string name
        {
            get { return "RL.Trainer"; }
        }

        protected override DatasetDescriptor get_dataset_override(int nProjectID)
        {
            if (m_igym == null)
                m_igym = m_colGyms.Find(m_strName);

            m_ds = m_igym.GetDataset(DATA_TYPE.BLOB);

            return m_ds;
        }

        protected override bool getData(GetDataArgs e)
        {
            Tuple<State, double, bool> state = null;

            if (e.Reset)
                state = m_igym.Reset();

            if (e.Action >= 0)
                state = m_igym.Step(e.Action);

            bool bIsOpen = (m_nUiId >= 0) ? true : false;
            Tuple<Bitmap, SimpleDatum> data = m_igym.Render(bIsOpen, 512, 512, true);
            int nDataLen = 0;
            SimpleDatum stateData = state.Item1.GetData(false, out nDataLen);
            Observation obs = new Observation(data.Item1, ImageData.GetImage(data.Item2), m_igym.RequiresDisplayImage, stateData.RealData, state.Item2, state.Item3);

            e.State = new StateBase(m_igym.GetActionSpace().Count());
            e.State.Reward = obs.Reward;
            e.State.Data = data.Item2;
            e.State.Done = obs.Done;
            e.State.IsValid = true;

            if (m_gymui != null && m_nUiId >= 0)
            {
                m_gymui.Render(m_nUiId, obs);
                Thread.Sleep(m_igym.UiDelay);
            }
            else
            {
                if (m_evtOpenUi.WaitOne(0))
                    openUi();
            }

            if (m_sw.Elapsed.TotalMilliseconds > 1000)
            {
                double dfPct = (GlobalEpisodeMax == 0) ? 0 : (double)GlobalEpisodeCount / (double)GlobalEpisodeMax;
                e.OutputLog.Progress = dfPct;
                e.OutputLog.WriteLine("(" + dfPct.ToString("P") + ") Global Episode #" + GlobalEpisodeCount.ToString() + "  Global Reward = " + GlobalRewards.ToString() + " Exploration Rate = " + ExplorationRate.ToString("P") + " Optimal Selection Rate = " + OptimalSelectionRate.ToString("P"));
                m_sw.Restart();
            }

            return true;
        }

        protected override void openUi()
        {
            m_gymui = new MyCaffeGymUiProxy(new InstanceContext(this));
            m_gymui.Open();

            string strName = m_strName;
            string strTrainer = m_properties.GetProperty("TrainerType");
            if (!string.IsNullOrEmpty(strTrainer))
                strName += ": " + strTrainer;

            m_nUiId = m_gymui.OpenUi(strName, m_nUiId);
        }

        public void Closing()
        {
            m_nUiId = -1;
            m_gymui.Close();
            m_gymui = null;
        }
    }

    class MyCaffeDataGeneralTrainer : MyCaffeTrainerRNN, IXMyCaffeGymUiCallback
    {
        Stopwatch m_sw = new Stopwatch();
        IXMyCaffeGym m_igym;
        Log m_log;
        int m_nUiId = -1;
        string m_strName = "DataGeneral";
        GymCollection m_colGyms = new GymCollection();
        DatasetDescriptor m_ds;
        Tuple<State, double, bool> m_firststate = null;

        public MyCaffeDataGeneralTrainer()
            : base()
        {
            m_colGyms.Load();
        }

        protected override void initialize(InitializeArgs e)
        {
            initialize(e.OutputLog);
            m_sw.Start();
        }

        private void initialize(Log log)
        {
            if (m_igym == null)
            {
                m_log = log;
                m_igym = m_colGyms.Find(m_strName);
                m_igym.Initialize(m_log, m_properties);
            }
        }

        protected override void shutdown()
        {
            if (m_igym != null)
            {
                m_igym.Close();
                m_igym = null;
            }
        }

        protected override void dispose()
        {
            base.dispose();
        }

        protected override string name
        {
            get { return "RNN.Trainer"; }
        }

        protected override DatasetDescriptor get_dataset_override(int nProjectID)
        {
            IXMyCaffeGym igym = m_igym;

            if (igym == null)
                igym = m_colGyms.Find(m_strName);

            m_ds = igym.GetDataset(DATA_TYPE.BLOB);

            return m_ds;
        }

        protected override bool getData(GetDataArgs e)
        {
            Tuple<State, double, bool> state = null;

            if (e.Reset)
            {
                if (m_firststate != null)
                {
                    state = m_firststate;
                    m_firststate = null;
                }
                else
                {
                    state = m_igym.Reset();
                }
            }

            if (e.Action >= 0)
                state = m_igym.Step(e.Action);

            bool bIsOpen = (m_nUiId >= 0) ? true : false;
            int nDataLen = 0;
            SimpleDatum stateData = state.Item1.GetData(false, out nDataLen);

            e.State = new StateBase(m_igym.GetActionSpace().Count());
            e.State.Reward = 0;
            e.State.Data = stateData;
            e.State.Done = state.Item3;
            e.State.IsValid = true;

            if (m_sw.Elapsed.TotalMilliseconds > 1000)
            {
                int nMax = (int)GetProperty("GlobalMaxIterations");
                int nIteration = (int)GetProperty("GlobalIteration");
                double dfPct = (nMax == 0) ? 0 : (double)nIteration / (double)nMax;
                e.OutputLog.Progress = dfPct;
                e.OutputLog.WriteLine("(" + dfPct.ToString("P") + ") Global Iteration #" + nIteration.ToString());
                m_sw.Restart();
            }

            return true;
        }

        protected override bool convertOutput(ConvertOutputArgs e)
        {
            IXMyCaffeGymData igym = m_igym as IXMyCaffeGymData;
            if (igym == null)
                throw new Exception("Output data conversion requires a gym that implements the IXMyCaffeGymData interface.");

            string type;
            byte[] rgOutput = igym.ConvertOutput(Stage, e.Output.Length, e.Output, out type);
            e.SetRawOutput(rgOutput, type);

            return true;
        }

        protected override void openUi()
        {
        }

        protected override BucketCollection preloaddata(Log log, CancelEvent evtCancel, int nProjectID)
        {
            initialize(log);
            IXMyCaffeGymData igym = m_igym as IXMyCaffeGymData;
            Tuple<State, double, bool> state = igym.Reset();
            int nDataLen;
            SimpleDatum sd = state.Item1.GetData(false, out nDataLen);
            BucketCollection rgBucketCollection = null;

            if (sd.IsRealData)
            {
                // Create the vocabulary bucket collection.
                rgBucketCollection = BucketCollection.Bucketize("Building vocabulary", 128, sd, log, evtCancel);
                if (rgBucketCollection == null)
                    return null;
            }
            else
            {
                List<int> rgVocabulary = new List<int>();

                for (int i = 0; i < sd.ByteData.Length; i++)
                {
                    int nVal = (int)sd.ByteData[i];

                    if (!rgVocabulary.Contains(nVal))
                        rgVocabulary.Add(nVal);
                }

                rgBucketCollection = new BucketCollection(rgVocabulary);
            }

            m_firststate = state;

            return rgBucketCollection;
        }

        public void Closing()
        {
        }
    }

    class MyCaffeCartPoleTrainerDual : MyCaffeTrainerDual, IXMyCaffeGymUiCallback
    {
        Stopwatch m_sw = new Stopwatch();
        IXMyCaffeGym m_igym;
        Log m_log;
        bool m_bNormalizeInput = false;
        int m_nUiId = -1;
        MyCaffeGymUiProxy m_gymui = null;
        string m_strName = "Cart-Pole";
        GymCollection m_colGyms = new GymCollection();
        EventWaitHandle m_evtOpenUi = new EventWaitHandle(false, EventResetMode.AutoReset, "_MyCaffeTrainer_OpenUi_");

        public MyCaffeCartPoleTrainerDual()
            : base()
        {
            m_colGyms.Load();
        }

        protected override void initialize(InitializeArgs e)
        {
            m_igym = m_colGyms.Find(m_strName);
            m_log = e.OutputLog;

            m_bNormalizeInput = m_properties.GetPropertyAsBool("NormalizeInput", false);
            m_igym.Initialize(m_log, m_properties);

            m_sw.Start();
        }

        protected override void shutdown()
        {
            if (m_igym != null)
            {
                m_igym.Close();
                m_igym = null;
            }
        }

        protected override void dispose()
        {
            base.dispose();
        }

        protected override string name
        {
            get { return "RL.Trainer.Dual"; }
        }

        protected override DatasetDescriptor get_dataset_override(int nProjectID)
        {
            if (m_igym == null)
                m_igym = m_colGyms.Find(m_strName);

            return m_igym.GetDataset(DATA_TYPE.VALUES);
        }

        protected override bool getData(GetDataArgs e)
        {
            Tuple<State, double, bool> state = null;

            if (e.Reset)
                state = m_igym.Reset();

            if (e.Action >= 0)
                state = m_igym.Step(e.Action);

            bool bIsOpen = (m_nUiId >= 0) ? true : false;
            Tuple<Bitmap, SimpleDatum> data = m_igym.Render(bIsOpen, 512, 512, true);
            int nDataLen = 0;
            SimpleDatum stateData = state.Item1.GetData(m_bNormalizeInput, out nDataLen);
            Observation obs = new Observation(null, ImageData.GetImage(data.Item2), m_igym.RequiresDisplayImage, stateData.RealData, state.Item2, state.Item3);

            e.State = new StateBase(m_igym.GetActionSpace().Count());
            e.State.Reward = obs.Reward;
            e.State.Data = new SimpleDatum(true, nDataLen, 1, 1, -1, DateTime.Now, null, stateData.RealData.ToList(), 0, false, 0);
            e.State.Done = obs.Done;
            e.State.IsValid = true;

            if (m_gymui != null && m_nUiId >= 0)
            {
                m_gymui.Render(m_nUiId, obs);
                Thread.Sleep(m_igym.UiDelay);
            }
            else
            {
                if (m_evtOpenUi.WaitOne(0))
                    openUi();
            }

            if (m_sw.Elapsed.TotalMilliseconds > 1000)
            {
                double dfPct = (GlobalEpisodeMax == 0) ? 0 : (double)GlobalEpisodeCount / (double)GlobalEpisodeMax;
                e.OutputLog.Progress = dfPct;
                e.OutputLog.WriteLine("(" + dfPct.ToString("P") + ") Global Episode #" + GlobalEpisodeCount.ToString() + "  Global Reward = " + GlobalRewards.ToString() + " Exploration Rate = " + ExplorationRate.ToString("P") + " Optimal Selection Rate = " + OptimalSelectionRate.ToString("P"));
                m_sw.Restart();
            }

            return true;
        }

        protected override void openUi()
        {
            m_gymui = new MyCaffeGymUiProxy(new InstanceContext(this));
            m_gymui.Open();

            string strName = m_strName;
            string strTrainer = m_properties.GetProperty("TrainerType");
            if (!string.IsNullOrEmpty(strTrainer))
                strName += ": " + strTrainer;

            m_nUiId = m_gymui.OpenUi(strName, m_nUiId);
        }

        public void Closing()
        {
            m_nUiId = -1;
            m_gymui.Close();
            m_gymui = null;
        }
    }

    class MyCaffeAtariTrainerDual : MyCaffeTrainerDual, IXMyCaffeGymUiCallback
    {
        Stopwatch m_sw = new Stopwatch();
        IXMyCaffeGym m_igym;
        Log m_log;
        int m_nUiId = -1;
        MyCaffeGymUiProxy m_gymui = null;
        string m_strName = "ATARI";
        GymCollection m_colGyms = new GymCollection();
        DatasetDescriptor m_ds;
        double m_dfLastPct = 0;
        double m_dfLastRewards = 0;
        double m_dfLastExploration = 0;
        double m_dfLastOptimal = 0;
        int m_nLastEpisode = 0;
        EventWaitHandle m_evtOpenUi = new EventWaitHandle(false, EventResetMode.AutoReset, "_MyCaffeTrainer_OpenUi_");

        public MyCaffeAtariTrainerDual()
            : base()
        {
            m_colGyms.Load();
        }

        protected override void initialize(InitializeArgs e)
        {
            m_igym = m_colGyms.Find(m_strName);
            m_log = e.OutputLog;

            m_igym.Initialize(m_log, m_properties);

            m_sw.Start();
        }

        protected override void shutdown()
        {
            if (m_igym != null)
            {
                m_igym.Close();
                m_igym = null;
            }
        }

        protected override void dispose()
        {
            base.dispose();
        }

        protected override string name
        {
            get { return "RL.Trainer"; }
        }

        protected override DatasetDescriptor get_dataset_override(int nProjectID)
        {
            if (m_igym == null)
                m_igym = m_colGyms.Find(m_strName);

            m_ds = m_igym.GetDataset(DATA_TYPE.BLOB);

            return m_ds;
        }

        protected override bool getData(GetDataArgs e)
        {
            Tuple<State, double, bool> state = null;

            if (e.Reset)
                state = m_igym.Reset();

            if (e.Action >= 0)
                state = m_igym.Step(e.Action);

            bool bIsOpen = (m_nUiId >= 0) ? true : false;
            Tuple<Bitmap, SimpleDatum> data = m_igym.Render(bIsOpen, 512, 512, true);
            int nDataLen = 0;
            SimpleDatum stateData = state.Item1.GetData(false, out nDataLen);
            Observation obs = new Observation(data.Item1, ImageData.GetImage(data.Item2), m_igym.RequiresDisplayImage, stateData.RealData, state.Item2, state.Item3);

            e.State = new StateBase(m_igym.GetActionSpace().Count());
            e.State.Reward = obs.Reward;
            e.State.Data = data.Item2;
            e.State.Done = obs.Done;
            e.State.IsValid = true;

            if (m_gymui != null && m_nUiId >= 0)
            {
                if (e.GetDataCallback != null)
                {
                    OverlayArgs args = new OverlayArgs(obs.ImageDisplay);
                    e.GetDataCallback.OnOverlay(args);
                    obs.ImageDisplay = args.DisplayImage;
                }

                m_gymui.Render(m_nUiId, obs);
                Thread.Sleep(m_igym.UiDelay);
            }
            else
            {
                if (m_evtOpenUi.WaitOne(0))
                    openUi();
            }

            if (m_sw.Elapsed.TotalMilliseconds > 1000)
            {
                double dfPct = (GlobalEpisodeMax == 0) ? 0 : (double)GlobalEpisodeCount / (double)GlobalEpisodeMax;
                e.OutputLog.Progress = dfPct;

                if (updateNeeded(dfPct))
                {
                    string strOut = "(" + dfPct.ToString("P") + ") Iteration: " + GlobalIteration.ToString() + " Global Episode #" + GlobalEpisodeCount.ToString() + "  Global Reward = " + GlobalRewards.ToString() + " Exploration Rate = " + ExplorationRate.ToString("P") + " Score = " + ImmediateRewards.ToString();

                    if (OptimalSelectionRate > 0)
                        strOut += " Optimal Selection Rate = " + OptimalSelectionRate.ToString("P");

                    e.OutputLog.WriteLine(strOut);
                }

                m_sw.Restart();
            }

            return true;
        }

        private bool updateNeeded(double dfPct)
        {
            int nEpisode = GlobalEpisodeCount;            
            double dfRewards = GlobalRewards;
            double dfExploration = ExplorationRate;
            double dfOptimal = OptimalSelectionRate;

            if (dfPct == m_dfLastPct && nEpisode == m_nLastEpisode && dfRewards == m_dfLastRewards && dfExploration == m_dfLastExploration && dfOptimal == m_dfLastOptimal)
                return false;

            m_dfLastPct = dfPct;
            m_nLastEpisode = nEpisode;
            m_dfLastRewards = dfRewards;
            m_dfLastExploration = dfExploration;
            m_dfLastOptimal = dfOptimal;

            return true;
        }

        protected override void openUi()
        {
            m_gymui = new MyCaffeGymUiProxy(new InstanceContext(this));
            m_gymui.Open();

            string strName = m_strName;
            string strTrainer = m_properties.GetProperty("TrainerType");
            if (!string.IsNullOrEmpty(strTrainer))
                strName += ": " + strTrainer;

            m_nUiId = m_gymui.OpenUi(strName, m_nUiId);
        }

        public void Closing()
        {
            m_nUiId = -1;
            m_gymui.Close();
            m_gymui = null;
        }
    }

    class MyCaffeDataGeneralTrainerDual : MyCaffeTrainerDual, IXMyCaffeGymUiCallback
    {
        Stopwatch m_sw = new Stopwatch();
        IXMyCaffeGym m_igym;
        Log m_log;
        int m_nUiId = -1;
        string m_strName = "DataGeneral";
        GymCollection m_colGyms = new GymCollection();
        DatasetDescriptor m_ds;
        Tuple<State, double, bool> m_firststate = null;

        public MyCaffeDataGeneralTrainerDual()
            : base()
        {
            m_colGyms.Load();
        }

        protected override void initialize(InitializeArgs e)
        {
            initialize(e.OutputLog);
            m_sw.Start();
        }

        private void initialize(Log log)
        {
            if (m_igym == null)
            {
                m_log = log;
                m_igym = m_colGyms.Find(m_strName);
                m_igym.Initialize(m_log, m_properties);
            }
        }

        protected override void shutdown()
        {
            if (m_igym != null)
            {
                m_igym.Close();
                m_igym = null;
            }
        }

        protected override void dispose()
        {
            base.dispose();
        }

        protected override string name
        {
            get { return "RNN.Trainer.Dual"; }
        }

        protected override DatasetDescriptor get_dataset_override(int nProjectID)
        {
            IXMyCaffeGym igym = m_igym;

            if (igym == null)
                igym = m_colGyms.Find(m_strName);

            m_ds = igym.GetDataset(DATA_TYPE.BLOB);

            return m_ds;
        }

        protected override bool getData(GetDataArgs e)
        {
            Tuple<State, double, bool> state = null;

            if (e.Reset)
            {
                if (m_firststate != null)
                {
                    state = m_firststate;
                    m_firststate = null;
                }
                else
                {
                    state = m_igym.Reset();
                }
            }

            if (e.Action >= 0)
                state = m_igym.Step(e.Action);

            bool bIsOpen = (m_nUiId >= 0) ? true : false;
            int nDataLen = 0;
            SimpleDatum stateData = state.Item1.GetData(false, out nDataLen);

            e.State = new StateBase(m_igym.GetActionSpace().Count());
            e.State.Reward = 0;
            e.State.Data = stateData;
            e.State.Done = state.Item3;
            e.State.IsValid = true;

            if (m_sw.Elapsed.TotalMilliseconds > 1000)
            {
                int nMax = (int)GetProperty("GlobalMaxIterations");
                int nIteration = (int)GetProperty("GlobalIteration");
                double dfPct = (nMax == 0) ? 0 : (double)nIteration / (double)nMax;
                e.OutputLog.Progress = dfPct;
                e.OutputLog.WriteLine("(" + dfPct.ToString("P") + ") Global Iteration #" + nIteration.ToString());
                m_sw.Restart();
            }

            return true;
        }

        protected override bool convertOutput(ConvertOutputArgs e)
        {
            IXMyCaffeGymData igym = m_igym as IXMyCaffeGymData;
            if (igym == null)
                throw new Exception("Output data conversion requires a gym that implements the IXMyCaffeGymData interface.");

            string type;
            byte[] rgOutput = igym.ConvertOutput(Stage, e.Output.Length, e.Output, out type);
            e.SetRawOutput(rgOutput, type);

            return true;
        }

        protected override void openUi()
        {
        }

        protected override BucketCollection preloaddata(Log log, CancelEvent evtCancel, int nProjectID, out bool bUsePreloadData)
        {
            initialize(log);
            IXMyCaffeGymData igym = m_igym as IXMyCaffeGymData;
            Tuple<State, double, bool> state = igym.Reset();
            int nDataLen;
            SimpleDatum sd = state.Item1.GetData(false, out nDataLen);
            BucketCollection rgBucketCollection = null;

            bUsePreloadData = true;

            if (sd.IsRealData)
            {
                // Create the vocabulary bucket collection.
                rgBucketCollection = BucketCollection.Bucketize("Building vocabulary", 128, sd, log, evtCancel);
                if (rgBucketCollection == null)
                    return null;
            }
            else
            {
                List<int> rgVocabulary = new List<int>();

                for (int i = 0; i < sd.ByteData.Length; i++)
                {
                    int nVal = (int)sd.ByteData[i];

                    if (!rgVocabulary.Contains(nVal))
                        rgVocabulary.Add(nVal);
                }

                rgBucketCollection = new BucketCollection(rgVocabulary);
            }

            m_firststate = state;

            return rgBucketCollection;
        }

        public void Closing()
        {
        }
    }

}
