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
using MyCaffe.imagedb;
using MyCaffe.basecode.descriptors;
using MyCaffe.gym;
using MyCaffe.trainers;

namespace MyCaffe.test
{
    [TestClass]
    public class TestMyCaffeCustomTrainer
    {
        [TestMethod]
        public void TrainA2CCartPoleWithUi()
        {
            MyCaffeCustomTrainerTest test = new MyCaffeCustomTrainerTest();

            try
            {
                foreach (IMyCaffeCustomTrainerTest t in test.Tests)
                {
                    t.TrainCartPole(true, 1);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TrainA2CCartPoleWithOutUi()
        {
            MyCaffeCustomTrainerTest test = new MyCaffeCustomTrainerTest();

            try
            {
                foreach (IMyCaffeCustomTrainerTest t in test.Tests)
                {
                    t.TrainCartPole(false, 1);
                }
            }
            finally
            {
                test.Dispose();
            }
        }
        [TestMethod]
        public void TrainA3CCartPoleWithUi()
        {
            MyCaffeCustomTrainerTest test = new MyCaffeCustomTrainerTest();

            try
            {
                foreach (IMyCaffeCustomTrainerTest t in test.Tests)
                {
                    t.TrainCartPole(true, 3);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TrainA3CCartPoleWithOutUi()
        {
            MyCaffeCustomTrainerTest test = new MyCaffeCustomTrainerTest();

            try
            {
                foreach (IMyCaffeCustomTrainerTest t in test.Tests)
                {
                    t.TrainCartPole(false, 3);
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
        void TrainCartPole(bool bShowUi, int nThreads);
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

    class MyCaffeCustomTrainerTest<T> : TestEx<T>, IMyCaffeCustomTrainerTest
    {
        SettingsCaffe m_settings = new SettingsCaffe();
        CancelEvent m_evtCancel = new CancelEvent();

        public MyCaffeCustomTrainerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_settings.ImageDbLoadMethod = IMAGEDB_LOAD_METHOD.LOAD_ALL;
            m_settings.GpuIds = nDeviceID.ToString();
        }

        protected override void dispose()
        {
            base.dispose();
        }

        public void TrainCartPole(bool bShowUi, int nThreads)
        {
            m_evtCancel.Reset();

            int nIterations = 1000;
            m_log.WriteHeader("Test Training Cart-Pole for " + nIterations.ToString("N0") + " iterations.");
            MyCaffeGymClient gym = new MyCaffeGymClient();
            MyCaffeControl<T> mycaffe = new MyCaffeControl<T>(m_settings, m_log, m_evtCancel);
            MyCaffeCartPoleTrainer trainer = new MyCaffeCartPoleTrainer(gym, bShowUi);
            ProjectEx project = getReinforcementProject(gym, nIterations);
            DatasetDescriptor ds = trainer.DatasetOverride;

            m_log.CHECK(ds != null, "The MyCaffeCartPoleTrainer should return its dataset override returned by the Gym that it uses.");

            // load the project to train (note the project must use the MemoryDataLayer for input).
            mycaffe.Load(Phase.TRAIN, project, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.NONE, false, null, (ds == null) ? true : false);

            // Train the network using the custom trainer
            //  - Iterations (maximum frames cumulative across all threads) = 1000 (normally this would be much higher such as 500,000)
            //  - Learning rate = 0.005 (defined in solver.prototxt)
            //  - Min Batch Size = 32 (defined in train_val.prototxt for MemoryDataLayer)
            //
            //  - Threads = 1 to 3 envrionment threads (normally this would be much higher such as 8)
            //  - Optimizers = 1 optimizer threads (normally this would be higher sucha s 2)
            //  - EpsSteps = 750 after which exploration will be set at EpsEnd (normally this would be much higher such as 75000)
            //  - EpsStart = 0.4, start exploration at 40%
            //  - EpsEnd = 0.1, end exploration (and remain at) 10 % after EpsSteps
            //  - NStepReturn = 8, get a sample and calculate reward after 8 steps.
            //  - Gamma = 0.99, discount factor.
            //  - LossCoefficient = 0.5, use 50% of the Loss Value when calculating total loss.
            //  - EntropyCoefficient = 0.01 = use 1% of the Entropy when calculating total loss.
            trainer.Initialize("Threads=" + nThreads.ToString() + ";Optimizers=1;EpsSteps=750;EpsStart=0.4;EpsEnd=0.0;NStepReturn=8;Gamma=0.99;LossCoefficient=0.5;EntropyCoefficient=0.01;NormalizeInput=True");
            trainer.Train(mycaffe, nIterations);
            trainer.CleanUp();
            // Close the gym.
            gym.CloseAll("Cart Pole");

            // Release the mycaffe resources.
            mycaffe.Dispose();
        }

        private ProjectEx getReinforcementProject(MyCaffeGymClient gym, int nIterations)
        {
            ProjectEx p = new ProjectEx("test");

            string strModelFile = getTestPath("\\MyCaffe\\test_data\\models\\reinforcement\\cartpole\\train_val.prototxt");
            string strSolverFile = getTestPath("\\MyCaffe\\test_data\\models\\reinforcement\\cartpole\\solver.prototxt");

            RawProto protoM = RawProtoFile.LoadFromFile(strModelFile);
            p.ModelDescription = protoM.ToString();

            RawProto protoS = RawProtoFile.LoadFromFile(strSolverFile);
            RawProto iter = protoS.FindChild("max_iter");
            iter.Value = nIterations.ToString();

            p.SolverDescription = protoS.ToString();

            p.SetDataset(gym.GetDataset("Cart Pole", 0));

            return p;
        }
    }

    class MyCaffeCartPoleTrainer : MyCaffeCustomTrainer
    {
        MyCaffeGymClient m_gym;
        Stopwatch m_sw = new Stopwatch();
        string m_strName = "Cart Pole";
        bool m_bNormalizeInput = false;
        bool m_bShowUi = true;

        public MyCaffeCartPoleTrainer(MyCaffeGymClient gym, bool bShowUi) 
            : base()
        {
            m_gym = gym;
            m_bShowUi = bShowUi;
        }

        protected override string name
        {
            get { return "Cart Pole Trainer"; }
        }

        protected override DatasetDescriptor dataset_override
        {
            get { return m_gym.GetDataset(m_strName, 0); }
        }

        protected override void initialize(InitializeArgs e)
        {
            m_bNormalizeInput = m_properties.GetPropertyAsBool("NormalizeInput", false);
            m_sw.Start();
        }

        protected override void dispose()
        {
            base.dispose();
        }

        protected override void getData(GetDataArgs e)
        {
            if (e.Reset)
            {
                if (e.Index == -1)
                    e.Index = m_gym.Open(m_strName, true, m_bShowUi, true);

                m_gym.Reset(m_strName, e.Index);
            }

            if (e.Action >= 0)
                m_gym.Run(m_strName, e.Index, e.Action);

            Thread.Sleep(30);

            int nRetries = 0;
            Observation obs = m_gym.GetObservation(m_strName, e.Index);
            while (obs == null && nRetries < 100)
            {
                Thread.Sleep(250);
                obs = m_gym.GetObservation(m_strName, e.Index);
                nRetries++;

                if (e.CancelEvent.WaitOne(0))
                    return;
            }

            if (obs == null)
                throw new Exception("Failed to get an observation from gym #" + e.Index.ToString() + "!");

            double[] rgState = Observation.GetValues(obs.State, m_bNormalizeInput);
            e.State = new StateBase(m_gym.GetActionSpace(m_strName).Count());
            e.State.Reward = obs.Reward;
            e.State.Data = new SimpleDatum(true, rgState.Length, 1, 1, -1, DateTime.Now, null, rgState.ToList(), 0, false, 0);
            e.State.Done = obs.Done;
            e.State.IsValid = true;

            if (m_sw.Elapsed.TotalMilliseconds > 1000)
            {
                double dfPct = (double)GlobalEpisodeCount / (double)GlobalEpisodeMax;
                e.OutputLog.Progress = dfPct;
                e.OutputLog.WriteLine("(" + dfPct.ToString("P") + ") Global Episode #" + GlobalEpisodeCount.ToString() + "  Global Reward = " + GlobalRewards.ToString() + " Exploration Rate = " + ExplorationRate.ToString("P"));
                m_sw.Restart();
            }
        }
    }
}
