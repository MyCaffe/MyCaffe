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

namespace MyCaffe.test
{
    [TestClass]
    public class TestMyCaffeControl
    {
        [TestMethod]
        public void TestLoad()
        {
            MyCaffeControlTest test = new MyCaffeControlTest();

            try
            {
                foreach (IMyCaffeControlTest t in test.Tests)
                {
                    t.TestLoad();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGetTestImage()
        {
            MyCaffeControlTest test = new MyCaffeControlTest();

            try
            {
                foreach (IMyCaffeControlTest t in test.Tests)
                {
                    t.TestGetTestImage();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTrain()
        {
            MyCaffeControlTest test = new MyCaffeControlTest();

            try
            {
                foreach (IMyCaffeControlTest t in test.Tests)
                {
                    t.TestTrain();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTrainMultiGpu2()
        {
            MyCaffeControlTest test = new MyCaffeControlTest();

            try
            {
                List<int> rgGpu = getGpus(2);
                foreach (IMyCaffeControlTest t in test.Tests)
                {
                    t.TestTrainMultiGpu(rgGpu.ToArray());
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTrainMultiGpu3()
        {
            MyCaffeControlTest test = new MyCaffeControlTest();

            try
            {
                List<int> rgGpu = getGpus(3);
                if (rgGpu.Count < 3)
                    return;

                foreach (IMyCaffeControlTest t in test.Tests)
                {
                    t.TestTrainMultiGpu(rgGpu.ToArray());
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTrainMultiGpu4()
        {
            MyCaffeControlTest test = new MyCaffeControlTest();

            try
            {
                List<int> rgGpu = getGpus(4);
                if (rgGpu.Count < 4)
                    return;

                foreach (IMyCaffeControlTest t in test.Tests)
                {
                    t.TestTrainMultiGpu(rgGpu.ToArray());
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTest()
        {
            MyCaffeControlTest test = new MyCaffeControlTest();

            try
            {
                foreach (IMyCaffeControlTest t in test.Tests)
                {
                    t.TestTest();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTestMany()
        {
            MyCaffeControlTest test = new MyCaffeControlTest();

            try
            {
                foreach (IMyCaffeControlTest t in test.Tests)
                {
                    t.TestTestMany();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTestManySimple()
        {
            MyCaffeControlTest test = new MyCaffeControlTest();

            try
            {
                foreach (IMyCaffeControlTest t in test.Tests)
                {
                    t.TestTestManySimple();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTestManyOnTrainingSet()
        {
            MyCaffeControlTest test = new MyCaffeControlTest();

            try
            {
                foreach (IMyCaffeControlTest t in test.Tests)
                {
                    t.TestTestManyOnTrainingSet();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        private List<int> getGpus(int nMax)
        {
            CudaDnn<float> cuda = new CudaDnn<float>(0);
            List<int> rgGpu = new List<int>();
            int nDevCount = cuda.GetDeviceCount();

            for (int i = 0; i < nDevCount; i++)
            {
                string strDevInfo = cuda.GetDeviceInfo(i, true);
                string strP2PInfo = cuda.GetDeviceP2PInfo(i);

                if (strP2PInfo.Contains("P2P Capable = YES"))
                    rgGpu.Add(i);

                if (rgGpu.Count == nMax)
                    break;
            }

            cuda.Dispose();

            return rgGpu;
        }
    }


    interface IMyCaffeControlTest : ITest
    {
        void TestLoad();
        void TestGetTestImage();
        void TestTrain();
        void TestTrainMultiGpu(params int[] rgGpu);
        void TestTest();
        void TestTestMany();
        void TestTestManySimple();
        void TestTestManyOnTrainingSet();
    }

    class MyCaffeControlTest : TestBase
    {
        public MyCaffeControlTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("MyCaffe Control Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new MyCaffeControlTest<double>(strName, nDeviceID, engine);
            else
                return new MyCaffeControlTest<float>(strName, nDeviceID, engine);
        }
    }

    class MyCaffeControlTest<T> : TestEx<T>, IMyCaffeControlTest
    {
        SettingsCaffe m_settings = new SettingsCaffe();
        CancelEvent m_evtCancel = new CancelEvent();
        AutoResetEvent m_evtForceSnapshot = new AutoResetEvent(false);
        AutoResetEvent m_evtForceTest = new AutoResetEvent(false);
        WaitHandle[] m_rgevtCancel;
        List<int> m_rgGpu = new List<int>();

        public MyCaffeControlTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_engine = engine;

            List<WaitHandle> rgWait = new List<WaitHandle>();
            rgWait.AddRange(m_evtCancel.Handles);

            m_rgevtCancel = rgWait.ToArray();
            m_settings.ImageDbLoadMethod = IMAGEDB_LOAD_METHOD.LOAD_ALL;
            m_settings.GpuIds = nDeviceID.ToString();
            m_rgGpu.Add(nDeviceID);
        }

        protected override void dispose()
        {
            base.dispose();
        }

        private ProjectEx getProject()
        {
            ProjectEx p = new ProjectEx("MNIST Project");

            DatasetFactory factory = new DatasetFactory();
            DatasetDescriptor ds = factory.LoadDataset("MNIST");

            p.SetDataset(ds);
            p.OnOverrideModel += new EventHandler<OverrideProjectArgs>(project_OnOverrideModel);
            p.OnOverrideSolver += new EventHandler<OverrideProjectArgs>(project_OnOverrideSolver);

            string strModelFile = getTestPath("\\MyCaffe\\test_data\\models\\mnist\\lenet_train_test.prototxt");
            string strSolverFile = getTestPath("\\MyCaffe\\test_data\\models\\mnist\\lenet_solver.prototxt");

            p.LoadModelFile(strModelFile);
            RawProto proto = RawProtoFile.LoadFromFile(strSolverFile);

            RawProto iter = proto.FindChild("max_iter");
            iter.Value = "1000";

            p.SolverDescription = proto.ToString();

            return p;
        }

        private ProjectEx getSimpleProject(string strDs)
        {
            ProjectEx p = new ProjectEx(strDs + " Project");

            p.OnOverrideModel += new EventHandler<OverrideProjectArgs>(project_OnOverrideModel);
            p.OnOverrideSolver += new EventHandler<OverrideProjectArgs>(project_OnOverrideSolver);

            DatasetFactory factory = new DatasetFactory();
            p.SetDataset(factory.LoadDataset(strDs));

            string strModelFile = getTestPath("\\MyCaffe\\test_data\\models\\simple\\train_test.prototxt");
            string strSolverFile = getTestPath("\\MyCaffe\\test_data\\models\\mnist\\lenet_solver.prototxt");

            p.LoadModelFile(strModelFile);
            RawProto proto = RawProtoFile.LoadFromFile(strSolverFile);

            RawProto iter = proto.FindChild("max_iter");
            iter.Value = "1000";

            p.SolverDescription = proto.ToString();

            return p;
        }

        void project_OnOverrideSolver(object sender, OverrideProjectArgs e)
        {
            RawProto proto = e.Proto;

            RawProto max_iter = proto.FindChild("max_iter");
            if (max_iter != null)
                max_iter.Value = "800";

            RawProto display = proto.FindChild("display");
            if (display != null)
                display.Value = "100";

            RawProto test_iter = proto.FindChild("test_iter");
            if (test_iter != null)
                test_iter.Value = "100";

            RawProto test_interval = proto.FindChild("test_interval");
            if (test_interval != null)
                test_interval.Value = "100";
        }

        void project_OnOverrideModel(object sender, OverrideProjectArgs e)
        {
            RawProto proto = e.Proto;
            RawProtoCollection colLayers = proto.FindChildren("layer");

            foreach (RawProto protoChild in colLayers)
            {
                RawProto name = protoChild.FindChild("name");

                if (name.Value == "fc8")
                {
                    RawProto inner_product_param = protoChild.FindChild("inner_product_param");
                    if (inner_product_param != null)
                    {
                        RawProto num_output = inner_product_param.FindChild("num_output");
                        if (num_output != null)
                        {
                            num_output.Value = "10";
                        }
                    }
                }
            }
        }

        public void TestLoad()
        {
            m_log.WriteHeader(m_dt.ToString() + " - Test Load");

            MyCaffeControl<T> ctrl = new MyCaffeControl<T>(m_settings, m_log, m_evtCancel, m_evtForceSnapshot, m_evtForceTest, null, m_rgGpu, m_cuda.Path);
            ProjectEx project;

            project = getProject();

            ctrl.Load(Phase.NONE, project, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.NONE);

            ctrl.Dispose();
        }

        public void TestGetTestImage()
        {
            m_log.WriteHeader(m_dt.ToString() + " - Test Get Test Image");

            MyCaffeControl<T> ctrl = new MyCaffeControl<T>(m_settings, m_log, m_evtCancel, m_evtForceSnapshot, m_evtForceTest, null, m_rgGpu, m_cuda.Path);
            ProjectEx project = getProject();

            ctrl.Load(Phase.NONE, project, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.NONE);

            int nLabel;
            int nLabel2;
            string strLabel;
            string strLabel2;

            Bitmap bmp1 = ctrl.GetTestImage(Phase.TEST, out nLabel, out strLabel);
            Bitmap bmp2 = ctrl.GetTestImage(Phase.TRAIN, out nLabel2, out strLabel2);

            Assert.AreEqual(bmp1.Width, bmp2.Width);
            Assert.AreEqual(bmp1.Height, bmp2.Height);

            int nDifferences = 0;

            for (int x = 0; x < bmp1.Width; x++)
            {
                for (int y = 0; y < bmp1.Height; y++)
                {
                    Color clr1 = bmp1.GetPixel(x, y);
                    Color clr2 = bmp2.GetPixel(x, y);

                    if (clr1.ToArgb() != clr2.ToArgb())
                        nDifferences++;
                }
            }

            Assert.AreNotEqual(nDifferences, 0);

            ctrl.Dispose();
        }

        public void TestTrain()
        {
            m_log.WriteHeader(m_dt.ToString() + " - Test Train");

            MyCaffeControl<T> ctrl = new MyCaffeControl<T>(m_settings, m_log, m_evtCancel, m_evtForceSnapshot, m_evtForceTest, null, m_rgGpu, m_cuda.Path);
            ProjectEx project = getProject();

            ctrl.Load(Phase.TRAIN, project, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.RANDOM);
            ctrl.Train();

            ctrl.Dispose();
        }

        public void TestTrainMultiGpu(params int[] rgGpu)
        {
            m_log.WriteHeader(m_dt.ToString() + " - Test Train Multi-Gpu");

            MyCaffeControl<T> ctrl = new MyCaffeControl<T>(m_settings, m_log, m_evtCancel, m_evtForceSnapshot, m_evtForceTest, null, rgGpu.ToList(), m_cuda.Path);
            ProjectEx project = getProject();

            ctrl.Load(Phase.TRAIN, project, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.RANDOM);
            ctrl.Train();

            ctrl.Dispose();
        }

        public void TestTest()
        {
            m_log.WriteHeader(m_dt.ToString() + " - Test Test");

            MyCaffeControl<T> ctrl = new MyCaffeControl<T>(m_settings, m_log, m_evtCancel, m_evtForceSnapshot, m_evtForceTest, null, m_rgGpu, m_cuda.Path);
            ProjectEx project = getProject();

            ctrl.Load(Phase.TEST, project, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.RANDOM);
            ctrl.Train();
            double dfLoss = ctrl.Test();

            m_log.WriteLine("dfLoss returned = " + dfLoss.ToString());

            ctrl.Dispose();
        }

        public void TestTestMany()
        {
            m_log.WriteHeader(m_dt.ToString() + " - Test Test Many (on testing set)");

            MyCaffeControl<T> ctrl = new MyCaffeControl<T>(m_settings, m_log, m_evtCancel, m_evtForceSnapshot, m_evtForceTest, null, m_rgGpu, m_cuda.Path);
            ProjectEx project = getProject();

            ctrl.Load(Phase.TRAIN, project, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.RANDOM);
            ctrl.Train();
            ctrl.TestMany(1000, false);

            ctrl.Dispose();
        }

        public void TestTestManyOnTrainingSet()
        {
            m_log.WriteHeader(m_dt.ToString() + " - Test Test Many (on training set)");

            MyCaffeControl<T> ctrl = new MyCaffeControl<T>(m_settings, m_log, m_evtCancel, m_evtForceSnapshot, m_evtForceTest, null, m_rgGpu, m_cuda.Path);
            ProjectEx project = getProject();

            ctrl.Load(Phase.TRAIN, project, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.RANDOM);
            ctrl.Train();
            ctrl.TestMany(1000, true);

            ctrl.Dispose();
        }

        public void TestTestManySimple()
        {
            m_log.WriteHeader(m_dt.ToString() + " - Test Simple");

            MyCaffeControl<T> ctrl = new MyCaffeControl<T>(m_settings, m_log, m_evtCancel, m_evtForceSnapshot, m_evtForceTest, null, m_rgGpu, m_cuda.Path);

            string strDs = createSimpleDataset();
            ProjectEx project = getSimpleProject(strDs);

            ctrl.Load(Phase.TRAIN, project, IMGDB_LABEL_SELECTION_METHOD.RANDOM, IMGDB_IMAGE_SELECTION_METHOD.RANDOM);
            ctrl.Train();
            ctrl.TestMany(1000, false);

            ctrl.Dispose();
        }

        private void deleteSimpleDataset(bool bFullDelete)
        {
            string strCmd;

            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                List<Dataset> rgDs = entities.Datasets.Where(p => p.Name == "simple").ToList();
                if (rgDs.Count > 0)
                {
                    int nSrcTestId = rgDs[0].TestingSourceID.GetValueOrDefault();
                    int nSrcTrainId = rgDs[0].TrainingSourceID.GetValueOrDefault();

                    if (bFullDelete)
                    {
                        strCmd = "DELETE FROM RawImages WHERE (SourceID = " + nSrcTestId.ToString() + ") OR (SourceID = " + nSrcTrainId.ToString() + ")";
                        entities.Database.ExecuteSqlCommand(strCmd);

                        strCmd = "DELETE FROM Sources WHERE (ID = " + nSrcTestId.ToString() + ") OR (ID = " + nSrcTrainId.ToString() + ")";
                        entities.Database.ExecuteSqlCommand(strCmd);

                        strCmd = "DELETE FROM Datasets WHERE (ID = " + rgDs[0].ID.ToString() + ")";
                        entities.Database.ExecuteSqlCommand(strCmd);
                    }

                    strCmd = "DELETE FROM RawImageResults WHERE (SourceID = " + nSrcTestId.ToString() + ") OR (SourceID = " + nSrcTrainId.ToString() + ")";
                    entities.Database.ExecuteSqlCommand(strCmd);
                }
            }
        }

        private string createSimpleDataset()
        {
            deleteSimpleDataset(false);

            DatasetFactory dsFactory = new DatasetFactory();
            SourceDescriptor srcTest = new SourceDescriptor(0, "simple.test", 2, 2, 1, false, false, 0, null, 4);
            SourceDescriptor srcTrain = new SourceDescriptor(0, "simple.train", 2, 2, 1, false, false, 0, null, 4);
            DatasetDescriptor ds = new DatasetDescriptor(0, "simple", null, null, srcTrain, srcTest, null, null);

            ds.ID = dsFactory.AddDataset(ds);
            ds = dsFactory.LoadDataset(ds.ID);

            if (ds.TestingSource.ImageCount == 0 || ds.TrainingSource.ImageCount == 0)
            {
                SimpleDatum sd1 = new SimpleDatum(false, 1, 2, 2, 0, DateTime.MinValue, new List<byte>() { 10, 10, 10, 10 }, null, 0, false, 0);
                SimpleDatum sd2 = new SimpleDatum(false, 1, 2, 2, 1, DateTime.MinValue, new List<byte>() { 10, 250, 10, 250 }, null, 0, false, 1);
                SimpleDatum sd3 = new SimpleDatum(false, 1, 2, 2, 2, DateTime.MinValue, new List<byte>() { 250, 10, 250, 10 }, null, 0, false, 2);
                SimpleDatum sd4 = new SimpleDatum(false, 1, 2, 2, 3, DateTime.MinValue, new List<byte>() { 250, 250, 250, 250 }, null, 0, false, 3);

                if (ds.TrainingSource.ImageCount == 0)
                {
                    dsFactory.Open(ds.TrainingSource);
                    dsFactory.PutRawImage(0, sd1);
                    dsFactory.PutRawImage(1, sd2);
                    dsFactory.PutRawImage(2, sd3);
                    dsFactory.PutRawImage(3, sd4);
                    dsFactory.UpdateSourceCounts();
                    dsFactory.Close();
                }

                if (ds.TestingSource.ImageCount == 0)
                {
                    dsFactory.Open(ds.TestingSource);
                    dsFactory.PutRawImage(0, sd1);
                    dsFactory.PutRawImage(1, sd2);
                    dsFactory.PutRawImage(2, sd3);
                    dsFactory.PutRawImage(3, sd4);
                    dsFactory.UpdateSourceCounts();
                    dsFactory.Close();
                }
            }

            return ds.Name;
        }
    }
}
