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
                    t.TestLoad(false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestLoadReinforcement()
        {
            MyCaffeControlTest test = new MyCaffeControlTest();

            try
            {
                foreach (IMyCaffeControlTest t in test.Tests)
                {
                    t.TestLoad(true);
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
                foreach (IMyCaffeControlTest t in test.Tests)
                {
                    t.TestTrainMultiGpu(1, 2);
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
                foreach (IMyCaffeControlTest t in test.Tests)
                {
                    t.TestTrainMultiGpu(1, 2, 3);
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
                foreach (IMyCaffeControlTest t in test.Tests)
                {
                    t.TestTrainMultiGpu(1, 2, 3, 4);
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
    }


    interface IMyCaffeControlTest : ITest
    {
        void TestLoad(bool bReinforcement);
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
            : base("Caffe Control Test", TestBase.DEFAULT_DEVICE_ID, engine)
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
        List<int> m_rgGpu = new List<int>() { TestBase.DEFAULT_DEVICE_ID };

        public MyCaffeControlTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_engine = engine;
            m_rgevtCancel = new WaitHandle[] { m_evtCancel.Handle };
            m_settings.ImageDbLoadMethod = SettingsCaffe.IMAGEDB_LOAD_METHOD.LOAD_ALL;
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
            p.OnOverrideSolver += new EventHandler<OverrideProjectArgs>(p_OnOverrideSolver);

            string strModelFile = getTestPath("\\test_data\\models\\mnist\\lenet_train_test.prototxt");
            string strSolverFile = getTestPath("\\test_data\\models\\mnist\\lenet_solver.prototxt");

            p.LoadModelFile(strModelFile);
            RawProto proto = RawProtoFile.LoadFromFile(strSolverFile);

            RawProto iter = proto.FindChild("max_iter");
            iter.Value = "1000";

            p.SolverDescription = proto.ToString();

            return p;
        }

        private ProjectEx getReinforcementProject()
        {
            string strModel = "name: MNIST_reinforcement " + Environment.NewLine +
                "layer { " + Environment.NewLine +
                " name: \"data\" " + Environment.NewLine +
                " type: \"BatchData\" " + Environment.NewLine +
                " top: \"data\" " + Environment.NewLine +
                " include { phase: TRAIN } " + Environment.NewLine +
                " transform_param { scale: 0.00390625 use_image_mean: True } " + Environment.NewLine +
                " batch_data_param { source: \"MNIST.training\" iterations: 1 batch_set_count: 1000 batch_size: 32 backend: IMAGEDB } " + Environment.NewLine +
                "} " + Environment.NewLine +
                "layer { " + Environment.NewLine +
                " name: \"data\" " + Environment.NewLine +
                " type: \"BatchData\" " + Environment.NewLine +
                " top: \"data\" " + Environment.NewLine +
                " include { phase: TEST } " + Environment.NewLine +
                " transform_param { scale: 0.00390625 use_image_mean: True } " + Environment.NewLine +
                " batch_data_param { source: \"MNIST.training\" iterations: 1 batch_set_count: 100 batch_size: 32 backend: IMAGEDB } " + Environment.NewLine +
                " data_param { source: \"MNIST.testing\" batch_size: 0 backend: IMAGEDB enable_random_selection: True } " + Environment.NewLine +
                "} " + Environment.NewLine +
                "layer { " + Environment.NewLine +
                " name: \"conv1\" " + Environment.NewLine +
                " type: \"Convolution\" " + Environment.NewLine +
                " bottom: \"data\" " + Environment.NewLine +
                " top: \"conv1\" " + Environment.NewLine +
                " param { lr_mult: 1 } " + Environment.NewLine +
                " param { lr_mult: 2 } " + Environment.NewLine +
                " convolution_param { kernel_size: 8 stride: 4 pad: 0 dilation: 1 num_output: 32 weight_filler { type: \"xavier\" variance_norm: FAN_IN } bias_filler { type: \"constant\" value: 0 } } " + Environment.NewLine +
                "} " + Environment.NewLine +
                "layer { " + Environment.NewLine +
                " name: \"relu1\" " + Environment.NewLine +
                " type: \"ReLU\" " + Environment.NewLine +
                " bottom: \"conv1\" " + Environment.NewLine +
                " top: \"conv1\" " + Environment.NewLine +
                "} " + Environment.NewLine +
                "layer { " + Environment.NewLine +
                " name: \"conv2\" " + Environment.NewLine +
                " type: \"Convolution\" " + Environment.NewLine +
                " bottom: \"conv1\" " + Environment.NewLine +
                " top: \"conv2\" " + Environment.NewLine +
                " param { lr_mult: 1 } " + Environment.NewLine +
                " param { lr_mult: 2 } " + Environment.NewLine +
                " convolution_param { kernel_size: 4 stride: 2 pad: 1 dilation: 1 num_output: 64 weight_filler { type: \"xavier\" variance_norm: FAN_IN } bias_filler { type: \"constant\" value: 0 } } " + Environment.NewLine +
                "} " + Environment.NewLine +
                "layer { " + Environment.NewLine +
                " name: \"relu2\" " + Environment.NewLine +
                " type: \"ReLU\" " + Environment.NewLine +
                " bottom: \"conv2\" " + Environment.NewLine +
                " top: \"conv2\" " + Environment.NewLine +
                "} " + Environment.NewLine +
                "layer { " + Environment.NewLine +
                " name: \"conv3\" " + Environment.NewLine +
                " type: \"Convolution\" " + Environment.NewLine +
                " bottom: \"conv2\" " + Environment.NewLine +
                " top: \"conv3\" " + Environment.NewLine +
                " param { lr_mult: 1 } " + Environment.NewLine +
                " param { lr_mult: 2 decay_mult: 0 } " + Environment.NewLine +
                " convolution_param { kernel_size: 3 stride: 2 pad: 0 dilation: 1 num_output: 64 weight_filler { type: \"xavier\" variance_norm: FAN_IN } bias_filler { type: \"constant\" value: 0.1 } } " + Environment.NewLine +
                "} " + Environment.NewLine +
                "layer { " + Environment.NewLine +
                " name: \"relu3\" " + Environment.NewLine +
                " type: \"ReLU\" " + Environment.NewLine +
                " bottom: \"conv3\" " + Environment.NewLine +
                " top: \"conv3\" " + Environment.NewLine +
                "} " + Environment.NewLine +
                "layer { " + Environment.NewLine +
                " name: \"lrn1\" " + Environment.NewLine +
                " type: \"LRN\" " + Environment.NewLine +
                " bottom: \"conv3\" " + Environment.NewLine +
                " top: \"lrn3\" " + Environment.NewLine +
                " lrn_param { local_size: 5 alpha: 1 beta: 0.75 norm_region: ACROSS_CHANNELS k: 1 } " + Environment.NewLine +
                "} " + Environment.NewLine +
                "layer { " + Environment.NewLine +
                " name: \"ip1\" " + Environment.NewLine +
                " type: \"InnerProduct\" " + Environment.NewLine +
                " bottom: \"lrn3\" " + Environment.NewLine +
                " top: \"ip1\" " + Environment.NewLine +
                " param { lr_mult: 1 } " + Environment.NewLine +
                " param { lr_mult: 2 } " + Environment.NewLine +
                " inner_product_param { num_output: 512 bias_term: True weight_filler { type: \"xavier\" variance_norm: FAN_IN } bias_filler { type: \"constant\" value: 0 } } " + Environment.NewLine +
                " axis: 1 " + Environment.NewLine +
                " transpose: False " + Environment.NewLine +
                "} " + Environment.NewLine +
                "layer { " + Environment.NewLine +
                " name: \"relu4\" " + Environment.NewLine +
                " type: \"ReLU\" " + Environment.NewLine +
                " bottom: \"ip1\" " + Environment.NewLine +
                " top: \"ip1\" " + Environment.NewLine +
                "} " + Environment.NewLine +
                "layer { " + Environment.NewLine +
                " name: \"ip2\" " + Environment.NewLine +
                " type: \"InnerProduct\" " + Environment.NewLine +
                " bottom: \"ip1\" " + Environment.NewLine +
                " top: \"ip2\" " + Environment.NewLine +
                " param { lr_mult: 1 } " + Environment.NewLine +
                " param { lr_mult: 2 } " + Environment.NewLine +
                " inner_product_param { num_output: 512 bias_term: True weight_filler { type: \"xavier\" variance_norm: FAN_IN } bias_filler { type: \"constant\" value: 0 } } " + Environment.NewLine +
                " axis: 1 " + Environment.NewLine +
                " transpose: False " + Environment.NewLine +
                "} " + Environment.NewLine +
                "layer { " + Environment.NewLine +
                " name: \"loss\" " + Environment.NewLine +
                " type: \"ReinforcementLoss\" " + Environment.NewLine +
                " bottom: \"ip2\" " + Environment.NewLine +
                " top: \"loss\" " + Environment.NewLine +
                " reinforcement_loss_param { exploration_rate_start: 0.4 exploration_rate_end: 0.1 training_step: 1 } " + Environment.NewLine +
                "} ";

            ProjectEx p = new ProjectEx("test");

            DatasetFactory factory = new DatasetFactory();
            p.SetDataset(factory.LoadDataset("MNIST"));

            p.ModelDescription = strModel;

            p.OnOverrideModel += new EventHandler<OverrideProjectArgs>(project_OnOverrideModel);
            p.OnOverrideSolver += new EventHandler<OverrideProjectArgs>(p_OnOverrideSolver);

            string strSolverFile = getTestPath("\\test_data\\models\\mnist\\lenet_solver.prototxt");

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
            p.OnOverrideSolver += new EventHandler<OverrideProjectArgs>(p_OnOverrideSolver);

            DatasetFactory factory = new DatasetFactory();
            p.SetDataset(factory.LoadDataset(strDs));

            string strModelFile = getTestPath("\\test_data\\models\\simple\\train_test.prototxt");
            string strSolverFile = getTestPath("\\test_data\\models\\mnist\\lenet_solver.prototxt");

            p.LoadModelFile(strModelFile);
            RawProto proto = RawProtoFile.LoadFromFile(strSolverFile);

            RawProto iter = proto.FindChild("max_iter");
            iter.Value = "1000";

            p.SolverDescription = proto.ToString();

            return p;
        }

        void p_OnOverrideSolver(object sender, OverrideProjectArgs e)
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

        public void TestLoad(bool bReinforcement)
        {
            m_log.WriteHeader(m_dt.ToString() + " - Test Load");

            MyCaffeControl<T> ctrl = new MyCaffeControl<T>(m_settings, m_log, m_evtCancel, m_evtForceSnapshot, m_evtForceTest, null, m_rgGpu, m_cuda.Path);
            ProjectEx project;

            if (bReinforcement)
                project = getReinforcementProject();
            else
                project = getProject();

            ctrl.Load(Phase.NONE, project, imagedb.IMGDB_LABEL_SELECTION_METHOD.NONE, imagedb.IMGDB_IMAGE_SELECTION_METHOD.NONE);

            ctrl.Dispose();
        }

        public void TestGetTestImage()
        {
            m_log.WriteHeader(m_dt.ToString() + " - Test Get Test Image");

            MyCaffeControl<T> ctrl = new MyCaffeControl<T>(m_settings, m_log, m_evtCancel, m_evtForceSnapshot, m_evtForceTest, null, m_rgGpu, m_cuda.Path);
            ProjectEx project = getProject();

            ctrl.Load(Phase.NONE, project, imagedb.IMGDB_LABEL_SELECTION_METHOD.NONE, imagedb.IMGDB_IMAGE_SELECTION_METHOD.NONE);

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

            ctrl.Load(Phase.TRAIN, project, imagedb.IMGDB_LABEL_SELECTION_METHOD.NONE, imagedb.IMGDB_IMAGE_SELECTION_METHOD.RANDOM);
            ctrl.Train();

            ctrl.Dispose();
        }

        public void TestTrainMultiGpu(params int[] rgGpu)
        {
            m_log.WriteHeader(m_dt.ToString() + " - Test Train Multi-Gpu");

            MyCaffeControl<T> ctrl = new MyCaffeControl<T>(m_settings, m_log, m_evtCancel, m_evtForceSnapshot, m_evtForceTest, null, rgGpu.ToList(), m_cuda.Path);
            ProjectEx project = getProject();

            ctrl.Load(Phase.TRAIN, project, imagedb.IMGDB_LABEL_SELECTION_METHOD.NONE, imagedb.IMGDB_IMAGE_SELECTION_METHOD.RANDOM);
            ctrl.Train();

            ctrl.Dispose();
        }

        public void TestTest()
        {
            m_log.WriteHeader(m_dt.ToString() + " - Test Test");

            MyCaffeControl<T> ctrl = new MyCaffeControl<T>(m_settings, m_log, m_evtCancel, m_evtForceSnapshot, m_evtForceTest, null, m_rgGpu, m_cuda.Path);
            ProjectEx project = getProject();

            ctrl.Load(Phase.TEST, project, imagedb.IMGDB_LABEL_SELECTION_METHOD.NONE, imagedb.IMGDB_IMAGE_SELECTION_METHOD.RANDOM);
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

            ctrl.Load(Phase.TRAIN, project, imagedb.IMGDB_LABEL_SELECTION_METHOD.NONE, imagedb.IMGDB_IMAGE_SELECTION_METHOD.RANDOM);
            ctrl.Train();
            ctrl.TestMany(1000, false);

            ctrl.Dispose();
        }

        public void TestTestManyOnTrainingSet()
        {
            m_log.WriteHeader(m_dt.ToString() + " - Test Test Many (on training set)");

            MyCaffeControl<T> ctrl = new MyCaffeControl<T>(m_settings, m_log, m_evtCancel, m_evtForceSnapshot, m_evtForceTest, null, m_rgGpu, m_cuda.Path);
            ProjectEx project = getProject();

            ctrl.Load(Phase.TRAIN, project, imagedb.IMGDB_LABEL_SELECTION_METHOD.NONE, imagedb.IMGDB_IMAGE_SELECTION_METHOD.RANDOM);
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

            ctrl.Load(Phase.TRAIN, project, imagedb.IMGDB_LABEL_SELECTION_METHOD.NONE, imagedb.IMGDB_IMAGE_SELECTION_METHOD.NONE);
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
            SourceDescriptor srcTest = new SourceDescriptor(0, "simple.test", 2, 2, 1, false, null, 4);
            SourceDescriptor srcTrain = new SourceDescriptor(0, "simple.train", 2, 2, 1, false, null, 4);
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
