using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.basecode;
using MyCaffe.param;
using MyCaffe.common;
using MyCaffe.layers;
using MyCaffe.fillers;
using System.IO;
using MyCaffe.db.image;
using System.Diagnostics;
using MyCaffe.basecode.descriptors;
using System.Drawing;

namespace MyCaffe.test
{
    [TestClass]
    public class TestPersistCaffe
    {
        [TestMethod]
        public void TestImport()
        {
            PersistCaffeTest test = new PersistCaffeTest();

            try
            {
                foreach (IPersistCaffeTest t in test.Tests)
                {
                    if (t.DataType == DataType.FLOAT)
                        t.TestImport();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestImportExport()
        {
            PersistCaffeTest test = new PersistCaffeTest();

            try
            {
                foreach (IPersistCaffeTest t in test.Tests)
                {
                    if (t.DataType == DataType.FLOAT)
                        t.TestImportExport();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestImportExportV1()
        {
            PersistCaffeTest test = new PersistCaffeTest();

            try
            {
                foreach (IPersistCaffeTest t in test.Tests)
                {
                    if (t.DataType == DataType.FLOAT)
                        t.TestImportExportV1();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestReadBlobProto()
        {
            PersistCaffeTest test = new PersistCaffeTest();

            try
            {
                foreach (IPersistCaffeTest t in test.Tests)
                {
                    t.TestReadBlobProto();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSolverState()
        {
            PersistCaffeTest test = new PersistCaffeTest();

            try
            {
                foreach (IPersistCaffeTest t in test.Tests)
                {
                    t.TestSolverState();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestReadWeightInfo()
        {
            PersistCaffeTest test = new PersistCaffeTest();

            try
            {
                foreach (IPersistCaffeTest t in test.Tests)
                {
                    t.TestReadWeightInfo();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }


    interface IPersistCaffeTest : ITest
    {
        void TestImport();
        void TestImportExport();
        void TestImportExportV1();
        void TestReadBlobProto();
        void TestSolverState();
        void TestReadWeightInfo();
    }

    class PersistCaffeTest : TestBase
    {
        public PersistCaffeTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Filter Persist Caffe", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new PersistCaffeTest<double>(strName, nDeviceID, engine);
            else
                return new PersistCaffeTest<float>(strName, nDeviceID, engine);
        }
    }

    class PersistCaffeTest<T> : TestEx<T>, IPersistCaffeTest
    {
        public PersistCaffeTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_engine = engine;
        }

        protected override void dispose()
        {
            base.dispose();
        }

        private void createPascalSbdDatasetShell(string strDs)
        {
            DatasetFactory factory = new DatasetFactory();

            DatasetDescriptor dsd = factory.LoadDataset(strDs);
            if (dsd != null)
                return;

            SourceDescriptor srcTrain = new SourceDescriptor(0, strDs + ".training", 500, 500, 3, false, true);
            SourceDescriptor srcTest = new SourceDescriptor(0, strDs + ".testing", 500, 500, 3, false, true);
            DatasetDescriptor ds = new DatasetDescriptor(0, strDs, null, null, srcTrain, srcTest, "", "Testing Dataset");

            factory.AddDataset(ds);

            Bitmap bmp = new Bitmap(500, 500);
            SimpleDatum sd = ImageData.GetImageData(bmp, 3, false, 0);
            int nLen = 500 * 500;
            int nItemCount = 1;
            int nWid = 500;
            int nHt = 500;
            byte[] rgData = new byte[nLen];

            List<byte> rgDataFinal = new List<byte>(rgData);
            rgDataFinal.AddRange(BitConverter.GetBytes(nLen));
            rgDataFinal.AddRange(BitConverter.GetBytes(nItemCount));
            rgDataFinal.AddRange(BitConverter.GetBytes(nWid));
            rgDataFinal.AddRange(BitConverter.GetBytes(nHt));

            sd.DataCriteria = rgDataFinal.ToArray();
            sd.DataCriteriaFormat = SimpleDatum.DATA_FORMAT.SEGMENTATION;

            factory.Open(ds.TrainingSource.ID);
            factory.PutRawImage(0, sd);
            factory.PutRawImage(1, sd);
            factory.Close();

            factory.Open(ds.TestingSource.ID);
            factory.PutRawImage(0, sd);
            factory.PutRawImage(1, sd);
            factory.Close();

            factory.UpdateDatasetCounts(ds.ID);
        }

        public void TestImport()
        {
            PersistCaffe<T> persist = new PersistCaffe<T>(m_log, true);
            string strModelFile = getTestPath("\\MyCaffe\\test_data\\models\\voc_fcns32\\train_val.prototxt");
            string strFile = getTestPath("\\MyCaffe\\test_data\\models\\voc_fcns32\\fcn32s-heavy-pascal.caffemodel");
            byte[] rgWeights = null;
            string strModelDesc = "";

            using (FileStream fs = new FileStream(strFile, FileMode.Open, FileAccess.Read))
            {
                using (BinaryReader br = new BinaryReader(fs))
                {
                    rgWeights = br.ReadBytes((int)fs.Length);
                }
            }

            using (StreamReader sr = new StreamReader(strModelFile))
            {
                strModelDesc = sr.ReadToEnd();
            }

            createPascalSbdDatasetShell("PASCAL_SBD");

            MyCaffeImageDatabase db = new MyCaffeImageDatabase();
            db.InitializeWithDsName(new SettingsCaffe(), "PASCAL_SBD");

            RawProto proto = RawProto.Parse(strModelDesc);
            NetParameter net_param = NetParameter.FromProto(proto);
            m_log.Enable = false;
            Net<T> net = new common.Net<T>(m_cuda, m_log, net_param, new CancelEvent(), db, Phase.TRAIN);
            m_log.Enable = true;

            net.LoadWeights(rgWeights, persist);
            net.Dispose();
        }


        public void TestImportExport()
        {
            PersistCaffe<T> persist = new PersistCaffe<T>(m_log, true);
            string strModelFile = getTestPath("\\MyCaffe\\test_data\\models\\voc_fcns32\\train_val.prototxt");
            string strFile = getTestPath("\\MyCaffe\\test_data\\models\\voc_fcns32\\fcn32s-heavy-pascal.caffemodel");
            byte[] rgWeights = null;
            string strModelDesc = "";

            using (FileStream fs = new FileStream(strFile, FileMode.Open, FileAccess.Read))
            {
                using (BinaryReader br = new BinaryReader(fs))
                {
                    rgWeights = br.ReadBytes((int)fs.Length);
                }
            }

            using (StreamReader sr = new StreamReader(strModelFile))
            {
                strModelDesc = sr.ReadToEnd();
            }

            createPascalSbdDatasetShell("PASCAL_SBD");

            MyCaffeImageDatabase db = new MyCaffeImageDatabase();
            db.InitializeWithDsName(new SettingsCaffe(), "PASCAL_SBD");

            RawProto proto = RawProto.Parse(strModelDesc);
            NetParameter net_param = NetParameter.FromProto(proto);
            m_log.Enable = false;
            Net<T> net = new common.Net<T>(m_cuda, m_log, net_param, new CancelEvent(), db, Phase.TRAIN);
            m_log.Enable = true;

            // Load Caffe weight format.
            net.LoadWeights(rgWeights, persist);

            // Save to MyCaffe format.
            byte[] rgWeights1 = net.SaveWeights(persist);

            // Reload in MyCaffe format.
            net.LoadWeights(rgWeights1, persist);

            net.Dispose();
        }


        public void TestImportExportV1()
        {
            PersistCaffe<T> persist = new PersistCaffe<T>(m_log, true);
            string strModelFile = getTestPath("\\MyCaffe\\test_data\\models\\bvlc_nin\\train_val.prototxt");
            string strFile = getTestPath("\\MyCaffe\\test_data\\models\\bvlc_nin\\cifar10_nin.caffemodel");
            byte[] rgWeights = null;
            string strModelDesc = "";

            using (FileStream fs = new FileStream(strFile, FileMode.Open, FileAccess.Read))
            {
                using (BinaryReader br = new BinaryReader(fs))
                {
                    rgWeights = br.ReadBytes((int)fs.Length);
                }
            }

            using (StreamReader sr = new StreamReader(strModelFile))
            {
                strModelDesc = sr.ReadToEnd();
            }

            MyCaffeImageDatabase db = new MyCaffeImageDatabase();
            db.InitializeWithDsName(new SettingsCaffe(), "CIFAR-10");

            RawProto proto = RawProto.Parse(strModelDesc);
            NetParameter net_param = NetParameter.FromProto(proto);
            m_log.Enable = false;
            Net<T> net1 = new common.Net<T>(m_cuda, m_log, net_param, new CancelEvent(), db, Phase.TRAIN);
            Net<T> net2 = new common.Net<T>(m_cuda, m_log, net_param, new CancelEvent(), db, Phase.TRAIN);
            m_log.Enable = true;

            // Load Caffe weight format.
            net1.LoadWeights(rgWeights, persist);

            // Save to native format.
            byte[] rgWeights1 = net1.SaveWeights(persist);

            // Reload native format.
            net2.LoadWeights(rgWeights1, persist);
            byte[] rgWeights2 = net2.SaveWeights(persist);

            //--------------------------------------
            //  Compare the weights.
            //--------------------------------------

            m_log.CHECK_EQ(rgWeights1.Length, rgWeights2.Length, "The weight sizes differ!");

            Stopwatch sw = new Stopwatch();
            sw.Start();

            for (int i = 0; i < rgWeights1.Length; i++)
            {
                m_log.CHECK_EQ(rgWeights1[i], rgWeights2[i], "The bytes at index " + i.ToString() + " are not equal!");

                if (sw.Elapsed.TotalMilliseconds > 1000)
                {
                    m_log.Progress = (double)i / (double)rgWeights1.Length;
                    m_log.WriteLine("Comparing weights...");
                    sw.Restart();
                }
            }

            m_log.WriteLine("The weights are the same!");

            net1.Dispose();
            net2.Dispose();
        }

        public void TestReadBlobProto()
        {
            PersistCaffe<T> persist = new PersistCaffe<T>(m_log, true);
            string strFile = getTestPath("\\MyCaffe\\test_data\\models\\bvlc_alexnet_imgnet\\imagenet_mean.binaryproto");
            byte[] rgData = null;

            using (FileStream fs = new FileStream(strFile, FileMode.Open, FileAccess.Read))
            {
                using (BinaryReader br = new BinaryReader(fs))
                {
                    rgData = br.ReadBytes((int)fs.Length);
                }
            }

            BlobProto proto = persist.LoadBlobProto(rgData, 1);
            int nCount = 1;

            if (proto.shape != null)
            {
                for (int i = 0; i < proto.shape.dim.Count; i++)
                {
                    nCount *= proto.shape.dim[i];
                }
            }
            else
            {
                if (proto.num.HasValue)
                {
                    m_log.CHECK_GT(proto.num.Value, 0, "There should be at least one num.");
                    nCount *= proto.num.Value;
                }

                if (proto.channels.HasValue)
                {
                    m_log.CHECK_GT(proto.channels.Value, 0, "There should be at least one channel.");
                    nCount *= proto.channels.Value;
                }

                if (proto.height.HasValue)
                {
                    m_log.CHECK_GT(proto.height.Value, 0, "There should be at least one height.");
                    nCount *= proto.height.Value;
                }

                if (proto.width.HasValue)
                {
                    m_log.CHECK_GT(proto.width.Value, 0, "There should be at least one width.");
                    nCount *= proto.width.Value;
                }
            }

            m_log.CHECK(proto.data.Count > 0 || proto.double_data.Count > 0, "Neither the 'data' or 'double_data' have any values.");

            if (proto.data.Count > 0)
                m_log.CHECK_EQ(nCount, proto.data.Count, "The data count does not match the shape.");
            else
                m_log.CHECK_EQ(nCount, proto.double_data.Count, "The double_data count does not match the shape.");

            // Test data conversion.
            if (proto.data.Count > 0)
            {
                double[] rg = new double[proto.data.Count];
                Array.Copy(proto.data.ToArray(), rg, rg.Length);
            }
        }

        public void TestSolverState()
        {
            PersistCaffe<T> persist = new PersistCaffe<T>(m_log, true);
            Random rand = new Random();
            SolverState state1 = new SolverState();

            state1.iter = 33;
            state1.current_step = 2;

            // Load the initial data
            for (int i = 0; i < 10; i++)
            {
                BlobProto bp = new BlobProto(new List<int>() { 1, i + 1, i + 1 + i + 1 });
                int nCount = 1 * (i + 1) * ((i + 1) + (i + 1));

                if (DataType == DataType.DOUBLE)
                {
                    for (int j = 0; j < nCount; j++)
                    {
                        bp.double_data.Add(rand.NextDouble());
                    }
                }
                else
                {
                    for (int j = 0; j < nCount; j++)
                    {
                        bp.data.Add((float)rand.NextDouble());
                    }
                }

                state1.history.Add(bp);
            }

            // Save the proto
            byte[] rgData = persist.SaveSolverState(state1);

            // Load the new state.
            SolverState state2 = persist.LoadSolverState(rgData);

            // Verify the data;

            m_log.CHECK_EQ(state1.iter, state2.iter, "The iterations are different.");
            m_log.CHECK_EQ(state1.current_step, state2.current_step, "The current steps are different");
            m_log.CHECK_EQ(state1.history.Count, state2.history.Count, "The history counts are different.");

            for (int i = 0; i < state1.history.Count; i++)
            {
                m_log.CHECK(state1.history[i].Compare(state2.history[i]), "The histories at " + i.ToString() + " are different!");
            }
        }

        public void TestReadWeightInfo()
        {
            PersistCaffe<T> persist = new PersistCaffe<T>(m_log, true);
            string strModelFile = getTestPath("\\MyCaffe\\test_data\\models\\voc_fcns32\\train_val.prototxt");
            string strFile = getTestPath("\\MyCaffe\\test_data\\models\\voc_fcns32\\fcn32s-heavy-pascal.caffemodel");
            byte[] rgWeights = null;
            string strModelDesc = "";

            using (FileStream fs = new FileStream(strFile, FileMode.Open, FileAccess.Read))
            {
                using (BinaryReader br = new BinaryReader(fs))
                {
                    rgWeights = br.ReadBytes((int)fs.Length);
                }
            }

            using (StreamReader sr = new StreamReader(strModelFile))
            {
                strModelDesc = sr.ReadToEnd();
            }

            createPascalSbdDatasetShell("PASCAL_SBD");

            MyCaffeImageDatabase db = new MyCaffeImageDatabase();
            db.InitializeWithDsName(new SettingsCaffe(), "PASCAL_SBD");

            RawProto proto = RawProto.Parse(strModelDesc);
            NetParameter net_param = NetParameter.FromProto(proto);
            m_log.Enable = false;
            Net<T> net = new common.Net<T>(m_cuda, m_log, net_param, new CancelEvent(), db, Phase.TRAIN);
            m_log.Enable = true;

            // Load Caffe weight format.
            net.LoadWeights(rgWeights, persist);
            WeightInfo<T> wi1 = persist.LoadWeightInfo(rgWeights);
            WeightInfo<T> wi2 = persist.LoadWeightInfo(net.learnable_parameters);

            m_log.CHECK_EQ(wi1.Blobs.Count, wi2.Blobs.Count, "The blob counts are not equal!");

            List<string> rgstrNames1 = new List<string>();
            List<List<int>> rgrgShape1 = new List<List<int>>();
            foreach (KeyValuePair<string, List<int>> kv in wi1.Blobs)
            {
                rgstrNames1.Add(kv.Key);
                rgrgShape1.Add(kv.Value);
            }

            List<string> rgstrNames2 = new List<string>();
            List<List<int>> rgrgShape2 = new List<List<int>>();
            foreach (KeyValuePair<string, List<int>> kv in wi2.Blobs)
            {
                rgstrNames2.Add(kv.Key);
                rgrgShape2.Add(kv.Value);
            }

            m_log.CHECK_EQ(rgstrNames1.Count, rgstrNames2.Count, "The names have a different count.");

            for (int i = 0; i < rgrgShape1.Count; i++)
            {
                m_log.CHECK_EQ(rgrgShape1[i].Count, rgrgShape2[i].Count, "The shape arrays at " + i.ToString() + " have different sizes.");

                for (int j = 0; j < rgrgShape1[i].Count; j++)
                {
                    m_log.CHECK_EQ(rgrgShape1[i][j], rgrgShape2[i][j], "The shape array element at " + j.ToString() + " in array " + i.ToString() + " doesn't match.");
                }
            }

            net.Dispose();
        }
    }
}
