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
using System.Net;
using OnnxControl;
using Onnx;
using MyCaffe.converter.onnx;

namespace MyCaffe.test
{
    [TestClass]
    public class TestPersistOnnx
    {
        [TestMethod]
        public void TestLoad()
        {
            PersistOnnxTest test = new PersistOnnxTest();

            try
            {
                foreach (IPersistOnnxTest t in test.Tests)
                {
                    if (t.DataType == DataType.FLOAT)
                        t.TestLoad();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSave()
        {
            PersistOnnxTest test = new PersistOnnxTest();

            try
            {
                foreach (IPersistOnnxTest t in test.Tests)
                {
                    if (t.DataType == DataType.FLOAT)
                        t.TestSave();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestConvertLeNetToOnnx()
        {
            PersistOnnxTest test = new PersistOnnxTest();

            try
            {
                foreach (IPersistOnnxTest t in test.Tests)
                {
                    if (t.DataType == DataType.FLOAT)
                        t.TestConvertLeNetToOnnx();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestConvertOnnxToLeNet()
        {
            PersistOnnxTest test = new PersistOnnxTest();

            try
            {
                foreach (IPersistOnnxTest t in test.Tests)
                {
                    if (t.DataType == DataType.FLOAT)
                        t.TestConvertOnnxToLeNet();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestImportOnnxAlexNetForRun()
        {
            PersistOnnxTest test = new PersistOnnxTest();

            try
            {
                foreach (IPersistOnnxTest t in test.Tests)
                {
                    t.TestImportOnnxAlexNet(null);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestImportOnnxAlexNetForTrain()
        {
            PersistOnnxTest test = new PersistOnnxTest();

            try
            {
                foreach (IPersistOnnxTest t in test.Tests)
                {
                    t.TestImportOnnxAlexNet("CIFAR-10");
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }


    interface IPersistOnnxTest : ITest
    {
        void TestLoad();
        void TestSave();
        void TestConvertLeNetToOnnx();
        void TestConvertOnnxToLeNet();
        void TestImportOnnxAlexNet(string strTrainingDs = null);
    }

    class PersistOnnxTest : TestBase
    {
        public PersistOnnxTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Test Persist Onnx", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new PersistOnnxTest<double>(strName, nDeviceID, engine);
            else
                return new PersistOnnxTest<float>(strName, nDeviceID, engine);
        }
    }

    class PersistOnnxTest<T> : TestEx<T>, IPersistOnnxTest
    {
        public PersistOnnxTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_engine = engine;
        }

        protected override void dispose()
        {
            base.dispose();
        }

        private void getModelFile(string strUrl, string strPath)
        {
            if (!File.Exists(strPath))
            {
                using (WebClient client = new WebClient())
                {
                    Trace.WriteLine("Downloading '" + strUrl + "' - this may take awhile...");
                    client.DownloadFile(strUrl, strPath);
                }
            }
        }

        public void TestLoad()
        {
            PersistOnnx persist = new PersistOnnx();

            string strTestPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\models\\onnx";
            if (!Directory.Exists(strTestPath))
                Directory.CreateDirectory(strTestPath);

            string strModelFileBig = strTestPath + "\\bvlcalexnet-9.onnx";
            string strDownloadPathBig = "https://github.com/onnx/models/raw/master/vision/classification/alexnet/model/bvlcalexnet-9.onnx";
            string strModelFileSmall = strTestPath + "\\mnist-1.onnx";
            string strDownloadPathSmall = "https://github.com/onnx/models/raw/master/vision/classification/mnist/model/mnist-1.onnx";
            string strDownloadPath = strDownloadPathSmall;
            string strModelFile = strModelFileSmall;

            getModelFile(strDownloadPath, strModelFile);

            ModelProto model = persist.Load(strModelFile);

            m_log.EnableTrace = true;
            m_log.WriteLine("Loaded model file '" + strModelFile + "'...");
            m_log.WriteLine("Version = " + model.IrVersion.ToString());
            m_log.WriteLine("Producer Name = " + model.ProducerName);
            m_log.WriteLine("Producer Version = " + model.ProducerVersion);
            m_log.WriteLine("Model Version = " + model.ModelVersion.ToString());
            m_log.WriteLine("Description = " + model.DocString);
            m_log.WriteLine("Domain = " + model.Domain);

            m_log.WriteHeader("Run Model");
            outputGraph("RUN GRAPH", model.Graph);

            m_log.WriteHeader("Training Model");
            foreach (TrainingInfoProto train in model.TrainingInfo)
            {
                outputGraph("TRAINING INIT", train.Initialization);
                outputGraph("TRAINING GRAPH ", train.Algorithm);
            }
        }

        private void outputGraph(string strName, GraphProto graph)
        {
            m_log.WriteLine("--- " + strName + " ------------------------");
            m_log.WriteLine("Name = " + graph.Name);

            m_log.WriteLine("Inputs:");
            foreach (ValueInfoProto val in graph.Input)
            {
                m_log.WriteLine(val.ToString());
            }

            m_log.WriteLine("Outputs:");
            foreach (ValueInfoProto val in graph.Output)
            {
                m_log.WriteLine(val.ToString());
            }

            m_log.WriteLine("Nodes:");
            foreach (NodeProto val in graph.Node)
            {
                m_log.WriteLine(val.ToString());
            }

            m_log.WriteLine("Quantization Annotation:");
            m_log.WriteLine(graph.QuantizationAnnotation.ToString());

            m_log.WriteLine("Initializer Tensors:");
            foreach (TensorProto t in graph.Initializer)
            {
                m_log.WriteLine(t.Name + " (data type = " + t.DataType.ToString() + ") " + t.Dims.ToString());
            }
        }

        public void TestSave()
        {
            PersistOnnx persist = new PersistOnnx();
            string strTestPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\models\\onnx";
            if (!Directory.Exists(strTestPath))
                Directory.CreateDirectory(strTestPath);

            string strModelFileBig = strTestPath + "\\bvlcalexnet-9.onnx";
            string strDownloadPathBig = "https://github.com/onnx/models/raw/master/vision/classification/alexnet/model/bvlcalexnet-9.onnx";
            string strModelFileSmall = strTestPath + "\\mnist-1.onnx";
            string strDownloadPathSmall = "https://github.com/onnx/models/raw/master/vision/classification/mnist/model/mnist-1.onnx";
            string strDownloadPath = strDownloadPathSmall;
            string strModelFile = strModelFileSmall;

            getModelFile(strDownloadPath, strModelFile);

            ModelProto model = persist.Load(strModelFile);
            string strPath = Path.GetDirectoryName(strModelFile);
            string strModelFile2 = strPath + "\\" + Path.GetFileNameWithoutExtension(strModelFile) + "2.onnx";

            if (File.Exists(strModelFile2))
                File.Delete(strModelFile2);

            persist.Save(model, strModelFile2);
            ModelProto model2 = persist.Load(strModelFile2);

            // Compare
            if (model.IrVersion != model2.IrVersion)
                m_log.FAIL("Versions do not match!");

            if (model.Graph.Name != model2.Graph.Name)
                m_log.FAIL("Graph names do not match!");

            m_log.WriteLine("Compare Inputs:");
            if (model.Graph.Input.Count != model2.Graph.Input.Count)
                m_log.FAIL("Graph Input counts differ!");

            for (int i = 0; i < model.Graph.Input.Count; i++)
            {
                string str1 = model.Graph.Input.ToString();
                string str2 = model2.Graph.Input.ToString();

                if (str1 != str2)
                    m_log.FAIL("Graph Inputs at " + i.ToString() + " do not match!");
            }

            m_log.WriteLine("Compare Outputs:");
            if (model.Graph.Output.Count != model2.Graph.Output.Count)
                m_log.FAIL("Graph Output counts differ!");

            for (int i = 0; i < model.Graph.Output.Count; i++)
            {
                string str1 = model.Graph.Output.ToString();
                string str2 = model2.Graph.Output.ToString();

                if (str1 != str2)
                    m_log.FAIL("Graph Outputs at " + i.ToString() + " do not match!");
            }

            m_log.WriteLine("Compare Nodes:");
            if (model.Graph.Node.Count != model2.Graph.Node.Count)
                m_log.FAIL("Graph Node counts differ!");

            for (int i = 0; i < model.Graph.Node.Count; i++)
            {
                string str1 = model.Graph.Node.ToString();
                string str2 = model2.Graph.Node.ToString();

                if (str1 != str2)
                    m_log.FAIL("Graph Node at " + i.ToString() + " do not match!");
            }


            m_log.WriteLine("Compare Quantization Annotation:");
            string strq1 = model.Graph.QuantizationAnnotation.ToString();
            string strq2 = model2.Graph.QuantizationAnnotation.ToString();
            if (strq1 != strq2)
                m_log.FAIL("Quantization Annotations do not match!");

            m_log.WriteLine("Compare Tensors:");
            if (model.Graph.Initializer.Count != model2.Graph.Initializer.Count)
                m_log.FAIL("Graph Tensors counts differ!");

            for (int i = 0; i < model.Graph.Initializer.Count; i++)
            {
                string str1 = model.Graph.Initializer[i].Name + " (data type = " + model.Graph.Initializer[i].DataType.ToString() + ") " + model.Graph.Initializer[i].Dims.ToString();
                string str2 = model2.Graph.Initializer[i].Name + " (data type = " + model2.Graph.Initializer[i].DataType.ToString() + ") " + model2.Graph.Initializer[i].Dims.ToString();

                if (str1 != str2)
                    m_log.FAIL("Graph Tensors at " + i.ToString() + " do not match!");
            }

            File.Delete(strModelFile2);
        }

        private string loadTextFile(string strFile)
        {
            using (StreamReader sr = new StreamReader(strFile))
            {
                return sr.ReadToEnd();
            }
        }

        private byte[] loadBinaryFile(string strFile)
        {
            using (FileStream fs = new FileStream(strFile, FileMode.Open, FileAccess.Read))
            using (BinaryReader br = new BinaryReader(fs))
            {
                return br.ReadBytes((int)fs.Length);
            }
        }

        private void saveTextToFile(string str, string strFile)
        {
            if (File.Exists(strFile))
                File.Delete(strFile);

            using (StreamWriter sw = new StreamWriter(strFile))
            {
                sw.WriteLine(str);
            }
        }

        private void saveBinaryToFile(byte[] rg, string strFile)
        {
            if (File.Exists(strFile))
                File.Delete(strFile);

            using (FileStream fs = new FileStream(strFile, FileMode.CreateNew, FileAccess.Write))
            using (BinaryWriter bw = new BinaryWriter(fs))
            {
                bw.Write(rg);
            }
        }

        public void TestConvertLeNetToOnnx()
        {
            string strTestPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\models\\onnx";
            if (!Directory.Exists(strTestPath))
                Directory.CreateDirectory(strTestPath);

            string strModelPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\models\\mnist";
            MyCaffeConversionControl<T> convert = new MyCaffeConversionControl<T>();

            string strOnnxModelFile = strTestPath + "\\lenet_from_mycaffe.onnx";
            if (!File.Exists(strOnnxModelFile))
                throw new Exception("You must first run 'TestConvertLeNetToOnnx' to create the .onnx file.");

            SettingsCaffe s = new SettingsCaffe();
            MyCaffeControl<T> mycaffe = new MyCaffeControl<T>(s, m_log, new CancelEvent());

            ProjectEx prj = new ProjectEx("AlexNet");
            prj.SolverDescription = loadTextFile(strModelPath + "\\lenet_solver.prototxt");
            prj.ModelDescription = loadTextFile(strModelPath + "\\lenet_train_test.prototxt");
            prj.WeightsState = loadBinaryFile(strModelPath + "\\my_weights.mycaffemodel");

            DatasetFactory factory = new DatasetFactory();
            prj.SetDataset(factory.LoadDataset("MNIST"));

            mycaffe.Load(Phase.TRAIN, prj);

            if (File.Exists(strOnnxModelFile))
                File.Delete(strOnnxModelFile);

            convert.ConvertMyCaffeToOnnxFile(mycaffe, strOnnxModelFile);
        }

        public void TestConvertOnnxToLeNet()
        {
            string strTestPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\models\\onnx";
            if (!Directory.Exists(strTestPath))
                Directory.CreateDirectory(strTestPath);

            string strModelPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\models\\mnist";
            MyCaffeConversionControl<T> convert = new MyCaffeConversionControl<T>();

            string strOnnxModelFile = strTestPath + "\\lenet_from_mycaffe.onnx";

            CudaDnn<T> cuda = new CudaDnn<T>(0);
            MyCaffeModelData model = convert.ConvertOnnxToMyCaffeFromFile(cuda, m_log, strOnnxModelFile);
            cuda.Dispose();

            saveTextToFile(model.ModelDescription, strModelPath + "\\onnx_to_lenet_runmodel.prototxt");
            saveBinaryToFile(model.Weights, strModelPath + "\\onnx_to_lenet_runmodel.mycaffemodel");
        }

        private string downloadOnnxAlexNetModel()
        {
            // Download a small onnx model from https://github.com/onnx (dowload is 233mb)
            string strModel = "bvlcalexnet-9.onnx";
            string strUrl = "https://github.com/onnx/models/raw/master/vision/classification/alexnet/model/" + strModel;
            string strFile = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\models\\onnx\\";

            if (!Directory.Exists(strFile))
                Directory.CreateDirectory(strFile);

            strFile += strModel;

            if (!File.Exists(strFile))
            {
                using (WebClient client = new WebClient())
                {
                    client.DownloadFile(strUrl, strFile);
                }
            }

            return strFile;
        }

        public void TestImportOnnxAlexNet(string strTrainingDs)
        {
            string strTestPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\models\\onnx\\imported\\alexnet";
            if (!Directory.Exists(strTestPath))
                Directory.CreateDirectory(strTestPath);

            string strOnnxFile = downloadOnnxAlexNetModel();
            MyCaffeConversionControl<T> convert = new MyCaffeConversionControl<T>();

            DatasetDescriptor dsTraining = null;
            if (strTrainingDs != null)
            {
                DatasetFactory factory = new DatasetFactory();
                dsTraining = factory.LoadDataset(strTrainingDs);
            }

            CudaDnn<T> cuda = new CudaDnn<T>(0);
            MyCaffeModelData data = convert.ConvertOnnxToMyCaffeFromFile(cuda, m_log, strOnnxFile, dsTraining);
            Trace.WriteLine(convert.ReportString);

            data.Save(strTestPath, "AlexNet" + ((dsTraining != null) ? ".train" : "") + "." + typeof(T).ToString());
        }
    }
}
