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
        public void TestImportOnnxLeNetVer1ForTrain()
        {
            PersistOnnxTest test = new PersistOnnxTest();

            try
            {
                foreach (IPersistOnnxTest t in test.Tests)
                {
                    t.TestImportOnnxLetNet("MNIST", 1);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestImportOnnxLeNetVer8ForTrain()
        {
            PersistOnnxTest test = new PersistOnnxTest();

            try
            {
                foreach (IPersistOnnxTest t in test.Tests)
                {
                    t.TestImportOnnxLetNet("MNIST", 8);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestImportOnnxSSDForTrain()
        {
            PersistOnnxTest test = new PersistOnnxTest();

            try
            {
                foreach (IPersistOnnxTest t in test.Tests)
                {
                    t.TestImportOnnxSSDModel("VOC0712");
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestImportOnnxEfficientNetForTrain()
        {
            PersistOnnxTest test = new PersistOnnxTest();

            try
            {
                foreach (IPersistOnnxTest t in test.Tests)
                {
                    t.TestImportOnnxEfficientNetModel("CIFAR-10");
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestImportOnnxEfficientNetForTrainScaled()
        {
            PersistOnnxTest test = new PersistOnnxTest();

            try
            {
                foreach (IPersistOnnxTest t in test.Tests)
                {
                    t.TestImportOnnxEfficientNetModel("CIFAR-10", -1, 1);
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

        [TestMethod]
        public void TestImportOnnxGoogNetForRun()
        {
            PersistOnnxTest test = new PersistOnnxTest();

            try
            {
                foreach (IPersistOnnxTest t in test.Tests)
                {
                    t.TestImportOnnxGoogNet(null);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestImportOnnxGoogNetForTrain()
        {
            PersistOnnxTest test = new PersistOnnxTest();

            try
            {
                foreach (IPersistOnnxTest t in test.Tests)
                {
                    t.TestImportOnnxGoogNet("CIFAR-10");
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestImportOnnxVGG19ForRun()
        {
            PersistOnnxTest test = new PersistOnnxTest();

            try
            {
                foreach (IPersistOnnxTest t in test.Tests)
                {
                    t.TestImportOnnxVGG19(null);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestImportOnnxVGG19ForTrain()
        {
            PersistOnnxTest test = new PersistOnnxTest();

            try
            {
                foreach (IPersistOnnxTest t in test.Tests)
                {
                    t.TestImportOnnxVGG19("CIFAR-10");
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestImportOnnxResNet50ForRun()
        {
            PersistOnnxTest test = new PersistOnnxTest();

            try
            {
                foreach (IPersistOnnxTest t in test.Tests)
                {
                    t.TestImportOnnxResNet50(null);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestImportOnnxResNet50ForTrain()
        {
            PersistOnnxTest test = new PersistOnnxTest();

            try
            {
                foreach (IPersistOnnxTest t in test.Tests)
                {
                    t.TestImportOnnxResNet50("CIFAR-10");
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestImportOnnxInceptionV1ForRun()
        {
            PersistOnnxTest test = new PersistOnnxTest();

            try
            {
                foreach (IPersistOnnxTest t in test.Tests)
                {
                    t.TestImportOnnxInceptionV1(null);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestImportOnnxInceptionV1ForTrain()
        {
            PersistOnnxTest test = new PersistOnnxTest();

            try
            {
                foreach (IPersistOnnxTest t in test.Tests)
                {
                    t.TestImportOnnxInceptionV1("CIFAR-10");
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestImportOnnxInceptionV2ForTrain()
        {
            PersistOnnxTest test = new PersistOnnxTest();

            try
            {
                foreach (IPersistOnnxTest t in test.Tests)
                {
                    t.TestImportOnnxInceptionV2("CIFAR-10");
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestImportOnnxInceptionV2ForTrainNoDummy()
        {
            PersistOnnxTest test = new PersistOnnxTest();

            try
            {
                foreach (IPersistOnnxTest t in test.Tests)
                {
                    t.TestImportOnnxInceptionV2("CIFAR-10", "DUMMY");
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestImportExportImportOnnxLeNet()
        {
            PersistOnnxTest test = new PersistOnnxTest();

            try
            {
                foreach (IPersistOnnxTest t in test.Tests)
                {
                    t.TestImportExportImportOnnxLeNet("MNIST", 8);
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
        void TestImportOnnxLetNet(string strTrainingDs = null, int nVersion = 8);
        void TestImportOnnxSSDModel(string strTrainingDs = null);
        void TestImportOnnxEfficientNetModel(string strTrainingDs = null, double? dfScaleMin = null, double? dfScaleMax = null);
        void TestImportOnnxAlexNet(string strTrainingDs = null);
        void TestImportOnnxGoogNet(string strTrainingDs = null);
        void TestImportOnnxVGG19(string strTrainingDs = null);
        void TestImportOnnxResNet50(string strTrainingDs = null);
        void TestImportOnnxInceptionV1(string strTrainingDs = null);
        void TestImportOnnxInceptionV2(string strTrainingDs = null, string strIgnoreLayer = null);
        void TestImportExportImportOnnxLeNet(string strTrainingDs, int nVersion);
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

            string strModelFileSmall = strTestPath + "\\mnist-1.onnx";
            string strDownloadPathSmall = "https://github.com/onnx/models/raw/main/vision/classification/mnist/model/mnist-1.onnx";
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
            //string strDownloadPathBig = "https://github.com/onnx/models/raw/main/vision/classification/alexnet/model/bvlcalexnet-9.onnx";
            string strModelFileSmall = strTestPath + "\\mnist-1.onnx";
            string strDownloadPathSmall = "https://github.com/onnx/models/raw/main/vision/classification/mnist/model/mnist-1.onnx";
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

            SettingsCaffe s = new SettingsCaffe();
            MyCaffeControl<T> mycaffe = new MyCaffeControl<T>(s, m_log, new CancelEvent());

            try
            {
                string strOnnxModelFile = strTestPath + "\\lenet_from_mycaffe.onnx";

                ProjectEx prj = new ProjectEx("LeNet");
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
            catch (Exception excpt)
            {
                throw excpt;
            }
            finally
            {
                mycaffe.Dispose();
                convert.Dispose();
            }
        }

        public void TestConvertOnnxToLeNet()
        {
            string strTestPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\models\\onnx";
            if (!Directory.Exists(strTestPath))
                Directory.CreateDirectory(strTestPath);

            string strModelPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\models\\mnist";
            MyCaffeConversionControl<T> convert = new MyCaffeConversionControl<T>();

            string strOnnxModelFile = strTestPath + "\\lenet_from_mycaffe.onnx";
            if (!File.Exists(strOnnxModelFile))
                throw new Exception("You must first run 'TestConvertLeNetToOnnx' to create the .onnx file.");

            CudaDnn<T> cuda = null;

            try
            {
                cuda = new CudaDnn<T>(0);
                MyCaffeModelData model = convert.ConvertOnnxToMyCaffeFromFile(cuda, m_log, strOnnxModelFile);

                saveTextToFile(model.ModelDescription, strModelPath + "\\onnx_to_lenet_runmodel.prototxt");
                saveBinaryToFile(model.Weights, strModelPath + "\\onnx_to_lenet_runmodel.mycaffemodel");
            }
            catch (Exception excpt)
            {
                throw excpt;
            }
            finally
            {
                cuda.Dispose();
                convert.Dispose();
            }
        }

        private string download(string strModel, string strUrl, double dfSizeInMb)
        {
            string strFolder = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\models\\onnx\\";

            if (!Directory.Exists(strFolder))
                Directory.CreateDirectory(strFolder);

            string strFile = strFolder + strModel;

            if (!File.Exists(strFile))
            {
                using (WebClient client = new WebClient())
                {
                    Trace.WriteLine("downloading '" + strUrl + "' (" + dfSizeInMb.ToString("N2") + " mb)...");
                    client.DownloadFile(strUrl, strFile);
                }
            }

            return strFile;
        }

        private string downloadOnnxLeNetModel(int nVersion)
        {
            // Download a small onnx model from https://github.com/onnx (dowload is 26mb)
            double dfSizeInMb = 26;
            string strModel = "mnist-" + nVersion.ToString() + ".onnx";
            string strUrl = "https://github.com/onnx/models/raw/main/vision/classification/mnist/model/" + strModel;
            return download(strModel, strUrl, dfSizeInMb);
        }

        public void TestImportOnnxLetNet(string strTrainingDs, int nVersion)
        {
            testImportOnnxLetNet(strTrainingDs, nVersion);
        }

        public MyCaffeModelData testImportOnnxLetNet(string strTrainingDs, int nVersion)
        {
            string strTestPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\models\\onnx\\imported\\mnist";
            if (!Directory.Exists(strTestPath))
                Directory.CreateDirectory(strTestPath);

            string strOnnxFile = downloadOnnxLeNetModel(nVersion);
            MyCaffeConversionControl<T> convert = new MyCaffeConversionControl<T>();

            DatasetDescriptor dsTraining = null;
            if (strTrainingDs != null)
            {
                DatasetFactory factory = new DatasetFactory();
                dsTraining = factory.LoadDataset(strTrainingDs);
            }

            CudaDnn<T> cuda = null;

            try
            {
                cuda = new CudaDnn<T>(0);
                MyCaffeModelData data = convert.ConvertOnnxToMyCaffeFromFile(cuda, m_log, strOnnxFile, true, false, dsTraining);
                Trace.WriteLine(convert.ReportString);

                data.Save(strTestPath, "LeNet" + ((dsTraining != null) ? ".train" : "") + "." + typeof(T).ToString());

                return data;
            }
            catch (Exception excpt)
            {
                throw excpt;
            }
            finally
            {
                cuda.Dispose();
                convert.Dispose();
            }
        }

        public void TestImportExportImportOnnxLeNet(string strTrainingDs, int nVersion)
        {
            MyCaffeModelData data = testImportOnnxLetNet(strTrainingDs, nVersion);

            DatasetDescriptor dsTraining = null;
            if (strTrainingDs == null)
                m_log.FAIL("The training dataset must be specified!");

            DatasetFactory factory = new DatasetFactory();
            dsTraining = factory.LoadDataset(strTrainingDs);

            List<int> rgShape = new List<int>() { 1, dsTraining.TrainingSource.ImageChannels, dsTraining.TrainingSource.ImageHeight, dsTraining.TrainingSource.ImageWidth };
            BlobShape shape = new BlobShape(rgShape);

            // Convert the model to a model for running (removes data layers)
            TransformationParameter transform;
            NetParameter netParam = MyCaffeControl<T>.CreateNetParameterForRunning(shape, data.ModelDescription, out transform);
            RawProto proto = netParam.ToProto("root");
            data.ModelDescription = proto.ToString();

            // Convert the mycaffe model (imported from an onnx model) back into an onnx model.
            MyCaffeConversionControl<T> converter = new MyCaffeConversionControl<T>();
            CudaDnn<T> cuda = null;

            string strOnnxFile = data.LastSavedModeDescriptionFileName + ".export.onnx";

            try
            {
                if (File.Exists(strOnnxFile))
                    File.Delete(strOnnxFile);

                cuda = new CudaDnn<T>(0);
                converter.ConvertMyCaffeToOnnxFile(cuda, m_log, data, strOnnxFile);
            }
            catch (Exception excpt)
            {
                throw excpt;
            }
            finally
            {
                cuda.Dispose();
                converter.Dispose();
            }


            //-----------------------------------------------------------------
            //  Verify the two files.
            //-----------------------------------------------------------------
            PersistOnnx onnxOriginal = new PersistOnnx();
            ModelProto modelOriginal = onnxOriginal.Load(data.OriginalDownloadFile);

            PersistOnnx onnxConvert = new PersistOnnx();
            ModelProto modelConvert = onnxConvert.Load(strOnnxFile);

            ModelProtoComparer compare = new ModelProtoComparer(m_log);
            compare.Compare(modelOriginal, modelConvert);
        }

        private string downloadOnnxSSDModel()
        {
            // Download a small onnx model from https://github.com/onnx (dowload is 76.6mb)
            double dfSizeInMb = 76.6;
            string strModel = "ssd-10.onnx";
            string strUrl = "https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/ssd/model/" + strModel;
            return download(strModel, strUrl, dfSizeInMb);
        }

        /// <summary>
        /// Test importing the ONNX SSD model.
        /// </summary>
        /// <param name="strTrainingDs">Specifies the dataset to use.</param>
        /// <remarks>
        /// NOTE: Currently this model is only imported in an incomplete form, primarily to use the weights in transfer learning.
        /// </remarks>
        public void TestImportOnnxSSDModel(string strTrainingDs)
        {
            string strTestPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\models\\onnx\\imported\\voc0712";
            if (!Directory.Exists(strTestPath))
                Directory.CreateDirectory(strTestPath);

            string strOnnxFile = downloadOnnxSSDModel();
            MyCaffeConversionControl<T> convert = new MyCaffeConversionControl<T>();

            DatasetDescriptor dsTraining = null;
            if (strTrainingDs != null)
            {
                DatasetFactory factory = new DatasetFactory();
                dsTraining = factory.LoadDataset(strTrainingDs);
            }

            CudaDnn<T> cuda = null;

            try
            {
                cuda = new CudaDnn<T>(0);
                MyCaffeModelData data = convert.ConvertOnnxToMyCaffeFromFile(cuda, m_log, strOnnxFile, true, false, dsTraining);
                Trace.WriteLine(convert.ReportString);

                data.Save(strTestPath, "SSD" + ((dsTraining != null) ? ".train" : "") + "." + typeof(T).ToString());
            }
            catch (Exception excpt)
            {
                throw excpt;
            }
            finally
            {
                cuda.Dispose();
                convert.Dispose();
            }
        }


        private string downloadOnnxEfficientNetModel()
        {
            // Download a small onnx model from https://github.com/onnx (dowload is 76.6mb)
            double dfSizeInMb = 51.9;
            string strModel = "efficientnet-lite4-11.onnx";
            string strUrl = "https://github.com/onnx/models/raw/main/vision/classification/efficientnet-lite4/model/" + strModel;
            return download(strModel, strUrl, dfSizeInMb);
        }

        public void TestImportOnnxEfficientNetModel(string strTrainingDs, double? dfScaledMin = null, double? dfScaledMax = null)
        {
            string strTestPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\models\\onnx\\imported\\efficientnet";
            if (!Directory.Exists(strTestPath))
                Directory.CreateDirectory(strTestPath);

            string strOnnxFile = downloadOnnxEfficientNetModel();
            MyCaffeConversionControl<T> convert = new MyCaffeConversionControl<T>();

            if (dfScaledMin.HasValue && dfScaledMax.HasValue)
                convert.SetWeightScaling(dfScaledMin.Value, dfScaledMax.Value);

            DatasetDescriptor dsTraining = null;
            if (strTrainingDs != null)
            {
                DatasetFactory factory = new DatasetFactory();
                dsTraining = factory.LoadDataset(strTrainingDs);
            }

            CudaDnn<T> cuda = null;

            try
            {
                cuda = new CudaDnn<T>(0);
                MyCaffeModelData data = convert.ConvertOnnxToMyCaffeFromFile(cuda, m_log, strOnnxFile, true, false, dsTraining);
                Trace.WriteLine(convert.ReportString);

                data.Save(strTestPath, "EfficientNetLite" + ((dsTraining != null) ? ".train" : "") + "." + typeof(T).ToString());
            }
            catch (Exception excpt)
            {
                throw excpt;
            }
            finally
            {
                cuda.Dispose();
                convert.Dispose();
            }
        }


        private string downloadOnnxAlexNetModel()
        {
            // Download a small onnx model from https://github.com/onnx (dowload is 233mb)
            double dfSizeInMb = 233;
            string strModel = "bvlcalexnet-9.onnx";
            string strUrl = "https://github.com/onnx/models/raw/main/vision/classification/alexnet/model/" + strModel;
            return download(strModel, strUrl, dfSizeInMb);
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

            CudaDnn<T> cuda = null;

            try
            {
                cuda = new CudaDnn<T>(0);
                MyCaffeModelData data = convert.ConvertOnnxToMyCaffeFromFile(cuda, m_log, strOnnxFile, true, false, dsTraining);
                Trace.WriteLine(convert.ReportString);

                data.Save(strTestPath, "AlexNet" + ((dsTraining != null) ? ".train" : "") + "." + typeof(T).ToString());
            }
            catch (Exception excpt)
            {
                throw excpt;
            }
            finally
            {
                cuda.Dispose();
                convert.Dispose();
            }
        }

        private string downloadOnnxGoogNetModel()
        {
            // Download a small onnx model from https://github.com/onnx (dowload is 26.7mb)
            double dfSizeInMb = 26.7;
            string strModel = "googlenet-9.onnx";
            string strUrl = "https://github.com/onnx/models/raw/main/vision/classification/inception_and_googlenet/googlenet/model/" + strModel;
            return download(strModel, strUrl, dfSizeInMb);
        }

        public void TestImportOnnxGoogNet(string strTrainingDs)
        {
            string strTestPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\models\\onnx\\imported\\goognet";
            if (!Directory.Exists(strTestPath))
                Directory.CreateDirectory(strTestPath);

            string strOnnxFile = downloadOnnxGoogNetModel();
            MyCaffeConversionControl<T> convert = new MyCaffeConversionControl<T>();

            DatasetDescriptor dsTraining = null;
            if (strTrainingDs != null)
            {
                DatasetFactory factory = new DatasetFactory();
                dsTraining = factory.LoadDataset(strTrainingDs);
            }

            CudaDnn<T> cuda = null;

            try
            {
                cuda = new CudaDnn<T>(0);
                MyCaffeModelData data = convert.ConvertOnnxToMyCaffeFromFile(cuda, m_log, strOnnxFile, true, false, dsTraining);
                Trace.WriteLine(convert.ReportString);

                data.Save(strTestPath, "GoogNet" + ((dsTraining != null) ? ".train" : "") + "." + typeof(T).ToString());
            }
            catch (Exception excpt)
            {
                throw excpt;
            }
            finally
            {
                cuda.Dispose();
                convert.Dispose();
            }
        }

        private string downloadOnnxVGG19Model()
        {
            // Download a small onnx model from https://github.com/onnx (dowload is 548mb)
            double dfSizeInMb = 548;
            string strModel = "vgg19-7.onnx";
            string strUrl = "https://github.com/onnx/models/raw/main/vision/classification/vgg/model/" + strModel;
            return download(strModel, strUrl, dfSizeInMb);
        }

        public void TestImportOnnxVGG19(string strTrainingDs)
        {
            string strTestPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\models\\onnx\\imported\\vgg";
            if (!Directory.Exists(strTestPath))
                Directory.CreateDirectory(strTestPath);

            string strOnnxFile = downloadOnnxVGG19Model();
            MyCaffeConversionControl<T> convert = new MyCaffeConversionControl<T>();

            DatasetDescriptor dsTraining = null;
            if (strTrainingDs != null)
            {
                DatasetFactory factory = new DatasetFactory();
                dsTraining = factory.LoadDataset(strTrainingDs);
            }

            CudaDnn<T> cuda = null;

            try
            {
                cuda = new CudaDnn<T>(0);
                MyCaffeModelData data = convert.ConvertOnnxToMyCaffeFromFile(cuda, m_log, strOnnxFile, true, false, dsTraining);
                Trace.WriteLine(convert.ReportString);

                data.Save(strTestPath, "VGG19" + ((dsTraining != null) ? ".train" : "") + "." + typeof(T).ToString());
            }
            catch (Exception excpt)
            {
                throw excpt;
            }
            finally
            {
                cuda.Dispose();
                convert.Dispose();
            }
        }

        private string downloadOnnxResNet50Model()
        {
            // Download a small onnx model from https://github.com/onnx (dowload is 548mb)
            double dfSizeInMb = 97.8;
            string strModel = "resnet50-v1-7.onnx";
            string strUrl = "https://github.com/onnx/models/raw/main/vision/classification/resnet/model/" + strModel;
            return download(strModel, strUrl, dfSizeInMb);
        }

        public void TestImportOnnxResNet50(string strTrainingDs)
        {
            string strTestPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\models\\onnx\\imported\\resnet";
            if (!Directory.Exists(strTestPath))
                Directory.CreateDirectory(strTestPath);

            string strOnnxFile = downloadOnnxResNet50Model();
            MyCaffeConversionControl<T> convert = new MyCaffeConversionControl<T>();

            DatasetDescriptor dsTraining = null;
            if (strTrainingDs != null)
            {
                DatasetFactory factory = new DatasetFactory();
                dsTraining = factory.LoadDataset(strTrainingDs);
            }

            CudaDnn<T> cuda = null;

            try
            {
                cuda = new CudaDnn<T>(0);
                MyCaffeModelData data = convert.ConvertOnnxToMyCaffeFromFile(cuda, m_log, strOnnxFile, true, false, dsTraining);
                Trace.WriteLine(convert.ReportString);

                data.Save(strTestPath, "ResNet50" + ((dsTraining != null) ? ".train" : "") + "." + typeof(T).ToString());
            }
            catch (Exception excpt)
            {
                throw excpt;
            }
            finally
            {
                cuda.Dispose();
                convert.Dispose();
            }
        }

        private string downloadOnnxInceptionV1Model()
        {
            // Download a small onnx model from https://github.com/onnx (dowload is 26.7mb)
            double dfSizeInMb = 26.7;
            string strModel = "inception-v1-9.onnx";
            string strUrl = "https://github.com/onnx/models/raw/main/vision/classification/inception_and_googlenet/inception_v1/model/" + strModel;
            return download(strModel, strUrl, dfSizeInMb);
        }

        public void TestImportOnnxInceptionV1(string strTrainingDs)
        {
            string strTestPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\models\\onnx\\imported\\inception";
            if (!Directory.Exists(strTestPath))
                Directory.CreateDirectory(strTestPath);

            string strOnnxFile = downloadOnnxInceptionV1Model();
            MyCaffeConversionControl<T> convert = new MyCaffeConversionControl<T>();

            DatasetDescriptor dsTraining = null;
            if (strTrainingDs != null)
            {
                DatasetFactory factory = new DatasetFactory();
                dsTraining = factory.LoadDataset(strTrainingDs);
            }

            CudaDnn<T> cuda = null;

            try
            {
                cuda = new CudaDnn<T>(0);
                MyCaffeModelData data = convert.ConvertOnnxToMyCaffeFromFile(cuda, m_log, strOnnxFile, true, false, dsTraining);
                Trace.WriteLine(convert.ReportString);

                data.Save(strTestPath, "Inception" + ((dsTraining != null) ? ".train" : "") + "." + typeof(T).ToString());
            }
            catch (Exception excpt)
            {
                throw excpt;
            }
            finally
            {
                cuda.Dispose();
                convert.Dispose();
            }
        }

        private string downloadOnnxInceptionV2Model()
        {
            // Download a small onnx model from https://github.com/onnx (dowload is 26.7mb)
            double dfSizeInMb = 26.7;
            string strModel = "inception-v2-9.onnx";
            string strUrl = "https://github.com/onnx/models/raw/main/vision/classification/inception_and_googlenet/inception_v2/model/" + strModel;
            return download(strModel, strUrl, dfSizeInMb);
        }

        public void TestImportOnnxInceptionV2(string strTrainingDs, string strIgnoreLayer)
        {
            string strTestPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\models\\onnx\\imported\\inception";
            if (!Directory.Exists(strTestPath))
                Directory.CreateDirectory(strTestPath);

            string strOnnxFile = downloadOnnxInceptionV2Model();
            MyCaffeConversionControl<T> convert = new MyCaffeConversionControl<T>();

            if (!string.IsNullOrEmpty(strIgnoreLayer))
                convert.IgnoreLayerNames.Add(strIgnoreLayer);

            DatasetDescriptor dsTraining = null;
            if (strTrainingDs != null)
            {
                DatasetFactory factory = new DatasetFactory();
                dsTraining = factory.LoadDataset(strTrainingDs);
            }

            CudaDnn<T> cuda = null;

            try
            {
                cuda = new CudaDnn<T>(0);
                MyCaffeModelData data = convert.ConvertOnnxToMyCaffeFromFile(cuda, m_log, strOnnxFile, true, false, dsTraining);
                Trace.WriteLine(convert.ReportString);

                data.Save(strTestPath, "Inception" + ((dsTraining != null) ? ".train" : "") + "." + typeof(T).ToString());
            }
            catch (Exception excpt)
            {
                throw excpt;
            }
            finally
            {
                cuda.Dispose();
                convert.Dispose();
            }
        }
    }

    class ModelProtoComparer
    {
        Log m_log;

        public ModelProtoComparer(Log log)
        {
            m_log = log;
        }

        public bool Compare(ModelProto m1, ModelProto m2)
        {
            if (m1.ModelVersion != m2.ModelVersion)
                m_log.FAIL("The model versions do not match!");

            if (m1.IrVersion != m2.IrVersion && m2.IrVersion != 1)
                m_log.FAIL("The IrVersions do not match!");

            if (!compareRepeatedField("OpSetIDProto", m1.OpsetImport, m2.OpsetImport))
                return false;

            if (!compareRepeatedField("Graph.Initializer", m1.Graph.Initializer, m2.Graph.Initializer))
                return false;

            if (!compareRepeatedField("Graph.Input", m1.Graph.Input, m2.Graph.Input))
                return false;

            if (!compareRepeatedField("Graph.Output", m1.Graph.Output, m2.Graph.Output))
                return false;

            return true;
        }

        private bool compareRepeatedField(string str, Google.Protobuf.Collections.RepeatedField<NodeProto> rg1, Google.Protobuf.Collections.RepeatedField<NodeProto> rg2)
        {
            m_log.CHECK_GE(rg1.Count, rg2.Count, str + ": The counts do not match!");
            for (int i = 0; i < rg2.Count; i++)
            {
                if (!compareNodeProto(rg1[i], rg2[i]))
                    return false;
            }
            return true;
        }

        private bool compareNodeProto(NodeProto t1, NodeProto t2)
        {
            if (t1.Name != t2.Name)
                m_log.FAIL("The value info names do not match!");

            if (!compareRepeatedField("Input", t1.Input, t2.Input))
                return false;

            if (!compareRepeatedField("Output", t1.Output, t2.Output))
                return false;

            if (t1.OpType != t2.OpType)
                m_log.FAIL("The OpTypes do not match!");

            if (!compareRepeatedField("Node.Attribute", t1.Attribute, t2.Attribute))
                return false;

            return true;
        }

        private bool compareRepeatedField(string str, Google.Protobuf.Collections.RepeatedField<AttributeProto> rg1, Google.Protobuf.Collections.RepeatedField<AttributeProto> rg2)
        {
            m_log.CHECK_EQ(rg1.Count, rg2.Count, str + ": The counts do not match!");
            for (int i = 0; i < rg1.Count; i++)
            {
                if (!compareAttributeProto(str, rg1[i], rg2[i]))
                    return false;
            }
            return true;
        }

        private bool compareAttributeProto(string str, AttributeProto t1, AttributeProto t2)
        {
            if (t1.Name != t2.Name)
                m_log.FAIL(str + ": The names do not match!");

            string str1 = t1.ToString();
            string str2 = t2.ToString();

            if (str1 != str2)
                m_log.FAIL(str + ": The attributes do not match!");

            return true;
        }

        private bool compareRepeatedField(string str, Google.Protobuf.Collections.RepeatedField<ValueInfoProto> rg1, Google.Protobuf.Collections.RepeatedField<ValueInfoProto> rg2, bool bExactCount = false)
        {
            if (bExactCount)
                m_log.CHECK_EQ(rg1.Count, rg2.Count, str + ": The counts do not match!");
            else
                m_log.CHECK_GE(rg1.Count, rg2.Count, str + ": The counts do not match!");

            for (int i = 0; i < rg2.Count; i++)
            {
                if (!compareValueInfoProto(rg1[i], rg2[i], false))
                    return false;
            }
            return true;
        }

        private bool compareValueInfoProto(ValueInfoProto t1, ValueInfoProto t2, bool bCompareNames)
        {
            if (!compareTypeProto("Type", t1.Type, t2.Type))
                return false;

            if (bCompareNames && t1.Name != t2.Name)
                m_log.FAIL("The value info names do not match!");

            return true;
        }

        private bool compareTypeProto(string str, TypeProto t1, TypeProto t2)
        {
            string str1 = t1.ToString();
            string str2 = t2.ToString();

            if (str1 != str2)
                m_log.FAIL(str + ": The type protos do not match!");

            return true;
        }

        private bool compareRepeatedField(string str, Google.Protobuf.Collections.RepeatedField<OperatorSetIdProto> rg1, Google.Protobuf.Collections.RepeatedField<OperatorSetIdProto> rg2)
        {
            m_log.CHECK_EQ(rg1.Count, rg2.Count, str + ": The counts do not match!");
            for (int i = 0; i < rg1.Count; i++)
            {
                if (!compareOperatorSetIdProto(rg1[i], rg2[i]))
                    return false;
            }
            return true;
        }

        private bool compareOperatorSetIdProto(OperatorSetIdProto o1, OperatorSetIdProto o2)
        {
            string str = "OperatorSetIDProto";
            m_log.CHECK_GE(o2.Version, o1.Version, str + ": The versions do not match.");

            if (o1.Domain != o2.Domain)
                m_log.FAIL(str + ": The domains do not match!");

            return true;
        }

        private bool compareRepeatedField(string str, Google.Protobuf.Collections.RepeatedField<TensorProto> rg1, Google.Protobuf.Collections.RepeatedField<TensorProto> rg2)
        {
            m_log.CHECK_LE(rg2.Count, rg1.Count, str + ": The counts do not match!");
            for (int i = 0; i < rg2.Count - 1; i++)
            {
                Google.Protobuf.Collections.RepeatedField<long> dim2 = rg2[i].Dims;
                int nIdx1 = -1;
                bool bResized = false;

                for (int j=0; j<rg1.Count; j++)
                {
                    Google.Protobuf.Collections.RepeatedField<long> dim1 = rg1[j].Dims;
                    if (compare(dim1, dim2, out bResized))
                    {
                        nIdx1 = j;
                        break;
                    }
                }

                if (nIdx1 >= 0)
                {
                    if (!compareTensorProto(rg1[nIdx1], rg2[i], bResized))
                        return false;
                }
            }

            return true;
        }

        private bool compare(Google.Protobuf.Collections.RepeatedField<long> rg1, Google.Protobuf.Collections.RepeatedField<long> rg2, out bool bReSized)
        {
            bReSized = false;

            if (rg1.Count != rg2.Count)
            {
                int nIdx = rg1.Count - 1;
                while (nIdx > 0)
                {
                    if (rg1[nIdx] == 1)
                        rg1.RemoveAt(nIdx);
                    else
                        break;

                    nIdx--;
                    bReSized = true;
                }

                nIdx = rg2.Count - 1;
                while (nIdx > 0)
                {
                    if (rg2[nIdx] == 1)
                        rg2.RemoveAt(nIdx);
                    else
                        break;

                    nIdx--;
                    bReSized = true;
                }

                if (rg1.Count != rg2.Count)
                    return false;
            }

            for (int i = 0; i < rg1.Count; i++)
            {
                if (rg1[i] != rg2[i])
                    return false;
            }

            return true;
        }

        private bool compareTensorProto(TensorProto t1, TensorProto t2, bool bResized = false)
        {
            double[] rgdf1 = MyCaffeConversionControl<double>.getDataAsDouble(t1);
            double[] rgdf2 = MyCaffeConversionControl<double>.getDataAsDouble(t2);

            m_log.CHECK_EQ(rgdf1.Length, rgdf2.Length, "The lengths do not match!");

            if (!bResized)
            {
                for (int i = 0; i < rgdf1.Length; i++)
                {
                    m_log.EXPECT_NEAR_FLOAT(rgdf1[i], rgdf2[i], 0.0000001, "The data items do not match!");
                }
            }

            return true;
        }

        private bool compareRepeatedField(string str, Google.Protobuf.Collections.RepeatedField<int> f1, Google.Protobuf.Collections.RepeatedField<int> f2, Google.Protobuf.ByteString b1, Google.Protobuf.ByteString b2)
        {
            m_log.CHECK_EQ(f1.Count, f2.Count, str + ": The counts do not match!");
            for (int i = 0; i < f1.Count; i++)
            {
                m_log.CHECK_EQ(f1[i], f2[i], str + ": The int fields do not match!");
            }
            return true;
        }

        private bool compareRepeatedField(string str, Google.Protobuf.Collections.RepeatedField<long> f1, Google.Protobuf.Collections.RepeatedField<long> f2, Google.Protobuf.ByteString b1, Google.Protobuf.ByteString b2)
        {
            m_log.CHECK_EQ(f1.Count, f2.Count, str + ": The counts do not match!");
            for (int i = 0; i < f1.Count; i++)
            {
                m_log.CHECK_EQ(f1[i], f2[i], str + ": The long fields do not match!");
            }
            return true;
        }

        private bool compareRepeatedField(string str, Google.Protobuf.Collections.RepeatedField<float> f1, Google.Protobuf.Collections.RepeatedField<float> f2, Google.Protobuf.ByteString b1, Google.Protobuf.ByteString b2)
        {
            m_log.CHECK_EQ(f1.Count, f2.Count, str + ": The counts do not match!");
            for (int i = 0; i < f1.Count; i++)
            {
                m_log.CHECK_EQ(f1[i], f2[i], str + ": The float fields do not match!");
            }
            return true;
        }

        private bool compareRepeatedField(string str, Google.Protobuf.Collections.RepeatedField<double> f1, Google.Protobuf.Collections.RepeatedField<double> f2, Google.Protobuf.ByteString b1, Google.Protobuf.ByteString b2)
        {
            m_log.CHECK_EQ(f1.Count, f2.Count, str + ": The counts do not match!");
            for (int i = 0; i < f1.Count; i++)
            {
                m_log.CHECK_EQ(f1[i], f2[i], str + ": The double fields do not match!");
            }
            return true;
        }

        private bool compareRepeatedField(string str, Google.Protobuf.Collections.RepeatedField<string> f1, Google.Protobuf.Collections.RepeatedField<string> f2)
        {
            m_log.CHECK_EQ(f1.Count, f2.Count, str + ": The counts do not match!");
            for (int i = 0; i < f1.Count; i++)
            {
                if (f1[i] != f2[i])
                    m_log.FAIL(str + ": The string fields do not match!");
            }
            return true;
        }
    }
}
