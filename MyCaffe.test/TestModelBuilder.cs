using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.param;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.fillers;
using MyCaffe.layers;
using MyCaffe.db.image;
using MyCaffe.basecode.descriptors;
using MyCaffe.data;
using MyCaffe.layers.beta;
using MyCaffe.model;
using System.IO;
using MyCaffe.param.gpt;
using MyCaffe.layers.gpt;
using System.Diagnostics;

/// <summary>
/// Testing the Model Builders.
/// </summary> 
namespace MyCaffe.test
{
    [TestClass]
    public class TestModelBuilder
    {
        [TestMethod]
        public void TestSSDPascal_CreateSolver()
        {
            SSDPascalModelBuilderTest test = new SSDPascalModelBuilderTest();

            try
            {
                foreach (IModelBuilderTest t in test.Tests)
                {
                    t.TestCreateSolver();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSSDPascal_CreateDeployModel()
        {
            SSDPascalModelBuilderTest test = new SSDPascalModelBuilderTest();

            try
            {
                foreach (IModelBuilderTest t in test.Tests)
                {
                    t.TestCreateDeployModel();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSSDPascal_CreateTrainingModel()
        {
            SSDPascalModelBuilderTest test = new SSDPascalModelBuilderTest();

            try
            {
                foreach (IModelBuilderTest t in test.Tests)
                {
                    t.TestCreateTrainingModel();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestResNet101_CreateSolver()
        {
            ResNetModelBuilderTest test = new ResNetModelBuilderTest(ResNetModelBuilder<float>.MODEL.RESNET101);

            try
            {
                foreach (IModelBuilderTest t in test.Tests)
                {
                    t.TestCreateSolver();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestResNet101_CreateDeployModel()
        {
            ResNetModelBuilderTest test = new ResNetModelBuilderTest(ResNetModelBuilder<float>.MODEL.RESNET101);

            try
            {
                foreach (IModelBuilderTest t in test.Tests)
                {
                    t.TestCreateDeployModel();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestResNet101_CreateTrainingModel()
        {
            ResNetModelBuilderTest test = new ResNetModelBuilderTest(ResNetModelBuilder<float>.MODEL.RESNET101);

            try
            {
                foreach (IModelBuilderTest t in test.Tests)
                {
                    t.TestCreateTrainingModel();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestResNet152_CreateSolver()
        {
            ResNetModelBuilderTest test = new ResNetModelBuilderTest(ResNetModelBuilder<float>.MODEL.RESNET152);

            try
            {
                foreach (IModelBuilderTest t in test.Tests)
                {
                    t.TestCreateSolver();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestResNet152_CreateDeployModel()
        {
            ResNetModelBuilderTest test = new ResNetModelBuilderTest(ResNetModelBuilder<float>.MODEL.RESNET152);

            try
            {
                foreach (IModelBuilderTest t in test.Tests)
                {
                    t.TestCreateDeployModel();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestResNet152_CreateTrainingModel()
        {
            ResNetModelBuilderTest test = new ResNetModelBuilderTest(ResNetModelBuilder<float>.MODEL.RESNET152);

            try
            {
                foreach (IModelBuilderTest t in test.Tests)
                {
                    t.TestCreateTrainingModel();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestResNet56_Siamese_CreateSolver()
        {
            ResNetModelBuilderTest test = new ResNetModelBuilderTest(ResNetModelBuilder<float>.MODEL.RESNET56, true);

            try
            {
                foreach (IModelBuilderTest t in test.Tests)
                {
                    t.TestCreateSolver();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestResNet56_Siamese_CreateDeployModel()
        {
            ResNetModelBuilderTest test = new ResNetModelBuilderTest(ResNetModelBuilder<float>.MODEL.RESNET56, true);

            try
            {
                foreach (IModelBuilderTest t in test.Tests)
                {
                    t.TestCreateDeployModel();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestResNet56_Siamease_CreateTrainingModel()
        {
            ResNetModelBuilderTest test = new ResNetModelBuilderTest(ResNetModelBuilder<float>.MODEL.RESNET56, true);

            try
            {
                foreach (IModelBuilderTest t in test.Tests)
                {
                    t.TestCreateTrainingModel();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestResNet101_Siamese_CreateSolver()
        {
            ResNetModelBuilderTest test = new ResNetModelBuilderTest(ResNetModelBuilder<float>.MODEL.RESNET101, true);

            try
            {
                foreach (IModelBuilderTest t in test.Tests)
                {
                    t.TestCreateSolver();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestResNet101_Siamese_CreateDeployModel()
        {
            ResNetModelBuilderTest test = new ResNetModelBuilderTest(ResNetModelBuilder<float>.MODEL.RESNET101, true);

            try
            {
                foreach (IModelBuilderTest t in test.Tests)
                {
                    t.TestCreateDeployModel();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestResNet101_Siamease_CreateTrainingModel()
        {
            ResNetModelBuilderTest test = new ResNetModelBuilderTest(ResNetModelBuilder<float>.MODEL.RESNET101, true);

            try
            {
                foreach (IModelBuilderTest t in test.Tests)
                {
                    t.TestCreateTrainingModel();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestResNet152_Siamese_CreateSolver()
        {
            ResNetModelBuilderTest test = new ResNetModelBuilderTest(ResNetModelBuilder<float>.MODEL.RESNET152, true);

            try
            {
                foreach (IModelBuilderTest t in test.Tests)
                {
                    t.TestCreateSolver();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestResNet152_Siamease_CreateDeployModel()
        {
            ResNetModelBuilderTest test = new ResNetModelBuilderTest(ResNetModelBuilder<float>.MODEL.RESNET152, true);

            try
            {
                foreach (IModelBuilderTest t in test.Tests)
                {
                    t.TestCreateDeployModel();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestResNet152_Siamese_CreateTrainingModel()
        {
            ResNetModelBuilderTest test = new ResNetModelBuilderTest(ResNetModelBuilder<float>.MODEL.RESNET152, true);

            try
            {
                foreach (IModelBuilderTest t in test.Tests)
                {
                    t.TestCreateTrainingModel();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestOctConvResNet26_CreateTrainingModel()
        {
            ResNetOctConvModelBuilderTest test = new ResNetOctConvModelBuilderTest(ResNetOctConvModelBuilder<float>.MODEL.RESNET26);

            try
            {
                foreach (IModelBuilderTest t in test.Tests)
                {
                    t.TestCreateTrainingModel();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestOctConvResNet26_CreateSolver()
        {
            ResNetOctConvModelBuilderTest test = new ResNetOctConvModelBuilderTest(ResNetOctConvModelBuilder<float>.MODEL.RESNET26);

            try
            {
                foreach (IModelBuilderTest t in test.Tests)
                {
                    t.TestCreateSolver();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestLlamaModel_CreateTrainingModel()
        {
            LlamaModelBuilderTest test = new LlamaModelBuilderTest("Llama7B");

            try
            {
                foreach (IModelBuilderTest t in test.Tests)
                {
                    if (t.DataType == DataType.DOUBLE)
                        continue;
                    t.TestCreateTrainingModel();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestLlamaModel_CreateSolver()
        {
            LlamaModelBuilderTest test = new LlamaModelBuilderTest("Llama7B");

            try
            {
                foreach (IModelBuilderTest t in test.Tests)
                {
                    if (t.DataType == DataType.DOUBLE)
                        continue;
                    t.TestCreateSolver();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface IModelBuilderTest : ITest
    {
        void TestCreateSolver();
        void TestCreateTrainingModel();
        void TestCreateDeployModel();
    }

    class ResNetModelBuilderTest : TestBase
    {
        public ResNetModelBuilderTest(ResNetModelBuilder<float>.MODEL model, bool bSiamese = false, EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("ResNet Model Builder Test" + ((bSiamese) ? " SIAMESE" : "") + " " + model.ToString(), TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (strName.Contains("SIAMESE"))
            {
                if (dt == common.DataType.DOUBLE)
                    return new ResNetSiameseModelBuilderTest<double>(strName, nDeviceID, engine);
                else
                    return new ResNetSiameseModelBuilderTest<float>(strName, nDeviceID, engine);
            }
            else
            {
                if (dt == common.DataType.DOUBLE)
                    return new ResNetModelBuilderTest<double>(strName, nDeviceID, engine);
                else
                    return new ResNetModelBuilderTest<float>(strName, nDeviceID, engine);
            }
        }
    }

    class ResNetSiameseModelBuilderTest<T> : ModelBuilderTest<T>
    {
        ResNetModelBuilder<T>.MODEL m_model;

        public ResNetSiameseModelBuilderTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, nDeviceID, engine)
        {
            if (strName.Contains(ResNetModelBuilder<T>.MODEL.RESNET101.ToString()))
                m_model = ResNetModelBuilder<T>.MODEL.RESNET101;
            else if (strName.Contains(ResNetModelBuilder<T>.MODEL.RESNET152.ToString()))
                m_model = ResNetModelBuilder<T>.MODEL.RESNET152;
            else
                m_model = ResNetModelBuilder<T>.MODEL.RESNET56;
        }

        protected override ModelBuilder<T> create()
        {
            List<Tuple<int, bool>> rgIP = new List<Tuple<int, bool>>();
            rgIP.Add(new Tuple<int, bool>(1024, false));
            rgIP.Add(new Tuple<int, bool>(512, true));
            rgIP.Add(new Tuple<int, bool>(10, false));
            int nBatch = (m_model == ResNetModelBuilder<T>.MODEL.RESNET56) ? 32 : (m_model == ResNetModelBuilder<T>.MODEL.RESNET101) ? 16 : 12;
            return new ResNetModelBuilder<T>(m_strBaseDir, "CIFAR-10", 3, true, rgIP, true, false, m_model, nBatch, nBatch);
        }
    }

    class ResNetOctConvModelBuilderTest : TestBase
    {
        public ResNetOctConvModelBuilderTest(ResNetOctConvModelBuilder<float>.MODEL model, EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("ResNet OctConv Model Builder Test" + " " + model.ToString(), TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new ResNetOctConvModelBuilderTest<double>(strName, nDeviceID, engine);
            else
                return new ResNetOctConvModelBuilderTest<float>(strName, nDeviceID, engine);
        }
    }

    class ResNetModelBuilderTest<T> : ModelBuilderTest<T>
    {
        ResNetModelBuilder<T>.MODEL m_model;

        public ResNetModelBuilderTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, nDeviceID, engine)
        {
            if (strName.Contains(ResNetModelBuilder<T>.MODEL.RESNET152.ToString()))
                m_model = ResNetModelBuilder<T>.MODEL.RESNET152;
            else if (strName.Contains(ResNetModelBuilder<T>.MODEL.RESNET101.ToString()))
                m_model = ResNetModelBuilder<T>.MODEL.RESNET101;
            else
                m_model = ResNetModelBuilder<T>.MODEL.RESNET56;
        }

        protected override ModelBuilder<T> create()
        {
            List<Tuple<int, bool>> rgIP = new List<Tuple<int, bool>>();
            rgIP.Add(new Tuple<int, bool>(10, false));
            return new ResNetModelBuilder<T>(m_strBaseDir, "CIFAR-10", 3, false, rgIP, true, false, m_model);
        }
    }

    class ResNetOctConvModelBuilderTest<T> : ModelBuilderTest<T>
    {
        ResNetOctConvModelBuilder<T>.MODEL m_model;

        public ResNetOctConvModelBuilderTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, nDeviceID, engine)
        {
            m_model = ResNetOctConvModelBuilder<T>.MODEL.RESNET26;
        }

        protected override ModelBuilder<T> create()
        {
            List<Tuple<int, bool>> rgIP = new List<Tuple<int, bool>>();
            rgIP.Add(new Tuple<int, bool>(10, false));
            return new ResNetOctConvModelBuilder<T>(m_strBaseDir, "CIFAR-10", rgIP, m_model);
        }
    }

    class SSDPascalModelBuilderTest : TestBase
    {
        public SSDPascalModelBuilderTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("SSD Pascal Model Builder Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new SSDModelBuilderTest<double>(strName, nDeviceID, engine);
            else
                return new SSDModelBuilderTest<float>(strName, nDeviceID, engine);
        }
    }

    class SSDModelBuilderTest<T> : ModelBuilderTest<T>
    {
        public SSDModelBuilderTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, nDeviceID, engine)
        {
            m_engine = engine;
            m_strBaseDir = TestBase.GetTestPath("\\MyCaffe\\test_data", true, true);
        }

        protected override ModelBuilder<T> create()
        {
            return new SsdPascalModelBuilder<T>(m_strBaseDir);
        }
    }

    class LlamaModelBuilderTest : TestBase
    {
        string m_strModel = "Llama7B";

        public LlamaModelBuilderTest(string strModel)
            : base("Llama 7B Model Builder Test", TestBase.DEFAULT_DEVICE_ID, EngineParameter.Engine.DEFAULT)
        {
            m_strModel = strModel;
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new LlamaModelBuilderTest<double>(strName, nDeviceID, m_strModel, engine);
            else
                return new LlamaModelBuilderTest<float>(strName, nDeviceID, m_strModel, engine);
        }
    }

    class LlamaModelBuilderTest<T> : ModelBuilderTest<T>
    {
        string m_strModel = "Llama7B";

        public LlamaModelBuilderTest(string strName, int nDeviceID, string strModel, EngineParameter.Engine engine)
            : base(strName, nDeviceID, engine)
        {
            m_engine = engine;
            m_strBaseDir = TestBase.GetTestPath("\\MyCaffe\\test_data", true, true);
            m_strModel = strModel;
        }

        protected override ModelBuilder<T> create()
        {
            return new LlamaModelBuilder<T>(m_strBaseDir, m_strModel);
        }
    }

    abstract class ModelBuilderTest<T> : TestEx<T>, IModelBuilderTest
    {
        protected string m_strBaseDir;
        protected string m_strName;

        public ModelBuilderTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_engine = engine;
            m_strName = strName;
            m_strBaseDir = TestBase.GetTestPath("\\MyCaffe\\test_data", true, true);
        }

        protected override void dispose()
        {
            base.dispose();
        }

        protected abstract ModelBuilder<T> create();

        protected void save(string strModel, string strSolver, bool bDeploy)
        {
            string strPath = m_strBaseDir + "\\models\\test\\";

            if (!Directory.Exists(strPath))
                Directory.CreateDirectory(strPath);

            string strModelFile = null;
            string strSolverFile = null;
            string strType = (bDeploy) ? "deploy" : "train_val";

            if (m_strName.Contains("SSD"))
            {
                strModelFile = strPath + "ssd_" + strType + ".prototxt";
                if (strSolver != null)
                    strSolverFile = strPath + "ssd_solver.prototxt";
            }
            else if (m_strName.Contains("ResNet"))
            {
                ResNetModelBuilder<T>.MODEL model = ResNetModelBuilder<T>.MODEL.RESNET101;
                if (m_strName.Contains("152"))
                    model = ResNetModelBuilder<T>.MODEL.RESNET152;
                else if (m_strName.Contains("101"))
                    model = ResNetModelBuilder<T>.MODEL.RESNET101;
                else
                    model = ResNetModelBuilder<T>.MODEL.RESNET56;

                // NOTE: These models are big and can require 40+ GB of Video Memory depending on the batch sizes used.
                if (m_strName.Contains("SIAMESE"))
                {
                    strModelFile = strPath + model.ToString().ToLower() + "_siamese_" + strType + ".prototxt";
                    if (strSolver != null)
                        strSolverFile = strPath + model.ToString().ToLower() + "_siamese_solver.prototxt";
                }
                else
                {
                    strModelFile = strPath + model.ToString().ToLower() + "_" + strType + ".prototxt";
                    if (strSolver != null)
                        strSolverFile = strPath + model.ToString().ToLower() + "_solver.prototxt";
                }
            }

            if (strModelFile != null)
            {
                using (StreamWriter sw = new StreamWriter(strModelFile))
                {
                    sw.Write(strModel);
                }
            }

            if (strSolverFile != null)
            {
                using (StreamWriter sw = new StreamWriter(strSolverFile))
                {
                    sw.Write(strSolver);
                }
            }
        }

        public void TestCreateSolver()
        {
            ModelBuilder<T> builder = create();

            SolverParameter solverParam = builder.CreateSolver();
            RawProto proto = solverParam.ToProto("root");
            string strSolver = proto.ToString();

            RawProto proto2 = RawProto.Parse(strSolver);
            SolverParameter solverParam2 = SolverParameter.FromProto(proto2);

            m_log.CHECK(solverParam2.Compare(solverParam), "The two solver parameters should be the same!");
        }

        public void TestCreateTrainingModel()
        {
            ModelBuilder<T> builder = create();

            PropertySet prop = new PropertySet();
            prop.SetProperty("VocabularyType", ((int)TokenizedDataParameter.VOCABULARY_TYPE.LLAMA2).ToString());
            NetParameter net_param = builder.CreateModel(prop);
            net_param.enable_memory_stats = true;
            RawProto proto = net_param.ToProto("root");
            string strNet = proto.ToString();

            RawProto proto2 = RawProto.Parse(strNet);
            NetParameter net_param2 = NetParameter.FromProto(proto2);

            m_log.CHECK(net_param2.Compare(net_param), "The two net parameters should be the same!");

            // verify creating the model.
            SolverParameter solver = builder.CreateSolver();
            RawProto protoSolver = solver.ToProto("root");
            string strSolver = protoSolver.ToString();

            CudaDnn<T> cuda = new CudaDnn<T>(0);
            int nDevCount = cuda.GetDeviceCount();
            cuda.Dispose();

            SettingsCaffe settings = new SettingsCaffe();
            settings.GpuIds = (nDevCount > 1) ? "1" : "0";
            CancelEvent evtCancel = new CancelEvent();
            MyCaffeControl<T> mycaffe = new MyCaffeControl<T>(settings, m_log, evtCancel);

            try
            {
                save(strNet, strSolver, false);

                mycaffe.LoadLite(Phase.TRAIN, strSolver, strNet, null, false, false);

                string strModelPath = "C:\\temp\\projects\\llama2\\llama2\\models\\llama2_7b_chat.bin";
                if (!File.Exists(strModelPath))
                    throw new Exception("Could not find the model file '" + strModelPath + "'!");

                Net<T> net = mycaffe.GetInternalNet(Phase.TRAIN);
                builder.LoadWeights(net.learnable_parameters, strModelPath, "KPTH1");

                TokenizedDataLayer<T> tok = net.FindLayer(LayerParameter.LayerType.TOKENIZED_DATA, "data") as TokenizedDataLayer<T>;

                PropertySet input = new PropertySet();
                string strPrompt = "What is your name?";
                int nMaxNewTokens = 50;
                Blob<T> blobTokdata = net.FindBlob("tokdata");
                Blob<T> blobLogits = net.FindBlob("logits");
                int nCurIdx = 0;
                float fTemperature = 0.1f;
                Stopwatch sw = new Stopwatch();

                int nSeqLen = 0;
                input.SetProperty("InputData", strPrompt);
                BlobCollection<T> colBtm = tok.PreProcessInput(input, out nSeqLen);
                blobTokdata.CopyFrom(colBtm[0], false, true);
                List<float> rgTokenIds = new List<float>();

                int[] rgShape = new int[2] { 1, 1 };

                rgTokenIds.AddRange(convertF(blobTokdata.update_cpu_data()));   

                sw.Start();
                double dfTotalTime = 0;

                for (int i = 0; i < nMaxNewTokens; i++)
                {                    
                    net.ForwardFromTo(3, 37);
                    blobLogits.scale_data(1.0f / fTemperature);

                    List<Tuple<string, int, double>> res = tok.PostProcessLogitsOutput(nCurIdx, blobLogits, null, 2, 10);
                    for (int j = 0; j < res.Count; j++)
                    {
                        rgTokenIds.Add(res[j].Item2);
                    }

                    while (rgTokenIds.Count > nSeqLen)
                    {
                        rgTokenIds.RemoveAt(0);
                    }

                    rgShape[1] = rgTokenIds.Count;
                    blobTokdata.Reshape(rgShape);
                    blobTokdata.mutable_cpu_data = convert(rgTokenIds.ToArray());

                    sw.Stop();
                    dfTotalTime += sw.Elapsed.TotalMilliseconds;

                    m_log.WriteLine("Processing prompt #" + i.ToString() + " average time " + (dfTotalTime / (i+1)).ToString("N3") + " ms.");

                    sw.Restart();
                }

                string strOutput = tok.Detokenize(rgTokenIds.ToArray(), 0, rgTokenIds.Count);
                m_log.WriteLine("Output: " + strOutput);
            }
            finally
            {
                mycaffe.Dispose();
            }
        }

        public void TestCreateDeployModel()
        {
            ModelBuilder<T> builder = create();

            NetParameter net_param = builder.CreateDeployModel();
            RawProto proto = net_param.ToProto("root");
            string strNet = proto.ToString();

            RawProto proto2 = RawProto.Parse(strNet);
            NetParameter net_param2 = NetParameter.FromProto(proto2);

            m_log.CHECK(net_param2.Compare(net_param), "The two net parameters should be the same!");

            // verify creating the model.
            SettingsCaffe settings = new SettingsCaffe();
            CancelEvent evtCancel = new CancelEvent();
            MyCaffeControl<T> mycaffe = new MyCaffeControl<T>(settings, m_log, evtCancel);

            save(strNet, null, true);

            mycaffe.LoadToRun(strNet, null, new BlobShape(1, 3, 300, 300));
            mycaffe.Dispose();
        }
    }
}
