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
using MyCaffe.solvers;

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
        public void TestSSDPascal_CreateInferenceModel()
        {
            SSDPascalModelBuilderTest test = new SSDPascalModelBuilderTest();

            try
            {
                foreach (IModelBuilderTest t in test.Tests)
                {
                    t.TestCreateInferenceModel(false);
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
                    t.TestCreateTrainingModel(false);
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
        public void TestResNet101_CreateInferenceModel()
        {
            ResNetModelBuilderTest test = new ResNetModelBuilderTest(ResNetModelBuilder<float>.MODEL.RESNET101);

            try
            {
                foreach (IModelBuilderTest t in test.Tests)
                {
                    t.TestCreateInferenceModel(false);
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
                    t.TestCreateTrainingModel(false);
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
        public void TestResNet152_CreateInferenceModel()
        {
            ResNetModelBuilderTest test = new ResNetModelBuilderTest(ResNetModelBuilder<float>.MODEL.RESNET152);

            try
            {
                foreach (IModelBuilderTest t in test.Tests)
                {
                    t.TestCreateInferenceModel(false);
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
                    t.TestCreateTrainingModel(false);
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
        public void TestResNet56_Siamese_CreateInferenceModel()
        {
            ResNetModelBuilderTest test = new ResNetModelBuilderTest(ResNetModelBuilder<float>.MODEL.RESNET56, true);

            try
            {
                foreach (IModelBuilderTest t in test.Tests)
                {
                    t.TestCreateInferenceModel(false);
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
                    t.TestCreateTrainingModel(false);
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
        public void TestResNet101_Siamese_CreateInferenceModel()
        {
            ResNetModelBuilderTest test = new ResNetModelBuilderTest(ResNetModelBuilder<float>.MODEL.RESNET101, true);

            try
            {
                foreach (IModelBuilderTest t in test.Tests)
                {
                    t.TestCreateInferenceModel(false);
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
                    t.TestCreateTrainingModel(false);
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
        public void TestResNet152_Siamease_CreateInferenceModel()
        {
            ResNetModelBuilderTest test = new ResNetModelBuilderTest(ResNetModelBuilder<float>.MODEL.RESNET152, true);

            try
            {
                foreach (IModelBuilderTest t in test.Tests)
                {
                    t.TestCreateInferenceModel(false);
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
                    t.TestCreateTrainingModel(false);
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
                    t.TestCreateTrainingModel(false);
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
        public void TestLlamaModel_CreateInferenceModel()
        {
            LlamaModelBuilderTest test = new LlamaModelBuilderTest("Llama7B");

            try
            {
                foreach (IModelBuilderTest t in test.Tests)
                {
                    if (t.DataType == DataType.DOUBLE)
                        continue;
                    t.TestCreateInferenceModel(true);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestLlamaModel_CreateInferenceModelNoLoRA()
        {
            LlamaModelBuilderTest test = new LlamaModelBuilderTest("Llama7B");

            try
            {
                foreach (IModelBuilderTest t in test.Tests)
                {
                    if (t.DataType == DataType.DOUBLE)
                        continue;
                    t.TestCreateInferenceModel(false);
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

        [TestMethod]
        public void TestTinyStoriesModel_CreateInferenceModel()
        {
            TinyStoriesModelBuilderTest test = new TinyStoriesModelBuilderTest("Stories15M");

            try
            {
                foreach (IModelBuilderTest t in test.Tests)
                {
                    if (t.DataType == DataType.DOUBLE)
                        continue;
                    t.TestCreateInferenceModel(true);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTinyStoriesModel_CreateTrainingModel_NoFineTune()
        {
            TinyStoriesModelBuilderTest test = new TinyStoriesModelBuilderTest("Stories15M");

            try
            {
                foreach (IModelBuilderTest t in test.Tests)
                {
                    if (t.DataType == DataType.DOUBLE)
                        continue;
                    t.TestCreateTrainingModel(false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTinyStoriesModel_CreateTrainingModel_FineTune()
        {
            TinyStoriesModelBuilderTest test = new TinyStoriesModelBuilderTest("Stories15M");

            try
            {
                foreach (IModelBuilderTest t in test.Tests)
                {
                    if (t.DataType == DataType.DOUBLE)
                        continue;
                    t.TestCreateTrainingModel(true);
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
        void TestCreateTrainingModel(bool bFineTune);
        void TestCreateInferenceModel(bool bEnableLoRA);
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
            return new LlamaModelBuilder<T>(m_strBaseDir, m_strModel, 1);
        }

        protected override void testCreateInferenceModel(bool bEnableLoRA)
        {
            ModelBuilder<T> builder = create();

            PropertySet prop = new PropertySet();
            prop.SetProperty("VocabularyType", ((int)TokenizedDataParameter.VOCABULARY_TYPE.LLAMA2).ToString());
            NetParameter net_param = builder.CreateModel(prop, Phase.TEST, bEnableLoRA);
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

                mycaffe.LoadLite(Phase.TRAIN, strSolver, strNet, null, null, false, false);

                string strModelPath = getTestDataLlamaPath("llama7b", "llama2_7b_chat.bin");

                Net<T> net = mycaffe.GetInternalNet(Phase.TRAIN);
                builder.LoadWeights(net.learnable_parameters, strModelPath, "KPTH1");

                TokenizedDataLayer<T> tok = net.FindLayer(LayerParameter.LayerType.TOKENIZED_DATA, "data") as TokenizedDataLayer<T>;

                PropertySet input = new PropertySet();
                string strPrompt = "[INST] What is your name? [/INST]";
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
                List<float> rgResponseTokens = new List<float>();

                rgTokenIds.AddRange(convertF(blobTokdata.update_cpu_data()));

                sw.Start();
                double dfTotalTime = 0;
                int nTokenId = (int)rgTokenIds[0];

                int[] rgShape = new int[2] { 1, 1 };
                blobTokdata.Reshape(rgShape);

                for (int i = 0; i < nMaxNewTokens; i++)
                {
                    blobTokdata.SetData(nTokenId, 0);

                    net.SetLayerOption("position", i);
                    net.ForwardFromTo(3, 37);

                    if (i < rgTokenIds.Count - 1)
                    {
                        nTokenId = (int)rgTokenIds[i + 1];
                    }
                    else
                    {
                        blobLogits.scale_data(1.0f / fTemperature);

                        List<Tuple<string, int, double>> res = tok.PostProcessLogitsOutput(nCurIdx, blobLogits, null, 2, 10);
                        nTokenId = res[0].Item2;

                        if (!tok.IsEOS(nTokenId))
                            rgResponseTokens.Add(nTokenId);
                    }

                    sw.Stop();
                    dfTotalTime += sw.Elapsed.TotalMilliseconds;

                    m_log.WriteLine("Processing prompt #" + i.ToString() + " average time " + (dfTotalTime / (i + 1)).ToString("N3") + " ms.", true);

                    sw.Restart();

                    if (tok.IsEOS(nTokenId))
                        break;
                }

                string strOutput = tok.Detokenize(rgResponseTokens.ToArray(), 0, rgResponseTokens.Count);
                m_log.WriteLine("Output: " + strOutput);
            }
            finally
            {
                mycaffe.Dispose();
            }
        }

        protected override void testCreateTrainingModel(bool bFineTune)
        {
        }
    }

    class TinyStoriesModelBuilderTest : TestBase
    {
        string m_strModel = "Stories15M";

        public TinyStoriesModelBuilderTest(string strModel)
            : base("TinyStories 15M Model Builder Test", TestBase.DEFAULT_DEVICE_ID, EngineParameter.Engine.DEFAULT)
        {
            m_strModel = strModel;
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new TinyStoriesModelBuilderTest<double>(strName, nDeviceID, m_strModel, engine);
            else
                return new TinyStoriesModelBuilderTest<float>(strName, nDeviceID, m_strModel, engine);
        }
    }

    class TinyStoriesModelBuilderTest<T> : ModelBuilderTest<T>
    {
        string m_strModel = "Stories15M";
        int m_nBatchSize = 1;
        int m_nSeqLen = 512;
        int m_nIterSize = 1;
        double m_dfDropout = 0.0;

        public TinyStoriesModelBuilderTest(string strName, int nDeviceID, string strModel, EngineParameter.Engine engine)
            : base(strName, nDeviceID, engine)
        {
            m_engine = engine;
            m_strBaseDir = TestBase.GetTestPath("\\MyCaffe\\test_data", true, true);
            m_strModel = strModel;
        }

        protected override ModelBuilder<T> create()
        {
            return new LlamaModelBuilder<T>(m_strBaseDir, m_strModel, m_nIterSize, (uint)m_nBatchSize, (uint)m_nSeqLen, 32000, m_dfDropout);
        }

        protected override void testCreateInferenceModel(bool bEnableLoRA)
        {
            m_strModel = "Stories15M";
            ModelBuilder<T> builder = create();

            PropertySet prop = new PropertySet();
            prop.SetProperty("VocabularyType", ((int)TokenizedDataParameter.VOCABULARY_TYPE.LLAMA2).ToString());
            NetParameter net_param = builder.CreateModel(prop, Phase.TEST, bEnableLoRA);
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

                mycaffe.LoadLite(Phase.TRAIN, strSolver, strNet, null, null, false, false);

                string strModelPath = getTestDataLlamaPath("stories", "stories15M.bin", "https://huggingface.co/karpathy/tinyllamas/resolve/main/");

                Net<T> net = mycaffe.GetInternalNet(Phase.TRAIN);
                builder.LoadWeights(net.learnable_parameters, strModelPath, "KPTH0");

                TokenizedDataLayer<T> tok = net.FindLayer(LayerParameter.LayerType.TOKENIZED_DATA, "data") as TokenizedDataLayer<T>;

                PropertySet input = new PropertySet();
                string strPrompt = "Once upon a time, ";
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
                List<float> rgResponseTokens = new List<float>();

                rgTokenIds.AddRange(convertF(blobTokdata.update_cpu_data()));

                sw.Start();
                double dfTotalTime = 0;
                int nTokenId = (int)rgTokenIds[0];

                int[] rgShape = new int[2] { 1, 1 };
                blobTokdata.Reshape(rgShape);

                for (int i = 0; i < nMaxNewTokens; i++)
                {
                    blobTokdata.SetData(nTokenId, 0);

                    net.SetLayerOption("position", i);
                    net.ForwardFromTo(3, 11);

                    if (i < rgTokenIds.Count - 1)
                    {
                        nTokenId = (int)rgTokenIds[i + 1];
                    }
                    else
                    {
                        blobLogits.scale_data(1.0f / fTemperature);

                        List<Tuple<string, int, double>> res = tok.PostProcessLogitsOutput(nCurIdx, blobLogits, null, 2, 10);
                        nTokenId = res[0].Item2;

                        if (!tok.IsEOS(nTokenId))
                            rgResponseTokens.Add(nTokenId);
                    }

                    sw.Stop();
                    dfTotalTime += sw.Elapsed.TotalMilliseconds;

                    m_log.WriteLine("Processing prompt #" + i.ToString() + " average time " + (dfTotalTime / (i + 1)).ToString("N3") + " ms.", true);

                    sw.Restart();

                    if (tok.IsEOS(nTokenId))
                        break;
                }

                string strOutput = tok.Detokenize(rgResponseTokens.ToArray(), 0, rgResponseTokens.Count);
                m_log.WriteLine("Output: " + strOutput);
            }
            finally
            {
                mycaffe.Dispose();
            }
        }

        protected override void testCreateTrainingModel(bool bFineTune)
        {
            m_strModel = "Stories15M_Instruct";
            m_nBatchSize = 64;
            m_nSeqLen = 256;
            m_nIterSize = (bFineTune) ? 8 : 1;
            ModelBuilder<T> builder = create();

            PropertySet prop = new PropertySet();
            prop.SetProperty("VocabularyType", ((int)TokenizedDataParameter.VOCABULARY_TYPE.LLAMA2).ToString());
            NetParameter net_param = builder.CreateModel(prop, Phase.TRAIN, bFineTune, 1);
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

                mycaffe.LoadLite(Phase.TRAIN, strSolver, strNet, null, null, false, false);
                mycaffe.OnTrainingIteration += MyCaffe_OnTrainingIteration;
                mycaffe.OnSnapshot += MyCaffe_OnSnapshot;

                string strModelPath = getTestDataLlamaPath("stories", "stories15M.bin", "https://huggingface.co/karpathy/tinyllamas/resolve/main/");

                Net<T> net = mycaffe.GetInternalNet(Phase.TRAIN);
                builder.LoadWeights(net.learnable_parameters, strModelPath, "KPTH0");

                // Train the model with fine-tuning.
                int nIter = 50;
#if DEBUG
                nIter = 5000;
#endif
                mycaffe.Train(nIter); // 5000 for full training.

                // Run inferencing test
                generate(net); 
            }
            finally
            {
                mycaffe.Dispose();
            }
        }

        private void MyCaffe_OnSnapshot(object sender, SnapshotArgs e)
        {
        }

        private int countWords(string str, string strWord)
        {
            int nCount = 0;
            int nIdx = 0;

            while (nIdx >= 0)
            {
                nIdx = str.IndexOf(strWord, nIdx + 1);
                if (nIdx >= 0)
                    nCount++;
            }

            return nCount;
        }

        private void generate(Net<T> net)
        {
            PreTokenizedDataLayer<T> tok = net.FindLayer(LayerParameter.LayerType.PRETOKENIZED_DATA, "data") as PreTokenizedDataLayer<T>;

            PropertySet input = new PropertySet();
            string strPrompt = "Write a story.  In the story, try to use the verb 'eat', the noun 'cat' and the adjective 'sad'.";
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
            List<float> rgResponseTokens = new List<float>();

            rgTokenIds.AddRange(convertF(blobTokdata.update_cpu_data()));

            sw.Start();
            double dfTotalTime = 0;
            int nTokenId = (int)rgTokenIds[0];

            int[] rgShape = new int[2] { 1, 1 };
            blobTokdata.Reshape(rgShape);

            for (int i = 0; i < nMaxNewTokens; i++)
            {
                blobTokdata.SetData(nTokenId, 0);

                net.SetLayerOption("position", i);
                net.ForwardFromTo(2, 11);

                if (i < rgTokenIds.Count - 1)
                {
                    nTokenId = (int)rgTokenIds[i + 1];
                }
                else
                {
                    blobLogits.scale_data(1.0f / fTemperature);

                    List<Tuple<string, int, double>> res = tok.PostProcessLogitsOutput(nCurIdx, blobLogits, null, 2, 10);
                    nTokenId = res[0].Item2;

                    if (!tok.IsEOS(nTokenId))
                        rgResponseTokens.Add(nTokenId);
                }

                sw.Stop();
                dfTotalTime += sw.Elapsed.TotalMilliseconds;

                m_log.WriteLine("Processing prompt #" + i.ToString() + " average time " + (dfTotalTime / (i + 1)).ToString("N3") + " ms.", true);

                sw.Restart();

                if (tok.IsEOS(nTokenId))
                    break;
            }

            string strOutput = tok.Detokenize(rgResponseTokens.ToArray(), 0, rgResponseTokens.Count);

            int nCountEat = countWords(strOutput, "eat");
            int nCountCat = countWords(strOutput, "cat");
            int nCountSad = countWords(strOutput, "sad");

            m_log.WriteLine("Output: " + strOutput);
            m_log.WriteLine("Word Counts: eat = " + nCountEat.ToString() + ", cat = " + nCountCat.ToString() + ", sad = " + nCountSad.ToString());
        }

        private void MyCaffe_OnTrainingIteration(object sender, TrainingIterationArgs<T> e)
        {
            m_log.WriteLine("Iteration " + e.Iteration.ToString() + " Loss = " + e.SmoothedLoss.ToString("N6"));

            if (e.Iteration % 100 == 0)
            {
                Solver<T> solver = sender as Solver<T>;
                Net<T> net = solver.net;
                generate(net);
            }
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

        protected string getTestDataLlamaPath(string strSubPath, string strFile, string strUrl = null)
        {
            string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\llama\\test\\" + strSubPath;

            if (!File.Exists(strPath + "\\" + strFile))
            {
                if (string.IsNullOrEmpty(strUrl))
                    throw new Exception("Could not find the test data file '" + strPath + strFile + "'.  You may need to run the 'Test|Download Test Data | Llama' menu item.");

                return downloadTestData(strUrl, strPath, strFile);
            }

            return strPath + "\\" + strFile;
        }

        public void TestCreateTrainingModel(bool bFineTune)
        {
            testCreateTrainingModel(bFineTune);
        }

        protected virtual void testCreateTrainingModel(bool bFineTune)
        {
            ModelBuilder<T> builder = create();

            NetParameter net_param = builder.CreateModel(null);
            RawProto proto = net_param.ToProto("root");
            string strNet = proto.ToString();

            RawProto proto2 = RawProto.Parse(strNet);
            NetParameter net_param2 = NetParameter.FromProto(proto2);

            m_log.CHECK(net_param2.Compare(net_param), "The two net parameters should be the same!");

            // verify creating the model.
            SolverParameter solver = builder.CreateSolver();
            RawProto protoSolver = solver.ToProto("root");
            string strSolver = protoSolver.ToString();

            SettingsCaffe settings = new SettingsCaffe();
            CancelEvent evtCancel = new CancelEvent();
            MyCaffeControl<T> mycaffe = new MyCaffeControl<T>(settings, m_log, evtCancel);

            save(strNet, strSolver, false);

            // mycaffe.LoadLite(Phase.TRAIN, strSolver, strNet, null);
            mycaffe.Dispose();
        }

        public void TestCreateInferenceModel(bool bEnableLoRA)
        {
            testCreateInferenceModel(bEnableLoRA);
        }

        protected virtual void testCreateInferenceModel(bool bEnableLoRA)
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

            mycaffe.LoadToRun(strNet, null, null, new BlobShape(1, 3, 300, 300));
            mycaffe.Dispose();
        }
    }
}
