using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.param;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.layers;
using MyCaffe.solvers;

namespace MyCaffe.test
{
    [TestClass]
    public class TestWeightAdapter
    {
        [TestMethod]
        public void TestLayerForwardLoRADisabled()
        {
            OutputAdapterTest test = new OutputAdapterTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IWeightAdapterTest t in test.Tests)
                {
                    t.TestLayerForward(false, "lora", SolverParameter.SolverType.ADAM);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestLayerGradientLoRADisabled()
        {
            OutputAdapterTest test = new OutputAdapterTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IWeightAdapterTest t in test.Tests)
                {
                    t.TestLayerGradient(false, "lora", SolverParameter.SolverType.ADAM);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestLayerForwardLoRAEnabled()
        {
            OutputAdapterTest test = new OutputAdapterTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IWeightAdapterTest t in test.Tests)
                {
                    t.TestLayerForward(true, "lora", SolverParameter.SolverType.ADAM);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestLayerGradientLoRAEnabled_ADAM()
        {
            OutputAdapterTest test = new OutputAdapterTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IWeightAdapterTest t in test.Tests)
                {
                    t.TestLayerGradient(true, "lora", SolverParameter.SolverType.ADAM);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestLayerForwardLoRAEnabledAtSize()
        {
            OutputAdapterTest test = new OutputAdapterTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IWeightAdapterTest t in test.Tests)
                {
                    t.TestLayerForward("lora", SolverParameter.SolverType.ADAM, 64, 128, 192, 1);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestLayerForwardLoRAEnabledAtSizeGradient()
        {
            OutputAdapterTest test = new OutputAdapterTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IWeightAdapterTest t in test.Tests)
                {
                    t.TestLayerGradient("lora", SolverParameter.SolverType.ADAM, 64, 128, 192, 1);
                }
            }
            finally
            {
                test.Dispose();
            }
        }


        [TestMethod]
        public void TestLayerGradientLoRAEnabled_ADAMW()
        {
            OutputAdapterTest test = new OutputAdapterTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IWeightAdapterTest t in test.Tests)
                {
                    t.TestLayerGradient(true, "lora", SolverParameter.SolverType.ADAMW);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestLayerGradientLoRAEnabled_SGD()
        {
            OutputAdapterTest test = new OutputAdapterTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IWeightAdapterTest t in test.Tests)
                {
                    t.TestLayerGradient(true, "lora", SolverParameter.SolverType.SGD);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestLayerGradientLoRAEnabled_RMSPROP()
        {
            OutputAdapterTest test = new OutputAdapterTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IWeightAdapterTest t in test.Tests)
                {
                    t.TestLayerGradient(true, "lora", SolverParameter.SolverType.RMSPROP);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestLayerGradientLoRAEnabled_ADAGRAD()
        {
            OutputAdapterTest test = new OutputAdapterTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IWeightAdapterTest t in test.Tests)
                {
                    t.TestLayerGradient(true, "lora", SolverParameter.SolverType.ADAGRAD);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestLayerGradientLoRAEnabled_NESTEROV()
        {
            OutputAdapterTest test = new OutputAdapterTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IWeightAdapterTest t in test.Tests)
                {
                    t.TestLayerGradient(true, "lora", SolverParameter.SolverType.NESTEROV);
                }
            }
            finally
            {
                test.Dispose();
            }
        }


        [TestMethod]
        public void TestLayerGradientLoRAEnabled_ADADELTA()
        {
            OutputAdapterTest test = new OutputAdapterTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IWeightAdapterTest t in test.Tests)
                {
                    t.TestLayerGradient(true, "lora", SolverParameter.SolverType.ADADELTA);
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface IWeightAdapterTest : ITest
    {
        void TestLayerForward(bool bEnabled, string strType, SolverParameter.SolverType type);
        void TestLayerGradient(bool bEnabled, string strType, SolverParameter.SolverType type);
        void TestLayerForward(string strType, SolverParameter.SolverType type, int nN, int nC, int nH, int nW);
        void TestLayerGradient(string strType, SolverParameter.SolverType type, int nN, int nC, int nH, int nW);
    }

    class OutputAdapterTest : TestBase
    {
        public OutputAdapterTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("OutputAdapter Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new WeightAdapterTest<double>(strName, nDeviceID, engine);
            else
                return new WeightAdapterTest<float>(strName, nDeviceID, engine);
        }
    }

    class WeightAdapterTest<T> : TestEx<T>, IWeightAdapterTest
    {
        Blob<T> m_blobTarget;

        public WeightAdapterTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 1, 1 }, nDeviceID)
        {
            m_engine = engine;
            m_blobTarget = new Blob<T>(m_cuda, m_log);
        }

        protected override void dispose()
        {
            base.dispose();
        }

        private string getDataPath()
        {
            string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\llama\\test\\instr_llama\\";
            return strPath;
        }

        private string buildSolver(SolverParameter.SolverType type)
        {
            SolverParameter solverParam = new SolverParameter();
            solverParam.base_lr = 0.01;
            solverParam.type = type;
            solverParam.test_iter[0] = 1;
            solverParam.test_interval = 1000;
            solverParam.test_initialization = false;

            if (type == SolverParameter.SolverType.ADADELTA)
                solverParam.momentum = 0.9;

            return solverParam.ToProto("root").ToString();
        }

        private string buildModel(bool bEnableLoRA, string strLoRAType, int nN, int nC, int nH, int nW, bool bFwdAndBwd, int nAxis = 2, int nNumOut = 1)
        {
            NetParameter pNet = new NetParameter();
            pNet.name = "lora_test";
            pNet.enable_lora = bEnableLoRA;

            LayerParameter data = new LayerParameter(LayerParameter.LayerType.INPUT);
            data.input_param.shape.Add(new BlobShape(new List<int>() { nN, nC, nH, nW }));
            data.top.Add("data");

            if (bFwdAndBwd)
            {
                data.input_param.shape.Add(new BlobShape(new List<int>() { nN, nC, nH, nW }));
                data.top.Add("target");
            }
            pNet.layer.Add(data);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT, "ip");
            p.inner_product_param.num_output = (uint)nNumOut;
            p.inner_product_param.bias_term = false;
            p.inner_product_param.axis = nAxis;
            p.freeze_learning = bEnableLoRA;
            p.weight_adapter.type = strLoRAType;
            p.weight_adapter.alpha = 1.0;
            p.weight_adapter.rank = 2;
            p.weight_adapter.enabled = bEnableLoRA;
            p.bottom.Add("data");
            p.top.Add("ip");
            pNet.layer.Add(p);

            if (bFwdAndBwd)
            {
                LayerParameter loss = new LayerParameter(LayerParameter.LayerType.MEAN_ERROR_LOSS, "loss");
                loss.loss_weight.Add(1);
                loss.loss_param.normalization = LossParameter.NormalizationMode.NONE;
                loss.mean_error_loss_param.axis = 2;
                loss.mean_error_loss_param.mean_error_type = MEAN_ERROR.MSE;
                loss.bottom.Add("ip");
                loss.bottom.Add("target");
                loss.top.Add("loss");
                loss.include.Add(new NetStateRule(Phase.TRAIN));
                pNet.layer.Add(loss);
            }

            return pNet.ToProto("root").ToString();
        }

        /// <summary>
        /// Test layer forward.
        /// </summary>
        /// <remarks>
        /// Test data generated using the following python code:
        ///     instruct_finetune.py with debug=True (data generated in model.py)
        /// </remarks>
        /// <param name="bEnableLoRA">Specifies to enable LoRA</param>
        /// <param name="strType">Specifies the weight type, should be "KPTH0".</param>
        /// <param name="type">Specifies the solver type.</param>
        public void TestLayerForward(bool bEnableLoRA, string strType, SolverParameter.SolverType type)
        {
            CancelEvent evtCancel = new CancelEvent();
            SettingsCaffe s = new SettingsCaffe();
            s.GpuIds = "0";
            MyCaffeControl<T> mycaffe = new MyCaffeControl<T>(s, m_log, evtCancel);
            Blob<T> blobVal = null;
            Blob<T> blobWork = null;
            string strPath = getDataPath();

            try
            {
                string strSolver = buildSolver(type);
                string strModel = buildModel(bEnableLoRA, strType, 64, 350, 288, 1, false, 2, 288);

                mycaffe.LoadLite(Phase.TRAIN, strSolver, strModel, null, null, false, false);

                blobVal = mycaffe.CreateBlob("val");
                blobWork = mycaffe.CreateBlob("work");

                Net<T> net = mycaffe.GetInternalNet(Phase.TRAIN);
                Blob<T> blobX = net.FindBlob("data");

                blobX.LoadFromNumpy(strPath + "x.npy");

                blobX.Unsqueeze(2);

                m_log.CHECK_EQ(blobX.num, 64, "The batch size should be 64.");
                m_log.CHECK_EQ(blobX.channels, 350, "The channels should be 350.");
                m_log.CHECK_EQ(blobX.height, 1, "The height should be 288.");
                m_log.CHECK_EQ(blobX.width, 288, "The width should be 1.");

                Layer<T> layer = net.FindLayer(LayerParameter.LayerType.INNERPRODUCT, "ip");

                layer.blobs[0].LoadFromNumpy(strPath + "wq.npy");

                if (bEnableLoRA)
                {
                    m_log.CHECK_EQ(layer.blobs_adapted.Count, 2, "The number of adapted blobs should be 2.");
                    m_log.CHECK_EQ(layer.blobs_adapted[0].num, layer.layer_param.weight_adapter.rank, "The num of the A adapted blob should be equal to rank " + layer.layer_param.weight_adapter.rank.ToString());
                    m_log.CHECK_EQ(layer.blobs_adapted[0].channels, 288, "The channels of the adapted blob should be 288.");
                    m_log.CHECK_EQ(layer.blobs_adapted[1].channels, layer.layer_param.weight_adapter.rank, "The channels of the B adapted blob should be equal to rank " + layer.layer_param.weight_adapter.rank.ToString());
                    m_log.CHECK_EQ(layer.blobs_adapted[1].num, 288, "The num of the B adapted blob should be 288.");

                    layer.blobs_adapted[0].LoadFromNumpy(strPath + "lora_a.npy");
                    layer.blobs_adapted[1].LoadFromNumpy(strPath + "lora_b.npy");
                }

                BlobCollection<T> colTop = net.Forward();

                blobVal.LoadFromNumpy(strPath + "xq.npy");
                m_log.CHECK(blobVal.Compare(colTop[0], blobWork), "The outputs are not as expected.");
            }
            finally
            {
                dispose(ref blobVal);
                dispose(ref blobWork);
                mycaffe.Dispose();
            }
        }

        public void TestLayerGradient(bool bEnabled, string strType, SolverParameter.SolverType type)
        {
            CancelEvent evtCancel = new CancelEvent();
            SettingsCaffe s = new SettingsCaffe();
            s.GpuIds = "0";
            MyCaffeControl<T> mycaffe = new MyCaffeControl<T>(s, m_log, evtCancel);

            try
            {
                string strSolver = buildSolver(type);
                string strModel = buildModel(bEnabled, strType, 2, 3, 1, 1, true);

                mycaffe.LoadLite(Phase.TRAIN, strSolver, strModel);

                Solver<T> solver = mycaffe.GetInternalSolver();
                Net<T> net = mycaffe.GetInternalNet(Phase.TRAIN);
                Net<T> netTest = mycaffe.GetInternalNet(Phase.TEST);
                Blob<T> blobX = net.FindBlob("data");
                Blob<T> blobTrg = net.FindBlob("target");

                float[] rgTarget = new float[] { 2, 4, 6, 8, 10, 12 };
                blobTrg.mutable_cpu_data = convert(rgTarget);

                float[] rgBottom = new float[] { 1, 2, 3, 4, 5, 6 };
                blobX.mutable_cpu_data = convert(rgBottom);

                Layer<T> layer = net.FindLayer(LayerParameter.LayerType.INNERPRODUCT, "ip");
                float[] rgWeight = new float[] { 0.0461305f };
                float[] rgBias = new float[] { 0.0f };
                layer.blobs[0].mutable_cpu_data = convert(rgWeight);
                layer.blobs[1].mutable_cpu_data = convert(rgBias);

                if (bEnabled)
                {
                    float[] rgLoraA = new float[] { 0.2012014f, -0.5057645f, 0.1083691f, -0.3061365f };
                    float[] rgLoraB = new float[] { 0.0f, 0.0f, 0.0f, 0.0f };
                    layer.blobs_adapted[0].mutable_cpu_data = convert(rgLoraA);
                    layer.blobs_adapted[1].mutable_cpu_data = convert(rgLoraB);
                }

                float[] rgLastTop = null;

                for (int i = 0; i < 1000; i++)
                {
                    net.Forward();
                    net.Backward();
                    solver.ApplyUpdate(i);

                    if (bEnabled)
                    {
                        // Verify that the bias and weight have not changed.
                        float[] rgActualWeight = convertF(layer.blobs[0].mutable_cpu_data);
                        float[] rgActualBias = convertF(layer.blobs[1].mutable_cpu_data);

                        for (int j = 0; j < rgWeight.Length; j++)
                        {
                            m_log.EXPECT_NEAR(rgWeight[j], rgActualWeight[j], 1e-12);
                        }
                        for (int j = 0; j < rgBias.Length; j++)
                        {
                            m_log.EXPECT_NEAR(rgBias[j], rgActualBias[j], 1e-12);
                        }
                    }
                    else
                    {
                        // Verify that the bias and weight have changed.
                        float[] rgActualWeight = convertF(layer.blobs[0].mutable_cpu_data);
                        float[] rgActualBias = convertF(layer.blobs[1].mutable_cpu_data);

                        for (int j = 0; j < rgWeight.Length; j++)
                        {
                            float fDiff = Math.Abs(rgWeight[j] - rgActualWeight[j]);
                            m_log.CHECK_GT(fDiff, 1e-05, "The weight difference is too small.");
                            rgWeight[j] = rgActualWeight[j];
                        }
                        for (int j = 0; j < rgBias.Length; j++)
                        {
                            float fDiff = Math.Abs(rgBias[j] - rgActualBias[j]);
                            m_log.CHECK_GT(fDiff, 1e-05, "The bias difference is too small.");
                            rgBias[j] = rgActualBias[j];
                        }
                    }

                    BlobCollection<T> colTop = netTest.Forward();
                    Blob<T> pred = colTop.FindBlob("ip");
                    float[] rgTop = convertF(pred.mutable_cpu_data);

                    if (rgLastTop != null)
                    {
                        if (i < 100)
                        {
                            for (int j = 0; j < rgTop.Length; j++)
                            {
                                double dfDiff = Math.Abs(rgTop[j] - rgLastTop[j]);
                                m_log.CHECK_GT(dfDiff, 1e-06, "The top difference is too small.");
                                rgLastTop[j] = rgTop[j];
                            }
                        }
                        else
                        {
                            double dfSum = 0;
                            for (int j = 0; j < rgTop.Length; j++)
                            {
                                double dfDiff = Math.Abs(rgTop[j] - rgLastTop[j]);
                                dfSum += dfDiff;
                            }

                            double dfAve = dfSum / rgTop.Length;
                            if (dfAve < 1e-06)
                                break;
                        }
                    }

                    rgLastTop = Utility.Clone<float>(rgTop);
                }

                for (int j = 0; j < rgLastTop.Length; j++)
                {
                    double dfDiff = Math.Abs(rgTarget[j] - rgLastTop[j]);
                    m_log.CHECK_LT(dfDiff, 1e-03, "The top difference is too small.");
                }
            }
            finally
            {
                mycaffe.Dispose();
            }
        }

        public void TestLayerForward(string strType, SolverParameter.SolverType type, int nN, int nC, int nH, int nW)
        {
            CancelEvent evtCancel = new CancelEvent();
            SettingsCaffe s = new SettingsCaffe();
            s.GpuIds = "0";
            MyCaffeControl<T> mycaffe = new MyCaffeControl<T>(s, m_log, evtCancel);

            try
            {
                string strSolver = buildSolver(type);
                string strModel = buildModel(true, strType, nN, nC, nH, nW, false, 2, nH);

                mycaffe.LoadLite(Phase.TRAIN, strSolver, strModel);

                Net<T> net = mycaffe.GetInternalNet(Phase.TRAIN);
                Blob<T> blobX = net.FindBlob("data");
                m_filler.Fill(blobX);

                BlobCollection<T> colTop = net.Forward();
                float[] rgTop = convertF(colTop[0].mutable_cpu_data);

                for (int i = 0; i < rgTop.Length; i++)
                {
                    m_log.CHECK(!float.IsNaN(rgTop[i]) && !float.IsInfinity(rgTop[i]), "The top value at " + i.ToString() + " is NaN!");
                }
            }
            finally
            {
                mycaffe.Dispose();
            }
        }

        public void TestLayerGradient(string strType, SolverParameter.SolverType type, int nN, int nC, int nH, int nW)
        {
            CancelEvent evtCancel = new CancelEvent();
            SettingsCaffe s = new SettingsCaffe();
            s.GpuIds = "0";
            MyCaffeControl<T> mycaffe = new MyCaffeControl<T>(s, m_log, evtCancel);

            try
            {
                string strSolver = buildSolver(type);
                string strModel = buildModel(true, strType, nN, nC, nH, nW, true, 2, nH);

                mycaffe.LoadLite(Phase.TRAIN, strSolver, strModel);

                Solver<T> solver = mycaffe.GetInternalSolver();
                Net<T> net = mycaffe.GetInternalNet(Phase.TRAIN);
                Blob<T> blobX = net.FindBlob("data");
                m_filler.Fill(blobX);

                BlobCollection<T> colTop = net.Forward();
                float[] rgTop = convertF(colTop[0].mutable_cpu_data);

                for (int i = 0; i < rgTop.Length; i++)
                {
                    m_log.CHECK(!float.IsNaN(rgTop[i]) && !float.IsInfinity(rgTop[i]), "The top value at " + i.ToString() + " is NaN!");
                }

                Blob<T> blobLoss = net.FindBlob("loss");
                blobLoss.SetDiff(1.0);

                net.Backward();
                solver.ApplyUpdate(1);

                colTop = net.Forward();
                rgTop = convertF(colTop[0].mutable_cpu_data);

                for (int i = 0; i < rgTop.Length; i++)
                {
                    m_log.CHECK(!float.IsNaN(rgTop[i]) && !float.IsInfinity(rgTop[i]), "The top value at " + i.ToString() + " is NaN!");
                }
            }
            finally
            {
                mycaffe.Dispose();
            }
        }
    }
}
