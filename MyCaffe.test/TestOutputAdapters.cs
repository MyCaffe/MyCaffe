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
    public class TestOutputAdapter
    {
        [TestMethod]
        public void TestLayerForwardLoRADisabled()
        {
            OutputAdapterTest test = new OutputAdapterTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IOutputAdapterTest t in test.Tests)
                {
                    t.TestLayerForward(false, "lora");
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
                foreach (IOutputAdapterTest t in test.Tests)
                {
                    t.TestLayerGradient(false, "lora");
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
                foreach (IOutputAdapterTest t in test.Tests)
                {
                    t.TestLayerForward(true, "lora");
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestLayerGradientLoRAEnabled()
        {
            OutputAdapterTest test = new OutputAdapterTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IOutputAdapterTest t in test.Tests)
                {
                    t.TestLayerGradient(true, "lora");
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface IOutputAdapterTest : ITest
    {
        void TestLayerForward(bool bEnabled, string strType);
        void TestLayerGradient(bool bEnabled, string strType);
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
                return new OutputAdapterTest<double>(strName, nDeviceID, engine);
            else
                return new OutputAdapterTest<float>(strName, nDeviceID, engine);
        }
    }

    class OutputAdapterTest<T> : TestEx<T>, IOutputAdapterTest
    {
        Blob<T> m_blobTarget;

        public OutputAdapterTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 1, 1 }, nDeviceID)
        {
            m_engine = engine;
            m_blobTarget = new Blob<T>(m_cuda, m_log);
        }

        protected override void dispose()
        {
            base.dispose();
        }

        private string buildSolver()
        {
            SolverParameter solverParam = new SolverParameter();
            solverParam.base_lr = 0.01;
            solverParam.type = SolverParameter.SolverType.ADAM;
            solverParam.test_iter.Clear();
            solverParam.test_interval = 1000;
            solverParam.test_initialization = false;

            return solverParam.ToProto("root").ToString();
        }

        private string buildModel(bool bEnableLoRA, string strLoRAType, int nN, int nC, int nH, int nW, bool bFwdAndBwd)
        {
            NetParameter pNet = new NetParameter();
            pNet.name = "lora_test";

            LayerParameter data = new LayerParameter(LayerParameter.LayerType.INPUT);
            data.input_param.shape.Add(new BlobShape(new List<int>() { nN, nC, nH, nW }));
            data.top.Add("x");

            if (bFwdAndBwd)
            {
                data.input_param.shape.Add(new BlobShape(new List<int>() { nN, nC, nH, nW }));
                data.top.Add("target");
            }
            pNet.layer.Add(data);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT, "ip");
            p.inner_product_param.num_output = 1;
            p.inner_product_param.bias_term = true;
            p.inner_product_param.axis = 2;
            p.freeze_learning = bEnableLoRA;
            p.output_adapter.type = strLoRAType;
            p.output_adapter.alpha = 1.0;
            p.output_adapter.rank = 4;
            p.output_adapter.enabled = bEnableLoRA;
            p.bottom.Add("x");
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
                pNet.layer.Add(loss);
            }

            return pNet.ToProto("root").ToString();
        }

        public void TestLayerForward(bool bEnabled, string strType)
        {
            CancelEvent evtCancel = new CancelEvent();
            SettingsCaffe s = new SettingsCaffe();
            s.GpuIds = "0";
            MyCaffeControl<T> mycaffe = new MyCaffeControl<T>(s, m_log, evtCancel);

            try
            {
                string strSolver = buildSolver();
                string strModel = buildModel(bEnabled, strType, 2, 3, 1, 1, false);

                mycaffe.LoadLite(Phase.TRAIN, strSolver, strModel);

                Net<T> net = mycaffe.GetInternalNet(Phase.TRAIN);
                Blob<T> blobX = net.FindBlob("x");

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

                BlobCollection<T> colTop = net.Forward();
                float[] rgTop = convertF(colTop[0].mutable_cpu_data);

                float[] rgExpected = new float[] { 0.0461305f, 0.0922609f, 0.1383914f, 0.1845219f, 0.2306523f, 0.2767828f };
                for (int i = 0; i < rgTop.Length; i++)
                {
                    m_log.EXPECT_NEAR(rgTop[i], rgExpected[i], 3e-07);
                }
            }
            finally
            {
                mycaffe.Dispose();
            }
        }

        // WORK IN PROGRESS
        public void TestLayerGradient(bool bEnabled, string strType)
        {
            CancelEvent evtCancel = new CancelEvent();
            SettingsCaffe s = new SettingsCaffe();
            s.GpuIds = "0";
            MyCaffeControl<T> mycaffe = new MyCaffeControl<T>(s, m_log, evtCancel);

            try
            {
                string strSolver = buildSolver();
                string strModel = buildModel(bEnabled, strType, 2, 3, 1, 1, true);

                mycaffe.LoadLite(Phase.TRAIN, strSolver, strModel);

                Solver<T> solver = mycaffe.GetInternalSolver();
                Net<T> net = mycaffe.GetInternalNet(Phase.TRAIN);
                Blob<T> blobX = net.FindBlob("x");
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

                for (int i = 0; i < 10; i++)
                {
                    BlobCollection<T> colTop = net.Forward();
                    float[] rgTop = convertF(colTop[0].mutable_cpu_data);

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

                    if (rgLastTop != null)
                    {
                        for (int j=0; j<rgTop.Length; j++)
                        {
                            double dfDiff = Math.Abs(rgTop[j] - rgLastTop[j]);
                            m_log.CHECK_GT(dfDiff, 1e-05, "The top difference is too small.");
                            rgLastTop[j] = rgTop[j];
                        }
                    }

                    rgLastTop = Utility.Clone<float>(rgTop);
                }
            }
            finally
            {
                mycaffe.Dispose();
            }
        }
    }
}
