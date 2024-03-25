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
using MyCaffe.fused_ops;
using MyCaffe.fillers;
using System.Diagnostics;

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
                    t.TestLayerForward(false, false, "lora", SolverParameter.SolverType.SGD);
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
                    t.TestLayerGradient(false, false, "lora", SolverParameter.SolverType.SGD);
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
                    t.TestLayerForward(true, false, "lora", SolverParameter.SolverType.SGD);
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
                foreach (IWeightAdapterTest t in test.Tests)
                {
                    t.TestLayerGradient(true, false, "lora", SolverParameter.SolverType.SGD);
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
                    t.TestLayerForward(false, "lora", SolverParameter.SolverType.SGD, 64, 128, 192, 1);
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
                    t.TestLayerGradient(false, "lora", SolverParameter.SolverType.SGD, 64, 128, 192, 1);
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
                    t.TestLayerGradient(true, false, "lora", SolverParameter.SolverType.ADAMW);
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
                    t.TestLayerGradient(true, false, "lora", SolverParameter.SolverType.ADAM);
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
                    t.TestLayerGradient(true, false, "lora", SolverParameter.SolverType.RMSPROP);
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
                    t.TestLayerGradient(true, false, "lora", SolverParameter.SolverType.ADAGRAD);
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
                    t.TestLayerGradient(true, false, "lora", SolverParameter.SolverType.NESTEROV);
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
                    t.TestLayerGradient(true, false, "lora", SolverParameter.SolverType.ADADELTA);
                }
            }
            finally
            {
                test.Dispose();
            }
        }


        [TestMethod]
        public void TestLayerForwardLoRADisabled_linear()
        {
            OutputAdapterTest test = new OutputAdapterTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IWeightAdapterTest t in test.Tests)
                {
                    t.TestLayerForward(false, true, "lora", SolverParameter.SolverType.SGD);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestLayerGradientLoRADisabled_linear()
        {
            OutputAdapterTest test = new OutputAdapterTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IWeightAdapterTest t in test.Tests)
                {
                    t.TestLayerGradient(false, true, "lora", SolverParameter.SolverType.SGD);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestLayerForwardLoRAEnabled_linear()
        {
            OutputAdapterTest test = new OutputAdapterTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IWeightAdapterTest t in test.Tests)
                {
                    t.TestLayerForward(true, true, "lora", SolverParameter.SolverType.SGD);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestLayerGradientLoRAEnabled_linear()
        {
            OutputAdapterTest test = new OutputAdapterTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IWeightAdapterTest t in test.Tests)
                {
                    t.TestLayerGradient(true, true, "lora", SolverParameter.SolverType.SGD);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestLayerForwardLoRAEnabledAtSize_linear()
        {
            OutputAdapterTest test = new OutputAdapterTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IWeightAdapterTest t in test.Tests)
                {
                    t.TestLayerForward(true, "lora", SolverParameter.SolverType.SGD, 64, 128, 192, 1);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestLayerForwardLoRAEnabledAtSizeGradient_linear()
        {
            OutputAdapterTest test = new OutputAdapterTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IWeightAdapterTest t in test.Tests)
                {
                    t.TestLayerGradient(true, "lora", SolverParameter.SolverType.SGD, 64, 128, 192, 1);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestLayerGradientLoRAEnabled_ADAMW_linear()
        {
            OutputAdapterTest test = new OutputAdapterTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IWeightAdapterTest t in test.Tests)
                {
                    t.TestLayerGradient(true, true, "lora", SolverParameter.SolverType.ADAMW);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestLayerGradientLoRAEnabled_ADAM_linear()
        {
            OutputAdapterTest test = new OutputAdapterTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IWeightAdapterTest t in test.Tests)
                {
                    t.TestLayerGradient(true, true, "lora", SolverParameter.SolverType.ADAM);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestLayerGradientLoRAEnabled_RMSPROP_linear()
        {
            OutputAdapterTest test = new OutputAdapterTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IWeightAdapterTest t in test.Tests)
                {
                    t.TestLayerGradient(true, true, "lora", SolverParameter.SolverType.RMSPROP);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestLayerGradientLoRAEnabled_ADAGRAD_linear()
        {
            OutputAdapterTest test = new OutputAdapterTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IWeightAdapterTest t in test.Tests)
                {
                    t.TestLayerGradient(true, true, "lora", SolverParameter.SolverType.ADAGRAD);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestLayerGradientLoRAEnabled_NESTEROV_linear()
        {
            OutputAdapterTest test = new OutputAdapterTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IWeightAdapterTest t in test.Tests)
                {
                    t.TestLayerGradient(true, true, "lora", SolverParameter.SolverType.NESTEROV);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestLayerGradientLoRAEnabled_ADADELTA_linear()
        {
            OutputAdapterTest test = new OutputAdapterTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IWeightAdapterTest t in test.Tests)
                {
                    t.TestLayerGradient(true, true, "lora", SolverParameter.SolverType.ADADELTA);
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
        void TestLayerForward(bool bEnableLoRa, bool bUseLinear, string strType, SolverParameter.SolverType type);
        void TestLayerGradient(bool bEnableLoRa, bool bUseLinear, string strType, SolverParameter.SolverType type);
        void TestLayerForward(bool bUseLinear, string strType, SolverParameter.SolverType type, int nN, int nC, int nH, int nW);
        void TestLayerGradient(bool bUseLinear, string strType, SolverParameter.SolverType type, int nN, int nC, int nH, int nW);
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
            //return Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\llama\\test\\instr_llama\\";
            return "C:\\temp\\projects\\llama2\\llama2\\llama2_instruct\\test\\";
        }

        private string buildSolver(SolverParameter.SolverType type)
        {
            SolverParameter solverParam = new SolverParameter();
            solverParam.base_lr = 0.01;
            solverParam.type = type;
            solverParam.test_iter[0] = 1;
            solverParam.test_interval = 1000;
            solverParam.test_initialization = false;
            solverParam.weight_decay = 0;
            solverParam.adamw_decay = 0;
            solverParam.momentum = 0.9;

            if (type == SolverParameter.SolverType.ADAM || type == SolverParameter.SolverType.ADAMW)
                solverParam.momentum2 = 0.999;

            if (type == SolverParameter.SolverType.ADAGRAD || type == SolverParameter.SolverType.RMSPROP)
                solverParam.momentum = 0.0;

            return solverParam.ToProto("root").ToString();
        }

        private string buildModel(bool bEnableLoRA, string strLoRAType, int nN, int nC, int nH, int nW, bool bFwdAndBwd, int nAxis = 2, int nNumOut = 1, bool bUseLinear = false)
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

            if (bUseLinear)
            {
                LayerParameter p = new LayerParameter(LayerParameter.LayerType.LINEAR, "ip");
                p.linear_param.num_output = (uint)nNumOut;
                p.linear_param.bias_term = false;
                p.linear_param.transpose = true;
                p.linear_param.axis = nAxis;
                p.freeze_learning = bEnableLoRA;
                p.weight_adapter.type = strLoRAType;
                p.weight_adapter.alpha = 1.0;
                p.weight_adapter.rank = 2;
                p.weight_adapter.enabled = bEnableLoRA;
                p.bottom.Add("data");
                p.top.Add("ip");
                pNet.layer.Add(p);
            }
            else
            {
                LayerParameter p = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT, "ip");
                p.inner_product_param.num_output = (uint)nNumOut;
                p.inner_product_param.bias_term = false;
                p.inner_product_param.transpose = false;
                p.inner_product_param.axis = nAxis;
                p.freeze_learning = bEnableLoRA;
                p.weight_adapter.type = strLoRAType;
                p.weight_adapter.alpha = 1.0;
                p.weight_adapter.rank = 2;
                p.weight_adapter.enabled = bEnableLoRA;
                p.bottom.Add("data");
                p.top.Add("ip");
                pNet.layer.Add(p);
            }

            if (bFwdAndBwd)
            {
                LayerParameter loss = new LayerParameter(LayerParameter.LayerType.MEAN_ERROR_LOSS, "loss");
                loss.loss_weight.Add(1);
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
        /// <param name="bUseLinear">Specifies to use the Linear layer instead of the InnerProduct.</param>
        /// <param name="type">Specifies the solver type.</param>
        public void TestLayerForward(bool bEnableLoRA, bool bUseLinear, string strType, SolverParameter.SolverType type)
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
                string strModel = buildModel(bEnableLoRA, strType, 64, 350, 1, 288, false, 3, 288, bUseLinear);

                mycaffe.LoadLite(Phase.TRAIN, strSolver, strModel, null, null, false, false);

                blobVal = mycaffe.CreateBlob("val");
                blobWork = mycaffe.CreateBlob("work");

                Net<T> net = mycaffe.GetInternalNet(Phase.TRAIN);
                Blob<T> blobX = net.FindBlob("data");

                blobX.LoadFromNumpy(strPath + "att.x.npy");
                blobX.Unsqueeze(2);

                m_log.CHECK_EQ(blobX.num, 64, "The batch size should be 64.");
                m_log.CHECK_EQ(blobX.channels, 350, "The channels should be 350.");
                m_log.CHECK_EQ(blobX.height, 1, "The height should be 1.");
                m_log.CHECK_EQ(blobX.width, 288, "The width should be 288.");

                LayerParameter.LayerType layerType = (bUseLinear) ? LayerParameter.LayerType.LINEAR : LayerParameter.LayerType.INNERPRODUCT;
                Layer<T> layer = net.FindLayer(layerType, "ip");

                blobVal.LoadFromNumpy(strPath + "wq.npy");
                layer.blobs[0].CopyFrom(blobVal);

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
                double dfErr = (typeof(T) == typeof(double)) ? 0.005 : (bUseLinear) ? 1e-10 : 0.005;    
                m_log.CHECK(blobVal.Compare(colTop[0], blobWork, false, dfErr), "The outputs are not as expected.");
            }
            finally
            {
                dispose(ref blobVal);
                dispose(ref blobWork);
                mycaffe.Dispose();
            }
        }

        public void TestLayerGradient(bool bEnableLoRA, bool bUseLinear, string strType, SolverParameter.SolverType type)
        {
            CancelEvent evtCancel = new CancelEvent();
            SettingsCaffe s = new SettingsCaffe();
            s.GpuIds = "0";
            MyCaffeControl<T> mycaffe = new MyCaffeControl<T>(s, m_log, evtCancel);
            Filler<T> filler = null;
            string strPath = getDataPath();

            try
            {
                string strSolver = buildSolver(type);
                string strModel = buildModel(bEnableLoRA, strType, 64, 350, 1, 288, true, 2, 288, bUseLinear);

                mycaffe.LoadLite(Phase.TRAIN, strSolver, strModel);

                Solver<T> solver = mycaffe.GetInternalSolver();
                Net<T> net = mycaffe.GetInternalNet(Phase.TRAIN);
                Net<T> netTest = mycaffe.GetInternalNet(Phase.TEST);
                Blob<T> blobX = net.FindBlob("data");
                Blob<T> blobTrg = net.FindBlob("target");

                // Make sure to use the filler on the same kernel as the net.
                filler = mycaffe.CreateFiller(m_filler.filler_param);
                filler.Fill(blobX);
                filler.Fill(blobTrg);

                LayerParameter.LayerType layerType = (bUseLinear) ? LayerParameter.LayerType.LINEAR : LayerParameter.LayerType.INNERPRODUCT;
                Layer<T> layer = net.FindLayer(layerType, "ip");
                layer.blobs[0].LoadFromNumpy(strPath + "wq.npy");

                if (bEnableLoRA)
                {
                    layer.blobs_adapted[0].LoadFromNumpy(strPath + "lora_a.npy");
                    layer.blobs_adapted[1].LoadFromNumpy(strPath + "lora_b.npy");
                }

                float[] rgTarget = convertF(blobTrg.update_cpu_data());

                for (int i = 0; i < 200; i++)
                {
                    double dfLoss;
                    net.Forward(out dfLoss);
                    Trace.WriteLine("Iter #" + i.ToString() + " Loss = " + dfLoss.ToString());  

                    float[] rgWeight = convertF(layer.blobs[0].mutable_cpu_data);

                    net.ClearParamDiffs();
                    net.Backward();
                    solver.ApplyUpdate(i);

                    if (bEnableLoRA)
                    {
                        // Verify that the weights have not changed.
                        float[] rgActualWeight = convertF(layer.blobs[0].mutable_cpu_data);

                        for (int j = 0; j < rgWeight.Length; j++)
                        {
                            m_log.EXPECT_NEAR(rgWeight[j], rgActualWeight[j], 1e-12);
                        }
                    }
                    else
                    {
                        // Verify that the weights have changed.
                        float[] rgActualWeight = convertF(layer.blobs[0].mutable_cpu_data);
                        int nDiffCount = 0;

                        for (int j = 0; j < rgWeight.Length; j++)
                        {
                            float fDiff = Math.Abs(rgWeight[j] - rgActualWeight[j]);
                            if (fDiff >= 1e-07)
                                nDiffCount++;

                            rgWeight[j] = rgActualWeight[j];
                        }

                        m_log.CHECK_GT(nDiffCount, 0, "The weights should have changed.");
                    }

                    Blob<T> pred1 = net.FindBlob("ip");
                    float[] rgTopTrain = convertF(pred1.mutable_cpu_data);

                    BlobCollection<T> colTop = netTest.Forward();
                    Blob<T> pred2 = colTop.FindBlob("ip");
                    float[] rgTopTest = convertF(pred2.mutable_cpu_data);

                    if (rgTopTest != null)
                    {
                        if (i < 100)
                        {
                            int nDiffCount = 0;
                            for (int j = 0; j < rgTopTest.Length; j++)
                            {
                                double dfDiff = Math.Abs(rgTopTest[j] - rgTopTrain[j]);
                                if (dfDiff >= 1e-07)
                                    nDiffCount++;
                                
                                rgTopTest[j] = rgTopTrain[j];
                            }

                            m_log.CHECK_GT(nDiffCount, 0, "The top should have changed.");
                        }
                        else
                        {
                            double dfSum = 0;
                            for (int j = 0; j < rgTopTest.Length; j++)
                            {
                                double dfDiff = Math.Abs(rgTopTest[j] - rgTopTrain[j]);
                                dfSum += dfDiff;
                            }

                            double dfAve = dfSum / rgTopTest.Length;
                            if (dfAve < 1e-06)
                                break;
                        }
                    }

                    double dfTotalDiff = 0;
                    for (int j = 0; j < rgTopTest.Length; j++)
                    {
                        double dfDiff = Math.Abs(rgTarget[j] - rgTopTest[j]);
                        dfTotalDiff += dfDiff;
                    }

                    double dfAveDiff = dfTotalDiff / rgTopTest.Length;
                    Trace.WriteLine("Iter #" + i.ToString() + " Ave Diff = " + dfAveDiff.ToString());
                }
            }
            finally
            {
                mycaffe.Dispose();
            }
        }

        public void TestLayerForward(bool bUseLinear, string strType, SolverParameter.SolverType type, int nN, int nC, int nH, int nW)
        {
            CancelEvent evtCancel = new CancelEvent();
            SettingsCaffe s = new SettingsCaffe();
            s.GpuIds = "0";
            MyCaffeControl<T> mycaffe = new MyCaffeControl<T>(s, m_log, evtCancel);

            try
            {
                string strSolver = buildSolver(type);
                string strModel = buildModel(true, strType, nN, nC, nH, nW, false, 2, nH, bUseLinear);

                mycaffe.LoadLite(Phase.TRAIN, strSolver, strModel);

                Net<T> net = mycaffe.GetInternalNet(Phase.TRAIN);
                Blob<T> blobX = net.FindBlob("data");

                // Make sure to use the filler on the same kernel as the net.
                Filler<T> filler = mycaffe.CreateFiller(m_filler.filler_param);
                filler.Fill(blobX);

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

        public void TestLayerGradient(bool bUseLinear, string strType, SolverParameter.SolverType type, int nN, int nC, int nH, int nW)
        {
            CancelEvent evtCancel = new CancelEvent();
            SettingsCaffe s = new SettingsCaffe();
            s.GpuIds = "0";
            MyCaffeControl<T> mycaffe = new MyCaffeControl<T>(s, m_log, evtCancel);

            try
            {
                string strSolver = buildSolver(type);
                string strModel = buildModel(true, strType, nN, nC, nH, nW, true, 2, nH, bUseLinear);

                mycaffe.LoadLite(Phase.TRAIN, strSolver, strModel);

                Solver<T> solver = mycaffe.GetInternalSolver();
                Net<T> net = mycaffe.GetInternalNet(Phase.TRAIN);
                Blob<T> blobX = net.FindBlob("data");

                // Make sure to use the filler on the same kernel as the net.
                Filler<T> filler = mycaffe.CreateFiller(m_filler.filler_param);
                filler.Fill(blobX);

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
