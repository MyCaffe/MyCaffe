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
using MyCaffe.layers.beta;
using System.Drawing;
using System.IO;
using SimpleGraphing;
using static MyCaffe.param.LayerParameter;

namespace MyCaffe.test
{
    [TestClass]
    public class TestActivations
    {
        [TestMethod]
        public void TestGelu()
        {
            ActivationsTest test = new ActivationsTest();

            try
            {
                foreach (IActivationsTest t in test.Tests)
                {
                    t.TestActivation(LayerParameter.LayerType.GELU, false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGeluBert()
        {
            ActivationsTest test = new ActivationsTest();

            try
            {
                foreach (IActivationsTest t in test.Tests)
                {
                    t.TestActivation(LayerParameter.LayerType.GELU, true);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestRelu()
        {
            ActivationsTest test = new ActivationsTest();

            try
            {
                foreach (IActivationsTest t in test.Tests)
                {
                    t.TestActivation(LayerParameter.LayerType.RELU);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSigmoid()
        {
            ActivationsTest test = new ActivationsTest();

            try
            {
                foreach (IActivationsTest t in test.Tests)
                {
                    t.TestActivation(LayerParameter.LayerType.SIGMOID);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTanh()
        {
            ActivationsTest test = new ActivationsTest();

            try
            {
                foreach (IActivationsTest t in test.Tests)
                {
                    t.TestActivation(LayerParameter.LayerType.TANH);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestElu()
        {
            ActivationsTest test = new ActivationsTest();

            try
            {
                foreach (IActivationsTest t in test.Tests)
                {
                    t.TestActivation(LayerParameter.LayerType.ELU);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestPRelu()
        {
            ActivationsTest test = new ActivationsTest();

            try
            {
                foreach (IActivationsTest t in test.Tests)
                {
                    t.TestActivation(LayerParameter.LayerType.PRELU);
                }
            }
            finally
            {
                test.Dispose();
            }
        }
        
        [TestMethod]
        public void TestSerf()
        {
            ActivationsTest test = new ActivationsTest();

            try
            {
                foreach (IActivationsTest t in test.Tests)
                {
                    t.TestActivation(LayerParameter.LayerType.SERF);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMish()
        {
            ActivationsTest test = new ActivationsTest();

            try
            {
                foreach (IActivationsTest t in test.Tests)
                {
                    t.TestActivation(LayerParameter.LayerType.MISH);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestAll()
        {
            ActivationsTest test = new ActivationsTest();

            try
            {
                foreach (IActivationsTest t in test.Tests)
                {
                    t.TestAllActivations();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface IActivationsTest : ITest
    {
        void TestActivation(LayerParameter.LayerType layerType, bool? bExtra = null);
        void TestAllActivations();
    }

    class ActivationsTest : TestBase
    {
        public ActivationsTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Test Activation", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new ActivationsTest<double>(strName, nDeviceID, engine);
            else
                return new ActivationsTest<float>(strName, nDeviceID, engine);
        }
    }

    class ActivationsTest<T> : TestEx<T>, IActivationsTest
    {
        public ActivationsTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_engine = engine;
        }

        protected override void dispose()
        {
            base.dispose();
        }

        public void FillDataFwd()
        {
            int nCount = 100;
            float[] rgX = new float[nCount];
            float fStart = -8.0f;
            float fStep = (16.0f / nCount);

            for (int i = 0; i < nCount; i++)
            {
                rgX[i] = fStart + (fStep * i);
            }

            m_blob_bottom.Reshape(1, 1, 1, nCount);
            m_blob_bottom.mutable_cpu_data = convert(rgX);
            
            BottomVec.Clear();
            BottomVec.Add(m_blob_bottom);

            m_blob_top.Reshape(1, 1, 1, nCount);
            m_blob_top.SetData(0);
        }
        public void FillDataBwd()
        {
            int nCount = 100;
            float[] rgY = new float[nCount];
            float fStart = -8.0f;
            float fStep = (16.0f / nCount);

            for (int i = 0; i < nCount; i++)
            {
                rgY[i] = fStart + (fStep * i);
            }

            m_blob_top.mutable_cpu_diff = convert(rgY);
        }


        private Image plotFunction(LayerParameter.LayerType layerType, Blob<T> bottom, Blob<T> top, Configuration cfg)
        {
            float[] rgX = convertF(bottom.mutable_cpu_data);
            float[] rgY = convertF(top.mutable_cpu_data);

            PlotCollection rgPlots = new PlotCollection();
            for (int i = 0; i < rgX.Length; i++)
            {
                rgPlots.Add(new Plot(rgX[i], rgY[i]));
            }

            rgPlots.Name = layerType.ToString() + " Activation";

            PlotCollectionSet set = new PlotCollectionSet(rgPlots);
            List<PlotCollectionSet> rgPlots1 = new List<PlotCollectionSet>() { set };

            return SimpleGraphingControl.QuickRender(rgPlots1, cfg);
        }

        private Image plotDerivative(LayerParameter.LayerType layerType, Blob<T> bottom, Blob<T> top, Configuration cfg)
        {
            // diff = x * grad and we just want to show grad, so we divide diff by x to get grad.
            m_cuda.div(bottom.count(), bottom.gpu_diff, bottom.gpu_data, bottom.mutable_gpu_diff);

            float[] rgX = convertF(bottom.mutable_cpu_data);
            float[] rgY = convertF(bottom.mutable_cpu_diff);

            PlotCollection rgPlots = new PlotCollection();
            for (int i = 0; i < rgX.Length; i++)
            {
                rgPlots.Add(new Plot(rgX[i], rgY[i]));
            }

            rgPlots.Name = layerType.ToString() + " Derivative";

            PlotCollectionSet set = new PlotCollectionSet(rgPlots);
            List<PlotCollectionSet> rgPlots1 = new List<PlotCollectionSet>() { set };

            return SimpleGraphingControl.QuickRender(rgPlots1, cfg);
        }

        private Image plotFunctions(List<Tuple<string, Tuple<float[], float[], float[]>, Color>> rgRes, Configuration cfg)
        {
            PlotCollectionSet set = new PlotCollectionSet();
            List<PlotCollectionSet> rgPlots1 = new List<PlotCollectionSet>() { set };

            cfg.Frames[0].Name = "Activation Comparison";

            foreach (Tuple<string, Tuple<float[], float[], float[]>, Color> item in rgRes)
            {
                PlotCollection rgPlots = new PlotCollection(item.Item1 + " Activation");
                float[] rgX = item.Item2.Item1;
                float[] rgY = item.Item2.Item2;

                for (int i = 0; i < item.Item2.Item1.Length; i++)
                {
                    rgPlots.Add(new Plot(rgX[i], rgY[i]));
                }

                set.Add(rgPlots);
            }

            return SimpleGraphingControl.QuickRender(rgPlots1, cfg);
        }

        private Image plotDerivatives(List<Tuple<string, Tuple<float[], float[], float[]>, Color>> rgRes, Configuration cfg)
        {
            PlotCollectionSet set = new PlotCollectionSet();
            List<PlotCollectionSet> rgPlots1 = new List<PlotCollectionSet>() { set };

            cfg.Frames[0].Name = "Derivative Comparison";

            foreach (Tuple<string, Tuple<float[], float[], float[]>, Color> item in rgRes)
            {
                PlotCollection rgPlots = new PlotCollection(item.Item1 + " Derivative");
                float[] rgX = item.Item2.Item1;
                float[] rgY = item.Item2.Item3;

                for (int i = 0; i < item.Item2.Item1.Length; i++)
                {
                    rgPlots.Add(new Plot(rgX[i], rgY[i]));
                }

                set.Add(rgPlots);
            }

            return SimpleGraphingControl.QuickRender(rgPlots1, cfg);
        }

        public Tuple<LayerParameter.LayerType, bool?, Blob<T>, Blob<T>> TestActivation(LayerParameter.LayerType layerType, bool? bExtra, Blob<T> bottom = null, Blob<T> top = null)
        {
            FillDataFwd();

            LayerParameter p = new LayerParameter(layerType);
            string strExtra = "";
            if (layerType == LayerParameter.LayerType.GELU && bExtra.HasValue)
            {
                p.gelu_param.enable_bert_version = bExtra.Value;
                if (bExtra.Value)
                    strExtra = "_BERT";
            }

            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, null);

            if (bottom != null)
            {
                BottomVec.Clear();
                BottomVec.Add(bottom);
            }

            if (top != null)
            {
                TopVec.Clear();
                TopVec.Add(top);
            }

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            FillDataBwd();

            layer.Backward(TopVec, new List<bool>() { true }, BottomVec);

            if (bottom == null && top == null)
            {
                string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\results\\activation_test";
                if (!Directory.Exists(strPath))
                    Directory.CreateDirectory(strPath);

                Configuration cfg = SimpleGraphingControl.GetQuickRenderConfiguration(layerType.ToString() + " Activation", m_blob_bottom.count());
                cfg.Frames[0].XAxis.ValueType = ConfigurationAxis.VALUE_TYPE.NUMBER;
                cfg.Frames[0].XAxis.Decimals = 3;
                cfg.Frames[0].Plots[0].LookaheadActive = false;

                Image bmpFwd = plotFunction(layerType, BottomVec[0], TopVec[0], cfg);
                Image bmpBwd = plotDerivative(layerType, BottomVec[0], TopVec[0], cfg);

                bmpFwd.Save(strPath + "\\" + layerType.ToString() + strExtra + "_fwd.png");
                bmpBwd.Save(strPath + "\\" + layerType.ToString() + strExtra + "_bwd.png");

                return null;
            }

            return new Tuple<LayerParameter.LayerType, bool?, Blob<T>, Blob<T>>(layerType, bExtra, bottom, top);
        }

        public void TestActivation(LayerParameter.LayerType layerType, bool? bExtra)
        {
            TestActivation(layerType, bExtra, null, null);
        }

        public void TestAllActivations()
        {
            try
            {
                List<Tuple<string, Tuple<float[], float[], float[]>, Color>> rgRes = new List<Tuple<string, Tuple<float[], float[], float[]>, Color>>();

                TestActivation(LayerParameter.LayerType.GELU, false, m_blob_bottom, m_blob_top);
                m_cuda.div(m_blob_bottom.count(), m_blob_bottom.gpu_diff, m_blob_bottom.gpu_data, m_blob_bottom.mutable_gpu_diff);
                Tuple<float[], float[], float[]> resGelu = new Tuple<float[], float[], float[]>(convertF(m_blob_bottom.mutable_cpu_data), convertF(m_blob_top.mutable_cpu_data), convertF(m_blob_bottom.mutable_cpu_diff));
                rgRes.Add(new Tuple<string, Tuple<float[], float[], float[]>, Color>("GELU", resGelu, Color.Blue));

                TestActivation(LayerParameter.LayerType.GELU, true, m_blob_bottom, m_blob_top);
                m_cuda.div(m_blob_bottom.count(), m_blob_bottom.gpu_diff, m_blob_bottom.gpu_data, m_blob_bottom.mutable_gpu_diff);
                Tuple<float[], float[], float[]> resGeluBert = new Tuple<float[], float[], float[]>(convertF(m_blob_bottom.mutable_cpu_data), convertF(m_blob_top.mutable_cpu_data), convertF(m_blob_bottom.mutable_cpu_diff));
                rgRes.Add(new Tuple<string, Tuple<float[], float[], float[]>, Color>("GELU_BERT", resGeluBert, Color.Red));

                TestActivation(LayerParameter.LayerType.SIGMOID, null, m_blob_bottom, m_blob_top);
                m_cuda.div(m_blob_bottom.count(), m_blob_bottom.gpu_diff, m_blob_bottom.gpu_data, m_blob_bottom.mutable_gpu_diff);
                Tuple<float[], float[], float[]> resSigmoid = new Tuple<float[], float[], float[]>(convertF(m_blob_bottom.mutable_cpu_data), convertF(m_blob_top.mutable_cpu_data), convertF(m_blob_bottom.mutable_cpu_diff));
                rgRes.Add(new Tuple<string, Tuple<float[], float[], float[]>, Color>("SIGMOID", resSigmoid, Color.Green));

                TestActivation(LayerParameter.LayerType.TANH, null, m_blob_bottom, m_blob_top);
                m_cuda.div(m_blob_bottom.count(), m_blob_bottom.gpu_diff, m_blob_bottom.gpu_data, m_blob_bottom.mutable_gpu_diff);
                Tuple<float[], float[], float[]> resTanh = new Tuple<float[], float[], float[]>(convertF(m_blob_bottom.mutable_cpu_data), convertF(m_blob_top.mutable_cpu_data), convertF(m_blob_bottom.mutable_cpu_diff));
                rgRes.Add(new Tuple<string, Tuple<float[], float[], float[]>, Color>("TANH", resTanh, Color.Orange));

                TestActivation(LayerParameter.LayerType.ELU, null, m_blob_bottom, m_blob_top);
                m_cuda.div(m_blob_bottom.count(), m_blob_bottom.gpu_diff, m_blob_bottom.gpu_data, m_blob_bottom.mutable_gpu_diff);
                Tuple<float[], float[], float[]> resElu = new Tuple<float[], float[], float[]>(convertF(m_blob_bottom.mutable_cpu_data), convertF(m_blob_top.mutable_cpu_data), convertF(m_blob_bottom.mutable_cpu_diff));
                rgRes.Add(new Tuple<string, Tuple<float[], float[], float[]>, Color>("ELU", resElu, Color.Lime));

                TestActivation(LayerParameter.LayerType.PRELU, null, m_blob_bottom, m_blob_top);
                m_cuda.div(m_blob_bottom.count(), m_blob_bottom.gpu_diff, m_blob_bottom.gpu_data, m_blob_bottom.mutable_gpu_diff);
                Tuple<float[], float[], float[]> resPRelu = new Tuple<float[], float[], float[]>(convertF(m_blob_bottom.mutable_cpu_data), convertF(m_blob_top.mutable_cpu_data), convertF(m_blob_bottom.mutable_cpu_diff));
                rgRes.Add(new Tuple<string, Tuple<float[], float[], float[]>, Color>("PRELU", resPRelu, Color.Purple));

                TestActivation(LayerParameter.LayerType.RELU, null, m_blob_bottom, m_blob_top);
                m_cuda.div(m_blob_bottom.count(), m_blob_bottom.gpu_diff, m_blob_bottom.gpu_data, m_blob_bottom.mutable_gpu_diff);
                Tuple<float[], float[], float[]> resRelu = new Tuple<float[], float[], float[]>(convertF(m_blob_bottom.mutable_cpu_data), convertF(m_blob_top.mutable_cpu_data), convertF(m_blob_bottom.mutable_cpu_diff));
                rgRes.Add(new Tuple<string, Tuple<float[], float[], float[]>, Color>("RELU", resRelu, Color.Brown));

                TestActivation(LayerParameter.LayerType.SERF, null, m_blob_bottom, m_blob_top);
                m_cuda.div(m_blob_bottom.count(), m_blob_bottom.gpu_diff, m_blob_bottom.gpu_data, m_blob_bottom.mutable_gpu_diff);
                Tuple<float[], float[], float[]> resSerf = new Tuple<float[], float[], float[]>(convertF(m_blob_bottom.mutable_cpu_data), convertF(m_blob_top.mutable_cpu_data), convertF(m_blob_bottom.mutable_cpu_diff));
                rgRes.Add(new Tuple<string, Tuple<float[], float[], float[]>, Color>("SERF", resSerf, Color.Cyan));

                TestActivation(LayerParameter.LayerType.MISH, null, m_blob_bottom, m_blob_top);
                m_cuda.div(m_blob_bottom.count(), m_blob_bottom.gpu_diff, m_blob_bottom.gpu_data, m_blob_bottom.mutable_gpu_diff);
                Tuple<float[], float[], float[]> resMish = new Tuple<float[], float[], float[]>(convertF(m_blob_bottom.mutable_cpu_data), convertF(m_blob_top.mutable_cpu_data), convertF(m_blob_bottom.mutable_cpu_diff));
                rgRes.Add(new Tuple<string, Tuple<float[], float[], float[]>, Color>("MISH", resMish, Color.SkyBlue));

                string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\results\\activation_test";
                if (!Directory.Exists(strPath))
                    Directory.CreateDirectory(strPath);

                Configuration cfg = SimpleGraphingControl.GetQuickRenderConfiguration("All Activations", m_blob_bottom.count());
                cfg.Frames[0].XAxis.ValueType = ConfigurationAxis.VALUE_TYPE.NUMBER;
                cfg.Frames[0].XAxis.Decimals = 3;
                cfg.Frames[0].Plots[0].PlotLineColor = Color.Transparent;
                cfg.Frames[0].Plots[0].PlotFillColor = Color.Transparent;
                cfg.Frames[0].Plots[0].LookaheadActive = false;
                cfg.Frames[0].Plots[0].LineColor = rgRes[0].Item3;
                cfg.Frames[0].Plots[0].FlagColor = rgRes[0].Item3;
                cfg.Frames[0].Plots[0].EnableLabel = true;
                cfg.Frames[0].Plots[0].EnableFlag = false;
                cfg.Frames[0].Plots[0].Name = rgRes[0].Item1;

                for (int i = 1; i < rgRes.Count; i++)
                {
                    ConfigurationPlot plot = cfg.Frames[0].Plots[0].Clone();
                    plot.Name = rgRes[i].Item1;
                    plot.LineColor = rgRes[i].Item3;
                    plot.FlagColor = rgRes[i].Item3;
                    plot.DataIndexOnRender = i;
                    plot.DataIndex = i;
                    cfg.Frames[0].Plots.Add(plot);
                }

                Image bmpFwd = plotFunctions(rgRes, cfg);
                Image bmpBwd = plotDerivatives(rgRes, cfg);

                bmpFwd.Save(strPath + "\\activations_all_fwd.png");
                bmpBwd.Save(strPath + "\\activations_all__bwd.png");
            }
            finally
            {
            }                
        }
    }
}
