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
using System.Diagnostics;

namespace MyCaffe.test
{
    [TestClass]
    public class TestMergeLayer
    {
        [TestMethod]
        public void TestForward()
        {
            MergeLayerTest2 test = new MergeLayerTest2(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IMergeLayerTest2 t in test.Tests)
                {
                    t.TestForward();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardNegativeIndexing()
        {
            MergeLayerTest2 test = new MergeLayerTest2(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IMergeLayerTest2 t in test.Tests)
                {
                    t.TestForwardNegativeIndexing();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradient()
        {
            MergeLayerTest2 test = new MergeLayerTest2(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IMergeLayerTest2 t in test.Tests)
                {
                    t.TestGradient();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface IMergeLayerTest2 : ITest
    {
        void TestForward();
        void TestForwardNegativeIndexing();
        void TestGradient();
    }

    class MergeLayerTest2 : TestBase
    {
        public MergeLayerTest2(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Merge Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new MergeLayerTest2<double>(strName, nDeviceID, engine);
            else
                return new MergeLayerTest2<float>(strName, nDeviceID, engine);
        }
    }

    class MergeLayerTest2<T> : TestEx<T>, IMergeLayerTest2
    {
        Blob<T> m_blobBottom2;

        public MergeLayerTest2(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 4, 5 }, nDeviceID)
        {
            m_engine = engine;
            m_blobBottom2 = new Blob<T>(m_cuda, m_log);
            m_blobBottom2.ReshapeLike(Bottom);
            BottomVec.Add(m_blobBottom2);
        }

        protected override FillerParameter getFillerParam()
        {
            return base.getFillerParam();
        }

        protected override void dispose()
        {
            if (m_blobBottom2 != null)
            {
                m_blobBottom2.Dispose();
                m_blobBottom2 = null;
            }

            base.dispose();
        }

        public void TestForward()
        {
            // Using SEQ_MAJOR ordering.
            // #. = sequence, max = 5; .# = batch, max = 4
            double[] rgSrcData1 = new double[] { 1.1, 2.1, 3.1, 4.1, 5.1, 1.2, 2.2, 3.2, 4.2, 5.2, 1.3, 2.3, 3.3, 4.3, 5.3, 1.4, 2.4, 3.4, 4.4, 5.4 }; // 5 x 4 x 1 x 1
            // #. = sequence, max = 2; .# = batch, max = 4
            double[] rgSrcData2 = new double[] { 10.11, 20.11, 10.22, 20.22, 10.33, 20.33, 10.44, 20.44 }; // 2 x 4 x 1 x 1

            // Copy src data 2 [0:1,4,1,1] -> C
            double[] rgExpected2 = new double[] { 5.1, 10.11, 20.11, 5.2, 10.22, 20.22, 5.3, 10.33, 20.33, 5.4, 10.44, 20.44 };

            m_blob_bottom.Reshape(5, 4, 1, 1);
            m_blob_bottom.mutable_cpu_data = convert(rgSrcData1);
            m_blobBottom2.Reshape(2, 4, 1, 1);
            m_blobBottom2.mutable_cpu_data = convert(rgSrcData2);
            m_blob_top.Reshape(1, 1, 1, 1);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.MERGE);
            p.merge_param.copy_count = 4; // batch
            p.merge_param.src_start_idx1 = 4; // last item (dim = 1)
            p.merge_param.dst_start_idx1 = 0; // first item (dim = 1)
            p.merge_param.src_start_idx2 = 0; // first two items (dim = 2)
            p.merge_param.dst_start_idx2 = 1; // second item (dim = 2)
            p.merge_param.copy_dim1 = 1;  // <---+
            p.merge_param.copy_dim2 = 2;  // <---+ combined should = dst shape(0)
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            m_log.CHECK(layer.type == LayerParameter.LayerType.MERGE, "The layer type is incorrect!");

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            m_log.CHECK_EQ(TopVec[0].num, 3, "The top num should = 3!");
            m_log.CHECK_EQ(TopVec[0].channels, 4, "The top num should = 4!");
            m_log.CHECK_EQ(TopVec[0].height, 1, "The top height should = 1!");
            m_log.CHECK_EQ(TopVec[0].width, 1, "The top width should = 1!");

            // Now, check values
            double[] rgBottomData1 = convert(Bottom.update_cpu_data());
            double[] rgBottomData2 = convert(m_blobBottom2.update_cpu_data());
            double[] rgTopData = convert(Top.update_cpu_data());

            for (int i = 0; i < rgTopData.Length; i++)
            {
                double dfExpected = rgExpected2[i];
                double dfActual = rgTopData[i];
                m_log.EXPECT_NEAR_FLOAT(dfExpected, dfActual, 0.000001, "The values are not as expected!");
            }

            layer.Dispose();
        }

        public void TestForwardNegativeIndexing()
        {
            // Using SEQ_MAJOR ordering.
            // #. = sequence, max = 5; .# = batch, max = 4
            double[] rgSrcData1 = new double[] { 1.1, 2.1, 3.1, 4.1, 5.1, 1.2, 2.2, 3.2, 4.2, 5.2, 1.3, 2.3, 3.3, 4.3, 5.3, 1.4, 2.4, 3.4, 4.4, 5.4 }; // 5 x 4 x 1 x 1
            // #. = sequence, max = 2; .# = batch, max = 4
            double[] rgSrcData2 = new double[] { 10.11, 20.11, 10.22, 20.22, 10.33, 20.33, 10.44, 20.44 }; // 2 x 4 x 1 x 1

            // Copy src data 2 [0:1,4,1,1] -> C
            double[] rgExpected2 = new double[] { 5.1, 10.11, 20.11, 5.2, 10.22, 20.22, 5.3, 10.33, 20.33, 5.4, 10.44, 20.44 };

            m_blob_bottom.Reshape(5, 4, 1, 1);
            m_blob_bottom.mutable_cpu_data = convert(rgSrcData1);
            m_blobBottom2.Reshape(2, 4, 1, 1);
            m_blobBottom2.mutable_cpu_data = convert(rgSrcData2);
            m_blob_top.Reshape(1, 1, 1, 1);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.MERGE);
            p.merge_param.copy_count = 4; // batch
            p.merge_param.src_start_idx1 = 4; // last item (dim = 1)
            p.merge_param.dst_start_idx1 = 0; // first item (dim = 1)
            p.merge_param.src_start_idx2 = 0; // first two items (dim = 2)
            p.merge_param.dst_start_idx2 = 1; // second item (dim = 2)
            p.merge_param.copy_dim1 = 1;  // <---+
            p.merge_param.copy_dim2 = 2;  // <---+ combined should = dst shape(0)
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            m_log.CHECK(layer.type == LayerParameter.LayerType.MERGE, "The layer type is incorrect!");

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            m_log.CHECK_EQ(TopVec[0].num, 3, "The top num should = 3!");
            m_log.CHECK_EQ(TopVec[0].channels, 4, "The top num should = 4!");
            m_log.CHECK_EQ(TopVec[0].height, 1, "The top height should = 1!");
            m_log.CHECK_EQ(TopVec[0].width, 1, "The top width should = 1!");

            // Now, check values
            double[] rgBottomData1 = convert(Bottom.update_cpu_data());
            double[] rgBottomData2 = convert(m_blobBottom2.update_cpu_data());
            double[] rgTopData = convert(Top.update_cpu_data());

            for (int i = 0; i < rgTopData.Length; i++)
            {
                double dfExpected = rgExpected2[i];
                double dfActual = rgTopData[i];
                m_log.EXPECT_NEAR_FLOAT(dfExpected, dfActual, 0.000001, "The values are not as expected!");
            }

            layer.Dispose();
        }

        public void TestGradient()
        {
            // Using SEQ_MAJOR ordering.
            // #. = sequence, max = 5; .# = batch, max = 4
            double[] rgSrcData1 = new double[] { 1.1, 2.1, 3.1, 4.1, 5.1, 1.2, 2.2, 3.2, 4.2, 5.2, 1.3, 2.3, 3.3, 4.3, 5.3, 1.4, 2.4, 3.4, 4.4, 5.4 }; // 5 x 4 x 1 x 1
            // #. = sequence, max = 2; .# = batch, max = 4
            double[] rgSrcData2 = new double[] { 10.11, 20.11, 10.22, 20.22, 10.33, 20.33, 10.44, 20.44 }; // 2 x 4 x 1 x 1

            // Copy src data 2 [0:1,4,1,1] -> C
            double[] rgExpected2 = new double[] { 5.1, 10.11, 20.11, 5.2, 10.22, 20.22, 5.3, 10.33, 20.33, 5.4, 10.44, 20.44 };

            m_blob_bottom.Reshape(5, 4, 1, 1);
            m_blob_bottom.mutable_cpu_data = convert(rgSrcData1);
            m_blobBottom2.Reshape(2, 4, 1, 1);
            m_blobBottom2.mutable_cpu_data = convert(rgSrcData2);
            m_blob_top.Reshape(1, 1, 1, 1);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.MERGE);
            p.merge_param.copy_count = 4; // batch
            p.merge_param.src_start_idx1 = 4; // last item (dim = 1)
            p.merge_param.dst_start_idx1 = 0; // first item (dim = 1)
            p.merge_param.src_start_idx2 = 0; // first two items (dim = 2)
            p.merge_param.dst_start_idx2 = 1; // second item (dim = 2)
            p.merge_param.copy_dim1 = 1;  // <---+
            p.merge_param.copy_dim2 = 2;  // <---+ combined should = dst shape(0)
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            m_log.CHECK(layer.type == LayerParameter.LayerType.MERGE, "The layer type is incorrect!");

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            m_log.CHECK_EQ(TopVec[0].num, 3, "The top num should = 3!");
            m_log.CHECK_EQ(TopVec[0].channels, 4, "The top num should = 4!");
            m_log.CHECK_EQ(TopVec[0].height, 1, "The top height should = 1!");
            m_log.CHECK_EQ(TopVec[0].width, 1, "The top width should = 1!");

            // Now, check values
            double[] rgBottomData1 = convert(Bottom.update_cpu_data());
            double[] rgBottomData2 = convert(m_blobBottom2.update_cpu_data());
            double[] rgTopData = convert(TopVec[0].update_cpu_data());

            for (int i = 0; i < rgTopData.Length; i++)
            {
                double dfExpected = rgExpected2[i];
                double dfActual = rgTopData[i];
                m_log.EXPECT_NEAR_FLOAT(dfExpected, dfActual, 0.000001, "The values are not as expected!");
            }

            m_blobBottom2.SetDiff(0);
            m_cuda.copy(m_blobBottom2.count(), m_blobBottom2.gpu_data, m_blobBottom2.mutable_gpu_diff);
            m_blob_bottom.SetDiff(0);
            m_cuda.copy(m_blob_bottom.count(), m_blob_bottom.gpu_data, m_blob_bottom.mutable_gpu_diff);
            m_cuda.copy(TopVec[0].count(), TopVec[0].gpu_data, TopVec[0].mutable_gpu_diff);

            layer.Backward(TopVec, new List<bool>() { true }, BottomVec);

            double[] rgSrcExpected1 = new double[] { 0.0, 0.0, 0.0, 0.0, 5.1, 0.0, 0.0, 0.0, 0.0, 5.2, 0.0, 0.0, 0.0, 0.0, 5.3, 0.0, 0.0, 0.0, 0.0, 5.4 }; // 5 x 4 x 1 x 1
                                                                                                                                                       // #. = sequence, max = 2; .# = batch, max = 4
            double[] rgSrcExpected2 = new double[] { 10.11, 20.11, 10.22, 20.22, 10.33, 20.33, 10.44, 20.44 }; // 2 x 4 x 1 x 1

            double[] rgActual1 = convert(m_blob_bottom.mutable_cpu_diff);
            double[] rgActual2 = convert(m_blobBottom2.mutable_cpu_diff);

            for (int i = 0; i < rgSrcExpected1.Length; i++)
            {
                double dfExpected = rgSrcExpected1[i];
                double dfActual = rgActual1[i];
                m_log.EXPECT_NEAR_FLOAT(dfExpected, dfActual, 0.000001, "The values are not as expected!");
            }

            for (int i = 0; i < rgSrcExpected2.Length; i++)
            {
                double dfExpected = rgSrcExpected2[i];
                double dfActual = rgActual2[i];
                m_log.EXPECT_NEAR_FLOAT(dfExpected, dfActual, 0.000001, "The values are not as expected!");
            }

            layer.Dispose();
        }
    }
}
