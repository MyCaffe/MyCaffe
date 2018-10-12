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
using MyCaffe.layers.alpha;

/// <summary>
/// Testing for simple triplet select layer.
/// 
/// TripletSelect Layer - this is the triplet select layer used to order the triplet tuples into matching
/// top blobs where:
/// 
///     colTop[0] = anchors
///     colTop[1] = positives
///     colTop[2] = negatives
///     
/// Positives are ordered by distance in decreasing order (larger to smaller),
/// Negatives are ordered by distance in increasing order (smaller to larger).
///     
/// Where Anchors and Positives are from the same class and Negatives are from a different class.  In the basic algorithm,
/// the distance between AP and AN are determined and the learning occurs by shrinking the distance between AP and increasing
/// the distance between AN.
/// 
/// </summary>
/// <remarks>
/// * Initial Python code for TripletDataLayer/TripletSelectionLayer/TripletSelectLayer by luhaofang/tripletloss on github. 
/// See https://github.com/luhaofang/tripletloss - for general architecture
/// </remarks> 
namespace MyCaffe.test
{
    [TestClass]
    public class TestTripletSelectLayer
    {
        [TestMethod]
        public void TestForward()
        {
            TripletSelectLayerTest test = new TripletSelectLayerTest();

            try
            {
                foreach (ITripletSelectLayerTest t in test.Tests)
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
        public void TestGradient()
        {
            TripletSelectLayerTest test = new TripletSelectLayerTest();

            try
            {
                foreach (ITripletSelectLayerTest t in test.Tests)
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

    interface ITripletSelectLayerTest : ITest
    {
        void TestForward();
        void TestGradient();
    }

    class TripletSelectLayerTest : TestBase
    {
        public TripletSelectLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("TripletSelect Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new TripletSelectLayerTest<double>(strName, nDeviceID, engine);
            else
                return new TripletSelectLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class TripletSelectLayerTest<T> : TestEx<T>, ITripletSelectLayerTest
    {
        Blob<T> m_blobAnchors;
        Blob<T> m_blobPositives;
        Blob<T> m_blobNegatives;
        Blob<T> m_blobAnchorsLabels;
        Blob<T> m_blobPositivesLabels;
        Blob<T> m_blobNegativesLabels;
        Blob<T> m_blobBottomLabels;
        Blob<T> m_blobTopAnchors;
        Blob<T> m_blobTopPositives;
        Blob<T> m_blobTopNegatives;
        DatasetDescriptor m_ds;
        MyCaffeImageDatabase m_db;

        public TripletSelectLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_engine = engine;
            m_blobAnchors = new Blob<T>(m_cuda, m_log);
            m_blobPositives = new Blob<T>(m_cuda, m_log);
            m_blobNegatives = new Blob<T>(m_cuda, m_log);
            m_blobAnchorsLabels = new Blob<T>(m_cuda, m_log);
            m_blobPositivesLabels = new Blob<T>(m_cuda, m_log);
            m_blobNegativesLabels = new Blob<T>(m_cuda, m_log);
            m_blobBottomLabels = new Blob<T>(m_cuda, m_log);
            m_blobTopAnchors = new Blob<T>(m_cuda, m_log);
            m_blobTopPositives = new Blob<T>(m_cuda, m_log);
            m_blobTopNegatives = new Blob<T>(m_cuda, m_log);
            m_db = new MyCaffeImageDatabase();

            DatasetFactory factory = new DatasetFactory();
            m_ds = factory.LoadDataset("MNIST");

            SettingsCaffe s = new SettingsCaffe();
            s.ImageDbLoadMethod = IMAGEDB_LOAD_METHOD.LOAD_ALL;
            m_db.InitializeWithDs(s, m_ds);

            int nBatchSize = 2;
            Fill(nBatchSize);
        }

        protected override void dispose()
        {
            m_blobAnchors.Dispose();
            m_blobPositives.Dispose();
            m_blobNegatives.Dispose();
            m_blobAnchorsLabels.Dispose();
            m_blobPositivesLabels.Dispose();
            m_blobNegativesLabels.Dispose();
            m_blobBottomLabels.Dispose();
            m_blobTopAnchors.Dispose();
            m_blobTopPositives.Dispose();
            m_blobTopNegatives.Dispose();
            m_db.Dispose();
            base.dispose();
        }

        public void Fill(int nBatchSize)
        {
            TransformationParameter transform_param = new TransformationParameter();
            SourceDescriptor src = m_ds.TrainingSource;
            DataTransformer<T> transformer = new DataTransformer<T>(m_log, transform_param, Phase.TRAIN);
            SimpleDatum sdAnchor = m_db.QueryImage(src.ID, 0, IMGDB_LABEL_SELECTION_METHOD.RANDOM, IMGDB_IMAGE_SELECTION_METHOD.RANDOM);
            List<T> rgAnchors = new List<T>();
            List<T> rgPositives = new List<T>();
            List<T> rgNegatives = new List<T>();
            List<T> rgAnchorLabels = new List<T>();
            List<T> rgPositiveLabels = new List<T>();
            List<T> rgNegativeLabels = new List<T>();

            List<T> rgAnchor = new List<T>(transformer.Transform(new Datum(sdAnchor)));

            for (int i = 0; i < nBatchSize; i++)
            {
                rgAnchors.AddRange(rgAnchor);

                // Get the positive that is not the same image, but of the same class.
                SimpleDatum sdPositive = m_db.QueryImage(m_ds.TrainingSource.ID, 0, IMGDB_LABEL_SELECTION_METHOD.RANDOM, IMGDB_IMAGE_SELECTION_METHOD.RANDOM, sdAnchor.Label);

                while (sdPositive.Index == sdAnchor.Index)
                {
                    sdPositive = m_db.QueryImage(m_ds.TrainingSource.ID, 0, IMGDB_LABEL_SELECTION_METHOD.RANDOM, IMGDB_IMAGE_SELECTION_METHOD.RANDOM, sdAnchor.Label);
                }

                rgPositives.AddRange(transformer.Transform(new Datum(sdPositive)));

                // Get the negative that is of a different class.
                SimpleDatum sdNegative = m_db.QueryImage(m_ds.TrainingSource.ID, 0, IMGDB_LABEL_SELECTION_METHOD.RANDOM, IMGDB_IMAGE_SELECTION_METHOD.RANDOM);

                while (sdNegative.Label == sdAnchor.Label)
                {
                    sdNegative = m_db.QueryImage(m_ds.TrainingSource.ID, 0, IMGDB_LABEL_SELECTION_METHOD.RANDOM, IMGDB_IMAGE_SELECTION_METHOD.RANDOM);
                }

                rgNegatives.AddRange(transformer.Transform(new Datum(sdNegative)));

                rgAnchorLabels.Add((T)Convert.ChangeType(sdAnchor.Label, typeof(T)));
                rgPositiveLabels.Add((T)Convert.ChangeType(sdPositive.Label, typeof(T)));
                rgNegativeLabels.Add((T)Convert.ChangeType(sdNegative.Label, typeof(T)));
            }


            m_blobAnchors.Reshape(nBatchSize, src.ImageChannels, src.ImageHeight, src.ImageWidth);
            m_blobPositives.ReshapeLike(m_blobAnchors);
            m_blobNegatives.ReshapeLike(m_blobAnchors);

            m_blobAnchorsLabels.Reshape(nBatchSize, 1, 1, 1);
            m_blobPositivesLabels.ReshapeLike(m_blobAnchorsLabels);
            m_blobNegativesLabels.ReshapeLike(m_blobAnchorsLabels);

            m_blobAnchors.SetData(rgAnchors.ToArray());
            m_blobPositives.SetData(rgPositives.ToArray());
            m_blobNegatives.SetData(rgNegatives.ToArray());

            m_blobAnchorsLabels.SetData(rgAnchorLabels.ToArray());
            m_blobPositivesLabels.SetData(rgPositiveLabels.ToArray());
            m_blobNegativesLabels.SetData(rgNegativeLabels.ToArray());

            Bottom.Reshape((int)nBatchSize * 3, m_ds.TrainingSource.ImageChannels, m_ds.TrainingSource.ImageHeight, m_ds.TrainingSource.ImageWidth);

            m_cuda.copy(m_blobAnchors.count(), m_blobAnchors.gpu_data, Bottom.mutable_gpu_data, 0, 0);
            m_cuda.copy(m_blobPositives.count(), m_blobPositives.gpu_data, Bottom.mutable_gpu_data, 0, m_blobAnchors.count());
            m_cuda.copy(m_blobNegatives.count(), m_blobNegatives.gpu_data, Bottom.mutable_gpu_data, 0, m_blobAnchors.count() + m_blobPositives.count());

            m_blobBottomLabels.Reshape(nBatchSize * 3, 1, 1, 1);

            m_cuda.copy(m_blobAnchorsLabels.count(), m_blobAnchorsLabels.gpu_data, m_blobBottomLabels.mutable_gpu_data, 0, 0);
            m_cuda.copy(m_blobPositivesLabels.count(), m_blobPositivesLabels.gpu_data, m_blobBottomLabels.mutable_gpu_data, 0, m_blobAnchorsLabels.count());
            m_cuda.copy(m_blobNegativesLabels.count(), m_blobNegativesLabels.gpu_data, m_blobBottomLabels.mutable_gpu_data, 0, m_blobAnchorsLabels.count() + m_blobPositivesLabels.count());

            BottomVec.Clear();
            BottomVec.Add(Bottom);

            TopVec.Clear();
            TopVec.Add(m_blobTopAnchors);
            TopVec.Add(m_blobTopPositives);
            TopVec.Add(m_blobTopNegatives);
        }

        protected override FillerParameter getFillerParam()
        {
            return new FillerParameter("gaussian");
        }

        public void TestForward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.TRIPLET_SELECT);
            TripletSelectLayer<T> layer = new TripletSelectLayer<T>(m_cuda, m_log, p);
            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            m_log.CHECK_EQ(3, TopVec.Count, "The TopVec should have three elements: (0) anchors, (1) positives and (2) negatives.");
            m_log.CHECK_EQ(TopVec[0].count(), TopVec[1].count(), "The Top[0] and Top[1] should have the same count.");
            m_log.CHECK_EQ(TopVec[0].count(), TopVec[2].count(), "The Top[0] and Top[2] should have the same count.");

            List<double> rgAps = new List<double>();
            List<double> rgAns = new List<double>();

            List<int> rgShape = new List<int>() { 1, m_ds.TrainingSource.ImageChannels, m_ds.TrainingSource.ImageHeight, m_ds.TrainingSource.ImageWidth };
            Blob<T> blobAnchor = new Blob<T>(m_cuda, m_log, rgShape);
            Blob<T> blobPositive = new Blob<T>(m_cuda, m_log, rgShape);
            Blob<T> blobNegative = new Blob<T>(m_cuda, m_log, rgShape);
            Blob<T> blobCompare = new Blob<T>(m_cuda, m_log, rgShape);
            double[] rgLabels = convert(m_blobBottomLabels.update_cpu_data());
            int nVectorSize = Bottom.count() / Bottom.num;

            for (int i = 0; i < layer.triplets.Count; i++)
            {
                int nAnchorIdx = layer.triplets[i].Item1;
                int nPositiveIdx = layer.triplets[i].Item2;
                int nNegativeIdx = layer.triplets[i].Item3;

                int nAnchorLabel = (int)rgLabels[nAnchorIdx];
                int nPositiveLabel = (int)rgLabels[nPositiveIdx];
                int nNegativeLabel = (int)rgLabels[nNegativeIdx];

                m_cuda.copy(blobAnchor.count(), Bottom.gpu_data, blobAnchor.mutable_gpu_data, nAnchorIdx * nVectorSize, 0);
                m_cuda.copy(blobPositive.count(), Bottom.gpu_data, blobPositive.mutable_gpu_data, nPositiveIdx * nVectorSize, 0);
                m_cuda.copy(blobNegative.count(), Bottom.gpu_data, blobNegative.mutable_gpu_data, nNegativeIdx * nVectorSize, 0);


                // Verify that anchor and postive are of same class, but different indexes.

                m_log.CHECK_EQ(nAnchorLabel, nPositiveLabel, "The anchor and positive should be of the same class.");
                m_cuda.sub(blobCompare.count(), blobAnchor.gpu_data, blobPositive.gpu_data, blobCompare.mutable_gpu_data);
                double[] rgData1 = convert(blobCompare.update_cpu_data());
                int nDiffCount1 = 0;

                for (int j = 0; j < rgData1.Length; j++)
                {
                    if (rgData1[j] != 0)
                        nDiffCount1++;
                }

                m_log.CHECK_GT(nDiffCount1, 0, "The anchor and positive should be different!");

                double dfAP = m_cuda.dot_double(blobCompare.count(), blobCompare.gpu_data, blobCompare.gpu_data);
                rgAps.Add(dfAP);


                // Verify that anchor and negative are of different classes.

                m_log.CHECK_NE(nAnchorLabel, nNegativeLabel, "The anchor and negative should be of different classes.");
                m_cuda.sub(blobCompare.count(), blobAnchor.gpu_data, blobNegative.gpu_data, blobCompare.mutable_gpu_data);
                double[] rgData2 = convert(blobCompare.update_cpu_data());
                int nDiffCount2 = 0;

                for (int j = 0; j < rgData2.Length; j++)
                {
                    if (rgData2[j] != 0)
                        nDiffCount2++;
                }

                m_log.CHECK_GT(nDiffCount2, 0, "The anchor and negative should be different!");

                double dfAN = m_cuda.dot_double(blobCompare.count(), blobCompare.gpu_data, blobCompare.gpu_data);
                rgAns.Add(dfAN);
            }


            // Verify that the Aps items are ordered from largest to smallest.
            var rgAps1 = from dbl in rgAps orderby dbl descending select dbl;
            List<double> rgApsEx = rgAps1.ToList();

            for (int i = 0; i < rgAps.Count; i++)
            {
                m_log.CHECK_EQ(rgAps[i], rgApsEx[i], "The items are not the same!");
            }


            // Verify that the Ans items are ordered from smallest to largest.
            var rgAns1 = from dbl in rgAns orderby dbl select dbl;
            List<double> rgAnsEx = rgAns1.ToList();

            for (int i = 0; i < rgAns.Count; i++)
            {
                m_log.CHECK_EQ(rgAns[i], rgAnsEx[i], "The items are not the same!");
            }


            // Cleanup.

            blobAnchor.Dispose();
            blobPositive.Dispose();
            blobNegative.Dispose();
            blobCompare.Dispose();
        }

        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.TRIPLET_SELECT);            
            TripletSelectLayer<T> layer = new TripletSelectLayer<T>(m_cuda, m_log, p);
            layer.Setup(BottomVec, TopVec);
            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-2, 1701);
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec, 0);
        }
    }
}
