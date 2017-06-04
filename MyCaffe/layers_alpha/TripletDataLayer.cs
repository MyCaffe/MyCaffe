using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Diagnostics;
using MyCaffe.basecode;
using MyCaffe.imagedb;
using MyCaffe.param;
using MyCaffe.common;
using MyCaffe.data;

namespace MyCaffe.layers.alpha
{
    /// <summary>
    /// <H3>PRE ALPHA</H3>
    /// 
    /// The TripletDataLayer loads batches of data similarly to the DataLayer, but orders the images where
    /// the first 1/3 of the inputs are the anchors, followed by another 1/3 of the positives, followed
    /// by the last 1/3 of the negatives.
    /// 
    /// This layer is initialized with the MyCaffe.param.DataParameter.
    /// </summary>
    /// <remarks>
    /// * Initial Python code for TripletDataLayer/TripletSelectionLayer/TripletLossLayer by luhaofang/tripletloss on github. 
    /// See https://github.com/luhaofang/tripletloss - for general architecture
    /// 
    /// @see [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832) by Florian Schroff, Dmitry Kalenichenko, and James Philbin, 2015.
    /// @see [Generalisation and Sharing in Triplet Convnets for Sketch based Visual Search](https://arxiv.org/abs/1611.05301v1) by Tu Bui, Leonardo Ribeiro, Moacir Ponti, and John Collomosse, 2016.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class TripletDataLayer<T> : DataLayer<T>
    {
        Random m_random = new Random();
        Dictionary<int, Datum> m_rgAnchors = new Dictionary<int, Datum>();

        /// <summary>
        /// The TripletDataLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type TRIPLET_DATA with parameter triplet_data_param.
        /// <param name="db">Specifies the image database.</param>
        /// <param name="evtCancel">Specifies the cancel event to cancel data loading operations.</param>
        /// </param>
        public TripletDataLayer(CudaDnn<T> cuda, Log log, LayerParameter p, IXImageDatabase db, CancelEvent evtCancel)
            : base(cuda, log, p, db, evtCancel)
        {
            log.CHECK(p.type == LayerParameter.LayerType.TRIPLET_DATA, "The layer type should be TRIPLET_DATA.");
            m_type = LayerParameter.LayerType.TRIPLET_DATA;
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            base.dispose();
        }

        /// <summary>
        /// Override this method to perform the actual data loading.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void DataLayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            base.DataLayerSetUp(colBottom, colTop);
        }

        /// <summary>
        /// Load a batch of data in the background (this is run on an internal thread within the BasePrefetchingDataLayer class).
        /// </summary>
        /// <remarks>
        /// Each batch is loaded using the ordering: N/3 anchors, N/3 positives, N/3 negatives.
        /// </remarks>
        /// <param name="batch">Specifies the Batch of data to load.</param>
        protected override void load_batch(Batch<T> batch)
        {
            if (m_param.data_param.display_timing)
            {
                m_swTimerBatch.Restart();
                m_dfReadTime = 0;
                m_dfTransTime = 0;
            }

            m_log.CHECK(batch.Data.count() > 0, "There is no space allocated for data!");
            int nBatchSize = (int)m_param.data_param.batch_size;
            m_log.CHECK_EQ(0, nBatchSize % 3, "The batch size must be a multiple of 3 for triplet select.");

            T[] rgTopLabel = null;
            if (m_bOutputLabels)
                rgTopLabel = batch.Label.mutable_cpu_data;

            List<T> rgAnchors = new List<T>();
            List<T> rgPositives = new List<T>();
            List<T> rgNegatives = new List<T>();
            List<T> rgTopData = new List<T>();
            Datum datum = null;

            //-----------------------------------
            //  Get the anchors
            //-----------------------------------

            Next();

            while (Skip())
            {
                Next();
            }

            datum = m_cursor.GetValue();

            if (!m_rgAnchors.ContainsKey(datum.label))
                m_rgAnchors.Add(datum.label, datum);

            int nIdx = m_random.Next(m_rgAnchors.Count);
            datum = m_rgAnchors.ElementAt(nIdx).Value;

            T[] rgAnchor = m_transformer.Transform(datum);
            int nAnchorLabel = datum.label;
            int nAnchorIndex = datum.index;

            for (int i = 0; i < nBatchSize/3; i++)
            {
                if (m_param.data_param.display_timing)
                    m_swTimerTransaction.Restart();

                rgAnchors.AddRange(rgAnchor);

                if (m_param.data_param.display_timing)
                    m_dfTransTime += m_swTimerTransaction.Elapsed.TotalMilliseconds;

                // Copy label.
                if (m_bOutputLabels)
                    rgTopLabel[i] = (T)Convert.ChangeType(nAnchorLabel, typeof(T));
            }

            rgTopData.AddRange(rgAnchors);


            //-----------------------------------
            //  Get the positives
            //-----------------------------------

            for (int i = nBatchSize / 3; i < 2 * nBatchSize / 3; i++)
            {
                if (m_param.data_param.display_timing)
                    m_swTimerTransaction.Restart();

                int nImageIdx = nAnchorIndex;
                int nImageLabel = -1;
                while (nImageIdx == nAnchorIndex || nAnchorLabel != nImageLabel)
                {
                    Next();

                    while (Skip())
                    {
                        Next();
                    }

                    datum = m_cursor.GetValue();
                    nImageIdx = datum.index;
                    nImageLabel = datum.label;
                }

                if (m_param.data_param.display_timing)
                {
                    m_dfReadTime += m_swTimerTransaction.Elapsed.TotalMilliseconds;
                    m_swTimerTransaction.Restart();
                }

                // Apply data transformations (mirrow, scaling, crop, etc)
                rgPositives.AddRange(m_transformer.Transform(datum));

                if (m_param.data_param.display_timing)
                    m_dfTransTime += m_swTimerTransaction.Elapsed.TotalMilliseconds;

                // Copy label.
                if (m_bOutputLabels)
                    rgTopLabel[i] = (T)Convert.ChangeType(datum.Label, typeof(T));
            }

            rgTopData.AddRange(rgPositives);


            //-----------------------------------
            //  Get the negatives
            //-----------------------------------

            for (int i = 2 * nBatchSize / 3; i < nBatchSize; i++)
            {
                if (m_param.data_param.display_timing)
                    m_swTimerTransaction.Restart();

                int nImageLabel = nAnchorLabel;
                while (nImageLabel == nAnchorLabel)
                {
                    Next();

                    while (Skip())
                    {
                        Next();
                    }

                    datum = m_cursor.GetValue();
                    nImageLabel = datum.label;
                }

                if (m_param.data_param.display_timing)
                {
                    m_dfReadTime += m_swTimerTransaction.Elapsed.TotalMilliseconds;
                    m_swTimerTransaction.Restart();
                }

                // Apply data transformations (mirrow, scaling, crop, etc)
                rgNegatives.AddRange(m_transformer.Transform(datum));

                if (m_param.data_param.display_timing)
                    m_dfTransTime += m_swTimerTransaction.Elapsed.TotalMilliseconds;

                // Copy label.
                if (m_bOutputLabels)
                    rgTopLabel[i] = (T)Convert.ChangeType(datum.Label, typeof(T));
            }

            rgTopData.AddRange(rgNegatives);

            batch.Data.SetCPUData(rgTopData.ToArray());

            if (m_bOutputLabels)
                batch.Label.SetCPUData(rgTopLabel);

            if (m_param.data_param.display_timing)
            {
                m_swTimerBatch.Stop();
                m_swTimerTransaction.Stop();
                m_log.WriteLine("Prefetch batch: " + m_swTimerBatch.ElapsedMilliseconds.ToString() + " ms.", true);
                m_log.WriteLine("     Read time: " + m_dfReadTime.ToString() + " ms.", true);
                m_log.WriteLine("Transform time: " + m_dfTransTime.ToString() + " ms.", true);
            }
        }
    }
}
