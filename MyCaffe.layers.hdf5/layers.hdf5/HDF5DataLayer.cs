using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;
using MyCaffe.fillers;
using System.IO;

namespace MyCaffe.layers.hdf5
{
    /// <summary>
    /// The HDF5DataLayer loads data from files in the HDF5 data format.
    /// This layer is initialized with the MyCaffe.param.HDF5DataParameter.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class HDF5DataLayer<T> : Layer<T>
    {
        List<string> m_rgstrFileNames = new List<string>();
        int m_nNumFiles = 0;
        int m_nCurrentFile = 0;
        int m_nCurrentRow = 0;
        BlobCollection<T> m_colHdfBlobs = new BlobCollection<T>();
        List<int> m_rgDataPermutation = new List<int>();
        List<int> m_rgFilePermutation = new List<int>();
        ulong m_lOffset = 0;

        /// <summary>
        /// The HDF5DataLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">
        /// Provides DummyDataParameter hdf5_data_param with options:
        ///  - data_filler. A list of Fillers to use.
        ///  
        ///  - shape.  A list of shapes to use.
        /// </param>
        public HDF5DataLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.HDF5_DATA;
        }

        /// <summary>
        /// Release all internal blobs.
        /// </summary>
        protected override void dispose()
        {
            m_colHdfBlobs.Dispose();
            base.dispose();
        }

        protected virtual void LoadHDF5FileData(string strFile)
        {
            m_log.WriteLine("Loading HDF5 file: '" + strFile + "'");
            HDF5<T> hdf5 = new HDF5<T>(m_cuda, m_log, strFile);

            int nTopCount = m_param.top.Count;
            for (int i = 0; i < nTopCount; i++)
            {
                // Allow reshape here, as we are loading data not params.
                Blob<T> blob = null;

                if (m_colHdfBlobs.Count < nTopCount)
                {
                    blob = new Blob<T>(m_cuda, m_log, false);
                    m_colHdfBlobs.Add(blob);
                }
                else
                {
                    blob = m_colHdfBlobs[i];
                }

                hdf5.load_nd_dataset(blob, m_param.top[i], true);
            }

            hdf5.Dispose();

            // MinTopBlobs=1 guarantees at least one top blob
            m_log.CHECK_GE(m_colHdfBlobs[0].num_axes, 1, "Input must have at least 1 axis.");
            int nNum = m_colHdfBlobs[0].shape(0);

            for (int i = 1; i < nTopCount; i++)
            {
                m_log.CHECK_EQ(m_colHdfBlobs[i].shape(0), nNum, "The 'num' on all blobs must be equal.");
            }

            // Default to identity permutation.
            m_rgDataPermutation = new List<int>();
            for (int i = 0; i < nNum; i++)
            {
                m_rgDataPermutation.Add(i);
            }

            // Shuffle if needed
            if (m_param.hdf5_data_param.shuffle)
            {
                m_rgDataPermutation = Utility.RandomShuffle(m_rgDataPermutation);
                m_log.WriteLine("Successfully loaded " + nNum.ToString() + " rows (shuffled).");
            }
            else
            {
                m_log.WriteLine("Successfully loaded " + nNum.ToString() + " rows.");
            }
        }

        /// <summary>
        /// Returns 0 for data layers have no bottom (input) Blobs.
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 0; }
        }

        /// <summary>
        /// Returns the minimum number of required top (output) Blobs: data
        /// </summary>
        public override int MinTopBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Not used.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            // Refuse transformation parameters since HDF5 is totally generic.
            if (m_param.transform_param != null)
                m_log.WriteLine("WARNING: " + m_type.ToString() + " does not transform data.");

            // Read the source to parse the filenames.
            m_log.WriteLine("Loading list of HDF5 file names from: " + m_param.hdf5_data_param.source);
            m_rgstrFileNames = Utility.LoadTextLines(m_param.hdf5_data_param.source, m_log, true);
            m_nNumFiles = m_rgstrFileNames.Count;
            m_nCurrentFile = 0;

            m_log.WriteLine("Number of HDF5 files: " + m_nNumFiles.ToString());
            m_log.CHECK_GE(m_nNumFiles, 1, "Must have at least one HDF5 filename listed in '" + m_param.hdf5_data_param.source + "'!");

            // Default to identity permutation.
            m_rgFilePermutation = new List<int>();
            for (int i = 0; i < m_nNumFiles; i++)
            {
                m_rgFilePermutation.Add(i);
            }

            // Shuffle if needed.
            if (m_param.hdf5_data_param.shuffle)
                m_rgFilePermutation = Utility.RandomShuffle(m_rgFilePermutation);

            // Load the first HDF5 file and initialize the line counter.
            LoadHDF5FileData(m_rgstrFileNames[m_rgFilePermutation[m_nCurrentFile]]);
            m_nCurrentRow = 0;

            // Reshape the blobs.
            int nBatchSize = (int)m_param.hdf5_data_param.batch_size;
            int nTopSize = m_param.top.Count;
            List<int> rgTopShape = new List<int>();

            for (int i = 0; i < nTopSize; i++)
            {
                rgTopShape = Utility.Clone<int>(m_colHdfBlobs[i].shape());
                rgTopShape[0] = nBatchSize;
                colTop[i].Reshape(rgTopShape);
            }
        }

        protected bool Skip()
        {
            ulong nSize = (ulong)m_param.solver_count;
            ulong nRank = (ulong)m_param.solver_rank;
            // In test mode, only rank 0 runs, so avoid skipping.
            bool bKeep = (m_lOffset % nSize) == nRank || m_param.phase == Phase.TEST;

            return !bKeep;
        }

        protected void Next()
        {
            m_nCurrentRow++;

            if (m_nCurrentRow == m_colHdfBlobs[0].shape(0))
            {
                if (m_nNumFiles > 1)
                {
                    m_nCurrentFile++;

                    if (m_nCurrentFile == m_nNumFiles)
                    {
                        m_nCurrentFile = 0;
                        if (m_param.hdf5_data_param.shuffle)
                            m_rgFilePermutation = Utility.RandomShuffle(m_rgFilePermutation);
                        m_log.WriteLine("Looping around to first file...");
                    }

                    LoadHDF5FileData(m_rgstrFileNames[m_rgFilePermutation[m_nCurrentFile]]);
                }

                m_nCurrentRow = 0;

                if (m_param.hdf5_data_param.shuffle)
                    m_rgDataPermutation = Utility.RandomShuffle(m_rgDataPermutation);
            }

            m_lOffset++;
        }

        /// <summary>
        /// Data layers have no bottoms, so reshaping is trivial.
        /// </summary>
        /// <param name="colBottom">Not used.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
        }

        /// <summary>
        /// Run the Forward computation, which fills the data into the top (output) Blobs.
        /// </summary>
        /// <param name="colBottom">Not used.</param>
        /// <param name="colTop">top output blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the data outputs.  
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            int nBatch = (int)m_param.hdf5_data_param.batch_size;

            for (int i = 0; i < nBatch; i++)
            {
                while (Skip())
                    Next();

                for (int j = 0; j < m_param.top.Count; j++)
                {
                    int nDataDim = colTop[j].count() / colTop[j].shape(0);
                    int nSrcIdx = m_rgDataPermutation[m_nCurrentRow] * nDataDim;
                    int nDstIdx = i * nDataDim;
                    m_cuda.copy(nDataDim, m_colHdfBlobs[j].gpu_data, colTop[j].mutable_gpu_data, nSrcIdx, nDstIdx);
                }

                Next();
            }
        }

        /// @brief Not implemented - data Layers do not perform backward..
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
        }
    }
}
