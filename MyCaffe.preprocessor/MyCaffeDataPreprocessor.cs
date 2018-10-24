using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.db.stream;

namespace MyCaffe.preprocessor
{
    /// <summary>
    /// The MyCaffeDataPreprocessor handles adding data from a streaming database to the GPU and then 
    /// pre-processing the data once on the GPU making it ready for use with the Data Gym.
    /// </summary>
    /// <typeparam name="T">Specfies the base type of <i>float</i> or <i>double</i>.</typeparam>
    public partial class MyCaffeDataPreprocessor<T> : Component, IXMyCaffePreprocessor<T>
    {
        MgrPreprocessor<T> m_mgrPreprocessor = null;

        /// <summary>
        /// The constructor.
        /// </summary>
        public MyCaffeDataPreprocessor()
        {
            InitializeComponent();
        }

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="container">Specifies the container that holds this compoent.</param>
        public MyCaffeDataPreprocessor(IContainer container)
        {
            container.Add(this);

            InitializeComponent();
        }

        private void dispose()
        {
            Shutdown();
        }

        /// <summary>
        /// Initialize the Data Processor.
        /// </summary>
        /// <param name="imycaffe">Specifies the instance of MyCaffe to use.</param>
        /// <param name="idb">Specifies the instance of the streaming database to use.</param>
        /// <param name="strPreProcessorDLLPath">Specifies the path to the preprocessing DLL to use.</param>
        public void Initialize(IXMyCaffe<T> imycaffe, IXStreamDatabase idb, string strPreProcessorDLLPath, PropertySet properties)
        {
            m_mgrPreprocessor = new MgrPreprocessor<T>(imycaffe, idb);

            int nFields = properties.GetPropertyAsInt("Fields", 0);
            int nDepth = properties.GetPropertyAsInt("Depth", 0);

            if (nFields == 0)
                throw new Exception("You must specify the 'Fields' property with a value greater than 0.");

            if (nDepth == 0)
                throw new Exception("You must specify the 'Depth' property with a value greater than 0.");

            m_mgrPreprocessor.Initialize(strPreProcessorDLLPath, nFields, nDepth);
        }

        /// <summary>
        /// Create and return the visualization data.
        /// </summary>
        /// <param name="sd">Specifies the visualization data to render, this is the SimpleDatum returned from the Step function.</param>
        /// <returns>A tuple containing a bitmap and SimpleDatum of the visualization data are returned.</returns>
        public Tuple<Bitmap, SimpleDatum> Render(SimpleDatum sd)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Reset to the start of the data set, or to an offset from the start.
        /// </summary>
        /// <param name="nStartOffset">Optionally, specifies an offset from the start to use.</param>
        public void Reset(int nStartOffset = 0)
        {
            m_mgrPreprocessor.Reset(nStartOffset);
        }

        /// <summary>
        /// Shutdown the preprocessor.
        /// </summary>
        public void Shutdown()
        {
            if (m_mgrPreprocessor != null)
            {
                m_mgrPreprocessor.Dispose();
                m_mgrPreprocessor = null;
            }
        }

        /// <summary>
        /// Step to the next data in the streaming database and process it.
        /// </summary>
        /// <param name="bGetSimpleDatum">Specifies to get the simple datum for visualization purposes.</param>
        /// <param name="nWait">Specifies an amount of time in ms. to wait for new data.</param>
        /// <returns>A tuple containing the Blob of processed data and optionally a SimpleDatum for visualiation is returned.</returns>
        public Tuple<Blob<T>, SimpleDatum> Step(bool bGetSimpleDatum, int nWait = 1000)
        {
            return m_mgrPreprocessor.Step(bGetSimpleDatum, nWait);
        }
    }
}
