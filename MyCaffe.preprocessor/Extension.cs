using MyCaffe.common;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.preprocessor
{
    /// <summary>
    /// The Extension class is used to add new pre-processor extension DLL's to MyCaffe.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class Extension<T> : IDisposable
    {
        IXMyCaffeExtension<T> m_iextension;
        long m_hExtension = 0;

        /// <summary>
        /// Defines the functions implemented by the extension pre-processor DLL.
        /// </summary>
        public enum FUNCTION
        {
            /// <summary>
            /// Initialize the pre-processor.
            /// </summary>
            INITIALIZE = 1,
            /// <summary>
            /// Clean-up the pre-processor.
            /// </summary>
            CLEANUP = 2,
            /// <summary>
            /// Set all internal memory handles based on an input and output blob.
            /// </summary>
            SETMEMORY = 3,
            /// <summary>
            /// Add new data to the input data.
            /// </summary>
            ADDDATA = 4,
            /// <summary>
            /// Process the data moving the results to the output data blob.
            /// </summary>
            PROCESSDATA = 5,
            /// <summary>
            /// Create the visualization data.
            /// </summary>
            GETVISUALIZATION = 6,
            /// <summary>
            /// Clear the input data.
            /// </summary>
            CLEAR = 7
        }

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="iextension">Specifies the IXMyCaffeExtension interface implemented by the MyCaffeControl.</param>
        public Extension(IXMyCaffeExtension<T> iextension)
        {
            m_iextension = iextension;
        }

        /// <summary>
        /// Release the processor extension.
        /// </summary>
        public void Dispose()
        {
            if (m_hExtension != 0)
            {
                Run(FUNCTION.CLEANUP);
                m_iextension.FreeExtension(m_hExtension);
                m_hExtension = 0;
            }
        }

        /// <summary>
        /// Initialize a new pre-processor extension and load it.
        /// </summary>
        /// <param name="strPath">Specifies the pre-processor DLL path to load.</param>
        public void Initialize(string strPath)
        {
            if (m_hExtension != 0)
                m_iextension.FreeExtension(m_hExtension);

            m_hExtension = m_iextension.CreateExtension(strPath);
        }

        /// <summary>
        /// Run a function on the pre-processor DLL without any arguments.
        /// </summary>
        /// <param name="fn">Specifies the function to run.</param>
        public void Run(FUNCTION fn)
        {
            m_iextension.RunExtension(m_hExtension, (long)fn, null);
        }

        /// <summary>
        /// Run a function on the pre-processor DLL with arguments.
        /// </summary>
        /// <param name="fn">Specifies the function to run.</param>
        /// <param name="rgParam">Specifies the arguments.</param>
        /// <returns>The return values from the pre-processor DLL are returned.</returns>
        public double[] Run(FUNCTION fn, double[] rgParam)
        {
            return m_iextension.RunExtensionD(m_hExtension, (long)fn, rgParam);
        }

        /// <summary>
        /// Run a function on the pre-processor DLL with arguments.
        /// </summary>
        /// <param name="fn">Specifies the function to run.</param>
        /// <param name="rgParam">Specifies the arguments.</param>
        /// <returns>The return values from the pre-processor DLL are returned.</returns>
        public float[] Run(FUNCTION fn, float[] rgParam)
        {
            return m_iextension.RunExtensionF(m_hExtension, (long)fn, rgParam);
        }
    }
}
