using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.db.stream;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.preprocessor
{
    /// <summary>
    /// The MgrPreprocessor manages the operations of the data pre-processor.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class MgrPreprocessor<T> : IDisposable
    {
        Extension<T> m_extension;
        MyCaffeControl<T> m_mycaffe;
        IXStreamDatabase m_idb;
        Blob<T> m_blobInput = null;
        Blob<T> m_blobOutput = null;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="imycaffe">Specifies the instance of the MyCaffeControl to use.</param>
        /// <param name="idb">Specifies the instance of the streaming database to use.</param>
        public MgrPreprocessor(IXMyCaffe<T> imycaffe, IXStreamDatabase idb)
        {
            m_mycaffe = (MyCaffeControl<T>)imycaffe;
            m_idb = idb;
            m_extension = new Extension<T>(imycaffe as IXMyCaffeExtension<T>);
            m_blobInput = new Blob<T>(m_mycaffe.Cuda, m_mycaffe.Log);
            m_blobOutput = new Blob<T>(m_mycaffe.Cuda, m_mycaffe.Log);
        }

        /// <summary>
        /// Release all resources used.
        /// </summary>
        public void Dispose()
        {
            if (m_blobInput != null)
            {
                m_blobInput.Dispose();
                m_blobInput = null;
            }

            if (m_blobOutput != null)
            {
                m_blobOutput.Dispose();
                m_blobOutput = null;
            }
        }

        /// <summary>
        /// Initialize the pre-processor.
        /// </summary>
        /// <param name="strExtPath">Specifies the path to the pre-processor extension DLL to use.</param>
        /// <param name="nFields">Specifies the number of fields that the pre-processor uses.</param>
        /// <param name="nDepth">Specifies the depth of the pre-processor.</param>
        public void Initialize(string strExtPath, int nFields, int nDepth)
        {
            List<float> rgParam;

            m_extension.Initialize(strExtPath);

            rgParam = new List<float>();
            rgParam.Add(nFields);
            rgParam.Add(nDepth);
            rgParam.Add(1);     // call ProcessData after AddData within the pre-processor DLL.
            float[] rgOut = m_extension.Run(Extension<T>.FUNCTION.INITIALIZE, rgParam.ToArray());
            int nOutputFields = (int)rgOut[0];

            m_blobInput.Reshape(1, 1, nFields, nDepth);
            m_blobOutput.Reshape(1, 1, nOutputFields, nDepth);

            rgParam = new List<float>();
            rgParam.Add(m_blobInput.count());
            rgParam.Add(m_blobInput.mutable_gpu_data);
            rgParam.Add(m_blobInput.count());
            rgParam.Add(m_blobInput.mutable_gpu_diff);
            rgParam.Add(m_blobOutput.count());
            rgParam.Add(m_blobOutput.mutable_gpu_data);
            rgParam.Add(m_blobOutput.count());
            rgParam.Add(m_blobOutput.mutable_gpu_diff);
            m_extension.Run(Extension<T>.FUNCTION.SETMEMORY, rgParam.ToArray());
        }

        /// <summary>
        /// Reset the streaming database to the data start or an offset from the start.
        /// </summary>
        /// <param name="nStartOffset">Specifies the offset from the start to use.</param>
        public void Reset(int nStartOffset)
        {
            m_idb.Reset(nStartOffset);
        }

        /// <summary>
        /// Step to the next data in the streaming database and process it.
        /// </summary>
        /// <param name="bGetSimpleDatum">When <i>true</i>, specifies to create the SimpleDatum for data visualization.</param>
        /// <param name="nWait">Specifies the amount of time in ms. to wait for data.</param>
        /// <returns>A tuple containing the output data Blob and optionally a SimpleDatum for visualization is returned.</returns>
        public Tuple<Blob<T>, SimpleDatum> Step(bool bGetSimpleDatum, int nWait)
        {
            SimpleDatum sd = m_idb.Query(nWait);
            if (sd == null)
                return null;

            if (typeof(T) == typeof(float))
            {
                m_extension.Run(Extension<T>.FUNCTION.ADDDATA, sd.RealDataF);
            }
            else
            {
                m_extension.Run(Extension<T>.FUNCTION.ADDDATA, sd.RealDataD);
            }

            return new Tuple<Blob<T>, SimpleDatum>(m_blobOutput, null);
        }
    }
}
