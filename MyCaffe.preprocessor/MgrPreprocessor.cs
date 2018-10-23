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
    public class MgrPreprocessor<T> : IDisposable
    {
        Extension<T> m_extension;
        MyCaffeControl<T> m_mycaffe;
        IXStreamDatabase m_idb;
        Blob<T> m_blobInput = null;
        Blob<T> m_blobOutput = null;

        public MgrPreprocessor(IXMyCaffe<T> imycaffe, IXStreamDatabase idb)
        {
            m_mycaffe = (MyCaffeControl<T>)imycaffe;
            m_idb = idb;
            m_extension = new Extension<T>(imycaffe as IXMyCaffeExtension<T>);
            m_blobInput = new Blob<T>(m_mycaffe.Cuda, m_mycaffe.Log);
            m_blobOutput = new Blob<T>(m_mycaffe.Cuda, m_mycaffe.Log);
        }

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

        public void Initialize(string strExtPath, int nFields, int nDepth)
        {
            List<float> rgParam;

            m_extension.Initialize(strExtPath);

            rgParam = new List<float>();
            rgParam.Add(nFields);
            rgParam.Add(nDepth);
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

        public Blob<T> Step()
        {
            SimpleDatum sd = m_idb.Query();
            if (sd == null)
                return null;

            if (typeof(T) == typeof(float))
            {
                float[] rgParam = sd.RealData.Select(p => (float)p).ToArray();
                m_extension.Run(Extension<T>.FUNCTION.ADDDATA, rgParam);
            }
            else
            {
                m_extension.Run(Extension<T>.FUNCTION.ADDDATA, sd.RealData);
            }

            return m_blobOutput;
        }
    }
}
