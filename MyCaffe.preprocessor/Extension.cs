using MyCaffe.common;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.preprocessor
{
    public class Extension<T> : IDisposable
    {
        IXMyCaffeExtension<T> m_iextension;
        long m_hExtension = 0;

        public enum FUNCTION
        {
            INITIALIZE = 1,
            CLEANUP = 2,
            SETMEMORY = 3,
            ADDDATA = 4,
            PROCESSDATA = 5,
            GETVISUALIZATION = 6,
            CLEAR = 7
        }

        public Extension(IXMyCaffeExtension<T> iextension)
        {
            m_iextension = iextension;
        }

        public void Dispose()
        {
            if (m_hExtension != 0)
            {
                Run(FUNCTION.CLEANUP);
                m_iextension.FreeExtension(m_hExtension);
                m_hExtension = 0;
            }
        }

        public void Initialize(string strPath)
        {
            if (m_hExtension != 0)
                m_iextension.FreeExtension(m_hExtension);

            m_hExtension = m_iextension.CreateExtension(strPath);
        }

        public void Run(FUNCTION fn)
        {
            m_iextension.RunExtension(m_hExtension, (long)fn, null);
        }

        public double[] Run(FUNCTION fn, double[] rgParam)
        {
            return m_iextension.RunExtensionD(m_hExtension, (long)fn, rgParam);
        }

        public float[] Run(FUNCTION fn, float[] rgParam)
        {
            return m_iextension.RunExtensionF(m_hExtension, (long)fn, rgParam);
        }
    }
}
