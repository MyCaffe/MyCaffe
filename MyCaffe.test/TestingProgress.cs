using System;
using System.Collections.Generic;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.test
{
    public class TestingProgressSet : IDisposable
    {
        MemoryMappedFile m_mmf = null;
        MemoryMappedViewStream m_mmvStrm = null;

        public TestingProgressSet()
        {
            m_mmf = MemoryMappedFile.CreateOrOpen("__TestProgress__", 8);
            m_mmvStrm = m_mmf.CreateViewStream(0, 8);
        }

        public void Dispose()
        {
            if (m_mmvStrm != null)
            {
                m_mmvStrm.Dispose();
                m_mmvStrm = null;
            }

            if (m_mmf != null)
            {
                m_mmf.Dispose();
                m_mmf = null;
            }
        }

        public void SetProgress(double dfProgress)
        {
            try
            {
                byte[] rgBuffer = new byte[8];

                using (MemoryStream ms = new MemoryStream(rgBuffer))
                using (BinaryWriter bw = new BinaryWriter(ms))
                {
                    bw.Write(dfProgress);
                }

                m_mmvStrm.Write(rgBuffer, 0, 8);
                m_mmvStrm.Seek(0, SeekOrigin.Begin);
            }
            catch (Exception excpt)
            {
                throw excpt;
            }
        }
    }
}
