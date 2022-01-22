using System;
using System.Collections.Generic;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MyCaffe.test.automated
{
    public class TestingActiveKernelHandleGet : IDisposable
    {
        EventWaitHandle m_evtEnabled = null;
        MemoryMappedFile m_mmf = null;
        MemoryMappedViewStream m_mmvStrm = null;

        public TestingActiveKernelHandleGet()
        {
            EventWaitHandle.TryOpenExisting("__TestingActiveKernelHandleEnabled__", out m_evtEnabled);
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

        public void Initialize()
        {
            m_evtEnabled = new EventWaitHandle(false, EventResetMode.ManualReset, "__TestingActiveKernelHandleEnabled__");
        }

        public long? GetActiveKernelHandle()
        {
            if (m_evtEnabled != null && !m_evtEnabled.WaitOne(0))
                return null;

            try
            {
                if (m_mmf == null)
                {
                    m_mmf = MemoryMappedFile.CreateOrOpen("__TestActiveKernelHandle__", 4);
                    m_mmvStrm = m_mmf.CreateViewStream(0, 4);
                }

                byte[] rgBuffer = new byte[8];

                m_mmvStrm.Read(rgBuffer, 0, 8);

                using (MemoryStream ms = new MemoryStream(rgBuffer))
                using (BinaryReader br = new BinaryReader(ms))
                {
                    return br.ReadInt64();
                }
            }
            catch (Exception)
            {
                if (m_evtEnabled != null)
                    m_evtEnabled.Reset();

                return null;
            }
            finally
            {
            }
        }
    }
}
