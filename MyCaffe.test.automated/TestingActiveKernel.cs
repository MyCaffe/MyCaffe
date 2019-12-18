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
    public class TestingActiveKernelHandleGet
    {
        EventWaitHandle m_evtEnabled = null;

        public TestingActiveKernelHandleGet()
        {
            EventWaitHandle.TryOpenExisting("__TestingActiveKernelHandleEnabled__", out m_evtEnabled);
        }

        public void Initialize()
        {
            m_evtEnabled = new EventWaitHandle(false, EventResetMode.ManualReset, "__TestingActiveKernelHandleEnabled__");
        }

        public long? GetActiveKernelHandle()
        {
            if (m_evtEnabled != null && !m_evtEnabled.WaitOne(0))
                return null;

            MemoryMappedFile mmf = null;
            MemoryMappedViewStream mmvStrm = null;

            try
            {
                mmf = MemoryMappedFile.CreateOrOpen("__TestActiveKernelHandle__", 4);
                mmvStrm = mmf.CreateViewStream(0, 8);
                byte[] rgBuffer = new byte[8];

                mmvStrm.Read(rgBuffer, 0, 8);

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
                if (mmvStrm != null)
                    mmvStrm.Dispose();

                if (mmf != null)
                    mmf.Dispose();
            }
        }
    }
}
