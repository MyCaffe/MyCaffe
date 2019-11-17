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
    public class TestingActiveGpuGet
    {
        EventWaitHandle m_evtEnabled = null;

        public TestingActiveGpuGet()
        {
            EventWaitHandle.TryOpenExisting("__TestingActiveGpuEnabled__", out m_evtEnabled);
        }

        public void Initialize()
        {
            m_evtEnabled = new EventWaitHandle(false, EventResetMode.ManualReset, "__TestingActiveGpuEnabled__");
        }

        public int? GetActiveGpuID()
        {
            if (m_evtEnabled != null && !m_evtEnabled.WaitOne(0))
                return null;

            MemoryMappedFile mmf = null;
            MemoryMappedViewStream mmvStrm = null;

            try
            {
                mmf = MemoryMappedFile.CreateOrOpen("__TestActiveGpu__", 4);
                mmvStrm = mmf.CreateViewStream(0, 4);
                byte[] rgBuffer = new byte[4];

                mmvStrm.Read(rgBuffer, 0, 4);

                using (MemoryStream ms = new MemoryStream(rgBuffer))
                using (BinaryReader br = new BinaryReader(ms))
                {
                    return br.ReadInt32();
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
