﻿using System;
using System.Collections.Generic;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MyCaffe.test
{
    public class TestingActiveKernelHandleSet : IDisposable
    {
        EventWaitHandle m_evtEnabled = null;
        MemoryMappedFile m_mmf = null;
        MemoryMappedViewStream m_mmvStrm = null;

        public TestingActiveKernelHandleSet()
        {
            m_mmf = MemoryMappedFile.CreateOrOpen("__TestActiveKernelHandle__", 8);
            m_mmvStrm = m_mmf.CreateViewStream(0, 8);

            if (EventWaitHandle.TryOpenExisting("__TestingActiveKernelHandleEnabled__", out m_evtEnabled))
                m_evtEnabled.Set();
        }

        public void Dispose()
        {
            if (m_evtEnabled != null)
            {
                m_evtEnabled.Reset();
                m_evtEnabled.Dispose();
            }

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

        public void SetActiveKernelHandle(long lKernelHandle)
        {
            try
            {
                byte[] rgBuffer = new byte[8];

                using (MemoryStream ms = new MemoryStream(rgBuffer))
                using (BinaryWriter bw = new BinaryWriter(ms))
                {
                    bw.Write(lKernelHandle);
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
