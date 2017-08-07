using MyCaffe.basecode;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MyCaffe.imagedb
{
    public class DeleteFiles /** @private */
    {
        Log m_log;
        CancelEvent m_evtCancel;

        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Auto, Pack = 1)]
        struct SHFILEOPSTRUCT  
        {
            public IntPtr hwnd;
            [MarshalAs(UnmanagedType.U4)]
            public int wFunc;
            public string pFrom;
            public string pTo;
            public short fFlags;
            [MarshalAs(UnmanagedType.Bool)]
            public bool fAnyOperationsAborted;
            public IntPtr hNameMappings;
            public string lpszProgressTitle;
        }

        [DllImport("shell32.dll", CharSet = CharSet.Auto)]
        static extern int SHFileOperation(ref SHFILEOPSTRUCT FileOp);

        const int FO_DELETE = 3;
        const int FOF_ALLOWUNDO = 0x40;
        const int FOF_NOCONFIRMATION = 0x10;

        public DeleteFiles(Log log, CancelEvent evtCancel)
        {
            m_log = log;
            m_evtCancel = evtCancel;
        }

        public bool DeleteDirectory(string strDir)
        {
            if (!Directory.Exists(strDir))
                return false;

            Stopwatch sw = new Stopwatch();
            string[] rgstrFiles = Directory.GetFiles(strDir);

            sw.Start();

            for (int i = 0; i < rgstrFiles.Length; i++)
            {
                SHFILEOPSTRUCT shf = new SHFILEOPSTRUCT();
                shf.wFunc = FO_DELETE;
                shf.fFlags = FOF_ALLOWUNDO + FOF_NOCONFIRMATION;
                shf.pFrom = strDir + '\0' + '\0';
                SHFileOperation(ref shf);

                if (m_evtCancel.WaitOne(0))
                    return false;

                if (sw.Elapsed.TotalMilliseconds > 1000)
                {
                    m_log.Progress = (double)i / (double)rgstrFiles.Length;
                    m_log.WriteLine("deleting " + i.ToString("N0") + " of " + rgstrFiles.Length.ToString("N0") + "...");
                    sw.Restart();
                }
            }

            rgstrFiles = Directory.GetFiles(strDir);
            if (rgstrFiles.Length == 0)
                Directory.Delete(strDir);

            return true;
        }
    }
}
