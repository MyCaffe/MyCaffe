using System;
using System.Collections.Generic;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.test.automated
{
    public class TestingProgressGet
    {
        public TestingProgressGet()
        {
        }

        public static double? GetProgress()
        {
            MemoryMappedFile mmf = null;
            MemoryMappedViewStream mmvStrm = null;

            try
            {
                mmf = MemoryMappedFile.OpenExisting("__TestProgress__");
                mmvStrm = mmf.CreateViewStream(0, 8);
                byte[] rgBuffer = new byte[8];

                mmvStrm.Read(rgBuffer, 0, 8);

                using (MemoryStream ms = new MemoryStream(rgBuffer))
                using (BinaryReader br = new BinaryReader(ms))
                {
                    return br.ReadDouble();
                }
            }
            catch (Exception)
            {
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
