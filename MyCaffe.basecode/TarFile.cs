using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.basecode
{
    /// <summary>
    /// The TarFile functions are used to expand tar files.
    /// </summary>
    /// <remarks>
    /// @see [Decompress tar files using C#](https://stackoverflow.com/questions/8863875/decompress-tar-files-using-c-sharp), StackOverflow, 2012
    /// @see [GitHub: ForeverZer0/ExtractTarGz](https://gist.github.com/ForeverZer0/a2cd292bd2f3b5e114956c00bb6e872b) ForeverZero0
    /// </remarks>
    public class TarFile
    {
        /// <summary>
        /// The constructor.
        /// </summary>
        public TarFile()
        {
        }

        /// <summary>
        /// Extract a Tar (*.tar) file to a specified output directory.
        /// </summary>
        /// <param name="strFileName">Specifies the name of the .tar file to extract.</param>
        /// <param name="strOutputDir">Specifies the output directory.</param>
        /// <param name="evtCancel">Optionally, specifies the cancel event used to cancel the extraction (default = null).</param>
        /// <param name="log">Optionally, specifies the Log used to output status of the extraction (default = null).</param>
        /// <param name="nExpectedTotal">Optionally, specifies the expected total number of files.</param>
        /// <returns>Upon a successful extraction, the number of files extracted offset by the index is returned, or 0 on abort.</returns>
        public static int ExtractTar(string strFileName, string strOutputDir, CancelEvent evtCancel = null, Log log = null, int nExpectedTotal = 0, int nIdx = 0)
        {
            using (FileStream fstrm = File.OpenRead(strFileName))
            {
                return ExtractTar(fstrm, strOutputDir, evtCancel, log, nExpectedTotal, nIdx);
            }
        }

        /// <summary>
        /// Extract Tar data from a stream to a specified output directory.
        /// </summary>
        /// <param name="stream">Specifies the stream containing the Tar data to extract.</param>
        /// <param name="strOutputDir">Specifies the output directory.</param>
        /// <param name="evtCancel">Optionally, specifies the cancel event used to cancel the extraction (default = null).</param>
        /// <param name="log">Optionally, specifies the Log used to output status of the extraction (default = null).</param>
        /// <param name="nExpectedTotal">Optionally, specifies the expected total number of files.</param>
        /// <returns>Upon a successful extraction, the number of files extracted offset by the index is returned, or 0 on abort.</returns>
        public static int ExtractTar(Stream stream, string strOutputDir, CancelEvent evtCancel = null, Log log = null, int nExpectedTotal = 0, int nIdx = 0)
        {
            byte[] rgBuffer = new byte[500];
            bool bDone = false;
            int nFileCount = 0;
            Stopwatch sw = new Stopwatch();

            sw.Start();

            try
            {
                while (!bDone)
                {
                    stream.Read(rgBuffer, 0, 100);
                    string strName = Encoding.ASCII.GetString(rgBuffer).Trim('\0');

                    if (string.IsNullOrWhiteSpace(strName))
                        break;

                    stream.Seek(24, SeekOrigin.Current);
                    stream.Read(rgBuffer, 0, 12);

                    long lSize = Convert.ToInt64(Encoding.UTF8.GetString(rgBuffer, 0, 12).Trim('\0').Trim(), 8);

                    stream.Seek(376L, SeekOrigin.Current);

                    string strOutput = Path.Combine(strOutputDir, strName);
                    string strPath = Path.GetDirectoryName(strOutput);

                    if (!Directory.Exists(strPath))
                        Directory.CreateDirectory(strPath);

                    if (!strName.EndsWith("/") && !strName.EndsWith("\\"))
                    {
                        if (sw.Elapsed.TotalMilliseconds > 1000)
                        {
                            if (log != null)
                            {
                                if (nExpectedTotal > 0)
                                    log.Progress = (double)(nIdx + nFileCount) / nExpectedTotal;

                                log.WriteLine("Extracting " + nFileCount.ToString("N0") + " files - '" + strName + "'...");
                            }

                            sw.Restart();
                        }

                        using (FileStream fstrm = File.Open(strOutput, FileMode.OpenOrCreate, FileAccess.Write))
                        {
                            byte[] rgData = new byte[lSize];
                            stream.Read(rgData, 0, rgData.Length);
                            fstrm.Write(rgData, 0, rgData.Length);
                            nFileCount++;
                        }
                    }

                    long lPos = stream.Position;
                    long lOffset = 512 - (lPos % 512);
                    if (lOffset == 512)
                        lOffset = 0;

                    stream.Seek(lOffset, SeekOrigin.Current);

                    if (evtCancel != null)
                    {
                        if (evtCancel.WaitOne(0))
                            return 0;
                    }
                }

                if (log != null)
                    log.WriteLine("Extracted a total of " + nFileCount.ToString("N0") + " files.");
            }
            catch (Exception excpt)
            {
                if (log != null)
                    log.WriteError(excpt);
                throw excpt;
            }

            return nFileCount + nIdx;
        }

        /// <summary>
        /// Extract a Gz zipped file to the output directory.
        /// </summary>
        /// <param name="strFileName">Specifize the .gz file to extract.</param>
        /// <param name="strOutputDir">Specifies the output directory.</param>
        public static void ExtractTarGz(string strFileName, string strOutputDir)
        {
            using (FileStream fstrm = File.OpenRead(strFileName))
            {
                ExtractTarGz(fstrm, strOutputDir);
            }
        }

        /// <summary>
        /// Extract a Gz stream to the output directory.
        /// </summary>
        /// <param name="stream">Specifies the Gz stream.</param>
        /// <param name="strOutputDir">Specifies the output directory.</param>
        public static void ExtractTarGz(Stream stream, string strOutputDir)
        {
            // A GZipStream is not seekable, so copy it first to a MemoryStream.
            using (Stream gzip = new GZipStream(stream, CompressionMode.Decompress))
            {
                const int nChunk = 4096;

                using (MemoryStream ms = new MemoryStream())
                {
                    byte[] rgBuffer = new byte[nChunk];
                    int nRead = gzip.Read(rgBuffer, 0, nChunk);

                    while (nRead > 0)
                    {
                        ms.Write(rgBuffer, 0, nRead);
                        nRead = gzip.Read(rgBuffer, 0, nChunk);
                    }

                    ms.Seek(0, SeekOrigin.Begin);
                    ExtractTar(ms, strOutputDir);
                }
            }
        }
    }
}
