using MyCaffe.basecode;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MyCaffe.layers.gpt.custom_input
{
    public class CustomInput : ICustomTokenInput
    {
        const int PAD = 999997;
        const int BOS = 999998;
        const int EOS = 999999;
        Dictionary<int, int> m_rgVocab = new Dictionary<int, int>();

        enum TYPE
        {
            ENC,
            DEC
        }

        public CustomInput()
        {
        }

        public List<Tuple<DateTime, int[], int[]>> LoadAllEncoderTokens(CancelEvent evtCancel, Log log, Phase phase, out int nVocabSize)
        {
            string strProgData = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData);
            string strFile = (phase == Phase.TRAIN) ? strProgData + "\\MyCaffe\\test_data\\data\\text\\encdec\\en_fr\\src\\train.txt" : strProgData + "\\MyCaffe\\test_data\\data\\text\\encdec\\en_fr\\src\\valid.txt";
            return loadTokens(evtCancel, log, TYPE.ENC, strFile, out nVocabSize);
        }

        public List<Tuple<DateTime, int[], int[]>> LoadAllDecoderTokens(CancelEvent evtCancel, Log log, Phase phase, out int nVocabSize)
        {
            string strProgData = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData);
            string strFile = (phase == Phase.TRAIN) ? strProgData + "\\MyCaffe\\test_data\\data\\text\\encdec\\en_fr\\trg\\train.txt" : strProgData + "\\MyCaffe\\test_data\\data\\text\\encdec\\en_fr\\trg\\valid.txt";
            return loadTokens(evtCancel, log, TYPE.DEC, strFile, out nVocabSize);
        }

        private List<Tuple<DateTime, int[], int[]>> loadTokens(CancelEvent evtCancel, Log log, TYPE type, string strFile, out int nVocabSize)
        {
            Stopwatch sw = new Stopwatch();
            List<Tuple<DateTime, int[], int[]>> rgTokens = new List<Tuple<DateTime, int[], int[]>>();
            string[] rgstrLines = File.ReadAllLines(strFile);
            DateTime dt = DateTime.MinValue;

            log.WriteLine(strFile);
            sw.Start();

            // Collect the unique characters for tokenization by character.
            log.WriteLine("Building vocabulary...");
            for (int i = 0; i < rgstrLines.Length; i++)
            {
                string strLine = rgstrLines[i];
                int[] rgChar = new int[strLine.Length];

                for (int j = 0; j < strLine.Length; j++)
                {
                    int nChar = (int)strLine[j];
                    rgChar[j] = nChar;

                    if (!m_rgVocab.ContainsKey(nChar))
                        m_rgVocab.Add(nChar, 1);
                    else
                        m_rgVocab[nChar]++;
                }

                if (sw.Elapsed.TotalMilliseconds > 1000)
                {
                    nVocabSize = 0;
                    if (evtCancel != null && evtCancel.WaitOne(0))
                        return null;

                    double dfPct = (double)i / rgstrLines.Length;
                    log.WriteLine("Loading vocabulary at " + dfPct.ToString("P") + "...", true);

                    sw.Restart();
                }
            }

            // Order the vocabulary by most common characters.
            List<KeyValuePair<int, int>> rgTok = m_rgVocab.OrderByDescending(p => p.Value).ToList();
            m_rgVocab = new Dictionary<int, int>();

            // Build the vocabulary.
            m_rgVocab.Add(PAD, 0);    // token = 0
            m_rgVocab.Add(BOS, 1);    // token = 1
            m_rgVocab.Add(EOS, 2);    // token = 2

            foreach (KeyValuePair<int, int> kv in rgTok)
            {
                m_rgVocab.Add(kv.Key, m_rgVocab.Count);
            }

            // Tokenize the data
            log.WriteLine("Tokenizing the data...");
            for (int i = 0; i < rgstrLines.Length; i++)
            {
                string strLine = rgstrLines[i];
                int[] rgSrcTokenSet = new int[strLine.Length + 1];
                int[] rgTgtTokenSet = new int[strLine.Length + 1];
                int nSrcStart = 0;
                int nTgtStart = 0;

                if (type == TYPE.DEC)
                {
                    nSrcStart = 1;
                    rgSrcTokenSet[0] = (int)SPECIAL_TOKENS.BOS;
                }

                for (int j = 0; j < strLine.Length; j++)
                {
                    int nChar = (int)strLine[j];
                    int nToken = m_rgVocab[nChar];
                    rgSrcTokenSet[nSrcStart + j] = nToken;
                    if (type == TYPE.DEC)
                        rgTgtTokenSet[nTgtStart + j] = nToken;
                }

                if (type == TYPE.ENC)
                    rgSrcTokenSet[strLine.Length + nSrcStart] = (int)SPECIAL_TOKENS.EOS;
                else
                    rgTgtTokenSet[strLine.Length + nTgtStart] = (int)SPECIAL_TOKENS.EOS;

                rgTokens.Add(new Tuple<DateTime, int[], int[]>(dt, rgSrcTokenSet, rgTgtTokenSet));
                dt += TimeSpan.FromMinutes(1);

                if (sw.Elapsed.TotalMilliseconds > 1000)
                {
                    nVocabSize = 0;
                    if (evtCancel != null && evtCancel.WaitOne(0))
                        return null;

                    double dfPct = (double)i / rgstrLines.Length;
                    log.WriteLine("Loading tokens at " + dfPct.ToString("P") + "...", true);

                    sw.Restart();
                }
            }

            nVocabSize = m_rgVocab.Count;
            return rgTokens;
        }
    }
}
