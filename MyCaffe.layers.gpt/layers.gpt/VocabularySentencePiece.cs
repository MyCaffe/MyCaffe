using MyCaffe.basecode;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.layers.gpt
{
    /// <summary>
    /// The VocabularySentencePieces class manages the data vocabulary of words.
    /// </summary>
    public class VocabularySentencePiece : IVocabulary
    {
        Random m_random;
        Dictionary<string, double> m_rgPieces = new Dictionary<string, double>();
        Dictionary<string, int> m_rgVocabKeyToIdx = new Dictionary<string, int>();
        Dictionary<int, string> m_rgVocabIdxToKey = new Dictionary<int, string>();
        bool m_bAddBos;
        bool m_bAddEos;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="random">Specifies the random number generator used.</param>
        /// <param name="bAddBos">Specifies to include the special BOS character in the vocabulary.</param>
        /// <param name="bAddEos">Specifies to include the special EOS character in the vocabulary.</param>
        /// <param name="strVocabFile">Specifies the vocabulary file created using the Python, SentencePieceProcess.</param>
        public VocabularySentencePiece(Random random, bool bAddBos, bool bAddEos, string strVocabFile)
        {
            string strProgData = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData);
            strVocabFile = Utility.ReplaceMacro(strVocabFile, "$ProgramData$", strProgData);
            
            string[] rgstrLines = File.ReadAllLines(strVocabFile);

            foreach (string strLine in rgstrLines)
            {
                string[] rgstr = strLine.Split(' ', '\t');
                if (rgstr.Length == 2)
                {
                    double dfVal;
                    if (double.TryParse(rgstr[1], out dfVal) && dfVal != 0)
                    {
                        string strKey = rgstr[0].Trim('_', (char)9601);

                        if (!m_rgPieces.ContainsKey(strKey))
                            m_rgPieces.Add(strKey, dfVal);
                    }
                }
            }

            m_random = random;
            m_bAddBos = bAddBos;
            m_bAddEos = bAddEos;
        }

        /// <summary>
        /// Returns the size of the vocabulary.
        /// </summary>
        public int Count
        {
            get { return m_rgVocabKeyToIdx.Count; }
        }

        private bool isSymbol(char ch)
        {
            if (char.IsDigit(ch))
                return true;

            if (char.IsPunctuation(ch))
                return true;

            if (char.IsSymbol(ch))
                return true;

            System.Globalization.UnicodeCategory cat = char.GetUnicodeCategory(ch);
            if (cat == System.Globalization.UnicodeCategory.OtherPunctuation ||
                cat == System.Globalization.UnicodeCategory.OtherSymbol ||
                cat == System.Globalization.UnicodeCategory.DecimalDigitNumber)
                return true;

            return false;
        }

        private string trim(string str)
        {
            string strOut = "";

            foreach (char ch in str)
            {
                System.Globalization.UnicodeCategory cat = char.GetUnicodeCategory(ch);

                if (!char.IsWhiteSpace(ch) && cat != System.Globalization.UnicodeCategory.SpaceSeparator)
                    strOut += ch;
            }

            return strOut;
        }

        /// <summary>
        /// Adds a new character to the vocabulary.
        /// </summary>
        /// <param name="str">Specifies the sentence or word to add.</param>
        public void Add(string str)
        {
            string[] rgstr = str.Split(' ');

            foreach (string strWord in rgstr)
            {
                string strWordA = strWord;
                string strWord1 = strWordA;

                while (strWord1.Length > 0)
                {
                    if (m_rgPieces.ContainsKey(strWord1))
                    {
                        if (!m_rgVocabKeyToIdx.ContainsKey(strWord1))
                            m_rgVocabKeyToIdx.Add(strWord1, 1);

                        strWord1 = strWordA.Substring(strWord1.Length);
                        strWordA = strWord1;
                    }
                    else
                    {
                        if (strWord1.Length > 0)
                            strWord1 = strWord1.Substring(0, strWord1.Length - 1);
                    }
                }
            }
        }

        /// <summary>
        /// Builds the vocabulary from all words added.
        /// </summary>
        /// <returns>The vocabulary size is returned.</returns>
        public int Build()
        {
            List<string> rgKeys = m_rgVocabKeyToIdx.Keys.ToList();
            rgKeys.Sort();

            m_rgVocabKeyToIdx.Clear();

            if (m_bAddEos)
                rgKeys.Insert(0, EOS.ToString());
            if (m_bAddBos)
                rgKeys.Insert(0, BOS.ToString());
            rgKeys.Insert(0, ((char)0).ToString());

            // index 0 reserved for pad.
            for (int i = 0; i < rgKeys.Count; i++)
            {
                if (i <= 2 || (rgKeys[i][0] != 0 && rgKeys[i][0] != BOS && rgKeys[i][0] != EOS))
                {
                    m_rgVocabKeyToIdx.Add(rgKeys[i], i);
                    m_rgVocabIdxToKey.Add(i, rgKeys[i]);
                }
            }

            return Count;
        }

        /// <summary>
        /// Build the vocabulary from a string.
        /// </summary>
        /// <param name="strData">Specifies the data to build the vocabulary from.</param>
        /// <returns>The vocabulary size is returned.</returns>
        public int BuildFromString(string strData)
        {
            string[] rgstrWords = strData.Split(' ');
            foreach (string strWord in rgstrWords)
            {
                Add(strWord);
            }

            return Build();
        }

        /// <summary>
        /// Returns the special BOS character.
        /// </summary>
        public char BOS
        {
            get { return (char)1; }
        }

        /// <summary>
        /// Returns the special EOS character.
        /// </summary>
        public char EOS
        {
            get { return (char)2; }
        }

        /// <summary>
        /// Create a target that is offset from the source by one and ends with a EOS.
        /// </summary>
        /// <param name="rgSrc">Specifies the source to create the target from.</param>
        /// <returns>The tokenized target is returned.</returns>
        public int[] CreateTarget(int[] rgSrc)
        {          
            List<int> rgTrg = new List<int>(rgSrc);

            if (rgSrc.Length > 0)
            {
                rgTrg.RemoveAt(0);
                rgTrg.Add(EOS);
            }

            return rgTrg.ToArray();
        }

        /// <summary>
        /// Tokenize a character into its corresponding index token.
        /// </summary>
        /// <param name="strWord">Specifies a single word to tokenize.</param>
        /// <param name="bMustExist">Optionally, specifies to throw an error if the item is not in the vocabulary (default = true).</param>
        /// <returns>The token corresponding to the character is returned.</returns>
        public List<int> Tokenize(string strWord, bool bMustExist = true)
        {
            List<int> rgTokens = new List<int>();
            string strWordA = strWord;
            string strWord1 = strWordA;

            while (strWord1.Length > 0)
            {
                if (m_rgPieces.ContainsKey(strWord1))
                {
                    if (m_rgVocabKeyToIdx.ContainsKey(strWord1))
                        rgTokens.Add(m_rgVocabKeyToIdx[strWord1]);

                    strWord1 = strWordA.Substring(strWord1.Length);
                    strWordA = strWord1;
                }
                else
                {
                    if (strWord1.Length > 0)
                        strWord1 = strWord1.Substring(0, strWord1.Length - 1);
                }
            }

            //if (rgTokens.Count == 0)
            //    Trace.WriteLine("No tokens found!");
            
            return rgTokens;
        }

        /// <summary>
        /// Tokenize a string of data.
        /// </summary>
        /// <param name="str">Specifies the string to tokenize.</param>
        /// <param name="bAddBos">Specifies to add the BOS at the start of the tokenized data.</param>
        /// <param name="bAddEos">Specifies to add the EOS to the end of the tokenized data.</param>
        /// <returns>The array of tokens is returned.</returns>
        public int[] Tokenize(string str, bool bAddBos, bool bAddEos)
        {
            List<int> rgTokens = new List<int>();

            if (string.IsNullOrEmpty(str))
                return rgTokens.ToArray();

            string[] rgstr = str.Split(' ');
            foreach (string strWord in rgstr)
            {                
                rgTokens.AddRange(Tokenize(strWord));
            }

            if (bAddBos)
                rgTokens.Insert(0, BOS);

            if (bAddEos)
                rgTokens.Add(EOS);

            return rgTokens.ToArray();
        }

        /// <summary>
        /// Detokenize an index token into its corresponding character.
        /// </summary>
        /// <param name="nIdxToken">Specifies the token to detokenize.</param>
        /// <param name="bIgnoreBos">Specifies to ignore the BOS token.</param>
        /// <param name="bIgnoreEos">Specifies to ignore the EOS token.</param>
        /// <returns>The detokenized string is returned (which may just be a character).</returns>
        public string Detokenize(int nIdxToken, bool bIgnoreBos, bool bIgnoreEos)
        {
            string str = null;
            
            if (nIdxToken == 0)
                return str;

            str = "";

            if (m_bAddBos && nIdxToken == BOS)
            {
                if (!bIgnoreBos)
                    str += "<BOS>";
            }

            else if (m_bAddEos && nIdxToken == EOS)
            {
                if (!bIgnoreEos)
                    str += "<EOS>";
            }

            else
            {
                if (!m_rgVocabIdxToKey.ContainsKey(nIdxToken))
                    throw new Exception("The token '" + nIdxToken.ToString() + "' is not in the vocabulary!");

                str += m_rgVocabIdxToKey[nIdxToken];
            }
            
            return str;
        }

        /// <summary>
        /// Detokenize an array into a string.
        /// </summary>
        /// <param name="rgf">Specifies the array of tokens to detokenize.</param>
        /// <param name="bIgnoreBos">Specifies to ignore the BOS token.</param>
        /// <param name="bIgnoreEos">Specifies to ignore the EOS token.</param>
        /// <returns>The detokenized string is returned.</returns>
        public string Detokenize(float[] rgf, bool bIgnoreBos, bool bIgnoreEos)
        {
            string str = "";

            foreach (float f in rgf)
            {
                string str1 = Detokenize((int)f, bIgnoreBos, bIgnoreEos);

                if (!string.IsNullOrEmpty(str1))
                    str += str1 + " ";
            }

            return str.TrimEnd(' ');
        }
    }
}
