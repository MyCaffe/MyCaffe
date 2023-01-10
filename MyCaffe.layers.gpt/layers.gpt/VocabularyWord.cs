using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.layers.gpt
{
    /// <summary>
    /// The VocabularyWords class manages the data vocabulary of words.
    /// </summary>
    public class VocabularyWord : IVocabulary
    {
        Random m_random;
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
        public VocabularyWord(Random random, bool bAddBos, bool bAddEos)
        {
            m_random = random;
            m_bAddBos = bAddBos;
            m_bAddEos = bAddEos;
            
            if (bAddBos)
                m_rgVocabKeyToIdx.Add(BOS.ToString(), 1);

            if (bAddEos)
                m_rgVocabKeyToIdx.Add(EOS.ToString(), 2);
        }

        /// <summary>
        /// Returns the size of the vocabulary.
        /// </summary>
        public int Count
        {
            get { return m_rgVocabKeyToIdx.Count + 1; }
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
                if (!string.IsNullOrEmpty(strWord))
                {
                    string strWord1 = trim(strWord.ToLower().Trim('\'', '\"'));                    
                    if (string.IsNullOrEmpty(strWord1))
                        continue;

                    while (strWord1.Length > 0 && isSymbol(strWord1[strWord1.Length-1]) && strWord1[strWord1.Length-1] != ' ')
                    {
                        string strLast = strWord1[strWord1.Length - 1].ToString();
                        if (!m_rgVocabKeyToIdx.ContainsKey(strLast))
                            m_rgVocabKeyToIdx.Add(strLast, 1);

                        strWord1 = strWord1.Substring(0, strWord1.Length - 1);
                    }

                    strWord1 = trim(strWord1);
                    if (string.IsNullOrEmpty(strWord1))
                        continue;

                    while (strWord1.Length > 0 && isSymbol(strWord1[0]) && strWord1[0] != ' ')
                    {
                        string strFirst = strWord1[0].ToString();
                        if (!m_rgVocabKeyToIdx.ContainsKey(strFirst))
                            m_rgVocabKeyToIdx.Add(strFirst, 1);

                        strWord1 = strWord1.Substring(1);
                    }

                    strWord1 = trim(strWord1);
                    if (string.IsNullOrEmpty(strWord1))
                        continue;

                    if (!m_rgVocabKeyToIdx.ContainsKey(strWord1))
                        m_rgVocabKeyToIdx.Add(strWord1, 1);
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

            // index 0 reserved for pad.
            for (int i = 0; i < rgKeys.Count; i++)
            {
                m_rgVocabKeyToIdx.Add(rgKeys[i], i + 1);
                m_rgVocabIdxToKey.Add(i + 1, rgKeys[i]);
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

            rgTrg.RemoveAt(0);
            rgTrg.Add(EOS);

            return rgTrg.ToArray();
        }

        /// <summary>
        /// Tokenize a character into its corresponding index token.
        /// </summary>
        /// <param name="strWord">Specifies a single word to tokenize.</param>
        /// <param name="bMustExist">Optionally, specifies to throw an error if the item is not in the vocabulary (default = true).</param>
        /// <returns>A list of tokens corresponding to the input is returned.</returns>
        public List<int> Tokenize(string strWord, bool bMustExist = true)
        {
            List<int> rgTokens = new List<int>();

            if (!string.IsNullOrEmpty(strWord))
            {
                string strWord1 = trim(strWord.ToLower().Trim('\'', '\"'));
                if (string.IsNullOrEmpty(strWord1))
                    return rgTokens;

                while (strWord1.Length > 0 && isSymbol(strWord1[strWord1.Length - 1]) && strWord1[strWord1.Length - 1] != ' ')
                {
                    string strLast = strWord1[strWord1.Length - 1].ToString();
                    if (m_rgVocabKeyToIdx.ContainsKey(strLast))
                        rgTokens.Add(m_rgVocabKeyToIdx[strLast]);

                    strWord1 = strWord1.Substring(0, strWord1.Length - 1);
                }

                strWord1 = trim(strWord1);
                if (string.IsNullOrEmpty(strWord1))
                    return rgTokens;

                while (strWord1.Length > 0 && isSymbol(strWord1[0]) && strWord1[0] != ' ')
                {
                    string strFirst = strWord1[0].ToString();
                    if (m_rgVocabKeyToIdx.ContainsKey(strFirst))
                        rgTokens.Add(m_rgVocabKeyToIdx[strFirst]);

                    strWord1 = strWord1.Substring(1);
                }

                strWord1 = trim(strWord1);
                if (string.IsNullOrEmpty(strWord1))
                    return rgTokens;

                if (m_rgVocabKeyToIdx.ContainsKey(strWord1))
                    rgTokens.Add(m_rgVocabKeyToIdx[strWord1]);
            }

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
                    str += str1;
            }

            return str;
        }
    }
}
