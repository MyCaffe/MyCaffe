using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.layers.gpt
{
    /// <summary>
    /// The VocabularyCharacters class manages the data vocabulary of characters.
    /// </summary>
    public class VocabularyCharacter : IVocabulary
    {
        bool m_bEnablePad = true;
        Random m_random;
        Dictionary<char, int> m_rgVocabKeyToIdx = new Dictionary<char, int>();
        Dictionary<int, char> m_rgVocabIdxToKey = new Dictionary<int, char>();
        bool m_bAddBos;
        bool m_bAddEos;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="random">Specifies the random number generator used.</param>
        /// <param name="bAddBos">Specifies to include the special BOS character in the vocabulary.</param>
        /// <param name="bAddEos">Specifies to include the special EOS character in the vocabulary.</param>
        /// <param name="bEnablePad">Specifies to enable the 0 based padding by adding the 0 pad key to the vocabulary.</param>
        public VocabularyCharacter(Random random, bool bAddBos, bool bAddEos, bool bEnablePad)
        {
            m_random = random;
            m_bAddBos = bAddBos;
            m_bAddEos = bAddEos;
            m_bEnablePad = bEnablePad;
            
            if (bAddBos)
                m_rgVocabKeyToIdx.Add(BOS, 1);

            if (bAddEos)
                m_rgVocabKeyToIdx.Add(EOS, 2);
        }

        /// <summary>
        /// Returns the size of the vocabulary.
        /// </summary>
        public int Count
        {
            get { return m_rgVocabKeyToIdx.Count + ((m_bEnablePad) ? 1 : 0); }
        }

        /// <summary>
        /// Adds a new character to the vocabulary.
        /// </summary>
        /// <param name="ch">Specifies the character</param>
        public void Add(char ch)
        {
            if (!m_rgVocabKeyToIdx.ContainsKey(ch))
                m_rgVocabKeyToIdx.Add(ch, 1);
        }

        /// <summary>
        /// Add a string of characters to the vocabulary.
        /// </summary>
        /// <param name="str">Specifies the string to add.</param>
        public void Add(string str)
        {
            foreach (char ch in str)
            {
                Add(ch);
            }
        }

        /// <summary>
        /// Builds the vocabulary from all characters added.
        /// </summary>
        /// <returns>The vocabulary size is returned.</returns>
        public int Build()
        {
            List<char> rgKeys = m_rgVocabKeyToIdx.Keys.ToList();
            rgKeys.Sort();

            m_rgVocabKeyToIdx.Clear();

            int nPadOffset = (m_bEnablePad) ? 1 : 0;
            
            // index 0 reserved for pad.
            for (int i = 0; i < rgKeys.Count; i++)
            {
                m_rgVocabKeyToIdx.Add(rgKeys[i], i + nPadOffset);
                m_rgVocabIdxToKey.Add(i + nPadOffset, rgKeys[i]);
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
            foreach (char ch in strData)
            {
                Add(ch);
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
        /// <param name="str1">Specifies a single element (character or word) to tokenize.</param>
        /// <param name="bMustExist">Optionally, specifies to throw an error if the item is not in the vocabulary (default = true).</param>
        /// <returns>A list of tokens corresponding to the character is returned (typically just a single token).</returns>
        public List<int> Tokenize(string str1, bool bMustExist = true)
        {
            if (str1.Length != 1)
                throw new Exception("The character must be a single character!");

            List<int> rgTokens = new List<int>();
            char ch = str1[0];

            if (!m_rgVocabKeyToIdx.ContainsKey(ch))
            {
                if (bMustExist)
                    throw new Exception("The character '" + ch.ToString() + " is not in the vocabulary!");
                else
                    rgTokens.Add(m_random.Next(Count));
            }

            rgTokens.Add(m_rgVocabKeyToIdx[ch]);
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

            foreach (char ch in str)
            {                
                rgTokens.AddRange(Tokenize(ch.ToString()));
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
            string str = "";

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
                if (nIdxToken == 0)
                    return str;

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
                {
                    char ch = str1[0];

                    if (ch == EOS)
                        break;

                    if (ch != 0 && ch != BOS && ch != EOS)
                        str += ch;
                }
            }

            return str;
        }
    }
}
