using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.layers.gpt
{
    /// <summary>
    /// The IVocabulary interface specifies the interface that all Vocabularies implement.
    /// </summary>
    public interface IVocabulary
    {
        /// <summary>
        /// Returns the size of the vocabulary.
        /// </summary>
        int Count { get; }
        /// <summary>
        /// Returns the special BOS character.
        /// </summary>
        char BOS { get; }
        /// <summary>
        /// Returns the special EOS character.
        /// </summary>
        char EOS { get; }
        /// <summary>
        /// Add a new string to the vocabulary.
        /// </summary>
        /// <param name="str">Specifies the string to add.</param>
        void Add(string str);
        /// <summary>
        /// Build the vocabulary.
        /// </summary>
        /// <returns>
        /// The vocabulary size is returned.
        /// </returns>
        int Build();
        /// <summary>
        /// Build the vocabulary from a string.
        /// </summary>
        /// <param name="strData">Specifies the data to build the vocabulary from.</param>
        /// <returns>The vocabulary size is returned.</returns>
        int BuildFromString(string strData);
        /// <summary>
        /// Create a target that is offset from the source by one and ends with a EOS.
        /// </summary>
        /// <param name="rgSrc">Specifies the source to create the target from.</param>
        /// <returns>The tokenized target is returned.</returns>
        int[] CreateTarget(int[] rgSrc);
        /// <summary>
        /// Tokenize a string of data.
        /// </summary>
        /// <param name="str">Specifies the string to tokenize.</param>
        /// <param name="bAddBos">Specifies to add the BOS at the start of the tokenized data.</param>
        /// <param name="bAddEos">Specifies to add the EOS to the end of the tokenized data.</param>
        /// <returns>The array of tokens is returned.</returns>
        int[] Tokenize(string str, bool bAddBos, bool bAddEos);
        /// <summary>
        /// Tokenize a character into its corresponding index token.
        /// </summary>
        /// <param name="str1">Specifies a single element (character or word) to tokenize.</param>
        /// <param name="bMustExist">Optionally, specifies to throw an error if the item is not in the vocabulary (default = true).</param>
        /// <returns>A list of tokens corresponding to the input is returned.</returns>
        List<int> Tokenize(string str1, bool bMustExist = true);
        /// <summary>
        /// Detokenize an array into a string.
        /// </summary>
        /// <param name="rgf">Specifies the array of tokens to detokenize.</param>
        /// <param name="bIgnoreBos">Specifies to ignore the BOS token.</param>
        /// <param name="bIgnoreEos">Specifies to ignore the EOS token.</param>
        /// <returns>The detokenized string is returned.</returns>
        string Detokenize(float[] rgf, bool bIgnoreBos, bool bIgnoreEos);
        /// <summary>
        /// Detokenize an index token into its corresponding character.
        /// </summary>
        /// <param name="nIdxToken">Specifies the token to detokenize.</param>
        /// <param name="bIgnoreBos">Specifies to ignore the BOS token.</param>
        /// <param name="bIgnoreEos">Specifies to ignore the EOS token.</param>
        /// <returns>The detokenized string is returned (which may just be a character).</returns>
        string Detokenize(int nIdxToken, bool bIgnoreBos, bool bIgnoreEos);
    }

    /// <summary>
    /// The InputData is an abstract class used to get training data and tokenize input data.
    /// </summary>
    public abstract class InputData
    {
        /// <summary>
        /// Specifies the random object made available to the derived classes.
        /// </summary>
        protected Random m_random;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="nRandomSeed">Optionally, specifies the seed to use for testing.</param>
        public InputData(int? nRandomSeed = null)
        {
            if (nRandomSeed.HasValue)
                m_random = new Random(nRandomSeed.Value);
            else
                m_random = new Random();
        }

        /// <summary>
        /// Returns the raw data.
        /// </summary>
        public abstract List<string> RawData { get; }
        /// <summary>
        /// Returns the size of a single token (e.g. 1 for character data)
        /// </summary>
        public abstract uint TokenSize { get; }
        /// <summary>
        /// Returns the size of the vocabulary.
        /// </summary>
        public abstract uint VocabularySize { get; }
        /// <summary>
        /// Gets a set of randomly selected source/target data, where the target may be null.
        /// </summary>
        /// <param name="nBatchSize">Specifies the number of blocks in the batch.</param>
        /// <param name="nBlockSize">Specifies the size of each block.</param>
        /// <param name="rgnIdx">Returns an array of the indexes of the data returned.</param>
        /// <returns>A tuple containing the data and target is returned.</returns>
        public abstract Tuple<float[], float[]> GetData(int nBatchSize, int nBlockSize, out int[] rgnIdx);
        /// <summary>
        /// Gets a set of source/target data from a specific index.
        /// </summary>
        /// <param name="nBatchSize">Specifies the number of blocks in the batch.</param>
        /// <param name="nBlockSize">Specifies the size of each block.</param>
        /// <param name="rgnIdx">Specifies the array of indexes of data to retrieve.</param>
        /// <returns>A tuple containing the data and target is returned.</returns>
        public abstract Tuple<float[], float[]> GetDataAt(int nBatchSize, int nBlockSize, int[] rgnIdx);
        /// <summary>
        /// Tokenize an input string using the internal vocabulary.
        /// </summary>
        /// <param name="str">Specifies the string to tokenize.</param>
        /// <returns>A list of tokens corresponding to the input is returned.</returns>
        public abstract List<int> Tokenize(string str);
        /// <summary>
        /// Detokenize a single token.
        /// </summary>
        /// <param name="nTokIdx">Specifies an index to the token to be detokenized.</param>
        /// <param name="bIgnoreBos">Specifies to ignore the BOS token.</param>
        /// <param name="bIgnoreEos">Specifies to ignore the EOS token.</param>
        /// <returns>The detokenized item is returned.</returns>
        public abstract string Detokenize(int nTokIdx, bool bIgnoreBos, bool bIgnoreEos);
        /// <summary>
        /// Detokenize an array into a string.
        /// </summary>
        /// <param name="rgf">Specifies the array of tokens to detokenize.</param>
        /// <param name="nStartIdx">Specifies the starting index where detokenizing begins.</param>
        /// <param name="nCount">Specifies the number of tokens to detokenize.</param>
        /// <param name="bIgnoreBos">Specifies to ignore the BOS token.</param>
        /// <param name="bIgnoreEos">Specifies to ignore the EOS token.</param>
        /// <returns>The detokenized string is returned.</returns>
        public abstract string Detokenize(float[] rgf, int nStartIdx, int nCount, bool bIgnoreBos, bool bIgnoreEos);
        /// <summary>
        /// Return the special begin of sequence character.
        /// </summary>
        public abstract char BOS { get; }
        /// <summary>
        /// Return the special end of sequence character.
        /// </summary>
        public abstract char EOS { get; }
    }
}
