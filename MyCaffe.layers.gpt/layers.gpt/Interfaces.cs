﻿using MyCaffe.basecode;
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
        int BOS { get; }
        /// <summary>
        /// Returns the special EOS character.
        /// </summary>
        int EOS { get; }
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
        /// <param name="bAddBos">Add the begin of sequence token.</param>
        /// <param name="bAddEos">Add the end of sequence token.</param>
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
        /// <param name="nStartIdx">Optionally, specifies a starting index (default = 0).</param>
        /// <param name="nCount">Optionally, specifies the number of items to process (default = -1, for all items).</param>
        /// <param name="nPadToken">Optionally, specifies a pad token that is ignored.</param>
        /// <returns>The detokenized string is returned.</returns>
        string Detokenize(float[] rgf, bool bIgnoreBos, bool bIgnoreEos, int nStartIdx = 0, int nCount = -1, int? nPadToken = null);
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
    /// The ICustomTokenInput interface specifies the interface that all custom token inputs implement.
    /// </summary>
    public interface ICustomTokenInput
    {
        /// <summary>
        /// Load all encoder tokens and their associated date/time.
        /// <paramref name="evtCancel">Specifies the cancel event.</paramref>
        /// <paramref name="log">Specifies the output log.</paramref>
        /// <paramref name="phase">Specifies the phase where the call is running.</paramref>
        /// <paramref name="nVocabSize">Specifies the source vocabulary size.</paramref>
        /// </summary>
        /// <returns>A tuple containing the encoder source, target and their date/time values is returned.</returns>
        List<Tuple<DateTime, int[], int[]>> LoadAllEncoderTokens(CancelEvent evtCancel, Log log, Phase phase, out int nVocabSize);
        /// <summary>
        /// Load all decoder tokens and their associated date/time.
        /// <paramref name="evtCancel">Specifies the cancel event.</paramref>
        /// <paramref name="log">Specifies the output log.</paramref>
        /// <paramref name="phase">Specifies the phase where the call is running.</paramref>
        /// <paramref name="nVocabSize">Specifies the target vocabulary size.</paramref>
        /// </summary>
        /// <returns>A tuple containing the decoder source, target and their date/time values is returned.</returns>
        List<Tuple<DateTime, int[], int[]>> LoadAllDecoderTokens(CancelEvent evtCancel, Log log, Phase phase, out int nVocabSize);
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
        /// Returns true if data is available at the given index.
        /// </summary>
        /// <param name="nIdx">Specifies the index to check</param>
        /// <param name="bIncludeSrc">Specifies to include the source in the check.</param>
        /// <param name="bIncludeTrg">Specifies to include the target in the check.</param>
        /// <returns>If the data is available, true is returned.</returns>
        public abstract bool GetDataAvailabilityAt(int nIdx, bool bIncludeSrc, bool bIncludeTrg);
        /// <summary>
        /// Gets a set of randomly selected source/target data, where the target may be null.
        /// </summary>
        /// <param name="nBatchSize">Specifies the number of blocks in the batch.</param>
        /// <param name="nBlockSize">Specifies the size of each block.</param>
        /// <param name="trgData">Specifies the target data used to see if data at index has data.</param>
        /// <param name="rgnIdx">Returns an array of the indexes of the data returned.</param>
        /// <returns>A tuple containing the data and target is returned.</returns>
        public abstract Tuple<float[], float[]> GetData(int nBatchSize, int nBlockSize, InputData trgData, out int[] rgnIdx);
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
        /// <param name="bAddBos">Add the begin of sequence token.</param>
        /// <param name="bAddEos">Add the end of sequence token.</param>
        /// <returns>A list of tokens corresponding to the input is returned.</returns>
        public abstract List<int> Tokenize(string str, bool bAddBos, bool bAddEos);
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
        public abstract int BOS { get; }
        /// <summary>
        /// Return the special end of sequence character.
        /// </summary>
        public abstract int EOS { get; }
    }
}
