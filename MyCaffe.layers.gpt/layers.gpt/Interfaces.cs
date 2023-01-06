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
        /// Detokenize an array into a string.
        /// </summary>
        /// <param name="rgf">Specifies the array of tokens to detokenize.</param>
        /// <returns>The detokenized string is returned.</returns>
        string Detokenize(float[] rgf);
    }
}
