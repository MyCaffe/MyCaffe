using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Drawing;
using System.Diagnostics;
using MyCaffe.basecode;
using MyCaffe.db.image;
using MyCaffe.common;
using MyCaffe.param;
using System.IO;
using System.Reflection;
using MyCaffe.output_adapters;
using System.ComponentModel;
using System.Diagnostics.SymbolStore;

/// <summary>
/// The MyCaffe.layers namespace contains all layers that have a solidified code base, including the Layer class.
/// </summary>
namespace MyCaffe.layers
{
    /// <summary>
    /// An base class for all tokenized data layers.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class BaseTokenizedDataLayer<T> : Layer<T>
    {
        /// <summary>
        /// The constructor.
        /// </summary>
        /// <remarks>
        /// Setup code for derivative classes should go into an override of the LayerSetup function where the 
        /// dimensionsn of the Blob%s are provided to the Layer.
        /// </remarks>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter that contains the settings of the Layer.</param>
        public BaseTokenizedDataLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
        }

        /// <summary>
        /// Should return true when PreProcessing methods are overriden.
        /// </summary>
        public virtual bool SupportsPreProcessing
        {
            get { return false; }
        }

        /// <summary>
        /// Should return true when pre PostProcessing methods are overriden.
        /// </summary>
        public virtual bool SupportsPostProcessing
        {
            get { return false; }
        }

        /// <summary>
        /// Should return true when pre PostProcessingLogits methods are overriden.
        /// </summary>
        public virtual bool SupportsPostProcessingLogits
        {
            get { return false; }
        }

        /// <summary>
        /// Should return true when PostProcessingFullOutput is supported.
        /// </summary>
        public virtual bool SupportsPostProcessingFullOutput
        {
            get { return false; }
        }

        /// <summary>
        /// The PreprocessInput allows derivative data layers to convert a property set of input
        /// data into the bottom blob collection used as intput.
        /// </summary>
        /// <param name="customInput">Specifies the custom input data.</param>
        /// <param name="nSeqLen">Returns the sequence length.</param>
        /// <param name="colBottom">Optionally, specifies the bottom data to fill.</param>
        /// <returns>The bottom data is returned.</returns>
        /// <remarks>The blobs returned should match the blob descriptions returned in the LayerParameter's
        /// overrides for 'PrepareRunModelInputs' and 'PrepareRunModel'.</remarks>
        public virtual BlobCollection<T> PreProcessInput(PropertySet customInput, out int nSeqLen, BlobCollection<T> colBottom = null)
        {
            nSeqLen = 0;
            return colBottom;
        }

        /// <summary>
        /// Preprocess the input data for the RUN phase.
        /// </summary>
        /// <param name="strEncInput">Specifies the encoder input.</param>
        /// <param name="nDecInput">Specifies the decoder input.</param>
        /// <param name="colBottom">Specifies the bottom blob where the preprocessed data is placed where
        /// colBottom[0] contains the preprocessed decoder input.
        /// colBottom[1] contains the preprocessed encoder input (depending on param settings),
        /// colBottom[2] contains the preprocessed encoder input reversed (depending on param settings)
        /// </param>
        /// <returns>
        /// If nDecInput == EOS, false is returned, otherwise true.
        /// </returns>
        /// <remarks>
        /// NOTE: the LayerSetup must be called before preprocessing input, for during LayerSetup the vocabulary is loaded.
        /// </remarks>
        public virtual bool PreProcessInput(string strEncInput, int? nDecInput, BlobCollection<T> colBottom)
        {
            return false;
        }

        /// <summary>
        /// The PostProcessOutput allows derivative data layers to post-process the results,
        /// converting them back into text results (e.g., detokenizing).
        /// </summary>
        /// <param name="blobSofmtax">Specifies the softmax blob output by the network.</param>
        /// <param name="nK">Optionally, specifies the K top items to return (default = 1).</param>
        /// <returns>The array of word string, index, propabilities and end of squence found boolean corresponding to the softmax output is returned.</returns>
        public virtual List<Tuple<string, int, double>> PostProcessOutput(Blob<T> blobSofmtax, int nK = 1)
        {
            return null;
        }
        /// <summary>
        /// The PostProcessLogitsOutput allows derivative data layers to post-process the results,
        /// converting them back into text results (e.g., detokenizing).
        /// </summary>
        /// <param name="nCurIdx">Specifies the current index being processed, or -1 for the last index.</param>
        /// <param name="blobLogits">Specifies the logits blob output by the last inner product layer of the network.</param>
        /// <param name="softmax">Specifies the softmax layer used to post process the logits.</param>
        /// <param name="nAxis">Specifies the axis of the softmax layer.</param>
        /// <param name="nK">Optionally, specifies the K top items to return (default = 1).</param>
        /// <param name="bSkipDetokenize">Optionally, skip detokenizing - set to true when detokenizing the entire set of tokens at the end (used with unicode tokens).</param>
        /// <returns>The array of word string, index, propabilities and end of sequence found boolean corresponding to the softmax output is returned.</returns>
        public virtual List<Tuple<string, int, double>> PostProcessLogitsOutput(int nCurIdx, Blob<T> blobLogits, Layer<T> softmax, int nAxis, int nK = 1, bool bSkipDetokenize = false)
        {
            return null;
        }
        /// <summary>
        /// The PostProcessFullOutput allows derivative data layers to post-process the results, usually be detokenizing the data in the blobSoftmax.
        /// </summary>
        /// <param name="blobSoftmax">Specifies the data to be post processed.</param>
        /// <returns>A string of the post processed data is returned.</returns>
        public virtual string PostProcessFullOutput(Blob<T> blobSoftmax)
        {
            return null;
        }

        /// <summary>
        /// Convert the index to the word.
        /// </summary>
        /// <param name="nIdx">Specifies the index to convert.</param>
        /// <returns>The corresponding word is returned.</returns>
        public virtual string PostProcessOutput(int nIdx)
        {
            return null;
        }

        /// <summary>
        /// Returns true if the token is an end of sequence token.
        /// </summary>
        /// <param name="nTokenId">Specifies the token to test.</param>
        /// <returns>If the token is an EOS token, true is returned.</returns>
        public virtual bool IsEOS(int nTokenId)
        {
            return false;
        }

        /// <summary>
        /// Detokenize a set of tokens from the data specified.
        /// </summary>
        /// <param name="rg">Specifies an array of tokens.</param>
        /// <param name="nStartIdx">Specifies the start index.</param>
        /// <param name="nCount">Specifies the number of tokens to detokenize.</param>
        /// <param name="bIgnoreBos">Specifies to ignore the BOS token.</param>
        /// <param name="bIgnoreEos">Specifies to ignore the EOS token.</param>
        /// <returns>The detokenized string is returned.</returns>
        public virtual string Detokenize(float[] rg, int nStartIdx, int nCount, bool bIgnoreBos = true, bool bIgnoreEos = true)
        {
            return null;
        }

#pragma warning disable 1591, 1587
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop) /** @private */
        {
        }

        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop) /** @private */
        {
        }

        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop) /** @private */
        {
        }

        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom) /** @private */
        {
        }
#pragma warning restore 1591, 1587
    }
}
