using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.common
{
    /// <summary>
    /// The IXDebugData interface is implemented by the DebugLayer to give access to the debug information managed by the layer.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public interface IXDebugData<T>
    {
        /// <summary>
        /// Returns the collection of data blobs.
        /// </summary>
        BlobCollection<T> data { get; }
        /// <summary>
        /// Returns the collection of label blobs.
        /// </summary>
        BlobCollection<T> labels { get; }
        /// <summary>
        /// Returns the name of the data set.
        /// </summary>
        string name { get; }
        /// <summary>
        /// Returns the handle to the kernel within the low-level Cuda Dnn DLL that where the data memory resides.
        /// </summary>
        long kernel_handle { get; }
        /// <summary>
        /// Returns the number of data items loaded into the collection.
        /// </summary>
        int load_count { get; }
    }

    /// <summary>
    /// The IXPersist interface is used by the CaffeControl to load and save weights.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public interface IXPersist<T>
    {
        /// <summary>
        /// Save the weights to a byte array.
        /// </summary>
        /// <remarks>
        /// NOTE: In order to maintain compatibility with the C++ %Caffe, extra MyCaffe features may be added to the <i>end</i> of the weight file.  After saving weights in the format
        /// used by the C++ %Caffe, MyCaffe writes the bytes "mycaffe.ai".  All information after these bytes are specific to MyCaffe and allow for loading weights for models by Blob name and shape
        /// and loosen the C++ %Caffe requirement that the 'number' of blobs match.  Adding this functionality allows for training model, changing the model structure, and then re-using the trained
        /// weights in the new model.  
        /// </remarks>
        /// <param name="colBlobs">Specifies the Blobs to save with the weights.</param>
        /// <param name="bSaveDiff">Specifies whether or not to save the diff values in addition to the data values.</param>
        /// <returns>The byte array containing the weights is returned.</returns>
        byte[] SaveWeights(BlobCollection<T> colBlobs, bool bSaveDiff);

        /// <summary>
        /// Loads new weights into a BlobCollection
        /// </summary>
        /// <remarks>
        /// NOTE: In order to maintain compatibility with the C++ %Caffe, extra MyCaffe features may be added to the <i>end</i> of the weight file.  After saving weights (see SaveWeights) in the format
        /// used by the C++ %Caffe, MyCaffe writes the bytes "mycaffe.ai".  All information after these bytes are specific to MyCaffe and allow for loading weights for models by Blob name and shape
        /// and loosen the C++ %Caffe requirement that the 'number' of blobs match.  Adding this functionality allows for training model, changing the model structure, and then re-using the trained
        /// weights in the new model.  
        /// </remarks>
        /// <param name="rgWeights">Specifies the weights themselves.</param>
        /// <param name="rgExpectedShapes">Specifies a list of expected shapes for each Blob where the weights are to be loaded.</param>
        /// <param name="colBlobs">Specifies the Blobs to load with the weights.</param>
        /// <param name="bLoadedDiffs">Returns whether or not the diffs were loaded.</param>
        /// <returns>The collection of Blobs with newly loaded weights is returned.</returns>
        BlobCollection<T> LoadWeights(byte[] rgWeights, List<string> rgExpectedShapes, BlobCollection<T> colBlobs, out bool bLoadedDiffs);
    }
}
