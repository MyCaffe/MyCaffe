using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.db.stream;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.ServiceModel;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.preprocessor
{
    /// <summary>
    /// The IXPreprocessor interface is used to query pre-processed data from a streaming database.
    /// </summary>
    [ServiceContract(CallbackContract = typeof(IXStreamDatabaseEvent), SessionMode = SessionMode.Required)]
    public interface IXPreprocessor<T>
    {
        /// <summary>
        /// Initialize the pre-processor with an existing instance of MyCaffe and a streaming database.
        /// </summary>
        /// <param name="imycaffe">Specifies the MyCaffe instance to use.</param>
        /// <param name="istrm">Specifies the streaming database instance to use.</param>
        /// <param name="strPreProcessorDLLPath">Specifies the path to the pre-processor DLL to use so that pre-processing can occur using CUDA.</param>
        void Initialize(IXMyCaffe<T> imycaffe, IXStreamDatabase istrm, string strPreProcessorDLLPath);
        /// <summary>
        /// Shutdown any internal threads used.
        /// </summary>
        void Shutdown();
        /// <summary>
        /// Reset the querying to the start specified within the streaming database, optionally with an offset.
        /// </summary>
        /// <param name="nStartOffset">Optionally, specifies an offset from the start (default = 0).</param>
        void Reset(int nStartOffset = 0);
        /// <summary>
        /// Step through the data of the streaming database, pre-process it, and return the data as a Blob.
        /// </summary>
        /// <param name="bGetSimpleDatum">Specifies whether or not to return the simple datum of data for rendering.</param>
        /// <param name="nWait">Specifies the maximum amount of time to wait for data.</param>
        /// <returns>A tuple containing a Blob of the data on the GPU and optionally the SimpleDatum with the data on the CPU (for rendering).</returns>
        Tuple<Blob<T>, SimpleDatum> Step(bool bGetSimpleDatum, int nWait = 1000);
        /// <summary>
        /// Render the data within the 
        /// </summary>
        /// <param name="sd">Specifies the data to be rendered.</param>
        /// <returns>A tuple containing the Bitmap of the data and the action data as a SimpleDatum is returned.</returns>
        Tuple<Bitmap, SimpleDatum> Render(SimpleDatum sd);
    }
}
