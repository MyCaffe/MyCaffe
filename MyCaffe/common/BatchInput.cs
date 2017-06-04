using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.common
{
    /// <summary>
    /// The BatchInput class stores the mini-batch index and input data.
    /// </summary>
    public class BatchInput
    {
        int m_nBatchIdx = 0;
        object m_objInput = null;

        /// <summary>
        /// The BatchInput constructor.
        /// </summary>
        /// <param name="nIdxBatch">Specifies the mini-batch index.</param>
        /// <param name="objInput">Specifies the mini-batch data.</param>
        public BatchInput(int nIdxBatch, object objInput)
        {
            m_nBatchIdx = nIdxBatch;
            m_objInput = objInput;
        }

        /// <summary>
        /// Returns the batch index.
        /// </summary>
        public int BatchIndex
        {
            get { return m_nBatchIdx; }
        }

        /// <summary>
        /// Returns the input data.
        /// </summary>
        public object InputData
        {
            get { return m_objInput; }
        }
    }
}
