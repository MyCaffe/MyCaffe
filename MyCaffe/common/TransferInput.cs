using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.common
{
    /// <summary>
    /// The TransferInput class is used to transfer get and set input data.  
    /// </summary>
    public class TransferInput
    {
        fnSetInputData m_fnSet = null;
        fnGetInputData m_fnGet = null;

        /// <summary>
        /// This delegate is used to set input data.
        /// </summary>
        /// <param name="bi">Specifies the batch input data.</param>
        public delegate void fnSetInputData(BatchInput bi);
        /// <summary>
        /// This delegate is used to get input data.
        /// </summary>
        /// <returns>The BatchInput previously set is returned.</returns>
        public delegate BatchInput fnGetInputData();

        /// <summary>
        /// The TransferInput constructor.
        /// </summary>
        /// <param name="getInput">Specifies the delegate to get the input.</param>
        /// <param name="setInput">Specifies the delegate to set the input.</param>
        public TransferInput(fnGetInputData getInput, fnSetInputData setInput)
        {
            m_fnGet = getInput;
            m_fnSet = setInput;
        }

        /// <summary>
        /// Returns the delegate used to get the batch input.
        /// </summary>
        public fnGetInputData Get
        {
            get { return m_fnGet; }
        }

        /// <summary>
        /// Returns the delegate used to set the batch input.
        /// </summary>
        public fnSetInputData Set
        {
            get { return m_fnSet; }
        }
    }
}
