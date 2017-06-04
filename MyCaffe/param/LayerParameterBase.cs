using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using MyCaffe.basecode;
using MyCaffe.common;

namespace MyCaffe.param
{
    /// <summary>
    /// The LayerParameterBase is the base class for all other layer specific parameters.
    /// </summary>
    public abstract class LayerParameterBase : BaseParameter, IBinaryPersist 
    {       
        /** @copydoc BaseParameter */
        public LayerParameterBase()
        {
        }

        /// <summary>
        /// Creates a new copy of this instance of the parameter.
        /// </summary>
        /// <returns>A new instance of this parameter is returned.</returns>
        public abstract LayerParameterBase Clone();

        /// <summary>
        /// Copy on parameter to another.
        /// </summary>
        /// <param name="src">Specifies the parameter to copy.</param>
        public abstract void Copy(LayerParameterBase src);

        /// <summary>
        /// Save this parameter to a binary writer.
        /// </summary>
        /// <param name="bw">Specifies the binary writer to use.</param>
        public void Save(BinaryWriter bw)
        {
            RawProto proto = ToProto("layer_param");
            bw.Write(proto.ToString());
        }

        /// <summary>
        /// Load the parameter from a binary reader.
        /// </summary>
        /// <param name="br">Specifies the binary reader.</param>
        /// <param name="bNewInstance">When <i>true</i> a new instance is created (the default), otherwise the existing instance is loaded from the binary reader.</param>
        /// <returns>Returns an instance of the parameter.</returns>
        public abstract object Load(BinaryReader br, bool bNewInstance = true);
    }
}
