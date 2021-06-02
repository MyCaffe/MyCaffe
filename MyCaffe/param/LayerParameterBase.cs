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
        /// <summary>
        /// Defines the label type.
        /// </summary>
        public enum LABEL_TYPE
        {
            /// <summary>
            /// Specifies that no label is used.
            /// </summary>
            NONE,
            /// <summary>
            /// Specifies a single label value per batch item so if the batch size = 8 there should be 8 labels, which is the default.
            /// </summary>
            SINGLE,
            /// <summary>
            /// Specifies multiple values for the label such as are used in segmentation problems, 
            /// the label's are stored in each Data Item's <i>DataCriteria</i> field.
            /// </summary>
            MULTIPLE,
            /// <summary>
            /// Specifies to force the label to a single value regardless of the batch size, so if the batch size is 8, the label count remains only 1.
            /// </summary>
            ONLY_ONE
        }

        /** @copydoc BaseParameter */
        public LayerParameterBase()
        {
        }

        /// <summary>
        /// This method gives derivative classes a chance specify model inputs required
        /// by the run model.
        /// </summary>
        /// <returns>The model inputs required by the layer (if any) or null.</returns>
        public virtual string PrepareRunModelInputs()
        {
            return null;
        }

        /// <summary>
        /// This method gives derivative classes a chance to prepare the layer for a run-model.
        /// </summary>
        /// <param name="p">Specifies the full layer parameter.</param>
        public virtual void PrepareRunModel(LayerParameter p)
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
