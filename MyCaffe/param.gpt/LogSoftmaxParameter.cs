using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters for the LogSoftmaxLayer
    /// </summary>
    /// <remarks>
    /// @see [Sofmax vs LogSoftmax](https://medium.com/@AbhiramiVS/softmax-vs-logsoftmax-eb94254445a2) by Abhirami V S, Medium, 2021.
    /// @see [Advantage of using LogSoftmax vs Softmax vs Crossentropyloss in PyTorch](https://androidkt.com/advantage-using-logs-softmax-softmax-crossentropyloss-in-pytorch/) by androidkt, 2022.
    /// </remarks>
    public class LogSoftmaxParameter : LayerParameterBase
    {
        int m_nAxis = 1;

        /** @copydoc LayerParameterBase */
        public LogSoftmaxParameter()
            : base()
        {
        }

        /// <summary>
        /// The axis along which to perform the logsoftmax -- may be negative to index
        /// from the end (e.g., -1 for the last axis).
        /// Any other axes will be evaluated as independent logsoftmaxes.
        /// </summary>
        [Description("Specifies the axis along which to perform the logsoftmax - may be negative to index from the end (e.g., -1 for the last axis).")]
        public int axis
        {
            get { return m_nAxis; }
            set { m_nAxis = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            LogSoftmaxParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            if (src is LogSoftmaxParameter)
            {
                LogSoftmaxParameter p = (LogSoftmaxParameter)src;
                m_nAxis = p.m_nAxis;
            }
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            LogSoftmaxParameter p = new LogSoftmaxParameter();
            p.Copy(this);
            return p;
        }

        /** @copydoc LayerParameterBase::ToProto */
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add("axis", axis.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static LogSoftmaxParameter FromProto(RawProto rp)
        {
            string strVal;
            LogSoftmaxParameter p = new LogSoftmaxParameter();

            if ((strVal = rp.FindValue("axis")) != null)
                p.axis = int.Parse(strVal);

            return p;
        }
    }
}
