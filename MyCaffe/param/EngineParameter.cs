using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies whether to use the [NVIDIA cuDnn](https://developer.nvidia.com/cudnn) version 
    /// or Caffe version of a given forward/backward operation.
    /// </summary>
    public class EngineParameter : LayerParameterBase 
    {
        Engine m_engine = Engine.DEFAULT;
        bool m_bUseHalfSize = false;

        /// <summary>
        /// Defines the type of engine to use.
        /// </summary>
        public enum Engine
        {
            /// <summary>
            /// Use the default engine that best suits the given layer.
            /// </summary>
            DEFAULT = 0,
            /// <summary>
            /// Use the Caffe version of the layer.
            /// </summary>
            CAFFE = 1,
            /// <summary>
            /// Use the [NVIDIA cuDnn](https://developer.nvidia.com/cudnn) version of the layer.
            /// </summary>
            CUDNN = 2
        }

        /** @copydoc LayerParameterBase */
        public EngineParameter()
        {
        }

        /// <summary>
        /// Specifies the Engine in use.
        /// </summary>
        [Description("Specifies the engine to use 'CAFFE' or CUDNN.  In most instances CUDNN is the default.")]
        public Engine engine
        {
            get { return m_engine; }
            set { m_engine = value; }
        }

        /// <summary>
        /// When true and using the CUDNN engine, half sizes are used on the weights (FP16), otherwise when using the CAFFE engine
        /// this setting is ignored.
        /// </summary>
        [Description("When true and using the CUDNN engine, half sizes (FP16) are used on the weights, otherwise when using the CAFFE engine, this setting is ignored.")]
        public bool cudnn_use_halfsize
        {
            get { return m_bUseHalfSize; }
            set { m_bUseHalfSize = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            EngineParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }


        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            EngineParameter p = (EngineParameter)src;
            m_engine = p.m_engine;
            m_bUseHalfSize = p.m_bUseHalfSize;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            EngineParameter p = new EngineParameter();
            p.Copy(this);
            return p;
        }

        /** @copydoc BaseParameter::ToProto */
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            if (engine != Engine.DEFAULT)
                rgChildren.Add("engine", engine.ToString());

            if (cudnn_use_halfsize)
                rgChildren.Add("cudnn_use_halfsize", cudnn_use_halfsize.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static EngineParameter FromProto(RawProto rp)
        {
            string strVal;
            EngineParameter p = new EngineParameter();

            if ((strVal = rp.FindValue("engine")) != null)
            {
                switch (strVal)
                {
                    case "DEFAULT":
                        p.engine = Engine.DEFAULT;
                        break;

                    case "CAFFE":
                        p.engine = Engine.CAFFE;
                        break;

                    case "CUDNN":
                        p.engine = Engine.CUDNN;
                        break;

                    default:
                        throw new Exception("Unknown 'engine' value: " + strVal);
                }
            }

            if ((strVal = rp.FindValue("cudnn_use_halfsize")) != null)
                p.cudnn_use_halfsize = bool.Parse(strVal);

            return p;
        }
    }
}
