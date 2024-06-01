using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param.ts
{
    /// <summary>
    /// Specifies the parameters for the NHitsBlockParameter (used by NHitsBlockLayer in N-HiTS models).  
    /// </summary>
    /// <remarks>
    /// The NHitsBlockLayer layer performs the Pooling, FC processing and Linear for backcast and forecast.
    /// 
    /// Note: Pooling parameters are specified in the PoolingParameter class.
    /// Note: FC parameters are specified in the FcParameter class.
    /// 
    /// @see [Understanding N-HiTS Time Series Prediction](https://www.signalpop.com/2024/05/29/n-hits/) by Brown, 2024, SignalPop
    /// @see [N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting](https://arxiv.org/abs/2201.12886) by Cristian Challu, Kin G. Olivares, Boris N. Oreshkin, Federico Garza, Max Mergenthaler-Canseco, and Artur Dubrawski, 2022, arXiv:2201.12886.
    /// @see [Darts: User-Friendly Modern Machine Learning for Time Series](https://jmlr.org/papers/v23/21-1177.html) by Julien Herzen, Francesco Lässig, Samuele Giuliano Piazzetta, Thomas Neuer, Léo Tafti, Guillaume Raille, Tomas Van Pottelbergh, Marek Pasieka, Andrzej Skrodzki, Nicolas Huguenin, Maxime Dumonal, Jan Kościsz, Dennis Bader, Frédérick Gusset, Mounir Benheddi, Camila Williamson, Michal Kosinski, Matej Petrik, and Gaël Grosch, 2022, JMLR
    /// @see [Github - unit8co/darts](https://github.com/unit8co/darts) by unit8co, 2022, GitHub.
    /// 
    /// WORK IN PROGRESS.
    /// </remarks>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class NHitsBlockParameter : LayerParameterBase
    {
        int m_nLayerCount = 2;
        int m_nInputChunks = 30;
        int m_nOutputChunks = 7;
        int m_nDownSampleSize = 1;

        /** @copydoc LayerParameterBase */
        public NHitsBlockParameter()
        {
        }

        /// <summary>
        /// Specifies the layer count.
        /// </summary>
        [Description("Specifies the number of the FC layers.")]
        public int num_layers
        {
            get { return m_nLayerCount; }
            set { m_nLayerCount = value; }
        }

        /// <summary>
        /// Specifies the number of input chunks.
        /// </summary>
        [Description("Specifies the number of input chunks.")]
        public int num_input_chunks
        {
            get { return m_nInputChunks; }
            set { m_nInputChunks = value; }
        }

        /// <summary>
        /// Specifies the number of output chunks.
        /// </summary>
        [Description("Specifies the number of input chunks.")]
        public int num_output_chunks
        {
            get { return m_nOutputChunks; }
            set { m_nOutputChunks = value; }
        }

        /// <summary>
        /// Specifies the downsampling size.
        /// </summary>
        [Description("Specifies the downsampling size.")]
        public int downsample_size
        {
            get { return m_nDownSampleSize; }
            set { m_nDownSampleSize = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            NHitsBlockParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            NHitsBlockParameter p = (NHitsBlockParameter)src;

            m_nLayerCount = p.m_nLayerCount;
            m_nInputChunks = p.m_nInputChunks;
            m_nOutputChunks = p.m_nOutputChunks;
            m_nDownSampleSize = p.m_nDownSampleSize;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            NHitsBlockParameter p = new NHitsBlockParameter();
            p.Copy(this);
            return p;
        }

        /// <summary>
        /// Convert the parameter into a RawProto.
        /// </summary>
        /// <param name="strName">Specifies the name to associate with the RawProto.</param>
        /// <returns>The new RawProto is returned.</returns>
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add(new RawProto("num_layers", num_layers.ToString()));
            rgChildren.Add(new RawProto("num_input_chunks", num_input_chunks.ToString()));
            rgChildren.Add(new RawProto("num_output_chunks", num_output_chunks.ToString()));
            rgChildren.Add(new RawProto("downsample_size", downsample_size.ToString()));

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static NHitsBlockParameter FromProto(RawProto rp)
        {
            string strVal;
            NHitsBlockParameter p = new NHitsBlockParameter();

            if ((strVal = rp.FindValue("num_layers")) != null)
                p.num_layers = int.Parse(strVal);

            if ((strVal = rp.FindValue("num_input_chunks")) != null)
                p.num_input_chunks = int.Parse(strVal);

            if ((strVal = rp.FindValue("num_output_chunks")) != null)
                p.num_output_chunks = int.Parse(strVal);

            if ((strVal = rp.FindValue("downsample_size")) != null)
                p.downsample_size = int.Parse(strVal);

            return p;
        }
    }
}
