using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;
using static MyCaffe.param.ts.NHitsBlockParameter;

namespace MyCaffe.param.ts
{
    /// <summary>
    /// Specifies the parameters for the NHitsStackParameter (used by NHitsStackLayer in N-HiTS models).  
    /// </summary>
    /// <remarks>
    /// The NHitsStackLayer layer performs the Block processing and prediction accumulation.
    /// 
    /// Note: Pooling parameters are specified in the PoolingParameter class.
    /// Note: FC parameters are specified in the FcParameter class.
    /// Note: NHitsBlock parameters are specified in the NHitsBlockParameter class.
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
    public class NHitsStackParameter : LayerParameterBase
    {
        int m_nBlockCount = 1;
        int m_nAutoPoolingDownsmapleIndex = -1;
        int m_nNumStacks = 1;
        DATA_ORDER m_dataOrder = DATA_ORDER.NTC;

        /** @copydoc LayerParameterBase */
        public NHitsStackParameter()
        {
        }

        /// <summary>
        /// Specifies the data order of each time-series input data stream.
        /// </summary>
        public DATA_ORDER data_order
        {
            get { return m_dataOrder; }
            set { m_dataOrder = value; }
        }

        /// <summary>
        /// Specifies the number of stacks.
        /// </summary>
        [Description("Specifies the number of stacks.")]
        public int num_stacks
        {
            get { return m_nNumStacks; }
            set { m_nNumStacks = value; }
        }

        /// <summary>
        /// When specified (e.g. >= 0), the pooling layer at the specified index has its pooling kernel, stride and downsamplingn sizes automatically calculated.
        /// </summary>
        /// <remarks>
        /// When used, the 'num_stacks' parameter must also be set with the total number of stacks used.  The auto pooling kernel, stride and downsampling sizes 
        /// are calculated based on the input data size and the number of stacks and replace the current pooling kernel, stride and downsampling sizes.
        /// </remarks>
        [Description("When specified (e.g. >= 0), the pooling layer at the specified index has its pooling kernel, stride and downsamplingn sizes automatically calculated.")]
        public int auto_pooling_downsample_index
        {
            get { return m_nAutoPoolingDownsmapleIndex; }
            set { m_nAutoPoolingDownsmapleIndex = value; }
        }

        /// <summary>
        /// Specifies the Block count.
        /// </summary>
        [Description("Specifies the number of the Block layers.")]
        public int num_blocks
        {
            get { return m_nBlockCount; }
            set { m_nBlockCount = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            NHitsStackParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            NHitsStackParameter p = (NHitsStackParameter)src;
            m_nBlockCount = p.m_nBlockCount;
            m_nAutoPoolingDownsmapleIndex = p.m_nAutoPoolingDownsmapleIndex;
            m_nNumStacks = p.m_nNumStacks;
            m_dataOrder = p.m_dataOrder;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            NHitsStackParameter p = new NHitsStackParameter();
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

            rgChildren.Add(new RawProto("num_blocks", num_blocks.ToString()));
            rgChildren.Add(new RawProto("auto_pooling_downsample_index", auto_pooling_downsample_index.ToString()));
            rgChildren.Add(new RawProto("num_stacks", num_stacks.ToString()));
            rgChildren.Add(new RawProto("data_order", data_order.ToString()));

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static NHitsStackParameter FromProto(RawProto rp)
        {
            string strVal;
            NHitsStackParameter p = new NHitsStackParameter();

            if ((strVal = rp.FindValue("num_blocks")) != null)
                p.num_blocks = int.Parse(strVal);

            if ((strVal = rp.FindValue("auto_pooling_downsample_index")) != null)
                p.auto_pooling_downsample_index = int.Parse(strVal);

            if ((strVal = rp.FindValue("num_stacks")) != null)
                p.num_stacks = int.Parse(strVal);

            if ((strVal = rp.FindValue("data_order")) != null)
                p.data_order = (DATA_ORDER)Enum.Parse(typeof(DATA_ORDER), strVal, true);

            return p;
        }
    }
}
