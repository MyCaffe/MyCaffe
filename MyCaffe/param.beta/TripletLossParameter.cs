using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param.beta
{
    /// <summary>
    /// Specifies the parameters for the TripletLossLayer.
    /// </summary>
    /// <remarks>
    /// @see [Deep Metric Learning Using Triplet Network](https://arxiv.org/pdf/1412.6622.pdf) by Hoffer and Ailon, 2018.
    /// 
    /// @see [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832) by Schroff, 2015.
    /// 
    /// * Initial Python code for TripletDataLayer/TripletSelectionLayer/TripletLossLayer by luhaofang/tripletloss on github. 
    /// @see https://github.com/luhaofang/tripletloss - for general architecture
    /// 
    /// * Initial C++ code for TripletLoss layer by eli-oscherovich in 'Triplet loss #3663' pull request on BVLC/caffe github.
    /// @see https://github.com/BVLC/caffe/pull/3663/commits/c6518fb5752344e1922eaa1b1eb686bae5cc3964 - for triplet loss layer implementation
    /// 
    /// For an explanation of the gradient calculations,
    /// @see http://stackoverflow.com/questions/33330779/whats-the-triplet-loss-back-propagation-gradient-formula/33349475#33349475 - for gradient calculations
    /// 
    /// @see [In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/abs/1703.07737v2) by Alexander Hermans, Lucas Beyer, and Bastian Leibe, 2017. 
    /// @see [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832) by Florian Schroff, and Dmitry Kalenichenko, and James Philbin, 2015.
    /// @see [Generalisation and Sharing in Triplet Convnets for Sketch based Visual Search](https://arxiv.org/abs/1611.05301v1) by Tu Bui, Leonardo Ribeiro, Moacir Ponti, and John Collomosse, 2016.
    /// </remarks> 
    public class TripletLossParameter : LayerParameterBase 
    {
        double m_dfAlpha = 1.1;

        /** @copydoc LayerParameterBase */
        public TripletLossParameter()
        {
        }

        /// <summary>
        /// Specifies the margin.
        /// </summary>
        public double alpha
        {
            get { return m_dfAlpha; }
            set { m_dfAlpha = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            TripletLossParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            TripletLossParameter p = (TripletLossParameter)src;
            m_dfAlpha = p.m_dfAlpha;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            TripletLossParameter p = new TripletLossParameter();
            p.Copy(this);
            return p;
        }

        /** @copydoc LayerParameterBase::ToProto */
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add("alpha", alpha.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static TripletLossParameter FromProto(RawProto rp)
        {
            string strVal;
            TripletLossParameter p = new TripletLossParameter();

            if ((strVal = rp.FindValue("alpha")) != null)
                p.alpha = double.Parse(strVal);

            return p;
        }
    }
}
