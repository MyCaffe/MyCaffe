using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// Simple triplet loss layer
    /// </summary>
    /// <remarks>
    /// @see https://github.com/freesouls/caffe
    /// @see [In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/abs/1703.07737v2) by Alexander Hermans, Lucas Beyer, and Bastian Leibe, 2017. 
    /// @see [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832) by Florian Schroff, and Dmitry Kalenichenko, and James Philbin, 2015.
    /// @see [Generalisation and Sharing in Triplet Convnets for Sketch based Visual Search](https://arxiv.org/abs/1611.05301v1) by Tu Bui, Leonardo Ribeiro, Moacir Ponti, and John Collomosse, 2016.
    /// </remarks> 
    public class TripletLossSimpleParameter : LayerParameterBase 
    {
        double m_dfAlpha = 1.1;
        int m_nSeparate = 10000;

        /** @copydoc LayerParameterBase */
        public TripletLossSimpleParameter()
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

        /// <summary>
        /// Label separate to distinguish the random sampled examples
        /// - should change according to the minibatch.
        /// </summary>
        public int separate
        {
            get { return m_nSeparate; }
            set { m_nSeparate = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            TripletLossSimpleParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            TripletLossSimpleParameter p = (TripletLossSimpleParameter)src;
            m_dfAlpha = p.m_dfAlpha;
            m_nSeparate = p.m_nSeparate;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            TripletLossSimpleParameter p = new TripletLossSimpleParameter();
            p.Copy(this);
            return p;
        }

        /** @copydoc LayerParameterBase::ToProto */
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add("alpha", alpha.ToString());
            rgChildren.Add("separate", separate.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static TripletLossSimpleParameter FromProto(RawProto rp)
        {
            string strVal;
            TripletLossSimpleParameter p = new TripletLossSimpleParameter();

            if ((strVal = rp.FindValue("alpha")) != null)
                p.alpha = double.Parse(strVal);

            if ((strVal = rp.FindValue("separate")) != null)
                p.separate = int.Parse(strVal);

            return p;
        }
    }
}
