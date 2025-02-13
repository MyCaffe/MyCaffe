﻿using System;
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
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class TripletLossParameter : LayerParameterBase 
    {
        double m_dfAlpha = 1.1;
        int m_nPreGenerateTargtStart = 0;

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

        /// <summary>
        /// Specifies the starting label for pre-generated targets, only used when 'colBottom.Count' = 5, which contains centroids.
        /// </summary>
        [Description("Specifies the starting label for pre-generated targets, only used when 'colBottom.Count' = 5, which contains centroids.")]
        public int pregen_label_start
        {
            get { return m_nPreGenerateTargtStart; }
            set { m_nPreGenerateTargtStart = value; }
        }

        /// <summary>
        /// Load the parameter from a binary reader.
        /// </summary>
        /// <param name="br">Specifies the binary reader.</param>
        /// <param name="bNewInstance">When <i>true</i> a new instance is created (the default), otherwise the existing instance is loaded from the binary reader.</param>
        /// <returns>Returns an instance of the parameter.</returns>
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            TripletLossParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /// <summary>
        /// Copy on parameter to another.
        /// </summary>
        /// <param name="src">Specifies the parameter to copy.</param>
        public override void Copy(LayerParameterBase src)
        {
            TripletLossParameter p = (TripletLossParameter)src;
            m_dfAlpha = p.m_dfAlpha;
            m_nPreGenerateTargtStart = p.m_nPreGenerateTargtStart;
        }

        /// <summary>
        /// Creates a new copy of this instance of the parameter.
        /// </summary>
        /// <returns>A new instance of this parameter is returned.</returns>
        public override LayerParameterBase Clone()
        {
            TripletLossParameter p = new TripletLossParameter();
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

            rgChildren.Add("alpha", alpha.ToString());
            rgChildren.Add("pregen_label_start", pregen_label_start.ToString());

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
                p.alpha = ParseDouble(strVal);

            if ((strVal = rp.FindValue("pregen_label_start")) != null)
                p.pregen_label_start = int.Parse(strVal);

            return p;
        }
    }
}
