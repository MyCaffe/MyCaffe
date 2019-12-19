using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyCaffe.basecode;

/// <summary>
/// The MyCaffe.param.ssd namespace contains all SSD related parameter objects that correspond to the native C++ %Caffe prototxt file.
/// </summary>
/// <remarks>
/// Using parameters within the MyCaffe.layer.ssd namespace are used by layers that require the MyCaffe.layers.ssd.dll.
/// 
/// @see [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg, 2016.
/// @see [GitHub: SSD: Single Shot MultiBox Detector](https://github.com/weiliu89/caffe/tree/ssd), by weiliu89/caffe, 2016
/// </remarks>
namespace MyCaffe.param.ssd
{
    /// <summary>
    /// Specifies the parameters for the InputLayer.
    /// </summary>
    /// <remarks>
    /// @see [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg, 2016.
    /// @see [GitHub: SSD: Single Shot MultiBox Detector](https://github.com/weiliu89/caffe/tree/ssd), by weiliu89/caffe, 2016
    /// </remarks>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class AnnotatedDataParameter : LayerParameterBase
    {
        SimpleDatum.ANNOTATION_TYPE m_type = SimpleDatum.ANNOTATION_TYPE.NONE;
        List<BatchSampler> m_rgBatchSampler = new List<BatchSampler>();
        string m_strLabelFile;


        /** @copydoc LayerParameterBase */
        public AnnotatedDataParameter()
        {
        }

        /// <summary>
        /// Get/set the annotation type.
        /// </summary>
        [Description("Get/set the annotation type.")]
        public SimpleDatum.ANNOTATION_TYPE anno_type
        {
            get { return m_type; }
            set { m_type = value; }
        }

        /// <summary>
        /// Get/set the batch sampler.
        /// </summary>
        [Description("Get/set the batch sampler.")]
        public List<BatchSampler> batch_sampler
        {
            get { return m_rgBatchSampler; }
            set { m_rgBatchSampler = value; }
        }

        /// <summary>
        /// Get/set the label map file.
        /// </summary>
        [Description("Get/set the label map file.")]
        public string label_map_file
        {
            get { return m_strLabelFile; }
            set { m_strLabelFile = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            AnnotatedDataParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            if (src is AnnotatedDataParameter)
            {
                AnnotatedDataParameter p = (AnnotatedDataParameter)src;
                m_type = p.m_type;
                m_strLabelFile = p.m_strLabelFile;

                m_rgBatchSampler = new List<BatchSampler>();
                foreach (BatchSampler bs in p.m_rgBatchSampler)
                {
                    m_rgBatchSampler.Add(bs.Clone());
                }
            }
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            AnnotatedDataParameter p = new param.ssd.AnnotatedDataParameter();
            p.Copy(this);
            return p;
        }

        /** @copydoc LayerParameterBase::ToProto */
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add(new RawProto("anno_type", ((int)m_type).ToString()));
            rgChildren.Add(new RawProto("label_map_file", m_strLabelFile));

            foreach (BatchSampler bs in m_rgBatchSampler)
            {
                rgChildren.Add(bs.ToProto("batch_sampler"));
            }

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static AnnotatedDataParameter FromProto(RawProto rp)
        {
            AnnotatedDataParameter p = new AnnotatedDataParameter();
            string strVal;

            if ((strVal = rp.FindValue("anno_type")) != null)
                p.m_type = (SimpleDatum.ANNOTATION_TYPE)int.Parse(strVal);

            if ((strVal = rp.FindValue("label_map_file")) != null)
                p.m_strLabelFile = strVal;

            RawProtoCollection col = rp.FindChildren("batch_sampler");
            foreach (RawProto rp1 in col)
            {
                p.m_rgBatchSampler.Add(BatchSampler.FromProto(rp1));
            }

            return p;
        }
    }
}
