using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;
using System.IO;

namespace MyCaffe.param.ssd
{
    /// <summary>
    /// Specifies the parameters for the SaveOutputLayer.
    /// </summary>
    /// <remarks>
    /// @see [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg, 2016.
    /// @see [GitHub: SSD: Single Shot MultiBox Detector](https://github.com/weiliu89/caffe/tree/ssd), by weiliu89/caffe, 2016
    /// </remarks>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class SaveOutputParameter : OptionalParameter
    {
        string m_strOutputDirectory;
        string m_strOutputNamePrefix;
        OUTPUT_FORMAT m_outputFormat = OUTPUT_FORMAT.VOC;
        string m_strLabelMapFile;
        string m_strNameSizeFile;
        uint? m_nNumTestImage = null;
        ResizeParameter m_resizeParam = null;

        /// <summary>
        /// Defines the output format.
        /// </summary>
        public enum OUTPUT_FORMAT
        {        
            /// <summary>
            /// Specifies the PASCAL VOC output format.
            /// </summary>
            VOC,
            /// <summary>
            /// Specifies the MS COCO output format.
            /// </summary>
            COCO,
            /// <summary>
            /// Specifies the ILSVRC output format.
            /// </summary>
            ILSVRC
        }

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="bActive">Specifies whether or not the parameter is active or not.</param>
        public SaveOutputParameter(bool bActive) : base(bActive)
        {
        }

        /// <summary>
        /// Specifies the output directory - if not empty, the results will be saved.
        /// </summary>
        [Description("Specifies output directory, if not empty, the results will be saved.")]
        public string output_directory
        {
            get { return m_strOutputDirectory; }
            set { m_strOutputDirectory = value; }
        }

        /// <summary>
        /// Specifies the output name prefix.
        /// </summary>
        [Description("Specifies output name prefix.")]
        public string output_name_prefix
        {
            get { return m_strOutputNamePrefix; }
            set { m_strOutputNamePrefix = value; }
        }

        /// <summary>
        /// Specifies the output format.
        /// </summary>
        [Description("Specifies output name prefix.")]
        public OUTPUT_FORMAT output_format
        {
            get { return m_outputFormat; }
            set { m_outputFormat = value; }
        }

        /// <summary>
        /// Optionally, specifies the output label map file.
        /// </summary>
        [Description("Optionally, specifies output label map file.")]
        public string label_map_file
        {
            get { return m_strLabelMapFile; }
            set { m_strLabelMapFile = value; }
        }

        /// <summary>
        /// Optionally, specifies the output name size file.
        /// </summary>
        /// <remarks>
        /// The name size file contains a list of names and sizes with the order of the input DB.  The
        /// file is in the following format:
        ///    name height width
        ///    ...
        /// </remarks>
        [Description("Optionally, specifies output name size file.")]
        public string name_size_file
        {
            get { return m_strNameSizeFile; }
            set { m_strNameSizeFile = value; }
        }

        /// <summary>
        /// Specifies the number of test images.
        /// </summary>
        /// <remarks>This setting can be less than the lines specified in the name_size_file.  For example,
        /// when we only want to evaluate on part of the test images.</remarks>
        [Description("Specifies the number of test images, which can be less than the name_size_file line count.")]
        public uint? num_test_image
        {
            get { return m_nNumTestImage; }
            set { m_nNumTestImage = value; }
        }

        /// <summary>
        /// Specifies the resize parameter used in saving the data.
        /// </summary>
        [Description("Specifies the resize parameter used in saving the data.")]
        public ResizeParameter resize_param
        {
            get { return m_resizeParam; }
            set { m_resizeParam = value; }
        }

        /// <summary>
        /// Load the and return a new ResizeParameter. 
        /// </summary>
        /// <param name="br"></param>
        /// <param name="bNewInstance"></param>
        /// <returns>The new object is returned.</returns>
        public SaveOutputParameter Load(BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            SaveOutputParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /// <summary>
        /// Copy the source object.
        /// </summary>
        /// <param name="src">Specifies the source data.</param>
        public override void Copy(OptionalParameter src)
        {
            base.Copy(src);

            if (src is SaveOutputParameter)
            {
                SaveOutputParameter p = (SaveOutputParameter)src;

                m_strOutputDirectory = p.m_strOutputDirectory;
                m_strOutputNamePrefix = p.m_strOutputNamePrefix;
                m_outputFormat = p.m_outputFormat;
                m_strLabelMapFile = p.m_strLabelMapFile;
                m_strNameSizeFile = p.m_strNameSizeFile;
                m_nNumTestImage = p.m_nNumTestImage;

                m_resizeParam = null;
                if (p.resize_param != null)
                    m_resizeParam = p.resize_param.Clone();
            }
        }

        /// <summary>
        /// Return a copy of this object.
        /// </summary>
        /// <returns>A new copy of the object is returned.</returns>
        public SaveOutputParameter Clone()
        {
            SaveOutputParameter p = new SaveOutputParameter(Active);
            p.Copy(this);
            return p;
        }

        /// <summary>
        /// Convert this object to a raw proto.
        /// </summary>
        /// <param name="strName">Specifies the name of the proto.</param>
        /// <returns>The new proto is returned.</returns>
        public override RawProto ToProto(string strName)
        {
            RawProto rpBase = base.ToProto("option");
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add(rpBase);
            rgChildren.Add(new RawProto("output_directory", m_strOutputDirectory));
            rgChildren.Add(new RawProto("output_name_prefix", m_strOutputNamePrefix));
            rgChildren.Add(new RawProto("output_format", m_outputFormat.ToString()));
            rgChildren.Add(new RawProto("label_map_file", m_strLabelMapFile));
            rgChildren.Add(new RawProto("name_size_file", m_strNameSizeFile));

            if (m_nNumTestImage.HasValue)
                rgChildren.Add(new RawProto("num_test_image", m_nNumTestImage.Value.ToString()));

            if (resize_param != null)
                rgChildren.Add(resize_param.ToProto("resize_param"));

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static new SaveOutputParameter FromProto(RawProto rp)
        {
            SaveOutputParameter p = new SaveOutputParameter(true);
            string strVal;

            RawProto rpOption = rp.FindChild("option");
            if (rpOption != null)
                ((OptionalParameter)p).Copy(OptionalParameter.FromProto(rpOption));

            if ((strVal = rp.FindValue("output_directory")) != null)
                p.output_directory = strVal;

            if ((strVal = rp.FindValue("output_name_prefix")) != null)
                p.output_name_prefix = strVal;

            if ((strVal = rp.FindValue("output_format")) != null)
            {
                if (strVal == OUTPUT_FORMAT.VOC.ToString())
                    p.output_format = OUTPUT_FORMAT.VOC;
                else if (strVal == OUTPUT_FORMAT.COCO.ToString())
                    p.output_format = OUTPUT_FORMAT.COCO;
                else
                    throw new Exception("Unknown output_format '" + strVal + "'!");
            }

            if ((strVal = rp.FindValue("label_map_file")) != null)
                p.label_map_file = strVal;

            if ((strVal = rp.FindValue("name_size_file")) != null)
                p.name_size_file = strVal;

            if ((strVal = rp.FindValue("num_test_image")) != null)
                p.num_test_image = uint.Parse(strVal);

            RawProto rpResize = rp.FindChild("resize_param");
            if (rpResize != null)
                p.resize_param = ResizeParameter.FromProto(rpResize);

            return p;
        }
    }
}
