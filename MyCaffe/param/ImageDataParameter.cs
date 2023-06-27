using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters for the ImageDataLayer
    /// </summary>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class ImageDataParameter : LayerParameterBase 
    {
        uint m_nRandomSkip = 0;
        bool m_bShuffle = false;
        uint m_nNewHeight = 0;
        uint m_nNewWidth = 0;
        bool m_bIsColor = true;
        string m_strRootFolder = "";


        /** @copydoc LayerParameterBase */
        public ImageDataParameter()
        {
        }

        /// <summary>
        /// Specifies the amount for the image data layer to skip a few points to avoid all asynchronous sgd clients to start at the same point.  The skip point should be set as rand_skip * rand(0,1).  Note that the rand_skip should not be larger than the number of keys in the database.
        /// </summary>
        [Description("Specifies the amount for the image data layer to skip a few points to avoid all asynchronous sgd clients to start at the same point.  The skip point should be set as rand_skip * rand(0,1).  Note that the rand_skip should not be larger than the number of keys in the database.")]
        public uint rand_skip
        {
            get { return m_nRandomSkip; }
            set { m_nRandomSkip = value; }
        }

        /// <summary>
        /// Specifies whether or not the ImageLayer should shuffle the list of files at each epoch.
        /// </summary>
        [Description("Specifies whether or not the ImageLayer should shuffle the list of files at each epoch.")]
        public bool shuffle
        {
            get { return m_bShuffle; }
            set { m_bShuffle = value; }
        }

        /// <summary>
        /// When > 0, specifies the new height of the images fed into the network (default = 0).
        /// </summary>
        [Description("When > 0, specifies the new height of the images fed into the network (default = 0).")]
        public uint new_height
        {
            get { return m_nNewHeight; }
            set { m_nNewHeight = value; }
        }

        /// <summary>
        /// When > 0, specifies the new width of the images fed into the network (default = 0).
        /// </summary>
        [Description("When > 0, specifies the new width of the images fed into the network (default = 0).")]
        public uint new_width
        {
            get { return m_nNewWidth; }
            set { m_nNewWidth = value; }
        }

        /// <summary>
        /// Specififies whether or not the image is color or gray-scale.
        /// </summary>
        [Description("Specifies whether or not the image is color or gray-scale.")]
        public bool is_color
        {
            get { return m_bIsColor; }
            set { m_bIsColor = value; }
        }

        /// <summary>
        /// Specifies the folder containing the image files.
        /// </summary>
        [Description("Specifies the folder containing the image files.")]
        public string root_folder
        {
            get { return m_strRootFolder; }
            set { m_strRootFolder = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            ImageDataParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            ImageDataParameter p = (ImageDataParameter)src;
            m_nRandomSkip = p.m_nRandomSkip;
            m_bShuffle = p.m_bShuffle;
            m_nNewHeight = p.m_nNewHeight;
            m_nNewWidth = p.m_nNewWidth;
            m_bIsColor = p.m_bIsColor;
            m_strRootFolder = p.m_strRootFolder;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            ImageDataParameter p = new ImageDataParameter();
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

            if (rand_skip > 0)
                rgChildren.Add("rand_skip", rand_skip.ToString());

            rgChildren.Add("shuffle", shuffle.ToString());

            if (new_height > 0)
                rgChildren.Add("new_height", new_height.ToString());

            if (new_width > 0)
                rgChildren.Add("new_width", new_width.ToString());

            rgChildren.Add("is_color", is_color.ToString());
            rgChildren.Add("root_folder", "\"" + root_folder + "\"");

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <param name="p">Optionally, specifies an instance to load.  If <i>null</i>, a new instance is created and loaded.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static ImageDataParameter FromProto(RawProto rp, ImageDataParameter p = null)
        {
            string strVal;

            if (p == null)
                p = new ImageDataParameter();

            if ((strVal = rp.FindValue("rand_skip")) != null)
                p.rand_skip = uint.Parse(strVal);

            if ((strVal = rp.FindValue("shuffle")) != null)
                p.shuffle = bool.Parse(strVal);

            if ((strVal = rp.FindValue("new_height")) != null)
                p.new_height = uint.Parse(strVal);

            if ((strVal = rp.FindValue("new_width")) != null)
                p.new_width = uint.Parse(strVal);

            if ((strVal = rp.FindValue("is_color")) != null)
                p.is_color = bool.Parse(strVal);

            if ((strVal = rp.FindValue("root_folder")) != null)
                p.root_folder = strVal.Trim('\"');

            return p;
        }
    }
}
