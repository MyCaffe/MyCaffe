using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;
using MyCaffe.common;

namespace MyCaffe.param
{
    /// <summary>
    /// This layer produces N >= 1 top blobs.  DummyDataParameter must specify 1 or 
    /// shape fields, and 0, 1 or N data fillers.
    /// This layer is initialized with the ReLUParameter.
    /// </summary>
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class DummyDataParameter : LayerParameterBase
    {
        List<FillerParameter> m_rgFillers = new List<FillerParameter>();
        List<BlobShape> m_rgShape = new List<BlobShape>();
        List<uint> m_rgNum = new List<uint>();
        List<uint> m_rgChannels = new List<uint>();
        List<uint> m_rgHeight = new List<uint>();
        List<uint> m_rgWidth = new List<uint>();
        bool m_bPrimaryData = true;
        bool m_bForceRefill = true;

        /** @copydoc LayerParameterBase */
        public DummyDataParameter()
        {
        }

        /// <summary>
        /// (\b optional, default = true) Specifies whether to force refill the data (even constant data) as opposed to only refilling once.
        /// </summary>
        /// <remarks>
        /// Given that the training and testing nets share memory, we have changed the default from <i>false</i> to <i>true</i> so that the 
        /// values are refilled on every forward pass.
        /// </remarks>
        public bool force_refill
        {
            get { return m_bForceRefill; }
            set { m_bForceRefill = value; }
        }

        /// <summary>
        /// (\b optional, default = true) Specifies whether or not the data is the primary datset as opposed to a secondary, target dataset.
        /// </summary>
        [Category("Data Selection"), Description("Specifies whether or not this data is the primary dataset as opposed to the target dataset.  By default, this is set to 'true'.")]
        public bool primary_data
        {
            get { return m_bPrimaryData; }
            set { m_bPrimaryData = value; }
        }

        /// <summary>
        /// If 0 data fillers are specified, ConstantFiller
        /// </summary>
        [Description("Specifies the data fillers used to fill the data.  If no data fillers are specified, the constant filler is used.")]
        public List<FillerParameter> data_filler
        {
            get { return m_rgFillers; }
            set { m_rgFillers = value; }
        }

        /// <summary>
        /// Specifies the shape of the dummy data where:
        ///   shape(0) = num
        ///   shape(1) = channels
        ///   shape(2) = height
        ///   shape(3) = width
        /// </summary>
        [Description("Specifies the shape of the dummy data where: shape(0) = num; shape(1) = channels; shape(2) = height; shape(3) = width.")]
        public List<BlobShape> shape
        {
            get { return m_rgShape; }
            set { m_rgShape = value; }
        }

        /// <summary>
        /// <b>DEPRECIATED</b> - 4D dimensions, use 'shape' instead.
        /// </summary>
        [Description("DEPRECIATED: use 'shape(0)' instead.")]
        public List<uint> num
        {
            get { return m_rgNum; }
            set { m_rgNum = value; }
        }

        /// <summary>
        /// <b>DEPRECIATED</b> - 4D dimensions, use 'shape' instead.
        /// </summary>
        [Description("DEPRECIATED: use 'shape(1)' instead.")]
        public List<uint> channels
        {
            get { return m_rgChannels; }
            set { m_rgChannels = value; }
        }

        /// <summary>
        /// <b>>DEPRECIATED</b> - 4D dimensions, use 'shape' instead.
        /// </summary>
        [Description("DEPRECIATED: use 'shape(2)' instead.")]
        public List<uint> height
        {
            get { return m_rgHeight; }
            set { m_rgHeight = value; }
        }

        /// <summary>
        /// <b>DEPRECIATED</b> - 4D dimensions, use 'shape' instead.
        /// </summary>
        [Description("DEPRECIATED: use 'shape(3)' instead.")]
        public List<uint> width
        {
            get { return m_rgWidth; }
            set { m_rgWidth = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            DummyDataParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            DummyDataParameter p = (DummyDataParameter)src;
            m_rgFillers = Utility.Clone<FillerParameter>(p.m_rgFillers);
            m_rgShape = Utility.Clone<BlobShape>(p.m_rgShape);
            m_rgNum = Utility.Clone<uint>(p.m_rgNum);
            m_rgChannels = Utility.Clone<uint>(p.m_rgChannels);
            m_rgHeight = Utility.Clone<uint>(p.m_rgHeight);
            m_rgWidth = Utility.Clone<uint>(p.m_rgWidth);
            m_bPrimaryData = p.m_bPrimaryData;
            m_bForceRefill = p.m_bForceRefill;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            DummyDataParameter p = new DummyDataParameter();
            p.Copy(this);
            return p;
        }

        /** @copydoc LayerParameterBase::ToProto */
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            foreach (FillerParameter fp in data_filler)
            {
                rgChildren.Add(fp.ToProto("data_filler"));
            }

            foreach (BlobShape bs in shape)
            {
                rgChildren.Add(bs.ToProto("shape"));
            }

            rgChildren.Add<uint>("num", num);
            rgChildren.Add<uint>("channels", channels);
            rgChildren.Add<uint>("height", height);
            rgChildren.Add<uint>("width", width);

            if (primary_data == false)
                rgChildren.Add("primary_data", primary_data.ToString());

            if (force_refill == true)
                rgChildren.Add("force_refill", force_refill.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static DummyDataParameter FromProto(RawProto rp)
        {
            DummyDataParameter p = new DummyDataParameter();
            RawProtoCollection rgp;
            
            rgp = rp.FindChildren("data_filler");
            foreach (RawProto child in rgp)
            {
                p.data_filler.Add(FillerParameter.FromProto(child));
            }

            rgp = rp.FindChildren("shape");
            foreach (RawProto child in rgp)
            {
                p.shape.Add(BlobShape.FromProto(child));
            }

            p.num = rp.FindArray<uint>("num");
            p.channels = rp.FindArray<uint>("channels");
            p.height = rp.FindArray<uint>("height");
            p.width = rp.FindArray<uint>("width");

            string strVal;
            if ((strVal = rp.FindValue("primary_data")) != null)
                p.primary_data = bool.Parse(strVal);

            if ((strVal = rp.FindValue("force_refill")) != null)
                p.force_refill = bool.Parse(strVal);

            return p;
        }
    }
}
