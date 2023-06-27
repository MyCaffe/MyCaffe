using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters used by the TileLayer
    /// </summary>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class TileParameter : LayerParameterBase
    {
        int m_nAxis = 1;
        int m_nTiles = 0;

        /** @copydoc LayerParameterBase */
        public TileParameter()
        {
        }

        /// <summary>
        /// Specifies the index of the axis to tile.
        /// </summary>
        [Description("Specifies the axis for which to tile.")]
        public int axis
        {
            get { return m_nAxis; }
            set { m_nAxis = value; }
        }

        /// <summary>
        /// Specifies the number of copies (tiles) of the blob to output.
        /// </summary>
        [Description("Specifies the number of copies (tiles) of the blob to output.")]
        public int tiles
        {
            get { return m_nTiles; }
            set { m_nTiles = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            TileParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            TileParameter p = (TileParameter)src;
            m_nAxis = p.m_nAxis;
            m_nTiles = p.m_nTiles;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            TileParameter p = new TileParameter();
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

            rgChildren.Add("axis", axis.ToString());
            rgChildren.Add("tiles", tiles.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static TileParameter FromProto(RawProto rp)
        {
            string strVal;
            TileParameter p = new TileParameter();

            if ((strVal = rp.FindValue("axis")) != null)
                p.axis = int.Parse(strVal);

            if ((strVal = rp.FindValue("tiles")) != null)
                p.tiles = int.Parse(strVal);

            return p;
        }
    }
}
