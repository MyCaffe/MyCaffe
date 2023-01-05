using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param.gpt
{
    /// <summary>
    /// Specifies the parameters for the TokenizedDataPairsLayer.
    /// </summary>
    /// <remarks>
    /// </remarks>
    public class TokenizedDataPairsParameter : TokenizedDataParameter
    {
        string m_strTarget = "";

        /** @copydoc LayerParameterBase */
        public TokenizedDataPairsParameter() : base()
        {            
        }


        /// <summary>
        /// Specifies the data source based on the INPUT_TYPE used.  Each dataset has both a training and testing data source and target consisting of matching lines.  
        /// For example, when using the TEXT_FILE input type, the source may be the eng.txt file of english phraes and the target may be the fra.txt file of french phrases.
        /// </summary>
        [Description("Specifies the data target (decoder input) based on the INPUT_TYPE used.  Each dataset has both a training and testing data source and target consisting of matching lines.")]
        public string target
        {
            get { return m_strTarget; }
            set { m_strTarget = value; }
        }
        
        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            TokenizedDataPairsParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            base.Copy(src);

            if (src is TokenizedDataPairsParameter)
            {
                TokenizedDataPairsParameter p = (TokenizedDataPairsParameter)src;
                m_strTarget = p.target;
            }
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            TokenizedDataPairsParameter p = new TokenizedDataPairsParameter();
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
            RawProto rpBase = base.ToProto("data");
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add(rpBase.Children);
            rgChildren.Add("target", "\"" + target + "\"");

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static new TokenizedDataPairsParameter FromProto(RawProto rp)
        {
            string strVal;
            TokenizedDataPairsParameter p = new TokenizedDataPairsParameter();

            ((TokenizedDataParameter)p).Copy(TokenizedDataParameter.FromProto(rp));
            
            if ((strVal = rp.FindValue("target")) != null)
                p.target = strVal;

            return p;
        }
    }
}
