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
        string m_strTarget;

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

            TokenizedDataPairsParameter p = (TokenizedDataPairsParameter)src;
            m_strTarget = p.target;
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
            RawProto proto = base.ToProto(strName);
            
            proto.Children.Add("target", m_strTarget);

            return proto;
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static new TokenizedDataPairsParameter FromProto(RawProto rp)
        {
            TokenizedDataParameter p1 = TokenizedDataParameter.FromProto(rp);

            string strVal;
            TokenizedDataPairsParameter p = new TokenizedDataPairsParameter();

            p.block_size = p1.block_size;
            p.batch_size = p1.batch_size;
            p.source = p1.source;
            p.seed = p1.seed;
            p.debug_index_file = p1.debug_index_file;
            p.tokenize_run_input = p1.tokenize_run_input;
            p.input_type = p1.input_type;

            if ((strVal = rp.FindValue("target")) != null)
                p.target = strVal;

            return p;
        }
    }
}
