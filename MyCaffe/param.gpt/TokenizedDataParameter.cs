﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param.gpt
{
    /// <summary>
    /// Specifies the parameters for the TokenizedDataLayer.
    /// </summary>
    /// <remarks>
    /// </remarks>
    public class TokenizedDataParameter : LayerParameterBase
    {
        uint m_nBatchSize;
        uint m_nBlockSize;
        INPUT_TYPE m_inputType;
        string m_strSource;
        int? m_nSeed = null;

        /// <summary>
        /// Defines the input type used.
        /// </summary>
        public enum INPUT_TYPE
        {
            /// <summary>
            /// Specifies that the source is a text filename.
            /// </summary>
            TEXT_FILE
        }

        /** @copydoc LayerParameterBase */
        public TokenizedDataParameter()
        {            
        }

        /// <summary>
        /// Specifies the seed used to initialize the random number generator (normally only for testing).
        /// </summary>
        public int? seed
        {
            get { return m_nSeed; }
            set { m_nSeed = value; }
        }

        /// <summary>
        /// Specifies data source input type.
        /// </summary>
        [Description("Specifies data source input type.")]
        public INPUT_TYPE input_type
        {
            get { return m_inputType; }
            set { m_inputType = value; }
        }

        /// <summary>
        /// Specifies the data source based on the INPUT_TYPE used.  Each dataset has both a training and testing data source.
        /// </summary>
        [Description("Specifies the data source based on the INPUT_TYPE used.  Each dataset has both a training and testing data source.")]
        public string source
        {
            get { return m_strSource; }
            set { m_strSource = value; }
        }

        /// <summary>
        /// The number of heads used.
        /// </summary>
        [Description("Specifies batch size.")]
        public uint batch_size
        {
            get { return m_nBatchSize; }
            set { m_nBatchSize = value; }
        }

        /// <summary>
        /// Specifies size of the block.
        /// </summary>
        public uint block_size
        {
            get { return m_nBlockSize; }
            set { m_nBlockSize = value; }
        }
        
        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            TokenizedDataParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            TokenizedDataParameter p = (TokenizedDataParameter)src;

            m_inputType = p.input_type;
            m_strSource = p.source;
            m_nBatchSize = p.batch_size;
            m_nBlockSize = p.block_size;
            m_nSeed = p.seed;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            TokenizedDataParameter p = new TokenizedDataParameter();
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

            rgChildren.Add("input_type", m_inputType.ToString());
            rgChildren.Add("source", "\"" + m_strSource + "\"");
            rgChildren.Add("batch_size", batch_size.ToString());
            rgChildren.Add("block_size", block_size.ToString());

            if (seed != null)
                rgChildren.Add("seed", seed.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static TokenizedDataParameter FromProto(RawProto rp)
        {
            string strVal;
            TokenizedDataParameter p = new TokenizedDataParameter();

            if ((strVal = rp.FindValue("block_size")) != null)
                p.block_size = uint.Parse(strVal);

            if ((strVal = rp.FindValue("batch_size")) != null)
                p.batch_size = uint.Parse(strVal);
            
            if ((strVal = rp.FindValue("source")) != null)
                p.source = strVal.Trim('\"');

            if ((strVal = rp.FindValue("seed")) != null)
                p.seed = int.Parse(strVal);

            if ((strVal = rp.FindValue("input_type")) != null)
            {
                if (strVal == INPUT_TYPE.TEXT_FILE.ToString())
                    p.input_type = INPUT_TYPE.TEXT_FILE;
                else
                    throw new Exception("Unknown input type '" + strVal + "'");
            }

            return p;
        }
    }
}