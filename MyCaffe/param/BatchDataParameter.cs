using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using System.Threading;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameter for the BatchDataLayer used to load a set of predefined images.
    /// </summary>
    public class BatchDataParameter : LayerParameterBase
    {
        string m_strSource = null;
        DataParameter.DB m_backend = DataParameter.DB.IMAGEDB;
        uint m_nBatchSetCount = 300;
        uint m_nBatchSize = 32;
        int m_nIterations = 1;
        AutoResetEvent m_evtCompleted = null;

        /** @copydoc LayerParameterBase */
        public BatchDataParameter()
        {
        }

        /// <summary>
        /// Specifies the batch set count.
        /// </summary>
        [Description("Specifies the batch set count - the batch set defines the number of batches to train.")]
        public uint batch_set_count
        {
            get { return m_nBatchSetCount; }
            set { m_nBatchSetCount = value; }
        }

        /// <summary>
        /// Specifies the batch size.
        /// </summary>
        [Description("Specifies the batch size of images to collect and train on each iteration of the network.")]
        public uint batch_size
        {
            get { return m_nBatchSize; }
            set { m_nBatchSize = value; }
        }

        /// <summary>
        /// Specifies the data source.
        /// </summary>
        [Description("Specifies the data 'source' within the database.  Some sources are used for training whereas others are used for testing.  Each dataset has both a training and testing data source.")]
        public string source
        {
            get { return m_strSource; }
            set { m_strSource = value; }
        }

        /// <summary>
        /// Specifies the number of iterations to 'forward' over the set of batches.
        /// </summary>
        [Description("Specifies the number of iterations to 'forward' over each batch within the set of batches before setting the completion event (if one exists).")]
        public int iterations
        {
            get { return m_nIterations; }
            set { m_nIterations = value; }
        }

        /// <summary>
        /// Specifies the backend database.
        /// </summary>
        /// <remarks>
        /// NOTE: Currently only the IMAGEDB is supported, which is a separate
        /// component used to load and manage all images within a given dataset.
        /// </remarks>
        [Description("Specifies the backend database type.  Currently only the IMAGEDB database type is supported.  However protofiles specifying the 'LMDB' backend are converted into the 'IMAGEDB' type.")]
        public DataParameter.DB backend
        {
            get { return m_backend; }
            set { m_backend = value; }
        }

        /// <summary>
        /// Specifies an optional event that is set after training 'iterations' over all batches.
        /// </summary>
        [Browsable(false)]
        public AutoResetEvent CompletedEvent
        {
            get { return m_evtCompleted; }
            set { m_evtCompleted = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            BatchDataParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            BatchDataParameter p = (BatchDataParameter)src;
            m_strSource = p.m_strSource;
            m_nBatchSetCount = p.m_nBatchSetCount;
            m_nBatchSize = p.m_nBatchSize;
            m_nIterations = p.m_nIterations;
            m_backend = p.m_backend;
            m_evtCompleted = p.m_evtCompleted;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            BatchDataParameter p = new BatchDataParameter();
            p.Copy(this);
            return p;
        }

        /** @copydoc LayerParameterBase::ToProto */
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add("source", "\"" + source + "\"");
            rgChildren.Add("iterations", m_nIterations.ToString());
            rgChildren.Add("batch_set_count", batch_set_count.ToString());
            rgChildren.Add("batch_size", batch_size.ToString());
            rgChildren.Add("backend", backend.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static BatchDataParameter FromProto(RawProto rp)
        {
            string strVal;
            BatchDataParameter p = new BatchDataParameter();

            if ((strVal = rp.FindValue("source")) != null)
                p.source = strVal.Trim('\"');

            if ((strVal = rp.FindValue("batch_set_count")) != null)
                p.batch_set_count = uint.Parse(strVal);

            if ((strVal = rp.FindValue("batch_size")) != null)
                p.batch_size = uint.Parse(strVal);

            if ((strVal = rp.FindValue("iterations")) != null)
                p.iterations = int.Parse(strVal);

            if ((strVal = rp.FindValue("backend")) != null)
            {
                switch (strVal)
                {
                    case "IMAGEDB":
                        p.backend = DataParameter.DB.IMAGEDB;
                        break;

                    case "LMDB":
                        p.backend = DataParameter.DB.IMAGEDB;
                        break;

                    default:
                        throw new Exception("Unknown 'backend' value " + strVal);
                }
            }

            return p;
        }
    }
}
