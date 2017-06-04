using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using System.Collections;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters used by the ReinforcementLossLayer.
    /// </summary>
    /// <remarks>
    /// @see [Deep Reinforcement Learning: An Overview](https://arxiv.org/abs/1701.07274v2) by Yuxi Li, 2017.
    /// @see [Reinforcement learning with raw image pixels as input state](http://www.montefiore.ulg.ac.be/services/stochastic/pubs/2006/EMW06/ernst-iwicpas-2006.pdf) by Damien Ernst, Raphaël Marée, and Louis Wehenkel, 2006.
    /// @see [Self-Optimizing Memory Controllers: A Reinforcement Learning Approach](https://www.csl.cornell.edu/~martinez/doc/isca08.pdf) by Engin Ipek, Onur Mutlu, José F. Martínez, and Rich Caruana, 2008.
    /// @see [Deep Auto-Encoder Neural Networks in Reinforcement Learning](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.172.1873&rank=1) by Sascha Lange and Martin Riedmiller, 2010.
    /// </remarks>
    public class ReinforcementLossParameter : LayerParameterBase
    {
        double m_dfDiscount = 0.99;
        BatchInformationCollection m_colBatchInfo = null;
        double m_dfExplorationRateStart = 0.6;
        double m_dfExplorationRateEnd = 0.4;
        double m_dfExplorationRateDecay = 6.0;
        uint m_nTrainingStep = 4;

        /** @copydoc LayerParameterBase */
        public ReinforcementLossParameter()
            : base()
        {
        }

        /// <summary>
        /// Specifies the discount rate applied to future 'estimated' rewards.  This value ranges between 0 and 1, where higher values weight future rewards more than near-term rewards.  The default value is 0.99.
        /// </summary>
        [Description("Specifies the discount rate applied to future 'estimated' rewards.  This value ranges between 0 and 1, where higher values weight future rewards more than near-term rewards.  The default value is 0.99.")]
        public double discount_rate
        {
            get { return m_dfDiscount; }
            set { m_dfDiscount = value; }
        }

        /// <summary>
        /// Get/set the batch information collection.
        /// </summary>
        [Browsable(false)]
        public BatchInformationCollection BatchInfoCollection
        {
            get { return m_colBatchInfo; }
            set { m_colBatchInfo = value; }
        }

        /// <summary>
        /// Specifies the starting exploration probability rate range (1.0 - 0.0), default = 0.6, where the exploration rate determines the probability of random selection vs. learned selection.
        /// </summary>
        [Description("Specifies the starting exploration probability rate range (1.0 - 0.0), default = 0.6, where the exploration rate determines the probability of random selection vs. learned selection.")]
        public double exploration_rate_start
        {
            get { return m_dfExplorationRateStart; }
            set { m_dfExplorationRateStart = value; }
        }

        /// <summary>
        /// Specifies the ending exploration probability rate range (1.0 - 0.0), default = 0.4, where the exploration rate determines the probability of random selection vs. learned selection.
        /// </summary>
        [Description("Specifies the ending exploration probability rate range (1.0 - 0.0), default = 0.4, where the exploration rate determines the probability of random selection vs. learned selection.")]
        public double exploration_rate_end
        {
            get { return m_dfExplorationRateEnd; }
            set { m_dfExplorationRateEnd = value; }
        }

        /// <summary>
        /// Specifies the exploration probability decay rate range (1.0 - 10.0), default = 6.0, where this value is used in the function @f$ exprate = 1.0 - tanh(iteration_pct * decayRate) @f$.  Larger numbers move the result closer to the end rate, whereas smaller numbers cause a more gradual shift from the start rate to the end rate.
        /// </summary>
        [Description("Specifies the exploration probability decay rate range (1.0 - 10.0), default = 6.0, where this value is used in the function 'exprate' = 1.0 - tanh(iteration_pct * 'decay rate').  Larger numbers move the result closer to the end rate, whereas smaller numbers cause a more gradual shift from the start rate to the end rate.")]
        public double exploration_rate_decay
        {
            get { return m_dfExplorationRateDecay; }
            set { m_dfExplorationRateDecay = value; }
        }

        /// <summary>
        /// Specifies the number of run steps that take place between each training step, range (1, +), default = 4
        /// </summary>
        [Description("Specifies the number of run steps that take place between each training step, range (1, +), default = 4")]
        public uint training_step
        {
            get { return m_nTrainingStep; }
            set { m_nTrainingStep = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            ReinforcementLossParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            if (src is ReinforcementLossParameter)
            {
                ReinforcementLossParameter p = (ReinforcementLossParameter)src;
                m_colBatchInfo = p.m_colBatchInfo;
                m_dfDiscount = p.m_dfDiscount;
                m_dfExplorationRateStart = p.m_dfExplorationRateStart;
                m_dfExplorationRateEnd = p.m_dfExplorationRateEnd;
                m_dfExplorationRateDecay = p.m_dfExplorationRateDecay;
                m_nTrainingStep = p.m_nTrainingStep;
            }
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            ReinforcementLossParameter p = new ReinforcementLossParameter();
            p.Copy(this);
            return p;
        }

        /** @copydoc LayerParameterBase::ToProto */
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            if (discount_rate != 0.99)
                rgChildren.Add("discount_rate", discount_rate.ToString());

            if (exploration_rate_start != 0.6)
                rgChildren.Add("exploration_rate_start", exploration_rate_start.ToString());

            if (exploration_rate_end != 0.4)
                rgChildren.Add("exploration_rate_end", exploration_rate_end.ToString());

            if (exploration_rate_decay != 6.0)
                rgChildren.Add("exploration_rate_decay", exploration_rate_decay.ToString());

            if (training_step != 4)
                rgChildren.Add("training_step", training_step.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static ReinforcementLossParameter FromProto(RawProto rp)
        {
            string strVal;
            ReinforcementLossParameter p = new ReinforcementLossParameter();

            if ((strVal = rp.FindValue("discount_rate")) != null)
                p.discount_rate = double.Parse(strVal);

            if ((strVal = rp.FindValue("exploration_rate_start")) != null)
                p.exploration_rate_start = double.Parse(strVal);

            if ((strVal = rp.FindValue("exploration_rate_end")) != null)
                p.exploration_rate_end = double.Parse(strVal);

            if ((strVal = rp.FindValue("exploration_rate_decay")) != null)
                p.exploration_rate_decay = double.Parse(strVal);

            if ((strVal = rp.FindValue("training_step")) != null)
                p.training_step = uint.Parse(strVal);

            return p;
        }
    }

    /// <summary>
    /// A collection of BatchInformation objects.
    /// </summary>
    public class BatchInformationCollection : IEnumerable<BatchInformation>
    {
        List<BatchInformation> m_rgBatches = new List<BatchInformation>();

        /// <summary>
        /// The BatchInformationCollection constructor.
        /// </summary>
        public BatchInformationCollection()
        {
        }

        /// <summary>
        /// The number of items in the collection.
        /// </summary>
        public int Count
        {
            get { return m_rgBatches.Count; }
        }

        /// <summary>
        /// Get/set an item within the collection.
        /// </summary>
        /// <param name="nIdx">Specifies the index of the item.</param>
        /// <returns>The item within the collection at <i>nIdx</i> is returned.</returns>
        public BatchInformation this[int nIdx]
        {
            get { return m_rgBatches[nIdx]; }
            set { m_rgBatches[nIdx] = value; }
        }

        /// <summary>
        /// Add a BatchInformation into the collection.
        /// </summary>
        /// <param name="b"></param>
        public void Add(BatchInformation b)
        {
            m_rgBatches.Add(b);
        }

        /// <summary>
        /// Remove a BatchInformation object from the collection if it exists.
        /// </summary>
        /// <param name="b">Specifies the object to remove.</param>
        /// <returns>If the object exists and is removed, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool Remove(BatchInformation b)
        {
            return m_rgBatches.Remove(b);
        }

        /// <summary>
        /// Removes the object at index <i>nIdx</i>.
        /// </summary>
        /// <param name="nIdx">Specifies the index of the object to remove.</param>
        public void RemoveAt(int nIdx)
        {
            m_rgBatches.RemoveAt(nIdx);
        }

        /// <summary>
        /// Removes all items from the collection.
        /// </summary>
        public void Clear()
        {
            m_rgBatches.Clear();
        }

        /// <summary>
        /// Retrieves the enumerator for the collection.
        /// </summary>
        /// <returns></returns>
        public IEnumerator<BatchInformation> GetEnumerator()
        {
            return m_rgBatches.GetEnumerator();
        }

        /// <summary>
        /// Retrieves the enumerator for the collection.
        /// </summary>
        IEnumerator IEnumerable.GetEnumerator()
        {
            return m_rgBatches.GetEnumerator();
        }
    }

    /// <summary>
    /// The BatchInformation object stores a mini batch of items. 
    /// </summary>
    public class BatchInformation : IEnumerable<BatchItem>
    {
        int m_nBatchIndex = 0;
        List<BatchItem> m_rgBatch = new List<BatchItem>();

        /// <summary>
        /// The BatchInformation constructor.
        /// </summary>
        /// <param name="nIdx">The mini-batch index.</param>
        public BatchInformation(int nIdx)
        {
            m_nBatchIndex = nIdx;
        }

        /// <summary>
        /// Returns the mini-batch index.
        /// </summary>
        public int BatchIndex
        {
            get { return m_nBatchIndex; }
        }

        /// <summary>
        /// Returns the number of items in the mini-batch.
        /// </summary>
        public int Count
        {
            get { return m_rgBatch.Count; }
        }

        /// <summary>
        /// Get/set an item at a given index within the mini-batch.
        /// </summary>
        /// <param name="nIdx">Specifies the index of the item.</param>
        /// <returns>Returns an item within the mini-batch.</returns>
        public BatchItem this[int nIdx]
        {
            get { return m_rgBatch[nIdx]; }
            set { m_rgBatch[nIdx] = value; }
        }

        /// <summary>
        /// Adds an item to the mini-batch.
        /// </summary>
        /// <param name="b"></param>
        public void Add(BatchItem b)
        {
            m_rgBatch.Add(b);
        }

        /// <summary>
        /// Removes an item from the mini-batch if it exists.
        /// </summary>
        /// <param name="b">Specifies the item to remove.</param>
        /// <returns>If the item is found and removed, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool Remove(BatchItem b)
        {
            return m_rgBatch.Remove(b);
        }

        /// <summary>
        /// Removes the item at a given index.
        /// </summary>
        /// <param name="nIdx">Specifies the index.</param>
        public void RemoveAt(int nIdx)
        {
            m_rgBatch.RemoveAt(nIdx);
        }

        /// <summary>
        /// Removes all items from the collection.
        /// </summary>
        public void Clear()
        {
            m_rgBatch.Clear();
        }

        /// <summary>
        /// Retrieves the mini-batch enumerator.
        /// </summary>
        /// <returns></returns>
        public IEnumerator<BatchItem> GetEnumerator()
        {
            return m_rgBatch.GetEnumerator();
        }

        /// <summary>
        /// Retrieves the mini-batch enumerator.
        /// </summary>
        /// <returns></returns>
        IEnumerator IEnumerable.GetEnumerator()
        {
            return m_rgBatch.GetEnumerator();
        }
    }

    /// <summary>
    /// The BatchItem contains all information about one item within a mini-batch.
    /// </summary>
    public class BatchItem
    {
        int m_nImgIdx0;
        int m_nImgIdx1;
        double m_dfReward;
        double m_dfQmax1;
        int m_nLabel;
        bool m_bTerminal;

        /// <summary>
        /// The BatchItem constructor.
        /// </summary>
        /// <param name="nImgIdx0">Specifies the image index.</param>
        /// <param name="nImgIdx1">Specifies the next image index.</param>
        public BatchItem(int nImgIdx0, int nImgIdx1)
        {
            m_nImgIdx0 = nImgIdx0;
            m_nImgIdx1 = nImgIdx1;
            m_dfReward = 0;
            m_dfQmax1 = 0;
            m_nLabel = 0;
            m_bTerminal = false;
        }

        /// <summary>
        /// Returns the current image index.
        /// </summary>
        public int ImageIndex0
        {
            get { return m_nImgIdx0; }
        }

        /// <summary>
        /// Returns the next image index.
        /// </summary>
        public int ImageIndex1
        {
            get { return m_nImgIdx1; }
        }

        /// <summary>
        /// Get/set the reward value.
        /// </summary>
        public double Reward
        {
            get { return m_dfReward; }
            set { m_dfReward = value; }
        }

        /// <summary>
        /// Get/set the q-max value.
        /// </summary>
        public double QMax1
        {
            get { return m_dfQmax1; }
            set { m_dfQmax1 = value; }
        }

        /// <summary>
        /// Get/set the label.
        /// </summary>
        public int Label
        {
            get { return m_nLabel; }
            set { m_nLabel = value; }
        }

        /// <summary>
        /// Get/set whether or not this is an end item.
        /// </summary>
        public bool Terminal
        {
            get { return m_bTerminal; }
            set { m_bTerminal = value; }
        }
    }
}
