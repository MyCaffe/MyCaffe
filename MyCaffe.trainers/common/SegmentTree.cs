using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.trainers.common
{
    /// <summary>
    /// Segment tree data structure
    /// </summary>
    /// <remarks>
    /// The segment tree can be used as a regular array, but with two important differences:
    /// 
    ///   a.) Setting an item's value is slightly slower: O(lg capacity) instead of O(1).
    ///   b.) User has access to an efficient 'reduce' operation which reduces the 'operation' 
    ///       over a contiguous subsequence of items in the array.
    /// 
    /// @see [Wikipedia: Segment tree](https://en.wikipedia.org/wiki/Segment_tree)
    /// @see [GitHub: openai/baselines](https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py) 2018
    /// @see [GitHub: higgsfield/RL-Adventure](https://github.com/higgsfield/RL-Adventure/blob/master/common/replay_buffer.py) 2018
    /// </remarks>
    public class SegmentTree
    {
        /// <summary>
        /// Specifies the capacity of the segment tree.
        /// </summary>
        protected int m_nCapacity;
        /// <summary>
        /// Specifies the operation to perform when reducing the tree.
        /// </summary>
        protected OPERATION m_op;
        /// <summary>
        /// Specifies the data of the tree.
        /// </summary>
        protected float[] m_rgfValues;

        /// <summary>
        /// Specifies the operations used during the reduction.
        /// </summary>
        public enum OPERATION
        {
            /// <summary>
            /// Sum the two elements together.
            /// </summary>
            SUM,
            /// <summary>
            /// Return the minimum of the two elements.
            /// </summary>
            MIN
        }

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="nCapacity">Specifies the total size of the array - must be a power of two.</param>
        /// <param name="oper">Specifies the operation for combining elements (e.g. sum, min)</param>
        /// <param name="fNeutralElement">Specifies the nautral element for the operation above (e.g. float.MaxValue for min and 0 for sum).</param>
        public SegmentTree(int nCapacity, OPERATION oper, float fNeutralElement)
        {
            if (nCapacity <= 0 || (nCapacity % 2) != 0)
                throw new Exception("The capacity must be positive and a power of 2.");

            m_nCapacity = nCapacity;
            m_op = oper;
            m_rgfValues = new float[2 * nCapacity];

            for (int i = 0; i < m_rgfValues.Length; i++)
            {
                m_rgfValues[i] = fNeutralElement;
            }
        }

        private float operation(float f1, float f2)
        {
            switch (m_op)
            {
                case OPERATION.MIN:
                    return Math.Min(f1, f2);

                case OPERATION.SUM:
                    return f1 + f2;

                default:
                    throw new Exception("Unknown operation '" + m_op.ToString() + "'!");
            }
        }

        private float reduce_helper(int nStart, int nEnd, int nNode, int nNodeStart, int nNodeEnd)
        {
            if (nStart == nNodeStart && nEnd == nNodeEnd)
                return m_rgfValues[nNode];

            int nMid = (int)Math.Floor((nNodeStart + nNodeEnd) / 2.0);

            if (nEnd <= nMid)
            {
                return reduce_helper(nStart, nEnd, 2 * nNode, nNodeStart, nMid);
            }
            else
            {
                if (nMid + 1 < nStart)
                {
                    return reduce_helper(nStart, nMid, 2 * nNode + 1, nMid + 1, nNodeEnd);
                }
                else
                {
                    float f1 = reduce_helper(nStart, nMid, 2 * nNode, nNodeStart, nMid);
                    float f2 = reduce_helper(nMid + 1, nEnd, 2 * nNode + 1, nMid + 1, nNodeEnd);
                    return operation(f1, f2);
                }
            }
        }

        /// <summary>
        /// Returns result of applying self.operation to a contiguous subsequence of the array.
        /// operation(arr[start], operation(ar[start+1], operation(..., arr[end])))
        /// </summary>
        /// <param name="nStart">Beginning of the subsequence.</param>
        /// <param name="nEnd">End of the subsequence</param>
        /// <returns></returns>
        public float reduce(int nStart, int? nEnd1 = null)
        {
            int nEnd = nEnd1.GetValueOrDefault(m_nCapacity);

            if (nEnd < 0)
                nEnd += m_nCapacity;

            nEnd -= 1;

            return reduce_helper(nStart, nEnd, 1, 0, m_nCapacity - 1);
        }

        /// <summary>
        /// Element accessor to get and set items.
        /// </summary>
        /// <param name="nIdx">Specifies the index of the item to access.</param>
        /// <returns>The item at the specified index is returned.</returns>
        public float this[int nIdx]
        {
            get
            {
                if (nIdx < 0 || nIdx >= m_nCapacity)
                    throw new Exception("The index is out of range!");

                return m_rgfValues[m_nCapacity + nIdx];
            }
            set
            {
                nIdx += m_nCapacity;
                m_rgfValues[nIdx] = value;

                nIdx = (int)Math.Floor(nIdx / 2.0);

                while (nIdx >= 1)
                {
                    m_rgfValues[nIdx] = operation(m_rgfValues[2 * nIdx], m_rgfValues[2 * nIdx + 1]);
                    nIdx = (int)Math.Floor(nIdx / 2.0);
                }
            }
        }
    }

    /// <summary>
    /// The SumSegmentTree provides a sum reduction of the items within the array.
    /// </summary>
    public class SumSegmentTree : SegmentTree
    {
        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="nCapacity">Specifies the total size of the array - must be a power of two.</param>
        public SumSegmentTree(int nCapacity)
            : base(nCapacity, OPERATION.SUM, 0.0f)
        {
        }

        /// <summary>
        /// Returns arr[start] + ... + arr[end]
        /// </summary>
        /// <param name="nStart">Beginning of the subsequence.</param>
        /// <param name="nEnd">End of the subsequence</param>
        /// <returns>Returns the sum of all items in the array.</returns>
        public float sum(int nStart = 0, int? nEnd = null)
        {
            return reduce(nStart, nEnd);
        }

        /// <summary>
        /// Finds the highest indes 'i' in the array such that sum(arr[0] + arr[1] + ... + arr[i-1]) less than or equal to the 'fPrefixSum'
        /// </summary>
        /// <remarks>
        /// If array values are probabilities, this function allows to sample indexes according to the discrete probability efficiently.
        /// </remarks>
        /// <param name="fPrefixSum">Specifies the upper bound on the sum of array prefix.</param>
        /// <returns>The highest index satisfying the prefixsum constraint is returned.</returns>
        public int find_prefixsum_idx(float fPrefixSum)
        {
            if (fPrefixSum < 0)
                throw new Exception("The prefix sum must be greater than zero.");

            float fSum = sum() + (float)1e-5;
            if (fPrefixSum > fSum)
                throw new Exception("The prefix sum cannot exceed the overall sum of '" + fSum.ToString() + "'!");

            int nIdx = 1;
            while (nIdx < m_nCapacity) // while non-leaf
            {
                if (m_rgfValues[2 * nIdx] > fPrefixSum)
                {
                    nIdx = 2 * nIdx;
                }
                else
                {
                    fPrefixSum -= m_rgfValues[2 * nIdx];
                    nIdx = 2 * nIdx + 1;
                }
            }

            return nIdx - m_nCapacity;
        }
    }

    /// <summary>
    /// The MinSegmentTree performs a reduction over the array and returns the minimum value.
    /// </summary>
    public class MinSegmentTree : SegmentTree
    {
        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="nCapacity">Specifies the total size of the array - must be a power of two.</param>
        public MinSegmentTree(int nCapacity)
            : base(nCapacity, OPERATION.MIN, float.MaxValue)
        {
        }

        /// <summary>
        /// Returns the minimum element in the array.
        /// </summary>
        /// <param name="nStart">Beginning of the subsequence.</param>
        /// <param name="nEnd">End of the subsequence</param>
        /// <returns>The minimum item in the sequence is returned.</returns>
        public float min(int nStart = 0, int? nEnd = null)
        {
            return reduce(nStart, nEnd);
        }
    }
}
