using MyCaffe.basecode;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.trainers
{
    /// <summary>
    /// The ObservationCollection contains a set of observations that make up an 'experience'.
    /// </summary>
    public class ObservationCollection : GenericList<Observation>
    {
        /// <summary>
        /// The constructor.
        /// </summary>
        public ObservationCollection()
        {
        }

        /// <summary>
        /// Returns the total reward of all observations in the collection.
        /// </summary>
        public double TotalReward
        {
            get
            {
                double dfTotal = 0;

                foreach (Observation o in m_rgItems)
                {
                    dfTotal += o.CurrentState.Reward;
                }

                return dfTotal;
            }
        }

        /// <summary>
        /// Returns the number of valid observations within the collection.
        /// </summary>
        public int TotalValidCount
        {
            get
            {
                int nCount = 0;

                foreach (Observation o in m_rgItems)
                {
                    if (o.CurrentState.IsValid)
                        nCount++;
                }

                return nCount;
            }
        }

        /// <summary>
        /// Returns a string representation of the collection.
        /// </summary>
        /// <returns>The string representation is returned.</returns>
        public override string ToString()
        {
            return "Total Reward = " + TotalReward.ToString();
        }
    }
}
