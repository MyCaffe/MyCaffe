using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.basecode.descriptors
{
    /// <summary>
    /// The StateDescriptor class contains the information related to the state of a project incuding the solver state and weights.
    /// </summary>
    [Serializable]
    public class StateDescriptor : BaseDescriptor
    {
        byte[] m_rgWeights;
        byte[] m_rgState;
        double m_dfAccuracy;
        double m_dfError;
        int m_nIterations;


        /// <summary>
        /// The StateDescriptor constructor.
        /// </summary>
        /// <param name="nId">Specifies the database ID of the item.</param>
        /// <param name="strName">Specifies the name of the item.</param>
        /// <param name="strOwner">Specifies the identifier of the item's owner.</param>
        public StateDescriptor(int nId, string strName, string strOwner)
            : base(nId, strName, strOwner)
        {
            m_rgWeights = null;
            m_rgState = null;
            m_dfAccuracy = -1;
            m_dfError = -1;
            m_nIterations = 0;
        }

        /// <summary>
        /// The StateDescriptor constructor.
        /// </summary>
        /// <param name="nId">Specifies the database ID of the item.</param>
        /// <param name="strName">Specifies the name of the item.</param>
        /// <param name="rgWeights">Specifies the weights of a trained Net.</param>
        /// <param name="rgState">Specifies the state of a Solver in training.</param>
        /// <param name="dfAccuracy">Specifies the accuracy observed while testing.</param>
        /// <param name="dfError">Specifies the error observed whiel training.</param>
        /// <param name="nIterations">Specifies the number of iterations run.</param>
        /// <param name="strOwner">Specifies the identifier of the item's owner.</param>
        public StateDescriptor(int nId, string strName, byte[] rgWeights, byte[] rgState, double dfAccuracy, double dfError, int nIterations, string strOwner)
            : base(nId, strName, strOwner)
        {
            m_rgWeights = rgWeights;
            m_rgState = rgState;
            m_dfAccuracy = dfAccuracy;
            m_dfError = dfError;
            m_nIterations = nIterations;
        }

        /// <summary>
        /// Returns whether or not the state has results (e.g. it has been trained at least to some degree).
        /// </summary>
        public bool HasResults
        {
            get { return (m_rgWeights == null) ? false : true; }
        }

        /// <summary>
        /// Get/set the weights of a trained Net.
        /// </summary>
        [Browsable(false)]
        public byte[] Weights
        {
            get { return m_rgWeights; }
            set { m_rgWeights = value; }
        }

        /// <summary>
        /// Get/set the state of a Solver in training.
        /// </summary>
        [Browsable(false)]
        public byte[] State
        {
            get { return m_rgState; }
            set { m_rgState = value; }
        }

        /// <summary>
        /// Returns the accuracy observed while testing.
        /// </summary>
        [ReadOnly(true)]
        public double Accuracy
        {
            get { return m_dfAccuracy; }
            set { m_dfAccuracy = value; }
        }

        /// <summary>
        /// Specifies the error observed whiel training.
        /// </summary>
        [ReadOnly(true)]
        public double Error
        {
            get { return m_dfError; }
            set { m_dfError = value; }
        }

        /// <summary>
        /// Specifies the number of iterations run.
        /// </summary>
        [ReadOnly(true)]
        public int Iterations
        {
            get { return m_nIterations; }
            set { m_nIterations = value; }
        }
    }
}
