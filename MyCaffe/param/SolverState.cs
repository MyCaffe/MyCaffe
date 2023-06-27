using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using MyCaffe.common;
using System.ComponentModel;

namespace MyCaffe.param
{
    /// <summary>
    /// The SolverState specifies the state of a given solver.
    /// </summary>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class SolverState
    {
        int m_nIter;
        int m_nCurrentStep = 0;
        List<BlobProto> m_history = new List<BlobProto>();
        // L-BFGS state
        int m_nStart = 0;
        int m_nEnd = 0;
        BlobProto m_gradients = null;
        BlobProto m_direction = null;
        List<BlobProto> m_rgHistoryS = new List<BlobProto>();
        List<double> m_rgHistoryRho = new List<double>();

        /// <summary>
        /// The SolverState constructor.
        /// </summary>
        public SolverState()
        {
        }

        /// <summary>
        /// The current iteration.
        /// </summary>
        public int iter
        {
            get { return m_nIter; }
            set { m_nIter = value; }
        }

        /// <summary>
        ///  Specifies the start used by L-BGFS
        /// </summary>
        public int start
        {
            get { return m_nStart; }
            set { m_nStart = value; }
        }

        /// <summary>
        /// Specifies the end used by L-BGFS
        /// </summary>
        public int end
        {
            get { return m_nEnd; }
            set { m_nEnd = value; }
        }

        /// <summary>
        /// The history for SGD solvers.
        /// </summary>
        public List<BlobProto> history
        {
            get { return m_history; }
            set { m_history = value; }
        }
        
        /// <summary>
        /// The current step for learning rate.
        /// </summary>
        public int current_step
        {
            get { return m_nCurrentStep; }
            set { m_nCurrentStep = value; }
        }

        /// <summary>
        /// Gradients used with L-BFGS state.
        /// </summary>
        public BlobProto gradients
        {
            get { return m_gradients; }
            set { m_gradients = value; }
        }

        /// <summary>
        /// Direction used with L-BFGS state.
        /// </summary>
        public BlobProto direction
        {
            get { return m_direction; }
            set { m_direction = value; }
        }

        /// <summary>
        /// S history used with L-BFGS state.
        /// </summary>
        public List<BlobProto> s_history
        {
            get { return m_rgHistoryS; }
            set { m_rgHistoryS = value; }
        }

        /// <summary>
        /// rho history used with L-BFGS state.
        /// </summary>
        public List<double> rho_history
        {
            get { return m_rgHistoryRho; }
            set { m_rgHistoryRho = value; }
        }
    }
}
