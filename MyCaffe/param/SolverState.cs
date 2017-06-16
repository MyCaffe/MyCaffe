using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using MyCaffe.common;

namespace MyCaffe.param
{
    /// <summary>
    /// The SolverState specifies the state of a given solver.
    /// </summary>
    public class SolverState
    {
        int m_nIter;
        int m_nCurrentStep = 0;
        List<BlobProto> m_history = new List<BlobProto>();

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
    }
}
