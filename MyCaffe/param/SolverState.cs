using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;

namespace MyCaffe.param
{
    /// <summary>
    /// The SolverState specifies the state of a given solver.
    /// </summary>
    public class SolverState
    {
        int m_nIter;
        List<BlobProto> m_rgHistory = new List<BlobProto>();
        int m_nCurrentStep = 0;

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
            get { return m_rgHistory; }
            set { m_rgHistory = value; }
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
        /// Saves the SolverState to a binary writer.
        /// </summary>
        /// <param name="bw">Specifies the binary writer.</param>
        public void Save(BinaryWriter bw)
        {
            bw.Write(m_nIter);
            bw.Write(m_nCurrentStep);
            bw.Write(m_rgHistory.Count);

            for (int i = 0; i < m_rgHistory.Count; i++)
            {
                m_rgHistory[i].Save(bw);
            }
        }

        /// <summary>
        /// Loads a SolverState from a binary reader.
        /// </summary>
        /// <param name="br">Specifies the binary reader to use.</param>
        /// <returns>A new SolverState instance is returned.</returns>
        public static SolverState Load(BinaryReader br)
        {
            SolverState s = new SolverState();

            s.m_nIter = br.ReadInt32();
            s.m_nCurrentStep = br.ReadInt32();

            int nCount = br.ReadInt32();

            for (int i = 0; i < nCount; i++)
            {
                s.m_rgHistory.Add(BlobProto.Load(br));
            }

            return s;
        }
    }
}
