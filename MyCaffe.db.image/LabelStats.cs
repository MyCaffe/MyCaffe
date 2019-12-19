using MyCaffe.basecode.descriptors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.db.image
{
#pragma warning disable 1591

    public class LabelStats /** @private */
    {
        List<LabelDescriptor> m_rgLabels;
        List<ulong> m_rgCounts;
        List<ulong> m_rgBoosts;
        object m_objSync = new object();
        object m_objSyncBoost = new object();

        public LabelStats(int nCount)
        {
            m_rgLabels = new List<LabelDescriptor>(nCount);
            m_rgCounts = new List<ulong>(nCount);
            m_rgBoosts = new List<ulong>();
        }

        public void Reset()
        {
            lock (m_objSyncBoost)
            {
                m_rgBoosts = new List<ulong>();
            }
        }

        private int find(int nLabel)
        {
            for (int i = 0; i < m_rgLabels.Count; i++)
            {
                if (m_rgLabels[i].Label == nLabel)
                    return i;
            }

            return -1;
        }

        public void Add(LabelDescriptor label)
        {
            lock (m_objSync)
            {
                m_rgLabels.Add(label);
                m_rgCounts.Add(0);
                m_rgLabels = m_rgLabels.OrderBy(p => p.Label).ToList();
            }
        }

        public void UpdateLabel(int nLabel)
        {
            int nIdx = find(nLabel);
            if (nIdx >= 0)
                m_rgCounts[nIdx]++;
        }

        public void UpdateBoost(int nBoost)
        {
            lock (m_objSyncBoost)
            {
                while (m_rgBoosts.Count <= nBoost)
                {
                    m_rgBoosts.Add(0);
                }

                m_rgBoosts[nBoost]++;
            }
        }

        public Dictionary<int, ulong> GetCounts()
        {
            lock (m_objSync)
            {
                Dictionary<int, ulong> rg = new Dictionary<int, ulong>();

                for (int i = 0; i < m_rgLabels.Count; i++)
                {
                    rg.Add(m_rgLabels[i].Label, m_rgCounts[i]);
                }

                return rg;
            }
        }

        public string GetQueryBoostHitPercentsAsText(int nMax = 10)
        {
            string strOut = "{";
            double dfTotal = 0;

            lock (m_objSyncBoost)
            {
                for (int i = 0; i < m_rgBoosts.Count; i++)
                {
                    dfTotal += m_rgBoosts[i];
                }

                for (int i = 0; i < m_rgBoosts.Count && i < nMax; i++)
                {
                    double dfPct = m_rgBoosts[i] / dfTotal;
                    strOut += dfPct.ToString("P");
                    strOut += ",";
                }
            }

            if (m_rgBoosts.Count > nMax)
                strOut += "...";
            else
                strOut = strOut.TrimEnd(',');

            strOut += "}";

            return strOut;
        }

        public string GetQueryLabelHitPercentsAsText(int nMax = 10)
        {
            string strOut = "{";
            double dfTotal = 0;

            lock (m_objSync)
            {
                for (int i = 0; i < m_rgCounts.Count; i++)
                {
                    dfTotal += m_rgCounts[i];
                }

                for (int i = 0; i < m_rgCounts.Count && i < nMax; i++)
                {
                    double dfPct = m_rgCounts[i] / dfTotal;
                    strOut += dfPct.ToString("P");
                    strOut += ",";
                }
            }

            if (m_rgCounts.Count > nMax)
                strOut += "...";
            else
               strOut = strOut.TrimEnd(',');

            strOut += "}";

            return strOut;
        }

        public string GetQueryLabelEpochAsText(int nMax = 10)
        {
            string strOut = "{";

            lock (m_objSync)
            {
                for (int i = 0; i < m_rgLabels.Count && i < nMax; i++)
                {
                    int nImageCount = m_rgLabels[i].ImageCount;
                    double dfPct = (nImageCount == 0) ? 0 : (double)m_rgCounts[i] / nImageCount;
                    strOut += dfPct.ToString("N2");
                    strOut += ",";
                }
            }

            if (m_rgCounts.Count > nMax)
                strOut += "...";
            else
                strOut = strOut.TrimEnd(',');

            strOut += "}";

            return strOut;
        }
    }

#pragma warning restore 1591
}
