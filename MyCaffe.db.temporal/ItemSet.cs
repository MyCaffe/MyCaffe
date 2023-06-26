using MyCaffe.basecode;
using MyCaffe.basecode.descriptors;
using SimpleGraphing;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.db.temporal
{
    /// <summary>
    /// The ItemSet manages the data for a single item (e.g., customer, station, stock symbol, etc.) and its associated streams.
    /// </summary>
    [Serializable]
    public class ItemSet
    {
        CryptoRandom m_random;
        int m_nValIdx = 0;
        DatabaseTemporal m_db;
        ValueItem m_item = null;
        PlotCollection m_plotsObservedNum = null;
        PlotCollection m_plotsObservedCat = null;
        PlotCollection m_plotsKnownNum = null;
        PlotCollection m_plotsKnownCat = null;
        List<RawValue> m_rgStaticNum = null;
        List<RawValue> m_rgStaticCat = null;
        List<ValueStream> m_rgStreams = null;
        SimpleDatum m_sdStaticNum = null;
        SimpleDatum m_sdStaticCat = null;
        int m_nRowCount = 0;
        List<int> m_rgRowCount = new List<int>(8);
        int m_nTargetStreamNumIdx = 0;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="random">Specifies the random number generator.</param>
        /// <param name="db">Specifies the database connection.</param>
        /// <param name="item">Specifies the value item.</param>
        /// <param name="rgStreams">Specifies the value streams associated with the item.</param>
        public ItemSet(CryptoRandom random, DatabaseTemporal db, ValueItem item, List<ValueStream> rgStreams)
        {
            m_random = random;
            m_db = db;
            m_item = item;
            m_rgStreams = rgStreams;
        }

        /// <summary>
        /// Release all resources used.
        /// </summary>
        public void CleanUp()
        {
            m_plotsObservedNum = null;
            m_plotsObservedCat = null;
            m_plotsKnownNum = null;
            m_plotsKnownCat = null;
            m_rgStaticNum = null;
            m_rgStreams = null;
            m_nRowCount = 0;
        }

        /// <summary>
        /// Reset the value index.
        /// </summary>
        public void Reset()
        {
            m_nValIdx = 0;
        }

        /// <summary>
        /// Returns the value item.
        /// </summary>
        public ValueItem Item
        {
            get { return m_item; }
        }

        /// <summary>
        /// Loads the data for the item starting at the specified date/time and loading the specified number of steps.
        /// </summary>
        /// <param name="dt">Specifies the start time to load the data.</param>
        /// <param name="nStepsToLoad">Specifies the number of temporal steps to load.</param>
        /// <param name="bNormalizedValue">Specifies whether to load the normalized values.</param>
        /// <param name="bEOD">Returns whether the end of data is found.</param>
        /// <returns>The end date is returned.</returns>
        /// <exception cref="Exception">An exception is thrown on error.</exception>
        public DateTime Load(DateTime dt, int nStepsToLoad, bool bNormalizedValue, out bool bEOD)
        {
            DateTime dtEnd = DateTime.MinValue;
            bEOD = false;

            if (m_rgStaticNum == null)
                m_rgStaticNum = m_db.GetStaticValuesNum(m_item.SourceID.Value, m_item.ID);

            if (m_rgStaticCat == null)
                m_rgStaticCat = m_db.GetStaticValuesCat(m_item.SourceID.Value, m_item.ID);

            bool? bEOD1;
            DateTime? dtEnd1;

            m_rgRowCount.Clear();

            PlotCollection plotsObservedNum = m_db.GetRawValuesObservedNum(m_item.SourceID.Value, m_item.ID, dt, nStepsToLoad, bNormalizedValue, out dtEnd1, out bEOD1);
            if (plotsObservedNum != null)
            {
                if (m_plotsObservedNum == null)
                    m_plotsObservedNum = plotsObservedNum;
                else
                    m_plotsObservedNum.Add(plotsObservedNum);

                m_rgRowCount.Add(m_plotsObservedNum.Count);

                if (bEOD1.GetValueOrDefault(false))
                    bEOD = true;

                if (dtEnd1.HasValue)
                    dtEnd = dtEnd1.Value;
            }

            PlotCollection plotsObservedCat = m_db.GetRawValuesObservedCat(m_item.SourceID.Value, m_item.ID, dt, nStepsToLoad, bNormalizedValue, out dtEnd1, out bEOD1);
            if (plotsObservedCat != null)
            {
                if (m_plotsObservedCat == null)
                    m_plotsObservedCat = plotsObservedCat;
                else
                    m_plotsObservedCat.Add(plotsObservedCat);

                m_rgRowCount.Add(m_plotsObservedCat.Count);

                if (bEOD1.GetValueOrDefault(false))
                    bEOD = true;

                if (dtEnd1.HasValue)
                    dtEnd = dtEnd1.Value;
            }

            PlotCollection plotsKnownNum = m_db.GetRawValuesKnownNum(m_item.SourceID.Value, m_item.ID, dt, nStepsToLoad, bNormalizedValue, out dtEnd1, out bEOD1);
            if (plotsKnownNum != null)
            {
                if (m_plotsKnownNum == null)
                    m_plotsKnownNum = plotsKnownNum;
                else
                    m_plotsKnownNum.Add(plotsKnownNum);

                m_rgRowCount.Add(m_plotsKnownNum.Count);

                if (bEOD1.GetValueOrDefault(false))
                    bEOD = true;

                if (dtEnd1.HasValue)
                    dtEnd = dtEnd1.Value;
            }

            PlotCollection plotsKnownCat = m_db.GetRawValuesKnownCat(m_item.SourceID.Value, m_item.ID, dt, nStepsToLoad, bNormalizedValue, out dtEnd1, out bEOD1);
            if (plotsKnownCat != null)
            {
                if (m_plotsKnownCat == null)
                    m_plotsKnownCat = plotsKnownCat;
                else
                    m_plotsKnownCat.Add(plotsKnownCat);

                m_rgRowCount.Add(m_plotsKnownCat.Count);

                if (bEOD1.GetValueOrDefault(false))
                    bEOD = true;

                if (dtEnd1.HasValue)
                    dtEnd = dtEnd1.Value;
            }

            m_nRowCount = m_rgRowCount[0];
            for (int i = 1; i < m_rgRowCount.Count; i++)
            {
                if (m_rgRowCount[i] != m_nRowCount)
                    throw new Exception("The number of rows in each plot must be the same.");
            }

            return dtEnd;
        }

        /// <summary>
        /// Clips the data to the load limit by removing older data items.
        /// </summary>
        /// <param name="nMax">Specifies the maximum number of steps to hold in memory.</param>
        public void LoadLimit(int nMax)
        {
            m_rgRowCount.Clear();

            if (m_plotsObservedNum != null)
            {
                while (m_plotsObservedNum.Count > nMax)
                {
                    m_plotsObservedNum.RemoveAt(0);
                }
                m_rgRowCount.Add(m_plotsObservedNum.Count);
            }

            if (m_plotsObservedCat != null)
            {
                while (m_plotsObservedCat.Count > nMax)
                {
                    m_plotsObservedCat.RemoveAt(0);
                }
                m_rgRowCount.Add(m_plotsObservedCat.Count);
            }

            if (m_plotsKnownNum != null)
            {
                while (m_plotsKnownNum.Count > nMax)
                {
                    m_plotsKnownNum.RemoveAt(0);
                }
                m_rgRowCount.Add(m_plotsKnownNum.Count);
            }

            if (m_plotsKnownCat != null)
            {
                while (m_plotsKnownCat.Count > nMax)
                {
                    m_plotsKnownCat.RemoveAt(0);
                }
                m_rgRowCount.Add(m_plotsKnownCat.Count);
            }

            m_nRowCount = m_rgRowCount[0];
            for (int i = 1; i < m_rgRowCount.Count; i++)
            {
                if (m_rgRowCount[i] != m_nRowCount)
                    throw new Exception("The number of rows in each plot must be the same.");
            }
        }

        /// <summary>
        /// Retreives the static, historical and future data at a selected step.
        /// </summary>
        /// <param name="nQueryIdx">Specifies the index of the query, used to show where this query is within a batch.</param>
        /// <param name="valueSelectionMethod">Specifies the value step selection method.</param>
        /// <param name="nHistSteps">Specifies the number of historical steps.</param>
        /// <param name="nFutSteps">Specifies the number of future steps.</param>
        /// <param name="nValueStepOffset">Specifes the step offset to apply when advancing the step index.</param>
        /// <param name="bEnableDebug">Optionally, specifies to enable debug output (default = false).</param>
        /// <param name="strDebugPath">Optionally, specifies the debug path where debug images are placed when 'EnableDebug' = true.</param>
        /// <returns>An array of SimpleDatum is returned where: [0] = static num, [1] = static cat, [2] = historical num, [3] = historical cat, [4] = future num, [5] = future cat, [6] = target, and [7] = target history
        /// for a given item at the temporal selection point.</returns>
        /// <remarks>Note, the ordering for historical value streams is: observed, then known.  Future value streams only contiain known value streams.  If a dataset does not have one of the data types noted above, null
        /// is returned in the array slot (for example, if the dataset does not produce static numeric values, the array slot is set to [0] = null.</remarks>
        public SimpleDatum[] GetData(int nQueryIdx, DB_ITEM_SELECTION_METHOD valueSelectionMethod, int nHistSteps, int nFutSteps, int nValueStepOffset = 1, bool bEnableDebug = false, string strDebugPath = null)
        {
            int nTotalSteps = nHistSteps + nFutSteps;

            if (m_nRowCount < nTotalSteps)
                return null;

            if (valueSelectionMethod == DB_ITEM_SELECTION_METHOD.RANDOM)
            {
                m_nValIdx = m_random.Next(m_nRowCount - nTotalSteps);
            }
            else if (valueSelectionMethod == DB_ITEM_SELECTION_METHOD.NONE)
            {
                if (m_nValIdx >= m_nRowCount - nTotalSteps)
                {
                    m_nValIdx = 0;
                    return null;
                }
            }

            if (m_sdStaticNum == null)
                m_sdStaticNum = getStaticDataNum();

            if (m_sdStaticCat == null)
                m_sdStaticCat = getStaticDataCat();

            SimpleDatum sdHistNum = getHistoricalDataNum(m_nValIdx, nHistSteps);
            SimpleDatum sdHistCat = getHistoricalDataCat(m_nValIdx, nHistSteps);
            SimpleDatum sdFutureNum = getFutureDataNum(m_nValIdx + nHistSteps, nFutSteps);
            SimpleDatum sdFutureCat = getFutureDataCat(m_nValIdx + nHistSteps, nFutSteps);
            SimpleDatum sdTarget = getTargetData(m_nValIdx + nHistSteps, nFutSteps);
            SimpleDatum sdTargetHist = getTargetData(m_nValIdx, nHistSteps);

            if (bEnableDebug)
                debug(nQueryIdx, strDebugPath, m_nValIdx, nHistSteps, nFutSteps, sdTargetHist, sdTarget);

            SimpleDatum[] rgData = new SimpleDatum[] { m_sdStaticNum, m_sdStaticCat, sdHistNum, sdHistCat, sdFutureNum, sdFutureCat, sdTarget, sdTargetHist };
            m_nValIdx++;

            return rgData;
        }

        private void debug(int nQueryIdx, string strDebugPath, int nIdx, int nHistSteps, int nFutSteps, SimpleDatum sd1, SimpleDatum sd2)
        {
            if (string.IsNullOrEmpty(strDebugPath))
                throw new Exception("You must specify a debug path, when 'EnableDebug' = true.");

            DateTime[] rgSync = getTimeSync(nIdx, nHistSteps + nFutSteps);
            SimpleDatum sd = getTargetData(nIdx, nHistSteps + nFutSteps);

            if (rgSync.Length != sd1.ItemCount + sd2.ItemCount)
                throw new Exception("The sync and data lengths do not match!");

            if (sd.ItemCount != sd1.ItemCount + sd2.ItemCount)
                throw new Exception("The target data length does not match the sum of the historical and future data lengths!");

            string strName = "TargetData: QueryIdx = " + nQueryIdx.ToString() + ", Idx = " + nIdx.ToString() + ", Hist = " + nHistSteps.ToString() + ", Fut = " + nFutSteps.ToString() + " Time: " + rgSync[0].ToString() + " - " + rgSync[rgSync.Length-1].ToString();
            PlotCollection plots = new PlotCollection(strName);

            float[] rgf1 = sd1.GetData<float>();
            float[] rgf2 = sd2.GetData<float>();
            float[] rgfE = sd.GetData<float>();

            for (int i=0; i<rgf1.Length; i++)
            {
                float fVal = rgf1[i];

                if (fVal != rgfE[i])
                    throw new Exception("The data values do not match!");

                Plot plot = new Plot(rgSync[i].ToFileTime(), fVal);
                plot.Tag = rgSync[i];
                plots.Add(plot);
            }

            for (int i = 0; i < rgf2.Length; i++)
            {
                float fVal = rgf2[i];

                if (fVal != rgfE[i + rgf1.Length])
                    throw new Exception("The data values do not match!");

                Plot plot = new Plot(rgSync[i + rgf1.Length].ToFileTime(), fVal);
                plot.Tag = rgSync[i];
                plots.Add(plot);
            }

            if (!Directory.Exists(strDebugPath))
                Directory.CreateDirectory(strDebugPath);

            DateTime dt = rgSync[rgf1.Length];

            strDebugPath = strDebugPath.TrimEnd('\\') + "\\";
            string strFile = strDebugPath + nQueryIdx.ToString() + "." + dt.Year.ToString() + "." + dt.Month.ToString() + "." + dt.Day.ToString() + "_" + dt.Hour.ToString() + "." + dt.Minute.ToString() + "." + dt.Second.ToString() + "." + nIdx.ToString() + ".png";
            Image img = SimpleGraphingControl.QuickRender(plots, 1000, 600, true, ConfigurationAxis.VALUE_RESOLUTION.MINUTE, null, true, null, true);

            using (Graphics g = Graphics.FromImage(img))
            {
                Brush br = new SolidBrush(Color.FromArgb(64, Color.Blue));
                int nWid = 5 * nFutSteps;
                int nLeft = img.Width - (55 + nWid);
                Rectangle rc = new Rectangle(nLeft, 0, nWid, img.Height);
                g.FillRectangle(br, rc);
                g.DrawLine(Pens.Lime, nLeft, 0, nLeft, img.Height);
                br.Dispose();
            }

            img.Save(strFile);
            img.Dispose();
        }

        private SimpleDatum getStaticDataNum()
        {
            if (m_rgStaticNum == null || m_rgStaticNum.Count == 0)
                return null;

            List<float> rgf = new List<float>();

            foreach (RawValue rv in m_rgStaticNum)
            {
                rgf.Add((float)rv.RawData.Value);
            }

            return new SimpleDatum(1, 1, rgf.Count, rgf.ToArray(), 0, rgf.Count);
        }

        private SimpleDatum getStaticDataCat()
        {
            if (m_rgStaticCat == null || m_rgStaticCat.Count == 0)
                return null;

            List<float> rgf = new List<float>();

            foreach (RawValue rv in m_rgStaticCat)
            {
                rgf.Add((float)rv.RawData.Value);
            }

            return new SimpleDatum(1, 1, rgf.Count, rgf.ToArray(), 0, rgf.Count);
        }

        private SimpleDatum getHistoricalDataNum(int nIdx, int nCount)
        {
            if (m_plotsObservedNum == null && m_plotsKnownNum == null)
                return null;

            List<float> rgf = new List<float>();

            for (int i = nIdx; i < nIdx + nCount; i++)
            {
                if (m_plotsObservedNum != null)
                    rgf.AddRange(m_plotsObservedNum[i].Y_values);
                if (m_plotsKnownNum != null)
                    rgf.AddRange(m_plotsKnownNum[i].Y_values);
            }

            int nStreams = 0;
            if (m_plotsObservedNum != null)
                nStreams += m_plotsObservedNum[0].Y_values.Length;
            if (m_plotsKnownNum != null)
                nStreams += m_plotsKnownNum[0].Y_values.Length;

            return new SimpleDatum(1, nStreams, nCount, rgf.ToArray(), 0, rgf.Count);
        }

        private SimpleDatum getHistoricalDataCat(int nIdx, int nCount)
        {
            if (m_plotsObservedCat == null && m_plotsKnownCat == null)
                return null;

            List<float> rgf = new List<float>();

            for (int i = nIdx; i < nIdx + nCount; i++)
            {
                if (m_plotsObservedCat != null)
                    rgf.AddRange(m_plotsObservedCat[i].Y_values);
                if (m_plotsKnownCat != null)
                    rgf.AddRange(m_plotsKnownCat[i].Y_values);
            }

            int nStreams = 0;
            if (m_plotsObservedCat != null)
                nStreams += m_plotsObservedCat[0].Y_values.Length;
            if (m_plotsKnownCat != null)
                nStreams += m_plotsKnownCat[0].Y_values.Length;

            return new SimpleDatum(1, nStreams, nCount, rgf.ToArray(), 0, rgf.Count);
        }

        private SimpleDatum getFutureDataNum(int nIdx, int nCount)
        {
            if (m_plotsKnownNum == null)
                return null;

            List<float> rgf = new List<float>();

            for (int i = nIdx; i < nIdx + nCount; i++)
            {
                rgf.AddRange(m_plotsKnownNum[i].Y_values);
            }

            int nStreams = m_plotsKnownNum[0].Y_values.Length;

            return new SimpleDatum(1, nStreams, nCount, rgf.ToArray(), 0, rgf.Count);
        }

        private SimpleDatum getFutureDataCat(int nIdx, int nCount)
        {
            if (m_plotsKnownCat == null)
                return null;

            List<float> rgf = new List<float>();

            for (int i = nIdx; i < nIdx + nCount; i++)
            {
                rgf.AddRange(m_plotsKnownCat[i].Y_values);
            }

            int nStreams = m_plotsKnownCat[0].Y_values.Length;

            return new SimpleDatum(1, nStreams, nCount, rgf.ToArray(), 0, rgf.Count);
        }

        private SimpleDatum getTargetData(int nIdx, int nCount)
        {
            List<float> rgf = new List<float>();

            for (int i = nIdx; i < nIdx + nCount; i++)
            {
                rgf.Add(m_plotsObservedNum[i].Y_values[m_nTargetStreamNumIdx]);
            }

            return new SimpleDatum(1, 1, nCount, rgf.ToArray(), 0, rgf.Count);
        }

        private DateTime[] getTimeSync(int nIdx, int nCount)
        {
            List<DateTime> rgDt = new List<DateTime>();

            for (int i = nIdx; i < nIdx + nCount; i++)
            {
                rgDt.Add((DateTime)m_plotsObservedNum[i].Tag);
            }

            return rgDt.ToArray();
        }
    }

    [Serializable]
    public partial class RawValue /**@private */
    {
    }

    [Serializable]
    public partial class ValueItem /**@private */
    {
    }

    [Serializable]
    public partial class ValueStream /**@private */
    {
    }
}
