using MyCaffe.basecode;
using SimpleGraphing;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.db.temporal
{
    /// <summary>
    /// The ItemSet manages the data for a single item (e.g., customer, station, stock symbol, etc.) and its associated streams.
    /// </summary>
    public class ItemSet
    {
        CryptoRandom m_random;
        int m_nValIdx = 0;
        DatabaseTemporal m_db;
        ValueItem m_item = null;
        PlotCollection m_plotsHitorical = null;
        PlotCollection m_plotsFuture = null;
        List<RawValue> m_rgStatic = null;
        List<ValueStream> m_rgStreams = null;
        SimpleDatum m_sdStatic = null;

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
            m_plotsHitorical = null;
            m_plotsFuture = null;
            m_rgStatic = null;
            m_rgStreams = null;
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
            bEOD = false;

            if (m_rgStatic == null)
                m_rgStatic = m_db.GetStaticValues(m_item.SourceID.Value, m_item.ID);

            bool bEOD1 = false;
            DateTime dtEnd;
            PlotCollection plotsObserved = m_db.GetRawValuesObserved(m_item.SourceID.Value, m_item.ID, dt, nStepsToLoad, bNormalizedValue, out dtEnd, out bEOD1);
            if (bEOD1)
                bEOD = true;
            PlotCollection plotsKnown = m_db.GetRawValuesKnown(m_item.SourceID.Value, m_item.ID, dt, nStepsToLoad, bNormalizedValue, out dtEnd, out bEOD1);
            if (bEOD1)
                bEOD = true;

            if (plotsObserved.Count != plotsKnown.Count)
                throw new Exception("The number of observed and known values must be the same.");

            int nObservedCount = plotsObserved[0].Y_values.Length;
            int nKnownCount = plotsKnown[0].Y_values.Length;

            for (int i = 0; i < plotsObserved.Count; i++)
            {
                float[] rgf = new float[nObservedCount + nKnownCount];
                Array.Copy(plotsObserved[i].Y_values, rgf, nObservedCount);
                Array.Copy(plotsKnown[i].Y_values, 0, rgf, nObservedCount, nKnownCount);
                plotsObserved[i].SetYValues(rgf);
            }

            for (int i=0; i<nKnownCount; i++)
            {
                string strKey = plotsKnown.Parameters.Keys.ToList()[i];
                plotsObserved.Parameters.Add(strKey, plotsKnown.Parameters[strKey]);
                plotsObserved.ParametersEx.Add(strKey, plotsKnown.ParametersEx[strKey]);
            }

            if (m_plotsHitorical == null)
                m_plotsHitorical = plotsObserved;
            else
                m_plotsHitorical.Add(plotsObserved);

            if (m_plotsFuture == null)
                m_plotsFuture = plotsKnown;
            else
                m_plotsFuture.Add(plotsKnown);

            return dtEnd;
        }

        /// <summary>
        /// Clips the data to the load limit by removing older data items.
        /// </summary>
        /// <param name="nMax">Specifies the maximum number of steps to hold in memory.</param>
        public void LoadLimit(int nMax)
        {
            while (m_plotsHitorical.Count > nMax)
            {
                m_plotsHitorical.RemoveAt(0);
            }

            while (m_plotsFuture.Count > nMax)
            {
                m_plotsFuture.RemoveAt(0);
            }
        }

        /// <summary>
        /// Retreives the static, historical and future data at a selected step.
        /// </summary>
        /// <param name="valueSelectionMethod">Specifies the value step selection method.</param>
        /// <param name="nHistSteps">Specifies the number of historical steps.</param>
        /// <param name="nFutSteps">Specifies the number of future steps.</param>
        /// <param name="nValueStepOffset">Specifes the step offset to apply when advancing the step index.</param>
        /// <returns>A tuple containing the Static, Historical and Future SimpleData is returned.</returns>
        public Tuple<SimpleDatum, SimpleDatum, SimpleDatum> GetData(DB_ITEM_SELECTION_METHOD valueSelectionMethod, int nHistSteps, int nFutSteps, int nValueStepOffset = 1)
        {
            int nTotalSteps = nHistSteps + nFutSteps;

            if (valueSelectionMethod == DB_ITEM_SELECTION_METHOD.RANDOM)
            {
                m_nValIdx = m_random.Next(m_plotsHitorical.Count - nTotalSteps);
            }
            else if (valueSelectionMethod == DB_ITEM_SELECTION_METHOD.NONE)
            {
                if (m_nValIdx >= m_plotsHitorical.Count - nTotalSteps)
                {
                    m_nValIdx = 0;
                    return null;
                }
            }

            if (m_sdStatic == null)
                m_sdStatic = getStaticData();

            SimpleDatum sdHist = getHistoricalData(m_nValIdx, nHistSteps);
            SimpleDatum sdFuture = getFutureData(m_nValIdx + nHistSteps, nFutSteps);

            Tuple<SimpleDatum, SimpleDatum, SimpleDatum> data = new Tuple<SimpleDatum, SimpleDatum, SimpleDatum>(m_sdStatic, sdHist, sdFuture);
            m_nValIdx++;

            return data;
        }

        private SimpleDatum getStaticData()
        {
            List<float> rgf = new List<float>();

            foreach (RawValue rv in m_rgStatic)
            {
                rgf.Add((float)rv.RawData.Value);
            }

            return new SimpleDatum(1, rgf.Count, 1, rgf.ToArray(), 0, rgf.Count);
        }

        private SimpleDatum getHistoricalData(int nIdx, int nCount)
        {
            List<float> rgf = new List<float>();

            for (int i = nIdx; i < nIdx + nCount; i++)
            {
                rgf.AddRange(m_plotsHitorical[i].Y_values);
            }

            return new SimpleDatum(1, nCount, m_plotsHitorical[0].Y_values.Length, rgf.ToArray(), 0, rgf.Count);
        }

        private SimpleDatum getFutureData(int nIdx, int nCount)
        {
            List<float> rgf = new List<float>();

            for (int i = nIdx; i < nIdx + nCount; i++)
            {
                rgf.AddRange(m_plotsFuture[i].Y_values);
            }

            return new SimpleDatum(1, nCount, m_plotsFuture[0].Y_values.Length, rgf.ToArray(), 0, rgf.Count);
        }
    }
}
