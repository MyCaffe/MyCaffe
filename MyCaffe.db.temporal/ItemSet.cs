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
using static MyCaffe.basecode.descriptors.ValueStreamDescriptor;

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
        OrderedValueStreamDescriptorSet m_rgStrm;
        int m_nColCount = 0;
        List<int> m_rgRowCount = new List<int>(8);
        int m_nTargetStreamNumIdx = 0;
        RawValueSet m_data = null;
        SimpleDatum m_sdStaticNum = null;
        SimpleDatum m_sdStaticCat = null;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="random">Specifies the random number generator.</param>
        /// <param name="db">Specifies the database connection.</param>
        /// <param name="item">Specifies the value item.</param>
        /// <param name="rgStrm">Specifies the value streams associated with the item set.</param>
        public ItemSet(CryptoRandom random, DatabaseTemporal db, ValueItem item, OrderedValueStreamDescriptorSet rgStrm)
        {
            m_random = random;
            m_db = db;
            m_item = item;
            m_rgStrm = rgStrm;
        }

        /// <summary>
        /// Release all resources used.
        /// </summary>
        public void CleanUp()
        {
            m_nColCount = 0;
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
        /// Returns the value streams.
        /// </summary>
        public OrderedValueStreamDescriptorSet Streams
        {
            get { return m_rgStrm; }
        }

        /// <summary>
        /// Loads the data for the item starting at the specified date/time and loading the specified number of steps.
        /// </summary>
        /// <param name="dt">Specifies the start time to load the data.</param>
        /// <param name="bEOD">Returns whether the end of data is found.</param>
        /// <returns>The end date is returned.</returns>
        /// <exception cref="Exception">An exception is thrown on error.</exception>
        public DateTime Load(DateTime dt, out bool bEOD)
        {
            int nSrcID = m_item.SourceID.Value;
            int nItemID = m_item.ID;

            if (m_data == null)
                m_data = m_db.GetValues(nSrcID, nItemID);

            bEOD = true;
            m_nColCount = m_data.ColCount;
            return m_data.EndTime;
        }

        /// <summary>
        /// Clips the data to the load limit by removing older data items.
        /// </summary>
        /// <param name="nMax">Specifies the maximum number of steps to hold in memory.</param>
        public void LoadLimit(int nMax)
        {
#warning("TBD: LoadLimit")
        }

        /// <summary>
        /// Retreives the static, historical and future data at a selected step.
        /// </summary>
        /// <param name="nQueryIdx">Specifies the index of the query, used to show where this query is within a batch.</param>
        /// <param name="nValueIdx">Specifies the value index override when not null, returns the index used with in the item.</param>
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
        public SimpleDatum[] GetData(int nQueryIdx, ref int? nValueIdx, DB_ITEM_SELECTION_METHOD valueSelectionMethod, int nHistSteps, int nFutSteps, int nValueStepOffset = 1, bool bEnableDebug = false, string strDebugPath = null)
        {
            int nTotalSteps = nHistSteps + nFutSteps;

            if (m_nColCount < nTotalSteps)
                return null;

            if (valueSelectionMethod == DB_ITEM_SELECTION_METHOD.RANDOM)
            {
                m_nValIdx = m_random.Next(m_nColCount - nTotalSteps);
            }
            else if (valueSelectionMethod == DB_ITEM_SELECTION_METHOD.NONE)
            {
                if (m_nValIdx >= m_nColCount - nTotalSteps)
                {
                    m_nValIdx = 0;
                    nValueIdx = m_nValIdx;
                    return null;
                }
            }

            if (nValueIdx.HasValue)
                m_nValIdx = nValueIdx.Value;
            nValueIdx = m_nValIdx;

            SimpleDatum sdStatNum = null;
            SimpleDatum sdStatCat = null;
            getStaticData(ref sdStatNum, ref sdStatCat);

            SimpleDatum sdHistNum = null;
            SimpleDatum sdHistCat = null;
            getHistoricalData(m_nValIdx, nHistSteps, out sdHistNum, out sdHistCat);

            SimpleDatum sdFutNum = null;
            SimpleDatum sdFutCat = null;
            getFutureData(m_nValIdx + nHistSteps, nFutSteps, out sdFutNum, out sdFutCat);

            SimpleDatum sdTarget = null;
            SimpleDatum sdTargetHist = null;
            getTargetData(m_nValIdx, nHistSteps, nFutSteps, m_nTargetStreamNumIdx, out sdTarget, out sdTargetHist);

            if (bEnableDebug)
            {
                debug("trg", nQueryIdx, strDebugPath, m_nValIdx, nHistSteps, nFutSteps, sdTargetHist, sdTarget);
                debugStatic("stat_cat", nQueryIdx, strDebugPath, m_nValIdx, sdStatCat);
                debugStatic("stat_num", nQueryIdx, strDebugPath, m_nValIdx, sdStatNum);
                debug("hist_cat", nQueryIdx, strDebugPath, m_nValIdx, sdHistCat, STREAM_CLASS_TYPE.OBSERVED);
                debug("hist_num", nQueryIdx, strDebugPath, m_nValIdx, sdHistNum, STREAM_CLASS_TYPE.OBSERVED);
                debug("fut_cat", nQueryIdx, strDebugPath, m_nValIdx, sdFutCat, STREAM_CLASS_TYPE.KNOWN);
                debug("fut_num", nQueryIdx, strDebugPath, m_nValIdx, sdFutNum, STREAM_CLASS_TYPE.KNOWN);
            }

            SimpleDatum[] rgData = new SimpleDatum[] { m_sdStaticNum, m_sdStaticCat, sdHistNum, sdHistCat, sdFutNum, sdFutCat, sdTarget, sdTargetHist };
            m_nValIdx++;

            return rgData;
        }

        private void debugStatic(string strTag, int nQueryIdx, string strDebugPath, int nIdx, SimpleDatum sd)
        {
            if (string.IsNullOrEmpty(strDebugPath))
                throw new Exception("You must specify a debug path, when 'EnableDebug' = true.");

            if (sd == null)
                return;

            string strName = strTag + ": QueryIdx = " + nQueryIdx.ToString() + ", Idx = " + nIdx.ToString() + ", Length = " + sd.ItemCount.ToString() + " Time: None - Static";
            PlotCollection plots = new PlotCollection(strName);

            float[] rgf = sd.GetData<float>();
            for (int i = 0; i < rgf.Length; i++)
            {
                float fVal = rgf[i];

                Plot plot = new Plot(i, fVal);
                plots.Add(plot);
            }

            if (!Directory.Exists(strDebugPath))
                Directory.CreateDirectory(strDebugPath);

            strDebugPath = strDebugPath.TrimEnd('\\') + "\\";
            string strFile = strDebugPath + nQueryIdx.ToString() + "." + nIdx.ToString() + "." + strTag + ".png";
            Image img = SimpleGraphingControl.QuickRender(plots, 1000, 600, true, ConfigurationAxis.VALUE_RESOLUTION.MINUTE, null, true, null, true);

            img.Save(strFile);
            img.Dispose();

        }

        private void debug(string strTag, int nQueryIdx, string strDebugPath, int nIdx, SimpleDatum sd, STREAM_CLASS_TYPE classType)
        {
            if (string.IsNullOrEmpty(strDebugPath))
                throw new Exception("You must specify a debug path, when 'EnableDebug' = true.");

            if (sd == null)
                return;

            DateTime[] rgSync = getTimeSync(nIdx, sd.Height);
            if (rgSync.Length != sd.Height)
                throw new Exception("The sync and data lengths do not match!");

            string strName = strTag + ": QueryIdx = " + nQueryIdx.ToString() + ", Idx = " + nIdx.ToString() + ", Length = " + sd.ItemCount.ToString() + " Time: " + rgSync[0].ToString() + " - " + rgSync[rgSync.Length - 1].ToString();
            PlotCollectionSet set = new PlotCollectionSet();
            float[] rgf = sd.GetData<float>();

            for (int i = 0; i < sd.Width; i++)
            {
                PlotCollection plots = new PlotCollection(strName + " strm #" + i.ToString());

                for (int j = 0; j < sd.Height; j++)
                {
                    int nDataIdx = j * sd.Width + i;
                    float fVal = rgf[nDataIdx];

                    Plot plot = new Plot(rgSync[j].ToFileTime(), fVal);
                    plot.Tag = rgSync[i];
                    plots.Add(plot);
                }

                set.Add(plots);
            }

            if (!Directory.Exists(strDebugPath))
                Directory.CreateDirectory(strDebugPath);

            DateTime dt = rgSync[0];

            if (classType == STREAM_CLASS_TYPE.OBSERVED)
                dt = rgSync[rgSync.Length - 1];

            strDebugPath = strDebugPath.TrimEnd('\\') + "\\";
            string strFile = strDebugPath + nQueryIdx.ToString() + "." + dt.Year.ToString() + "." + dt.Month.ToString() + "." + dt.Day.ToString() + "_" + dt.Hour.ToString() + "." + dt.Minute.ToString() + "." + dt.Second.ToString() + "." + nIdx.ToString() + "." + strTag + ".png";
            Image img = SimpleGraphingControl.QuickRender(set, 1000, 600, true, ConfigurationAxis.VALUE_RESOLUTION.MINUTE, null, true, null, true);

            img.Save(strFile);
            img.Dispose();
        }

        private void debug(string strTag, int nQueryIdx, string strDebugPath, int nIdx, int nHistSteps, int nFutSteps, SimpleDatum sd1, SimpleDatum sd2)
        {
            if (string.IsNullOrEmpty(strDebugPath))
                throw new Exception("You must specify a debug path, when 'EnableDebug' = true.");

            DateTime[] rgSync = getTimeSync(nIdx, nHistSteps + nFutSteps);
            SimpleDatum sd = getTargetData(nIdx, nHistSteps + nFutSteps, m_nTargetStreamNumIdx);

            if (rgSync.Length != sd1.ItemCount + sd2.ItemCount)
                throw new Exception("The sync and data lengths do not match!");

            if (sd.ItemCount != sd1.ItemCount + sd2.ItemCount)
                throw new Exception("The target data length does not match the sum of the historical and future data lengths!");

            string strName = "TargetData: QueryIdx = " + nQueryIdx.ToString() + ", Idx = " + nIdx.ToString() + ", Hist = " + nHistSteps.ToString() + ", Fut = " + nFutSteps.ToString() + "\nTime: " + rgSync[0].ToString() + " - " + rgSync[rgSync.Length-1].ToString();
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
                int nDataIdx = i + rgf1.Length;

                if (fVal != rgfE[nDataIdx])
                    throw new Exception("The data values do not match!");

                Plot plot = new Plot(rgSync[nDataIdx].ToFileTime(), fVal);
                plot.Tag = rgSync[nDataIdx];
                plots.Add(plot);
            }

            if (!Directory.Exists(strDebugPath))
                Directory.CreateDirectory(strDebugPath);

            DateTime dt = rgSync[rgf1.Length];

            strDebugPath = strDebugPath.TrimEnd('\\') + "\\";
            string strFile = strDebugPath + nQueryIdx.ToString() + "." + dt.Year.ToString() + "." + dt.Month.ToString() + "." + dt.Day.ToString() + "_" + dt.Hour.ToString() + "." + dt.Minute.ToString() + "." + dt.Second.ToString() + "." + nIdx.ToString() + "." + strTag + ".png";
            int nImgWid = 1000;
            int nActualWid = plots.Count * 5 + 55;

            if (nImgWid > nActualWid)
                nImgWid = nActualWid;

            Image img = SimpleGraphingControl.QuickRender(plots, nImgWid, 600, true, ConfigurationAxis.VALUE_RESOLUTION.MINUTE, null, true, null, true);

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

        private void getStaticData(ref SimpleDatum sdNum, ref SimpleDatum sdCat)
        {
            if (m_sdStaticNum != null && m_sdStaticCat != null)
            {
                sdNum = m_sdStaticNum;
                sdCat = m_sdStaticCat;
                return;
            }   

            Tuple<float[], float[]> data = m_data.GetStaticValues();

            List<ValueStreamDescriptor> rgDesc = m_rgStrm.GetStreamDescriptors(STREAM_CLASS_TYPE.STATIC, STREAM_VALUE_TYPE.NUMERIC);
            int nC = 1;

            if (rgDesc != null && rgDesc.Count > 0)
            {
                int nH = rgDesc.Count;
                int nW = rgDesc.Max(p => p.Steps);
                sdNum = new SimpleDatum(nC, nW, nH, data.Item1, 0, data.Item1.Length);
                sdNum.TagName = "StaticNumeric";
            }
            else
            {
                sdNum = null;
            }

            rgDesc = m_rgStrm.GetStreamDescriptors(STREAM_CLASS_TYPE.STATIC, STREAM_VALUE_TYPE.CATEGORICAL);
            nC = 1;

            if (rgDesc != null && rgDesc.Count > 0)
            {
                int nH = rgDesc.Count;
                int nW = rgDesc.Max(p => p.Steps);
                sdCat = new SimpleDatum(nC, nW, nH, data.Item2, 0, data.Item2.Length);
                sdCat.TagName = "StaticCategorical";
            }
            else
            {
                sdCat = null;
            }

            m_sdStaticNum = sdNum;
            m_sdStaticCat = sdCat;
        }

        private void getHistoricalData(int nIdx, int nCount, out SimpleDatum sdNum, out SimpleDatum sdCat)
        {
            Tuple<float[], float[], DateTime[]> dataObs = m_data.GetObservedValues(nIdx, nCount);
            Tuple<float[], float[], DateTime[]> dataKnown = m_data.GetKnownValues(nIdx, nCount);

            // Collect the numeric data
            {
                List<ValueStreamDescriptor> rgDescO = m_rgStrm.GetStreamDescriptors(STREAM_CLASS_TYPE.OBSERVED, STREAM_VALUE_TYPE.NUMERIC);
                int nObsCount = rgDescO == null ? 0 : rgDescO.Count;
                List<ValueStreamDescriptor> rgDescK = m_rgStrm.GetStreamDescriptors(STREAM_CLASS_TYPE.KNOWN, STREAM_VALUE_TYPE.NUMERIC);
                int nKnownCount = rgDescK == null ? 0 : rgDescK.Count;
                int nItemCount = nObsCount + nKnownCount;

                float[] rgfNum = null;
                if (dataObs.Item1.Length > 0 && dataKnown.Item1.Length > 0)
                {
                    rgfNum = new float[dataObs.Item1.Length + dataKnown.Item1.Length];

                    for (int i=0; i<nCount; i++)
                    {
                        for (int j = 0; j < nObsCount; j++)
                        {
                            int nDataIdx = i * nItemCount + j;
                            rgfNum[nDataIdx] = dataObs.Item1[i * nObsCount + j];
                        }

                        for (int j = 0; j < nKnownCount; j++)
                        {
                            int nDataIdx = i * nItemCount + nObsCount + j;
                            rgfNum[nDataIdx] = dataKnown.Item1[i * nKnownCount + j];
                        }
                    }   
                }
                else if (dataObs.Item1.Length > 0)
                {
                    rgfNum = dataObs.Item1;
                }
                else if (dataKnown.Item1.Length > 0)
                {
                    rgfNum = dataKnown.Item1;
                }

                if (rgfNum != null)
                {
                    int nC = 1;
                    int nH = nCount;
                    int nW = nItemCount;

                    sdNum = new SimpleDatum(nC, nW, nH, rgfNum, 0, rgfNum.Length);
                    sdNum.TagName = "HistoricalNumeric";
                    sdNum.Tag = dataObs.Item3;
                }
                else
                {
                    sdNum = null;
                }
            }

            // Collect the categorical data
            {
                List<ValueStreamDescriptor> rgDescO = m_rgStrm.GetStreamDescriptors(STREAM_CLASS_TYPE.OBSERVED, STREAM_VALUE_TYPE.CATEGORICAL);
                int nObsCount = rgDescO == null ? 0 : rgDescO.Count;
                List<ValueStreamDescriptor> rgDescK = m_rgStrm.GetStreamDescriptors(STREAM_CLASS_TYPE.KNOWN, STREAM_VALUE_TYPE.CATEGORICAL);
                int nKnownCount = rgDescK == null ? 0 : rgDescK.Count;
                int nItemCount = nObsCount + nKnownCount;

                float[] rgfCat = null;
                if (dataObs.Item2.Length > 0 && dataKnown.Item2.Length > 0)
                {
                    rgfCat = new float[dataObs.Item2.Length + dataKnown.Item2.Length];

                    for (int i = 0; i < nCount; i++)
                    {
                        for (int j = 0; j < nObsCount; j++)
                        {
                            int nDataIdx = i * nItemCount + j;
                            rgfCat[nDataIdx] = dataObs.Item2[i * nObsCount + j];
                        }

                        for (int j = 0; j < nKnownCount; j++)
                        {
                            int nDataIdx = i * nItemCount + nObsCount + j;
                            rgfCat[nDataIdx] = dataKnown.Item2[i * nKnownCount + j];
                        }
                    }
                }
                else if (dataObs.Item2.Length > 0)
                {
                    rgfCat = dataObs.Item2;
                }
                else if (dataKnown.Item2.Length > 0)
                {
                    rgfCat = dataKnown.Item2;
                }

                if (rgfCat != null)
                {
                    int nC = 1;
                    int nH = nCount;
                    int nW = nItemCount;

                    sdCat = new SimpleDatum(nC, nW, nH, rgfCat, 0, rgfCat.Length);
                    sdCat.TagName = "HistoricalCategorical";
                    sdCat.Tag = dataObs.Item3;
                }
                else
                {
                    sdCat = null;
                }
            }
        }

        private void getFutureData(int nIdx, int nCount, out SimpleDatum sdNum, out SimpleDatum sdCat)
        {
            Tuple<float[], float[], DateTime[]> dataKnown = m_data.GetKnownValues(nIdx, nCount);

            float[] rgfNum = null;
            if (dataKnown.Item1.Length > 0)
                rgfNum = dataKnown.Item1;

            float[] rgfCat = null;
            if (dataKnown.Item2.Length > 0)
                rgfCat = dataKnown.Item2;

            if (rgfNum != null)
            {
                List<ValueStreamDescriptor> rgDescK = m_rgStrm.GetStreamDescriptors(STREAM_CLASS_TYPE.KNOWN, STREAM_VALUE_TYPE.NUMERIC);
                int nC = 1;
                int nH = nCount;
                int nW = rgDescK.Count;

                sdNum = new SimpleDatum(nC, nW, nH, rgfNum, 0, rgfNum.Length);
                sdNum.TagName = "FutureNumeric";
                sdNum.Tag = dataKnown.Item3;
            }
            else
            {
                sdNum = null;
            }

            if (rgfCat != null)
            {
                List<ValueStreamDescriptor> rgDescK = m_rgStrm.GetStreamDescriptors(STREAM_CLASS_TYPE.KNOWN, STREAM_VALUE_TYPE.CATEGORICAL);
                int nC = 1;
                int nH = nCount;
                int nW = rgDescK.Count;

                sdCat = new SimpleDatum(nC, nW, nH, rgfCat, 0, rgfCat.Length);
                sdCat.TagName = "FutureCategorical";
            }
            else
            {
                sdCat = null;
            }
        }

        private void getTargetData(int nIdx, int nHistSteps, int nFutSteps, int nTargetIdx, out SimpleDatum sdTarget, out SimpleDatum sdTargetHist)
        {
            Tuple<float[], DateTime[]> rgfTgt = m_data.GetObservedNumValues(nIdx + nHistSteps, nFutSteps, nTargetIdx);
            Tuple<float[], DateTime[]> rgfTgtH = m_data.GetObservedNumValues(nIdx, nHistSteps, nTargetIdx);

            int nC = 1;
            int nH = rgfTgt.Item1.Length;
            int nW = 1;
            sdTarget = new SimpleDatum(nC, nW, nH, rgfTgt.Item1, 0, rgfTgt.Item1.Length);
            sdTarget.TagName = "Target";
            sdTarget.Tag = rgfTgt.Item2;

            nH = rgfTgtH.Item1.Length;
            sdTargetHist = new SimpleDatum(nC, nW, nH, rgfTgtH.Item1, 0, rgfTgtH.Item1.Length);
            sdTargetHist.TagName = "TargetHist";
            sdTargetHist.Tag = rgfTgtH.Item2;
        }

        private SimpleDatum getTargetData(int nIdx, int nCount, int nTargetIdx)
        {
            Tuple<float[], DateTime[]> rgfTgt = m_data.GetObservedNumValues(nIdx, nCount, nTargetIdx);

            int nC = 1;
            int nH = 1;
            int nW = rgfTgt.Item1.Length;
            SimpleDatum sd = new SimpleDatum(nC, nW, nH, rgfTgt.Item1, 0, rgfTgt.Item1.Length);
            sd.TagName = "Target";

            return sd;
        }

        private DateTime[] getTimeSync(int nIdx, int nCount)
        {
            return m_data.GetTimeSyncValues(nIdx, nCount);
        }

        /// <summary>
        /// Return the total number of queries available in the item set.
        /// </summary>
        /// <param name="nSteps">Specifies the number of steps in the query.</param>
        /// <returns>The number of queries available is returned.</returns>
        public int GetCount(int nSteps)
        {
            int nCount = m_nColCount - nSteps;
            if (nCount < 0)
                return 0;

            return nCount;
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
