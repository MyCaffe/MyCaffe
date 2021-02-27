using MyCaffe.basecode;
using MyCaffe.basecode.descriptors;
using MyCaffe.param;
using MyCaffe;
using MyCaffe.db.image;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using MyCaffe.common;
using System.Reflection;
using System.Collections;

namespace MyCaffe.gym
{
    /// <summary>
    /// The Model Gym runs a given Project over the dataset specified within the project where each step advanced
    /// to another data item within the project's dataset.
    /// </summary>
    public class ModelGym : IXMyCaffeGymData, IDisposable
    {
        string m_strName = "Model";
        string m_strModelDesc;
        string m_strDataset;
        int m_nGpuID = 0;
        byte[] m_rgWeights;
        Log m_log;
        Dictionary<string, int> m_rgActions = new Dictionary<string, int>();
        DATA_TYPE m_dt = DATA_TYPE.VALUES;
        Phase m_phase = Phase.NONE;
        MyCaffeControl<float> m_mycaffe = null;
        Blob<float> m_blobWork = null;
        IXImageDatabaseBase m_imgdb = null;
        CancelEvent m_evtCancel = new CancelEvent();
        int m_nCurrentIdx = 0;
        DatasetDescriptor m_ds = null;
        int m_nBatchSize = 16;
        bool m_bRecreateData = true;
        ScoreCollection m_scores = new ScoreCollection();
        int m_nWidth = 0;
        int m_nDim = 0;
        

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="gym">Optionally, specifies another Gym to copy.</param>
        public ModelGym(ModelGym gym = null)
        {
            if (gym != null)
            {
                m_strName = gym.m_strName;
                m_strModelDesc = gym.m_strModelDesc;
                m_strDataset = gym.m_strDataset;
                m_nGpuID = gym.m_nGpuID;
                m_rgWeights = gym.m_rgWeights;
                m_log = gym.m_log;

                m_mycaffe = gym.m_mycaffe;
                gym.m_mycaffe = null;

                m_imgdb = gym.m_imgdb;
                gym.m_imgdb = null;

                m_evtCancel = gym.m_evtCancel;
                gym.m_evtCancel = null;
            }
        }

        /// <summary>
        /// Release all resources used.
        /// </summary>
        public void Dispose()
        {
            Close();
        }

        /// <summary>
        /// Initialize the gym with the specified properties.
        /// </summary>
        /// <param name="log">Specifies the output log to use.</param>
        /// <param name="properties">Specifies the properties containing Gym specific initialization parameters.</param>
        /// <remarks>
        /// The ModelGym uses the following initialization properties.
        /// 
        /// 'GpuID' - the GPU to run on. 
        /// 'ModelDescription' - the model description of the model to use.
        /// 'Dataset' - the name of the dataset to use.
        /// 'Weights' - the model trained weights.
        /// 'CudaPath' - the path of the CudaDnnDLL to use.
        /// 'BatchSize' - the batch size used when running images through the model (default = 16).
        /// 'RecreateData' - when 'True' the data is re-run through the model, otherwise if already run the data is loaded from file (faster).
        /// </remarks>
        public void Initialize(Log log, PropertySet properties)
        {
            m_nGpuID = properties.GetPropertyAsInt("GpuID");
            m_strModelDesc = properties.GetProperty("ModelDescription");
            m_strDataset = properties.GetProperty("Dataset");
            m_rgWeights = properties.GetPropertyBlob("Weights");
            m_nBatchSize = properties.GetPropertyAsInt("BatchSize", 16);
            m_bRecreateData = properties.GetPropertyAsBool("RecreateData", false);

            string strCudaPath = properties.GetProperty("CudaPath");

            SettingsCaffe s = new SettingsCaffe();
            s.GpuIds = m_nGpuID.ToString();
            s.ImageDbLoadMethod = IMAGEDB_LOAD_METHOD.LOAD_ON_DEMAND_BACKGROUND;

            m_imgdb = new MyCaffeImageDatabase2(log);
            m_imgdb.InitializeWithDsName1(s, m_strDataset);
            m_ds = m_imgdb.GetDatasetByName(m_strDataset);

            SimpleDatum sd = m_imgdb.QueryImage(m_ds.TrainingSource.ID, 0, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.NONE);
            BlobShape shape = new BlobShape(1, sd.Channels, sd.Height, sd.Width);

            if (m_evtCancel == null)
                m_evtCancel = new CancelEvent();

            m_mycaffe = new MyCaffeControl<float>(s, log, m_evtCancel, null, null, null, null, strCudaPath);
            m_mycaffe.LoadToRun(m_strModelDesc, m_rgWeights, shape);

            m_log = log;
        }

        /// <summary>
        /// Create a new copy of the gym.
        /// </summary>
        /// <param name="properties">Optionally, specifies the properties to initialize the new copy with.</param>
        /// <returns>The new Gym copy is returned.</returns>
        public IXMyCaffeGym Clone(PropertySet properties = null)
        {
            return new ModelGym(this);
        }

        /// <summary>
        /// Returns <i>true</i> indicating that this Gym requires a display image.
        /// </summary>
        public bool RequiresDisplayImage
        {
            get { return false; }
        }

        /// <summary>
        /// Returns the selected data type.
        /// </summary>
        public DATA_TYPE SelectedDataType
        {
            get { return m_dt; }
        }

        /// <summary>
        /// Returns the data types supported by this gym.
        /// </summary>
        public DATA_TYPE[] SupportedDataType
        {
            get { return new DATA_TYPE[] { DATA_TYPE.VALUES }; }
        }

        /// <summary>
        /// Returns the gym's name.
        /// </summary>
        public string Name
        {
            get { return m_strName; }
        }

        /// <summary>
        /// Returns the delay to use (if any) when the user-display is visible.
        /// </summary>
        public int UiDelay
        {
            get { return 0; }
        }

        /// <summary>
        /// Returns the testinng percent of 0, which will cause the training data to be used during testing.
        /// </summary>
        public double TestingPercent
        {
            get { return 0; }
        }

        /// <summary>
        /// Returns the action space as a dictionary of name,actionid pairs.
        /// </summary>
        /// <returns>The action space is returned.</returns>
        public Dictionary<string, int> GetActionSpace()
        {
            return m_rgActions;
        }

        /// <summary>
        /// Shutdown and close the gym.
        /// </summary>
        public void Close()
        {
            if (m_blobWork != null)
            {
                m_blobWork.Dispose();
                m_blobWork = null;
            }

            if (m_mycaffe != null)
            {
                m_mycaffe.Dispose();
                m_mycaffe = null;
            }

            if (m_imgdb != null)
            {
                ((MyCaffeImageDatabase2)m_imgdb).Dispose();
                m_imgdb = null;
            }
        }

        /// <summary>
        /// Render the gym's current state on a bitmap and SimpleDatum.
        /// </summary>
        /// <param name="bShowUi">When <i>true</i> the Bitmap is drawn.</param>
        /// <param name="nWidth">Specifies the width used to size the Bitmap.</param>
        /// <param name="nHeight">Specifies the height used to size the Bitmap.</param>
        /// <param name="bGetAction">When <i>true</i> the action data is returned as a SimpleDatum.</param>
        /// <returns>A tuple optionally containing a Bitmap and/or Simpledatum is returned.</returns>
        public Tuple<Bitmap, SimpleDatum> Render(bool bShowUi, int nWidth, int nHeight, bool bGetAction)
        {
            List<double> rgData = new List<double>();
            return Render(bShowUi, nWidth, nHeight, rgData.ToArray(), bGetAction);
        }

        /// <summary>
        /// Render the gyms specified data.
        /// </summary>
        /// <param name="bShowUi">When <i>true</i> the Bitmap is drawn.</param>
        /// <param name="nWidth">Specifies the width used to size the Bitmap.</param>
        /// <param name="nHeight">Specifies the height used to size the Bitmap.</param>
        /// <param name="rgData">Specifies the gym data to render.</param>
        /// <param name="bGetAction">When <i>true</i> the action data is returned as a SimpleDatum.</param>
        /// <returns>A tuple optionally containing a Bitmap and/or Simpledatum is returned.</returns>
        public Tuple<Bitmap, SimpleDatum> Render(bool bShowUi, int nWidth, int nHeight, double[] rgData, bool bGetAction)
        {
            return null;
        }

        /// <summary>
        /// Reset the state of the gym.
        /// </summary>
        /// <param name="bGetLabel">Not used.</param>
        /// <returns>A tuple containing state data, the reward, and the done state is returned.</returns>
        public Tuple<State, double, bool> Reset(bool bGetLabel)
        {
            m_nCurrentIdx = 0;
            return Step(-1, bGetLabel);
        }

        /// <summary>
        /// Step the gym one step in the data.
        /// </summary>
        /// <param name="nAction">Specifies the action to run on the gym.</param>
        /// <param name="bGetLabel">Not used.</param>
        /// <param name="extraProp">Optionally, specifies extra properties.</param>
        /// <returns>A tuple containing state data, the reward, and the done state is returned.</returns>
        public Tuple<State, double, bool> Step(int nAction, bool bGetLabel = false, PropertySet extraProp = null)
        {
            DataState data = new DataState();
            ScoreCollection scores = null;

            if (ActivePhase == Phase.RUN)
            {
                if (extraProp == null)
                    throw new Exception("The extra properties are needed when querying data during the RUN phase.");

                int nDataCount = extraProp.GetPropertyAsInt("DataCountRequested");
                string strStartTime = extraProp.GetProperty("SeedTime");


                int nStartIdx = m_scores.Count - nDataCount;                
                DateTime dt;
                if (DateTime.TryParse(strStartTime, out dt))
                    nStartIdx = m_scores.FindIndexAt(dt, nDataCount);

                scores = m_scores.CopyFrom(nStartIdx, nDataCount);
            }
            else
            {
                int nCount = 0;

                m_scores = load(out m_nDim, out m_nWidth);

                if (m_bRecreateData || m_scores.Count != m_ds.TrainingSource.ImageCount)
                {
                    Stopwatch sw = new Stopwatch();
                    sw.Start();

                    m_scores = new ScoreCollection();

                    while (m_nCurrentIdx < m_ds.TrainingSource.ImageCount)
                    {
                        // Query images sequentially by index in batches
                        List<SimpleDatum> rgSd = new List<SimpleDatum>();

                        for (int i = 0; i < m_nBatchSize; i++)
                        {
                            SimpleDatum sd = m_imgdb.QueryImage(m_ds.TrainingSource.ID, m_nCurrentIdx + i, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.NONE);
                            rgSd.Add(sd);
                            nCount++;

                            if (nCount == m_ds.TrainingSource.ImageCount)
                                break;
                        }

                        List<ResultCollection> rgRes = m_mycaffe.Run(rgSd, ref m_blobWork);

                        if (m_nWidth == 0)
                        {
                            m_nWidth = rgRes[0].ResultsOriginal.Count;
                            m_nDim = rgRes[0].ResultsOriginal.Count * 2;
                        }

                        // Fill SimpleDatum with the ordered label,score pairs starting with the detected label.
                        for (int i = 0; i < rgRes.Count; i++)
                        {
                            m_scores.Add(new Score(rgSd[i].TimeStamp, rgSd[i].Index, rgRes[i]));
                            m_nCurrentIdx++;
                        }

                        if (sw.Elapsed.TotalMilliseconds > 1000)
                        {
                            m_log.Progress = (double)m_nCurrentIdx / (double)m_ds.TrainingSource.ImageCount;
                            m_log.WriteLine("Running model on image " + m_nCurrentIdx.ToString() + " of " + m_ds.TrainingSource.ImageCount.ToString() + " of '" + m_strDataset + "' dataset.");

                            if (m_evtCancel.WaitOne(0))
                                return null;
                        }
                    }

                    save(m_nDim, m_nWidth, m_scores);
                }
                else
                {
                    m_nCurrentIdx = m_scores.Count;
                }

                scores = m_scores;
            }

            float[] rgfRes = scores.Data;
            SimpleDatum sdRes = new SimpleDatum(scores.Count, m_nWidth, 2, rgfRes, 0, rgfRes.Length);
            data.SetData(sdRes);
            m_nCurrentIdx = 0;

            return new Tuple<State, double, bool>(data, 0, false);
        }

        private string save_file
        {
            get { return Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location) + "\\data.bin"; }
        }

        private void save(int nDim, int nWid, ScoreCollection col)
        {
            string strFile = save_file;

            using (FileStream fs = File.OpenWrite(strFile))
            using (BinaryWriter bw = new BinaryWriter(fs))
            {
                bw.Write(nDim);
                bw.Write(nWid);

                col.Save(bw);
            }
        }

        private ScoreCollection load(out int nDim, out int nWid)
        {
            string strFile = save_file;

            nDim = 0;
            nWid = 0;

            if (!File.Exists(strFile))
                return new ScoreCollection();

            m_log.WriteLine("Loading pre-run data from '" + strFile + "'.");

            using (FileStream fs = File.OpenRead(strFile))
            using (BinaryReader br = new BinaryReader(fs))
            {
                nDim = br.ReadInt32();
                nWid = br.ReadInt32();

                return ScoreCollection.Load(br);
            }
        }

        /// <summary>
        /// Returns the dataset descriptor of the dynamic dataset produced by the Gym.
        /// </summary>
        /// <param name="dt">Specifies the data-type to use.</param>
        /// <param name="log">Optionally, specifies the output log to use (default = <i>null</i>).</param>
        /// <returns>The dataset descriptor is returned.</returns>
        public DatasetDescriptor GetDataset(DATA_TYPE dt, Log log = null)
        {
            if (dt == DATA_TYPE.DEFAULT)
                dt = DATA_TYPE.VALUES;

            if (dt != DATA_TYPE.VALUES)
            {
                if (log == null)
                    log = m_log;

                if (log != null)
                    log.WriteLine("WARNING: This gym only supports the VALUE type, the datatype will be changed to VALUE.");
                else
                    throw new Exception("This gym only supports the VALUE type.");

                dt = DATA_TYPE.VALUES;
            }

            int nC = 1;
            int nH = 1;
            int nW = 0;

            SourceDescriptor srcTrain = new SourceDescriptor((int)GYM_DS_ID.MODEL, Name + ".training", nW, nH, nC, false, false);
            SourceDescriptor srcTest = new SourceDescriptor((int)GYM_SRC_TEST_ID.MODEL, Name + ".testing", nW, nH, nC, false, false);
            DatasetDescriptor ds = new DatasetDescriptor((int)GYM_SRC_TRAIN_ID.MODEL, Name, null, null, srcTrain, srcTest, "ModelGym", "Model Gym", null, GYM_TYPE.DATA);

            m_dt = dt;

            return ds;
        }

        /// <summary>
        /// Converts the output values into the native type used by the Gym during queries.
        /// </summary>
        /// <param name="stage">Specifies the stage under which the conversion is run.</param>
        /// <param name="nN">Specifies the number of outputs.</param>
        /// <param name="rg">Specifies the raw output data.</param>
        /// <param name="type">Returns the output type.</param>
        /// <returns>The converted output data is returned in a byte stream.</returns>
        /// <remarks>Note: Currently, only data gym's implement this function.</remarks>
        public byte[] ConvertOutput(Stage stage, int nN, float[] rg, out string type)
        {
            type = "String";

            Dictionary<int, LabelDescriptor> rgLabels = new Dictionary<int, LabelDescriptor>();
            foreach (LabelDescriptor lbl in m_ds.TrainingSource.Labels)
            {
                if (!rgLabels.ContainsKey(lbl.ActiveLabel))
                    rgLabels.Add(lbl.ActiveLabel, lbl);
            }

            string str = "";
            for (int i = 0; i < rg.Length; i++)
            {
                int nLabel = (int)rg[i];

                if (rgLabels.ContainsKey(nLabel))
                    str += rgLabels[nLabel].Name;
                else
                    str += "n/a";

                str += ",";
            }

            str = str.TrimEnd(',');

            using (MemoryStream ms = new MemoryStream())
            {
                foreach (char ch in str)
                {
                    ms.WriteByte((byte)ch);
                }

                ms.WriteByte(0);

                return ms.ToArray();
            }
        }

        /// <summary>
        /// Get/set the active phase under which the reset and next run.
        /// </summary>
        public Phase ActivePhase
        {
            get { return m_phase; }
            set { m_phase = value; }
        }
    }

    class ScoreCollection : IEnumerable<Score> /** @private */
    {
        List<Score> m_rgItems = new List<Score>();

        public ScoreCollection()
        {
        }

        public float[] Scores
        {
            get { return m_rgItems.Select(p => p.ScoreValue).ToArray(); }
        }

        public float[] Labels
        {
            get { return m_rgItems.Select(p => (float)p.Label).ToArray(); }
        }

        public float[] Data
        {
            get
            {
                float[] rgf = new float[m_rgItems.Count * m_rgItems[0].Results.Count * 2];

                for (int i = 0; i < m_rgItems.Count; i++)
                {
                    for (int j = 0; j < m_rgItems[i].Results.Count; j++)            // results ordered by score (highest first)
                    {
                        int nCount = m_rgItems[i].Results.Count;
                        int nIdx = i * nCount * 2;
                        rgf[nIdx + j] = m_rgItems[i].Results[j].Item1;              // label
                        rgf[nIdx + nCount + j] = m_rgItems[i].Results[j].Item2;     // score
                    }
                }

                return rgf;
            }
        }

        public int Count
        {
            get { return m_rgItems.Count; }
        }

        public Score this[int nIdx]
        {
            get { return m_rgItems[nIdx]; }
            set { m_rgItems[nIdx] = value; }
        }

        public void Add(Score r)
        {
            m_rgItems.Add(r);
        }

        public void Clear()
        {
            m_rgItems.Clear();
        }

        public int FindIndexAt(DateTime dt, int nCount)
        {
            int nIdx = m_rgItems.Count - nCount;

            for (int i = m_rgItems.Count - 1; i >= 0; i--)
            {
                if (m_rgItems[i].TimeStamp >= dt)
                {
                    if (i < nIdx)
                        return i;
                    else
                        return nIdx;
                }
            }

            return nIdx;
        }

        public ScoreCollection CopyFrom(int nStartIdx, int nCount)
        {
            ScoreCollection col = new ScoreCollection();

            for (int i = 0; i < nCount; i++)
            {
                if (nStartIdx + i < m_rgItems.Count)
                    col.Add(m_rgItems[nStartIdx + i]);
            }

            return col;
        }

        public IEnumerator<Score> GetEnumerator()
        {
            return m_rgItems.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return m_rgItems.GetEnumerator();
        }

        public static ScoreCollection Load(BinaryReader br)
        {
            ScoreCollection col = new ScoreCollection();
            int nCount = br.ReadInt32();

            for (int i = 0; i < nCount; i++)
            {
                col.Add(Score.Load(br));
            }

            return col;
        }

        public void Save(BinaryWriter bw)
        {
            bw.Write(m_rgItems.Count);

            for (int i = 0; i < m_rgItems.Count; i++)
            {
                m_rgItems[i].Save(bw);
            }
        }
    }

    class Score /** @private */
    {
        int m_nIdx;
        DateTime m_dt;
        int m_nLabel;
        float m_fScore;
        List<Tuple<int, float>> m_rgResults = new List<Tuple<int, float>>();

        public Score(DateTime dt, int nIdx, ResultCollection res)
        {
            m_nIdx = nIdx;
            m_dt = dt;
            m_nLabel = res.DetectedLabel;
            m_fScore = (float)res.DetectedLabelOutput;

            foreach (Result res1 in res.ResultsSorted)
            {
                m_rgResults.Add(new Tuple<int, float>(res1.Label, (float)res1.Score));
            }
        }

        public Score(DateTime dt, int nIdx, int nLabel, float fScore, List<Tuple<int, float>> rgRes)
        {
            m_nIdx = nIdx;
            m_dt = dt;
            m_nLabel = nLabel;
            m_fScore = fScore;
            m_rgResults = rgRes;
        }

        public List<Tuple<int, float>> Results
        {
            get { return m_rgResults; }
        }

        public int Index
        {
            get { return m_nIdx; }
        }

        public int Label
        {
            get { return m_nLabel; }
        }

        public DateTime TimeStamp
        {
            get { return m_dt; }
        }

        public float ScoreValue
        {
            get { return m_fScore; }
        }

        public static Score Load(BinaryReader br)
        {
            int nIdx = br.ReadInt32();
            long lTime = br.ReadInt64();
            int nLabel = br.ReadInt32();
            float fScore = br.ReadSingle();

            List<Tuple<int, float>> rgResults = new List<Tuple<int, float>>();
            int nCount = br.ReadInt32();

            for (int i = 0; i < nCount; i++)
            {
                int nLabel1 = br.ReadInt32();
                float fScore1 = br.ReadSingle();
                rgResults.Add(new Tuple<int, float>(nLabel1, fScore1));
            }

            return new Score(DateTime.FromFileTime(lTime), nIdx, nLabel, fScore, rgResults);
        }

        public void Save(BinaryWriter bw)
        {
            bw.Write(m_nIdx);
            bw.Write(m_dt.ToFileTime());
            bw.Write(m_nLabel);
            bw.Write(m_fScore);

            bw.Write(m_rgResults.Count);

            for (int i = 0; i < m_rgResults.Count; i++)
            {
                bw.Write(m_rgResults[i].Item1);
                bw.Write(m_rgResults[i].Item2);
            }
        }
    }

    class ModelDataState : State /** @private */
    {
        SimpleDatum m_sd = null;

        public ModelDataState()
        {
        }

        public ModelDataState(ModelDataState s)
        {
            m_sd = s.m_sd;
        }

        public override State Clone()
        {
            ModelDataState data = new ModelDataState(this);
            data.SetData(m_sd);
            return data;
        }

        public void SetData(SimpleDatum sd)
        {
            m_sd = sd;
        }

        public override SimpleDatum GetData(bool bNormalize, out int nDataLen)
        {
            nDataLen = (m_sd == null) ? 0 : m_sd.ItemCount;
            return m_sd;
        }
    }
}
