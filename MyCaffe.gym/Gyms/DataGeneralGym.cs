using MyCaffe.basecode;
using MyCaffe.basecode.descriptors;
using MyCaffe.db.stream;
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

namespace MyCaffe.gym
{
    /// <summary>
    /// The DataGeneral Gym provides access to the MyCaffe Streaming Database with GENERAL query types.
    /// </summary>
    public class DataGeneralGym : IXMyCaffeGymData, IDisposable
    {
        string m_strName = "DataGeneral";
        Log m_log;
        CryptoRandom m_random;
        Dictionary<string, int> m_rgActions = new Dictionary<string, int>();
        DATA_TYPE m_dt = DATA_TYPE.BLOB;
        MyCaffeStreamDatabase m_db;

        /// <summary>
        /// The constructor.
        /// </summary>
        public DataGeneralGym()
        {
        }

        /// <summary>
        /// Release all resources used.
        /// </summary>
        public void Dispose()
        {
        }

        /// <summary>
        /// Initialize the gym with the specified properties.
        /// </summary>
        /// <param name="log">Specifies the output log to use.</param>
        /// <param name="properties">Specifies the properties containing Gym specific initialization parameters.</param>
        /// <remarks>
        /// The DataGym uses the following initialization properties.
        /// 
        /// 'DbSettings' - returns the database settings based on the QUERY_TYPE used.
        /// 'DbSchema' - returns the database schema.
        /// </remarks>
        public void Initialize(Log log, PropertySet properties)
        {
            m_log = log;
            m_random = new CryptoRandom();
            m_db = new MyCaffeStreamDatabase(m_log);
            m_db.Initialize(QUERY_TYPE.GENERAL, properties.ToString());
            m_db.Reset();
        }

        /// <summary>
        /// Create a new copy of the gym.
        /// </summary>
        /// <param name="properties">Optionally, specifies the properties to initialize the new copy with.</param>
        /// <returns>The new Gym copy is returned.</returns>
        public IXMyCaffeGym Clone(PropertySet properties = null)
        {
            DataGeneralGym gym = new DataGeneralGym();

            if (properties != null)
                gym.Initialize(m_log, properties);

            return gym;
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
            get { return new DATA_TYPE[] { DATA_TYPE.BLOB }; }
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
            m_db.Shutdown();
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
        /// <returns>A tuple containing state data, the reward, and the done state is returned.</returns>
        public Tuple<State, double, bool> Reset()
        {
            m_db.Reset();
            return Step(-1);
        }

        /// <summary>
        /// Step the gym one step in the data.
        /// </summary>
        /// <param name="nAction">Specifies the action to run on the gym.</param>
        /// <returns>A tuple containing state data, the reward, and the done state is returned.</returns>
        public Tuple<State, double, bool> Step(int nAction)
        {
            DataState data = new DataState();

            SimpleDatum sd = m_db.Query(1000);
            data.SetData(sd);

            return new Tuple<State, double, bool>(data, 0, (sd == null) ? true : false);
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
                dt = DATA_TYPE.BLOB;

            if (dt != DATA_TYPE.BLOB)
            {
                if (log == null)
                    log = m_log;

                if (log != null)
                    log.WriteLine("WARNING: This gym only supports the BLOB type, the datatype will be changed to BLOB.");
                else
                    throw new Exception("This gym only supports the BLOB type.");

                dt = DATA_TYPE.BLOB;
            }

            int nC = 1;
            int nH = 1;
            int nW = 0;

            SourceDescriptor srcTrain = new SourceDescriptor((int)GYM_DS_ID.DATAGENERAL, Name + ".training", nW, nH, nC, false, false);
            SourceDescriptor srcTest = new SourceDescriptor((int)GYM_SRC_TEST_ID.DATAGENERAL, Name + ".testing", nW, nH, nC, false, false);
            DatasetDescriptor ds = new DatasetDescriptor((int)GYM_SRC_TRAIN_ID.DATAGENERAL, Name, null, null, srcTrain, srcTest, "DataGym", "Data Gym", null, GYM_TYPE.DATA);

            m_dt = dt;

            return ds;
        }

        /// <summary>
        /// Converts the output values into the native type used by the Gym during queries.
        /// </summary>
        /// <param name="rg">Specifies the raw output data.</param>
        /// <param name="type">Returns the output type.</param>
        /// <returns>The converted output data is returned in a byte stream.</returns>
        /// <remarks>Note: Currently, only data gym's implement this function.</remarks>
        public byte[] ConvertOutput(float[] rg, out Type type)
        {
            return m_db.ConvertOutput(rg, out type);
        }
    }

    class DataState : State /** @private */
    {
        SimpleDatum m_sd = null;

        public DataState()
        {
        }

        public DataState(DataState s)
        {
            m_sd = s.m_sd;
        }

        public override State Clone()
        {
            DataState data = new DataState(this);
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
