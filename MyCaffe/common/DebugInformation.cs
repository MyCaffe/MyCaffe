using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyCaffe.layers;

namespace MyCaffe.common
{
    /// <summary>
    /// The DebugInformation contains information used to help debug the Layers of a Net
    /// while it is training.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class DebugInformation<T> : IDisposable
    {
        string m_strName;
        double m_dfExtraInfo = 0;
        Blob<T> m_blobWork;
        bool m_bDetectNans = false;
        List<LayerDebugInformation<T>> m_rgLayerInfo = new List<LayerDebugInformation<T>>();

        /// <summary>
        /// The DebugInformation constructor.
        /// </summary>
        /// <param name="strName">Specifies a name for the DebugInformation.</param>
        /// <param name="work">Specifies a workspace Blob.</param>
        /// <param name="bDetectNans">Specifies whether or not to detect Nan's in the data.</param>
        public DebugInformation(string strName, Blob<T> work, bool bDetectNans)
        {
            m_strName = strName;
            m_blobWork = work;
            m_bDetectNans = bDetectNans;
        }

        /// <summary>
        /// Releases the memory (GPU and Host) used by the DebugInformation including the Workspace.
        /// </summary>
        public void Dispose()
        {
            if (m_blobWork != null)
            {
                m_blobWork.Dispose();
                m_blobWork = null;
            }
        }

        /// <summary>
        /// Adds a Layer and its Bottom and Top blob collections.
        /// </summary>
        /// <param name="layer">Specifies the Layer.</param>
        /// <param name="colBottomBlobs">Specifies the Bottom Blobs flowing into the Layer.</param>
        /// <param name="colTopBlobs">Specifies the Top Blobs flowing out of the Layer.</param>
        public void Add(Layer<T> layer, BlobCollection<T> colBottomBlobs, BlobCollection<T> colTopBlobs)
        {
            m_rgLayerInfo.Add(new LayerDebugInformation<T>(layer, colBottomBlobs, colTopBlobs, m_blobWork, m_bDetectNans));
        }

        /// <summary>
        /// Returns the name of the DebugInformation.
        /// </summary>
        public string Name
        {
            get { return m_strName; }
        }

        /// <summary>
        /// Get/set extra information about the DebugInformation.
        /// </summary>
        public double ExtraInfo
        {
            get { return m_dfExtraInfo; }
            set { m_dfExtraInfo = value; }
        }

        /// <summary>
        /// Returns an array of LayerDebugInformation corresponding to each Layer added.
        /// </summary>
        public List<LayerDebugInformation<T>> LayerInfoList
        {
            get { return m_rgLayerInfo; }
        }

        /// <summary>
        /// Compares this DebugInformation to another.
        /// </summary>
        /// <param name="dbg">Specifies the other DebugInformation to compare to.</param>
        /// <returns>If the two DebugInformations are the same, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool Compare(DebugInformation<T> dbg)
        {
            if (dbg.Name != m_strName)
                return false;

            if (dbg.m_rgLayerInfo.Count != m_rgLayerInfo.Count)
                return false;

            for (int i = 0; i < m_rgLayerInfo.Count; i++)
            {
                if (!dbg.m_rgLayerInfo[i].Compare(m_rgLayerInfo[i]))
                    return false;
            }

            return true;
        }

        /// <summary>
        /// Searches for the first NaN within any of the Layers.
        /// </summary>
        /// <param name="strType">Returns the Layer type for which a NaN was detected (if any).</param>
        /// <returns>If found, the name of the Blob in which a NaN was detected is returned, otherwise <i>null</i> is returned.</returns>
        public string DetectFirstNaN(out string strType)
        {
            for (int i = 0; i< m_rgLayerInfo.Count; i++)
            {
                LayerDebugInformation<T> dbg = m_rgLayerInfo[i];
                string strBlobNan = dbg.DetectFirstNaN(out strType);

                if (strBlobNan != null)
                    return strBlobNan;
            }

            strType = null;
            return null;
        }

        /// <summary>
        /// Searches for the last NaN within any of the Layers.
        /// </summary>
        /// <param name="strType">Returns the Layer type for which a NaN was detected (if any).</param>
        /// <returns>If found, the name of the Blob in which a NaN was detected is returned, otherwise <i>null</i> is returned.</returns>
        public string DetectLastNaN(out string strType)
        {
            for (int i = m_rgLayerInfo.Count - 1; i >= 0; i--)
            {
                LayerDebugInformation<T> dbg = m_rgLayerInfo[i];
                string strBlobNan = dbg.DetectLastNaN(out strType);

                if (strBlobNan != null)
                    return strBlobNan;
            }

            strType = null;
            return null;
        }

        /// <summary>
        /// Returns a string representation of the DebugInformation.
        /// </summary>
        /// <returns>The string representation is returned.</returns>
        public override string ToString()
        {
            return m_strName;
        }
    }

    /// <summary>
    /// The LayerDebugInformation describes debug information relating to a given Layer in the Net.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class LayerDebugInformation<T>
    {
        string m_strName;
        string m_strType;
        double m_dfForwardTiming = 0;
        double m_dfForwardTimingAve = 0;
        double m_dfBackwardTiming = 0;
        double m_dfBackwardTimingAve = 0;
        List<BlobDebugInformation<T>> m_rgBlobInfo = new List<BlobDebugInformation<T>>();

        /// <summary>
        /// The LayerDebugInformation constructor.
        /// </summary>
        /// <param name="layer">Specifies the Layer.</param>
        /// <param name="colBottom">Specifies the Bottom blobs flowing into the Layer.</param>
        /// <param name="colTop">Specifies the Top blobs flowing out of the Layer.</param>
        /// <param name="work">Specifies the Workspace data.</param>
        /// <param name="bDetectNans">Specifies whether or not to detect Nan's in the data.</param>
        public LayerDebugInformation(Layer<T> layer, BlobCollection<T> colBottom, BlobCollection<T> colTop, Blob<T> work, bool bDetectNans)
        {
            m_strName = layer.layer_param.name;
            m_strType = layer.type.ToString();
            m_dfForwardTiming = layer.forward_timing;
            m_dfForwardTimingAve = layer.forward_timing_average;
            m_dfBackwardTiming = layer.backward_timing;
            m_dfBackwardTimingAve = layer.backward_timing_average;

            foreach (Blob<T> b in colBottom)
            {
                m_rgBlobInfo.Add(new BlobDebugInformation<T>(b, work, BlobDebugInformation<T>.BLOBTYPE.DATA, BlobDebugInformation<T>.LOCATION.BOTTOM, bDetectNans));
            }

            foreach (Blob<T> b in colTop)
            {
                m_rgBlobInfo.Add(new BlobDebugInformation<T>(b, work, BlobDebugInformation<T>.BLOBTYPE.DATA, BlobDebugInformation<T>.LOCATION.TOP, bDetectNans));
            }

            foreach (Blob<T> b in layer.blobs)
            {
                m_rgBlobInfo.Add(new BlobDebugInformation<T>(b, work, BlobDebugInformation<T>.BLOBTYPE.PARAM, BlobDebugInformation<T>.LOCATION.NONE, bDetectNans));
            }

            foreach (Blob<T> b in layer.internal_blobs)
            {
                m_rgBlobInfo.Add(new BlobDebugInformation<T>(b, work, BlobDebugInformation<T>.BLOBTYPE.INTERNAL, BlobDebugInformation<T>.LOCATION.NONE, bDetectNans));
            }
        }

        /// <summary>
        /// Returns the name of the Layer managed by the LayerDebugInformation.
        /// </summary>
        public string Name
        {
            get { return m_strName; }
        }

        /// <summary>
        /// Returns the type of the Layer managed by the LayerDebugInformation.
        /// </summary>
        public string Type
        {
            get { return m_strType; }
        }

        /// <summary>
        /// Returns the timing of the last forward pass made by the Layer.
        /// </summary>
        public double ForwardTiming
        {
            get { return m_dfForwardTiming; }
        }

        /// <summary>
        /// Returns the average timing of the forward passes made by the Layer.
        /// </summary>
        public double ForwardTimingAverage
        {
            get { return m_dfForwardTimingAve; }
        }

        /// <summary>
        /// Returns the timing of the last backward pass made by the Layer.
        /// </summary>
        public double BackwardTiming
        {
            get { return m_dfBackwardTiming; }
        }

        /// <summary>
        /// Returns the average timing of the backward passes made by the Layer.
        /// </summary>
        public double BackwardTimingAverage
        {
            get { return m_dfBackwardTimingAve; }
        }

        /// <summary>
        /// Returns the collection of BlobDebugInformation for the Layer.
        /// </summary>
        public List<BlobDebugInformation<T>> BlobInfoList
        {
            get { return m_rgBlobInfo; }
        }

        /// <summary>
        /// Compare this LayerDebugInformation to another.
        /// </summary>
        /// <param name="dbg">Specifies the other LayerDebugInformation to compare to.</param>
        /// <returns>If the two LayerDebugInformation's are the same, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool Compare(LayerDebugInformation<T> dbg)
        {
            if (dbg.Name != m_strName)
                return false;

            if (dbg.Type != m_strType)
                return false;

            if (dbg.m_rgBlobInfo.Count != m_rgBlobInfo.Count)
                return false;

            for (int i = 0; i < m_rgBlobInfo.Count; i++)
            {
                if (!dbg.m_rgBlobInfo[i].Compare(m_rgBlobInfo[i]))
                    return false;
            }

            return true;
        }

        /// <summary>
        /// Searches for the first NaN within the Layer.
        /// </summary>
        /// <param name="strType">Returns the Layer type for which a NaN was detected (if any).</param>
        /// <returns>If found, the name of the Blob in which a NaN was detected is returned, otherwise <i>null</i> is returned.</returns>
        public string DetectFirstNaN(out string strType)
        {
            for (int i = 0; i < m_rgBlobInfo.Count; i++)
            {
                BlobDebugInformation<T> dbg = m_rgBlobInfo[i];
                string strBlobNan = dbg.DetectFirstNaN(out strType);

                if (strBlobNan != null)
                    return strBlobNan;
            }

            strType = null;

            return null;
        }

        /// <summary>
        /// Searches for the last NaN within the Layer.
        /// </summary>
        /// <param name="strType">Returns the Layer type for which a NaN was detected (if any).</param>
        /// <returns>If found, the name of the Blob in which a NaN was detected is returned, otherwise <i>null</i> is returned.</returns>
        public string DetectLastNaN(out string strType)
        {
            for (int i=m_rgBlobInfo.Count-1; i>=0; i--)
            {
                BlobDebugInformation<T> dbg = m_rgBlobInfo[i];
                string strBlobNan = dbg.DetectLastNaN(out strType);

                if (strBlobNan != null)
                    return strBlobNan;
            }

            strType = null;

            return null;
        }

        /// <summary>
        /// Returns the string representation of the LayerDebugInformation.
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            return m_strName + " (" + m_strType + ")";
        }
    }

    /// <summary>
    /// The BlobDebugInformation describes debug information relating to a given Blob in a given Layer.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class BlobDebugInformation<T>
    {
        BLOBTYPE m_type;
        LOCATION m_location;
        string m_strName;
        string m_strSize;
        double m_dfDataMin;
        double m_dfDataMax;
        double m_dfDiffMin;
        double m_dfDiffMax;
        double m_dfDataNanCount = 0;
        double m_dfDataInfCount = 0;
        double m_dfDiffNanCount = 0;
        double m_dfDiffInfCount = 0;

        /// <summary>
        /// Defines the Blob type.
        /// </summary>
        public enum BLOBTYPE
        {
            /// <summary>
            /// The Blob contains Data.
            /// </summary>
            DATA,
            /// <summary>
            /// The Blob is a parameter blob (e.g. learnable parameters).
            /// </summary>
            PARAM,
            /// <summary>
            /// The Blob is an internal blob to the Layer.
            /// </summary>
            INTERNAL
        }

        /// <summary>
        /// Defines the location of the Blob.
        /// </summary>
        public enum LOCATION
        {
            /// <summary>
            /// No location is specified.
            /// </summary>
            NONE,
            /// <summary>
            /// The Blob is a Bottom Blob.
            /// </summary>
            BOTTOM,
            /// <summary>
            /// The Blob is a Top Blob.
            /// </summary>
            TOP
        }

        /// <summary>
        /// The BlobDebugInformation constructor.
        /// </summary>
        /// <param name="b">Specifies the Blob.</param>
        /// <param name="work">Specifies the workspace.</param>
        /// <param name="type">Specifies the Blob type.</param>
        /// <param name="loc">Specifies the Blob location.</param>
        /// <param name="bDetectNans">Specifies whether or not to detect Nan's in the data.</param>
        public BlobDebugInformation(Blob<T> b, Blob<T> work, BLOBTYPE type, LOCATION loc = LOCATION.NONE, bool bDetectNans = false)
        {
            m_type = type;
            m_location = loc;
            m_strName = b.Name;
            m_strSize = b.ToSizeString();

            Tuple<double, double, double, double> data = b.minmax_data(work, bDetectNans);
            Tuple<double, double, double, double> diff = b.minmax_diff(work, bDetectNans);

            m_dfDataMin = data.Item1;
            m_dfDataMax = data.Item2;
            m_dfDataNanCount = data.Item3;
            m_dfDataInfCount = data.Item4;
            m_dfDiffMin = diff.Item1;
            m_dfDiffMax = diff.Item2;
            m_dfDiffNanCount = diff.Item3;
            m_dfDiffInfCount = diff.Item4;
        }

        /// <summary>
        /// Returns the name of the Blob.
        /// </summary>
        public string Name
        {
            get { return m_strName; }
        }

        /// <summary>
        /// Returns the size of the Blob.
        /// </summary>
        public string Size
        {
            get { return m_strSize; }
        }

        /// <summary>
        /// Returns the minimum value of the Blob data.
        /// </summary>
        public double DataMinValue
        {
            get
            {
                if (DiffNanCount > 0)
                    return double.NaN;

                if (DiffInfCount > 0)
                    return double.PositiveInfinity;

                return m_dfDataMin;
            }
        }

        /// <summary>
        /// Returns the maximum value of the Blob data.
        /// </summary>
        public double DataMaxValue
        {
            get
            {
                if (DiffNanCount > 0)
                    return double.NaN;

                if (DiffInfCount > 0)
                    return double.PositiveInfinity;

                return m_dfDataMax;
            }
        }

        /// <summary>
        /// Returns the number of nans detected in the Blob data.
        /// </summary>
        public double DataNanCount
        {
            get { return m_dfDataNanCount; }
        }

        /// <summary>
        /// Returns the number of infinity values detected in the Blob data.
        /// </summary>
        public double DataInfCount
        {
            get { return m_dfDataInfCount; }
        }

        /// <summary>
        /// Returns the minimum value of the Blob diff.
        /// </summary>
        public double DiffMinValue
        {
            get
            {
                if (DiffNanCount > 0)
                    return double.NaN;

                if (DiffInfCount > 0)
                    return double.PositiveInfinity;

                return m_dfDiffMin;
            }
        }

        /// <summary>
        /// Returns the maximum value of the Blob diff.
        /// </summary>
        public double DiffMaxValue
        {
            get
            {
                if (DiffNanCount > 0)
                    return double.NaN;

                if (DiffInfCount > 0)
                    return double.PositiveInfinity;

                return m_dfDiffMax;
            }
        }

        /// <summary>
        /// Returns the number of nans detected in the Blob diff.
        /// </summary>
        public double DiffNanCount
        {
            get { return m_dfDiffNanCount; }
        }

        /// <summary>
        /// Returns the number of infinity values detected in the Blob diff.
        /// </summary>
        public double DiffInfCount
        {
            get { return m_dfDiffInfCount; }
        }

        /// <summary>
        /// Returns the Blob type.
        /// </summary>
        public BLOBTYPE BlobType
        {
            get { return m_type; }
        }

        /// <summary>
        /// Returns the Blob location.
        /// </summary>
        public LOCATION BlobLocation
        {
            get { return m_location; }
        }

        /// <summary>
        /// Compares this BlobDebugInformation to another one.
        /// </summary>
        /// <param name="dbg">Specifies the other BlobDebugInformation to compare to.</param>
        /// <returns>If the two BlobDebugInformation's are the same, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool Compare(BlobDebugInformation<T> dbg)
        {
            if (dbg.Name != m_strName)
                return false;

            if (dbg.Size != m_strSize)
                return false;

            if (dbg.BlobLocation != m_location)
                return false;

            if (dbg.BlobType != m_type)
                return false;

            if (dbg.DataMinValue != m_dfDataMin)
                return false;

            if (dbg.DataMaxValue != m_dfDataMax)
                return false;

            if (dbg.DataNanCount != m_dfDataNanCount)
                return false;

            if (dbg.DataInfCount != m_dfDiffInfCount)
                return false;

            if (dbg.DiffMinValue != m_dfDiffMin)
                return false;

            if (dbg.DiffMaxValue != m_dfDiffMax)
                return false;

            if (dbg.DiffNanCount != m_dfDiffNanCount)
                return false;

            if (dbg.DiffInfCount != m_dfDiffInfCount)
                return false;

            return true;
        }

        /// <summary>
        /// Search for the first NaN in the Blob.
        /// </summary>
        /// <param name="strType">Returns the type of the Blob.</param>
        /// <returns>If a NaN is found, the name of the Blob is returned, otherwise <i>null</i> is returned.</returns>
        public string DetectFirstNaN(out string strType)
        {
            if (!isValid(m_dfDataMin))
            {
                strType = "data min";
                return m_strName;
            }

            if (!isValid(m_dfDataMax))
            {
                strType = "data max";
                return m_strName;
            }

            if (m_dfDataNanCount > 0)
            {
                strType = "data nan";
                return m_strName;
            }

            if (m_dfDataInfCount > 0)
            {
                strType = "data inf";
                return m_strName;
            }

            if (!isValid(m_dfDiffMin))
            {
                strType = "diff min";
                return m_strName;
            }

            if (!isValid(m_dfDiffMax))
            {
                strType = "diff max";
                return m_strName;
            }

            if (m_dfDiffNanCount > 0)
            {
                strType = "diff nan";
                return m_strName;
            }

            if (m_dfDiffInfCount > 0)
            {
                strType = "diff inf";
                return m_strName;
            }

            strType = null;
            return null;
        }

        /// <summary>
        /// Search for the last NaN in the Blob.
        /// </summary>
        /// <param name="strType">Returns the type of the Blob.</param>
        /// <returns>If a NaN is found, the name of the Blob is returned, otherwise <i>null</i> is returned.</returns>
        public string DetectLastNaN(out string strType)
        {
            if (!isValid(m_dfDiffMin))
            {
                strType = "diff min";
                return m_strName;
            }

            if (!isValid(m_dfDiffMax))
            {
                strType = "diff max";
                return m_strName;
            }

            if (m_dfDiffNanCount > 0)
            {
                strType = "diff nan";
                return m_strName;
            }

            if (m_dfDiffInfCount > 0)
            {
                strType = "diff inf";
                return m_strName;
            }

            if (!isValid(m_dfDataMin))
            {
                strType = "data min";
                return m_strName;
            }

            if (!isValid(m_dfDataMax))
            {
                strType = "data max";
                return m_strName;
            }

            if (m_dfDataNanCount > 0)
            {
                strType = "data nan";
                return m_strName;
            }

            if (m_dfDataInfCount > 0)
            {
                strType = "data inf";
                return m_strName;
            }

            strType = null;
            return null;
        }

        private bool isValid(double df)
        {
            if (double.IsNaN(df))
                return false;

            if (double.IsInfinity(df))
                return false;

            return true;
        }

        /// <summary>
        /// Returns a string representation of the BlobDebugInformation.
        /// </summary>
        /// <returns>The string representation is returned.</returns>
        public override string ToString()
        {
            return m_strName + " (" + m_strSize + ")";
        }
    }
}
