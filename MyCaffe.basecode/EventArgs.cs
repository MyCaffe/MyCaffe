using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using System.Text;

/// <summary>
/// The MyCaffe.basecode contains all generic types used throughout MyCaffe.
/// </summary>

namespace MyCaffe.basecode
{
    /// <summary>
    /// The LogProgressArg is passed as an argument to the Log::OnProgress event.
    /// </summary>
    public class LogProgressArg : EventArgs
    {
        string m_strSrc;
        double m_dfProgress = 0;

        /// <summary>
        /// The LogProgressArg constructor.
        /// </summary>
        /// <param name="strSrc">Specifies the Log source name.</param>
        /// <param name="dfProgress">Specifies the progress value.</param>
        public LogProgressArg(string strSrc, double dfProgress)
        {
            m_strSrc = strSrc;
            m_dfProgress = dfProgress;
        }

        /// <summary>
        /// Returns the Log source name.
        /// </summary>
        public string Source
        {
            get { return m_strSrc; }
        }

        /// <summary>
        /// Returns the progress value.
        /// </summary>
        public double Progress
        {
            get { return m_dfProgress; }
        }
    }

    /// <summary>
    /// The LogArg is passed as an argument to the Log::OnWriteLine event.
    /// </summary>
    public class LogArg : LogProgressArg
    {
        string m_strMsg;
        object m_tag = null;
        bool m_bError;
        bool m_bOverrideEnabled = false;
        bool m_bConsumed = false;
        bool m_bDisable = false;

        /// <summary>
        /// The LogArg constructor.
        /// </summary>
        /// <param name="strSrc">Specifies the Log source name.</param>
        /// <param name="strMsg">Specifies the message written when calling the Log::WriteLine function (which triggers the event).</param>
        /// <param name="dfProgress">Specifies the progress value specifies when setting the Log::Progress value.</param>
        /// <param name="bError">Specifies whether or not the message is the result of a call from Log::WriteError.</param>
        /// <param name="bOverrideEnabled">Specifies whether or not the log override was used.</param>
        /// <param name="bDisable">Specifies whether or not to disable the output of the message (e.g. used internally).</param>
        public LogArg(string strSrc, string strMsg, double dfProgress = 0.0, bool bError = false, bool bOverrideEnabled = false, bool bDisable = false)
            : base(strSrc, dfProgress)
        {
            m_strMsg = strMsg;
            m_bError = bError;
            m_bOverrideEnabled = bOverrideEnabled;
            m_bDisable = bDisable;
        }

        /// <summary>
        /// Specifies whether or not the message has already been consumed.
        /// </summary>
        public bool Consumed
        {
            get { return m_bConsumed; }
            set { m_bConsumed = value; }
        }

        /// <summary>
        /// Specifies whether or not to mark this log entry as disabled so that it is not output.
        /// </summary>
        public bool Disable
        {
            get { return m_bDisable; }
            set { m_bDisable = value; }
        }

        /// <summary>
        /// Returns the message logged.
        /// </summary>
        public string Message
        {
            get { return m_strMsg; }
        }

        /// <summary>
        /// Returns whether or not this is an error message.
        /// </summary>
        public bool Error
        {
            get { return m_bError; }
        }

        /// <summary>
        /// Returns whether or not the override was enabled or not.
        /// </summary>
        public bool OverrideEnabled
        {
            get { return m_bOverrideEnabled; }
        }

#pragma warning disable 1591

        public object Tag /** @private */
        {
            get { return m_tag; }
            set { m_tag = value; }
        }

#pragma warning restore 1591
    }

    /// <summary>
    /// The CalculateImageMeanArgs is passed as an argument to the MyCaffeImageDatabase::OnCalculateImageMean event.
    /// </summary>
    public class CalculateImageMeanArgs : EventArgs
    {
        SimpleDatum[] m_rgImg;
        SimpleDatum m_mean;
        bool m_bCancelled = false;

        /// <summary>
        /// The CalculateImageMeanArgs constructor.
        /// </summary>
        /// <param name="rgImg">Specifies the list of images from which the mean should be calculated.</param>
        public CalculateImageMeanArgs(SimpleDatum[] rgImg)
        {
            m_rgImg = rgImg;
        }

        /// <summary>
        /// Specifies the list of images from which the mean should be calculated.
        /// </summary>
        public SimpleDatum[] Images
        {
            get { return m_rgImg; }
        }

        /// <summary>
        /// Get/set the image mean calculated from the <i>Images</i>.
        /// </summary>
        public SimpleDatum ImageMean
        {
            get { return m_mean; }
            set { m_mean = value; }
        }

        /// <summary>
        /// Get/set a flag indicating to cancel the operation.
        /// </summary>
        public bool Cancelled
        {
            get { return m_bCancelled; }
            set { m_bCancelled = value; }
        }
    }

    /// <summary>
    /// The OverrideProjectArgs is passed as an argument to the OnOverrideModel and OnOverrideSolver events fired by the ProjectEx class.
    /// </summary>
    public class OverrideProjectArgs : EventArgs
    {
        RawProto m_proto;

        /// <summary>
        /// The OverrideProjectArgs constructor.
        /// </summary>
        /// <param name="proto">Specifies the RawProtot.</param>
        public OverrideProjectArgs(RawProto proto)
        {
            m_proto = proto;
        }

        /// <summary>
        /// Get/set the RawProto used.
        /// </summary>
        public RawProto Proto
        {
            get { return m_proto; }
            set { m_proto = value; }
        }
    }

    /// <summary>
    /// The LossArgs contains the loss values for a given batch.
    /// </summary>
    public class LossArgs : EventArgs
    {
        List<int> m_rgShape;
        float[] m_rgfData;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="nCount">Specifies the batch size used.</param>
        /// <param name="rgShape">Specifies the shape of the data.</param>
        public LossArgs(int nCount, List<int> rgShape)
        {
            m_rgShape = rgShape;
            m_rgfData = new float[nCount];
        }

        /// <summary>
        /// Specifies the shape of the data.
        /// </summary>
        public List<int> Shape
        {
            get { return m_rgShape; }
        }

        /// <summary>
        /// Specifies the loss values for a given batch.
        /// </summary>
        public float[] Data
        {
            get { return m_rgfData; }
        }
    }
}
