using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param.ssd
{
    /// <summary>
    /// Specifies the parameters for the VideoDataLayer.
    /// </summary>
    /// <remarks>
    /// @see [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg, 2016.
    /// @see [GitHub: SSD: Single Shot MultiBox Detector](https://github.com/weiliu89/caffe/tree/ssd), by weiliu89/caffe, 2016
    /// </remarks>
    public class VideoDataParameter : LayerParameterBase 
    {
        VideoType m_videoType = VideoType.WEBCAM;
        int m_nDeviceID = 0;
        string m_strVideoFile;
        uint m_nSkipFrames = 0;
        uint m_nVideoWidth = 400;
        uint m_nVideoHeight = 300;

        /// <summary>
        /// Defines the video type.
        /// </summary>
        public enum VideoType
        {
            /// <summary>
            /// Specifies to use the web cam if available.
            /// </summary>
            WEBCAM = 0,
            /// <summary>
            /// Specifies to use a video file.
            /// </summary>
            VIDEO = 1
        }

        /** @copydoc LayerParameterBase */
        public VideoDataParameter()
        {
        }

        /// <summary>
        /// Specifies the video type (default = WEBCAM).
        /// </summary>
        [Description("Specifies the video type (default = WEBCAM).")]
        public VideoType video_type
        {
            get { return m_videoType; }
            set { m_videoType = value; }
        }

        /// <summary>
        /// Specifies the device ID (default = 0).
        /// </summary>
        [Description("Specifies the device ID (default = 0).")]
        public int device_id
        {
            get { return m_nDeviceID; }
            set { m_nDeviceID = value; }
        }

        /// <summary>
        /// Specifies the video file when using the VIDEO type.
        /// </summary>
        [Description("Specifies the video file when using the VIDEO type.")]
        public string video_file
        {
            get { return m_strVideoFile; }
            set { m_strVideoFile = value; }
        }

        /// <summary>
        /// Optionally, specifies the number of frames to be skipped before processing a frame (default = 0).
        /// </summary>
        [Description("Optionally, specifies the number of frames to be skipped before processing a frame (default = 0).")]
        public uint skip_frames
        {
            get { return m_nSkipFrames; }
            set { m_nSkipFrames = value; }
        }

        /// <summary>
        /// Optionally, specifies the video width (default = 400).
        /// </summary>
        [Description("Optionally, specifies the video width (default = 400).")]
        public uint video_width
        {
            get { return m_nVideoWidth; }
            set { m_nVideoWidth = value; }
        }

        /// <summary>
        /// Optionally, specifies the video height (default = 300).
        /// </summary>
        [Description("Optionally, specifies the video height (default = 300).")]
        public uint video_height
        {
            get { return m_nVideoHeight; }
            set { m_nVideoHeight = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            VideoDataParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            VideoDataParameter p = (VideoDataParameter)src;

            m_videoType = p.m_videoType;
            m_nDeviceID = p.m_nDeviceID;
            m_strVideoFile = p.m_strVideoFile;
            m_nSkipFrames = p.m_nSkipFrames;
            m_nVideoWidth = p.m_nVideoWidth;
            m_nVideoHeight = p.m_nVideoHeight;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            VideoDataParameter p = new VideoDataParameter();
            p.Copy(this);
            return p;
        }

        /** @copydoc LayerParameterBase::ToProto */
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add(new RawProto("video_type", m_videoType.ToString()));
            rgChildren.Add(new RawProto("device_id", m_nDeviceID.ToString()));

            if (!string.IsNullOrEmpty(m_strVideoFile))
                rgChildren.Add(new RawProto("video_file", m_strVideoFile.ToString()));

            rgChildren.Add(new RawProto("skip_frames", m_nSkipFrames.ToString()));
            rgChildren.Add(new RawProto("video_width", m_nVideoWidth.ToString()));
            rgChildren.Add(new RawProto("video_height", m_nVideoHeight.ToString()));

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static VideoDataParameter FromProto(RawProto rp)
        {
            VideoDataParameter p = new VideoDataParameter();
            string strVal;

            if ((strVal = rp.FindValue("video_type")) != null)
            {
                if (strVal == VideoType.VIDEO.ToString())
                    p.video_type = VideoType.VIDEO;
                else
                    p.video_type = VideoType.WEBCAM;
            }

            if ((strVal = rp.FindValue("device_id")) != null)
                p.device_id = int.Parse(strVal);

            if ((strVal = rp.FindValue("video_file")) != null)
                p.video_file = strVal;

            if ((strVal = rp.FindValue("skip_frames")) != null)
                p.skip_frames = uint.Parse(strVal);

            if ((strVal = rp.FindValue("video_width")) != null)
                p.video_width = uint.Parse(strVal);

            if ((strVal = rp.FindValue("video_height")) != null)
                p.video_height = uint.Parse(strVal);

            return p;
        }
    }
}
