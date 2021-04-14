using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using System.ComponentModel;

/// <summary>
/// The MyCaffe.param.beta parameters are used by the MyCaffe.layer.beta layers.
/// </summary>
/// <remarks>
/// Using parameters within the MyCaffe.layer.beta namespace are used by layers that require the MyCaffe.layers.beta.dll.
/// </remarks>
namespace MyCaffe.param.beta
{
    /// <summary>
    /// Specifies the parameters for the MergeLayer.
    /// </summary>
    public class MergeParameter : LayerParameterBase 
    {
        int m_nCopyAxis = 0;
        int m_nOrderingMajorAxis = 1;
        int m_nCopyCount = 0;
        int m_nCopyDim1 = 0;
        int m_nSrcStartIdx1 = 0;
        int m_nDstStartIdx1 = 0;
        int m_nCopyDim2 = 0;
        int m_nSrcStartIdx2 = 0;
        int m_nDstStartIdx2 = 0;
        int m_nSrcSpatialDimStartIdx1 = 0;
        int m_nDstSpatialDimStartIdx1 = 0;
        int m_nSrcSpatialDimStartIdx2 = 0;
        int m_nDstSpatialDimStartIdx2 = 0;
        int m_nSpatialDimCopyCount = -1;
        int m_nDstSpatialDim = 0;

        /** @copydoc LayerParameterBase */
        public MergeParameter()
        {
        }

        /// <summary>
        /// Specifies axis providing the major ordering (e.g. axis=1 uses axis 1 as the major ordering with axis 0 following).
        /// </summary>
        /// <remarks>
        /// LSTM layers using sequnce ordering have shapes (T,N,...) where T = the sequence and N = the batch.  These layers use ordering_major_axis = 1.
        /// </remarks>
        [Description("Specifies axis providing the major ordering (e.g. axis=1 uses axis 1 as the major ordering with axis 0 following).")]
        public int order_major_axis
        {
            get { return m_nOrderingMajorAxis; }
//            set { m_nOrderingMajorAxis = value; } // Currenly LSTM type ordering only supported with order_major_axis=1.
        }

        /// <summary>
        /// Specifies axis along which the indexing is applied when copying.
        /// </summary>
        /// <remarks>
        /// LSTM layers using sequnce ordering have shapes (T,N,...) where T = the sequence and N = the batch.  These layers use copying_axis = 0.
        /// </remarks>
        [Description("Specifies axis providing the major ordering (e.g. axis=1 uses axis 1 as the major ordering with axis 0 following).")]
        public int copy_axis
        {
            get { return m_nCopyAxis; }
//            set { m_nCopyAxis = value; } // Currently LSTM type ordering only supported with copy_axis=0.
        }

        /// <summary>
        /// Specifies the number of skip copies along the copy_axis to copy (e.g. this is the number of skips to perform and is usually = the batch size).
        /// </summary>
        [Description("Specifies the number of skip copies along the copy_axis to copy (e.g. this is the number of skips to perform and is usually = the batch size).")]
        public int copy_count
        {
            get { return m_nCopyCount; }
            set { m_nCopyCount = value; }
        }

        /// <summary>
        /// Specifies the src start index where copying begins in the first blob in bottom(0).
        /// </summary>
        [Description("Specifies the src start index where copying begins in the first blob in bottom(0).")]
        public int src_start_idx1
        {
            get { return m_nSrcStartIdx1; }
            set { m_nSrcStartIdx1 = value; }
        }

        /// <summary>
        /// Specifies the dst start index where copying begins in the destination blob in top(0).
        /// </summary>
        [Description("Specifies the dst start index where copying begins in the destination blob in top(0).")]
        public int dst_start_idx1
        {
            get { return m_nDstStartIdx1; }
            set { m_nDstStartIdx1 = value; }
        }

        /// <summary>
        /// Specifies the src1 spatial dim start index (only used when > 0).
        /// </summary>
        [Description("Specifies the src1 spatial dim start index (only used when > 0)")]
        public int src_spatialdim_start_idx1
        {
            get { return m_nSrcSpatialDimStartIdx1; }
            set { m_nSrcSpatialDimStartIdx1 = value; }
        }

        /// <summary>
        /// Specifies the dst1 spatial dim start index (only used when > 0).
        /// </summary>
        [Description("Specifies the dst1 spatial dim start index (only used when > 0)")]
        public int dst_spatialdim_start_idx1
        {
            get { return m_nDstSpatialDimStartIdx1; }
            set { m_nDstSpatialDimStartIdx1 = value; }
        }

        /// <summary>
        /// Specifies the src2 spatial dim start index (only used when > 0).
        /// </summary>
        [Description("Specifies the src2 spatial dim start index (only used when > 0)")]
        public int src_spatialdim_start_idx2
        {
            get { return m_nSrcSpatialDimStartIdx2; }
            set { m_nSrcSpatialDimStartIdx2 = value; }
        }

        /// <summary>
        /// Specifies the dst2 spatial dim start index (only used when > 0).
        /// </summary>
        [Description("Specifies the dst2 spatial dim start index (only used when > 0)")]
        public int dst_spatialdim_start_idx2
        {
            get { return m_nDstSpatialDimStartIdx2; }
            set { m_nDstSpatialDimStartIdx2 = value; }
        }

        /// <summary>
        /// Specifies the spatial dim copy count, used when less than the entire spatial dim is to be copied.
        /// </summary>
        [Description("Specifies the spatial dim copy count, used when less than the entire spatial dim is to be copied.")]
        public int spatialdim_copy_count
        {
            get { return m_nSpatialDimCopyCount; }
            set { m_nSpatialDimCopyCount = value; }
        }

        /// <summary>
        /// Specifies the dst spatial dim which if not copied into is set to zero.
        /// </summary>
        [Description("Specifies the dst spatial dim which if not copied into is set to zero.")]
        public int dst_spatialdim
        {
            get { return m_nDstSpatialDim; }
            set { m_nDstSpatialDim = value; }
        }

        /// <summary>
        /// Specifies the dimension (sans the spatial dimension) to be copied (the full copy size = copy_dim * spatial_dim which is calculated using axis dims after the copy axis).
        /// </summary>
        [Description("Specifies the dimension (sans the spatial dimension) to be copied from bottom(0) (the full copy size = copy_dim1 * spatial_dim which is calculated using axis dims after the copy axis).")]
        public int copy_dim1
        {
            get { return m_nCopyDim1; }
            set { m_nCopyDim1 = value; }
        }

        /// <summary>
        /// Specifies the src start index where copying begins in the second input blob in bottom(1).
        /// </summary>
        [Description("Specifies src the start index where copying begins in the second input blob in bottom(1).")]
        public int src_start_idx2
        {
            get { return m_nSrcStartIdx2; }
            set { m_nSrcStartIdx2 = value; }
        }

        /// <summary>
        /// Specifies the dst start index where copying begins for the second copy to dst blob in top(0).
        /// </summary>
        [Description("Specifies the dst start index where copying begins for the second copy to dst blob in top(0).")]
        public int dst_start_idx2
        {
            get { return m_nDstStartIdx2; }
            set { m_nDstStartIdx2 = value; }
        }

        /// <summary>
        /// Specifies the dimension (sans the spatial dimension) to be copied (the full copy size = copy_dim * spatial_dim which is calculated using axis dims after the copy axis).
        /// </summary>
        [Description("Specifies the dimension (sans the spatial dimension) to be copied from bottom(0) (the full copy size = copy_dim1 * spatial_dim which is calculated using axis dims after the copy axis).")]
        public int copy_dim2
        {
            get { return m_nCopyDim2; }
            set { m_nCopyDim2 = value; }
        }

        /// <summary>
        /// Load the parameter from a binary reader.
        /// </summary>
        /// <param name="br">Specifies the binary reader.</param>
        /// <param name="bNewInstance">When <i>true</i> a new instance is created (the default), otherwise the existing instance is loaded from the binary reader.</param>
        /// <returns>Returns an instance of the parameter.</returns>
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            MergeParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /// <summary>
        /// Copy on parameter to another.
        /// </summary>
        /// <param name="src">Specifies the parameter to copy.</param>
        public override void Copy(LayerParameterBase src)
        {
            MergeParameter p = (MergeParameter)src;
            m_nCopyAxis = p.m_nCopyAxis;
            m_nOrderingMajorAxis = p.m_nOrderingMajorAxis;
            m_nCopyCount = p.m_nCopyCount;
            m_nSrcStartIdx1 = p.m_nSrcStartIdx1;
            m_nSrcStartIdx2 = p.m_nSrcStartIdx2;
            m_nDstStartIdx1 = p.m_nDstStartIdx1;
            m_nDstStartIdx2 = p.m_nDstStartIdx2;
            m_nCopyDim1 = p.m_nCopyDim1;
            m_nCopyDim2 = p.m_nCopyDim2;
            m_nSrcSpatialDimStartIdx1 = p.m_nSrcSpatialDimStartIdx1;
            m_nDstSpatialDimStartIdx1 = p.m_nDstSpatialDimStartIdx1;
            m_nSrcSpatialDimStartIdx2 = p.m_nSrcSpatialDimStartIdx2;
            m_nDstSpatialDimStartIdx2 = p.m_nDstSpatialDimStartIdx2;
            m_nSpatialDimCopyCount = p.m_nSpatialDimCopyCount;
            m_nDstSpatialDim = p.m_nDstSpatialDim;
        }

        /// <summary>
        /// Creates a new copy of this instance of the parameter.
        /// </summary>
        /// <returns>A new instance of this parameter is returned.</returns>
        public override LayerParameterBase Clone()
        {
            MergeParameter p = new MergeParameter();
            p.Copy(this);
            return p;
        }

        /// <summary>
        /// Convert the parameter into a RawProto.
        /// </summary>
        /// <param name="strName">Specifies the name to associate with the RawProto.</param>
        /// <returns>The new RawProto is returned.</returns>
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add("copy_axis", copy_axis.ToString());
            rgChildren.Add("order_major_axis", order_major_axis.ToString());
            rgChildren.Add("copy_count", copy_count.ToString());
            rgChildren.Add("src_start_idx1", src_start_idx1.ToString());
            rgChildren.Add("dst_start_idx1", dst_start_idx1.ToString());
            rgChildren.Add("copy_dim1", copy_dim1.ToString());
            rgChildren.Add("src_start_idx2", src_start_idx2.ToString());
            rgChildren.Add("dst_start_idx2", dst_start_idx2.ToString());
            rgChildren.Add("copy_dim2", copy_dim2.ToString());
            rgChildren.Add("src_spatialdim_idx1", src_spatialdim_start_idx1.ToString());
            rgChildren.Add("dst_spatialdim_idx1", dst_spatialdim_start_idx1.ToString());
            rgChildren.Add("src_spatialdim_idx2", src_spatialdim_start_idx2.ToString());
            rgChildren.Add("dst_spatialdim_idx2", dst_spatialdim_start_idx2.ToString());
            rgChildren.Add("spatialdim_copy_count", spatialdim_copy_count.ToString());
            rgChildren.Add("dst_spatialdim", dst_spatialdim.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static MergeParameter FromProto(RawProto rp)
        {
            string strVal;
            MergeParameter p = new MergeParameter();

//            if ((strVal = rp.FindValue("copy_axis")) != null)
//                p.copy_axis = int.Parse(strVal);

//            if ((strVal = rp.FindValue("order_major_axis")) != null)
//                p.order_major_axis = int.Parse(strVal);

            if ((strVal = rp.FindValue("copy_count")) != null)
                p.copy_count = int.Parse(strVal);

            if ((strVal = rp.FindValue("src_start_idx1")) != null)
                p.src_start_idx1 = int.Parse(strVal);

            if ((strVal = rp.FindValue("dst_start_idx1")) != null)
                p.dst_start_idx1 = int.Parse(strVal);

            if ((strVal = rp.FindValue("copy_dim1")) != null)
                p.copy_dim1 = int.Parse(strVal);

            if ((strVal = rp.FindValue("src_start_idx2")) != null)
                p.src_start_idx2 = int.Parse(strVal);

            if ((strVal = rp.FindValue("dst_start_idx2")) != null)
                p.dst_start_idx2 = int.Parse(strVal);

            if ((strVal = rp.FindValue("copy_dim2")) != null)
                p.copy_dim2 = int.Parse(strVal);

            if ((strVal = rp.FindValue("src_spatialdim_start_idx1")) != null)
                p.src_spatialdim_start_idx1 = int.Parse(strVal);

            if ((strVal = rp.FindValue("dst_spatialdim_start_idx1")) != null)
                p.dst_spatialdim_start_idx1 = int.Parse(strVal);

            if ((strVal = rp.FindValue("src_spatialdim_start_idx2")) != null)
                p.src_spatialdim_start_idx2 = int.Parse(strVal);

            if ((strVal = rp.FindValue("dst_spatialdim_start_idx2")) != null)
                p.dst_spatialdim_start_idx2 = int.Parse(strVal);

            if ((strVal = rp.FindValue("spatialdim_copy_count")) != null)
                p.spatialdim_copy_count = int.Parse(strVal);

            if ((strVal = rp.FindValue("dst_spatialdim")) != null)
                p.dst_spatialdim = int.Parse(strVal);

            return p;
        }

        /// <summary>
        /// Calculate the new shape based on the merge parameter settings and the specified input shapes.
        /// </summary>
        /// <param name="p">Specifies the merge parameter.</param>
        /// <param name="rgShape1">Specifies the shape of the first input.</param>
        /// <param name="rgShape2">Specifies the shape of the second input.</param>
        /// <param name="log">Specifies the output log.</param>
        /// <returns>The new shape is returned.</returns>
        public static List<int> Reshape(Log log, MergeParameter p, List<int> rgShape1, List<int> rgShape2)
        {
            while (rgShape2.Count > rgShape1.Count && rgShape2.Count > 0)
            {
                if (rgShape2[rgShape2.Count - 1] == 1)
                    rgShape2.RemoveAt(rgShape2.Count - 1);
            }

            while (rgShape1.Count > rgShape2.Count && rgShape1.Count > 0)
            {
                if (rgShape1[rgShape1.Count - 1] == 1)
                    rgShape1.RemoveAt(rgShape1.Count - 1);
            }

            log.CHECK_EQ(rgShape1.Count, rgShape2.Count, "The inputs must have the same number of axes.");
            log.CHECK_LT(p.copy_axis, rgShape1.Count, "There must be more axes than the copy axis!");

            int nSrcStartIdx1 = Utility.CanonicalAxisIndex(p.src_start_idx1, rgShape1[p.copy_axis]);
            int nSrcStartIdx2 = Utility.CanonicalAxisIndex(p.src_start_idx2, rgShape2[p.copy_axis]);
            int nDstStartIdx1 = Utility.CanonicalAxisIndex(p.dst_start_idx1, rgShape1[p.copy_axis]);
            int nDstStartIdx2 = Utility.CanonicalAxisIndex(p.dst_start_idx2, rgShape2[p.copy_axis]);

            List<int> rgNewShape = new List<int>();
            for (int i = 0; i < rgShape1.Count; i++)
            {
                rgNewShape.Add(1);
            }

            for (int i = 0; i < p.copy_axis; i++)
            {
                log.CHECK_EQ(rgShape1[i], rgShape2[i], "Inputs must have the same dimensions up to the copy axis.");
                rgNewShape[i] = rgShape1[i];
            }

            int nCopy1 = p.copy_dim1;
            int nCopy2 = p.copy_dim2;
            int nIdx = p.copy_axis;

            rgNewShape[nIdx] = nCopy1 + nCopy2;
            nIdx++;
            rgNewShape[nIdx] = rgShape1[nIdx];
            nIdx++;

            for (int i = nIdx; i < rgNewShape.Count; i++)
            {
                if (p.m_nDstSpatialDim > 0)
                {
                    rgNewShape[i] = p.m_nDstSpatialDim;
                    break;
                }

                if (p.spatialdim_copy_count <= 0)
                {
                    log.CHECK_EQ(rgShape1[i], rgShape2[i], "Inputs must have the same dimensions after the copy axis.");
                    rgNewShape[i] = rgShape1[i];
                }
            }

            return rgNewShape;
        }
    }
}
