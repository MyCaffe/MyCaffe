﻿using System;
using System.Collections;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Serialization;

namespace MyCaffe.basecode
{
    /// <summary>
    /// The Annotation class is used by annotations attached to SimpleDatum's and used in SSD.
    /// </summary>
    /// <remarks>
    /// @see [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg, 2016.
    /// @see [GitHub: SSD: Single Shot MultiBox Detector](https://github.com/weiliu89/caffe/tree/ssd), by weiliu89/caffe, 2016
    /// </remarks>
    [Serializable]
    public class Annotation
    {
        int m_nInstanceId = 0;
        NormalizedBBox m_bbox;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="bbox">Specifies the bounding box.</param>
        /// <param name="nInstanceId">Specifies the instance ID.</param>
        public Annotation(NormalizedBBox bbox, int nInstanceId = 0)
        {
            m_bbox = bbox;
            m_nInstanceId = nInstanceId;
        }

        /// <summary>
        /// Returns a copy of the Annotation.
        /// </summary>
        /// <returns>A new copy of the annotation is returned.</returns>
        public Annotation Clone()
        {
            NormalizedBBox bbox = null;

            if (m_bbox != null)
                bbox = m_bbox.Clone();

            return new Annotation(bbox, m_nInstanceId);
        }

        /// <summary>
        /// Get/set the instance ID.
        /// </summary>
        public int instance_id
        {
            get { return m_nInstanceId; }
            set { m_nInstanceId = value; }
        }

        /// <summary>
        /// Get/set the bounding box.
        /// </summary>
        public NormalizedBBox bbox
        {
            get { return m_bbox; }
            set { m_bbox = value; }
        }

        /// <summary>
        /// Normalize all annotations to the given rectangle.
        /// </summary>
        /// <param name="rc">Specifies the rectangle used to normalize all annotations.</param>
        public void Normalize(Rectangle rc)
        {
            float fxmin = 0;
            float fymin = 0;
            float fxmax = 1;
            float fymax = 1;

            if (m_bbox.xmin > rc.Left)
                fxmin = (m_bbox.xmin - rc.Left) / rc.Width;

            if (m_bbox.ymin > rc.Top)
                fymin = (m_bbox.ymin - rc.Top) / rc.Height;

            if (m_bbox.xmax < rc.Right)
                fxmax = (m_bbox.xmax - rc.Left) / rc.Width;

            if (m_bbox.ymax < rc.Bottom)
                fymax = (m_bbox.ymax - rc.Top) / rc.Height;

            m_bbox.Set(fxmin, fymin, fxmax, fymax);
        }

        /// <summary>
        /// Save the annotation data using the binary writer.
        /// </summary>
        /// <param name="bw">Specifies the binary writer used to save the data.</param>
        public void Save(BinaryWriter bw)
        {
            bw.Write(m_nInstanceId);
            m_bbox.Save(bw);
        }

        /// <summary>
        /// Load the annotation using a binary reader.
        /// </summary>
        /// <param name="br">Specifies the binary reader used to load the data.</param>
        /// <returns>The newly loaded annoation is returned.</returns>
        public static Annotation Load(BinaryReader br)
        {
            int nInstanceId = br.ReadInt32();
            NormalizedBBox bbox = NormalizedBBox.Load(br);

            return new Annotation(bbox, nInstanceId);
        }
    }

    /// <summary>
    /// The AnnoationGroup class manages a group of annotations.
    /// </summary>
    [Serializable]
    public class AnnotationGroup
    {
        int m_nGroupLabel = 0;
        List<Annotation> m_rgAnnotations = new List<Annotation>();

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="rgAnnotations">Optionally, specifies the list of group annotations.</param>
        /// <param name="nGroupLabel">Specifies the group label.</param>
        public AnnotationGroup(List<Annotation> rgAnnotations = null, int nGroupLabel = 0)
        {
            if (rgAnnotations != null && rgAnnotations.Count > 0)
                m_rgAnnotations.AddRange(rgAnnotations);

            m_nGroupLabel = nGroupLabel;
        }

        /// <summary>
        /// Create a copy of the annotation group.
        /// </summary>
        /// <returns>A copy of the annotation group is returned.</returns>
        public AnnotationGroup Clone()
        {
            List<Annotation> rg = null;

            if (m_rgAnnotations != null)
            {
                rg = new List<Annotation>();

                foreach (Annotation a in m_rgAnnotations)
                {
                    rg.Add(a.Clone());
                }
            }

            return new AnnotationGroup(rg, m_nGroupLabel);
        }

        /// <summary>
        /// Returns the maximum scoring annotation within a group.
        /// </summary>
        /// <returns>The annotation with the maximum score is returned, or null if there are no annotations.</returns>
        public Annotation GetMaxScoringAnnotation()
        {
            float fMaxScore = 0;
            int nMaxIdx = -1;

            for (int i = 0; i < m_rgAnnotations.Count; i++)
            {
                if (m_rgAnnotations[i].bbox.score >= fMaxScore)
                {
                    nMaxIdx = i;
                    fMaxScore = m_rgAnnotations[i].bbox.score;
                }
            }

            if (nMaxIdx < 0)
                return null;

            return m_rgAnnotations[nMaxIdx];
        }

        /// <summary>
        /// Normalize all annotations to the given rectangle.
        /// </summary>
        /// <param name="rc">Specifies the rectangle used to normalize all annotations.</param>
        public void Normalize(Rectangle rc)
        {
            foreach (Annotation a in m_rgAnnotations)
            {
                a.Normalize(rc);
            }
        }

        /// <summary>
        /// Get/set the group annoations.
        /// </summary>
        public List<Annotation> annotations
        {
            get { return m_rgAnnotations; }
            set { m_rgAnnotations = value; }
        }

        /// <summary>
        /// Get/set the group label.
        /// </summary>
        public int group_label
        {
            get { return m_nGroupLabel; }
            set { m_nGroupLabel = value; }
        }

        /// <summary>
        /// Save the annotation group to the binary writer.
        /// </summary>
        /// <param name="bw">Specifies the binary writer used to write the data.</param>
        public void Save(BinaryWriter bw)
        {
            bw.Write(m_nGroupLabel);
            bw.Write(m_rgAnnotations.Count);

            for (int i = 0; i < m_rgAnnotations.Count; i++)
            {
                m_rgAnnotations[i].Save(bw);
            }
        }

        /// <summary>
        /// Load an annotation group using the binary reader.
        /// </summary>
        /// <param name="br">Specifies the binary reader used to load the data.</param>
        /// <returns>The new AnnotationGroup loaded is returned.</returns>
        public static AnnotationGroup Load(BinaryReader br)
        {
            int nGroupLabel = br.ReadInt32();
            int nCount = br.ReadInt32();
            List<Annotation> rgAnnotations = new List<Annotation>();

            for (int i = 0; i < nCount; i++)
            {
                rgAnnotations.Add(Annotation.Load(br));
            }

            return new AnnotationGroup(rgAnnotations, nGroupLabel);
        }
    }

    /// <summary>
    /// Defines a collection of AnnotationGroups.
    /// </summary>
    [Serializable]
    public class AnnotationGroupCollection : IEnumerable<AnnotationGroup>
    {
        List<AnnotationGroup> m_rgItems = new List<AnnotationGroup>();
        Dictionary<int, string> m_rgLabels = new Dictionary<int, string>();
        int m_nImageID = 0;
        int m_nImageIdx = 0;
        int m_nCreatorID = 0;
        int m_nDatasetID = 0;
        int m_nSourceID = 0;
        bool m_bHasDataCriteria = false;
        bool m_bHasDebugData = false;

        /// <summary>
        /// The constructor.
        /// </summary>
        public AnnotationGroupCollection()
        {
        }

        /// <summary>
        /// Get/set whether or not the image has a data criteria associated with it.
        /// </summary>
        public bool HasDataCriteria
        {
            get { return m_bHasDataCriteria; }
            set { m_bHasDataCriteria = value; }
        }

        /// <summary>
        /// Get/set whether or not the image has debug data associated with it.
        /// </summary>
        public bool HasDebugData
        {
            get { return m_bHasDebugData; }
            set { m_bHasDebugData = value; }
        }

        /// <summary>
        /// Specifies the ImageID.
        /// </summary>
        public int ImageID
        {
            get { return m_nImageID; }
            set { m_nImageID = value; }
        }

        /// <summary>
        /// Specifies the Image Index.
        /// </summary>
        public int ImageIdx
        {
            get { return m_nImageIdx; }
            set { m_nImageIdx = value; }
        }

        /// <summary>
        /// Specifies the Dataset Creator ID.
        /// </summary>
        public int CreatorID
        {
            get { return m_nCreatorID; }
            set { m_nCreatorID = value; }
        }

        /// <summary>
        /// Specifies the Dataset ID.
        /// </summary>
        public int DatasetID
        {
            get { return m_nDatasetID; }
            set { m_nDatasetID = value; }
        }

        /// <summary>
        /// Specifies the Data Source ID.
        /// </summary>
        public int SourceID
        {
            get { return m_nSourceID; }
            set { m_nSourceID = value; }
        }

        /// <summary>
        /// Get/set the label name mappings.
        /// </summary>
        public Dictionary<int, string> Labels
        {
            get { return m_rgLabels; }
            set { m_rgLabels = value; }
        }

        /// <summary>
        /// Specifies the number of items in the collection.
        /// </summary>
        public int Count
        {
            get { return m_rgItems.Count; }
        }

        /// <summary>
        /// Get/set a specific item within the collection at a given index.
        /// </summary>
        /// <param name="nIdx">Specifies the index of the item to get/set.</param>
        /// <returns>The item at the index is returned.</returns>
        public AnnotationGroup this[int nIdx]
        {
            get { return m_rgItems[nIdx]; }
            set { m_rgItems[nIdx] = value; }
        }

        /// <summary>
        /// Add another AnnotationGroupCollection to this one.
        /// </summary>
        /// <param name="col">Specifies the annotation group to add.</param>
        public void Add(AnnotationGroupCollection col)
        {
            foreach (AnnotationGroup g in col)
            {
                m_rgItems.Add(g);
            }
        }

        /// <summary>
        /// Add a new AnnotationGroup to the collection.
        /// </summary>
        /// <param name="g">Specifies the AnnotationGroup to add.</param>
        public void Add(AnnotationGroup g)
        {
            m_rgItems.Add(g);
        }

        /// <summary>
        /// Remove an AnnotationGroup from the collection if it exists.
        /// </summary>
        /// <param name="g">Specifies the AnnotationGroup to remove.</param>
        /// <returns>If found and removed <i>true</i> is returned, otherwise <i>false</i>.</returns>
        public bool Remove(AnnotationGroup g)
        {
            return m_rgItems.Remove(g);
        }

        /// <summary>
        /// Remove an item at a given index.
        /// </summary>
        /// <param name="nIdx">Specifies the index of the item to remove.</param>
        public void RemoveAt(int nIdx)
        {
            m_rgItems.RemoveAt(nIdx);
        }

        /// <summary>
        /// Clear all items from the collection.
        /// </summary>
        public void Clear()
        {
            m_rgItems.Clear();
        }

        /// <summary>
        /// Returns the enumerator for the collection.
        /// </summary>
        /// <returns>The enumerator is returned.</returns>
        public IEnumerator<AnnotationGroup> GetEnumerator()
        {
            return m_rgItems.GetEnumerator();
        }

        /// <summary>
        /// Returns the enumerator for the collection.
        /// </summary>
        /// <returns>The enumerator is returned.</returns>
        IEnumerator IEnumerable.GetEnumerator()
        {
            return m_rgItems.GetEnumerator();
        }

        /// <summary>
        /// Returns the maximum scoring annotation within a collection of groups.
        /// </summary>
        /// <returns>The annotation with the maximum score is returned, or null if there are no annotations.</returns>
        public Annotation GetMaxScoringAnnotation()
        {
            Annotation bestAnnotation = null;
            float fMaxScore = 0;

            foreach (AnnotationGroup g in m_rgItems)
            {
                Annotation a = g.GetMaxScoringAnnotation();

                if (a != null && a.bbox.score >= fMaxScore)
                {
                    fMaxScore = a.bbox.score;
                    bestAnnotation = a;
                }
            }

            return bestAnnotation;
        }

        /// <summary>
        /// Find the annotation group with the given label.
        /// </summary>
        /// <param name="nLabel">Specifies the label to look for.</param>
        /// <returns>Either the AnnotationGroup with the group label is returned, or <i>null</i>.</returns>
        public AnnotationGroup Find(int nLabel)
        {
            foreach (AnnotationGroup g in m_rgItems)
            {
                if (g.group_label == nLabel)
                    return g;
            }

            return null;
        }

        /// <summary>
        /// Return a copy of the collection.
        /// </summary>
        /// <returns>A copy of the collection is returned.</returns>
        public AnnotationGroupCollection Clone()
        {
            AnnotationGroupCollection col = new AnnotationGroupCollection();

            foreach (AnnotationGroup g in m_rgItems)
            {
                col.Add(g.Clone());
            }

            foreach (KeyValuePair<int, string> kv in m_rgLabels)
            {
                col.m_rgLabels.Add(kv.Key, kv.Value);
            }

            col.ImageID = m_nImageID;
            col.ImageIdx = m_nImageIdx;
            col.CreatorID = m_nCreatorID;
            col.DatasetID = m_nDatasetID;
            col.SourceID = m_nSourceID;
            col.HasDataCriteria = m_bHasDataCriteria;
            col.HasDebugData = m_bHasDebugData;

            return col;
        }

        /// <summary>
        /// Return the min/max labels found int the collection.
        /// </summary>
        /// <returns>A tuple containing the min/max lable found is returned.</returns>
        public Tuple<int, int> GetMinMaxLabels()
        {
            int nMin = int.MaxValue;
            int nMax = -int.MaxValue;

            for (int i = 0; i < m_rgItems.Count; i++)
            {
                nMin = Math.Min(nMin, m_rgItems[i].group_label);
                nMax = Math.Max(nMax, m_rgItems[i].group_label);
            }

            return new Tuple<int, int>(nMin, nMax);
        }

        /// <summary>
        /// Normalize all annotations to the given rectangle.
        /// </summary>
        /// <param name="rc">Specifies the rectangle used to normalize all annotations.</param>
        public void Normalize(Rectangle rc)
        {
            foreach (AnnotationGroup g in m_rgItems)
            {
                g.Normalize(rc);
            }
        }

        /// <summary>
        /// Save an AnnotationGroupCollection to a binary writer.
        /// </summary>
        /// <param name="bw">Specifies the binary writer.</param>
        /// <param name="rg">Specifies the AnnotationGroupCollection to save.</param>
        public static void SaveList(BinaryWriter bw, AnnotationGroupCollection rg)
        {
            bw.Write(rg.Count);

            foreach (AnnotationGroup g in rg)
            {
                g.Save(bw);
            }

            bw.Write(rg.ImageID);
            bw.Write(rg.ImageIdx);
            bw.Write(rg.CreatorID);
            bw.Write(rg.DatasetID);
            bw.Write(rg.SourceID);
            bw.Write(rg.HasDataCriteria);
            bw.Write(rg.HasDebugData);
        }

        /// <summary>
        /// Save the labels to a binary writer.
        /// </summary>
        /// <param name="bw">Specifies the binary writer.</param>
        /// <param name="rg">Specifies the AnnotationGroupCollection to save.</param>
        public static void SaveLabels(BinaryWriter bw, AnnotationGroupCollection rg)
        {
            bw.Write(rg.Labels.Count);
            foreach (KeyValuePair<int, string> kv in rg.Labels)
            {
                bw.Write(kv.Key);
                bw.Write(kv.Value);
            }
        }

        /// <summary>
        /// Load an AnnotationGroupCollection from a binary stream.
        /// </summary>
        /// <param name="br">Specifies the binary reader.</param>
        /// <returns>The list of annotation groups is returned.</returns>
        public static AnnotationGroupCollection LoadList(BinaryReader br)
        {
            AnnotationGroupCollection rg = new AnnotationGroupCollection();
            int nCount = br.ReadInt32();

            for (int i = 0; i < nCount; i++)
            {
                rg.Add(AnnotationGroup.Load(br));
            }

            rg.ImageID = br.ReadInt32();
            rg.ImageIdx = br.ReadInt32();
            rg.CreatorID = br.ReadInt32();
            rg.DatasetID = br.ReadInt32();
            rg.SourceID = br.ReadInt32();
            rg.HasDataCriteria = br.ReadBoolean();
            rg.HasDebugData = br.ReadBoolean();

            return rg;
        }

        /// <summary>
        /// Load the labels from a binary reader.
        /// </summary>
        /// <param name="br">Specifies the binary reader.</param>
        /// <returns>The labels are returned in a Dictionary.</returns>
        public static Dictionary<int, string> LoadLabels(BinaryReader br)
        {
            Dictionary<int, string> rg = new Dictionary<int, string>();

            int nCount = br.ReadInt32();
            for (int i = 0; i < nCount; i++)
            {
                int nKey = br.ReadInt32();
                string strVal = br.ReadString();

                rg.Add(nKey, strVal);
            }

            return rg;
        }

        /// <summary>
        /// Saves a AnnotationGroupCollection to a byte array.
        /// </summary>
        /// <param name="rg">Specifies the list of AnnotationGroup to save.</param>
        /// <param name="bIncludeLabels">Optionally, include the labels.</param>
        /// <returns>The byte array is returned.</returns>
        public static byte[] ToByteArray(AnnotationGroupCollection rg, bool bIncludeLabels = false)
        {
            using (MemoryStream ms = new MemoryStream())
            using (BinaryWriter bw = new BinaryWriter(ms))
            {
                SaveList(bw, rg);

                bw.Write(bIncludeLabels);
                if (bIncludeLabels)
                    SaveLabels(bw, rg);

                bw.Write(rg.ImageID);
                bw.Write(rg.ImageIdx);
                bw.Write(rg.CreatorID);
                bw.Write(rg.DatasetID);
                bw.Write(rg.SourceID);
                bw.Write(rg.HasDataCriteria);
                bw.Write(rg.HasDebugData);

                ms.Flush();
                return ms.ToArray();
            }
        }

        /// <summary>
        /// Returns an AnnotationGroupCollection from a byte array.
        /// </summary>
        /// <param name="rg">Specifies the byte array to load.</param>
        /// <returns>The list of annotation groups is returned.</returns>
        public static AnnotationGroupCollection FromByteArray(byte[] rg)
        {
            using (MemoryStream ms = new MemoryStream(rg))
            using (BinaryReader br = new BinaryReader(ms))
            {
                AnnotationGroupCollection col = LoadList(br);

                if (br.ReadBoolean())
                    col.Labels = LoadLabels(br);

                col.ImageID = br.ReadInt32();
                col.ImageIdx = br.ReadInt32();
                col.CreatorID = br.ReadInt32();
                col.DatasetID = br.ReadInt32();
                col.SourceID = br.ReadInt32();
                col.HasDataCriteria = br.ReadBoolean();
                col.HasDebugData = br.ReadBoolean();

                return col;
            }
        }
    }
}
