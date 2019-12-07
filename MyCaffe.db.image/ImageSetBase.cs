using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Drawing;
using MyCaffe.basecode;
using System.Threading;
using System.Diagnostics;
using MyCaffe.basecode.descriptors;

namespace MyCaffe.db.image
{
    /// <summary>
    /// The ImageSetBase class contains the list of image for a data source as well as a list of LabelSets that map into it.
    /// </summary>
    public class ImageSetBase : IDisposable
    {
        /// <summary>
        /// Specifies the DatasetFactory used to work with the underlying database.
        /// </summary>
        protected DatasetFactory m_factory;
        /// <summary>
        /// Specifies the data source used with this Image Set.
        /// </summary>
        protected SourceDescriptor m_src;

        /// <summary>
        /// The ImageSet constructor.
        /// </summary>
        /// <param name="factory">Specifies the DatasetFactory.</param>
        /// <param name="src">Specifies the data source.</param>
        public ImageSetBase(DatasetFactory factory, SourceDescriptor src)
        {
            m_src = new SourceDescriptor(src);
            m_factory = new DatasetFactory(factory);
            m_factory.Open(src.ID);
        }

        /// <summary>
        /// Releases the resouces used.
        /// </summary>
        /// <param name="bDisposing">Set to <i>true</i> when called by Dispose()</param>
        protected virtual void Dispose(bool bDisposing)
        {
            if (m_factory != null)
            {
                m_factory.Close();
                m_factory.Dispose();
                m_factory = null;
            }
        }

        /// <summary>
        /// Releases the resouces used.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
        }

        /// <summary>
        /// Returns the data source of the image set.
        /// </summary>
        public SourceDescriptor Source
        {
            get { return m_src; }
        }

        /// <summary>
        /// Set the label mapping of the ImageSet.
        /// </summary>
        /// <param name="map">Specifies the label map.</param>
        public void SetLabelMapping(LabelMapping map)
        {
            m_factory.SetLabelMapping(map, m_src.ID);
        }

        /// <summary>
        /// Update the label mapping on the ImageSet.
        /// </summary>
        /// <param name="nNewLabel">Specifies the new label.</param>
        /// <param name="rgOriginalLabels">Specifies the labels to be mapped to the new label.</param>
        public void UpdateLabelMapping(int nNewLabel, List<int> rgOriginalLabels)
        {
            m_factory.UpdateLabelMapping(nNewLabel, rgOriginalLabels, m_src.ID);
        }

        /// <summary>
        /// Resets the labels for a project.
        /// </summary>
        /// <param name="nProjectId">Specifies the ID of the project.</param>
        public void ResetLabels(int nProjectId)
        {
            m_factory.ResetLabels(nProjectId, m_src.ID);
        }

        /// <summary>
        /// Deletes the label boosts for a project.
        /// </summary>
        /// <param name="nProjectId">Specifies the ID of the project.</param>
        public void DeleteLabelBoosts(int nProjectId)
        {
            m_factory.DeleteLabelBoosts(nProjectId, m_src.ID);
        }

        /// <summary>
        /// Adds a label boost for a project.
        /// </summary>
        /// <param name="nProjectId">Specifies the ID of the project.</param>
        /// <param name="nLabel">Specifies the label.</param>
        /// <param name="dfBoost">Specifies the label boost.</param>
        public void AddLabelBoost(int nProjectId, int nLabel, double dfBoost)
        {
            m_factory.AddLabelBoost(nProjectId, nLabel, dfBoost, m_src.ID);
        }

        /// <summary>
        /// Returns the label boosts as text.
        /// </summary>
        /// <param name="nProjectId">Specifies the ID of the project.</param>
        /// <returns>The label boosts are returned.</returns>
        public string GetLabelBoostsAsText(int nProjectId)
        {
            return m_factory.GetLabelBoostsAsText(nProjectId, m_src.ID);
        }

        /// <summary>
        /// Returns the label counts as a dictionary of item pairs (int nLabel, int nCount).
        /// </summary>
        /// <returns>The label counts are returned as item pairs (int nLabel, int nCount).</returns>
        public Dictionary<int, int> LoadLabelCounts()
        {
            return m_factory.LoadLabelCounts(m_src.ID);
        }

        /// <summary>
        /// Updates the label counts for a project.
        /// </summary>
        /// <param name="nProjectId">Specifies the ID of the project.</param>
        public void UpdateLabelCounts(int nProjectId)
        {
            m_factory.UpdateLabelCounts(m_src.ID, nProjectId);
        }

        /// <summary>
        /// Returns the label counts for the ImageList as text.
        /// </summary>
        /// <returns>The label counts are retuned.</returns>
        public string GetLabelCountsAsText()
        {
            return m_factory.GetLabelCountsAsText(m_src.ID);
        }
    }
}
