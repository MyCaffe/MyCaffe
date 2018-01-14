using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Drawing;
using System.Data.SqlClient;
using System.Diagnostics;
using System.Net;
using System.Runtime.InteropServices;
using System.Threading;
using MyCaffe.basecode.descriptors;
using MyCaffe.basecode;
using System.Data.Entity.Infrastructure;
using System.IO;
using System.Threading.Tasks;

namespace MyCaffe.imagedb
{
    /// <summary>
    /// The Database class manages the actual connection to the physical database using <a href="https://msdn.microsoft.com/en-us/library/aa937723(v=vs.113).aspx">Entity Framworks</a> from Microsoft.
    /// </summary>
    public class Database
    {
        Source m_src = null;
        DNNEntities m_entities = null;
        List<Label> m_rgLabelCache;
        int m_nSecondarySourceID = 0;
        /// <summary>
        /// Specifies the base path to the file based data.
        /// </summary>
        protected string m_strPrimaryImgPath = null;
        /// <summary>
        /// Specifies the secondary base path to the file based data (used when copying a data source)
        /// </summary> 
        protected string m_strSecondaryImgPath = null;
        /// <summary>
        /// Specifies whether or not file based data is enabled.
        /// </summary>
        protected bool m_bEnableFileBasedData = false;

        /// <summary>
        /// The Database constructor.
        /// </summary>
        public Database()
        {
        }

        private string convertWs(string str, char chReplacement)
        {
            string strOut = "";

            foreach (char ch in str)
            {
                if (char.IsWhiteSpace(ch))
                    strOut += chReplacement;
                else
                    strOut += ch;
            }

            return strOut;
        }

        /// <summary>
        /// Saves any changes on the open satabase.
        /// </summary>
        public void SaveChanges()
        {
            if (m_entities != null)
                m_entities.SaveChanges();
        }

        /// <summary>
        /// Returns the current entity framwork Source object set during the previous call to Open().
        /// </summary>
        public Source CurrentSource
        {
            get { return m_src; }
        }

        /// <summary>
        /// Opens a data source.
        /// </summary>
        /// <param name="nSrcId">Specifies the ID of the data source to open.</param>
        /// <param name="bForceLoadImageFilePath">Optionally, specifies to force load the image file path (default = <i>false</i>).</param>
        public void Open(int nSrcId, bool bForceLoadImageFilePath = false)
        {
            m_src = GetSource(nSrcId);
            if (m_src == null)
                throw new Exception("Could not find the source with ID = " + nSrcId.ToString());

            m_entities = EntitiesConnection.CreateEntities();
            m_rgLabelCache = loadLabelCache(m_src.ID);

            setImagePath(bForceLoadImageFilePath);
        }

        /// <summary>
        /// Opens a data source.
        /// </summary>
        /// <param name="strSrc">Specifies the name of the data source to open.</param>
        /// <param name="bForceLoadImageFilePath">Optionally, specifies to force load the image file path (default = <i>false</i>) and use file-based data.</param>
        public void Open(string strSrc, bool bForceLoadImageFilePath = false)
        {
            m_src = GetSource(strSrc);
            m_entities = EntitiesConnection.CreateEntities();
            m_rgLabelCache = loadLabelCache(m_src.ID);

            setImagePath(bForceLoadImageFilePath);
        }

        /// <summary>
        /// Sets the image path member to the path used when saving binary data to the file system.
        /// </summary>
        /// <param name="bForceLoadImageFilePath">Specifies whether or not to enable saving binary data to the file system.</param>
        protected virtual void setImagePath(bool bForceLoadImageFilePath)
        {
            m_strPrimaryImgPath = getImagePath();

            if (m_src.SaveImagesToFile.GetValueOrDefault(false) || bForceLoadImageFilePath)
            {
                m_bEnableFileBasedData = true;

                if (!Directory.Exists(m_strPrimaryImgPath))
                    Directory.CreateDirectory(m_strPrimaryImgPath);
            }
        }

        /// <summary>
        /// Returns the base image path used when saving binary data to the file system.
        /// </summary>
        /// <returns>The base image path is returned.</returns>
        protected virtual string getImagePath(string strSrcName = null)
        {
            return getImagePathBase(strSrcName);
        }

        string getImagePathBase(string strSrcName = null, DNNEntities entities = null)
        {
            if (strSrcName == null)
            {
                if (m_src == null)
                    return null;

                strSrcName = m_src.Name;

                if (m_src.CopyOfSourceID > 0)
                    strSrcName = GetSourceName(m_src.CopyOfSourceID.GetValueOrDefault());
            }

            if (entities == null)
                entities = m_entities;

            return GetDatabaseImagePath(entities.Database.Connection.Database) + strSrcName + "\\";
        }

        /// <summary>
        /// Close the previously opened data source.
        /// </summary>
        public void Close()
        {
            m_src = null;
            m_entities.Dispose();
            m_entities = null;
            m_strPrimaryImgPath = null;
            m_bEnableFileBasedData = false;
        }

        /// <summary>
        /// Close and re Open with the current data source.
        /// </summary>
        public void Refresh()
        {
            int nSrcId = m_src.ID;
            Close();
            Open(nSrcId);
        }

        /// <summary>
        /// Query the physical database file path.
        /// </summary>
        /// <param name="strName">Specifies the name of the database.</param>
        /// <returns>The physical file path is returned.</returns>
        public string GetDatabaseFilePath(string strName)
        {
            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                string strCmd = "SELECT physical_name FROM sys.master_files WHERE name = '" + strName + "'";
                DbRawSqlQuery<string> qry = entities.Database.SqlQuery<string>(strCmd);
                List<string> rgStr = qry.ToList();

                if (rgStr.Count == 0)
                    return null;

                FileInfo fi = new FileInfo(rgStr[0]);
                string strDir = fi.DirectoryName;

                return strDir + "\\";
            }
        }

        /// <summary>
        /// Query the physical database file path for Images.
        /// </summary>
        /// <param name="strName">Specifies the name of the database.</param>
        /// <returns>The physical file path is returned.</returns>
        public string GetDatabaseImagePath(string strName)
        {
            string strDir = GetDatabaseFilePath(strName);

            return strDir + "Images\\" + strName + "\\";
        }


        //---------------------------------------------------------------------
        //  Labels
        //---------------------------------------------------------------------
        #region Labels

        private List<Label> loadLabelCache(int nSrcId)
                {
                    return m_entities.Labels.Where(p => p.SourceID == nSrcId).ToList();
                }

                /// <summary>
                /// Update the name of a label.
                /// </summary>
                /// <param name="nLabel">Specifies the label.</param>
                /// <param name="strName">Specifies the new name.</param>
                /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
                public void UpdateLabelName(int nLabel, string strName, int nSrcId = 0)
                {
                    if (nSrcId == 0)
                        nSrcId = m_src.ID;

                    using (DNNEntities entities = EntitiesConnection.CreateEntities())
                    {
                        List<Label> rg = entities.Labels.Where(p => p.SourceID == nSrcId && p.ActiveLabel == nLabel).ToList();
                        Label l;

                        if (rg.Count == 0)
                        {
                            l = new Label();
                            l.Label1 = nLabel;
                            l.ActiveLabel = nLabel;
                            l.ImageCount = 0;
                            l.SourceID = nSrcId;

                            entities.Labels.Add(l);
                        }
                        else
                        {
                            l = rg[0];
                        }

                        l.Name = strName;

                        entities.SaveChanges();
                    }
                }

                /// <summary>
                /// Return the Label with the given ID.
                /// </summary>
                /// <param name="nID">Specifies the Label ID.</param>
                /// <returns>When found, the Label with the ID is returned, otherwise <i>null</i> is returned.</returns>
                public Label GetLabel(int nID)
                {
                    foreach (Label l in m_rgLabelCache)
                    {
                        if (l.ID == nID)
                            return l;
                    }

                    return null;
                }

                /// <summary>
                /// Get the Label name of a label within a data source.
                /// </summary>
                /// <param name="nLabel">Specifies the label.</param>
                /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
                /// <returns>When found, the Label is returned, otherwise <i>null</i> is returned.</returns>
                public string GetLabelName(int nLabel, int nSrcId = 0)
                {
                    if (nSrcId == 0)
                    {
                        foreach (Label l in m_rgLabelCache)
                        {
                            if (l.Label1 == nLabel)
                                return l.Name;
                        }

                        return null;
                    }
                    else
                    {
                        using (DNNEntities entities = EntitiesConnection.CreateEntities())
                        {
                            List<Label> rg = entities.Labels.Where(p => p.SourceID == nSrcId && p.Label1 == nLabel).ToList();

                            if (rg.Count == 0)
                                return null;

                            return rg[0].Name;
                        }
                    }
                }

                /// <summary>
                /// Search for a Label in the label cache.
                /// </summary>
                /// <param name="nLabel">Specifies the label.</param>
                /// <returns>When found, the Label is returned, otherwise <i>null</i> is returned.</returns>
                public Label FindLabelInCache(int nLabel)
                {
                    foreach (Label l in m_rgLabelCache)
                    {
                        if (l.ActiveLabel == nLabel)
                            return l;
                    }

                    return null;
                }

                /// <summary>
                /// Returns the label ID associated with a label value.
                /// </summary>
                /// <param name="nLabel">Specifies the label.</param>
                /// <returns>The ID of the Label is returned.</returns>
                public int GetLabelID(int nLabel)
                {
                    foreach (Label l in m_rgLabelCache)
                    {
                        if (l.ActiveLabel == nLabel)
                            return l.ID;
                    }

                    return 0;
                }

                /// <summary>
                /// Returns the number of images under a given label.
                /// </summary>
                /// <param name="nLabel">Specifies the label.</param>
                /// <returns>The number of images is returned.</returns>
                public int GetLabelCount(int nLabel)
                {
                    foreach (Label l in m_rgLabelCache)
                    {
                        if (l.ActiveLabel == nLabel)
                            return l.ImageCount.GetValueOrDefault(0);
                    }

                    return 0;
                }

                /// <summary>
                /// Adds a label to the label cache.
                /// </summary>
                /// <param name="nLabel">Specifies the label.</param>
                public void AddLabelToCache(int nLabel)
                {
                    Label l = FindLabelInCache(nLabel);

                    if (l == null)
                    {
                        l = new Label();
                        l.ImageCount = 1;
                        l.Name = "";
                        l.SourceID = m_src.ID;
                        l.Label1 = nLabel;
                        l.ActiveLabel = nLabel;
                        m_rgLabelCache.Add(l);
                    }
                    else
                    {
                        l.ImageCount++;
                    }
                }

                /// <summary>
                /// Saves the label cache to the database.
                /// </summary>
                public void SaveLabelCache()
                {
                    using (DNNEntities entities = EntitiesConnection.CreateEntities())
                    {
                        foreach (Label l in m_rgLabelCache)
                        {
                            List<Label> rgLabels = entities.Labels.Where(p => p.SourceID == m_src.ID && p.Label1 == l.Label1).ToList();

                            if (rgLabels.Count == 0)
                            {
                                if (!l.ActiveLabel.HasValue)
                                    l.ActiveLabel = l.Label1;

                                entities.Labels.Add(l);
                            }
                        }

                        entities.SaveChanges();
                    }
                }

                /// <summary>
                /// Updates the label counts in the database for the open data source.
                /// </summary>
                /// <param name="rgCounts">Specifies a dictionary containing (int nLabel, int nCount) pairs.</param>
                public void UpdateLabelCounts(Dictionary<int, int> rgCounts)
                {
                    foreach (Label l in m_rgLabelCache)
                    {
                        l.ImageCount = 0;
                    }

                    foreach (KeyValuePair<int, int> kv in rgCounts)
                    {
                        List<Label> rgLabel = m_rgLabelCache.Where(p => p.ActiveLabel == kv.Key).ToList();

                        if (rgLabel.Count > 0)
                            rgLabel[0].ImageCount = kv.Value;
                    }

                    m_entities.SaveChanges();
                }

                /// <summary>
                /// Load the label counts from the database for a data source.
                /// </summary>
                /// <param name="rgCounts">Specifies where the counts are loaded.</param>
                /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
                public void LoadLabelCounts(Dictionary<int, int> rgCounts, int nSrcId = 0)
                {
                    if (nSrcId == 0)
                        nSrcId = m_src.ID;

                    using (DNNEntities entities = EntitiesConnection.CreateEntities())
                    {
                        List<Label> rgLabels = entities.Labels.Where(p => p.SourceID == nSrcId).ToList();

                        foreach (Label l in rgLabels)
                        {
                            int nCount = entities.RawImages.Where(p => p.SourceID == nSrcId && p.ActiveLabel == l.ActiveLabel && p.Active == true).Count();
                            rgCounts.Add(l.ActiveLabel.GetValueOrDefault(), nCount);
                        }
                    }
                }

                /// <summary>
                /// Returns the label counts for a given data source.
                /// </summary>
                /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
                /// <returns>A string containing the label counts is returned.</returns>
                public string GetLabelCountsAsText(int nSrcId = 0)
                {
                    if (nSrcId == 0)
                        nSrcId = m_src.ID;

                    using (DNNEntities entities = EntitiesConnection.CreateEntities())
                    {
                        List<Label> rgLabels = entities.Labels.Where(p => p.SourceID == nSrcId).OrderBy(p => p.Label1).ToList();

                        string strOut = "{";

                        foreach (Label l in rgLabels)
                        {
                            strOut += l.ImageCount.GetValueOrDefault().ToString();
                            strOut += ",";
                        }

                        strOut = strOut.TrimEnd(',');
                        strOut += "}";

                        return strOut;
                    }
                }

                /// <summary>
                /// Update the label counts for a given data source.
                /// </summary>
                /// <param name="rgCounts">Specifies the counts.</param>
                /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
                public void UpdateLabelCounts(Dictionary<int, int> rgCounts, int nSrcId = 0)
                {
                    if (nSrcId == 0)
                        nSrcId = m_src.ID;

                    using (DNNEntities entities = EntitiesConnection.CreateEntities())
                    {
                        List<Label> rgLabels = entities.Labels.Where(p => p.SourceID == nSrcId).ToList();

                        foreach (Label l in rgLabels)
                        {
                            l.ImageCount = 0;
                        }

                        foreach (KeyValuePair<int, int> kv in rgCounts)
                        {
                            List<Label> rgLabel = rgLabels.Where(p => p.ActiveLabel == kv.Key).ToList();

                            if (rgLabel.Count > 0)
                                rgLabel[0].ImageCount = kv.Value;
                        }

                        entities.SaveChanges();
                    }
                }

                /// <summary>
                /// Update the label counts for a given data source and project (optionally) by querying the database for the actual counts.
                /// </summary>
                /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
                /// <param name="nProjectId">Optionally, specifies the ID of a project to use (default = 0).</param>
                public void UpdateLabelCounts(int nSrcId = 0, int nProjectId = 0)
                {
                    Dictionary<int, double> rgLabelBoosts = null;
                    double dfTotal = 0;

                    if (nSrcId == 0)
                        nSrcId = m_src.ID;

                    if (nProjectId > 0)
                        rgLabelBoosts = new Dictionary<int, double>();

                    using (DNNEntities entities = EntitiesConnection.CreateEntities())
                    {
                        List<Label> rgLabels = entities.Labels.Where(p => p.SourceID == nSrcId).ToList();

                        foreach (Label l in rgLabels)
                        {
                            int nCount = entities.RawImages.Where(p => p.SourceID == nSrcId && p.ActiveLabel == l.ActiveLabel && p.Active == true).Count();
                            l.ImageCount = nCount;

                            if (nProjectId > 0)
                            {
                                rgLabelBoosts.Add(l.ActiveLabel.GetValueOrDefault(0), nCount);
                                dfTotal += nCount;
                            }
                        }

                        entities.SaveChanges();
                    }

                    if (nProjectId > 0)
                    {
                        if (dfTotal == 0)
                            throw new Exception("There are no images for label boost!");

                        foreach (int nKey in rgLabelBoosts.Keys)
                        {
                            AddLabelBoost(nProjectId, nKey, rgLabelBoosts[nKey] / dfTotal, nSrcId);
                        }
                    }
                }

                /// <summary>
                /// Update the label counts for the currently open data source by querying the database for the actual counts.
                /// </summary>
                public void UpdateLabelCounts()
                {
                    UpdateLabelCounts(m_src.ID, 0);
                }

                /// <summary>
                /// Returns a list of all labels used by a data source.
                /// </summary>
                /// <param name="bSort">Specifies to sort the labels by label.</param>
                /// <param name="bWithImagesOnly">Specifies to only return labels that actually have images associated with them.</param>
                /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
                /// <returns></returns>
                public List<Label> GetLabels(bool bSort = true, bool bWithImagesOnly = false, int nSrcId = 0)
                {
                    if (nSrcId == 0)
                        nSrcId = m_src.ID;

                    using (DNNEntities entities = EntitiesConnection.CreateEntities())
                    {
                        List<Label> rgLabels = entities.Labels.Where(p => p.SourceID == nSrcId).ToList();

                        if (bWithImagesOnly)
                            rgLabels = rgLabels.Where(p => p.ImageCount > 0).ToList();

                        if (bSort)
                            rgLabels = rgLabels.OrderBy(p => p.Label1).ToList();

                        return rgLabels;
                    }
                }

                /// <summary>
                /// Delete the labels of a data source from the database.
                /// </summary>
                /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
                public void DeleteLabels(int nSrcId = 0)
                {
                    if (nSrcId == 0)
                        nSrcId = m_src.ID;

                    using (DNNEntities entities = EntitiesConnection.CreateEntities())
                    {
                        string strCmd = "DELETE FROM [DNN].[dbo].[Labels] WHERE (SourceID = " + nSrcId.ToString() + ")";
                        entities.Database.ExecuteSqlCommand(strCmd);
                    }
                }

                /// <summary>
                /// Add a label to the database for a data source.
                /// </summary>
                /// <param name="nLabel">Specifies the label.</param>
                /// <param name="strName">Optionally, specifies a label name (default = "").</param>
                /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
                /// <returns>The ID of the added label is returned.</returns>
                public int AddLabel(int nLabel, string strName = "", int nSrcId = 0)
                {
                    if (nSrcId == 0)
                        nSrcId = m_src.ID;

                    using (DNNEntities entities = EntitiesConnection.CreateEntities())
                    {
                        List<Label> rgLabel = entities.Labels.Where(p => p.SourceID == nSrcId && p.Label1 == nLabel).ToList();
                        Label l; 
                
                        if (rgLabel.Count > 0)
                        {
                            l = rgLabel[0];
                        }
                        else
                        {
                            l = new Label();
                            l.Label1 = nLabel;
                            l.SourceID = nSrcId;
                        }

                        l.ActiveLabel = nLabel;
                        l.ImageCount = 0;
                        l.Name = strName;

                        if (rgLabel.Count == 0)
                            entities.Labels.Add(l);

                        entities.SaveChanges();

                        return l.ID;
                    }
                }

                /// <summary>
                /// Add a label boost to the database for a given project.
                /// </summary>
                /// <param name="nProjectId">Specifies the ID of the project for which the label boost is to be added.</param>
                /// <param name="nLabel">Specifies the label.</param>
                /// <param name="dfBoost">Specifies the boost.</param>
                /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
                public void AddLabelBoost(int nProjectId, int nLabel, double dfBoost, int nSrcId = 0)
                {
                    if (nSrcId == 0)
                        nSrcId = m_src.ID;

                    using (DNNEntities entities = EntitiesConnection.CreateEntities())
                    {
                        List<LabelBoost> rgLabelBoosts = entities.LabelBoosts.Where(p => p.ProjectID == nProjectId && p.SourceID == nSrcId && p.ActiveLabel == nLabel).ToList();
                        LabelBoost lb;

                        if (rgLabelBoosts.Count > 0)
                        {
                            lb = rgLabelBoosts[0];
                        }
                        else
                        {
                            lb = new LabelBoost();
                            lb.ActiveLabel = nLabel;
                            lb.ProjectID = nProjectId;
                            lb.SourceID = nSrcId;
                        }

                        lb.Boost = (decimal)dfBoost;

                        if (rgLabelBoosts.Count == 0)
                            entities.LabelBoosts.Add(lb);

                        entities.SaveChanges();
                    }
                }

                /// <summary>
                /// Saves a label mapping in the database for a data source.
                /// </summary>
                /// <param name="map">Specifies the label mapping.</param>
                /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
                public void SetLabelMapping(LabelMapping map, int nSrcId = 0)
                {
                    if (nSrcId == 0)
                        nSrcId = m_src.ID;

                    using (DNNEntities entities = EntitiesConnection.CreateEntities())
                    {
                        entities.Database.CommandTimeout = 180;
                        string strCmd = "UPDATE [DNN].[dbo].[RawImages] SET [ActiveLabel] = " + map.NewLabel.ToString() + " WHERE (SourceID = " + nSrcId.ToString() + ")";
                        strCmd += " AND (OriginalLabel = " + map.OriginalLabel.ToString() + ")";

                        if (map.ConditionBoostEquals.HasValue)
                            strCmd += " AND (ActiveBoost = " + map.ConditionBoostEquals.Value.ToString() + ")";

                        entities.Database.ExecuteSqlCommand(strCmd);

                        if (map.ConditionBoostEquals.HasValue)
                        {
                            if (map.NewLabelConditionFalse.HasValue)
                            {
                                strCmd = "UPDATE [DNN].[dbo].[RawImages] SET [ActiveLabel] = " + map.NewLabelConditionFalse.Value.ToString() + " WHERE (SourceID = " + nSrcId.ToString() + ")";

                                strCmd += " AND (OriginalLabel = " + map.OriginalLabel.ToString() + ")";
                                strCmd += " AND (ActiveBoost != " + map.ConditionBoostEquals.Value.ToString() + ")";
                                entities.Database.ExecuteSqlCommand(strCmd);
                            }
                        }
                    }
                }

                /// <summary>
                /// Update a label mapping in the database for a data source.
                /// </summary>
                /// <param name="nNewLabel">Specifies the new label.</param>
                /// <param name="rgOriginalLabels">Specifies the original labels that are to be mapped to the new label.</param>
                /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
                public void UpdateLabelMapping(int nNewLabel, List<int> rgOriginalLabels, int nSrcId = 0)
                {
                    if (nSrcId == 0)
                        nSrcId = m_src.ID;

                    using (DNNEntities entities = EntitiesConnection.CreateEntities())
                    {
                        string strCmd = "UPDATE [DNN].[dbo].[RawImages] SET [ActiveLabel] = " + nNewLabel.ToString() + " WHERE (SourceID = " + nSrcId.ToString() + ") AND (";

                        for (int i = 0; i < rgOriginalLabels.Count; i++)
                        {
                            strCmd += "OriginalLabel = " + rgOriginalLabels[i].ToString();

                            if (i < rgOriginalLabels.Count - 1)
                                strCmd += " OR ";
                        }

                        strCmd += ")";

                        entities.Database.ExecuteSqlCommand(strCmd);
                    }
                }

                /// <summary>
                /// Resets all labels back to their original labels for a project.
                /// </summary>
                /// <param name="nProjectId">Specifies the ID of a project.</param>
                /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
                public void ResetLabels(int nProjectId, int nSrcId = 0)
                {
                    if (nSrcId == 0)
                        nSrcId = m_src.ID;

                    using (DNNEntities entities = EntitiesConnection.CreateEntities())
                    {
                        entities.Database.CommandTimeout = 180;

                        string strResetCmd = "UPDATE [DNN].[dbo].[RawImages] SET [ActiveLabel] = [OriginalLabel] WHERE (SourceID = " + nSrcId.ToString() + ")";
                        entities.Database.ExecuteSqlCommand(strResetCmd);

                        DeleteLabelBoosts(nProjectId, nSrcId);
                    }
                }

                /// <summary>
                /// Delete all label boosts for a project.
                /// </summary>
                /// <param name="nProjectId">Specifies the ID of a project.</param>
                /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
                public void DeleteLabelBoosts(int nProjectId, int nSrcId = 0)
                {
                    if (nSrcId == 0)
                        nSrcId = m_src.ID;

                    using (DNNEntities entities = EntitiesConnection.CreateEntities())
                    {
                        string strCmd = "DELETE FROM [DNN].[dbo].[LabelBoosts] WHERE (ProjectID = " + nProjectId.ToString() + ") AND (SourceID = " + nSrcId.ToString() + ")";
                        entities.Database.ExecuteSqlCommand(strCmd);
                    }
                }

                /// <summary>
                /// Delete all label boosts for a project.
                /// </summary>
                /// <param name="nProjectId">Specifies the ID of a project.</param>
                public void DeleteLabelBoosts(int nProjectId)
                {
                    using (DNNEntities entities = EntitiesConnection.CreateEntities())
                    {
                        string strCmd = "DELETE FROM [DNN].[dbo].[LabelBoosts] WHERE (ProjectID = " + nProjectId.ToString() + ")";
                        entities.Database.ExecuteSqlCommand(strCmd);
                    }
                }

                /// <summary>
                /// Reset all label boosts to their orignal settings for a project.
                /// </summary>
                /// <param name="nProjectId">Specifies the ID of a project.</param>
                public void ResetLabelBoosts(int nProjectId)
                {
                    using (DNNEntities entities = EntitiesConnection.CreateEntities())
                    {
                        List<LabelBoost> rgLabels = entities.LabelBoosts.Where(p => p.ProjectID == nProjectId).ToList();

                        foreach (LabelBoost l in rgLabels)
                        {
                            l.Boost = 1;
                        }

                        entities.SaveChanges();
                    }
                }

                /// <summary>
                /// Returns a list of all label boosts set on a project.
                /// </summary>
                /// <param name="nProjectId">Specifies the ID of a project.</param>
                /// <param name="bSort">Optionally, specifies whether or not to sort the labels by active label (default = true).</param>
                /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
                /// <returns>A list of LabelBoosts is returned.</returns>
                public List<LabelBoost> GetLabelBoosts(int nProjectId, bool bSort = true, int nSrcId = 0)
                {
                    if (nSrcId == 0)
                        nSrcId = m_src.ID;

                    using (DNNEntities entities = EntitiesConnection.CreateEntities())
                    {
                        List<LabelBoost> rgBoosts = entities.LabelBoosts.Where(p => p.ProjectID == nProjectId && p.SourceID == nSrcId).ToList();

                        if (bSort)
                            rgBoosts = rgBoosts.OrderBy(p => p.ActiveLabel).ToList();

                        return rgBoosts;
                    }
                }

                /// <summary>
                /// Returns the Label boosts as a string.
                /// </summary>
                /// <param name="nProjectId">Specifies the ID of a project.</param>
                /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
                /// <param name="bSort">Optionally, specifies whether or not to sort the labels by active label (default = true).</param>
                /// <returns></returns>
                public string GetLabelBoostsAsText(int nProjectId, int nSrcId = 0, bool bSort = true)
                {
                    if (nSrcId == 0)
                        nSrcId = m_src.ID;

                    List<LabelBoost> rgLb = GetLabelBoosts(nProjectId, bSort, nSrcId);
                    string strOut = "";

                    foreach (LabelBoost lb in rgLb)
                    {
                        strOut += lb.Boost.GetValueOrDefault().ToString("N2");
                        strOut += ",";
                    }

                    return strOut.TrimEnd(',');
                }

                #endregion


        //---------------------------------------------------------------------
        //  Images
        //---------------------------------------------------------------------
        #region RawImages

        /// <summary>
        /// Returns the number of raw images in the database for the open data source.
        /// </summary>
        /// <returns></returns>
        public int GetImageCount()
        {
            return m_entities.RawImages.Where(p => p.SourceID == m_src.ID).Count();
        }

        /// <summary>
        /// Returns the list of raw images that have a source ID from a selected list.
        /// </summary>
        /// <param name="rgSrcId">Specifies the list of source ID.</param>
        /// <returns>The list of RawImage's is returned.</returns>
        public List<RawImage> QueryRawImages(params int[] rgSrcId)
        {
            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                List<int?> rgSrcId1 = new List<int?>();

                foreach (int id in rgSrcId)
                {
                    rgSrcId1.Add(id);
                }

                return entities.RawImages.Where(p => rgSrcId1.Contains(p.SourceID)).ToList();   
            }
        }

        /// <summary>
        /// Returns a list of RawImages from the database for a data source.
        /// </summary>
        /// <param name="nIdx">Specifies the starting image index.</param>
        /// <param name="nCount">Specifies the number of images to retrieve from the starting index <i>nIdx</i>.</param>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <param name="strDescription">Optionally, specifies a description to filter the images retrieved (when specified, only images matching the filter are returned) (default = null).</param>
        /// <returns></returns>
        public List<RawImage> GetRawImagesAt(int nIdx, int nCount, int nSrcId = 0, string strDescription = null)
        {
            if (nSrcId == 0)
                nSrcId = m_src.ID;

            string strCmd = "SELECT TOP " + nCount.ToString() + " * FROM [DNN].[dbo].[RawImages] WHERE (SourceID = " + nSrcId.ToString() + ") AND (Idx >= " + nIdx.ToString() + ") AND (Active = 1)";

            if (!String.IsNullOrEmpty(strDescription))
                strCmd += " AND (Description = " + strDescription + ")";

            strCmd += " ORDER BY Idx";

            return m_entities.Database.SqlQuery<RawImage>(strCmd).ToList();
        }

        /// <summary>
        /// Returns the RawImage at a given image index.
        /// </summary>
        /// <param name="nIdx">Specifies the image index to retrieve.</param>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <returns></returns>
        public RawImage GetRawImageAt(int nIdx, int nSrcId = 0)
        {
            if (nSrcId == 0)
                nSrcId = m_src.ID;

            string strCmd = "SELECT TOP 1 * FROM [DNN].[dbo].[RawImages] WHERE (SourceID = " + nSrcId.ToString() + ") AND (Idx = " + nIdx.ToString() + ") AND (Active = 1)";
            List<RawImage> rgImg = m_entities.Database.SqlQuery<RawImage>(strCmd).ToList();

            if (rgImg.Count == 0)
                return null;

            return rgImg[0];
        }

        /// <summary>
        /// Returns the RawImage ID for the image with the given time-stamp. 
        /// </summary>
        /// <param name="dt">Specifies the image time-stamp.</param>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <returns>The ID of the RawImage is returned.</returns>
        public int GetRawImageID(DateTime dt, int nSrcId = 0)
        {
            if (nSrcId == 0)
                nSrcId = m_src.ID;

            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                List<RawImage> rgImg = entities.RawImages.Where(p => p.SourceID == nSrcId && p.TimeStamp == dt && p.Active == true).ToList();

                if (rgImg.Count == 0)
                    return 0;

                return rgImg[0].ID;
            }
        }

        /// <summary>
        /// Returns the raw data of the RawImage.
        /// </summary>
        /// <remarks>
        /// If the RawImage uses its Virtual ID, the RawImage with that ID is queried from the database and its raw data is returned.
        /// </remarks>
        /// <param name="img">Specifies the RawImage to use.</param>
        /// <param name="rgDataCriteria">Returns the image data criteria (if any).</param>
        /// <param name="nDataCriteriaFmtId">Returns the image data criteria format (if any).</param>
        /// <param name="rgDebugData">Returns the image debug data (if any).</param>
        /// <param name="nDebugDataFmtId">Returns the debug data format (if any).</param>
        /// <returns>The raw data is returned as a array of <i>byte</i> values.</returns>
        public byte[] GetRawImageData(RawImage img, out byte[] rgDataCriteria, out int? nDataCriteriaFmtId, out byte[] rgDebugData, out int? nDebugDataFmtId)
        {
            if (img.VirtualID == 0)
            {
                rgDataCriteria = getRawImage(img.DataCriteria, img.OriginalSourceID);
                nDataCriteriaFmtId = img.DataCriteriaFormatID;
                rgDebugData = getRawImage(img.DebugData, img.OriginalSourceID);
                nDebugDataFmtId = img.DebugDataFormatID;
                return getRawImage(img.Data, img.OriginalSourceID);
            }

            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                List<RawImage> rgImg = entities.RawImages.Where(p => p.ID == img.VirtualID).ToList();

                if (rgImg.Count == 0)
                {
                    rgDataCriteria = null;
                    nDataCriteriaFmtId = null;
                    rgDebugData = null;
                    nDebugDataFmtId = null;
                    return null;
                }

                rgDataCriteria = getRawImage(rgImg[0].DataCriteria, img.OriginalSourceID);
                nDataCriteriaFmtId = rgImg[0].DataCriteriaFormatID;
                rgDebugData = getRawImage(rgImg[0].DebugData, img.OriginalSourceID);
                nDebugDataFmtId = rgImg[0].DebugDataFormatID;

                return getRawImage(rgImg[0].Data, img.OriginalSourceID);
            }
        }

        /// <summary>
        /// Converts a set of bytes from a file path\name by loading its bytes and returning them, or if the original bytes do not
        /// contain a path, just returns the original bytes.
        /// </summary>
        /// <param name="rgData">Specifies the original bytes.</param>
        /// <returns>The actual data bytes (whether direct or loaded from file) are returned.</returns>
        protected byte[] getRawImage(byte[] rgData, int? nSecondarySrcId = null)
        {
            if (rgData == null || rgData.Length < 5)
                return rgData;

            string strFile = getImagePath(rgData);
            if (strFile == null)
                return rgData;

            if (m_strPrimaryImgPath == null)
                throw new Exception("You must open the database on a datasource.");

            // Get the file.
            if (nSecondarySrcId == null)
                return File.ReadAllBytes(m_strPrimaryImgPath + strFile);

            string strPath = m_strPrimaryImgPath;

            if (!File.Exists(strPath + strFile))
            {
                if (nSecondarySrcId.Value != m_nSecondarySourceID)
                {
                    m_nSecondarySourceID = nSecondarySrcId.Value;
                    m_strSecondaryImgPath = getImagePath(GetSourceName(m_nSecondarySourceID));
                }

                strPath = m_strSecondaryImgPath;
            }

            return File.ReadAllBytes(strPath + strFile);
        }

        /// <summary>
        /// Returns the file path contained within a byte array or <i>null</i> if no path is found.
        /// </summary>
        /// <param name="rgData">Specifies the bytes to check.</param>
        /// <returns>The actual embedded file path is returned if found, otherwise, <i>null</i> is returned.</returns>
        protected string getImagePath(byte[] rgData)
        {
            if (rgData == null)
                return null;

            if (rgData.Length < 5)
                return null;

            if (rgData[0] != 'F' ||
                rgData[1] != 'I' ||
                rgData[2] != 'L' ||
                rgData[3] != 'E' ||
                rgData[4] != ':')
                return null;

            return Encoding.ASCII.GetString(rgData, 5, rgData.Length - 5);
        }

        /// <summary>
        /// Update the label value of a label.
        /// </summary>
        /// <param name="nID">Specifies the ID of the label.</param>
        /// <param name="nLabel">Specifies the new label value.</param>
        /// <param name="bActivate">Specifies whether or not to activate the image, the default = <i>true</i>.</param>
        /// <param name="bSaveChanges">Specifies whether or not to save the changes, the default = <i>true</i>.</param>
        /// <returns>If the Label is found and set, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool UpdateActiveLabel(int nID, int nLabel, bool bActivate = true, bool bSaveChanges = true)
        {
            List<RawImage> rg = m_entities.RawImages.Where(p => p.ID == nID).ToList();

            if (rg.Count == 0)
                return false;

            rg[0].ActiveLabel = nLabel;
            rg[0].Active = bActivate;

            if (bSaveChanges)
                m_entities.SaveChanges();

            return true;
        }

        /// <summary>
        /// Update the description of a RawImage.
        /// </summary>
        /// <param name="nID">Specifies the ID of the RawImage.</param>
        /// <param name="strDescription">Specifies the new description.</param>
        /// <returns>If the RawImage is found and updated, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool UpdateRawImageDescription(int nID, string strDescription)
        {
            List<RawImage> rg = m_entities.RawImages.Where(p => p.ID == nID).ToList();

            if (rg.Count == 0)
                return false;

            rg[0].Description = strDescription;
            m_entities.SaveChanges();

            return true;
        }

        /// <summary>
        /// Update the RawImage description from a RawImage parameter.
        /// </summary>
        /// <param name="nID">Specifies the ID of the RawImage.</param>
        /// <param name="strParamName">Specifies the RawImage parameter name.</param>
        /// <returns>If the RawImage is found and updated, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool UpdateRawImageDescriptionFromParameter(int nID, string strParamName)
        {
            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                List<RawImage> rg = entities.RawImages.Where(p => p.ID == nID).ToList();

                if (rg.Count == 0)
                    return false;

                List<RawImageParameter> rgP = entities.RawImageParameters.Where(p => p.RawImageID == nID && p.Name == strParamName).ToList();

                if (rgP.Count == 0)
                    return false;

                if (rgP[0].TextValue == null || rgP[0].TextValue.Length == 0)
                    return false;

                if (rg[0].Description == rgP[0].TextValue)
                    return false;

                rg[0].Description = rgP[0].TextValue;
                entities.SaveChanges();

                return true;
            }
        }

        /// <summary>
        /// Returns the ID's of all RawImages within a data source.
        /// </summary>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <param name="nMax">Specifies the maximum number of ID's to query, the default is int max.</param>
        /// <param name="nLabel">Specifies a label from which images are to be queried, default is to ignore (-1).</param>
        /// <returns>The List of RawImage ID's is returned.</returns>
        public List<int> QueryAllRawImageIDs(int nSrcId = 0, int nMax = int.MaxValue, int nLabel = -1)
        {
            if (nSrcId == 0)
                nSrcId = m_src.ID;

            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                string strTop = (nMax == int.MaxValue) ? "" : "TOP " + nMax.ToString();
                string strCmd = "SELECT " + strTop + " ID FROM RawImages WHERE (SourceID = " + nSrcId.ToString() + ")";

                if (nLabel != -1)
                    strCmd += " AND (ActiveLabel = " + nLabel.ToString() + ")";

                return entities.Database.SqlQuery<int>(strCmd).ToList();
            }
        }

        /// <summary>
        /// Create a new RawImage but do not add it to the database.
        /// </summary>
        /// <param name="nIdx">Specifies the image index.</param>
        /// <param name="d">Specifies the SimpleDatum containing the data.</param>
        /// <param name="strDescription">Optionally, specifies the description (default = null).</param>
        /// <returns>The RawImage is returned.</returns>
        public RawImage CreateRawImage(int nIdx, SimpleDatum d, string strDescription = null)
        {
            DateTime dtMin = new DateTime(1980, 1, 1);
            RawImage img = new RawImage();
            bool bEncoded = false;
            img.Channels = d.Channels;
            img.Height = d.Height;
            img.Width = d.Width;
            img.SourceID = m_src.ID;
            img.TimeStamp = (d.TimeStamp < dtMin) ? dtMin : d.TimeStamp;
            img.Idx = nIdx;
            img.OriginalBoost = (short)d.Boost;
            img.ActiveBoost = (short)d.Boost;
            img.GroupID = d.GroupID;
            img.ActiveLabel = d.Label;
            img.OriginalLabel = d.Label;
            img.SourceID = m_src.ID;
            img.Active = true;
            img.AutoLabel = d.AutoLabeled;
            img.Description = strDescription;

            string strGuid = Guid.NewGuid().ToString();

            if (d.VirtualID > 0)
            {
                img.VirtualID = d.VirtualID;
                img.Encoded = d.IsRealData;
            }
            else
            {
                img.VirtualID = 0;
                img.Data = setImageByteData(d.GetByteData(out bEncoded), null, strGuid);
                img.Encoded = bEncoded;
            }

            if (d.DebugData != null)
            {
                img.DebugData = setImageByteData(d.DebugData, "dbg", strGuid);
                img.DebugDataFormatID = (byte)d.DebugDataFormat;
            }

            if (d.DataCriteria != null)
            {
                img.DataCriteria = setImageByteData(d.DataCriteria, "criteria", strGuid);
                img.DataCriteriaFormatID = (byte)d.DataCriteriaFormat;
            }

            return img;
        }

        /// <summary>
        /// When enabled, saves the bytes to file and returns the file name of the binary file saved as an
        /// array of bytes..
        /// </summary>
        /// <remarks>
        /// The path format returned is 'FILE:filepath'
        /// </remarks>
        /// <param name="rgImg">Specifies the bytes to check for a path.</param>
        /// <param name="strType">Specifies an extra name to add to the file name.</param>
        /// <param name="strGuid">Specifies an optional guid string to use as the file name.</param>
        /// <returns></returns>
        protected byte[] setImageByteData(byte[] rgImg, string strType = null, string strGuid = null)
        {
            if (rgImg == null)
                return null;

            if (!m_bEnableFileBasedData || rgImg.Length < 100)
                return rgImg;

            if (strGuid == null)
                strGuid = Guid.NewGuid().ToString();

            if (!Directory.Exists(m_strPrimaryImgPath))
                Directory.CreateDirectory(m_strPrimaryImgPath);

            string strTypeExt = (strType == null) ? "" : "." + strType;
            string strFile = strGuid + strTypeExt + ".bin";
            File.WriteAllBytes(m_strPrimaryImgPath + strFile, rgImg);

            string strTag = "FILE:" + strFile;
            return Encoding.ASCII.GetBytes(strTag);
        }

        /// <summary>
        /// The ConvertRawImagesSaveToFile method saves the image in the database to the file system and replaces the database data with the 
        /// path to the saved image, thus saving database space.
        /// </summary>
        /// <param name="nIdx">Specifies the first index of a RawImage to convert.</param>
        /// <param name="nCount">Specifies the number of RawImages to convert including and following the RawImage at the index.</param>
        /// <param name="evtCancel">Optionally, specifies a cancellation event.</param>
        /// <returns>Upon full completion, <i>true</i> is returned, otherwise <i>false</i> is returned when cancelled.</returns>
        public bool ConvertRawImagesSaveToFile(int nIdx, int nCount, CancelEvent evtCancel = null)
        {
            if (m_strPrimaryImgPath == null)
                m_strPrimaryImgPath = getImagePath();

            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                List<RawImage> rgImg = entities.RawImages.Where(p => p.SourceID == m_src.ID && p.Idx >= nIdx && p.Active == true).OrderBy(p => p.Idx).Take(nCount).ToList();
                string strPath;
                string strImgPath;
                string strTag;
                string strName;
                byte[] rgData;
                List<int?> rgId = new List<int?>();
                Dictionary<int, string> rgNames = new Dictionary<int, string>();

                for (int i=0; i<rgImg.Count; i++)
                {
                    if (evtCancel != null && evtCancel.WaitOne(0))
                        return false;

                    strName = Guid.NewGuid().ToString();
                    rgNames.Add(rgImg[i].ID, strName);

                    rgId.Add(rgImg[i].ID);
                    rgData = rgImg[i].Data;
                    if (rgData != null)
                    {
                        strPath = getImagePath(rgData);
                        if (strPath == null)
                        {
                            strImgPath = strName + ".bin";
                            File.WriteAllBytes(m_strPrimaryImgPath + strImgPath, rgData);
                            strTag = "FILE:" + strImgPath;
                            rgImg[i].Data = Encoding.ASCII.GetBytes(strTag);
                        }
                    }

                    rgData = rgImg[i].DebugData;
                    if (rgData != null)
                    {
                        strPath = getImagePath(rgData);
                        if (strPath == null)
                        {
                            strImgPath = strName + ".dbg.bin";
                            File.WriteAllBytes(m_strPrimaryImgPath + strImgPath, rgData);
                            strTag = "FILE:" + strImgPath;
                            rgImg[i].DebugData = Encoding.ASCII.GetBytes(strTag);
                        }
                    }

                    rgData = rgImg[i].DataCriteria;
                    if (rgData != null)
                    {
                        strPath = getImagePath(rgData);
                        if (strPath == null)
                        {
                            strImgPath = strName + ".criteria.bin";
                            File.WriteAllBytes(m_strPrimaryImgPath + strImgPath, rgData);
                            strTag = "FILE:" + strImgPath;
                            rgImg[i].DataCriteria = Encoding.ASCII.GetBytes(strTag);
                        }
                    }
                }

                entities.SaveChanges();

                List<RawImageParameter> rgParam = entities.RawImageParameters.Where(p => rgId.Contains(p.RawImageID)).ToList();
                for (int i = 0; i < rgParam.Count; i++)
                {
                    rgData = rgParam[i].Value;
                    if (rgData != null)
                    {
                        strPath = getImagePath(rgData);
                        if (strPath == null)
                        {
                            int nRawImgId = rgParam[i].RawImageID.GetValueOrDefault(0);
                            string strName1 = rgNames[nRawImgId];
                            strImgPath = strName1 + ".param_" + rgParam[i].Name + ".bin";
                            File.WriteAllBytes(m_strPrimaryImgPath + strImgPath, rgData);
                            strTag = "FILE:" + strImgPath;
                            rgParam[i].Value = Encoding.ASCII.GetBytes(strTag);
                        }
                    }
                }

                entities.SaveChanges();
            }

            return true;
        }

        /// <summary>
        /// The ConvertRawImagesSaveToDatabase method saves the image in the file system to the database and deletes the file from
        /// the file system.
        /// </summary>
        /// <param name="nIdx">Specifies the first index of a RawImage to convert.</param>
        /// <param name="nCount">Specifies the number of RawImages to convert including and following the RawImage at the index.</param>
        /// <param name="evtCancel">Optionally, specifies a cancellation event.</param>
        /// <returns>Upon full completion, <i>true</i> is returned, otherwise <i>false</i> is returned when cancelled.</returns>
        public bool ConvertRawImagesSaveToDatabase(int nIdx, int nCount, CancelEvent evtCancel = null)
        {
            if (m_strPrimaryImgPath == null)
                m_strPrimaryImgPath = getImagePath();

            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                List<RawImage> rgImg = entities.RawImages.Where(p => p.SourceID == m_src.ID && p.Idx >= nIdx && p.Active == true).OrderBy(p => p.Idx).Take(nCount).ToList();
                List<string> rgstrFiles = new List<string>();
                string strPath;
                byte[] rgData;
                List<int?> rgId = new List<int?>();

                for (int i = 0; i < rgImg.Count; i++)
                {
                    if (evtCancel != null && evtCancel.WaitOne(0))
                        return false;

                    rgData = rgImg[i].Data;
                    if (rgData != null)
                    {
                        strPath = getImagePath(rgData);
                        if (strPath != null)
                        {
                            rgImg[i].Data = File.ReadAllBytes(m_strPrimaryImgPath + strPath);
                            rgstrFiles.Add(m_strPrimaryImgPath + strPath);
                        }
                    }

                    rgData = rgImg[i].DebugData;
                    if (rgData != null)
                    {
                        strPath = getImagePath(rgData);
                        if (strPath != null)
                        {
                            rgImg[i].DebugData = File.ReadAllBytes(m_strPrimaryImgPath + strPath);
                            rgstrFiles.Add(m_strPrimaryImgPath + strPath);
                        }
                    }

                    rgData = rgImg[i].DataCriteria;
                    if (rgData != null)
                    {
                        strPath = getImagePath(rgData);
                        if (strPath != null)
                        {
                            rgImg[i].DataCriteria = File.ReadAllBytes(m_strPrimaryImgPath + strPath);
                            rgstrFiles.Add(m_strPrimaryImgPath + strPath);
                        }
                    }
                }

                entities.SaveChanges();

                List<RawImageParameter> rgParam = entities.RawImageParameters.Where(p => rgId.Contains(p.RawImageID)).ToList();
                for (int i = 0; i < rgParam.Count; i++)
                {
                    rgData = rgParam[i].Value;
                    if (rgData != null)
                    {
                        strPath = getImagePath(rgData);
                        if (strPath != null)
                        {
                            rgParam[i].Value = File.ReadAllBytes(m_strPrimaryImgPath + strPath);
                            rgstrFiles.Add(m_strPrimaryImgPath + strPath);
                        }
                    }
                }

                entities.SaveChanges();

                foreach (string strFile in rgstrFiles)
                {
                    File.Delete(strFile);
                }

                if (Directory.Exists(m_strPrimaryImgPath))
                {
                    if (Directory.GetFiles(m_strPrimaryImgPath).Length == 0)
                        Directory.Delete(m_strPrimaryImgPath);
                }
            }

            return true;
        }

        /// <summary>
        /// The FixupRawImageCopy method is used to fixup the OriginalSourceId by setting it to a secondary
        /// source ID in the event that the path created using the PrimarySourceID does not have the image
        /// data file.
        /// </summary>
        /// <remarks>
        /// When creating a copy of a Data Source that uses both training and testing Data Sources (e.g., 
        /// re-arranging the time period used for training vs testing), it is important that the 
        /// OriginalSourceID be set with the Data Source ID that holds the data file.
        /// </remarks>
        /// <param name="nImageID">Specifies the image to update.</param>
        /// <param name="nSecondarySrcId">Specifies the secondary Source ID to use if the data file is not found.</param>
        public void FixupRawImageCopy(int nImageID, int nSecondarySrcId)
        {
            if (m_strPrimaryImgPath == null)
                m_strPrimaryImgPath = getImagePath();

            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                List<RawImage> rgImg = entities.RawImages.Where(p => p.ID == nImageID).ToList();
                if (rgImg.Count > 0)
                {
                    string strPath;
                    int nVirtId = rgImg[0].VirtualID.GetValueOrDefault(0);

                    if (nVirtId > 0)
                    {
                        List<RawImage> rgImg2 = entities.RawImages.Where(p => p.ID == nVirtId).ToList();
                        if (rgImg2.Count > 0)
                        {
                            strPath = getImagePath(rgImg2[0].Data);
                            if (!File.Exists(m_strPrimaryImgPath + strPath))
                            {
                                rgImg[0].OriginalSourceID = nSecondarySrcId;
                                entities.SaveChanges();
                            }
                        }
                    }
                    else if (rgImg[0].Data != null)
                    {
                        strPath = getImagePath(rgImg[0].Data);
                        if (!File.Exists(m_strPrimaryImgPath + strPath))
                        {
                            rgImg[0].OriginalSourceID = nSecondarySrcId;
                            entities.SaveChanges();
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Saves a List of RawImages to the database.
        /// </summary>
        /// <param name="rgImg">Specifies the list of RawImages.</param>
        /// <param name="rgrgParam">Optionally, specifies the List of parameters to also save for each RawImage (default = null).</param>
        public void PutRawImages(List<RawImage> rgImg, List<List<ParameterData>> rgrgParam = null)
        {
            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                entities.Configuration.AutoDetectChangesEnabled = false;
                entities.Configuration.ValidateOnSaveEnabled = false;

                foreach (RawImage img in rgImg)
                {
                    entities.RawImages.Add(img);
                    AddLabelToCache(img.ActiveLabel.GetValueOrDefault());
                }

                entities.SaveChanges();

                if (rgrgParam != null && rgrgParam.Count == rgImg.Count)
                {
                    for (int i = 0; i < rgImg.Count; i++)
                    {
                        for (int j = 0; j < rgrgParam[i].Count; j++)
                        {
                            ParameterData p = rgrgParam[i][j];
                            string strName = p.Name;
                            string strVal = p.Value;
                            byte[] rgData = p.Data;

                            if (p.QueryValueFromImageID != 0)
                            {
                                RawImageParameter imgParam = GetRawImageParameterEx(p.QueryValueFromImageID, strName);

                                if (imgParam != null)
                                {
                                    strVal = imgParam.TextValue;
                                    rgData = imgParam.Value;
                                }
                            }

                            if (!String.IsNullOrEmpty(strVal) || rgData != null)
                                SetRawImageParameter(rgImg[i].ID, strName, strVal, rgData, false, entities);
                        }
                    }

                    entities.SaveChanges();
                }
            }
        }

        /// <summary>
        /// Save a SimpleDatum as a RawImage in the database.
        /// </summary>
        /// <param name="nIdx">Specifies the image index.</param>
        /// <param name="d">Specifies the SimpleDatum containing the data.</param>
        /// <param name="strDescription">Optionally, specifies a description for the RawImage (default = null).</param>
        /// <returns>The ID of the RawImage is returned.</returns>
        public int PutRawImage(int nIdx, SimpleDatum d, string strDescription = null)
        {
            RawImage img = CreateRawImage(nIdx, d, strDescription);
            m_entities.RawImages.Add(img);
            m_entities.SaveChanges();

            if (d is Datum)
            {
                string str = ((Datum)d).Tag as string;
                string strName = ((Datum)d).TagName;

                if (str != null && str.Length > 0 && strName != null && strName.Length > 0)
                    SetRawImageParameter(m_src.ID, img.ID, strName, str);
            }

            AddLabelToCache(d.Label);
 
            return img.ID;
        }

        /// <summary>
        /// Returns the RawImage with a given ID.
        /// </summary>
        /// <param name="nID">Specifies the RawImage ID.</param>
        /// <returns>The RawImage is returned, or <i>null</i> if not found.</returns>
        public RawImage GetRawImage(int nID)
        {
            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                List<RawImage> rgItems = entities.RawImages.Where(p => p.ID == nID).ToList();

                if (rgItems.Count == 0)
                    return null;

                return rgItems[0];
            }
        }

        /// <summary>
        /// Returns the number of RawImages in a data source.
        /// </summary>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <returns>The number of RawImages is returned.</returns>
        public int QueryRawImageCount(int nSrcId = 0)
        {
            if (nSrcId == 0)
                nSrcId = m_src.ID;

            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                return entities.RawImages.Where(p => p.SourceID == nSrcId).Count();
            }
        }

        /// <summary>
        /// Delete all RawImages in a data source.
        /// </summary>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        public void DeleteRawImages(int nSrcId = 0)
        {
            if (nSrcId == 0)
                nSrcId = m_src.ID;

            string strCmd = "DELETE FROM RawImages WHERE (SourceID = " + nSrcId.ToString() + ")";

            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                entities.Database.ExecuteSqlCommand(strCmd);

                string strImgPath = getImagePath();
                if (strImgPath != null)
                    deleteImages(strImgPath);
            }
        }

        private bool deleteImages(string strPath)
        {
            if (!Directory.Exists(strPath))
                return true;

            string[] rgstrFiles = Directory.GetFiles(strPath);
            foreach (string strFile in rgstrFiles)
            { 
                File.Delete(strFile);
            }

            Directory.Delete(strPath);

            return true;
        }

        /// <summary>
        /// Delete all RawImageResults for a data source.
        /// </summary>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        public void DeleteRawImageResults(int nSrcId = 0)
        {
            if (nSrcId == 0)
                nSrcId = m_src.ID;

            string strCmd = "DELETE FROM RawImageResults WHERE (SourceID = " + nSrcId.ToString() + ")";

            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                entities.Database.ExecuteSqlCommand(strCmd);
            }
        }

        /// <summary>
        /// Return the RawImageMean for the image mean from the open data source.
        /// </summary>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <returns>The RawImageMean is returned if found, otherwise <i>null</i> is returned.</returns>
        public RawImageMean GetRawImageMean(int nSrcId = 0)
        {
            if (nSrcId == 0)
                nSrcId = m_src.ID;

            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                List<RawImageMean> rgImg = entities.RawImageMeans.Where(p => p.SourceID == nSrcId).ToList();

                if (rgImg.Count == 0)
                    return null;

                return rgImg[0];
            }
        }
        
        /// <summary>
        /// Save the SimpleDatum as a RawImageMean in the database.
        /// </summary>
        /// <param name="sd">Specifies the data.</param>
        /// <param name="bUpdate">Specifies whether or not to update the mean image.</param>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <returns>The ID of the RawImageMean is returned.</returns>
        public int PutRawImageMean(SimpleDatum sd, bool bUpdate, int nSrcId = 0)
        {
            if (nSrcId == 0)
                nSrcId = m_src.ID;

            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                IQueryable<RawImageMean> iQuery = entities.RawImageMeans.Where(p => p.SourceID == nSrcId);
                if (iQuery != null)
                {
                    List<RawImageMean> rgMean = iQuery.ToList();
                    RawImageMean im = null;

                    if (rgMean.Count == 0)
                        im = new RawImageMean();
                    else
                        im = rgMean[0];

                    if (bUpdate || rgMean.Count == 0)
                    {
                        bool bEncoded = false;
                        im.Channels = sd.Channels;
                        im.Height = sd.Height;
                        im.Width = sd.Width;
                        im.SourceID = nSrcId;
                        im.Encoded = sd.IsRealData;
                        im.Data = sd.GetByteData(out bEncoded);
                    }

                    if (rgMean.Count == 0)
                        entities.RawImageMeans.Add(im);

                    if (rgMean.Count == 0 || bUpdate)
                        entities.SaveChanges();

                    return im.ID;
                }

                return 0;
            }
        }

        /// <summary>
        /// Delete all RawImageMeans for a data source.
        /// </summary>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        public void DeleteRawImageMeans(int nSrcId = 0)
        {
            if (nSrcId == 0)
                nSrcId = m_src.ID;

            string strCmd = "DELETE FROM RawImageMeans WHERE (SourceID = " + nSrcId.ToString() + ")";

            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                entities.Database.ExecuteSqlCommand(strCmd);
            }
        }

        /// <summary>
        /// Return the number of boosted images for a data source.
        /// </summary>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <returns>The number of boosted images is returned.</returns>
        public int GetBoostCount(int nSrcId = 0)
        {
            if (nSrcId == 0)
                nSrcId = m_src.ID;

            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                return entities.RawImages.Where(p => p.SourceID == nSrcId && p.ActiveBoost > 0).Count();
            }
        }

        /// <summary>
        /// Reset all image boosts for a data set.
        /// </summary>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        public void ResetAllBoosts(int nSrcId = 0)
        {
            if (nSrcId == 0)
                nSrcId = m_src.ID;

            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                string strCmd = "UPDATE RawImages SET ActiveBoost = OriginalBoost WHERE(SourceID = " + nSrcId.ToString() + ")";
                entities.Database.ExecuteSqlCommand(strCmd);
            }
        }

        /// <summary>
        /// Update the image boost of a given image.
        /// </summary>
        /// <param name="nImageID">Specifies the ID of the RawImage.</param>
        /// <param name="nBoost">Specifies the new boost value.</param>
        public void UpdateBoost(int nImageID, int nBoost)
        {
            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                string strCmd = "UPDATE RawImages SET ActiveBoost = " + nBoost.ToString() + " WHERE (ID = " + nImageID.ToString() + ")";
                entities.Database.ExecuteSqlCommand(strCmd);
            }
        }

        /// <summary>
        /// Reindex the RawImages of a data source.
        /// </summary>
        /// <param name="log">Specifies the Log to use for status output.</param>
        /// <param name="evtCancel">Specifies the cancel event used to cancel the operation.</param>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <returns>Upon completion <i>true</i> is returned, otherwise <i>false</i> is returned when cancelled.</returns>
        public bool ReindexRawImages(Log log, CancelEvent evtCancel, int nSrcId = 0)
        {
            if (nSrcId == 0 && m_src != null)
                nSrcId = m_src.ID;

            if (nSrcId == 0)
                return false;

            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                log.WriteLine("Querying images...");

                Stopwatch sw = new Stopwatch();
                List<RawImage> rgImg = entities.RawImages.Where(p => p.SourceID == nSrcId).OrderBy(p => p.TimeStamp).ThenBy(p => p.ID).ToList();

                sw.Start();
                int nIdx = 0;

                for (int i = 0; i < rgImg.Count; i++)
                {
                    if (rgImg[i].Active.GetValueOrDefault(false))
                    {
                        rgImg[i].Idx = nIdx;
                        nIdx++;
                    }
                    else
                    {
                        rgImg[i].Idx = -1;
                    }

                    if (i % 1000 == 0)
                        entities.SaveChanges();

                    if (sw.Elapsed.TotalMilliseconds > 1000)
                    {
                        log.Progress = ((double)i / (double)rgImg.Count);
                        log.WriteLine("reindexing at " + log.Progress.ToString("P"));
                        sw.Restart();
                    }

                    if (evtCancel.WaitOne(0))
                        return false;
                }

                entities.SaveChanges();

                return true;
            }
        }

        /// <summary>
        /// Updates a given image's source ID.
        /// </summary>
        /// <param name="nImageID">Specifies the ID of the image to update.</param>
        /// <param name="nSrcID">Specifies the new source ID.</param>
        /// <returns>If the source ID is updated, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool UpdateRawImageSourceID(int nImageID, int nSrcID)
        {
            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                List<RawImage> rgImg = entities.RawImages.Where(p => p.ID == nImageID).ToList();
                if (rgImg.Count > 0)
                {
                    if (rgImg[0].SourceID != nSrcID)
                    {
                        rgImg[0].SourceID = nSrcID;
                        entities.SaveChanges();
                        return true;
                    }
                }
            }

            return false;
        }

        /// <summary>
        /// Activate/Deactivate a given image.
        /// </summary>
        /// <param name="nImageID">Specifies the ID of the image to activate/deactivate.</param>
        /// <param name="bActivate">Specifies whether to activate (<i>true</i>) or deactivate (<i>false</i>) the image.</param>
        /// <returns>If the active state is changed, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool ActivateRawImage(int nImageID, bool bActivate)
        {
            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                List<RawImage> rgImg = entities.RawImages.Where(p => p.ID == nImageID).ToList();
                if (rgImg.Count == 0)
                    return false;

                if (rgImg[0].Active == bActivate)
                    return false;

                rgImg[0].Active = bActivate;
                entities.SaveChanges();
            }

            return true;
        }

        /// <summary>
        /// Activate all raw images associated with a set of source ID's.
        /// </summary>
        /// <param name="rgSrcId">Specifies the source ID's.</param>
        public void ActivateAllRawImages(params int[] rgSrcId)
        {
            if (rgSrcId.Length == 0)
                throw new Exception("You must specify at least one source iD.");

            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                string strCmd = "UPDATE RawImages SET [Active] = 1 WHERE (";

                for (int i=0; i<rgSrcId.Length; i++)
                {
                    strCmd += "SourceID = " + rgSrcId[i].ToString();

                    if (i < rgSrcId.Length - 1)
                        strCmd += " OR ";
                }

                strCmd += ")";

                entities.Database.ExecuteSqlCommand(strCmd);
            }
        }

        #endregion


        //---------------------------------------------------------------------
            //  RawImage Results
            //---------------------------------------------------------------------
        #region RawImage Results

        /// <summary>
            /// Save the results of a Run as a RawImageResult.
            /// </summary>
            /// <param name="nSrcId">Specifies the ID of the data source.</param>
            /// <param name="nIdx">Specifies the index of the result.</param>
            /// <param name="nLabel">Specifies the expected label of the result.</param>
            /// <param name="dt">Specifies the time-stamp of the result.</param>
            /// <param name="rgResults">Specifies the results of the run as a list of (int nLabel, double dfReult) values.</param>
            /// <param name="bInvert">Specifies whether or not the results are inverted.</param>
            /// <returns></returns>
        public int PutRawImageResults(int nSrcId, int nIdx, int nLabel, DateTime dt, List<KeyValuePair<int, double>> rgResults, bool bInvert)
        {
            if (rgResults.Count == 0)
                throw new Exception("You must have at least one result!");

            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                List<RawImageResult> rg = entities.RawImageResults.Where(p => p.SourceID == nSrcId && p.Idx == nIdx).ToList();
                RawImageResult r;

                if (rg.Count == 0)
                {
                    r = new RawImageResult();
                    r.Idx = nIdx;
                    r.SourceID = nSrcId;
                }
                else
                {
                    r = rg[0];
                }

                r.Label = nLabel;
                r.ResultCount = rgResults.Count;
                r.Results = ResultDescriptor.CreateResults(rgResults, bInvert);
                r.TimeStamp = dt;

                if (rg.Count == 0)
                    entities.RawImageResults.Add(r);

                entities.SaveChanges();

                return r.ID;
            }
        }

        /// <summary>
        /// Returns the RawImageResults for a data source.
        /// </summary>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <returns>The list of RawImageResults is returned.</returns>
        public List<RawImageResult> GetRawImageResults(int nSrcId = 0)
        {
            if (nSrcId == 0)
                nSrcId = m_src.ID;

            using (DNNEntities entities = MyCaffe.imagedb.EntitiesConnection.CreateEntities())
            {
                return entities.RawImageResults.Where(p => p.SourceID == nSrcId).OrderBy(p => p.TimeStamp).ToList();
            }
        }

        #endregion


        //---------------------------------------------------------------------
        //  RawImage Parameters
        //---------------------------------------------------------------------
        #region RawImage Parameters

        /// <summary>
        /// Return the string value of a RawImage parameter.
        /// </summary>
        /// <param name="nRawImageID">Specifies the ID of the RawImage.</param>
        /// <param name="strName">Specifies the name of the parameter.</param>
        /// <param name="strDefault">Specifies the default value to return if the RawImage or parameter are not found.</param>
        /// <returns>The parameter value is returned as a string.</returns>
        public string GetRawImageParameter(int nRawImageID, string strName, string strDefault)
        {
            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                List<RawImageParameter> rgP = entities.RawImageParameters.Where(p => p.RawImageID == nRawImageID && p.Name == strName).ToList();

                if (rgP.Count == 0)
                    return strDefault;

                return rgP[0].TextValue;
            }
        }

        /// <summary>
        /// Return the <i>int</i> value of a RawImage parameter.
        /// </summary>
        /// <param name="nRawImageID">Specifies the ID of the RawImage.</param>
        /// <param name="strName">Specifies the name of the parameter.</param>
        /// <param name="nDefault">Specifies the default value to return if the RawImage or parameter are not found.</param>
        /// <returns>The parameter value is returned as a <i>int</i>.</returns>
        public int GetRawImageParameter(int nRawImageID, string strName, int nDefault)
        {
            string str = GetRawImageParameter(nRawImageID, strName, null);

            if (str == null)
                return nDefault;

            return int.Parse(str);
        }

        /// <summary>
        /// Return the <i>double</i> value of a RawImage parameter.
        /// </summary>
        /// <param name="nRawImageID">Specifies the ID of the RawImage.</param>
        /// <param name="strName">Specifies the name of the parameter.</param>
        /// <param name="dfDefault">Specifies the default value to return if the RawImage or parameter are not found.</param>
        /// <returns>The parameter value is returned as a <i>double</i>.</returns>
        public double GetRawImageParameter(int nRawImageID, string strName, double dfDefault)
        {
            string str = GetRawImageParameter(nRawImageID, strName, null);

            if (str == null)
                return dfDefault;

            return double.Parse(str);
        }

        /// <summary>
        /// Return the <i>bool</i> value of a RawImage parameter.
        /// </summary>
        /// <param name="nRawImageID">Specifies the ID of the RawImage.</param>
        /// <param name="strName">Specifies the name of the parameter.</param>
        /// <param name="bDefault">Specifies the default value to return if the RawImage or parameter are not found.</param>
        /// <returns>The parameter value is returned as a <i>bool</i>.</returns>
        public bool GetRawImageParameter(int nRawImageID, string strName, bool bDefault)
        {
            string str = GetRawImageParameter(nRawImageID, strName, null);

            if (str == null)
                return bDefault;

            return bool.Parse(str);
        }

        /// <summary>
        /// Return the <i>byte</i> array data of a RawImage parameter.
        /// </summary>
        /// <param name="nRawImageID">Specifies the ID of the RawImage.</param>
        /// <param name="strName">Specifies the name of the parameter.</param>
        /// <returns>The parameter <i>byte</i> array data is returned.</returns>
        public byte[] GetRawImageParameterData(int nRawImageID, string strName)
        {
            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                List<RawImageParameter> rgP = entities.RawImageParameters.Where(p => p.RawImageID == nRawImageID && p.Name == strName).ToList();

                if (rgP.Count == 0)
                    return null;

                return getRawImage(rgP[0].Value);
            }
        }

        /// <summary>
        /// Returns the RawImageParameter entity given the image ID and parameter name..
        /// </summary>
        /// <param name="nRawImageID">Specifies the ID of the RawImage.</param>
        /// <param name="strName">Specifies the parameter name.</param>
        /// <returns>The RawImageParameter entity is returned.</returns>
        public RawImageParameter GetRawImageParameterEx(int nRawImageID, string strName)
        {
            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                List<RawImageParameter> rgP = entities.RawImageParameters.Where(p => p.RawImageID == nRawImageID && p.Name == strName).ToList();

                if (rgP.Count == 0)
                    return null;

                return rgP[0];
            }
        }

        /// <summary>
        /// Add a new RawImage parameter (or update an existing if found).
        /// </summary>
        /// <param name="nRawImageID">Specifies the ID of the RawImage.</param>
        /// <param name="strName">Specifies the name of the parameter.</param>
        /// <param name="strValue">Specifies the value of the parameter as a string.</param>
        /// <param name="rgData">Optionally, specifies the <i>byte</i> data associated with the parameter (default = null).</param>
        /// <param name="bSave">Optionally, specifies to save the data to the database (default = true).</param>
        /// <param name="entities">Optionally, specifies the entities to use (default = null in which case the open data source entities are used).</param>
        /// <returns>The ID of the RawImageParameter is returned.</returns>
        public int SetRawImageParameter(int nRawImageID, string strName, string strValue, byte[] rgData = null, bool bSave = true, DNNEntities entities = null)
        {
            if (entities == null)
                entities = m_entities;

            List<RawImageParameter> rgP = entities.RawImageParameters.Where(p => p.RawImageID == nRawImageID && p.Name == strName).ToList();
            RawImageParameter riP = null;

            if (rgP.Count == 0)
            {
                riP = new RawImageParameter();
                riP.RawImageID = nRawImageID;
                riP.SourceID = m_src.ID;
                riP.Name = strName;
            }
            else
            {
                riP = rgP[0];
            }

            riP.TextValue = strValue;
            riP.Value = setImageByteData(rgData, "param_" + strName);

            if (rgP.Count == 0)
                entities.RawImageParameters.Add(riP);

            if (bSave)
                entities.SaveChanges();

            return riP.ID;
        }

        /// <summary>
        /// Add a new RawImage parameter (or update an existing if found).
        /// </summary>
        /// <param name="nSrcId">Specifies the ID of the data source.</param>
        /// <param name="nRawImageID">Specifies the ID of the RawImage.</param>
        /// <param name="strName">Specifies the name of the parameter.</param>
        /// <param name="strValue">Specifies the value of the parameter as a string.</param>
        /// <param name="rgData">Optionally, specifies the <i>byte</i> data associated with the parameter (default = null).</param>
        /// <returns>The ID of the RawImageParameter is returned.</returns>
        public int SetRawImageParameter(int nSrcId, int nRawImageID, string strName, string strValue, byte[] rgData = null)
        {
            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                List<RawImageParameter> rgP = entities.RawImageParameters.Where(p => p.RawImageID == nRawImageID && p.Name == strName).ToList();
                RawImageParameter riP = null;

                if (rgP.Count == 0)
                {
                    riP = new RawImageParameter();
                    riP.RawImageID = nRawImageID;
                    riP.SourceID = nSrcId;
                    riP.Name = strName;
                }
                else
                {
                    riP = rgP[0];
                }

                riP.TextValue = strValue;
                riP.Value = setImageByteData(rgData, "param_" + strName);

                if (rgP.Count == 0)
                    entities.RawImageParameters.Add(riP);

                entities.SaveChanges();

                return riP.ID;
            }
        }

        /// <summary>
        /// Set the RawImage parameter for all RawImages with the given time-stamp in the data source.
        /// </summary>
        /// <param name="dt">Specifies the time-stamp.</param>
        /// <param name="strName">Specifies the name of the parameter.</param>
        /// <param name="strValue">Specifies the value of the parameter as a string.</param>
        /// <param name="rgData">Optionally, specifies the <i>byte</i> data associated with the parameter (default = null).</param>
        /// <returns>The ID of the RawImageParameter is returned.</returns>
        public int SetRawImageParameterAt(DateTime dt, string strName, string strValue, byte[] rgData)
        {
            List<RawImage> rg = m_entities.RawImages.Where(p => p.SourceID == m_src.ID && p.TimeStamp == dt).ToList();

            if (rg.Count == 0)
                return 0;

            return SetRawImageParameter(m_src.ID, rg[0].ID, strName, strValue, rgData);
        }

        /// <summary>
        /// Delete all RawImage parameters within a data source.
        /// </summary>
        /// <param name="nSrcId">Specifies the ID of the data source.</param>
        public void DeleteRawImageParameters(int nSrcId)
        {
            string strCmd = "DELETE FROM RawImageParameters WHERE (SourceID = " + nSrcId.ToString() + ")";

            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                entities.Database.ExecuteSqlCommand(strCmd);
            }
        }

        /// <summary>
        /// Returns the RawImage parameter count for a data source.
        /// </summary>
        /// <param name="strName">Specifies the parameter name.</param>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <param name="strType">Optionally, specifies the parameter type (default = "TEXT").</param>
        /// <returns>The number of RawImage parameters is returned.</returns>
        public int GetRawImageParameterCount(string strName, int nSrcId = 0, string strType = "TEXT")
        {
            if (nSrcId == 0)
                nSrcId = m_src.ID;

            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                if (strType.ToLower() == "text")
                    return entities.RawImageParameters.Where(p => p.SourceID == nSrcId && p.Name == strName && p.TextValue != null).Count();

                if (strType.ToLower() == "value")
                    return entities.RawImageParameters.Where(p => p.SourceID == nSrcId && p.Name == strName && p.Value != null).Count();
            }

            return 0;
        }

        /// <summary>
        /// Returns whether or not a given RawImage parameter exists.
        /// </summary>
        /// <param name="strName">Specifies the parameter name.</param>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <param name="strType">Optionally, specifies the parameter type (default = "TEXT").</param>
        /// <returns>Returns <i>true</i> if the parameter exists, <i>false</i> otherwise.</returns>
        public bool GetRawImageParameterExist(string strName, int nSrcId = 0, string strType = "TEXT")
        {
            if (nSrcId == 0)
                nSrcId = m_src.ID;

            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                if (strType.ToLower() == "text")
                {
                    int nCount = entities.RawImageParameters.Where(p => p.SourceID == nSrcId && p.Name == strName && p.TextValue != null).Take(1).Count();
                    return (nCount > 0) ? true : false;
                }

                if (strType.ToLower() == "value")
                {
                    int nCount = entities.RawImageParameters.Where(p => p.SourceID == nSrcId && p.Name == strName && p.Value != null).Take(1).Count();
                    return (nCount > 0) ? true : false;
                }
            }

            return false;
        }

        /// <summary>
        /// Returns a list of distinct RawImage parameter descriptions for a data source.
        /// </summary>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <returns>The list of distinct descriptions is returned.</returns>
        public List<string> GetRawImageDistinctParameterDescriptions(int nSrcId = 0)
        {
            if (nSrcId == 0)
                nSrcId = m_src.ID;

            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                string strCmd = "SELECT DISTINCT [TextValue] FROM RawImageParameters WHERE SourceID = " + nSrcId.ToString();
                DbRawSqlQuery<string> res = entities.Database.SqlQuery<string>(strCmd);
                List<string> rgstr = res.ToList<string>();

                List<string> rgstr1 = new List<string>();

                foreach (string str in rgstr)
                {
                    if (str.Trim().Length > 0)
                        rgstr1.Add(str);
                }

                return rgstr1;
            }
        }

        #endregion


        //---------------------------------------------------------------------
            //  RawImage Groups
            //---------------------------------------------------------------------
        #region RawImage Groups

        /// <summary>
        /// Adds a new RawImage group to the database.
        /// </summary>
        /// <param name="img">Specifies an image associated with the group.</param>
        /// <param name="nIdx">Specifies an index associated with the group.</param>
        /// <param name="dtStart">Specifies the start time stamp for the group.</param>
        /// <param name="dtEnd">Specifies the end time stamp for the group.</param>
        /// <param name="rgProperties">Specifies the properties of the group.</param>
        /// <returns>The ID of the RawImageGroup is returned.</returns>
        public int AddRawImageGroup(Image img, int nIdx, DateTime dtStart, DateTime dtEnd, List<double> rgProperties)
        {
            RawImageGroup g = new RawImageGroup();

            g.RawData = ImageTools.ImageToByteArray(img);
            g.Idx = nIdx;
            g.StartDate = dtStart;
            g.EndDate = dtEnd;

            if (rgProperties != null)
            {
                if (rgProperties.Count > 0)
                    g.Property1 = (decimal)rgProperties[0];

                if (rgProperties.Count > 1)
                    g.Property2 = (decimal)rgProperties[1];

                if (rgProperties.Count > 2)
                    g.Property3 = (decimal)rgProperties[2];

                if (rgProperties.Count > 3)
                    g.Property4 = (decimal)rgProperties[3];

                if (rgProperties.Count > 4)
                    g.Property5 = (decimal)rgProperties[4];
            }

            return PutRawImageGroup(g);
        }

        /// <summary>
        /// Adds a RawImageGroup to the database.
        /// </summary>
        /// <param name="g">Specifies the group to add.</param>
        /// <returns>The ID of the RawImageGroup is returned.</returns>
        public int PutRawImageGroup(RawImageGroup g)
        {
            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                List<RawImageGroup> rgG = null;
                
                if (g.ID != 0)
                    rgG = entities.RawImageGroups.Where(p => p.ID == g.ID).ToList();

                if (rgG != null && rgG.Count > 0)
                {
                    rgG[0].EndDate = g.EndDate;
                    rgG[0].Idx = g.Idx;
                    rgG[0].Property1 = g.Property1;
                    rgG[0].Property2 = g.Property2;
                    rgG[0].Property3 = g.Property3;
                    rgG[0].Property4 = g.Property4;
                    rgG[0].Property5 = g.Property5;
                    rgG[0].Rating = g.Rating;
                    rgG[0].RawData = g.RawData;
                    rgG[0].StartDate = g.StartDate;                    
                }
                else
                {
                    entities.RawImageGroups.Add(g);
                }

                entities.SaveChanges();

                return g.ID;
            }
        }

        /// <summary>
        /// Searches for a RawImageGroup by index, start time-stamp and end time-stamp.
        /// </summary>
        /// <param name="nIdx">Specifies the index of the raw image group.</param>
        /// <param name="dtStart">Specifies the start time-stamp of the image group.</param>
        /// <param name="dtEnd">Specifies the end time-stamp of the image group.</param>
        /// <returns>If found, the RawImageGroup is returned, otherwise <i>null</i> is returned.</returns>
        public RawImageGroup FindRawImageGroup(int nIdx, DateTime dtStart, DateTime dtEnd)
        {
            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                List<RawImageGroup> rgG = entities.RawImageGroups.Where(p => p.Idx == nIdx && p.StartDate == dtStart && p.EndDate == dtEnd).ToList();

                if (rgG.Count == 0)
                    return null;

                return rgG[0];
            }
        }

        /// <summary>
        /// Searches fro the RawImageGroup ID.
        /// </summary>
        /// <param name="nIdx">Specifies the index of the raw image group.</param>
        /// <param name="dtStart">Specifies the start time-stamp of the image group.</param>
        /// <param name="dtEnd">Specifies the end time-stamp of the image group.</param>
        /// <returns>If found, the RawImageGroup ID is returned, otherwise 0 is returned.</returns>
        public int FindRawImageGroupID(int nIdx, DateTime dtStart, DateTime dtEnd)
        {
            RawImageGroup g = FindRawImageGroup(nIdx, dtStart, dtEnd);

            if (g == null)
                return 0;

            return g.ID;
        }

        /// <summary>
        /// Deletes all RawImage groups
        /// </summary>
        public void DeleteRawImageGroups()
        {
            string strCmd = "DELETE FROM RawImageGroups";

            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                entities.Database.ExecuteSqlCommand(strCmd);
            }
        }

        #endregion


        //---------------------------------------------------------------------
        //  Sources
        //---------------------------------------------------------------------
        #region Sources

        public List<int> GetAllDataSourcesIDs() /** @private */
        {
            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                return entities.Sources.Select(p => p.ID).ToList();
            }
        }

        /// <summary>
        /// Deletes the data source data for the open data source.
        /// </summary>
        public void DeleteSourceData()
        {
            DeleteSourceData(m_src.ID);
        }

        /// <summary>
        /// Update the SaveImagesToFile flag in a given Data Source.
        /// </summary>
        /// <param name="bSaveToFile">Specifies whether images are saved to the file system (<i>true</i>), or the directly to the database (<i>false</i>).</param>
        /// <param name="nSrcId">Optionally, specifies a source ID to use.  When 0, this parameter is ignored and the open Source is used instead.</param>
        public void UpdateSaveImagesToFile(bool bSaveToFile, int nSrcId = 0)
        {
            if (nSrcId == 0)
                nSrcId = m_src.ID;

            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                List<Source> rg = entities.Sources.Where(p => p.ID == nSrcId).ToList();

                if (rg.Count > 0)
                {
                    rg[0].SaveImagesToFile = bSaveToFile;
                    entities.SaveChanges();
                }
            }

            if (m_src != null)
                m_src.SaveImagesToFile = bSaveToFile;
        }

        /// <summary>
        /// Updates a data source.
        /// </summary>
        /// <param name="nChannels">Specifies the number of channels per item.</param>
        /// <param name="nWidth">Specifies the width of each item.</param>
        /// <param name="nHeight">Specifies the height of each item.</param>
        /// <param name="bDataIsReal">Specifies whether or not the item uses real or <i>byte</i> data.</param>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        public void UpdateSource(int nChannels, int nWidth, int nHeight, bool bDataIsReal, int nSrcId = 0)
        {
            if (nSrcId == 0)
                nSrcId = m_src.ID;

            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                List<Source> rg = entities.Sources.Where(p => p.ID == nSrcId).ToList();

                if (rg.Count > 0)
                {
                    rg[0].ImageChannels = nChannels;
                    rg[0].ImageWidth = nWidth; 
                    rg[0].ImageHeight = nHeight;
                    rg[0].ImageEncoded = bDataIsReal;
                    entities.SaveChanges();
                }
            }

            if (m_src != null)
            {
                m_src.ImageChannels = nChannels;
                m_src.ImageWidth = nWidth;
                m_src.ImageHeight = nHeight;
                m_src.ImageEncoded = bDataIsReal;
            }
        }

        /// <summary>
        /// Updates the source counts for the open data source.
        /// </summary>
        /// <param name="nImageCount">Specifies the new image count.</param>
        public void UpdateSourceCounts(int nImageCount)
        {
            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                int nSrcId = m_src.ID;
                List<Source> rg = entities.Sources.Where(p => p.ID == nSrcId).ToList();

                if (rg.Count > 0)
                {
                    rg[0].ImageCount = nImageCount;
                    entities.SaveChanges();
                }
            }

            m_src.ImageCount = nImageCount;
        }

        /// <summary>
        /// Updates the source counts for the open data source by querying the database for the counts.
        /// </summary>
        public void UpdateSourceCounts()
        {
            string strCmd = "SELECT COUNT(ID) FROM RawImages WHERE (SourceID = " + m_src.ID.ToString() + " AND Active=1)";
            DbRawSqlQuery<int> result = m_entities.Database.SqlQuery<int>(strCmd);
            List<int> rgResult = result.ToList();
            int nCount = 0;

            if (rgResult.Count > 0)
                nCount = rgResult[0];

            UpdateSourceCounts(nCount);
        }

        /// <summary>
        /// Returns the ID of a data source given its name.
        /// </summary>
        /// <param name="strName">Specifies the data source name.</param>
        /// <returns>The ID of the data source is returned.</returns>
        public int GetSourceID(string strName)
        {
            strName = convertWs(strName, '_');
            Source src = GetSource(strName);

            if (src == null)
                return 0;

            return src.ID;
        }

        /// <summary>
        /// Returns the name of a data source given its ID.
        /// </summary>
        /// <param name="nID">Specifies the ID of the data source.</param>
        /// <returns>The data source name is returned.</returns>
        public string GetSourceName(int nID)
        {
            Source src = GetSource(nID);

            if (src == null)
                return null;

            return src.Name;
        }

        /// <summary>
        /// Returns the Source entity given a data source name.
        /// </summary>
        /// <param name="strName">Specifies the data source name.</param>
        /// <returns>The Source entity is returned.</returns>
        public Source GetSource(string strName)
        {
            strName = convertWs(strName, '_');
            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                List<Source> rgSrc = entities.Sources.Where(p => p.Name == strName).ToList();

                if (rgSrc.Count == 0)
                    return null;

                return rgSrc[0];
            }
        }

        /// <summary>
        /// Returns the Source entity given a data source ID.
        /// </summary>
        /// <param name="nID">Specifies the ID of the data source.</param>
        /// <returns>The data source name is returned.</returns>
        public Source GetSource(int nID)
        {
            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                List<Source> rgSrc = entities.Sources.Where(p => p.ID == nID).ToList();

                if (rgSrc.Count == 0)
                    return null;

                return rgSrc[0];
            }
        }

        /// <summary>
        /// Adds or updates (if exists) a data source to the database.
        /// </summary>
        /// <param name="src">Specifies the Source entity to add.</param>
        /// <returns>The ID of the data source added is returned.</returns>
        public int PutSource(Source src)
        {
            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                List<Source> rgSrc = null;

                if (src.ID != 0)
                    rgSrc = entities.Sources.Where(p => p.ID == src.ID).ToList();
                else
                    rgSrc = entities.Sources.Where(p => p.Name == src.Name).ToList();

                if (rgSrc != null && rgSrc.Count > 0)
                {
                    rgSrc[0].ImageChannels = src.ImageChannels;
                    rgSrc[0].ImageCount = src.ImageCount;
                    rgSrc[0].ImageEncoded = src.ImageEncoded;
                    rgSrc[0].ImageHeight = src.ImageHeight;
                    rgSrc[0].ImageWidth = src.ImageWidth;
                    rgSrc[0].SaveImagesToFile = src.SaveImagesToFile;
                    src.ID = rgSrc[0].ID;
                }
                else
                {
                    entities.Sources.Add(src);
                }

                entities.SaveChanges();

                return src.ID;
            }
        }

        /// <summary>
        /// Adds a new data source to the database.
        /// </summary>
        /// <param name="strName">Specifies the data source name.</param>
        /// <param name="nChannels">Specifies the number of channels per item.</param>
        /// <param name="nWidth">Specifies the width of each item.</param>
        /// <param name="nHeight">Specifies the height of each item.</param>
        /// <param name="bDataIsReal">Specifies whether or not the item uses real or <i>byte</i> data.</param>
        /// <param name="nCopyOfSourceID">Optionally, specifies the ID of the source from which this source was copied.  If this is an original source, this parameter should be 0.</param>
        /// <param name="bSaveImagesToFile">Optionally, specifies whether or not to save the images to the file system (<i>true</i>) or directly into the database (<i>false</i>)  The default = <i>true</i>.</param>
        /// <returns>The ID of the data source added is returned.</returns>
        public int AddSource(string strName, int nChannels, int nWidth, int nHeight, bool bDataIsReal, int nCopyOfSourceID = 0, bool bSaveImagesToFile = true)
        {
            Source src = new Source();

            src.Name = convertWs(strName, '_');
            src.ImageChannels = nChannels;
            src.ImageHeight = nHeight;
            src.ImageWidth = nWidth;
            src.ImageEncoded = bDataIsReal;
            src.ImageCount = 0;
            src.SaveImagesToFile = bSaveImagesToFile;
            src.CopyOfSourceID = nCopyOfSourceID;

            return PutSource(src);
        }

        /// <summary>
        /// Delete a data source from the database.
        /// </summary>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        public void DeleteSource(int nSrcId = 0)
        {
            string strCmd;

            if (nSrcId == 0)
                nSrcId = m_src.ID;

            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                strCmd = "DELETE FROM Sources WHERE (ID = " + nSrcId.ToString() + ")";
                entities.Database.ExecuteSqlCommand(strCmd);

                strCmd = "DELETE FROM SourceParameters WHERE (SourceID = " + nSrcId.ToString() + ")";
                entities.Database.ExecuteSqlCommand(strCmd);

                strCmd = "DELETE FROM Labels WHERE (SourceID = " + nSrcId.ToString() + ")";
                entities.Database.ExecuteSqlCommand(strCmd);
            }

            DeleteSourceData(nSrcId);
        }

        /// <summary>
        /// Delete the list of data sources, listed by name, from the database.
        /// </summary>
        /// <param name="rgstrSrc">Specifies the list of data sources.</param>
        public void DeleteSources(params string[] rgstrSrc)
        {
            List<string> rgstrSrc1 = new List<string>();

            foreach (string str in rgstrSrc)
            {
                rgstrSrc1.Add(convertWs(str, '_'));
            }

            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                foreach (string str in rgstrSrc1)
                {                
                    List<Source> rgSrc = entities.Sources.Where(p => rgstrSrc.Contains(p.Name)).ToList();

                    foreach (Source src in rgSrc)
                    {
                        DeleteSource(src.ID);
                    }
                }
            }
        }

        /// <summary>
        /// Delete the data source data (images, means, results and parameters) from the database.
        /// </summary>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        public void DeleteSourceData(int nSrcId = 0)
        {
            if (nSrcId == 0)
                nSrcId = m_src.ID;

            DeleteRawImageMeans(nSrcId);
            DeleteRawImageResults(nSrcId);
            DeleteRawImageParameters(nSrcId);
            DeleteRawImages(nSrcId);
        }

        /// <summary>
        /// Delete the data source data (images, means, results and parameters) from the database.
        /// </summary>
        /// <param name="strSrc">Specifies the data source name.</param>
        public void DeleteSourceData(string strSrc)
        {
            DeleteSourceData(GetSourceID(convertWs(strSrc, '_')));
        }

        /// <summary>
        /// Returns a dictionary of the data source parameters.
        /// </summary>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <returns>The dictionary of source parameter values is returned.</returns>
        public Dictionary<string, string> GetSourceParameters(int nSrcId = 0)
        {
            if (nSrcId == 0)
                nSrcId = m_src.ID;

            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                List<SourceParameter> rgP = entities.SourceParameters.Where(p => p.SourceID == nSrcId).ToList();
                Dictionary<string, string> rgPval = new Dictionary<string, string>();

                foreach (SourceParameter p in rgP)
                {
                    rgPval.Add(p.Name, p.Value);
                }

                return rgPval;
            }
        }

        /// <summary>
        /// Return the data source parameter as a string.
        /// </summary>
        /// <param name="strName">Specifies the parameter name.</param>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <returns>The parameter value is returned as a string.</returns>
        public string GetSourceParameter(string strName, int nSrcId = 0)
        {
            strName = convertWs(strName, '_');

            if (nSrcId == 0)
                nSrcId = m_src.ID;

            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                List<SourceParameter> rgP = entities.SourceParameters.Where(p => p.SourceID == nSrcId && p.Name == strName).ToList();

                if (rgP.Count == 0)
                    return null;

                return rgP[0].Value;
            }
        }

        /// <summary>
        /// Return the data source parameter as an <i>int</i>.
        /// </summary>
        /// <param name="strName">Specifies the parameter name.</param>
        /// <param name="nDefault">Specifies the default value returned if the parameter is not found.</param>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <returns>The parameter value is returned as an <i>int</i>.</returns>
        public int GetSourceParameter(string strName, int nDefault, int nSrcId = 0)
        {
            strName = convertWs(strName, '_');
            string strVal = GetSourceParameter(strName, nSrcId);

            if (strVal == null)
                return nDefault;

            return int.Parse(strVal);
        }

        /// <summary>
        /// Return the data source parameter as a <i>double</i>.
        /// </summary>
        /// <param name="strName">Specifies the parameter name.</param>
        /// <param name="dfDefault">Specifies the default value returned if the parameter is not found.</param>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <returns>The parameter value is returned as a <i>double</i>.</returns>
        public double GetSourceParameter(string strName, double dfDefault, int nSrcId = 0)
        {
            strName = convertWs(strName, '_');
            string strVal = GetSourceParameter(strName, nSrcId);

            if (strVal == null)
                return dfDefault;

            return double.Parse(strVal);
        }

        /// <summary>
        /// Return the data source parameter as a <i>bool</i>.
        /// </summary>
        /// <param name="strName">Specifies the parameter name.</param>
        /// <param name="bDefault">Specifies the default value returned if the parameter is not found.</param>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <returns>The parameter value is returned as a <i>bool</i>.</returns>
        public bool GetSourceParameter(string strName, bool bDefault, int nSrcId = 0)
        {
            strName = convertWs(strName, '_');
            string strVal = GetSourceParameter(strName, nSrcId);

            if (strVal == null)
                return bDefault;

            return bool.Parse(strVal);
        }

        /// <summary>
        /// Set the value of a data source parameter.
        /// </summary>
        /// <param name="strName">Specifies the parameter name.</param>
        /// <param name="strValue">Specifies the value of the parameter.</param>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        public void SetSourceParameter(string strName, string strValue, int nSrcId = 0)
        {
            strName = convertWs(strName, '_');

            if (nSrcId == 0)
                nSrcId = m_src.ID;

            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                List<SourceParameter> rgP = entities.SourceParameters.Where(p => p.SourceID == nSrcId && p.Name == strName).ToList();
                SourceParameter sP = null;

                if (rgP.Count == 0)
                {
                    sP = new SourceParameter();
                    sP.Name = strName;
                    sP.SourceID = nSrcId;
                }
                else
                {
                    sP = rgP[0];
                }

                sP.Value = strValue;

                if (rgP.Count == 0)
                    entities.SourceParameters.Add(sP);

                entities.SaveChanges();
            }
        }

        /// <summary>
        /// Returns the first time-stamp in the data source.
        /// </summary>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <param name="strDesc">Optionally, specifies a description to filter the values with (default = null, no filter).</param>
        /// <returns>If found, the time-stamp is returned, otherwise, DateTime.MinValue is returned.</returns>
        public DateTime GetFirstTimeStamp(int nSrcId = 0, string strDesc = null)
        {
            if (nSrcId == 0)
                nSrcId = m_src.ID;

            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                IQueryable<RawImage> iquery = entities.RawImages.Where(p => p.SourceID == nSrcId).OrderBy(p => p.TimeStamp);
                if (strDesc != null)
                    iquery = iquery.Where(p => p.Description == strDesc);

                List<RawImage> rgImages = iquery.Take(1).ToList();
                if (rgImages.Count == 0)
                    return DateTime.MinValue;

                return rgImages[0].TimeStamp.GetValueOrDefault();
            }
        }

        /// <summary>
        /// Returns the last time-stamp in the data source.
        /// </summary>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <param name="strDesc">Optionally, specifies a description to filter the values with (default = null, no filter).</param>
        /// <returns>If found, the time-stamp is returned, otherwise, DateTime.MinValue is returned.</returns>
        public DateTime GetLastTimeStamp(int nSrcId = 0, string strDesc = null)
        {
            if (nSrcId == 0)
                nSrcId = m_src.ID;

            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                IQueryable<RawImage> iquery = entities.RawImages.Where(p => p.SourceID == nSrcId).OrderByDescending(p => p.TimeStamp);
                if (strDesc != null)
                    iquery = iquery.Where(p => p.Description == strDesc);

                List<RawImage> rgImages = iquery.Take(1).ToList();
                if (rgImages.Count == 0)
                    return DateTime.MinValue;

                return rgImages[0].TimeStamp.GetValueOrDefault();
            }
        }

        #endregion


        //---------------------------------------------------------------------
        //  Datasets
        //---------------------------------------------------------------------
        #region Datasets

        /// <summary>
        /// Searches for the data set name based on the training and testing source names.
        /// </summary>
        /// <param name="strTrainSrc">Specifies the data source name for training.</param>
        /// <param name="strTestSrc">Specifies the data source name for testing.</param>
        /// <returns>If found, the dataset name is returned, otherwise <i>null</i> is returned.</returns>
        public string FindDatasetNameFromSourceName(string strTrainSrc, string strTestSrc)
        {
            int nTrainSrcID = 0;
            int nTestSrcID = 0;

            if (!String.IsNullOrEmpty(strTestSrc))
                nTestSrcID = GetSourceID(strTestSrc);

            if (!String.IsNullOrEmpty(strTrainSrc))
                nTrainSrcID = GetSourceID(strTrainSrc);

            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                List<Dataset> rgDs = null;

                if (nTestSrcID != 0 && nTrainSrcID != 0)
                    rgDs = entities.Datasets.Where(p => p.TrainingSourceID == nTrainSrcID && p.TestingSourceID == nTestSrcID).ToList();
                else if (nTestSrcID != 0)
                    rgDs = entities.Datasets.Where(p => p.TestingSourceID == nTestSrcID).ToList();
                else if (nTrainSrcID != 0)
                    rgDs = entities.Datasets.Where(p => p.TrainingSourceID == nTrainSrcID).ToList();

                if (rgDs != null && rgDs.Count > 0)
                    return rgDs[0].Name;
            }

            return null;
        }

        /// <summary>
        /// Returns a datasets ID given its name.
        /// </summary>
        /// <param name="strName">Specifies the dataset name.</param>
        /// <returns>The ID of the dataset is returned.</returns>
        public int GetDatasetID(string strName)
        {
            strName = convertWs(strName, '_');
            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                List<Dataset> rgDs = entities.Datasets.Where(p => p.Name == strName).ToList();

                if (rgDs.Count == 0)
                    return 0;

                return rgDs[0].ID;
            }
        }

        /// <summary>
        /// Returns the name of a dataset given its ID.
        /// </summary>
        /// <param name="nID">Specifies the dataset ID.</param>
        /// <returns>The dataset name is returned.</returns>
        public string GetDatasetName(int nID)
        {
            Dataset ds = GetDataset(nID);

            if (ds == null)
                return null;

            return ds.Name;
        }

        /// <summary>
        /// Returns the Dataset entity for a dataset ID.
        /// </summary>
        /// <param name="nID">Specifies the dataset ID.</param>
        /// <returns>The Dataset entity is returned.</returns>
        public Dataset GetDataset(int nID)
        {
            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                IQueryable<Dataset> iQuery = entities.Datasets.Where(p => p.ID == nID);

                if (iQuery.Count() == 0)
                    return null;

                List<Dataset> rgDs = iQuery.ToList();

                if (rgDs.Count == 0)
                    return null;

                return rgDs[0];
            }
        }

        /// <summary>
        /// Returns the Dataset entity for a dataset name.
        /// </summary>
        /// <param name="strName">Specifies the dataset name.</param>
        /// <returns>The Dataset entity is returned.</returns>
        public Dataset GetDataset(string strName)
        {
            strName = convertWs(strName, '_');
            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                List<Dataset> rgDs = entities.Datasets.Where(p => p.Name == strName).ToList();

                if (rgDs.Count == 0)
                    return null;

                return rgDs[0];
            }
        }

        /// <summary>
        /// Returns the Dataset entity containing the training and testing source names.
        /// </summary>
        /// <param name="strTestingSrc">Specifies the data source name for testing.</param>
        /// <param name="strTrainingSrc">Specifies the data source name for training.</param>
        /// <returns>If found the Dataset entity is returned, otherwise <i>null</i> is returned.</returns>
        public Dataset GetDataset(string strTestingSrc, string strTrainingSrc)
        {
            int nTestingSrcId = GetSourceID(strTestingSrc);
            int nTrainingSrcId = GetSourceID(strTrainingSrc);

            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                List<Dataset> rgDs = entities.Datasets.Where(p => p.TestingSourceID == nTestingSrcId && p.TrainingSourceID == nTrainingSrcId).ToList();

                if (rgDs.Count == 0)
                    return null;

                return rgDs[0];
            }
        }

        /// <summary>
        /// Add a new (or update an existing if exists) dataset to the database.
        /// </summary>
        /// <param name="nDsCreatorID">Specifies the ID of the creator.</param>
        /// <param name="strName">Specifies the name of the dataset.</param>
        /// <param name="nTestSrcId">Specifies the ID of the testing data source.</param>
        /// <param name="nTrainSrcId">Specifies the ID of the training data source.</param>
        /// <param name="nDsGroupID">Optionally, specifies the ID of the dataset group (default = 0).</param>
        /// <param name="nModelGroupID">Optionally, specifies the ID of the model group (default = 0).</param>
        /// <returns></returns>
        public int AddDataset(int nDsCreatorID, string strName, int nTestSrcId, int nTrainSrcId, int nDsGroupID = 0, int nModelGroupID = 0)
        {
            strName = convertWs(strName, '_');

            Source srcTest = GetSource(nTestSrcId);
            if (srcTest == null)
                throw new Exception("Could not find either the test source with ID = " + nTestSrcId.ToString() + "!");

            Source srcTrain = GetSource(nTrainSrcId);
            if (srcTrain == null)
                throw new Exception("Could not find either the train source with ID = " + nTrainSrcId.ToString() + "!");

            if (srcTest.ImageChannels.GetValueOrDefault() != srcTrain.ImageChannels.GetValueOrDefault())
                throw new Exception("The test and train sources have different image channels!");

            if (srcTest.ImageHeight.GetValueOrDefault() != srcTrain.ImageHeight.GetValueOrDefault())
                throw new Exception("The test and train sources have different image heights!");

            if (srcTest.ImageWidth.GetValueOrDefault() != srcTrain.ImageWidth.GetValueOrDefault())
                throw new Exception("The test and train sources have different image widths!");

            if (srcTest.ImageEncoded.GetValueOrDefault() != srcTrain.ImageEncoded.GetValueOrDefault())
                throw new Exception("The test and train sources have different image encodings!");

            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                List<Dataset> rgDs = entities.Datasets.Where(p => p.Name == strName && p.DatasetCreatorID == nDsCreatorID).ToList();
                Dataset ds;

                if (rgDs.Count > 0)
                {
                    ds = rgDs[0];
                }
                else
                {
                    ds = new Dataset();
                    ds.Name = strName;
                    ds.DatasetCreatorID = nDsCreatorID;
                }

                double dfPct = 0;
                double dfTotal = srcTest.ImageCount.GetValueOrDefault() + srcTrain.ImageCount.GetValueOrDefault();

                if (dfTotal > 0)
                    dfPct = srcTest.ImageCount.GetValueOrDefault() / dfTotal;

                ds.ImageChannels = srcTrain.ImageChannels;
                ds.ImageHeight = srcTrain.ImageHeight;
                ds.ImageWidth = srcTrain.ImageWidth;
                ds.ImageEncoded = srcTrain.ImageEncoded;
                ds.DatasetGroupID = nDsGroupID;
                ds.ModelGroupID = nModelGroupID;
                ds.TestingTotal = srcTest.ImageCount;
                ds.TrainingTotal = srcTrain.ImageCount;
                ds.TestingPercent = (decimal)dfPct;
                ds.Relabeled = false;
                ds.TestingSourceID = srcTest.ID;
                ds.TrainingSourceID = srcTrain.ID;

                if (rgDs.Count == 0)
                    entities.Datasets.Add(ds);

                entities.SaveChanges();

                UpdateLabelCounts(srcTest.ID);
                UpdateLabelCounts(srcTrain.ID);

                return ds.ID;
            }
        }

        /// <summary>
        /// Update the description of a given dataset.
        /// </summary>
        /// <param name="nDsId">Specifies the ID of the dataset to update.</param>
        /// <param name="strDesc">Specifies the new description.</param>
        public void UpdateDatasetDescription(int nDsId, string strDesc)
        {
            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                List<Dataset> rgDs = entities.Datasets.Where(p => p.ID == nDsId).ToList();

                if (rgDs.Count == 0)
                    return;

                rgDs[0].Description = strDesc;
                entities.SaveChanges();
            }
        }

        /// <summary>
        /// Update the dataset counts.
        /// </summary>
        /// <param name="nDsId">Specifies the ID of the dataset to update.</param>
        public void UpdateDatasetCounts(int nDsId)
        {
            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                List<Dataset> rgDs = entities.Datasets.Where(p => p.ID == nDsId).ToList();

                if (rgDs.Count == 0)
                    return;

                int nTestSrcId = rgDs[0].TestingSourceID.GetValueOrDefault();
                List<Source> rgSrcTest = entities.Sources.Where(p => p.ID == nTestSrcId).ToList();

                int nTrainSrcId = rgDs[0].TrainingSourceID.GetValueOrDefault();
                List<Source> rgSrcTrain = entities.Sources.Where(p => p.ID == nTrainSrcId).ToList();

                int nTestingTotal = rgSrcTest[0].ImageCount.GetValueOrDefault(0);
                int nTrainingTotal = rgSrcTrain[0].ImageCount.GetValueOrDefault(0);

                rgDs[0].TestingTotal = nTestingTotal;
                rgDs[0].TrainingTotal = nTrainingTotal;
                rgDs[0].TestingPercent = 0;

                if (nTrainingTotal + nTrainingTotal > 0)
                    rgDs[0].TestingPercent = (decimal)nTestingTotal / (decimal)(nTestingTotal + nTrainingTotal);

                entities.SaveChanges();
            }
        }

        /// <summary>
        /// Save the Dataset entity to the database.
        /// </summary>
        /// <param name="ds">Specifies the Dataset entity.</param>
        /// <returns>The ID of the dataset is returned.</returns>
        public int PutDataset(Dataset ds)
        {
            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                List<Dataset> rgDs = null;
                
                if (ds.ID != 0)
                    rgDs = entities.Datasets.Where(p => p.ID == ds.ID).ToList();

                if (rgDs != null && rgDs.Count > 0)
                {
                    rgDs[0].DatasetGroupID = ds.DatasetGroupID;
                    rgDs[0].Name = ds.Name;
                    rgDs[0].TestingSourceID = ds.TestingSourceID;
                    rgDs[0].TrainingSourceID = ds.TrainingSourceID;
                    rgDs[0].DatasetCreatorID = ds.DatasetCreatorID;
                    rgDs[0].ImageChannels = ds.ImageChannels;
                    rgDs[0].ImageEncoded = ds.ImageEncoded;
                    rgDs[0].ImageHeight = ds.ImageHeight;
                    rgDs[0].ImageWidth = ds.ImageWidth;
                    rgDs[0].ModelGroupID = ds.ModelGroupID;
                    rgDs[0].TestingPercent = ds.TestingPercent;
                    rgDs[0].TrainingTotal = ds.TrainingTotal;
                    rgDs[0].TestingTotal = ds.TestingTotal;
                }
                else
                {
                    entities.Datasets.Add(ds);
                }

                entities.SaveChanges();
            }

            return ds.ID;
        }

        /// <summary>
        /// Returns the DatasetGroup entity given a group ID.
        /// </summary>
        /// <param name="nGroupID">Specifies the ID of the group.</param>
        /// <returns>If found, the DatasetGroup is returned, otherwise <i>null</i> is returned.</returns>
        public DatasetGroup GetDatasetGroup(int nGroupID)
        {
            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                List<DatasetGroup> rgGroups = entities.DatasetGroups.Where(p => p.ID == nGroupID).ToList();

                if (rgGroups.Count == 0)
                    return null;

                return rgGroups[0];
            }
        }

        /// <summary>
        /// Returns the name of a dataset group given its ID.
        /// </summary>
        /// <param name="nGroupID">Specifies the ID of the group.</param>
        /// <returns>The dataset group name is returned.</returns>
        public string GetDatasetGroupName(int nGroupID)
        {
            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                List<DatasetGroup> rgGroups = entities.DatasetGroups.Where(p => p.ID == nGroupID).ToList();

                if (rgGroups.Count == 0)
                    return null;

                return rgGroups[0].Name;
            }
        }

        /// <summary>
        /// Returns all dataset parameters for a given dataset.
        /// </summary>
        /// <param name="nDsId">Specifies the ID of the dataset.</param>
        /// <returns>A dictionary of the dataset parameters is returned.</returns>
        public Dictionary<string, string> GetDatasetParameters(int nDsId)
        {
            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                List<DatasetParameter> rgP = entities.DatasetParameters.Where(p => p.DatasetID == nDsId).ToList();
                Dictionary<string, string> rgDsP = new Dictionary<string, string>();

                foreach (DatasetParameter p in rgP)
                {
                    rgDsP.Add(p.Name, p.Value);
                }

                return rgDsP;
            }
        }

        /// <summary>
        /// Returns the value of a dataset parameter as a string.
        /// </summary>
        /// <param name="nDsId">Specifies the ID of the dataset.</param>
        /// <param name="strName">Specifies the name of the parameter.</param>
        /// <returns>If the parameter is found it is returned as a string, otherwise <i>null</i> is returned.</returns>
        public string GetDatasetParameter(int nDsId, string strName)
        {
            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                List<DatasetParameter> rgP = entities.DatasetParameters.Where(p => p.DatasetID == nDsId && p.Name == strName).ToList();

                if (rgP.Count == 0)
                    return null;

                return rgP[0].Value;
            }
        }

        /// <summary>
        /// Returns the value of a dataset parameter as an <i>int</i>.
        /// </summary>
        /// <param name="nDsId">Specifies the ID of the dataset.</param>
        /// <param name="strName">Specifies the name of the parameter.</param>
        /// <param name="nDefault">Specifies the default value to return if not found.</param>
        /// <returns>If the parameter is found it is returned as an <i>int</i>, otherwise the default value is returned.</returns>
        public int GetDatasetParameter(int nDsId, string strName, int nDefault)
        {
            string strVal = GetDatasetParameter(nDsId, strName);

            if (strVal == null)
                return nDefault;

            return int.Parse(strVal);
        }

        /// <summary>
        /// Returns the value of a dataset parameter as a <i>double</i>.
        /// </summary>
        /// <param name="nDsId">Specifies the ID of the dataset.</param>
        /// <param name="strName">Specifies the name of the parameter.</param>
        /// <param name="dfDefault">Specifies the default value to return if not found.</param>
        /// <returns>If the parameter is found it is returned as a <i>double</i>, otherwise the default value is returned.</returns>
        public double GetDatasetParameter(int nDsId, string strName, double dfDefault)
        {
            string strVal = GetDatasetParameter(nDsId, strName);

            if (strVal == null)
                return dfDefault;

            return double.Parse(strVal);
        }

        /// <summary>
        /// Returns the value of a dataset parameter as a <i>bool</i>.
        /// </summary>
        /// <param name="nDsId">Specifies the ID of the dataset.</param>
        /// <param name="strName">Specifies the name of the parameter.</param>
        /// <param name="bDefault">Specifies the default value to return if not found.</param>
        /// <returns>If the parameter is found it is returned as a <i>bool</i>, otherwise the default value is returned.</returns>
        public bool GetDatasetParameter(int nDsId, string strName, bool bDefault)
        {
            string strVal = GetDatasetParameter(nDsId, strName);

            if (strVal == null)
                return bDefault;

            return bool.Parse(strVal);
        }

        /// <summary>
        /// Adds a new parameter or Sets the value of an existing dataset parameter.
        /// </summary>
        /// <param name="nDsId">Specifies the ID of the dataset.</param>
        /// <param name="strName">Specifies the name of the parameter.</param>
        /// <param name="strValue">Specifies the value of the parameter.</param>
        public void SetDatasetParameter(int nDsId, string strName, string strValue)
        {
            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                List<DatasetParameter> rgP = entities.DatasetParameters.Where(p => p.DatasetID == nDsId && p.Name == strName).ToList();
                DatasetParameter dsP = null;

                if (rgP.Count == 0)
                {
                    dsP = new DatasetParameter();
                    dsP.Name = strName;
                    dsP.DatasetID = nDsId;
                }
                else
                {
                    dsP = rgP[0];
                }

                dsP.Value = strValue;

                if (rgP.Count == 0)
                    entities.DatasetParameters.Add(dsP);

                entities.SaveChanges();
            }
        }

        /// <summary>
        /// Delete a dataset.
        /// </summary>
        /// <param name="strDsName">Specifies the dataset name.</param>
        /// <param name="bDeleteRelatedProjects">Specifies whether or not to also delete all projects using the dataset.  <b>WARNING!</b> Use this with caution for it will permenantly delete the projects and their results.</param>
        /// <param name="log">Specifies the Log object for status output.</param>
        /// <param name="evtCancel">Specifies the cancel event used to cancel the delete.</param>
        public virtual void DeleteDataset(string strDsName, bool bDeleteRelatedProjects, Log log, CancelEvent evtCancel)
        {
            Dataset ds = GetDataset(strDsName);
            Source srcTraining = GetSource(ds.TrainingSourceID.GetValueOrDefault());
            Source srcTesting = GetSource(ds.TestingSourceID.GetValueOrDefault());
            string strCmd;

            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                entities.Database.CommandTimeout = 180;

                strCmd = "DELETE RawImageParameters WHERE (SourceID = " + ds.TestingSourceID.GetValueOrDefault().ToString() + ") OR (SourceID = " + ds.TrainingSourceID.GetValueOrDefault().ToString() + ")";
                entities.Database.ExecuteSqlCommand(strCmd);

                strCmd = "DELETE Labels WHERE (SourceID = " + ds.TestingSourceID.GetValueOrDefault().ToString() + ") OR (SourceID = " + ds.TrainingSourceID.GetValueOrDefault().ToString() + ")";
                entities.Database.ExecuteSqlCommand(strCmd);

                strCmd = "DELETE RawImageMeans WHERE (SourceID = " + ds.TestingSourceID.GetValueOrDefault().ToString() + ") OR (SourceID = " + ds.TrainingSourceID.GetValueOrDefault().ToString() + ")";
                entities.Database.ExecuteSqlCommand(strCmd);

                strCmd = "DELETE RawImageResults WHERE (SourceID = " + ds.TestingSourceID.GetValueOrDefault().ToString() + ") OR (SourceID = " + ds.TrainingSourceID.GetValueOrDefault().ToString() + ")";
                entities.Database.ExecuteSqlCommand(strCmd);

                strCmd = "DELETE RawImages WHERE (SourceID = " + ds.TestingSourceID.GetValueOrDefault().ToString() + ") OR (SourceID = " + ds.TrainingSourceID.GetValueOrDefault().ToString() + ")";
                entities.Database.ExecuteSqlCommand(strCmd);

                strCmd = "DELETE SourceParameters WHERE (SourceID = " + ds.TestingSourceID.GetValueOrDefault().ToString() + ") OR (SourceID = " + ds.TrainingSourceID.GetValueOrDefault().ToString() + ")";
                entities.Database.ExecuteSqlCommand(strCmd);

                strCmd = "DELETE Sources WHERE (ID = " + ds.TestingSourceID.GetValueOrDefault().ToString() + ") OR (ID = " + ds.TrainingSourceID.GetValueOrDefault().ToString() + ")";
                entities.Database.ExecuteSqlCommand(strCmd);

                strCmd = "DELETE DatasetParameters WHERE (DatasetID = " + ds.ID.ToString() + ")";
                entities.Database.ExecuteSqlCommand(strCmd);

                strCmd = "DELETE Datasets WHERE (ID = " + ds.ID.ToString() + ")";
                entities.Database.ExecuteSqlCommand(strCmd);

                DeleteFiles del = new DeleteFiles(log, evtCancel);
                del.DeleteDirectory(getImagePathBase(srcTesting.Name, entities));
                del.DeleteDirectory(getImagePathBase(srcTraining.Name, entities));
            }
        }

        /// <summary>
        /// Returns a list of all datasets within a group.
        /// </summary>
        /// <param name="nDatasetGroupID">Specifies the ID of the dataset group.</param>
        /// <returns>The list of Dataset entities is returned.</returns>
        public List<Dataset> GetAllDatasets(int nDatasetGroupID)
        {
            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                if (nDatasetGroupID > 0)
                    return entities.Datasets.Where(p => p.DatasetGroupID == nDatasetGroupID).ToList();

                return entities.Datasets.ToList();
            }
        }

        /// <summary>
        /// Returns a list of all datasets within a group with dataset creators.
        /// </summary>
        /// <param name="nDatasetGroupID">Specifies the ID of the dataset group.</param>
        /// <returns>The list of Dataset entities is returned.</returns>
        public List<Dataset> GetAllDatasetsWithCreators(int nDatasetGroupID)
        {
            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                List<Dataset> rgDs = new List<Dataset>();
                IQueryable<Dataset> iQuery = entities.Datasets.Where(p => p.DatasetCreatorID > 0);

                if (iQuery.Count() > 0)
                {
                    rgDs = iQuery.ToList();

                    if (nDatasetGroupID > 0)
                        return rgDs.Where(p => p.DatasetGroupID == nDatasetGroupID).ToList();
                }

                return rgDs;
            }
        }

        /// <summary>
        /// Returns a list of all datasets within a group with dataset creators.
        /// </summary>
        /// <param name="nDsCreatorID">Specifies the ID of the dataset creator.</param>
        /// <param name="bRelabeled">Optionally, specifies whether or not only re-labeled datasets should be returned.</param>
        /// <returns>The list of Dataset entities is returned.</returns>
        public List<Dataset> GetAllDatasetsWithCreator(int nDsCreatorID, bool? bRelabeled = null)
        {
            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                if (bRelabeled.HasValue)
                {
                    bool bRelabeledValue = bRelabeled.Value;
                    return entities.Datasets.Where(p => p.DatasetCreatorID == nDsCreatorID && p.Relabeled == bRelabeled).ToList();
                }
                else
                {
                    return entities.Datasets.Where(p => p.DatasetCreatorID == nDsCreatorID).ToList();
                }
            }
        }

        /// <summary>
        /// Returns the ID of a dataset group given its name.
        /// </summary>
        /// <param name="strName">Specifies the name of the group.</param>
        /// <returns>Returns the ID of the dataset group, or 0 if not found.</returns>
        public int GetDatasetGroupID(string strName)
        {
            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                List<DatasetGroup> rgGroups = entities.DatasetGroups.Where(p => p.Name == strName).ToList();

                if (rgGroups.Count == 0)
                    return 0;

                return rgGroups[0].ID;
            }
        }

        /// <summary>
        /// Returns the name of a dataset creator given its ID.
        /// </summary>
        /// <param name="nDatasetCreatorID">Specifies the ID of the dataset creator.</param>
        /// <returns>Returns name of the dataset creator, or <i>null</i> if not found.</returns>
        public string GetDatasetCreatorName(int nDatasetCreatorID)
        {
            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                List<DatasetCreator> rgDsc = entities.DatasetCreators.Where(p => p.ID == nDatasetCreatorID).ToList();

                if (rgDsc.Count == 0)
                    return "";

                return rgDsc[0].Name;
            }
        }

        /// <summary>
        /// Returns the ID of a dataset creator given its name.
        /// </summary>
        /// <param name="strName">Specifies the name of the dataset creator.</param>
        /// <returns>Returns the ID of the dataset creator, or 0 if not found.</returns>
        public int GetDatasetCreatorID(string strName)
        {
            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                List<DatasetCreator> rgDsc = entities.DatasetCreators.Where(p => p.Name == strName).ToList();

                if (rgDsc.Count == 0)
                    return 0;

                return rgDsc[0].ID;
            }
        }

        /// <summary>
        /// Reset all dataset relabel flags with a given creator.
        /// </summary>
        /// <param name="nDsCreatorID">Specifies the ID of the dataset creator.</param>
        public void ResetAllDatasetRelabelWithCreator(int nDsCreatorID)
        {
            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                List<Dataset> rgDs = entities.Datasets.Where(p => p.DatasetCreatorID == nDsCreatorID && p.Relabeled == true).ToList();

                foreach (Dataset ds in rgDs)
                {
                    ds.Relabeled = false;
                }

                entities.SaveChanges();
            }
        }

        /// <summary>
        /// Update the dataset relabel flag for a dataset.
        /// </summary>
        /// <param name="nDsID">Specifies the ID of the dataset.</param>
        /// <param name="bRelabel">Specifies the re-label flag.</param>
        public void UpdateDatasetRelabel(int nDsID, bool bRelabel)
        {
            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                List<Dataset> rgDs = entities.Datasets.Where(p => p.ID == nDsID).ToList();

                foreach (Dataset ds in rgDs)
                {
                    ds.Relabeled = bRelabel;
                }

                entities.SaveChanges();
            }
        }

        /// <summary>
        /// Returns the minimum time-stamp for a dataset.
        /// </summary>
        /// <param name="nDsID">Specifies the ID of the dataset.</param>
        /// <returns>IF found, the minimum time-stamp is returned, otherwiseo the DateTime.MinValue is returned.</returns>
        public DateTime GetDatasetMinimumTimestamp(int nDsID)
        {
            Dataset ds = GetDataset(nDsID);

            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                int nSrcTestId = ds.TestingSourceID.GetValueOrDefault();
                int nSrcTrainId = ds.TrainingSourceID.GetValueOrDefault();

                List<RawImage> rgImg = entities.RawImages.Where(p => p.SourceID == nSrcTestId || p.SourceID == nSrcTrainId).OrderBy(p => p.TimeStamp).Take(1).ToList();
                if (rgImg.Count == 0)
                    return DateTime.MinValue;

                return rgImg[0].TimeStamp.GetValueOrDefault(DateTime.MinValue);
            }
        }

        /// <summary>
        /// Returns the maximum time-stamp for a dataset.
        /// </summary>
        /// <param name="nDsID">Specifies the ID of the dataset.</param>
        /// <returns>IF found, the maximum time-stamp is returned, otherwiseo the DateTime.MinValue is returned.</returns>
        public DateTime GetDatasetMaximumTimestamp(int nDsID)
        {
            Dataset ds = GetDataset(nDsID);

            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                int nSrcTestId = ds.TestingSourceID.GetValueOrDefault();
                int nSrcTrainId = ds.TrainingSourceID.GetValueOrDefault();

                List<RawImage> rgImg = entities.RawImages.Where(p => p.SourceID == nSrcTestId || p.SourceID == nSrcTrainId).OrderByDescending(p => p.TimeStamp).Take(1).ToList();
                if (rgImg.Count == 0)
                    return DateTime.MinValue;

                return rgImg[0].TimeStamp.GetValueOrDefault(DateTime.MinValue);
            }
        }

        /// <summary>
        /// Updates the dataset counts for a set of datasets.
        /// </summary>
        /// <param name="evtCancel">Specifies a cancel event used to abort the process.</param>
        /// <param name="log">Specifies the Log used for status output.</param>
        /// <param name="nDatasetCreatorID">Specifies the ID of the dataset creator.</param>
        /// <param name="rgstrDs">Specifies a list of the dataset names to update.</param>
        /// <param name="strParamNameForDescription">Specifies the parameter name used for descriptions.</param>
        public void UpdateDatasetCounts(CancelEvent evtCancel, Log log, int nDatasetCreatorID, List<string> rgstrDs, string strParamNameForDescription)
        {
            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                List<Dataset> rgDs = entities.Datasets.Where(p => p.DatasetCreatorID == nDatasetCreatorID).ToList();

                foreach (Dataset ds in rgDs)
                {
                    bool bContainsDs = false;
                    int nIdx;
                    Stopwatch sw = new Stopwatch();

                    if (evtCancel.WaitOne(0))
                    {
                        log.WriteLine("Aborting dataset count update.");
                        return;
                    }

                    if (rgstrDs != null)
                    {
                        foreach (string str in rgstrDs)
                        {
                            if (ds.Name.Contains(str))
                            {
                                bContainsDs = true;
                                break;
                            }
                        }

                        if (!bContainsDs)
                            continue;
                    }

                    log.WriteLine("Updating '" + ds.Name + "'...");

                    int nSrcIdTraining = ds.TrainingSourceID.GetValueOrDefault();
                    if (nSrcIdTraining> 0)
                    {
                        List<Source> rgSrc = entities.Sources.Where(p => p.ID == nSrcIdTraining).ToList();
                        if (rgSrc.Count > 0)
                        {
                            Source src = rgSrc[0];
                            int nImageCount = entities.RawImages.Where(p => p.SourceID == nSrcIdTraining).Count();

                            if (src.ImageCount != nImageCount)
                                src.ImageCount = nImageCount;

                            if (ds.TrainingTotal != nImageCount)
                                ds.TrainingTotal = nImageCount;

                            UpdateLabelCounts(src.ID);

                            if (bContainsDs)
                            {
                                List<int> rgId = QueryAllRawImageIDs(src.ID);

                                nIdx = 0;
                                sw.Restart();

                                foreach (int nID in rgId)
                                {
                                    if (evtCancel.WaitOne(0))
                                        return;

                                    UpdateRawImageDescriptionFromParameter(nID, strParamNameForDescription);
                                    nIdx++;

                                    if (sw.Elapsed.TotalMilliseconds > 1000)
                                    {
                                        log.Progress = ((double)nIdx / (double)rgId.Count);
                                        sw.Restart();
                                    }
                                }
                            }
                        }
                    }

                    int nSrcIdTesting = ds.TestingSourceID.GetValueOrDefault();
                    if (nSrcIdTesting > 0)
                    {
                        List<Source> rgSrc = entities.Sources.Where(p => p.ID == nSrcIdTesting).ToList();
                        if (rgSrc.Count > 0)
                        {
                            Source src = rgSrc[0];
                            int nImageCount = entities.RawImages.Where(p => p.SourceID == nSrcIdTesting).Count();

                            if (src.ImageCount != nImageCount)
                                src.ImageCount = nImageCount;

                            if (ds.TestingTotal != nImageCount)
                                ds.TestingTotal = nImageCount;

                            UpdateLabelCounts(src.ID);

                            if (bContainsDs)
                            {
                                List<int> rgId = QueryAllRawImageIDs(src.ID);

                                nIdx = 0;
                                sw.Restart();

                                foreach (int nID in rgId)
                                {
                                    if (evtCancel.WaitOne(0))
                                        return;

                                    UpdateRawImageDescriptionFromParameter(nID, strParamNameForDescription);
                                    nIdx++;

                                    if (sw.Elapsed.TotalMilliseconds > 1000)
                                    {
                                        log.Progress = ((double)nIdx / (double)rgId.Count);
                                        sw.Restart();
                                    }
                                }
                            }
                        }
                    }

                    double dfTestingPct = (double)ds.TestingTotal / (double)(ds.TestingTotal + ds.TrainingTotal);
                    if (ds.TestingPercent != (decimal)dfTestingPct)
                        ds.TestingPercent = (decimal)dfTestingPct;

                    entities.SaveChanges();
                }
            }
        }

        /// <summary>
        /// Returns the ModelGroup entity given the ID of a model group.
        /// </summary>
        /// <param name="nGroupID">Specifies the ID of the model group.</param>
        /// <returns>If found, the ModelGroup entity is returned, otherwise <i>null</i> is returned.</returns>
        public ModelGroup GetModelGroup(int nGroupID)
        {
            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                List<ModelGroup> rgGroups = entities.ModelGroups.Where(p => p.ID == nGroupID).ToList();

                if (rgGroups.Count == 0)
                    return null;

                return rgGroups[0];
            }
        }

        /// <summary>
        /// Returns the name of a model group given its ID.
        /// </summary>
        /// <param name="nGroupID">Specifies the ID of the model group.</param>
        /// <returns>The model group name is returned.</returns>
        public string GetModelGroupName(int nGroupID)
        {
            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                List<ModelGroup> rgGroups = entities.ModelGroups.Where(p => p.ID == nGroupID).ToList();

                if (rgGroups.Count == 0)
                    return null;

                return rgGroups[0].Name;
            }
        }

        /// <summary>
        /// Retruns the ID of a model group given its name.
        /// </summary>
        /// <param name="strGroup">Specifies the name of the model group.</param>
        /// <returns>The ID of the model group is returned.</returns>
        public int GetModelGroupID(string strGroup)
        {
            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                List<ModelGroup> rgGroup = entities.ModelGroups.Where(p => p.Name == strGroup).ToList();
                if (rgGroup.Count == 0)
                    return 0;

                return rgGroup[0].ID;
            }
        }

        /// <summary>
        /// Returns all Dataset entities within a given model group.
        /// </summary>
        /// <param name="nModelGroupId">Specifies the ID of a model group.</param>
        /// <returns>The list of Dataset entities is returned.</returns>
        public List<Dataset> GetAllDatasetsInModelGroup(int nModelGroupId)
        {
            using (DNNEntities entities = MyCaffe.imagedb.EntitiesConnection.CreateEntities())
            {
                return entities.Datasets.Where(p => p.ModelGroupID == nModelGroupId).ToList();
            }
        }

        /// <summary>
        /// Deletes a model group from the database.
        /// </summary>
        /// <param name="strGroup">Specifies the name of the group.</param>
        /// <param name="log">Specifies the Log object for status output.</param>
        /// <param name="evtCancel">Specifies the cancel event used to cancel the delete.</param>
        public void DeleteModelGroup(string strGroup, Log log, CancelEvent evtCancel)
        {
            int nModelGroupId = GetModelGroupID(strGroup);
            List<Dataset> rgDs = GetAllDatasetsInModelGroup(nModelGroupId);

            foreach (Dataset ds in rgDs)
            {
                DeleteDataset(ds.Name, false, log, evtCancel);
            }

            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                string strCmd = "DELETE FROM ModelGroups WHERE ([Name] = '" + strGroup + "')";
                entities.Database.ExecuteSqlCommand(strCmd);
            }

        }

        #endregion
    }

    /// <summary>
    /// The ParameterData class is used to save and retrieve parameter data.
    /// </summary>
    public class ParameterData
    {
        string m_strName;
        string m_strValue = null;
        byte[] m_rgValue = null;
        int m_nQueryFromImageID = 0;

        /// <summary>
        /// The ParameterData constructor.
        /// </summary>
        /// <param name="strName">Specifies the name of the parameter.</param>
        /// <param name="strValue">Specifies the value of the parameter.</param>
        /// <param name="rgData">Specifies the raw data associated with the parameter.</param>
        public ParameterData(string strName, string strValue, byte[] rgData)
        {
            m_strName = strName;
            m_strValue = strValue;
            m_rgValue = rgData;
        }

        /// <summary>
        /// The ParameterData constructor.
        /// </summary>
        /// <param name="strName">Specifies the name of the parameter.</param>
        /// <param name="nQueryFromImageID">Specifies a RawImage ID from which the parameter should be queried.</param>
        public ParameterData(string strName, int nQueryFromImageID)
        {
            m_strName = strName;
            m_nQueryFromImageID = nQueryFromImageID;
        }

        /// <summary>
        /// Returns the parameter name.
        /// </summary>
        public string Name
        {
            get { return m_strName; }
        }

        /// <summary>
        /// Returns the parameter value.
        /// </summary>
        public string Value
        {
            get { return m_strValue; }
        }

        /// <summary>
        /// Returns the raw data associated with the parameter.
        /// </summary>
        public byte[] Data
        {
            get { return m_rgValue; }
        }

        /// <summary>
        /// When specified, returns the RawImage ID from which the parameter is to be queried, otherwise returns 0.
        /// </summary>
        public int QueryValueFromImageID
        {
            get { return m_nQueryFromImageID; }
        }
    }
}
