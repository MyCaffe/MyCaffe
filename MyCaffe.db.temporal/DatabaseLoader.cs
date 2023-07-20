using MyCaffe.basecode.descriptors;
using MyCaffe.db.image;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static MyCaffe.basecode.descriptors.ValueStreamDescriptor;

namespace MyCaffe.db.temporal
{
    /// <summary>
    /// The DatasetLoader is used to load descriptors from the database.
    /// </summary>
    public class DatabaseLoader
    {
        DatabaseTemporal m_db = new DatabaseTemporal();

        /// <summary>
        /// The constructor.
        /// </summary>
        public DatabaseLoader()
        {
        }

        /// <summary>
        /// Load a dataset with the specified name from the database.
        /// </summary>
        /// <param name="strDs">Specifies the name of the dataset to load.</param>
        /// <returns>The dataset descriptor is returned.</returns>
        public DatasetDescriptor LoadDatasetFromDb(string strDs)
        {
            int nDsID = m_db.GetDatasetID(strDs);
            return LoadDatasetFromDb(nDsID);
        }

        /// <summary>
        /// Load a dataset with the specified ID from the database.
        /// </summary>
        /// <param name="nDsID">Specifies the ID of the dataset to load.</param>
        /// <returns>The dataset descriptor is returned.</returns>
        public DatasetDescriptor LoadDatasetFromDb(int nDsID)
        {
            Dataset ds = m_db.GetDataset(nDsID);
            if (ds == null)
                return null;

            GroupDescriptor grpd = LoadGroupFromDb(ds.DatasetGroupID);
            SourceDescriptor srcTrain = LoadSourceFromDb(ds.TrainingSourceID);
            SourceDescriptor srcTest = LoadSourceFromDb(ds.TestingSourceID);

            string strCreatorName = m_db.GetDatasetCreatorName(ds.DatasetCreatorID.GetValueOrDefault(0));

            return new DatasetDescriptor(ds.ID, ds.Name, grpd, null, srcTrain, srcTest, strCreatorName, ds.Description);
        }

        /// <summary>
        /// Load a data source with the specified ID from the database.
        /// </summary>
        /// <param name="nSrcId">Specifies the ID of the data source to load.</param>
        /// <returns>The source descriptor is returned.</returns>
        public SourceDescriptor LoadSourceFromDb(int? nSrcId)
        {
            Source src = m_db.GetSource(nSrcId.GetValueOrDefault(0));
            if (src == null)
                return null;

            int nW = src.ImageWidth.GetValueOrDefault(0);
            int nH = src.ImageHeight.GetValueOrDefault(0);
            int nC = src.ImageCount.GetValueOrDefault(0);
            bool bIsReal = src.ImageEncoded.GetValueOrDefault(true);
            int nCount = src.ImageCount.GetValueOrDefault(0);

            SourceDescriptor srcd = new SourceDescriptor(src.ID, src.Name, nW, nH, nC, bIsReal, false, 0, null, nCount);
            srcd.TemporalDescriptor = LoadTemporalFromDb(src.ID);

            return srcd;
        }

        /// <summary>
        /// Load a dataset group with the specified ID from the database.
        /// </summary>
        /// <param name="nGrpId">Specifies the ID of the group to load.</param>
        /// <returns>The group descriptor is returned.</returns>
        public GroupDescriptor LoadGroupFromDb(int? nGrpId)
        {
            DatasetGroup grp = m_db.GetDatasetGroup(nGrpId.GetValueOrDefault(0));
            if (grp == null)
                return null;

            return new GroupDescriptor(grp.ID, grp.Name, null);
        }

        /// <summary>
        /// Load the temporal descriptor for the specified source ID from the database.
        /// </summary>
        /// <param name="nSrcID">Specifies the data source ID.</param>
        /// <returns>The TemporalDescriptor is returned.</returns>
        public TemporalDescriptor LoadTemporalFromDb(int nSrcID)
        {
            TemporalDescriptor td = new TemporalDescriptor();
            List<ValueStream> rgStrm = m_db.GetAllValueStreams(nSrcID);
            List<ValueItem> rgItem = m_db.GetAllValueItems(nSrcID);

            if (rgItem.Count == 0 || rgStrm.Count == 0)
                return null;


            foreach (ValueStream strm in rgStrm)
            {
                ValueStreamDescriptor vsd = new ValueStreamDescriptor(strm.ID,
                                                                      strm.Name,
                                                                      strm.Ordering.GetValueOrDefault(0),
                                                                      (STREAM_CLASS_TYPE)strm.ClassTypeID,
                                                                      (STREAM_VALUE_TYPE)strm.ValueTypeID,
                                                                      strm.StartTime,
                                                                      strm.EndTime,
                                                                      strm.SecondsPerStep,
                                                                      strm.TotalSteps.GetValueOrDefault(0));

                td.ValueStreamDescriptors.Add(vsd);
            }

            foreach (ValueItem vi in rgItem)
            {
                td.ValueItemDescriptors.Add(new ValueItemDescriptor(vi.ID, vi.Name, vi.StartTime, vi.EndTime, vi.Steps));
            }

            return td;            
        }

        /// <summary>
        /// Load the temporal descriptors for the specified source ID from the database.
        /// </summary>
        /// <param name="ds">Specifies the dataset descriptor where the temporal descriptors are loaded.</param>
        public void LoadTemporalFromDb(DatasetDescriptor ds)
        {
            if (ds == null)
                return;

            if (ds.TrainingSource != null)
                ds.TrainingSource.TemporalDescriptor = LoadTemporalFromDb(ds.TrainingSource.ID);

            if (ds.TestingSource != null)
                ds.TestingSource.TemporalDescriptor = LoadTemporalFromDb(ds.TestingSource.ID);
        }
    }
}
