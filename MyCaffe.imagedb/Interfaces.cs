using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ServiceModel;
using System.Runtime.Serialization;
using System.Threading;
using MyCaffe.basecode;
using MyCaffe.basecode.descriptors;

namespace MyCaffe.imagedb
{
    /// <summary>
    /// Defines the image selection method.
    /// </summary>
    [Serializable]
    public enum IMGDB_IMAGE_SELECTION_METHOD
    {
        /// <summary>
        /// No selection method used, select sequentially by index.
        /// </summary>
        NONE = 0x0000,
        /// <summary>
        /// Randomly select the images, ignore the input index.
        /// </summary>
        RANDOM = 0x0001,
        /// <summary>
        /// Pair select the images where the first query returns a randomly selected image,
        /// and the next query returns the image just following the last queried image.
        /// </summary>
        PAIR = 0x0002,
        /// <summary>
        /// Randomly select, but given higher priority to boosted images using the super-boost setting.
        /// </summary>
        BOOST = 0x0004,
        /// <summary>
        /// Specifically select based on the input index.
        /// </summary>
        FIXEDINDEX = 0x0008,
        /// <summary>
        /// Clear the fixed index.
        /// </summary>
        CLEARFIXEDINDEX = 0x0010
    }

    /// <summary>
    /// Defines the label selection method.
    /// </summary>
    [Serializable]
    public enum IMGDB_LABEL_SELECTION_METHOD
    {
        /// <summary>
        /// Don't use label selection and instead select from the general list of all images.
        /// </summary>
        NONE = 0x0000,
        /// <summary>
        /// Randomly select the label set.
        /// </summary>
        RANDOM = 0x0001,
        /// <summary>
        /// Randomly select the label set but give a higher priority to boosted label sets using their boost values.
        /// </summary>
        BOOST = 0x0002
    }

    /// <summary>
    /// The IXImageDatabase interface defines the eneral interface to the in-memory image database.
    /// </summary>
    [ServiceContract]
    public interface IXImageDatabase
    {
        /// <summary>
        /// Initializes the image database.
        /// </summary>
        /// <param name="s">Specifies the caffe settings.</param>
        /// <param name="strDs">Specifies the data set to load.</param>
        /// <param name="evtCancel">Specifies the CancelEvent used to cancel load operations.</param>
        /// <returns>Returns <i>true</i> on success, <i>false</i> otherwise.</returns>
        [OperationContract]
        [FaultContract(typeof(ImageDatabaseErrorData))]
        bool Initialize(SettingsCaffe s, string strDs, CancelEvent evtCancel = null);

        /// <summary>
        /// Initializes the image database.
        /// </summary>
        /// <param name="s">Specifies the caffe settings.</param>
        /// <param name="ds">Specifies the data set to load.</param>
        /// <param name="evtCancel">Specifies the CancelEvent used to cancel load operations.</param>
        /// <returns>Returns <i>true</i> on success, <i>false</i> otherwise.</returns>
        [OperationContract]
        [FaultContract(typeof(ImageDatabaseErrorData))]
        bool Initialize(SettingsCaffe s, DatasetDescriptor ds, CancelEvent evtCancel = null);

        /// <summary>
        /// Initializes the image database.
        /// </summary>
        /// <param name="s">Specifies the caffe settings.</param>
        /// <param name="nDataSetID">Specifies the database ID of the data set to load.</param>
        /// <param name="evtCancel">Specifies the CancelEvent used to cancel load operations.</param>
        /// <param name="nPadW">Specifies the padding to add to each image width (default = 0).</param>
        /// <param name="nPadH">Specifies the padding to add to each image height (default = 0).</param>
        /// <returns>Returns <i>true</i> on success, <i>false</i> otherwise.</returns>
        [OperationContract]
        [FaultContract(typeof(ImageDatabaseErrorData))]
        bool Initialize(SettingsCaffe s, int nDataSetID, CancelEvent evtCancel = null, int nPadW = 0, int nPadH = 0);

        /// <summary>
        /// Releases the image database, and if this is the last instance using the in-memory database, frees all memory used.
        /// </summary>
        [OperationContract]
        [FaultContract(typeof(ImageDatabaseErrorData))]
        void CleanUp();

        /// <summary>
        /// Updates the label boosts for the images based on the label boosts set for the given project.
        /// </summary>
        /// <param name="nProjectId">Specifies the project ID in the database.</param>
        /// <param name="nSrcId">Specifies the data source ID.</param>
        [OperationContract]
        [FaultContract(typeof(ImageDatabaseErrorData))]
        void UpdateLabelBoosts(int nProjectId, int nSrcId);

        /// <summary>
        /// Returns the number of images in a given data source.
        /// </summary>
        /// <param name="nSrcId">Specifies the data source ID.</param>
        /// <returns>The number of images is returned.</returns>
        [OperationContract]
        [FaultContract(typeof(ImageDatabaseErrorData))]
        int ImageCount(int nSrcId);

        /// <summary>
        /// Query an image in a given data source.
        /// </summary>
        /// <param name="nSrcId">Specifies the databse ID of the data source.</param>
        /// <param name="nIdx">Specifies the image index to query.  Note, the index is only used in non-random image queries.</param>
        /// <param name="labelSelectionOverride">Optionally, specifies the label selection method override.  The default = null, which directs the method to use the label selection method specified during Initialization.</param>
        /// <param name="imageSelectionOverride">Optionally, specifies the image selection method override.  The default = null, which directs the method to use the image selection method specified during Initialization.</param>
        /// <param name="nLabel">Optionally, specifies a label set to use for the image selection.  When specified only images of this label are returned using the image selection method.</param>
        /// <returns>The image SimpleDatum is returned.</returns>
        [OperationContract]
        [FaultContract(typeof(ImageDatabaseErrorData))]
        SimpleDatum QueryImage(int nSrcId, int nIdx, IMGDB_LABEL_SELECTION_METHOD? labelSelectionOverride = null, IMGDB_IMAGE_SELECTION_METHOD? imageSelectionOverride = null, int? nLabel = null);

        /// <summary>
        /// Returns a list of LabelDescriptor%s associated with the labels within a data source.
        /// </summary>
        /// <param name="nSrcId">Specifies the data source ID.</param>
        /// <returns>The list of LabelDescriptor%s is returned.</returns>
        [OperationContract]
        [FaultContract(typeof(ImageDatabaseErrorData))]
        List<LabelDescriptor> GetLabels(int nSrcId);

        /// <summary>
        /// Returns the text name of a given label within a data source. 
        /// </summary>
        /// <param name="nSrcId">Specifies the data source ID.</param>
        /// <param name="nLabel">Specifies the label.</param>
        /// <returns>The laben name is returned as a string.</returns>
        [OperationContract]
        [FaultContract(typeof(ImageDatabaseErrorData))]
        string GetLabelName(int nSrcId, int nLabel);

        /// <summary>
        /// Returns the DatasetDescriptor for a given data set ID.
        /// </summary>
        /// <param name="nDsId">Specifies the data set ID.</param>
        /// <returns>The dataset Descriptor is returned.</returns>
        [OperationContract]
        [FaultContract(typeof(ImageDatabaseErrorData))]
        DatasetDescriptor GetDataset(int nDsId);

        /// <summary>
        /// Returns the DatasetDescriptor for a given data set name.
        /// </summary>
        /// <param name="strDs">Specifies the data set name.</param>
        /// <returns>The dataset Descriptor is returned.</returns>
        [OperationContract]
        [FaultContract(typeof(ImageDatabaseErrorData))]
        DatasetDescriptor GetDataset(string strDs);

        /// <summary>
        /// Returns a data set name given its ID.
        /// </summary>
        /// <param name="nDsId">Specifies the data set ID.</param>
        /// <returns>The data set name is returned.</returns>
        [OperationContract]
        [FaultContract(typeof(ImageDatabaseErrorData))]
        string GetDatasetName(int nDsId);

        /// <summary>
        /// Returns a data set ID given its name.
        /// </summary>
        /// <param name="strDs">Specifies the data set name.</param>
        /// <returns>The data set ID is returned.</returns>
        [OperationContract]
        [FaultContract(typeof(ImageDatabaseErrorData))]
        int GetDatasetID(string strDs);

        /// <summary>
        /// Returns the SourceDescriptor for a given data source ID.
        /// </summary>
        /// <param name="nSrcId">Specifies the data source ID.</param>
        /// <returns>The SourceDescriptor is returned.</returns>
        [OperationContract]
        [FaultContract(typeof(ImageDatabaseErrorData))]
        SourceDescriptor GetSource(int nSrcId);

        /// <summary>
        /// Returns the SourceDescriptor for a given data source name.
        /// </summary>
        /// <param name="strSrc">Specifies the data source name.</param>
        /// <returns>The SourceDescriptor is returned.</returns>
        [OperationContract]
        [FaultContract(typeof(ImageDatabaseErrorData))]
        SourceDescriptor GetSource(string strSrc);

        /// <summary>
        /// Returns a data source name given its ID.
        /// </summary>
        /// <param name="nSrcId">Specifies the data source ID.</param>
        /// <returns>The data source name is returned.</returns>
        [OperationContract]
        [FaultContract(typeof(ImageDatabaseErrorData))]
        string GetSourceName(int nSrcId);

        /// <summary>
        /// Returns a data source ID given its name.
        /// </summary>
        /// <param name="strSrc">Specifies the data source name.</param>
        /// <returns>The data source ID is returned.</returns>
        [OperationContract]
        [FaultContract(typeof(ImageDatabaseErrorData))]
        int GetSourceID(string strSrc);

        /// <summary>
        /// Queries the image mean for a data source from the database on disk.
        /// </summary>
        /// <param name="nSrcId">Specifies the ID of the data source.</param>
        /// <returns>The image mean is returned as a SimpleDatum.</returns>
        [OperationContract]
        [FaultContract(typeof(ImageDatabaseErrorData))]
        SimpleDatum QueryImageMean(int nSrcId);

        /// <summary>
        /// Returns the image mean for a data source.
        /// </summary>
        /// <param name="nSrcId">Specifies the ID of the data source.</param>
        /// <returns>The image mean is returned as a SimpleDatum.</returns>
        [OperationContract]
        [FaultContract(typeof(ImageDatabaseErrorData))]
        SimpleDatum GetImageMean(int nSrcId);

        /// <summary>
        /// Returns the image mean for the Training data source of a given data set.
        /// </summary>
        /// <param name="nDatasetId">Specifies the data set to use.</param>
        /// <returns>The image mean is returned as a SimpleDatum.</returns>
        [OperationContract]
        [FaultContract(typeof(ImageDatabaseErrorData))]
        SimpleDatum QueryImageMeanFromDataset(int nDatasetId);

        /// <summary>
        /// Sets the label mapping to the database for a given data source.
        /// </summary>
        /// <param name="nSrcId">Specifies the ID of the data source.</param>
        /// <param name="map">Specifies the label mapping to set.</param>
        [OperationContract]
        [FaultContract(typeof(ImageDatabaseErrorData))]
        void SetLabelMapping(int nSrcId, LabelMapping map);

        /// <summary>
        /// Updates the label mapping in the database for a given data source.
        /// </summary>
        /// <param name="nSrcId">Specifies the ID of the data source.</param>
        /// <param name="nNewLabel">Specifies a new label.</param>
        /// <param name="rgOriginalLabels">Specifies the original lables that are mapped to the new label.</param>
        [OperationContract]
        [FaultContract(typeof(ImageDatabaseErrorData))]
        void UpdateLabelMapping(int nSrcId, int nNewLabel, List<int> rgOriginalLabels);

        /// <summary>
        /// Resets all labels within a data source, used by a project, to their original labels.
        /// </summary>
        /// <param name="nProjectId">Specifies the ID of the project.</param>
        /// <param name="nSrcId">Specifies the ID of the data source.</param>
        [OperationContract]
        [FaultContract(typeof(ImageDatabaseErrorData))]
        void ResetLabels(int nProjectId, int nSrcId);

        /// <summary>
        /// Delete all label boosts for a given data source associated with a given project.
        /// </summary>
        /// <param name="nProjectId">Specifies the project ID.</param>
        /// <param name="nSrcId">Specifies the ID of the data source.</param>
        [OperationContract]
        [FaultContract(typeof(ImageDatabaseErrorData))]
        void DeleteLabelBoosts(int nProjectId, int nSrcId);

        /// <summary>
        /// Updates the number of images of each label within a data source.
        /// </summary>
        /// <param name="nProjectId">Specifies a project ID.</param>
        /// <param name="nSrcId">Specifies the ID of the data source.</param>
        [OperationContract]
        [FaultContract(typeof(ImageDatabaseErrorData))]
        void UpdateLabelCounts(int nProjectId, int nSrcId);

        /// <summary>
        /// Add a label boost for a data source associated with a given project.
        /// </summary>
        /// <param name="nProjectId">Specifies the project ID.</param>
        /// <param name="nSrcId">Specifies the ID of the data source.</param>
        /// <param name="nLabel">Specifies the label.</param>
        /// <param name="dfBoost">Specifies the new boost for the label.</param>
        [OperationContract]
        [FaultContract(typeof(ImageDatabaseErrorData))]
        void AddLabelBoost(int nProjectId, int nSrcId, int nLabel, double dfBoost);

        /// <summary>
        /// Returns a label lookup of counts for a given data source.
        /// </summary>
        /// <param name="nSrcId">Specifies the ID of the data source.</param>
        /// <returns>A dictionary containing label,count pairs is returned.</returns>
        [OperationContract]
        [FaultContract(typeof(ImageDatabaseErrorData))]
        Dictionary<int, int> LoadLabelCounts(int nSrcId);

        /// <summary>
        /// Returns a string with all label counts for a data source.
        /// </summary>
        /// <param name="nSrcId">Specifies the ID of the data source.</param>
        /// <returns>A string containing all label counts is returned.</returns>
        [OperationContract]
        [FaultContract(typeof(ImageDatabaseErrorData))]
        string GetLabelCountsAsText(int nSrcId);

        /// <summary>
        /// Returns a string with all label counts for a data source.
        /// </summary>
        /// <param name="strSource">Specifies the name of the data source.</param>
        /// <returns>A string containing all label counts is returned.</returns>
        [OperationContract]
        [FaultContract(typeof(ImageDatabaseErrorData))]
        string GetLabelCountsAsText(string strSource);

        /// <summary>
        /// Returns the label boosts as a text string for all boosted labels within a data source associated with a given project. 
        /// </summary>
        /// <param name="nProjectId">Specifies the project ID.</param>
        /// <param name="nSrcId">Specifies the ID of the data source.</param>
        /// <returns>The label boosts are returned as a text string.</returns>
        [OperationContract]
        [FaultContract(typeof(ImageDatabaseErrorData))]
        string GetLabelBoostsAsText(int nProjectId, int nSrcId);

        /// <summary>
        /// Reload a data set.
        /// </summary>
        /// <param name="nDsId">Specifies the ID of the data set.</param>
        /// <returns>If the data set is found, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        [OperationContract]
        [FaultContract(typeof(ImageDatabaseErrorData))]
        bool ReloadDataset(int nDsId);

        /// <summary>
        /// Reloads the images of a data source.
        /// </summary>
        /// <param name="nSrcId">Specifies the ID of the data source.</param>
        /// <returns>If the data source is found, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        [OperationContract]
        [FaultContract(typeof(ImageDatabaseErrorData))]
        bool ReloadImageSet(int nSrcId);

        /// <summary>
        /// Searches fro the image index of an image within a data source matching a DateTime/description pattern.
        /// </summary>
        /// <remarks>
        /// Optionally, images may have a time-stamp and/or description associated with each image.  In such cases
        /// searching by the time-stamp + description can be useful in some instances.
        /// </remarks>
        /// <param name="nSrcId">Specifies the data source ID of the data source to be searched.</param>
        /// <param name="dt">Specifies the time-stamp to search for.</param>
        /// <param name="strDescription">Specifies the description to search for.</param>
        /// <returns>If found the zero-based index of the image is returned, otherwise -1 is returned.</returns>
        [OperationContract]
        [FaultContract(typeof(ImageDatabaseErrorData))]
        int FindImageIndex(int nSrcId, DateTime dt, string strDescription);

        /// <summary>
        /// When using a <i>Load Limit</i> that is greater than 0, this function loads the next set of images.
        /// </summary>
        /// <param name="evtCancel">Specifies the cance event to abort loading the images.</param>
        /// <returns>Returns <i>true</i> on success, <i>false</i> otherwise.</returns>
        [OperationContract]
        [FaultContract(typeof(ImageDatabaseErrorData))]
        bool LoadNextSet(CancelEvent evtCancel);

        /// <summary>
        /// Returns the image selection method.
        /// </summary>
        IMGDB_IMAGE_SELECTION_METHOD ImageSelectionMethod { get; set; }
        /// <summary>
        /// Returns the label selection method.
        /// </summary>
        IMGDB_LABEL_SELECTION_METHOD LabelSelectionMethod { get; set; }
    }

    [DataContract]
    public class ImageDatabaseErrorData /** @private */
    {
        [DataMember]
        public bool Result { get; set; }
        [DataMember]
        public string ErrorMessage { get; set; }
        [DataMember]
        public string ErrorDetails { get; set; }
    }
}
