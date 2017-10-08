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
    [DataContract]
    public enum IMGDB_IMAGE_SELECTION_METHOD
    {
        /// <summary>
        /// No selection method used, select sequentially by index.
        /// </summary>
        [EnumMember]
        NONE = 0x0000,
        /// <summary>
        /// Randomly select the images, ignore the input index.
        /// </summary>
        [EnumMember]
        RANDOM = 0x0001,
        /// <summary>
        /// Pair select the images where the first query returns a randomly selected image,
        /// and the next query returns the image just following the last queried image.
        /// </summary>
        [EnumMember]
        PAIR = 0x0002,
        /// <summary>
        /// Combines RANDOM + PAIR for marshalling.
        /// </summary>
        [EnumMember]
        RANDOM_AND_PAIR = 0x0003,
        /// <summary>
        /// Randomly select, but given higher priority to boosted images using the super-boost setting.
        /// </summary>
        [EnumMember]
        BOOST = 0x0004,
        /// <summary>
        /// Combines RANDOM + BOOST for marshalling.
        /// </summary>
        [EnumMember]
        RANDOM_AND_BOOST = 0x0005,
        /// <summary>
        /// Combines RANDOM + PAIR + BOOST for marshalling.
        /// </summary>
        [EnumMember]
        RANDOM_AND_PAIR_AND_BOOST = 0x0007,
        /// <summary>
        /// Specifically select based on the input index.
        /// </summary>
        [EnumMember]
        FIXEDINDEX = 0x0008,
        /// <summary>
        /// Clear the fixed index.
        /// </summary>
        [EnumMember]
        CLEARFIXEDINDEX = 0x0010
    }

    /// <summary>
    /// Defines the label selection method.
    /// </summary>
    [Serializable]
    [DataContract]
    public enum IMGDB_LABEL_SELECTION_METHOD
    {
        /// <summary>
        /// Don't use label selection and instead select from the general list of all images.
        /// </summary>
        [EnumMember]
        NONE = 0x0000,
        /// <summary>
        /// Randomly select the label set.
        /// </summary>
        [EnumMember]
        RANDOM = 0x0001,
        /// <summary>
        /// Randomly select the label set but give a higher priority to boosted label sets using their boost values.
        /// </summary>
        [EnumMember]
        BOOST = 0x0002
    }

    [ServiceContract]
    public interface IXImageDatabaseEvent /** @private */
    {
        [OperationContract(IsOneWay = false)]
        void OnResult(string strMsg, double dfProgress);

        [OperationContract(IsOneWay = false)]
        void OnError(ImageDatabaseErrorData err);
    }


    /// <summary>
    /// The IXImageDatabase interface defines the eneral interface to the in-memory image database.
    /// </summary>
    [ServiceContract(CallbackContract = typeof(IXImageDatabaseEvent), SessionMode = SessionMode.Required)]
    public interface IXImageDatabase
    {
        /// <summary>
        /// Set the database instance to use.
        /// </summary>
        /// <param name="strInstance">Specifies the instance name to use in '.\\name' format.</param>
        [OperationContract(IsOneWay = false)]
        void SetInstance(string strInstance);

        /// <summary>
        /// Initializes the image database.
        /// </summary>
        /// <param name="s">Specifies the caffe settings.</param>
        /// <param name="strDs">Specifies the data set to load.</param>
        /// <param name="strEvtCancel">Specifies the name of the CancelEvent used to cancel load operations.</param>
        /// <returns>Returns <i>true</i> on success, <i>false</i> otherwise.</returns>
        [OperationContract(IsOneWay = false)]
        bool InitializeWithDsName(SettingsCaffe s, string strDs, string strEvtCancel = null);

        /// <summary>
        /// Initializes the image database.
        /// </summary>
        /// <param name="s">Specifies the caffe settings.</param>
        /// <param name="ds">Specifies the data set to load.</param>
        /// <param name="strEvtCancel">Specifies the name of the CancelEvent used to cancel load operations.</param>
        /// <returns>Returns <i>true</i> on success, <i>false</i> otherwise.</returns>
        [OperationContract(IsOneWay = false)]
        bool InitializeWithDs(SettingsCaffe s, DatasetDescriptor ds, string strEvtCancel = null);

        /// <summary>
        /// Initializes the image database.
        /// </summary>
        /// <param name="s">Specifies the caffe settings.</param>
        /// <param name="nDataSetID">Specifies the database ID of the data set to load.</param>
        /// <param name="strEvtCancel">Specifies the name of the CancelEvent used to cancel load operations.</param>
        /// <param name="nPadW">Specifies the padding to add to each image width (default = 0).</param>
        /// <param name="nPadH">Specifies the padding to add to each image height (default = 0).</param>
        /// <returns>Returns <i>true</i> on success, <i>false</i> otherwise.</returns>
        [OperationContract(IsOneWay = false)]
        bool InitializeWithDsId(SettingsCaffe s, int nDataSetID, string strEvtCancel = null, int nPadW = 0, int nPadH = 0);

        /// <summary>
        /// Releases the image database, and if this is the last instance using the in-memory database, frees all memory used.
        /// </summary>
        /// <param name="nDsId">Optionally, specifies the dataset previously used.</param>
        [OperationContract(IsOneWay = false)]
        void CleanUp(int nDsId = 0);

        /// <summary>
        /// Updates the label boosts for the images based on the label boosts set for the given project.
        /// </summary>
        /// <param name="nProjectId">Specifies the project ID in the database.</param>
        /// <param name="nSrcId">Specifies the data source ID.</param>
        [OperationContract(IsOneWay = false)]
        void UpdateLabelBoosts(int nProjectId, int nSrcId);

        /// <summary>
        /// Returns the number of images in a given data source.
        /// </summary>
        /// <param name="nSrcId">Specifies the data source ID.</param>
        /// <returns>The number of images is returned.</returns>
        [OperationContract(IsOneWay = false)]
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
        [OperationContract(IsOneWay = false)]
        SimpleDatum QueryImage(int nSrcId, int nIdx, IMGDB_LABEL_SELECTION_METHOD? labelSelectionOverride = null, IMGDB_IMAGE_SELECTION_METHOD? imageSelectionOverride = null, int? nLabel = null);

        /// <summary>
        /// Get the image with a given Raw Image ID.
        /// </summary>
        /// <param name="nImageID">Specifies the Raw Image ID of the image to get.</param>
        /// <param name="rgSrcId">Specifies a list of Source ID's to search for the image.</param>
        /// <returns>The SimpleDatum of the image is returned.</returns>
        [OperationContract(IsOneWay = false)]
        SimpleDatum GetImage(int nImageID, params int[] rgSrcId);

        /// <summary>
        /// Returns a list of LabelDescriptor%s associated with the labels within a data source.
        /// </summary>
        /// <param name="nSrcId">Specifies the data source ID.</param>
        /// <returns>The list of LabelDescriptor%s is returned.</returns>
        [OperationContract(IsOneWay = false)]
        List<LabelDescriptor> GetLabels(int nSrcId);

        /// <summary>
        /// Returns the text name of a given label within a data source. 
        /// </summary>
        /// <param name="nSrcId">Specifies the data source ID.</param>
        /// <param name="nLabel">Specifies the label.</param>
        /// <returns>The laben name is returned as a string.</returns>
        [OperationContract(IsOneWay = false)]
        string GetLabelName(int nSrcId, int nLabel);

        /// <summary>
        /// Returns the DatasetDescriptor for a given data set ID.
        /// </summary>
        /// <param name="nDsId">Specifies the data set ID.</param>
        /// <returns>The dataset Descriptor is returned.</returns>
        [OperationContract(IsOneWay = false)]
        DatasetDescriptor GetDatasetById(int nDsId);

        /// <summary>
        /// Returns the DatasetDescriptor for a given data set name.
        /// </summary>
        /// <param name="strDs">Specifies the data set name.</param>
        /// <returns>The dataset Descriptor is returned.</returns>
        [OperationContract(IsOneWay = false)]
        DatasetDescriptor GetDatasetByName(string strDs);

        /// <summary>
        /// Returns a data set name given its ID.
        /// </summary>
        /// <param name="nDsId">Specifies the data set ID.</param>
        /// <returns>The data set name is returned.</returns>
        [OperationContract(IsOneWay = false)]
        string GetDatasetName(int nDsId);

        /// <summary>
        /// Returns a data set ID given its name.
        /// </summary>
        /// <param name="strDs">Specifies the data set name.</param>
        /// <returns>The data set ID is returned.</returns>
        [OperationContract(IsOneWay = false)]
        int GetDatasetID(string strDs);

        /// <summary>
        /// Returns the SourceDescriptor for a given data source ID.
        /// </summary>
        /// <param name="nSrcId">Specifies the data source ID.</param>
        /// <returns>The SourceDescriptor is returned.</returns>
        [OperationContract(IsOneWay = false)]
        SourceDescriptor GetSourceById(int nSrcId);

        /// <summary>
        /// Returns the SourceDescriptor for a given data source name.
        /// </summary>
        /// <param name="strSrc">Specifies the data source name.</param>
        /// <returns>The SourceDescriptor is returned.</returns>
        [OperationContract(IsOneWay = false)]
        SourceDescriptor GetSourceByName(string strSrc);

        /// <summary>
        /// Returns a data source name given its ID.
        /// </summary>
        /// <param name="nSrcId">Specifies the data source ID.</param>
        /// <returns>The data source name is returned.</returns>
        [OperationContract(IsOneWay = false)]
        string GetSourceName(int nSrcId);

        /// <summary>
        /// Returns a data source ID given its name.
        /// </summary>
        /// <param name="strSrc">Specifies the data source name.</param>
        /// <returns>The data source ID is returned.</returns>
        [OperationContract(IsOneWay = false)]
        int GetSourceID(string strSrc);

        /// <summary>
        /// Queries the image mean for a data source from the database on disk.
        /// </summary>
        /// <param name="nSrcId">Specifies the ID of the data source.</param>
        /// <returns>The image mean is returned as a SimpleDatum.</returns>
        [OperationContract(IsOneWay = false)]
        [FaultContract(typeof(ImageDatabaseErrorData))]
        SimpleDatum QueryImageMean(int nSrcId);

        /// <summary>
        /// Returns the image mean for a data source.
        /// </summary>
        /// <param name="nSrcId">Specifies the ID of the data source.</param>
        /// <returns>The image mean is returned as a SimpleDatum.</returns>
        [OperationContract(IsOneWay = false)]
        [FaultContract(typeof(ImageDatabaseErrorData))]
        SimpleDatum GetImageMean(int nSrcId);

        /// <summary>
        /// Returns the image mean for the Training data source of a given data set.
        /// </summary>
        /// <param name="nDatasetId">Specifies the data set to use.</param>
        /// <returns>The image mean is returned as a SimpleDatum.</returns>
        [OperationContract(IsOneWay = false)]
        SimpleDatum QueryImageMeanFromDataset(int nDatasetId);

        /// <summary>
        /// Sets the label mapping to the database for a given data source.
        /// </summary>
        /// <param name="nSrcId">Specifies the ID of the data source.</param>
        /// <param name="map">Specifies the label mapping to set.</param>
        [OperationContract(IsOneWay = false)]
        void SetLabelMapping(int nSrcId, LabelMapping map);

        /// <summary>
        /// Updates the label mapping in the database for a given data source.
        /// </summary>
        /// <param name="nSrcId">Specifies the ID of the data source.</param>
        /// <param name="nNewLabel">Specifies a new label.</param>
        /// <param name="rgOriginalLabels">Specifies the original lables that are mapped to the new label.</param>
        [OperationContract(IsOneWay = false)]
        void UpdateLabelMapping(int nSrcId, int nNewLabel, List<int> rgOriginalLabels);

        /// <summary>
        /// Resets all labels within a data source, used by a project, to their original labels.
        /// </summary>
        /// <param name="nProjectId">Specifies the ID of the project.</param>
        /// <param name="nSrcId">Specifies the ID of the data source.</param>
        [OperationContract(IsOneWay = false)]
        void ResetLabels(int nProjectId, int nSrcId);

        /// <summary>
        /// Delete all label boosts for a given data source associated with a given project.
        /// </summary>
        /// <param name="nProjectId">Specifies the project ID.</param>
        /// <param name="nSrcId">Specifies the ID of the data source.</param>
        [OperationContract(IsOneWay = false)]
        void DeleteLabelBoosts(int nProjectId, int nSrcId);

        /// <summary>
        /// Updates the number of images of each label within a data source.
        /// </summary>
        /// <param name="nProjectId">Specifies a project ID.</param>
        /// <param name="nSrcId">Specifies the ID of the data source.</param>
        [OperationContract(IsOneWay = false)]
        void UpdateLabelCounts(int nProjectId, int nSrcId);

        /// <summary>
        /// Add a label boost for a data source associated with a given project.
        /// </summary>
        /// <param name="nProjectId">Specifies the project ID.</param>
        /// <param name="nSrcId">Specifies the ID of the data source.</param>
        /// <param name="nLabel">Specifies the label.</param>
        /// <param name="dfBoost">Specifies the new boost for the label.</param>
        [OperationContract(IsOneWay = false)]
        void AddLabelBoost(int nProjectId, int nSrcId, int nLabel, double dfBoost);

        /// <summary>
        /// Returns a label lookup of counts for a given data source.
        /// </summary>
        /// <param name="nSrcId">Specifies the ID of the data source.</param>
        /// <returns>A dictionary containing label,count pairs is returned.</returns>
        [OperationContract(IsOneWay = false)]
        [FaultContract(typeof(ImageDatabaseErrorData))]
        Dictionary<int, int> LoadLabelCounts(int nSrcId);

        /// <summary>
        /// Returns a string with all label counts for a data source.
        /// </summary>
        /// <param name="nSrcId">Specifies the ID of the data source.</param>
        /// <returns>A string containing all label counts is returned.</returns>
        [OperationContract(IsOneWay = false)]
        string GetLabelCountsAsTextFromSourceId(int nSrcId);

        /// <summary>
        /// Returns a string with all label counts for a data source.
        /// </summary>
        /// <param name="strSource">Specifies the name of the data source.</param>
        /// <returns>A string containing all label counts is returned.</returns>
        [OperationContract(IsOneWay = false)]
        string GetLabelCountsAsTextFromSourceName(string strSource);

        /// <summary>
        /// Returns the label boosts as a text string for all boosted labels within a data source associated with a given project. 
        /// </summary>
        /// <param name="nProjectId">Specifies the project ID.</param>
        /// <param name="nSrcId">Specifies the ID of the data source.</param>
        /// <returns>The label boosts are returned as a text string.</returns>
        [OperationContract(IsOneWay = false)]
        string GetLabelBoostsAsTextFromProject(int nProjectId, int nSrcId);

        /// <summary>
        /// Reload a data set.
        /// </summary>
        /// <param name="nDsId">Specifies the ID of the data set.</param>
        /// <returns>If the data set is found, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        [OperationContract(IsOneWay = false)]
        bool ReloadDataset(int nDsId);

        /// <summary>
        /// Reloads the images of a data source.
        /// </summary>
        /// <param name="nSrcId">Specifies the ID of the data source.</param>
        /// <returns>If the data source is found, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        [OperationContract(IsOneWay = false)]
        bool ReloadImageSet(int nSrcId);

        /// <summary>
        /// Searches for the image index of an image within a data source matching a DateTime/description pattern.
        /// </summary>
        /// <remarks>
        /// Optionally, images may have a time-stamp and/or description associated with each image.  In such cases
        /// searching by the time-stamp + description can be useful in some instances.
        /// </remarks>
        /// <param name="nSrcId">Specifies the data source ID of the data source to be searched.</param>
        /// <param name="dt">Specifies the time-stamp to search for.</param>
        /// <param name="strDescription">Specifies the description to search for.</param>
        /// <returns>If found the zero-based index of the image is returned, otherwise -1 is returned.</returns>
        [OperationContract(IsOneWay = false)]
        int FindImageIndex(int nSrcId, DateTime dt, string strDescription);

        /// <summary>
        /// When using a <i>Load Limit</i> that is greater than 0, this function loads the next set of images.
        /// </summary>
        /// <param name="evtCancel">Specifies the name of the Cancel Event to abort loading the images.</param>
        /// <returns>Returns <i>true</i> on success, <i>false</i> otherwise.</returns>
        [OperationContract(IsOneWay = false)]
        bool LoadNextSet(string strEvtCancel);

        /// <summary>
        /// <summary>
        /// Returns the label and image selection method used.
        /// </summary>
        /// <returns>A tuple containing the Label and Image selection method.</returns>
        [OperationContract(IsOneWay = false)]
        Tuple<IMGDB_LABEL_SELECTION_METHOD, IMGDB_IMAGE_SELECTION_METHOD> GetSelectionMethod();

        /// <summary>
        /// Sets the label and image selection methods.
        /// </summary>
        /// <param name="lbl">Specifies the label selection method or <i>null</i> to ignore.</param>
        /// <param name="img">Specifies the image selection method or <i>null</i> to ignore.</param>
        [OperationContract(IsOneWay = false)]
        void SetSelectionMethod(IMGDB_LABEL_SELECTION_METHOD? lbl, IMGDB_IMAGE_SELECTION_METHOD? img);

        /// <summary>
        /// The UnloadDataset function unloads a given dataset from memory.
        /// </summary>
        /// <param name="strDataset">Specifies the name of the dataset to unload.</param>
        /// <returns>If the dataset is found and removed, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        [OperationContract(IsOneWay = false)]
        bool UnloadDatasetByName(string strDataset);

        /// <summary>
        /// The UnloadDataset function unloads a given dataset from memory.
        /// </summary>
        /// <param name="nDatasetID">Specifies the ID of the dataset to unload.</param>
        /// <remarks>Specifiying a dataset ID of -1 directs the UnloadDatasetById to unload ALL datasets loaded.</remarks>
        /// <returns>If the dataset is found and removed, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        [OperationContract(IsOneWay = false)]
        bool UnloadDatasetById(int nDatasetID);

        /// <summary>
        /// Returns the percentage that a dataset is loaded into memory.
        /// </summary>
        /// <param name="strDataset">Specifies the name of the dataset.</param>
        /// <param name="dfTraining">Specifies the percent of training images that are loaded.</param>
        /// <param name="dfTesting">Specifies the percent of testing images that are loaded.</param>
        /// <returns>The current image load percent for the dataset is returned..</returns>
        [OperationContract(IsOneWay = false)]
        double GetDatasetLoadedPercentByName(string strDataset, out double dfTraining, out double dfTesting);

        /// <summary>
        /// Returns the percentage that a dataset is loaded into memory.
        /// </summary>
        /// <param name="nDatasetID">Specifies the ID of the dataset.</param>
        /// <param name="dfTraining">Specifies the percent of training images that are loaded.</param>
        /// <param name="dfTesting">Specifies the percent of testing images that are loaded.</param>
        /// <returns>The current image load percent for the dataset is returned..</returns>
        [OperationContract(IsOneWay = false)]
        double GetDatasetLoadedPercentById(int nDatasetID, out double dfTraining, out double dfTesting);
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
