﻿using MyCaffe.basecode.descriptors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;
using System.ServiceModel;
using System.Text;

namespace MyCaffe.basecode
{
    /// <summary>
    /// Specifies the special tokens.
    /// </summary>
    public enum SPECIAL_TOKENS
    {
        /// <summary>
        /// The PAD token is used to pad the input to the same length.
        /// </summary>
        PAD = 0,
        /// <summary>
        /// The BOS token indicates the begin of sequence.
        ///</summary>
        BOS = 1,
        /// <summary>
        /// The EOS token indicates the end of sequence.
        /// </summary>
        EOS = 2
    }

    /// <summary>
    /// Defines the category of training.
    /// </summary>
    public enum TRAINING_CATEGORY
    {
        /// <summary>
        /// No training category specified.
        /// </summary>
        NONE,
        /// <summary>
        /// Defines a purely custom training method.
        /// </summary>
        CUSTOM,
        /// <summary>
        /// Defines the reinforcement training method such as PG.
        /// </summary>
        REINFORCEMENT,
        /// <summary>
        /// Defines the recurrent training method.
        /// </summary>
        RECURRENT,
        /// <summary>
        /// Defines the reinforcement training method such as PG that also uses a recurrent model such as LSTM.
        /// </summary>
        DUAL
    }

    /// <summary>
    /// Defines the type of score normalization to perform when using the score as the label.
    /// </summary>
    public enum SCORE_AS_LABEL_NORMALIZATION
    {
        /// <summary>
        /// Specifies to not do any normalization.
        /// </summary>
        NONE = 0,
        /// <summary>
        /// Specifies to run z-score normalization on all values usng the 'Mean' and 'StdDev' values from the mean image.
        /// </summary>
        Z_SCORE = 1,
        /// <summary>
        /// Specifies to run pos/neg z-score normalization on all values using the 'PosMean' and 'PosStdDev' for all positive values, and 'NegMean' and 'NegStdDev' for all negative values..
        /// </summary>
        Z_SCORE_POSNEG = 2,
        /// <summary>
        /// Specifies to run a positive shift normalization on all values by adding 1 and multiplying by 100 to fit in a range of [0,200]
        /// </summary>
        POS_SHIFT = 3
    }

    /// <summary>
    /// Defines the Phase under which to run a Net.
    /// </summary>
    public enum Phase
    {
        /// <summary>
        /// No phase defined.
        /// </summary>
        NONE = 0,
        /// <summary>
        /// Run a training phase.
        /// </summary>
        TRAIN = 1,
        /// <summary>
        /// Run a testing phase.
        /// </summary>
        TEST = 2,
        /// <summary>
        /// Run on an image given to the Net.
        /// </summary>
        RUN = 3,
        /// <summary>
        /// Applies to all phases.
        /// </summary>
        ALL = 5
    }

    /// <summary>
    /// Specifies the stage underwhich to run a custom trainer.
    /// </summary>
    public enum Stage
    {
        /// <summary>
        /// No stage defined.
        /// </summary>
        NONE = 0,
        /// <summary>
        /// Run the trainer in RNN mode.
        /// </summary>
        RNN = 1,
        /// <summary>
        /// Run the trainer in RL mode.
        /// </summary>
        RL = 2,
        /// <summary>
        /// Applies to all stages.
        /// </summary>
        ALL = 3,
        /// <summary>
        /// Applies to stages specified within the solver prototxt.
        /// </summary>
        CUSTOM = 4
    }


    /// <summary>
    /// Defines the gym type (if any).
    /// </summary>
    public enum GYM_TYPE
    {
        /// <summary>
        /// Specifies that the type is not a gym.
        /// </summary>
        NONE,
        /// <summary>
        /// Specifies a dynamic gym type that dynamically produces its data.
        /// </summary>
        DYNAMIC,
        /// <summary>
        /// Specifies a data gym that collects data from a data source, such as a database.
        /// </summary>
        DATA
    }

    /// <summary>
    /// Defines the gym data type.
    /// </summary>
    public enum DATA_TYPE
    {
        /// <summary>
        /// Specifies to use the default data type of the gym used.
        /// </summary>
        DEFAULT,
        /// <summary>
        /// Specifies to use the raw state values of the gym (if supported).
        /// </summary>
        VALUES,
        /// <summary>
        /// Specifies to use a SimpleDatum blob of data of the gym (if supported).
        /// </summary>
        BLOB
    }

    /// <summary>
    /// Defines how to laod the items into the in-memory database.
    /// </summary>
    public enum DB_LOAD_METHOD
    {
        /// <summary>
        /// Load the items as they are queried - this option cahces items into memory as needed, training speeds are slower up until all items are loaded into memory.
        /// </summary>
        LOAD_ON_DEMAND,
        /// <summary>
        /// Load all of the items into memory - this option provides the highest training speeds, but can use a lot of memory and takes time to load.
        /// </summary>
        LOAD_ALL,
        /// <summary>
        /// Load the items from an external source such as a Windows Service - this option provides the best balance of speed and short load times for once loaded all applications share the in-memory data.
        /// </summary>
        LOAD_EXTERNAL,
        /// <summary>
        /// Load the image as they are queried AND start the background loading at the same time.
        /// </summary>
        LOAD_ON_DEMAND_BACKGROUND,
        /// <summary>
        /// Load the items on demand, but do not cache the items - this option loads items from disk as needed and does not cache them thus saving memory use.
        /// </summary>
        LOAD_ON_DEMAND_NOCACHE
    }

    /// <summary>
    /// Defines the snapshot weight update method.
    /// </summary>
    public enum SNAPSHOT_WEIGHT_UPDATE_METHOD
    {
        /// <summary>
        /// Disables all snapshots.
        /// </summary>
        DISABLED = -1,
        /// <summary>
        /// Update the snapshot weights when the accuracy increases.
        /// </summary>
        FAVOR_ACCURACY,
        /// <summary>
        /// Update the snapshot weights when the error decreases.
        /// </summary>
        FAVOR_ERROR,
        /// <summary>
        /// Update the snapshot weights when the accuracy increases or the error decreases.
        /// </summary>
        FAVOR_BOTH,
        /// <summary>
        /// Update the snapshow on each test cycle regardless of the accuracy or error change. 
        /// </summary>
        ALWAYS_ON_TEST
    }

    /// <summary>
    /// Defines the snapshot load method.
    /// </summary>
    public enum SNAPSHOT_LOAD_METHOD
    {
        /// <summary>
        /// Load the last solver state snapshotted.
        /// </summary>
        LAST_STATE = 0,
        /// <summary>
        /// Load the weights with the best accuracy (which may not be the last).
        /// </summary>
        WEIGHTS_BEST_ACCURACY = 1,
        /// <summary>
        /// Load the weights with the best error (which may not be the last).
        /// </summary>
        WEIGHTS_BEST_ERROR = 2,
        /// <summary>
        /// Load the state with the best accuracy (which may not be the last).
        /// </summary>
        STATE_BEST_ACCURACY = 4,
        /// <summary>
        /// Load the state with the best error (which may not be the last).
        /// </summary>
        STATE_BEST_ERROR = 8
    }

    /// <summary>
    /// Defines the different way of computing average precision.
    /// </summary>
    /// <remarks>
    /// @see [Tag: Average Precision](https://sanchom.wordpress.com/tag/average-precision) by Sanchom
    /// </remarks>
    public enum ApVersion
    {
        /// <summary>
        /// Specifies the 11-point interpolated average precision, used in VOC2007.
        /// </summary>
        ELEVENPOINT,
        /// <summary>
        /// Specifies the maximally interpolated AP, used in VOC2012/ILSVRC.
        /// </summary>
        MAXINTEGRAL,
        /// <summary>
        /// Specifies the natural integral of the precision-recall curve.
        /// </summary>
        INTEGRAL
    }

    /// <summary>
    /// Defines the ITest interface used by the Test module to return its known failures.
    /// </summary>
    public interface ITestKnownFailures
    {
        /// <summary>
        /// Get the known failures of the test module.
        /// </summary>
        List<Tuple<string, string, string>> KnownFailures { get; }
        /// <summary>
        /// Get the priority of a class::method pair.
        /// </summary>
        /// <param name="strClass">Specifies the class.</param>
        /// <param name="strMethod">Specifies the method.</param>
        /// <returns>The priority is returned with 0 being the top priority.</returns>
        int GetPriority(string strClass, string strMethod);
    }


    //-------------------------------------------------------------------------------------------------
    //  IXImageDatabase Interfaces
    //-------------------------------------------------------------------------------------------------

    /// <summary>
    /// Defines the item (e.g., image or temporal item) selection method.
    /// </summary>
    [Serializable]
    [DataContract]
    public enum DB_ITEM_SELECTION_METHOD
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
    /// Specifies the the items are to be selected from the database when using 'NONE' as the selection for both the item and value.
    /// </summary>
    [Serializable]
    [DataContract]
    public enum DB_INDEX_ORDER
    {
        /// <summary>
        /// Specifies to use a row-major ordering where indexes along each row are indexed until the end of the row is reached, then the next row is indexed.
        /// </summary>
        [EnumMember]
        ROW_MAJOR,
        /// <summary>
        /// Specifies to use a column-major ordering where indexes along each column are indexed until the end of the column is reached, then the next column is indexed.
        /// </summary>
        [EnumMember]
        COL_MAJOR,
        /// <summary>
        /// Specifies to use the default ordering (currently ROW_MAJOR).
        /// </summary>
        [EnumMember]
        DEFAULT = ROW_MAJOR
    }

    /// <summary>
    /// Defines the label selection method.
    /// </summary>
    [Serializable]
    [DataContract]
    public enum DB_LABEL_SELECTION_METHOD
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

    /// <summary>
    /// Defines the sorting method.
    /// </summary>
    /// <remarks>BYDESC and BYDATE can be compbined which causes the images to be sorted by description first and then by time.  BYID cannot be combined with other sorting methods.</remarks>
    [Serializable]
    [DataContract]
    public enum IMGDB_SORT
    {
        /// <summary>
        /// No sorting performed.
        /// </summary>
        [EnumMember]
        NONE = 0x0000,
        /// <summary>
        /// Sort by description first.
        /// </summary>
        [EnumMember]
        BYDESC = 0x0001,
        /// <summary>
        /// Sort by time.
        /// </summary>
        [EnumMember]
        BYTIME = 0x0002,
        /// <summary>
        /// Sort by image ID.
        /// </summary>
        [EnumMember]
        BYID = 0x0004,
        /// <summary>
        /// Sort by image ID in decending order.
        /// </summary>
        [EnumMember]
        BYID_DESC = 0x0008,
        /// <summary>
        /// Sort by image Index.
        /// </summary>
        [EnumMember]
        BYIDX = 0x0010
    }

    /// <summary>
    /// Defines the image database version to use.
    /// </summary>
    [Serializable]
    [DataContract]
    public enum DB_VERSION
    {
        /// <summary>
        /// Specifies to not use an in-memory database.
        /// </summary>
        [EnumMember]
        NONE,
        /// <summary>
        /// Specifies to use the original image database.
        /// </summary>
        [EnumMember]
        IMG_V1,
        /// <summary>
        /// Specifies to use the new image database v2.
        /// </summary>
        [EnumMember]
        IMG_V2,
        /// <summary>
        /// Specifies to use the tempoal database.
        /// </summary>
        [EnumMember]
        TEMPORAL,
        /// <summary>
        /// Specifies the default version (currently V2)
        /// </summary>
        [EnumMember]
        DEFAULT = IMG_V2
    }

#pragma warning disable 1591

    [ServiceContract]
    public interface IXDatabaseEvent /** @private */
    {
        [OperationContract(IsOneWay = false)]
        void OnResult(string strMsg, double dfProgress);

        [OperationContract(IsOneWay = false)]
        void OnError(DatabaseErrorData err);
    }

#pragma warning restore 1591

    /// <summary>
    /// The IXDatabaseBase interface defines the general interface to the in-memory database.
    /// </summary>
    [ServiceContract(CallbackContract = typeof(IXDatabaseEvent), SessionMode = SessionMode.Required)]
    public interface IXDatabaseBase
    {
        #region Initialization and Cleanup

        /// <summary>
        /// Set the database connection to use.
        /// </summary>
        /// <param name="ci">Specifies the connection information.</param>
        [OperationContract(IsOneWay = false)]
        void SetConnection(ConnectInfo ci);

        /// <summary>
        /// Initializes the image database.
        /// </summary>
        /// <param name="s">Specifies the caffe settings.</param>
        /// <param name="strDs">Specifies the data set to load.</param>
        /// <param name="strEvtCancel">Optionally, specifies the name of the CancelEvent used to cancel load operations (default = null).</param>
        /// <param name="prop">Optionally, specifies the properties for the initialization (default = null).</param>
        /// <returns>Returns <i>true</i> on success, <i>false</i> otherwise.</returns>
        [OperationContract(IsOneWay = false)]
        bool InitializeWithDsName1(SettingsCaffe s, string strDs, string strEvtCancel = null, PropertySet prop = null);

        /// <summary>
        /// Initializes the image database.
        /// </summary>
        /// <param name="s">Specifies the caffe settings.</param>
        /// <param name="ds">Specifies the data set to load.</param>
        /// <param name="strEvtCancel">Optionally, specifies the name of the CancelEvent used to cancel load operations (default = null).</param>
        /// <param name="prop">Optionally, specifies the properties for the initialization (default = null).</param>
        /// <returns>Returns <i>true</i> on success, <i>false</i> otherwise.</returns>
        [OperationContract(IsOneWay = false)]
        bool InitializeWithDs1(SettingsCaffe s, DatasetDescriptor ds, string strEvtCancel = null, PropertySet prop = null);

        /// <summary>
        /// Initializes the image database.
        /// </summary>
        /// <param name="s">Specifies the caffe settings.</param>
        /// <param name="nDataSetID">Specifies the database ID of the data set to load.</param>
        /// <param name="strEvtCancel">Specifies the name of the CancelEvent used to cancel load operations.</param>
        /// <param name="nPadW">Optionally, specifies the padding to add to each image width (default = 0).</param>
        /// <param name="nPadH">Optionally, specifies the padding to add to each image height (default = 0).</param>
        /// <param name="prop">Optionally, specifies the properties for the initialization (default = null).</param>
        /// <returns>Returns <i>true</i> on success, <i>false</i> otherwise.</returns>
        [OperationContract(IsOneWay = false)]
        bool InitializeWithDsId1(SettingsCaffe s, int nDataSetID, string strEvtCancel = null, int nPadW = 0, int nPadH = 0, PropertySet prop = null);

        /// <summary>
        /// Load another, 'secondary' dataset.
        /// </summary>
        /// <remarks>
        /// The primary dataset should be loaded using one of the 'Initialize' methods.  This method is provided to allow for loading
        /// multiple datasets.
        /// </remarks>
        /// <param name="nDsId">Specifies the ID of the data set.</param>
        /// <param name="strEvtCancel">Specifies the name of the CancelEvent used to cancel load operations.</param>
        /// <returns>When loaded, the handle to the default query state is returned, otherwise 0 is returned.</returns>
        [OperationContract(IsOneWay = false)]
        bool LoadDatasetByID1(int nDsId, string strEvtCancel = null);

        /// <summary>
        /// Load another, 'secondary' dataset.
        /// </summary>
        /// <remarks>
        /// The primary dataset should be loaded using one of the 'Initialize' methods.  This method is provided to allow for loading
        /// multiple datasets.
        /// </remarks>
        /// <param name="strDs">Specifies the name of the data set.</param>
        /// <param name="strEvtCancel">Specifies the name of the CancelEvent used to cancel load operations.</param>
        /// <returns>When loaded, the handle to the default query state is returned, otherwise 0 is returned.</returns>
        [OperationContract(IsOneWay = false)]
        bool LoadDatasetByName1(string strDs, string strEvtCancel = null);

        /// <summary>
        /// Releases the image database, and if this is the last instance using the in-memory database, frees all memory used.
        /// </summary>
        /// <param name="nDsId">Optionally, specifies the dataset previously used.</param>
        /// <param name="bForce">Optionally, force the cleanup even if other users are using the database.</param>
        [OperationContract(IsOneWay = false)]
        void CleanUp(int nDsId = 0, bool bForce = false);

        #endregion // Initialization and Cleanup

        #region Properties
        /// <summary>
        /// Returns the version of the MyCaffe Image Database being used.
        /// </summary>
        /// <returns>Returns the version.</returns>
        [OperationContract(IsOneWay = false)]
        DB_VERSION GetVersion();

        /// <summary>
        /// Returns whether or not the item data criteria is loaded with each item.
        /// </summary>
        [OperationContract(IsOneWay = false)]
        bool GetLoadItemDataCriteria();

        /// <summary>
        /// Returns whether or not the item debug data is loaded with each item.
        /// </summary>
        [OperationContract(IsOneWay = false)]
        bool GetLoadItemDebugData();

        /// <summary>
        /// Returns the percentage that a dataset is loaded into memory.
        /// </summary>
        /// <param name="strDataset">Specifies the name of the dataset.</param>
        /// <param name="dfTraining">Specifies the percent of training items that are loaded.</param>
        /// <param name="dfTesting">Specifies the percent of testing items that are loaded.</param>
        /// <returns>The current image load percent for the dataset is returned..</returns>
        [OperationContract(IsOneWay = false)]
        double GetDatasetLoadedPercentByName(string strDataset, out double dfTraining, out double dfTesting);

        /// <summary>
        /// Returns the percentage that a dataset is loaded into memory.
        /// </summary>
        /// <param name="nDatasetID">Specifies the ID of the dataset.</param>
        /// <param name="dfTraining">Specifies the percent of training items that are loaded.</param>
        /// <param name="dfTesting">Specifies the percent of testing items that are loaded.</param>
        /// <returns>The current image load percent for the dataset is returned..</returns>
        [OperationContract(IsOneWay = false)]
        double GetDatasetLoadedPercentById(int nDatasetID, out double dfTraining, out double dfTesting);

        /// <summary>
        /// Returns the number of items (e.g., images, or temporal items) in a given data source.
        /// </summary>
        /// <param name="nSrcId">Specifies the data source ID.</param>
        /// <param name="strFilterVal">Optionally, specifies the filter value that the description must match (default = <i>null</i>, which ignores this parameter).</param>
        /// <param name="nBoostVal">Optionally, specifies the boost value that the boost must match (default = <i>null</i>, which ignores this parameter).</param>
        /// <param name="bBoostValIsExact">Optionally, specifies whether or the boost value (if specified) is to be used literally (exact = true), or as a minimum boost value.</param>
        /// <returns>The number of items (e.g., images or temporal items) is returned.</returns>
        /// <remarks>When using the 'nBoostValue' negative values are used to test the exact match of the boost value with the absolute value of the 'nBoostValue', ande
        /// positive values are used to test for boost values that are greater than or equal to the 'nBoostValue'.</remarks>
        [OperationContract(IsOneWay = false)]
        int GetItemCount(int nSrcId, string strFilterVal = null, int? nBoostVal = null, bool bBoostValIsExact = false);

        /// <summary>
        /// Returns the label and image selection method used.
        /// </summary>
        /// <returns>A tuple containing the Label and Image selection method.</returns>
        [OperationContract(IsOneWay = false)]
        Tuple<DB_LABEL_SELECTION_METHOD, DB_ITEM_SELECTION_METHOD> GetSelectionMethod();

        /// <summary>
        /// Sets the label and image selection methods.
        /// </summary>
        /// <param name="lbl">Specifies the label selection method or <i>null</i> to ignore.</param>
        /// <param name="img">Specifies the image selection method or <i>null</i> to ignore.</param>
        [OperationContract(IsOneWay = false)]
        void SetSelectionMethod(DB_LABEL_SELECTION_METHOD? lbl, DB_ITEM_SELECTION_METHOD? img);
        #endregion

        #region Sources
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
        #endregion

        #region Datasets
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
        /// Reload a data set.
        /// </summary>
        /// <param name="nDsId">Specifies the ID of the data set.</param>
        /// <returns>If the data set is found, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        [OperationContract(IsOneWay = false)]
        bool ReloadDataset(int nDsId);

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
        #endregion

        #region Image Acquisition

        /// <summary>
        /// Query an image in a given data source.
        /// </summary>
        /// <param name="nSrcId">Specifies the databse ID of the data source.</param>
        /// <param name="nIdx">Specifies the image index to query.  Note, the index is only used in non-random image queries.</param>
        /// <param name="labelSelectionOverride">Optionally, specifies the label selection method override.  The default = null, which directs the method to use the label selection method specified during Initialization.</param>
        /// <param name="imageSelectionOverride">Optionally, specifies the image selection method override.  The default = null, which directs the method to use the image selection method specified during Initialization.</param>
        /// <param name="nLabel">Optionally, specifies a label set to use for the image selection.  When specified only images of this label are returned using the image selection method.</param>
        /// <param name="bLoadDataCriteria">Specifies to load the data criteria data (default = false).</param>
        /// <param name="bLoadDebugData">Specifies to load the debug data (default = false).</param>
        /// <param name="bThrowExceptions">Optionally, specifies to throw exceptions on error (default = true).</param>
        /// <returns>The image SimpleDatum is returned.</returns>
        [OperationContract(IsOneWay = false)]
        SimpleDatum QueryItem(int nSrcId, int nIdx, DB_LABEL_SELECTION_METHOD? labelSelectionOverride = null, DB_ITEM_SELECTION_METHOD? imageSelectionOverride = null, int? nLabel = null, bool bLoadDataCriteria = false, bool bLoadDebugData = false, bool bThrowExceptions = true);

        /// <summary>
        /// Query an image in a given data source.
        /// </summary>
        /// <param name="nSrcId">Specifies the databse ID of the data source.</param>
        /// <param name="dt">Specifies the image time to query.</param>
        /// <param name="labelSelectionOverride">Optionally, specifies the label selection method override.  The default = null, which directs the method to use the label selection method specified during Initialization.</param>
        /// <param name="imageSelectionOverride">Optionally, specifies the image selection method override.  The default = null, which directs the method to use the image selection method specified during Initialization.</param>
        /// <param name="nLabel">Optionally, specifies a label set to use for the image selection.  When specified only images of this label are returned using the image selection method.</param>
        /// <param name="bLoadDataCriteria">Specifies to load the data criteria data (default = false).</param>
        /// <param name="bLoadDebugData">Specifies to load the debug data (default = false).</param>
        /// <param name="bThrowExceptions">Optionally, specifies to throw exceptions on error (default = true).</param>
        /// <returns>The image SimpleDatum is returned.</returns>
        [OperationContract(IsOneWay = false)]
        SimpleDatum QueryItem(int nSrcId, DateTime dt, DB_LABEL_SELECTION_METHOD? labelSelectionOverride = null, DB_ITEM_SELECTION_METHOD? imageSelectionOverride = null, int? nLabel = null, bool bLoadDataCriteria = false, bool bLoadDebugData = false, bool bThrowExceptions = true);

        /// <summary>
        /// Returns the array of items (e.g., images or temporal items) in the item set, possibly filtered with the filtering parameters.
        /// </summary>
        /// <param name="nSrcId">Specifies the data source ID.</param>
        /// <param name="nStartIdx">Specifies a starting index from which the query is to start within the set of items.</param>
        /// <param name="nQueryCount">Optionally, specifies a number of items to retrieve within the set (default = int.MaxValue).</param>
        /// <param name="strFilterVal">Optionally, specifies the filter value that the description must match (default = <i>null</i>, which ignores this parameter).</param>
        /// <param name="nBoostVal">Optionally, specifies the boost value that the boost must match (default = <i>null</i>, which ignores this parameter).</param>
        /// <param name="bBoostValIsExact">Optionally, specifies whether or the boost value (if specified) is to be used literally (exact = true), or as a minimum boost value.</param>
        /// <param name="bAttemptDirectLoad">Optionaly, specifies to directly load all images not already loaded.</param>
        /// <returns>The list of items (e.g., images or temporal items) is returned.</returns>
        /// <remarks>When using the 'nBoostValue' negative values are used to test the exact match of the boost value with the absolute value of the 'nBoostValue', and
        /// positive values are used to test for boost values that are greater than or equal to the 'nBoostValue'.</remarks>
        [OperationContract(IsOneWay = false)]
        List<SimpleDatum> GetItemsFromIndex(int nSrcId, int nStartIdx, int nQueryCount = int.MaxValue, string strFilterVal = null, int? nBoostVal = null, bool bBoostValIsExact = false, bool bAttemptDirectLoad = false);

        /// <summary>
        /// Returns the array of items (e.g., images or temporal items) in the item set, possibly filtered with the filtering parameters.
        /// </summary>
        /// <param name="nSrcId">Specifies the data source ID.</param>
        /// <param name="dtStart">Specifies a starting time from which the query is to start within the set of images.</param>
        /// <param name="nQueryCount">Optionally, specifies a number of items to retrieve within the set (default = int.MaxValue).</param>
        /// <param name="strFilterVal">Optionally, specifies the filter value that the description must match (default = <i>null</i>, which ignores this parameter).</param>
        /// <param name="nBoostVal">Optionally, specifies the boost value that the boost must match (default = <i>null</i>, which ignores this parameter).</param>
        /// <param name="bBoostValIsExact">Optionally, specifies whether or the boost value (if specified) is to be used literally (exact = true), or as a minimum boost value.</param>
        /// <returns>The list of items (e.g., images or temporal items) is returned.</returns>
        /// <remarks>When using the 'nBoostValue' negative values are used to test the exact match of the boost value with the absolute value of the 'nBoostValue', ande
        /// positive values are used to test for boost values that are greater than or equal to the 'nBoostValue'.</remarks>
        [OperationContract(IsOneWay = false)]
        List<SimpleDatum> GetItemsFromTime(int nSrcId, DateTime dtStart, int nQueryCount = int.MaxValue, string strFilterVal = null, int? nBoostVal = null, bool bBoostValIsExact = false);

        /// <summary>
        /// Returns the array of items (e.g., images or temporal items) in the item set, possibly filtered with the filtering parameters.
        /// </summary>
        /// <param name="nSrcId">Specifies the data source ID.</param>
        /// <param name="rgIdx">Specifies an array of indexes to query.</param>
        /// <param name="strFilterVal">Optionally, specifies the filter value that the description must match (default = <i>null</i>, which ignores this parameter).</param>
        /// <param name="nBoostVal">Optionally, specifies the boost value that the boost must match (default = <i>null</i>, which ignores this parameter).</param>
        /// <param name="bBoostValIsExact">Optionally, specifies whether or the boost value (if specified) is to be used literally (exact = true), or as a minimum boost value.</param>
        /// <returns>The list of items (e.g., image or temporal item) is returned.</returns>
        /// <remarks>When using the 'nBoostValue' negative values are used to test the exact match of the boost value with the absolute value of the 'nBoostValue', ande
        /// positive values are used to test for boost values that are greater than or equal to the 'nBoostValue'.</remarks>
        [OperationContract(IsOneWay = false)]
        List<SimpleDatum> GetItems(int nSrcId, int[] rgIdx, string strFilterVal = null, int? nBoostVal = null, bool bBoostValIsExact = false);

        /// <summary>
        /// Get the item (e.g., image or temporal item) with a given Raw Item ID.
        /// </summary>
        /// <param name="nItemID">Specifies the Raw Image ID of the image to get.</param>
        /// <param name="rgSrcId">Specifies a list of Source ID's to search for the image.</param>
        /// <returns>The SimpleDatum of the item is returned.</returns>
        /// <remarks>Note, temporal items are alwasy returned as a set of temporal items as defined by past and future steps defined in the temporal specific interface.</remarks>
        [OperationContract(IsOneWay = false)]
        SimpleDatum GetItem(int nItemID, params int[] rgSrcId);

        /// <summary>
        /// Searches for the item (e.g., image or temporal item) index of an image within a data source matching a DateTime/description pattern.
        /// </summary>
        /// <remarks>
        /// Optionally, items may have a time-stamp and/or description associated with each item.  In such cases
        /// searching by the time-stamp + description can be useful in some instances.
        /// </remarks>
        /// <param name="nSrcId">Specifies the data source ID of the data source to be searched.</param>
        /// <param name="dt">Specifies the time-stamp to search for.</param>
        /// <param name="strDescription">Specifies the description to search for.</param>
        /// <returns>If found the zero-based index of the item (e.g., image or temporal item) is returned, otherwise -1 is returned.</returns>
        [OperationContract(IsOneWay = false)]
        int FindItemIndex(int nSrcId, DateTime dt, string strDescription);

        #endregion

        #region Item Mean

        /// <summary>
        /// Queries the item (e.g., image or temporal item) mean for a data source from the database on disk.
        /// </summary>
        /// <param name="nSrcId">Specifies the ID of the data source.</param>
        /// <returns>The item mean is returned as a SimpleDatum.</returns>
        /// <remarks>Note, the mean for a temporal item is a set of values where one mean value exists for each data stream.</remarks>
        [OperationContract(IsOneWay = false)]
        [FaultContract(typeof(DatabaseErrorData))]
        SimpleDatum QueryItemMean(int nSrcId);

        /// <summary>
        /// Queries the item (e.g., image or temporal item) mean for a data source from the database on disk.
        /// </summary>
        /// <remarks>
        /// If the item mean does not exist in the database, one is created, saved
        /// and then returned.
        /// </remarks>
        /// <param name="nSrcId">Specifies the ID of the data source.</param>
        /// <returns>The item (e.g., image or temporal item) mean is returned as a SimpleDatum.</returns>
        /// <remarks>Note, the mean for a temporal item is a set of values where one mean value exists for each data stream.</remarks>
        [OperationContract(IsOneWay = false)]
        [FaultContract(typeof(DatabaseErrorData))]
        SimpleDatum QueryItemMeanFromDb(int nSrcId);

        /// <summary>
        /// Returns the item (e.g., image or temporal item) mean for a data source.
        /// </summary>
        /// <param name="nSrcId">Specifies the ID of the data source.</param>
        /// <param name="rgParams">Optionally, specifies image mean parameters to query (default = none)</param>
        /// <returns>The item (e.g., image or temporal item) mean is returned as a SimpleDatum.</returns>
        /// <remarks>Note, the mean for a temporal item is a set of values where one mean value exists for each data stream.</remarks>
        [OperationContract(IsOneWay = false)]
        [FaultContract(typeof(DatabaseErrorData))]
        SimpleDatum GetItemMean(int nSrcId, params string[] rgParams);

        /// <summary>
        /// Returns the item (e.g., image or temporal item) mean for the Training data source of a given data set.
        /// </summary>
        /// <param name="nDatasetId">Specifies the data set to use.</param>
        /// <returns>The item (e.g., image or temporal item) mean is returned as a SimpleDatum.</returns>
        /// <remarks>Note, the mean for a temporal item is a set of values where one mean value exists for each data stream.</remarks>
        [OperationContract(IsOneWay = false)]
        SimpleDatum QueryItemMeanFromDataset(int nDatasetId);

        #endregion // Image Mean
    }

    /// <summary>
    /// Teh IXTemporalDatabaseBase interface defines the general interface to the in-memory temporal database.
    /// </summary>
    [ServiceContract(CallbackContract = typeof(IXDatabaseEvent), SessionMode = SessionMode.Required)]
    public interface IXTemporalDatabaseBase : IXDatabaseBase
    {
        /// <summary>
        /// The SetInitializationProperties method is used to set the initialization properties of the database.
        /// </summary>
        /// <param name="prop">Specifies the initialization properties.</param>
        /// <remarks>This method must be called before any other initialization methods.</remarks>
        [OperationContract(IsOneWay = false)]
        void SetInitializationProperties(PropertySet prop);

        /// <summary>
        /// Returns the total number of blocks in the database where one block is a set of (historical and future steps).
        /// </summary>
        /// <param name="nDsId">Specifies the dataset ID.</param>
        /// <param name="phase">Specifies the phase who's data size is to be returned.</param>
        /// <param name="nHistoricalSteps">Specifies the number of historical steps.</param>
        /// <param name="nFutureSteps">Specifies the number of future steps.</param>
        /// <returns>The total number of blocks is returned.</returns>
        [OperationContract(IsOneWay = false)]
        int GetTotalSize(int nDsId, Phase phase, int nHistoricalSteps, int nFutureSteps);

        /// <summary>
        /// Returns a block of static, observed and known data from the database where one block is a set of (historical and future steps).
        /// </summary>
        /// <param name="nQueryIdx">Specifies the index of the query within a batch.</param>
        /// <param name="nSrcId">Specifies the source ID of the data source.</param>
        /// <param name="nItemIdx">Specifies the item index override when not null, returns the item index used.</param>
        /// <param name="nValueIdx">Specifies the value index override when not null, returns the index used with in the item.</param>
        /// <param name="itemSelectionOverride">Optionally, specifies the item selection method used to select the item (e.g., customer, station, stock symbol)</param>
        /// <param name="valueSelectionOverride">Optionally, specifies the value selection method used to select the index within the temporal data of the selected item.</param>
        /// <param name="ordering">Optionally, specifies the ordering of item selection (only applies when itemSelectionOverride and valueSelectionOverride are both set to 'NONE').</param>
        /// <param name="bOutputTime">Optionally, output the time data.</param>
        /// <param name="bOutputMask">Optionally, output the mask data.</param>
        /// <param name="bOutputItemIDs">Optionally, output the item IDs.</param>
        /// <param name="bEnableDebug">Optionally, specifies to enable debug output (default = false).</param>
        /// <param name="strDebugPath">Optionally, specifies the debug path where debug images are placed when 'EnableDebug' = true.</param>
        /// <param name="bIgnoreFuture">Optionally, specifies to ignore the future data.</param>
        /// <returns>An collection of SimpleTemporalDatum containing the static num, statuc cat, historical num, historical cat, future num, future cat, target and target hist data is returned. 
        /// If one of the value types are not produced, null is filled in the array slot.</returns>
        [OperationContract(IsOneWay = false)]
        SimpleTemporalDatumCollection QueryTemporalItem(int nQueryIdx, int nSrcId, ref int? nItemIdx, ref int? nValueIdx, DB_LABEL_SELECTION_METHOD? itemSelectionOverride = null, DB_ITEM_SELECTION_METHOD? valueSelectionOverride = null, DB_INDEX_ORDER? ordering = null, bool bOutputTime = false, bool bOutputMask = false, bool bOutputItemIDs = false, bool bEnableDebug = false, string strDebugPath = null, bool bIgnoreFuture = false);

        /// <summary>
        /// Reset the database indexes.
        /// </summary>
        /// <remarks>
        /// This method is only used when using sequential selection.
        /// </remarks>
        [OperationContract(IsOneWay = false)]
        void Reset();

        /// <summary>
        /// Checks whether or not the value index is valid.  An index is considered invalid if the value index + nStepsForward is greater than the number of values in the items.
        /// </summary>
        /// <param name="strSource">Specifies the source name of the temporal dataset.</param>
        /// <param name="nItemIndex">Specifies the item index.</param>
        /// <param name="nValueIndex">Specifies the value index.</param>
        /// <param name="nStepsForward">Specifies the number of steps (hist + fut) forward from the value index.</param>
        /// <returns>If there is enough data from the value index + steps, true is returned, otherwise false.</returns>
        [OperationContract(IsOneWay = false)]
        bool IsValueIndexValid(string strSource, int nItemIndex, int nValueIndex, int nStepsForward);

        /// <summary>
        /// Returns the master time sync for a given phase of a dataset.
        /// </summary>
        /// <param name="nDataSetID">Specifies the dataset ID.</param>
        /// <param name="phase">Specifies the phase of the dataset.</param>
        /// <returns>The master time sync is returned.</returns>
        [OperationContract(IsOneWay = false)]
        List<DateTime> GetMasterTimeSync(int nDataSetID, Phase phase);
    }

    /// <summary>
    /// The IXImageDatabaseBase interface defines the general interface to the in-memory image database.
    /// </summary>
    [ServiceContract(CallbackContract = typeof(IXDatabaseEvent), SessionMode = SessionMode.Required)]
    public interface IXImageDatabaseBase : IXDatabaseBase
    {
        #region Properties

        /// <summary>
        /// Returns a string with the query hit percent for each boost (e.g. the percentage that each boost value has been queried).
        /// </summary>
        /// <param name="strSource">Specifies the data source who's hit percentages are to be retrieved.</param>
        /// <returns>A string representing the query hit percentages is returned.</returns>
        [OperationContract(IsOneWay = false)]
        string GetBoostQueryHitPercentsAsTextFromSourceName(string strSource);

        /// <summary>
        /// Returns a string with the query hit percent for each label (e.g. the percentage that each label has been queried).
        /// </summary>
        /// <param name="strSource">Specifies the data source who's hit percentages are to be retrieved.</param>
        /// <returns>A string representing the query hit percentages is returned.</returns>
        [OperationContract(IsOneWay = false)]
        string GetLabelQueryHitPercentsAsTextFromSourceName(string strSource);

        /// <summary>
        /// Returns a string with the query epoch counts for each label (e.g. the number of times all images with the label have been queried).
        /// </summary>
        /// <param name="strSource">Specifies the data source who's query epochs are to be retrieved.</param>
        /// <returns>A string representing the query epoch counts is returned.</returns>
        [OperationContract(IsOneWay = false)]
        string GetLabelQueryEpocsAsTextFromSourceName(string strSource);

        #endregion // Properties

        #region Datasets

        /// <summary>
        /// Reloads the images of a data source.
        /// </summary>
        /// <param name="nSrcId">Specifies the ID of the data source.</param>
        /// <returns>If the data source is found, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        [OperationContract(IsOneWay = false)]
        bool ReloadImageSet(int nSrcId);

        #endregion // Datasets

        #region Labels

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
        /// Updates the number of images of each label within a data source.
        /// </summary>
        /// <param name="nProjectId">Specifies a project ID.</param>
        /// <param name="nSrcId">Specifies the ID of the data source.</param>
        [OperationContract(IsOneWay = false)]
        void UpdateLabelCounts(int nProjectId, int nSrcId);

        /// <summary>
        /// Returns a label lookup of counts for a given data source.
        /// </summary>
        /// <param name="nSrcId">Specifies the ID of the data source.</param>
        /// <returns>A dictionary containing label,count pairs is returned.</returns>
        [OperationContract(IsOneWay = false)]
        [FaultContract(typeof(DatabaseErrorData))]
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

        #endregion // Labels
    }

    /// <summary>
    /// The IXImageDatabase interface defines the eneral interface to the in-memory image database.
    /// </summary>
    [ServiceContract(CallbackContract = typeof(IXDatabaseEvent), SessionMode = SessionMode.Required)]
    public interface IXImageDatabase1 : IXImageDatabaseBase
    {
        /// <summary>
        /// Updates the label boosts for the images based on the label boosts set for the given project.
        /// </summary>
        /// <param name="nProjectId">Specifies the project ID in the database.</param>
        /// <param name="nSrcId">Specifies the data source ID.</param>
        [OperationContract(IsOneWay = false)]
        void UpdateLabelBoosts(int nProjectId, int nSrcId);

        /// <summary>
        /// Reset the query for the given data set ID.
        /// </summary>
        /// <param name="nDsID">Specifies the data set ID whos query indexes are to be reset.</param>
        [OperationContract(IsOneWay = false)]
        void ResetQuery(int nDsID);

        /// <summary>
        /// Sort the internal images.
        /// </summary>
        /// <param name="nSrcId">Specifies the databse ID of the data source.</param>
        /// <param name="method">Specifies the sorting method.</param>
        /// <returns>If the sorting is successful, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        [OperationContract(IsOneWay = false)]
        bool Sort(int nSrcId, IMGDB_SORT method);

        /// <summary>
        /// Create a dynamic dataset organized by time from a pre-existing dataset.
        /// </summary>
        /// <param name="nDsId">Specifies the database ID of the dataset to copy.</param>
        /// <returns>The dataset ID of the newly created dataset is returned.</returns>
        [OperationContract(IsOneWay = false)]
        int CreateDatasetOranizedByTime(int nDsId);

        /// <summary>
        /// Delete a dataset created with CreateDatasetOrganizedByTime.
        /// </summary>
        /// <param name="nDsId">Specifies the dataset ID of the created dataset.</param>
        /// <returns>If successful, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        [OperationContract(IsOneWay = false)]
        bool DeleteCreatedDataset(int nDsId);

        /// <summary>
        /// Delete all datasets created with CreateDatasetOrganizedByTime.
        /// </summary>
        [OperationContract(IsOneWay = false)]
        void DeleteAllCreatedDatasets();

        /// <summary>
        /// Delete all label boosts for a given data source associated with a given project.
        /// </summary>
        /// <param name="nProjectId">Specifies the project ID.</param>
        /// <param name="nSrcId">Specifies the ID of the data source.</param>
        [OperationContract(IsOneWay = false)]
        void DeleteLabelBoosts(int nProjectId, int nSrcId);

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
        /// Returns the label boosts as a text string for all boosted labels within a data source associated with a given project. 
        /// </summary>
        /// <param name="nProjectId">Specifies the project ID.</param>
        /// <param name="nSrcId">Specifies the ID of the data source.</param>
        /// <returns>The label boosts are returned as a text string.</returns>
        [OperationContract(IsOneWay = false)]
        string GetLabelBoostsAsTextFromProject(int nProjectId, int nSrcId);

        /// <summary>
        /// When using a <i>Load Limit</i> that is greater than 0, this function loads the next set of images.
        /// </summary>
        /// <param name="strEvtCancel">Specifies the name of the Cancel Event to abort loading the images.</param>
        /// <returns>Returns <i>true</i> on success, <i>false</i> otherwise.</returns>
        [OperationContract(IsOneWay = false)]
        bool LoadNextSet(string strEvtCancel);
    }

    /// <summary>
    /// The IXImageDatabase2 interface defines the general interface to the in-memory image database (v2).
    /// </summary>
    [ServiceContract(CallbackContract = typeof(IXDatabaseEvent), SessionMode = SessionMode.Required)]
    public interface IXImageDatabase2 : IXImageDatabaseBase
    {
        #region Initialization and Cleanup

        /// <summary>
        /// Initializes the image database.
        /// </summary>
        /// <param name="s">Specifies the caffe settings.</param>
        /// <param name="strDs">Specifies the data set to load.</param>
        /// <param name="strEvtCancel">Specifies the name of the CancelEvent used to cancel load operations.</param>
        /// <returns>Upon loading the dataset a handle to the default QueryState is returned (which is ordered by Index), or 0 on cancel.</returns>
        [OperationContract(IsOneWay = false)]
        long InitializeWithDsName(SettingsCaffe s, string strDs, string strEvtCancel = null);

        /// <summary>
        /// Initializes the image database.
        /// </summary>
        /// <param name="s">Specifies the caffe settings.</param>
        /// <param name="ds">Specifies the data set to load.</param>
        /// <param name="strEvtCancel">Specifies the name of the CancelEvent used to cancel load operations.</param>
        /// <returns>Upon loading the dataset a handle to the default QueryState is returned (which is ordered by Index), or 0 on cancel.</returns>
        [OperationContract(IsOneWay = false)]
        long InitializeWithDs(SettingsCaffe s, DatasetDescriptor ds, string strEvtCancel = null);

        /// <summary>
        /// Initializes the image database.
        /// </summary>
        /// <param name="s">Specifies the caffe settings.</param>
        /// <param name="nDataSetID">Specifies the database ID of the data set to load.</param>
        /// <param name="strEvtCancel">Specifies the name of the CancelEvent used to cancel load operations.</param>
        /// <param name="nPadW">Specifies the padding to add to each image width (default = 0).</param>
        /// <param name="nPadH">Specifies the padding to add to each image height (default = 0).</param>
        /// <returns>Upon loading the dataset a handle to the default QueryState is returned (which is ordered by Index), or 0 on cancel.</returns>
        [OperationContract(IsOneWay = false)]
        long InitializeWithDsId(SettingsCaffe s, int nDataSetID, string strEvtCancel = null, int nPadW = 0, int nPadH = 0);

        /// <summary>
        /// Load another, 'secondary' dataset.
        /// </summary>
        /// <remarks>
        /// The primary dataset should be loaded using one of the 'Initialize' methods.  This method is provided to allow for loading
        /// multiple datasets.
        /// </remarks>
        /// <param name="nDsId">Specifies the ID of the data set.</param>
        /// <param name="strEvtCancel">Specifies the name of the CancelEvent used to cancel load operations.</param>
        /// <returns>When loaded, the handle to the default query state is returned, otherwise 0 is returned.</returns>
        [OperationContract(IsOneWay = false)]
        long LoadDatasetByID(int nDsId, string strEvtCancel = null);

        /// <summary>
        /// Load another, 'secondary' dataset.
        /// </summary>
        /// <remarks>
        /// The primary dataset should be loaded using one of the 'Initialize' methods.  This method is provided to allow for loading
        /// multiple datasets.
        /// </remarks>
        /// <param name="strDs">Specifies the name of the data set.</param>
        /// <param name="strEvtCancel">Specifies the name of the CancelEvent used to cancel load operations.</param>
        /// <returns>When loaded, the handle to the default query state is returned, otherwise 0 is returned.</returns>
        [OperationContract(IsOneWay = false)]
        long LoadDatasetByName(string strDs, string strEvtCancel = null);

        /// <summary>
        /// Reload the indexing for a data set.
        /// </summary>
        /// <param name="nDsId">Specifies the dataset ID.</param>
        /// <returns>If the data source(s) have their indexing reloaded, <i>true</i> is returned, otherwise <i>false</i>.</returns>
        [OperationContract(IsOneWay = false)]
        bool ReloadIndexing(int nDsId);

        /// <summary>
        /// Reload the indexing for a data set.
        /// </summary>
        /// <param name="strDs">Specifies the dataset name.</param>
        /// <returns>If the data source(s) have their indexing reloaded, <i>true</i> is returned, otherwise <i>false</i>.</returns>
        [OperationContract(IsOneWay = false)]
        bool ReloadIndexing(string strDs);

        /// <summary>
        /// Wait for the dataset loading to complete.
        /// </summary>
        /// <param name="nDsId">Specifies the dataset ID.</param>
        /// <param name="bTraining">Specifies to wait for the training data source to load.</param>
        /// <param name="bTesting">Specifies to wait for the testing data source to load.</param>
        /// <param name="nWait">Specifies the amount of time to wait in ms. (default = int.MaxValue).</param>
        /// <returns>If the data source(s) complete loading, <i>true</i> is returned, otherwise <i>false</i>.</returns>
        bool WaitForDatasetToLoad(int nDsId, bool bTraining, bool bTesting, int nWait = int.MaxValue);

        /// <summary>
        /// Wait for the dataset loading to complete.
        /// </summary>
        /// <param name="strDs">Specifies the dataset name.</param>
        /// <param name="bTraining">Specifies to wait for the training data source to load.</param>
        /// <param name="bTesting">Specifies to wait for the testing data source to load.</param>
        /// <param name="nWait">Specifies the amount of time to wait in ms. (default = int.MaxValue).</param>
        /// <returns>If the data source(s) complete loading, <i>true</i> is returned, otherwise <i>false</i>.</returns>
        bool WaitForDatasetToLoad(string strDs, bool bTraining, bool bTesting, int nWait = int.MaxValue);

        #endregion // Initialization and Cleanup

        #region Query States

        /// <summary>
        /// Create a new query state, optionally with a certain ordering.
        /// </summary>
        /// <param name="nDsId">Specifies the dataset on which the query states are to be created.</param>
        /// <param name="bUseUniqueLabelIndexes">Optionally, specifies to use unique label indexes which is slightly slower, but ensures each label is hit per epoch equally (default = true).</param>
        /// <param name="bUseUniqueImageIndexes">Optionally, specifies to use unique image indexes which is slightly slower, but ensures each image is hit per epoch (default = true).</param>
        /// <param name="sort">Optionally, specifies an index ordering (default = NONE).</param>
        /// <returns>A handle to the new query state is returned.</returns>
        [OperationContract(IsOneWay = false)]
        long CreateQueryState(int nDsId, bool bUseUniqueLabelIndexes = true, bool bUseUniqueImageIndexes = true, IMGDB_SORT sort = IMGDB_SORT.NONE);

        /// <summary>
        /// Create a new query state, optionally with a certain ordering.
        /// </summary>
        /// <param name="strDs">Specifies the dataset on which the query states are to be created.</param>
        /// <param name="bUseUniqueLabelIndexes">Optionally, specifies to use unique label indexes which is slightly slower, but ensures each label is hit per epoch equally (default = true).</param>
        /// <param name="bUseUniqueImageIndexes">Optionally, specifies to use unique image indexes which is slightly slower, but ensures each image is hit per epoch (default = true).</param>
        /// <param name="sort">Optionally, specifies an index ordering (default = NONE).</param>
        /// <returns>A handle to the new query state is returned.</returns>
        [OperationContract(IsOneWay = false)]
        long CreateQueryState(string strDs, bool bUseUniqueLabelIndexes = true, bool bUseUniqueImageIndexes = true, IMGDB_SORT sort = IMGDB_SORT.NONE);

        /// <summary>
        /// Set the default query state to the query state specified for the dataset specified.
        /// </summary>
        /// <param name="nDsId">Specifies the dataset ID.</param>
        /// <param name="lQueryState">Specifies the query state to set.</param>
        /// <returns>Returns <i>true</i> on success, <i>false</i> on failure.</returns>
        [OperationContract(IsOneWay = false)]
        bool SetDefaultQueryState(int nDsId, long lQueryState);

        /// <summary>
        /// Set the default query state to the query state specified for the dataset specified.
        /// </summary>
        /// <param name="strDs">Specifies the dataset name.</param>
        /// <param name="lQueryState">Specifies the query state to set.</param>
        /// <returns>Returns <i>true</i> on success, <i>false</i> on failure.</returns>
        [OperationContract(IsOneWay = false)]
        bool SetDefaultQueryState(string strDs, long lQueryState);

        /// <summary>
        /// Frees a query state from a given dataset.
        /// </summary>
        /// <param name="nDsId">Specifies the dataset on which to free the query state.</param>
        /// <param name="lHandle">Specifies the handle to the query state to free.</param>
        /// <returns>If found and freed, <i>true</i> is returned, otherwise <i>false</i>.</returns>
        [OperationContract(IsOneWay = false)]
        bool FreeQueryState(int nDsId, long lHandle);

        /// <summary>
        /// Frees a query state from a given dataset.
        /// </summary>
        /// <param name="strDs">Specifies the dataset name on which to free the query state.</param>
        /// <param name="lHandle">Specifies the handle to the query state to free.</param>
        /// <returns>If found and freed, <i>true</i> is returned, otherwise <i>false</i>.</returns>
        [OperationContract(IsOneWay = false)]
        bool FreeQueryState(string strDs, long lHandle);

        /// <summary>
        /// Returns a string with the query hit percent for each label (e.g. the percentage that each label has been queried).
        /// </summary>
        /// <param name="lQueryState">Specifies the handle to the query state.</param>
        /// <param name="strSource">Specifies the data source who's hit percentages are to be retrieved.</param>
        /// <returns>A string representing the query hit percentages is returned.</returns>
        [OperationContract(IsOneWay = false)]
        string GetLabelQueryHitPercentsAsTextFromSourceName(long lQueryState, string strSource);

        /// <summary>
        /// Returns a string with the query epoch counts for each label (e.g. the number of times all images with the label have been queried).
        /// </summary>
        /// <param name="lQueryState">Specifies the handle to the query state.</param>
        /// <param name="strSource">Specifies the data source who's query epochs are to be retrieved.</param>
        /// <returns>A string representing the query epoch counts is returned.</returns>
        [OperationContract(IsOneWay = false)]
        string GetLabelQueryEpocsAsTextFromSourceName(long lQueryState, string strSource);

        /// <summary>
        /// Returns a string with the query hit percent for each boost (e.g. the percentage that each boost value has been queried).
        /// </summary>
        /// <param name="lQueryState">Specifies the handle to the query state.</param>
        /// <param name="strSource">Specifies the data source who's hit percentages are to be retrieved.</param>
        /// <returns>A string representing the query hit percentages is returned.</returns>
        [OperationContract(IsOneWay = false)]
        string GetBoostQueryHitPercentsAsTextFromSourceName(long lQueryState, string strSource);

        #endregion // Query States

        #region Properties

        /// <summary>
        /// Returns the number of images in a given data source.
        /// </summary>
        /// <param name="lQueryState">Specifies a handle to the query state to use.</param>
        /// <param name="nSrcId">Specifies the data source ID.</param>
        /// <param name="strFilterVal">Optionally, specifies the filter value that the description must match (default = <i>null</i>, which ignores this parameter).</param>
        /// <param name="nBoostVal">Optionally, specifies the boost value that the boost must match (default = <i>null</i>, which ignores this parameter).</param>
        /// <param name="bBoostValIsExact">Optionally, specifies whether or the boost value (if specified) is to be used literally (exact = true), or as a minimum boost value.</param>
        /// <returns>The number of images is returned.</returns>
        /// <remarks>When using the 'nBoostValue' negative values are used to test the exact match of the boost value with the absolute value of the 'nBoostValue', ande
        /// positive values are used to test for boost values that are greater than or equal to the 'nBoostValue'.</remarks>
        [OperationContract(IsOneWay = false)]
        int GetImageCount(long lQueryState, int nSrcId, string strFilterVal = null, int? nBoostVal = null, bool bBoostValIsExact = false);

        #endregion // Properties

        #region Image Acquisition

        /// <summary>
        /// Returns the array of images in the image set, possibly filtered with the filtering parameters.
        /// </summary>
        /// <param name="lQueryState">Specifies a handle to the query state to use.</param>
        /// <param name="nSrcId">Specifies the data source ID.</param>
        /// <param name="nStartIdx">Specifies a starting index from which the query is to start within the set of images.</param>
        /// <param name="nQueryCount">Optionally, specifies a number of images to retrieve within the set (default = int.MaxValue).</param>
        /// <param name="strFilterVal">Optionally, specifies the filter value that the description must match (default = <i>null</i>, which ignores this parameter).</param>
        /// <param name="nBoostVal">Optionally, specifies the boost value that the boost must match (default = <i>null</i>, which ignores this parameter).</param>
        /// <param name="bBoostValIsExact">Optionally, specifies whether or the boost value (if specified) is to be used literally (exact = true), or as a minimum boost value.</param>
        /// <param name="bAttemptDirectLoad">Optionaly, specifies to directly load all images not already loaded.</param>
        /// <returns>The list of images is returned.</returns>
        /// <remarks>When using the 'nBoostValue' negative values are used to test the exact match of the boost value with the absolute value of the 'nBoostValue', ande
        /// positive values are used to test for boost values that are greater than or equal to the 'nBoostValue'.</remarks>
        [OperationContract(IsOneWay = false)]
        List<SimpleDatum> GetImagesFromIndex(long lQueryState, int nSrcId, int nStartIdx, int nQueryCount = int.MaxValue, string strFilterVal = null, int? nBoostVal = null, bool bBoostValIsExact = false, bool bAttemptDirectLoad = false);

        /// <summary>
        /// Returns the array of images in the image set, possibly filtered with the filtering parameters.
        /// </summary>
        /// <param name="lQueryState">Specifies a handle to the query state to use.</param>
        /// <param name="nSrcId">Specifies the data source ID.</param>
        /// <param name="dtStart">Specifies a starting time from which the query is to start within the set of images.</param>
        /// <param name="nQueryCount">Optionally, specifies a number of images to retrieve within the set (default = int.MaxValue).</param>
        /// <param name="strFilterVal">Optionally, specifies the filter value that the description must match (default = <i>null</i>, which ignores this parameter).</param>
        /// <param name="nBoostVal">Optionally, specifies the boost value that the boost must match (default = <i>null</i>, which ignores this parameter).</param>
        /// <param name="bBoostValIsExact">Optionally, specifies whether or the boost value (if specified) is to be used literally (exact = true), or as a minimum boost value.</param>
        /// <returns>The list of images is returned.</returns>
        /// <remarks>When using the 'nBoostValue' negative values are used to test the exact match of the boost value with the absolute value of the 'nBoostValue', ande
        /// positive values are used to test for boost values that are greater than or equal to the 'nBoostValue'.</remarks>
        [OperationContract(IsOneWay = false)]
        List<SimpleDatum> GetImagesFromTime(long lQueryState, int nSrcId, DateTime dtStart, int nQueryCount = int.MaxValue, string strFilterVal = null, int? nBoostVal = null, bool bBoostValIsExact = false);

        /// <summary>
        /// Query an image in a given data source.
        /// </summary>
        /// <param name="lQueryState">Specifies a handle to the query state to use.</param>
        /// <param name="nSrcId">Specifies the databse ID of the data source.</param>
        /// <param name="nIdx">Specifies the image index to query.  Note, the index is only used in non-random image queries.</param>
        /// <param name="labelSelectionOverride">Optionally, specifies the label selection method override.  The default = null, which directs the method to use the label selection method specified during Initialization.</param>
        /// <param name="imageSelectionOverride">Optionally, specifies the image selection method override.  The default = null, which directs the method to use the image selection method specified during Initialization.</param>
        /// <param name="nLabel">Optionally, specifies a label set to use for the image selection.  When specified only images of this label are returned using the image selection method.</param>
        /// <param name="bLoadDataCriteria">Specifies to load the data criteria data (default = false).</param>
        /// <param name="bLoadDebugData">Specifies to load the debug data (default = false).</param>
        /// <param name="bThrowExceptions">Optionally, specifies to throw exceptions on error (default = true).</param>
        /// <returns>The image SimpleDatum is returned.</returns>
        [OperationContract(IsOneWay = false)]
        SimpleDatum QueryImage(long lQueryState, int nSrcId, int nIdx, DB_LABEL_SELECTION_METHOD? labelSelectionOverride = null, DB_ITEM_SELECTION_METHOD? imageSelectionOverride = null, int? nLabel = null, bool bLoadDataCriteria = false, bool bLoadDebugData = false, bool bThrowExceptions = true);

        #endregion // Image Acquisition

        #region Boosts

        /// <summary>
        /// Reset all in-memory image boosts.
        /// </summary>
        /// <remarks>
        /// This does not impact the boost setting within the physical database.
        /// </remarks>
        /// <param name="nSrcId">Specifies the source ID of the data set to reset.</param>
        [OperationContract(IsOneWay = false)]
        void ResetAllBoosts(int nSrcId);

        #endregion // Boosts

        #region Load Limit Refresh

        /// <summary>
        /// Wait for the dataset refreshing to complete.
        /// </summary>
        /// <param name="nDsId">Specifies the dataset ID.</param>
        /// <param name="bTraining">Specifies to wait for the training data source to refresh.</param>
        /// <param name="bTesting">Specifies to wait for the testing data source to refresh.</param>
        /// <param name="nWait">Specifies the amount of time to wait in ms. (default = int.MaxValue).</param>
        /// <returns>If the data source(s) complete refreshing, <i>true</i> is returned, otherwise <i>false</i>.</returns>
        bool WaitForDatasetToRefresh(int nDsId, bool bTraining, bool bTesting, int nWait = int.MaxValue);

        /// <summary>
        /// Wait for the dataset refreshing to complete.
        /// </summary>
        /// <param name="strDs">Specifies the dataset name.</param>
        /// <param name="bTraining">Specifies to wait for the training data source to refresh.</param>
        /// <param name="bTesting">Specifies to wait for the testing data source to refresh.</param>
        /// <param name="nWait">Specifies the amount of time to wait in ms. (default = int.MaxValue).</param>
        /// <returns>If the data source(s) complete refreshing, <i>true</i> is returned, otherwise <i>false</i>.</returns>
        bool WaitForDatasetToRefresh(string strDs, bool bTraining, bool bTesting, int nWait = int.MaxValue);

        /// <summary>
        /// Returns true if the refresh operation running.
        /// </summary>
        /// <param name="nDsId">Specifies the dataset ID.</param>
        /// <param name="bTraining">Specifies to check the training data source for refresh.</param>
        /// <param name="bTesting">Specifies to check the testing data source for refresh.</param>
        /// <returns>If the refresh is running, true is returned, otherwise false.</returns>
        bool IsRefreshRunning(int nDsId, bool bTraining, bool bTesting);

        /// <summary>
        /// Returns true if the refresh operation running.
        /// </summary>
        /// <param name="strDs">Specifies the dataset name.</param>
        /// <param name="bTraining">Specifies to check the training data source for refresh.</param>
        /// <param name="bTesting">Specifies to check the testing data source for refresh.</param>
        /// <returns>If the refresh is running, true is returned, otherwise false.</returns>
        bool IsRefreshRunning(string strDs, bool bTraining, bool bTesting);

        /// <summary>
        /// Start a refresh on the dataset by replacing a specified percentage of the images with images from the physical database.
        /// </summary>
        /// <remarks>
        /// Note, this method is only valid when initialized with LoadLimit > 0.
        /// </remarks>
        /// <param name="strDs">Specifies the dataset name.</param>
        /// <param name="bTraining">Specifies the training data source to refresh.</param>
        /// <param name="bTesting">Specifies the testing data source to refresh.</param>
        /// <param name="dfReplacementPct">Optionally, specifies the replacement percentage to use (default = 0.25 for 25%).</param>
        /// <returns>On succes, true is returned, otherwise false is returned.</returns>
        bool StartRefresh(string strDs, bool bTraining, bool bTesting, double dfReplacementPct);

        /// <summary>
        /// Stop a refresh operation running on the dataset.
        /// </summary>
        /// <remarks>
        /// Note, this method is only valid when initialized with LoadLimit > 0.
        /// </remarks>
        /// <param name="strDs">Specifies the dataset name.</param>
        /// <param name="bTraining">Specifies the training data source to strop refreshing.</param>
        /// <param name="bTesting">Specifies the testing data source to stop refreshing.</param>
        /// <returns>On succes, true is returned, otherwise false is returned.</returns>
        bool StopRefresh(string strDs, bool bTraining, bool bTesting);

        /// <summary>
        /// Start a refresh on the dataset by replacing a specified percentage of the images with images from the physical database.
        /// </summary>
        /// <remarks>
        /// Note, this method is only valid when initialized with LoadLimit > 0.
        /// </remarks>
        /// <param name="nDsID">Specifies the dataset ID.</param>
        /// <param name="bTraining">Specifies the training data source to refresh.</param>
        /// <param name="bTesting">Specifies the testing data source to refresh.</param>
        /// <param name="dfReplacementPct">Optionally, specifies the replacement percentage to use (default = 0.25 for 25%).</param>
        /// <returns>On succes, true is returned, otherwise false is returned.</returns>
        bool StartRefresh(int nDsID, bool bTraining, bool bTesting, double dfReplacementPct);

        /// <summary>
        /// Stop a refresh operation running on the dataset.
        /// </summary>
        /// <remarks>
        /// Note, this method is only valid when initialized with LoadLimit > 0.
        /// </remarks>
        /// <param name="nDsID">Specifies the dataset ID.</param>
        /// <param name="bTraining">Specifies the training data source to strop refreshing.</param>
        /// <param name="bTesting">Specifies the testing data source to stop refreshing.</param>
        /// <returns>On succes, true is returned, otherwise false is returned.</returns>
        bool StopRefresh(int nDsID, bool bTraining, bool bTesting);

        /// <summary>
        /// Start the automatic refresh cycle to occur on specified period increments.
        /// </summary>
        /// <param name="strDs">Specifies the dataset name for which the automatic refresh cycle is to run.</param>
        /// <param name="bTraining">Specifies the training data source to start refreshing.</param>
        /// <param name="bTesting">Specifies the testing data source to start refreshing.</param>
        /// <param name="nPeriodInMs">Specifies the period in milliseconds over which the auto refresh cycle is to run.</param>
        /// <param name="dfReplacementPct">Specifies the percentage of replacement to use on each cycle.</param>
        /// <returns>If successfully started, true is returned, otherwise false.</returns>
        bool StartAutomaticRefreshSchedule(string strDs, bool bTraining, bool bTesting, int nPeriodInMs, double dfReplacementPct);

        /// <summary>
        /// Stop the automatic refresh schedule running on a dataset.
        /// </summary>
        /// <param name="strDs">Specifies the dataset name for which the automatic refresh cycle is to run.</param>
        /// <param name="bTraining">Specifies the training data source to stop refreshing.</param>
        /// <param name="bTesting">Specifies the testing data source to stop refreshing.</param>
        /// <returns>If successfully stopped, true is returned, otherwise false.</returns>
        bool StopAutomaticRefreshSchedule(string strDs, bool bTraining, bool bTesting);

        /// <summary>
        /// Returns whether or not a scheduled refresh is running and if so at what period and replacement percent.
        /// </summary>
        /// <param name="strDs">Specifies the dataset name for which the automatic refresh cycle is to run.</param>
        /// <param name="nPeriodInMs">Returns the period in milliseconds over which the auto refresh cycle is run.</param>
        /// <param name="dfReplacementPct">Returns the percentage of replacement to use on each cycle.</param>
        /// <param name="nTrainingRefreshCount">Returns the training refrsh count.</param>
        /// <param name="nTestingRefreshCount">Returns the testing refresh count.</param>
        /// <returns>If the refresh schedule is running, true is returned, otherwise false.</returns>
        bool GetScheduledAutoRefreshInformation(string strDs, out int nPeriodInMs, out double dfReplacementPct, out int nTrainingRefreshCount, out int nTestingRefreshCount);

        /// <summary>
        /// Start the automatic refresh cycle to occur on specified period increments.
        /// </summary>
        /// <param name="nDsID">Specifies the dataset ID for which the automatic refresh cycle is to run.</param>
        /// <param name="bTraining">Specifies the training data source to start refreshing.</param>
        /// <param name="bTesting">Specifies the testing data source to start refreshing.</param>
        /// <param name="nPeriodInMs">Specifies the period in milliseconds over which the auto refresh cycle is to run.</param>
        /// <param name="dfReplacementPct">Specifies the percentage of replacement to use on each cycle.</param>
        /// <returns>If successfully started, true is returned, otherwise false.</returns>
        bool StartAutomaticRefreshSchedule(int nDsID, bool bTraining, bool bTesting, int nPeriodInMs, double dfReplacementPct);

        /// <summary>
        /// Stop the automatic refresh schedule running on a dataset.
        /// </summary>
        /// <param name="nDsID">Specifies the dataset ID for which the automatic refresh cycle is to run.</param>
        /// <param name="bTraining">Specifies the training data source to stop refreshing.</param>
        /// <param name="bTesting">Specifies the testing data source to stop refreshing.</param>
        /// <returns>If successfully stopped, true is returned, otherwise false.</returns>
        bool StopAutomaticRefreshSchedule(int nDsID, bool bTraining, bool bTesting);

        /// <summary>
        /// Returns whether or not a scheduled refresh is running and if so at what period and replacement percent.
        /// </summary>
        /// <param name="nDsID">Specifies the dataset name for which the automatic refresh cycle is to run.</param>
        /// <param name="nPeriodInMs">Returns the period in milliseconds over which the auto refresh cycle is run.</param>
        /// <param name="dfReplacementPct">Returns the percentage of replacement to use on each cycle.</param>
        /// <param name="nTrainingRefreshCount">Returns the training refrsh count.</param>
        /// <param name="nTestingRefreshCount">Returns the testing refresh count.</param>
        /// <returns>If the refresh schedule is running, true is returned, otherwise false.</returns>
        bool GetScheduledAutoRefreshInformation(int nDsID, out int nPeriodInMs, out double dfReplacementPct, out int nTrainingRefreshCount, out int nTestingRefreshCount);

        #endregion

        #region Results

        /// <summary>
        /// Query all results for a given data source.
        /// </summary>
        /// <param name="strSource">Specifies the data source who's results are to be returned.</param>
        /// <param name="bRequireExtraData">Specifies whether or not the Extra 'target' data is required or not.</param>
        /// <param name="nMax">Optionally, specifies the maximum number of items to load.</param>
        /// <returns>Each result is returned in a SimpleResult object.</returns>
        List<SimpleResult> GetAllResults(string strSource, bool bRequireExtraData, int nMax = -1);

        #endregion
    }

    /// <summary>
    /// Specifies the accuracy test interface implemented by some of the accuracy layers.
    /// </summary>
    public interface IXAccuracyTest
    {
        /// <summary>
        /// Reset the testing accuracy - must be called at least once.
        /// </summary>
        void ResetTesting();
        /// <summary>
        /// Add the testing predicted and ground truth results.
        /// </summary>
        /// <param name="fPredicted">Specifies the predicted value.</param>
        /// <param name="fGroundTruth">Specifies the ground truth value.</param>
        /// <param name="fSecondaryGroundTruth">Optionally, specifies a secondary ground truth value.</param>
        /// <param name="bNormalizeGt">Optionally, specifies to normalize the ground truth (default = true).</param>
        void AddTesting(float fPredicted, float fGroundTruth, float? fSecondaryGroundTruth = null, bool bNormalizeGt = true);
        /// <summary>
        /// Calculate and return the accuracy.
        /// </summary>
        /// <param name="bGetDetails">Specifies to get the details.</param>
        /// <param name="strDetails">Specifies details on the testing.</param>
        /// <returns>The accuracy value is returned as a percentage.</returns>
        double CalculateTestingAccuracy(bool bGetDetails, out string strDetails);
    }

    /// <summary>
    /// Specifies the normalize interface implemented by some normalizing layers.
    /// </summary>
    public interface IXNormalize<T>
    {
        /// <summary>
        /// Specifies to normalize the value.
        /// </summary>
        /// <param name="fVal">Specifies the unormalized value.</param>
        /// <returns>The normalized value is returned.</returns>
        T Normalize(T fVal);
        /// <summary>
        /// Specifies to unnormalize a normalized value.
        /// </summary>
        /// <param name="fVal">Specifies the normalized value.</param>
        /// <returns>The unormalized value is returned.</returns>
        T Unnormalize(T fVal);
    }

#pragma warning disable 1591

    [DataContract]
    public class DatabaseErrorData /** @private */
    {
        [DataMember]
        public bool Result { get; set; }
        [DataMember]
        public string ErrorMessage { get; set; }
        [DataMember]
        public string ErrorDetails { get; set; }
    }

#pragma warning restore 1591
}
