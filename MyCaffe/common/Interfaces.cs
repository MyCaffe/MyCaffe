using MyCaffe.basecode;
using MyCaffe.basecode.descriptors;
using MyCaffe.db.image;
using MyCaffe.param;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MyCaffe.common
{
    /// <summary>
    /// Delegate used to set the OnSetworkspace event.
    /// </summary>
    /// <param name="sender">Specifies the sender.</param>
    /// <param name="e">Specifies the arguments.</param>
    public delegate void onSetWorkspace(object sender, WorkspaceArgs e);
    /// <summary>
    /// Delegate used to set the OnGetworkspace event.
    /// </summary>
    /// <param name="sender">Specifies the sender.</param>
    /// <param name="e">Specifies the arguments.</param>
    public delegate void onGetWorkspace(object sender, WorkspaceArgs e);

    /// <summary>
    /// Defines the type of weight to target in re-initializations.
    /// </summary>
    [Serializable]
    public enum WEIGHT_TARGET
    {
        /// <summary>
        /// No weights are targeted.
        /// </summary>
        NONE,
        /// <summary>
        /// Generic weights are targeted.
        /// </summary>
        WEIGHTS,
        /// <summary>
        /// Bias weights are targeted.
        /// </summary>
        BIAS,
        /// <summary>
        /// Both weights and bias are targeted.
        /// </summary>
        BOTH
    }

    /// <summary>
    /// Defines the tpe of data held by a given Blob.
    /// </summary>
    [Serializable]
    public enum BLOB_TYPE
    {
        /// <summary>
        /// The blob is an unknown type.
        /// </summary>
        UNKNOWN,
        /// <summary>
        /// The Blob holds Data.
        /// </summary>
        DATA,
        /// <summary>
        /// The Blob holds an inner product weight.
        /// </summary>
        IP_WEIGHT,
        /// <summary>
        /// The Blob holds a general weight.
        /// </summary>
        WEIGHT,
        /// <summary>
        /// The Blob holds Loss Data.
        /// </summary>
        LOSS,
        /// <summary>
        /// The Blob holds Accuracy Data.
        /// </summary>
        ACCURACY,
        /// <summary>
        /// The blob holds Clip data.
        /// </summary>
        CLIP,
        /// <summary>
        /// The blob holds multi-boundingbox data.
        /// </summary>
        /// <remarks>
        /// The multi-box data ordering is as follows:
        /// [0] index of num.
        /// [1] label
        /// [2] score
        /// [3] bbox xmin
        /// [4] bbox ymin
        /// [5] bbox xmax
        /// [6] bbox ymax
        /// 
        /// continues for each of the top 'n' bboxes output.
        /// </remarks>
        MULTIBBOX,
        /// <summary>
        /// The blob is an internal blob used within the layer.
        /// </summary>
        INTERNAL
    }

    /// <summary>
    /// Defines the training stepping method (if any).
    /// </summary>
    [Serializable]
    [DataContract]
    public enum TRAIN_STEP
    {
        /// <summary>
        /// No stepping.
        /// </summary>
        [EnumMember]
        NONE = 0x0000,
        /// <summary>
        /// Step only in the forward direction.
        /// </summary>
        [EnumMember]
        FORWARD = 0x0001,
        /// <summary>
        /// Step only in the backward direction.
        /// </summary>
        [EnumMember]
        BACKWARD = 0x0002,
        /// <summary>
        /// Step in both directions (one forward and one backward).
        /// </summary>
        [EnumMember]
        BOTH = 0x0003
    }

    /// <summary>
    /// The IXDebugData interface is implemented by the DebugLayer to give access to the debug information managed by the layer.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public interface IXDebugData<T>
    {
        /// <summary>
        /// Returns the collection of data blobs.
        /// </summary>
        BlobCollection<T> data { get; }
        /// <summary>
        /// Returns the collection of label blobs.
        /// </summary>
        BlobCollection<T> labels { get; }
        /// <summary>
        /// Returns the name of the data set.
        /// </summary>
        string name { get; }
        /// <summary>
        /// Returns the handle to the kernel within the low-level Cuda Dnn DLL that where the data memory resides.
        /// </summary>
        long kernel_handle { get; }
        /// <summary>
        /// Returns the number of data items loaded into the collection.
        /// </summary>
        int load_count { get; }
    }

    /// <summary>
    /// The IXPersist interface is used by the CaffeControl to load and save weights.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public interface IXPersist<T>
    {
        /// <summary>
        /// Save the weights to a byte array.
        /// </summary>
        /// <remarks>
        /// NOTE: In order to maintain compatibility with the C++ %Caffe, extra MyCaffe features may be added to the <i>end</i> of the weight file.  After saving weights in the format
        /// used by the C++ %Caffe, MyCaffe writes the bytes "mycaffe.ai".  All information after these bytes are specific to MyCaffe and allow for loading weights for models by Blob name and shape
        /// and loosen the C++ %Caffe requirement that the 'number' of blobs match.  Adding this functionality allows for training model, changing the model structure, and then re-using the trained
        /// weights in the new model.  
        /// </remarks>
        /// <param name="colBlobs">Specifies the Blobs to save with the weights.</param>
        /// <param name="bSaveDiffs">Optionally, specifies to save the diff values.</param>
        /// <returns>The byte array containing the weights is returned.</returns>
        byte[] SaveWeights(BlobCollection<T> colBlobs, bool bSaveDiffs = false);

        /// <summary>
        /// Loads new weights into a BlobCollection
        /// </summary>
        /// <remarks>
        /// NOTE: In order to maintain compatibility with the C++ %Caffe, extra MyCaffe features may be added to the <i>end</i> of the weight file.  After saving weights (see SaveWeights) in the format
        /// used by the C++ %Caffe, MyCaffe writes the bytes "mycaffe.ai".  All information after these bytes are specific to MyCaffe and allow for loading weights for models by Blob name and shape
        /// and loosen the C++ %Caffe requirement that the 'number' of blobs match.  Adding this functionality allows for training model, changing the model structure, and then re-using the trained
        /// weights in the new model.  
        /// </remarks>
        /// <param name="rgWeights">Specifies the weights themselves.</param>
        /// <param name="rgExpectedShapes">Specifies a list of expected shapes for each Blob where the weights are to be loaded.</param>
        /// <param name="colBlobs">Specifies the Blobs to load with the weights.</param>
        /// <param name="bSizeToFit">Optionally, specifies wether or not the weights should be re-sized.  Note: resizing can render the weights useless, especially in deeper, layers.</param>
        /// <param name="bLoadedDiffs">Returns whether or not the diffs were loaded.</param>
        /// <param name="inputWtInfo">Optionally, specifies the weight info describing the input weight blobs to import by name.  Note when used the number of blobs must match the number of <i>targetWtInfo</i> blobs.  Otherwise, when <i>null</i> this parameter is ignored.</param>
        /// <param name="targetWtInfo">Optionally, specifies the weight info describing the target weight blobs to import by name.  Note when used the number of blobs must match the number of <i>inputWtInfo</i> blobs.  Otherwise, when <i>null</i> this parameter is ignored.</param>
        /// <param name="strSkipBlobType">Optionally, specifies a blob type where weights are NOT loaded.  See Blob.BLOB_TYPE for the types of Blobs.</param>
        /// <returns>The collection of Blobs with newly loaded weights is returned.</returns>
        BlobCollection<T> LoadWeights(byte[] rgWeights, List<string> rgExpectedShapes, BlobCollection<T> colBlobs, bool bSizeToFit, out bool bLoadedDiffs, List<string> inputWtInfo = null, List<string> targetWtInfo = null, string strSkipBlobType = null);

        /// <summary>
        /// Save the solver state to a byte array.
        /// </summary>
        /// <param name="state">Specifies the solver state to save.</param>
        /// <param name="type">Specifies the solver type.</param>
        /// <returns>A byte array containing the solver state is returned.</returns>
        byte[] SaveSolverState(SolverState state, SolverParameter.SolverType type = SolverParameter.SolverType.SGD);

        /// <summary>
        /// Load the solver state from a byte array.
        /// </summary>
        /// <param name="rgState">Specifies the byte array containing the solver state.</param>
        /// <param name="type">Specifies the solver type.</param>
        /// <returns>The SolverState loaded is returned.</returns>
        SolverState LoadSolverState(byte[] rgState, SolverParameter.SolverType type = SolverParameter.SolverType.SGD);

        /// <summary>
        /// Returns the weight information describing the weights containined within the weight bytes.
        /// </summary>
        /// <param name="rgWeights">Specifies the bytes containing the weights.</param>
        /// <returns>The weight information is returned.</returns>
        WeightInfo<T> LoadWeightInfo(byte[] rgWeights);

        /// <summary>
        /// Returns the weight information describing the weights containined within the Blob collection.
        /// </summary>
        /// <param name="colBlobs">Specifies the Blob collection containing the weights.</param>
        /// <returns>The weight information is returned.</returns>
        WeightInfo<T> LoadWeightInfo(BlobCollection<T> colBlobs);
    }

    /// <summary>
    /// The IXMyCaffeState interface contains functions related to the MyCaffeComponent state.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public interface IXMyCaffeState<T>
    {
        /// <summary>
        /// Sets the root solver's onTest event function.
        /// </summary>
        /// <param name="onTest">Specifies the event handler called when testing.</param>
        void SetOnTestOverride(EventHandler<TestArgs> onTest);

        /// <summary>
        /// Add a cancel override.
        /// </summary>
        /// <param name="strEvtCancel">Specifies the name of the new cancel event to add.</param>
        void AddCancelOverrideByName(string strEvtCancel);

        /// <summary>
        /// Add a cancel override.
        /// </summary>
        /// <param name="evtCancel">Specifies the cancel event to add.</param>
        void AddCancelOverride(CancelEvent evtCancel);

        /// <summary>
        /// Remove a cancel override.
        /// </summary>
        /// <param name="strEvtCancel">Specifies the name of the new cancel event to remove.</param>
        void RemoveCancelOverrideByName(string strEvtCancel);

        /// <summary>
        /// Remove a cancel override.
        /// </summary>
        /// <param name="evtCancel">Specifies the cancel event to remove.</param>
        void RemoveCancelOverride(CancelEvent evtCancel);

        /// <summary>
        /// Enable/disable blob debugging.
        /// </summary>
        /// <remarks>
        /// Note, when enabled, training will dramatically slow down.
        /// </remarks>
        bool EnableBlobDebugging { get; set; }
        /// <summary>
        /// Enable/disable break training after first detecting a NaN.
        /// </summary>
        /// <remarks>
        /// This option requires that EnableBlobDebugging == <i>true</i>.
        /// </remarks>
        bool EnableBreakOnFirstNaN { get; set; }
        /// <summary>
        /// When enabled (requires EnableBlobDebugging = <i>true</i>), the detailed Nan (and Infinity) detection is perofmed on each blob when training Net.
        /// </summary>
        bool EnableDetailedNanDetection { get; set; }
        /// <summary>
        /// Enable/disable single step training.
        /// </summary>
        /// <remarks>
        /// This option requires that EnableBlobDebugging == true.
        /// </remarks>
        bool EnableSingleStep { get; set; }
        /// <summary>
        /// Enable/disable layer debugging which causes each layer to check for NAN/INF on each forward/backward pass and throw an exception when found.
        /// </summary>
        /// <remarks>
        /// This option dramatically slows down training and is only recommended during debugging.
        /// </remarks>
        bool EnableLayerDebugging { get; set; }

        /// <summary>
        /// Returns the persist used to load and save weights.
        /// </summary>
        IXPersist<T> Persist { get; }
        /// <summary>
        /// Returns the MyCaffeImageDatabase used.
        /// </summary>
        IXImageDatabaseBase ImageDatabase { get; }

        /// <summary>
        /// Returns the CancelEvent used.
        /// </summary>
        CancelEvent CancelEvent { get; }
        /// <summary>
        /// Returns a list of Active GPU's used by the control.
        /// </summary>
        List<int> ActiveGpus { get; }
        /// <summary>
        /// Returns a string describing the active label counts observed during training.
        /// </summary>
        /// <remarks>
        /// This string can help diagnos label balancing issue.
        /// </remarks>
        string ActiveLabelCounts { get; }
        /// <summary>
        /// Returns a string describing the label query hit percentages observed during training.
        /// </summary>
        /// <remarks>
        /// This string can help diagnose label balancing issue.
        /// </remarks>
        string LabelQueryHitPercents { get; }
        /// <summary>
        /// Returns a string describing the label query epochs observed during training.
        /// </summary>
        /// <remarks>
        /// This string can help diagnose label balancing issue.
        /// </remarks>
        string LabelQueryEpochs { get; }
        /// <summary>
        /// Returns the name of the current device used.
        /// </summary>
        string CurrentDevice { get; }
        /// <summary>
        /// Returns the name of the currently loaded project.
        /// </summary>
        ProjectEx CurrentProject { get; }
        /// <summary>
        /// Returns the current iteration.
        /// </summary>
        int CurrentIteration { get; }
        /// <summary>
        /// Returns the maximum iteration.
        /// </summary>
        int MaximumIteration { get; }
        /// <summary>
        /// Returns the total number of devices installed on this computer.
        /// </summary>
        /// <returns></returns>
        int GetDeviceCount();
        /// <summary>
        /// Returns the device name of a given device ID.
        /// </summary>
        /// <param name="nDeviceID">Specifies the device ID.</param>
        /// <returns></returns>
        string GetDeviceName(int nDeviceID);
        /// <summary>
        /// Re-initializes each of the specified layers by re-running the filler (if any) specified by the layer.  
        /// When the 'rgstr' parameter is <i>null</i> or otherwise empty, the blobs of all layers are re-initialized. 
        /// </summary>
        /// <param name="target">Specifies the weights to target (e.g. weights, bias or both).</param>
        /// <param name="rgstrLayers">Specifies the layers to reinitialize, when <i>null</i> or empty, all layers are re-initialized</param>
        /// <returns>If a layer is specified and found, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        bool ReInitializeParameters(WEIGHT_TARGET target, params string[] rgstrLayers);
        /// <summary>
        /// VerifyCompute compares the current compute of the current device (or device specified) against the required compute of the current CudaDnnDLL.dll used.
        /// </summary>
        /// <param name="strExtra">Optionally, specifies extra information for the exception if one is thrown.</param>
        /// <param name="nDeviceID">Optionally, specifies a specific device ID to check, otherwise uses the current device used (default = -1, which uses the current device).</param>
        /// <param name="bThrowException">Optionally, specifies whether or not to throw an exception on a compute mis-match (default = true).</param>
        /// <returns>If the device's compute is >= to the required compute fo the CudaDnnDll.dll used, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        bool VerifyCompute(string strExtra = null, int nDeviceID = -1, bool bThrowException = true);
    }

    /// <summary>
    /// The IXMyCaffe interface contains functions used to perform MyCaffe operations that work with the MyCaffeImageDatabase.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public interface IXMyCaffe<T>
    {
        /// <summary>
        /// Load a project and optionally the MyCaffeImageDatabase.
        /// </summary>
        /// <remarks>
        /// This load function uses the MyCaffeImageDatabase.
        /// </remarks>
        /// <param name="phase">Specifies the Phase for which the load should focus.</param>
        /// <param name="p">Specifies the Project to load.</param>
        /// <param name="labelSelectionOverride">Optionally, specifies the label selection override (overides the label selection in SettingsCaffe).  The label selection dictates how the label sets are selected.</param>
        /// <param name="imageSelectionOverride">Optionally, specifies the image selection override (overides the image selection in SettingsCaffe).  The image selection dictates how the images are selected from each label set.</param>
        /// <param name="bResetFirst">Optionally, resets the device before loading.  IMPORTANT: this functionality is only recommendned during testing, for resetting the device will throw off all other users of the device.</param>
        /// <param name="imgdb">Optionally, specifies the MyCaffeImageDatabase to use.  When <i>null</i>, an instance if the MyCaffeImageDatabase is created internally.</param>
        /// <param name="bUseImageDb">Optionally, specifies whehter or not to use the image database (default = true).</param>
        /// <param name="bCreateRunNet">Optionally, specifies whether or not to create the Run net.</param>
        /// <param name="strStage">Optionally, specifies the stage under which to load the model.</param>
        /// <param name="bEnableMemTrace">Optionally, specifies to enable the memory tracing (only available in debug builds).</param>
        /// <returns>If the project is loaded the function returns <i>true</i>, otherwise <i>false</i> is returned.</returns>
        bool Load(Phase phase, ProjectEx p, IMGDB_LABEL_SELECTION_METHOD? labelSelectionOverride = null, IMGDB_IMAGE_SELECTION_METHOD? imageSelectionOverride = null, bool bResetFirst = false, IXImageDatabaseBase imgdb = null, bool bUseImageDb = true, bool bCreateRunNet = true, string strStage = null, bool bEnableMemTrace = false);
        /// <summary>
        /// Load a project and optionally the MyCaffeImageDatabase.
        /// </summary>
        /// <remarks>
        /// This load function uses the MyCaffeImageDatabase.
        /// </remarks>
        /// <param name="phase">Specifies the Phase for which the load should focus.</param>
        /// <param name="strSolver">Specifies the solver descriptor.</param>
        /// <param name="strModel">Specifies the model desciptor.</param>
        /// <param name="rgWeights">Optionally, specifies the weights to load, or <i>null</i> to ignore.</param>
        /// <param name="labelSelectionOverride">Optionally, specifies the label selection override (overides the label selection in SettingsCaffe).  The label selection dictates how the label sets are selected.</param>
        /// <param name="imageSelectionOverride">Optionally, specifies the image selection override (overides the image selection in SettingsCaffe).  The image selection dictates how the images are selected from each label set.</param>
        /// <param name="bResetFirst">Optionally, resets the device before loading.  IMPORTANT: this functionality is only recommendned during testing, for resetting the device will throw off all other users of the device.</param>
        /// <param name="imgdb">Optionally, specifies the MyCaffeImageDatabase to use.  When <i>null</i>, an instance if the MyCaffeImageDatabase is created internally.</param>
        /// <param name="bUseImageDb">Optionally, specifies whehter or not to use the image database (default = true).</param>
        /// <param name="bCreateRunNet">Optionally, specifies whether or not to create the Run net.</param>
        /// <param name="strStage">Optionally, specifies the stage under which to load the model.</param>
        /// <param name="bEnableMemTrace">Optionally, specifies to enable the memory tracing (only available in debug builds).</param>
        /// <returns>If the project is loaded the function returns <i>true</i>, otherwise <i>false</i> is returned.</returns>
        bool Load(Phase phase, string strSolver, string strModel, byte[] rgWeights, IMGDB_LABEL_SELECTION_METHOD? labelSelectionOverride = null, IMGDB_IMAGE_SELECTION_METHOD? imageSelectionOverride = null, bool bResetFirst = false, IXImageDatabaseBase imgdb = null, bool bUseImageDb = true, bool bCreateRunNet = true, string strStage = null, bool bEnableMemTrace = false);
        /// <summary>
        /// Unload the currently loaded project.
        /// </summary>
        /// <param name="bUnloadImageDb">Optionally, specifies whether or not to unload the image database. The default = <i>true</i>.</param>
        /// <param name="bIgnoreExceptions">Optionally, specifies to ignore exceptions that occur (default = <i>false</i>).</param>
        void Unload(bool bUnloadImageDb = true, bool bIgnoreExceptions = false);
        /// <summary>
        /// Train the network a set number of iterations and allow for single stepping.
        /// </summary>
        /// <param name="nIterationOverride">Optionally, specifies number of iterations to run that override the iterations specified in the solver desctiptor.</param>
        /// <param name="nTrainingTimeLimitInMinutes">Optionally, specifies a maximum number of minutes to train.  When set to 0, this parameter is ignored and no time limit is imposed.</param>
        /// <param name="step">Optionally, specifies whether or not to single step the training on the forward pass, backward pass or both.  The default is <i>TRAIN_STEP.NONE</i> which runs the training to the maximum number of iterations specified.</param>
        /// <param name="dfLearningRateOverride">Optionally, specifies a learning rate override (default = 0 which ignores this parameter)</param>
        /// <param name="bReset">Optionally, reset the iterations to zero.</param>
        void Train(int nIterationOverride = -1, int nTrainingTimeLimitInMinutes = 0, TRAIN_STEP step = TRAIN_STEP.NONE, double dfLearningRateOverride = 0, bool bReset = false);
        /// <summary>
        /// Test the network a given number of iterations.
        /// </summary>
        /// <param name="nIterationOverride">Optionally, specifies number of iterations to run that override the iterations specified in the solver desctiptor.</param>
        /// <returns>The accuracy value from the test is returned.</returns>
        double Test(int nIterationOverride = -1);
        /// <summary>
        /// Test on a number of images by selecting random images from the database, running them through the Run network, and then comparing the results with the 
        /// expected results.
        /// </summary>
        /// <param name="nCount">Specifies the number of cycles to run.</param>
        /// <param name="bOnTrainingSet">Specifies on whether to select images from the training set, or when <i>false</i> the testing set of data.</param>
        /// <param name="bOnTargetSet">Optionally, specifies to test on the target dataset (if exists) as opposed to the source dataset.</param>
        /// <param name="imgSelMethod">Optionally, specifies the image selection method (default = RANDOM).</param>
        /// <param name="nImageStartIdx">Optionally, specifies the image start index (default = 0).</param>
        /// <returns>The list of SimpleDatum and their ResultCollections (after running the model on each) is returned.</returns>
        List<Tuple<SimpleDatum, ResultCollection>> TestMany(int nCount, bool bOnTrainingSet, bool bOnTargetSet = false, IMGDB_IMAGE_SELECTION_METHOD imgSelMethod = IMGDB_IMAGE_SELECTION_METHOD.RANDOM, int nImageStartIdx = 0);
        /// <summary>
        /// Run on a given image in the MyCaffeImageDatabase based on its image index.
        /// </summary>
        /// <param name="nImageIdx">Specifies the image index.</param>
        /// <returns>The result of the run is returned.</returns>
        ResultCollection Run(int nImageIdx);
        /// <summary>
        /// Run on a set of images in the MyCaffeImageDatabase based on their image indexes.
        /// </summary>
        /// <param name="rgImageIdx">Specifies a list of image indexes.</param>
        /// <returns>A list of results from the run is returned - one result per image.</returns>
        List<ResultCollection> Run(List<int> rgImageIdx);
        /// <summary>
        /// Run on a given Datum. 
        /// </summary>
        /// <param name="d">Specifies the Datum to run.</param>
        /// <param name="bSort">Optionally, specifies whether or not to sor the results.</param>
        /// <param name="bUseSolverNet">Optionally, specifies whether or not to use the training net vs. the run net.</param>
        /// <returns>The results of the run are returned.</returns>
        ResultCollection Run(SimpleDatum d, bool bSort = true, bool bUseSolverNet = false);
        /// <summary>
        /// Retrieves a random image from either the training or test set depending on the Phase specified.
        /// </summary>
        /// <param name="phase">Specifies whether to select images from the training set or testing set.</param>
        /// <param name="nLabel">Returns the expected label for the image.</param>
        /// <param name="strLabel">Returns the expected label name for the image.</param>
        /// <returns>The image queried is returned.</returns>
        Bitmap GetTestImage(Phase phase, out int nLabel, out string strLabel);
        /// <summary>
        /// Retrieves a random image from either the training or test set depending on the Phase specified.
        /// </summary>
        /// <param name="phase">Specifies whether to select images from the training set or testing set.</param>
        /// <param name="nLabel">Returns the expected label for the image.</param>
        /// <returns>The image queried is returned.</returns>
        Bitmap GetTestImage(Phase phase, int nLabel);
        /// <summary>
        /// Returns the image mean used by the solver network used during training.
        /// </summary>
        /// <returns>The image mean is returned as a SimpleDatum.</returns>
        SimpleDatum GetImageMean();
        /// <summary>
        /// Returns the current dataset used when training and testing.
        /// </summary>
        /// <returns>The DatasetDescriptor is returned.</returns>
        DatasetDescriptor GetDataset();
        /// <summary>
        /// Retrieves the weights of the training network.
        /// </summary>
        /// <returns>The weights are returned.</returns>
        byte[] GetWeights();
        /// <summary>
        /// Loads the weights from the training net into the Net used for running.
        /// </summary>
        void UpdateRunWeights();
        /// <summary>
        /// Loads the training Net with new weights.
        /// </summary>
        /// <param name="rgWeights">Specifies the weights to load.</param>
        void UpdateWeights(byte[] rgWeights);
        /// <summary>
        /// Returns the license text for MyCaffe.
        /// </summary>
        /// <param name="strOtherLicenses">Specifies other licenses to append to the license text.</param>
        /// <returns></returns>
        string GetLicenseText(string strOtherLicenses);
    }

    /// <summary>
    /// The IXMyCaffeNoDb interface contains functions used to perform MyCaffe operations that run in a light-weight manner without the MyCaffeImageDatabase.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public interface IXMyCaffeNoDb<T>
    {
        /// <summary>
        /// The LoadToRun method loads the MyCaffeControl for running only (e.g. deployment).
        /// </summary>
        /// <remarks>
        /// This method does not use the MyCaffeImageDatabase.
        /// </remarks>
        /// <param name="strModel">Specifies the model description to load.</param>
        /// <param name="rgWeights">Specifies the trained weights to load.</param>
        /// <param name="shape">Specifies the expected shape to run on.</param>
        /// <param name="sdMean">Optionally, specifies the simple datum mean to subtract from input images that are run.</param>
        /// <param name="bForceBackward">Optionally, enables the force backward.</param>
        /// <param name="transParam">Optionally, specifies the TransformationParameter to use.  When using a 'deployment' model that has no data layers, you should supply a transformation parameter
        /// that matches the transformation used during training.</param>
        void LoadToRun(string strModel, byte[] rgWeights, BlobShape shape, SimpleDatum sdMean = null, TransformationParameter transParam = null, bool bForceBackward = false);
        /// <summary>
        /// Create a data blob from a SimpleDatum by transforming the data and placing the results in the blob returned.
        /// </summary>
        /// <param name="d">Specifies the datum to load into the blob.</param>
        /// <param name="blob">Optionally, specifies a blob to use instead of creating a new one.</param>
        /// <returns>The data blob containing the transformed data is returned.</returns>
        Blob<T> CreateDataBlob(SimpleDatum d, Blob<T> blob = null);
        /// <summary>
        /// Run on a given bitmap image.
        /// </summary>
        /// <remarks>
        /// This method does not use the MyCaffeImageDatabase.
        /// </remarks>
        /// <param name="img">Specifies the input image.</param>
        /// <param name="bSort">Specifies whether or not to sort the results.</param>
        /// <returns>The results of the run are returned.</returns>
        ResultCollection Run(Bitmap img, bool bSort = true);
        /// <summary>
        /// Run on a given Datum. 
        /// </summary>
        /// <param name="d">Specifies the Datum to run.</param>
        /// <param name="bSort">Optionally, specifies whether or not to sort the results.</param>
        /// <returns>The results of the run are returned.</returns>
        ResultCollection Run(SimpleDatum d, bool bSort = true);
    }

    /// <summary>
    /// The IXMyCaffeExtension interface allows for easy extension management of the low-level software that interacts directly with CUDA.
    /// </summary>
    /// <typeparam name="T">Specifies the base type of <i>float</i> or <i>double</i>.</typeparam>
    public interface IXMyCaffeExtension<T>
    {
        /// <summary>
        /// Create and load a new extension DLL.
        /// </summary>
        /// <param name="strExtensionDLLPath">Specifies the path to the extension DLL.</param>
        /// <returns>The handle to the extension is returned.</returns>
        long CreateExtension(string strExtensionDLLPath);
        /// <summary>
        /// Free an existing extension and unload it.
        /// </summary>
        /// <param name="hExtension">Specifies the handle to the extension to free.</param>
        void FreeExtension(long hExtension);
        /// <summary>
        /// Run a function on an existing extension.
        /// </summary>
        /// <param name="hExtension">Specifies the extension.</param>
        /// <param name="lfnIdx">Specifies the function to run on the extension.</param>
        /// <param name="rgParam">Specifies the parameters.</param>
        /// <returns>The return values of the function are returned.</returns>
        T[] RunExtension(long hExtension, long lfnIdx,  T[] rgParam);
        /// <summary>
        /// Run a function on an existing extension using the <i>double</i> base type.
        /// </summary>
        /// <param name="hExtension">Specifies the extension.</param>
        /// <param name="lfnIdx">Specifies the function to run on the extension.</param>
        /// <param name="rgParam">Specifies the parameters.</param>
        /// <returns>The return values of the function are returned.</returns>
        double[] RunExtensionD(long hExtension, long lfnIdx, double[] rgParam);
        /// <summary>
        /// Run a function on an existing extension using the <i>float</i> base type.
        /// </summary>
        /// <param name="hExtension">Specifies the extension.</param>
        /// <param name="lfnIdx">Specifies the function to run on the extension.</param>
        /// <param name="rgParam">Specifies the parameters.</param>
        /// <returns>The return values of the function are returned.</returns>
        float[] RunExtensionF(long hExtension, long lfnIdx, float[] rgParam);
    }
}
