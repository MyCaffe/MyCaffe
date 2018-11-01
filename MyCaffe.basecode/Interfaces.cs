using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MyCaffe.basecode
{
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
    /// Defines how to laod the images into the image database.
    /// </summary>
    public enum IMAGEDB_LOAD_METHOD
    {
        /// <summary>
        /// Load the images as they are queried - this option cahces images into memory as needed, training speeds are slower up until all images are loaded into memory.
        /// </summary>
        LOAD_ON_DEMAND,
        /// <summary>
        /// Load all of the images into memory - this option provides the highest training speeds, but can use a lot of memory and takes time to load.
        /// </summary>
        LOAD_ALL,
        /// <summary>
        /// Load the images from an external source such as a Windows Service - this option provides the best balance of speed and short load times for once loaded all applications share the in-memory data.
        /// </summary>
        LOAD_EXTERNAL
    }

    /// <summary>
    /// Defines the snapshot weight update method.
    /// </summary>
    public enum SNAPSHOT_WEIGHT_UPDATE_METHOD
    {
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
        FAVOR_BOTH
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
    /// Defines the ITest interface used by the Test module to return its known failures.
    /// </summary>
    public interface ITestKnownFailures
    {
        /// <summary>
        /// Get the known failures of the test module.
        /// </summary>
        List<Tuple<string, string, string>> KnownFailures { get; }
    }
}
