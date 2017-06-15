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
    /// Defines how to laod the images into the image database.
    /// </summary>
    public enum IMAGEDB_LOAD_METHOD
    {
        /// <summary>
        /// Load all of the images into memory.
        /// </summary>
        LOAD_ALL,
        /// <summary>
        /// Load the images as they are queried.
        /// </summary>
        LOAD_ON_DEMAND
    }

    /// <summary>
    /// Defines the snapshot update method.
    /// </summary>
    public enum SNAPSHOT_UPDATE_METHOD
    {
        /// <summary>
        /// Update the snapshot when the accuracy increases.
        /// </summary>
        FAVOR_ACCURACY,
        /// <summary>
        /// Update the snapshot when the error decreases.
        /// </summary>
        FAVOR_ERROR,
        /// <summary>
        /// Update the snapshot when the accuracy increases or the error decreases.
        /// </summary>
        FAVOR_BOTH
    }

    /// <summary>
    /// Defines the snapshot load method.
    /// </summary>
    public enum SNAPSHOT_LOAD_METHOD
    {
        /// <summary>
        /// Load the last snapshot taken.
        /// </summary>
        LAST,
        /// <summary>
        /// Load the snapshot with the best accuracy (which may not be the last).
        /// </summary>
        BEST_ACCURACY,
        /// <summary>
        /// Load the snapshot with the best error (which may not be the last).
        /// </summary>
        BEST_ERROR
    }
}
