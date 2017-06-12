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
}
