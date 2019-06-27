using MyCaffe.basecode;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.trainers.common
{
    /// <summary>
    /// Specifies the type of memory collection to use.
    /// </summary>
    public enum MEMTYPE
    {
        /// <summary>
        /// Specifies the randomly sampled memory collection.
        /// </summary>
        RANDOM,
        /// <summary>
        /// Specifies the prioritized sampled memory collection.
        /// </summary>
        PRIORITY,
        /// <summary>
        /// Specifies a memory collection loaded from file (used during debugging).
        /// </summary>
        LOADING,
        /// <summary>
        /// Specifies a randomly sampled memory collection that saves to file (used during debugging).
        /// </summary>
        SAVING
    }

    /// <summary>
    /// The IMemoryCollection interface is implemented by all memory collection types.
    /// </summary>
    public interface IMemoryCollection
    {
        /// <summary>
        /// Add a new item to the memory collection.
        /// </summary>
        /// <param name="m">Specifies the memory item to add.</param>
        void Add(MemoryItem m);
        /// <summary>
        /// Retrieve a set of samples from the collection.
        /// </summary>
        /// <param name="random">Specifies the random number generator.</param>
        /// <param name="nCount">Specifies the number of samples to retrieve.</param>
        /// <param name="dfBeta">Specifies a value used by the prioritized memory collection.</param>
        /// <returns></returns>
        MemoryCollection GetSamples(CryptoRandom random, int nCount, double dfBeta);
        /// <summary>
        /// Updates the memory collection - currently only used by the Prioritized memory collection to update its priorities.
        /// </summary>
        /// <param name="rgSamples">Specifies the samples with updated priorities (if used).</param>
        void Update(MemoryCollection rgSamples);
        /// <summary>
        /// Returns the number of items in the memory collection.
        /// </summary>
        int Count { get; }
        /// <summary>
        /// Performs final clean-up tasks.
        /// </summary>
        void CleanUp();
    }
}
