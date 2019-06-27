using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.trainers.common
{
    /// <summary>
    /// The MemoryCollectionFactory is used to create various memory collection types.
    /// </summary>
    public class MemoryCollectionFactory
    {
        /// <summary>
        /// CreateMemory creates the memory collection type based on the MEMTYPE parameter.
        /// </summary>
        /// <param name="type">Specifies the memory collection type to create.</param>
        /// <param name="nMax">Specifies the maximum count for the memory collection.</param>
        /// <param name="fAlpha">Specifies the alpha value used with prioritized memory collections.</param>
        /// <param name="strFile">Specifies the input/output file used with save and load memory collections.</param>
        /// <returns>The IMemoryCollection interface implemented by the memory collection created is returned.</returns>
        public static IMemoryCollection CreateMemory(MEMTYPE type, int nMax, float fAlpha = 0, string strFile = null)
        {
            switch (type)
            {
                case MEMTYPE.LOADING:
                    if (string.IsNullOrEmpty(strFile))
                        throw new Exception("You must specify a file.");
                    return new FileMemoryCollection(nMax, true, false, strFile);

                case MEMTYPE.SAVING:
                    if (string.IsNullOrEmpty(strFile))
                        throw new Exception("You must specify a file.");
                    return new FileMemoryCollection(nMax, false, true, strFile);

                case MEMTYPE.PRIORITY:
                    return new PrioritizedMemoryCollection(nMax, fAlpha);

                default:
                    return new RandomMemoryCollection(nMax);
            }
        }
    }
}
