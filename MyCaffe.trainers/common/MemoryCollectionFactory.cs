using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.trainers.common
{
    public class MemoryCollectionFactory
    {
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
