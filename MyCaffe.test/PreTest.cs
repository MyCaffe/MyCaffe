using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.test
{
    class PreTest
    {
        public PreTest()
        {
        }

        public static void Init()
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
        }
    }
}
