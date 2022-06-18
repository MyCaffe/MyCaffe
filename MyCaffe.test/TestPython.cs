using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.basecode;
using MyCaffe.param;
using MyCaffe.common;
using MyCaffe.layers;
using MyCaffe.gym.python;
using System.Diagnostics;
using MyCaffe.python;
using System.IO;

namespace MyCaffe.test
{
    [TestClass]
    public class TestPython
    {
        /// <summary>
        /// The TestPythonInterop runs the GPT2 model from https://huggingface.co to generate a
        /// short text dialog from input text.
        /// 
        /// To run the TestPythonInterop test, the following requirements must be installed:
        /// 
        /// Python 3.9.7 (64-bit)
        /// pip install tensorflow
        /// pip install torch --pre --extra-index-url https://download.pytorch.org/whl/nightly/cu116 
        /// pip install transformers        
        /// </summary>
        /// <remarks>
        /// The torch installation above installs torch with CUDA 11.6 which is located at:
        /// C:\Users\Windows\AppData\Local\Programs\Python\Python39\Lib\site-packages\torch\lib
        /// 
        /// The torch cuDNN DLLs must match the DLL versions in the 'cuda_11.6' directory
        /// </remarks>
        [TestMethod]
        public void TestPythonInterop()
        {
            string strUserName = Environment.UserName;
            string strPythonPath = @"C:\Users\" + strUserName + @"\AppData\Local\Programs\Python\Python39\python39.dll";

            if (!File.Exists(strPythonPath))
                return;

            PythonInterop py = new PythonInterop(strPythonPath);

            string strPy = "import os" + Environment.NewLine +
                            "os.environ['CUDA_VISIBLE_DEVICES'] = \'strGpu\'" + Environment.NewLine +
                            "from transformers import pipeline" + Environment.NewLine +
                            "generator = pipeline(task = 'text-generation', max_length=500)" + Environment.NewLine +
                            "res = generator(strInput)";

            KeyValuePair<string, object>[] rgArg = new KeyValuePair<string, object>[]
            {
                new KeyValuePair<string, object>("strGpu", "0"),
                new KeyValuePair<string, object>("strInput", "To be or not to be; that is the question.")
            };

            object obj = py.RunPythonCodeAndReturn(strPy, "res", rgArg);

            object[] rgRes = obj as object[];
            if (rgRes != null)
            {
                foreach (object obj1 in rgRes)
                {
                    string strJson = obj1.ToString();
                    Trace.WriteLine(strJson);
                }
            }
            else
            {
                string strJson = obj.ToString();
                Trace.WriteLine(strJson);
            }
        }
    }
}
