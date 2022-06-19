using Python.Runtime;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

/// <summary>
/// The MyCaffe.python namespace contains python related classes to help improve interop between C# and Python.  
/// </summary>
namespace MyCaffe.python
{
    /// <summary>
    /// The PythonInterop uses PythonNet to execute Python code.
    /// </summary>
    /// <remarks>
    /// @see [Calling Python from C#:an Introduction to PythonNet](https://somegenericdev.medium.com/calling-python-from-c-an-introduction-to-pythonnet-c3d45f7d5232)
    /// </remarks>
    public class PythonInterop : IDisposable
    {
        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="strPythonDllPath">Specifies the path to the Python DLL (e.g. Python39.dll) which is usually located
        /// at: @"C:\Users\" + strUserName + @"\AppData\Local\Programs\Python\Python39\python39.dll"</param>
        /// <remarks>
        /// See https://github.com/MyCaffe/MyCaffe/blob/master/MyCaffe.app/FormGptTest.cs (dowork, line 85) for an example 
        /// that demonstrates how to use the PythonInterop class to run the GPT2 pre-trained model from HuggingFace to generate text.
        /// </remarks>
        public PythonInterop(string strPythonDllPath)
        {
            Initialize(strPythonDllPath);
        }

        /// <summary>
        /// Free all resources used.
        /// </summary>
        public void Dispose()
        {
            Shutdown();
        }

        /// <summary>
        /// Initialize the Python Engine with the version of Python used.
        /// </summary>
        /// <param name="strPythonDllPath">Specifies the path to the Python DLL (e.g. Python39.dll) which is usually located
        /// at: @"C:\Users\" + strUserName + @"\AppData\Local\Programs\Python\Python39\python39.dll"</param>
        public void Initialize(string strPythonDllPath)
        {
            Environment.SetEnvironmentVariable("PYTHONNET_PYDLL", strPythonDllPath);
            PythonEngine.Initialize();
        }

        /// <summary>
        /// Shutdown the Python engine.
        /// </summary>
        public void Shutdown()
        {
            PythonEngine.Shutdown();
        }

        /// <summary>
        /// Run the Python code specified.
        /// </summary>
        /// <param name="pycode">Specifies a block of Python code to run.</param>
        public void RunPythonCode(string pycode)
        {
            using (Py.GIL())
            {
                PythonEngine.RunSimpleString(pycode);
            }
        }

        /// <summary>
        /// Run the Python code specified, passing a parameter to it from C#
        /// </summary>
        /// <param name="pycode">Specifies a block of Python code to run.</param>
        /// <param name="parameter">Specifies the parameter value.</param>
        /// <param name="parameterName">Specifies the parameter name within the Python code.</param>
        public void RunPythonCode(string pycode, object parameter, string parameterName)
        {
            using (Py.GIL())
            {
                using (PyModule scope = Py.CreateScope())
                {
                    scope.Set(parameterName, parameter.ToPython());
                    scope.Exec(pycode);
                }

            }
        }

        /// <summary>
        /// Run the Python code specified, passing a parameter to it from C# and receiving a return value
        /// </summary>
        /// <param name="pycode">Specifies a block of Python code to run.</param>
        /// <param name="parameter">Specifies the parameter value.</param>
        /// <param name="parameterName">Specifies the parameter name within the Python code.</param>
        /// <param name="returnedVariableName">Specifies the name of the return value in the Python code.</param>
        /// <returns>The Python return value is returned.</returns>
        public object RunPythonCodeAndReturn(string pycode, object parameter, string parameterName, string returnedVariableName)
        {
            object returnedVariable = new object();
            using (Py.GIL())
            {
                using (PyModule scope = Py.CreateScope())
                {
                    scope.Set(parameterName, parameter.ToPython());
                    scope.Exec(pycode);
                    returnedVariable = scope.Get<object>(returnedVariableName);
                }
            }
            return returnedVariable;
        }

        /// <summary>
        /// Run the Python code specified, passing a set of parameters to it from C# and receiving a return value
        /// </summary>
        /// <param name="pycode">Specifies a block of Python code to run.</param>
        /// <param name="returnedVariableName">Specifies the name of the return value in the Python code.</param>
        /// <param name="rgArg">Specifies a list of parameter name, value pairs.</param>
        /// <returns>The Python return value is returned.</returns>
        public object RunPythonCodeAndReturn(string pycode, string returnedVariableName, params KeyValuePair<string, object>[] rgArg)
        {
            object returnedVariable = new object();
            using (Py.GIL())
            {
                using (PyModule scope = Py.CreateScope())
                {
                    foreach (KeyValuePair<string, object> arg in rgArg)
                    {
                        scope.Set(arg.Key, arg.Value.ToPython());
                    }

                    scope.Exec(pycode);
                    returnedVariable = scope.Get<object>(returnedVariableName);
                }
            }

            return returnedVariable;
        }
    }
}
