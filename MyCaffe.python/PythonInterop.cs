using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.python
{
    /// <summary>
    /// The PythonInterop uses PythonNet to execute Python code.
    /// </summary>
    /// <remarks>
    /// @see [Calling Python from C#:an Introduction to PythonNet](https://somegenericdev.medium.com/calling-python-from-c-an-introduction-to-pythonnet-c3d45f7d5232)
    /// </remarks>
    public class PythonInterop
    {
        public PythonInterop(string strPythonDllPath)
        {
            Initialize(strPythonDllPath);
        }

        public void Initialize(string strPythonDllPath)
        {
            Environment.SetEnvironmentVariable("PYTHONNET_PYDLL", strPythonDllPath);
            PythonEngine.Initialize();
        }

        public void RunPythonCode(string pycode)
        {
            using (Py.GIL())
            {
                PythonEngine.RunSimpleString(pycode);
            }
        }

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
