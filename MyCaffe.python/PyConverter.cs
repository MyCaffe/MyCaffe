using System;
using System.Collections.Generic;
using System.Reflection;
using System.Linq;

namespace Python.Runtime
{
    /// <summary>
    /// use PyConverter to convert between python object and clr object.
    /// </summary>
    /// <remarks>
    /// Open-source public code originally shared from:
    /// @see [PyConverter](https://github.com/yagweb/pythonnetLab/blob/master/pynetLab/PyConverter.cs) by Wenguang Yang, 2018
    /// </remarks>
    public class PyConverter
    {
        /// <summary>
        /// Constructor
        /// </summary>
        public PyConverter()
        {
            this.Converters = new List<PyClrTypeBase>();
            this.PythonConverters = new Dictionary<IntPtr, Dictionary<Type, PyClrTypeBase>>();
            this.ClrConverters = new Dictionary<Type, Dictionary<IntPtr, PyClrTypeBase>>();
        }

        private List<PyClrTypeBase> Converters;

        private Dictionary<IntPtr, Dictionary<Type, PyClrTypeBase>> PythonConverters;

        private Dictionary<Type, Dictionary<IntPtr, PyClrTypeBase>> ClrConverters;

        /// <summary>
        /// Add a new type for conversion.
        /// </summary>
        /// <param name="converter">Specifies the converter.</param>
        public void Add(PyClrTypeBase converter)
        {
            this.Converters.Add(converter);

            Dictionary<Type, PyClrTypeBase> py_converters;
            var state = this.PythonConverters.TryGetValue(converter.PythonType.Handle, out py_converters);
            if (!state)
            {
                py_converters = new Dictionary<Type, PyClrTypeBase>();
                this.PythonConverters.Add(converter.PythonType.Handle, py_converters);
            }
            py_converters.Add(converter.ClrType, converter);

            Dictionary<IntPtr, PyClrTypeBase> clr_converters;
            state = this.ClrConverters.TryGetValue(converter.ClrType, out clr_converters);
            if (!this.ClrConverters.ContainsKey(converter.ClrType))
            {
                clr_converters = new Dictionary<IntPtr, PyClrTypeBase>();
                this.ClrConverters.Add(converter.ClrType, clr_converters);
            }
            clr_converters.Add(converter.PythonType.Handle, converter);
        }

        /// <summary>
        /// Add a new PyObjec type to the converter.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="pyType">Specifies the type to add.</param>
        /// <param name="converter">Specifies the converter.</param>
        public void AddObjectType<T>(PyObject pyType, PyConverter converter = null)
        {
            if (converter == null)
            {
                converter = this;
            }
            this.Add(new ObjectType<T>(pyType, converter));
        }

        /// <summary>
        /// Add a new list type to the converter.
        /// </summary>
        /// <param name="converter">Specifies the converter.</param>
        public void AddListType(PyConverter converter = null)
        {
            this.AddListType<object>(converter);
        }

        /// <summary>
        /// Add a new list type to the converter.
        /// </summary>
        /// <typeparam name="T">Specifies the element type.</typeparam>
        /// <param name="converter">Specifies the converter.</param>
        public void AddListType<T>(PyConverter converter = null)
        {
            if (converter == null)
            {
                converter = this;
            }
            this.Add(new PyListType<T>(converter));
        }

        /// <summary>
        /// Add a new dictionary type of K key types and V value types.
        /// </summary>
        /// <typeparam name="K">Specifies the key type.</typeparam>
        /// <typeparam name="V">Specifies the value type.</typeparam>
        /// <param name="converter">Specifies the converter.</param>
        public void AddDictType<K, V>(PyConverter converter = null)
        {
            if (converter == null)
            {
                converter = this;
            }
            this.Add(new PyDictType<K, V>(converter));
        }

        /// <summary>
        /// Convert the PyObject to a specifict type T
        /// </summary>
        /// <typeparam name="T">Specifies the type to convert to.</typeparam>
        /// <param name="obj">Specifies the PyObject to convert.</param>
        /// <returns>The clr object is returned.</returns>
        public T ToClr<T>(PyObject obj)
        {
            return (T)ToClr(obj, typeof(T));
        }

        /// <summary>
        /// Convert a PyObject to a clr type of 't'.
        /// </summary>
        /// <param name="obj">Specifies the PyObject to convert.</param>
        /// <param name="t">Specifies the expected clr type.</param>
        /// <returns>The clr object is returned.</returns>
        public object ToClr(PyObject obj, Type t = null)
        {
            if (obj == null)
            {
                return null;
            }
            PyObject type = obj.GetPythonType();
            Dictionary<Type, PyClrTypeBase> converters;
            var state = PythonConverters.TryGetValue(type.Handle, out converters);
            if (!state)
            {
                return obj;
            }
            if (t == null || !converters.ContainsKey(t))
            {
                return converters.Values.First().ToClr(obj);
            }
            else
            {
                return converters[t].ToClr(obj);
            }
        }

        /// <summary>
        /// Convert a clr object to a PyObject.
        /// </summary>
        /// <param name="clrObj">Specifies the clr object to covert.</param>
        /// <param name="t">Specifies the expected Python type.</param>
        /// <returns>The converted PyObject is returned.</returns>
        /// <exception cref="Exception">An exception is thrown on error.</exception>
        public PyObject ToPython(object clrObj, IntPtr? t = null)
        {
            if (clrObj == null)
            {
                return null;
            }
            Type type = clrObj.GetType();
            Dictionary<IntPtr, PyClrTypeBase> converters;
            var state = ClrConverters.TryGetValue(type, out converters);
            if (!state)
            {
                throw new Exception($"Type {type.ToString()} not recognized");
            }
            if (t == null || !converters.ContainsKey(t.Value))
            {
                return converters.Values.First().ToPython(clrObj);
            }
            else
            {
                return converters[t.Value].ToPython(clrObj);
            }
        }
    }

    /// <summary>
    /// The PyClrTypeBase is the base class for other types.
    /// </summary>
    /// <remarks>
    /// Open-source public code originally shared from:
    /// @see [PyConverter](https://github.com/yagweb/pythonnetLab/blob/master/pynetLab/PyConverter.cs) by Wenguang Yang, 2018
    /// </remarks>
    public abstract class PyClrTypeBase
    {
        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="pyType">Specifies the Python type.</param>
        /// <param name="clrType">Specifies the clr type.</param>
        protected PyClrTypeBase(string pyType, Type clrType)
        {
            this.PythonType = PythonEngine.Eval(pyType);
            this.ClrType = clrType;
        }

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="pyType">Specifies the Python type.</param>
        /// <param name="clrType">Specifies the clr type.</param>
        protected PyClrTypeBase(PyObject pyType, Type clrType)
        {
            this.PythonType = pyType;
            this.ClrType = clrType;
        }

        /// <summary>
        /// Returns the Python type.
        /// </summary>
        public PyObject PythonType
        {
            get;
            private set;
        }

        /// <summary>
        /// Returns the clr type.
        /// </summary>
        public Type ClrType
        {
            get;
            private set;
        }

        /// <summary>
        /// Converts the PyObject type to the clr object.
        /// </summary>
        /// <param name="pyObj"></param>
        /// <returns></returns>
        public abstract object ToClr(PyObject pyObj);

        /// <summary>
        /// Converts the clr object to a PyObject type.
        /// </summary>
        /// <param name="clrObj"></param>
        /// <returns></returns>
        public abstract PyObject ToPython(object clrObj);
    }

    /// <summary>
    /// The PyClrType class defines a Python clr type.
    /// </summary>
    /// <remarks>
    /// Open-source public code originally shared from:
    /// @see [PyConverter](https://github.com/yagweb/pythonnetLab/blob/master/pynetLab/PyConverter.cs) by Wenguang Yang, 2018
    /// </remarks>
    public class PyClrType : PyClrTypeBase
    {
        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="pyType">Specifies the Python type.</param>
        /// <param name="clrType">Specifies the clr type.</param>
        /// <param name="py2clr">Specifies the Python 2 clr object.</param>
        /// <param name="clr2py">Specifies the clr 2 Python object.</param>
        public PyClrType(PyObject pyType, Type clrType,
            Func<PyObject, object> py2clr, Func<object, PyObject> clr2py)
            : base(pyType, clrType)
        {
            this.Py2Clr = py2clr;
            this.Clr2Py = clr2py;
        }

        /// <summary>
        /// Returns the PyObject, clr object pair.
        /// </summary>
        private Func<PyObject, object> Py2Clr;

        /// <summary>
        /// Returns the clr object, PyObject pair.
        /// </summary>
        private Func<object, PyObject> Clr2Py;

        /// <summary>
        /// Converts a PyObject to a clr object.
        /// </summary>
        /// <param name="pyObj"></param>
        /// <returns></returns>
        public override object ToClr(PyObject pyObj)
        {
            return this.Py2Clr(pyObj);
        }

        /// <summary>
        /// Converts a clr object to a PyObject.
        /// </summary>
        /// <param name="clrObj"></param>
        /// <returns></returns>
        public override PyObject ToPython(object clrObj)
        {
            return this.Clr2Py(clrObj);
        }
    }

    /// <summary>
    /// The StringType represents a clr string type.
    /// </summary>
    /// <remarks>
    /// Open-source public code originally shared from:
    /// @see [PyConverter](https://github.com/yagweb/pythonnetLab/blob/master/pynetLab/PyConverter.cs) by Wenguang Yang, 2018
    /// </remarks>
    public class StringType : PyClrTypeBase
    {
        /// <summary>
        /// The constructor.
        /// </summary>
        public StringType()
            : base("str", typeof(string))
        {
        }

        /// <summary>
        /// Converts a PyObject to a string.
        /// </summary>
        /// <param name="pyObj">Specifies the Python object to convert.</param>
        /// <returns>A clr string is returned.</returns>
        public override object ToClr(PyObject pyObj)
        {
            return pyObj.As<string>();
        }

        /// <summary>
        /// Converts a string to a PyObject.
        /// </summary>
        /// <param name="clrObj">Specifies the string to convert.</param>
        /// <returns>A PyObject representing the string is returned.</returns>
        public override PyObject ToPython(object clrObj)
        {
            return new PyString(Convert.ToString(clrObj));
        }
    }

    /// <summary>
    /// The BooleanType represents a clr bool type.
    /// </summary>
    /// <remarks>
    /// Open-source public code originally shared from:
    /// @see [PyConverter](https://github.com/yagweb/pythonnetLab/blob/master/pynetLab/PyConverter.cs) by Wenguang Yang, 2018
    /// </remarks>
    public class BooleanType : PyClrTypeBase
    {
        /// <summary>
        /// The constructor.
        /// </summary>
        public BooleanType()
            : base("bool", typeof(bool))
        {
        }

        /// <summary>
        /// Converts a PyObject to a bool.
        /// </summary>
        /// <param name="pyObj">Specifies the Python object to convert.</param>
        /// <returns>A clr bool is returned.</returns>
        public override object ToClr(PyObject pyObj)
        {
            return pyObj.As<bool>();
        }

        /// <summary>
        /// Converts a bool to a PyObject.
        /// </summary>
        /// <param name="clrObj">Specifies the bool to convert.</param>
        /// <returns>A PyObject representing the bool is returned.</returns>
        public override PyObject ToPython(object clrObj)
        {
            //return new PyBoolean(Convert.ToString(clrObj));
            throw new NotImplementedException();
        }
    }

    /// <summary>
    /// The Int32Type represents a clr int type.
    /// </summary>
    /// <remarks>
    /// Open-source public code originally shared from:
    /// @see [PyConverter](https://github.com/yagweb/pythonnetLab/blob/master/pynetLab/PyConverter.cs) by Wenguang Yang, 2018
    /// </remarks>
    public class Int32Type : PyClrTypeBase
    {
        /// <summary>
        /// The constructor.
        /// </summary>
        public Int32Type()
            : base("int", typeof(int))
        {
        }

        /// <summary>
        /// Converts a PyObject to a int.
        /// </summary>
        /// <param name="pyObj">Specifies the Python object to convert.</param>
        /// <returns>A clr int is returned.</returns>
        public override object ToClr(PyObject pyObj)
        {
            return pyObj.As<int>();
        }

        /// <summary>
        /// Converts a int to a PyObject.
        /// </summary>
        /// <param name="clrObj">Specifies the int to convert.</param>
        /// <returns>A PyObject representing the int is returned.</returns>
        public override PyObject ToPython(object clrObj)
        {
            return new PyInt(Convert.ToInt32(clrObj));
        }
    }

    /// <summary>
    /// The Int64Type represents a clr long type.
    /// </summary>
    /// <remarks>
    /// Open-source public code originally shared from:
    /// @see [PyConverter](https://github.com/yagweb/pythonnetLab/blob/master/pynetLab/PyConverter.cs) by Wenguang Yang, 2018
    /// </remarks>
    public class Int64Type : PyClrTypeBase
    {
        /// <summary>
        /// The constructor.
        /// </summary>
        public Int64Type()
            : base("int", typeof(long))
        {
        }

        /// <summary>
        /// Converts a PyObject to a long.
        /// </summary>
        /// <param name="pyObj">Specifies the Python object to convert.</param>
        /// <returns>A clr long is returned.</returns>
        public override object ToClr(PyObject pyObj)
        {
            return pyObj.As<long>();
        }

        /// <summary>
        /// Converts a long to a PyObject.
        /// </summary>
        /// <param name="clrObj">Specifies the long to convert.</param>
        /// <returns>A PyObject representing the long is returned.</returns>
        public override PyObject ToPython(object clrObj)
        {
            return new PyInt(Convert.ToInt64(clrObj));
        }
    }

    /// <summary>
    /// The FloatType represents a clr float type.
    /// </summary>
    /// <remarks>
    /// Open-source public code originally shared from:
    /// @see [PyConverter](https://github.com/yagweb/pythonnetLab/blob/master/pynetLab/PyConverter.cs) by Wenguang Yang, 2018
    /// </remarks>
    public class FloatType : PyClrTypeBase
    {
        /// <summary>
        /// The constructor.
        /// </summary>
        public FloatType()
            : base("float", typeof(float))
        {
        }

        /// <summary>
        /// Converts a PyObject to a float.
        /// </summary>
        /// <param name="pyObj">Specifies the Python object to convert.</param>
        /// <returns>A clr float is returned.</returns>
        public override object ToClr(PyObject pyObj)
        {
            return pyObj.As<float>();
        }

        /// <summary>
        /// Converts a float to a PyObject.
        /// </summary>
        /// <param name="clrObj">Specifies the float to convert.</param>
        /// <returns>A PyObject representing the float is returned.</returns>
        public override PyObject ToPython(object clrObj)
        {
            return new PyFloat(Convert.ToSingle(clrObj));
        }
    }

    /// <summary>
    /// The DoubleType represents a clr double type.
    /// </summary>
    /// <remarks>
    /// Open-source public code originally shared from:
    /// @see [PyConverter](https://github.com/yagweb/pythonnetLab/blob/master/pynetLab/PyConverter.cs) by Wenguang Yang, 2018
    /// </remarks>
    public class DoubleType : PyClrTypeBase
    {
        /// <summary>
        /// The constructor.
        /// </summary>
        public DoubleType()
            : base("float", typeof(double))
        {
        }

        /// <summary>
        /// Converts a PyObject to a float.
        /// </summary>
        /// <param name="pyObj">Specifies the Python object to convert.</param>
        /// <returns>A clr double is returned.</returns>
        public override object ToClr(PyObject pyObj)
        {
            return pyObj.As<double>();
        }

        /// <summary>
        /// Converts a double to a PyObject.
        /// </summary>
        /// <param name="clrObj">Specifies the double to convert.</param>
        /// <returns>A PyObject representing the double is returned.</returns>
        public override PyObject ToPython(object clrObj)
        {
            return new PyFloat(Convert.ToDouble(clrObj));
        }
    }

    /// <summary>
    /// The PyPropertyAttribute represents a Python attribute.
    /// </summary>
    /// <remarks>
    /// Open-source public code originally shared from:
    /// @see [PyConverter](https://github.com/yagweb/pythonnetLab/blob/master/pynetLab/PyConverter.cs) by Wenguang Yang, 2018
    /// </remarks>
    public class PyPropetryAttribute : Attribute
    {
        /// <summary>
        /// The constructor.
        /// </summary>
        public PyPropetryAttribute()
        {
            this.Name = null;
        }

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="name">Specifies the attribute name.</param>
        /// <param name="py_type">Specifies the attribute type.</param>
        public PyPropetryAttribute(string name, string py_type = null)
        {
            this.Name = name;
            this.PythonTypeName = py_type;
        }

        /// <summary>
        /// Returns the attribute name.
        /// </summary>
        public string Name
        {
            get;
            private set;
        }

        /// <summary>
        /// Returns the atttribute type name.
        /// </summary>
        public string PythonTypeName
        {
            get;
            private set;
        }

        /// <summary>
        /// Returns the Python type.
        /// </summary>
        public IntPtr? PythonType
        {
            get;
            set;
        }
    }

    /// <summary>
    /// The ClrMemberInfo represents clr information.
    /// </summary>
    /// <remarks>
    /// Open-source public code originally shared from:
    /// @see [PyConverter](https://github.com/yagweb/pythonnetLab/blob/master/pynetLab/PyConverter.cs) by Wenguang Yang, 2018
    /// </remarks>
    abstract class ClrMemberInfo
    {
        /// <summary>
        /// Returns the Python property name.
        /// </summary>
        public string PyPropertyName;

        /// <summary>
        /// Returns the Python type.
        /// </summary>
        public IntPtr? PythonType;

        /// <summary>
        /// Returns the clr property name.
        /// </summary>
        public string ClrPropertyName;

        /// <summary>
        /// Returns the clr type.
        /// </summary>
        public Type ClrType;

        /// <summary>
        /// Returns the converter used.
        /// </summary>
        public PyConverter Converter;

        /// <summary>
        /// Set the Python object attribute.
        /// </summary>
        /// <param name="pyObj">Specifies the PyObject.</param>
        /// <param name="clrObj">Specifies the clr object.</param>
        public abstract void SetPyObjAttr(PyObject pyObj, object clrObj);

        /// <summary>
        /// Set the clr object attribute.
        /// </summary>
        /// <param name="clrObj">Specifies the clr object.</param>
        /// <param name="pyObj">Specifies the PyObject.</param>
        public abstract void SetClrObjAttr(object clrObj, PyObject pyObj);
    }

    /// <summary>
    /// The ClrPropertyInfo specifies the clr property information.
    /// </summary>
    /// <remarks>
    /// Open-source public code originally shared from:
    /// @see [PyConverter](https://github.com/yagweb/pythonnetLab/blob/master/pynetLab/PyConverter.cs) by Wenguang Yang, 2018
    /// </remarks>
    class ClrPropertyInfo : ClrMemberInfo
    {
        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="info">Specifies the property information.</param>
        /// <param name="py_info">Specifies the Python property information.</param>
        /// <param name="converter">Specifies the converter.</param>
        public ClrPropertyInfo(PropertyInfo info, PyPropetryAttribute py_info, PyConverter converter)
        {
            this.PropertyInfo = info;
            this.ClrPropertyName = info.Name;
            this.ClrType = info.PropertyType;
            this.PyPropertyName = py_info.Name;
            if (string.IsNullOrEmpty(this.PyPropertyName))
            {
                this.PyPropertyName = info.Name;
            }
            //this.PythonType = converter.Get();
            this.Converter = converter;
        }

        /// <summary>
        /// Return the clr Property information.
        /// </summary>
        public PropertyInfo PropertyInfo
        {
            get;
            private set;
        }

        /// <summary>
        /// Sets the Python object attribute.
        /// </summary>
        /// <param name="pyObj">Specifies the PyObject.</param>
        /// <param name="clrObj">Specifies the clr object.</param>
        public override void SetPyObjAttr(PyObject pyObj, object clrObj)
        {
            var clr_value = this.PropertyInfo.GetValue(clrObj, null);
            var py_value = this.Converter.ToPython(clr_value);
            pyObj.SetAttr(this.PyPropertyName, py_value);
        }

        /// <summary>
        /// Sets the clr object attribute.
        /// </summary>
        /// <param name="clrObj">Specifies the clr object.</param>
        /// <param name="pyObj">Specifies the PyObject.</param>
        public override void SetClrObjAttr(object clrObj, PyObject pyObj)
        {
            var py_value = pyObj.GetAttr(this.PyPropertyName);
            var clr_value = this.Converter.ToClr(py_value);
            this.PropertyInfo.SetValue(clrObj, clr_value, null);
        }
    }

    /// <summary>
    /// The ClrFieldInfo defines the clr field information.
    /// </summary>
    /// <remarks>
    /// Open-source public code originally shared from:
    /// @see [PyConverter](https://github.com/yagweb/pythonnetLab/blob/master/pynetLab/PyConverter.cs) by Wenguang Yang, 2018
    /// </remarks>
    class ClrFieldInfo : ClrMemberInfo
    {
        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="info">Specifies the field information.</param>
        /// <param name="py_info">Specifies the Python property attribute.</param>
        /// <param name="converter">Specifies the converter.</param>
        public ClrFieldInfo(FieldInfo info, PyPropetryAttribute py_info, PyConverter converter)
        {
            this.FieldInfo = info;
            this.ClrPropertyName = info.Name;
            this.ClrType = info.FieldType;
            this.PyPropertyName = py_info.Name;
            if (string.IsNullOrEmpty(this.PyPropertyName))
            {
                this.PyPropertyName = info.Name;
            }
            //this.PythonType = converter.Get();
            this.Converter = converter;
        }

        /// <summary>
        /// Returns the field information.
        /// </summary>
        public FieldInfo FieldInfo;

        /// <summary>
        /// Sets the Python object attribute.
        /// </summary>
        /// <param name="pyObj">Specifies the PyObject.</param>
        /// <param name="clrObj">Specifies the clr object.</param>
        public override void SetPyObjAttr(PyObject pyObj, object clrObj)
        {
            var clr_value = this.FieldInfo.GetValue(clrObj);
            var py_value = this.Converter.ToPython(clr_value);
            pyObj.SetAttr(this.PyPropertyName, py_value);
        }

        /// <summary>
        /// Sets the clr object attribute.
        /// </summary>
        /// <param name="clrObj">Specifies the clr object.</param>
        /// <param name="pyObj">Specifies the PyObject.</param>
        public override void SetClrObjAttr(object clrObj, PyObject pyObj)
        {
            var py_value = pyObj.GetAttr(this.PyPropertyName);
            var clr_value = this.Converter.ToClr(py_value);
            this.FieldInfo.SetValue(clrObj, clr_value);
        }
    }

    /// <summary>
    /// Convert between Python object and clr object
    /// </summary>
    /// <remarks>
    /// Open-source public code originally shared from:
    /// @see [PyConverter](https://github.com/yagweb/pythonnetLab/blob/master/pynetLab/PyConverter.cs) by Wenguang Yang, 2018
    /// </remarks>
    public class ObjectType<T> : PyClrTypeBase
    {
        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="pyType">Specifies the Python type.</param>
        /// <param name="converter">Specifies the converter.</param>
        public ObjectType(PyObject pyType, PyConverter converter)
            : base(pyType, typeof(T))
        {
            this.Converter = converter;
            this.Properties = new List<ClrMemberInfo>();

            // Get all attributes
            foreach (var property in this.ClrType.GetProperties())
            {
                var attr = property.GetCustomAttributes(typeof(PyPropetryAttribute), true);
                if (attr.Length == 0)
                {
                    continue;
                }
                var py_info = attr[0] as PyPropetryAttribute;
                this.Properties.Add(new ClrPropertyInfo(property, py_info, this.Converter));
            }

            foreach (var field in this.ClrType.GetFields())
            {
                var attr = field.GetCustomAttributes(typeof(PyPropetryAttribute), true);
                if (attr.Length == 0)
                {
                    continue;
                }
                var py_info = attr[0] as PyPropetryAttribute;
                this.Properties.Add(new ClrFieldInfo(field, py_info, this.Converter));
            }
        }

        /// <summary>
        /// Returns the converter.
        /// </summary>
        private PyConverter Converter;

        /// <summary>
        /// Returns a list of the member information.
        /// </summary>
        private List<ClrMemberInfo> Properties;

        /// <summary>
        /// Converts the PyObject to a clr object.
        /// </summary>
        /// <param name="pyObj">Specifies the PyObject to convert.</param>
        /// <returns>The converted clr object is returned.</returns>
        public override object ToClr(PyObject pyObj)
        {
            var clrObj = Activator.CreateInstance(this.ClrType);
            foreach (var item in this.Properties)
            {
                item.SetClrObjAttr(clrObj, pyObj);
            }
            return clrObj;
        }

        /// <summary>
        /// Converts a clr object to a PyObject.
        /// </summary>
        /// <param name="clrObj">Specifies the clr object.</param>
        /// <returns>The converted PyObject is returned.</returns>
        public override PyObject ToPython(object clrObj)
        {
            var pyObj = this.PythonType.Invoke();
            foreach (var item in this.Properties)
            {
                item.SetPyObjAttr(pyObj, clrObj);
            }
            return pyObj;
        }
    }

    /// <summary>
    /// Defines a PyListType of type 'T'
    /// </summary>
    /// <typeparam name="T">Specifies the base type of the list.</typeparam>
    /// <remarks>
    /// Open-source public code originally shared from:
    /// @see [PyConverter](https://github.com/yagweb/pythonnetLab/blob/master/pynetLab/PyConverter.cs) by Wenguang Yang, 2018
    /// </remarks>
    public class PyListType<T> : PyClrTypeBase
    {
        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="converter">Specifies the converter.</param>
        public PyListType(PyConverter converter)
            : base("list", typeof(List<T>))
        {
            this.Converter = converter;
        }

        /// <summary>
        /// Returns the converter.
        /// </summary>
        private PyConverter Converter;

        /// <summary>
        /// Converts the PyObject to a clr object.
        /// </summary>
        /// <param name="pyObj">Specifies the PyObject to convert.</param>
        /// <returns>The converted clr object is returned.</returns>
        public override object ToClr(PyObject pyObj)
        {
            var dict = this._ToClr(new PyList(pyObj));
            return dict;
        }

        /// <summary>
        /// Converts a PyList object to a clr List.
        /// </summary>
        /// <param name="pyList">Specifies the list object to convert.</param>
        /// <returns>The clr List is returned.</returns>
        private object _ToClr(PyList pyList)
        {
            var list = new List<T>();
            foreach (PyObject item in pyList)
            {
                var _item = this.Converter.ToClr<T>(item);
                list.Add(_item);
            }
            return list;
        }

        /// <summary>
        /// Converts a clr object to a PyObject.
        /// </summary>
        /// <param name="clrObj">Specifies the clr object.</param>
        /// <returns>The converted PyObject is returned.</returns>
        public override PyObject ToPython(object clrObj)
        {
            return this._ToPython(clrObj as List<T>);
        }

        /// <summary>
        /// Converts a clr List to a PyList.
        /// </summary>
        /// <param name="clrObj">Specifies the clr List.</param>
        /// <returns>The PyList is returned as a PyObject.</returns>
        public PyObject _ToPython(List<T> clrObj)
        {
            var pyList = new PyList();
            foreach (var item in clrObj)
            {
                PyObject _item = this.Converter.ToPython(item);
                pyList.Append(_item);
            }
            return pyList;
        }
    }

    /// <summary>
    /// Specifies a PyDictionType dictionary.
    /// </summary>
    /// <typeparam name="K">Specifies the key type.</typeparam>
    /// <typeparam name="V">Specifies the value type.</typeparam>
    /// <remarks>
    /// Open-source public code originally shared from:
    /// @see [PyConverter](https://github.com/yagweb/pythonnetLab/blob/master/pynetLab/PyConverter.cs) by Wenguang Yang, 2018
    /// </remarks>
    public class PyDictType<K, V> : PyClrTypeBase
    {
        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="converter">Specifies the converter.</param>
        public PyDictType(PyConverter converter)
            : base("dict", typeof(Dictionary<K, V>))
        {
            this.Converter = converter;
        }

        /// <summary>
        /// Returns the converter.
        /// </summary>
        private PyConverter Converter;

        /// <summary>
        /// Convert a PyObject to a clr object.
        /// </summary>
        /// <param name="pyObj">Specifies the PyObject to convert.</param>
        /// <returns>The converted clr object is returned.</returns>
        public override object ToClr(PyObject pyObj)
        {
            var dict = this._ToClr(new PyDict(pyObj));
            return dict;
        }

        /// <summary>
        /// Convert a PyDict dictionary to a clr Dictionary,
        /// </summary>
        /// <param name="pyDict">Specifies the Python PyDict dictionary to convert.</param>
        /// <returns>The clr Dictionary is returned as a object.</returns>
        private object _ToClr(PyDict pyDict)
        {
            var dict = new Dictionary<K, V>();
            foreach (PyObject key in pyDict.Keys())
            {
                var _key = this.Converter.ToClr<K>(key);
                PyObject objVal = pyDict[key];
                
                var _value = this.Converter.ToClr<V>(objVal);
                dict.Add(_key, _value);
            }
            return dict;
        }

        /// <summary>
        /// Convert a clr object to a PyObjec.t
        /// </summary>
        /// <param name="clrObj">Specifies the clr object to convert.</param>
        /// <returns>The conveted PyObject is returned.</returns>
        public override PyObject ToPython(object clrObj)
        {
            return this._ToPython(clrObj as Dictionary<K, V>);
        }

        /// <summary>
        /// Convert a clr Dictionary to a PyDict and return it as a PyObject.
        /// </summary>
        /// <param name="clrObj">Specifies the clr Dictionry to convert.</param>
        /// <returns>The converted PyDict is returned as a PyObject.</returns>
        public PyObject _ToPython(Dictionary<K, V> clrObj)
        {
            var pyDict = new PyDict();
            foreach (var item in clrObj)
            {
                PyObject _key = this.Converter.ToPython(item.Key);
                PyObject _value = this.Converter.ToPython(item.Value);
                pyDict[_key] = _value;
            }
            return pyDict;
        }
    }
}