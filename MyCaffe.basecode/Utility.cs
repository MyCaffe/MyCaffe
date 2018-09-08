using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;

/// <summary>
/// The MyCaffe.common namespace contains all common objects that make up MyCaffe.
/// </summary>
namespace MyCaffe.basecode
{
    /// <summary>
    /// The IBinaryPersist interface provides generic save and load functionality.
    /// </summary>
    public interface IBinaryPersist
    {
        /// <summary>
        /// Save to a binary writer.
        /// </summary>
        /// <param name="bw">Specifies the binary writer.</param>
        void Save(BinaryWriter bw);
        /// <summary>
        /// Load from a binary reader.
        /// </summary>
        /// <param name="br">Specifies the binary reader.</param>
        /// <param name="bNewInstance">When <i>true</i>, creates a new instance of the object, otherwise the current instance is loaded.</param>
        /// <returns></returns>
        object Load(BinaryReader br, bool bNewInstance = true);
    }

    /// <summary>
    /// The Utility class provides general utility funtions.
    /// </summary>
    public class Utility
    {
        /// <summary>
        /// The Utility constructor.
        /// </summary>
        public Utility()
        {
        }

        /// <summary>
        /// Save a list of items to a binary writer.
        /// </summary>
        /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
        /// <param name="bw">Specifies the binary writer.</param>
        /// <param name="rg">Specifies the list of items.</param>
        public static void Save<T>(BinaryWriter bw, List<T> rg)
        {
            int nCount = (rg != null) ? rg.Count : 0;

            bw.Write(nCount);

            if (nCount == 0)
                return;

            if (typeof(T) == typeof(string))
            {
                foreach (T t in rg)
                {
                    bw.Write(t.ToString());
                }

                return;
            }

            if (typeof(T) == typeof(double))
            {
                foreach (T t in rg)
                {
                    bw.Write((double)Convert.ChangeType(t, typeof(double)));
                }

                return;
            }

            if (typeof(T) == typeof(float))
            {
                foreach (T t in rg)
                {
                    bw.Write((float)Convert.ChangeType(t, typeof(float)));
                }

                return;
            }

            if (typeof(T) == typeof(int))
            {
                foreach (T t in rg)
                {
                    bw.Write((int)Convert.ChangeType(t, typeof(int)));
                }

                return;
            }

            if (typeof(T) == typeof(long))
            {
                foreach (T t in rg)
                {
                    bw.Write((long)Convert.ChangeType(t, typeof(long)));
                }

                return;
            }

            if (typeof(T) == typeof(uint))
            {
                foreach (T t in rg)
                {
                    bw.Write((uint)Convert.ChangeType(t, typeof(uint)));
                }

                return;
            }

            if (typeof(T) == typeof(bool))
            {
                foreach (T t in rg)
                {
                    bw.Write((bool)Convert.ChangeType(t, typeof(bool)));
                }

                return;
            }

            bool bSaved = true;

            foreach (T t in rg)
            {
                IBinaryPersist p = t as IBinaryPersist;

                if (p != null)
                {
                    p.Save(bw);
                    bSaved = true;
                }

                return;
            }

            if (!bSaved)
                throw new Exception("Persistance for type " + (typeof(T)).ToString() + " is not supported!");
        }

        /// <summary>
        /// Loads a list of items from a binary reader.
        /// </summary>
        /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
        /// <param name="br">Specifies the binary reader.</param>
        /// <returns>The list of items is returned.</returns>
        public static List<T> Load<T>(BinaryReader br)
        {
            int nCount = br.ReadInt32();
            List<T> rg = new List<T>(nCount);

            if (nCount == 0)
                return rg;

            if (typeof(T) == typeof(string))
            {
                for (int i = 0; i < nCount; i++)
                {
                    string val = br.ReadString();
                    rg.Add((T)Convert.ChangeType(val, typeof(T)));
                }

                return rg;
            }

            if (typeof(T) == typeof(double))
            {
                for (int i = 0; i < nCount; i++)
                {
                    double val = br.ReadDouble();
                    rg.Add((T)Convert.ChangeType(val, typeof(T)));
                }

                return rg;
            }

            if (typeof(T) == typeof(float))
            {
                for (int i = 0; i < nCount; i++)
                {
                    float val = br.ReadSingle();
                    rg.Add((T)Convert.ChangeType(val, typeof(T)));
                }

                return rg;
            }

            if (typeof(T) == typeof(int))
            {
                for (int i = 0; i < nCount; i++)
                {
                    int val = br.ReadInt32();
                    rg.Add((T)Convert.ChangeType(val, typeof(T)));
                }

                return rg;
            }

            if (typeof(T) == typeof(long))
            {
                for (int i = 0; i < nCount; i++)
                {
                    long val = br.ReadInt64();
                    rg.Add((T)Convert.ChangeType(val, typeof(T)));
                }

                return rg;
            }

            if (typeof(T) == typeof(uint))
            {
                for (int i = 0; i < nCount; i++)
                {
                    uint val = br.ReadUInt32();
                    rg.Add((T)Convert.ChangeType(val, typeof(T)));
                }

                return rg;
            }

            if (typeof(T) == typeof(bool))
            {
                for (int i = 0; i < nCount; i++)
                {
                    bool val = br.ReadBoolean();
                    rg.Add((T)Convert.ChangeType(val, typeof(T)));
                }

                return rg;
            }

            object obj = Activator.CreateInstance(typeof(T));
            IBinaryPersist p = obj as IBinaryPersist;

            if (p != null)
            {
                for (int i = 0; i < nCount; i++)
                {
                    object val = p.Load(br);
                    rg.Add((T)Convert.ChangeType(val, typeof(T)));
                }

                return rg;
            }

            throw new Exception("Persistance for type " + (typeof(T)).ToString() + " is not supported!");
        }

        /// <summary>
        /// Save a list of <i>double</i> to a binary writer.
        /// </summary>
        /// <param name="bw">Specifies the binary writer.</param>
        /// <param name="rg">Specifies the list of items.</param>
        public static void Save(BinaryWriter bw, List<double> rg)
        {
            bw.Write(rg.Count);

            for (int i=0; i<rg.Count; i++)
            {
                bw.Write(rg[i]);
            }
        }

        /// <summary>
        /// Loads a list of <i>double</i> from a binary reader.
        /// </summary>
        /// <param name="br">Specifies the binary reader.</param>
        /// <returns>The list of items is returned.</returns>
        public static List<double> LoadDouble(BinaryReader br)
        {
            List<double> rg = new List<double>();
            int nCount = br.ReadInt32();

            for (int i = 0; i < nCount; i++)
            {
                rg.Add(br.ReadDouble());
            }

            return rg;
        }

        /// <summary>
        /// Save a list of <i>float</i> to a binary writer.
        /// </summary>
        /// <param name="bw">Specifies the binary writer.</param>
        /// <param name="rg">Specifies the list of items.</param>
        public static void Save(BinaryWriter bw, List<float> rg)
        {
            bw.Write(rg.Count);

            for (int i = 0; i < rg.Count; i++)
            {
                bw.Write(rg[i]);
            }
        }

        /// <summary>
        /// Loads a list of <i>float</i> from a binary reader.
        /// </summary>
        /// <param name="br">Specifies the binary reader.</param>
        /// <returns>The list of items is returned.</returns>
        public static List<float> LoadFloat(BinaryReader br)
        {
            List<float> rg = new List<float>();
            int nCount = br.ReadInt32();

            for (int i = 0; i < nCount; i++)
            {
                rg.Add(br.ReadSingle());
            }

            return rg;
        }

        /// <summary>
        /// Saves a nullable int to a binary writer.
        /// </summary>
        /// <param name="bw">Specifies the binary writer.</param>
        /// <param name="nVal">Specifies the value to write.</param>
        public static void Save(BinaryWriter bw, int? nVal)
        {
            bw.Write(nVal.HasValue);

            if (nVal.HasValue)
                bw.Write(nVal.Value);
        }

        /// <summary>
        /// Loads a nullable int from a binary reader.
        /// </summary>
        /// <param name="br">Specifies the binary reader.</param>
        /// <returns>The value read in is returned.</returns>
        public static int? LoadInt(BinaryReader br)
        {
            bool bHasVal = br.ReadBoolean();

            if (!bHasVal)
                return null;

            return br.ReadInt32();
        }

        /// <summary>
        /// Returns the base type size, where <i>double</i> = 8, <i>float</i> = 4.
        /// </summary>
        /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
        /// <returns>The base type size (in bytes) is returned.</returns>
        public static int BaseTypeSize<T>()
        {
            if (typeof(T) == typeof(float))
                return sizeof(float);

            return sizeof(double);
        }

        /// <summary>
        /// Convert a generic to a <i>double</i>.
        /// </summary>
        /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
        /// <param name="fVal">Specifies the generic value.</param>
        /// <returns>The <i>double</i> value is returned.</returns>
        public static double ConvertVal<T>(T fVal)
        {
            return (double)Convert.ChangeType(fVal, typeof(double));
        }

        /// <summary>
        /// Convert a double to a generic.
        /// </summary>
        /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
        /// <param name="dfVal">Specifies the <i>double</i> value.</param>
        /// <returns>The generic value is returned.</returns>
        public static T ConvertVal<T>(double dfVal)
        {
            return (T)Convert.ChangeType(dfVal, typeof(T));
        }

        /// <summary>
        /// Convert an array of generics to an array of <i>double</i>.
        /// </summary>
        /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
        /// <param name="rg">Specifies the array of generics.</param>
        /// <returns>The array of <i>double</i> is returned.</returns>
        public static double[] ConvertVec<T>(T[] rg)
        {
            if (typeof(T) == typeof(double))
                return (double[])Convert.ChangeType(rg, typeof(double[]));

            double[] rgdf = new double[rg.Length];
            Array.Copy(rg, rgdf, rg.Length);

            return rgdf;
        }

        /// <summary>
        /// Convert an array of generics to an array of <i>float</i>.
        /// </summary>
        /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
        /// <param name="rg">Specifies the array of generics.</param>
        /// <returns>The array of <i>float</i> is returned.</returns>
        public static float[] ConvertVecF<T>(T[] rg)
        {
            if (typeof(T) == typeof(float))
                return (float[])Convert.ChangeType(rg, typeof(float[]));

            float[] rgf = new float[rg.Length];
            Array.Copy(Array.ConvertAll(rg, p => Convert.ToSingle(p)), rgf, rgf.Length);

            return rgf;
        }

        /// <summary>
        /// Convert an array of <i>double</i> to an array of generics.
        /// </summary>
        /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
        /// <param name="rgdf">Specifies the array of <i>double</i>.</param>
        /// <returns>The array of generics is returned.</returns>
        public static T[] ConvertVec<T>(double[] rgdf)
        {
            if (typeof(T) == typeof(double))
                return (T[])Convert.ChangeType(rgdf, typeof(T[]));

            T[] rgt = new T[rgdf.Length];

            if (typeof(T) == typeof(float))
                Array.Copy(Array.ConvertAll(rgdf, p => Convert.ToSingle(p)), rgt, rgdf.Length);
            else
                Array.Copy(rgdf, rgt, rgdf.Length);

            return rgt;
        }

        /// <summary>
        /// Convert an array of <i>float</i> to an array of generics.
        /// </summary>
        /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
        /// <param name="rgf">Specifies the array of <i>float</i>.</param>
        /// <returns>The array of generics is returned.</returns>
        public static T[] ConvertVec<T>(float[] rgf)
        {
            if (typeof(T) == typeof(float))
                return (T[])Convert.ChangeType(rgf, typeof(T[]));

            T[] rgt = new T[rgf.Length];
            Array.Copy(rgf, rgt, rgf.Length);

            return rgt;
        }

        /// <summary>
        /// Resize a List and fill the new elements with the default value.
        /// </summary>
        /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
        /// <param name="rg">Specifies the List to resize.</param>
        /// <param name="nCount">Specifies the new count.</param>
        /// <param name="tDefault">Specifies the default value used when expanding the list.</param>
        public static void Resize<T>(ref List<T> rg, int nCount, T tDefault)
        {
            if (rg == null)
                rg = new List<T>();

            while (rg.Count < nCount)
            {
                rg.Add(tDefault);
            }

            while (rg.Count > nCount)
            {
                rg.RemoveAt(rg.Count - 1);
            }
        }

        /// <summary>
        /// Copy an array.
        /// </summary>
        /// <typeparam name="T">Specifies the base type.</typeparam>
        /// <param name="rg">Specifies the source array.</param>
        /// <returns>The new array is returned.</returns>
        public static T[] Clone<T>(T[] rg)
        {
            if (rg == null)
                return null;

            T[] rg1 = new T[rg.Length];
            Array.Copy(rg, rg1, rg.Length);

            return rg1;
        }

        /// <summary>
        /// Copy a List up to a maximum count.
        /// </summary>
        /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
        /// <param name="rg">Specifies the list to copy.</param>
        /// <param name="nMaxCount">Optionally, specifies a maximum count to copy.</param>
        /// <returns>The new copy of the List is returned.</returns>
        public static List<T> Clone<T>(List<T> rg, int nMaxCount = int.MaxValue)
        {
            List<T> rg1 = new List<T>();

            for (int i=0; i<rg.Count && i<nMaxCount; i++)
            {
                ICloneable cloneable = rg[i] as ICloneable;

                if (cloneable != null)
                    rg1.Add((T)Convert.ChangeType(cloneable.Clone(), typeof(T)));
                else
                    rg1.Add(rg[i]);
            }

            return rg1;
        }

        /// <summary>
        /// Compares one List to another.
        /// </summary>
        /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
        /// <param name="rg1">Specifies the first List.</param>
        /// <param name="rg2">Specifies the second List.</param>
        /// <returns>If the Lists are the same, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public static bool Compare<T>(List<T> rg1, List<T> rg2)
        {
            if (rg1.Count != rg2.Count)
                return false;

            for (int i = 0; i < rg1.Count; i++)
            {
                IComparable compare1 = rg1[i] as IComparable;

                if (compare1 != null)
                {
                    if (compare1.CompareTo((object)Convert.ChangeType(rg2[i], typeof(object))) != 0)
                        return false;
                }
                else
                {
                    if (rg1[i].ToString() != rg2[i].ToString())
                        return false;
                }
            }

            return true;
        }

        /// <summary>
        /// Create a new List and fill it with default values up to a given count.
        /// </summary>
        /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
        /// <param name="nCount">Specifies the number of items.</param>
        /// <param name="fDefault">Specifies the default value.</param>
        /// <returns>The new List is returned.</returns>
        public static List<T> Create<T>(int nCount, T fDefault)
        {
            List<T> rg = new List<T>();

            for (int i = 0; i < nCount; i++)
            {
                rg.Add(fDefault);
            }

            return rg;
        }

        /// <summary>
        /// Set all values of a List with a given value.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="rg">Specifies the List.</param>
        /// <param name="fVal">Specifies the value.</param>
        public static void Set<T>(List<T> rg, T fVal)
        {
            for (int i = 0; i < rg.Count; i++)
            {
                rg[i] = fVal;
            }
        }

        /// <summary>
        /// Set all values within an array with a given value.
        /// </summary>
        /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
        /// <param name="rg">Specifies the array to set.</param>
        /// <param name="fVal">Specifies the value.</param>
        public static void Set<T>(T[] rg, T fVal)
        {
            for (int i = 0; i < rg.Length; i++)
            {
                rg[i] = fVal;
            }
        }

        /// <summary>
        /// Parses a string into a number, or if the string does not contain a number returns 0.
        /// </summary>
        /// <param name="str">Specifies the string to parse.</param>
        /// <returns>The parsed number is returned, or if the string does not contan a number, 0 is returned.</returns>
        public static int GetNumber(string str)
        {
            if (str == null || str.Length == 0)
                return 0;

            if (!char.IsNumber(str[str.Length - 1]))
                return 0;

            for (int i = str.Length - 1; i > 0; i--)
            {
                if (!char.IsNumber(str[str.Length - 1]))
                {
                    string strNum = str.Substring(i);
                    return int.Parse(strNum);
                }
            }

            return 0;
        }
    }
}
