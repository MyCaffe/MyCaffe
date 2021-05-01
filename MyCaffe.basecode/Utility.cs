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
        /// Returns the 'canonical' version of a (usually) user-specified axis,
        /// allowing for negative indexing (e.g., -1 for the last axis).
        /// </summary>
        /// <param name="nIdx">The axis index.</param> 
        /// <param name="nNumAxes">The total number of axes.</param>
        /// <returns>The zero based index is returned.</returns>
        public static int CanonicalAxisIndex(int nIdx, int nNumAxes)
        {
            if (nIdx < 0)
                return nIdx + nNumAxes;
            else
                return nIdx;
        }

        /// <summary>
        /// Calculate the spatial dimension of an array starting at a given axis.
        /// </summary>
        /// <param name="rg">Specifies the shape to measure.</param>
        /// <param name="nStartIdx">Specifies the starting axis.</param>
        /// <returns>The spacial dimension is returned.</returns>
        public static int GetSpatialDim(List<int> rg, int nStartIdx = 0)
        {
            int nDim = 1;

            for (int i = nStartIdx; i < rg.Count; i++)
            {
                nDim *= rg[i];
            }

            return nDim;
        }

        /// <summary>
        /// Return the count of items given the shape.
        /// </summary>
        /// <param name="rgShape">Specifies the shape to count from the start index through the end index.</param>
        /// <param name="nStartIdx">Specifies the start index (default = 0).</param>
        /// <param name="nEndIdx">Specifies the end index (default = -1, which uses length of rgShape).</param>
        /// <returns>The count is returned.</returns>
        public static int Count(List<int> rgShape, int nStartIdx = 0, int nEndIdx = -1)
        {
            int nCount = 1;

            if (nEndIdx == -1)
                nEndIdx = rgShape.Count;

            for (int i = nStartIdx; i < nEndIdx; i++)
            {
                nCount *= rgShape[i];
            }

            return nCount;
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
        /// <param name="nStart">Specifies a start offset (default = 0).</param>
        /// <returns>The array of <i>float</i> is returned.</returns>
        public static float[] ConvertVecF<T>(T[] rg, int nStart = 0)
        {
            if (typeof(T) == typeof(float) && nStart == 0)
                return (float[])Convert.ChangeType(rg, typeof(float[]));

            float[] rgf = new float[rg.Length - nStart];
            Array.Copy(Array.ConvertAll(rg, p => Convert.ToSingle(p)), nStart, rgf, 0, rgf.Length);

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
        /// Copy a List up to a maximum count.
        /// </summary>
        /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
        /// <param name="rg">Specifies the array to copy.</param>
        /// <param name="nMaxCount">Optionally, specifies a maximum count to copy.</param>
        /// <returns>The new copy of the List is returned.</returns>
        public static List<T> Clone<T>(T[] rg, int nMaxCount = int.MaxValue)
        {
            List<T> rg1 = new List<T>();

            for (int i = 0; i < rg.Length && i < nMaxCount; i++)
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
        /// <param name="bExact">Optionally, specifies to look for an exact match.  When <i>false</i> a trailing one is accepted if the count()'s match.</param>
        /// <returns>If the Lists are the same, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public static bool Compare<T>(List<T> rg1, List<T> rg2, bool bExact = true)
        {
            if (rg1.Count != rg2.Count)
            {
                if (bExact)
                    return false;

                if (Math.Abs(rg1.Count - rg2.Count) > 1)
                    return false;

                T tOne = (T)Convert.ChangeType(1, typeof(T));

                if (rg1.Count > rg2.Count && !rg1[rg1.Count - 1].Equals(tOne))
                    return false;

                if (rg2.Count > rg1.Count && !rg2[rg2.Count - 1].Equals(tOne))
                    return false;                
            }

            for (int i = 0; i < rg1.Count && i < rg2.Count; i++)
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
        /// Create a new List and fill it with values starting with start and incrementing by inc.
        /// </summary>
        /// <param name="nCount">Specifies the number of items.</param>
        /// <param name="nStart">Specifies the start value.</param>
        /// <param name="nInc">Specifies the increment added to the last value added.</param>
        /// <returns>The new List is returned.</returns>
        public static List<int> Create(int nCount, int nStart, int nInc)
        {
            List<int> rg = new List<int>();
            int nVal = nStart;

            for (int i = 0; i < nCount; i++)
            {
                rg.Add(nVal);
                nVal += nInc;
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
        /// Convert an array to a string.
        /// </summary>
        /// <typeparam name="T">Specifies the base type of the array.</typeparam>
        /// <param name="rg">Specifies the array.</param>
        /// <param name="nDecimals">Optionally, specifies the number of decimals (default = -1, ignored)</param>
        /// <param name="nIdxHighight">Optionally, specifies the index to highlight (default = -1, ignored)</param>
        /// <returns>The string representation of the array is returned.</returns>
        public static string ToString<T>(List<T> rg, int nDecimals = -1, int nIdxHighight = -1)
        {
            string strOut = "{";

            for (int i = 0; i < rg.Count; i++)
            {
                if (nIdxHighight >= 0 && i == nIdxHighight)
                    strOut += "[*";

                if (nDecimals >= 0)
                {
                    if (typeof(T) == typeof(float))
                    {
                        float f = (float)Convert.ChangeType(rg[i], typeof(float));
                        strOut += f.ToString("N" + nDecimals.ToString());
                    }
                    else if (typeof(T) == typeof(double))
                    {
                        double f = (double)Convert.ChangeType(rg[i], typeof(double));
                        strOut += f.ToString("N" + nDecimals.ToString());
                    }
                    else
                    {
                        strOut += rg[i].ToString();
                    }
                }
                else
                    strOut += rg[i].ToString();

                if (nIdxHighight >= 0 && i == nIdxHighight)
                    strOut += "*]";

                strOut += ",";
            }

            strOut = strOut.TrimEnd(',');
            strOut += "}";

            return strOut;
        }

        /// <summary>
        /// Parses a comma delimited string into an array of int.
        /// </summary>
        /// <param name="str">Specifies the string to parse.</param>
        /// <returns>The array of int is returned.</returns>
        public static List<int> ParseListToInt(string str)
        {
            string[] rg = str.Split(',');
            List<int> rgVal = new List<int>();

            foreach (string str1 in rg)
            {
                rgVal.Add(int.Parse(str1));
            }

            return rgVal;
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

        /// <summary>
        /// Replaces each instance of one character with another character in a given string.
        /// </summary>
        /// <param name="str">Specifies the string.</param>
        /// <param name="ch1">Specifies the character to find.</param>
        /// <param name="ch2">Specifies the character replacement.</param>
        /// <returns>The new string is returned.</returns>
        public static string Replace(string str, char ch1, char ch2)
        {
            if (str == null)
                return null;

            string strOut = "";

            foreach (char ch in str)
            {
                if (ch == ch1)
                    strOut += ch2;
                else
                    strOut += ch;
            }

            return strOut;
        }

        /// <summary>
        /// Replaces each instance of one character with another string in a given string.
        /// </summary>
        /// <param name="str">Specifies the string.</param>
        /// <param name="ch1">Specifies the character to find.</param>
        /// <param name="str2">Specifies the string replacement.</param>
        /// <returns>The new string is returned.</returns>
        public static string Replace(string str, char ch1, string str2)
        {
            if (str == null)
                return null;

            string strOut = "";

            foreach (char ch in str)
            {
                if (ch == ch1)
                    strOut += str2;
                else
                    strOut += ch;
            }

            return strOut;
        }

        /// <summary>
        /// Replaces each instance of one character with another string in a given string.
        /// </summary>
        /// <param name="str">Specifies the string.</param>
        /// <param name="str1">Specifies the string to find.</param>
        /// <param name="ch2">Specifies the char replacement.</param>
        /// <returns>The new string is returned.</returns>
        public static string Replace(string str, string str1, char ch2)
        {
            if (str == null)
                return null;

            string strOut = "";

            while (str.Length > 0)
            {
                int nPos = str.IndexOf(str1);
                if (nPos >= 0)
                {
                    strOut += str.Substring(0, nPos);
                    strOut += ch2;
                    str = str.Substring(nPos + str1.Length);
                }
                else
                {
                    strOut += str;
                    str = "";
                }
            }

            return strOut;
        }

        /// <summary>
        /// The <c>ConvertMacro</c> method is used to replace a set of macros in a given string.
        /// </summary>
        /// <param name="strRaw">Specifies the raw string.</param>
        /// <param name="strMacroName">Specifies the macro to be replaced.</param>
        /// <param name="strReplacement">Specifies the replacement string.</param>
        /// <returns>The new string with all macros replaced is returned.</returns>
        public static string ReplaceMacro(string strRaw, string strMacroName, string strReplacement)
        {
            int nPos = 0;
            string strOut = "";

            while (strRaw.Length > 0)
            {
                nPos = strRaw.IndexOf(strMacroName);

                if (nPos >= 0)
                {
                    strOut += strRaw.Substring(0, nPos);
                    strOut += strReplacement;
                    strRaw = strRaw.Substring(nPos + strMacroName.Length);
                }
                else
                {
                    strOut += strRaw;
                    strRaw = "";
                }
            }

            return strOut;
        }

        /// <summary>
        /// The <c>ReplaceMacros</c> method is used to replace a set of macros in a given string.
        /// </summary>
        /// <param name="strRaw">Specifies the raw string.</param>
        /// <param name="rgMacros">Specifies the set of macros.</param>
        /// <returns>The new string with the macros replaced, is returned.</returns>
        public static string ReplaceMacros(string strRaw, List<KeyValuePair<string, string>> rgMacros)
        {
            string strOut = strRaw;

            foreach (KeyValuePair<string, string> kv in rgMacros)
            {
                strOut = ReplaceMacro(strOut, kv.Key, kv.Value);
            }

            return strOut;
        }


        /// <summary>
        /// Convert a date time into minutes since 1/1/1980
        /// </summary>
        /// <param name="dt">Specifies the datetime to convert.</param>
        /// <returns>The minutes since 1/1/1980 is returned.</returns>
        public static double ConvertTimeToMinutes(DateTime dt)
        {
            DateTime dt1 = new DateTime(1980, 1, 1);
            TimeSpan ts = dt - dt1;
            return ts.TotalMinutes;
        }

        /// <summary>
        /// Convert a number of minutes into the date time equivalent to 1/1/1980 + the minutes.
        /// </summary>
        /// <param name="dfMin">Specifies the minutes since 1/1/1980.</param>
        /// <returns>The datetime is returned.</returns>
        public static DateTime ConvertTimeFromMinutes(double dfMin)
        {
            DateTime dt1 = new DateTime(1980, 1, 1);
            TimeSpan ts = TimeSpan.FromMinutes(dfMin);
            return dt1 + ts;
        }

        /// <summary>
        /// Randomly shuffle the entries in the specified list.
        /// </summary>
        /// <param name="rg">Specifies the input list to shuffle.</param>
        /// <param name="nSeed">Optionally, specifies a seed for the random generator.</param>
        /// <returns>The newly shuffled list is returned.</returns>
        public static List<int> RandomShuffle(List<int> rg, int? nSeed = null)
        {
            if (!nSeed.HasValue)
                nSeed = (int)DateTime.Now.Ticks;

            Random random = new Random(nSeed.Value);
            List<int> rg1 = new List<int>();

            while (rg.Count > 0)
            {
                if (rg.Count == 1)
                {
                    rg1.Add(rg[0]);
                    rg.Clear();
                }
                else
                {
                    int nIdx = random.Next(rg.Count);
                    rg1.Add(rg[nIdx]);
                    rg.RemoveAt(nIdx);
                }
            }

            return rg1;
        }

        /// <summary>
        /// Load each line of a text file and return the contents as a list.
        /// </summary>
        /// <param name="strFile">Specifies the text file to load.</param>
        /// <param name="log">Optionally, specifies the output log used to output errors (default = null).  When null, any errors are thrown as exceptions.</param>
        /// <param name="bPrependPath">Optionallly, specifies to prepend the path of the 'strFile' to each file within the file, but only do so with entries starting with '.'</param>
        /// <returns>A list containing each line of the text file is returned.</returns>
        public static List<string> LoadTextLines(string strFile, Log log = null, bool bPrependPath = true)
        {
            List<string> rgstr = new List<string>();

            try
            {
                string strPath = Path.GetDirectoryName(strFile);

                using (StreamReader sr = new StreamReader(strFile))
                {
                    string strLine = sr.ReadLine();

                    while (strLine != null)
                    {
                        if (strLine.Length > 0)
                        {
                            if (strLine[0] == '.' && bPrependPath)
                            {
                                int nPos = strLine.LastIndexOf('/');
                                if (nPos < 0)
                                    nPos = strLine.LastIndexOf('\\');

                                if (nPos >= 0)
                                    strLine = strLine.Substring(nPos + 1);

                                strLine = strPath + "\\" + Replace(strLine, '/', '\\');
                            }

                            rgstr.Add(strLine);
                        }

                        strLine = sr.ReadLine();
                    }
                }
            }
            catch (Exception excpt)
            {
                if (log != null)
                    log.FAIL("Failed to load '" + strFile + "'!  Error: " + excpt.Message);
                else
                    throw excpt;
            }

            return rgstr;
        }
    }
}
