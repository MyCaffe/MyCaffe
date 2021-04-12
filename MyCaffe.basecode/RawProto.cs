using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;

namespace MyCaffe.basecode
{
    /// <summary>
    /// The RawProto class is used to parse and output Google prototxt file data.
    /// </summary>
    /// <remarks>
    /// Each RawProto instance forms a tree of RawProto instances where each Leaf contains the data and branches
    /// contain a collection of child RawProto's.
    /// </remarks>
    public class RawProto
    {
        TYPE m_type = TYPE.NONE;
        string m_strName;
        string m_strValue;
        RawProtoCollection m_rgChildren = new RawProtoCollection();

        /// <summary>
        /// Defines the type of a RawProto node.
        /// </summary>
        public enum TYPE
        {
            /// <summary>
            /// Brach node.
            /// </summary>
            NONE,
            /// <summary>
            /// Numeric leaf node.
            /// </summary>
            NUMERIC,
            /// <summary>
            /// Numeric string node.
            /// </summary>
            STRING
        }

        enum STATE
        {
            NAME,
            VALUE,
            BLOCKSTART,
            BLOCKEND
        }

        /// <summary>
        /// The RawProto constructor.
        /// </summary>
        /// <param name="strName">Specifies the name of the node.</param>
        /// <param name="strValue">Specifies the value of the node.</param>
        /// <param name="rgChildren">Specifies the children nodes of this node (if any).</param>
        /// <param name="type">Specifies the type of the node.</param>
        public RawProto(string strName, string strValue, RawProtoCollection rgChildren = null, TYPE type = TYPE.NONE)
        {
            m_type = type;
            m_strName = strName;
            m_strValue = strValue;

            if (rgChildren != null)
                m_rgChildren = rgChildren;
        }

        /// <summary>
        /// Returns the name of the node.
        /// </summary>
        public string Name
        {
            get { return m_strName; }
        }

        /// <summary>
        /// Get/set the value of the node.
        /// </summary>
        public string Value
        {
            get { return m_strValue; }
            set { m_strValue = value; }
        }

        /// <summary>
        /// Returns the type of the node.
        /// </summary>
        public TYPE Type
        {
            get { return m_type; }
        }

        /// <summary>
        /// Returns a collection of this nodes child nodes.
        /// </summary>
        public RawProtoCollection Children
        {
            get { return m_rgChildren; }
        }

        /// <summary>
        /// Searches for a falue of a node within this nodes children.
        /// </summary>
        /// <param name="strName">Specifies the name of the node to look for.</param>
        /// <returns>If found, the value of the child is returned, otherwise <i>null</i> is returned.</returns>
        public string FindValue(string strName)
        {
            foreach (RawProto p in m_rgChildren)
            {
                if (p.Name == strName)
                    return p.Value.Trim('\"');
            }

            return null;
        }

        /// <summary>
        /// Searches for a value of a node within this nodes children and return it as a given type.
        /// </summary>
        /// <param name="strName">Specifies the name of the node to look for.</param>
        /// <param name="t">Specifies the type to convert the value, if found.</param>
        /// <returns>If found, the value of the child is returned, otherwise <i>null</i> is returned.</returns>
        public object FindValue(string strName, Type t)
        {
            string strVal = FindValue(strName);

            if (strVal == null)
                return null;

            return convert(strVal, t);
        }

        /// <summary>
        /// Searches for all values of a given name within this nodes children and return it as a generic List.
        /// </summary>
        /// <typeparam name="T">Specifies the type of item to return.</typeparam>
        /// <param name="strName">Specifies the name of the nodes to look for.</param>
        /// <returns>If found, the generic List of values of the children found is returned.</returns>
        public List<T> FindArray<T>(string strName)
        {
            List<T> rg = new List<T>();

            foreach (RawProto rp in m_rgChildren)
            {
                if (rp.Name == strName)
                {
                    object obj = convert(rp.Value, typeof(T));
                    T tVal = (T)Convert.ChangeType(obj, typeof(T));
                    rg.Add(tVal);
                }
            }

            return rg;
        }

        private object convert(string strVal, Type t)
        {
            strVal = strVal.TrimEnd('}');

            if (t == typeof(string))
                return strVal;

            if (t == typeof(bool))
                return bool.Parse(strVal);

            if (t == typeof(double))
                return BaseParameter.ParseDouble(strVal);

            if (t == typeof(float))
                return BaseParameter.ParseFloat(strVal);

            if (t == typeof(long))
                return long.Parse(strVal);

            if (t == typeof(int))
                return int.Parse(strVal);

            if (t == typeof(uint))
                return uint.Parse(strVal);

            throw new Exception("The type '" + t.ToString() + "' is not supported by the FindArray<T> function!");
        }

        /// <summary>
        /// Removes a given child from this node's children.
        /// </summary>
        /// <param name="p">Specifies the RawProto to remove.</param>
        /// <returns>If found and removed, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool RemoveChild(RawProto p)
        {
            return m_rgChildren.Remove(p);
        }

        /// <summary>
        /// Removes a given child with a set value from this node's children.
        /// </summary>
        /// <param name="strName">Specifes the name of the node.</param>
        /// <param name="strValue">Specifies the value to match.</param>
        /// <param name="bContains">Optionally, specifies whether just 'containing' the value (as opposed to equallying the value) is enough to delete the bottom.</param>
        /// <returns>If the named node is found and its value matches <i>strValue</i>, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool RemoveChild(string strName, string strValue, bool bContains = false)
        {
            int nIdx = -1;

            for (int i = 0; i < m_rgChildren.Count; i++)
            {
                if (m_rgChildren[i].Name == strName)
                {
                    if ((bContains && m_rgChildren[i].Value.Contains(strValue)) ||
                        (!bContains && m_rgChildren[i].Value == strValue))
                    {
                        nIdx = i;
                        break;
                    }
                }
            }

            if (nIdx >= 0)
            {
                m_rgChildren.RemoveAt(nIdx);
                return true;
            }

            return false;
        }

        /// <summary>
        /// Searches for a given node.
        /// </summary>
        /// <param name="strName">Specifies the name of the node to look for.</param>
        /// <returns>If found, the node is returned, otherwise <i>null</i> is returned.</returns>
        public RawProto FindChild(string strName)
        {
            foreach (RawProto p in m_rgChildren)
            {
                if (p.Name == strName)
                    return p;
            }

            return null;
        }

        /// <summary>
        /// Searches for the index to a given node's child.
        /// </summary>
        /// <param name="strName">Specifies the name of the child node to look for.</param>
        /// <returns>If found, the index of the child node is returned, otherwise -1 is returned.</returns>
        public int FindChildIndex(string strName)
        {
            for (int i = 0; i < m_rgChildren.Count; i++)
            {
                if (m_rgChildren[i].Name == strName)
                    return i;
            }

            return -1;
        }

        /// <summary>
        /// Searches for all children with a given name in this node's children.
        /// </summary>
        /// <param name="rgstrName">Specifies a array of names to look for.</param>
        /// <returns>The collection of child nodes found is returned.</returns>
        public RawProtoCollection FindChildren(params string[] rgstrName)
        {
            RawProtoCollection rg = new RawProtoCollection();

            foreach (RawProto p in m_rgChildren)
            {
                if (rgstrName.Contains(p.Name))
                    rg.Add(p);
            }

            return rg;
        }

        /// <summary>
        /// Parses a prototxt from a file and returns it as a RawProto.
        /// </summary>
        /// <param name="strFileName">Specifies the file name.</param>
        /// <returns>The new RawProto is returned.</returns>
        public static RawProto FromFile(string strFileName)
        {
            using (StreamReader sr = new StreamReader(strFileName))
            {
                return Parse(sr.ReadToEnd());
            }
        }

        /// <summary>
        /// Saves a RawProto to a file.
        /// </summary>
        /// <param name="strFileName">Specifies the file name.</param>
        public void ToFile(string strFileName)
        {
            using (StreamWriter sw = new StreamWriter(strFileName))
            {
                sw.Write(ToString());
            }
        }

        /// <summary>
        /// Parses a prototxt and places it in a new RawProto.
        /// </summary>
        /// <param name="str">Specifies the prototxt to parse.</param>
        /// <returns>The new RawProto is returned.</returns>
        public static RawProto Parse(string str)
        {
            List<RawProto> rgParent = new List<RawProto>() { new RawProto("root", "") };
            RawProto child = new RawProto("", "");

            str = strip_comments(str);

            List<KeyValuePair<string, int>> rgstrTokens = tokenize(str);

            parse(rgParent, child, rgstrTokens, 0, STATE.NAME);

            return rgParent[0];
        }

        private static string strip_comments(string str)
        {
            if (!str.Contains('#'))
                return str;

            string[] rgstr = str.Split('\n', '\r');
            string strOut = "";

            for (int i = 0; i < rgstr.Length; i++)
            {
                if (rgstr[i].Length > 0)
                {
                    int nPos = rgstr[i].IndexOf('#');
                    if (nPos >= 0)
                        rgstr[i] = rgstr[i].Substring(0, nPos);
                }

                if (rgstr[i].Length > 0)
                    strOut += rgstr[i] + "\r\n";
            }

            return strOut;
        }

        private static string strip_commentsOld(string str)
        {
            List<string> rgstr = new List<string>();
            int nPos = str.IndexOf('\n');

            while (nPos >= 0)
            {
                string strLine = str.Substring(0, nPos);
                strLine = strLine.Trim('\n', '\r');

                int nPosComment = strLine.IndexOf('#');
                if (nPosComment >= 0)
                    strLine = strLine.Substring(0, nPosComment);

                if (strLine.Length > 0)
                    rgstr.Add(strLine);

                str = str.Substring(nPos + 1);
                nPos = str.IndexOf('\n');
            }

            str = str.Trim('\n', '\r');
            if (str.Length > 0)
                rgstr.Add(str);

            str = "";

            foreach (string strLine in rgstr)
            {
                str += strLine;
                str += " \r\n";
            }

            return str;
        }

        private static List<KeyValuePair<string, int>> tokenize(string str)
        {
            List<KeyValuePair<string, int>> rgstrTokens = new List<KeyValuePair<string, int>>();
            string[] rgLines = str.Split('\n');

            for (int i=0; i<rgLines.Length; i++)
            {
                string strLine = rgLines[i].Trim(' ', '\r', '\t');
                string[] rgTokens = strLine.Split(' ', '\t');
                List<string> rgFinalTokens = new List<string>();
                bool bAdding = false;
                string strItem1 = "";

                foreach (string strItem in rgTokens)
                {
                    if (strItem.Length > 0 && strItem[0] == '\'')
                    {
                        bAdding = true;
                        strItem1 = strItem.TrimStart('\'');

                        if (strItem1.Contains('\''))
                        {
                            strItem1.TrimEnd('\'');
                            rgFinalTokens.Add(strItem1);
                            bAdding = false;
                        }
                    }
                    else if (bAdding && strItem.Contains('\''))
                    {
                        int nPos = strItem.IndexOf('\'');
                        strItem1 += strItem.Substring(0, nPos);
                        rgFinalTokens.Add(strItem1);

                        strItem1 = strItem.Substring(nPos + 1);
                        strItem1 = strItem1.Trim(' ', '\n', '\r', '\t');
                        if (strItem1.Length > 0)
                            rgFinalTokens.Add(strItem1);

                        strItem1 = "";
                        bAdding = false;
                    }
                    else
                    {
                        rgFinalTokens.Add(strItem);
                    }
                }

                foreach (string strItem in rgFinalTokens)
                {
                    string strItem0 = strItem.Trim(' ', '\n', '\r', '\t');

                    if (strItem0.Length > 0)
                        rgstrTokens.Add(new KeyValuePair<string, int>(strItem0, i));
                }
            }

            return rgstrTokens;
        }

        private static string removeCommentsOld(string str)
        {
            string[] rgstr = str.Split('\n');
            string strOut = "";

            foreach (string strLine in rgstr)
            {
                string strLine0 = strLine.Trim(' ', '\n', '\r', '\t');

                if (strLine0.Length > 0 && strLine0[0] != '#')
                {
                    strOut += strLine;
                    strOut += Environment.NewLine;
                }
            }

            return strOut;
        }

        private static void parse(List<RawProto> rgParent, RawProto child, List<KeyValuePair<string, int>> rgstrTokens, int nIdx, STATE s)
        {
            while (nIdx < rgstrTokens.Count)
            {
                KeyValuePair<string, int> kvToken = rgstrTokens[nIdx];
                string strToken = kvToken.Key;
                int nLine = kvToken.Value;

                if (s == STATE.NAME)
                {
                    if (!char.IsLetter(strToken[0]))
                        throw new Exception("Expected a name and instead have: " + rgstrTokens[nIdx]);

                    STATE sNext;

                    if (strToken[strToken.Length - 1] == ':')
                    {
                        child.m_strName = strToken.TrimEnd(':');
                        sNext = STATE.VALUE;
                    }
                    else
                    {
                        child.m_strName = strToken;
                        sNext = STATE.BLOCKSTART;
                    }

                    nIdx++;

                    if (nIdx >= rgstrTokens.Count)
                        return;

                    if (rgstrTokens[nIdx].Key == "{")
                        s = STATE.BLOCKSTART;
                    else if (sNext == STATE.VALUE)
                        s = STATE.VALUE;
                    else
                        throw new Exception("line (" + rgstrTokens[nIdx].Value.ToString() + ") - Unexpected token: '" + rgstrTokens[nIdx].Key + "'");
                }
                else if (s == STATE.VALUE)
                {
                    TYPE type = TYPE.NUMERIC;

                    strToken = strToken.Trim(' ', '\t');

                    if (strToken[0] == '"' || strToken[0] == '\'')
                        type = TYPE.STRING;

                    child.m_strValue = strToken.Trim('"', '\'');
                    child.m_type = type;
                    nIdx++;

                    rgParent[0].Children.Add(child);

                    if (nIdx >= rgstrTokens.Count)
                        return;

                    if (char.IsLetter(rgstrTokens[nIdx].Key[0]))
                    {
                        child = new RawProto("", "");
                        s = STATE.NAME;
                    }
                    else if (rgstrTokens[nIdx].Key == "}")
                    {
                        s = STATE.BLOCKEND;
                    }
                    else
                        throw new Exception("line (" + rgstrTokens[nIdx].Value.ToString() + ") - Unexpected token: '" + rgstrTokens[nIdx].Key + "'");
                }
                else if (s == STATE.BLOCKSTART)
                {
                    rgParent[0].Children.Add(child);
                    rgParent.Insert(0, child);
                    child = new RawProto("", "");
                    nIdx++;

                    if (char.IsLetter(rgstrTokens[nIdx].Key[0]))
                        s = STATE.NAME;
                    else if (rgstrTokens[nIdx].Key == "}")
                        s = STATE.BLOCKEND;
                    else
                        throw new Exception("line (" + rgstrTokens[nIdx].Value.ToString() + ") - Unexpected token: '" + rgstrTokens[nIdx].Key + "'");
                }
                else if (s == STATE.BLOCKEND)
                {
                    child = rgParent[0];
                    rgParent.RemoveAt(0);

                    nIdx++;

                    if (nIdx >= rgstrTokens.Count)
                        return;

                    if (char.IsLetter(rgstrTokens[nIdx].Key[0]))
                    {
                        child = new RawProto("", "");
                        s = STATE.NAME;
                    }
                    else if (rgstrTokens[nIdx].Key == "}")
                    {
                        s = STATE.BLOCKEND;
                    }
                    else
                        throw new Exception("line (" + rgstrTokens[nIdx].Value.ToString() + ") - Unexpected token: '" + rgstrTokens[nIdx].Key + "'");
                }
            }
        }

        private static void parse2(List<RawProto> rgParent, RawProto child, List<KeyValuePair<string, int>> rgstrTokens, int nIdx, STATE s)
        {
            if (nIdx >= rgstrTokens.Count)
                return;

            string strToken = rgstrTokens[nIdx].Key;

            if (s == STATE.NAME)
            {
                if (!char.IsLetter(strToken[0]))
                    throw new Exception("Expected a name and instead have: " + rgstrTokens[nIdx]);

                STATE sNext;

                if (strToken[strToken.Length - 1] == ':')
                {
                    child.m_strName = strToken.TrimEnd(':');
                    sNext = STATE.VALUE;
                }
                else
                {
                    child.m_strName = strToken;
                    sNext = STATE.BLOCKSTART;
                }

                nIdx++;

                if (nIdx >= rgstrTokens.Count)
                    return;

                if (sNext == STATE.VALUE)
                    parse(rgParent, child, rgstrTokens, nIdx, STATE.VALUE);
                else if (rgstrTokens[nIdx].Key == "{")
                    parse(rgParent, child, rgstrTokens, nIdx, STATE.BLOCKSTART);
                else
                    throw new Exception("line (" + rgstrTokens[nIdx].Value.ToString() + ") - Unexpected token: '" + rgstrTokens[nIdx].Key + "'");
            }
            else if (s == STATE.VALUE)
            {
                TYPE type = TYPE.NUMERIC;

                strToken = strToken.Trim(' ', '\t');

                if (strToken[0] == '"' || strToken[0] == '\'')
                    type = TYPE.STRING;

                child.m_strValue = strToken.Trim('"', '\'');
                child.m_type = type;
                nIdx++;

                rgParent[0].Children.Add(child);

                if (nIdx >= rgstrTokens.Count)
                    return;

                if (char.IsLetter(rgstrTokens[nIdx].Key[0]))
                {
                    child = new RawProto("", "");
                    parse(rgParent, child, rgstrTokens, nIdx, STATE.NAME);
                }
                else if (rgstrTokens[nIdx].Key == "}")
                {
                    parse(rgParent, child, rgstrTokens, nIdx, STATE.BLOCKEND);
                }
                else
                    throw new Exception("line (" + rgstrTokens[nIdx].Value.ToString() + ") - Unexpected token: '" + rgstrTokens[nIdx].Key + "'");
            }
            else if (s == STATE.BLOCKSTART)
            {
                rgParent[0].Children.Add(child);
                rgParent.Insert(0, child);
                child = new RawProto("", "");
                nIdx++;

                if (char.IsLetter(rgstrTokens[nIdx].Key[0]))
                    parse(rgParent, child, rgstrTokens, nIdx, STATE.NAME);
                else if (rgstrTokens[nIdx].Key == "}")
                    parse(rgParent, child, rgstrTokens, nIdx, STATE.BLOCKEND);
                else
                    throw new Exception("line (" + rgstrTokens[nIdx].Value.ToString() + ") - Unexpected token: '" + rgstrTokens[nIdx].Key + "'");
            }
            else if (s == STATE.BLOCKEND)
            {
                child = rgParent[0];
                rgParent.RemoveAt(0);

                nIdx++;

                if (nIdx >= rgstrTokens.Count)
                    return;

                if (char.IsLetter(rgstrTokens[nIdx].Key[0]))
                {
                    child = new RawProto("", "");
                    parse(rgParent, child, rgstrTokens, nIdx, STATE.NAME);
                }
                else if (rgstrTokens[nIdx].Key == "}")
                {
                    parse(rgParent, child, rgstrTokens, nIdx, STATE.BLOCKEND);
                }
                else
                    throw new Exception("line (" + rgstrTokens[nIdx].Value.ToString() + ") - Unexpected token: '" + rgstrTokens[nIdx].Key + "'");
            }
        }

        /// <summary>
        /// Returns the RawProto as its full prototxt string.
        /// </summary>
        /// <returns>The full prototxt string representing the RawProto is returned.</returns>
        public override string ToString()
        {
            if (m_strName != "root")
                return toString(this, "");

            string str = "";

            foreach (RawProto child in m_rgChildren)
            {
                str += child.ToString();
            }

            return str;
        }

        private string toString(RawProto rp, string strIndent)
        {
            if ((rp.Value == null || rp.Value.Length == 0) && rp.Children.Count == 0)
                return "";

            string str = strIndent + rp.Name;

            if (rp.Value.Length > 0)
            {
                str += ": ";

                if (rp.Type == TYPE.STRING)
                    str += "\"";

                str += rp.Value;

                if (rp.Type == TYPE.STRING)
                    str += "\"";
            }
            else
            {
                str += " ";
            }

            str += Environment.NewLine;

            if (rp.Children.Count == 0)
                return str;

            str += strIndent + "{";
            str += Environment.NewLine;

            foreach (RawProto child in rp.m_rgChildren)
            {
                str += toString(child, strIndent + "   ");
            }

            str += strIndent + "}";
            str += Environment.NewLine;

            return str;
        }
    }
}
