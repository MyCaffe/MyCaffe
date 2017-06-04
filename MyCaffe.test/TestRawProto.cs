using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.basecode;

namespace MyCaffe.test
{
    [TestClass]
    public class TestRawProto
    {
        [TestMethod]
        public void TestSimpleParsing1()
        {
            RawProto rp;
            string str;
            
            str = "name: 'foo'";
            rp = RawProto.Parse(str);

            Assert.AreEqual(rp.Name, "root");
            Assert.AreEqual(rp.Value, "");
            Assert.AreEqual(rp.Children.Count, 1);
            Assert.AreEqual(rp.Children[0].Name, "name");
            Assert.AreEqual(rp.Children[0].Value, "foo");

            str = "name { type: 'boo' }";
            rp = RawProto.Parse(str);

            Assert.AreEqual(rp.Name, "root");
            Assert.AreEqual(rp.Value, "");
            Assert.AreEqual(rp.Children.Count, 1);
            Assert.AreEqual(rp.Children[0].Name, "name");
            Assert.AreEqual(rp.Children[0].Value, "");
            Assert.AreEqual(rp.Children[0].Children.Count, 1);
            Assert.AreEqual(rp.Children[0].Children[0].Name, "type");
            Assert.AreEqual(rp.Children[0].Children[0].Value, "boo");

            str =  "name {" + Environment.NewLine;
            str += "   type: 'boo'" + Environment.NewLine;
            str += "}";
            rp = RawProto.Parse(str);

            Assert.AreEqual(rp.Name, "root");
            Assert.AreEqual(rp.Value, "");
            Assert.AreEqual(rp.Children.Count, 1);
            Assert.AreEqual(rp.Children[0].Name, "name");
            Assert.AreEqual(rp.Children[0].Value, "");
            Assert.AreEqual(rp.Children[0].Children.Count, 1);
            Assert.AreEqual(rp.Children[0].Children[0].Name, "type");
            Assert.AreEqual(rp.Children[0].Children[0].Value, "boo");

            str = "name " + Environment.NewLine;
            str += "{ " + Environment.NewLine;
            str += "   type: 'boo'" + Environment.NewLine;
            str += "}";
            rp = RawProto.Parse(str);

            Assert.AreEqual(rp.Name, "root");
            Assert.AreEqual(rp.Value, "");
            Assert.AreEqual(rp.Children.Count, 1);
            Assert.AreEqual(rp.Children[0].Name, "name");
            Assert.AreEqual(rp.Children[0].Value, "");
            Assert.AreEqual(rp.Children[0].Children.Count, 1);
            Assert.AreEqual(rp.Children[0].Children[0].Name, "type");
            Assert.AreEqual(rp.Children[0].Children[0].Value, "boo");

            str = "name " + Environment.NewLine;
            str += "{ " + Environment.NewLine;
            str += "   type: 'boo' }" + Environment.NewLine;
            rp = RawProto.Parse(str);

            Assert.AreEqual(rp.Name, "root");
            Assert.AreEqual(rp.Value, "");
            Assert.AreEqual(rp.Children.Count, 1);
            Assert.AreEqual(rp.Children[0].Name, "name");
            Assert.AreEqual(rp.Children[0].Value, "");
            Assert.AreEqual(rp.Children[0].Children.Count, 1);
            Assert.AreEqual(rp.Children[0].Children[0].Name, "type");
            Assert.AreEqual(rp.Children[0].Children[0].Value, "boo");
        }

        [TestMethod]
        public void TestSimpleParsing2()
        {
            RawProto rp;
            string str;

            str = "name { type: 'boo' type2: 'moo' }";
            rp = RawProto.Parse(str);

            string str1 = rp.ToString();

            Assert.AreEqual(rp.Name, "root");
            Assert.AreEqual(rp.Value, "");
            Assert.AreEqual(rp.Children.Count, 1);
            Assert.AreEqual(rp.Children[0].Name, "name");
            Assert.AreEqual(rp.Children[0].Value, "");
            Assert.AreEqual(rp.Children[0].Children.Count, 2);
            Assert.AreEqual(rp.Children[0].Children[0].Name, "type");
            Assert.AreEqual(rp.Children[0].Children[0].Value, "boo");
            Assert.AreEqual(rp.Children[0].Children[1].Name, "type2");
            Assert.AreEqual(rp.Children[0].Children[1].Value, "moo");

            str = "name {" + Environment.NewLine;
            str += "   type: 'boo'" + Environment.NewLine;
            str += "   type2: 'moo'" + Environment.NewLine;
            str += "}";
            rp = RawProto.Parse(str);

            Assert.AreEqual(rp.Name, "root");
            Assert.AreEqual(rp.Value, "");
            Assert.AreEqual(rp.Children.Count, 1);
            Assert.AreEqual(rp.Children[0].Name, "name");
            Assert.AreEqual(rp.Children[0].Value, "");
            Assert.AreEqual(rp.Children[0].Children.Count, 2);
            Assert.AreEqual(rp.Children[0].Children[0].Name, "type");
            Assert.AreEqual(rp.Children[0].Children[0].Value, "boo");
            Assert.AreEqual(rp.Children[0].Children[1].Name, "type2");
            Assert.AreEqual(rp.Children[0].Children[1].Value, "moo");

            str = "name " + Environment.NewLine;
            str += "{ " + Environment.NewLine;
            str += "   type: 'boo' type2: 'moo'" + Environment.NewLine;
            str += "}";
            rp = RawProto.Parse(str);

            Assert.AreEqual(rp.Name, "root");
            Assert.AreEqual(rp.Value, "");
            Assert.AreEqual(rp.Children.Count, 1);
            Assert.AreEqual(rp.Children[0].Name, "name");
            Assert.AreEqual(rp.Children[0].Value, "");
            Assert.AreEqual(rp.Children[0].Children.Count, 2);
            Assert.AreEqual(rp.Children[0].Children[0].Name, "type");
            Assert.AreEqual(rp.Children[0].Children[0].Value, "boo");
            Assert.AreEqual(rp.Children[0].Children[1].Name, "type2");
            Assert.AreEqual(rp.Children[0].Children[1].Value, "moo");

            str = "name " + Environment.NewLine;
            str += "{ " + Environment.NewLine;
            str += "   type: 'boo' type2: 'moo' }" + Environment.NewLine;
            rp = RawProto.Parse(str);

            Assert.AreEqual(rp.Name, "root");
            Assert.AreEqual(rp.Value, "");
            Assert.AreEqual(rp.Children.Count, 1);
            Assert.AreEqual(rp.Children[0].Name, "name");
            Assert.AreEqual(rp.Children[0].Value, "");
            Assert.AreEqual(rp.Children[0].Children.Count, 2);
            Assert.AreEqual(rp.Children[0].Children[0].Name, "type");
            Assert.AreEqual(rp.Children[0].Children[0].Value, "boo");
            Assert.AreEqual(rp.Children[0].Children[1].Name, "type2");
            Assert.AreEqual(rp.Children[0].Children[1].Value, "moo");
        }

        [TestMethod]
        public void TestSimpleParsing3()
        {
            RawProto rp;
            string str;

            str = "name { innername { type: 'boo' type2: 'moo' } }";
            rp = RawProto.Parse(str);

            string str1 = rp.ToString();

            Assert.AreEqual(rp.Name, "root");
            Assert.AreEqual(rp.Value, "");
            Assert.AreEqual(rp.Children.Count, 1);
            Assert.AreEqual(rp.Children[0].Name, "name");
            Assert.AreEqual(rp.Children[0].Value, "");
            Assert.AreEqual(rp.Children[0].Children.Count, 1);
            Assert.AreEqual(rp.Children[0].Children[0].Name, "innername");
            Assert.AreEqual(rp.Children[0].Children[0].Value, "");
            Assert.AreEqual(rp.Children[0].Children[0].Children.Count, 2);
            Assert.AreEqual(rp.Children[0].Children[0].Children[0].Name, "type");
            Assert.AreEqual(rp.Children[0].Children[0].Children[0].Value, "boo");
            Assert.AreEqual(rp.Children[0].Children[0].Children[1].Name, "type2");
            Assert.AreEqual(rp.Children[0].Children[0].Children[1].Value, "moo");

            str = "name {" + Environment.NewLine;
            str += "   innername { " + Environment.NewLine;
            str += "      type: 'boo'" + Environment.NewLine;
            str += "      type2: 'moo'" + Environment.NewLine;
            str += "   }" + Environment.NewLine;
            str += "}";
            rp = RawProto.Parse(str);

            Assert.AreEqual(rp.Name, "root");
            Assert.AreEqual(rp.Value, "");
            Assert.AreEqual(rp.Children.Count, 1);
            Assert.AreEqual(rp.Children[0].Name, "name");
            Assert.AreEqual(rp.Children[0].Value, "");
            Assert.AreEqual(rp.Children[0].Children.Count, 1);
            Assert.AreEqual(rp.Children[0].Children[0].Name, "innername");
            Assert.AreEqual(rp.Children[0].Children[0].Value, "");
            Assert.AreEqual(rp.Children[0].Children[0].Children.Count, 2);
            Assert.AreEqual(rp.Children[0].Children[0].Children[0].Name, "type");
            Assert.AreEqual(rp.Children[0].Children[0].Children[0].Value, "boo");
            Assert.AreEqual(rp.Children[0].Children[0].Children[1].Name, "type2");
            Assert.AreEqual(rp.Children[0].Children[0].Children[1].Value, "moo");

            str = "name " + Environment.NewLine;
            str += "{ " + Environment.NewLine;
            str += "   innername" + Environment.NewLine;
            str += "   {" + Environment.NewLine;
            str += "   type: 'boo' type2: 'moo'" + Environment.NewLine;
            str += "}" + Environment.NewLine;
            str += "}";
            rp = RawProto.Parse(str);

            Assert.AreEqual(rp.Name, "root");
            Assert.AreEqual(rp.Value, "");
            Assert.AreEqual(rp.Children.Count, 1);
            Assert.AreEqual(rp.Children[0].Name, "name");
            Assert.AreEqual(rp.Children[0].Value, "");
            Assert.AreEqual(rp.Children[0].Children.Count, 1);
            Assert.AreEqual(rp.Children[0].Children[0].Name, "innername");
            Assert.AreEqual(rp.Children[0].Children[0].Value, "");
            Assert.AreEqual(rp.Children[0].Children[0].Children.Count, 2);
            Assert.AreEqual(rp.Children[0].Children[0].Children[0].Name, "type");
            Assert.AreEqual(rp.Children[0].Children[0].Children[0].Value, "boo");
            Assert.AreEqual(rp.Children[0].Children[0].Children[1].Name, "type2");
            Assert.AreEqual(rp.Children[0].Children[0].Children[1].Value, "moo");

            str = "name " + Environment.NewLine;
            str += "{ " + Environment.NewLine;
            str += "   innername { type: 'boo' type2: 'moo' } }" + Environment.NewLine;
            rp = RawProto.Parse(str);

            Assert.AreEqual(rp.Name, "root");
            Assert.AreEqual(rp.Value, "");
            Assert.AreEqual(rp.Children.Count, 1);
            Assert.AreEqual(rp.Children[0].Name, "name");
            Assert.AreEqual(rp.Children[0].Value, "");
            Assert.AreEqual(rp.Children[0].Children.Count, 1);
            Assert.AreEqual(rp.Children[0].Children[0].Name, "innername");
            Assert.AreEqual(rp.Children[0].Children[0].Value, "");
            Assert.AreEqual(rp.Children[0].Children[0].Children.Count, 2);
            Assert.AreEqual(rp.Children[0].Children[0].Children[0].Name, "type");
            Assert.AreEqual(rp.Children[0].Children[0].Children[0].Value, "boo");
            Assert.AreEqual(rp.Children[0].Children[0].Children[1].Name, "type2");
            Assert.AreEqual(rp.Children[0].Children[0].Children[1].Value, "moo");
        }

        [TestMethod]
        public void TestSimpleParsingArrays()
        {
            RawProto rp;
            string str;

            str = "name { innername { dim: 1 dim: 2 dim: 3 dim: 4 } }";
            rp = RawProto.Parse(str);

            string str1 = rp.ToString();

            Assert.AreEqual(rp.Name, "root");
            Assert.AreEqual(rp.Value, "");
            Assert.AreEqual(rp.Children.Count, 1);
            Assert.AreEqual(rp.Children[0].Name, "name");
            Assert.AreEqual(rp.Children[0].Value, "");
            Assert.AreEqual(rp.Children[0].Children.Count, 1);
            Assert.AreEqual(rp.Children[0].Children[0].Name, "innername");
            Assert.AreEqual(rp.Children[0].Children[0].Value, "");
            Assert.AreEqual(rp.Children[0].Children[0].Children.Count, 4);
            Assert.AreEqual(rp.Children[0].Children[0].Children[0].Name, "dim");
            Assert.AreEqual(rp.Children[0].Children[0].Children[0].Value, "1");
            Assert.AreEqual(rp.Children[0].Children[0].Children[1].Name, "dim");
            Assert.AreEqual(rp.Children[0].Children[0].Children[1].Value, "2");
            Assert.AreEqual(rp.Children[0].Children[0].Children[2].Name, "dim");
            Assert.AreEqual(rp.Children[0].Children[0].Children[2].Value, "3");
            Assert.AreEqual(rp.Children[0].Children[0].Children[3].Name, "dim");
            Assert.AreEqual(rp.Children[0].Children[0].Children[3].Value, "4");

            str = "name {" + Environment.NewLine;
            str += "   innername { " + Environment.NewLine;
            str += "      dim: 1" + Environment.NewLine;
            str += "      dim: 2" + Environment.NewLine;
            str += "      dim: 3" + Environment.NewLine;
            str += "      dim: 4" + Environment.NewLine;
            str += "   }" + Environment.NewLine;
            str += "}";
            rp = RawProto.Parse(str);

            Assert.AreEqual(rp.Name, "root");
            Assert.AreEqual(rp.Value, "");
            Assert.AreEqual(rp.Children.Count, 1);
            Assert.AreEqual(rp.Children[0].Name, "name");
            Assert.AreEqual(rp.Children[0].Value, "");
            Assert.AreEqual(rp.Children[0].Children.Count, 1);
            Assert.AreEqual(rp.Children[0].Children[0].Name, "innername");
            Assert.AreEqual(rp.Children[0].Children[0].Value, "");
            Assert.AreEqual(rp.Children[0].Children[0].Children.Count, 4);
            Assert.AreEqual(rp.Children[0].Children[0].Children[0].Name, "dim");
            Assert.AreEqual(rp.Children[0].Children[0].Children[0].Value, "1");
            Assert.AreEqual(rp.Children[0].Children[0].Children[1].Name, "dim");
            Assert.AreEqual(rp.Children[0].Children[0].Children[1].Value, "2");
            Assert.AreEqual(rp.Children[0].Children[0].Children[2].Name, "dim");
            Assert.AreEqual(rp.Children[0].Children[0].Children[2].Value, "3");
            Assert.AreEqual(rp.Children[0].Children[0].Children[3].Name, "dim");
            Assert.AreEqual(rp.Children[0].Children[0].Children[3].Value, "4");

            str = "name " + Environment.NewLine;
            str += "{ " + Environment.NewLine;
            str += "   innername" + Environment.NewLine;
            str += "   {" + Environment.NewLine;
            str += "   dim: 1 dim: 2 dim: 3 dim: 4" + Environment.NewLine;
            str += "}" + Environment.NewLine;
            str += "}";
            rp = RawProto.Parse(str);

            Assert.AreEqual(rp.Name, "root");
            Assert.AreEqual(rp.Value, "");
            Assert.AreEqual(rp.Children.Count, 1);
            Assert.AreEqual(rp.Children[0].Name, "name");
            Assert.AreEqual(rp.Children[0].Value, "");
            Assert.AreEqual(rp.Children[0].Children.Count, 1);
            Assert.AreEqual(rp.Children[0].Children[0].Name, "innername");
            Assert.AreEqual(rp.Children[0].Children[0].Value, "");
            Assert.AreEqual(rp.Children[0].Children[0].Children.Count, 4);
            Assert.AreEqual(rp.Children[0].Children[0].Children[0].Name, "dim");
            Assert.AreEqual(rp.Children[0].Children[0].Children[0].Value, "1");
            Assert.AreEqual(rp.Children[0].Children[0].Children[1].Name, "dim");
            Assert.AreEqual(rp.Children[0].Children[0].Children[1].Value, "2");
            Assert.AreEqual(rp.Children[0].Children[0].Children[2].Name, "dim");
            Assert.AreEqual(rp.Children[0].Children[0].Children[2].Value, "3");
            Assert.AreEqual(rp.Children[0].Children[0].Children[3].Name, "dim");
            Assert.AreEqual(rp.Children[0].Children[0].Children[3].Value, "4");

            str = "name " + Environment.NewLine;
            str += "{ " + Environment.NewLine;
            str += "   innername { dim: 1 dim: 2 dim: 3 dim: 4 } }" + Environment.NewLine;
            rp = RawProto.Parse(str);

            Assert.AreEqual(rp.Name, "root");
            Assert.AreEqual(rp.Value, "");
            Assert.AreEqual(rp.Children.Count, 1);
            Assert.AreEqual(rp.Children[0].Name, "name");
            Assert.AreEqual(rp.Children[0].Value, "");
            Assert.AreEqual(rp.Children[0].Children.Count, 1);
            Assert.AreEqual(rp.Children[0].Children[0].Name, "innername");
            Assert.AreEqual(rp.Children[0].Children[0].Value, "");
            Assert.AreEqual(rp.Children[0].Children[0].Children.Count, 4);
            Assert.AreEqual(rp.Children[0].Children[0].Children[0].Name, "dim");
            Assert.AreEqual(rp.Children[0].Children[0].Children[0].Value, "1");
            Assert.AreEqual(rp.Children[0].Children[0].Children[1].Name, "dim");
            Assert.AreEqual(rp.Children[0].Children[0].Children[1].Value, "2");
            Assert.AreEqual(rp.Children[0].Children[0].Children[2].Name, "dim");
            Assert.AreEqual(rp.Children[0].Children[0].Children[2].Value, "3");
            Assert.AreEqual(rp.Children[0].Children[0].Children[3].Name, "dim");
            Assert.AreEqual(rp.Children[0].Children[0].Children[3].Value, "4");
        }
    }
}
