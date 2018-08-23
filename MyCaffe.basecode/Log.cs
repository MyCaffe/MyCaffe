using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;

namespace MyCaffe.basecode
{
    /// <summary>
    /// The Log class provides general output in text form.
    /// </summary>
    public class Log
    {
        string m_strPreText = "";
        string m_strSource;
        double m_dfProgress = 0;
        bool m_bEnable = true;
        bool m_bEnableTrace = false;

        /// <summary>
        /// The OnWriteLine event fires each time the WriteLine, WriteHeader or WriteError functions are called.
        /// </summary>
        public event EventHandler<LogArg> OnWriteLine;
        /// <summary>
        /// The OnProgress event fires each time the Progress value is set.
        /// </summary>
        public event EventHandler<LogProgressArg> OnProgress;

        /// <summary>
        /// The Log constructor.
        /// </summary>
        /// <param name="strSrc">Specifies the source of the Log.</param>
        public Log(string strSrc)
        {
            m_strSource = strSrc;
        }

        /// <summary>
        /// Enables/disables the Log.  When disabled, the Log does not output any data.
        /// </summary>
        public bool Enable
        {
            set { m_bEnable = value; }
        }

        /// <summary>
        /// Returns whether or not the Log is enabled.
        /// </summary>
        public bool IsEnabled
        {
            get { return m_bEnable; }
        }

        /// <summary>
        /// Get/set the pre-text prepended to each output line when set.
        /// </summary>
        public string PreText
        {
            get { return m_strPreText; }
            set { m_strPreText = value; }
        }

        /// <summary>
        /// Enables/disables the Trace.  When enabled, the .Net Trace.WriteLine is called in addition to the normal Log output.  This is primarily used when debugging.
        /// </summary>
        public bool EnableTrace
        {
            get { return m_bEnableTrace; }
            set { m_bEnableTrace = value; }
        }

        /// <summary>
        /// Write a line of output.
        /// </summary>
        /// <param name="str">Specifies the string to output.</param>
        /// <param name="bOverrideEnabled">Specifies whether or not to override the enabled state (default = false).</param>
        /// <param name="bHeader">Specifies whether or not the output text represents a header (default = false).</param>
        /// <param name="bError">Specfifies whether or not the output text represents an error (default = false).</param>
        public void WriteLine(string str, bool bOverrideEnabled = false, bool bHeader = false, bool bError = false)
        {
            // Check for enabled and not overridden
            if (!m_bEnable && !bOverrideEnabled)
                return;

            string strLine;

            if (!bHeader && m_strPreText != null && m_strPreText.Length > 0)
                strLine = m_strPreText + str;
            else
                strLine = str;

            if (OnWriteLine != null)
                OnWriteLine(this, new LogArg(m_strSource, strLine, m_dfProgress, bError, bOverrideEnabled));

            if (m_bEnableTrace)
            {
                if (bHeader)
                    Trace.WriteLine(strLine);
                else
                    Trace.WriteLine(m_dfProgress.ToString("P") + "   " + strLine);
            }
        }

        /// <summary>
        /// Write a header as output.
        /// </summary>
        /// <param name="str">Specifies the header text.</param>
        public void WriteHeader(string str)
        {
            if (!m_bEnable)
                return;

            string strLine = "";

            strLine += "=============================================";
            strLine += Environment.NewLine;
            strLine += str;
            strLine += Environment.NewLine;
            strLine += "=============================================";
            strLine += Environment.NewLine;

            WriteLine(strLine, false, true);
        }

        /// <summary>
        /// Write an error as output.
        /// </summary>
        /// <param name="e">Specifies the error.</param>
        public void WriteError(Exception e)
        {
            string strErr = e.Message;

            if (e.InnerException != null)
                strErr += " " + e.InnerException.Message;

            if (strErr.Trim().Length == 0)
                strErr = "No error message!";

            WriteLine("ERROR! " + strErr, false, false, true);
        }

        /// <summary>
        /// Get/set the progress associated with the Log.
        /// </summary>
        public double Progress
        {
            get { return m_dfProgress; }
            set 
            { 
                m_dfProgress = value;

                if (OnProgress != null)
                    OnProgress(this, new LogProgressArg(m_strSource, m_dfProgress));
            }
        }

        /// <summary>
        /// Test whether two values are equal using a given type 'T'.
        /// </summary>
        /// <typeparam name="T">Specifies the resolution for the test: either <i>double</i> or <i>float</i>.</typeparam>
        /// <param name="df1">Specifies the first value to test.</param>
        /// <param name="df2">Specifies the second value to test.</param>
        /// <param name="str">Specifies the descriptive text to output if the test fails.</param>
        public void EXPECT_EQUAL<T>(double df1, double  df2, string str = null)
        {
            if (typeof(T) == typeof(float))
            {
                if (str == null)
                    str = "Float Values " + df1.ToString() + " and " + df2.ToString() + " are NOT FLOAT equal!";

                EXPECT_NEAR(df1, df2, 1e-2, str);
            }
            else
            {
                if (df1 != df2)
                {
                    if (str == null)
                        str = "Values " + df1.ToString() + " and " + df2.ToString() + " are NOT equal!";

                    throw new Exception(str);
                }
            }
        }

        /// <summary>
        /// Test whether two numbers are within a range (<i>dfErr</i>) of one another using the <i>float</i> resolution.
        /// </summary>
        /// <param name="df1">Specifies the first value to test.</param>
        /// <param name="df2">Specifies the second value to test.</param>
        /// <param name="dfErr">Specifies the acceptable error for the test.</param>
        /// <param name="str">Specifies the descriptive text to output if the test fails.</param>
        public void EXPECT_NEAR_FLOAT(double df1, double df2, double dfErr, string str = "")
        {
            float f1 = (float)df1;
            float f2 = (float)df2;
            float fErr = (float)dfErr;
            float fDiff = Math.Abs(f1 - f2);

            if (fDiff > fErr)
                throw new Exception("Values " + f1.ToString() + " and " + f2.ToString() + " are NOT within the range " + fErr.ToString() + " of one another.  " + str);
        }

        /// <summary>
        /// Test whether two numbers are within a range (<i>dfErr</i>) of one another using the <i>double</i> resolution.
        /// </summary>
        /// <param name="df1">Specifies the first value to test.</param>
        /// <param name="df2">Specifies the second value to test.</param>
        /// <param name="dfErr">Specifies the acceptable error for the test.</param>
        /// <param name="str">Specifies the descriptive text to output if the test fails.</param>
        public void EXPECT_NEAR(double df1, double df2, double dfErr, string str = "")
        {
            double dfDiff = Math.Abs(df1 - df2);

            if (dfDiff > dfErr)
                throw new Exception("Values " + df1.ToString() + " and " + df2.ToString() + " are NOT within the range " + dfErr.ToString() + " of one another.  " + str);
        }

        /// <summary>
        /// Test a flag for <i>true</i>.
        /// </summary>
        /// <param name="b">Specifies the flag to test.</param>
        /// <param name="str">Specifies the description text to output if the flag is <i>false</i>.</param>
        public void CHECK(bool b, string str)
        {
            if (!b)
                throw new Exception(str);
        }

        /// <summary>
        /// Test whether one number is equal to another.
        /// </summary>
        /// <param name="df1">Specifies the first value to test.</param>
        /// <param name="df2">Specifies the second value to test.</param>
        /// <param name="str">Specifies the descriptive text to output if the test fails.</param>
        public void CHECK_EQ(double df1, double df2, string str)
        {
            if (df1 != df2)
                throw new Exception(str);
        }

        /// <summary>
        /// Test whether one number is not-equal to another.
        /// </summary>
        /// <param name="df1">Specifies the first value to test.</param>
        /// <param name="df2">Specifies the second value to test.</param>
        /// <param name="str">Specifies the descriptive text to output if the test fails.</param>
        public void CHECK_NE(double df1, double df2, string str)
        {
            if (df1 == df2)
                throw new Exception(str);
        }

        /// <summary>
        /// Test whether one number is less than or equal to another.
        /// </summary>
        /// <param name="df1">Specifies the first value to test.</param>
        /// <param name="df2">Specifies the second value to test.</param>
        /// <param name="str">Specifies the descriptive text to output if the test fails.</param>
        public void CHECK_LE(double df1, double df2, string str)
        {
            if (df1 > df2)
                throw new Exception(str);
        }

        /// <summary>
        /// Test whether one number is less than another.
        /// </summary>
        /// <param name="df1">Specifies the first value to test.</param>
        /// <param name="df2">Specifies the second value to test.</param>
        /// <param name="str">Specifies the descriptive text to output if the test fails.</param>
        public void CHECK_LT(double df1, double df2, string str)
        {
            if (df1 >= df2)
                throw new Exception(str);
        }

        /// <summary>
        /// Test whether one number is greater than or equal to another.
        /// </summary>
        /// <param name="df1">Specifies the first value to test.</param>
        /// <param name="df2">Specifies the second value to test.</param>
        /// <param name="str">Specifies the descriptive text to output if the test fails.</param>
        public void CHECK_GE(double df1, double df2, string str)
        {
            if (df1 < df2)
                throw new Exception(str);
        }

        /// <summary>
        /// Test whether one number is greater than another.
        /// </summary>
        /// <param name="df1">Specifies the first value to test.</param>
        /// <param name="df2">Specifies the second value to test.</param>
        /// <param name="str">Specifies the descriptive text to output if the test fails.</param>
        public void CHECK_GT(double df1, double df2, string str)
        {
            if (df1 <= df2)
                throw new Exception(str);
        }

        /// <summary>
        /// Test whether one number is equal to another.
        /// </summary>
        /// <param name="f1">Specifies the first value to test.</param>
        /// <param name="f2">Specifies the second value to test.</param>
        /// <param name="str">Specifies the descriptive text to output if the test fails.</param>
        public void CHECK_EQ(float f1, float f2, string str)
        {
            if (f1 != f2)
                throw new Exception(str);
        }

        /// <summary>
        /// Test whether one number is not equal to another.
        /// </summary>
        /// <param name="f1">Specifies the first value to test.</param>
        /// <param name="f2">Specifies the second value to test.</param>
        /// <param name="str">Specifies the descriptive text to output if the test fails.</param>
        public void CHECK_NE(float f1, float f2, string str)
        {
            if (f1 == f2)
                throw new Exception(str);
        }

        /// <summary>
        /// Test whether one number is less-than or equal to another.
        /// </summary>
        /// <param name="f1">Specifies the first value to test.</param>
        /// <param name="f2">Specifies the second value to test.</param>
        /// <param name="str">Specifies the descriptive text to output if the test fails.</param>
        public void CHECK_LE(float f1, float f2, string str)
        {
            if (f1 > f2)
                throw new Exception(str);
        }

        /// <summary>
        /// Test whether one number is less-than another.
        /// </summary>
        /// <param name="f1">Specifies the first value to test.</param>
        /// <param name="f2">Specifies the second value to test.</param>
        /// <param name="str">Specifies the descriptive text to output if the test fails.</param>
        public void CHECK_LT(float f1, float f2, string str)
        {
            if (f1 >= f2)
                throw new Exception(str);
        }

        /// <summary>
        /// Test whether one number is greater-than or equal to another.
        /// </summary>
        /// <param name="f1">Specifies the first value to test.</param>
        /// <param name="f2">Specifies the second value to test.</param>
        /// <param name="str">Specifies the descriptive text to output if the test fails.</param>
        public void CHECK_GE(float f1, float f2, string str)
        {
            if (f1 < f2)
                throw new Exception(str);
        }

        /// <summary>
        /// Test whether one number is greater-than another.
        /// </summary>
        /// <param name="f1">Specifies the first value to test.</param>
        /// <param name="f2">Specifies the second value to test.</param>
        /// <param name="str">Specifies the descriptive text to output if the test fails.</param>
        public void CHECK_GT(float f1, float f2, string str)
        {
            if (f1 <= f2)
                throw new Exception(str);
        }

        /// <summary>
        /// Causes a failure which throws an exception with the desciptive text.
        /// </summary>
        /// <param name="str">Specifies the descriptive text to output.</param>
        public void FAIL(string str)
        {
            throw new Exception(str);
        }
    }
}
