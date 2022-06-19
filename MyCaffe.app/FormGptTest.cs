using MyCaffe.data;
using MyCaffe.python;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace MyCaffe.app
{
    public partial class FormGptTest : Form
    {
        Tuple<string, string, string> m_arg = null;
        AutoResetEvent m_evtCancel = new AutoResetEvent(false);
        AutoResetEvent m_evtReady = new AutoResetEvent(false);
        ManualResetEvent m_evtDone = new ManualResetEvent(false);
        ManualResetEvent m_evtRunning = new ManualResetEvent(false);
        Thread m_threadRun;

        delegate void fnOutput(string str);

        public FormGptTest()
        {
            InitializeComponent();

            m_threadRun = new Thread(new ThreadStart(dowork));
            m_threadRun.Start();
        }

        private void linkLabel1_LinkClicked(object sender, LinkLabelLinkClickedEventArgs e)
        {
            Process.Start(linkLabel1.Text);
        }

        private void setStatus(string str)
        {
            edtOutput.Text += Environment.NewLine + str;
            edtOutput.SelectionLength = 0;
            edtOutput.SelectionStart = edtOutput.Text.Length;
            edtOutput.ScrollToCaret();
        }

        private void btnRun_Click(object sender, EventArgs e)
        {
            string strUserName = Environment.UserName;
            string strPythonPath = @"C:\Users\" + strUserName + @"\AppData\Local\Programs\Python\Python39\python39.dll";

            if (!File.Exists(strPythonPath))
            {
                MessageBox.Show("Could not find Python at: '" + strPythonPath + "'.  You must have Python 3.9 installed!", "Missing Python 3.9", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }

            btnRun.Enabled = false;
            m_arg = new Tuple<string, string, string>(strPythonPath, edtGpu.Text, edtInput.Text);
            m_evtReady.Set();
        }

        private void stdout_OnOutput(object sender, OutputArgs e)
        {
            if (InvokeRequired)
                Invoke(new fnOutput(output), e.Message);
            else
                output(e.Message);
        }

        private void output(string str)
        {
            setStatus(str);

            if (str.StartsWith("RESULT:") ||
                str.StartsWith("ERROR:"))
            {
                btnRun.Enabled = true;
            }
        }

        private void dowork()
        {
            PythonInterop py = null;
            Output stdout1 = null;

            try
            {
                WaitHandle[] rgWait = new WaitHandle[] { m_evtCancel, m_evtReady };

                while (WaitHandle.WaitAny(rgWait) > 0)
                {
                    m_evtRunning.Set();
                    Tuple<string, string, string> arg = m_arg;
                    string strPythonPath = arg.Item1;
                    string strGpu = arg.Item2;
                    string strInput = arg.Item3;

                    string strPy;
                    KeyValuePair<string, object>[] rgArg;

                    if (py == null)
                    {
                        py = new PythonInterop(strPythonPath);
                        stdout1 = new Output();
                        stdout1.OnOutput += stdout_OnOutput;

                        strPy = "import sys" + Environment.NewLine +
                                "sys.stdout = sys.stderr = output" + Environment.NewLine +
                                "res = 1";

                        rgArg = new KeyValuePair<string, object>[]
                        {
                            new KeyValuePair<string, object>("output", stdout1)
                        };

                        py.RunPythonCodeAndReturn(strPy, "res", rgArg);
                    }

                    strPy = "import os" + Environment.NewLine +
                            "os.environ['CUDA_VISIBLE_DEVICES'] = strGpu" + Environment.NewLine +
                            "from transformers import pipeline" + Environment.NewLine +
                            "generator = pipeline(task = 'text-generation', max_length=500)" + Environment.NewLine +
                            "res = generator(strInput)";

                    rgArg = new KeyValuePair<string, object>[]
                    {
                        new KeyValuePair<string, object>("strGpu", strGpu),
                        new KeyValuePair<string, object>("strInput", strInput)
                    };

                    Invoke(new fnOutput(output), "Running GPT2 transformer from https://huggingface.co...");

                    object obj = py.RunPythonCodeAndReturn(strPy, "res", rgArg);

                    Invoke(new fnOutput(output), "-----------------------------");

                    string strResult = "";
                    object[] rgRes = obj as object[];
                    if (rgRes != null)
                    {
                        foreach (object obj1 in rgRes)
                        {
                            string strJson = obj1.ToString();
                            strResult += strJson;
                            strResult += Environment.NewLine;
                        }
                    }
                    else
                    {
                        string strJson = obj.ToString();
                        strResult = strJson;
                    }

                    Invoke(new fnOutput(output), "RESULT:" + Environment.NewLine + strResult);
                    m_evtRunning.Reset();
                }
            }
            catch (Exception excpt)
            {
                Invoke(new fnOutput(output), "ERROR: " + excpt.Message);
            }
            finally
            {
                m_evtRunning.Reset();
                if (py != null)
                {
                    py.Dispose();
                    py = null;
                }

                m_evtDone.Set();
            }
        }

        private void FormGptInput_FormClosing(object sender, FormClosingEventArgs e)
        {
            if (e.CloseReason != CloseReason.WindowsShutDown)
            {
                if (m_evtRunning.WaitOne(0))
                {
                    MessageBox.Show("The process is still running.", "Process Running", MessageBoxButtons.OK, MessageBoxIcon.Information);
                    e.Cancel = true;
                    return;
                }
            }

            m_evtCancel.Set();

            int nWait = 0;
            while (!m_evtDone.WaitOne(250) && nWait < 10)
            {
                Application.DoEvents();
                nWait++;
            }

            if (m_threadRun.IsAlive)
                m_threadRun.Abort();
        }
    }

    /// <summary>
    /// All Python output will be sent to this output object, which
    /// inturn routes the output to the OnOutput event.
    /// </summary>
    public class Output
    {
        public event EventHandler<OutputArgs> OnOutput;

        public void write(String str)
        {
            if (OnOutput != null)
                OnOutput(this, new OutputArgs(str));
        }

        public void writelines(String[] str)
        {
            foreach (String line in str)
            {
                if (OnOutput != null)
                    OnOutput(this, new OutputArgs(line));
            }
        }
        public void flush() { }
        public void close() { }
    }

    public class OutputArgs : EventArgs
    {
        string m_str;

        public OutputArgs(string str)
        {
            m_str = str;
        }

        public string Message
        {
            get { return m_str; }
        }
    }
}
