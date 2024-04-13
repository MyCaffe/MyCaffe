using MyCaffe.basecode;
using MyCaffe.common;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MyCaffe.extras
{
    /// <summary>
    /// The LLM Inference class uses the LLM low-level extension to perform LLM inference.
    /// </summary>
    public class LlmInference<T> : IDisposable
    {
        CudaDnn<T> m_cuda = null;
        Log m_log = null;
        string m_strExtensionPath;
        long m_hExtension = 0;
        T[] m_rgLlm = null;
        ManualResetEvent m_evtLoading = new ManualResetEvent(false);
        ManualResetEvent m_evtGenerating = new ManualResetEvent(false);
        AutoResetEvent m_evtCancelQuery = new AutoResetEvent(false);
        string m_strModelFiles;
        string m_strPrompt;

        /// <summary>
        /// The OnStatus event fires when the status of the LlmInference Load changes.
        /// </summary>
        public event EventHandler<LlmInferenceStatusArgs> OnStatus;
        /// <summary>
        /// The OnResults event fires when the results of the LlmInference Generate produces results.
        /// </summary>
        public event EventHandler<LlmInferenceResultsArgs> OnResults;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="cuda">Specifies the link to the low-level CUDA inferencing.</param>
        /// <param name="log">Specifies the output log.</param>
        public LlmInference(CudaDnn<T> cuda, Log log)
        {
            m_cuda = cuda;
            m_log = log;
        }

        /// <summary>
        /// Release all resources used.
        /// </summary>
        public void Dispose()
        {
            CleanUp();
        }

        /// <summary>
        /// Initialize the LlmInference with a given extension path, temperature, topp, and seed.
        /// </summary>
        /// <param name="strExtensionPath">Specifies the path to the extension dll for LLM inferencing.</param>
        /// <param name="fTemperature">Specifies the temperature where 0.0 = greedy deterministic. 1.0 = original (range = 0.0 to 1.0).</param>
        /// <param name="fTopp">Specifies the top-p in nucleus sampling. 1.0 = 0ff. 0.9 works well but slower (range = 0.0 to 1.0).</param>
        /// <param name="lSeed">Specifies the random seed, or 0 to ignore.</param>
        /// <exception cref="Exception">An exception is thrown on error.</exception>
        public void Initialize(string strExtensionPath, float fTemperature, float fTopp, long lSeed)
        {
            try
            {
                m_strExtensionPath = strExtensionPath;
                m_hExtension = m_cuda.CreateExtension(strExtensionPath);

                T[] rgParam = new T[3];
                rgParam[0] = Utility.ConvertVal<T>(fTemperature);
                rgParam[1] = Utility.ConvertVal<T>(fTopp);
                rgParam[2] = Utility.ConvertVal<T>(lSeed);

                m_rgLlm = m_cuda.RunExtension(m_hExtension, (int)CUDAFN_EXTENSION_LLM.CREATE, rgParam);
            }
            catch (Exception excpt)
            {
                throw new Exception("Initializing LlmInference failed with '" + strExtensionPath + "'!", excpt);
            }
        }

        /// <summary>
        /// Release all resources used.
        /// </summary>
        public void CleanUp()
        {
            if (m_hExtension != 0)
            {
                m_cuda.RunExtension(m_hExtension, (int)CUDAFN_EXTENSION_LLM.DESTROY, m_rgLlm);
                m_cuda.FreeExtension(m_hExtension);
                m_hExtension = 0;
            }
        }

        /// <summary>
        /// Asynchonously load the model files.  When done the OnStatus event is fired with bLoaded = true.
        /// </summary>
        /// <param name="strModelFile">Specifies the Llama2_7B_chat.bin file created using the Karpathy scripts.</param>
        /// <param name="strTokenizerFile">Specifies the tokenizer.bin file used by the Llama2 models.</param>
        /// <returns>If loading starts 'true' is returned, otherwise 'false' is returned when already loading.</returns>
        /// <exception cref="Exception">An exception is thrown on error.</exception>
        public bool LoadAsync(string strModelFile, string strTokenizerFile)
        {
            if (m_hExtension == 0 || m_rgLlm == null)
                throw new Exception("The LlmInference has not been initialized!");

            if (m_evtLoading.WaitOne(0))
                return false;

            m_strModelFiles = strModelFile + ";" + strTokenizerFile;
            Task.Factory.StartNew(new Action(load));

            if (OnStatus != null)
                OnStatus(this, new LlmInferenceStatusArgs(false));

            return true;
        }

        private void load()
        {
            try
            {
                m_evtLoading.Set();
                m_cuda.RunExtensionEx(m_hExtension, (int)CUDAFN_EXTENSION_LLM.LOAD, m_rgLlm, m_strModelFiles);

                if (OnStatus != null)
                    OnStatus(this, new LlmInferenceStatusArgs(true));   
            }
            catch (Exception excpt)
            {
                if (OnStatus != null)
                    OnStatus(this, new LlmInferenceStatusArgs(false, excpt));
            }
            finally
            {
                m_evtLoading.Reset();
            }
        }

        /// <summary>
        /// Generate a response to a sys/user prompt asynchronously.  When done the OnResults event is fired with the results.
        /// </summary>
        /// <param name="strSystemPrompt">Optionally specifies the system prompt, or null to ignore.</param>
        /// <param name="strUserPrompt">Specifies the user prompt.</param>
        /// <returns>If the generation starts, 'true' is returned. Otherwise if already generating, 'false' is returned.</returns>
        public bool GenerateAsync(string strSystemPrompt, string strUserPrompt)
        {
            try
            {
                if (m_hExtension == 0 || m_rgLlm == null)
                    throw new Exception("The LlmInference has not been initialized!");

                if (m_evtGenerating.WaitOne(0))
                    return false;

                m_evtGenerating.Set();

                m_strPrompt = "[INST]";
                if (!string.IsNullOrEmpty(strSystemPrompt))
                {
                    m_strPrompt += "<<SYS>>";
                    m_strPrompt += strSystemPrompt;
                    m_strPrompt += "<</SYS>>";
                }

                m_strPrompt += strUserPrompt;
                m_strPrompt += "[/INST]";

                Task.Factory.StartNew(new Action(generate));
                Thread.Sleep(250);
                Task.Factory.StartNew(new Action(query));
            }
            catch (Exception excpt)
            {
                if (OnResults != null)
                    OnResults(this, new LlmInferenceResultsArgs("ERROR", true, excpt));
            }
            finally
            {
                m_evtGenerating.Reset();
            }

            return true;
        }

        private void generate()
        {
            try
            {
                m_cuda.RunExtensionEx(m_hExtension, (int)CUDAFN_EXTENSION_LLM.GENERATE, m_rgLlm, m_strPrompt);
                m_evtCancelQuery.Set();
                query();
            }
            finally
            {
                m_evtGenerating.Reset();
            }
        }

        private void query()
        {
            string strEnd = "\n[END]";

            while (!m_evtCancelQuery.WaitOne(1000))
            {
                int[] rgLlm1 = new int[1];
                rgLlm1[0] = (int)Utility.ConvertValF(m_rgLlm[0]);
                string[] rgText = m_cuda.QueryExtensionStrings(m_hExtension, (int)CUDAFN_EXTENSION_LLM.QUERY_RESPONSE, rgLlm1);

                if (rgText != null && rgText.Length > 0)
                {
                    string strResult = "";
                    int nPos = rgText[0].LastIndexOf(strEnd);
                    bool bEnd = false;

                    if (nPos >= 0)
                    {
                        strResult = rgText[0].Substring(0, nPos);
                        bEnd = true;
                    }
                    else
                    {
                        strResult = rgText[0];
                    }

                    if (OnResults != null)
                        OnResults(this, new LlmInferenceResultsArgs(strResult, bEnd));
                }
            }
        }
    }

    /// <summary>
    /// The LlmInferenceStatusArgs class provides the arguments for the OnStatus event.
    /// </summary>
    public class  LlmInferenceStatusArgs : EventArgs
    {
        bool m_bLoaded = false;
        Exception m_err;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="bLoaded">Specifies 0 for loading or 1 for loaded.</param>
        /// <param name="err">Optionally, specifies an error if one occurs.</param>
        public LlmInferenceStatusArgs(bool bLoaded, Exception err = null)
        {
            m_bLoaded = bLoaded;
            m_err = err;
        }

        /// <summary>
        /// Specifies whether the LlmInference has been loaded.
        /// </summary>
        public bool Loaded
        {
            get { return m_bLoaded; }
        }

        /// <summary>
        /// Specifies any error that occurred, or null if no error occurred.
        /// </summary>
        public Exception Error
        {
            get { return m_err; }
        }
    }

    /// <summary>
    /// The LlmInferenceResultsArgs class provides the arguments for the OnResults event.
    /// </summary>
    public class LlmInferenceResultsArgs : EventArgs
    {
        string m_strResults = "";
        bool m_bEnd = false;
        Exception m_err;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="strResults">Specifies the current result part.</param>
        /// <param name="bEnd">Specifies whether or not this is the end result.</param>
        /// <param name="err">If an error occurs it is returned here.</param>
        public LlmInferenceResultsArgs(string strResults, bool bEnd, Exception err = null)
        {
            m_strResults = strResults;
            m_bEnd = bEnd;
            m_err = err;
        }

        /// <summary>
        /// Specifies the partial results.
        /// </summary>
        public string Results
        {
            get { return m_strResults; }
        }

        /// <summary>
        /// Specifies whether or not this is the end result.
        /// </summary>
        public bool End
        {
            get { return m_bEnd; }
        }

        /// <summary>
        /// Specifies any error that occurred, or null if no error occurred.
        /// </summary>
        public Exception Error
        {
            get { return m_err; }
        }
    }
}
