using MyCaffe.basecode;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Google.Protobuf;
using System.Collections;
using MyCaffe.param;

namespace MyCaffe.common
{
    /// <summary>
    /// The PersistCaffe class is used to load and save weight files in the .caffemodel format.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class PersistCaffe<T> : IXPersist<T>
    {
        Log m_log;
        bool m_bFailOnFirstTry = false;
        const string m_strWeightMyCaffeTag = "mycaffe.ai";

        /// <summary>
        /// The PersistCaffe constructor.
        /// </summary>
        /// <param name="log">Specifies the log used for output.</param>
        /// <param name="bFailOnFirstTry">Specifies whether or not to try to load the weights file.  On the first try the Caffe model format is attempted, and on the second the MyCaffe format is used.</param>
        public PersistCaffe(Log log, bool bFailOnFirstTry)
        {
            m_log = log;
            m_bFailOnFirstTry = bFailOnFirstTry;
        }

        /// <summary>
        /// This tag is used to mark the ending section of each weighting file with 
        /// 'MyCaffe' specific information.
        /// </summary>
        public string MyCaffeTag
        {
            get { return m_strWeightMyCaffeTag; }
        }

        /// <summary>
        /// This method returns whether or not the weights have been marked as 'mycaffe.ai'.
        /// </summary>
        /// <param name="rgWeights">Specifies the weights.</param>
        /// <param name="strVer">Returns the version of this file.</param>
        /// <returns>If the weights contain mycaffe weights, <i>true</i> is returned, false otherwise.</returns>
        public bool IsMyCaffe(byte[] rgWeights, out string strVer)
        {
            strVer = null;

            if (rgWeights == null || rgWeights.Length < 10)
                return false;

            string strCaffeNet = Encoding.ASCII.GetString(rgWeights, rgWeights.Length - 10, 10);
            if (strCaffeNet == m_strWeightMyCaffeTag)
            {
                long lCaffeStart = BitConverter.ToInt64(rgWeights, rgWeights.Length - (10 + sizeof(long)));
                strVer = Encoding.ASCII.GetString(rgWeights, (int)lCaffeStart + 10, 5);
                return true;
            }

            return false;
        }

        /// <summary>
        /// Save the solver state to a byte array.
        /// </summary>
        /// <param name="state">Specifies the solver state to save.</param>
        /// <param name="type">Specifies the solver type.</param>
        /// <returns>A byte array containing the solver state is returned.</returns>
        public byte[] SaveSolverState(SolverState state, SolverParameter.SolverType type = SolverParameter.SolverType.SGD)
        {
            FieldDescriptor fd = FieldDescriptor.CreateSolverStateFieldDesc();
            ProtoBufWriter writer = new ProtoBufWriter(m_log);

            m_log.WriteLine("Saving state...");

            writer.WriteField(fd, "iter", new int[] { state.iter });
            writer.WriteField(fd, "current_step", new int[] { state.current_step });

            if (type == SolverParameter.SolverType.LBFGS)
            {
                writer.WriteField(fd, "start", new int[] { state.start });
                writer.WriteField(fd, "end", new int[] { state.end });
            }

            for (int i = 0; i < state.history.Count; i++)
            {
                writer.WriteField(fd, "history", saveBlobProto(fd.FindFirstChild("history"), state.history[i]));
            }

            if (type == SolverParameter.SolverType.LBFGS)
            {
                for (int i = 0; i < state.s_history.Count; i++)
                {
                    writer.WriteField(fd, "s_history", saveBlobProto(fd.FindFirstChild("s_history"), state.s_history[i]));
                }

                writer.WriteField(fd, "gradients", saveBlobProto(fd.FindFirstChild("gradient"), state.gradients));
                writer.WriteField(fd, "direction", saveBlobProto(fd.FindFirstChild("direction"), state.direction));
            }

            return writer.GetBytes(true);
        }

        /// <summary>
        /// Load the solver state from a byte array.
        /// </summary>
        /// <param name="rgState">Specifies the byte array containing the solver state.</param>
        /// <param name="type">Specifies the solver type.</param>
        /// <returns>The SolverState loaded is returned.</returns>
        public SolverState LoadSolverState(byte[] rgState, SolverParameter.SolverType type = SolverParameter.SolverType.SGD)
        {
            SolverState state = new SolverState();
            FieldDescriptor fd = FieldDescriptor.CreateSolverStateFieldDesc();
            ProtoBufReader reader = new ProtoBufReader(rgState);
            ProtoBufFieldCollection fields = reader.ReadFields(fd, false);
            Stopwatch sw = new Stopwatch();

            m_log.WriteLine("Loading the Solver state...");

            if (fields == null || fields.Count == 0)
                return null;


            //---------------------------------------------
            //  Load the state.
            //---------------------------------------------

            ProtoBufField pbIter = fields.FindFirstChild("iter");
            state.iter = (pbIter == null || pbIter.IntValues == null || pbIter.IntValues.Length == 0) ? 0 : pbIter.IntValues[0];

            ProtoBufField pbCurStep = fields.FindFirstChild("current_step");
            state.current_step = (pbCurStep == null || pbCurStep.IntValues == null || pbCurStep.IntValues.Length == 0) ? 1 : pbCurStep.IntValues[0];

            if (type == SolverParameter.SolverType.LBFGS)
            {
                ProtoBufField pbStart = fields.FindFirstChild("start");
                state.start = (pbStart == null || pbStart.IntValues == null || pbStart.IntValues.Length == 0) ? 0 : pbStart.IntValues[0];

                ProtoBufField pbEnd = fields.FindFirstChild("end");
                state.end = (pbEnd == null || pbEnd.IntValues == null || pbEnd.IntValues.Length == 0) ? 1 : pbEnd.IntValues[0];
            }

            ProtoBufFieldCollection col = fields.FindAllChildren("history");
            if (col != null && col.Count > 0)
            {
                FieldDescriptor fdHist = fd.FindFirstChild("history");

                for (int i = 0; i < col.Count; i++)
                {
                    state.history.Add(LoadBlobProto(col[i].Bytes, fdHist.FieldId));
                }
            }

            if (type == SolverParameter.SolverType.LBFGS)
            {
                ProtoBufFieldCollection colS = fields.FindAllChildren("s_history");
                if (colS != null && colS.Count > 0)
                {
                    FieldDescriptor fdHist = fd.FindFirstChild("s_history");

                    for (int i = 0; i < colS.Count; i++)
                    {
                        state.s_history.Add(LoadBlobProto(colS[i].Bytes, fdHist.FieldId));
                    }
                }

                ProtoBufField pbGrad = fields.FindFirstChild("gradients");
                if (pbGrad != null)
                {
                    FieldDescriptor fdGrad = fd.FindFirstChild("gradients");
                    state.gradients = LoadBlobProto(pbGrad.Bytes, fdGrad.FieldId);
                }

                ProtoBufField pbDir = fields.FindFirstChild("direction");
                if (pbDir != null)
                {
                    FieldDescriptor fdDir = fd.FindFirstChild("direction");
                    state.direction = LoadBlobProto(pbDir.Bytes, fdDir.FieldId);
                }
            }

            return state;
        }

        /// <summary>
        /// Loads new weights into a BlobCollection
        /// </summary>
        /// <remarks>
        /// NOTE: In order to maintain compatibility with the C++ %Caffe, extra MyCaffe features may be added to the <i>end</i> of the weight file.  After saving weights (see SaveWeights) in the format
        /// used by the C++ %Caffe, MyCaffe writes the bytes "mycaffe.ai".  All information after these bytes are specific to MyCaffe and allow for loading weights for models by Blob name and shape
        /// and loosen the C++ %Caffe requirement that the 'number' of blobs match.  Adding this functionality allows for training model, changing the model structure, and then re-using the trained
        /// weights in the new model.  
        /// </remarks>
        /// <param name="rgWeights">Specifies the weights themselves.</param>
        /// <param name="rgExpectedShapes">Specifies a list of expected shapes for each Blob where the weights are to be loaded.</param>
        /// <param name="colBlobs">Specifies the Blobs to load with the weights.</param>
        /// <param name="bSizeToFit">Specifies wether or not the weights should be re-sized.  Note: resizing can render the weights useless, especially in deeper, layers.</param>
        /// <param name="bLoadedDiffs">Returns whether or not the diffs were loaded.</param>
        /// <param name="inputWtInfo">Optionally, specifies the weight info describing the input weight blobs to import by name.  Note when used the number of blobs must match the number of <i>targetWtInfo</i> blobs.  Otherwise, when <i>null</i> this parameter is ignored.</param>
        /// <param name="targetWtInfo">Optionally, specifies the weight info describing the target weight blobs to import by name.  Note when used the number of blobs must match the number of <i>inputWtInfo</i> blobs.  Otherwise, when <i>null</i> this parameter is ignored.</param>
        /// <param name="strSkipBlobType">Optionally, specifies a blob type where weights are NOT loaded.  See Blob.BLOB_TYPE for the types of Blobs.</param>
        /// <returns>The collection of Blobs with newly loaded weights is returned.</returns>
        public BlobCollection<T> LoadWeights(byte[] rgWeights, List<string> rgExpectedShapes, BlobCollection<T> colBlobs, bool bSizeToFit, out bool bLoadedDiffs, List<string> inputWtInfo = null, List<string> targetWtInfo = null, string strSkipBlobType = null)
        {
            BlobCollection<T> colBlob1;
            m_log.WriteLine("Attempting to load the weights in Caffe model format...");
            string strVer;

            if (!IsMyCaffe(rgWeights, out strVer))
            {
                colBlob1 = loadFromCaffe(rgWeights, rgExpectedShapes, colBlobs, bSizeToFit, out bLoadedDiffs, inputWtInfo, targetWtInfo, strSkipBlobType);
                if (colBlob1 != null)
                {
                    m_log.WriteLine("Weights loaded in Caffe model format.");
                    return colBlob1;
                }

                if (m_bFailOnFirstTry)
                    throw new Exception("Failed to load the weights from the caffe model.");
            }
            else if (strVer == "v.1.0")
            {
                m_log.FAIL("Loading weights with 'depreciated' native v.1.0 format...");
            }

            m_log.WriteLine("Attempting to load weights in MyCaffe model format...");
            colBlob1 = loadFromMyCaffe(rgWeights, rgExpectedShapes, colBlobs, bSizeToFit, out bLoadedDiffs, inputWtInfo, targetWtInfo, strSkipBlobType);
            if (colBlob1 != null)
            {
                m_log.WriteLine("Weights loaded in MyCaffe model format.");
                return colBlob1;
            }

            if (m_bFailOnFirstTry)
                throw new Exception("Failed to load the weights from the MyCaffe model.");

            m_log.FAIL("Loading weights with 'depreciated' native format...");
            return null;
        }

        /// <summary>
        /// Returns the weight information describing the weights containined within the weight bytes.
        /// </summary>
        /// <param name="rgWeights">Specifies the bytes containing the weights.</param>
        /// <returns>The weight information is returned.</returns>
        public WeightInfo<T> LoadWeightInfo(byte[] rgWeights)
        {
            string strVer;

            if (!IsMyCaffe(rgWeights, out strVer))
                return loadInfoFromCaffe(rgWeights);
            else
                return loadInfoFromMyCaffe(rgWeights);
        }

        /// <summary>
        /// Returns the weight information describing the weights containined within the Blob collection.
        /// </summary>
        /// <param name="colBlobs">Specifies the Blob collection containing the weights.</param>
        /// <returns>The weight information is returned.</returns>
        public WeightInfo<T> LoadWeightInfo(BlobCollection<T> colBlobs)
        {
            WeightInfo<T> info = new common.WeightInfo<T>();

            foreach (Blob<T> b in colBlobs)
            {
                info.AddBlob(b);
            }

            return info;
        }

        /// <summary>
        /// Save the weights to a byte array.
        /// </summary>
        /// <remarks>
        /// NOTE: In order to maintain compatibility with the C++ %Caffe, extra MyCaffe features may be added to the <i>end</i> of the weight file.  After saving weights in the format
        /// used by the C++ %Caffe, MyCaffe writes the bytes "mycaffe.ai".  All information after these bytes are specific to MyCaffe and allow for loading weights for models by Blob name and shape
        /// and loosen the C++ %Caffe requirement that the 'number' of blobs match.  Adding this functionality allows for training model, changing the model structure, and then re-using the trained
        /// weights in the new model.  
        /// </remarks>
        /// <param name="colBlobs">Specifies the Blobs to save with the weights.</param>
        /// <param name="bSaveDiffs">Optionally, specifies to save the diff values - currently this parameter is not used.</param>
        /// <returns>The byte array containing the weights is returned.</returns>
        public byte[] SaveWeights(BlobCollection<T> colBlobs, bool bSaveDiffs = false)
        {
            FieldDescriptor fd = FieldDescriptor.CreateNetworkParamFieldDesc();
            ProtoBufWriter writer = new ProtoBufWriter(m_log);
            Dictionary<string, BlobCollection<T>> rgLayers = new Dictionary<string, BlobCollection<T>>();

            foreach (Blob<T> blob in colBlobs)
            {
                string strLayer = (string)blob.Tag;
                if (strLayer == null || strLayer.Length == 0)
                    throw new Exception("Invalid blob specification - missing layer name.");

                if (!rgLayers.ContainsKey(strLayer))
                    rgLayers.Add(strLayer, new BlobCollection<T>());

                rgLayers[strLayer].Add(blob);
            }

            writer.WriteField(fd, "name", "");

            foreach (KeyValuePair<string, BlobCollection<T>> kv in rgLayers)
            {
                m_log.WriteLine("Saving layer '" + kv.Key + "'...");
                writer.WriteField(fd, "LayerParameter", saveLayerParameter(fd.FindFirstChild("LayerParameter"), kv.Key, kv.Value));
            }

            writer.Flush();

            long lCaffeNetStart = writer.Length;
            byte[] rgPad = new byte[256];

            using (BinaryWriter bw = new BinaryWriter(writer.Stream))
            {
                bw.Write(rgPad);
                lCaffeNetStart += rgPad.Length;

                string strCaffeNet = MyCaffeTag;
                byte[] rgCaffeNet = Encoding.ASCII.GetBytes(strCaffeNet);

                string strVer = "1.0.1";
                byte[] rgV = Encoding.ASCII.GetBytes(strVer);
                byte[] rgVer = new byte[32];
                Array.Copy(rgV, rgVer, rgV.Length);

                bw.Write(rgCaffeNet);
                bw.Write(rgVer);
                bw.Write(lCaffeNetStart);
                bw.Write(rgCaffeNet);
            }

            return writer.GetBytes(false);
        }

        /// <summary>
        /// The LoadBlobProto function loads a BlobProto from a proto buffer.
        /// </summary>
        /// <param name="rg">Specifies the bytes containing the BlobProto in proto buffer format.</param>
        /// <param name="nFieldId">Specifies the field ID to use for the BlobProto.</param>
        /// <returns>The new BlobProt is returned.</returns>
        public BlobProto LoadBlobProto(byte[] rg, int nFieldId)
        {
            FieldDescriptor fd = FieldDescriptor.CreateBlobProtoDesc(nFieldId);
            ProtoBufReader reader = new ProtoBufReader(rg);
            ProtoBufFieldCollection fields = reader.ReadFields(fd, false);

            if (fields == null || fields.Count == 0)
                return null;

            for (int i = 0; i < fields.Count; i++)
            {
                ProtoBufField field = fields[i];
                field.LoadSubFields(0, 4);
            }

            List<int> rgShape = new List<int>();

            ProtoBufField pbShape = fields.FindFirstChild("shape");
            if (pbShape != null)
            {
                if (pbShape.Type != ProtoBufField.TYPE.ARRAY)
                    throw new Exception("Invalid proto buf: invalid type 'shape'");

                ProtoBufField pbDim = pbShape.Array.FindFirstChild("dim");
                if (pbDim == null || pbDim.Type != ProtoBufField.TYPE.LONG_ARRAY)
                    throw new Exception("Invalid proto buf: missing 'dim' type.");

                for (int i = 0; i < pbDim.LongValues.Length; i++)
                {
                    rgShape.Add((int)pbDim.LongValues[i]);
                }
            }
            else
            {
                ProtoBufField pbNum = fields.FindFirstChild("num");
                if (pbNum != null)
                {
                    if (pbNum.Type != ProtoBufField.TYPE.BIT32)
                        throw new Exception("Invalid proto buf: invalid type 'num'");

                    rgShape.Add(pbNum.IntValue);

                    ProtoBufField pbChannels = fields.FindFirstChild("channels");
                    if (pbChannels != null)
                    {
                        if (pbChannels.Type != ProtoBufField.TYPE.BIT32)
                            throw new Exception("Invalid proto buf: invalid type 'channels'");

                        rgShape.Add(pbChannels.IntValue);

                        ProtoBufField pbHeight = fields.FindFirstChild("height");
                        if (pbHeight != null)
                        {
                            if (pbHeight.Type != ProtoBufField.TYPE.BIT32)
                                throw new Exception("Invalid proto buf: invalid type 'height'");

                            rgShape.Add(pbHeight.IntValue);

                            ProtoBufField pbWidth = fields.FindFirstChild("width");
                            if (pbWidth != null)
                            {
                                if (pbWidth.Type != ProtoBufField.TYPE.BIT32)
                                    throw new Exception("Invalid proto buf: invalid type 'width'");

                                rgShape.Add(pbWidth.IntValue);
                            }
                        }
                    }
                }
            }

            ProtoBufField pbData = fields.FindFirstChild("data");
            if (pbData == null)
            {
                pbData = fields.FindFirstChild("double_data");
                if (pbData == null)
                    throw new Exception("Invalid proto buf: missing 'data' or 'double_data'");
            }

            BlobProto proto = new param.BlobProto(rgShape);

            if (pbData.Type == ProtoBufField.TYPE.FLOAT_ARRAY)
                proto.data = new List<float>(pbData.FloatValues);
            else if (pbData.Type == ProtoBufField.TYPE.DOUBLE_ARRAY)
                proto.double_data = new List<double>(pbData.DoubleValues);
            else
                throw new Exception("Invalid proto buf: invalid data type '" + pbData.Type.ToString() + "'.");

            return proto;
        }

        /// <summary>
        /// The LoadBlobProto function loads a BlobProto from a file.
        /// </summary>
        /// <param name="strFile">Specifies the binary file containing the blob proto.</param>
        /// <param name="nFieldId">Specifies the field ID to use for the BlobProto.</param>
        /// <returns>The new BlobProt is returned.</returns>
        public BlobProto LoadBlobProto(string strFile, int nFieldId)
        {
            byte[] rgBytes;

            using (FileStream fs = new FileStream(strFile, FileMode.Open, FileAccess.Read))
            {
                using (BinaryReader br = new BinaryReader(fs))
                {
                    rgBytes = br.ReadBytes((int)fs.Length);
                }
            }

            return LoadBlobProto(rgBytes, nFieldId);
        }

        private byte[] saveLayerParameter(FieldDescriptor fd, string strName, BlobCollection<T> col)
        {
            ProtoBufWriter writer = new common.ProtoBufWriter(m_log);

            writer.WriteField(fd, "name", strName);

            foreach (Blob<T> blob in col)
            {
                writer.WriteField(fd, "blobs", saveBlobProto(fd.FindFirstChild("blobs"), blob));
                m_log.WriteLine("  - saved blob '" + blob.Name + "'");
            }

            return writer.GetBytes();
        }

        private byte[] saveBlobProto(FieldDescriptor fd, BlobProto bp)
        {
            ProtoBufWriter writer = new ProtoBufWriter(m_log);

            writer.WriteField(fd, "shape", saveBlobShape(fd.FindFirstChild("shape"), bp.shape.dim));

            if (bp.double_data != null && bp.double_data.Count > 0)
                writer.WriteField(fd, "double_data", bp.double_data.ToArray());
            else
                writer.WriteField(fd, "data", bp.data.ToArray());

            return writer.GetBytes();
        }

        private byte[] saveBlobProto(FieldDescriptor fd, Blob<T> blob)
        {
            ProtoBufWriter writer = new ProtoBufWriter(m_log);

            writer.WriteField(fd, "shape", saveBlobShape(fd.FindFirstChild("shape"), blob.shape()));

            T[] rg = blob.update_cpu_data();

            if (typeof(T) == typeof(double))
            {
                double[] rgD = (double[])Convert.ChangeType(rg, typeof(double[]));
                writer.WriteField(fd, "double_data", rgD);
            }
            else
            {
                float[] rgD = (float[])Convert.ChangeType(rg, typeof(float[]));
                writer.WriteField(fd, "data", rgD);
            }

            return writer.GetBytes();
        }

        private byte[] saveBlobShape(FieldDescriptor fd, List<int> rg)
        {
            ProtoBufWriter writer = new ProtoBufWriter(m_log);
            List<long> rgLong = new List<long>();

            for (int i = 0; i < rg.Count; i++)
            {
                rgLong.Add(rg[i]);
            }

            writer.WriteField(fd, "dim", rgLong.ToArray());

            return writer.GetBytes();
        }

        private BlobCollection<T> loadFromMyCaffe(byte[] rgWeights, List<string> rgExpectedShapes, BlobCollection<T> colBlobs, bool bSizeToFit, out bool bLoadedDiffs, List<string> inputWtInfo = null, List<string> targetWtInfo = null, string strSkipBlobType = null)
        {
            BlobCollection<T> colBlobs1 = loadFromCaffe(rgWeights, rgExpectedShapes, colBlobs, bSizeToFit, out bLoadedDiffs, inputWtInfo, targetWtInfo, strSkipBlobType);
            return colBlobs1;
        }

        private WeightInfo<T> loadInfoFromMyCaffe(byte[] rgWeights)
        {
            return loadInfoFromCaffe(rgWeights);
        }

        private BlobCollection<T> loadFromCaffe(byte[] rgWeights, List<string> rgExpectedShapes, BlobCollection<T> colBlobs, bool bSizeToFit, out bool bLoadedDiffs, List<string> inputWtInfo = null, List<string> targetWtInfo = null, string strSkipBlobType = null)
        {
            FieldDescriptor fd = FieldDescriptor.CreateNetworkParamFieldDesc();
            ProtoBufReader reader = new ProtoBufReader(rgWeights);
            ProtoBufFieldCollection fields = reader.ReadFields(fd, true);
            Stopwatch sw = new Stopwatch();
            BlobName name = new BlobName();

            bLoadedDiffs = false;

            if (fields == null || fields.Count == 0)
                return null;

            sw.Start();

            for (int i=0; i<fields.Count; i++)
            {
                ProtoBufField field = fields[i];
                field.LoadSubFields(0, 4);

                if (sw.Elapsed.TotalMilliseconds > 1000)
                {
                    m_log.Progress = (double)i / (double)fields.Count;
                    m_log.WriteLine("(" + m_log.Progress.ToString("P") + ") loading fields...");
                    sw.Restart();
                }
            }

            //---------------------------------------------
            //  Find all the blobs containing learnable
            //  parameters.
            //---------------------------------------------

            ProtoBufFieldCollection colFieldBlobs = new common.ProtoBufFieldCollection();
            int nLayerIdx = 0;

            for (int i = 0; i < fields.Count; i++)
            {
                if (fields[i].FieldDesc != null) // Ignore null entries which can occur in V1.
                {
                    if (fields[i].FieldDesc.Name == "LayerParameter")
                    {
                        ProtoBufField pbName = fields[i].Array.FindFirstChild("name");
                        ProtoBufFieldCollection col = fields[i].Array.FindAllChildren("blobs");
                        string strName = (pbName != null) ? pbName.StringValue : ("layer_" + nLayerIdx.ToString());

                        if (col != null && col.Count > 0)
                        {
                            col.SetTag(strName);
                            colFieldBlobs.AddRange(col);
                        }

                        nLayerIdx++;
                    }
                    else if (fields[i].FieldDesc.Name == "V1LayerParameter")
                    {
                        ProtoBufField pbName = fields[i].Array.FindFirstChild("name");
                        ProtoBufFieldCollection col = fields[i].Array.FindAllChildren("blobs");
                        string strName = (pbName != null) ? pbName.StringValue : ("layer_" + nLayerIdx.ToString());

                        if (col != null && col.Count > 0)
                        {
                            col.SetTag(strName);
                            col.SetLegacy(true);
                            colFieldBlobs.AddRange(col);
                        }

                        nLayerIdx++;
                    }
                }
            }

            //---------------------------------------------
            //  Find the first learnable parameter that
            //  matches the size of the first colBlob.
            //---------------------------------------------

            m_log.Progress = 0;
            m_log.WriteLine("Loading the weights...");

            if (colBlobs.Count != colFieldBlobs.Count)
                m_log.WriteLine("The number of learnable blobs within the weights does not match the number within the network, attempting to load by size...");

            int nFieldIdx = 0;
            int nBlobIdx = 0;
            int nInfoIdx = 0;
            int nTargetIdx = 0;

            List<long> rgBlobShape = null;

            while (nFieldIdx < colFieldBlobs.Count && nBlobIdx < colBlobs.Count)
            {
                Blob<T> blob = colBlobs[nBlobIdx];
                string strName = name.GetName(blob.Name);

                if (targetWtInfo != null)
                {
                    while (strName != targetWtInfo[nTargetIdx] && nBlobIdx < colBlobs.Count)
                    {
                        blob = colBlobs[nBlobIdx];
                        strName = name.GetName(blob.Name);
                        nBlobIdx++;
                    }

                    if (nBlobIdx == colBlobs.Count)
                        m_log.WriteError(new Exception("Could not find the target blob '" + targetWtInfo[nTargetIdx] + "'!"));

                    nTargetIdx++;
                }

                string strShapeB = rgExpectedShapes[nBlobIdx];
                string strShapeW = "";
                long lCount = 0;
                bool bResizeNeeded = false;
                bool bMisSized = false;

                //-----------------------------------------
                //  Find the first matching size.
                //-----------------------------------------
                while (nFieldIdx < colFieldBlobs.Count)
                {
                    strName = null;

                    ProtoBufField pbName = colFieldBlobs[nFieldIdx].Array.FindFirstChild("name");
                    if (pbName != null && pbName.Type == ProtoBufField.TYPE.STRING)
                    {
                        strName = pbName.StringValue;
                    }
                    else
                    {
                        ProtoBufField pbType = colFieldBlobs[nFieldIdx].Array.FindFirstChild("type");
                        if (pbType != null && pbType.Type == ProtoBufField.TYPE.STRING)
                            strName = pbType.StringValue + "_" + nFieldIdx.ToString();
                        else
                            strName = "blob_" + nFieldIdx.ToString();
                    }

                    if (inputWtInfo == null || strName == inputWtInfo[nInfoIdx])
                    {
                        nInfoIdx++;

                        ProtoBufField pbShape = colFieldBlobs[nFieldIdx].Array.FindFirstChild("shape");
                        if (pbShape != null && pbShape.Type == ProtoBufField.TYPE.ARRAY)
                        {
                            ProtoBufField pbDim = pbShape.Array.FindFirstChild("dim");
                            if (pbDim != null && pbDim.Type == ProtoBufField.TYPE.LONG_ARRAY)
                            {
                                strShapeW = createShapeString(pbDim.LongValues, out lCount);

                                if (compareShapes(strShapeB, strShapeW))
                                {
                                    rgBlobShape = new List<long>(pbDim.LongValues);
                                    break;
                                }

                                if (bSizeToFit && compareShapes(strShapeB, strShapeW, 2))
                                {
                                    rgBlobShape = new List<long>(pbDim.LongValues);
                                    break;
                                }

                                bMisSized = true;
                                break;
                            }
                        }
                        else
                        {
                            ProtoBufField pbNum = colFieldBlobs[nFieldIdx].Array.FindFirstChild("num");
                            if (pbNum != null && pbNum.Type == ProtoBufField.TYPE.BIT32)
                            {
                                List<long> rgShape = new List<long>();
                                rgShape.Add(pbNum.IntValue);

                                ProtoBufField pbChannels = colFieldBlobs[nFieldIdx].Array.FindFirstChild("channels");
                                if (pbChannels != null && pbChannels.Type == ProtoBufField.TYPE.BIT32)
                                {
                                    rgShape.Add(pbChannels.IntValue);

                                    ProtoBufField pbHeight = colFieldBlobs[nFieldIdx].Array.FindFirstChild("height");
                                    if (pbHeight != null && pbHeight.Type == ProtoBufField.TYPE.BIT32)
                                    {
                                        rgShape.Add(pbHeight.IntValue);

                                        ProtoBufField pbWidth = colFieldBlobs[nFieldIdx].Array.FindFirstChild("width");
                                        if (pbWidth != null && pbWidth.Type == ProtoBufField.TYPE.BIT32)
                                        {
                                            rgShape.Add(pbWidth.IntValue);
                                        }
                                    }
                                }

                                strShapeW = createShapeString(rgShape.ToArray(), out lCount);

                                if (compareShapes(strShapeB, strShapeW) || bSizeToFit)
                                {
                                    rgBlobShape = rgShape;
                                    break;
                                }

                                if (bSizeToFit && compareShapes(strShapeB, strShapeW, 2))
                                {
                                    rgBlobShape = rgShape;
                                    bResizeNeeded = true;
                                    break;
                                }

                                bMisSized = true;
                                break;
                            }
                        }
                    }

                    nFieldIdx++;
                }

                if (nFieldIdx == colFieldBlobs.Count)
                    continue;

                //-----------------------------------------
                //  Copy the data, but only for blobs 
                //  that are not missized and ones that do 
                //  not match the skip type, if specified.
                //-----------------------------------------

                if (!bMisSized && (strSkipBlobType == null || blob.type.ToString() != strSkipBlobType))
                {
                    ProtoBufField pbData = colFieldBlobs[nFieldIdx].Array.FindFirstChild("data");
                    FieldDescriptor.TYPE type = FieldDescriptor.TYPE.FLOAT;
                    long lDataCount = 0;
                    if (pbData == null)
                    {
                        pbData = colFieldBlobs[nFieldIdx].Array.FindFirstChild("double_data");
                        type = FieldDescriptor.TYPE.DOUBLE;
                        lDataCount = pbData.DoubleValues.Length;
                    }
                    else
                    {
                        lDataCount = pbData.FloatValues.Length;
                    }

                    if (pbData == null || (lDataCount != lCount && !bSizeToFit))
                        m_log.FAIL("Could not find the weights matching the data size '" + strShapeB + "'!");

                    if (bSizeToFit && !compareShapes(strShapeB, strShapeW, 4))
                        m_log.FAIL("Could not find the weights matching the first two items of the shape '" + strShapeB + "'!");

                    T[] rgData = copyData(pbData, type, lDataCount, rgBlobShape);

                    blob.mutable_cpu_data = rgData;
                    blob.Tag = colFieldBlobs[nFieldIdx].Tag;

                    if (bSizeToFit && bResizeNeeded)
                    {
                        List<int> rgNewShape = parseShape(strShapeB);
                        Blob<T> blobResized = blob.Resize(rgNewShape);
                        blob.Dispose();
                        colBlobs[nBlobIdx] = blobResized;
                    }
                }

                m_log.Progress = (double)nBlobIdx / (double)colBlobs.Count;
                m_log.WriteLine("(" + m_log.Progress.ToString("P") + ") loaded blob '" + colBlobs[nBlobIdx].Name + "' size = " + strShapeB);

                nFieldIdx++;
                nBlobIdx++;

                if ((targetWtInfo != null && nTargetIdx == targetWtInfo.Count) ||
                    (inputWtInfo != null && nInfoIdx == inputWtInfo.Count))
                    break;
            }

            return colBlobs;
        }

        private WeightInfo<T> loadInfoFromCaffe(byte[] rgWeights)
        {
            WeightInfo<T> info = new common.WeightInfo<T>();
            FieldDescriptor fd = FieldDescriptor.CreateNetworkParamFieldDesc();
            ProtoBufReader reader = new ProtoBufReader(rgWeights);
            ProtoBufFieldCollection fields = reader.ReadFields(fd, true);
            Stopwatch sw = new Stopwatch();

            if (fields == null || fields.Count == 0)
                return null;

            sw.Start();

            for (int i = 0; i < fields.Count; i++)
            {
                ProtoBufField field = fields[i];
                field.LoadSubFields(0, 4);

                if (sw.Elapsed.TotalMilliseconds > 1000)
                {
                    m_log.Progress = (double)i / (double)fields.Count;
                    m_log.WriteLine("(" + m_log.Progress.ToString("P") + ") loading fields...");
                    sw.Restart();
                }
            }

            //---------------------------------------------
            //  Find all the blobs containing learnable
            //  parameters.
            //---------------------------------------------

            ProtoBufFieldCollection colFieldBlobs = new common.ProtoBufFieldCollection();
            int nLayerIdx = 0;

            for (int i = 0; i < fields.Count; i++)
            {
                if (fields[i].FieldDesc != null)  // Ignore null entries which can occur in V1
                {
                    if (fields[i].FieldDesc.Name == "LayerParameter")
                    {
                        ProtoBufField pbName = fields[i].Array.FindFirstChild("name");
                        ProtoBufFieldCollection col = fields[i].Array.FindAllChildren("blobs");
                        string strName = (pbName != null) ? pbName.StringValue : ("layer_" + nLayerIdx.ToString());

                        if (col != null && col.Count > 0)
                        {
                            col.SetTag(strName);
                            colFieldBlobs.AddRange(col);
                        }

                        nLayerIdx++;
                    }
                    else if (fields[i].FieldDesc.Name == "V1LayerParameter")
                    {
                        ProtoBufField pbName = fields[i].Array.FindFirstChild("name");
                        ProtoBufFieldCollection col = fields[i].Array.FindAllChildren("blobs");
                        string strName = (pbName != null) ? pbName.StringValue : ("layer_" + nLayerIdx.ToString());

                        if (col != null && col.Count > 0)
                        {
                            col.SetTag(strName);
                            col.SetLegacy(true);
                            colFieldBlobs.AddRange(col);
                        }

                        nLayerIdx++;
                    }
                }
            }

            //---------------------------------------------
            //  Find the first learnable parameter that
            //  matches the size of the first colBlob.
            //---------------------------------------------

            m_log.Progress = 0;

            int nFieldIdx = 0;

            while (nFieldIdx < colFieldBlobs.Count)
            {
                string strName = null;

                ProtoBufField pbName = colFieldBlobs[nFieldIdx].Array.FindFirstChild("name");
                if (pbName != null && pbName.Type == ProtoBufField.TYPE.STRING)
                {
                    strName = pbName.StringValue;
                }
                else
                {
                    ProtoBufField pbType = colFieldBlobs[nFieldIdx].Array.FindFirstChild("type");
                    if (pbType != null && pbType.Type == ProtoBufField.TYPE.STRING)
                        strName = pbType.StringValue + "_" + nFieldIdx.ToString();
                    else
                        strName = "blob_" + nFieldIdx.ToString();
                }

                List<int> rgShape = new List<int>();

                ProtoBufField pbShape = colFieldBlobs[nFieldIdx].Array.FindFirstChild("shape");
                if (pbShape != null && pbShape.Type == ProtoBufField.TYPE.ARRAY)
                {
                    ProtoBufField pbDim = pbShape.Array.FindFirstChild("dim");
                    if (pbDim != null && pbDim.Type == ProtoBufField.TYPE.LONG_ARRAY)
                    {
                        for (int i = 0; i < pbDim.LongValues.Length; i++)
                        {
                            rgShape.Add((int)pbDim.LongValues[i]);
                        }
                    }
                }
                else
                {
                    ProtoBufField pbNum = colFieldBlobs[nFieldIdx].Array.FindFirstChild("num");
                    if (pbNum != null && pbNum.Type == ProtoBufField.TYPE.BIT32)
                    {
                        rgShape.Add(pbNum.IntValue);

                        ProtoBufField pbChannels = colFieldBlobs[nFieldIdx].Array.FindFirstChild("channels");
                        if (pbChannels != null && pbChannels.Type == ProtoBufField.TYPE.BIT32)
                        {
                            rgShape.Add(pbChannels.IntValue);

                            ProtoBufField pbHeight = colFieldBlobs[nFieldIdx].Array.FindFirstChild("height");
                            if (pbHeight != null && pbHeight.Type == ProtoBufField.TYPE.BIT32)
                            {
                                rgShape.Add(pbHeight.IntValue);

                                ProtoBufField pbWidth = colFieldBlobs[nFieldIdx].Array.FindFirstChild("width");
                                if (pbWidth != null && pbWidth.Type == ProtoBufField.TYPE.BIT32)
                                {
                                    rgShape.Add(pbWidth.IntValue);
                                }
                            }
                        }
                    }
                }

                info.AddBlob(strName, rgShape);

                nFieldIdx++;
            }

            return info;
        }

        private T[] copyData(ProtoBufField pb, FieldDescriptor.TYPE type, long lCount, List<long> rgBlobShape)
        {
            T[] rgData = new T[lCount];

            if (type == FieldDescriptor.TYPE.FLOAT)
                Array.Copy(pb.FloatValues, rgData, lCount);
            else
            {
                if (typeof(T) == typeof(double))
                    Array.Copy(pb.DoubleValues, rgData, lCount);
                else
                    return Utility.ConvertVec<T>(pb.DoubleValues);
            }

            return rgData;
        }

        private List<int> parseShape(string strShape, int nCount = int.MaxValue)
        {
            List<int> rg1 = new List<int>();
            string[] rgstr1 = strShape.Split(' ');

            for (int i = 0; i < rgstr1.Length - 1 && i < nCount; i++)
            {
                int nVal = int.Parse(rgstr1[i]);

                if (nVal > 1)
                    rg1.Add(nVal);
            }

            return rg1;
        }

        private bool compareShapes(string strA, string strB, int nCount = int.MaxValue)
        {
            if (strA == strB)
                return true;

            List<int> rg1 = parseShape(strA, nCount);
            List<int> rg2 = parseShape(strB, nCount);

            if (rg1.Count != rg2.Count)
                return false;

            if (rg1.Count == 0)
            {
                if (strA != strB)
                    return false;
                else
                    return true;
            }

            for (int i = 0; i < rg1.Count; i++)
            {
                if (rg1[i] != rg2[i])
                    return false;
            }

            return true;
        }

        private string createShapeString(long[] rg, out long lCount)
        {
            lCount = 1;
            string str = "";

            for (int i = 0; i < rg.Length; i++)
            {
                if (rg[i] >= 1)
                {
                    str += rg[i].ToString();
                    str += " ";
                    lCount *= rg[i];
                }
            }

            str += "(" + rg.Length.ToString() + ")";

            return str;
        }
    }

    class ProtoBufWriter : IDisposable /** @private */
    {
        MemoryStream m_ms = null;
        CodedOutputStream m_strm = null;
        bool m_bOwnStream = true;
        Log m_log;
        static int m_nUnknownFieldID = 5000;
        Dictionary<string, int> m_rgUnknownFields = new Dictionary<string, int>();

        public ProtoBufWriter(Log log)
        {
            m_log = log;
            m_ms = new MemoryStream();
            m_strm = new CodedOutputStream(m_ms);
        }

        public ProtoBufWriter(Log log, CodedOutputStream strm)
        {
            m_strm = strm;
            m_bOwnStream = false;
        }

        public void Dispose()
        {
            if (m_strm != null && m_bOwnStream)
            {
                m_strm.Dispose();
                m_strm = null;
            }

            if (m_ms != null)
            {
                m_ms.Dispose();
                m_ms = null;
            }
        }

        public int Length
        {
            get { return (int)m_ms.Length; }
        }

        public byte[] GetBytes(bool bFlush = true)
        {
            if (m_strm != null && bFlush)
                m_strm.Flush();

            byte[] rg = m_ms.ToArray();
            return rg;
        }

        public void Flush()
        {
            m_strm.Flush();
        }

        public MemoryStream Stream
        {
            get { return m_ms; }
        }

        private int getFieldId(FieldDescriptor fd, string strName, out FieldDescriptor.TYPE type)
        {
            type = FieldDescriptor.TYPE.UNKNOWN;

            fd = fd.FindFirstChild(strName);
            if (fd != null)
            {
                type = fd.Type;
                return fd.FieldId;
            }

            if (m_rgUnknownFields.ContainsKey(strName))
                return m_rgUnknownFields[strName];

            int nId = m_nUnknownFieldID;
            m_nUnknownFieldID++;

            m_rgUnknownFields.Add(strName, nId);

            return nId;
        }

        public void WriteField(FieldDescriptor fd, string strName, string strVal)
        {
            FieldDescriptor.TYPE type;
            int nFieldId = getFieldId(fd, strName, out type);
            uint tag = WireFormat.MakeTag(nFieldId, WireFormat.WireType.LengthDelimited);

            m_strm.WriteUInt32(tag);
            m_strm.WriteString(strVal);
        }

        public void WriteField(FieldDescriptor fd, string strName, byte[] rg)
        {
            FieldDescriptor.TYPE type;
            int nFieldId = getFieldId(fd, strName, out type);
            uint tag = WireFormat.MakeTag(nFieldId, WireFormat.WireType.LengthDelimited);

            m_strm.WriteUInt32(tag);
            m_strm.WriteBytes(ByteString.CopyFrom(rg));
        }

        public void WriteField(FieldDescriptor fd, string strName, double dfVal)
        {
            FieldDescriptor.TYPE type;
            int nFieldId = getFieldId(fd, strName, out type);
            uint tag;

            switch (type)
            {
                case FieldDescriptor.TYPE.DOUBLE:
                    tag = WireFormat.MakeTag(nFieldId, WireFormat.WireType.Fixed64);
                    m_strm.WriteUInt32(tag);
                    m_strm.WriteDouble(dfVal);
                    break;

                case FieldDescriptor.TYPE.FLOAT:
                    tag = WireFormat.MakeTag(nFieldId, WireFormat.WireType.Fixed32);
                    m_strm.WriteUInt32(tag);
                    m_strm.WriteFloat((float)dfVal);
                    break;

                case FieldDescriptor.TYPE.LONG:
                case FieldDescriptor.TYPE.ULONG:
                    tag = WireFormat.MakeTag(nFieldId, WireFormat.WireType.Fixed64);
                    m_strm.WriteUInt32(tag);
                    m_strm.WriteFixed64((ulong)(long)dfVal);
                    break;

                case FieldDescriptor.TYPE.INT:
                case FieldDescriptor.TYPE.UINT:
                    tag = WireFormat.MakeTag(nFieldId, WireFormat.WireType.Fixed32);
                    m_strm.WriteUInt32(tag);
                    m_strm.WriteFixed32((uint)(int)dfVal);
                    break;

                default:
                    throw new Exception("Unknown type '" + type.ToString() + "'");
            }
        }

        public void WriteField(FieldDescriptor fd, string strName, long[] rgVal)
        {
            FieldDescriptor.TYPE type;
            int nFieldId = getFieldId(fd, strName, out type);
            uint tag;

            if (type != FieldDescriptor.TYPE.LONG &&
                type != FieldDescriptor.TYPE.ULONG)
                throw new Exception("Invalid type '" + type.ToString() + "'");

            tag = WireFormat.MakeTag(nFieldId, WireFormat.WireType.LengthDelimited);
            m_strm.WriteUInt32(tag);

            ProtoBufWriter pbWriter = new ProtoBufWriter(m_log);
            byte[] rg = pbWriter.WriteArray(type, rgVal);
            m_strm.WriteBytes(ByteString.CopyFrom(rg));
        }

        public byte[] WriteArray(FieldDescriptor.TYPE type, long[] rgVal)
        {
            for (int i = 0; i < rgVal.Length; i++)
            {
                if (type == FieldDescriptor.TYPE.ULONG)
                    m_strm.WriteUInt64((uint)rgVal[i]);
                else
                    m_strm.WriteInt64(rgVal[i]);
            }

            return GetBytes();
        }

        public void WriteField(FieldDescriptor fd, string strName, int[] rgVal)
        {
            FieldDescriptor.TYPE type;
            int nFieldId = getFieldId(fd, strName, out type);
            uint tag;

            if (type != FieldDescriptor.TYPE.INT &&
                type != FieldDescriptor.TYPE.UINT)
                throw new Exception("Invalid type '" + type.ToString() + "'");

            tag = WireFormat.MakeTag(nFieldId, WireFormat.WireType.LengthDelimited);
            m_strm.WriteUInt32(tag);

            ProtoBufWriter pbWriter = new ProtoBufWriter(m_log);
            byte[] rg = pbWriter.WriteArray(type, rgVal);
            m_strm.WriteBytes(ByteString.CopyFrom(rg));
        }

        public byte[] WriteArray(FieldDescriptor.TYPE type, int[] rgVal)
        {
            for (int i = 0; i < rgVal.Length; i++)
            {
                if (type == FieldDescriptor.TYPE.UINT)
                    m_strm.WriteUInt32((uint)rgVal[i]);
                else
                    m_strm.WriteInt64(rgVal[i]);
            }

            return GetBytes();
        }

        public void WriteField(FieldDescriptor fd, string strName, double[] rgVal)
        {
            FieldDescriptor.TYPE type;
            int nFieldId = getFieldId(fd, strName, out type);
            uint tag;

            if (type != FieldDescriptor.TYPE.DOUBLE &&
                type != FieldDescriptor.TYPE.FLOAT)
                throw new Exception("Invalid type '" + type.ToString() + "'");          

            tag = WireFormat.MakeTag(nFieldId, WireFormat.WireType.LengthDelimited);
            m_strm.WriteUInt32(tag);

            ProtoBufWriter pbWriter = new ProtoBufWriter(m_log);
            byte[] rg = pbWriter.WriteArray(type, rgVal);
            m_strm.WriteBytes(ByteString.CopyFrom(rg));
        }

        public byte[] WriteArray(FieldDescriptor.TYPE type, double[] rgVal)
        {
            for (int i = 0; i < rgVal.Length; i++)
            {
                m_strm.WriteDouble(rgVal[i]);
            }

            return GetBytes();
        }

        public void WriteField(FieldDescriptor fd, string strName, float[] rgVal)
        {
            FieldDescriptor.TYPE type;
            int nFieldId = getFieldId(fd, strName, out type);
            uint tag;

            if (type != FieldDescriptor.TYPE.DOUBLE &&
                type != FieldDescriptor.TYPE.FLOAT)
                throw new Exception("Invalid type '" + type.ToString() + "'");

            tag = WireFormat.MakeTag(nFieldId, WireFormat.WireType.LengthDelimited);
            m_strm.WriteUInt32(tag);

            ProtoBufWriter pbWriter = new ProtoBufWriter(m_log);
            byte[] rg = pbWriter.WriteArray(type, rgVal);
            m_strm.WriteBytes(ByteString.CopyFrom(rg));
        }

        public byte[] WriteArray(FieldDescriptor.TYPE type, float[] rgVal)
        {
            for (int i = 0; i < rgVal.Length; i++)
            {
                m_strm.WriteFloat(rgVal[i]);
            }

            return GetBytes();
        }
    }

    class ProtoBufReader : IDisposable /** @private */
    {
        CodedInputStream m_strm = null;
        bool m_bOwnStream = true;

        public ProtoBufReader(byte[] rg)
        {
            m_strm = new CodedInputStream(rg);
        }

        public ProtoBufReader(CodedInputStream strm)
        {
            m_strm = strm;
            m_bOwnStream = false;
        }

        public void Dispose()
        {
            if (m_strm != null && m_bOwnStream)
            {
                m_strm.Dispose();
                m_strm = null;
            }
        }

        public ProtoBufFieldCollection ReadFields(FieldDescriptor fd, bool bFirstRead)
        {
            ProtoBufFieldCollection fields = new common.ProtoBufFieldCollection();
            ProtoBufField field = ReadField(fd, bFirstRead);

            while (field != null)
            {
                if (field.Length > 0 || (field.Type != ProtoBufField.TYPE.BYTES && field.Type != ProtoBufField.TYPE.STRING))
                    fields.Add(field);

                field = ReadField(fd, bFirstRead);
            }

            return fields;
        }

        public ProtoBufField ReadField(FieldDescriptor fd, bool bFirstRead)
        {
            if (m_strm.IsAtEnd)
                return null;

            uint tag = m_strm.ReadUInt32();
            int nField = WireFormat.GetTagFieldNumber(tag);

            if (nField <= 0)
                return null;

            int nWireFmt = (int)WireFormat.GetTagWireType(tag);
            if (bFirstRead && nWireFmt != (int)WireFormat.WireType.LengthDelimited)
                return null;

            if (fd != null)
                fd = fd.FindFirstChild(nField);

            ProtoBufField field = new ProtoBufField(m_strm, nField, fd);
            if (!field.Load((WireFormat.WireType)nWireFmt))
                return null;

            return field;
        }
    }

    class ProtoBufFieldCollection : IEnumerable<ProtoBufField> /** @private */
    {
        List<ProtoBufField> m_rgFields = new List<ProtoBufField>();

        public ProtoBufFieldCollection()
        {
        }

        public int Count
        {
            get { return m_rgFields.Count; }
        }

        public ProtoBufField this[int nIdx]
        {
            get { return m_rgFields[nIdx]; }
        }

        public void SetTag(string str)
        {
            foreach (ProtoBufField field in m_rgFields)
            {
                field.Tag = str;
            }
        }

        public void SetLegacy(bool bLegacy)
        {
            foreach (ProtoBufField field in m_rgFields)
            {
                field.Legacy = bLegacy;
            }
        }

        public void Add(ProtoBufField p)
        {
            m_rgFields.Add(p);
        }

        public void AddRange(ProtoBufFieldCollection col)
        {
            m_rgFields.AddRange(col.m_rgFields);
        }

        public ProtoBufFieldCollection FindAllChildren(string strName)
        {
            ProtoBufFieldCollection col = new common.ProtoBufFieldCollection();

            foreach (ProtoBufField field in m_rgFields)
            {
                if (field.FieldDesc != null && field.FieldDesc.Name == strName)
                    col.Add(field);
            }

            return col;
        }

        public ProtoBufField FindFirstChild(string strName)
        {
            foreach (ProtoBufField field in m_rgFields)
            {
                if (field.FieldDesc != null && field.FieldDesc.Name == strName)
                    return field;
            }

            return null;
        }

        public IEnumerator<ProtoBufField> GetEnumerator()
        {
            return m_rgFields.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return m_rgFields.GetEnumerator();
        }
    }


    class ProtoBufField /** @private */
    {
        FieldDescriptor m_fd;
        CodedInputStream m_strm;
        byte[] m_rgBytes;
        string m_strVal;
        int m_nVal = 0;
        long m_lVal = 0;
        float m_fVal = 0;
        double m_dfVal = 0;
        int[] m_rgnVal = null;
        long[] m_rglVal = null;
        float[] m_rgfVal = null;
        double[] m_rgdfVal = null;
        string m_strTag = null;
        bool m_bLegacy = false;

        TYPE m_type = TYPE.BYTES;
        ProtoBufFieldCollection m_col = new ProtoBufFieldCollection();
        int m_nField;
        WireFormat.WireType m_wireType;

        public enum TYPE
        {
            BYTES,
            STRING,
            BIT32,
            BIT64,
            ARRAY,
            FLOAT_ARRAY,
            DOUBLE_ARRAY,
            INT_ARRAY,
            LONG_ARRAY
        }

        public ProtoBufField(CodedInputStream strm, int nField, FieldDescriptor fd)
        {
            m_nField = nField;
            m_fd = fd;
            m_strm = strm;
        }

        public bool Load(WireFormat.WireType wireType)
        {
            m_wireType = wireType;

            switch (wireType)
            {
                case WireFormat.WireType.Varint:
                    m_lVal = m_strm.ReadInt32();
                    m_nVal = (int)m_lVal;
                    m_type = TYPE.BIT32;
                    break;

                case WireFormat.WireType.LengthDelimited:
                    ByteString bs = m_strm.ReadBytes(); 
                    if (bs.Length > 0)
                    {
                        m_rgBytes = bs.ToByteArray();

                        if (m_fd == null || m_fd.Type == FieldDescriptor.TYPE.STRING)
                            m_strVal = getString(m_rgBytes, out m_type);

                        if (m_type == TYPE.BYTES && m_fd != null && m_fd.Type != FieldDescriptor.TYPE.FIELDDESC)
                        {
                            switch (m_fd.Type)
                            {
                                case FieldDescriptor.TYPE.INT:
                                case FieldDescriptor.TYPE.UINT:
                                    m_rgnVal = readIntArray(m_rgBytes, m_fd.Type);
                                    m_type = TYPE.INT_ARRAY;
                                    break;

                                case FieldDescriptor.TYPE.LONG:
                                case FieldDescriptor.TYPE.ULONG:
                                    m_rglVal = readLongArray(m_rgBytes, m_fd.Type);
                                    m_type = TYPE.LONG_ARRAY;
                                    break;

                                case FieldDescriptor.TYPE.FLOAT:
                                    m_rgfVal = readFloatArray(m_rgBytes);
                                    m_type = TYPE.FLOAT_ARRAY;
                                    break;

                                case FieldDescriptor.TYPE.DOUBLE:
                                    m_rgdfVal = readDoubleArray(m_rgBytes);
                                    m_type = TYPE.DOUBLE_ARRAY;
                                    break;
                            }
                        }
                    }
                    break;

                case WireFormat.WireType.Fixed32:
                    float fVal = m_strm.ReadFloat();
                    m_nVal = (int)fVal;
                    m_fVal = (float)fVal;
                    m_type = TYPE.BIT32;
                    break;

                case WireFormat.WireType.Fixed64:
                    double dfVal = m_strm.ReadDouble();
                    m_lVal = (long)dfVal;
                    m_dfVal = (double)dfVal;
                    m_type = TYPE.BIT64;
                    break;

                default:
                    return false;
            }

            return true;
        }

        private int[] readIntArray(byte[] rgBytes, FieldDescriptor.TYPE type)
        {
            CodedInputStream strm = new CodedInputStream(rgBytes);
            List<int> rg = new List<int>();

            while (!strm.IsAtEnd)
            {
                int lVal = (type == FieldDescriptor.TYPE.INT) ? (int)strm.ReadInt32() : (int)strm.ReadUInt32();
                rg.Add(lVal);
            }

            return rg.ToArray();
        }

        private long[] readLongArray(byte[] rgBytes, FieldDescriptor.TYPE type)
        {
            CodedInputStream strm = new CodedInputStream(rgBytes);
            List<long> rg = new List<long>();

            while (!strm.IsAtEnd)
            {
                long lVal = (type == FieldDescriptor.TYPE.LONG) ? (long)strm.ReadInt64() : (long)strm.ReadUInt64();
                rg.Add(lVal);
            }

            return rg.ToArray();
        }

        private float[] readFloatArray(byte[] rgBytes)
        {
            int nCount = rgBytes.Length / sizeof(float);
            int nErr = rgBytes.Length % sizeof(float);

            if (nErr != 0)
                throw new Exception("Invalid " + m_fd.Type.ToString() + " data - not aligned.");

            CodedInputStream strm = new CodedInputStream(rgBytes);
            float[] rg = new float[nCount];

            for (int i = 0; i < nCount; i++)
            {
                rg[i] = strm.ReadFloat();
            }

            return rg;
        }

        private double[] readDoubleArray(byte[] rgBytes)
        {
            int nCount = rgBytes.Length / sizeof(double);
            int nErr = rgBytes.Length % sizeof(double);

            if (nErr != 0)
                throw new Exception("Invalid " + m_fd.Type.ToString() + " data - not aligned.");

            CodedInputStream strm = new CodedInputStream(rgBytes);
            double[] rg = new double[nCount];

            for (int i = 0; i < nCount; i++)
            {
                rg[i] = strm.ReadDouble();
            }

            return rg;
        }

        public void LoadSubFields(int nDepth = 0, int nMaxDepth = int.MaxValue, List<KeyValuePair<int, string>> rgIgnore = null)
        {
            ProtoBufFieldCollection col = null;

            if (m_type == TYPE.BYTES)
            {
                ProtoBufReader reader = new common.ProtoBufReader(m_rgBytes);
                col = reader.ReadFields(m_fd, false);
                m_col = col;
                m_type = TYPE.ARRAY;
            }
            else if (m_type == TYPE.ARRAY)
            {
                col = m_col;
            }

            if (col != null && col.Count > 0)
            {
                nDepth += 1;

                if (nDepth < nMaxDepth)
                {
                    if (rgIgnore != null)
                    {
                        foreach (KeyValuePair<int, string> kv in rgIgnore)
                        {
                            if (kv.Key <= m_col.Count &&
                                m_col[kv.Key].Type == TYPE.STRING &&
                                m_col[kv.Key].StringValue == kv.Value)
                                return;
                        }
                    }

                    foreach (ProtoBufField field in m_col)
                    {
                        field.LoadSubFields(nDepth, nMaxDepth);
                    }
                }
            }
        }

        private string getString(byte[] rg, out TYPE type)
        {
            string strOut = null;

            type = TYPE.BYTES;

            for (int i = 0; i < rg.Length; i++)
            {
                char ch = (char)rg[i];
                if (char.IsControl(ch))
                    return null;

                strOut += ch;
            }

            type = TYPE.STRING;

            return strOut;
        }

        private byte[] getBytes(string str, out TYPE type)
        {
            byte[] rg = new byte[str.Length];

            type = TYPE.STRING;

            for (int i = 0; i < str.Length; i++)
            {
                rg[i] = (byte)str[i];

                if (char.IsControl(str[i]))
                    type = TYPE.BYTES;
            }

            return rg;
        }

        public bool Legacy
        {
            get { return m_bLegacy; }
            set { m_bLegacy = value; }
        }

        public string Tag
        {
            get { return m_strTag; }
            set { m_strTag = value; }
        }

        public byte[] Bytes
        {
            get { return m_rgBytes; }
        }

        public int Length
        {
            get { return (m_rgBytes == null) ? 0 : m_rgBytes.Length; }
        }

        public TYPE Type
        {
            get { return m_type; }
        }

        public string StringValue
        {
            get { return m_strVal; }
        }

        public long LongValue
        {
            get { return m_lVal; }
        }

        public long[] LongValues
        {
            get { return m_rglVal; }
        }

        public int IntValue
        {
            get { return m_nVal; }
        }

        public int[] IntValues
        {
            get { return m_rgnVal; }
        }

        public float FloatValue
        {
            get { return m_fVal; }
        }

        public float[] FloatValues
        {
            get { return m_rgfVal; }
        }

        public double DoubleValue
        {
            get { return m_dfVal; }
        }

        public double[] DoubleValues
        {
            get { return m_rgdfVal; }
        }

        public ProtoBufFieldCollection Array
        {
            get { return m_col; }
        }

        public int FieldId
        {
            get { return m_nField; }
        }

        public FieldDescriptor FieldDesc
        {
            get { return m_fd; }
        }

        public override string ToString()
        {
            string strName = (m_fd == null) ? "NO FLDESC!" : m_fd.Name;
            string str = strName + "(" + m_nField.ToString() + ")[" + m_wireType.ToString() + "] " + m_type.ToString() + ": ";

            if (m_type == TYPE.STRING)
                return str + m_strVal;

            if (m_type == TYPE.BIT32)
                return str + m_nVal.ToString() + " (float = " + m_fVal.ToString() + ")";

            if (m_type == TYPE.BIT64)
                return str + m_lVal.ToString() + " (double = " + m_dfVal.ToString() + ")";

            if (m_type == TYPE.ARRAY)
                return str + " Count = " + m_col.Count.ToString();

            return str + " bytes = " + ((m_rgBytes == null) ? "0" : m_rgBytes.Length.ToString());
        }
    }

    public class FieldDescriptor /** @private */
    {
        List<FieldDescriptor> m_rgChildren = new List<FieldDescriptor>();
        int m_nFieldID = 0;
        string m_strName = "";
        TYPE m_type = TYPE.UNKNOWN;

        public enum TYPE
        {
            UNKNOWN,
            STRING,
            BOOL,
            INT,
            UINT,
            LONG,
            ULONG,
            FLOAT,
            DOUBLE,
            FIELDDESC
        }

        public FieldDescriptor(int nField, string strName, TYPE type, List<FieldDescriptor> rgChildren = null)
        {
            m_nFieldID = nField;
            m_strName = strName;
            m_type = type;

            if (rgChildren != null)
                m_rgChildren = rgChildren;
        }

        public FieldDescriptor FindFirstChild(int nFieldId)
        {
            foreach (FieldDescriptor fd in m_rgChildren)
            {
                if (fd.FieldId == nFieldId)
                    return fd;
            }

            return null;
        }

        public FieldDescriptor FindFirstChild(string strName)
        {
            foreach (FieldDescriptor fd in m_rgChildren)
            {
                if (fd.Name == strName)
                    return fd;
            }

            return null;
        }

        public int FieldId
        {
            get { return m_nFieldID; }
        }

        public string Name
        {
            get { return m_strName; }
        }

        public TYPE Type
        {
            get { return m_type; }
        }

        public List<FieldDescriptor> Children
        {
            get { return m_rgChildren; }
        }

        public override string ToString()
        {
            return m_strName + " (" + m_nFieldID.ToString() + ") - " + m_type.ToString();
        }

        public static FieldDescriptor CreateSolverStateFieldDesc()
        {
            return new common.FieldDescriptor(0, "SolverState", TYPE.FIELDDESC, loadSolverState());
        }

        public static FieldDescriptor CreateNetworkParamFieldDesc()
        {
            return new common.FieldDescriptor(0, "NetParameter", TYPE.FIELDDESC, loadNetParameter());
        }

        public static FieldDescriptor CreateBlobProtoDesc(int nFieldId)
        {
            return new FieldDescriptor(nFieldId, "BlobProto", TYPE.FIELDDESC, loadBlobProto());
        }

        private static List<FieldDescriptor> loadSolverState()
        {
            List<FieldDescriptor> rgF = new List<common.FieldDescriptor>();
            rgF.Add(new FieldDescriptor(1, "iter", TYPE.INT));
            rgF.Add(new FieldDescriptor(3, "history", TYPE.FIELDDESC, loadBlobProto()));
            rgF.Add(new FieldDescriptor(4, "current_step", TYPE.INT));
            return rgF;
        }

        private static List<FieldDescriptor> loadNetParameter()
        {
            List<FieldDescriptor> rgF = new List<common.FieldDescriptor>();
            rgF.Add(new FieldDescriptor(1, "name", TYPE.STRING));
            rgF.Add(new FieldDescriptor(100, "LayerParameter", TYPE.FIELDDESC, loadLayerParameter()));
            rgF.Add(new FieldDescriptor(2, "V1LayerParameter", TYPE.FIELDDESC, loadV1LayerParameter()));
            return rgF;
        }

        private static List<FieldDescriptor> loadLayerParameter()
        {
            List<FieldDescriptor> rgF = new List<common.FieldDescriptor>();
            rgF.Add(new FieldDescriptor(1, "name", FieldDescriptor.TYPE.STRING));
            rgF.Add(new FieldDescriptor(2, "type", FieldDescriptor.TYPE.STRING));
            rgF.Add(new FieldDescriptor(3, "bottom", FieldDescriptor.TYPE.STRING));
            rgF.Add(new FieldDescriptor(4, "top", FieldDescriptor.TYPE.STRING));
            rgF.Add(new FieldDescriptor(10, "phase", FieldDescriptor.TYPE.INT));
            rgF.Add(new FieldDescriptor(5, "loss_weight", FieldDescriptor.TYPE.FLOAT));
            rgF.Add(new FieldDescriptor(6, "param", FieldDescriptor.TYPE.FIELDDESC, loadParamSpec()));
            rgF.Add(new FieldDescriptor(7, "blobs", FieldDescriptor.TYPE.FIELDDESC, loadBlobProto()));
            rgF.Add(new FieldDescriptor(11, "prop_down", FieldDescriptor.TYPE.BOOL));
            rgF.Add(new FieldDescriptor(8, "include", FieldDescriptor.TYPE.FIELDDESC, loadNetStateRule()));
            rgF.Add(new FieldDescriptor(9, "exclude", FieldDescriptor.TYPE.FIELDDESC, loadNetStateRule()));
            rgF.Add(new FieldDescriptor(100, LayerParameter.LayerType.TRANSFORM.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(101, LayerParameter.LayerType.LOSS.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));

            rgF.Add(new FieldDescriptor(102, LayerParameter.LayerType.ACCURACY.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(103, LayerParameter.LayerType.ARGMAX.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(139, LayerParameter.LayerType.BATCHNORM.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(141, LayerParameter.LayerType.BIAS.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(104, LayerParameter.LayerType.CONCAT.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(105, LayerParameter.LayerType.CONTRASTIVE_LOSS.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(106, LayerParameter.LayerType.CONVOLUTION.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC, loadConvolutionParam()));
            rgF.Add(new FieldDescriptor(144, LayerParameter.LayerType.CROP.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(107, LayerParameter.LayerType.DATA.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(108, LayerParameter.LayerType.DROPOUT.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(109, LayerParameter.LayerType.DUMMYDATA.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(110, LayerParameter.LayerType.ELTWISE.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(140, LayerParameter.LayerType.ELU.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(137, LayerParameter.LayerType.EMBED.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(111, LayerParameter.LayerType.EXP.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(135, LayerParameter.LayerType.FLATTEN.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(112, "hdf5_input_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(113, "hdf5_output_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(114, LayerParameter.LayerType.HINGE_LOSS.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(115, "image_data_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(116, LayerParameter.LayerType.INFOGAIN_LOSS.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(117, LayerParameter.LayerType.INNERPRODUCT.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(143, LayerParameter.LayerType.INPUT.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(134, LayerParameter.LayerType.LOG.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(118, LayerParameter.LayerType.LRN.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(119, LayerParameter.LayerType.MEMORYDATA.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(120, LayerParameter.LayerType.MVN.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(121, LayerParameter.LayerType.PARAMETER.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(121, LayerParameter.LayerType.POOLING.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(122, LayerParameter.LayerType.POWER.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(131, LayerParameter.LayerType.PRELU.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(130, "python_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(146, LayerParameter.LayerType.RECURRENT.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(136, LayerParameter.LayerType.REDUCTION.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(123, LayerParameter.LayerType.RELU.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(133, LayerParameter.LayerType.RESHAPE.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(142, LayerParameter.LayerType.SCALE.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(124, LayerParameter.LayerType.SIGMOID.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(125, LayerParameter.LayerType.SOFTMAX.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(132, LayerParameter.LayerType.SPP.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(126, LayerParameter.LayerType.SLICE.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(127, LayerParameter.LayerType.TANH.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(128, LayerParameter.LayerType.THRESHOLD.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(138, LayerParameter.LayerType.TILE.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(129, "window_data_param", FieldDescriptor.TYPE.FIELDDESC));

            rgF.Add(new FieldDescriptor(900, LayerParameter.LayerType.BINARYHASH.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));

            return rgF;
        }

        private static List<FieldDescriptor> loadV1LayerParameter()
        {
            List<FieldDescriptor> rgF = new List<common.FieldDescriptor>();
            rgF.Add(new FieldDescriptor(2, "bottom", FieldDescriptor.TYPE.STRING));
            rgF.Add(new FieldDescriptor(3, "top", FieldDescriptor.TYPE.STRING));
            rgF.Add(new FieldDescriptor(4, "name", FieldDescriptor.TYPE.STRING));
            rgF.Add(new FieldDescriptor(32, "include", FieldDescriptor.TYPE.FIELDDESC, loadNetStateRule()));
            rgF.Add(new FieldDescriptor(33, "exclude", FieldDescriptor.TYPE.FIELDDESC, loadNetStateRule()));
            rgF.Add(new FieldDescriptor(5, "type", FieldDescriptor.TYPE.INT));
            rgF.Add(new FieldDescriptor(6, "blobs", FieldDescriptor.TYPE.FIELDDESC, loadBlobProto()));
            rgF.Add(new FieldDescriptor(1001, "param", FieldDescriptor.TYPE.FIELDDESC, loadParamSpec()));
            rgF.Add(new FieldDescriptor(1002, "blob_share_mode", FieldDescriptor.TYPE.INT));
            rgF.Add(new FieldDescriptor(7, "blobs_lr", FieldDescriptor.TYPE.FLOAT));
            rgF.Add(new FieldDescriptor(8, "weight_decay", FieldDescriptor.TYPE.FLOAT));
            rgF.Add(new FieldDescriptor(35, "loss_weight", FieldDescriptor.TYPE.FLOAT));

            rgF.Add(new FieldDescriptor(27, LayerParameter.LayerType.ACCURACY.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(23, LayerParameter.LayerType.ARGMAX.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(9, LayerParameter.LayerType.CONCAT.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(40, LayerParameter.LayerType.CONTRASTIVE_LOSS.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(10, LayerParameter.LayerType.CONVOLUTION.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC, loadConvolutionParam()));
            rgF.Add(new FieldDescriptor(11, LayerParameter.LayerType.DATA.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(12, LayerParameter.LayerType.DROPOUT.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(26, LayerParameter.LayerType.DUMMYDATA.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(24, LayerParameter.LayerType.ELTWISE.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(41, LayerParameter.LayerType.EXP.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(13, "hdf5_input_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(14, "hdf5_output_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(29, LayerParameter.LayerType.HINGE_LOSS.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(15, "image_data_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(16, LayerParameter.LayerType.INFOGAIN_LOSS.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(17, LayerParameter.LayerType.INNERPRODUCT.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(18, LayerParameter.LayerType.LRN.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(22, LayerParameter.LayerType.MEMORYDATA.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(34, LayerParameter.LayerType.MVN.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(19, LayerParameter.LayerType.POOLING.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(21, LayerParameter.LayerType.POWER.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(30, LayerParameter.LayerType.RELU.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(38, LayerParameter.LayerType.SIGMOID.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(39, LayerParameter.LayerType.SOFTMAX.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(31, LayerParameter.LayerType.SLICE.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(37, LayerParameter.LayerType.TANH.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(25, LayerParameter.LayerType.THRESHOLD.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(20, "window_data_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(36, LayerParameter.LayerType.TRANSFORM.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));
            rgF.Add(new FieldDescriptor(42, LayerParameter.LayerType.LOSS.ToString() + "_param", FieldDescriptor.TYPE.FIELDDESC));

            return rgF;
        }

        private static List<FieldDescriptor> loadParamSpec()
        {
            List<FieldDescriptor> rgF = new List<common.FieldDescriptor>();
            rgF.Add(new FieldDescriptor(1, "name", FieldDescriptor.TYPE.STRING));
            rgF.Add(new FieldDescriptor(2, "share_mode", FieldDescriptor.TYPE.INT));
            rgF.Add(new FieldDescriptor(3, "lr_mult", FieldDescriptor.TYPE.FLOAT));
            rgF.Add(new FieldDescriptor(4, "decay_mult", FieldDescriptor.TYPE.FLOAT));
            return rgF;
        }

        private static List<FieldDescriptor> loadBlobShape()
        {
            List<FieldDescriptor> rgF = new List<common.FieldDescriptor>();
            rgF.Add(new FieldDescriptor(1, "dim", FieldDescriptor.TYPE.LONG));
            return rgF;
        }

        private static List<FieldDescriptor> loadBlobProto()
        {
            List<FieldDescriptor> rgF = new List<common.FieldDescriptor>();
            rgF.Add(new FieldDescriptor(7, "shape", FieldDescriptor.TYPE.FIELDDESC, loadBlobShape()));
            rgF.Add(new FieldDescriptor(5, "data", FieldDescriptor.TYPE.FLOAT));
            rgF.Add(new FieldDescriptor(6, "diff", FieldDescriptor.TYPE.FLOAT));
            rgF.Add(new FieldDescriptor(8, "double_data", FieldDescriptor.TYPE.DOUBLE));
            rgF.Add(new FieldDescriptor(9, "double_diff", FieldDescriptor.TYPE.DOUBLE));
            rgF.Add(new FieldDescriptor(1, "num", FieldDescriptor.TYPE.INT));
            rgF.Add(new FieldDescriptor(2, "channels", FieldDescriptor.TYPE.INT));
            rgF.Add(new FieldDescriptor(3, "height", FieldDescriptor.TYPE.INT));
            rgF.Add(new FieldDescriptor(4, "width", FieldDescriptor.TYPE.INT));
            return rgF;
        }

        private static List<FieldDescriptor> loadNetStateRule()
        {
            List<FieldDescriptor> rgF = new List<common.FieldDescriptor>();
            rgF.Add(new FieldDescriptor(1, "phase", FieldDescriptor.TYPE.INT));
            rgF.Add(new FieldDescriptor(2, "min_level", FieldDescriptor.TYPE.INT));
            rgF.Add(new FieldDescriptor(3, "max_level", FieldDescriptor.TYPE.INT));
            rgF.Add(new FieldDescriptor(4, "stage", FieldDescriptor.TYPE.STRING));
            rgF.Add(new FieldDescriptor(5, "not_stage", FieldDescriptor.TYPE.STRING));
            return rgF;
        }

        private static List<FieldDescriptor> loadFillerParam()
        {
            List<FieldDescriptor> rgF = new List<common.FieldDescriptor>();
            rgF.Add(new FieldDescriptor(1, "type", FieldDescriptor.TYPE.STRING));
            rgF.Add(new FieldDescriptor(2, "value", FieldDescriptor.TYPE.FLOAT));
            rgF.Add(new FieldDescriptor(3, "min", FieldDescriptor.TYPE.FLOAT));
            rgF.Add(new FieldDescriptor(4, "max", FieldDescriptor.TYPE.FLOAT));
            rgF.Add(new FieldDescriptor(5, "mean", FieldDescriptor.TYPE.FLOAT));
            rgF.Add(new FieldDescriptor(6, "std", FieldDescriptor.TYPE.FLOAT));
            rgF.Add(new FieldDescriptor(7, "sparse", FieldDescriptor.TYPE.INT));
            rgF.Add(new FieldDescriptor(8, "variance_norm", FieldDescriptor.TYPE.INT));
            return rgF;
        }

        private static List<FieldDescriptor> loadConvolutionParam()
        {
            List<FieldDescriptor> rgF = new List<common.FieldDescriptor>();
            rgF.Add(new FieldDescriptor(1, "num_output", FieldDescriptor.TYPE.UINT));
            rgF.Add(new FieldDescriptor(2, "bias_term", FieldDescriptor.TYPE.BOOL));
            rgF.Add(new FieldDescriptor(3, "pad", FieldDescriptor.TYPE.UINT));
            rgF.Add(new FieldDescriptor(4, "kernel_size", FieldDescriptor.TYPE.UINT));
            rgF.Add(new FieldDescriptor(6, "stride", FieldDescriptor.TYPE.UINT));
            rgF.Add(new FieldDescriptor(18, "dilation", FieldDescriptor.TYPE.UINT));
            rgF.Add(new FieldDescriptor(9, "pad_h", FieldDescriptor.TYPE.UINT));
            rgF.Add(new FieldDescriptor(10, "pad_w", FieldDescriptor.TYPE.UINT));
            rgF.Add(new FieldDescriptor(11, "kernel_h", FieldDescriptor.TYPE.UINT));
            rgF.Add(new FieldDescriptor(12, "kernel_w", FieldDescriptor.TYPE.UINT));
            rgF.Add(new FieldDescriptor(13, "stride_h", FieldDescriptor.TYPE.UINT));
            rgF.Add(new FieldDescriptor(14, "stride_w", FieldDescriptor.TYPE.UINT));
            rgF.Add(new FieldDescriptor(5, "group", FieldDescriptor.TYPE.UINT));
            rgF.Add(new FieldDescriptor(7, "weight_filler", FieldDescriptor.TYPE.FIELDDESC, loadFillerParam()));
            rgF.Add(new FieldDescriptor(8, "bias_filler", FieldDescriptor.TYPE.FIELDDESC, loadFillerParam()));
            rgF.Add(new FieldDescriptor(15, "engine", FieldDescriptor.TYPE.INT));
            rgF.Add(new FieldDescriptor(16, "axis", FieldDescriptor.TYPE.INT));
            rgF.Add(new FieldDescriptor(17, "force_nd", FieldDescriptor.TYPE.BOOL));
            return rgF;
        }
    }
}
