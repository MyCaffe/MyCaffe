using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using HDF5DotNet;
using MyCaffe.basecode;
using MyCaffe.common;

namespace MyCaffe.layers.hdf5
{
    /// <summary>
    /// The HDF5Load interface is used to load weights into objects like a Net.
    /// </summary>
    /// <typeparam name="T">Specifies the base type of 'double' or 'float'.</typeparam>
    public interface IHDF5Load<T>
    {
        /// <summary>
        /// Copy the weights from an HDF5 file into a Net.
        /// </summary>
        /// <param name="net">Specifies the destination Net.</param>
        /// <param name="log">Specifies the output log.</param>
        /// <param name="strFile">Specifies the HDF5 file containing the source weights.</param>
        void CopyTrainedLayersFromHDF5(Net<T> net, Log log, string strFile);
    }

    /// <summary>
    /// The HDF5 object provides HDF5 dataset support to the HDF5DataLayer.
    /// </summary>
    /// <typeparam name="T">Specifies the base type.</typeparam>
    public class HDF5<T> : IDisposable, IHDF5Load<T>
    {
        Log m_log;
        CudaDnn<T> m_cuda;
        H5FileId m_file;
        string m_strFile;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="strFile">Specifies the HDF5 file to load.</param>
        public HDF5(CudaDnn<T> cuda, Log log, string strFile)
        {
            m_strFile = strFile;
            m_cuda = cuda;
            m_log = log;

            m_file = H5F.open(strFile, H5F.OpenMode.ACC_RDONLY);
            if (m_file == null)
                m_log.FAIL("Failed opening HDF5 file: '" + strFile + "'!");
        }

        /// <summary>
        /// The constructor used when loading weights into a Net.
        /// </summary>
        public HDF5()
        {
        }

        private int get_num_links(H5GroupId hG)
        {
            H5GInfo info = H5G.getInfo(hG);
            return (int)info.nLinks;
        }

        private string get_name_by_idx(H5GroupId hg, int i)
        {
            return H5L.getNameByIndex(hg, ".", H5IndexType.NAME, H5IterationOrder.NATIVE, i);
        }

        private Tuple<H5DataSetId, int> load_nd_datasetEx(Blob<T> blob, string strDatasetName, bool bReshape, int nMinDim = 1, int nMaxDim = int.MaxValue, H5GroupId id = null, bool bAllowSingleItems = false)
        {
            H5DataSetId ds = null;
            int nSingleItemSize = 0;

            try
            {
                if (id != null)
                    ds = H5D.open(id, strDatasetName);
                else
                    ds = H5D.open(m_file, strDatasetName);

                if (ds == null)
                    m_log.FAIL("Failed to find the dataset '" + strDatasetName + "'!");

                // Verify that the number of dimensions are in the accepted range.
                H5DataSpaceId dsSpace = H5D.getSpace(ds);
                if (dsSpace == null)
                    m_log.FAIL("Failed to get the dataset space!");

                int nDims = H5S.getSimpleExtentNDims(dsSpace);
                m_log.CHECK_GE(nDims, nMinDim, "The dataset dim is out of range!");
                m_log.CHECK_LE(nDims, nMaxDim, "The dataset dim is out of range!");

                long[] rgDims = H5S.getSimpleExtentDims(dsSpace);

                // Verify that the data format is what we expect: float or double
                H5DataTypeId dsType = H5D.getType(ds);
                if (dsType == null)
                    m_log.FAIL("Failed to get the dataset type!");

                H5T.H5TClass dataClass = H5T.getClass(dsType);
                switch (dataClass)
                {
                    case H5T.H5TClass.FLOAT:
                        m_log.WriteLine("Datatype class: H5T_FLOAT");
                        break;

                    case H5T.H5TClass.INTEGER:
                        m_log.WriteLine("Datatype class: H5T_INTEGER");
                        break;

                    default:
                        m_log.FAIL("Unsupported datatype class: " + dataClass.ToString());
                        break;
                }

                List<int> rgBlobDims = new List<int>();
                for (int i = 0; i < nDims; i++)
                {
                    rgBlobDims.Add((int)rgDims[i]);
                }

                if (bReshape)
                {
                    blob.Reshape(rgBlobDims);
                }
                else
                {
                    if (!Utility.Compare<int>(rgBlobDims, blob.shape()))
                    {
                        if (!bAllowSingleItems || (rgBlobDims.Count == 1 && rgBlobDims[0] != 1))
                        {
                            string strSrcShape = Utility.ToString<int>(rgBlobDims);
                            m_log.FAIL("Cannot load blob from  hdf5; shape mismatch.  Source shape = " + strSrcShape + ", target shape = " + blob.shape_string);
                        }

                        if (rgBlobDims.Count == 1)
                            nSingleItemSize = rgBlobDims[0];
                    }
                }
            }
            catch (Exception excpt)
            {
                if (ds != null)
                {
                    H5D.close(ds);
                    ds = null;
                }

                throw excpt;
            }

            return new Tuple<H5DataSetId, int>(ds, nSingleItemSize);
        }

        /// <summary>
        /// Creates a new dataset from an HDF5 data file.
        /// </summary>
        /// <param name="blob">The input blob is reshaped to the dataset item shape.</param>
        /// <param name="strDatasetName">Specifies the new dataset name.</param>
        /// <param name="bReshape">Specifies whether to reshape the 'blob' parameter.</param>
        /// <param name="nMinDim">Specifies the minimum dimension.</param>
        /// <param name="nMaxDim">Specifies the maximum dimension.</param>
        /// <param name="id">Optional, group ID to use instead of internal file (default = null).</param>
        /// <param name="bAllowSingleItems">When true single item values are allowed and used to copy across entire blob.</param>
        public void load_nd_dataset(Blob<T> blob, string strDatasetName, bool bReshape = false, int nMinDim = 1, int nMaxDim = int.MaxValue, H5GroupId id = null, bool bAllowSingleItems = false)
        {
            H5DataSetId ds = null;
            int nSingleItemSize = 0;

            try
            {
                Tuple<H5DataSetId, int> ds1 = load_nd_datasetEx(blob, strDatasetName, bReshape, nMinDim, nMaxDim, id, bAllowSingleItems);
                ds = ds1.Item1;
                nSingleItemSize = ds1.Item2;

                H5DataTypeId dsType = H5D.getType(ds);
                int nSize = H5T.getSize(dsType);

                if (nSize == sizeof(double))
                {
                    double[] rgBuffer = new double[blob.count()];
                    H5Array<double> rgData = new H5Array<double>(rgBuffer);

                    H5D.read<double>(ds, dsType, rgData);

                    if (!bAllowSingleItems || nSingleItemSize == 0)
                        blob.mutable_cpu_data = Utility.ConvertVec<T>(rgBuffer);
                    else
                        blob.SetData(rgBuffer[0]);
                }
                else if (nSize == sizeof(float))
                {
                    float[] rgBuffer = new float[blob.count()];
                    H5Array<float> rgData = new H5Array<float>(rgBuffer);

                    H5D.read<float>(ds, dsType, rgData);

                    if (!bAllowSingleItems || nSingleItemSize == 0)
                        blob.mutable_cpu_data = Utility.ConvertVec<T>(rgBuffer);
                    else
                        blob.SetData(rgBuffer[0]);
                }
                else if (nSize == sizeof(byte))
                {
                    byte[] rgBuffer = new byte[blob.count()];
                    H5Array<byte> rgData = new H5Array<byte>(rgBuffer);

                    H5D.read<byte>(ds, dsType, rgData);

                    float[] rgf = rgBuffer.Select(p1 => (float)p1).ToArray();
                    blob.mutable_cpu_data = Utility.ConvertVec<T>(rgf);
                }
                else
                    m_log.FAIL("The dataset size of '" + nSize.ToString() + "' is not supported!");
            }
            catch (Exception excpt)
            {
                m_log.FAIL(excpt.Message);
            }
            finally
            {
                if (ds != null)
                    H5D.close(ds);
            }
        }

        /// <summary>
        /// Copy the weights from an HDF5 file into a Net.
        /// </summary>
        /// <param name="net">Specifies the destination Net.</param>
        /// <param name="log">Specifies the output log.</param>
        /// <param name="strFile">Specifies the HDF5 file containing the source weights.</param>
        public void CopyTrainedLayersFromHDF5(Net<T> net, Log log, string strFile)
        {
            m_log = log;

            if (m_file == null)
            {
                m_file = H5F.open(strFile, H5F.OpenMode.ACC_RDONLY);
                if (m_file == null)
                    m_log.FAIL("Failed opening HDF5 weights file: '" + strFile + "'!");
            }

            H5GroupId hData = null;
            H5GroupId hLayer = null;

            try
            {
                hData = H5G.open(m_file, "data");
                if (hData == null || hData.Id <= 0)
                    m_log.FAIL("Failed to open the 'data' data stream within the HDF5 weights file: '" + strFile + "'!");

                int nNumLayers = get_num_links(hData);

                for (int i = 0; i < nNumLayers; i++)
                {
                    string strSrcLayerName = get_name_by_idx(hData, i);

                    if (net.layer_index_by_name(strSrcLayerName) < 0)
                    {
                        m_log.WriteLine("Ignoring source layer '" + strSrcLayerName + "'.");
                        continue;
                    }

                    int nTargetLayerId = net.layer_index_by_name(strSrcLayerName);
                    m_log.WriteLine("Copying source layer '" + strSrcLayerName + "'.");

                    BlobCollection<T> targetBlobs = net.layers[nTargetLayerId].blobs;

                    hLayer = H5G.open(hData, strSrcLayerName);
                    if (hLayer == null || hLayer.Id <= 0)
                        m_log.FAIL("Failed to open '" + strSrcLayerName + "' layer in data stream within the HDF5 weights file: '" + strFile + "'!");

                    // Check that source layer doesnt have more params than target layer.
                    int nNumSourceParams = get_num_links(hLayer);
                    m_log.CHECK_LE(nNumSourceParams, targetBlobs.Count, "Incompatible number of blobs for layer '" + strSrcLayerName + "'!");

                    for (int j = 0; j < nNumSourceParams; j++)
                    {
                        string strDatasetName = j.ToString();

                        if (!H5L.Exists(hLayer, strDatasetName))
                        {
                            // Target param doesnt exist in source weights...
                            int nTargetNetParamId = net.param_names_index[strDatasetName];
                            if (net.param_owners.Contains(nTargetNetParamId))
                            {
                                // ...but its weight-shared in target, thats fine.
                                continue;
                            }
                            else
                            {
                                m_log.FAIL("Incompatible number of blobs for layer '" + strSrcLayerName + "'!");
                            }
                        }

                        load_nd_dataset(targetBlobs[j], strDatasetName, false, 1, int.MaxValue, hLayer, true);
                    }

                    H5G.close(hLayer);
                    hLayer = null;
                }

                H5G.close(hData);
                hData = null;
            }
            finally
            {
                if (hLayer != null)
                    H5G.close(hLayer);

                if (hData != null)
                    H5G.close(hData);
            }
        }

        /// <summary>
        /// Release all resources uses.
        /// </summary>
        public void Dispose()
        {
            if (m_file != null)
            {
                H5F.close(m_file);
                m_file = null;
            }
        }
    }
}
