﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.db.image;
using MyCaffe.layers.python.layers.python;
using MyCaffe.param;

/// <summary>
/// The MyCaffe.layers.python namespace contains all python layers where the layer provides a wrapper to the python implementation.
/// </summary>
namespace MyCaffe.layers.python
{
    /// <summary>
    /// The LayerFactor is responsible for creating all layers implemented in the MyCaffe.layers.ssd namespace.
    /// </summary>
    public class LayerFactory : ILayerCreator
    {
        /// <summary>
        /// Create the layers when using the <i>double</i> base type.
        /// </summary>
        /// <param name="cuda">Specifies the connection to the low-level CUDA interfaces.</param>
        /// <param name="log">Specifies the output log.</param>
        /// <param name="p">Specifies the layer parameter.</param>
        /// <param name="evtCancel">Specifies the cancellation event.</param>
        /// <param name="db">Specifies an interface to the in-memory database, who's use is optional.</param>
        /// <returns>If supported, the layer is returned, otherwise <i>null</i> is returned.</returns>
        public Layer<double> CreateDouble(CudaDnn<double> cuda, Log log, LayerParameter p, CancelEvent evtCancel, IXDatabaseBase db)
        {
            switch (p.type)
            {
                case LayerParameter.LayerType.TOKENIZED_DATA_PAIRS_PY:
                    return new TokenizedDataPairsLayerPy<double>(cuda, log, p, db, evtCancel);

                default:
                    return null;
            }
        }

        /// <summary>
        /// Create the layers when using the <i>float</i> base type.
        /// </summary>
        /// <param name="cuda">Specifies the connection to the low-level CUDA interfaces.</param>
        /// <param name="log">Specifies the output log.</param>
        /// <param name="p">Specifies the layer parameter.</param>
        /// <param name="evtCancel">Specifies the cancellation event.</param>
        /// <param name="db">Specifies an interface to the in-memory database, who's use is optional.</param>
        /// <returns>If supported, the layer is returned, otherwise <i>null</i> is returned.</returns>
        public Layer<float> CreateSingle(CudaDnn<float> cuda, Log log, LayerParameter p, CancelEvent evtCancel, IXDatabaseBase db)
        {
            switch (p.type)
            {
                case LayerParameter.LayerType.TOKENIZED_DATA_PAIRS_PY:
                    return new TokenizedDataPairsLayerPy<float>(cuda, log, p, db, evtCancel);
                    
                default:
                    return null;
            }
        }
    }
}
