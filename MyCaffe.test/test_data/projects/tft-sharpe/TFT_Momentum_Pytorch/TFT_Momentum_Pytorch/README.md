========================================
Original Source Files - TFT
========================================

https://github.com/PlaytikaOSS/tft-torch/blob/main/docs/build/html/tutorials/TrainingExample.ipynb
https://github.com/PlaytikaOSS/tft-torch

License MIT: https://github.com/PlaytikaOSS/tft-torch/blob/main/LICENSE

Includes files in the following directories:

	\tft_torch

========================================
Original Source Files - Trading Momentum Transformer
========================================

https://github.com/kieranjwood/trading-momentum-transformer

License MIT: https://github.com/kieranjwood/trading-momentum-transformer/blob/master/LICENSE

Includes files in the following directories:

    \data
	\examples
	\mom_trans
	\settings

NOTE: The file TFT_Momentum_Pytorch.py merges the data input from the Trading Momentum Transformer with the TFT model in tft_torch.  These files have 
been modified to output test data and optionally work with portions of MyCaffe.

========================================
Test Data Creation Process
========================================

Follow the steps #1-3 under 'Using the code' of https://github.com/kieranjwood/trading-momentum-transformer, in summary:

1.) Create a Nasdaq Data Link account located at https://data.nasdaq.com/data/CHRIS-wiki-continuous-futures/documentation
2.) Download the Nasdaq Data Link commodity data by running 'data.download_data.py' located in the 'data' directory.
3.) Run the 'examples.create_features_quandl.py' located in the 'examples' directory to create the 'quandl_cpd_nonelbw.csv' file. (Note script parameters are set to empty '').


========================================
General Use
========================================

run the 'TFT_Momentum_Pytorch.py' script to train and test the model.  The script will output the test results to the console.

When running the script uses the ModelFeatures from the Trading Momentum Transformer project to load the data from the 'quandl_cpd_nonelbw.csv' file.  
The script then uses the TFT model from the tft_torch project to train and test the model.  The TFT model has been altered to support the decoder only
configuration expected by the Trading Momentum Transformer project.

