The TokenizedDataPairsLayerPy uses data created using the 'generate_data.py' Python script.

To create the dataset for training, follow these steps.

1.) Installing the MyCaffe Test Application also installs the pre-trained sentence piece model (spm.model) and the 
Python script (generate_data.py) used to create the dataset. The script is located in the 'examples\python\mycaffe\test\text' folder.
The original sentencepeice model was created by following the steps outlined at: https://github.com/devjwsong/transformer-translator-pytorch
A pre-trained sentence piece model is provided at C:\ProgramData\MyCaffe\test_data\data\text\encdec

2.) Create the dataset by running the 'generate_data.py' script. The script will create the dataset as a set of numpy arrays in the 
C:\ProgramData\MyCaffe\test_data\data\text\encdec\cache\train and
C:\ProgramData\MyCaffe\test_data\data\text\encdec\cahce\valid folders.

3.) Each numpy array has a size (40,200) where the first dimension is the number of samples and the second dimension is the number of tokens. An array
matching array is created for each the encoder input ('enc'), decoder input ('dec') and decoder target ('tgt').  When loading the dataset for training
these are all loaded and organized together.

4.) Before training, 
set the 'm_param.tokenized_data_pairs_param.source' parameter of the TokenizedDataPairsLayerPy running in the TRAIN phase
to the 'C:\ProgramData\MyCaffe\test_data\data\text\encdec\cache\train' folder 
and the 'm_param.tokenized_data_pairs_param.target' parameter of the TokenizedDataPairsLayerPy running in the TEST phase
to the 'C:\ProgramData\MyCaffe\test_data\data\text\encdec\cache\valid' folder.


