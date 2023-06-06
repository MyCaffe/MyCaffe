========================================
Original Source Files
========================================

https://github.com/PlaytikaOSS/tft-torch/blob/main/docs/build/html/tutorials/TrainingExample.ipynb
https://github.com/PlaytikaOSS/tft-torch

License MIT: https://github.com/PlaytikaOSS/tft-torch/blob/main/LICENSE

========================================
Test Data Creation Process
========================================

1. Create the raw data by running ‘dataset_creation.py’ which produces ‘data.pickle’ file.  This takes a long time (e.g. a day) and places the ‘data.pickle’ file in \data\favorita directory
2. Run ‘data.py’ with nMax = 10000 to create ‘data.small.10000.pickle’ which is a smaller subset of the main ‘data.pickle’ file with only the first 10000 entries.  This also creates the data numpy files used when testing with MyCaffe.
3. Run each of the scripts in the following order:

	a.) training_flavorita.py, run up to line 640 'end_iter_time = time.time()' with settings:
		# Settings for debugging and generating data
		debug = True
		use_mycaffe = True
		use_mycaffe_data = False
		use_mycaffe_model_direct = False
		lstm_use_mycaffe = True
		use_mycaffe_model = False
		tag = "tft.all"
		test = False
		train_batch = 16
		test_batch = 16

	b.) test_1_inputchannelembedding_future.py
	c.) test_1_inputchannelembedding_hist.py
	d.) test_1_inputchannelembedding_stat.py
	e.) test_1_transforminputs.py
	f.) test_10_loss_focused.py
	g.) test_11_tft_full.py
	h.) test_1a_categoricalinputtransformation.py
	i.) test_1a_numericalinputtransformation.py
	j.) test_2_variableselectionnetwork_fut.py
	k.) test_2_variableselectionnetwork_hist.py
	l.) test_2_variableselectionnetwork_stat.py
	m.) test_3b_grn.py
	n.) test_4_sequential_processing.py
	o.) test_5_static_enrichment_focused.py
	p.) test_8_interpmultheadattn_hist_focused.py
	q.) test_8a_glu_imha.py
	r.) test_8b_gateaddnorm_imha.py

Generated files are placed in separate directories for each script and are created in the '\test' directory.  Tests may use data found within the '\data' directory.
To generate data in the '\data'directory do the following:

	a.) '\data\electricity\preprocessed' - run SignalPop AI Designer Electricity Dataset Creator
	b.) '\data\traffic\preprocessed' - run SignalPop AI Designer Traffic Dataset Creator
	c.) '\data\volatility\preprocessed' - run SignalPop AI Designer Volatility Dataset Creator
