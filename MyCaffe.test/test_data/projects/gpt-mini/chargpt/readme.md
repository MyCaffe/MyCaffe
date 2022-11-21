This version of the minGPT project is highly augmented to generate test data used to test
the MyCaffe implementation of GPT.

(original minGPT version by Andrej Karpathy located at https://github.com/karpathy/minGPT 
and distributed under the MIT license https://github.com/karpathy/minGPT/blob/master/LICENSE) 

All generated files are placed in the 'c:\temp\snap' directory.

To generate test data, the following configuration settings have been added to the project.

chargpt.py, line 34 - by default the 'gpt-mini' project is used which produces a 6 layer, 6 head,
192 embed model.  For other model configurations, see model.py, lines 1328-1351.  The 'gpt-mini'
model is the default model used by the original project.

Testing model configurations include:

gpt-pico - this is a tiny 1-layer model with 1 head and 3 sized embed, with a 4 sized block, batch 1.
gpt-pico3 - this is a tiny 1-layer model with 3 heads and 3 sized embed, with a 4 sized block, batch 1.
gpt-pico3B - this is a tiny 1 layer model with 3 haeds and 3 sized embed, with a 4 sized block, batch 2.
gpt-pico3B5 - this is a tiny 1 layer model with 3 haeds and 3 sized embed, with a 4 sized block, batch 5.

These models are used to generate the Pico test data for the following pico related tests:

 TestCausalSelfAttentionLayer, (Turn on data creation by setting CausalSelfAttention.debug=True at model.py @ line 106)
 TestLayerNormLayer,           (Turn on data creation by setting LayerNormalization[2].debug=True at model.py @ line 310)
 TestTransformerBlock          (Turn on data creation by setting Block.debug=True at model.py @ line 558)

gpt-mini1, this is a 2-layer, 6-head, 192-embed model with all dropout turned off for testing.

This model is used to generate the GptMini1 data for the following Gpt related tests:

 TestTransformerBlock::TestTrainingGptMini1 (Turn on data creation by setting Trainer.debug=True at trainer.py @ line 50)

NOTE: Some of the data generation settings generate a lot of data and dramatically slow down training.  When generating
data, typically the data generation is turned on for one of the tests and a break point is set at the start and end of
the corresponding 'forward' function.  During the first pass of the 'forward' all in-bound data to each layer is saved
and then between the first 'forward' and second 'forward' call, the gradient values are saved during the backward pass.

Data generation is not needed for the auto tests to opererate for the installation includes pre-generated data. 



 