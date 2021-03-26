## Students Names : 
  Mohamed Amine HACHICHA : mohamed-amine.hachicha@student.ecp.fr
  Khalil BERGAOUI: khalil.bergaoui@student.ecp.fr
  Firas DHAHA : firas.dhaha@student.ecp.fr
  Hamza GHARBI : hamza.gharbi@supelec.fr
  

## Final System
  After trying a couple of methods (mainly based on sentence parsing and dependency graphs), 
the system that gave the best results and that we're submitting is based on the fine-tuning
of a BERT model using the transformers library of HuggingFace and PyTorch.

  To make use of the target term, we used a configuration that is inspired from question answering
system : we take as input the review and the target term separated by a [SEP] token and we add 
segment IDs (or token_type_ids) to let BERT know what segment of the input corresponds to the review
(segment A) and which corresponds to the target term (segment B).

  To fine tune BERT for text classification (to predict sentiment), we have to tokenize the input
using a BERT tokenizer to extract the input_ids, token_type_ids and the attention_mask (to prevent
training on the padded tokens). Then we feed these inputs to our pre-trained BERT model to extract
the embeddings. Then, to do classification, we stack a classifying layer on top of our BERT layers.
This layer has 3 output neurons (number of target classes) and we use a Cross Entropy Loss for
training.

  The model needs a GPU to speed up execution time. We used a batch size of 12 observations for training
and a learning rate of 5 e-5. We run our training for 5 epochs and we keep the model state that gives
the best accuracy on the DEV set.


## Accuracy results
  After running tester.py, we get an accuracy of 85.32% on the DEV dataset, which is much better than
the other methods that we tried.
  
  Execution takes around 500 seconds with a P100 Nvidia GPU and around 1650 seconds with a K80 Tesla GPU
