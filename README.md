# pytorch-retain
This is PyTorch re-implementation of the following paper:

***Choi, Edward, Mohammad Taha Bahadori, Jimeng Sun, Joshua Kulas, Andy Schuetz, and Walter Stewart. 2016. “RETAIN: An Interpretable Predictive Model for Healthcare Using Reverse Time Attention Mechanism.” In Advances in Neural Information Processing Systems 29, edited by D. D. Lee, M. Sugiyama, U. V. Luxburg, I. Guyon, and R. Garnett, 3504–12. Curran Associates, Inc.***

# Contents
retain.py - Contains the implimentation of RETAIN from the paper RETAIN: An Interpretable Predictive Model for Healthcare Using Reverse Time Attention Mechanism
Not originally Create By Me. However, it is modified by me with the addition of comments

transfrom_retain.py - Contains another implimentation of RETAIN, with the RNNs being
replaced with Transformers. Copied and Modified version of retain.py, created by me

transform_retain_optimization - Finds the best possible hyperparameters for the
Transformer RETAIN model. Copied and Modified version of transform_retain.py created by me.

Data/group_codes.py - Groups the medical codes from CPTEVENTS.csv, AppendixASingleDX.txt, and AppendixCMultiDX.txt. Created by me

Data/mimic.py - Produces a dataset that use patient's visits as features and whether the patient died or not as labels. Not originally created be me

Data/heartFailure.py - Produces a dataset that uses patient's visits as features and whether the patient has heart failure or not as labels. Heart Failure Dataset used as the main dataset. Created by me. 

histogram.py - Produces an histogram that checks the frequency of the dataset in Data folder. It also prints out additional information about the dataset

train_result.txt - Contains the results for the best epoch of the original RETAIN model after running it for 50 epcohs

train_result20.txt - Contains the results for the best epoch of the Transformer RETAIN model after running it for 50 epochs. This model was run with the goal of lowering validation loss. 

hfDataInfo.txt - Contains additional information on the heart failure dataset

Data/Codes_With_Group_HF_Seqs_Statistics.txt - Statistics on number of codes in each visit in heart failure sequences with code grouping

Data/Codes_With_Group_NonHF_Seqs_Statistics.txt - Statistics on number of codes in each visit in non heart failure sequecnes without code grouping 

Data/Codes_With_Goup_All_Seqs_Statistics.txt - Statistics on number of codes in each visit in every sequence in the dataset with code grouping

Data/Codes_Without_Group_HF_Seqs_Statisitcs.txt -  Statistics on number of codes in each visit in heart failure sequences without code grouping

Data/Codes_Without_Group_NonHF_Seqs_Statistics.txt - Statistics on number of codes in each visit in non heart failure sequecnes with code grouping 

Data/Codes_Without_Group_All_Seqs_Statistics.txt - Statistics on number of codes in each visit in every sequence in the dataset without code grouping

Data/Additional_Codes_Statistics.txt - Additional statistics on numbber of codes in each visit in every sequence with and without code grouping

# How to use repository 
# Step 1: Obtain Files
Obtain the following files from the MIMIC-III Dataset, ADMISSIONS.csv, DIAGONSES.csv, CPTEVENTS.csv, PATIENTS.csv, and PRESECRIPTIONS.csv and put them in Data Folder. These files will be used to creating the heart failure dataset. You need access to  MIMIC-III before accessing these files. They can be obtained from https://physionet.org/content/mimiciii/1.4/

You will also need CCS_services_procedures_v2022-1_052422.csv and AppendixASingleDX.txt for grouping codes. AppendixSingleDX.txt is obtained from https://www.hcup-us.ahrq.gov/toolssoftware/ccs/ccs.jsp and CCS_services_procedures_v2022-1_052422.csv is obtained from https://www.hcup-us.ahrq.gov/toolssoftware/ccs_svcsproc/ccssvcproc.jsp

# Step 2: Creating Datasets
If you want to create a dataset that determine patient mortality, you can run mimic.py with ADMISSIONS.csv, DIAGNOSES_ICD.csv, and PATIENTS.csv

If you want to create a dataset that determine if a patient will have have heart failure or not in their next visit, firstly, group diagnoses and procedure groups by running group_codes.py with AppendixASingleDX.txt and CCS_services_procedures_v2022-1_052422.csv to form diagnoses.groups and procedure.groups. Afterwards, run heartFailure.py with ADMISSIONS.csv, DIAGNOSES_ICD.csv, CPTEVENTS.csv, PRESCRIPTIONS.csv, procdure.groups, and diagnoses.groups.

Running either mimic.py or heartFailure.py will produce the following: 

train.seqs - Input sequences for training set

train.labels - Output labels for training set

valid.seqs - Input sequences for validation set

valid.labels - Output labels for validation set

test.seqs - Input sequences for test set

test.labels - Output labels for test set

Additional files that are produced by heartFailure.py will be discussed in Step 3

# Step 3: Understanding the Heart Failure Dataset
To have a better understanding of the heart failure dataset, you can run histogram.py which takes in the dataset created and produces a histogram as well as other metrics. 
The histogram created is called Frequency of Visits.png, which determines the frequency of the number of visits for inputs of the dataset. 

hfDataInfo.txt contains additional metrics about the dataset, including the number of patients in total, the number and percentage of patients with heart failure, the number and percentage of pateints without heart failure, the mean number of visits for all inputs, the highest number of visits for all inputs.

Additionally, if you already run heartFailure.py, the program will produce additional information on the codes of the heart failure dataset. Each files contains statistics on the number of codes for each visit in each file respective dataset. This include statistics involving the number of diagnoses, medication, and procedure codes for each visit in addition to all types of codes in general. Each file also include a table to better organize the statistics of each dataset. The files produced includes:

Codes_With_Group_HF_Seqs_Statistics.txt - Statistics on number of codes in each visit in heart failure sequences with code grouping

Codes_With_Group_NonHF_Seqs_Statistics.txt - Statistics on number of codes in each visit in non heart failure sequecnes without code grouping 

Codes_With_Goup_All_Seqs_Statistics.txt - Statisitcs on number of codes in each visit in every sequence in the dataset with code grouping

Codes_Without_Group_HF_Seqs_Statisitcs.txt -  Statistics on number of codes in each visit in heart failure sequences without code grouping

Codes_Without_Group_NonHF_Seqs_Statistics.txt - Statistics on number of codes in each visit in non heart failure sequecnes with code grouping 

Codes_Without_Group_All_Seqs_Statistics.txt - Statisitcs on number of codes in each visit in every sequence in the dataset without code grouping

In addition, heartFailure.py produced Additional_Codes_Statistics.txt, which includes additional statistics about the number of codes in both datasets with and without code grouping. Additional_Codes_Statistics currently contain the number of codes being used in each dataset. 

# Step 4: Training RETAIN on dataset
Afterwards, you can use the dataset created to train either RETAIN or Transformer RETAIN. 
RETAIN - REverse Time AttentIoN model implimented form the paper RETAIN: An Interpretable Predictive Model for Healthcare using Reverse Time Attention Mechanism

Transformer RETAIN - RETAIN which replaces the RNNs with Transformer Encoders 

If you want to train dataset on RETAIN, run retain.py. You can run it with the following arguments: 

data_path - Path of the dataset. It is likely the datasets is in Data folder, so you should run Data/ as the argument. 

lr - learning rate

weight-decay - Weight Decay/Number of weights to drop

epochs - Number of epochs 

batch-size - Size of training batch

eval-batcb-size - Size of validation batch size

no-cuda - Doesn't use cuda gpu

no-plot - Doens't use plot

threads - Number of threads to use

save - Folder/Location to save model and checkpoints

Othwerise, if you want to train dataset on Transformer RETAIN. You can run it with the following arguments: 

data_path - Path of the dataset. It is likely the datasets is in Data folder, so you should run Data/ as the argument. 

lr - learning rate

weight-decay - Weight Decay/Number of weights to drop

epochs - Number of epochs 

batch-size - Size of training batch

eval-batcb-size - Size of validation batch size

no-cuda - Doesn't use cuda gpu

no-plot - Doens't use plot

threads - Number of threads to use 

save - Folder/Location to save model and checkpoints

embSize - Size of the embedding layer 

embDropout - Dropout of the embedding layer

contextDropout - Dropout of the context layer

heads - Number of heads for the attention layer in the Alpha Transformer

attentionDropout - Dropout for the attention layer in the Alpha Transformer

ffDropout - Dropout for the feed forward layer in the Alpha Transformer

normEps - The epsilon value for the norm layer in the Alpha Transformer

attEps - The epsilon value for the attention layer in the Alpha Transformer

ffSize - The size of the feed forward lyaer in the Alpha Transformer

transDropout - The dropout of the transformer layer in the Alpha Transformer

numLayers - The number of transformer layers in the Alpha Transformer

headsBeta - Number of heads for the attention layer in the Beta Transformer

attentionDropoutBeta - Dropout for the attention layer in the Beta Transformer

ffDropoutBeta - Dropout for the feed forward layer in the Beta Transformer

normEpsBeta - The epsilon value for the norm layer in the Beta Transformer

attEpsBeta - The epsilon value for the attention layer in the Beta Transformer

ffSizeBeta - The size of the feed forward lyaer in the Beta Transformer

transDropoutBeta - The dropout of the transformer layer in the Beta Transformer

numLayersBeta - The number of transformer layers in the Beta Transformer

metric - The metric used to determine the best epoch for model. The options are valid_loss, train_loss, and valid_auc

name - Name of the training result file

gpu - The GPUs being use for training for the model. This can be use if some GPUs are already being used by another program. 

Both retain.py and transformer_retain.py will save any metrics and hyperameters in train_result{name}.txt and will save model as best_model.pth and best_model_params.pth

# Step 5: Optimizing Transformer RETAIN
This step only applies Transformer RETAIN. To optimize Transformer RETAIN with the best possible parameters, you run transform_retain_optimization.py with ray and the following parameters:

data_path - Path of the dataset. It is likely the datasets is in Data folder, so you should run Data/ as the argument. 

epochs - Number of epochs for each trial

no-cuda - Doesn't use cuda gpu

threads - Number of threads to use 

save - Folder/Location to save metrics

trials - Number of trials to run tunner to optimize model

metric - The metric used to determine how should the tunner optimze the model. For example, if the metric is valid_loss, the tuner goal is to find the set of hyper parameters where the model have the lowest validation loss. The options are valid_loss for minimizing validation loss, train_loss for minimizing training loss, valid_auc for maximizing validation auc, and train_auc for maximizing training auc

name - Name of the resulting file 

gpu - The GPUs to be used in train_retain_optmization.py

Once all trials are completed, the program return transform_optimize{name}.txt, which contains a series of best possible hyperparamters of all possible metrics, including the main metric that the program is trying to find the best model for. 