# Import packages
import os
import csv
from tqdm import tqdm
from datetime import datetime
import numpy as np
import pandas as pd
from time import time, sleep

from datasets import Dataset, Value
from datasets.utils.logging import disable_progress_bar
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import (AutoModel, AutoModelForSequenceClassification, AutoConfig, AutoTokenizer, TrainingArguments, 
                          Trainer, EarlyStoppingCallback, set_seed)
from transformers.modeling_outputs import TokenClassifierOutput

from sentence_transformers import SentenceTransformer, InputExample, losses, models, util
from sentence_transformers.evaluation import SentenceEvaluator, SimilarityFunction

from debater_python_api.api.debater_api import DebaterApi

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, brier_score_loss, accuracy_score, f1_score, average_precision_score

import gc
import logging



###########################################################################################################################
### Similarity Scorer ###
###########################################################################################################################

# Define functions
def get_similarity_matrix(model, sentences_1, sentences_2):
    # Load model
    model = model.to('mps')
    # Compute embedding for both lists
    embeddings_1 = model.encode(sentences_1, convert_to_tensor = True)
    embeddings_2 = model.encode(sentences_2, convert_to_tensor = True)
    # Compute matrix of cosine-similarities
    similarity_matrix = util.cos_sim(embeddings_1, embeddings_2)
    # Return similarity matrix
    return similarity_matrix

###########################################################################################################################
### Quality Scorer ###
###########################################################################################################################

class quality_scorer(nn.Module):
    '''
    "The official code repository of BERT supports fine-tuning to classification tasks, which is done by applying a 
    linear layer on the [CLS] token of the last layer of BERT's model, which is then passed through a soft-max layer. 
    The weights of the preceding layers are initialized with BERT's pre-trained model, and the entire network is then 
    trained on the new data. To adapt the fine-tuning process to a regression task, the following were performed: 
    (1) Changing the label type to represent real values instead of integers; 
    (2) Replacing the softmax layer with a sigmoid function, to support a single output holding values in the range of [0,1]; 
    (3) Modifying the loss function to calculate the Mean Squared Error of the logits compared to the labels.
    '''
    def __init__(self, checkpoint):
        super(quality_scorer, self).__init__()
        self.model = AutoModel.from_pretrained(checkpoint, config = AutoConfig.from_pretrained(checkpoint,
                                                                                               output_attention = True,
                                                                                               output_hidden_state = True))
        self.output_layer = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids = None, attention_mask = None, token_type_ids = None, labels = None):
        outputs = self.model(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
        x = outputs['last_hidden_state'][:, 0, :] # Last hidden states of [CLS] token
        logits = self.output_layer(x)
        
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(self.sigmoid(logits.view(-1, 1)), labels.view(-1, 1).to(dtype = torch.float32))
        
        return TokenClassifierOutput(loss = loss, logits = logits, hidden_states = outputs.hidden_states, 
                                     attentions = outputs.attentions)
   
def save_log_history_quality_scorer(path, log_history_dict):
    '''
    Saves model performance for train/dev data during training
    '''
    # Set column names for the output file
    col_names = ['epoch', 'step', 'loss_train', 'loss_dev', 'rmse_dev', 'p_corr_dev', 's_corr_dev']
    epoch, step, loss_train, loss_dev, rmse_dev, p_corr_dev, s_corr_dev = [],[] ,[] ,[] ,[] ,[] ,[] 
    # Iterate over the dict including taining information and collect relevant data
    for i in range(0, len(log_history_dict)-1, 2):
        loss_train.append(log_history_dict[i]['loss'])
    # Iterate over the dict including the development information and collect relevant data
    for i in range(1, len(log_history_dict)-1, 2):
        epoch.append(log_history_dict[i]['epoch'])
        step.append(log_history_dict[i]['step'])
        loss_dev.append(log_history_dict[i]['eval_loss'])
        rmse_dev.append(log_history_dict[i]['eval_rmse'])
        p_corr_dev.append(log_history_dict[i]['eval_p_corr'])
        s_corr_dev.append(log_history_dict[i]['eval_s_corr'])
    # Create output folder if not existing
    if not os.path.exists(path):
        os.makedirs(path)
    # Save collected data
    with open(path + '/train_report.csv','w') as report:
        writer = csv.writer(report, delimiter = ',', lineterminator = '\n')
        writer.writerow(col_names)
        for row in zip(epoch, step, loss_train, loss_dev, rmse_dev, p_corr_dev, s_corr_dev):
            writer.writerow(row)
        report.close()

def train_quality_scorer(df_train, df_dev, target_name = 'WA', num_train_epochs = 5, batch_size = 32, learning_rate = 2e-5, 
                         weight_decay = 0,save_logging_steps = 200, seed = 573, pooling = False, model = None):
    # Set seed
    set_seed(seed)
    # Set timestamp
    timestamp = datetime.now().strftime('%Y-%b-%d_%H-%M-%S')
    # Load model and tokenizer path
    model_path = 'bert-base-uncased'
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = model_path)
    # Load model
    if model == None:
        if pooling == False:
            model = quality_scorer(model_path)
            suffix = 'np'
        elif pooling == True:
            model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path = model_path, 
                                                                       num_labels = 1)
            suffix = 'p'
    # Create tokenized datasets
    tokenized_ds_train = Dataset.from_pandas(df_train).map(lambda example: tokenizer(example['argument'], example['topic'], 
                                                                                     max_length = 128, truncation = True, 
                                                                                     padding = 'max_length'), 
                                                            batched = True)
    tokenized_ds_dev = Dataset.from_pandas(df_dev).map(lambda example: tokenizer(example['argument'], example['topic'], 
                                                                                 max_length = 128, truncation = True, 
                                                                                 padding = 'max_length'), 
                                                        batched = True)
    # Copy target variable and name it as lable
    tokenized_ds_train = tokenized_ds_train.add_column('label', tokenized_ds_train[target_name])
    tokenized_ds_dev = tokenized_ds_dev.add_column('label', tokenized_ds_dev[target_name])
    # Set dataset formats to pytorch
    tokenized_ds_train.set_format(type = 'pt', columns = ['input_ids', 'attention_mask', 'token_type_ids', 'label'])
    tokenized_ds_dev.set_format(type = 'pt', columns = ['input_ids', 'attention_mask', 'token_type_ids', 'label'])
    # Set checkpoint path, batch size as well as save and logging steps
    model_output_dir = f'models/quality_scorer/bert_ft_topic_{suffix}_{target_name.lower()}/{timestamp}'
    # Set training arguments
    training_args = TrainingArguments(output_dir = model_output_dir + '/checkpoints',
                                      overwrite_output_dir = True,
                                      num_train_epochs = num_train_epochs,
                                      evaluation_strategy = 'steps',
                                      save_strategy = 'steps',
                                      per_device_train_batch_size = batch_size,
                                      per_device_eval_batch_size= batch_size,
                                      save_steps = save_logging_steps, 
                                      save_total_limit = 2,
                                      logging_steps = save_logging_steps,
                                      learning_rate = learning_rate,
                                      weight_decay = weight_decay,
                                      load_best_model_at_end = True)
    # Set training metric
    def compute_metrics(eval_pred): 
        s = nn.Sigmoid()
        logits, labels = eval_pred
        predictions = s(torch.tensor(logits))
        rmse = mean_squared_error(labels, predictions, squared = False)
        p_corr = pearsonr(labels, predictions.flatten()).statistic
        s_corr = spearmanr(labels, predictions.flatten()).statistic
        return {'rmse':rmse, 'p_corr':p_corr, 's_corr':s_corr}
    # Create trainer
    trainer = Trainer(model = model,
                      args = training_args,
                      train_dataset = tokenized_ds_train,
                      eval_dataset = tokenized_ds_dev,
                      tokenizer = tokenizer,
                      compute_metrics = compute_metrics,
                      callbacks = [EarlyStoppingCallback(early_stopping_patience = 5)])

    # Train the model
    if trainer.args.device == torch.device('mps'):
        torch.mps.empty_cache()
        trainer.train()
        # Save best model
        torch.save(trainer.model, model_output_dir + '/best_model.pt')
        # Save logging history
        report = trainer.state.log_history
        save_log_history_quality_scorer(path = model_output_dir, log_history_dict = report)

def get_quality_scores(model, arguments, topic, stance = None, file_name = None, path = None, sleep_time = 0, n = 6):
    '''
    Quality scoring pipeline
    '''
    if str(type(model)) in ["<class 'argsum.tools.quality_scorer'>",
                            "<class 'transformers.models.bert.modeling_bert.BertForSequenceClassification'>"]:
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        disable_progress_bar()
        # Load tokenizer
        tokenizer_path = 'bert-base-uncased'
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        # Load model
        model = model.to('mps')
        # Concate arguments and topic
        if type(topic) == str:
            topic_ = [topic for i in range(len(arguments))]
        else:
            topic_ = topic
        arg_top_dict = {'argument':arguments, 'topic':topic_}
        # Tokenize
        tokenized_ds = Dataset.from_dict(arg_top_dict).map(lambda example: tokenizer(example['argument'], example['topic'],
                                                                                    max_length = 128, truncation = True,
                                                                                    padding = 'max_length'),
                                                            batched = True)
        # Set dataset formats to pytorch
        tokenized_ds.set_format(type = 'pt', columns = ['input_ids', 'attention_mask', 'token_type_ids'])
        # Create data loader 
        data_loader = DataLoader(tokenized_ds, batch_size = 32, shuffle = False)
        # Create sigmoid object
        sigmoid = nn.Sigmoid()
        # Set model to evaluation mode
        model.eval()
        # Create empty array for scores
        scores = []
        # Get scores 
        with torch.no_grad():
            for batch in tqdm(data_loader, position = 0, desc = 'Get Quality Scores', leave = False, colour = 'green'):
                tokenized_dict = {'input_ids':batch['input_ids'].to('mps'),
                                'attention_mask':batch['attention_mask'].to('mps'),
                                'token_type_ids':batch['token_type_ids'].to('mps')}
                logits = model(**tokenized_dict).logits
                scores += sigmoid(logits).to('cpu').detach().numpy().flatten().tolist()
                del tokenized_dict, logits
                gc.collect()
                torch.mps.empty_cache()
        del model
        quality_scorer
        gc.collect()
        torch.mps.empty_cache()   

    elif model == 'debater_api':
        api_key = 'd8dffadfbdd2630db78885818106fcf5L05'
        debater_api = DebaterApi(api_key)
        argument_quality_client = debater_api.get_argument_quality_client()
        if type(topic) == str:
            topic_ = [topic for i in range(len(arguments))]
        else:
            topic_ = topic
        arg_top_dicts = [{'sentence': arguments[i], 'topic': topic_[i]} for i in range(len(arguments))]
        scores = []
        runtimes = []
        for i in range(0, len(arg_top_dicts), n):
            start_time = time()
            sleep(sleep_time)
            scores += argument_quality_client.run(arg_top_dicts[i:i+n])
            runtimes += [np.round((time() - start_time) / n, 2)] * len(arg_top_dicts[i:i+n])
        
        if (file_name != None) and (path != None):
            # Create output folder if not existing
            if not os.path.exists(path):
                os.makedirs(path)
            scores_df = pd.DataFrame({'argument':arguments,
                                      'topic':topic_, 
                                      'stance':stance,
                                      'score':scores,
                                      'runtime':runtimes})
            scores_df.to_csv(path + '/' + file_name, index = False)
    # Return scores
    return scores

def eval_quality_scorer(model, df_test, output_dir = None):
    '''
    Evaluate quality scorer on test data.
    '''
    # Extract arguments, topics, labels
    args = df_test['argument'].to_list()
    tops = df_test['topic'].to_list()
    if model != 'debater_api':
        # Get scores
        start_time = time()
        scores = get_quality_scores(model, arguments = args, topic = tops)
        runtime = np.round(time() - start_time, 4)
    else:
        # Get scores         
        start_time = time()
        scores = get_quality_scores('debater_api', arguments = args, topic = tops, sleep_time = 0, n = len(args))
        #scores = df_test['ibm_pro_deb_qs'].to_list()
        runtime = np.round(time() - start_time, 4)
    # Compute evaluation metrics
    macep = df_test['MACE-P'].to_list()
    wa = df_test['WA'].to_list()
    macep_rmse = np.round(mean_squared_error(macep, scores, squared = False), 4)
    macep_p_corr = np.round(pearsonr(macep, scores).statistic, 4)
    macep_s_corr = np.round(spearmanr(macep, scores).statistic, 4)
    wa_rmse = np.round(mean_squared_error(wa, scores, squared = False), 4)
    wa_p_corr = np.round(pearsonr(wa, scores).statistic, 4)
    wa_s_corr = np.round(spearmanr(wa, scores).statistic, 4)
    # Save evaluation report
    col_names = ['macep_rmse_eval', 'macep_p_corr_eval', 'macep_s_corr_eval',
                 'wa_rmse_eval', 'wa_p_corr_eval', 'wa_s_corr_eval', 'runtime']
    # Create output folder if not existing
    if output_dir != None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_dir + '/eval_report.csv','w') as report:
            writer = csv.writer(report, delimiter = ',', lineterminator = '\n')
            writer.writerow(col_names)
            writer.writerow([macep_rmse, macep_p_corr, macep_s_corr, wa_rmse, wa_p_corr, wa_s_corr, runtime])
            report.close()

###########################################################################################################################
### Match Scorer ###
###########################################################################################################################

class match_scorer_np(nn.Module):
    '''
    "We fine tuned the BERTbase-uncased and BERT-large-uncased models (Devlin et al., 2019) to predict matches between 
    argument and key point pairs. We added a linear fully connected layer of size 1 followed by a sigmoid layer to the 
    special [CLS] token in the BERT model, and trained it for three epochs with a learning rate of 2e-5 and a binary 
    cross entropy loss.
    '''
    def __init__(self, checkpoint):
        super(match_scorer_np, self).__init__()
        self.model = AutoModel.from_pretrained(checkpoint, config = AutoConfig.from_pretrained(checkpoint,
                                                                                               output_attention = True,
                                                                                               output_hidden_state = True))
        self.config = self.model.config
        self.output_layer = nn.Linear(1024, 1)
    
    def forward(self, input_ids = None, attention_mask = None, labels = None):
        outputs = self.model(input_ids = input_ids, attention_mask = attention_mask)
        x = outputs['last_hidden_state'][:, 0, :] # Last hidden states of <s> token
        logits = self.output_layer(x)
            
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, 1), labels.view(-1, 1).to(dtype = torch.float32))
            
        return TokenClassifierOutput(loss = loss, logits = logits, hidden_states = outputs.hidden_states, 
                                     attentions = outputs.attentions)

class BCETrainer(Trainer):
    '''
    Trainer object for pooling case
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs = False):
        loss_fct = nn.BCEWithLogitsLoss()
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = loss_fct(logits.view(-1, 1), labels.view(-1, 1).to(dtype = torch.float32))
        return (loss, outputs) if return_outputs else loss
    
def save_log_history_match_scorer(path, log_history_dict):
    '''
    Saves model performance for train/dev data during training.
    '''
    # Set column names for the output file
    col_names = ['epoch', 'step', 'loss_train', 'loss_dev', 'brier_dev']
    epoch, step, loss_train, loss_dev, brier_dev = [], [], [], [], []
    # Iterate over the dict including taining information and collect relevant data
    for i in range(0, len(log_history_dict)-1, 2):
        loss_train.append(log_history_dict[i]['loss'])
    # Iterate over the dict including the development information and collect relevant data
    for i in range(1, len(log_history_dict)-1, 2):
        epoch.append(log_history_dict[i]['epoch'])
        step.append(log_history_dict[i]['step'])
        loss_dev.append(log_history_dict[i]['eval_loss'])
        brier_dev.append(log_history_dict[i]['eval_brier'])
    # Create output folder if not existing
    if not os.path.exists(path):
        os.makedirs(path)
    # Save collected data
    with open(path + '/train_report.csv','w') as report:
        writer = csv.writer(report, delimiter = ',', lineterminator = '\n')
        writer.writerow(col_names)
        for row in zip(epoch, step, loss_train, loss_dev, brier_dev):
            writer.writerow(row)
        report.close()

def train_cross_match_scorer(model_path, df_train, df_dev, num_train_epochs = 9, batch_size = 32, learning_rate = 5e-6, 
                             weight_decay = 0, save_logging_steps = 200, n_early_stop = 3, seed = 531, pooling = False, 
                             include_topic = False, augmented_data = False):
    '''
    Trains cross-encoder based match scorer.
    '''
    # Set seed
    set_seed(seed)
    # Set timestamp
    timestamp = datetime.now().strftime('%Y-%b-%d_%H-%M-%S')
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = model_path)
    # Load model
    if pooling == False:
        model = match_scorer_np(model_path) 
        pooling_suffix = '_np'
    elif pooling == True:
        model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path = model_path, 
                                                                    num_labels = 1)
        pooling_suffix = '_p'
    # Create tokenized datasets
    if augmented_data == False:
        augmented_data_suffix = ''
        if include_topic == False:
            include_topic_suffix = ''
            tokenized_ds_train = Dataset.from_pandas(df_train).map(lambda example: tokenizer(example['argument'],
                                                                                            example['key_point'], 
                                                                                            max_length = 128, truncation = True,
                                                                                            padding = 'max_length'), 
                                                                    batched = True)
            tokenized_ds_dev = Dataset.from_pandas(df_dev).map(lambda example: tokenizer(example['argument'], 
                                                                                        example['key_point'], 
                                                                                        max_length = 128, truncation = True,
                                                                                        padding = 'max_length'), 
                                                                batched = True)
        elif include_topic == True:
            include_topic_suffix = '_tp'
            tokenized_ds_train = Dataset.from_pandas(df_train).map(lambda example: tokenizer(example['argument'],
                                                                                            [example['topic'][i] + ' ' + example['key_point'][i] for i in range(len(example['key_point']))], 
                                                                                            max_length = 128, truncation = True,
                                                                                            padding = 'max_length'), 
                                                                    batched = True)
            tokenized_ds_dev = Dataset.from_pandas(df_dev).map(lambda example: tokenizer(example['argument'], 
                                                                                        [example['topic'][i] + ' ' + example['key_point'][i] for i in range(len(example['key_point']))], 
                                                                                        max_length = 128, truncation = True,
                                                                                        padding = 'max_length'), 
                                                                batched = True)            
    elif augmented_data == True:
        augmented_data_suffix = '_aug'
        if include_topic == False:
            include_topic_suffix = ''
            tokenized_ds_train = Dataset.from_pandas(df_train).map(lambda example: tokenizer(example['sentence_1'],
                                                                                            example['sentence_2'], 
                                                                                            max_length = 128, truncation = True,
                                                                                            padding = 'max_length'), 
                                                                    batched = True)
            tokenized_ds_dev = Dataset.from_pandas(df_dev).map(lambda example: tokenizer(example['sentence_1'], 
                                                                                        example['sentence_2'], 
                                                                                        max_length = 128, truncation = True, 
                                                                                        padding = 'max_length'), 
                                                                batched = True)
        elif include_topic == True:
            include_topic_suffix = '_tp'
            tokenized_ds_train = Dataset.from_pandas(df_train).map(lambda example: tokenizer(example['sentence_1'],
                                                                                            [example['topic'][i] + ' ' + example['sentence_2'][i] for i in range(len(example['sentence_2']))], 
                                                                                            max_length = 128, truncation = True,
                                                                                            padding = 'max_length'), 
                                                                    batched = True)
            tokenized_ds_dev = Dataset.from_pandas(df_dev).map(lambda example: tokenizer(example['sentence_1'], 
                                                                                        [example['topic'][i] + ' ' + example['sentence_2'][i] for i in range(len(example['sentence_2']))], 
                                                                                        max_length = 128, truncation = True, 
                                                                                        padding = 'max_length'), 
                                                                batched = True)            
    # Change datatype of label variable 
    new_features = tokenized_ds_train.features.copy()  
    new_features['label'] = Value('float')
    tokenized_ds_train = tokenized_ds_train.cast(new_features)
    new_features = tokenized_ds_dev.features.copy()  
    new_features['label'] = Value('float')
    tokenized_ds_dev = tokenized_ds_dev.cast(new_features)
    # Set dataset formats to pytorch
    tokenized_ds_train.set_format(type = 'pt',columns = ['input_ids', 'attention_mask', 'label'])
    tokenized_ds_dev.set_format(type = 'pt',columns = ['input_ids', 'attention_mask', 'label'])
    # Set checkpoint path, batch size as well as save and logging steps
    model_output_dir = f'models/match_scorer/cross_encoder/{model.config.model_type}{pooling_suffix}{include_topic_suffix}{augmented_data_suffix}/{timestamp}'
    # Set training arguments
    training_args = TrainingArguments(output_dir = model_output_dir + '/checkpoints',
                                      overwrite_output_dir = True,
                                      num_train_epochs = num_train_epochs,
                                      evaluation_strategy = 'steps',
                                      save_strategy = 'steps',
                                      per_device_train_batch_size = batch_size,
                                      per_device_eval_batch_size= batch_size,
                                      save_steps = save_logging_steps, 
                                      save_total_limit = 2,
                                      logging_steps = save_logging_steps,
                                      learning_rate = learning_rate,
                                      weight_decay = weight_decay,
                                      load_best_model_at_end = True)
    # Set training metric
    def compute_metrics(eval_pred): 
        s = nn.Sigmoid()
        logits, labels = eval_pred
        predictions = s(torch.tensor(logits))
        brier = brier_score_loss(labels, predictions)
        return {'brier':brier}
    # Create trainer
    if pooling == False:
        trainer = Trainer(model = model,
                        args = training_args,
                        train_dataset = tokenized_ds_train,
                        eval_dataset = tokenized_ds_dev,
                        tokenizer = tokenizer,
                        compute_metrics = compute_metrics,
                        callbacks = [EarlyStoppingCallback(early_stopping_patience = n_early_stop)])
    elif pooling == True:
        trainer = BCETrainer(model = model,
                                args = training_args,
                                train_dataset = tokenized_ds_train,
                                eval_dataset = tokenized_ds_dev,
                                tokenizer = tokenizer,
                                compute_metrics = compute_metrics,
                                callbacks = [EarlyStoppingCallback(early_stopping_patience = n_early_stop)])
    # Train the model
    if trainer.args.device == torch.device('mps'):
        # Collect training information 
        col_names = ['num_train_epochs', 'batch_size', 'learning_rate', 'weight_decay', 'seed', 'pooling', 'n_early_stop', 'augmented_data']
        train_values = [num_train_epochs, batch_size, learning_rate, weight_decay, seed, pooling, n_early_stop, augmented_data]
        # Create output folder if not existing
        if not os.path.exists(model_output_dir):
            os.makedirs(model_output_dir)
        # Save collected data
        with open(model_output_dir + '/train_information.csv','w') as report:
            writer = csv.writer(report, delimiter = ',', lineterminator = '\n')
            writer.writerow(col_names)
            writer.writerow(train_values)
        torch.mps.empty_cache()
        trainer.train()
        # Save best model
        torch.save(trainer.model, model_output_dir + '/best_model.pt')
        # Save logging history
        report = trainer.state.log_history
        save_log_history_match_scorer(path = model_output_dir, log_history_dict = report)

def get_match_scores(model, arguments, candidates, topic = None):
    '''
    Computes match scores.
    '''
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    disable_progress_bar()
    if str(type(model)) in ["<class 'argsum.tools.match_scorer_np'>", 
                            "<class 'transformers.models.roberta.modeling_roberta.RobertaForSequenceClassification'>",
                            "<class 'transformers.models.deberta_v2.modeling_deberta_v2.DebertaV2ForSequenceClassification'>"]:
        # Load tokenizer
        tokenizer_path = model.config._name_or_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        # Load model
        model = model.to('mps')
        # Create dict containing arguments and candidates
        if topic == None:
            arg_cand_dict = {'argument':arguments, 'candidate':candidates}
        elif topic != None:
            arg_cand_dict = {'argument':arguments, 'candidate':[topic + ' ' + cand for cand in candidates]}
        # Tokenize
        tokenized_ds = Dataset.from_dict(arg_cand_dict).map(lambda example: tokenizer(example['argument'], example['candidate'],
                                                                                      max_length = 128, truncation = True,
                                                                                      padding = 'max_length'),
                                                            batched = True)
        # Set dataset formats to pytorch
        tokenized_ds.set_format(type = 'pt', columns = ['input_ids', 'attention_mask'])
        # Create data loader 
        data_loader = DataLoader(tokenized_ds, batch_size = 32, shuffle = False)
        # Create sigmoid object
        sigmoid = nn.Sigmoid()
        # Set model to evaluation mode
        model.eval()
        # Create empty array for scores
        scores = np.array([])
        # Get scores 
        with torch.no_grad():
            for batch in tqdm(data_loader, position = 0, desc = 'Get Match Scores', leave = False, colour = 'green'):
                tokenized_dict = {'input_ids':batch['input_ids'].to('mps'),
                                'attention_mask':batch['attention_mask'].to('mps')}
                logits = model(**tokenized_dict).logits
                scores = np.append(scores, sigmoid(logits).to('cpu').detach().numpy().flatten())

    elif str(type(model)) == "<class 'sentence_transformers.SentenceTransformer.SentenceTransformer'>":
        if topic != None:
            if str(type(model.tokenizer)) in ["<class 'transformers.models.mpnet.tokenization_mpnet_fast.RobertaTokenizerFast'>",
                                              "<class 'transformers.models.roberta.tokenization_roberta_fast.RobertaTokenizerFast'>"]:
                sep = model.tokenizer.sep_token
                candidates = [topic + ' ' + sep + sep + ' ' + cand for cand in candidates]
            elif str(type(model.tokenizer)) == "<class 'transformers.models.deberta_v2.tokenization_deberta_v2_fast.DebertaV2TokenizerFast'>":
                sep = model.tokenizer.sep_token
                candidates = [topic + ' ' + sep + ' ' + cand for cand in candidates]
        argument_embeddings = model.encode(arguments)
        candidate_embeddings  = model.encode(candidates)
        sim_matrix = util.cos_sim(argument_embeddings, candidate_embeddings)
        scores = sim_matrix.to('cpu').diagonal().detach().numpy().flatten()
    del model
    gc.collect()
    torch.mps.empty_cache()
    # Return scores
    return scores

def get_match_score_matrix(model, arguments, candidates, comp_strategy = 'standard', topic = None):
    '''
    Computes match score matrix.
    '''
    if str(type(model)) in ["<class 'argsum.tools.match_scorer_np'>", 
                            "<class 'transformers.models.roberta.modeling_roberta.RobertaForSequenceClassification'>",
                            "<class 'transformers.models.deberta_v2.modeling_deberta_v2.DebertaV2ForSequenceClassification'>"]:
        # Get each combination of (argument, candidate) pairs
        if comp_strategy == 'standard':
            args = np.array([])
            cands = np.array([])
            for arg in arguments:
                for cand in candidates:
                    args = np.append(args, arg)
                    cands = np.append(cands, cand)
            # Perform match scoring to get match score matrix of shape (arguments, candidates)
            score_matrix = get_match_scores(model, args, cands, topic).reshape(len(arguments), len(candidates))
            if np.all(arguments == candidates):
                np.fill_diagonal(score_matrix, 1)
        else:
            idx_pairs_upper_diag = np.triu_indices(len(arguments), k = 1)
            idx_pairs_lower_diag = np.tril_indices(len(arguments), k = -1)
            args_upper_diag = [arguments[i] for i in idx_pairs_upper_diag[0]]
            cands_upper_diag = [candidates[i] for i in idx_pairs_upper_diag[1]]
            args_lower_diag = [arguments[i] for i in idx_pairs_lower_diag[0]]
            cands_lower_diag = [candidates[i] for i in idx_pairs_lower_diag[1]]
            score_matrix = np.ones((len(arguments), len(arguments)))
            if comp_strategy == 'quadratic_all':
                scores = get_match_scores(model, args_upper_diag + args_lower_diag, cands_upper_diag + cands_lower_diag, topic)
                score_matrix[idx_pairs_upper_diag] = scores[0:int(len(scores)/2)]
                score_matrix[idx_pairs_lower_diag] = scores[int(len(scores)/2):]
            elif comp_strategy == 'quadratic_upper':
                scores = get_match_scores(model, args_upper_diag, cands_upper_diag, topic)
                score_matrix[idx_pairs_upper_diag] = scores
                score_matrix[idx_pairs_lower_diag] = score_matrix.T[idx_pairs_lower_diag]   
            elif comp_strategy == 'quadratic_lower':
                scores = get_match_scores(model, args_lower_diag, cands_lower_diag, topic)
                score_matrix[idx_pairs_lower_diag] = scores
                score_matrix[idx_pairs_upper_diag] = score_matrix.T[idx_pairs_upper_diag]
            elif comp_strategy == 'quadratic_mean':
                scores = get_match_scores(model,args_upper_diag + args_lower_diag, cands_upper_diag + cands_lower_diag, topic)
                upper_scores = scores[0:int(len(scores)/2)]
                lower_scores = scores[int(len(scores)/2):]
                reverse_idx_tuple_pairs_lower_diag = [(idx_pairs_lower_diag[1][i], 
                                                    idx_pairs_lower_diag[0][i]) for i in range(len(idx_pairs_upper_diag[0]))]
                mean_scores = []
                for i in range(len(idx_pairs_upper_diag[0])):
                    upper_score = upper_scores[i]
                    upper_idx_n, upper_idx_m = idx_pairs_upper_diag[0][i], idx_pairs_upper_diag[1][i]
                    lower_score_idx = reverse_idx_tuple_pairs_lower_diag.index((upper_idx_n, upper_idx_m))
                    lower_score = lower_scores[lower_score_idx]
                    mean_scores.append((upper_score + lower_score) / 2)
                score_matrix[idx_pairs_upper_diag] = mean_scores
                score_matrix[idx_pairs_lower_diag] = score_matrix.T[idx_pairs_lower_diag]

    elif str(type(model)) == "<class 'sentence_transformers.SentenceTransformer.SentenceTransformer'>":      
        fill_diag = False
        if np.all(arguments == candidates):
            fill_diag = True
        if topic != None:
            if str(type(model.tokenizer)) in ["<class 'transformers.models.mpnet.tokenization_mpnet_fast.RobertaTokenizerFast'>",
                                              "<class 'transformers.models.roberta.tokenization_roberta_fast.RobertaTokenizerFast'>"]:
                sep = model.tokenizer.sep_token
                candidates = [topic + ' ' + sep + sep + ' ' + cand for cand in candidates]
            elif str(type(model.tokenizer)) == "<class 'transformers.models.deberta_v2.tokenization_deberta_v2_fast.DebertaV2TokenizerFast'>":
                sep = model.tokenizer.sep_token
                candidates = [topic + ' ' + sep + ' ' + cand for cand in candidates]
        argument_embeddings = model.encode(arguments)
        candidate_embeddings  = model.encode(candidates)
        score_matrix = util.cos_sim(argument_embeddings, candidate_embeddings).to('cpu').detach().numpy()
        if fill_diag == True:
            np.fill_diagonal(score_matrix, 1)
        score_matrix[score_matrix > 1] = 1
    # Return match score matrix
    return score_matrix

def get_best_match_plus_threshold_dict(arguments, candidates, match_scorer, match_scorer_t = 0.856, 
                                       include_not_matching = False, return_ids = False, topic = None):
    '''
    
    '''
    # Get match score matrix
    score_matrix = get_match_score_matrix(match_scorer, arguments, candidates, topic = topic)
    # Create array containing idx of best matching candidate for each argument
    bm = np.argmax(score_matrix, axis = -1)
    # Apply threshold policy to best matching candidates: Return dict containing matched candidates as keys and 
    # array of corresponding matched arguments as values
    bm_plus_t = {}
    if return_ids == False:
        for arg_idx, cand_idx in enumerate(bm):
            arg = arguments[arg_idx]
            cand = candidates[cand_idx]
            if score_matrix[arg_idx, cand_idx] > match_scorer_t:
                if cand not in bm_plus_t.keys():
                    bm_plus_t[cand] = np.array([arg])
                elif cand in bm_plus_t.keys():
                    bm_plus_t[cand] = np.append(bm_plus_t[cand], arg)
            else:
                if include_not_matching == True:
                    if '' not in bm_plus_t.keys():
                        bm_plus_t[''] = np.array([arg])
                    elif '' in bm_plus_t.keys():
                        bm_plus_t[''] = np.append(bm_plus_t[''], arg)
    elif return_ids == True:
        for arg_idx, cand_idx in enumerate(bm):
            if score_matrix[arg_idx, cand_idx] > match_scorer_t:
                if cand_idx not in bm_plus_t.keys():
                    bm_plus_t[cand_idx] = np.array([arg_idx])
                elif cand_idx in bm_plus_t.keys():
                    bm_plus_t[cand_idx] = np.append(bm_plus_t[cand_idx], arg_idx)
            else:
                if include_not_matching == True:
                    if -1 not in bm_plus_t.keys():
                        bm_plus_t[-1] = np.array([arg_idx])
                    elif -1 in bm_plus_t.keys():
                        bm_plus_t[-1] = np.append(bm_plus_t[-1], arg_idx)      
    # Return dict containing matched candidates with corresponding arguments
    return bm_plus_t

def get_best_match_plus_threshold(match_scorer, arguments, key_points, match_scorer_t = 0.856, topic = None):
    '''
    Match arguments to key points.
    '''
    bm = get_best_match_plus_threshold_dict(match_scorer = match_scorer, arguments = arguments, 
                                            candidates = key_points, topic = topic, match_scorer_t = match_scorer_t, 
                                            include_not_matching = True, return_ids = True)
    bm_revered = dict((v1, k) for k, v in bm.items() for v1 in v)
    return [bm_revered[id] for id in range(len(arguments))]

def eval_match_scorer(model, df_test, include_topic, output_dir = None):
    # Get unique topics and stances
    topics = df_test['topic'].unique()
    stances = df_test['stance'].unique()    
    # Create empty dict to store references and predictions 
    results = {'acc':[],
               'f1':[],
               'runtime':[],
               'topic':[],
               'stance':[]}
    # Iterate over topics and stances
    for topic in topics:
        # Set topic variable
        if include_topic == False:
            topic_to_include = None
        else:
            topic_to_include = topic
        for stance in stances:   
            # Get data for topic and stance
            mask = (df_test['topic'] == topic) & (df_test['stance'] == stance)
            df_kp_sta = df_test[mask]
            # Get arguments and keypoints
            args = df_kp_sta['argument'].to_list()
            kps = dict(zip(df_kp_sta['key_point'].unique(), [i for i in range(len(df_kp_sta['key_point'].unique()))]))
            # Get references
            refs = [kps[kp] for kp in df_kp_sta['key_point']]
            # Get matches
            start_time = time()
            preds = get_best_match_plus_threshold(match_scorer = model, arguments = args, key_points = list(kps.keys()), 
                                                  topic = topic_to_include, match_scorer_t = 0)
            runtime = time() - start_time        
            # Get accuracy and f1 score
            results['acc'].append(np.round(accuracy_score(refs, preds), 4))
            results['f1'].append(np.round(f1_score(refs, preds, average = 'weighted'), 4))
            results['runtime'].append(np.round(runtime, 4))
            results['topic'].append(topic)
            results['stance'].append(stance)
    results['acc'].append(np.round(np.mean(results['acc']), 4))
    results['f1'].append(np.round(np.mean(results['f1']), 4))
    results['runtime'].append(np.round(np.sum(results['runtime']), 4))
    results['topic'].append('All')
    results['stance'].append('All')
    results_df = pd.DataFrame(results)
    if output_dir != None:
        # Create output folder if not existing
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Save evalaution report
        results_df.to_csv(output_dir + 'eval_report.csv', index = False)
    return results_df
 
##### METHODS FROM THE KPA SHARED TASK ##########

logger = logging.getLogger(__name__)

def get_ap(df, label_column, top_percentile=0.5):
    top = int(len(df)*top_percentile)
    df = df.sort_values('score', ascending=False).head(top)
    # after selecting top percentile candidates, we set the score for the dummy kp to 1, to prevent it from increasing the precision.
    df.loc[df['key_point_id'] == "dummy_id", 'score'] = 0.99
    return average_precision_score(y_true=df[label_column], y_score=df["score"])

def calc_mean_average_precision(df, label_column):
    precisions = [get_ap(group, label_column) for _, group in df.groupby(["topic", "stance"])]
    return np.mean(precisions)

def evaluate_predictions(merged_df):
    mAP_strict = calc_mean_average_precision(merged_df, "label_strict")
    mAP_relaxed = calc_mean_average_precision(merged_df, "label_relaxed")
    return mAP_strict, mAP_relaxed

def load_kpm_data(gold_data_dir, subset):
    arguments_file = os.path.join(gold_data_dir, f"arguments_{subset}.csv")
    key_points_file = os.path.join(gold_data_dir, f"key_points_{subset}.csv")
    labels_file = os.path.join(gold_data_dir, f"labels_{subset}.csv")

    arguments_df = pd.read_csv(arguments_file)
    key_points_df = pd.read_csv(key_points_file)
    labels_file_df = pd.read_csv(labels_file)

    return arguments_df, key_points_df, labels_file_df

def get_predictions(preds, labels_df, arg_df):
    arg_df = arg_df[["arg_id", "topic", "stance"]]
    predictions_df = load_predictions(preds)
    #make sure each arg_id has a prediction
    predictions_df = pd.merge(arg_df, predictions_df, how="left", on="arg_id")

    #handle arguements with no matching key point
    predictions_df["key_point_id"] = predictions_df["key_point_id"].fillna("dummy_id")
    predictions_df["score"] = predictions_df["score"].fillna(0)

    #merge each argument with the gold labels
    merged_df = pd.merge(predictions_df, labels_df, how="left", on=["arg_id", "key_point_id"])

    merged_df.loc[merged_df['key_point_id'] == "dummy_id", 'label'] = 0
    merged_df["label_strict"] = merged_df["label"].fillna(0)
    merged_df["label_relaxed"] = merged_df["label"].fillna(1)
    return merged_df

def load_predictions(preds):
    arg =[]
    kp = []
    scores = []
    for arg_id, kps in preds.items():
        best_kp = max(kps.items(), key=lambda x: x[1])
        arg.append(arg_id)
        kp.append(best_kp[0])
        scores.append(best_kp[1])
    return pd.DataFrame({"arg_id" : arg, "key_point_id": kp, "score": scores})

####### METHODS FROM SMATCHTOPR #######

def match_argument_with_keypoints(result, kp_dict, arg_dict):    
        for arg, arg_embedding in arg_dict.items():
            result[arg] = {}
            for kp, kp_embedding in kp_dict.items():
                result[arg][kp] = util.pytorch_cos_sim(arg_embedding, kp_embedding).item()

        return result

def perform_preds(model, arg_df, kp_df, include_topic):
    argument_keypoints = {}
    for topic in arg_df.topic.unique():
        for stance in [-1, 1]:
            topic_keypoints_ids = kp_df[(kp_df.topic==topic) & (kp_df.stance==stance)]['key_point_id'].tolist()
            topic_keypoints = kp_df[(kp_df.topic==topic) & (kp_df.stance==stance)]['key_point'].tolist()
            
            if include_topic:
                topic_keypoints = [topic + ' </s> </s> ' + x for x in topic_keypoints]
                
            topic_keypoints_embeddings = model.encode(topic_keypoints, show_progress_bar=False)
            topic_kp_embed = dict(zip(topic_keypoints_ids, topic_keypoints_embeddings))

            topic_arguments_ids = arg_df[(arg_df.topic==topic)&(arg_df.stance==stance)]['arg_id'].tolist()
            topic_arguments = arg_df[(arg_df.topic==topic)&(arg_df.stance==stance)]['argument'].tolist()
            topic_arguments_embeddings = model.encode(topic_arguments, show_progress_bar=False)
            topic_arg_embed= dict(zip(topic_arguments_ids, topic_arguments_embeddings))

            argument_keypoints = match_argument_with_keypoints(argument_keypoints, topic_kp_embed, topic_arg_embed)

    return argument_keypoints

class KeyPointEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on a triplet: (sentence, positive_example, negative_example). Checks if distance(sentence,positive_example) < distance(sentence, negative_example).
    """
    def __init__(self, arg_df, kp_df, labels_df, include_topic, main_distance_function: SimilarityFunction = None, name: str = '', batch_size: int = 16, show_progress_bar: bool = False, write_csv: bool = True):
        """
        Constructs an evaluator based for the dataset
        :param dataloader:
            the data for the evaluation
        :param main_similarity:
            the similarity metric that will be used for the returned score
        """
        self.arg_df = arg_df
        self.kp_df = kp_df
        self.labels_df = labels_df
        self.name = name
        self.include_topic=include_topic
        self.main_distance_function = main_distance_function

        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

        self.csv_file: str = "triplet_evaluation"+("_"+name if name else '')+"_results.csv"
        self.csv_headers = ["epoch", "steps", "mAP_relaxed", "mAP_strict"]
        self.write_csv = write_csv


    @classmethod
    def from_eval_data_path(cls, eval_data_path, subset_name, include_topic, **kwargs):
        arg_df, kp_df, labels_df = load_kpm_data(eval_data_path, subset=subset_name)
        
        return cls(arg_df, kp_df, labels_df, include_topic, **kwargs)

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("TripletEvaluator: Evaluating the model on "+self.name+" dataset"+out_txt)

        
        #Perform prediction on the validation/test dataframes
        preds = perform_preds(model, self.arg_df, self.kp_df, self.include_topic)

        merged_df = get_predictions(preds, self.labels_df, self.arg_df)
        
        #Perform evaluation
        mAP_strict, mAP_relaxed = evaluate_predictions(merged_df)
        
        print(f"mAP strict= {mAP_strict} ; mAP relaxed = {mAP_relaxed}")
        
        logger.info("mAP strict:   \t{:.2f}".format(mAP_strict*100))
        logger.info("mAP relaxed:   \t{:.2f}".format(mAP_relaxed*100))
        
        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps, mAP_relaxed, mAP_strict])

            else:
                with open(csv_path, mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, steps, mAP_relaxed, mAP_strict])


        return (mAP_strict + mAP_relaxed)/2

def train_bi_match_scorer_smatchtopr(df_train, include_topic = True, epochs = 10):
    # Set timestamp
    timestamp = datetime.now().strftime('%Y-%b-%d_%H-%M-%S')
    # Set embedding model path 
    model_path = 'roberta-large'
    # Set embedding model
    word_embedding_model = models.Transformer(model_path)
    # Set max sequence length
    word_embedding_model.max_seq_length = 70
    # Add special token ############################################################ proove effect
    word_embedding_model.tokenizer.add_tokens(['<SEP>'], special_tokens = True)
    word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))
    # Set mean pooling to get one fixed sized sentence vector ###################### proove effect
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens = True,
                                   pooling_mode_cls_token = False,
                                   pooling_mode_max_tokens = False)
    # Combine both to sentence transformer model
    model = SentenceTransformer(modules = [word_embedding_model, pooling_model], device = 'mps').to('mps')
    # Create train examples
    if include_topic == True:
        train_examples = [InputExample(texts = [df_train.iloc[i]['argument'], 
                                                df_train.iloc[i]['topic'] + ' </s> </s> ' + df_train.iloc[i]['key_point']], 
                                                label = df_train.iloc[i]['label']) for i in range(len(df_train))]
    else:
         train_examples = [InputExample(texts = [df_train.iloc[i]['argument'], df_train.iloc[i]['key_point']], 
                                                label = df_train.iloc[i]['label']) for i in range(len(df_train))]       
    # Create data loader
    train_dataloader = DataLoader(train_examples, shuffle = False, batch_size = 32)
    # Set train loss
    train_loss = losses.ContrastiveLoss(model)
    # Set warmup steps
    warmup_steps = int(len(train_dataloader) * 10 * 0.1)
    
    evaluator = KeyPointEvaluator.from_eval_data_path('data/KPA_2021_shared_task/kpm_data', 'dev', name = 'dev', show_progress_bar=False, include_topic = include_topic)

    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator = evaluator,
              epochs = epochs,
              warmup_steps = warmup_steps,
              output_path = f'models/match_scorer/bi_encoder/roberta_tp/{timestamp}',
              checkpoint_path = f'models/match_scorer/bi_encoder/roberta_tp/{timestamp}/checkpoints',
              evaluation_steps = 200,
              checkpoint_save_steps = 200, 
              save_best_model = True)

def train_bi_match_scorer_barh(df_train, model_path = None, include_topic = True, epochs = 10, augmented = False, loss = 'cos', lr = 5e-6):
    # Set timestamp
    timestamp = datetime.now().strftime('%Y-%b-%d_%H-%M-%S')
    # Set embedding model path 
    if model_path == None:
        model_path = 'all-mpnet-base-v2'
    # Set embedding model
    model = SentenceTransformer(model_path, device = 'mps').to('mps')
    # Set max sequence length
    model.max_seq_length = 256
    # Create train examples
    if augmented == True:
        train_examples = [InputExample(texts = [df_train.iloc[i]['sentence_1'], df_train.iloc[i]['sentence_2']], 
                                                        label = int(df_train.iloc[i]['label'])) for i in range(len(df_train))]      
    else:
        if loss == 'cos':
            train_examples = [InputExample(texts = [df_train.iloc[i]['argument'], df_train.iloc[i]['key_point']], 
                                                        label = float(df_train.iloc[i]['label'])) for i in range(len(df_train))]   
        else:
             train_examples = [InputExample(texts = [df_train.iloc[i]['argument'], df_train.iloc[i]['key_point']], 
                                                        label = df_train.iloc[i]['label']) for i in range(len(df_train))]                  
    # Create data loader
    train_dataloader = DataLoader(train_examples, shuffle = True, batch_size = 32)
    # Set train loss
    if loss == 'cos':
        train_loss = losses.CosineSimilarityLoss(model)
    elif loss == 'con':
        train_loss = losses.ContrastiveLoss(model)
    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              optimizer_params = {'lr':lr},
              epochs = epochs,
              output_path = f'models/match_scorer/bi_encoder/all-mpnet-base-v2/{timestamp}',
              checkpoint_path = f'models/match_scorer/bi_encoder/all-mpnet-base-v2/{timestamp}/checkpoints',
              evaluation_steps = 200,
              checkpoint_save_steps = 200, 
              save_best_model = True)