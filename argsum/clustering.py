# Import packages
import numpy as np
import pandas as pd
import os
import requests
import csv
import json
import re
from tqdm.notebook import tqdm
import gc

from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from sklearn.metrics.pairwise import cosine_similarity

import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import (AutoTokenizer, 
                          AutoModelForCausalLM, 
                          pipeline, 
                          AutoModelForSeq2SeqLM, 
                          DataCollatorForSeq2Seq,
                          set_seed, 
                          Seq2SeqTrainingArguments, 
                          Seq2SeqTrainer, 
                          DataCollatorForSeq2Seq)

from sklearn.cluster import AgglomerativeClustering
from openai import OpenAI

from argsum._utils_ import set_summetix_api_login, get_summetix_api_login, get_summetix_api_key_id
from argsum.tools import get_quality_scores

###########################################################################################################################
### Clustering approaches ###
###########################################################################################################################

# Summetix
def get_summetix_clusters(arguments, threshold = 0.675, min_cluster_size = 4, topic = None):
    '''
    Performs soft argument clustering utilizing the API of summetix.
    arguments:          Arguments to cluster
    argument_ids:       Optional IDs of arguments
    threshold:          Similarity threshold that is necessary between two arguments to cluster them into the same cluster
    min_cluster_size:   All clusters must hold at least that many arguments
    '''
    # Update Summetix API key
    #set_summetix_api_login('...', '...')
    get_summetix_api_login()
    # Set API key and id
    api_key, api_id = get_summetix_api_key_id()
    # Set API url 
    api_url = 'https://api.summetix.com/en/cluster_arguments'
    # Assign argument ids
    arguments_with_ids = [[str(id), argument] for id, argument in enumerate(arguments)]
    # Set payload
    payload = {'arguments':arguments_with_ids,
               'threshold':threshold,
               'min_cluster_size':min_cluster_size,
               'compute_labels': False,
               'apiKeyId': api_id,
               'apiKey': api_key}         
    # Get response
    response = requests.post(api_url, data = json.dumps(payload), headers = {'Content-Type': 'application/json'}).json()
    # Get clusters with arguments
    clusters = response['clusters_argument_id']
    cluster_ids = []
    clustered_argument_ids = []
    for id, clu in enumerate(clusters):
        cluster_ids += [id] * clu['size']
        clustered_argument_ids += [int(id) for id in clu['sentence_ids']]
    # Get unclustered argument ids
    unclustered_argument_ids = [int(arg_with_id[0]) for arg_with_id in arguments_with_ids if int(arg_with_id[0]) not in clustered_argument_ids]
    # Get sorted (argument_id, cluster_id) pairs of clustered and unclustered arguments
    sorted_argument_ids_cluster_ids = sorted(list(zip(clustered_argument_ids, cluster_ids)) + list(zip(unclustered_argument_ids, [-1 for i in range(len(unclustered_argument_ids))])))
    # Return sorted cluster_ids
    return [id[1] for id in sorted_argument_ids_cluster_ids]

# USKPM
def get_bertopic_clusters(arguments, n_neighbors = 3, n_components = 2, min_dist = 0.0, min_cluster_size = 4, seed = 753):
    '''
    Performs soft argument clustering utilizing BERTopic.
    n_neighbors:        Number of neighboring samples used when making the manifold approximation
                            Large -> More global view of the embedding structure (larger clusters)
                            Small -> More local view (smaller clusters)
    n_components:       Dimensionality of embeddings after reducing them
    min_dist:           Minimum distance that the points are allowed to have in the low dimensional representation
                        (Controls how tightly UMAP is allowed to pack points together)
                            Larger -> Will prevent UMAP from packing points together
                            Small -> Clumpier embeddings
    min_cluster_size:   Minimum size of a cluster (and thereby the number of clusters that will be generated) 
                        (in the SKPM code with a value of 6)
                            Large -> Fewer clusters of larger size 
                            Small -> More clusters of small size
    '''
    # Set sentence transformers model
    sentence_model = SentenceTransformer('all-mpnet-base-v2')
    # Set dimensionality reduction model
    umap_model = UMAP(n_neighbors = n_neighbors, 
                      n_components = n_components, 
                      min_dist = min_dist, 
                      metric = 'cosine', 
                      random_state = seed)
    # Set clustering model
    np.random.seed(seed)
    hdbscan_model = HDBSCAN(min_cluster_size = min_cluster_size, 
                            metric = 'euclidean', 
                            cluster_selection_method = 'eom', 
                            prediction_data = True)
    # Set vectorizer model for topic tokenization 
    vectorizer_model = CountVectorizer(ngram_range = (1,2), stop_words = 'english')
    # Create bertopic object
    topic_model = BERTopic(embedding_model = sentence_model, 
                           umap_model = umap_model, 
                           hdbscan_model = hdbscan_model,
                           vectorizer_model = vectorizer_model, 
                           calculate_probabilities = False, 
                           low_memory = False,
                           n_gram_range = (1,1), # Obmitted before recent code updates
                           nr_topics = 'auto')
    # Get topics and their probabilities
    cluster_ids, probs = topic_model.fit_transform(arguments)
    # Return cluser_ids
    return cluster_ids

# USKPM
def get_uskpm_clusters(arguments, n_neighbors = 3, n_components = 2, min_dist = 0.0, min_cluster_size = 4, seed = 473):
    '''
    Performs soft argument clustering utilizing BERTopic.
    n_neighbors:        Number of neighboring samples used when making the manifold approximation
                            Large -> More global view of the embedding structure (larger clusters)
                            Small -> More local view (smaller clusters)
    n_components:       Dimensionality of embeddings after reducing them
    min_dist:           Minimum distance that the points are allowed to have in the low dimensional representation
                        (Controls how tightly UMAP is allowed to pack points together)
                            Larger -> Will prevent UMAP from packing points together
                            Small -> Clumpier embeddings
    min_cluster_size:   Minimum size of a cluster (and thereby the number of clusters that will be generated) 
                        (in the SKPM code with a value of 6)
                            Large -> Fewer clusters of larger size 
                            Small -> More clusters of small size
    '''
    # Set sentence transformers model
    sentence_model = SentenceTransformer('all-mpnet-base-v2')
    # Set dimensionality reduction model
    umap_model = UMAP(n_neighbors = n_neighbors, 
                      n_components = n_components, 
                      min_dist = min_dist, 
                      metric = 'cosine', 
                      random_state = seed)
    # Set clustering model
    np.random.seed(seed)
    hdbscan_model = HDBSCAN(min_cluster_size = min_cluster_size, 
                            metric = 'euclidean', 
                            cluster_selection_method = 'eom', 
                            prediction_data = True)
    # Set vectorizer model for topic tokenization 
    vectorizer_model = CountVectorizer(ngram_range = (1,2), stop_words = 'english')
    # Create bertopic object
    topic_model = BERTopic(embedding_model = sentence_model, 
                           umap_model = umap_model, 
                           hdbscan_model = hdbscan_model,
                           vectorizer_model = vectorizer_model, 
                           calculate_probabilities = False, 
                           low_memory = False,
                           n_gram_range = (1,1), # Obmitted before recent code updates
                           nr_topics = 'auto')
    # Get topics and their probabilities
    cluster_ids, probs = topic_model.fit_transform(arguments)
    # Return cluser_ids
    return cluster_ids

def get_iterative_clusters(arguments, cluster_ids, threshold, sentence_model = None):
    '''
    Performs iterative clustering by assigning each unclustered argument to its most similar cluster.
    threshold:          Similarity threshold for argument and cluster embeddings which must be exceeded for assigning the 
                        argument the cluster
    '''
    # Assign argument ids
    arguments_with_ids = [[id, argument] for id, argument in enumerate(arguments)]
    if sentence_model == None:
        # Create sentence transformers object
        sentence_model = SentenceTransformer('all-mpnet-base-v2', device = 'mps')
    # Initialize current clusters
    clustered_arguments = [arguments_with_ids[i] for i in range(len(arguments_with_ids)) if cluster_ids[i] != -1]
    clustered_arguments_cluster_ids = [cluster_ids[i] for i in range(len(cluster_ids)) if cluster_ids[i] != -1]
    # Create dict with unique cluster ids as keys and empty lists as values
    current_clusters = dict(zip(list(set(clustered_arguments_cluster_ids)), [[] for i in range(len(list(set(clustered_arguments_cluster_ids))))]))
    if len(current_clusters) > 0:
        # Iterate over arguments and append these to corresponding cluster id list
        for i, arg in enumerate(clustered_arguments):
            current_clusters[clustered_arguments_cluster_ids[i]].append(clustered_arguments[i])
        # Get unclustered arguments
        unclustered_arguments = [arguments_with_ids[i] for i in range(len(arguments_with_ids)) if cluster_ids[i] == -1]
        # Iterate over unclustered arguments
        for arg in unclustered_arguments:
            # Get sentence embedding of argument
            arg_embedding = sentence_model.encode(arg[1])
            # Iterate over clusters
            sims = []
            for clu in list(current_clusters.keys()):
                clu_embedding = np.mean(sentence_model.encode([arg[1] for arg in current_clusters[clu]]), axis = 0)
                sims.append(cosine_similarity(arg_embedding.reshape(1, -1), clu_embedding.reshape(1,-1)))
            # Get cluster with highest similarity
            best_sim = np.max(sims)
            best_cluster_id = list(current_clusters.keys())[np.argmax(sims)]
            # Assign argument to cluster with highest similarity if it is above threshold or new cluster
            if best_sim > threshold:
                current_clusters[best_cluster_id].append(arg)
            else:
                new_cluster_id = np.max(list(current_clusters.keys())) + 1
                current_clusters[new_cluster_id] = [arg]
        reversed_current_clusters = dict((v1[0], k) for k, v in current_clusters.items() for v1 in v)
        sorted_reversed_current_clusters = dict(sorted(reversed_current_clusters.items()))
        # Return cluster_ids
        try:
            del sentence_model, arg_embedding, clu_embedding
            gc.collect()
            torch.mps.empty_cache()
        except:
            pass
        return list(sorted_reversed_current_clusters.values())
    else:
        del sentence_model
        gc.collect()
        torch.mps.empty_cache()
        return cluster_ids

# MCArgSum
def get_match_clusters(match_score_matrix, threshold = 0.675, min_cluster_size = 4, method = 1):
    '''
    Performs soft argument clustering utilizing agglomerative clustering with custom match (or similarity) matrix.
    threshold:          Match (or similarity) threshold that is necessary between two arguments to cluster them into the same 
                        cluster
    min_cluster_size:   All clusters must hold at least that many arguments
    '''
    # Set cluster method and parameters
    clustering_params = {'n_clusters':None, 
                         'distance_threshold':1 - threshold,
                         'linkage':'average', 
                         'metric':'precomputed'}
    clustering_algorithm = AgglomerativeClustering(**clustering_params)
    # Fit clusters and get cluster ids
    if method == 1:
        cluster_ids = clustering_algorithm.fit(1 - match_score_matrix).labels_
    elif method == 2:
        cluster_ids = clustering_algorithm.fit(np.sqrt(1- match_score_matrix)).labels_
    elif method == 3:
        cluster_ids = clustering_algorithm.fit(-np.log(match_score_matrix)).labels_
    elif method == 4:
        cluster_ids = clustering_algorithm.fit((1/match_score_matrix)-1).labels_
    # Check cluster sizes
    unique_cluster_ids, cluster_sizes = np.unique(cluster_ids, return_counts = True)
    cluster_sizes_dict = dict(zip(unique_cluster_ids, cluster_sizes))
    cluster_ids_to_delete = []
    # Set labe of clusters with size below min_cluster_size to -1
    for id in unique_cluster_ids:
        if cluster_sizes_dict[id] < min_cluster_size:
            cluster_ids_to_delete.append(id)
    # Get new cluster ids
    cluster_ids = [id if id not in cluster_ids_to_delete else -1 for id in cluster_ids]
    # Change cluster id labels
    labels_dict = dict(zip(sorted(list(set(cluster_ids))), [i for i in range(-1, len(list(set(cluster_ids))) -1)]))
    # Return new labled cluster ids
    return [labels_dict[id] for id in cluster_ids]

###########################################################################################################################
### Cluster summarization approaches ###
###########################################################################################################################

# Summetix
def get_summetix_cluster_sums(arguments, cluster_ids, topic = None, stance = None):
    '''
    Get aspect detection based cluster summaries utilizing the API of summetix.
    '''
    # Update Summetix API key
    #set_summetix_api_login('...', '...')
    get_summetix_api_login()
    # Set API key and id
    api_key, api_id = get_summetix_api_key_id()
    # Set API urls
    api_url_get_aspects = 'https://api.summetix.com/en/get_aspects'
    api_url_get_aspect_topics = 'https://api.summetix.com/en/get_aspect_topics'
    # Set argument ids 
    ids_args = [[str(id), arg] for id, arg in enumerate(arguments)]
    # Order arguments ids according to cluster ids
    unique_cluster_ids = sorted(list(set(cluster_ids)))
    #print([ids_args[i][0] for i in range(len(ids_args))])# if cluster_ids[i] == 1])
    argument_ids_clusters = [[ids_args[i][0] for i in range(len(ids_args)) if cluster_ids[i] == cluster_id] for cluster_id in unique_cluster_ids]
    # Get aspects for each sentence of each cluster
    aspects_clusters = []
    for arg_ids_clu in argument_ids_clusters:
        # Set payload
        payload_get_aspects = {'arguments':dict([arg for arg in ids_args if arg[0] in arg_ids_clu]),
                   'apiKeyId': api_id,
                   'apiKey': api_key}
        # Get response
        response_get_aspects = requests.post(api_url_get_aspects, data = json.dumps(payload_get_aspects), headers = {'Content-Type': 'application/json'}).json()
        aspects_clusters.append(response_get_aspects['arguments'])
    # Get aspects dict with argument ids as keys and corresponding aspects as values
    aspects_sentence_ids = {}
    for asp_clu in aspects_clusters:
        for arg_id in list(asp_clu.keys()):
            aspects_sentence_ids[arg_id] = {'aspects':asp_clu[arg_id]['aspects']}
    # Set payload
    payload_get_aspect_topics = {'clusters':dict(zip(unique_cluster_ids, argument_ids_clusters)),
                                 'aspects':aspects_sentence_ids,
                                 'arguments':dict(ids_args),
                                 'label_stopwords':topic.split(),
                                 'apiKeyId': api_id,
                                 'apiKey': api_key}
    # Get response
    response_get_aspect_topics = requests.post(api_url_get_aspect_topics, data = json.dumps(payload_get_aspect_topics), headers = {'Content-Type': 'application/json'}).json()
    # Return summaries
    return dict(sorted(zip([int(id) for id in list(response_get_aspect_topics['clusterLabelMap'].keys())], [[sum] for sum in list(response_get_aspect_topics['clusterLabelMap'].values())])))

# USKPM
def preprocess_input_t5(df):
    stance_dict = {1:'Positive', -1:'Negative'}
    quality_scorer = torch.load('models/quality_scorer/bert_ft_topic_np_mace-p/2024-Feb-24_13-34-37/best_model.pt')
    summaries_unique = df['key_point'].unique()
    preprocessed_df = pd.DataFrame(columns = ['text', 'label'])
    for summary in summaries_unique:
        mask = (df['label'] == 1) & (df['key_point'] == summary)
        df_summary = df[mask]
        topic_summary = df_summary['topic'].unique()[0]
        stance_summary = df_summary['stance'].unique()[0]
        args_summary = df_summary['argument'].to_list()
        aqs = get_quality_scores(quality_scorer, args_summary, topic_summary)
        sorted_args_summary = [arg for aq, arg in sorted(zip(aqs, args_summary), reverse = True)]
        text = 'summarize: ' + stance_dict[stance_summary] + ' ' + topic_summary + ' ' + ' '.join(sorted_args_summary)
        #text = 'summarize: ' + ' '.join(sorted_args_summary)
        preprocessed_df = pd.concat([preprocessed_df, pd.DataFrame({'text':text, 'label':[summary]})], ignore_index = True)
    return preprocessed_df

def train_t5(df_train, df_dev, num_train_epochs = 15, batch_size = 16, learning_rate = 4e-4, weight_decay = 0, save_logging_steps = 20, 
             max_length_text = 512, max_length_label = 128, seed = 748, load_best_model_at_end = True):
    # Load model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base').to('mps')
    tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')
    # Create tokenized datasets
    tokenized_text_train = tokenizer(df_train['text'].to_list(), max_length = max_length_text, truncation = True, 
                                     padding = 'max_length')
    tokenized_text_dev = tokenizer(df_dev['text'].to_list(), max_length = max_length_text, truncation = True, 
                                   padding = 'max_length')
    tokenized_label_train = tokenizer(df_train['label'].to_list(), max_length = max_length_label, truncation = True, 
                                      padding = 'max_length')['input_ids']
    tokenized_label_dev = tokenizer(df_dev['label'].to_list(), max_length = max_length_label, truncation = True, 
                                    padding = 'max_length')['input_ids']
    tokenized_ds_train = Dataset.from_dict({'input_ids':tokenized_text_train['input_ids'], 'attention_mask':tokenized_text_train['attention_mask'], 
                                            'labels': tokenized_label_train})
    tokenized_ds_dev = Dataset.from_dict({'input_ids':tokenized_text_dev['input_ids'], 'attention_mask':tokenized_text_dev['attention_mask'], 
                                          'labels': tokenized_label_dev})
    tokenized_ds_train.set_format(type = 'pt', columns = ['input_ids', 'attention_mask', 'labels'])
    tokenized_ds_dev.set_format(type = 'pt', columns = ['input_ids', 'attention_mask', 'labels'])
    # Set seed
    set_seed(seed)
    model_output_dir = 'models/summary_generator/flan_t5_cluster'
    data_collator = DataCollatorForSeq2Seq(tokenizer, model = model)
    # Set training arguments
    training_args = Seq2SeqTrainingArguments(output_dir = model_output_dir + '/checkpoints',
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
                                             predict_with_generate = True,
                                             load_best_model_at_end = load_best_model_at_end)
    # Create trainer
    trainer = Seq2SeqTrainer(model = model,
                             args = training_args,
                             train_dataset = tokenized_ds_train,
                             eval_dataset = tokenized_ds_dev,
                             data_collator = data_collator,
                             tokenizer = tokenizer)#,
                             #callbacks = [EarlyStoppingCallback(early_stopping_patience = 3)])
    # Train the model
    if trainer.args.device == torch.device('mps'):
        torch.mps.empty_cache()
        trainer.train()
    torch.save(trainer.model, model_output_dir + '/best_model.pt')
    # Save train report 
    log_history_dict = trainer.state.log_history
    # Set column names for the output file
    col_names = ['epoch', 'step', 'loss_train', 'loss_dev']
    epoch, step, loss_train, loss_dev = [], [] ,[] ,[]
    # Iterate over the dict including taining information and collect relevant data
    for i in range(0, len(log_history_dict)-1, 2):
        loss_train.append(log_history_dict[i]['loss'])
    # Iterate over the dict including the development information and collect relevant data
    for i in range(1, len(log_history_dict)-1, 2):
        epoch.append(log_history_dict[i]['epoch'])
        step.append(log_history_dict[i]['step'])
        loss_dev.append(log_history_dict[i]['eval_loss'])
    # Create output folder if not existing
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)
    # Save collected data
    with open(model_output_dir + '/train_report.csv','w') as report:
        writer = csv.writer(report, delimiter = ',', lineterminator = '\n')
        writer.writerow(col_names)
        for row in zip(epoch, step, loss_train, loss_dev):
            writer.writerow(row)
        report.close()

def get_t5_cluster_sums(arguments, cluster_ids, topic, stance = None, max_length_text = 512, max_length_label = 128, n = 1, num_beams = 6, temperature = None, do_sample = False, p = None):
    quality_scorer = torch.load('models/quality_scorer/bert_ft_topic_np_mace-p/2024-May-29_15-58-03/best_model.pt')
    quality_scorer.to('mps')
    model = torch.load('models/summary_generator/flan_t5_cluster/best_model.pt')
    model.to('mps')

    tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')
    stance_dict = {1:'Positive', -1:'Negative'}
    unique_cluster_ids = sorted(list(set(cluster_ids)))
    texts = []
    for cluster_id in unique_cluster_ids:
        args = [arguments[i] for i in range(len(arguments)) if cluster_ids[i] == cluster_id]
        aqs = get_quality_scores(quality_scorer, args, topic)
        sorted_args_cluster = [arg for aq, arg in sorted(zip(aqs, args), reverse = True)]
        text = 'summarize: ' + stance_dict[stance] + ' ' + topic + ' ' + ' '.join(sorted_args_cluster)
        #text = 'summarize: ' + ' '.join(sorted_args_cluster)
        texts.append(text)
    tokenized_texts = tokenizer(texts, max_length = max_length_text, truncation = True, padding = 'max_length')
    tokenized_ds = Dataset.from_dict({'input_ids':tokenized_texts['input_ids'],
                                      'attention_mask':tokenized_texts['attention_mask']})
    
    tokenized_ds.set_format(type = 'pt', columns = ['input_ids', 'attention_mask'])

    del aqs
    gc.collect()
    torch.mps.empty_cache()

    # Create data loader 
    data_loader = DataLoader(tokenized_ds, batch_size = 16, shuffle = False)
    # Set model to evaluation mode
    model.eval()
    # Get summaries
    sums = []
    with torch.no_grad():
        for batch in tqdm(data_loader, position = 0, desc = 'Get Flan-T5 Cluster Sums', leave = False, colour = 'green'):
            tokenized_dict = {'input_ids':batch['input_ids'].to('mps'),
                              'attention_mask':batch['attention_mask'].to('mps')}
            sum = model.generate(**tokenized_dict, num_beams = num_beams, max_length = max_length_label, 
                                 num_return_sequences = n, temperature = temperature, do_sample = do_sample, top_p = p)
            sums += tokenizer.batch_decode(sum, skip_special_tokens = True)
            del tokenized_dict
            gc.collect()
            torch.mps.empty_cache()
    sums_splitted = []
    start_idx = 0
    for i in range(0, len(sums), n):
        sums_splitted.append(sums[start_idx:start_idx+n])
        start_idx += n
    
    del model, quality_scorer, data_loader, tokenized_ds
    gc.collect()
    torch.mps.empty_cache()

    return dict(zip(unique_cluster_ids, sums_splitted))

# MCArgSum
def get_llm_messages(clusters, topic, stance, optimization = 'local', sum_token_length = 10, sum_min_num = 1, sum_max_num = 2, 
                     few_shot = True, exclude_topic = False, generate_less = False):
    '''
    
    '''
    # Create dict with numerical stance as key and textual stance as value
    stance_dict = {-1:'opposing', 1:'supporting'}
    # Create dict to map numerical number to written sring
    number_dict = {0:'zero', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five', 6:'six', 7:'seven', 8:'eight', 9:'nine', 10:'ten',              
                   11:'eleven', 12:'twelve', 13:'thirteen', 14:'fourteen', 15:'fifteen', 16:'sixteen', 17:'seventeen', 
                   18:'eighteen', 19:'nineteen', 20:'twenty'}
    # Create key point token in singular or plural depending on the maximal allowed number of summaries
    if sum_max_num == 1:
        num_token = 'a single'
        kp_token = 'key point'
    elif sum_min_num < sum_max_num:
        num_token =  number_dict[sum_min_num] + ' to ' + number_dict[sum_max_num]
        kp_token = 'key points'
    elif sum_min_num == sum_max_num:
        num_token = number_dict[sum_max_num]
        kp_token = 'key points'       
    # Create system message
    system_message = '''
You are a professional debater and you can express yourself succinctly. If you are given a cluster of similar arguments on a 
certain topic and a certain stance, you find {num_token} appropriate salient single sentences, called {kp_token}, capturing 
the main statement that is shared between most of the clustered arguments and providing a textual and quantitative view of the data. 
A key point can be seen as a meta argument why one is for or against a certain topic. Since argument clusters are not perfect, they
may contain arguments that do not actually belong together. Therefore, make sure that a generated key point summarizes the majority 
of the arguments contained in the cluster. A key point should not exceed a length of {sum_token_length} tokens.
'''[1:-1].format(num_token = num_token, kp_token = kp_token, sum_token_length = number_dict[sum_token_length])
    # Add examples if few shot prompting should be performed
    if few_shot == True:
        system_message += '''
Here are two examples of good key points: "School uniform reduces bullying" is an opposing key point on the topic "We should 
abandon the use of school uniform" and "Guns lead to accidental deaths" is a supporting key point on the topic "We should 
abolish the right to keep and bear arms".
'''[:-1]
    # Create system and user messages if optimation should be performed locally: One message per cluster
    system_messages = []
    user_messages = []

    
    if optimization == 'local':
        # Iterate over clusters
        for cluster in clusters.keys():
            # Create user message 
            user_message = '''
Please generate {num_token} short (maximal length of {sum_token_length} tokens), salient and high quality {stance} 
{kp_token} on the topic "{topic}" so that it captures the main statement that is shared between most of the clustered arguments 
based on the following cluster of similar arguments: "{arguments}".
'''[1:-1].format(num_token = num_token, sum_token_length = number_dict[sum_token_length], stance = stance_dict[stance],
               kp_token = kp_token, topic = topic, arguments = '", "'.join(clusters[cluster]))  
##### Alternative: so that it summarize most of the clustered arguments
            user_message += '''
Since argument clusters are not optimal, they may contain arguments that do not actually belong together. Therefore, make sure that 
the generated key point summarizes the majority of the arguments contained in the cluster.  
'''[:-1]
            # Add note about the preference to generate fewer key points than overlapping key points in terms of content
            if (sum_max_num > 1) and (generate_less == True):
                user_message += '''
You should only generate as many key points as really necessary to summarize the arguments contained in the cluster. This 
means you should preferably generate fewer key points than the maximum permitted number of {sum_max_num} key points instead of
generating overlapping key points in terms of content.
'''[:-1].format(sum_max_num = number_dict[sum_max_num])
            # Add note about explanation of key point generation
            user_message += '''
Do not deliver an explanation why you generated the key point or any other information about it. Only return the 
individual key point.
'''[:-1]
            if exclude_topic == True:
                user_message +='''
Do not include any words from topic "{topic}" in the generated key points. For example, for the topic "We should end affirmative action", the 
key point "Reduces quality" would be a good choice. In contrast, the key point "We should end affirmative action 
because affirmative action reduces quality" would be a poor choice.
'''[:-1].format(topic = topic)
            system_messages.append(system_message)
            user_messages.append(user_message)


    elif optimization == 'global':
        system_messages.append(system_message)
        # Create user message
        user_message = '''
Please generate {num_token} short (maximal length of {sum_token_length} tokens), salient and high quality {stance} 
{kp_token} on the topic "{topic}" so that it captures the main statement that is shared among most of the clustered arguments 
for each of the following {num_clusters} clusters of similar arguments.
'''[1:-1].format(num_token = num_token, sum_token_length = number_dict[sum_token_length], stance = stance_dict[stance],
               kp_token = kp_token, topic = topic, num_clusters = len(list(clusters.keys())))
        for cluster_id in list(clusters.keys()):
            user_message += '''
Cluster {cluster_id}: "{arguments}".
'''[:-1].format(cluster_id = cluster_id, arguments = '", "'.join(clusters[cluster_id])) 
        user_message += '''
Since argument clusters are not optimal, they may contain arguments that do not actually belong together. Therefore, make sure that 
each generated key point summarizes the majority of the arguments contained in the respective cluster. In addition ensure that 
the generated key points do not overlap in terms of content.  
'''[:-1]
        # Add note about the preference to generate fewer key points than overlapping key points in terms of content
        if (sum_max_num > 1) and (generate_less == True):
            user_message += '''
You should only generate as many key points as really necessary to summarize the arguments contained in the clusters. This 
means you should preferably generate fewer key points than the maximum permitted number of {sum_max_num} key points instead of
generating overlapping key points in terms of content.
'''[:-1].format(sum_max_num = number_dict[sum_max_num])
        # Add note about explanation of key point generation
        user_message += '''
Do not deliver an explanation why you generated the key points or any other information. Only return the cluster 
ids and corresponding individual key points.
'''[:-1]
        if exclude_topic == True:
            user_message +='''
Do not include any words from topic "{topic}" in the generated key points. For example, for the topic "We should end affirmative action", the 
key point "Reduces quality" would be a good choice. In contrast, the key point "We should end affirmative action 
because affirmative action reduces quality" would be a poor choice.
'''[:-1].format(topic = topic)
        user_messages.append(user_message)
    # Return messages
    return system_messages, user_messages

def modify_gpt_output(sums):
    '''
    
    '''
    # Create empty list for storing modified summaries
    modified_sums = []
    # iterate over summaries
    for sum in sums:
        if sum not in ['', ' ', '  ']:
            # Split lines of the output and delete empty lines
            splitted_sum = [s for s in sum.splitlines()]
            # Iterate over sub-summaries (sum_max_num > 1)
            subsums = []
            for subsum in splitted_sum:
                # Check for colons
                if subsum.find(':') != -1:
                    subsum = subsum[subsum.find(':') + 1:]
                # Remove leading and trailing whitespaces
                subsum = subsum.strip()
                # Check for quotation marks
                if (subsum[0] == '"') or (subsum[0] == "'"):
                    subsum = subsum[1:]
                if (subsum[-1] == '"') or (subsum[-1] == "'"):
                    subsum = subsum[:-1]
                # Check for bullets
                if subsum[0] == '-':
                    subsum = subsum[2:]
                try:
                    if type(int(subsum[0])) == int:
                        subsum = subsum[3:]
                except:
                    pass
                # Check again for quotation marks
                if (subsum[0] == '"') or (subsum[0] == "'"):
                    subsum = subsum[1:]
                if (subsum[-1] == '"') or (subsum[-1] == "'"):
                    subsum = subsum[:-1]
                subsums.append(subsum)
            modified_sums.append(subsums)
    return modified_sums

def get_llm_cluster_sums(arguments, cluster_ids, topic, stance, llm = 'gpt-3.5-turbo', optimization = 'global', sum_token_length = 7, 
                sum_min_num = 1, sum_max_num = 1, few_shot = True, exclude_topic = False, generate_less = False, temperature = 0.5, frequency_penalty = None, n = 1, p = None):
    '''
    
    '''
    # Create dictionary with cluster ids as keys and list of corresponding arguments as values
    unique_cluster_ids = sorted(list(set(cluster_ids)))
    clusters = dict(zip(unique_cluster_ids, [[arguments[i] for i in range(len(arguments)) if cluster_ids[i] == cluster_id] for cluster_id in unique_cluster_ids]))
    noise = False
    if -1 in clusters.keys():
        noise = True
        new_key = max(list(clusters.keys())) + 1
        clusters[new_key] = clusters.pop(-1)
    # Get system messages and user messages for LLM
    system_messages, user_messages = get_llm_messages(clusters = clusters, 
                                                      topic = topic, stance = stance, 
                                                      optimization = optimization, 
                                                      sum_token_length = sum_token_length, 
                                                      sum_min_num = sum_min_num,
                                                      sum_max_num = sum_max_num, 
                                                      few_shot = few_shot, 
                                                      exclude_topic = exclude_topic,
                                                      generate_less = generate_less)
    # Create empty list to store summaries of individual clusters
    outputs = []
    # Get summaries
    if llm in ['gpt-3.5-turbo']:
        # Set enviroment variable 
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        # Set API client
        client = OpenAI(api_key = '...')
        # Iterate over system and user messages
        modified_outputs = [[] for i in range(len(unique_cluster_ids))]
        for i in tqdm(range(len(system_messages)), position = 0, desc = 'Get GPT-3.5 Cluster Sums', leave = False, colour = 'green'):
            # Create model input
            input = [{'role':'system', 
                      'content':system_messages[i]},
                      {'role':'user',
                       'content':user_messages[i]}]
            success = False
            n_trails = 0
            while (success == False) and (n_trails <= 2):
                try:
                    # Get model output
                    output = client.chat.completions.create(model = llm, 
                                                            messages = input, 
                                                            temperature = temperature, 
                                                            frequency_penalty = frequency_penalty, 
                                                            n = n, 
                                                            top_p = p,
                                                            timeout = 10)
                    # Append cluster summaries to output list
                    if optimization == 'local':
                        output = [output.choices[i].message.content for i in range(len(output.choices))]
                        
                        modified_output = modify_gpt_output(output)
                        modified_outputs[i] += np.array(modified_output).flatten().tolist()
                        if len(modified_output) == n:
                            success = True
                        else:
                            n_trails += 1
                            print('failed')

                    elif optimization == 'global':
                        output = [output.choices[i].message.content for i in range(len(output.choices))]
                        output = [[sum.strip() for sum in re.split(r'[*c*C]*luster* *\d*:', output[i]) if sum.strip() != ''] for i in range(len(output))]
                        outputs = [[] for i in range(len(unique_cluster_ids))]
                        for i in range(len(output)):
                            for j in range(len(unique_cluster_ids)):
                                outputs[j].append(output[i][j])
                        modified_outputs = [[] for i in range(len(unique_cluster_ids))]
                        for j in range(len(outputs)):
                            modified_outputs[j] += np.array(modify_gpt_output(outputs[j])).flatten().tolist()
                        if len(modified_outputs) == len(unique_cluster_ids):
                            for i in range(len(unique_cluster_ids)):
                                if len(modified_outputs[i]) == n:
                                    success = True

                                else:
                                    n_trails += 1
                                    print('failed')
                        else:
                            n_trails += 1
                            print('failed')
                except:
                    n_trails += 1
                    print('failed')
        
    elif llm == 'llama-2':
        # Set model path
        model_path = 'models/llms/llama-2-7b-chat-hf'
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Load model
        torch.mps.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(model_path)
        # Create pipeline
        pipe = pipeline(task = 'text-generation',
                        model = model,
                        tokenizer = tokenizer,
                        torch_dtype = torch.float16,
                        device_map = 'auto')
        # Iterate over system and user messages
        for i in tqdm(range(len(system_messages)), position = 0, desc = 'Get Llama-2 Cluster Sums', leave = False, colour = 'green'):
            # Create model input
            input = f'''<s>[INST] <<SYS>>\n{system_messages[i]}\n<</SYS>>\n{user_messages[i]} [/INST]'''
            # Get model output
            output = pipe(input,
                          do_sample = True,
                          top_k = 10,
                          num_return_sequences = 1,
                          eos_token_id = tokenizer.eos_token_id,
                          max_length = 4096)
            # Cut input tokens from output
            output = output[0]['generated_text'][len(input):]
            # Append cluster summaries to output list
            outputs.append(output)
        # Clear and filter model output
        modified_outputs = modify_llama_output(outputs)
    # Return summaries
    summaries = dict(zip(list(clusters.keys()), [out[0] if len(out) == 1 else out for out in modified_outputs]))
    if noise == True:
        summaries[-1] = summaries.pop(max(list(summaries.keys())))
    return summaries

###########################################################################################################################
### Old code ###
###########################################################################################################################

def modify_llama_output(sums):
    '''
    
    '''
    # Create empty list for storing modified summaries
    modified_sums = []
    # iterate over summaries
    for sum in sums:
        # Check for colons
        if sum.find(':') != -1:
            sum = sum[sum.find(':') + 1:]
        # Split lines of the output, delete empty lines and remove leading and trailing whitespaces from remaining lines
        splitted_sum = [s.strip() for s in sum.splitlines() if s != '']
        # Iterate over sub-summaries (sum_max_num > 1)
        subsums = []
        for subsum in splitted_sum:
            # Delete number of token note
            regex = re.search('\([0-9]+[0-9]*.*\)', subsum)
            if regex != None:
                span = regex.span()
                subsum = subsum[:span[0]] + subsum[span[1]:]
            # Delete explanation
            regex = re.search('[tT]his ([kK]ey|[aA]rg|[eE]xp)', subsum)
            if regex != None:
                span = regex.span()
                subsum = subsum[:span[0]]
            # Remove leading and trailing whitespaces
            subsum = subsum.strip()
            # Check for quotation marks
            if (subsum[0] == '"') or (subsum[0] == "'"):
                subsum = subsum[1:]
            if (subsum[-1] == '"') or (subsum[-1] == "'"):
                subsum = subsum[:-1]
            # Check for bullets
            if subsum[0] == '-':
                subsum = subsum[2:]
            try:
                if type(int(subsum[0])) == int:
                    subsum = subsum[3:]
            except:
                pass
            # Check again for quotation marks
            if (subsum[0] == '"') or (subsum[0] == "'"):
                subsum = subsum[1:]
            if (subsum[-1] == '"') or (subsum[-1] == "'"):
                subsum = subsum[:-1]
            subsums.append(subsum)
        modified_sums.append(subsums)
    return modified_sums