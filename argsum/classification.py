# Import packages
import numpy as np
import pandas as pd
import nltk
import torch
from sentence_transformers import SentenceTransformer
from argsum.tools import (get_quality_scores, 
                          get_match_scores, 
                          get_match_score_matrix, 
                          get_best_match_plus_threshold_dict, 
                          get_best_match_plus_threshold)
from fast_pagerank import pagerank
import gc
from openai import OpenAI
from argsum.clustering import modify_gpt_output

###########################################################################################################################
### Classification approaches ###
###########################################################################################################################

# BarH 
def get_barh_candidates(arguments, topic, quality_scorer = None, quality_scorer_t = 0.7, min_proportion_candidates = 0.1):
    '''
    Extracts candidates from arguments given the topic.
    '''
    # Load quality scorer
    if quality_scorer == None:
        quality_scorer = torch.load('models/quality_scorer/bert_ft_topic_np_mace-p/2024-May-29_15-58-03/best_model.pt')
    # Create set of pronuons
    pronuons = {'all', 'another', 'any', 'anybody', 'anyone', 'anything', 'as', 'aught', 'both', 'each', 'each other', 'either', 
                'enough', 'everybody', 'everyone', 'everything', 'few', 'he', 'her', 'hers', 'herself', 'him', 'himself', 'his', 'I', 
                'idem', 'it', 'its', 'itself', 'many', 'me', 'mine', 'most', 'my', 'myself', 'naught', 'neither', 'no one', 'nobody', 
                'none', 'nothing', 'nought', 'one', 'one another', 'other', 'others', 'ought', 'our', 'ours', 'ourself', 'ourselves', 
                'several', 'she', 'some', 'somebody', 'someone', 'something', 'somewhat', 'such', 'suchlike', 'that', 'thee', 'their', 
                'theirs', 'theirself', 'theirselves', 'them', 'themself', 'themselves', 'there', 'these', 'they', 'thine', 'this', 
                'those', 'thou', 'thy', 'thyself', 'us', 'we', 'what', 'whatever', 'whatnot', 'whatsoever', 'whence', 'where', 'whereby', 
                'wherefrom', 'wherein', 'whereinto', 'whereof', 'whereon', 'wherever', 'wheresoever', 'whereto', 'whereunto', 'wherewith', 
                'wherewithal', 'whether', 'which', 'whichever', 'whichsoever', 'who', 'whoever', 'whom', 'whomever', 'whomso', 
                'whomsoever', 'whose', 'whosever', 'whosesoever', 'whoso', 'whosoever', 'ye', 'yon', 'yonder', 'you', 'your', 'yours', 
                'yourself', 'yourselves'}
    # Create NLTK Punkt Sentence Tokenizer and Regular-Expression Tokenizer objects
    punkt_sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    regexp_tokenizer = nltk.RegexpTokenizer(r'\w+')
    # Create empty array for key point candidates
    candidates = []
    candidates_without_quality_score = []
    remaining_arguments = []
    # Compute argument quality scores
    aqs = np.array(get_quality_scores(model = quality_scorer, arguments = arguments, topic = topic))
    
    del quality_scorer
    gc.collect()

    torch.mps.empty_cache()
    # Iterate over arguments 
    for i, arg in enumerate(arguments):
        # Filter out arguments consisting of multiple sentences
        sentences = punkt_sentence_tokenizer.tokenize(arg)
        if len(sentences) == 1:
            # Filter out arguments consisting of more than 15 tokens (less than 5 or more than 20 tokens)
            tokens = regexp_tokenizer.tokenize(arg)
            if len(tokens) <= 15: #((len(tokens) > 5) or (len(tokens) < 20)):
                # Filter out arguments starting with pronouns
                if tokens[0].lower() not in pronuons:
                    # Get argument quality score and filter out low-quality arguments
                    if aqs[i] > quality_scorer_t:
                        candidates.append((i, arg))
                    else:
                        candidates_without_quality_score.append((i, arg))
                else:
                    remaining_arguments.append((i, arg))
            else:
                remaining_arguments.append((i, arg))
        else:
            pass
    # If there are too few candidates, fill candidates with arguments which have a quality score above 80% quantile
    min_n_candidates = np.round(min_proportion_candidates * len(arguments))
    if len(candidates) < min_n_candidates:
        diff_n_candidates = int(min_n_candidates - len(candidates))
        candidates_without_quality_score_aqs = aqs[[cand[0] for cand in candidates_without_quality_score]]
        sorted_candidates_without_quality_score = [cand for _, cand in sorted(zip(candidates_without_quality_score_aqs, candidates_without_quality_score), reverse = True)]
        if len(sorted_candidates_without_quality_score) > diff_n_candidates:
            candidates += [sorted_candidates_without_quality_score[j] for j in [i for i in range(diff_n_candidates)]]
        else:
            candidates += sorted_candidates_without_quality_score
            diff_n_candidates = int(min_n_candidates - len(candidates))
            remaining_arguments_aqs = aqs[[arg[0] for arg in remaining_arguments]]
            sorted_remaining_arguments = [arg for _, arg in sorted(zip(remaining_arguments_aqs, remaining_arguments), reverse = True)]
            candidates += [sorted_remaining_arguments[j] for j in [i for i in range(diff_n_candidates)]]
    return [cand[1] for cand in candidates]

def get_barh_key_points(arguments, candidates, match_scorer = None, match_scorer_t = 0.856, scorer_cands = None, scorer_cands_t = None):
    '''
    Determins key points from candidates.
    '''
    # Load match scorer
    if match_scorer == None:
        match_scorer = torch.load('models/match_scorer/bi_encoder/roberta_tp/2024-Feb-20_16-23-49')
    if scorer_cands == None:
        scorer_cands = match_scorer
    if scorer_cands_t == None:
        scorer_cands_t = match_scorer_t
    # Update remaining arguments
    arguments = np.array(list(set(arguments).difference(candidates)))
    # Get matched candidates with corresponding matched arguments as well as unmatched candidates
    matched_cands_args = get_best_match_plus_threshold_dict(match_scorer = match_scorer, arguments = arguments, 
                                                            candidates = candidates, match_scorer_t = match_scorer_t)
    unmatched_cands = [cand for cand in candidates if cand not in list(matched_cands_args.keys())]
    # Sort matched candidates with corresponding matched arguments according to number of matches
    matched_cands_args_sorted = {} 
    for k in sorted(matched_cands_args, key = lambda k: len(matched_cands_args[k]), reverse = True):
        matched_cands_args_sorted[k] = matched_cands_args[k]
    # Get match score matrix for each combination of sorted matched candidates
    matched_cands_sorted = np.array(list(matched_cands_args_sorted.keys()))
    if len(matched_cands_sorted) > 0:
        ms_matched_cands_sorted = get_match_score_matrix(scorer_cands, matched_cands_sorted, matched_cands_sorted)
        # Remove each matched candidate (and corresponding matched arguments) whose match score with a 
        # higher-ranked matched candidate exceeded the threshold
        matched_cands_to_remove = np.array([])
        matched_args_to_remove = np.array([])
        for i, cand_1 in enumerate(matched_cands_sorted):
            for j, cand_2 in enumerate(matched_cands_sorted):
                if i < j:
                    ms = (ms_matched_cands_sorted[i, j] + ms_matched_cands_sorted[i,j]) / 2
                    if ms > scorer_cands_t:
                        # Check whether the lower-ranked candidate was already removed in a previous comparison or not
                        if cand_2 in matched_cands_args.keys():
                            matched_cands_to_remove = np.append(matched_cands_to_remove, cand_2)
                            matched_args_to_remove = np.append(matched_args_to_remove, matched_cands_args[cand_2])
                            matched_cands_args.pop(cand_2)  
        # Get resulting key points as well as removed matched candidates with corresponding matched arguments 
        # (both + unmatched candidates referred to as remaining arguments)
        key_points = np.array(list(matched_cands_args.keys()))
        rem_args = np.concatenate([matched_cands_to_remove, matched_args_to_remove, unmatched_cands])
        # Get matched key points with corresponding matched arguments for remaining arguments
        if len(rem_args) > 0:
            matched_kps_rem_args = get_best_match_plus_threshold_dict(match_scorer = match_scorer, arguments = rem_args, 
                                                                    candidates = key_points, match_scorer_t = match_scorer_t)
            # Merge initially matched arguments and matched remaining arguments
            for kp in matched_kps_rem_args.keys():
                matched_cands_args[kp] = np.append(matched_cands_args[kp], matched_kps_rem_args[kp])
        # Sort matched key points with corresponding matched arguments according to number of matches
        matched_kps_args_sorted = {}
        for k in sorted(matched_cands_args, key = lambda k: len(matched_cands_args[k]), reverse = True):
            matched_kps_args_sorted[k] = matched_cands_args[k]   
        kps = list(matched_kps_args_sorted.keys())
        kps = dict(zip([id for id in range(len(kps))], kps))
    else:
        kps = None
    return kps

def get_barh_classification_sums(arguments, topic, stance = None, quality_scorer = None, match_scorer = None, 
                                quality_scorer_t = 0.7, min_proportion_candidates = 0.1, match_scorer_t = 0.856, 
                                final_match_scorer_t = None, scorer_cands = None, scorer_cands_t = None, use_llm = False,
                                sum_token_length = 7, sum_min_num = 3, sum_min_num_plus = 2, sum_max_num = None, temperature = 0.5, 
                                frequency_penalty = None, few_shot = True, return_kps = True):

    if match_scorer == None:
        match_scorer = torch.load('models/match_scorer/cross_encoder/roberta_np/2024-Feb-20_08-20-28/best_model.pt')
 
    # Set final match scorer threshold
    if final_match_scorer_t == None:
        final_match_scorer_t = match_scorer_t
    
    # Get candidates and key points
    if use_llm == False:
        # Get candidates
        candidates = get_barh_candidates(arguments = arguments, topic = topic, quality_scorer = quality_scorer, 
                                         quality_scorer_t = quality_scorer_t, 
                                         min_proportion_candidates = min_proportion_candidates)
        # Extract key points
        key_points = get_barh_key_points(arguments = arguments, candidates = candidates, match_scorer = match_scorer, 
                                         match_scorer_t = match_scorer_t, scorer_cands = scorer_cands, 
                                         scorer_cands_t = scorer_cands_t)
    elif use_llm == 'candidates':
        # Get candidates
        candidates = get_llm_classification_sums(arguments, topic, stance, purpose = 'candidates',  
                                                sum_token_length = sum_token_length, sum_min_num = sum_min_num, 
                                                sum_min_num_plus = sum_min_num_plus,
                                                sum_max_num = sum_max_num, temperature = temperature, 
                                                frequency_penalty = frequency_penalty, few_shot = few_shot)
        # Extract key points
        key_points = get_barh_key_points(arguments = arguments, candidates = candidates, match_scorer = match_scorer, 
                                         match_scorer_t = match_scorer_t, scorer_cands = scorer_cands, 
                                         scorer_cands_t = scorer_cands_t)
    
    elif use_llm == 'key_points':
        # Generate key points
        key_points = get_llm_classification_sums(arguments, topic, stance, purpose = 'key_points',  
                                                sum_token_length = sum_token_length, sum_min_num = sum_min_num, 
                                                sum_min_num_plus = sum_min_num_plus,
                                                sum_max_num = sum_max_num, temperature = temperature, 
                                                frequency_penalty = frequency_penalty, few_shot = few_shot)
        key_points = dict(zip([i for i in range(len(key_points))], key_points))
    
    # Get matches
    if key_points != None:

        matches = get_best_match_plus_threshold(arguments = arguments, key_points = list(key_points.values()), 
                                                match_scorer = match_scorer, match_scorer_t = final_match_scorer_t)
        
    else: 
        matches = None
    
    # Return matches and key points
    if return_kps == False:
        return matches
    elif return_kps == True:
        return matches, key_points

# SMatchToPr
def get_smatchtopr_candidates(arguments, topic, quality_scorer_t = 0.8, min_proportion_candidates = 0.1):
    '''
    Extracts candidates from arguments given the topic.
    They splitted arguments of multiple sentences. Here: Only consider single-sentence arguments
    '''
    # Load quality scores
    aqs = np.array(get_quality_scores(model = 'debater_api', arguments = arguments, topic = topic, n = len(arguments)))
    #quality_scores = quality_scores_df[(quality_scores_df['topic'] == topic) & (quality_scores_df['stance'] == stance)]['score'].to_list()
    # Create set of pronuons
    pronuons = {'all', 'another', 'any', 'anybody', 'anyone', 'anything', 'as', 'aught', 'both', 'each', 'each other', 'either', 
                'enough', 'everybody', 'everyone', 'everything', 'few', 'he', 'her', 'hers', 'herself', 'him', 'himself', 'his', 'I', 
                'idem', 'it', 'its', 'itself', 'many', 'me', 'mine', 'most', 'my', 'myself', 'naught', 'neither', 'no one', 'nobody', 
                'none', 'nothing', 'nought', 'one', 'one another', 'other', 'others', 'ought', 'our', 'ours', 'ourself', 'ourselves', 
                'several', 'she', 'some', 'somebody', 'someone', 'something', 'somewhat', 'such', 'suchlike', 'that', 'thee', 'their', 
                'theirs', 'theirself', 'theirselves', 'them', 'themself', 'themselves', 'there', 'these', 'they', 'thine', 'this', 
                'those', 'thou', 'thy', 'thyself', 'us', 'we', 'what', 'whatever', 'whatnot', 'whatsoever', 'whence', 'where', 'whereby', 
                'wherefrom', 'wherein', 'whereinto', 'whereof', 'whereon', 'wherever', 'wheresoever', 'whereto', 'whereunto', 'wherewith', 
                'wherewithal', 'whether', 'which', 'whichever', 'whichsoever', 'who', 'whoever', 'whom', 'whomever', 'whomso', 
                'whomsoever', 'whose', 'whosever', 'whosesoever', 'whoso', 'whosoever', 'ye', 'yon', 'yonder', 'you', 'your', 'yours', 
                'yourself', 'yourselves'}
    # Create NLTK Punkt Sentence Tokenizer and Regular-Expression Tokenizer objects
    punkt_sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    regexp_tokenizer = nltk.RegexpTokenizer(r'\w+')
    # Create empty array for key point candidates
    candidates = []
    candidate_qualities = []

    candidates_without_quality_score = []
    remaining_arguments = []
    # Iterate over arguments 
    for i, arg in enumerate(arguments):
        # Filter out arguments consisting of multiple sentences
        sentences = punkt_sentence_tokenizer.tokenize(arg)
        if len(sentences) == 1:
            # Filter out arguments consisting of less than 5 or more than 20 tokens
            tokens = regexp_tokenizer.tokenize(arg)
            if ((len(tokens) > 5) and (len(tokens) < 20)):
                # Filter out arguments starting with pronouns
                if tokens[0].lower() not in pronuons:
                    # Get argument quality score and filter out low-quality arguments
                    if aqs[i] > quality_scorer_t:
                        candidates.append((i, arg))
                        candidate_qualities.append(aqs[i])
                    else:
                        candidates_without_quality_score.append((i, arg))
                else:
                    remaining_arguments.append((i, arg))
            else:
                remaining_arguments.append((i, arg))
        else:
            pass
    
    # If there are too few candidates, fill candidates with arguments which have a quality score above 80% quantile
    min_n_candidates = np.round(min_proportion_candidates * len(arguments))
    if len(candidates) < min_n_candidates:
        diff_n_candidates = int(min_n_candidates - len(candidates))
        candidates_without_quality_score_aqs = aqs[[cand[0] for cand in candidates_without_quality_score]]
        sorted_candidates_without_quality_score = [(aqs, cand) for aqs, cand in sorted(zip(candidates_without_quality_score_aqs, candidates_without_quality_score), reverse = True)]
        if len(sorted_candidates_without_quality_score) > diff_n_candidates:
            candidates += [sorted_candidates_without_quality_score[j][1] for j in [i for i in range(diff_n_candidates)]]
            candidate_qualities += [sorted_candidates_without_quality_score[j][0] for j in [i for i in range(diff_n_candidates)]]
        else:
            
            candidates += [sorted_candidates_without_quality_score[j][1] for j in [i for i in range(diff_n_candidates)]]
            candidate_qualities += [sorted_candidates_without_quality_score[j][0] for j in [i for i in range(diff_n_candidates)]]

            diff_n_candidates = int(min_n_candidates - len(candidates))
            remaining_arguments_aqs = aqs[[arg[0] for arg in remaining_arguments]]
            sorted_remaining_arguments = [(aqs,arg) for aqs, arg in sorted(zip(remaining_arguments_aqs, remaining_arguments), reverse = True)]

            candidates += [sorted_remaining_arguments[j][1] for j in [i for i in range(diff_n_candidates)]]
            candidate_qualities += [sorted_remaining_arguments[j][0] for j in [i for i in range(diff_n_candidates)]]

    return [cand[1] for cand in candidates], candidate_qualities

def get_smatchtopr_key_points(candidates, topic, candidate_qualities, match_scorer = None, match_scorer_pr_t = 0.4, 
                              damping_factor = 0.2, scorer_cands = None, scorer_cands_t = 0.8):
    if match_scorer == None:
        match_scorer = SentenceTransformer('models/match_scorer/bi_encoder/roberta_tp/2024-Feb-20_16-23-49')
    if scorer_cands == None:
        scorer_cands = match_scorer
    # Compute match scores of candidates
    match_score_matrix = get_match_score_matrix(model = match_scorer, arguments = candidates, candidates = candidates, topic = topic)
    match_score_matrix[match_score_matrix < match_scorer_pr_t] = 0
    # Apply pagerank and get ordered candidates
    pr_importance_scores = pagerank(match_score_matrix, personalize = np.array(candidate_qualities), p = damping_factor)
    ranked_candidates = sorted(list(zip(candidates, pr_importance_scores)), key = lambda x: -x[1])   
    # Filter candidates and get key points
    kps = [ranked_candidates[0][0]]
    for cand in ranked_candidates[1:]:
        match_scores = get_match_scores(model = scorer_cands, arguments = [cand[0] for i in range(len(kps))], candidates = kps, topic = topic)
        if np.max(match_scores) < scorer_cands_t:
            kps.append(cand[0])
    return dict(zip([i for i in range(len(kps))], kps))

def get_smatchtopr_classification_sums(arguments, topic, stance, match_scorer = None, quality_scorer_t = 0.8, 
                   match_scorer_pr_t = 0.4, damping_factor = 0.2, scorer_cands = None, scorer_cands_t = 0.8,
                   final_match_scorer_t = 0, use_llm = False, sum_token_length = 7, sum_min_num = 3, sum_min_num_plus = 2,
                   sum_max_num = None, temperature = 0.5, frequency_penalty = None, few_shot = True, return_kps = True, min_proportion_candidates = 0.1):
    
    if match_scorer == None:
        match_scorer = SentenceTransformer('models/match_scorer/bi_encoder/roberta_tp/2024-Feb-20_16-23-49')

    # Get candidates and key points
    if use_llm == False:
        # Get candidates
        candidates, candidate_qualities = get_smatchtopr_candidates(arguments = arguments, topic = topic, 
                                                                    quality_scorer_t = quality_scorer_t, min_proportion_candidates = min_proportion_candidates)
        # Extract key points 
        key_points = get_smatchtopr_key_points(candidates = candidates, topic = topic, candidate_qualities = candidate_qualities, 
                                               match_scorer = match_scorer, match_scorer_pr_t = match_scorer_pr_t, 
                                               damping_factor = damping_factor, scorer_cands = scorer_cands, 
                                               scorer_cands_t = scorer_cands_t)
    elif use_llm == 'candidates':
        # Get candidates
        candidates = get_llm_classification_sums(arguments, topic, stance, purpose = 'candidates',  
                                                sum_token_length = sum_token_length, sum_min_num = sum_min_num, 
                                                sum_min_num_plus = sum_min_num_plus,
                                                sum_max_num = sum_max_num, temperature = temperature, 
                                                frequency_penalty = frequency_penalty, few_shot = few_shot)
        candidate_qualities = get_quality_scores(model = 'debater_api', arguments = candidates, topic = topic, n = len(candidates))
        # Extract key points 
        key_points = get_smatchtopr_key_points(candidates = candidates, topic = topic, candidate_qualities = candidate_qualities, 
                                               match_scorer = match_scorer, match_scorer_pr_t = match_scorer_pr_t, 
                                               damping_factor = damping_factor, scorer_cands = scorer_cands, 
                                               scorer_cands_t = scorer_cands_t)
    elif use_llm == 'key_points':
        # Generate key points
        print(sum_min_num + sum_min_num_plus)
        key_points = get_llm_classification_sums(arguments, topic, stance, purpose = 'key_points',  
                                                sum_token_length = sum_token_length, sum_min_num = sum_min_num, 
                                                sum_min_num_plus = sum_min_num_plus,
                                                sum_max_num = sum_max_num, temperature = temperature, 
                                                frequency_penalty = frequency_penalty, few_shot = few_shot)
        key_points = dict(zip([i for i in range(len(key_points))], key_points)) 

    # Get matches
    matches = get_best_match_plus_threshold(match_scorer = match_scorer, arguments = arguments, key_points = list(key_points.values()), 
                                            match_scorer_t = final_match_scorer_t, topic = topic)

    # Return matches and optionally key points
    if return_kps == False:
        return matches
    elif return_kps == True:
        return matches, key_points

###########################################################################################################################
### LLM-based candidates and key points ###
###########################################################################################################################

def get_llm_classification_sums(arguments, topic, stance, purpose = 'candidates', sum_token_length = 7, sum_min_num = 3, sum_max_num = None, 
                                sum_min_num_plus = 2, temperature = 0.5, frequency_penalty =  None, few_shot = True):    
    # Create dict with numerical stance as key and textual stance as value
    stance_dict = {-1:'opposing', 1:'supporting'}
    # Create dict to map numerical number to written sring
    number_dict = {0:'zero', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five', 6:'six', 7:'seven', 8:'eight', 9:'nine', 10:'ten',              
                   11:'eleven', 12:'twelve', 13:'thirteen', 14:'fourteen', 15:'fifteen', 16:'sixteen', 17:'seventeen', 
                   18:'eighteen', 19:'nineteen', 20:'twenty'}
    if sum_max_num != None:
        num_token =  number_dict[sum_min_num] + ' to ' + number_dict[sum_max_num]
    elif sum_max_num == None:
        num_token =  number_dict[sum_min_num] + ' to ' + number_dict[sum_min_num + sum_min_num_plus]
        sum_max_num = sum_min_num + sum_min_num_plus
    system_message = '''
You are a professional debater and you can express yourself succinctly. If you are given a corpus of arguments on a certain debate 
topic and stance, you find {num_token} appropriate salient single sentences, called key points, summarizing most of the 
arguments and providing a textual and quantitative view of the data. A key point can be seen as a meta argument why one is for or 
against a certain topic. Make sure that the generated key points summarize the majority of the arguments contained in the corpus.
A key point should not exceed a length of {sum_token_length} tokens. 
'''[1:-1].format(num_token = num_token, sum_token_length = number_dict[sum_token_length])
    if few_shot == True:
        system_message += '''
Here are two examples of good key points: "School uniform reduces bullying" is an opposing key point on the topic "We should 
abandon the use of school uniform" and "Guns lead to accidental deaths" is a supporting key point on the topic "We should abolish 
the right to keep and bear arms".
'''[:-1]
    if purpose == 'candidates':
        user_message = '''
Please generate {num_token} short (maximal length of {sum_token_length} tokens), salient and high quality {stance} key points on 
the topic "{topic}" so that they capture the main statements that are shared between most of the arguments based on the following 
corpus of arguments: "{arguments}". 
'''[:-1].format(sum_token_length = number_dict[sum_token_length], stance = stance_dict[stance], topic = topic, arguments = arguments, 
                num_token = num_token)
    elif purpose == 'key_points':
        user_message = '''
Please generate {num_token} short (maximal length of {sum_token_length} tokens), salient and high quality {stance} key points on 
the topic "{topic}" so that they capture the main statements that are shared between most of the arguments based on the following 
corpus of arguments: "{arguments}". 
'''[:-1].format(sum_token_length = number_dict[sum_token_length], stance = stance_dict[stance], topic = topic, arguments = arguments, 
                num_token = num_token)
        user_message += '''
You should only generate as many key points as necessary to summarize the arguments contained in the corpus. This means you should 
preferably generate fewer key points than the maximum permitted number of {sum_max_num} key points instead of generating overlapping 
key points in terms of content.
'''[:-1].format(sum_max_num = number_dict[sum_max_num])
    # Set API client
    client = OpenAI(api_key = '...')
    # Create model input
    input = [{'role':'system', 
            'content':system_message},
            {'role':'user',
            'content':user_message}]
    # Get model output
    output = client.chat.completions.create(model = 'gpt-3.5-turbo', messages = input, temperature = temperature, frequency_penalty = frequency_penalty).choices[0].message.content
    modified_output = [out[0] for out in modify_gpt_output(output.splitlines())]
    return modified_output



    topics = df['topic'].unique().tolist()
    stances = [str(int(sta)) for sta in sorted(df['stance'].unique())]
    best_aris = []
    comps = []
    props_matched = []
    best_params = []
    tops = []
    stas = []
    best_sums = []
    best_scores = []

    for topic in topics:
        for stance in stances:

            mask = (df['topic'] == topic) & (df['stance'] == int(stance))
            arguments = df[mask]['argument'].to_list()

            best_ari = 0
            comp = None
            prop_matched = None
            best_param = None
            sum = None

            if specification == 'bm':
                for comb in classification_separation_dict['bm'].keys():
                    if len(list(set(classification_sums_dict['bm'][comb][topic][stance]['sum_ids']))) >= min_n_clusters:
                        if classification_separation_dict['bm'][comb][topic][stance]['ari'] > best_ari:
                            best_ari = classification_separation_dict['bm'][comb][topic][stance]['ari']
                            comp = classification_separation_dict['bm'][comb][topic][stance]['comp']
                            prop_matched = classification_separation_dict['bm'][comb][topic][stance]['prop_matched']
                            best_param = comb
                            sum = classification_sums_dict['bm'][comb][topic][stance]['sums']
                            sum_ids = classification_sums_dict['bm'][comb][topic][stance]['sum_ids']
                       
            elif specification == 'bm_plus_th_noise':
                for comb in classification_separation_dict['bm_plus_th'].keys():
                    if len(list(set([id for id in classification_sums_dict['bm_plus_th'][comb][topic][stance]['sum_ids'] if id != -1]))) >= min_n_clusters:
                        if classification_separation_dict['bm_plus_th'][comb][topic][stance]['noise']['ari'] > best_ari:
                            best_ari = classification_separation_dict['bm_plus_th'][comb][topic][stance]['noise']['ari']
                            comp = classification_separation_dict['bm_plus_th'][comb][topic][stance]['noise']['comp']
                            prop_matched = classification_separation_dict['bm_plus_th'][comb][topic][stance]['noise']['prop_matched']
                            best_param = comb
                            sum = classification_sums_dict['bm_plus_th'][comb][topic][stance]['sums']
                            sum_ids = classification_sums_dict['bm_plus_th'][comb][topic][stance]['sum_ids']
                        
            elif specification == 'bm_plus_th_no_noise':
                for comb in classification_separation_dict['bm_plus_th'].keys():
                    if len(list(set([id for id in classification_sums_dict['bm_plus_th'][comb][topic][stance]['sum_ids'] if id != -1]))) >= min_n_clusters:
                        if classification_separation_dict['bm_plus_th'][comb][topic][stance]['no_noise']['ari'] > best_ari:
                            best_ari = classification_separation_dict['bm_plus_th'][comb][topic][stance]['no_noise']['ari']
                            comp = classification_separation_dict['bm_plus_th'][comb][topic][stance]['no_noise']['comp']
                            prop_matched = classification_separation_dict['bm_plus_th'][comb][topic][stance]['no_noise']['prop_matched']
                            best_param = comb
                            sum = classification_sums_dict['bm_plus_th'][comb][topic][stance]['sums']
                            sum_ids = classification_sums_dict['bm_plus_th'][comb][topic][stance]['sum_ids']
                        
            elif specification == 'bm_plus_th_no_noise_prop':
                for comb in classification_separation_dict['bm_plus_th'].keys():
                    if len(list(set([id for id in classification_sums_dict['bm_plus_th'][comb][topic][stance]['sum_ids'] if id != -1]))) >= min_n_clusters:
                        if (classification_separation_dict['bm_plus_th'][comb][topic][stance]['no_noise']['ari'] > best_ari) and (classification_separation_dict['bm_plus_th'][comb][topic][stance]['no_noise']['prop_matched'] >= prop):
                            best_ari = classification_separation_dict['bm_plus_th'][comb][topic][stance]['no_noise']['ari']
                            comp = classification_separation_dict['bm_plus_th'][comb][topic][stance]['no_noise']['comp']
                            prop_matched = classification_separation_dict['bm_plus_th'][comb][topic][stance]['no_noise']['prop_matched']
                            best_param = comb
                            sum = classification_sums_dict['bm_plus_th'][comb][topic][stance]['sums']
                            sum_ids =  classification_sums_dict['bm_plus_th'][comb][topic][stance]['sum_ids']

            if best_ari == 0:
                best_ari = None                        
            best_aris.append(best_ari)
            comps.append(comp)
            props_matched.append(prop_matched)
            best_params.append(best_param)
            tops.append(topic)
            stas.append(stance)

            if sum != None:
                summaries_concat = []
                arguments_concat = []
                idx_concat = []
                cluster_sizes = []

                for sum_id in sum.keys():
                    if int(sum_id) in sum_ids: 
                        # Get unique summaries for each cluster
                        if type(sum[sum_id]) == str:
                            summaries_classification = list(set([sum[sum_id]]))
                        elif type(sum[sum_id]) == list:
                            summaries_classification = list(set(sum[sum_id]))
                        # Get arguments for each cluster
                        arguments_classification = [arguments[i] for i in range(len(arguments)) if str(sum_ids[i]) == sum_id]
                        cluster_sizes.append(len(arguments_classification))
                        # Concatenate unique cluster summaries and cluster arguments 
                        summaries_concat += np.array([[summary] * len(arguments_classification) for summary in summaries_classification]).flatten().tolist()
                        arguments_concat += arguments_classification * len(summaries_classification)
                        idx_concat.append([i for i in range(0,len(arguments_classification * len(summaries_classification)) + 1, len(arguments_classification))])
                        scores_concat = match_evaluation_callable(arguments = arguments_concat, summaries = summaries_concat)

                best_sum = []
                best_score = []
                current_idx_concat = 0

                for i, sum_id in enumerate([id for id in list(sum.keys()) if int(id) in sum_ids]):
                    if int(sum_id) in sum_ids: 
                        classification_idx_concat = idx_concat[i]
                        classification_scores_concat = scores_concat[current_idx_concat:current_idx_concat + classification_idx_concat[-1]]
                        scores_splitted = [float(np.round(np.mean(classification_scores_concat[classification_idx_concat[i]:classification_idx_concat[i+1]]), 4)) for i in range(len(classification_idx_concat) - 1)]
                        if type(sum[sum_id]) == str:
                            best = list(set([sum[sum_id]]))[np.argmax(scores_splitted)]
                        elif type(sum[sum_id]) == list:
                            best = list(set(sum[sum_id]))[np.argmax(scores_splitted)]
    
                        best_sum.append(best)
                        best_score.append(np.max(scores_splitted))
                        current_idx_concat = classification_idx_concat[-1]
                        
                best_sums.append(dict(zip([id for id in list(sum.keys()) if int(id) in sum_ids], best_sum)))
                best_scores.append(np.average(best_score, weights = cluster_sizes))

            else:
                best_sums.append(None)
                best_scores.append(None)
    
    results = pd.DataFrame([tops, stas, best_aris, comps, props_matched, best_params, best_sums, best_scores], index = ['topic', 'stance', 'best_ari', 'comp', 'prop_matched', 'param', 'sum', 'sum_score']).T
    results = pd.concat([results, pd.DataFrame([dict(zip(list(results.columns), ['', '',np.round(results['best_ari'].mean(), 4), np.round(results['comp'].mean(), 4), np.round(results['prop_matched'].mean(), 4), '', '', np.round(results['sum_score'].mean(), 4) ]))])], ignore_index = True)
    return results