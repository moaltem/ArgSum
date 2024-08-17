# Import packages
import os
import numpy as np
import math 
import re
import itertools
from time import time
from collections import defaultdict
from scipy.stats import hmean
import torch
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI

from rouge_score import rouge_scorer
from bert_score import score as bert_score
from moverscore_v2 import word_mover_score
from ._utils_ import BARTScorer
from .bleurt_score import score as bleurt_score
from menli.MENLI import MENLI

from argsum import get_match_scores, get_match_score_matrix
from argsum.bleurt_score import score as bleurt_score

###########################################################################################################################
### Automatic Evaluation ###
###########################################################################################################################

def rescale_bart(x): 
    rescaled = math.tanh(math.exp((np.mean(x)/2)+1.3))
    return rescaled

def llm_sim_eval(sentences_1, sentences_2, model = 'gpt-3.5-turbo', scale_type = 'continuous', scale_min = 0, scale_max = 100, user_message_add_corr = True, user_message_few_shot = True, 
            user_message_single_num_out = False, temperature = 0.7):

    api_key = '...'
    client = OpenAI(api_key = api_key)

    user_message = f'''You act as an automatic evaluation metric for scoring two sentences with respect to their semantic similarity on a {scale_type} scale from {scale_min} to 
{scale_max}, where a score of {scale_min} means "sentences do not overlap semantically at all" and a score of {scale_max} means "sentences are semantically absolutely identical".
'''
    if user_message_add_corr == True:
        user_message += 'Your generated scores should have a high correlation with human judgments.'

    if user_message_few_shot == True:
        user_message += f''' Here are six examples: 

Sentence 1: A man is smoking.
Sentence 2: A man is skating.
Semantic Similarity Score: {(0.5 / 5) * scale_max}

Sentence 1: North Korea says to restart nuclear reactor
Sentence 2: North Korea Offers Talks: Complex May Re-Open
Semantic Similarity Score: {(3.5 / 5) * scale_max}

Sentence 1: Philippe becomes king of Belgium
Sentence 2: Philippe ascends throne of divided Belgium
Semantic Similarity Score: {(4 / 5) * scale_max}

Sentence 1: 4 dead, 3 injured in east China road accident
Sentence 2: 2 dead, 8 injured in central Israel traffic accident
Semantic Similarity Score: {(0 / 5) * scale_max}

Sentence 1: How the Dow Jones Industrial Average Did Wednesday
Sentence 2:How the Dow Jones Industrial Average Fared on Monday
Semantic Similarity Score: {(2 / 5) * scale_max}

Sentence 1: US top diplomat Kerry's wife rushed to hospital
Sentence 2: US Secretary of State John Kerry's wife rushed to hospital
Semantic Similarity Score: {(5 / 5) * scale_max}

'''

    outputs = []

    for i in range(len(sentences_1)):

        add_user_message = 'Score the following two sentences.'

        if user_message_single_num_out == True:
            add_user_message += ' Output a single score without explanation.'  

        add_user_message += f'''

Sentence 1: {sentences_1[i]}
Sentence 2: {sentences_2[i]}
Semantic Similarity Score:
'''

        input = [{'role':'user', 'content':user_message + add_user_message}]

        success = False
        while success == False:
            try:
                output = client.chat.completions.create(model = model, 
                                                        messages = input, 
                                                        temperature = temperature,
                                                        n = 10, 
                                                        timeout = 10)
        
                outputs.append(np.mean([float(re.findall(r'\b\d+(?:\.\d+)?\b', i)[0]) for i in [output.choices[i].message.content for i in range(len(output.choices))]]))
                success = True
            except:
                print('failed')
    
    return outputs

def get_soft_scores(reference, candidate, similarity_functions = 'all'):
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Create output dict
    if similarity_functions == 'all':
        similarity_functions = ['rouge_1', 'rouge_2', 'rouge_l', 
                                'bert_score_p', 'bert_score_r', 'bert_score_f', 
                                'mover_score', 
                                'bart_score_cnndm', 'bart_score_parabank', 'bart_score_cnndm_rs', 'bart_score_parabank_rs',
                                'bleurt',
                                'menli',
                                'all_mpnet_base_v2', 'sbert_roberta_match', 's_bert_all_mpnet_base_v2_match',
                                'deberta_match_1', 'deberta_match_2', 'deberta_match_mean',
                                'llm_sim']
    
    
    results = dict(zip(similarity_functions, [dict(zip(['sP', 'sR', 'sF1', 'runtime'], [np.nan for i in range(len(['sP', 'sR', 'sF1', 'runtime']))])) for i in range(len(similarity_functions))]))
    # Create two list such that all combinations of references and candidates are covered 
    refs = list(itertools.chain.from_iterable(itertools.repeat(x, len(candidate)) for x in reference))
    cands = candidate * len(reference)
    all_scores = []

    ################################################################################################################
    # Compute scores for each combination of references and candidates according to different similarity functions #
    ################################################################################################################

    # ROUGE
    if 'rouge_1' in similarity_functions:
        r_1 =  np.array([])
        r_1_scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer = True)
        results['rouge_1']['runtime'] = 0

        for comb in zip(refs, cands):
            start_time = time()
            r_1_scores = r_1_scorer.score(target = comb[0], prediction = comb[1])
            results['rouge_1']['runtime'] += time() - start_time
            r_1 = np.append(r_1, r_1_scores['rouge1'].fmeasure)
            results['rouge_1']['runtime'] = np.sum(results['rouge_1']['runtime'])
        all_scores.append(r_1)

    if 'rouge_2' in similarity_functions:
        r_2 =  np.array([])
        r_2_scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer = True)
        results['rouge_2']['runtime'] = 0

        for comb in zip(refs, cands):
            start_time = time()
            r_2_scores = r_2_scorer.score(target = comb[0], prediction = comb[1])
            results['rouge_2']['runtime'] += time() - start_time
            r_2 = np.append(r_2, r_2_scores['rouge2'].fmeasure)
            results['rouge_2']['runtime'] = np.sum(results['rouge_2']['runtime'])
        all_scores.append(r_2)

    if 'rouge_l' in similarity_functions:
        r_l =  np.array([])
        r_l_scorer = rouge_scorer.RougeScorer(['rougel'], use_stemmer = True)
        results['rouge_l']['runtime'] = 0

        for comb in zip(refs, cands):
            start_time = time()
            r_l_scores = r_l_scorer.score(target = comb[0], prediction = comb[1])
            results['rouge_l']['runtime'] += time() - start_time
            r_l = np.append(r_l, r_l_scores['rougel'].fmeasure)
            results['rouge_l']['runtime'] = np.sum(results['rouge_l']['runtime'])
        all_scores.append(r_l)

    # BERTScore
    if  ('bert_score_p' in similarity_functions) or ('bert_score_r' in similarity_functions) or ('bert_score_f' in similarity_functions):
        start_time = time()
        bert_score_p, bert_score_r, bert_score_f = bert_score(cands = cands, refs = refs, lang = 'en', verbose = False, rescale_with_baseline = True, device = 'mps')        
        bert_score_p, bert_score_r, bert_score_f = bert_score_p.numpy(),  bert_score_r.numpy(),  bert_score_f.numpy()
        if ('bert_score_p' in similarity_functions):
            results['bert_score_p']['runtime'] = time() - start_time
            all_scores.append(bert_score_p)
        if ('bert_score_r' in similarity_functions):
            results['bert_score_r']['runtime'] = time() - start_time
            all_scores.append(bert_score_r)
        if ('bert_score_f' in similarity_functions):
            results['bert_score_f']['runtime'] = time() - start_time
            all_scores.append(bert_score_f)

    # MoverScore
    if 'mover_score' in similarity_functions:
        start_time = time()
        mover_score = np.array(word_mover_score(refs = refs, hyps = cands, idf_dict_ref = defaultdict(lambda: 1.), idf_dict_hyp = defaultdict(lambda: 1.), 
                                                stop_words = [], n_gram = 1, remove_subwords = False, device = 'mps'))
        results['mover_score']['runtime'] = time() - start_time
        all_scores.append(mover_score)

    # BARTScore
    if ('bart_score_cnndm' in similarity_functions) or  ('bart_score_cnndm_rs' in similarity_functions) or ('bart_score_parabank' in similarity_functions) or ('bart_score_parabank_rs' in similarity_functions):
        
        if ('bart_score_cnndm' in similarity_functions) or  ('bart_score_cnndm_rs' in similarity_functions):
            bart_scorer_cnndm = BARTScorer(device = 'mps', checkpoint = 'facebook/bart-large-cnn')

            start_time = time()
            bart_score_cnndm = np.array(bart_scorer_cnndm.score(cands = cands, refs = refs))
            bart_score_cnndm_runtime = time() - start_time

            if 'bart_score_cnndm' in similarity_functions:
                all_scores.append(bart_score_cnndm)        
                results['bart_score_cnndm']['runtime'] = bart_score_cnndm_runtime

            if 'bart_score_cnndm_rs' in similarity_functions:
                bart_score_cnndm_rs = np.array([rescale_bart(x) for x in bart_score_cnndm])
                results['bart_score_cnndm_rs']['runtime'] = bart_score_cnndm_runtime
                all_scores.append(bart_score_cnndm_rs)

        elif ('bart_score_parabank' in similarity_functions) or  ('bart_score_parabank_rs' in similarity_functions):
            bart_scorer_parabank = BARTScorer(device = 'mps', checkpoint = 'facebook/bart-large-cnn')
            bart_scorer_parabank.load(path = 'models/metrics/bart_score.pth')

            start_time = time()
            bart_score_parabank = np.array(bart_scorer_parabank.score(cands = cands, refs = refs))
            bart_score_parabank_runtime = time() - start_time

            if 'bart_score_parabank' in similarity_functions:
                all_scores.append(bart_score_parabank)    
                results['bart_score_parabank']['runtime'] = bart_score_parabank_runtime

            if 'bart_score_parabank_rs' in similarity_functions:
                bart_score_parabank_rs = np.array([rescale_bart(x) for x in bart_score_parabank])
                results['bart_score_parabank_rs']['runtime'] = bart_score_parabank_runtime
                all_scores.append(bart_score_parabank_rs)

    # BLEURT
    if 'bleurt' in similarity_functions:
        start_time = time()
        bleurt = np.array(bleurt_score(refs = refs, cands = cands))
        results['bleurt']['runtime'] = time() - start_time
        all_scores.append(bleurt)

    # MENLI
    if 'menli' in similarity_functions:
        menli_scorer = MENLI(direction = 'hr', formula = 'e-c', nli_weight = 1.0, combine_with = 'None', model = 'D')
        start_time = time()
        menli = np.array(menli_scorer.score_nli(refs = refs, hyps = cands))
        results['menli']['runtime'] = time() - start_time
        all_scores.append(menli)

    # SBERT
    if 'all_mpnet_base_v2' in similarity_functions:
        all_mpnet_base_v2_scorer = SentenceTransformer('all-mpnet-base-v2')
        start_time = time()
        embeddings_cands = all_mpnet_base_v2_scorer.encode(refs, convert_to_tensor = True) # Semantic Similarity: all-mpnet-base-v2 
        embeddings_refs = all_mpnet_base_v2_scorer.encode(cands, convert_to_tensor = True)
        all_mpnet_base_v2 = util.cos_sim(embeddings_cands, embeddings_refs).diag().detach().numpy()
        results['all_mpnet_base_v2']['runtime'] = time() - start_time
        all_scores.append(all_mpnet_base_v2)

    if 'sbert_roberta_match' in similarity_functions:
        sbert_roberta_match_scorer = SentenceTransformer('models/match_scorer/bi_encoder/roberta_tp/2024-Feb-20_16-23-49')
        start_time = time()
        embeddings_cands = sbert_roberta_match_scorer.encode(refs, convert_to_tensor = True) # Match Score: roberta_tp
        embeddings_refs = sbert_roberta_match_scorer.encode(cands, convert_to_tensor = True)
        sbert_roberta_match = util.cos_sim(embeddings_cands, embeddings_refs).diag().detach().numpy()
        results['sbert_roberta_match']['runtime'] = time() - start_time
        all_scores.append(sbert_roberta_match)

    if 's_bert_all_mpnet_base_v2_match' in similarity_functions:
        s_bert_all_mpnet_base_v2_match_scorer = SentenceTransformer('models/match_scorer/bi_encoder/all-mpnet-base-v2/2024-Jun-05_22-29-27/checkpoints/6200')
        start_time = time()
        embeddings_cands = s_bert_all_mpnet_base_v2_match_scorer.encode(refs, convert_to_tensor = True) # Match Score: all-mpnet-base-v2
        embeddings_refs = s_bert_all_mpnet_base_v2_match_scorer.encode(cands, convert_to_tensor = True)
        s_bert_all_mpnet_base_v2_match = util.cos_sim(embeddings_cands, embeddings_refs).diag().detach().numpy()
        results['s_bert_all_mpnet_base_v2_match']['runtime'] = time() - start_time
        all_scores.append(s_bert_all_mpnet_base_v2_match)
    
    # DeBERTa Match Score
    if ('deberta_match_1' in similarity_functions) or ('deberta_match_2' in similarity_functions) or ('deberta_match_mean' in similarity_functions):
        deberta_match_scorer = torch.load('models/match_scorer/cross_encoder/deberta-v2_np/2024-Feb-20_13-41-02/best_model.pt')

        if 'deberta_match_1' in similarity_functions:
            start_time = time()
            deberta_match_1 = get_match_scores(arguments = refs, candidates = cands, model = deberta_match_scorer)
            deberta_match_1_runtime = time() - start_time
            results['deberta_match_1']['runtime'] = deberta_match_1_runtime
            all_scores.append(deberta_match_1)

        if 'deberta_match_2' in similarity_functions:
            start_time = time()
            deberta_match_2 = get_match_scores(arguments = cands, candidates = refs, model = deberta_match_scorer)
            deberta_match_2_runtime = time() - start_time
            results['deberta_match_2']['runtime'] = deberta_match_2_runtime
            all_scores.append(deberta_match_2)

        if 'deberta_match_mean' in similarity_functions:    
            deberta_match_mean = np.mean([deberta_match_1, deberta_match_2], axis = 0)
            results['deberta_match_mean']['runtime'] = deberta_match_1_runtime + deberta_match_2_runtime
            all_scores.append(deberta_match_mean)
    
    # LLM Sim
    if 'llm_sim' in similarity_functions:    
        start_time = time()
        llm_sim = np.array(llm_sim_eval(sentences_1 = refs, sentences_2 = cands, user_message_single_num_out = False, user_message_few_shot = True, temperature = 0.7, 
                            user_message_add_corr = True, scale_max = 100))
        results['llm_sim']['runtime'] = time() - start_time
        all_scores.append(llm_sim)
    
    # Extract SoftScores from scores of each similarity function
    for i, scores in enumerate(all_scores):
        # Get score matrix with references on y-axis and candidates on x-axis
        score_matrix = scores.reshape(len(reference), len(candidate))
        # SP: Finding the most suitable reference summary for each candidate
        sP = np.mean(np.max(score_matrix, axis = 0))
        results[similarity_functions[i]]['sP'] = float(sP)
        # SR: Finding the most suitable candidate summary for each reference
        sR = np.mean(np.max(score_matrix, axis = 1))
        results[similarity_functions[i]]['sR'] = float(sR)
        # SF1: Harmonic mean
        if sR < 0:
            sR = 0
        if sP <0:
            sP = 0
        try:
            results[similarity_functions[i]]['sF1'] = float(hmean([sP, sR]))
        except:
            results[similarity_functions[i]]['sF1'] = float(-hmean([-sP, -sR]))

    return results

def match_scorer_cov_eval(reference, candidate, match_scorer = None, threshold = 0.85):

    if match_scorer == None:
        match_scorer = torch.load('models/match_scorer/cross_encoder/roberta_np/2024-Feb-20_08-20-28/best_model.pt').to('mps')

    match_score_matrix = get_match_score_matrix(arguments = reference, candidates = candidate, model = match_scorer)
    coverage_matrix = match_score_matrix > threshold
    n_covered_refs = np.sum(coverage_matrix, axis = 1)
    n_cov = np.sum(n_covered_refs != 0)
    coverage = n_cov / len(reference)
    return coverage

def llm_cov_eval(reference, candidate, temperature = 0.6, topic = None, stance = None, include_top_sta = False):

    reference_str = ''
    for i, ref in enumerate(reference):
        reference_str += f'{i+1}. {ref}\n'
    
    candidate_str = ''
    for i, cand in enumerate(candidate):
        candidate_str += f'{i+1}. {cand}\n'
    
    stance_dict = {1:'Supporting', -1:'Opposing'}
    if include_top_sta == True:
        user_message = '''Your task is to evaluate a set of generated summaries obtained from a collection of arguments on a certain debate topic and stance against a set of reference 
summaries. The evaluation is conducted according to the criteria of coverage, meaning that the set of generated summaries aims to cover the main statements contained in the set of
reference summaries. Since each reference summary addresses a unique main statement, you are asked to count the number of reference summaries that are covered by the set of generated 
summaries. If a reference summary is only partially covered by the set of generated summaries, an increase of the count by 0.5 is allowed. Your counts aim to correlate well with human
judgments. In the following you are provided with an example, instructions for the evaluation procedure and finally with your evaluation task.

Example: 

Topic: We should abolish the right to keep and bear arms
Stance: Supporting

Set of Reference Summaries:
1. Banning guns would save lives
2. Guns can fall into the wrong hands
3. Guns lead to accidental deaths
4. Gun ownership allows for mass-shootings/general gun violence

Set of Generated Summaries:
1. Banning guns would save thousands of lives
2. Some people do not know how to handle firearms. This is a danger to them and others. 
3. Guns kill people, they should be banned
4. Firearms can fall into the hands of potential murderers
5. Firearms are a disgrace to humanity.
6. Without weapons there would be no war.

Coverage count: 3.5

Explanation:
- Reference 1 is covered by generated summaries 1 and 3
- Reference 2 is covered by generated summary 4
- Reference 3 is covered by generated summary 2
- Reference 4 is partially covered by generated summary 6

Note: Generated summary 5 does not cover any reference. It is very general.

Evaluation Procedure:

1. Read the reference summaries.
2. Read the generated summaries.
3. Go through the set of reference summaries and determine whether the reference summary at hand is covered by at least one generated summary. 
4. Once you have done this for each reference summary, count the number of covered reference summaries and return the resulting coverage count.

Evaluation Task:

Topic: {topic}
Stance: {stance_}

Set of Reference Summaries:
{reference_str}

Set of Generated Summaries:
{candidate_str}

Coverage count: 
'''.format(topic = topic, stance_ = stance_dict[stance], reference_str = reference_str, candidate_str = candidate_str, max = len(reference_str.splitlines()))
    
    elif include_top_sta == False:
        user_message = '''Your task is to evaluate a set of generated summaries obtained from a collection of arguments on a certain debate topic and stance against a set of reference 
summaries. The evaluation is conducted according to the criteria of coverage, meaning that the set of generated summaries aims to cover the main statements contained in the set of
reference summaries. Since each reference summary addresses a unique main statement, you are asked to count the number of reference summaries that are covered by the set of generated 
summaries. If a reference summary is only partially covered by the set of generated summaries, an increase of the count by 0.5 is allowed. Your counts aim to correlate well with human
judgments. In the following you are provided with an example, instructions for the evaluation procedure and finally with your evaluation task.

Example: 

Set of Reference Summaries:
1. Banning guns would save lives
2. Guns can fall into the wrong hands
3. Guns lead to accidental deaths
4. Gun ownership allows for mass-shootings/general gun violence

Set of Generated Summaries:
1. Banning guns would save thousands of lives
2. Some people do not know how to handle firearms. This is a danger to them and others. 
3. Guns kill people, they should be banned
4. Firearms can fall into the hands of potential murderers
5. Firearms are a disgrace to humanity.
6. Without weapons there would be no war.

Coverage count: 3.5

Explanation:
- Reference 1 is covered by generated summaries 1 and 3
- Reference 2 is covered by generated summary 4
- Reference 3 is covered by generated summary 2
- Reference 4 is partially covered by generated summary 6

Note: Generated summary 5 does not cover any reference. It is very general.

Evaluation Procedure:

1. Read the reference summaries.
2. Read the generated summaries.
3. Go through the set of reference summaries and determine whether the reference summary at hand is covered by at least one generated summary. 
4. Once you have done this for each reference summary, count the number of covered reference summaries and return the resulting coverage count.

Evaluation Task:

Set of Reference Summaries:
{reference_str}

Set of Generated Summaries:
{candidate_str}

Coverage count: 
'''.format(reference_str = reference_str, candidate_str = candidate_str, max = len(reference_str.splitlines()))

    client = OpenAI(api_key = '...')

    input = [{'role':'user',
               'content':user_message}]

    success = False
    n_failed = 0
    while (success == False) and (n_failed < 3):
        try:
            output = client.chat.completions.create(model = 'gpt-3.5-turbo', 
                                                    messages = input, 
                                                    temperature = temperature,
                                                    n = 10)
            filtered_output = [float(re.findall(r'\b\d+(?:\.\d+)?\b', i)[0]) for i in [output.choices[i].message.content for i in range(len(output.choices))]]
            
            for i in range(len(filtered_output)):
                if filtered_output[i] >= len(reference_str.splitlines()):
                    filtered_output[i] = len(reference_str.splitlines())

            answer =  np.nanmean(filtered_output) / len(reference_str.splitlines())
            success = True
            
        except:
             n_failed += 1
             print('failed')
             answer = None

    return answer

def llm_red_eval(candidate, temperature =  0.5, reference = None,):

    candidate_str = ''
    for i, cand in enumerate(candidate):
        candidate_str += f'{i+1}. {cand}\n'

    user_message = '''Your task is to evaluate a set of arguments on a certain debate topic and stance according to their uniqueness. Since arguments can be formulated differently, 
but address the same aspect of a debate, your task is to count the number of unique main statements addressed by the set of arguments. If a main statement addressed by an argument is 
only partially unique because it is also in parts covered by an other argument, an increase of the count by 0.5 is allowed. Your counts aim to correlate well with human judgments. In 
the following you are provided with an example, instructions for the evaluation procedure and finally with your evaluation task.

Example: 

Set of Arguments:
1. Banning guns would save lives 
2. Guns can fall into the wrong hands 
3. Guns lead to accidental deaths
4. Guns kill people, they should be banned 
5. Gun ownership allows for mass-shootings/general gun violence
6. Some people do not know how to handle firearms. This is a danger to them and others.
7. Banning guns would save thousands of lives 
8. Firearms can fall into the hands of potential murderers

Number of Unique Main Statements: 4

Explanation:
- Argument 1, 4 and 7 address the same main statement (guns kill people so without guns lives could be saved)
- Argument 2, 6 and 8 address the same main statement (guns could fall into the wrong hands, such as murders or people don't knowing how to handle guns)
- Argument 3 addresses a unique main statement, focusing on accidents with guns
- Argument 5 addresses a unique main statement, focusing intentional killing like terrorism or running amok

Notes: 
- Arguments 1, 4 and 7 are quite general, and therefore differ from the others
- E.g., argument 3 could also be assigned to 1, 4 and 7. Nevertheless it focuses on accidents and is more specific

Evaluation Procedure:

1. Read the arguments.
1. Go through the list of arguments, starting with the first argument.
2. Determine whether the argument at hand addresses a main statement of the debate.
3. Move on to the next one and consider whether it addresses a main statement and whether it has already been covered by previous arguments in the list.
4. Once you have done this for each argument, count the total number of unique main statements. 
5. Return your uniqueness count.

Evaluation Task:

Set of Arguments:
{candidate_str}

Number of Unique Main Statements: 
'''.format(candidate_str = candidate_str)

    client = OpenAI(api_key = '...')

    input = [{'role':'user',
               'content':user_message}]

    success = False
    n_failed = 0
    while (success == False) and (n_failed < 3):
        try:
            output = client.chat.completions.create(model = 'gpt-3.5-turbo', 
                                                    messages = input, 
                                                    temperature = temperature,
                                                    n = 10)
            filtered_output = [float(re.findall(r'\b\d+(?:\.\d+)?\b', i)[0]) for i in [output.choices[i].message.content for i in range(len(output.choices))]]
            
            for i in range(len(filtered_output)):
                if filtered_output[i] >= len(candidate_str.splitlines()):
                    filtered_output[i] = len(candidate_str.splitlines())
            answer =  1 - (np.nanmean(filtered_output) / len(candidate_str.splitlines()))
            success = True

        except:
             n_failed += 1
             print('failed')
             answer =  None

    return answer