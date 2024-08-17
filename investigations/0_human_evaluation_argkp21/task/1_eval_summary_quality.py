import pandas as pd
import os

if '1_summary_quality.csv' not in os.listdir('./results'):
    results = pd.DataFrame(columns = ['topic', 'stance', 'reference','prediction', 'n_covered', 'n_unique', 'coverage', 'redundancy', 'file_name'])
    file_names = os.listdir('./data')
    for file_name in file_names:
        if file_name[-4:] == '.csv':
            file = pd.read_csv('./data/' + file_name)
            topics = file['topic'].unique()
            stances = file['stance'].unique()
            for topic in topics:
                for stance in stances:
                    mask = (file['topic'] == topic) & (file['stance'] == stance)
                    reference = '\n'.join([f'{i+1}' + ': ' + file[mask]['key_point'].unique().tolist()[i] for i in range(len(file[mask]['key_point'].unique().tolist()))])
                    prediction = '\n'.join([f'{i+1}' + ': ' + file[mask][file_name].unique().tolist()[i] for i in range(len(file[mask][file_name].unique().tolist()))])
                    results = pd.concat([results, pd.DataFrame([dict(zip(results.columns, [topic, stance, reference, prediction, -2, -2, -2.0, -2.0, file_name]))])], ignore_index = True)
    results.sort_values(['topic', 'stance'], inplace = True)
    results.to_csv('./results/1_summary_quality.csv', sep = ',', index = False)

results = pd.read_csv('./results/1_summary_quality.csv')

try:
    idx = results.loc[(results['n_covered'] == -2) & (results['n_unique'] == -2), :].index[0]
except:
    idx = len(results)

while idx != len(results):

    results = pd.read_csv('./results/1_summary_quality.csv')
    idx = results.loc[(results['n_covered'] == -2) & (results['n_unique'] == -2), :].index[0]
    print('\n#############################################################################################################')
    print('##### COVERAGE:   Number of reference summaries covered by the set of generated summaries               #####')
    print('##### UNIQUENESS: Number of distinct/unique main statements contained in the set of generated summaries #####')
    print('#############################################################################################################\n')
    print(f'{idx+1}/{len(results)}\n')
    print(f'Topic:\t\t{results.loc[idx, "topic"]}\n')
    print(f'Stance:\t\t{dict(zip([-1,1], ["Opposing", "Supporting"]))[results.loc[idx, "stance"]]}\n')
    print('Set of reference summaries:')
    print(results.loc[idx, 'reference'], '\n')
    print('Set of generated summaries:')
    print(results.loc[idx, 'prediction'], '\n')

    n_covered = -2
    n_unique = -2

    n_refs = len(results.loc[idx, 'reference'].splitlines())
    n_preds = len(results.loc[idx, 'prediction'].splitlines())

    while n_covered not in [i for i in range(-1, n_refs + 1)] + [i + 0.5 for i in range(0, n_refs)]:
        try:
            n_covered = float(input('COVERAGE: '))
            coverage = n_covered / n_refs
            if n_covered not in [i for i in range(-1, n_refs + 1)] + [i + 0.5 for i in range(0, n_refs)]:
                print(f'\nYour input should be a value in the range [0,{n_refs}] with a steps size of 0.5. For the case you are not sure, you can answer with -1. Please try it again.\n')
        except:
            print(f'\nYour input should be a value in the range [0,{n_refs}] with a steps size of 0.5. For the case you are not sure, you can answer with -1. Please try it again.\n')
    
    while n_unique not in [i for i in range(-1, n_preds + 1)] + [i + 0.5 for i in range(0, n_preds)]:    
        try:
            n_unique = float(input('UNIQUENESS: '))
            redundancy = 1 - (n_unique / n_preds)
            if n_unique not in [i for i in range(-1, n_preds + 1)] + [i + 0.5 for i in range(0, n_preds)]:
                print(f'\nYour input should be a value in the range [0,{n_preds}] with a steps size of 0.5. For the case you are not sure, you can answer with -1. Please try it again.\n')
        except:
            print(f'\nYour input should be a value in the range [0,{n_preds}] with a steps size of 0.5. For the case you are not sure, you can answer with -1. Please try it again.\n')

    results.at[idx, 'n_covered'] = n_covered
    results.at[idx, 'n_unique'] = n_unique
    results.at[idx, 'coverage'] = coverage
    results.at[idx, 'redundancy'] = redundancy
    results.to_csv('./results/1_summary_quality.csv', sep = ',', index = False)

print('Thank you :)')





