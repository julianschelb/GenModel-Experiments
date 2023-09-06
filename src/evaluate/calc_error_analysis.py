import shared_functions as sf
import numpy as np
import pandas as pd
import nltk
import string

PUNCTUATION = string.punctuation
FILES = sf.get_files_from_folder('../output','json')

def calculate_time_cost():
    summary = [['model','total_time','average','min','max','std dev']]
    for file in FILES:
        data = sf.import_json(file)
        meta = data['metadata']
        title = f'{meta["model"]} {meta["prompt"]} {meta["splitter"]}'
        content = data['content']

        duration = [sum(list(x['duration'].values())) for x in content if 'duration' in x.keys()]


        summary.append([title, sum(duration), np.mean(duration), min(duration), max(duration), np.std(duration)])
    df = pd.DataFrame(summary[1:],columns=summary[0])
    return df

def remove_punctuation(string_form):
    for i in PUNCTUATION:
        string_form= string_form.replace(i,'')
    return string_form

def calculate_true_positives(data):
    pm = find_perfect_matches(data)
    nm = find_noisy_matches(data)
    partm = find_partial_matches(data)
    pm_gte_50 = len([x for x in partm if x >=0.5])
    return pm + nm + pm_gte_50

def find_perfect_matches(data):
    perfect_matches = []
    for i in range(len(data)):
        elt = data[i]
        for hvv in ['hero','villain','victim']:
            if hvv not in elt or f'model_{hvv}' not in elt.keys():
                continue

            if elt[hvv]=='None':
                continue

            if isinstance(elt[f'model_{hvv}'],list):
                elt[f'model_{hvv}'] = " ".join(elt[f'model_{hvv}'])

            true_answers = elt[hvv].split(',')

            for a in true_answers:
                np_truth = remove_punctuation(a.lower())
                np_model = remove_punctuation(elt[f'model_{hvv}'].lower())  # todo remove <pad> and </s>

                token_truth = nltk.word_tokenize(np_truth)
                token_model = nltk.word_tokenize(np_model)
                if len(token_model) == 0:
                    continue

                intersection = set(token_truth).intersection(set(token_model))
                if len(intersection) == len(token_model):
                    perfect_matches.append(1)
    return len(perfect_matches)

def find_noisy_matches(data):
    noisy_matches = []
    for i in range(len(data)):
        elt = data[i]
        for hvv in ['hero','villain','victim']:
            if hvv not in elt or f'model_{hvv}' not in elt.keys():
                continue

            if elt[hvv]=='None':
                continue

            if isinstance(elt[f'model_{hvv}'],list):
                elt[f'model_{hvv}'] = " ".join(elt[f'model_{hvv}'])

            true_answers = elt[hvv].split(',')

            for a in true_answers:
                np_truth = remove_punctuation(a.lower())
                np_model = remove_punctuation(elt[f'model_{hvv}'].lower())  # todo remove <pad> and </s>

                token_truth = nltk.word_tokenize(np_truth)
                token_model = nltk.word_tokenize(np_model)

                intersection = set(token_truth).intersection(set(token_model))
                if len(intersection) == len(token_truth):
                    noisy_matches.append(1)
    return len(noisy_matches)

def find_partial_matches(data):
    partial = []
    for i in range(len(data)):
        elt = data[i]
        for hvv in ['hero','villain','victim']:
            if hvv not in elt or f'model_{hvv}' not in elt.keys():
                continue

            if elt[hvv]=='None':
                continue

            if isinstance(elt[f'model_{hvv}'],list):
                elt[f'model_{hvv}'] = " ".join(elt[f'model_{hvv}'])

            true_answers = elt[hvv].split(',')

            for a in true_answers:
                np_truth = remove_punctuation(a.lower())
                np_model = remove_punctuation(elt[f'model_{hvv}'].lower())  # todo remove <pad> and </s>

                token_truth = nltk.word_tokenize(np_truth)
                token_model = nltk.word_tokenize(np_model)

                intersection = set(token_truth).intersection(set(token_model))
                if len(intersection) > 0 and len(intersection)!= len(token_truth) and len(intersection)!=len(token_model):
                    partial.append(len(intersection)/len(token_truth))
    return partial

def calculate_false_negatives(data):
    fns = []
    for i in range(len(data)):
        elt = data[i]
        for hvv in ['hero', 'villain', 'victim']:
            if hvv not in elt or f'model_{hvv}' not in elt.keys():
                continue

            if elt[hvv] == 'None':
                continue


            np_model = remove_punctuation(elt[f'model_{hvv}'].lower())
            if np_model =='' or np_model ==' ' or ('none' in np_model and len(np_model) < 50):
                print(f"Match:{np_model}")
                fns.append(1)
            else:
                print(f'\nNo Match: {np_model}')
    return len(fns)

def calculate_false_positives(data):
    fps = []
    for i in range(len(data)):
        elt = data[i]
        for hvv in ['hero', 'villain', 'victim']:
            if hvv not in elt or f'model_{hvv}' not in elt.keys():
                continue

            if elt[hvv] != 'None':
                continue

            if isinstance(elt[f'model_{hvv}'],list):
                elt[f'model_{hvv}'] = " ".join(elt[f'model_{hvv}'])

            np_model = remove_punctuation(elt[f'model_{hvv}'].lower())
            if np_model == '' or np_model == ' ' or ('none' in np_model and len(np_model) < 50):
                print(f"Match:{np_model}")

            else:
                print(f'\nNo Match: {np_model}')
                fps.append(1)
    return len(fps)

def calculate_precision():
    summary = [['model', 'precision']]
    for file in FILES:
        data = sf.import_json(file)
        meta = data['metadata']
        title = f'{meta["model"]} {meta["prompt"]} {meta["splitter"]}'
        content = data['content']
        tp = calculate_true_positives(content)
        fn = calculate_false_negatives(content)
        fp = calculate_false_positives(content)
        summary.append([title,tp / (tp + fp)])
    df = pd.DataFrame(summary[1:], columns=summary[0])
    return df

def calculate_recall():
    summary = [['model', 'recall']]
    for file in FILES:
        data = sf.import_json(file)
        meta = data['metadata']
        title = f'{meta["model"]} {meta["prompt"]} {meta["splitter"]}'
        content = data['content']
        tp = calculate_true_positives(content)
        fn = calculate_false_negatives(content)
        summary.append([title, tp / (tp+fn)])
    df = pd.DataFrame(summary[1:],columns=summary[0])
    return df


def calculate_true_positive_overlap():
    summary = [['model', 'perfect_matches', 'noisy_match','partial_matches','partial_gt_20', 'partial_gt_40','partial_gt_60', 'partial_gt_80']]
    for file in FILES:
        data = sf.import_json(file)
        meta = data['metadata']
        title = f'{meta["model"]} {meta["prompt"]} {meta["splitter"]}'
        content = data['content']
        pm = find_perfect_matches(content)
        noisy_match = find_noisy_matches(content)
        partial_matches = find_partial_matches(content)
        gt_20 = len([x for x in partial_matches if x >0.2])
        gt_40 = len([x for x in partial_matches if x >0.4])
        gt_60 = len([x for x in partial_matches if x > 0.6])
        gt_80 = len([x for x in partial_matches if x > 0.8])
        summary.append([title, pm, noisy_match, len(partial_matches), gt_20, gt_40, gt_60, gt_80])
    df = pd.DataFrame(summary[1:], columns=summary[0])
    return df


def compare_splitters(metric:str):
    error_analysis = sf.import_csv('Full_Error_Analysis.csv')
    metric_index = [i for i in range(len(error_analysis[0])) if error_analysis[0][i]==metric][0]
    models = [x[1] for x in error_analysis[1:]]
    langchain = [x[metric_index] for x in error_analysis[1:] if 'langchain' in x[1]]
    truncated = [x[metric_index] for x in error_analysis[1:] if 'langchain' not in x[1]]
    model_names = sf.remove_duplicates([x.replace('langchain','').replace('truncated','') for x in models])
    df = pd.DataFrame([langchain,truncated],columns=model_names, index=['langchain','truncated'])
    return df

def compare_prompts(metric:str):
    error_analysis = sf.import_csv('Full_Error_Analysis.csv')
    metric_index = [i for i in range(len(error_analysis[0])) if error_analysis[0][i] == metric][0]
    models = [x[1] for x in error_analysis[1:]]
    a = [float(x[metric_index]) for x in error_analysis[1:] if 'A' in x[1]]
    b = [float(x[metric_index]) for x in error_analysis[1:] if 'A' not in x[1]]
    model_names = sf.remove_duplicates([x.replace('A', '').replace('B', '') for x in models])
    df = pd.DataFrame([a, b], columns=model_names, index=['A', 'B'])
    return df

def compare_models(metric:str):
    error_analysis = sf.import_csv('Full_Error_Analysis.csv')
    metric_index = [i for i in range(len(error_analysis[0])) if error_analysis[0][i] == metric][0]
    models = [x[1] for x in error_analysis[1:]]
    flan_base = [x[metric_index] for x in error_analysis[1:] if 'google/flan-t5-base' in x[1]]
    flan_large = [x[metric_index] for x in error_analysis[1:] if 'google/flan-t5-large' in x[1]]
    vicuna = [x[metric_index] for x in error_analysis[1:] if 'vicuna-13b-v1.3' in x[1]]
    model_names = sf.remove_duplicates([x.replace('google/flan-t5-base', '').replace('google/flan-t5-large', '').replace('vicuna-13b-v1.3','') for x in models])
    df = pd.DataFrame([flan_base,flan_large, vicuna], columns=model_names, index=['FlanBase', 'FlanLarge','Vicuna'])
    return df

time_df = calculate_time_cost()
p_df = calculate_precision()
r_df = calculate_recall()
tp_df = calculate_true_positive_overlap()

all_models = pd.merge(time_df[['model','average','total_time']],  p_df, on='model')
all_models = pd.merge(all_models, r_df, on='model')
all_models = pd.merge(all_models, tp_df, on='model')
all_models.to_csv('Full_Error_Analysis.csv')


# for m in ['total_time','perfect_matches','noisy_match','partial_matches','precision','recall']:
#     split_df = compare_splitters(m)
#     split_df.to_csv(f'tables/{m}_splitters.csv')
#     prompt_df = compare_prompts(m)
#     prompt_df.to_csv(f'tables/{m}_prompts.csv')
#     models_df = compare_models(m)
#     models_df.to_csv(f'tables/{m}_models.csv')