from explore_gen_models import humans_in_social_science_pipline as h
import json
import time 


def import_json(file):
    with open(file, 'r') as j:
        content = json.loads(j.read())
    return content

def prep_data():
    annotations = import_json('/home/kmadole/model_pipeline/explore_gen_models/Data_for_annotation_v4.json')['content']
    return annotations

def generate_prompt(hvv:str, data:dict, prompt_type:str, split_method:str):
    if prompt_type == 'A':
        prompt = "Output a response given the Output rules and Article.\nOutput Rules: Identify if" \
                " there is one, multiple, or zero <elt>s in the article.\nIf the number of <elt>s == 0, then output " \
                "'None'.\nIf the number of <elt>s > 0, then output the names of the <elt>s as a python list.\n" \
                "Article: <article_text>"
    elif prompt_type == 'B':
        prompt =f"Who is the <elt> in the following text?\n<article_text>"
    template = prompt.replace('<elt>',hvv).split('<article')[0]
    w_data = prompt.replace('<article_text>', data['article_text']).replace('<elt>',hvv)
    return {'prompt': w_data,'prompt_type':prompt_type,'template':template, 'split_method':split_method}

def export_as_json(export_filename:str, output):
    with open(export_filename, "w") as outfile:
        outfile.write(json.dumps(output))


def run_model(MODEL_NAME,PROMPT_TYPE,SPLITTER_TYPE):
    model = h.TFG(model_name=MODEL_NAME, connect_to_gpu=True, memory_saver=True)

    data = prep_data()

    for elt in data:
        duration  = {k:None for k in ['hero', 'villain','victim'] }
        for obj in ['hero', 'villain','victim']:
            prompt = generate_prompt(obj, elt, PROMPT_TYPE, SPLITTER_TYPE) # TODO: how to handle token max exceeded?

            a = time.time()
            response = model.generate(prompt)
            b = time.time()

            duration[obj] = b-a

            elt[f'model_{obj}'] = response[0]
            elt[f"model_{obj}_score"] = response[1]
            elt[f"prompt_length_exceeded"] = response[2]

        elt['duration'] = duration
        
  
    export_as_json(f"/home/kmadole/model_pipeline/output/{MODEL_NAME.replace('/','_')}_{PROMPT_TYPE}_{SPLITTER_TYPE}_annotations.json",
                    {'content':data,
                     'metadata':{'model':MODEL_NAME,'prompt':PROMPT_TYPE,'splitter':SPLITTER_TYPE}})
    

if __name__ == '__main__':
    models = ["lmsys/vicuna-7b-v1.5","meta-llama/Llama-2-7b-hf",'google/flan-t5-base','lmsys/vicuna-13b-v1.3','google/flan-t5-large','lmsys/vicuna-7b-delta-v0']
    
    prompts = ['A','B']
    splitters = ['truncated','langchain'] # options: langchain, truncated
    for m in models:
    #m = models[1]
        for p in prompts:
            for s in splitters:
                print(f"Running {m}, {p}, {s}")
                run_model(MODEL_NAME=m,PROMPT_TYPE=p,SPLITTER_TYPE=s)