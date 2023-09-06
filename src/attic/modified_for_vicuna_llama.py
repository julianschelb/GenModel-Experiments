import argparse
import time
import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer

from fastchat.conversation import conv_templates, SeparatorStyle

torch.cuda.empty_cache()

def import_json(file):
    with open(file, 'r') as j:
        content = json.loads(j.read())
    return content

def get_data():
    annotations = import_json('/home/kmadole/model_pipeline/explore_gen_models/Data_for_annotation_v4.json')['content']
    return annotations

def export_as_json(export_filename:str, output):
    with open(export_filename, "w") as outfile:
        outfile.write(json.dumps(output))


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

def load_model(model_name, device, num_gpus, load_8bit=False):
    if device == "cpu":
        kwargs = {}
    elif device == "cuda":
        kwargs = {"torch_dtype": torch.float16}
        if load_8bit:
            if num_gpus != "auto" and int(num_gpus) != 1:
                print("8-bit weights are not supported on multiple GPUs. Revert to use one GPU.")
            kwargs.update({"load_in_8bit": True, "device_map": "auto"})
        else:
            if num_gpus == "auto":
                kwargs["device_map"] = "auto"
            else:
                num_gpus = int(num_gpus)
                if num_gpus != 1:
                    kwargs.update({
                        "device_map": "auto",
                        "max_memory": {i: "13GiB" for i in range(num_gpus)},
                    })
    #elif device == "mps":
    #    # Avoid bugs in mps backend by not using in-place operations.
    #    kwargs = {"torch_dtype": torch.float16}
    #    replace_llama_attn_with_non_inplace_operations()
    else:
        raise ValueError(f"Invalid device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_name,
        low_cpu_mem_usage=True, **kwargs)

    # calling model.cuda() mess up weights if loading 8-bit weights
    if device == "cuda" and num_gpus == 1 and not load_8bit:
        model.to("cuda")
    elif device == "mps":
        model.to("mps")

    return model, tokenizer

@torch.inference_mode()
def generate_stream(tokenizer, model, params, device,
                    context_len=2048, stream_interval=2):
    """Adapted from fastchat/serve/model_worker.py::generate_stream"""

    prompt = params["prompt"]['prompt']
    l_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.0))
    max_new_tokens = int(params.get("max_new_tokens", 256))
    stop_str = params.get("stop", None)

    input_ids = tokenizer(prompt).input_ids
    output_ids = list(input_ids)

    max_src_len = context_len - max_new_tokens - 8
    input_ids = input_ids[-max_src_len:]

    for i in range(max_new_tokens):
        if i == 0:
            out = model(
                torch.as_tensor([input_ids], device=device), use_cache=True)
            logits = out.logits
            past_key_values = out.past_key_values
        else:
            attention_mask = torch.ones(
                1, past_key_values[0][0].shape[-2] + 1, device=device)
            out = model(input_ids=torch.as_tensor([[token]], device=device),
                        use_cache=True,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values)
            logits = out.logits
            past_key_values = out.past_key_values

        last_token_logits = logits[0][-1]

        if device == "mps":
            # Switch to CPU by avoiding some bugs in mps backend.
            last_token_logits = last_token_logits.float().to("cpu")

        if temperature < 1e-4:
            token = int(torch.argmax(last_token_logits))
        else:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))

        output_ids.append(token)

        if token == tokenizer.eos_token_id:
            stopped = True
        else:
            stopped = False

        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            output = tokenizer.decode(output_ids, skip_special_tokens=True)
            pos = output.rfind(stop_str, l_prompt)
            if pos != -1:
                output = output[:pos]
                stopped = True
            yield output

        if stopped:
            break

    del past_key_values

def check_prompt_length(tokenizer, prompt):
    prompt_size = len(tokenizer(prompt).input_ids)
    if prompt_size > tokenizer.model_max_length:
        return True
    else:
        return False


def run_model(model_name,prompt_type,splitter_type):
    data = get_data()
 
    args = dict(
    model_name=model_name,
    device='cuda',
    num_gpus='1',
    load_8bit=True,
    conv_template='vicuna_v1.1' if 'vicuna' in model_name else 'llama-2',
    temperature=0.7,
    max_new_tokens=512,
    debug=False)

    args = argparse.Namespace(**args)

    conv = conv_templates[args.conv_template].copy()

    model, tokenizer = load_model(args.model_name, args.device,args.num_gpus, args.load_8bit)
    
    for i in range(len(data)):
        duration  = {k:None for k in ['hero', 'villain','victim'] }
        for hvv in ['hero','villain','victim']:
            prompt = generate_prompt(hvv, data[i], prompt_type, splitter_type)
            data[i]['prompt_length_exceeded'] = check_prompt_length(prompt=prompt['prompt'], tokenizer=tokenizer)
            # Chat
            
            params = {
            "model": model_name,
            "prompt": prompt,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens, 
            "stop": conv.sep if conv.sep_style == SeparatorStyle.ADD_COLON_SINGLE else conv.sep2}
            print('STOP',params['stop'])
            # generate
            #output = generate_stream(tokenizer, model, params, args.device)
            #print(f"{conv.roles[1]}: ", end="", flush=True)
            a = time.time()
            pre = 0
            for outputs in generate_stream(tokenizer, model, params, args.device):
                outputs = outputs[len(prompt['prompt']) + 1:].strip()
                outputs = outputs.split(" ")
                now = len(outputs)
                if now - 1 > pre:
                    #print(" ".join(outputs[pre:now-1]), end=" ", flush=True)
                    pre = now - 1
            #print(" ".join(outputs[pre:]), flush=True)
            b = time.time()
            data[i][f"model_{hvv}"] = " ".join(outputs[pre:])
            duration[hvv] = b-a
        data[i]['duration'] = duration
        
    export_as_json(f"/home/kmadole/model_pipeline/output/{model_name.replace('/','_')}_{prompt_type}_{splitter_type}_annotations.json",
                    {'content':data,
                     'metadata':{'model':model_name,'prompt':prompt_type,'splitter':splitter_type}})

MODEL_NAME="meta-llama/Llama-2-7b-hf"#'lmsys/vicuna-13b-v1.3'
PROMPT_TYPE='B'
SPLITTER_METHOD='truncated'
response = run_model(model_name=MODEL_NAME, prompt_type=PROMPT_TYPE, splitter_type=SPLITTER_METHOD)
