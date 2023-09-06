# ===========================================================================
#                            Run  Experiments
# ===========================================================================

# from utils.model import *
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import torch
import pickle

# Load the dataset from the pickle file
with open("data/input/articles_tokenized.pkl", 'rb') as file:
    dataset = pickle.load(file)


# ------------------- Load Model -------------------


tokenizer = AutoTokenizer.from_pretrained(
    "facebook/genre-kilt", add_prefix_space=True)
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/genre-kilt").eval()

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"There are {torch.cuda.device_count()} GPU(s) available.")
    print("Device name:", torch.cuda.get_device_name(0))

# Otherwise use the CPU
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")

model.to(device)
print(f"Model loaded: {model.config}")


def select_columns(example):
    return {
        "input_ids": example["input_ids"],
        "attention_mask": example["attention_mask"]
    }


desired_dataset = dataset.map(select_columns, remove_columns=[
                              col for col in dataset.column_names if col not in ["input_ids", "attention_mask"]])


# Format dataset to PyTorch
desired_dataset.set_format(type='torch', columns=[
                           'input_ids', 'attention_mask'])


BATCH_SIZE = 128

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Create dataloader without explicit sampler for sequential loading
dataloader = DataLoader(desired_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ------------------- Make Prediction -------------------
outputs = []
with torch.no_grad():
    for batch in tqdm(dataloader, desc="Batches"):

        test = batch.items()

        # Your prediction code here
        # Ensure your batch data is sent to the same device as your model
        batch = {k: v.to(device) for k, v in batch.items()}

        # Generate outputs
        outputs = model.generate(
            input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], max_new_tokens=100)

        # input_decoded = tokenizer.decode(
        #     batch['input_ids'][0], skip_special_tokens=True)
        # print(input_decoded)

        # Decode and store predictions
        decoded_outputs = [tokenizer.decode(
            output_id, skip_special_tokens=True) for output_id in outputs]

        outputs.extend(decoded_outputs)


exit()

# Create dataloader
BATCH_SIZE = 12
sampler = SequentialSampler(desired_dataset)
dataloader = DataLoader(
    desired_dataset, sampler=sampler, batch_size=BATCH_SIZE)

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-base", device_map="auto")
model.eval()

# ------------------- Make Prediction -------------------

candidates = []
for batch in tqdm(dataloader, desc="Batches"):

    # batch = [r.to(device) for r in batch]  # Push batch to GPU
    input_ids, mask = batch  # Extract id, attention mask, and labels from batch

    outputs = model.generate(input_ids)
    print(tokenizer.decode(outputs[0]))

exit()


model = ModelForTripletExtraction.from_pretrained(
    "google/flan-t5-base",  load_in_8bit=False)
model.eval()


print(type(model))


response = model.generateFromText(
    "translate English to German: How old are you?")
print(response)


tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-base", device_map="auto")
model.eval()

input_text = "translate English to German: How old are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))


def import_json(file):
    with open(file, 'r') as j:
        content = json.loads(j.read())
    return content


def generate_prompt(hvv: str, data: dict, prompt_type: str, split_method: str):
    if prompt_type == 'A':
        prompt = "Output a response given the Output rules and Article.\nOutput Rules: Identify if" \
                 " there is one, multiple, or zero <elt>s in the article.\nIf the number of <elt>s == 0, then output " \
                 "'None'.\nIf the number of <elt>s > 0, then output the names of the <elt>s as a python list.\n" \
                 "Article: <article_text>"
    elif prompt_type == 'B':
        prompt = f"Who is the <elt> in the following text?\n<article_text>"

    template = prompt.replace('<elt>', hvv).split('<article')[0]
    w_data = prompt.replace(
        '<article_text>', data['article_text']).replace('<elt>', hvv)
    return {'prompt': w_data, 'prompt_type': prompt_type, 'template': template, 'split_method':  split_method}


def export_as_json(export_filename: str, output):
    with open(export_filename, "w") as outfile:
        outfile.write(json.dumps(output))


def prep_model(MODEL_NAME='google/flan-t5-base'):
    return h.TFG(model_name=MODEL_NAME, connect_to_gpu=True, memory_saver=True)


def run_model(data, model, MODEL_NAME='google/flan-t5-base', PROMPT_TYPE='B', SPLITTER_TYPE='langchain'):
    annotations = []
    for i in range(len(data)):
        elt = data[i]
        duration = {k: None for k in ['hero', 'villain', 'victim']}
        for obj in ['hero', 'villain', 'victim']:
            prompt = generate_prompt(obj, elt, PROMPT_TYPE, SPLITTER_TYPE)

            a = time.time()
            response = model.generate(prompt)
            b = time.time()

            duration[obj] = b - a

            elt[f'model_{obj}'] = response[0]
            elt[f"model_{obj}_score"] = response[1]
            elt[f"prompt_length_exceeded"] = response[2]

        elt['duration'] = duration
        annotations.append(elt)

    return annotations


if __name__ == '__main__':
    MODEL_NAME = 'google/flan-t5-base'
    PROMPT_TYPE = 'B'
    SPLITTER_TYPE = 'langchain'

    sample_size = 50

    model = prep_model(MODEL_NAME)

    while True:
        a = time.time()
        data, start, end = f.fetch_data(sample_size)

        annotations = run_model(model=model, data=data)
        b = time.time()

        # export_as_json(
        #     f"/home/kmadole/model_pipeline/true_output/{start}_{end}_annotations.json",
        #     {'content': data,
        #      'metadata': {'model': MODEL_NAME, 'prompt': PROMPT_TYPE, 'splitter': SPLITTER_TYPE,
        #                   'start_date':start, 'end_date':end}})

        print(f"Exporting {len(annotations)}, took {(b-a)/60} minutes")
