# ===========================================================================
#                            Fetch and Prepare Dataset
# ===========================================================================

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from transformers import AutoTokenizer
from multiprocessing import Pool
from utils.database import *
from utils.files import *
from utils.preprocessing import *
from datasets import Dataset, load_from_disk
import transformers

# import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
transformers.utils.logging.set_verbosity_error()

# ------------------- Import Raw Dataset  -------------------

# # Load the dataset from the pickle file
# with open("data/input/articles.pkl", 'rb') as file:
#     dataset = pickle.load(file)

dataset = load_from_disk('data/input/articles')

describeDataset(dataset)


# ------------------- Prompt Template  -------------------

# PROMPT_TEMPLATE = "Output a response given the Output rules and Article.\nOutput Rules: Identify if" \
#     " there is one, multiple, or zero {elt}s in the article.\nIf the number of {elt}s == 0, then output " \
#     "'None'.\nIf the number of {elt}s > 0, then output the names of the {elt}s as a python list.\n" \
#     "Article: {article_text}"

PROMPT_TEMPLATE = "Who is the {elt} in the following text?\nText: {article_text}"

# Test the template with a dummy text
print(PROMPT_TEMPLATE.format(elt='hero',
      article_text='Lorem ipsum dolor sit amet, consectetur adipiscing elit.'))


# ------------------- Expand Dataset  -------------------

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")


def calcInputLength(prompt):
    """Calculate the length of the input after"""
    return tokenizer(prompt, return_tensors="pt").input_ids.real.shape[1]


template_length = calcInputLength(
    PROMPT_TEMPLATE.format(elt='villain', article_text=' '))
print(template_length)

chunk_size = tokenizer.model_max_length - template_length
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=30,
    # separators=['.', '?', '!', "\n\n", "\n", " ", ""],
    length_function=calcInputLength)


def split_text(text, n_tokens, tokenizer, overlap=10):
    """Splits the input text into chunks with n_tokens tokens using HuggingFace tokenizer, with an overlap of overlap tokens."""
    tokens = tokenizer.tokenize(text)
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i+n_tokens]
        chunks.append(tokenizer.convert_tokens_to_string(chunk))
        i += n_tokens - overlap

    return chunks


def expandRow(row):
    """
    Generate prompts based on various roles and text chunks from the input row.
    """
    roles = ['hero', 'villain', 'victim']
    prompts = []

    # Split the text into chunks
    # text_splitter.split_text(row.get('text'))
    text_chunks = split_text(row.get('text'), 450, tokenizer, overlap=10)

    # Generate prompts for each role and text chunk
    for role in roles:
        for chunk_id, text_chunk in enumerate(text_chunks):
            prompt = PROMPT_TEMPLATE.format(elt=role, article_text=text_chunk)
            new_row = {
                **row,
                'prompt': prompt,
                'role': role,
                'chunk': chunk_id,
                'chunk_length': calcInputLength(text_chunk)
            }
            prompts.append(new_row)

    return prompts


if __name__ == "__main__":
    # Define the number of processes you want to spawn.
    # Usually, you'd use the number of cores in your machine.
    num_processes = 16

    with Pool(processes=num_processes) as pool:
        # The pool.map function applies the expandRow function to each row in dataset
        # and returns a list of results. Each result is a list, so we flatten the list using itertools.chain.
        dataset_hvv = list(pool.map(expandRow, dataset))

    # Flatten the resulting list of lists
    dataset_hvv = [item for sublist in dataset_hvv for item in sublist]

    # Convert the list of dictionaries into a Dataset
    dataset_hvv = Dataset.from_dict(
        {key: [dic[key] for dic in dataset_hvv] for key in dataset_hvv[0]})

    # Save dataset as a pickle file
    # with open("data/input/articles_chunkified.pkl", "wb") as file:
    #    pickle.dump(dataset_hvv, file)

    dataset_hvv.save_to_disk('data/input/articles_chunkified')

    # ------------------- Tokenize -------------------

    def tokenizeInputs(example):
        """Tokenize the inputs"""

        tokenized_inputs = tokenizer(example["prompt"], max_length=tokenizer.model_max_length,
                                     truncation=True, is_split_into_words=False, add_special_tokens=True, padding="max_length")

        # Combine original data with the tokenized inputs
        example.update(tokenized_inputs)
        return example

    tokenized_dataset = dataset_hvv.map(tokenizeInputs)

    def calculate_prompt_length(row):
        row['prompt_length'] = calcInputLength(row['prompt'])
        return row

    # Assuming the dataset object supports a map operation
    tokenized_dataset = tokenized_dataset.map(calculate_prompt_length)

    # Assuming the dataset object can be iterated like a list
    min_length = min(row['prompt_length'] for row in tokenized_dataset)
    max_length = max(row['prompt_length'] for row in tokenized_dataset)
    total_length = sum(row['prompt_length'] for row in tokenized_dataset)
    avg_length = total_length / len(tokenized_dataset)

    print("Minimum prompt length:", min_length)
    print("Maximum prompt length:", max_length)
    print("Average prompt length:", avg_length)

    print(tokenized_dataset[0]["prompt"])

    # Save dataset as a pickle file
    # with open("data/input/articles_tokenized.pkl", "wb") as file:
    #    pickle.dump(tokenized_dataset, file)

    tokenized_dataset.save_to_disk('data/input/articles_tokenized')
