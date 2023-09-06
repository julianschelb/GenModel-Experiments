# ===========================================================================
#                            Run  Experiments
# ===========================================================================

# from utils.model import *
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import torch
import pickle
import threading
from datasets import Dataset, load_from_disk, concatenate_datasets
import time
import copy


def split_dataset(dataset: Dataset, split_ratio=0.5) -> (Dataset, Dataset):
    """Splits a dataset into two parts."""
    if not (0 <= split_ratio <= 1):
        raise ValueError("split_ratio should be between 0 and 1.")

    split_point = int(len(dataset) * split_ratio)

    # Split the dataset into two parts
    first_half = Dataset.from_dict(dataset[:split_point])
    second_half = Dataset.from_dict(dataset[split_point:])

    return first_half, second_half


def generate_predictions_from_dataset(id, dataset,  device):
    # Load tokenizer and model for generation
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    model.eval()
    model.to(device)

    print("Device:", torch.cuda.get_device_name())

    dataset_full = copy.copy(dataset)
    dataset.set_format(type='torch', columns=[
        'input_ids', 'attention_mask'])

    # Create dataloader without explicit sampler for sequential loading
    BATCH_SIZE = 192
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False)

    params = {'do_sample': True,
              'early_stopping': False,
              # 'max_length': 100,
              # 'min_length': 1,
              # 'num_beam_groups': 2,
              # 'num_beams': 2,
              # 'max_tokens': 32,
              # 'min_tokens': 1,
              # 'output_scores': False,
              'repetition_penalty': 1.0,
              # 'return_dict_in_generate': False,
              'temperature': 1.0,
              'top_k': 50,
              'top_p': 1.0, }

    # Make predictions
    predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Batches"):
            batch = {k: v.to(device) for k, v in batch.items()}

            # Generate outputs
            batch_outputs = model.generate(
                input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], **params, max_new_tokens=100)

            # Decode and store predictions
            decoded_outputs = [tokenizer.decode(
                output_id, skip_special_tokens=True) for output_id in batch_outputs]
            predictions.extend(decoded_outputs)

    # results.extend(predictions)

    # Ensure the new column has the same number of items as the dataset
    assert len(dataset_full) == len(
        predictions), "The length of new_column_values must match the dataset's length"

    # Add new column
    dataset_full = dataset_full.add_column('Answers', predictions)
    dataset_full.save_to_disk('data/output/articles_processed_' + str(id))
    # with open("data/output/articles_processed.pkl" + device, "wb") as file:
    #     pickle.dump(dataset_full, file)

    return predictions


# def generate_predictions_from_device(dataset, device):
#     """A wrapper around generate_predictions_from_dataset to specify device."""
#     # torch.cuda.set_device(device)  # Set the current device
#     return generate_predictions_from_dataset(dataset,  device)


if __name__ == "__main__":

    # # Load the dataset from the pickle file
    # with open("data/input/articles_tokenized.pkl", 'rb') as file:
    #     dataset = pickle.load(file)

    dataset = load_from_disk('data/input/articles_tokenized')
    print("Dataset length:", len(dataset))

    # Split the dataset into two halves
    dataset1, dataset2 = split_dataset(dataset)

    # predictions = generate_predictions_from_dataset(dataset)
    # print(predictions)

    print("Dataset length:", len(dataset))

    # Use threading to process each half on a different GPU
    thread1 = threading.Thread(
        target=generate_predictions_from_dataset, args=(0, dataset1, 'cuda:0'))
    thread2 = threading.Thread(
        target=generate_predictions_from_dataset, args=(1, dataset2, 'cuda:1'))

    # Start threads
    thread1.start()
    thread2.start()

    # Wait for threads to finish
    thread1.join()
    thread2.join()

    results_1 = load_from_disk('data/output/articles_processed_0')
    results_2 = load_from_disk('data/output/articles_processed_1')
    merged_dataset = concatenate_datasets([results_1, results_2])
    merged_dataset.save_to_disk('data/output/articles_processed')

    print("Processing on both GPUs completed!")
    print("Results:", len(merged_dataset))
