# ===========================================================================
#                            Upload Results to DB
# ===========================================================================
from utils.preprocessing import *
from utils.database import *
from datasets import load_from_disk
from tqdm import tqdm

dataset = load_from_disk('data/output/articles_processed')

# Sort
dataset = dataset.map(lambda row: {'sort_key':
                                   str(row['id']) + str(row['role']) +
                                   str(row['chunk'])
                                   })
sorted_dataset = dataset.sort('sort_key')
sorted_dataset = sorted_dataset.remove_columns('sort_key')

# ------------------- Connect to Database  -------------------

_, db = getConnection(use_dotenv=True)


# ------------------- Update Database  -------------------
def process_dataset(dataset):
    # Initial processing results
    processing_result = {"hero": [], "villain": [], "victim": []}
    object_id_prev = None

    for item in dataset:
        object_id = item['id']
        role = item['role']
        answer = item['Answers']

        # If the object_id changes, reset the processing_result
        if object_id_prev is not None and object_id_prev != object_id:
            yield object_id_prev, processing_result
            processing_result = {"hero": [], "villain": [], "victim": []}

        processing_result[role].append(answer)
        object_id_prev = object_id

    # Yield the final processing_result if any
    if processing_result["hero"] or processing_result["villain"] or processing_result["victim"]:
        yield object_id_prev, processing_result


# Assuming `ds` is your dataset object
unique_ids = set(dataset["id"])

# Count of unique ids
count_unique_ids = len(unique_ids)
# print(count_unique_ids)

for object_id, result in tqdm(process_dataset(dataset), total=count_unique_ids, desc="Uploading results"):
    updateProcessingResults(db, object_id, {"processing_result": result})
