import re


def cleanText(text):
    """Replace line breaks with a spaceoc currences of more than one space"""
    text = text.replace('\n', ' ').replace('\r', ' ')
    return re.sub(' +', ' ', text)


def describeDataset(dataset):
    """
    Print basic descriptive information about the given dataset.
    """

    # Basic information
    print("Number of rows:", len(dataset))
    print("Column names:", dataset.column_names)
    print("Features (schema):", dataset.features)

    # Check if dataset has a 'set' column (e.g., 'train', 'validation', 'test')
    if "set" in dataset.column_names:
        set_counts = dataset["set"].value_counts()
        for set_name, count in set_counts.items():
            print(f"Number of samples in {set_name}: {count}")