import json
import glob
import csv

def import_json(file):
    with open(file, 'r') as j:
        content = json.loads(j.read())
    return content


def export_as_json(export_filename:str, output):
    if export_filename.endswith('.json') is not True:
        raise Exception(f"{export_filename} should be a .json file")

    with open(export_filename, "w") as outfile:
        outfile.write( json.dumps(output))


def import_csv(csv_file:str):
    # Given a file location, imports the data as a nested list, each row is a new list

    nested_list = []  # initialize list
    with open(csv_file, newline='', encoding='utf-8') as csvfile:  # open csv file
        reader = csv.reader(csvfile, delimiter=',')

        for row in reader:
            nested_list.append(row)  # add each row of the csv file to a list

    return nested_list

def get_files_from_folder(folder_name:str, file_endings:str)->list:
    return [x for x in glob.glob(folder_name + f"/*.{file_endings}")]

def remove_duplicates(data:list)->list:
    new_list  =[]
    for elt in data:
        if elt not in new_list:
            new_list.append(elt)
    return new_list