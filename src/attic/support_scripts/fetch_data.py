import datetime
import gridfs
from dotenv import load_dotenv
import os
import pymongo as pm
import shared_functions as sf
from bson import ObjectId
from run_models.support_scripts.process_html import extract_text as e

def getConnection(
    connection_string: str = "", database_name: str = "", use_dotenv: bool = False
):
    "Returns MongoDB and GridFS connection"

    # Load config from config file
    if use_dotenv:
        load_dotenv()
        connection_string = os.getenv("CONNECTION_STRING")
        database_name = os.getenv("DATABASE_NAME")

    # Use connection string
    conn = pm.MongoClient(connection_string)
    db = conn[database_name]
    fs = gridfs.GridFS(db)

    return fs, db


def getPageContent(fs: gridfs, id: str, encoding="UTF-8"):
    """Retrieves a file from GridFS"""
    try:
        f = fs.get(ObjectId(id))
        content = f.read().decode(encoding)
        return content
    except gridfs.errors.NoFile:
        return None
    except UnicodeDecodeError:
        return None

def reformat_element(data, text):
    return  {'article_id': data['article_id'],
               'file_id':data['_id'],
               'text':text,
               'file_uploadDate':data['uploadDate']}

def fetch_data(n):
    fs, db = getConnection()
    start_date = datetime.datetime(year=2023,month=8,day=1)
    # start_date = datetime.datetime.fromtimestamp(sf.import_json('last_logged_date.json')['last_logged_date'])
    c = db['fs.files'].find({'uploadDate': {"$gt": start_date}})  # find documents uploaded since last_date
    c = c.sort("uploadDate", 1)  # sort these from oldest to newest
    c = list(c.limit(n))  # fetch the top n
    last_logged_date = c[0]['uploadDate']

    data = []
    for elt in c:
        html = getPageContent(fs, id=elt['_id'])
        if html is None:
            continue

        text, error = e.extractText(url=elt['target_url'], response=html)
        if error != '':
            continue
        data.append(reformat_element(elt, text))

    return data, datetime.datetime.timestamp(start_date), datetime.datetime.timestamp(last_logged_date)