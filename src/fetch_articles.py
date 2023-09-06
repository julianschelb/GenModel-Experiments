# ===========================================================================
#                            Fetch and Prepare Dataset
# ===========================================================================

from utils.database import *
from utils.files import *
from utils.preprocessing import *
from datasets import Dataset
# import pickle


# ------------------- Connect to Database  -------------------

_, db = getConnection(use_dotenv=True)

# ------------------- Fetch Articles -------------------

# Fetch articles from database
fields = {"url": 1, "title": 1, "parsing_result.text": 1}
articles = fetchArticleTexts(db,  limit=100_000, skip=0, fields=fields, query={
                             "processing_result": {"$exists": False}})

# Clean text
for article in articles:
    text = article.get("parsing_result", {}).get("text", "")
    if "parsing_result" in article:
        article["parsing_result"]["text"] = cleanText(text)
    else:
        print("No parsing result for article", article["_id"])

articles = [article for article in articles if article.get(
    "title", "") and article.get("parsing_result", "")]

print("Number of articles:", len(articles))

# ------------------- Export as JSON  -------------------

exportAsJSON("data/input/articles.json",  articles)

# ------------------- Dataset Object -------------------

# Iterate through each article in the list of articles
# and convert the "_id" field of the article to a string
for article in articles:
    article["_id"] = str(article["_id"])

# Convert the list of JSON objects into a dictionary format
dataset_dict = {
    "id": [_["_id"] for _ in articles],
    "title": [_["title"] for _ in articles],
    "url": [_["url"] for _ in articles],
    "text": [_["parsing_result"]["text"] for _ in articles],
}

dataset = Dataset.from_dict(dataset_dict)

# ------------------- Export as Pickle  -------------------

# Save dataset as a pickle file
# with open("data/input/articles.pkl", "wb") as file:
#    pickle.dump(dataset, file)

dataset.save_to_disk('data/input/articles')

# Print the first row
print(dataset[0]["text"][:100])
