#DA0 - Data Access Object
# Access - CRUD(Create, Red, Update, Delete)

from pymongo import MongoClient

# MongoDB Connection
def conn_momgo():
    client = MongoClient('localhost', 27017)
    db = client['local']
    collection = db.get_collection('movie')
    return collection


# Create review data(데이터 등록)
def add_review(data):
    collection = conn_momgo()    # MongoDB Connection
    collection.insert_one(data)  # Data save


# Select review data(데이터 조회)
def get_reviews():
    pass