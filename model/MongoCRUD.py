# MongoDB Access and CRUD test

from pymongo import MongoClient

# conda install -c anaconda pymongo

# 1. MongoDB Connection
client = MongoClient('localhost', 27017)    # (IP address, Port number)
db = client['local']                        # Allocating 'local' DB
collection = db.get_collection('test')      # Allocating 'review' Collection

data = {'name': 'cherry', 'age': 8}
collection.insert_one(data)

# MongoDB > database > collection > document
# 우리은행 > 우리은행 광주지점 > 예금 > 50,000 입금

# CRUD - Create, Read, Update, Delete