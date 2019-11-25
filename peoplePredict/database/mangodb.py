import time
from pymongo import MongoClient

username = "qh_rw"
password = "BEPQf67s"
host = "dds-2zefd5cd204049441717-pub.mongodb.rds.aliyuncs.com/qh_area_forecast"
port = 3717

uri = "mongodb://{}:{}@{}".format(username, password, host)

conn = MongoClient(uri, port=port)
db = conn.qh_area_forecast
people_num_set = db.hist_loc_unum
poi_set = db.XiAn_poi

count = 0
people_l = []
poi_l = []

for i in people_num_set.find():
    people_l.append(i)
    count += 1
    # if poi is not None:
    #     print(poi)
    if count % 1000 == 0:
        print(count)

count = 0
for i in poi_set.find():
    # print(i)
    # print(type(i['_id']))
    # print(i['_id'])
    # poi = poi_set.find_one({'_id': i['_id']})
    poi_l.append(i)
    count += 1
    # if poi is not None:
    #     print(poi)
    if count % 1000 == 0:
        print(count)

time.sleep(10000)
