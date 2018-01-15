import json

with open("data/timit/dev.json", 'r') as fid:
    data = [json.loads(l) for l in fid]

with open("data/timit/train.json", 'r') as fid:
    train_data = [json.loads(l) for l in fid]

# Add the training data we didn't use for more eval data
train_data = filter(lambda d: d['duration'] > 3, train_data)

data = data+train_data

ranges = [(0,3), (3,4), (4,5), (5,float("inf"))]

for mint, maxt in ranges:
    filtd = filter(lambda d: mint <=  d['duration'] < maxt, data)[:300]
    save_file = "d{}-{}.json".format(mint, maxt)
    with open(save_file, 'w') as fid:
        for d in filtd:
            json.dump(d, fid)
            fid.write("\n")
