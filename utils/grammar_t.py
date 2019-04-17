import json

dict = {
    "pre_train_flag": True,
    "pre_known_flag": False
}

json_str = json.dumps(dict)
print(json_str)

with open("status.json", "w") as f:
    json.dump(dict, f)
    print("write finish")

with open("status.json", "r") as f:
    load_dict = json.load(f)
    res = load_dict['pre_train_flag']
    print(res)
    print(res == "True")
    print(res is True)
