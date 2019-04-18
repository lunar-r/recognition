# JSON
# import json
#
# dict = {
#     "pre_train_flag": True,
#     "pre_known_flag": False
# }
#
# json_str = json.dumps(dict)
# print(json_str)
#
# with open("status.json", "w") as f:
#     json.dump(dict, f)
#     print("write finish")
#
# with open("status.json", "r") as f:
#     load_dict = json.load(f)
#     res = load_dict['pre_train_flag']
#     print(res)
#     print(res == "True")
#     print(res is True)

#
# string = "E:/QQ/youtube/myself.jpg"
# sub = string.rfind("/")
# print(string[(sub + 1):])


def abot(a=None, b=None, c=None):
    print(a)
    print(b)
    print(c)


a = "you"
abot()
abot(a)
abot(a, a, a)
