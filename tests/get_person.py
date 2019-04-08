import requests
import re
import os

headers = {
    'user-agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36'
    }
url = "http://www.uimaker.com/uimakerhtml/uidesign/uibs/2018/0928/131203.html"


response = requests.get(url, headers=headers)
html = None
if response.status_code == 200:
    html = response.text

# 1. Re
name = "person"
title = "..\source\images\\" + name
if not os.path.exists(title):
    os.mkdir(title)

# google img
pattern = re.compile('<img .*?src="(.*?)".*?jsaction', re.S)
items = re.findall(pattern, html)
count = 0
for i, item in enumerate(items):
    if count > 20:
        break
    url = item
    if url[:4] != "http":
        continue
    response = requests.get(url, headers=headers)
    file_path = "{0}\{1}_{2}.{3}".format(title, name, i + 1000, 'jpg')

    with open(file_path, 'wb') as f:
        f.write(response.content)
        print("picture " + str(i) + " is finished.")
        count = count + 1

# other
# pattern = re.compile('<img src="(.*?)".*?style', re.S)
#
# items = re.findall(pattern, html)
# # print(items)
#
# for i, item in enumerate(items):
#     if item[:8] == "/uploads":
#         item = "http://www.uimaker.com" + item
#         response = requests.get(url, headers=headers)
#         file_path = "{0}\{1}".format(title, name, i + 1000, 'jpg')
#
#         with open(file_path, 'wb') as f:
#             f.write(response.content)
#             print("picture " + str(i) + " is finished.")
