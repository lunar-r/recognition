import requests
import re
import os

headers = {
    'user-agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36'
    }
url = "https://www.google.com/search?safe=active&tbm=isch&q=Eva+Green&chips=q:eva+green,g_1:casino+royale:4MUygCs8Cw0%3D&usg=AI4_-kSGeqMCmeu_Jz6pds7sf_2Z-wqwzQ&sa=X&ved=0ahUKEwii_uiSubXhAhUv7HMBHSeUCfYQ4lYIKCgB&biw=1707&bih=880&dpr=1.13"


response = requests.get(url, headers=headers)
html = None
if response.status_code == 200:
    html = response.text

# 1. Re
name = "Eva_Green"
title = "..\source\images\\" + name
if not os.path.exists(title):
    os.mkdir(title)

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


