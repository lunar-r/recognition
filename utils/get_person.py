import requests
import re
import os
import cv2
headers = {
    'user-agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36'
    }
urls = [
    "https://www.google.com/search?safe=active&biw=1707&bih=880&tbm=isch&sa=1&ei=4kqwXN64L47W0gSnhb9Q&q=+Kaley+Cuoco&oq=+Kaley+Cuoco&gs_l=img.3..0i67l3j0l7.33245.33245..129114...0.0..0.258.258.2-1......1....2j1..gws-wiz-img.61nniqVKd5g",
    "https://www.google.com/search?safe=active&biw=1707&bih=880&tbm=isch&sa=1&ei=ZUuwXNu1CcHFmAWxr6mgDw&q=Simon+Helberg&oq=Simon+Helberg&gs_l=img.3..0l4j0i67j0l5.36667.38554..39298...0.0..0.199.379.0j2......1....2j1..gws-wiz-img.xvZL35vtiOA",
    "https://www.google.com/search?safe=active&biw=1707&bih=880&tbm=isch&sa=1&ei=jUuwXMLCFpCMr7wP38ibuAI&q=Kunal+Nayyar&oq=Kunal+Nayyar&gs_l=img.3..0i67j0j0i67j0l7.24630.28364..28676...0.0..0.187.1093.0j6......1....2j1..gws-wiz-img.3iY248A4vMc",
    "https://www.google.com/search?safe=active&biw=1707&bih=880&tbm=isch&sa=1&ei=q0uwXMeEBIe4mAXFvpC4AQ&q=Melissa+Rauch&oq=Melissa+Rauch&gs_l=img.3..0i67j0l9.15100.22811..23596...0.0..0.191.373.0j2......1....2j1..gws-wiz-img.pe0YZsPjo6U",
    "https://www.google.com/search?safe=active&biw=1707&bih=880&tbm=isch&sa=1&ei=w0uwXOaiGaiNr7wPrrm2yAY&q=Mayim+Bialik&oq=Mayim+Bialik&gs_l=img.3..0l10.15556.17388..18203...0.0..0.191.558.0j3......1....2j1..gws-wiz-img.1trNam9BnNw",
    "https://www.google.com/search?safe=active&biw=1707&bih=880&tbm=isch&sa=1&ei=1kuwXJiqHuqumAXB8q2IAQ&q=Kevin+Sussman&oq=Kevin+Sussman&gs_l=img.3..0l10.118467.127707..128218...0.0..0.323.528.2-1j1......1....2j1..gws-wiz-img.......0i30.yqiLa7Y7N8o",
    "https://www.google.com/search?safe=active&biw=1707&bih=880&tbm=isch&sa=1&ei=V0ywXOrqJdzKmAXg25v4BA&q=Giovanni+Bejarano&oq=Giovanni+Bejarano&gs_l=img.3..0.19174.19174..19429...0.0..0.185.185.0j1......1....2j1..gws-wiz-img.CqFpAra8080",
    "https://www.google.com/search?safe=active&biw=1707&bih=880&tbm=isch&sa=1&ei=bEywXO7OEYKSr7wPm661wAw&q=Ciara+Ren%C3%A9e&oq=Ciara+Ren%C3%A9e&gs_l=img.3..0l6j0i5i30j0i24l3.28054.28054..28436...0.0..0.194.194.0j1......1....2j1..gws-wiz-img.Ah4gg15oRBc",
    "https://www.google.com/search?safe=active&biw=1707&bih=880&tbm=isch&sa=1&ei=iUywXMyLLsCUr7wPv5CwsA4&q=Raymond+Joseph+Teller&oq=Raymond+Joseph+Teller&gs_l=img.3..0l2j0i24l8.21760.21760..22134...0.0..0.183.183.0j1......1....2j1..gws-wiz-img.cxYjLAzzhe4",
    "https://www.google.com/search?safe=active&biw=1707&bih=880&tbm=isch&sa=1&ei=oUywXIXdB4PVmAWnjJSgCw&q=Neil+deGrasse+Tyson&oq=Neil+deGrasse+Tyson&gs_l=img.3..0l10.22347.22347..23042...0.0..0.195.195.0j1......1....2j1..gws-wiz-img.JsO0V1GTcvc",
    "https://www.google.com/search?safe=active&biw=1707&bih=880&tbm=isch&sa=1&ei=uUywXLnPDr6Vr7wPrJOHuAg&q=Kathy+Bates&oq=Kathy+Bates&gs_l=img.3..0l10.18976.18976..19319...0.0..0.203.203.2-1......1....2j1..gws-wiz-img.43cjHmJp-Jw",
    "https://www.google.com/search?safe=active&biw=1707&bih=880&tbm=isch&sa=1&ei=zkywXObtGI6wmAXow5Qo&q=Bill+Nye&oq=Bill+Nye&gs_l=img.3..0l10.15127.15127..15482...0.0..0.188.188.0j1......1....2j1..gws-wiz-img.Ed-5PnGXVis",
    "https://www.google.com/search?safe=active&biw=1707&bih=880&tbm=isch&sa=1&ei=3kywXPbGONeMr7wP6c6JgAI&q=Mark+Hamill&oq=Mark+Hamill&gs_l=img.3..0l10.15163.15163..15546...0.0..0.206.206.2-1......1....2j1..gws-wiz-img.ZsLHKAeTWmk",
    "https://www.google.com/search?safe=active&biw=1707&bih=880&tbm=isch&sa=1&ei=70ywXJbyG-2kmAX4i6ioBQ&q=Christine+Baranski&oq=Christine+Baranski&gs_l=img.3..0l10.11817.11817..12024...0.0..0.191.191.0j1......1....2j1..gws-wiz-img.FBJveuiEyNw"
]

names = [
    "Kaley_Cuoco", "Simon_Helberg", "Kunal_Nayyar",
    "Melissa_Rauch", "Mayim_Bialik", "Kevin_Sussman",
    "Giovanni_Bejarano", "Ciara_Renée", "Raymond_Joseph_Teller",
    "Neil_deGrasse_Tyson", "Kathy_Bates", "Bill_Nye",
    "Mark_Hamill", "Christine_Baranski"
]

urls_less = [
    "https://www.google.com/search?safe=active&biw=1707&bih=880&tbm=isch&sa=1&ei=_UywXMhZp7CYBbbRnfAC&q=Johnny+Galecki&oq=Johnny+Galecki&gs_l=img.3..0l10.2246655.2246655..2247053...0.0..0.192.192.0j1......1....2j1..gws-wiz-img.B0oxPG0_Yx0",
    "https://www.google.com/search?safe=active&biw=1707&bih=880&tbm=isch&sa=1&ei=xVWwXOvIBe6Lr7wPjPGU8Aw&q=Jim+Parsons&oq=Jim+Parsons&gs_l=img.3..35i39l2j0l2j0i67j0j0i67j0l3.47421.47421..47804...0.0..0.193.193.0j1......1....2j1..gws-wiz-img.RHGUlDUNDvc"

]

names_less = [
    "Johnny_Galecki", "Jim_Parsons"
]


def download(name, url):
    response = requests.get(url, headers=headers)
    html = None
    if response.status_code == 200:
        html = response.text

    # 1. Re
    title = "..\source\images\\" + name
    if not os.path.exists(title):
        os.mkdir(title)

    # google img
    pattern = re.compile('<img .*?src="(.*?)".*?jsaction', re.S)
    items = re.findall(pattern, html)
    count = 0
    for i, item in enumerate(items):
        if count > 30:
            break
        url = item
        if url[:4] != "http":
            continue
        response = requests.get(url, headers=headers)
        file_path = "{0}\{1}_{2}.{3}".format(title, name, i + 1000, 'jpg')
        print("still search: " + str(i))
        with open(file_path, 'wb') as f:
            f.write(response.content)
            print("picture " + str(i) + " is finished.")
            count = count + 1


url = "https://www.google.com/search?safe=active&biw=1707&bih=880&tbm=isch&sa=1&ei=3r-1XJ3LGfKsmAXNnLaICg&q=Hillary+Clinton&oq=Hillary+Clinton&gs_l=img.3..0i67l4j0l6.16891.16993..17210...0.0..0.191.376.0j2......1....1..gws-wiz-img.odR4QXZjkPk"
download("Hillary_Clinton", url)

# for (name, url) in zip(names_less, urls_less):
#     download(name, url)
#     print("Finish task of: " + name)

    # img = cv2.imread(file_path)
    # height, width = img.shape[:2]
    # if height > 300 and width > 300:

    # else:
    #     os.remove(file_path)

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
