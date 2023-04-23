import requests

url = "https://drive.google.com/u/0/uc?id=1hc1Xh_1qBkU4iGzWxkThpUa5_W9t7GZ_&export=download"
r = requests.get(url, allow_redirects=True)
open('cunet_weight.pth', 'wb').write(r.content)