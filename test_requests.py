import requests
import json


r = requests.get('https://api-web.nhle.com/v1/gamecenter/2022030411/play-by-play',params = {'details': 'game_detail'})
print(r.json().get('plays').get('typeCode'))