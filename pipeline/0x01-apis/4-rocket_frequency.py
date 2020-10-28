#!/usr/bin/env python3
"""
Displays the number of launches per rocket
"""
import requests


if __name__ == '__main__':

    rockets = {}

    url = 'https://api.spacexdata.com/v4/launches'
    r = requests.get(url)
    launches = r.json()

    for launch in launches:
        rocket_id = launch['rocket']
        url_r = "https://api.spacexdata.com/v4/rockets/{}".\
            format(rocket_id)
        req_r = requests.get(url_r)
        json_r = req_r.json()
        rocket_name = json_r['name']

        if rocket_name in rockets.keys():
            rockets[rocket_name] += 1
        else:
            rockets[rocket_name] = 1

    sort = sorted(rockets.items(), key=lambda x: x[0])
    sort = sorted(sort, key=lambda x: x[1], reverse=True)

    for i in sort:
        print("{}: {}".format(i[0], i[1]))
