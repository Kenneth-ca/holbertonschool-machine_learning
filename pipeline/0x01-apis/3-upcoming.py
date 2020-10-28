#!/usr/bin/env python3
"""
Displays the upcoming launch information
"""
import requests

if __name__ == '__main__':
    url = "https://api.spacexdata.com/v4/launches/upcoming"
    r = requests.get(url)
    json = r.json()

    dates = [x['date_unix'] for x in json]
    index = dates.index(min(dates))
    next_launch = json[index]

    name = next_launch['name']
    date = next_launch['date_local']
    rocket_id = next_launch['rocket']
    launchpad_id = next_launch['launchpad']

    url_r = "https://api.spacexdata.com/v4/rockets/" + rocket_id
    req_r = requests.get(url_r)
    json_r = req_r.json()
    rocket_name = json_r['name']

    url_l = "https://api.spacexdata.com/v4/launchpads/" + launchpad_id
    req_l = requests.get(url_l)
    json_l = req_l.json()
    launchpad_name = json_l['name']
    launchpad_loc = json_l['locality']

    info = (name + ' (' + date + ') ' + rocket_name + ' - ' +
            launchpad_name + ' (' + launchpad_loc + ')')
    print(info)
