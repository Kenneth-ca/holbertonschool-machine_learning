#!/usr/bin/env python3
"""
Prints the location of a specific Github user
"""
import requests
import sys
import time


if __name__ == '__main__':
    if len(sys.argv) != 2:
        exit()
    url = sys.argv[1]
    headers = {'Accept': 'application/vnd.github.v3+json'}
    r = requests.get(url, headers=headers)

    if r.status_code == 200:
        print(r.json()['location'])

    if r.status_code == 404:
        print("Not found")

    if r.status_code == 403:
        rate_limit = int(r.headers['X-Ratelimit-Reset'])
        now = int(time.time())
        minutes = int((rate_limit - now) / 60)
        print("Reset in {} min".format(minutes))
