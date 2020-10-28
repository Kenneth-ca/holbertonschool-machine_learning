#!/usr/bin/env python3
"""
Create a method that returns the list of ships that can hold a given number
of passengers
"""
import requests


def availableShips(passengerCount):
    """
    Returns the list of ships that can hold a given number of passengers
    :param passengerCount: number of passengers
    :return: If no ship available, return an empty list
    """
    url = "https://swapi-api.hbtn.io/api/starships/"
    r = requests.get(url)
    json = r.json()
    results = json["results"]
    ships = []
    while json["next"]:
        for res in results:
            if res["passengers"] == 'n/a' or res["passengers"] == 'unknown':
                continue
            if int(res["passengers"].replace(',', '')) >= passengerCount:
                ships.append(res["name"])
        url = json["next"]
        r = requests.get(url)
        json = r.json()
        results = json["results"]
    return ships
