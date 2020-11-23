#!/usr/bin/env python3
"""
Change school topics
"""


def update_topics(mongo_collection, name, topics):
    """
    Change school topics
    """
    search = {"name": name}
    new = {"$set": {"topics": topics}}

    mongo_collection.update_many(search, new)
