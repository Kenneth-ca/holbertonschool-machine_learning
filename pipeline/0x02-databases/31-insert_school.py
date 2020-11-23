#!/usr/bin/env python3
"""
Insert documents in Python
"""


def insert_school(mongo_collection, **kwargs):
    """
    Insert documents
    """
    return mongo_collection.insert_one(kwargs).inserted_id
