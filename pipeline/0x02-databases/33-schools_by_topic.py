#!/usr/bin/env python3
"""
Returns list of school having specific topics
"""


def schools_by_topic(mongo_collection, topic):
    """
    Returns list of school having specific topics
    """
    return mongo_collection.find({"topics": {"$in": [topic]}})
