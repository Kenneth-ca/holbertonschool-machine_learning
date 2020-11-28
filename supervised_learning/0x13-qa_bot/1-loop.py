#!/usr/bin/env python3
"""
Takes in input from the user
"""


while 1:
    question = input("Q: ")
    words = ["exit", "quit", "goodbye", "bye"]

    if question.lower().strip() in words:
        print("A: Goodbye")
        exit(0)
    else:
        print("A: ")
