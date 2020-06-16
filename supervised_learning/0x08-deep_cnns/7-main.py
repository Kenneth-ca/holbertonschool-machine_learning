#!/usr/bin/env python3

densenet121 = __import__('7-densenet121').densenet121

if __name__ == '__main__':
    model = densenet121(32, 0.5)
    model.summary()
