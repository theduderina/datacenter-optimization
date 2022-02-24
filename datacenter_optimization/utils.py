#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
"""
Utility functions.

Student name: Jay Bhavesh Doshi , Anna Lebowsky
Student matriculation number: 4963577 , 5143788
"""
from datetime import datetime
from pkg_resources import resource_filename


INPUT_PATH = resource_filename(__name__, "../input")

def utcfromtimestamp(timestamp):
    """
    Construct a naive UTC datetime from a POSIX timestamp.
    
    Same as `datetime.utcfromtimestamp` but can also handle str as
    input.
    """

    return datetime.utcfromtimestamp(int(timestamp))