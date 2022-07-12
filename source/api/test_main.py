"""
Creator: Francisval G. & Hareton G., Adapted from version: Ivanovitch Silva
Date: 30 may 2022
API testing
"""
from fastapi.testclient import TestClient
import os
import sys
import pathlib
from source.api.main import app

# Instantiate the testing client with our app.
client = TestClient(app)

# a unit test that tests the status code of the root path
def test_root():
    r = client.get("/")
    assert r.status_code == 200

# a unit test that tests the status code and response 
# for an instance with a low income
def test_get_inference_no():

    person = {
        "age"       :  58,
        "job"       :  "management",
        "marital"   :  "married",
        "education" :  "tertiary",
        "default"   :  "no",
        "balance"   :  2143,
        "housing"   :  "yes",
        "loan"      :  "no",
        "contact"   :  "unknown",
        "day"       :  5, 
        "month"     :  "may",
        "duration"  :  261, 
        "campaign"  :  1, 
        "pdays"     :  -1, 
        "previous"  :  0, 
        "poutcome"  :  "unknown"
    }

    r = client.post("/predict", json=person)
    # print(r.json())
    assert r.status_code == 200
    assert r.json() == "no"

# a unit test that tests the status code and response 
# for an instance with a high income
def test_get_inference_yes():

    person = {
        "age"       :  53,
        "job"       :  "management",
        "marital"   :  "married",
        "education" :  "tertiary",
        "default"   :  "no",
        "balance"   :  583,
        "housing"   :  "no",
        "loan"      :  "no",
        "contact"   :  "cellular",
        "day"       :  17, 
        "month"     :  "nov",
        "duration"  :  226, 
        "campaign"  :  1, 
        "pdays"     :  184, 
        "previous"  :  4, 
        "poutcome"  :  "success"
    }

    r = client.post("/predict", json=person)
    print(r.json())
    assert r.status_code == 200
    assert r.json() == "yes"