"""
Adapted: Francisval G. Hareton G.
Creator: Ivanovitch Silva
Date: 30 may 2022
Script that POSTS to the API using the requests 
module and returns both the result of 
model inference and the status code
"""
import requests
import json
# import pprint

#no
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

#yes
# person = {
#     "age"       :  53,
#     "job"       :  "management",
#     "marital"   :  "married",
#     "education" :  "tertiary",
#     "default"   :  "no",
#     "balance"   :  583,
#     "housing"   :  "no",
#     "loan"      :  "no",
#     "contact"   :  "cellular",
#     "day"       :  17, 
#     "month"     :  "nov",
#     "duration"  :  226, 
#     "campaign"  :  1, 
#     "pdays"     :  184, 
#     "previous"  :  4, 
#     "poutcome"  :  "success"
#     }

# url = "http://127.0.0.1:8000"
url = "https://bank-marketing-mlp.herokuapp.com"
response = requests.post(f"{url}/predict",
                         json=person)

print(f"Request: {url}/predict")
print(f"Person: \n age: {person['age']}\n job: {person['job']}\n"\
      f" marital: {person['marital']}\n education: {person['education']}\n"\
      f" default: {person['default']}\n"\
      f" balance: {person['balance']}\n"\
      f" housing: {person['housing']}\n"\
      f" loan: {person['loan']}\n"\
      f" contact: {person['contact']}\n"\
      f" day: {person['day']}\n"\
      f" month: {person['month']}\n"\
      f" duration: {person['duration']}\n"\
      f" campaign: {person['campaign']}\n"\
      f" pdays: {person['pdays']}\n"
      f" previous: {person['previous']}\n"
      f" poutcome: {person['poutcome']}\n"
     )
print(f"Result of model inference: {response.json()}")
print(f"Status code: {response.status_code}")