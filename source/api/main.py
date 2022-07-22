"""
Creator: Francisval G. & Hareton G., Adapted from version: Ivanovitch Silva
Date: 30 may 2022
Create API
"""

# <img src="{{ url_for('static', path='images/marketing.png') }}" alt="" width="30" height="24">


# from typing import Union
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi import FastAPI
import pandas as pd
import joblib
import os
import wandb
import sys
from source.api.pipeline import FeatureSelector, CategoricalTransformer, NumericalTransformer

import tensorflow as tf
import tensorflow_addons as tfa

from fastapi.templating import Jinja2Templates
#from fastapi import APIRouter
from fastapi import Request
from fastapi.staticfiles import StaticFiles

# global variables
setattr(sys.modules["__main__"], "FeatureSelector", FeatureSelector)
setattr(sys.modules["__main__"], "CategoricalTransformer", CategoricalTransformer)
setattr(sys.modules["__main__"], "NumericalTransformer", NumericalTransformer)

# name of the model artifact
artifact_model_name = "mlops_ivan/bank_mlp/model_export:latest"
artifact_keras_model_name = "mlops_ivan/bank_mlp/keras_model_export:latest"
# keras_model_run_path = "mlops_ivan/bank_mlp/2nz728iv"

# initiate the wandb project
run = wandb.init(project="bank_mlp", entity="mlops_ivan",job_type="api")

# create the api
app = FastAPI()
# app.include_router(general_pages_router)
app.mount("/static", StaticFiles(directory="source/api/static"), name="static")

templates = Jinja2Templates(directory="source/api/templates")
# general_pages_router = APIRouter()

# declare request example data using pydantic
# a person in our dataset has the following attributes
class Person(BaseModel):
    age       :  int 
    job       :  str
    marital   :  str
    education :  str
    default   :  str
    balance   :  int 
    housing   :  str
    loan      :  str
    contact   :  str
    day       :  int 
    month     :  str
    duration  :  int 
    campaign  :  int 
    pdays     :  int 
    previous  :  int 
    poutcome  :  str

    class Config:
        schema_extra = {
            "example": {
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
        }

# give a greeting using GET
#@app.get("/", response_class=HTMLResponse)
# @general_pages_router.get("/")
# async def root(request: Request):
#     result = "Type a number"
#     return templates.TemplateResponse("root.html", context={'request': request, 'result': result})
@app.get("/")
async def home(request: Request):
	return templates.TemplateResponse("root.html",{"request":request})
      
    # """
    # <p><span style="font-size:28px"><strong>Bank Marketing - A Multilayer Perceptron (MLP) Approach</strong></span></p>"""\
    # """<p><span style="font-size:20px">The data is related with direct marketing campaigns of"""\
    #     """ a Portuguese banking institution. The marketing campaigns were based on phone calls."""\
    #     """ Often, more than one contact to the same client was required, in order to access if the"""\
    #     """ product (bank term deposit) would be ('yes') or not ('no') subscribed.</span></p>"""\
    # """ <p><span style="font-size:20px">The data is publicly available:"""\
    #     """<a href="http://archive.ics.uci.edu/ml/datasets/Bank+Marketing"> Bank Marketing</a>."""\
    #     """ Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]."""\
    #     """ Irvine, CA: University of California, School of Information and Computer Science.</span></p>"""\
    # """ <p><span style="font-size:20px">The model repository is publicly available at:"""\
    #     """ <a href="https://github.com/francisvalguedes/bank_marketing_mlp"> bank_marketing repository</a>.</span></p>"""\
    # """ <p><span style="font-size:20px">The classification goal is to predict if the client will subscribe (yes/no) a term deposit (variable y)."""\
    #     """ You can access the documentation and test the API from the link:"""\
    #     """ <a href="../../docs"> documentation</a>.</span></p>"""

# run the model inference and use a Person data structure via POST to the API.
@app.post("/predict")
async def get_inference(person: Person):
    
    # Download pipe artifact
    model_export_path = run.use_artifact(artifact_model_name).file()
    pipe = joblib.load(model_export_path)

    # Download inference artifact keras model
    # use the latest version of the keras model
    model_at = run.use_artifact(artifact_keras_model_name)
    # download the directory in which the model is saved
    model_dir= model_at.download()
    model = tf.keras.models.load_model(model_dir)

    # # restore the raw model file "model.h5" from a specific run
    # best_model = wandb.restore('model.h5', run_path="mlops_ivan/bank_mlp/330l6zaz")
    # # restore the model for tf.keras
    # model = tf.keras.models.load_model(best_model.name) 
   
    # Create a dataframe from the input feature
    # note that we could use pd.DataFrame.from_dict
    # but due be only one instance, it would be necessary to
    # pass the Index.

    person_df = pd.DataFrame([person.dict()])

    person_transformed = pipe.transform(person_df)
    
    # Predict test data
    predict = model.predict(person_transformed)

    return "no" if predict[0] <= 0.5 else "yes"