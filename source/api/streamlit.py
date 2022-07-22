
import streamlit as st

api = "http://fastapi:8000/docs"


# def process(image, server_url: str):

#     m = MultipartEncoder(fields={"file": ("filename", image, "image/jpeg")})

#     r = requests.post(
#         server_url, data=m, headers={"Content-Type": m.content_type}, timeout=8000
#     )

#     return r

# construct UI layout
st.title("Bank Marketing")

st.subheader('**A Multilayer Perceptron (MLP) Approach**')

st.markdown("The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls.\
        Often, more than one contact to the same client was required, in order to access if the\
        product (bank term deposit) would be ('yes') or not ('no') subscribed.") 

st.markdown("Links:")

st.markdown("[Bank Marketing Data Set](http://archive.ics.uci.edu/ml/datasets/Bank+Marketing)")
st.markdown("[Repository](https://github.com/francisvalguedes/bank_marketing_mlp)")
st.markdown("[Documentation and API](http://127.0.0.1:8000/docs)")



st.markdown('Por: Francisval Guedes (<francisvalg@gmail.com>), Hareton Gomes')