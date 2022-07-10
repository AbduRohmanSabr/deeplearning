import streamlit as st
from fastai.vision.all import *
import plotly.express as px


st.title("This program differs from the "
         "Rams of animals in the category of "
         "horse, rabbit, sheep, zebra, monkey, elephant.")

# upload image
file = st.file_uploader("Upload image", type={'png', 'jpeg', 'jpg'})
if file:
    st.image(file)
    # PIL convert
    img = PILImage.create(file)

    # model
    model = load_learner("animals.pkl")
    # prediction
    pred, pred_id, probs = model.predict(img)
    st.success(f"Prediction: {pred}")
    st.info(f"Probability : {probs[pred_id]*100:.1f}%")
    # plotting
    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)

