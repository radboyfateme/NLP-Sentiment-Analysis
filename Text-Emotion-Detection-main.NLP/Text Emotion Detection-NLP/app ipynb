import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib

# Load the model once at the start
pipe_lr = joblib.load(open("C:\\Users\\Fatemeh Radboy\\Desktop\\4_5987739612456424556\\Text-Emotion-Detection-main\\Text Emotion Detection\\model\\text_emotion.pkl", "rb"))

# Dictionary mapping emotions to emojis
emotions_emoji_dict = {
    "joy": "😊", "anger": "😡", "surprise": "😲", "disgust": "😒", "fear": "😨", 
    "happy": "😃", "neutral": "😐", "sad": "😢", "sadness": "😢", "shame": "😳"
}

def predict_emotions(docx):
    """Predict the emotion of a given text."""
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    """Get the prediction probabilities of a given text."""
    results = pipe_lr.predict_proba([docx])
    return results

def main():
    st.set_page_config(page_title="Text Emotion Detection", layout="wide")
    
    st.title("Text Emotion Detection")
    st.markdown("### Detect Emotions In Text")
    st.markdown("#### Enter a text to analyze its emotional content. The model will predict the emotion and display the confidence levels for each emotion category.")

    with st.sidebar:
        st.header("Options")
        uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
        theme = st.selectbox("Choose a theme", ["Light", "Dark"])
        
        if theme == "Dark":
            st.markdown(
                """
                <style>
                .main {
                    background-color: #0e1117;
                    color: #fafafa;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <style>
                .main {
                    background-color: #fafafa;
                    color: #0e1117;
                }
                </style>
                """,
                unsafe_allow_html=True
            )

    with st.form(key='my_form'):
        if uploaded_file is not None:
            raw_text = uploaded_file.read().decode("utf-8")
        else:
            raw_text = st.text_area("Type your text here", height=150)
        submit_text = st.form_submit_button(label='Submit')

    if submit_text and raw_text.strip() != "":
        with st.spinner("Analyzing..."):
            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Original Text")
            st.info(raw_text)

            st.markdown("### Prediction")
            emoji_icon = emotions_emoji_dict.get(prediction, "")
            st.success(f"{prediction} {emoji_icon}")
            st.write(f"Confidence: {np.max(probability):.2f}")

        with col2:
            st.markdown("### Prediction Probability")
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]

            fig = alt.Chart(proba_df_clean).mark_bar().encode(
                x='emotions', 
                y='probability', 
                color='emotions'
            ).properties(
                width=400,
                height=300
            )
            st.altair_chart(fig, use_container_width=True)
    else:
        st.warning("Please enter some text to analyze.")

if __name__ == '__main__':
    main()
