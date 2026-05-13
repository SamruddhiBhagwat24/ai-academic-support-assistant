from chatbot import ask_question
import os

from rag_pipeline import process_pdf

import streamlit as st

from predictor import predict_score


st.set_page_config(
    page_title="AI Academic Assistant",
    layout="centered"
)

st.title("AI Academic Support Assistant")

st.subheader("Student Performance Prediction System")


# ================= PREDICTION SECTION =================

study_hours = st.slider(
    "Weekly Self Study Hours",
    0.0,
    50.0,
    10.0
)

attendance = st.slider(
    "Attendance Percentage",
    0.0,
    100.0,
    75.0
)

participation = st.slider(
    "Class Participation",
    0.0,
    10.0,
    5.0
)

grade = st.selectbox(
    "Current Grade",
    ["A", "B", "C", "D"]
)


if "prediction" not in st.session_state:
    st.session_state.prediction = None


if st.button("Predict Performance"):

    st.session_state.prediction = predict_score(
        study_hours,
        attendance,
        participation,
        grade
    )


if st.session_state.prediction is not None:

    st.success(
        f"Predicted Total Score: {st.session_state.prediction:.2f}"
    )


# ================= PDF UPLOAD SECTION =================

st.divider()

st.header("Upload Study Material")

uploaded_file = st.file_uploader(
    "Upload PDF Notes",
    type=["pdf"]
)


if uploaded_file is not None:

    save_path = os.path.join(
        "documents",
        uploaded_file.name
    )

    with open(save_path, "wb") as f:

        f.write(uploaded_file.getbuffer())

    st.success("PDF uploaded successfully.")

    with st.spinner("Processing PDF and creating embeddings..."):

        process_pdf(save_path)

    st.success("Vector database created successfully.")


# ================= CHATBOT SECTION =================

st.divider()

st.header("Ask Questions From Notes")

user_question = st.text_input(
    "Ask anything from uploaded study material"
)


if st.button("Ask AI"):

    if user_question.strip() == "":

        st.warning("Please enter a question.")

    else:

        with st.spinner("Searching relevant content..."):

            answer = ask_question(user_question)

        st.success("Answer Generated")

        st.markdown(answer)