import gradio as gr
import joblib
import numpy as np

def load_model():
    model = joblib.load('student_performance_trainedmodel.joblib')
    return model

model = load_model()

def predict_performance(hours_studied, previous_scores, extracurricular_activities, sleep_hours, question_papers):
    try:
        input_data = np.array([[hours_studied, previous_scores, extracurricular_activities, sleep_hours, question_papers]])

        print("Input Data:", input_data)

        prediction = model.predict(input_data)

        return int(prediction[0])

    except Exception as e:

        return f"Error: {str(e)}"

interface = gr.Interface(
    fn=predict_performance,
    inputs=[
        gr.Number(label="Hours Studied", value=10),
        gr.Number(label="Previous Scores", value=95),
        gr.Number(label="Extracurricular Activities", value=0),
        gr.Number(label="Sleep Hours", value=10),
        gr.Number(label="Sample Question Papers Practiced", value=10)
    ],
    outputs=gr.Textbox(label="Predicted Performance Score"),
    title="Student Performance Prediction",
    description="Predict student performance based on study habits, scores, and activities."
)

interface.launch()