import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
import matplotlib.pyplot as plt
import seaborn as sns
import os
from forecasting import linear_regression
from matplotlib.backends.backend_pdf import PdfPages
from fpdf import FPDF
from PyPDF2 import PdfMerger
import base64

client = OpenAI(
    api_key=st.secrets["OPENAI_API"],
)

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "robot" not in st.session_state:
    st.session_state.robot = True

student_name, response = None, None
temperature=0.3
frequency_penalty=0.0

# Local Exam Descriptions 
descriptions = {
    "UNT": "UNT (Unified National Testing) is a system for assessing the knowledge of graduates used in Kazakhstan.",
    "NUET": "NUET (Nazarbayev University Entrance Test) is a test used for entering Nazarbayev University in Kazakhstan.",
    "MESC": "MESC (Mathematics, English, Science, and Computer) is a test used for assessing the knowledge of students in Kazakhstan."
}

# 2023-2024 last official stats
average_scores = {
    "UNT": 73,
    "NUET": 120,
    "IELTS": 6.3,
    "SAT": 1000,
    "TOEFL": 90 
}

# Letter-based Scores
letter_scores = {
    "A*": 9,
    "A": 8,
    "B": 7,
    "C": 5,
    "D": 4,
    "E": 3,
    "F": 2,
    "G": 1,
    "U": 0 
}

def encode_score(score):
    return letter_scores.get(score, 0)

def get_dataframe(file_path):
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path, engine="python")
    elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
        return pd.read_excel(file_path, engine="openpyxl")
    return None

def generate_prompt(df, exam_type, student_name):
    text_output = f'The following lines describe the performance of {student_name} on {exam_type} practice tests. I need you to analyze the progress of {student_name}: indicate their strong points and weak points if there are any, give statistics such as their mean scores, minimum and maximum scores, identify trends, comment on their progress, and give recommendations. Design your response in markdown format.\n'

    if exam_type in average_scores:
        text_output += f"The average score for {exam_type} is {average_scores[exam_type]}, add comparative analysis for the student's total score at the end.\n\n"
    
    for _, row in df.iterrows():
      text = f"On {row[df.columns[0]]}, the student scored "
      for column in df.columns[1:-1]:
        text += f"{row[column]} for the {column} section, "
      text += f"with a total score of {row[df.columns[-1]]}."
      text_output += text + '\n'
    
    return text_output

def generate_gpt4_response(prompt, exam_type, api_key):
    client = OpenAI(api_key=api_key)
    gpt_assistant_prompt = f"You are a professional {exam_type} tutor!"

    if exam_type in descriptions:
        gpt_assistant_prompt += ("\n" + descriptions[exam_type])

    message=[{"role": "assistant", "content": gpt_assistant_prompt}, {"role": "user", "content": prompt}]

    st.session_state.messages.append({"role": "assistant", "content": gpt_assistant_prompt})
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=st.session_state.openai_model,
        messages = message,
        temperature=temperature,
        frequency_penalty=frequency_penalty
    )

    return response.choices[0].message.content

def draw_analysis_lineplot(df):        
    id = df.columns[0]
    linestyles = ['--', '-.', ':', '-']
    for i, column in enumerate(df.columns[1:]):
        linestyle = linestyles[i % len(linestyles)]
        sns.lineplot(data=df, x=id, y=column, label=column, marker='o', linestyle=linestyle, markersize=5)
    plt.legend()
    plt.xlabel('Test ID')
    plt.ylabel('Test Score')
    st.pyplot(plt)

st.set_page_config(
    page_title="Beyim AI",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Beyim AI")

st.markdown("#### Welcome to Beyim AI, a platfom that helps you to analyze your academic performance!")

exam_type = st.selectbox("Select an exam", ["", "UNT", "NUET", "IELTS", "MESC", "TOEFL", "NUET", "SAT", "OTHER"], index=0)

if exam_type == "OTHER":
    exam_type = st.text_input("Enter exam name:")

if exam_type:
    student_name = st.text_input("Enter student's name:")

if student_name:
    file = st.file_uploader("Upload a file", type=["xlsx", "xls", "csv"])

    st.checkbox("Show sample data", key="show_sample_data")
    if st.session_state.show_sample_data:
        exam_type = exam_type.strip()
        sample_df = pd.read_excel("data/data_samples.xlsx", engine="openpyxl", sheet_name="General" if exam_type not in {"UNT", "NUET", "IELTS", "MESC", "TOEFL", "NUET", "SAT"} else exam_type)
        st.dataframe(sample_df)
    
    if file:
        with open(f"temp/{file.name}", "wb") as f:
            f.write(file.getbuffer())
        file_path = f"temp/{file.name}"
        df = get_dataframe(file_path)

        if file:
            with open(f"temp/{file.name}", "wb") as f:
                f.write(file.getbuffer())
            file_path = f"temp/{file.name}"
            df = get_dataframe(file_path)
            os.remove(file_path)

            analyzing_button = st.button("Analyze", key="analyze_button")

            if analyzing_button:
                st.subheader("Exploratory Data Analysis:")
                
                if any(df.apply(lambda col: col.isin(['A*', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'U']).any())):
                    df_copy = df.copy()
                    df_copy.iloc[:, 1:] = df_copy.iloc[:, 1:].applymap(encode_score)
                    colA, colB = st.columns(2)
                    with colA:
                        with st.spinner("🤖 Loading..."):
                            st.dataframe(df)
                            fig, ax = plt.subplots(figsize=(12,4))
                            ax.axis('tight')
                            ax.axis('off')
                            the_table = ax.table(cellText=df.values,colLabels=df.columns,loc='center')
                            with PdfPages("temp/table1.pdf") as pp:
                                pp.savefig(fig, bbox_inches='tight')
                            plt.close()
                    with colB:
                        with st.spinner("🤖 Plotting..."):
                            st.scatter_chart(data=df_copy.iloc[:, 1:])
                            with PdfPages("temp/plot2.pdf") as pdf:
                                pdf.savefig()
                            st.pyplot()
                            plt.close()
                else:
                    colA, colB = st.columns(2)

                    with colA:
                        with st.spinner("🤖 Loading..."):
                            st.dataframe(df)
                            fig, ax = plt.subplots(figsize=(12,4))
                            ax.axis('tight')
                            ax.axis('off')
                            the_table = ax.table(cellText=df.values,colLabels=df.columns,loc='center')
                            with PdfPages("temp/table1.pdf") as pp:
                                pp.savefig(fig, bbox_inches='tight')
                            plt.close()
                    with colB:
                        with st.spinner("🤖 Calculating..."):
                            df_copy = df.describe().T.drop(columns=["count", "std", "25%", "50%", "75%"])
                            st.write(df_copy)
                            fig, ax = plt.subplots(figsize=(12,4))
                            ax.axis('tight')
                            ax.axis('off')
                            the_table = ax.table(cellText=df_copy.values, colLabels=df_copy.columns, loc='center')
                            with PdfPages("temp/table2.pdf") as pp:
                                pp.savefig(fig, bbox_inches='tight')
                            plt.close()
                    colA, colB = st.columns(2)
                    with colA:
                        with st.spinner("🤖 Plotting..."):
                            draw_analysis_lineplot(df)

                    with colB:
                        with st.spinner("🤖 Plotting..."):
                            if df.columns[-1].strip().lower() == 'total':
                                df.plot(kind="bar", stacked=True, x=df.columns[0], y=df.columns[1:-1], title="Total Score")
                                for i, row in df.iterrows():
                                    total_score = row[df.columns[-1]]
                                    plt.text(i, total_score, total_score, ha='center', va='bottom')
                                with PdfPages("temp/plot2.pdf") as pdf:
                                    pdf.savefig()
                                st.pyplot(plt)
                                plt.close()
                            elif df.columns[-1].strip().lower() == 'average':
                                df_scaled = df.copy()
                                df_scaled[df.columns[1:-1]] = df_scaled[df.columns[1:-1]] * 0.25
                                df_scaled.plot(kind="bar", stacked=True, x=df.columns[0], y=df.columns[1:-1], title="Band Score")
                                for i, row in df_scaled.iterrows():
                                    total_score = row[df_scaled.columns[-1]]
                                    plt.text(i, total_score, total_score, ha='center', va='bottom')
                                with PdfPages("temp/plot2.pdf") as pdf:
                                    pdf.savefig()
                                st.pyplot(plt)
                                plt.close()
                            else:
                                st.snow()
                    
                    with st.spinner("🤖 Forecasting..."):
                        linear_regression(df)
            
                    with st.spinner("🤖 Analyzing..."):
                        try:
                            response = generate_gpt4_response(generate_prompt(df, exam_type, student_name), exam_type, st.secrets["OPENAI_API"])
                            st.markdown(response)

                            pdf = FPDF()
                            pdf.add_page()
                            pdf.set_font("Arial", size=12)
                            pdf.cell(200, 10, txt="Analysis Report", ln=True, align="C")
                            pdf.multi_cell(0, 10, txt = response)
                            pdf_file_path = "temp/text_document.pdf"
                            pdf.output(pdf_file_path)
                        
                            merger = PdfMerger()
                            for filename in os.listdir('temp'):
                                if filename.endswith('.pdf'):
                                    if filename == 'final_analysis.pdf':
                                        continue
                                    with open(f'temp/{filename}', 'rb') as file:
                                        merger.append(file)
                                    os.remove(f'temp/{filename}')

                            with open('temp/final_analysis.pdf', 'wb') as output:
                                merger.write(output)
                            
                            merger.close()

                            st.success("Analysis completed successfully!")

                            file_content = open("temp/final_analysis.pdf", "rb").read()

                            st.markdown(
                                f'<a href="data:application/pdf;base64,{base64.b64encode(file_content).decode()}" download="analysis_report.pdf">Download Analysis Report</a>',
                                unsafe_allow_html=True
                            )

                            st.session_state.robot = False
                                              
                        except Exception as e:
                            st.error(f"An error occurred: {str(e)}")
                            st.stop()

# Accept user input
if prompt := st.chat_input("🤖 The bot is offline. Submit your file to turn him up!" if st.session_state.robot else "🤖 The bot is on. You can ask him questions!", disabled=st.session_state.robot):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})