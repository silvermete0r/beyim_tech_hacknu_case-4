import streamlit as st
from openai import OpenAI

temperature=0.4
frequency_penalty=0.0

# 2023-2024 last official stats
average_scores = {
    "UNT": 73,
    "NUET": 120,
    "IELTS": 6.3,
    "SAT": 1000,
    "TOEFL": 90 
}

# Local Exam Descriptions 
descriptions = {
    "UNT": "UNT (Unified National Testing) is a system for assessing the knowledge of graduates used in Kazakhstan.",
    "NUET": "NUET (Nazarbayev University Entrance Test) is a test used for entering Nazarbayev University in Kazakhstan.",
    "MESC": "MESC (Mathematics, English, Science, and Computer) is a test used for assessing the knowledge of students in Kazakhstan."
}

def generate_prompt(df, exam_type, student_name):
    text_output = f'The following lines describe the performance of {student_name} on {exam_type} practice tests. I need you to analyze the progress of {student_name}: indicate their strong points and weak points if there are any, give the main statistics, identify trends, comment on their progress, and give recommendations. Design your response in markdown format, do not include any graphs or tables.\n'

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