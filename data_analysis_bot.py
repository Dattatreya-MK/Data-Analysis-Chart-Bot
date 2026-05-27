import streamlit as st
import pandas as pd
import plotly.express as px
import os
import re

from dotenv import load_dotenv
from langchain_groq import ChatGroq

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="AI Data Analyst Bot",
    layout="wide"
)

st.title("📊 AI Data Analyst Bot")

# ---------------- LOAD ENV ---------------- #
load_dotenv()

# ---------------- LOAD GROQ MODEL ---------------- #
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

# ---------------- CLEAN GENERATED CODE ---------------- #
def clean_code(code):
    code = re.sub(r"```python", "", code)
    code = re.sub(r"```", "", code)
    return code.strip()

# ---------------- FILE UPLOAD ---------------- #
file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if file:

    # Read CSV
    df = pd.read_csv(file)

    # Preview Data
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # Dataset Info
    st.subheader("📌 Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Rows:", df.shape[0])
        st.write("Columns:", df.shape[1])

    with col2:
        st.write("Column Names:")
        st.write(list(df.columns))

    # ---------------- USER QUESTION ---------------- #
    question = st.text_input(
        "Ask a question about your data"
    )

    if question:

        # ---------------- PROMPT ---------------- #
        prompt = f"""
You are a senior AI Data Analyst.

Dataset Columns:
{list(df.columns)}

Write ONLY Python code.

Rules:
- Dataframe name is df
- Store final answer in variable named result
- Use pandas operations
- Use plotly.express as px for charts if needed
- Figure variable name must be fig
- Display charts using:
    st.plotly_chart(fig)

STRICT RULES:
- Do NOT import any libraries
- No markdown
- No explanation
- No print()
- Output ONLY executable Python code

Question:
{question}
"""

        # ---------------- LLM RESPONSE ---------------- #
        response = llm.invoke(prompt)

        # Clean code
        code = clean_code(response.content)

        # ---------------- SHOW GENERATED CODE ---------------- #
        st.subheader("🧠 Generated Code")
        st.code(code, language="python")

        # ---------------- EXECUTE GENERATED CODE ---------------- #
        try:

            # Variables available to AI-generated code
            local_vars = {
                "df": df,
                "pd": pd,
                "px": px,
                "st": st
            }

            # Execute generated code
            exec(code, {}, local_vars)

            # ---------------- SHOW RESULT ---------------- #
            if "result" in local_vars:

                st.subheader("📊 Answer")
                st.write(local_vars["result"])

            else:
                st.warning("⚠️ No result variable found.")

        except Exception as e:

            st.error(f"❌ Execution Error: {e}")

else:
    st.info("📂 Please upload a CSV file to begin.")