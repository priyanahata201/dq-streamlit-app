import streamlit as st
import pandas as pd
import yaml
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Gemini API key from Streamlit Secrets
google_api_key = st.secrets["GEMINI_API_KEY"]

# Gemini LLM (1.5 Flash)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key, temperature=0)

# Prompt template for rule conversion
template = """
You are a data quality expert.

Convert the following plain English data quality rule into a YAML-formatted rule for a Python-based data quality engine.

YAML Format:
- rule_id: <unique_rule_id>
  description: <detailed_description>
  condition: <optional pandas query condition>  # optional
  check: <boolean pandas expression that returns True for valid rows>

EXAMPLES:
- rule_id: dq_kyc_number_length_check
  description: "If KYCType is IDP6 then KYCNumber length should be 12"
  condition: "df['KYCType'] == 'IDP6'"
  check: "df['KYCNumber'].astype(str).str.len() == 12"

{question}
"""

prompt = PromptTemplate(
    input_variables=["question"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)

# Streamlit UI
st.set_page_config(page_title="AI Data Quality Checker", layout="wide")
st.title("📊 AI-Driven Data Quality Checker (Gemini)")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
user_rule = st.text_area("Enter your data quality rule (in English):", height=100)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### 🔍 Preview of Uploaded Data")
    st.dataframe(df)

    if user_rule and st.button("✅ Generate & Apply Rule"):
        with st.spinner("Generating YAML rule using Gemini..."):
            yaml_text = chain.run({"question": user_rule})

        st.code(yaml_text, language="yaml")

        try:
            rules = yaml.safe_load(yaml_text)
            if not isinstance(rules, list):
                rules = [rules]
        except Exception as e:
            st.error(f"❌ Failed to parse YAML: {e}")
            st.stop()

        results_summary = []

        for rule in rules:
            st.subheader(f"🔎 Rule: {rule.get('rule_id', 'Unnamed')}")
            try:
                condition = rule.get("condition", None)
                check_expr = rule["check"]
                description = rule.get("description", "")

                subset = df.query(condition) if condition else df
                validity = eval(check_expr, {"df": subset, "pd": pd})

                if isinstance(validity, pd.Series):
                    failed_rows = subset[~validity]
                    violation_count = failed_rows.shape[0]
                else:
                    violation_count = 0 if validity else subset.shape[0]
                    failed_rows = subset if violation_count > 0 else subset[[]]

                percent = (violation_count / subset.shape[0] * 100) if subset.shape[0] > 0 else 0.0

                st.write(f"❌ Failed Rows: {violation_count}")
                if violation_count > 0:
                    st.dataframe(failed_rows)

                results_summary.append({
                    "rule_id": rule.get("rule_id", ""),
                    "description": description,
                    "violations": violation_count,
                    "percentage": round(percent, 2)
                })

            except Exception as e:
                st.error(f"⚠️ Error applying rule `{rule.get('rule_id', 'unknown')}`: {e}")

        # Display summary table
        if results_summary:
            st.markdown("### 📋 Rule Validation Summary")
            summary_df = pd.DataFrame(results_summary)
            st.dataframe(summary_df)
