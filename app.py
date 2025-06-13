import streamlit as st
import pandas as pd
import yaml
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load Gemini API key from Streamlit secrets
google_api_key = st.secrets["GEMINI_API_KEY"]

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    google_api_key=google_api_key,
    model="gemini-pro",
    temperature=0.3
)

# Prompt template for rule generation
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
st.title("📊 AI-Driven Data Quality Checker (Gemini)")

uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])
user_rule = st.text_area("Describe your data quality rule (in English):", "")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of your data:")
    st.dataframe(df)

    if user_rule.strip() and st.button("Generate and Apply Rule"):
        try:
            yaml_text = chain.run({"question": user_rule})
            st.code(yaml_text, language="yaml")

            rules = yaml.safe_load(yaml_text)
            if not isinstance(rules, list):
                rules = [rules]

            for rule in rules:
                st.subheader(f"Rule: {rule.get('rule_id', 'Unnamed Rule')}")
                try:
                    condition = rule.get("condition", None)
                    check_expr = rule["check"]

                    subset = df.query(condition) if condition else df
                    validity = eval(check_expr, {"df": subset, "pd": pd})

                    if isinstance(validity, pd.Series):
                        failed_rows = subset[~validity]
                    else:
                        failed_rows = subset if not validity else subset[[]]

                    st.write(f"❌ Failed Rows: {len(failed_rows)}")
                    st.dataframe(failed_rows)
                except Exception as e:
                    st.error(f"⚠️ Error applying rule `{rule.get('rule_id', 'unknown')}`: {e}")
        except Exception as e:
            st.error(f"Error generating or applying rule: {e}")
