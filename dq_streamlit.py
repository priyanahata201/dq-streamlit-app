import streamlit as st
import pandas as pd
import yaml
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load API key from Streamlit secrets
openai_key = st.secrets["API_KEY"]

# Initialize LLM
llm = ChatOpenAI(openai_api_key=openai_key, temperature=0, model="gpt-4")

# YAML Prompt Template
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

{input}
"""

prompt = PromptTemplate(template=template, input_variables=["input"])
chain = LLMChain(llm=llm, prompt=prompt)

st.title("📊 AI-Driven Data Quality Checker")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])

# User Rule Prompt
user_rule = st.text_area("Describe your data quality rule (in English):", "")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of your data:")
    st.dataframe(df)

    if user_rule and st.button("Generate and Apply Rule"):
        # Generate YAML rule from user input
        yaml_text = chain.run(input=user_rule)
        st.code(yaml_text, language="yaml")

        # Save YAML temporarily
        rules = yaml.safe_load(yaml_text)

        # Apply rule(s)
        for rule in rules:
            st.subheader(f"Rule: {rule['rule_id']}")
            try:
                condition = rule.get("condition", None)
                check_expr = rule["check"]

                if condition:
                    subset = df.query(condition)
                else:
                    subset = df

                validity = eval(check_expr, {"df": subset, "pd": pd})
                if isinstance(validity, pd.Series):
                    failed_rows = subset[~validity]
                else:
                    failed_rows = subset[[]] if validity else subset

                st.write(f"❌ Failed Rows: {len(failed_rows)}")
                st.dataframe(failed_rows)

            except Exception as e:
                st.error(f"Error applying rule `{rule['rule_id']}`: {e}")
