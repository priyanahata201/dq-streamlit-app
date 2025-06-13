import streamlit as st
import pandas as pd
import yaml
import re
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain

# Load Gemini model using Streamlit secrets
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=st.secrets["GEMINI_API_KEY"]
)

# Prompt template to convert English rule to YAML
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

prompt = PromptTemplate(input_variables=["input"], template=template)
chain = LLMChain(llm=llm, prompt=prompt)

# UI
st.set_page_config(page_title="AI Data Quality Checker", layout="centered")
st.title("📊 AI Data Quality Validator")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
user_rule = st.text_area("Write your data quality rule in English:")

if uploaded_file and user_rule:
    df = pd.read_csv(uploaded_file)

    if st.button("Generate & Apply Rule"):
        with st.spinner("Applying AI-powered rule..."):
            try:
                # Step 1: Generate YAML
                yaml_output = chain.run({"input": user_rule})

                # Step 2: Strip markdown formatting if present
                if "```" in yaml_output:
                    yaml_clean = re.sub(r"```(?:yaml)?\s*(.*?)```", r"\1", yaml_output, flags=re.DOTALL).strip()
                else:
                    yaml_clean = yaml_output.strip()

                # Step 3: Parse YAML
                try:
                    rules = yaml.safe_load(yaml_clean)
                except yaml.YAMLError as e:
                    st.error(f"YAML Parsing Failed: {e}")
                    rules = []

                result_rows = []

                # Step 4: Apply each rule
                for rule in rules:
                    rule_id = rule.get("rule_id", "unknown_rule")
                    description = rule.get("description", "")
                    condition = rule.get("condition")
                    check = rule.get("check")

                    try:
                        subset = df.query(condition) if condition else df
                        check_expr = check.replace("df", "subset")
                        validity = eval(check_expr, {"subset": subset, "pd": pd})

                        if isinstance(validity, pd.Series):
                            failed_rows = subset[~validity]
                            violations = len(failed_rows)
                        else:
                            violations = 0 if validity else len(subset)

                        percentage = round((violations / len(df)) * 100, 2) if len(df) > 0 else 0.0

                        result_rows.append({
                            "rule_id": rule_id,
                            "description": description,
                            "violations": violations,
                            "percentage": percentage
                        })

                    except Exception as e:
                        result_rows.append({
                            "rule_id": rule_id,
                            "description": f"Error applying rule: {e}",
                            "violations": "N/A",
                            "percentage": "N/A"
                        })

                # Step 5: Display Results Only
                if result_rows:
                    result_df = pd.DataFrame(result_rows)
                    st.dataframe(result_df, use_container_width=True)
                else:
                    st.warning("No valid rules applied.")

            except Exception as e:
                st.error(f"Error: {e}")
