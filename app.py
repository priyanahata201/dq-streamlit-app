import streamlit as st
import pandas as pd
import yaml
import os
import google.generativeai as genai
import re

# Load Gemini API key
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")

# Streamlit UI
st.title("📊 AI-Driven Data Quality Checker (Gemini 1.5 Flash)")

prompt_template = """
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

RULE:
{text}
"""

uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])
user_rule = st.text_area("Describe your data quality rule (in English):", "")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of your data:")
    st.dataframe(df)

    if user_rule and st.button("Generate and Apply Rule"):
        prompt = prompt_template.format(text=user_rule)
        response = model.generate_content(prompt)
        raw_output = response.text.strip()

        # 🔍 Clean out Markdown, triple backticks, and leading junk
        cleaned_output = re.sub(r"```(?:yaml)?", "", raw_output)
        cleaned_output = cleaned_output.strip().strip("`")

        try:
            rules = yaml.safe_load(cleaned_output)
            if not isinstance(rules, list):
                rules = [rules]

            st.subheader("📄 Generated YAML Rule:")
            st.code(cleaned_output, language="yaml")

            summary_rows = []

            for rule in rules:
                rule_id = rule.get("rule_id", "unknown_rule")
                description = rule.get("description", "")
                condition = rule.get("condition", None)
                check_expr = rule["check"]

                try:
                    subset = df.query(condition) if condition else df
                    validity = eval(check_expr, {"df": subset, "pd": pd})
                    failed_rows = subset[~validity] if isinstance(validity, pd.Series) else pd.DataFrame()

                    violations = len(failed_rows)
                    percent = round((violations / len(df)) * 100, 2) if len(df) else 0.0

                    summary_rows.append({
                        "rule_id": rule_id,
                        "description": description,
                        "violations": violations,
                        "percentage": percent
                    })

                except Exception as e:
                    st.error(f"⚠️ Error applying rule `{rule_id}`: {e}")

            if summary_rows:
                st.write("### ✅ Rule Summary:")
                st.dataframe(pd.DataFrame(summary_rows))
            else:
                st.info("No valid rules were applied.")

        except Exception as e:
            st.error(f"❌ Failed to parse YAML: {e}")
            st.text_area("Raw Gemini Output (for debugging)", value=raw_output, height=300)
