import streamlit as st
import pandas as pd
import yaml
import re
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain

# Use Gemini 1.5 Flash
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=st.secrets["GEMINI_API_KEY"])

# Prompt template to convert English rule into YAML
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
st.title("📊 AI-Driven Data Quality Checker (Gemini 1.5 Flash)")

uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])
user_rule = st.text_area("Describe your data quality rule (in English):", "")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of your data:")
    st.dataframe(df)

    if user_rule and st.button("Generate and Apply Rule"):
        with st.spinner("🔧 Generating rule and applying checks..."):
            try:
                yaml_text = chain.run({"input": user_rule})
                st.subheader("📄 Generated Rule (YAML):")
                st.code(yaml_text, language="yaml")

                # Remove markdown code block if present
                yaml_text_clean = re.sub(r"```.*?\n(.*?)```", r"\1", yaml_text, flags=re.DOTALL).strip()

                try:
                    rules = yaml.safe_load(yaml_text_clean)
                except yaml.YAMLError as e:
                    st.error(f"❌ Failed to parse YAML: {e}")
                    rules = []

                result_rows = []

                for rule in rules:
                    rule_id = rule.get("rule_id", "unknown_rule")
                    description = rule.get("description", "")
                    condition = rule.get("condition")
                    check_expr = rule.get("check")

                    try:
                        subset = df.query(condition) if condition else df
                        validity = eval(check_expr, {"df": subset, "pd": pd})
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
                        st.warning(f"⚠️ Error applying rule {rule_id}: {e}")

                if result_rows:
                    st.subheader("📊 Rule Results Summary")
                    result_df = pd.DataFrame(result_rows)
                    st.dataframe(result_df)
                else:
                    st.warning("⚠️ No valid rules were applied.")

            except Exception as e:
                st.error(f"❌ Failed to process rule: {e}")
