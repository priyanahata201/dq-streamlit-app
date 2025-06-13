import streamlit as st
import pandas as pd
import yaml
import re
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=st.secrets["GEMINI_API_KEY"]
)

# Prompt to convert plain English rule into YAML
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
st.title("📊 AI-Powered Data Quality Validator")
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
user_rule = st.text_area("Describe your data quality rule (in English):")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of your data:")
    st.dataframe(df)

    if user_rule and st.button("Generate and Apply Rule"):
        with st.spinner("🧠 Thinking and applying rule..."):
            try:
                # Generate YAML rule
                yaml_output = chain.run({"input": user_rule})
                st.subheader("🧾 Generated YAML Rule")
                st.code(yaml_output, language="yaml")

                # Clean markdown formatting if present
                yaml_clean = re.sub(r"```.*?\n(.*?)```", r"\1", yaml_output, flags=re.DOTALL).strip()

                try:
                    rules = yaml.safe_load(yaml_clean)
                except yaml.YAMLError as e:
                    st.error(f"❌ Invalid YAML: {e}")
                    rules = []

                result_rows = []

                for rule in rules:
                    rule_id = rule.get("rule_id", "unknown_rule")
                    description = rule.get("description", "")
                    condition = rule.get("condition")
                    check = rule.get("check")

                    try:
                        # Apply condition if any
                        subset = df.query(condition) if condition else df

                        # Replace 'df' with 'subset' so check works properly
                        check_expr = check.replace("df", "subset")
                        validity = eval(check_expr, {"subset": subset, "pd": pd})

                        if isinstance(validity, pd.Series):
                            failed = subset[~validity]
                            violations = len(failed)
                        else:
                            violations = 0 if validity else len(subset)

                        percent = round((violations / len(df)) * 100, 2) if len(df) > 0 else 0

                        result_rows.append({
                            "rule_id": rule_id,
                            "description": description,
                            "violations": violations,
                            "percentage": percent
                        })

                    except Exception as e:
                        st.warning(f"⚠️ Failed to apply rule {rule_id}: {e}")

                if result_rows:
                    st.subheader("📈 Validation Summary")
                    result_df = pd.DataFrame(result_rows)
                    st.dataframe(result_df)
                else:
                    st.warning("⚠️ No rules applied successfully.")

            except Exception as e:
                st.error(f"❌ Error: {e}")
