import streamlit as st
import pandas as pd
import yaml
import google.generativeai as genai
import re

# Load Gemini API key
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")

# Title
st.title("📊 AI-Powered Data Quality Rule Checker")

uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])
user_rule = st.text_area("📝 Describe your data quality rule (in plain English):", "")

def clean_yaml_response(text):
    # Remove markdown artifacts like ```yaml or ``` from output
    return re.sub(r"^```.*?[\r\n]+|```$", "", text.strip(), flags=re.MULTILINE)

def get_yaml_from_gemini(prompt):
    full_prompt = f"""
You are a data quality assistant.

Convert the following user input into a Python YAML-based rule format.

Required YAML format:
- rule_id: <unique_rule_id>
  description: <rule explanation>
  condition: <pandas query string>  # optional
  check: <boolean pandas expression using df>

Input:
{prompt}

Return only valid YAML format.
"""
    response = model.generate_content(full_prompt)
    return clean_yaml_response(response.text)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of your data")
    st.dataframe(df)

    if user_rule and st.button("🚀 Generate and Apply Rule"):
        with st.spinner("Generating and applying rule..."):
            try:
                yaml_text = get_yaml_from_gemini(user_rule)
                try:
                    rules = yaml.safe_load(yaml_text)
                    if not isinstance(rules, list):
                        raise ValueError("YAML is not a list of rules.")
                except Exception as parse_err:
                    st.error(f"❌ Failed to parse YAML: {parse_err}")
                    st.code(yaml_text)
                    rules = []

                summary_rows = []

                for rule in rules:
                    rule_id = rule.get("rule_id", "unknown_rule")
                    desc = rule.get("description", "")
                    condition = rule.get("condition", None)
                    check_expr = rule.get("check", None)

                    try:
                        subset = df.query(condition) if condition else df
                        validity = eval(check_expr, {"df": subset, "pd": pd})
                        failed_rows = subset[~validity] if isinstance(validity, pd.Series) else subset if not validity else subset[[]]

                        summary_rows.append({
                            "rule_id": rule_id,
                            "description": desc,
                            "violations": len(failed_rows),
                            "percentage": round((len(failed_rows) / len(subset)) * 100, 2) if len(subset) > 0 else 0.0
                        })

                        st.subheader(f"🛠 Rule: {rule_id}")
                        st.write(desc)
                        st.write(f"❌ Failed Rows: {len(failed_rows)}")
                        st.dataframe(failed_rows)

                    except Exception as rule_error:
                        st.warning(f"⚠️ Error applying rule {rule_id}: {rule_error}")

                if summary_rows:
                    st.markdown("## ✅ Summary of Results")
                    st.dataframe(pd.DataFrame(summary_rows))

            except Exception as e:
                st.error(f"Something went wrong: {e}")
