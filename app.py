import streamlit as st
import pandas as pd
import yaml
import google.generativeai as genai

# Load Gemini API key
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")

# YAML conversion prompt
GENERATION_PROMPT = """
You are a data quality expert.

Convert the following plain English data quality rule into a YAML-formatted rule.

YAML Format:
- rule_id: <unique_rule_id>
  description: <detailed_description>
  condition: <optional pandas query condition>  # optional
  check: <boolean pandas expression that returns True for valid rows>

Plain English Rule:
{user_rule}
"""

st.title("📊 AI Data Quality Rule Checker")

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
user_rule = st.text_area("Enter a data quality rule in English")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.dataframe(df)

    if user_rule and st.button("Generate and Apply Rule"):
        with st.spinner("Generating YAML rule..."):
            prompt = GENERATION_PROMPT.format(user_rule=user_rule)
            response = model.generate_content(prompt)
            yaml_text = response.text.strip()

            # Clean yaml formatting if markdown artifacts exist
            if yaml_text.startswith("```"):
                yaml_text = yaml_text.strip("`").split("yaml")[-1].strip()

            st.code(yaml_text, language="yaml")

        try:
            rules = yaml.safe_load(yaml_text)
            if not isinstance(rules, list):
                rules = [rules]

            results = []

            for rule in rules:
                rule_id = rule.get("rule_id", "unknown_rule")
                description = rule.get("description", "")
                condition = rule.get("condition", None)
                check_expr = rule.get("check", None)

                if check_expr is None:
                    st.warning(f"No 'check' found in rule {rule_id}")
                    continue

                try:
                    # Evaluate check over full df
                    validity = eval(check_expr, {"df": df.copy(), "pd": pd})

                    # Apply condition filter
                    subset = df.query(condition) if condition else df
                    failed = subset[~validity.loc[subset.index]] if isinstance(validity, pd.Series) else pd.DataFrame()

                    results.append({
                        "rule_id": rule_id,
                        "description": description,
                        "violations": len(failed),
                        "percentage": round(len(failed) / len(subset) * 100, 2) if len(subset) > 0 else 0.0
                    })

                except Exception as e:
                    st.error(f"⚠️ Error applying rule `{rule_id}`: {e}")

            if results:
                st.write("### ✅ Rule Violation Summary")
                st.dataframe(pd.DataFrame(results))
            else:
                st.warning("No valid rules were applied.")

        except Exception as e:
            st.error(f"❌ Failed to parse YAML: {e}")
