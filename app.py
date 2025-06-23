import streamlit as st
import pandas as pd
import yaml
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# --- Configuration ---
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# OpenAI API key from secrets
OPENAI_API_KEY = st.secrets["MY_API_KEY"]

# --- Prompt Template for LangChain ---
template = """
You are a data quality expert.

Convert the following plain English data quality rule into a YAML-formatted rule for a Python-based data quality engine.

YAML Format:
- rule_id: <unique_rule_id>
  description: <detailed_description>
  condition: <optional pandas query condition>  # optional
  check: <boolean pandas expression that returns True for valid rows>

EXAMPLES :
- rule_id: dq_kyc_number_length_check
  description: "If KYCType is IDP6 then KYCNumber length should be 12"
  condition: "df['KYCType'] == 'IDP6'"
  check: "df['KYCNumber'].astype(str).str.len() == 12"

- rule_id: dq_null_values_check
  description: "Column 'Name' should not be null"
  check: "df['Name'].notnull()"

- rule_id: dq_incorrect_date
  description: "Replace '1900-01-01' with NULL"
  check: "df['Date'] != '1900-01-01'"

- rule_id: dq_date_after_2020
  description: "Date should be after the year 2020"
  check: "pd.to_datetime(df['Date']).dt.year > 2020"

Rule:
"{dq_rule}"

Return only the YAML block, no explanation.
"""
prompt = PromptTemplate.from_template(template)
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, model="gpt-4")
chain = prompt | llm

# --- Functions ---

def get_yaml_path(csv_filename):
    base_name = os.path.splitext(csv_filename)[0]
    return os.path.join(UPLOAD_DIR, f"{base_name}.yaml")

def load_rules(csv_filename):
    path = get_yaml_path(csv_filename)
    if not os.path.exists(path):
        return []
    with open(path, "r") as file:
        return yaml.safe_load(file)

def append_rule_to_yaml(yaml_text, csv_filename):
    path = get_yaml_path(csv_filename)
    try:
        new_rules = yaml.safe_load(yaml_text)
        if not isinstance(new_rules, list):
            new_rules = [new_rules]
        with open(path, "a") as f:
            yaml.dump(new_rules, f, sort_keys=False)
        return True, "Rule added successfully!"
    except yaml.YAMLError as e:
        return False, f"YAML parse error: {e}"
    except Exception as e:
        return False, f"Failed to write rule: {e}"

def delete_rule(rule_id, csv_filename):
    path = get_yaml_path(csv_filename)
    try:
        rules = load_rules(csv_filename)
        new_rules = [r for r in rules if r.get("rule_id") != rule_id]
        with open(path, "w") as f:
            yaml.dump(new_rules, f, sort_keys=False)
        return True, "Rule deleted successfully."
    except Exception as e:
        return False, f"Error deleting rule: {e}"

def apply_rules(df, rules):
    results = []
    local_env = {"df": df, "pd": pd}
    for rule in rules:
        try:
            temp_df = df.copy()
            if "condition" in rule:
                mask = eval(rule["condition"], {}, local_env)
                temp_df = temp_df[mask]
            check_mask = eval(rule["check"], {}, {"df": temp_df, "pd": pd})
            failed = temp_df[~check_mask]
            results.append({
                "rule_id": rule["rule_id"],
                "description": rule["description"],
                "violations": len(failed),
                "percentage": round(len(failed) / len(df) * 100, 2)
            })
        except Exception as e:
            results.append({
                "rule_id": rule.get("rule_id", "unknown"),
                "description": f"Error: {str(e)}",
                "violations": -1,
                "percentage": 0
            })
    return pd.DataFrame(results)

# --- Streamlit Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["Data Source", "Profiling", "Monitoring Check", "Data Quality Check"])

# --- Tab 1: Data Source ---
with tab1:
    st.header("Upload CSV Data")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    current_csv_name = None
    df = None

    if uploaded_file is not None:
        current_csv_name = uploaded_file.name
        csv_path = os.path.join(UPLOAD_DIR, current_csv_name)
        with open(csv_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File '{current_csv_name}' uploaded and saved.")

    uploaded_files = os.listdir(UPLOAD_DIR)
    csv_files = [f for f in uploaded_files if f.endswith(".csv")]

    if csv_files:
        current_csv_name = csv_files[0]
        csv_path = os.path.join(UPLOAD_DIR, current_csv_name)
        df = pd.read_csv(csv_path)
        st.info(f"Loaded file: {current_csv_name}")
        st.dataframe(df)

        if st.button("Remove Uploaded File"):
            os.remove(csv_path)
            st.warning("Uploaded file removed. Refresh to clear view.")
            current_csv_name = None
            df = None
    else:
        st.warning("No file uploaded.")

# --- Tab 2: Profiling ---
with tab2:
    st.header("Profiling")

# --- Tab 3: Monitoring Check ---
with tab3:
    subtab1, subtab2, subtab3 = st.tabs(["Add", "Update", "Show"])

    if current_csv_name:
        with subtab1:
            st.subheader("Add New Rule")
            user_rule = st.text_input("Enter a rule in plain English:")
            if st.button("Generate and Add Rule"):
                if user_rule.strip():
                    with st.spinner("Generating YAML..."):
                        yaml_rule = chain.invoke({"dq_rule": user_rule}).content
                        st.code(yaml_rule, language="yaml")
                        success, message = append_rule_to_yaml(yaml_rule, current_csv_name)
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
                else:
                    st.warning("Please enter a rule.")

        with subtab2:
            st.subheader("Delete Existing Rule")
            existing_rules = load_rules(current_csv_name)
            rule_ids = [r.get("rule_id") for r in existing_rules]
            if rule_ids:
                selected_rule = st.selectbox("Select a rule to delete", rule_ids)
                if st.button("Delete Rule"):
                    success, msg = delete_rule(selected_rule, current_csv_name)
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)
            else:
                st.info("No rules available to delete.")

        with subtab3:
            st.subheader("Violations Summary")
            rules = load_rules(current_csv_name)
            if df is not None and rules:
                result_df = apply_rules(df, rules)
                st.dataframe(result_df)
                csv_out = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Report as CSV",
                    data=csv_out,
                    file_name="dq_report.csv",
                    mime="text/csv"
                )
            elif df is not None:
                st.info("No rules defined for this file.")
            else:
                st.warning("No data loaded.")
    else:
        st.warning("Upload a CSV file to manage monitoring rules.")

# --- Tab 4: Data Quality Check ---
with tab4:
    st.header("Data Quality Check")
