import streamlit as st
import pandas as pd
import yaml
import os
import google.generativeai as genai
import matplotlib.pyplot as plt
import plotly.express as px
from sentence_transformers import SentenceTransformer, util

# --- Setup ---
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load Gemini API Key from Streamlit secrets
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Load sentence transformer model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Prompt Builder ---
def build_prompt(dq_rule):
    return f"""
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
\"{dq_rule}\"

Return only the YAML block, no explanation.
"""

# --- Gemini Call ---
def generate_yaml_from_gemini(prompt_text):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt_text)
    yaml_raw = response.text.strip()
    if yaml_raw.startswith("```yaml"):
        yaml_raw = yaml_raw.replace("```yaml", "").replace("```", "").strip()
    return yaml_raw

# --- Utility Functions ---
def get_yaml_path(csv_filename):
    base_name = os.path.splitext(csv_filename)[0]
    return os.path.join(UPLOAD_DIR, f"{base_name}.yaml")

def load_rules(csv_filename):
    path = get_yaml_path(csv_filename)
    if not os.path.exists(path):
        return []
    with open(path, "r") as file:
        try:
            rules = yaml.safe_load(file)
            return rules if rules else []
        except yaml.YAMLError:
            return []

def get_combined_text(rule):
    return f"{rule.get('description', '')} {rule.get('condition', '')} {rule.get('check', '')}"

def check_rule_similarity(new_description, existing_rules, threshold=0.8):
    input_embedding = sentence_model.encode(new_description, convert_to_tensor=True)
    existing_texts = [get_combined_text(rule) for rule in existing_rules]
    existing_embeddings = sentence_model.encode(existing_texts, convert_to_tensor=True)
    cosine_scores = util.cos_sim(input_embedding, existing_embeddings)[0]
    return any(score >= threshold for score in cosine_scores)

def append_rule_to_yaml(yaml_text, csv_filename):
    path = get_yaml_path(csv_filename)
    try:
        new_rules = yaml.safe_load(yaml_text)
        if not isinstance(new_rules, list):
            new_rules = [new_rules]

        existing = []
        if os.path.exists(path):
            with open(path, "r") as f:
                existing = yaml.safe_load(f) or []
        if not isinstance(existing, list):
            existing = [existing]

        for new_rule in new_rules:
            if check_rule_similarity(new_rule.get("description", ""), existing):
                return False, "Rule already exists."

        all_rules = existing + new_rules
        with open(path, "w") as f:
            yaml.dump(all_rules, f, sort_keys=False)

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
    all_failed_indices = set()

    for rule in rules:
        try:
            temp_df = df.copy()
            if "condition" in rule:
                mask = eval(rule["condition"], {}, local_env)
                temp_df = temp_df[mask]
            check_mask = eval(rule["check"], {}, {"df": temp_df, "pd": pd})
            failed = temp_df[~check_mask]
            all_failed_indices.update(failed.index.tolist())
            results.append({
                "rule_id": rule["rule_id"],
                "description": rule["description"],
                "violations": len(failed),
                "percentage": round(len(failed) / len(df) * 100, 2)
            })
        except Exception as e:
            all_failed_indices.update(df.index.tolist())
            results.append({
                "rule_id": rule.get("rule_id", "unknown"),
                "description": f"Error: {str(e)}",
                "violations": -1,
                "percentage": 0
            })
    return pd.DataFrame(results), all_failed_indices

# --- Streamlit UI ---
tab1, tab2, tab3, tab4 = st.tabs(["Data Source", "Profiling", "Monitoring Check", "Data Quality Check"])

# --- Tab 1: Upload CSV ---
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
    st.info("Profiling features coming soon.")

# --- Tab 3: Monitoring Rules ---
with tab3:
    subtab1, subtab2, subtab3 = st.tabs(["Add", "Update", "Show"])

    if current_csv_name:
        with subtab1:
            st.subheader("Add New Rule")
            user_rule = st.text_input("Enter a rule in plain English:")
            if st.button("Generate and Add Rule"):
                if user_rule.strip():
                    with st.spinner("Generating YAML using Gemini..."):
                        prompt = build_prompt(user_rule)
                        yaml_rule = generate_yaml_from_gemini(prompt)
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
            st.subheader("Existing Rules")
            rules = load_rules(current_csv_name)
            if rules:
                simplified = pd.DataFrame([{ "rule_id": r["rule_id"], "description": r["description"] } for r in rules])
                st.dataframe(simplified)
            else:
                st.info("No rules defined yet.")
    else:
        st.warning("Upload a CSV file to manage monitoring rules.")

# --- Tab 4: Data Quality Check ---
with tab4:
    st.header("Data Quality Check")
    if current_csv_name and df is not None:
        rules = load_rules(current_csv_name)
        if rules:
            results_df, failed_indices = apply_rules(df, rules)
            total = len(df)
            invalid = len(failed_indices)
            valid = total - invalid

            st.subheader("Validation Summary")
            pie_df = pd.DataFrame({
                'Status': ['Valid', 'Invalid'],
                'Count': [valid, invalid]
            })

            fig = px.pie(
                pie_df,
                names='Status',
                values='Count',
                color='Status',
                color_discrete_map={'Valid': 'green', 'Invalid': 'red'},
                title='Valid vs Invalid Rows',
                hole=0.4
            )
            fig.update_traces(textinfo='label+percent', hoverinfo='label+percent')
            clicked = st.plotly_chart(fig, use_container_width=True)

            selected_status = st.radio("Show Rules for:", ["All", "Valid", "Invalid"], horizontal=True)
            if selected_status == "Valid":
                st.subheader(" Rules with 0 Violations")
                st.dataframe(results_df[results_df['violations'] == 0])
            elif selected_status == "Invalid":
                st.subheader(" Rules with Violations")
                st.dataframe(results_df[results_df['violations'] > 0])
            else:
                st.subheader(" All Rule Check Results")
                st.dataframe(results_df)
        else:
            st.warning("No rules defined to apply.")
    else:
        st.warning("Upload a CSV file to perform data quality checks.")
