import streamlit as st
import pandas as pd
import yaml
import os
import google.generativeai as genai
import matplotlib.pyplot as plt
import plotly.express as px
from sentence_transformers import SentenceTransformer, util
import streamlit.components.v1 as components
from ydata_profiling import ProfileReport
import torch
import re

# --- Setup ---
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
device = 'cuda' if torch.cuda.is_available() else 'cpu'
sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# --- Prompt Builder ---
def build_prompt(dq_rule, column_names):
    return f"""
You are a data quality expert.

Convert the following plain English data quality rule into a YAML-formatted rule for a Python-based data quality engine.

Please use below columns for applying condition based on if user prompted spelling mistake or blank space between while specifying column name. {column_names} Please use similarity search and use most appropriate column for Rule during check.

YAML Format:
- rule_id: <unique_rule_id>
  description: <detailed_description>
  condition: <optional pandas query condition>
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

Rule: \"{dq_rule}\"

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

def check_rule_similarity(new_description, existing_rules, new_check=None, threshold=0.6):
    if not existing_rules:
        return False
    input_embedding = sentence_model.encode(new_description, convert_to_tensor=True)
    existing_texts = [get_combined_text(rule) for rule in existing_rules]
    existing_embeddings = sentence_model.encode(existing_texts, convert_to_tensor=True)
    cosine_scores = util.cos_sim(input_embedding, existing_embeddings)[0]

    if any(score >= threshold for score in cosine_scores):
        return True

    if new_check:
        for rule in existing_rules:
            if rule.get("check", "").strip() == new_check.strip():
                return True

    return False

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

def extract_column_name(check_expr):
    """
    Tries to extract column name from a check expression of form df['colname']
    """
    match = re.search(r"df\[['\"]([^'\"]+)['\"]\]", check_expr)
    if match:
        return match.group(1)
    return "Unknown"

def apply_rules(df, rules, dataset_name=None):
    results = []
    local_env = {"df": df, "pd": pd}
    all_failed_indices = set()
    total_rows = len(df)

    for rule in rules:
        try:
            temp_df = df.copy()
            if "condition" in rule and rule["condition"]:
                mask = eval(rule["condition"], {}, local_env)
                temp_df = temp_df[mask]
            check_mask = eval(rule["check"], {}, {"df": temp_df, "pd": pd})
            failed = temp_df[~check_mask]
            all_failed_indices.update(failed.index.tolist())

            num_invalid = len(failed)
            perc_invalid = round((num_invalid / total_rows) * 100, 2) if total_rows > 0 else 0
            perc_valid = round(100 - perc_invalid, 2)

            results.append({
                "rule_id": rule.get("rule_id", "unknown"),
                "description": rule.get("description", ""),
                "dataset_name": dataset_name or "Unknown",
                "column_name": extract_column_name(rule.get("check", "")),
                "num_records_scanned": total_rows,
                "num_invalid_records": num_invalid,
                "percentage_invalid": perc_invalid,
                "percentage_valid": perc_valid
            })
        except Exception as e:
            all_failed_indices.update(df.index.tolist())
            results.append({
                "rule_id": rule.get("rule_id", "unknown"),
                "description": f"Error: {str(e)}",
                "dataset_name": dataset_name or "Unknown",
                "column_name": "Error",
                "num_records_scanned": total_rows,
                "num_invalid_records": -1,
                "percentage_invalid": 0,
                "percentage_valid": 0
            })

    return pd.DataFrame(results), all_failed_indices

def suggest_rules_from_data(df):
    suggestions = []
    for col in df.columns:
        col_lower = col.lower()
        if pd.api.types.is_numeric_dtype(df[col]):
            if "age" in col_lower:
                suggestions.append({
                    "rule_id": f"dq_{col}_age_min_check",
                    "description": f"{col} should be greater than 18",
                    "check": f"df['{col}'] > 18"
                })
            elif "salary" in col_lower or "amount" in col_lower:
                suggestions.append({
                    "rule_id": f"dq_{col}_positive_check",
                    "description": f"{col} should be greater than 0",
                    "check": f"df['{col}'] > 0"
                })
            else:
                suggestions.append({
                    "rule_id": f"dq_{col}_non_negative_check",
                    "description": f"{col} should be non-negative",
                    "check": f"df['{col}'] >= 0"
                })
        elif pd.api.types.is_string_dtype(df[col]):
            suggestions.append({
                "rule_id": f"dq_{col}_not_null",
                "description": f"{col} should not be null or empty",
                "check": f"df['{col}'].notnull() & (df['{col}'].astype(str).str.strip() != '')"
            })
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            suggestions.append({
                "rule_id": f"dq_{col}_after_2000",
                "description": f"{col} should be after 01-01-2000",
                "check": f"pd.to_datetime(df['{col}']) > pd.to_datetime('2000-01-01')"
            })
    return suggestions

# NEW FUNCTION (for requirement 1)
def suggest_rules_from_profiling(df):
    suggestions = []
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            suggestions.append({
                "rule_id": f"dq_{col}_not_null",
                "description": f"{col} should not be null",
                "check": f"df['{col}'].notnull()"
            })
    return suggestions

def delete_uploaded_file(filename):
    csv_path = os.path.join(UPLOAD_DIR, filename)
    yaml_path = get_yaml_path(filename)
    if os.path.exists(csv_path):
        os.remove(csv_path)
    if os.path.exists(yaml_path):
        os.remove(yaml_path)

# --- Tab Layout ---
tab1, tab2, tab3, tab4 = st.tabs(["Data Source", "Profiling", "Monitoring Check", "Data Quality Check"])

# --- Tab 1: Data Source ---
with tab1:
    subtab1, subtab2, subtab3 = st.tabs(["Upload File", "Select File", "Delete Files"])

    if "all_files" not in st.session_state:
        st.session_state["all_files"] = sorted([f for f in os.listdir(UPLOAD_DIR) if f.endswith(".csv")])

    with subtab1:
        st.subheader("Upload New CSV")
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], label_visibility="collapsed")
        if uploaded_file:
            save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state["current_csv_name"] = uploaded_file.name
            st.session_state["all_files"] = sorted([f for f in os.listdir(UPLOAD_DIR) if f.endswith(".csv")])
            st.success(f"Uploaded and selected `{uploaded_file.name}`.")

    with subtab2:
        st.subheader("Uploaded Files")
        all_files = st.session_state.get("all_files", [])
        if all_files:
            selected_file = st.radio("Select a file:", all_files,
                                     index=all_files.index(st.session_state.get("current_csv_name"))
                                     if "current_csv_name" in st.session_state and st.session_state["current_csv_name"] in all_files else 0)
            if selected_file:
                st.session_state["current_csv_name"] = selected_file
                st.success(f"`{selected_file}` selected.")
        else:
            st.info("No uploaded files available.")

    with subtab3:
        st.subheader("Delete Files")
        all_files = st.session_state.get("all_files", [])
        if all_files:
            files_to_delete = st.multiselect("Select files to delete:", options=all_files)
            if st.button("Delete Selected Files"):
                for file in files_to_delete:
                    delete_uploaded_file(file)
                    if st.session_state.get("current_csv_name") == file:
                        del st.session_state["current_csv_name"]
                st.session_state["all_files"] = sorted([f for f in os.listdir(UPLOAD_DIR) if f.endswith(".csv")])
                st.rerun()  # <<< NEW LINE for requirement 2
        else:
            st.info("No uploaded files to delete.")

# --- Tab 2: Profiling ---
with tab2:
    st.header("Data Profiling Report")
    if "current_csv_name" in st.session_state:
        current_csv = st.session_state["current_csv_name"]
        profile_path = os.path.join(UPLOAD_DIR, "profile_report.html")
        generate_clicked = st.button("Generate Profile")

        if generate_clicked:
            df = pd.read_csv(os.path.join(UPLOAD_DIR, current_csv))
            with st.spinner("Generating profiling report..."):
                profile = ProfileReport(df, title="Data Profile Report", explorative=True)
                profile.to_file(profile_path)
                st.session_state["profile_generated"] = True

        if st.session_state.get("profile_generated") and os.path.exists(profile_path):
            with open(profile_path, 'r', encoding='utf-8') as f:
                html_report = f.read()
            components.html(html_report, height=1000, scrolling=True)
            with open(profile_path, "rb") as f:
                st.download_button("Download Report", f, file_name="data_profile_report.html", mime="text/html")
        elif not st.session_state.get("profile_generated"):
            st.info("Click 'Generate Profile' to create a data profiling report.")
    else:
        st.warning("Please select a file from Data Source tab.")


# --- Tab 3: Monitoring Rules ---
with tab3:
    subtab1, subtab2, subtab3 = st.tabs(["Add", "Update", "Show"])
    if "current_csv_name" in st.session_state:
        current_csv = st.session_state["current_csv_name"]
        df = pd.read_csv(os.path.join(UPLOAD_DIR, current_csv))

        # --- Add New Rule ---
        with subtab1:
            st.subheader("Add New Rule")
            user_rule = st.text_input("Enter a rule in plain English:")
            if st.button("Generate and Add Rule"):
                if user_rule.strip():
                    prompt = build_prompt(user_rule, ", ".join(df.columns))
                    yaml_rule = generate_yaml_from_gemini(prompt)
                    st.code(yaml_rule, language="yaml")

                    from yaml import safe_load
                    new_rules = safe_load(yaml_rule)
                    if not isinstance(new_rules, list):
                        new_rules = [new_rules]
                    existing_rules = load_rules(current_csv)

                    duplicate_found = False
                    for new_rule in new_rules:
                        new_desc = new_rule.get("description", "")
                        if check_rule_similarity(new_desc, existing_rules):
                            st.error("Rule already exists.")
                            duplicate_found = True
                            break

                    if not duplicate_found:
                        all_rules = existing_rules + new_rules
                        with open(get_yaml_path(current_csv), "w") as f:
                            yaml.dump(all_rules, f, sort_keys=False)
                        st.success("Rule added successfully!")

            st.markdown("---")
            st.subheader("Suggested Rules based on Columns")
            suggested_rules = suggest_rules_from_data(df)

            if suggested_rules:
                selected_suggestions = []
                for i, rule in enumerate(suggested_rules):
                    if st.checkbox(f"{rule['description']}", key=f"suggested_{i}"):
                        selected_suggestions.append(rule)

                if st.button("Add Selected Suggested Rules"):
                    existing_rules = load_rules(current_csv)
                    to_add = []
                    duplicates = []
                    for rule in selected_suggestions:
                        if check_rule_similarity(rule.get("description", ""), existing_rules, rule.get("check", "")):
                            duplicates.append(rule.get("description", "Unnamed Rule"))
                        else:
                            to_add.append(rule)
                    if to_add:
                        all_rules = existing_rules + to_add
                        with open(get_yaml_path(current_csv), "w") as f:
                            yaml.dump(all_rules, f, sort_keys=False)
                        st.success("Selected suggested rules added.")
                    if duplicates:
                        st.warning("The following rules already exist and were not added:\n" + "\n".join([f"- {desc}" for desc in duplicates]))
                    if not to_add and not duplicates:
                        st.info("No new rules to add or all duplicates.")

                if st.button("Add All Suggested Rules"):
                    existing_rules = load_rules(current_csv)
                    to_add = []
                    duplicates = []
                    for rule in suggested_rules:
                        if check_rule_similarity(rule.get("description", ""), existing_rules):
                            duplicates.append(rule.get("description", "Unnamed Rule"))
                        else:
                            to_add.append(rule)
                    if to_add:
                        all_rules = existing_rules + to_add
                        with open(get_yaml_path(current_csv), "w") as f:
                            yaml.dump(all_rules, f, sort_keys=False)
                        st.success("All suggested rules added.")
                    if duplicates:
                        st.warning("The following rules already exist and were not added:\n" + "\n".join([f"- {desc}" for desc in duplicates]))
                    if not to_add and not duplicates:
                        st.info("No new rules to add or all duplicates.")
            else:
                st.info("No column-based rule suggestions available.")

            st.markdown("---")
            st.subheader("Suggested Rules based on Profiling")
            profiling_suggestions = suggest_rules_from_profiling(df)

            if profiling_suggestions:
                selected_profiling = []
                for i, rule in enumerate(profiling_suggestions):
                    if st.checkbox(f"{rule['description']}", key=f"profile_suggested_{i}"):
                        selected_profiling.append(rule)

                if st.button("Add Selected Profiling Rules"):
                    existing_rules = load_rules(current_csv)
                    to_add = []
                    duplicates = []
                    for rule in selected_profiling:
                        if check_rule_similarity(rule.get("description", ""), existing_rules, rule.get("check", "")):
                            duplicates.append(rule.get("description", "Unnamed Rule"))
                        else:
                            to_add.append(rule)
                    if to_add:
                        all_rules = existing_rules + to_add
                        with open(get_yaml_path(current_csv), "w") as f:
                            yaml.dump(all_rules, f, sort_keys=False)
                        st.success("Selected profiling rules added.")
                    if duplicates:
                        st.warning("The following rules already exist and were not added:\n" + "\n".join([f"- {desc}" for desc in duplicates]))
                    if not to_add and not duplicates:
                        st.info("No new rules to add or all duplicates.")

                if st.button("Add All Profiling Rules"):
                    existing_rules = load_rules(current_csv)
                    to_add = []
                    duplicates = []
                    for rule in profiling_suggestions:
                        if check_rule_similarity(rule.get("description", ""), existing_rules):
                            duplicates.append(rule.get("description", "Unnamed Rule"))
                        else:
                            to_add.append(rule)
                    if to_add:
                        all_rules = existing_rules + to_add
                        with open(get_yaml_path(current_csv), "w") as f:
                            yaml.dump(all_rules, f, sort_keys=False)
                        st.success("All profiling rules added.")
                    if duplicates:
                        st.warning("The following rules already exist and were not added:\n" + "\n".join([f"- {desc}" for desc in duplicates]))
                    if not to_add and not duplicates:
                        st.info("No new rules to add or all duplicates.")
            else:
                st.info("No profiling-based rule suggestions available.")

        # --- Update / Delete Existing Rule ---
        with subtab2:
            st.subheader("Edit Existing Rule Description")
            rules = load_rules(current_csv)
            if rules:
                rule_options = {r.get("rule_id"): r.get("description") for r in rules}
                selected_rule_id = st.selectbox("Select Rule ID to Edit", list(rule_options.keys()), key="edit_select")
                current_desc = rule_options.get(selected_rule_id, "")
                new_desc = st.text_area("Edit Description", value=current_desc, key="edit_text")

                if st.button("Update Rule"):
                    if not new_desc.strip():
                        st.error("Description cannot be empty.")
                    else:
                        # Delete old rule
                        updated_rules = [r for r in rules if r.get("rule_id") != selected_rule_id]
                        with open(get_yaml_path(current_csv), "w") as f:
                            yaml.dump(updated_rules, f, sort_keys=False)

                        # Use AI to generate new rule
                        prompt = build_prompt(new_desc, ", ".join(df.columns))
                        yaml_rule = generate_yaml_from_gemini(prompt)
                        st.code(yaml_rule, language="yaml")

                        try:
                            new_rules = yaml.safe_load(yaml_rule)
                            if not isinstance(new_rules, list):
                                new_rules = [new_rules]
                        except yaml.YAMLError:
                            st.error("Generated YAML is invalid.")
                            # Restore old rules
                            with open(get_yaml_path(current_csv), "w") as f:
                                yaml.dump(rules, f, sort_keys=False)
                            st.warning("Please correct the description and try again.")

                        # Check for duplicates
                        duplicate_found = False
                        for new_rule in new_rules:
                            new_desc_check = new_rule.get("description", "")
                            new_check_expr = new_rule.get("check", "")
                            if check_rule_similarity(new_desc_check, updated_rules, new_check_expr):
                                duplicate_found = True
                                break

                        if duplicate_found:
                            st.error("Duplicate rule exists. Update aborted. No changes made.")
                            # Restore old rules
                            with open(get_yaml_path(current_csv), "w") as f:
                                yaml.dump(rules, f, sort_keys=False)
                        else:
                            # Save updated rules with new rule
                            final_rules = updated_rules + new_rules
                            with open(get_yaml_path(current_csv), "w") as f:
                                yaml.dump(final_rules, f, sort_keys=False)
                            st.success("Rule updated successfully!")
                            rules = load_rules(current_csv)
            else:
                st.info("No rules available to edit.")

            st.markdown("---")
            st.subheader("Delete Existing Rule")
            rules = load_rules(current_csv)
            rule_ids = [r.get("rule_id") for r in rules]
            if rule_ids:
                selected_rule = st.selectbox("Select a rule to delete", rule_ids, key="delete_select")
                if st.button("Delete Rule"):
                    new_rules = [r for r in rules if r.get("rule_id") != selected_rule]
                    with open(get_yaml_path(current_csv), "w") as f:
                        yaml.dump(new_rules, f, sort_keys=False)
                    st.success("Rule deleted successfully.")
            else:
                st.info("No rules available to delete.")

        # --- Show Existing Rules ---
        with subtab3:
            st.subheader("Existing Rules")
            rules = load_rules(current_csv)
            if rules:
                st.dataframe(pd.DataFrame([
                    {"rule_id": r["rule_id"], "description": r["description"]}
                    for r in rules
                ]))
            else:
                st.info("No rules defined yet.")
    else:
        st.warning("Please select a file from Data Source tab.")

# --- Tab 4: Data Quality Check ---
with tab4:
    st.header("Data Quality Check")
    if "current_csv_name" in st.session_state:
        current_csv = st.session_state["current_csv_name"]
        df = pd.read_csv(os.path.join(UPLOAD_DIR, current_csv))
        rules = load_rules(current_csv)
        if rules:
            results_df, failed_indices = apply_rules(df, rules, dataset_name=current_csv)
            total = len(df)
            invalid = len(failed_indices)
            valid = total - invalid
            st.subheader("Validation Summary")
            pie_df = pd.DataFrame({
                'Status': ['Valid', 'Invalid'],
                'Count': [valid, invalid]
            })
            fig = px.pie(pie_df, names='Status', values='Count', color='Status',
                         color_discrete_map={'Valid': 'green', 'Invalid': 'red'},
                         title='Valid vs Invalid Rows', hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
            status_option = st.radio("Show Rules for:", ["All", "Valid", "Invalid"], horizontal=True)
            if status_option == "Valid":
                st.dataframe(results_df[results_df['num_invalid_records'] == 0])
            elif status_option == "Invalid":
                st.dataframe(results_df[results_df['num_invalid_records'] > 0])
            else:
                st.dataframe(results_df)

            # NEW: Bar chart of rules with violations > 0
            st.subheader("Rules with Violations - Percentage Chart")
            violated_rules_df = results_df[results_df['num_invalid_records'] > 0]
            if not violated_rules_df.empty:
                fig_bar = px.bar(
                    violated_rules_df,
                    x='rule_id',
                    y='percentage_invalid',
                    title='Violation Percentage by Rule',
                    labels={'rule_id': 'Rule ID', 'percentage_invalid': 'Violation %'},
                    color='percentage_invalid',
                    color_continuous_scale='reds'
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No rules with violations to show in bar chart.")
        else:
            st.warning("No rules defined to apply.")
    else:
        st.warning("Please select a file from Data Source tab.")
