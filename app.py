import streamlit as st

st.set_page_config(
    page_title="AWS Cost Optimization Dashboard",
    page_icon="💸",
    layout="wide"
)

st.title("💡 AWS Cost Optimization Assistant")
st.sidebar.success("👈 Choose a page above to get started")

st.markdown(
    """
Welcome to the **AWS Cost Optimization Dashboard** 🚀

This app helps you inspect predicted AWS costs, get LLM-driven insights, and optionally trigger remediation actions (for example, stopping an instance) via Lambda.

Key areas:
- 💬 **Chatbot** — Ask cost-related questions, request summaries, or ask the assistant to identify high-cost resources. The assistant receives a compact preview of the current flagged predictions so replies are grounded in your data.
- 💰 **Cost Analyzer** — Visualize predicted costs in a full table and a compact summary. Select rows and trigger Lambda actions to take remediation steps.
- ⚙️ **Settings / Permissions** — Make sure the runtime environment has appropriate AWS permissions (S3 read/write, Lambda invoke, and Bedrock/LLM access if used). Avoid hardcoding credentials; use IAM roles, profiles, or environment variables.

Quick start (local):
1. Install requirements: pip install -r requirements.txt
2. Run: 
    ```bash
    streamlit run app.py
    ```
3. Ensure AWS credentials are available (e.g. AWS_PROFILE or environment variables) or run from an environment with an IAM role.

Configuration pointers:
- S3 bucket and object keys for cost/predictions are set via environment variables or config in the pages; confirm they point to the expected CSVs.
- If using an LLM (Bedrock / OpenAI), ensure API keys/permissions are configured and not embedded in source.

Suggested checks if things fail:
- Data not loading: verify S3 bucket/key and that the caller has s3:GetObject permission.
- Lambda actions not working: verify lambda:InvokeFunction permission and that the payload matches the target Lambda's expected schema.
- LLM errors: verify network access, API key, model id, and that response parsing matches the model's response format.

Security note:
- Do not commit AWS credentials or API keys into source control. Use environment variables or IAM roles.
- Test any automatic remediation in a safe environment before applying to production.

Need more help? Check your repository README or contact your cloud admin for required IAM roles and model access.
"""
)