from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.plugins.save_files_as_artifacts_plugin import SaveFilesAsArtifactsPlugin
from google.adk.tools import FunctionTool, ToolContext

import pathlib
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow

import asyncio
from email.message import EmailMessage
from email import message_from_bytes

from base64 import urlsafe_b64decode

from datetime import datetime

import os
from dotenv import load_dotenv

from google import genai
from google.genai import types

import base64

import json
import numpy as np
import pandas as pd

load_dotenv()

GOOGLE_GENAI_USE_VERTEXAI = os.getenv('GOOGLE_GENAI_USE_VERTEXAI')
GOOGLE_CLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')
GOOGLE_CLOUD_LOCATION = os.getenv('GOOGLE_CLOUD_LOCATION')
GEMINI_MODEL_ID = os.getenv('GEMINI_MODEL_ID')

KEYFILE_PATH = os.getcwd() + "/invoiceprocessing/credentials/gcp-oauth.keys.json"
GMAIL_CREDENTIALS_PATH = os.getcwd() + "/invoiceprocessing/credentials/.gmail-server-credentials.json"
GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]

def authenticate_and_save(app: str = "gmail"):
    
    if(app == "gmail"):
        if os.path.exists(GMAIL_CREDENTIALS_PATH):
            return
        
        flow = InstalledAppFlow.from_client_secrets_file(KEYFILE_PATH, GMAIL_SCOPES)
        creds = flow.run_local_server(port=8080)
        pathlib.Path(os.path.dirname(GMAIL_CREDENTIALS_PATH)).mkdir(parents=True, exist_ok=True)
        with open(GMAIL_CREDENTIALS_PATH, "w") as f:
            f.write(creds.to_json())
        print(f"Credentials saved to {GMAIL_CREDENTIALS_PATH}")

# -- Gmail Client --
def get_gmail_client():
    authenticate_and_save("gmail")
    creds = Credentials.from_authorized_user_file(GMAIL_CREDENTIALS_PATH, GMAIL_SCOPES)
    return build("gmail", "v1", credentials=creds)

def get_current_user_email_id():
    """Get current user's email address"""
    client = get_gmail_client()
    profile = client.users().getProfile(userId='me').execute()
    emailId = profile.get("emailAddress", "")

    return {
        "content": {
            "emailId": emailId,
        }
    }

async def send_email(sender_id: str, recipient_id: str, subject: str, message: str,) -> dict:
    """Creates and sends an email message"""
    client = get_gmail_client()
    message_obj = EmailMessage()
    message_obj.set_content(message)
    
    message_obj['To'] = recipient_id
    message_obj['From'] = sender_id
    message_obj['Subject'] = subject

    encoded_message = base64.urlsafe_b64encode(message_obj.as_bytes()).decode()
    create_message = {'raw': encoded_message}
    
    send_message = await asyncio.to_thread( 
        client.users().messages().send(userId="me", body=create_message).execute
    )
    return {"status": "success", "message_id": send_message["id"]}



genai_client = genai.Client(vertexai=GOOGLE_GENAI_USE_VERTEXAI,
    project=GOOGLE_CLOUD_PROJECT, location=GOOGLE_CLOUD_LOCATION)

async def get_pdf_from_artifact(
    filename: str, 
    tool_context: ToolContext
    ) -> bytes:
    """
    Loads a specific PDF from saved artifacts and returns its byte content

    Args:
        filename: Name of PDF artifact file to load
        tool_context: context object provided by ADK framework

    Returns:
        Byte content (bytes) of PDF file

    Raises:
        ValueError: If the artifact is not found or is not a PDF
        RuntimeError: For unexpected storage or other errors
    """
    try:
        # Load the specified artifact
        pdf_artifact = await tool_context.load_artifact(filename=filename)

        # Check if the artifact was found and contains data
        if (pdf_artifact and hasattr(pdf_artifact, 'inline_data') and 
            pdf_artifact.inline_data):
            # Validate that it is a PDF
            if pdf_artifact.inline_data.mime_type == "application/pdf":
                print(f"✅ Successfully loaded PDF artifact '{filename}'.")
                # Extract and eturn the raw byte content
                pdf_bytes = pdf_artifact.inline_data.data
                return pdf_bytes
            else:
                # Raise an error if the file type is wrong
                raise ValueError(
                    f"Artifact '{filename}' is not a PDF. "
                    f"Found type: '{pdf_artifact.inline_data.mime_type}'."
                )
        else:
            # Raise an error if the artifact wasn't found or was empty
            raise ValueError(f"Artifact '{filename}' not found or is empty.")

    except ValueError as e:
        # This will catch errors from load_artifact or the checks above
        print(f"❌ Error loading artifact: {e}")
        raise e
    except Exception as e:
        # Handle other potential storage or unexpected errors
        raise RuntimeError(f"An unexpected error occurred: {e}")

async def get_table_schema_from_pdf(
    filename: str,
    tool_context: ToolContext
    ) -> str:
    """Returns table schema from given PDF to be used in data extraction

    Args:
        tool_context: context object provided by ADK framework

    Returns:
        str: schema to be used in PDF data extraction
    """

    try:
        pdf_data = await get_pdf_from_artifact(filename, tool_context)
    except ValueError as e:
        return str(e)

    document = types.Part.from_bytes(
        data=pdf_data,
        mime_type="application/pdf"
    )

    text = types.Part.from_text(text=
        """
        Looking closely at the tables or other such structured data in the PDF,
        create a JSON output as per the below format with the respective fields.
        If any of the field is not present in the PDF, return None for that field.

        Below is the json format example with the respective fields.
        {
        "vendor_name": "ABC Technologies Pvt Ltd",
        "vendor_gstin": "27AAECS1234F1Z9",
        "invoice_number": "INV-2345",
        "invoice_date": "2025-01-02",
        "place_of_supply": "Maharashtra",
        "currency": "INR",
        "line_items": [
            {
            "description": "Cloud Services",
            "hsn": "9983",
            "quantity": 1,
            "unit_price": 100000,
            "line_total": 100000
            }
        ],
        "subtotal": 100000,
        "taxes": {
            "cgst": 9000,
            "sgst": 9000,
            "igst": 0
        },
        "total_tax": 18000,
        "invoice_total": 118000,
        "confidence_score": 0.94
        }
        
        Return that JSON schema by itself in a format that can be used in
        downstream data extraction.
        """)

    contents = [
        types.Content(
        role="user",
        parts=[text,
            document
            ]
        )
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature = 0,
        top_p = 1,
        seed = 0,
        max_output_tokens = 65535,
        response_mime_type = "application/json",
        thinking_config=types.ThinkingConfig(
            thinking_budget=128,
        )
    )

    response = genai_client.models.generate_content(
        model = GEMINI_MODEL_ID,
        contents = contents,
        config = generate_content_config,
        )

    return response.text


async def save_structured_response(structured_response: str, file_name: str, tool_context: ToolContext, file_type: str = "csv") -> str:
    """
    Saves structured response (e.g. from Gemini w/ controlled generation)
    as CSV or JSON artifact in tool_context

    Args:
        structured_response: response from Gemini in text 
        file_name: name of file to save (without ".csv" or ".json" in name)
        tool_context: context object provided by ADK framework containing user
          content

    Returns:
        A success message with the name & version of the CSV/JSON file saved as
          an artifact
    """

    parsed_json = json.loads(structured_response)

    # Convert structured response to pandas df
    df = pd.DataFrame(parsed_json)

    file_name_with_ext = f"{file_name}.{file_type}"

   # Use the file_type parameter to determine the output format
    if file_type.lower() == "csv":
        output_bytes = df.to_csv(index=False).encode('utf-8')
        mime_type = "text/csv"
        extension = "csv"
    elif file_type.lower() == "json":
        output_bytes = df.to_json(orient="records", indent=2).encode('utf-8')
        mime_type = "application/json"
        extension = "json"
    else:
        raise ValueError(f"Unsupported file type: '{file_type}'. "
            "Please use 'csv' or 'json'.")

    # Save the content as an artifact
    version = await tool_context.save_artifact(
        filename=f"{file_name_with_ext}", 
        artifact=types.Part.from_bytes(data=output_bytes, mime_type=mime_type)
    )

    return f"Saved data to artifact {file_name_with_ext} w/ version {version}."


async def generate_data_from_pdf_and_schema(
    pdf_file_name: str,
    schema: str, 
    tool_context: ToolContext,
    output_file_name: str = "extracted_data",
    output_file_type: str = "csv"    
    ):
    """Extracts data from PDF with specified schema

    Args:
        pdf_file_name: name of PDF file to read in   
        schema: JSON schema for use in PDF data extraction
        tool_context: context object provided by ADK framework containing user
          content

    Returns:
        A success message with the name & version of the CSV/JSON file saved as
          an artifact
    """

    try:
        pdf_data = await get_pdf_from_artifact(pdf_file_name, tool_context)
    except ValueError as e:
        return str(e)

    document = types.Part.from_bytes(
        data=pdf_data,
        mime_type="application/pdf"
    )

    text = types.Part.from_text(text=
        f"""Extract all data from the included PDF into the following schema:
        {schema}
        """)

    contents = [
        types.Content(
        role="user",
        parts=[text,
            document
            ]
        )
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature = 0,
        top_p = 1,
        seed = 0,
        max_output_tokens = 65535,
        response_mime_type = "application/json",
        thinking_config=types.ThinkingConfig(
            thinking_budget=128,
        )
    )

    response = genai_client.models.generate_content(
        model = GEMINI_MODEL_ID,
        contents = contents,
        config = generate_content_config,
        )

    response_text = response.text.replace('\n', ' ')   

    file_save_result = await save_structured_response(
        structured_response=response_text,
        file_name=output_file_name,
        tool_context=tool_context,
        file_type=output_file_type
        )

    return file_save_result


invoiceprocessing = Agent(
    name="invoiceprocessing",
    model=GEMINI_MODEL_ID,
    description="""
        Invoice processing agent to extract data from provided PDF into structured format
        """,
    instruction="""
        You are a highly accurate Indian invoice data extraction agent.

        Your task is to extract structured data from Indian GST invoices.
        You must strictly follow Indian GST rules and the output schema.

        You are NOT allowed to:
        - Guess missing values
        - Fabricate numbers
        - Infer totals that are not explicitly present

        If a field is not found with high confidence, return null.

        
        Supported invoice types:
        - Indian GST tax invoices
        - Credit notes and debit notes
        - Service invoices
        - Goods invoices

        Not supported invoice types:
        - Proforma invoices
        - Quotes
        - Delivery challans

        Extraction constraints are as below:
        - Currency must be INR
        - Dates must be ISO format: DD-MM-YYYY
        - Numbers must be numeric (no commas, no symbols)
        - GST amounts must be separated as CGST, SGST, IGST
        - Output must be valid JSON only
        - Do NOT add commentary or explanation

        The field extraction must be done with high confidence. 
        If a field is not found with high confidence, return null.

        vendor_name:
        - Extract the registered business name
        - Ignore branding slogans or logos

        vendor_gstin:
        - Must be 15 characters
        - Example format: 27ABCDE1234F1Z5
        - Do NOT extract PAN or CIN here
        - If this is not present in the PDF, return a message that GSTIN is either not present or format is incorrect.Hence the invoice cannot be processed.

        invoice_number:
        - Look for labels: "Invoice No", "Tax Invoice No", "Bill No"
        - Do NOT use Order No or PO No
        - If this is not present in the PDF, return a message that invoice number is not present and cannot be processed.

        invoice_date:
        - Look for "Invoice Date", "Bill Date"
        - Convert DD/MM/YYYY or DD-MM-YYYY to ISO format
        - If this is not present in the PDF, return a message that invoice date is not present and cannot be processed.

        place_of_supply:
        - Extract Indian state name
        - Normalize abbreviations (MH → Maharashtra)

        Line Items:
        For each line item:
        - description: Product or service name
        - hsn: Extract HSN or SAC if present
        - quantity: Numeric only
        - unit_price: Price per unit before tax
        - line_total: quantity × unit_price (ONLY if explicitly printed)

        taxes:
        - cgst: CGST amount
        - sgst: SGST amount
        - igst: IGST amount

        subtotal:
        - Sum of line totals

        total_tax:
        - Sum of CGST, SGST, IGST

        invoice_total:
        - subtotal + total_tax
        - if the invoice total is less than 5000 INR, return a message that invoice total is less than 5000 INR and hence is auto approved and processed
        - if the invoice total is greater than 5000 INR, return a message that invoice total is greater than 5000 INR and hence needs manual approval. The invoice will be processed only after manual approval.

        confidence_score:
        - 0.95 if all fields are extracted with high confidence
        - 0.85 if some fields are extracted with low confidence
        - 0.75 if most fields are extracted with low confidence
        - If the concfidence score is less than 0.95, then return a message that invoice data extraction agent is not confident and will be processed only after manual approval.
        
        Extract GST amounts exactly as printed:
        - CGST amount
        - SGST amount
        - IGST amount

        Do NOT:
        - Calculate GST
        - Split combined GST unless explicitly shown

    """,
    output_key = "invoice_data_extraction_agent_output",
    tools=[
        get_table_schema_from_pdf,
        # get_current_user_email_id,
        # send_email
        ]
)

root_agent = invoiceprocessing

app = App(
    name='invoiceprocessing',
    root_agent=root_agent,
    plugins=[SaveFilesAsArtifactsPlugin()],
)