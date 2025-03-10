import fitz
from openai import OpenAI
import json
import pandas as pd
import streamlit as st
from io import StringIO
import base64
import os
from datetime import date, timedelta
from PIL import Image
import cv2
import numpy as np
from pytesseract import pytesseract
import re
import json
from openai import OpenAI
import io
import streamlit as st
import pandas as pd
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from openai import AzureOpenAI
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

from flask import Flask, request, jsonify
from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)

# streamlit_url = 'https://invoice-automate.streamlit.app/'

# @app.route('/trigger_streamlit', methods=['GET'])
# def trigger_streamlit():
#     # Sending a GET request to your Streamlit app's public URL (or you can pass query parameters)
    
#     blob_name = request.args.get('blob_name')
#     st.info(f'Blob Name: {blob_name}')
#     # response = requests.get(f'{streamlit_url}?blob_name={blob_name}')

# # @app.route('/extract', methods=['GET'])
# # def extract():
# #     data = request.get_json()
# #     prompt = data['blob_name']
# #     print(prompt)
# #     # response['response']['summary_of_summaries'] = summ_of_summ
# #     return prompt


# if __name__ == '__main__':
#     app.run(debug=False, port=5002)

connection_string = os.environ['connection_string']
input_container = "input"
output_container = "output"
input_blob = 'sample.txt'

# # query_params = st.query_params()
# st.info(f'Query Params: {st.query_params}')
# # Extract the 'blob_name' parameter
# blob_name = st.query_params.get("blob_name", None)

# if not blob_name:
#     blob_name = "invoice5.PDF"
# st.info(blob_name)

# st.set_page_config(page_title="Invoice OCR App", layout="wide", initial_sidebar_state="collapsed")
# st.image("tp.png",use_column_width=True)

# st.markdown("""
#     <style>
#     .stButton>button {
#
#
#         border-radius: 5px;
#         border: 2px solid #BB1CCC; /* Set the border color to pink */
#         padding: 10px 20px;
#         color: #BB1CCC
#     }
#     </style>""", unsafe_allow_html=True)

endpoint = os.environ['endpoint']
key = os.environ['azure_key']

# col1, col2 = st.columns(2)
api_key = os.environ['openai_api_key']
api_version = os.environ['api_version']
azure_endpoint = os.environ['azure_endpoint']

client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=azure_endpoint,

)


def upload_to_blob(csv_data, connection_string, output_container, output_blob):
    # Create a BlobServiceClient using the connection string
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    # Get a ContainerClient for the container
    container_client = blob_service_client.get_container_client(output_container)

    # Create a BlobClient for the specific blob
    blob_client = container_client.get_blob_client(output_blob)

    # Convert the CSV data to a byte stream
    byte_data = io.BytesIO(csv_data.encode('utf-8'))  # Encoding to bytes

    # Upload the byte stream to Azure Blob Storage
    blob_client.upload_blob(byte_data, overwrite=True)
    st.info(f"CSV file '{output_blob}' uploaded successfully to {output_container}.")


def download_file_from_blob(connection_string, input_container):
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    container_client = blob_service_client.get_container_client(input_container)

    # Create a BlobServiceClient
    blobs = container_client.list_blobs()

    # Find the most recent blob by comparing the last modified times
    most_recent_blob = None
    latest_modified_time = None

    for blob in blobs:
        if most_recent_blob is None or blob['last_modified'] > latest_modified_time:
            most_recent_blob = blob
            latest_modified_time = blob['last_modified']

    st.info(latest_modified_time)

    if most_recent_blob:
        # Get the blob client for the most recent blob
        input_blob = most_recent_blob['name']
        blob_client = container_client.get_blob_client(input_blob)

        # Download the blob and read it into a byte stream
        blob_data = blob_client.download_blob()
        file_data = io.BytesIO(blob_data.readall())

        st.info(input_blob)

        return file_data, input_blob

    return None

def df_date(df):
    try:
        return pd.to_datetime(df["Invoice Date (DD-MMM-YYYY)"])
    except Exception as e:
        return pd.to_datetime(df["Invoice Date"])

def highlight_text(row):
    color = 'color: red' if row['Remark'] in ["Data Mismatch Detected", "Payment Date Mismatch as per SoW"] else ''
    return [color if col in ['Extracted Values', 'Remark'] else '' for col in row.index]


def highlight_mismatches(vendor_row, master_df):
    vendor_name = vendor_row['Vendor Name']
    matched_row = master_df[master_df['Vendor Name'] == vendor_name]

    status_list = []
    master_data = []
    columns_to_compare = ['Vendor Address', 'Vendor VAT ID', 'Vendor Country', 'Currency', 'Bill to VAT ID']

    if not matched_row.empty:
        matched_row = matched_row.iloc[0]  # Get the first match
        # columns_to_compare = ['Vendor Address', 'Vendor VAT ID', 'Vendor Country', 'Currency', 'Due By', 'Bill to VAT ID']

        # Check for mismatches
        for col in columns_to_compare:
            vendor_value = vendor_row[col]
            master_value = matched_row[col]

            if pd.isna(vendor_value) and pd.isna(master_value):
                status_list.append("Matched")  # Both are NaN
                master_data.append("")  # No master data for matched rows
            elif vendor_value != master_value:
                status_list.append("Data Mismatch Detected")
                master_data.append(master_value)  # Add master data for mismatched columns
            else:
                status_list.append("Matched")
                master_data.append("")  # No master data for matched rows

        return status_list, master_data

    return "", ""


def safe_load_json(json_string):
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        # st.error(f"Error in JSON parsing: {e}")
        return None


def clean_temp_folder(directory_path):
    if os.path.exists(directory_path):
        # List all items in the given directory
        items = os.listdir(directory_path)

        # Check if the list is not empty
        if items:
            for item in items:
                item_path = os.path.join(directory_path, item)
                # Check if it's a file and not a directory
                if os.path.isfile(item_path):
                    os.remove(item_path)  # Delete the file
            print("All files deleted.")
        else:
            print("No files found in the directory.")
    else:
        print("Directory does not exist.")


def pdf_to_images(pdf_document, output_folder):
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        dpi = 300
        image = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
        pil_image = Image.frombytes("RGB", [image.width, image.height], image.samples)

        image_filename = f"{output_folder}/page_{page_num + 1}.png"
        pil_image.save(image_filename, "PNG")

        print(f"Page {page_num + 1} saved as {image_filename}")
    pdf_document.close()


def sort_pngfiles():
    folder_path = 'temp_images'  # Replace with your folder path
    files = os.listdir(folder_path)
    png_files = [file for file in files if file.endswith('.png')]
    png_files.sort(key=lambda f: int(f.split('_')[1].split('.')[0]))
    return png_files


def clean_country_of_origin(row):
    # Using strip to remove leading/trailing spaces after replacement
    cleaned_country = row['country_of_origin'].replace(row['port_of_loading'], "").strip()
    # Further clean-up to handle potential multiple spaces within the string
    cleaned_country = ' '.join(cleaned_country.split())
    return cleaned_country


def ap_invoice(text):
    response = client.chat.completions.create(
        # model="gpt-4-1106-preview",
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": f"Here is the complete document,All numbers should be in UK format and use list if find multiple values,:{text}"
            },
            {
                "role": "user",
                "content": "Extract following fields 'Invoice Header', 'Vendor Name','Vendor Address', 'Vendor Country', 'Vendor VAT ID', 'Bill to Name','Bill to Address', 'Bill to Country', 'Bill to VAT ID', 'Due By', 'Ship to Name','Ship to Address', 'Ship to Country', 'Invoice Number', 'Invoice Date (DD-MMM-YYYY)','PO Number', 'Currency', 'NET', 'VAT', 'VAT Rate', 'Gross Amount','Local Curr', 'Fx Rate', 'Net (Local Curr)', 'VAT (Local Curr)','Gross (Local Curr)', 'Payment Term', 'Invoice Item Description', 'VAT Verbage'"
                # "content": "Extract following fields 'Entity Number', 'SAP Document Number', 'Invoice Header', 'Vendor Name','Vendor Address', 'Vendor Country', 'Vendor VAT ID', 'Bill to Name','Bill to Address', 'Bill to Country', 'Bill to VAT ID', 'Due By', 'Ship to Name','Ship to Address', 'Ship to Country', 'Invoice Number', 'Invoice Date (DD-MMM-YYYY)','PO Number', 'Currency', 'NET', 'VAT', 'VAT Rate', 'Gross Amount','Local Curr', 'Fx Rate', 'Net (Local Curr)', 'VAT (Local Curr)','Gross (Local Curr)', 'Payment Term', 'Invoice Item Description', 'VAT Verbage'"
            },
            {
                "role": "assistant",
                "content": "put in json format,fill blank where field not found, Juniper Networks is not vendor name look for other company,Invoice item description is generally found on 1st page,.Translate all data in English.Numbers should be clean"
            }

        ],
        temperature=0,
        max_tokens=4095,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content


# boe_eval = pd.read_csv('BOE_final.csv')
# boe_eval['be_no'] = boe_eval['be_no'].astype('str')
# cache_boes = boe_eval['be_no'].tolist()


def extract_entrysummary(text):
    response = client.chat.completions.create(
        # model="gpt-4-1106-preview",
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": f"Here is the complete document:{text}"
            },
            {
                "role": "user",
                "content": "Extract these fields only, if multiple add as a list, Page title,Subtitle ,PORT address, BE No, BE Date, Country of Origin, PORT OF LOADING,Country of Cossignment, Invoice Number, Invoice Amount,CUR, GSTIN/TYPE,CB CODE,TOT.ASS VAL,IGST"
            },
            {
                "role": "assistant",
                "content": """put in json format,fill blank where field not found. except, invoice number and inv amount, all are scalers not list only.For example format must be like this:
                                    {
                                         "page_title":"BILL OF ENTRY FOR HOME CONSUMPTION",
                                         "subtitle":"PART - I -BILL OF ENTRY SUMMARY",
                                         "port_address":"ACC BANGALORE BENGALURU INTERNATIONAL AIRPORT BILL OF ENTRY FOR HOME CONSUMPTION",
                                         "be_no":"8768951",
                                         "be_date":"15/11/2023",
                                         "GSTIN/TYPE":"29AAECJ1345A1ZZ/G",
                                         "cb_code":"AAACZ3050ACH002",
                                         "country_of_origin":["Austria"],
                                         "port_of_loading":"HONG KONG",
                                         "country_of_consignment":"HONG KONG",
                                         "tot.ass_val":"17060",
                                         "igst":"5346.10"
                                         "invoice_number":[5001081942,5001081943],
                                         "inv.amt":[43.44,134.01]
                                         "cur":"USD"
                                    }


                                    """
            }

        ],
        temperature=0,
        max_tokens=4095,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0)
    return response.choices[0].message.content


def create_dataframe_from_json(json_obj):
    # Check if any value in the JSON object is a list
    if any(isinstance(value, list) for value in json_obj.values()):
        # If there's at least one list, use the JSON object directly
        df = pd.DataFrame(json_obj)
    else:
        # If all values are scalar, wrap the JSON object in a list
        df = pd.DataFrame([json_obj])

    return df


def analyze_read(path):
    # sample document
    # formUrl = "/Users/nitingupta/Desktop/SNOW/Visiongpt/Vendor invoice -2/2810_2700000315_7940.PDF"

    document_analysis_client = DocumentAnalysisClient(
        endpoint=endpoint, credential=AzureKeyCredential(key)
    )

    # path_to_sample_documents = "/Users/nitingupta/Desktop/SNOW/Visiongpt/Vendor_Invoice/2810_2023_5100007574_COMPUT_1157993AA.PDF"
    with open(path, "rb") as f:
        poller = document_analysis_client.begin_analyze_document(
            "prebuilt-read", document=f, locale="en-US"
        )
    result = poller.result()

    #
    # print ("Document contains content: ", result.content)

    return result.content


def extract_invoice(text):
    response = client.chat.completions.create(
        # model="gpt-4-1106-preview",
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": f"Here is the complete document:{text}"
            },
            {
                "role": "user",
                "content": "Extract these fields only, Page title, Subtitle,A.INVOICE 1.S.NO,CTH,DESCRIPTION,QUANTITY,UNIT PRICE,AMOUNT,Invoice Number, Invoice Date, ASS. Value"
            },
            {
                "role": "assistant",
                "content": """put in json format,fill blank where field not found.For example format must be like this
                                    {
                                        "page_title":"BILL OF ENTRY FOR HOME CONSUMPTION",
                                        "subtitle":"PART - II - INVOICE & VALUATION DETAILS (Invoice 1/2)",
                                        "invoice_s.no":"2",
                                        "invoice_invoice_number":"5001085781",
                                        "invoice_invoice_date:"10-NOV-23",
                                        "invoice_ass.value":"2995402.06",
                                        "invoice_cth":[85444299,85444299],
                                        "invoice_description":["MX2K-MPC11E PART OF ROUTER (MPC/LINE CARD)(MX2K-MPC11E) 40X100GE ZT BASED LINE CARD FOR MX2K.","JNP10K-RE1 PART OF ROUTER (ROUTING ENGINE)(JNP10K-RE1) JNP10K RE, SPARE."],
                                        "invoice_unit_price":[2000,1500],
                                        "invoice_quantity":[1,1],
                                        "invoice_amount":[2000,1500]


                                    }


                            """
            }

        ],
        temperature=0,
        max_tokens=4095,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0)
    return response.choices[0].message.content


def extract_duties(text):
    response = client.chat.completions.create(
        # model="gpt-4-1106-preview",
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": f"Here is the complete document:{text}"
            },
            {
                "role": "user",
                "content": "Extract these fields only, Page title, Subtitle,INVSNO,CTH,ITEM DESCRIPTION,C.QTY,S.QTY,ASSESS VALUE, TOTAL DUTY,1.BCD AMOUNT,2.ACD Amount,3.SWS AMOUNT,4.SAD, 5.IGST AMOUNT,6.G.CESS AMOUNT"
            },
            {
                "role": "assistant",
                "content": """put in json format,fill blank where field not found.For example format must be like this
                                      {
                                        "page_title":"BILL OF ENTRY FOR HOME CONSUMPTION",
                                        "subtitle":"PART III-DUTIES",
                                        "duties_invsno":"1",
                                        "duties_cth":"85444299",
                                        "duties_item_description":"PART ID: CBL-EX-PWR-C13-IN POWER CABLE- INDIA (10A/250V.2.5M)",
                                        "duties_c.qty":"4",
                                        "duties_s.qty":"10",
                                        "duties_assess_value":"4176.3",
                                        "duties_total_duty":"1293.8",
                                        "duties_bcd_amount":"476.6",
                                        "duties_acd_amount":"0",
                                        "duties_sws_amount":"1125",
                                        "duties_sad_amount":"0",
                                        "duties_igst_amount":"834.4",
                                        "duties_g.cess_amount":"0"}
                                      """
            }

        ],
        temperature=0,
        max_tokens=4095,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0)
    return response.choices[0].message.content


def extract_packaging(text):
    response = client.chat.completions.create(
        # model="gpt-4-1106-preview",
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": f"Here is the complete document:{text}"
            },
            {
                "role": "user",
                "content": "Extract these fields only, Invoice number,Courier Reference,Ship from Name,Ship from Country,	Entry Number,Date,Delivery Name,Delivery Address,Delivery Country,Delivery country -EU/Non-EU,Incoterms,Sold to, Customer VAT ID,On Hand,MAWB Number,SO number,	HAWB Number,Customs Status,	Customer Reference,	Total Packages,	Signature (Y / N),	License plate number?"
            },
            {
                "role": "assistant",
                "content": """put in json format,fill blank where field not found.For example format must be like this:
                                            {
                                            'invoice_number':'5000608142',
                                            'courier_reference':'8400061807',
                                            'ship_from_name':'Expeditors International Italia Sr',
                                            'ship_from_country':'Netherlands',
                                            'entry_number':'8001478396',
                                            'date':'06/12/2019',
                                            'delivery_name':'RETELIT S.P.A',
                                            'delivery_address':'VIA VIVIANI, 8, 20124 MILANO MI',
                                            'delivery_country':'Denmark',
                                            'Delivery country -EU/Non-EU':'EU',
                                            'incoterms':'DDP DEST',
                                            'sold_to':'TEXOR S.R.L.',
                                            'customer_vat':'12862140154',
                                            'on_hand':'F484864528',
                                            'mawb_number':'8001478396',
                                            'hawb_number':'16136483',
                                            'customs_status':'C',
                                            'customer_reference':'JH-251-RTL-2019',
                                            'total_package':'1',
                                            'signature_y_n':Y,
                                            'license_plate_number':'26 BARI'


                                            }






                                      """
            }

        ],
        temperature=0,
        max_tokens=4095,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0)
    return response.choices[0].message.content


# with col1:
    # st.write("Image Upload")

    # st.write(uploaded_file)
    # option = st.selectbox(
    #     "Choose the type of Invoice",
    #     ("AP Invoice", "Packaging list", "BOE")
    #
    # )
    # with st.sidebar:
    #     selected_option = st.selectbox("Select Comparison Type", ["Compare with Excel", "Compare with PDF"])

# with col2:
    # st.write("Prompt")
uploaded_file, input_blob = download_file_from_blob(connection_string, input_container)
# show_details = st.toggle("Advanced Options", value=False)

# if show_details:
#     # Text input for additional details about the image, shown only if toggle is True
#     placeholder_text = "By Default the smart extracter will format data in US date and number format. Add instructions in natural language to overwrite these options with your preferred options.For example : format date in UK format."
#     additional_details = st.text_area(
#         "Add any additional details or context about the image here:",
#         disabled=not show_details,
#         placeholder=placeholder_text
#     )
# analyze_button = st.button("Extract", type="secondary")
if uploaded_file is not None and api_key:
    # with st.spinner("Analysing the file ..."):
    document_analysis_client = DocumentAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))

    pdf_path = uploaded_file
    poller = document_analysis_client.begin_analyze_document("prebuilt-read", document=pdf_path.read(),
                                                             locale="en-US")
    result = poller.result()
    text = result.content
    json_string = ap_invoice(text)
    data_dict = safe_load_json(json_string.split('```json')[1].split('```')[0].strip())
    del data_dict['Payment Term']
    # st.write(data_dict)
    invoice_date = df_date(data_dict)
    df = pd.json_normalize(data_dict)

    master_df = pd.read_excel('Vendor_Master.xlsx')

    # Read PDF
    if "Apple" in data_dict['Vendor Name']:
        input_pdf = open('Sample_SoW_invoice5.pdf', 'rb')
    else:
        input_pdf = open('Sample_SoW_invoice4.pdf', 'rb')
    input_poller = document_analysis_client.begin_analyze_document("prebuilt-read",
                                                                   document=input_pdf.read(),
                                                                   locale="en-US")
    input_result = input_poller.result()
    input_text = input_result.content
    input_json_string = ap_invoice(input_text)
    input_data_dict = safe_load_json(input_json_string.split('```json')[1].split('```')[0].strip())
    input_df = pd.json_normalize(input_data_dict)

    # retrieve Payemnt terms from PDF
    match = input_df['Payment Term'].values[0]
    days = re.search(r'\b\d+\b', match)
    number_of_days = days.group()
    # st.info(f"Payment should be made within {number_of_days} days of Invoice as per SoW")

    status_list, master_data_list = highlight_mismatches(df.iloc[0], master_df)

    if status_list == "":
        csv = input_df.to_csv(index=False)
        output_blob = f"{input_blob.split('.')[0]}_output.csv"
        
        upload_to_blob(csv, connection_string, output_container, output_blob)
        print("No Matching records found in Database")
    else:
        transposed_df = df.T
        transposed_df.columns = ['Extracted Values']
        transposed_df['Attribute'] = transposed_df.index

        if number_of_days:
            due_date = invoice_date + timedelta(days=int(number_of_days))
            status_list.append(" ")
            status_df = pd.DataFrame(status_list, columns=['Remark'],
                                     index=['Vendor Address', 'Vendor VAT ID', 'Vendor Country', 'Currency',
                                            'Bill to VAT ID', 'Due By'])
            master_data_list.append(due_date.date())
        else:
            status_df = pd.DataFrame(status_list, columns=['Remark'],
                                     index=['Vendor Address', 'Vendor VAT ID', 'Vendor Country', 'Currency',
                                            'Bill to VAT ID'])

        status_df['Expected Value'] = master_data_list
        status_df.reset_index(inplace=True)
        status_df.rename(columns={'index': 'Attribute'}, inplace=True)

        # Merge status information into transposed_df
        merged_df = transposed_df.merge(status_df, on='Attribute', how='left')

        # Reorder columns
        merged_df = merged_df[['Attribute', 'Extracted Values', 'Remark', 'Expected Value']]
        merged_df.loc[merged_df['Remark'] == 'Matched', 'Remark'] = ''
        merged_df.fillna('', inplace=True)
        # merged_df = merged_df[merged_df['Attribute'] != 'Payment Term']

        date_row = merged_df[merged_df['Attribute'] == 'Due By']
        if not date_row.empty:
            row_index = date_row.index[0]

            if pd.to_datetime(merged_df.at[row_index, 'Extracted Values']).date() \
                    != merged_df.at[row_index, 'Expected Value']:
                merged_df.at[row_index, 'Remark'] = "Payment Date Mismatch as per SoW"

        # Apply the highlighting function to the DataFrame
        styled_df = merged_df.style.apply(highlight_text, axis=1)

        csv = merged_df.to_csv(index=False)
        output_blob = f"{input_blob.split('.')[0]}_output.csv"
        
        upload_to_blob(csv, connection_string, output_container, output_blob)
