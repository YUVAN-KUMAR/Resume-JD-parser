import docx
import PyPDF2
import csv
from json import JSONDecodeError
import os
import json
import openai
from fastapi import HTTPException
#import mammoth
import subprocess
import pdfplumber
import fitz  # PyMuPDF
from pdfminer.high_level import extract_text
from pdf2image import convert_from_path
import pytesseract
from typing import Optional
"""
from openai import OpenAI
#from dotenv import load_dotenv

from google.generativeai import types
import openai
from fastapi import HTTPException
import google.generativeai as genai
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables")

#genai.configure(api_key=GEMINI_API_KEY)
#model = genai.GenerativeModel("gemini-2.0-flash-exp")"""



def convert_docx_to_text(file_path: str) -> str:
    doc = docx.Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text


def is_single_page_pdf(pdf_path: str) -> bool:
    try:
        doc = fitz.open(pdf_path)
        return len(doc) == 1
    except Exception as e:
        raise ValueError(f"Failed to open PDF: {e}")

# Step 2: Check if PDF is scanned (no embedded text)
def is_scanned_pdf(pdf_path: str) -> bool:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            first_page = pdf.pages[0]
            text = first_page.extract_text()
            return not bool(text or first_page.extract_words())  # Fallback to word detection
    except Exception as e:
        print(f"[WARNING] Error checking if scanned: {e}")
        return True  # Assume scanned if check fails

# Step 3: Detect column layout
def detect_columns_pymupdf(pdf_path: str) -> str:
    try:
        doc = fitz.open(pdf_path)
        column_counts = []

        for page in doc:
            blocks = page.get_text("blocks")
            if not blocks:
                column_counts.append("single-column")
                continue
            x_positions = sorted(set(block[0] for block in blocks if block[4].strip()))  # Ignore empty blocks
            column_counts.append("multi-column" if len(x_positions) > 1 else "single-column")

        if all(count == "single-column" for count in column_counts):
            print("[INFO] Detected Single-Column PDF.")
            return "single-column"
        elif all(count == "multi-column" for count in column_counts):
            print("[INFO] Detected Multi-Column PDF.")
            return "multi-column"
        else:
            print("[INFO] Detected Mixed Layout PDF.")
            return "mixed"
    except Exception as e:
        print(f"[WARNING] Error detecting columns: {e}")
        return "mixed"  # Default to mixed if detection fails

# Step 4: Extract text from scanned PDFs using OCR
def extract_text_ocr(pdf_path: str) -> Optional[str]:
    try:
        print("[INFO] Extracting text using OCR...")
        images = convert_from_path(pdf_path)
        text = "".join(pytesseract.image_to_string(img) + "\n" for img in images)
        return text if text.strip() else None
    except Exception as e:
        print(f"[ERROR] OCR failed: {e}")
        return None

# Step 5: Extract text using PyMuPDF (Best for single-column)
def extract_text_pymupdf(pdf_path: str) -> str:
    try:
        doc = fitz.open(pdf_path)
        return "".join(page.get_text("text") + "\n" for page in doc)
    except Exception as e:
        raise ValueError(f"PyMuPDF extraction failed: {e}")

# Step 6: Extract text using pdfminer.six (Best for multi-column)
def extract_text_pdfminer(pdf_path: str) -> str:
    try:
        return extract_text(pdf_path)
    except Exception as e:
        raise ValueError(f"pdfminer extraction failed: {e}")

# Step 7: Main function - Process PDF based on its layout
def convert_pdf_to_text(file_path: str) -> Optional[str]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    # Step 1: Check single-page or multi-page
    try:
        is_single = is_single_page_pdf(file_path)
        print(f"[INFO] {'Single-Page' if is_single else 'Multi-Page'} PDF detected.")
    except Exception as e:
        print(f"[ERROR] {e}")
        return None

    # Step 2: Check if scanned
    if is_scanned_pdf(file_path):
        return extract_text_ocr(file_path)

    # Step 3: Detect layout
    layout = detect_columns_pymupdf(file_path)

    # Step 4: Extract text based on layout
    try:
        if layout == "single-column":
            print("[INFO] Using PyMuPDF for Single-Column PDF.")
            return extract_text_pymupdf(file_path)
        elif layout == "multi-column":
            print("[INFO] Using pdfminer for Multi-Column PDF.")
            return extract_text_pdfminer(file_path)
        else:  # Mixed layout
            print("[INFO] Mixed Layout detected. Trying both methods.")
            text_pdfminer = extract_text_pdfminer(file_path)
            text_pymupdf = extract_text_pymupdf(file_path)
            # Return the one with more meaningful content (non-whitespace chars)
            return (text_pdfminer if len(text_pdfminer.strip()) > len(text_pymupdf.strip())
                    else text_pymupdf)
    except Exception as e:
        print(f"[ERROR] Text extraction failed: {e}")
        return None
    
def convert_doc_to_text(file_path: str) -> str:
    """Converts a .doc file to plain text using antiword."""
    try:
        result = subprocess.run(["antiword", file_path], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception as e:
        print(f"Error converting .doc file: {e}")
        return ""
    
def convert_csv_to_text(file_path: str) -> str:
    extracted_data = ""
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            extracted_data += "\n".join([f"{key}: {value}" for key, value in row.items()]) + "\n\n"
    return extracted_data

def convert_txt_to_text(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8-sig") as file:
            return file.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, "r", encoding="latin-1") as file:
                return file.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, "r", encoding="ISO-8859-1") as file:
                    return file.read()
            except UnicodeDecodeError as e2:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Error decoding file with multiple encodings. The file might be corrupted or have an unsupported encoding. Error: {e2}"
                )

def convert_text_to_json(text: str) -> dict:
    print(os.getenv("OPENAI_API_KEY"))
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    if not text or not isinstance(text, str):
        return {"error": "Invalid input: Text must be a non-empty string"}
        
    if not os.getenv("OPENAI_API_KEY"):
        return {"error": "OpenAI API key not found in environment variables"}

    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except Exception as e:
        return {"error": f"Failed to initialize OpenAI client: {str(e)}"}

    prompt = f"""
    Extract specific details from the provided resume text and return them as a structured JSON object, following the format and data extraction rules specified below.  If an entire object (like location, expected_salary) has all its fields as null, return an empty dictionary instead.If any value is missing or unavailable, return null for that field.


Data Fields and Requirements:

Personal Information:
name:Extract the full name of the person from the resume. Ensure there are no spelling mistakes or missing initials. Convert all letters to uppercase. If the name is missing, return null.
email: List all email addresses mentioned in the resume. Capture them exactly as they appear without missing any characters, as accuracy is crucial. If no email addresses are found, return an empty list [].
phone: Extract all phone numbers provided in the resume. Ensure no numbers are missed. If no phone numbers are found, return an empty list [].

location: Parse the given address carefully and categorize its components into street, district, state, and country. If any details are missing (e.g., state or country), infer them based on the available district. Use the provided district name to determine the corresponding state and country.
  - street:Extract the door number, street name, and village name if explicitly mentioned in the address. Ensure accurate extraction. If no such details are available, return null.
  - district:Extract the district name from the address if provided.If no district is mentioned, set this field to null.
  - state:Infer the state name based on the district or any other available address details. If neither a state nor a district can be determined, return null.
  - country:Infer the country name based on the district or state. If no state, district, or country can be determined, return null.

expected_salary: Extract and format the salary details if mentioned in the resume. If any details are missing, return null.
   -salary: Extract the salary value.If a single value is provided (e.g., 50000), format it as 50,000.If a salary range is given (e.g., 50000-70000), format it as 50,000-70,000.If no salary is mentioned, return null.
   -currency: Extract the currency used for the salary (e.g., USD, INR). If missing, return null.
   -per: Extract the salary unit (e.g., per year, per month, per week). If missing, return null

summary: Extract a detailed summary of all key elements from the resume, ensuring accuracy and completeness. If the resume already contains a summary, use it. Otherwise, generate a comprehensive summary including all key details such as skills, technologies, certifications, education, experience, languages, projects, location, and any other significant details.

experience: Extract and structure job experience details while ensuring accurate calculations and proper formatting.
   -Internships should NOT be included. If no experience is mentioned in the resume, return job: [],total_duration: 0
   - total_duration: Sum the duration of all job experiences and represent it in decimal years (e.g., 5.4). If missing, calculate it based on the duration field in the job list. If no experience is found, return 0.
   -job:A list of dictionaries containing the following details for each job experience:
        - job_title: Extract the job title or designation. If missing, return null..
        - company: Extract the company or organization name. If missing, return null.
        - technology: List all skills and technologies (e.g., Java, Python, SQL) used in the job. Extract relevant technologies from the requirements and responsibilities section.Ensure consistent naming conventions (e.g., "React.js" should always be extracted as "React JS", not "React.js").
        - start_year: Extract the start year from the provided details and convert it to the format mm/yyyy. If missing, return null.
        - end_year: Extract the end year and format it as mm/yyyy. If the end year is missing but a duration is mentioned (e.g., "3 years"), infer the end year by adding the duration to the start year.If neither the end year nor duration is available, assume the experience is ongoing and use the current year as the end_year.If no details are available, return null.
        - duration: Calculate the total duration of each job experience in decimal years (e.g., 2.6). If missing, return null.
        - responsibilities: Extract and list key job responsibilities. If missing, return an empty list [].

education:Extract education details from the resume and return a structured list of dictionaries with the following fields:
-degree (Unique Degree Type):
    -Extract the degree type as one of the predefined values:["Bachelor", "Master", "Doctorate", "Secondary", "Higher Secondary", "Diploma"]
    -Mappings: examples are
    "Bachelor of Engineering" or "B.E" → "Bachelor"
    "Master of Technology" or "M.E" → "Master"
    "Ph.D. in Computer Science" or "Ph.d" → "Doctorate"
    "12th", "HSC", "Senior Secondary", "Intermediate" or School-related mentions (e.g., Higher Secondary, Schooling,...) based ont that infer → "Higher Secondary"
    "10th", "SSLC", "Matriculation", "High School"	 → "Secondary"
    -Inference:
    -If only a field of study is mentioned (e.g., "Studied CSE"), infer "Bachelor" if CSE/IT-related.
    -If the degree is not inferable, return null.
-degree_name (Short Form):
    -Extract the short form of the degree:examples are
    "Bachelor of Engineering" or "B.E" → "B.E"
    "Bachelor of Technology" or "B.Tech" → "B.Tech"
    "Diploma" remains the same "Diploma"
    "Master of Science" or "M.Sc"→ "M.Sc"
    "10th" or "Secondary" or "SSLC"→ "SSLC"
    "12th" or "Higher Secondary" → "Higher Secondary"
    -Inference:
    -If degree_name is missing but department suggests a known degree, infer accordingly.
    -If both are missing, return null.
-department (Short Form):
    -Extract the short form of the department:examples are
    "Computer Science Engineering" or "CSE"→ "CSE"
    "Information Technology" or "IT" → "IT"
    "Electronics and Communication Engineering" or "ECE"→ "ECE"
    -If missing, return null.
-institution (College/University Name):
    -Extract and return the name of the institution.
    -If SSLC (10th) does **not mention** an institution but Higher Secondary (12th) does, **infer** that the SSLC institution is the same.
    -If missing, return null.
-start_year (Date Extraction):
    -Extract the start year in mm/yyyy or yyyy format.
    -If only a single year is mentioned, infer if it represents start_year or end_year.
    -If missing, return null.
-end_year (Date Extraction):
    -Extract the end year in mm/yyyy or yyyy format.
    -If only one year is mentioned, infer whether it is start_year or end_year.
    -If missing, return null.
- duration (Integer):
    -Calculate the number of years for the degree.
    -It should always be an integer, not a float.
    -Example:
    -start_year = 2018, end_year = 2022 → duration = 4
    -If data is missing, return null.
-cgpa: If CGPA is mentioned, convert it to a percentage using the formula:
    -Percentage=CGPA×10
    -If percentage is directly mentioned, use it as is.
    -If CGPA is missing, return null.
    -example:
    input to output(cgpa)
    1."8.5" to "85%"
    2."7.2" to "72%"
    3."92% or 92 " to "92%"
    4."GPA: 3.8 (out of 4)" to "95%" (scaled to 10-point CGPA)
    5."First class with distinction" to null (CGPA not mentioned)
    6."No CGPA or percentage given" to null.

languages: Extract and structure language proficiency details from the resume. If no languages are found, return an empty list [].
    - language: Extract the language name (e.g., "English", "Spanish").
    - proficiency: The proficiency level classified as: values should only be fluent or intermediate or basic not other.(based on the context you should understand and give a value on any(fluent,intermediate,basic))
        - fluent: If the resume mentions high proficiency, advanced, fluent, quick learner, Native/Mother Tongue, Professional, classify it as "fluent".
        - intermediate: If the resume mentions medium proficiency, conversational, classify it as "intermediate".
        - basic: If the resume mentions low proficiency, advantage, understandable, manageable, beginner, classify it as "basic".
        - If no proficiency level is specified, default to "basic".

skills:Extract and categorize all skills mentioned in the resume into "technical_skills" and "soft_skills". Ensure there are no duplicate entries, and maintain consistent formatting.
    - **technical_skills**:
        - Identify all technologies, programming languages, frameworks, databases, tools, cloud services, AI/ML models, and software mentioned anywhere in the resume.
        - Search across **all sections** of the resume, including **experience, projects, certifications, achievements, education, and skills sections**.
        - Ensure consistent naming conventions (e.g., "React.js" should always be extracted as "React JS", not "React.js").
        - Include all **relevant** technical skills, ensuring none are omitted.
        - Remove duplicate entries to avoid repetition.
        - If no technical skills are found, return an empty list `[]`.
    - **soft_skills**:
        - Extract all non-technical and interpersonal skills such as: **Communication, Teamwork, Leadership, Problem-Solving, Adaptability, Time Management, Collaboration, Critical Thinking, Creativity, Decision-Making, etc.**
        - These should be collected from any relevant section of the resume.
        - Ensure that only genuine soft skills are included, avoiding misclassification.

projects:Extract all project details from the resume, ensuring structured and accurate extraction. Each project should contain the following fields:
    -title (Project Name): Extract the project name. If missing, return null.
    -skills (Technologies Used): Identify technologies, programming languages, frameworks, and tools explicitly mentioned as part of the project. Only extract skills that are directly related to the project, excluding skills from other projects or any other fields.
        -Ensure that only the technologies used in the project are included, not general skills from other sections like skills or certifications.
        -Extracted technologies should be stored as a list of strings, e.g., ["Python", "TensorFlow"]
        - Ensure consistent naming conventions (e.g., "React.js" should always be extracted as "React JS", not "React.js").
        -If no technologies are explicitly stated, return [].
        - Identify the technologies, programming languages, frameworks, or tools explicitly mentioned as part of the project.
    -responsibilities (Key Contributions):Extract key contributions, tasks, or work done in the project, such as development, testing, deployment, optimization, integration, etc.
        -Look for action verbs such as: developed, designed, implemented, integrated, optimized, automated, built, tested, deployed, enhanced, configured, etc.
        -If responsibilities are implied but not explicitly mentioned, infer them from the project description.
        -If a workflow or process is mentioned but no specific skills are listed, ensure it is included under responsibilities.
        -If responsibilities are written in a single sentence, split them into a list of strings based on sentence-ending punctuation (dot .).
        -If no responsibilities are found, return [].
        
Certificates:Extract all certificates, courses, or achievements mentioned in the resume. Each entry should include:
  - title: Extract the name of the certificate or course. If missing, return null.
  - issuer: Extract the name of the issuing organization or institute exactly as mentioned in the resume. If missing, return null.
  - year: Extract the issuance year from the certificate details. If missing, return null.

Search Summary:Extract and summarize all key elements(name,email,phone number,education,Skills & Technologies,Certifications,experience,projects,Languages Known,location) from the resume to ensure completeness, clarity, and standardization. This summary is essential for resume searching, matching, and sorting. The summary should include the following sections:
-name:Extract the full name of the person from the resume. Ensure there are no spelling mistakes or missing initials. Convert all letters to uppercase.
-email: List all email addresses mentioned in the resume. Capture them exactly as they appear without missing any characters, as accuracy is crucial.
-phone: Extract all phone numbers provided in the resume.
-Education:strictly follow the rules of both short and full forms (e.g., 'B.E (Bachelor of Engineering)'), infer 'B.E (Bachelor of Engineering) in CSE (Computer Science Engineering)' for CSE-related degrees, and standard milestones '10th (SSLC)', '12th (Higher Secondary)' when present, with institution and years.
-Skills & Technologies:List all programming languages, frameworks, tools, databases, and cloud platforms, expanding abbreviations (e.g., 'SQL (Structured Query Language)').
-Certifications:Extract and list all certifications with exact names to ensure accuracy.
-Experience (if any):Mention job roles, company names, and durations in a structured format.
-Projects:List key projects with a brief but clear description.(e.g., 'E-commerce Platform: Developed using Python and SQL').
-Languages Known:Include all spoken languages mentioned in the resume.
-Location:Ensure proper formatting by splitting District, State, and Country correctly (e.g., Krishnagiri, Tamilnadu, India).
-Key Identifiers for Matching:Ensure the summary is rich in keywords for resume matching, searching, and sorting.
-final output:Provide a structured, detailed summary that aligns with job descriptions for efficient categorization.

raw_data:Include an additional field, raw_data, where each key corresponds to a structured field (e.g., name, email, location) and contains the exact full raw text extracted from the resume. Ensure proper spacing and correct sentence structuring with appropriate punctuation.
-name: Extract the full raw name in the original format.
-email:A string containing all email addresses, separated by commas if multiple (e.g., "example1@email.com, example2@email.com").
-phone:A string containing all phone numbers, separated by commas if multiple (e.g., "9876543210, 8765432109"). two comes with string for example(string1,string2) and if responsibilities is a string if there are two or more points comes with dot.
-expected_salary:Format the salary with proper currency and range (e.g., "$40,000-$50,000 per annum" for a range, "$40,000 per annum" for a single value).
-location:Store the extracted location as a single formatted string (e.g., "Krishnagiri, Tamilnadu, India").
-experience:it is list of dict if technology has
-education:It is list of dict 
-languages:It is list of string if more languages comes with comma (for example proficiency in english and comes with "English-proficiency").
-skills:It is a dictionary fields like technical and soft skills that should contain list of skills.
-certificates:A list of strings, containing only certificate titles (e.g., ["Java Full Stack Development", "AWS Certified Developer"]).
-projects:it is list of dict if skills are stored as a list (e.g., ["Python", "Java"]).and responsibilities are stored as a list of strings, with each responsibility ending with a dot (.) if multiple.
    <Resume>
    {text}
    </Resume>

    **Return JSON Format:**  
    (Ensure all values are properly structured as strings and lists)
    
    {{
        "name": "string",
        "email": ["string", "string"],
        "phone": ["string", "string"],
        "location": {{
            "street": "string",
            "district": "string",
            "state": "string",
            "country": "string"
        }},
        "expected_salary": {{
            "salary": null,
            "currency": "string",
            "per": "string"
        }},
        "summary": "string",
        "experience": {{
            "job": [
            {{
            "job_title": "string",  
            "company": "string",  
            "Technology": ["string", "string"],  
            "start_year": "mm/yyyy",  
            "end_year": "mm/yyyy",  
            "duration": "X.Y",  
            "responsibilities": ["string", "string", "string"]  
           }}
        ],
            "total_duration":"X.Y"
        }},
        "education": [
            {{
            "degree":"string",
            "degree_name": "string",
            "department": "string",
            "institution": "string",
            "start_year": "mm/yyyy",
            "end_year": "mm/yyyy",
            "duration": x,
            "cgpa": "string"
            }}
        ],
        "languages": [
            {{
            "language": "string",
            "proficiency": "string"
            }}
        ],
        "skills": {{
        "technical_skills": [
            "string",
            "string",
            "string"
        ],
        "soft_skills": [
            "string",
            "string",
            "string"
        ]
        }},
        "projects": [
            {{
            "title": "string",
            "skills": [
                "string",
                "string"
            ],
            "responsibilities": [
                "string",
                "string",
                "string"
            ]
            }}
        ],
        "certificates": [
            {{
            "title": "string",
            "issuer": "string",
            "year": "string"
            }}
        ],
        "search_summary": "string",
        "raw_data":{{
        "name": "string",
        "email": "string",
        "phone": "string",
        "location": "string",
        "expected_salary": "string",
        "summary": "string",
        "experience": [
            {{
            "job_title": "string",
            "company": "string",
            "Technology":"string",
            "start_year": "mm/yyyy",
            "end_year": "mm/yyyy",
            "duration": "X.Y",
            "responsibilities":"string"
            }}
        ],
        "education": [
            {{
            "degree_name": "string",
            "department": "string",
            "institution": "string",
            "start_year": "mm/yyyy",
            "end_year": "mm/yyyy",
            "duration": "x",
            "cgpa": "string"
            }}
        ],
        "languages":["string"],
        "skills":{{"technical_skills":["string"],"soft_skills":["string"]}},
        "projects": [
            {{
            "title": "string",
            "skills": "string",
            "responsibilities":["string"]
            }}
        ],
        "certificates": ["string"],
        "search_summary": "string"
        }}
    }}
    Extract details accurately. Do not add extra fields.
    """

    try:
        print("start1")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an AI that extracts structured data from resumes."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}

        )
        print(response)
        print("start2")
        if not response.choices[0].message.content:
            return {"error": "Empty response from API"}
        try:
            return json.loads(response.choices[0].message.content)
        except JSONDecodeError as jde:
            print(f"JSON Parse Error: {jde}")
            return {"error": "Invalid JSON received from API"}
        """structured_data = json.loads(response.choices[0].message.content)
        print("hi")
        print(structured_data)

        return structured_data"""

    except Exception as e:
        print(f"Error: {e}")
        return {"error": "Failed to extract job description data"}


def convert_text_to_jsonjd(text: str) -> dict:
    print(os.getenv("OPENAI_API_KEY"))
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    if not text or not isinstance(text, str):
        return {"error": "Invalid input: Text must be a non-empty string"}
        
    if not os.getenv("OPENAI_API_KEY"):
        return {"error": "OpenAI API key not found in environment variables"}

    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except Exception as e:
        return {"error": f"Failed to initialize OpenAI client: {str(e)}"}

    prompt = f"""
    Please extract the following details from the provided Job Description (JD) text and return them in a structured JSON format.Please extract the following details from the provided Job Description (JD) text. Return them in the structured JSON object format specified below(not a list of objects). Ensure each key in the JSON is a string, and values accurately reflect the details from the JD text.

Data Fields and Requirements:
All value starting first letter should be capital.
job_titles: A list of all job titles mentioned in the JD.If no titles are found, return an empty list ([]).

company: The name of the hiring organization or company.If not mentioned, return null.

locations: A list of job locations, where each location is represented as a dictionary containing the following keys: Follow these detailed rules and instructions for parsing and inferring:If any of these details are missing (e.g., state or country), infer them based on the district. Use the provided district name to find the corresponding state and country.If no valid locations exist, return an empty list ([]) instead of null values and also If the location contains only "Hybrid", "Remote", "Work from Home", "Anywhere", etc., return [] (an empty list).
  - street:This field should contain the door number, street name, and village name, if explicitly mentioned in the address.If no such details are available, set this field to null.
  - district:Extract the district name from the address if provided.If no district is mentioned, set this field to null.
  - state:Infer the state name based on the district or any other information in the address.If no state,district can be determined, set this field to null
  - country:Infer the country name based on the district or state or any other information in the address.If no state,district,country can be determined, set this field to null

employment_type: A list of employment types explicitly identified in the JD (e.g., Full-time, Part-time, Internship, Contractual etc.). you should understand the employment_type mentioned in jd and modify and infer Return only the types mentioned in the JD(Full-time, Part-time, Internship, Contractual). If none are found, return an empty list ([]).

salary_details: A list of dictionaries, where each dictionary represents details about the salary structure. Each dictionary should have the following keys:
- salary: The numeric salary amount (if available).if single like 50000 comes with 50,000 and range of salary comes with 50,000-70,000) comma seperation based on salary curreny 
- currency: The currency of the salary (e.g., USD, EUR, INR).comes with proper currency codes.
- per: The time period for the salary (e.g., per hour, per month, per year).
- position: Whether the salary applies to specific job positions mentioned in the JD. If yes, return the position(s); otherwise, set it as "all positions".
-If no salary details are found, return an empty list ([]).

responsibilities: A list of key responsibilities for the job.If none are found,comes with what they are provide without any modifications return an empty list ([]).

skills_required: A dictionary with the following structure:
- technical: A list of technical skills required for the role (e.g., programming languages, software tools, frameworks).for example (If an abbreviation is given ML , it should be converted to its full form (e.g., "Machine Learning").likewise you should find out these type provide all in full form).If a job title is provided (e.g., "Python Developer"), extract only the core skill (e.g., "Python").Ensure consistent naming conventions (e.g., "React.js" should always be extracted as "React JS", not "React.js").
- soft: A list of soft skills required for the role (e.g., communication, teamwork, leadership).
-If no skills are mentioned, both lists should be empty ([]).

experience_required: A list of dictionaries where each dictionary contains:
-job_titles: Job title or designation or position based on the jd mention in their experience you should infer the job_titles. If missing, return null.
- min_experience: The minimum years of experience required for that skill or job_titles.It should be numeric value
- max_experience: The maximum years of experience required for that skill or job_titles.It should be numeric value
- experience_range: take the min_experience and max_experience give in the range like min_experience-max_experience(for example min is 0 and max is 3 and then experience_range should be 0-3)
- skills: The list of skills for which the experience is required (e.g., "Java", "Python").list of skills based on the specific job_titles.if job_titles is missing and then only based on the common experience_range list of string will assign.If missing return [].
-experience_type:It should be Fresher or Experienced based on the content you should find out that.Ensure consistent naming conventions (e.g., "React.js" should always be extracted as "React JS", not "React.js").
-Experience rules:
- If the JD mentions months, convert them into years (e.g., "6 months" becomes min_experience: 0.5, max_experience: 1).
- If a range (e.g., 3 to 5 years) is provided, set min_experience and max_experience accordingly.
- If only a single year (e.g., 2 years) is mentioned, treat both min_experience and max_experience as that value.
- If the JD does not mention any experience or years and saying fresher , set min_experience: 0 and max_experience: 0 and experience_type:"Fresher" by default.
- If the jd mention only experience not giving years, set min_experience:0 and max_experience:1 and experience_type:"Experienced"by default
-experience_range whatever you should take from the min_experience and max_experience and give in the format (min_experience-max_experience)
-if the jd doesn't mention any experience and only mentioning fresher or not mentioning anything.
-let us avoid of providing empty list in experience_required and return should be skills should be empty string, min_experience:0,max_experience:0,experience_range:0-0,experience_type:"Fresher".
-example:
1. 2 year experience in java,sql.fresher in python.
output:
job_titles:'',skills:["java","sql"],experience_range:"2-2"
job_titles:'',skills:["python"],experience_range:"0-0"

2. 2.5 year experience in java,sql.fresher in python.
output:
job_titles:'',skills:["java","sql"],experience_range:"2.5-2.5",min=2.5 ,max=2.5
job_titles:'',skills:["python"],experience_range:"0-0",min=0,max=0

3. experience in software developer include skills like python,java
output:
job_titles:'software developer',skills:["java","python"],experience_range:"0-1"

4. experience in software developer include skills like  1 year in python, 2 year in java
output:
job_titles:'software developer',skills:["python"],experience_range:"1-1"
job_titles:'software developer',skills:["java"],experience_range:"2-2"

education_required: A list of dictionaries containing the following: 
-degree field should be unique based on that list of dictionary can happen.carefully get all the degree like  ("Bachelor", "Master", "Doctorate","Secondary("10th" or "sslc")","Higher Secondary("12th" or "Higher Secondary")","Diploma") in the jd they had mentioned.
-Degree Name Conversion "B.E." for Bachelor and "M.E." for Master
-degree:  -Extract the degree type as one of the predefined values:["Bachelor", "Master", "Doctorate", "Secondary", "Higher Secondary", "Diploma"]
    -Mappings: These are the few exmples based on that you should infer the other also.
        -"Bachelor of Engineering" or "B.E" or "Bachelor" → "Bachelor"
        -"Master of Technology" or "M.E" or "Master" → "Master"
        -"Ph.D. in Computer Science" or "Ph.d" or "Doctorate" → "Doctorate"
        -"10th", "SSLC", "Matriculation", "High School","Class X","Junior Secondary" based on that you should find out and infer  → "Secondary"
        -"12th", "HSC", "Senior Secondary", "Intermediate","Class XII"  based on that you should find out and infer → "Higher Secondary"      
        -"If a common term like examples 'Schooling' likewise is mentioned without directly specifying '10th' or '12th', generate two dictionaries: One for 'Secondary' (representing 10th) and one for 'Higher Secondary' (representing 12th)."
        - Special Rule for Vague School References:
            - If a common term like "School" likewise is mentioned without explicitly stating "10th" or "12th", **always generate two separate dictionaries**:
            - "Secondary" with degree_name: ["SSLC"] (representing 10th).
            - "Higher Secondary" with degree_name: ["Higher Secondary"] (representing 12th).
            - If "10th" (or equivalent) is explicitly mentioned, include only "Secondary".
            - If "12th" (or equivalent) is explicitly mentioned, include only "Higher Secondary".
    -Inference:
        -If only a field of study is mentioned (e.g., "Studied CSE"), infer "Bachelor" if CSE/IT-related.
        -If "10th" is explicitly mentioned, only include "Secondary".
        -If "12th" (or equivalent) is explicitly mentioned, only include "Higher Secondary".        
        -If the degree is not inferable, return null.
        -"Studied CSE" (degree not mentioned) to "Bachelor" (inferred from "B.E")
        -"Studied AI" (degree not mentioned) to null (degree cannot be inferred)

-degree_name: A list of degree names based on the specific degree.
The degree name should always be in short form, even if the job description (JD) contains the full form.
    -MAPPINGS:These are the few mapping exmples based on that you should infer the other also.
        -"Bachelor of Engineering" or "B.E" and output "degree":["Bachelor"],"degree_name": ["B.E"]
        - "Master of Technology" or "M.Tech" and output "degree":["Master"],"degree_name":["M.Tech"]
        - For 10th and 12th grades, use "SSLC" and "Higher Secondary" respectively.
        -"10th" and ouput "degree_name":["SSLC"]
        -"Higher Secondary" and output "degree_name":["Higher Secondary"]
        -"Studied CSE" (Degree name not mentioned, infer from department) output "degree":"Bachelor",degree_name": ["B.E"]
    -Inference:
        -If degree_name is missing but department suggests a known degree, infer accordingly.
        -If missing, return an empty list [].

- department: A list of department names based on the degree name.strictly define the asked department in jd.
The department name should always be in short form, even if the job description (JD) contains the full form.
    -Mappings: These are the few mapping exmples based on that you should infer the other also.
        -"Computer Science Engineering" or "CSE" and output "department":["CSE"]
        -"Mechanical Engineering" or "ME" and output "department":["ME"]
        -"Studied B.E. but department not mentioned" "department":[]
        -"Computer Science Engineering" → "CSE"
        -If department is missing, return an empty list [].

- year: All the specific years mentioned for the degree (e.g., "2020", "2023").If a range is provided (e.g., "2019 to 2023"), include all years in that range (["2019", "2020", "2021", "2022", "2023"]). if they provide before 2023 or after 2023 comes with ["before 2023"] or ["after 2023"].if they metion only one year complete in 2023 then comes with ["2023"].If not provided, return an empty list ([]).
- min_year: The minimum year from the year list (as a string). If the year list is empty, return null.
- max_year: The maximum year from the year list (as a string). If the year list is empty, return null.
-Before a specific year (e.g., "before 2023"): Set the max_year to that year and leave min_year as null.
-After a specific year (e.g., "after 2023"): Set the min_year to that year and leave max_year as null.
-Exact year (e.g., "2023"): Set both min_year and max_year to that year.
-A range of years (e.g., "2019 to 2023"): Set min_year and max_year based on the range.
-If no specific years or conditions are mentioned, both min_year and max_year should be null.
- cgpa: 
If CGPA is mentioned, convert it to a percentage using the formula:
Percentage=CGPA×10
If percentage is directly mentioned, use it as is.
If CGPA is missing, return null.
    -Mappings:These are the few mapping exmples based on that you should infer the other also.
        input to output(cgpa)
        1."8.5" to "85%"
        2."7.2" to "72%"
        3."92%" to "92%"
        4."GPA: 3.8 (out of 4)" to "95%" (scaled to 10-point CGPA)
        5."First class with distinction" to null (CGPA not mentioned)
        6."No CGPA or percentage given" to null

-example:
-input:
1.Qualification: B.E/B.Tech/M.E/Ph.d in cse,ece,it,eee before 2023 at 60%.10th and 12th should be above 80%
-output:
-according to the input it contain b.e/B.tech so there should be one bachelor dictionary and m.e so there should be master dictionary and ph.d so there will be doctorate dictionary and 10th so there will be secondary dictionary and 12th so there will be higher secondary dictionary.if they have then only compulsory provide unique degree dictionary for each degree.
 Bachelor's Degree (B.E/B.Tech)
Degree: "Bachelor" (since it is a Bachelor's degree)
Degree Name: ["B.E", "B.Tech"] (short form of Bachelor of Engineering and Bachelor of Technology)
Department: ["CSE", "ECE", "IT", "EEE"] (all eligible departments)
Year: ["before 2023"] (since it's mentioned that graduation should be before 2023)
Min Year: null (since it's before a specific year)
Max Year: "2023" (since it's before 2023)
CGPA: "60%" (minimum required percentage)

 Master's Degree (M.E)
Degree: "M.E" (since it is a Master's degree)
Degree Name: ["M.E"] (short form of Master of Engineering)
Department: ["CSE", "ECE", "IT", "EEE"] (all eligible departments)
Year: ["before 2023"] (since it's mentioned that graduation should be before 2023)
Min Year: null (since it's before a specific year)
Max Year: "2023" (since it's before 2023)
CGPA: "60%" (minimum required percentage)

 Doctorate Degree (Ph.d)
Degree: "Doctorate" (since it is a Doctorate degree)
Degree Name: ["Ph.d"] (short form of Doctor of Philosophy)
Department: ["CSE", "ECE", "IT", "EEE"] (all eligible departments)
Year: ["before 2023"] (since it's mentioned that graduation should be before 2023)
Min Year: null (since it's before a specific year)
Max Year: "2023" (since it's before 2023)
CGPA: "60%" (minimum required percentage)

 10th  (SSLC)
Degree: "Secondary" (since it represents 10th )
Degree Name: ["SSLC"] (short form of Secondary School Leaving Certificate)
Department: [] (no department for SSLC)
Year: [] (no year specified)
Min Year / Max Year: null (since no specific years are given)
CGPA: "80%" (minimum required percentage)

 12th  (Higher Secondary)
Degree: "Higher Secondary" (since it represents 12th )
Degree Name: ["Higher Secondary"] (standard short form)
Department: [] (no department specified)
Year: [] (no year specified)
Min Year / Max Year: null (since no specific years are given)
CGPA: "80%" (minimum required percentage)

Input:
Bachelor’s or Master’s degree in Computer Science engineering, Information Technology
B.E/M.E in Computer Science, Information Technology

output:
here computer science engineering department refer it is B.E 
here Information Technology  department refer it is B.Tech
"degree":"Bachelor","degree_name":["B.E","B.Tech"],"department":["CSE","IT"]
"degree":"Master","degree_name":["M.E"],"department":["CSE","IT"]
don't get confused and give a different answer and if you confused in degree_name and just focus on degree and department and then find out degree_name.

Input:
Schooling should be above 80%

output:
"degree":"Higher Secondary","degree_name":["Higher Secondary"],"department":[]
"degree":"Secondary","degree_name":["SSLC"],"department":[]


languages_required: A list of dictionaries containing the following:
- language: The language name (e.g., "English", "Spanish").
- proficiency: The proficiency level classified as:
  - fluent (for high proficiency,advance,fluent,quick,Native / Mother Tongue ,Professional)
  - intermediate (for medium proficiency,Conversational,)
  - basic (for low proficiency,advantage,understandable,manageable,Beginner) based on the content you find out properly. If not specified, default to "basic".
-If no languages are found, return an empty list ([]).

post_date: The application deadline or post date for the job posting. Return the date in dd/mm/yyyy format. If no date is provided, return null.

end_date: The application deadline or end date for the job posting. Return the date in dd/mm/yyyy format. If no date is provided, return null.

contact_us: A list of contact methods, which could include one or more ways to get in touch with the company (e.g., phone number, website). if there is multiple number,email,website it should comes like type:"email",value:["email1","email2"]

additional_information: Any other relevant details or notes in the JD.

jd_summary:A detailed, structured summary of the job description covering:
Skills, Technologies, Certifications, Education, Experience, Languages, Locations, Employment Type, Salary, and Additional Information.belowe i am mention rules for these field according to that rule it should be convert and comes into a sentence for example (b.e in degree_name comes and convert to the rule and provide B.E(Bachelor of Engineering))
-Education Requirements:
-Strictly mention degrees,degree_name and departments in both full form and short form.
-Example: B.E (Bachelor of Engineering) in CSE (Computer Science Engineering), B.Tech (Bachelor of Technology) in IT (Information Technology).
-Do NOT use vague terms like "or a related field". Always mention specific degrees and departments explicitly.
-If only a department is mentioned (e.g., "Computer Science"), infer the most relevant degree.
-10th and 12th should always be written as 10th (SSLC) and 12th (Higher Secondary).

-Skills & Technologies:
-List all programming languages, frameworks, and tools explicitly.
-Example: Python, Java, SQL, React JS, AWS (Amazon Web Services), Kubernetes, Terraform.

-Certifications:
-Extract and mention all certifications in full form.
-Example: AWS Certified Solutions Architect - Associate, Microsoft Azure Fundamentals, Google Professional Cloud Architect.

-Work Location & Employment Model:
-Always mention both work location and work model (onsite, remote, hybrid).
-Example: Bangalore, India (Hybrid) or New York, USA (Onsite).
-Salary Details:
-Specify salary in the correct currency and format.
-Example: ₹8-12 LPA (Indian Rupees), $100,000-$130,000 USD per year.

-Employment Type:
-Specify whether it is full-time, part-time, contract, or internship.

-Additional Details:
-Include any preferred experience (e.g., Microservices, REST APIs, DevOps, Docker, Kubernetes).
-List soft skills (e.g., Problem-solving, Communication, Teamwork).
-Mention language proficiency in a structured format (e.g., English - Fluent, Tamil - Native).
-Highlight if candidates with open-source contributions or AI/ML expertise will be preferred.

-Structured Format:
-The JD summary should be detailed, structured as a long paragraph, and contain all extracted elements explicitly.
-Ensure that education, certifications, salary, and location are not omitted.

raw_data:where each key corresponds to a structured field (e.g., job_titles, company, locations) and contains the exact raw text from the JD used to extract that data
-location:It should represent the list of locations. for example 
-if there is a more district then should comes in ["district1","district2"]
-if there is more district district comes with state then should comes in ["district1,state1","district2,state2"]
-language:It should represent the list of language required.based on the above languages_required field get the values fro that.for example
-inside field like language:english,proficiency:fluent and that should be comes in ["English-Native"].
-contact: if there is multiple number,email,website it should comes like type:"email",value:["email1","email2"]
-experience:It should represent the list of string.
-education:It should represent the list of string.
-language:It should represent the list of string.
-jd_summary:It respresnt a string that should contain a sentence of the jd and that should refer from above jd_summary field. 
    <Job Description>
    {text}
    </Job Description>

    **Return JSON Format:**  
    (Ensure all values are properly structured as strings and lists)
    
    {{"job_titles": ["string1", "string2"],
    "company": "string",
    "locations": [
    {{"street": "string", "district": "string", "state": "string", "country": "string"}}
    ],
    "employment_type": ["string"],
    "salary_details": [
    {{"salary": "string", "currency": "string", "per": "string", "position": "string"}}
    ],
    "responsibilities": ["string"],
    "skills_required": {{"technical": ["string"], "soft": ["string"]}},
    "experience_required": [
    {{"job_titles":"string", "min_experience": 0, "max_experience": 0, "experience_range":"0-0","skills":["string","string"],"experience_type": "string"}}
    ],
    "education_required": [
    {{"degree": "string", "degree_name":["string"], "department":["string"], "year": ["string"], "min_year": "string", "max_year": "string", "cgpa": "string"}}
    ],
    "languages_required": [
    {{"language": "string", "proficiency": "fluent"}}
    ],
    "post_date": "dd/mm/yyyy",
    "end_date": "dd/mm/yyyy",
    "contact_us": [{{
      "type": "string", "value": ["string"]
    }}],
    "additional_information": "string",
    "jd_summary": "string",
    
    "raw_data": {{
        "job_title":["string"],
        "company": "string",
        "location":["string"],
        "employment_types":["string"],
        "salary_detail":["string","string"],
        "responsibilitie":["string"],
        "skills":{{
            "technical":["string"],
            "soft":["string"]
        }},
        "education":["string"],
        "experience":["string"],
        "language":["string"],
        "post_date":"string",
        "end_date":"string",
        "contact": [{{
            "type": "string", 
            "value": ["string"]
        }}],
        "additional_information": "string",
        "jd_summary": "string"   
    }}
    }}
    Extract details accurately. Do not add extra fields.
    """

    try:
        print("start1")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an AI that extracts structured data from job descriptions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}

        )
        print(response)
        print("start2")
        if not response.choices[0].message.content:
            return {"error": "Empty response from API"}
        try:
            return json.loads(response.choices[0].message.content)
        except JSONDecodeError as jde:
            print(f"JSON Parse Error: {jde}")
            return {"error": "Invalid JSON received from API"}
    except Exception as e:
        print(f"Error: {e}")
        return {"error": "Failed to extract job description data"}