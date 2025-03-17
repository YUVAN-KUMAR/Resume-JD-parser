import redis
import os
import time
import magic
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from models import Job, Resume, JobDescription
import uuid
from datetime import datetime
import json
from utility import convert_pdf_to_text,convert_doc_to_text,convert_docx_to_text,convert_csv_to_text,convert_txt_to_text,convert_text_to_json,convert_text_to_jsonjd
from dotenv import load_dotenv
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_real_file_type(file_path: str) -> str:
    """Detects the actual MIME type of the file and returns the corresponding extension."""
    mime = magic.Magic(mime=True)
    mime_type = mime.from_file(file_path)
    mime_mapping = {
        "application/pdf": "pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
#       "application/msword": "doc",
        "text/plain": "txt",
        "text/csv": "csv",
    }
    if file_path.endswith(".doc"):
        print("yes")
        return "doc"
    return mime_mapping.get(mime_type, "unknown")

def background_worker():
    print("Background worker started.....")
    
    
    db = SessionLocal()
    
    try:
        while True:
            job_data = redis_client.blpop("job_queue", timeout=30)
            if job_data is None:
                continue

            job_id = job_data[1].decode("utf-8")

            job = db.query(Job).filter(Job.job_id == job_id).first()
            if not job:
                print(f"Job {job_id} not found in database.")
                continue

            try:
                job.status = "processing"
                db.commit()

                file_path = job.filepath
                
                real_mime_type = get_real_file_type(file_path)
                
                print(f"Detected MIME Type: {real_mime_type}")
                print(job.file_type)
                if real_mime_type != job.file_type:
                    raise Exception(f"File type mismatch: Expected {job.file_type}, but detected {real_mime_type}.")
                real_mime_type = job.file_type
                def process_text(text):
                    return text.lower().replace(" ","") if text else None
                """
                def process_list_of_strings(data_list):
                    return [process_text(item) for item in data_list] if data_list else []
                def process_dict_of_strings(data_dict):
                    if isinstance(data_dict, dict):
                        for key, value in data_dict.items():
                            if isinstance(value, list):  # If the value is a list, process each item
                                data_dict[key] = process_list_of_strings(value)
                            elif isinstance(value, str):  # If the value is a string, apply process_text
                                data_dict[key] = process_text(value)
                    return data_dict
                def process_dict_of_strings(data_dict):
                    return {key: process_text(value) for key, value in data_dict.items()} if data_dict else {}"""
                if job.jobtype == "Resume":
                    
                    if real_mime_type =="docx":
                        text = convert_docx_to_text(file_path)
                        print(text)
                    elif real_mime_type == "doc":
                        print("Begin")
                        text=convert_doc_to_text(file_path)
                        print(text)
                    elif real_mime_type == "pdf":
                        text = convert_pdf_to_text(file_path)
                        print(text)
                    elif real_mime_type == "csv":
                        text = convert_csv_to_text(file_path)
                        print(text)
                    elif real_mime_type == "txt":
                        text = convert_txt_to_text(file_path)
                        print(text)
                    else:
                        raise Exception("Unsupported file format for resume")

                    json_data = convert_text_to_json(text)
                    print(json_data)
                    name = json_data.get("name")
                    email = json_data.get("email",[]) 
                    phone = json_data.get("phone",[])
                    location = json_data.get("location",{}) 
                    expected_salary = json_data.get("expected_salary",{})
                    summary = json_data.get("summary")
                    skills = json_data.get("skills",{})  
                    experience = json_data.get("experience", []) 
                    education = json_data.get("education", [])  
                    languages = json_data.get("languages", [])
                    projects = json_data.get("projects", [])  
                    certificates = json_data.get("certificates", [])
                    search_summary = json_data.get("search_summary")  
                    processed_search_summary=process_text(json_data.get("search_summary") )
                    raw_data=json_data.get("raw_data",{})
                    resume = Resume(
                        name=name,
                        email=email,
                        phone=phone,
                        location=location,  
                        expected_salary=expected_salary, 
                        summary=summary, 
                        skills=skills,  
                        experience=experience, 
                        education=education,  
                        languages=languages, 
                        projects=projects,  
                        certificates=certificates, 
                        search_summary=search_summary,  
                        processed_search_summary=processed_search_summary,
                        raw_data=raw_data
                    )
                    db.add(resume)
                    db.commit()

                elif job.jobtype == "JD":
                    if real_mime_type == "docx":
                        text = convert_docx_to_text(file_path)
                    elif real_mime_type == "doc":
                        print("Begins")
                        text = convert_doc_to_text(file_path)
                        print(text)
                    elif real_mime_type == "pdf":
                        text = convert_pdf_to_text(file_path)
                    elif real_mime_type == "csv":
                        text = convert_csv_to_text(file_path)
                    elif real_mime_type == "txt":
                        text = convert_txt_to_text(file_path)
                    
                    else:
                        raise Exception("Unsupported file format for JD")
                    
                    json_data = convert_text_to_jsonjd(text)
                    print("Type of json_data:", type(json_data))
                    print("starts")
                    print(json_data)
                    job_titles = json_data.get("job_titles", [])
                    print(job_titles)

                    company = json_data.get("company",None)
                    print(company)

                    locations = json_data.get("locations", [])
                    print(locations)
                    employment_type = json_data.get("employment_type", [])
                    print(employment_type)
                    salary_details =  json_data.get("salary_details", [])
                    print(salary_details)

                    responsibilities = json_data.get("responsibilities", [])
                    print(responsibilities)
                    skills_required = json_data.get("skills_required", {})
                    print(skills_required)
                    experience_required =  json_data.get("experience_required", [])

                    languages_required = json_data.get("languages_required", [])

                    education_required =json_data.get("education_required", [])
                    print(education_required)
                    
                    post_date=json_data.get("post_date","")
                    print("post_date")
                    print(post_date)
                    if not post_date:
                        post_date=datetime.now().strftime("%d/%m/%Y")
                        
                    print(post_date)

                    end_date = json_data.get("end_date", "")

                    contact_us = json_data.get("contact_us", [])

                    additional_information = json_data.get("additional_information", "")

                    search_summary = json_data.get("jd_summary")
                    processed_search_summary=process_text(json_data.get("jd_summary") )

                    raw_data= json_data.get("raw_data", {})

                    jd = JobDescription(
                        job_titles=job_titles,
                        company=company,
                        locations=locations,
                        employment_type=employment_type,
                        salary_details=salary_details,
                        responsibilities=responsibilities,
                        skills_required=skills_required,
                        experience_required=experience_required,
                        languages_required=languages_required,
                        education_required=education_required,
                        post_date=post_date,
                        end_date=end_date,
                        contact_us=contact_us,
                        additional_information=additional_information,
                        search_summary=search_summary,
                        processed_search_summary=processed_search_summary,
                        raw_data=raw_data
                    )
                    db.add(jd)
                    db.commit()
                    print("success")
                job.status = "success"
                db.commit()

            except Exception as e:
                job.status = "failed"
                job.err_message = str(e)
                db.commit()
                print(f"Error processing job {job_id}: {str(e)}")

            time.sleep(1)

    except KeyboardInterrupt:
        print("Worker shutting down.")
    finally:
        db.close()

if __name__ == "__main__":
    background_worker()
