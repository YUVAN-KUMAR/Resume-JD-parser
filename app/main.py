from fastapi import FastAPI, UploadFile, File, HTTPException, Depends,Query,Body
from sqlalchemy.orm import Session,sessionmaker,aliased
from sqlalchemy import String,Integer,cast,Date,asc,desc,create_engine,text,JSON,ARRAY
from sqlalchemy.dialects.postgresql import ARRAY, JSON,JSONB,FLOAT,TEXT
from app.db import get_db
from sqlalchemy.sql import func,or_, select, exists,and_
from sqlalchemy.sql.expression import cast
from sqlalchemy.types import String
import redis
import uuid
import re
from rq import Queue
import os
#import docx
#import csv
#import PyPDF2
from wordsegment import load,segment
from fuzzywuzzy import fuzz
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.requests import Request
import google.generativeai as genai
from dotenv import load_dotenv
import json
from app.models import Resume,JobDescription,Job
from app.schemas import ResumeResponseSchema,ResumeDetailResponseSchema,JobDescriptionDetailResponseSchema,JobDescriptionResponseSchema
from typing import List,Dict,Optional
from datetime import datetime
from app.db import SessionLocal
import spacy
"""
redis = Redis(host='localhost', port=6379, db=0)
queue = Queue(connection=redis)"""
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
load_dotenv()
app=FastAPI()
load()
nlp = spacy.load("en_core_web_md")

templates = Jinja2Templates(directory="app/templates")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
upload_dir = "uploads"

def split_words(keyword:str)->str:
    split_words = segment(keyword)
    print(split_words)
    return split_words

if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)
    
def compare_education(resume_education: list, jd_education: list) -> float:
    if not resume_education:
        return 0.0
    
    primary_degrees = {"bachelor", "master", "doctorate"}
    secondary_degrees = {"diploma", "higher secondary", "sslc"}
    
    jd_primary = []
    jd_secondary = []
    
    for jd in jd_education:
        jd_degree = (jd.get('degree') or '').lower()
        if jd_degree in primary_degrees:
            jd_primary.append(jd)
        elif jd_degree in secondary_degrees:
            jd_secondary.append(jd)
    
    # Group resume education into primary and secondary
    resume_primary = []
    resume_secondary = []
    
    for resume in resume_education:
        resume_degree = (resume.get('degree') or '').lower()
        if resume_degree in primary_degrees:
            resume_primary.append(resume)
        elif resume_degree in secondary_degrees:
            resume_secondary.append(resume)
    
    total_score = 0.0
    total_jd_categories = 0
    
    
    if jd_primary:
        total_jd_categories += 1
        primary_score = compare_degree_group(resume_primary, jd_primary)
        total_score += primary_score
        print("jd_p:",total_score)
    
    if jd_secondary:
        total_jd_categories += 1
        secondary_score = compare_degree_group(resume_secondary, jd_secondary)
        total_score += secondary_score
        print("jd_s:",total_score)
    
    print("total:",total_jd_categories)
    print(total_score)
    final_score = total_score / total_jd_categories if total_jd_categories > 0 else 1.
    print(final_score)
    return final_score

def compare_degree_group(resume_degrees: list, jd_degrees: list) -> float:
    best_score = 0.0
    for jd in jd_degrees:
       
        jd_degree = (jd.get('degree') or '').lower()
        jd_degree_names = [(name or '').lower() for name in jd.get('degree_name', [])]
        jd_departments = [(dept or '').lower() for dept in jd.get('department', [])]
        
        jd_cgpa = jd.get('cgpa', None)
        if isinstance(jd_cgpa, str):
            jd_cgpa = jd_cgpa.replace('%', '').strip()
            jd_cgpa = float(jd_cgpa) if jd_cgpa and jd_cgpa.replace('.', '', 1).isdigit() else None
        
        jd_min_year = jd.get('min_year', None)
        jd_max_year = jd.get('max_year', None)
        jd_min_year = int(jd_min_year) if jd_min_year and str(jd_min_year).isdigit() else None
        jd_max_year = int(jd_max_year) if jd_max_year and str(jd_max_year).isdigit() else None
        
        for resume in resume_degrees:
            current_score = 0.0
            resume_degree = (resume.get('degree') or '').lower()
            resume_degree_name = (resume.get('degree_name') or '').lower()
            resume_department = (resume.get('department') or '').lower()
            
            resume_cgpa = resume.get('cgpa', 0) or 0
            if isinstance(resume_cgpa, str):
                resume_cgpa = resume_cgpa.replace('%', '').strip()
            resume_cgpa = float(resume_cgpa) if resume_cgpa and resume_cgpa.replace('.', '', 1).isdigit() else None
            
            resume_end_year = resume.get('end_year', '')
            if resume_end_year:
                if '/' in resume_end_year:
                    resume_end_year = resume_end_year.split('/')[-1]
                resume_end_year = resume_end_year.split()[0]
                resume_end_year = int(resume_end_year) if resume_end_year.isdigit() else None
            else:
                resume_end_year = None
            
            if jd_degree == resume_degree:
                if not jd_degree_names:
                    current_score += 0.3
                elif not resume_degree_name:
                    current_score += 0.0
                elif resume_degree_name in jd_degree_names:
                    current_score += 0.3
                
                if not jd_departments:
                    current_score += 0.3
                elif not resume_department:
                    current_score += 0.0
                elif resume_department in jd_departments:
                    current_score += 0.3
                
                if not jd_cgpa or (resume_cgpa and float(resume_cgpa) >= float(jd_cgpa)):
                    current_score += 0.2
                
                if jd_min_year is None and jd_max_year is None:
                    current_score += 0.2
                elif resume_end_year is None:
                    current_score += 0.0
                elif jd_min_year is not None and jd_max_year is not None:
                    if jd_min_year == jd_max_year and resume_end_year == jd_min_year:
                        current_score += 0.2
                    elif resume_end_year >= jd_min_year and resume_end_year <= jd_max_year:
                        current_score += 0.2
                elif jd_min_year is None or jd_max_year is None:
                    if jd_min_year:
                        if resume_end_year>=jd_min_year:
                            current_score +=0.2
                    elif jd_max_year:
                        if resume_end_year <=jd_max_year:
                            current_score +=0.2        
                print("4",current_score)
                best_score = max(best_score, current_score)
    
    return best_score

def job_similarity(title1, title2):
    doc1 = nlp(title1)
    doc2 = nlp(title2)
    print("yesssss")
    return doc1.similarity(doc2)

def compare_experience(resume_experience: dict, jd_experience: list) -> float:
    
    def is_valid_partial_match(jd_skill, resume_skill):
        if jd_skill in resume_skill or resume_skill in jd_skill:
            return True  
        similarity = fuzz.partial_ratio(jd_skill, resume_skill)
        return similarity > 90  
    resume_jobs = resume_experience.get("job", [])
    if not resume_jobs:
        return 0.0
    max_score = 0.0  
    for jd in jd_experience:
        jd_job_title = jd.get("job_titles", "").strip().lower() if jd.get("job_titles") else ""
        jd_skills = [skill.lower().strip() for skill in jd.get("skills", []) if skill] 
        #jd_skills = jd.get('skills', '').strip().lower() if jd.get('skills') else ''
        jd_min = float(jd.get('min_experience', 0) or 0)
        jd_max = float(jd.get('max_experience', 0) or 0)
        jd_experience_type = jd.get('experience_type', '').strip().lower()

        for resume in resume_jobs:
            resume_skills = [skill.lower() for skill in resume.get('Technology', [])]
            resume_duration = resume.get('duration', "0")
            resume_title = resume.get('job_title', '').lower()
            try:
                resume_duration = float(resume_duration) if resume_duration is not None else 0
            except ValueError:
                resume_duration = 0
            score = 0.0  
            print(resume_skills)
            print(jd_skills)
            #skill_match = any(is_valid_partial_match(jd, skill) for jd in jd_skills for skill in resume_skills) or fuzz.token_set_ratio(jd_job_title, resume_title) > 75 or jd_experience_type == "fresher"
            skill_match = any(is_valid_partial_match(jd_skill, skill) for jd_skill in jd_skills for skill in resume_skills)
            #title_match = fuzz.token_set_ratio(jd_job_title, resume_title) > 75 
            print(jd_job_title) 
            print(resume_title)
            title_match = job_similarity(jd_job_title, resume_title) > 0.75
            print(title_match)
            if skill_match or title_match or jd_experience_type == "fresher":
                print("Skills match!")
                score = 0.7 
                print(resume_duration)
                print(jd_min)
                print(jd_max)  
                if jd_min <= resume_duration <= jd_max:
                    print("Experience matches within range")
                    score += 0.3  
                elif resume_duration >= jd_max:
                    score += 0.3
                elif resume_duration >= jd_min:
                    score += 0.3
            max_score = max(max_score, score) 
            print(max_score)
    return max_score 

def compare_location(resume_location: dict, jd_locations: list) -> float:
    if not resume_location:
        return 0.0
    resume_district = (resume_location.get('district') or '').lower()
    resume_state = (resume_location.get('state') or '').lower()
    resume_country = (resume_location.get('country') or '').lower()
    best_match = 0.0
    for jd_loc in jd_locations:
        
        jd_district = (jd_loc.get('district') or '').lower()  
        jd_state = (jd_loc.get('state') or '').lower()
        jd_country = (jd_loc.get('country') or '').lower()
        if jd_district:
            if jd_district == resume_district:
                return 1.0
            elif jd_state and resume_state and jd_state == resume_state:
                best_match = max(best_match, 0.5)
            elif jd_country and resume_country and jd_country == resume_country:
                best_match = max(best_match, 0.25)
            continue
        elif jd_state:
            if resume_state and jd_state == resume_state:
                return 1.0
            elif jd_country and resume_country and jd_country == resume_country:
                best_match = max(best_match, 0.5)
            continue
        elif jd_country and resume_country:
            if jd_country == resume_country:
                return 1.0   
    return best_match

def compare_skills(resume_skills: dict, jd_skills: dict) -> float:
    #if not jd_skills.get('technical'):
        #return 1.0
    def normalize_skill(skill: str) -> str:
        return skill.lower().replace(' ', '').replace('-', '').replace('.', '').replace('_','')
    if not resume_skills.get('technical_skills'):
        return 0.0
    resume_skills_set = set(normalize_skill(skill) for skill in (resume_skills['technical_skills'] or []))
    jd_skills_set = set(normalize_skill(skill) for skill in (jd_skills['technical'] or []))
    matched_skills = len(resume_skills_set.intersection(jd_skills_set))
    total_skills_in_jd = len(jd_skills_set)
    return matched_skills / total_skills_in_jd if total_skills_in_jd > 0 else 0.0

def compare_languages(resume_languages: list, jd_languages: list) -> float:
    #if not jd_languages:
        #return 1.0
    if not resume_languages:
        return 0.0
    matched_languages = 0
    partial_match = 0
    proficiency_levels = {"fluent": 3, "intermediate": 2, "basic": 1}
    #print(resume_languages)
    #print(jd_languages)
    for resume_lang in resume_languages:
        for jd_lang in jd_languages:
            if (resume_lang.get('language') or '').lower() == (jd_lang.get('language') or '').lower():
                if not jd_lang.get('proficiency'):
                    matched_languages+=1
                elif proficiency_levels.get((resume_lang.get('proficiency') or '').lower(), 0) >= proficiency_levels.get((jd_lang.get('proficiency') or '').lower(), 0):
                    matched_languages += 1
                else:
                    partial_match += 1
    full_match_percentage = matched_languages / len(jd_languages) if jd_languages else 0
    partial_match_percentage = (partial_match / len(jd_languages) if jd_languages else 0) * 0.7
    loc=round(full_match_percentage + partial_match_percentage, 2)
    return round(full_match_percentage + partial_match_percentage, 2)

def compare_resume_to_jd(resume_data: dict, jd_data: dict) -> float:
    default_weights = {
        'location': 10,
        'language': 20,
        'skills': 25,
        'education': 25,
        'experience': 20
    }
    fields = {
        'location': bool(jd_data.get('location')),
        'language': bool(jd_data.get('language_required')),
        'skills': bool(jd_data.get('skills_required', {}).get('technical')),
        'education': bool(jd_data.get('education_required')),
        'experience': bool(jd_data.get('experience_required'))
    }
    present_fields = {field for field, present in fields.items() if present}
    if not present_fields:
        return 0.0  
    total_default_weight = sum(default_weights[field] for field in present_fields)
    scores = {}
    if fields['location']:
        scores['location'] = compare_location(resume_data.get('location', {}), jd_data.get('location', []))
        #print(f"location: {scores['location']}")
    if fields['language']:
        scores['language'] = compare_languages(resume_data.get('languages', []), jd_data.get('language_required', []))
        #print(f"language: {scores['language']}")
    if fields['skills']:
        scores['skills'] = compare_skills(resume_data.get('skills', {}), jd_data.get('skills_required', {}))
        #print(f"skills: {scores['skills']}")
    if fields['education']:
        scores['education'] = compare_education(resume_data.get('education', []), jd_data.get('education_required', []))
        #print(f"education: {scores['education']}")
    if fields['experience']:
        scores['experience'] = compare_experience(resume_data.get('experience', {}), jd_data.get('experience_required', []))
        print(f"experience: {scores['experience']}")
    weights = {}
    total_weight = 100
    for field in default_weights:
        if field in present_fields:
            weights[field] = (default_weights[field] / total_default_weight) * total_weight
        else:
            weights[field] = 0
    total_score = sum(scores[field] * weights[field] for field in scores) / 100
    print(f"Weights: {weights}")
    return round(total_score * 100, 2)


def get_resume_data(resume)-> dict:
    return {
        'id': resume.id,
        'name': resume.name or '',
        'email': resume.email or [],
        'phone': resume.phone or [],
        'location': resume.location or {},
        'expected_salary': resume.expected_salary or {},
        'summary': resume.summary or '',
        'skills': resume.skills or {},
        'experience': resume.experience or {},
        'education': resume.education or [],
        'languages': resume.languages or [],
        'projects': resume.projects or [],
        'certificates': resume.certificates or [],
        'search_summary': resume.search_summary or '',
        'raw_data':resume.raw_data or {}
    }
     
def get_job_description_data(jd) -> dict:
    return {
        'job_titles': jd.job_titles or [],
        'company': jd.company or '',
        'location': jd.locations or [],
        'employment_type': jd.employment_type or [],
        'salary': jd.salary_details or [],
        'skills_required': jd.skills_required or {},
        'experience_required': jd.experience_required or [],
        'education_required': jd.education_required or [],
        'language_required': jd.languages_required or [],
        'responsibilities': jd.responsibilities or [],
        'post_date': jd.post_date or '',
        'end_date': jd.end_date or '',
        'contact': jd.contact_us or '',
        'additional_information': jd.additional_information or '',
        'jd_summary': jd.search_summary or '',
        'raw_data':jd.raw_data or {},
        'id': jd.id
    }

@app.get("/all", response_class=HTMLResponse)
async def get_upload_resume(request: Request):
    return templates.TemplateResponse("uploadall.html", {"request": request})

@app.get("/", response_class=HTMLResponse)
async def get_upload_resume(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/uploadresume/")
async def upload_resume(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        upload_dir = "uploads"
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        
        file_location = os.path.join(upload_dir, file.filename)
        file_type=file.filename.split('.')[-1].lower()
        with open(file_location, "wb") as f:
            f.write(await file.read())
       
        job_id = str(uuid.uuid4())
        status="queued"
        job = Job(
            job_id=job_id,
            filepath=file_location,
            file_type=file_type,
            jobtype="Resume",
            status=status,
            err_message=""
        )
        db.add(job)
        db.commit()
        
        redis_client.rpush("job_queue", job_id)
        db.refresh(job)
        updated_status = job.status
        return {
            "job_id": job_id,
            "file_location": file_location,
            "file_type": file_type,
            "status": updated_status,
            "message": "Resume uploaded and processing started."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/jd/", response_class=HTMLResponse)
async def get_upload_jd(request:Request):
    return templates.TemplateResponse("uploadjd.html", {"request": request})

@app.post("/uploadjd/")
async def upload_jd(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        upload_dir = "uploads"
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        
        file_location = os.path.join(upload_dir, file.filename)
        file_type = file.filename.split('.')[-1].lower() 

        with open(file_location, "wb") as f:
            f.write(await file.read())

        job_id = str(uuid.uuid4())
        status = "queued"

        job = Job(
            job_id=job_id,
            filepath=file_location,
            file_type=file_type,
            jobtype="JD",
            status=status, 
            err_message=""
        )
        db.add(job)
        db.commit()

        redis_client.rpush("job_queue", job_id)

        db.refresh(job)
        updated_status = job.status

        return {
            "job_id": job_id,
            "file_location": file_location,
            "file_type": file_type,
            "status": updated_status,
            "message": "Job Description uploaded and processing started."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

abbreviations_dict = {
    "b.e": "bachelor of engineering",
    "b.tech": "bachelor of technology",
    "m.e": "master of engineering",
    "m.tech": "master of technology",
    "bsc": "bachelor of science",
    "ba": "bachelor of arts",
    "m.sc": "master of science",
    "m.a": "master of arts",
    "mba": "master of business administration",
    "phd": "doctor of philosophy",
    "cse": "computer science and engineering",
    "it": "information technology",
    "ece": "electronics and communication engineering",
    "eee": "electrical and electronics engineering",
    "me": "mechanical engineering",
    "civil": "civil engineering",
    "chem": "chemical engineering",
    "bio": "biological sciences",
    "ai": "artificial intelligence",
    "ds": "data science"
}

def replace_abbreviations_in_keyword(keyword: str) -> str:
    print(keyword)
    for abbr, full_form in abbreviations_dict.items():
        keyword = re.sub(r'\b' + abbr + r'\b', full_form, keyword, flags=re.IGNORECASE)
    return keyword

@app.get("/resumes/",response_model=ResumeResponseSchema)
async def resumes(
    keyword:Optional[str]=None,
    skills: Optional[List[str]] = Query(None, description="List of skills"),
    experience: Optional[List[str]] = Query(None, description="Experience in years"),
    education: Optional[List[str]] = Query(None, description="Education degrees"),
    location: Optional[List[str]] = Query(None, description="Preferred locations"),
    languages: Optional[List[str]] = Query(None, description="Known languages"),
    page: int = Query(1, alias="page", description="Page number (starting from 1)"),
    limit: int = Query(2, alias="limit", description="Number of resumes per page"), 
    db:Session=Depends(get_db)
):
    query=db.query(Resume)
    if skills:
        if isinstance(skills, list) and len(skills) == 1 and ',' in skills[0]:
            skills_list = [skill.strip() for skill in skills[0].split(',')]
        else:
            skills_list = skills
        query = apply_resume_filter(query, "skills", skills_list)
    if experience:
        if isinstance(experience, list) and len(experience) == 1 and ',' in experience[0]:
            experience_list = [experience.strip() for experience in experience[0].split(',')]
        else:
            experience_list = experience
        query = apply_resume_filter(query, "experience", experience_list)
    if education:
        if isinstance(education, list) and len(education) == 1 and ',' in education[0]:
            education_list = [education.strip() for education in education[0].split(',')]
        else:
            education_list = education
        query = apply_resume_filter(query, "education", education_list)
    if location:
        if isinstance(location, list) and len(location) == 1 and ',' in location[0]:
            location_list = [location.strip() for location in location[0].split(',')]
        else:
            location_list = location
        query = apply_resume_filter(query, "location", location_list)
        
    if languages:
        if isinstance(languages, list) and len(languages) == 1 and ',' in languages[0]:
            languages_list = [language.strip() for language in languages[0].split(',')]
        else:
            languages_list = languages
        query = apply_resume_filter(query, "languages", languages_list)
    if keyword:
        keyword=keyword.lower()
        processed_keyword=replace_abbreviations_in_keyword(keyword)
        processed_keyword=processed_keyword.replace(" ","")
        print("hi")
        print(processed_keyword)
        keyword=keyword.lower()
        keyword=replace_abbreviations_in_keyword(keyword)
        keyword=split_words(keyword)
        print("bye")
        print(keyword)
        query = query.filter(
            or_(
                func.to_tsvector('english', Resume.search_summary).op('@@')(
                    func.plainto_tsquery('english',  f"{keyword}:*")  
                ),
                ~func.to_tsvector('english', Resume.search_summary).op('@@')(
                    func.plainto_tsquery('english', f"{keyword}:*")
                ) & Resume.processed_search_summary.ilike(f"%{processed_keyword}%")  
            )
        )
        """
        query=query.filter(
            func.to_tsvector('english',Resume.search_summary).op('@@')(
                    func.plainto_tsquery('english', f"{keyword}:*")
                )
        )"""
    all_resume = query.all()
    total_count = len(all_resume)
    total_pages = (total_count + limit - 1) // limit
    offset = (page - 1) * limit
    paginated_resume = all_resume[offset: offset + limit]

    results = [get_resume_data(resume) for resume in paginated_resume]

    return {
        "message": "Displaying filtered and paginated job descriptions.",
        "data": results,
        "pagination": {
            "current_page": page,
            "next_page": page + 1 if page < total_pages else None,
            "total_pages": total_pages,
            "prev_page": page - 1 if page > 1 else None
        }
    }
    
@app.get("/resumes/{id}",response_model=ResumeDetailResponseSchema)
async def resume_id(
    id:int,
    db:Session=Depends(get_db)
):
    query=db.query(Resume).filter(Resume.id==id).first()
    if not query:
        return{
            "message":f"Not the jd details with specific {id}",
            "data":None
        }
    result=get_resume_data(query)
    return{
        "message":f"Display the resume with specific id {id}",
        "data":result
    }
@app.get("/jds/")
async def jds(
    keyword: Optional[str] = None,
    skills_required: Optional[List[str]] = Query(None, description="List of required skills"),
    experience_required: Optional[List[str]] = Query(None, description="Required years of experience"),
    education_required: Optional[List[str]] = Query(None, description="Required education levels"),
    location_required: Optional[List[str]] = Query(None, description="Preferred locations"),
    employment_type: Optional[List[str]] = Query(None, description="Preferred employment type"),
    job_titles:Optional[List[str]] = Query(None, description="Preferred job_titles"),
    company:Optional[List[str]]=Query(None,description="Company name"),
    page: int = Query(1, ge=1, description="Page number (starting from 1)"),
    limit: int = Query(2, description="Number of job descriptions per page"),
    db: Session = Depends(get_db)
):
    query = db.query(JobDescription)
    
    if skills_required:
        if isinstance(skills_required, list) and len(skills_required) == 1 and ',' in skills_required[0]:
            skills_list = [skill.strip() for skill in skills_required[0].split(',')]
        else:
            skills_list = skills_required
        query = apply_jd_filter(query, "skills_required", skills_list)
    
    if experience_required:
        if isinstance(experience_required, list) and len(experience_required) == 1 and ',' in experience_required[0]:
            experience_list = [experience.strip() for experience in experience_required[0].split(',')]
        else:
            experience_list = experience_required
        query = apply_jd_filter(query, "experience_required", experience_list)
    
    if education_required:
        if isinstance(education_required, list) and len(education_required) == 1 and ',' in education_required[0]:
            education_list = [education.strip() for education in education_required[0].split(',')]
        else:
            education_list = education_required
        query = apply_jd_filter(query, "education_required", education_list)
    
    if location_required:
        if isinstance(location_required, list) and len(location_required) == 1 and ',' in location_required[0]:
            location_list = [location.strip() for location in location_required[0].split(',')]
        else:
            location_list = location_required
        query = apply_jd_filter(query, "location_required", location_list)
    
    if employment_type:
        if isinstance(employment_type, list) and len(employment_type) == 1 and ',' in employment_type[0]:
            employment_type_list = [type.strip() for type in employment_type[0].split(',')]
        else:
            employment_type_list = employment_type
        query = apply_jd_filter(query, "employment_type", employment_type_list)
        
    if job_titles:
        if isinstance(job_titles, list) and len(job_titles) == 1 and ',' in job_titles[0]:
            job_titles_list = [title.strip() for title in job_titles[0].split(',')]
        else:
            job_titles_list = job_titles
        query = apply_jd_filter(query, "job_titles", job_titles_list)
        
    if company:
        if isinstance(company, list) and len(company) == 1 and ',' in company[0]:
            company_list = [com.strip() for com in company[0].split(',')]
        else:
            company_list = company
        query = apply_jd_filter(query, "company", company_list)
        
    if keyword:
        keyword=keyword.lower()
        processed_keyword=replace_abbreviations_in_keyword(keyword)
        processed_keyword=processed_keyword.replace(" ","")
        print(processed_keyword)
        keyword=keyword.lower()
        keyword=replace_abbreviations_in_keyword(keyword)
        keyword=split_words(keyword)
        print(keyword)
        query = query.filter(
            or_(
                func.to_tsvector('english', JobDescription.search_summary).op('@@')(
                    func.plainto_tsquery('english',  f"{keyword}:*")  
                ),
                ~func.to_tsvector('english', JobDescription.search_summary).op('@@')(
                    func.plainto_tsquery('english', f"{keyword}:*")
                ) & JobDescription.processed_search_summary.ilike(f"%{processed_keyword}%")
            )
        )
        print(query)
    all_jds = query.all()
    total_count = len(all_jds)
    total_pages = (total_count + limit - 1) // limit
    offset = (page - 1) * limit
    paginated_jds = all_jds[offset: offset + limit]

    results = [get_job_description_data(jd) for jd in paginated_jds]

    return {
        "message": "Displaying filtered and paginated job descriptions.",
        "data": results,
        "pagination": {
            "current_page": page,
            "next_page": page + 1 if page < total_pages else None,
            "total_pages": total_pages,
            "prev_page": page - 1 if page > 1 else None
        }
    }

    
@app.get("/jds/{id}",response_model=JobDescriptionDetailResponseSchema)
async def jds_id(
    id:int,
    db:Session=Depends(get_db)
):
    query=db.query(JobDescription).filter(JobDescription.id==id).first()
    if not query:
        return{
            "message":f"Not the jd details with specific {id}",
            "data":None
        }
    result=get_job_description_data(query)
    return{
        "message":f"Display the jd details with specific {id}",
        "data":result
    }
@app.get("/match_resumes/",response_model=ResumeResponseSchema)
async def match_resume(
    jd_id:Optional[int]=None,
    skills: Optional[List[str]] = Query(None, description="List of skills"),
    experience: Optional[List[str]] = Query(None, description="Experience in years"),
    education: Optional[List[str]] = Query(None, description="Education degrees"),
    location: Optional[List[str]] = Query(None, description="Preferred locations"),
    languages: Optional[List[str]] = Query(None, description="Known languages"),
    page: int = Query(1, alias="page", description="Page number (starting from 1)"),
    limit: int = Query(2, alias="limit", description="Number of resumes per page"),
    db:Session=Depends(get_db)
    
):  
    query=db.query(Resume)
    """def process_filter(query, field_name, field_value):
        if isinstance(field_value, list) and len(field_value) == 1 and ',' in field_value[0]:
            field_value = [item.strip() for item in field_value[0].split(',')]
        query = apply_resume_filter(query, field_name, field_value)
        return query

    if skills:
        query = process_filter(query, "skills", skills)
    if experience:
        query = process_filter(query, "experience", experience)
    if education:
        query = process_filter(query, "education", education)
    if location:
        query = process_filter(query, "location", location)
    if languages:
        query = process_filter(query, "languages", languages)"""
    if skills:
        if isinstance(skills, list) and len(skills) == 1 and ',' in skills[0]:
            skills_list = [skill.strip() for skill in skills[0].split(',')]
        else:
            skills_list = skills
        query = apply_resume_filter(query, "skills", skills_list)
    if experience:
        if isinstance(experience, list) and len(experience) == 1 and ',' in experience[0]:
            experience_list = [experience.strip() for experience in experience[0].split(',')]
        else:
            experience_list = experience
        query = apply_resume_filter(query, "experience", experience_list)
    if education:
        if isinstance(education, list) and len(education) == 1 and ',' in education[0]:
            education_list = [education.strip() for education in education[0].split(',')]
        else:
            education_list = education
        query = apply_resume_filter(query, "education", education_list)
    if location:
        if isinstance(location, list) and len(location) == 1 and ',' in location[0]:
            location_list = [location.strip() for location in location[0].split(',')]
        else:
            location_list = location
        query = apply_resume_filter(query, "location", location_list)
        
    if languages:
        if isinstance(languages, list) and len(languages) == 1 and ',' in languages[0]:
            languages_list = [language.strip() for language in languages[0].split(',')]
        else:
            languages_list = languages
        query = apply_resume_filter(query, "languages", languages_list)
    
    query=query.all()
    resume_data=[]
    
    for resume in query:
        resume_data.append(get_resume_data(resume))
        
    if jd_id:
        jdquery=db.query(JobDescription).filter(JobDescription.id==jd_id).first()
        if not jdquery:
            return {"message": f"Job Description with ID {jd_id} not found.", "data": []}
        jd_data=get_job_description_data(jdquery)
        matches = [
                {
                    **resume,
                    'match_percentage': compare_resume_to_jd(resume, jd_data)
                }
                for resume in resume_data
            ]
        matches.sort(key=lambda x: x['match_percentage'], reverse=True)
        total_resumes = len(matches)
        total_pages = (total_resumes + limit - 1) // limit
        offset = (page - 1) * limit
        paginated_matches = matches[offset: offset + limit]
        return {
            "message": "Displaying resumes with matching percentages.",
            "data": paginated_matches,
            "pagination": {
                "current_page": page,
                "next_page": page + 1 if page < total_pages else None,
                "total_pages": total_pages,
                "prev_page": page - 1 if page > 1 else None
            }
        }
    total_resumes = len(resume_data)
    total_pages = (total_resumes + limit - 1) // limit
    offset = (page - 1) * limit
    paginated_resumes = resume_data[offset: offset + limit]
    return {
        "message": "Displaying all resumes.",
        "data": paginated_resumes,
        "pagination": {
            "current_page": page,
            "next_page": page + 1 if page < total_pages else None,
            "total_pages": total_pages,
            "prev_page": page - 1 if page > 1 else None
        }
    
    }
@app.get("/match_jds/",response_model=JobDescriptionResponseSchema)
async def match_jds(
    resume_id: Optional[int] = None,
    skills_required: Optional[List[str]] = Query(None, description="List of skills"),
    experience_required: Optional[List[str]] = Query(None, description="Experience in years"),
    education_required: Optional[List[str]] = Query(None, description="Education degrees"),
    location_required: Optional[List[str]] = Query(None, description="Preferred locations"),
    employment_type: Optional[List[str]] = Query(None, description="Preferred employment type"),
    job_titles:Optional[List[str]] = Query(None, description="Preferred job_titles"),
    company:Optional[str]=Query(None,description="Company name"),
    page: int = Query(1, alias="page", description="Page number (starting from 1)"),
    limit: int = Query(2, alias="limit", description="Number of job descriptions per page"),
    db: Session = Depends(get_db)
):
    
    query = db.query(JobDescription)
    
    if skills_required:
        if isinstance(skills_required, list) and len(skills_required) == 1 and ',' in skills_required[0]:
            skills_list = [skill.strip() for skill in skills_required[0].split(',')]
        else:
            skills_list = skills_required
        query = apply_jd_filter(query, "skills_required", skills_list)
    
    if experience_required:
        if isinstance(experience_required, list) and len(experience_required) == 1 and ',' in experience_required[0]:
            experience_list = [experience.strip() for experience in experience_required[0].split(',')]
        else:
            experience_list = experience_required
        print("start")
        query = apply_jd_filter(query, "experience_required", experience_list)
    
    if education_required:
        if isinstance(education_required, list) and len(education_required) == 1 and ',' in education_required[0]:
            education_list = [education.strip() for education in education_required[0].split(',')]
        else:
            education_list = education_required
        query = apply_jd_filter(query, "education_required", education_list)
    
    if location_required:
        if isinstance(location_required, list) and len(location_required) == 1 and ',' in location_required[0]:
            location_list = [location.strip() for location in location_required[0].split(',')]
        else:
            location_list = location_required
        query = apply_jd_filter(query, "location_required", location_list)

    if employment_type:
        if isinstance(employment_type, list) and len(employment_type) == 1 and ',' in employment_type[0]:
            employment_type_list = [type.strip() for type in employment_type[0].split(',')]
        else:
            employment_type_list = employment_type
        query = apply_jd_filter(query, "employment_type", employment_type_list)
    if job_titles:
        if isinstance(job_titles, list) and len(job_titles) == 1 and ',' in job_titles[0]:
            job_titles_list = [title.strip() for title in job_titles[0].split(',')]
        else:
            job_titles_list = job_titles
        query = apply_jd_filter(query, "job_titles", job_titles_list)
    if company:
        if isinstance(company, list) and len(company) == 1 and ',' in company[0]:
            company_list = [com.strip() for com in company[0].split(',')]
        else:
            company_list = company
        query = apply_jd_filter(query, "company", company_list)
    
    all_results = query.all()
    jd_data_list = [get_job_description_data(jd) for jd in all_results]
    

    if resume_id:
        resume = db.query(Resume).filter(Resume.id == resume_id).first()
        if not resume:
            return {"message": f"Resume with ID {resume_id} not found.", "data": []}
        
        resume_data = get_resume_data(resume)
        
        sorted_matches = [
            {
                **jd_data,
                'match_percentage': compare_resume_to_jd(resume_data, jd_data)
            }
            for jd_data in jd_data_list
        ]
        sorted_matches.sort(key=lambda x: x['match_percentage'], reverse=True)
    else:
        sorted_matches = jd_data_list 

    total_count = len(sorted_matches)
    total_pages = (total_count + limit - 1) // limit
    offset = (page - 1) * limit
    paginated_results = sorted_matches[offset: offset + limit]

    return {
        "message": "Displaying job descriptions with matching percentages." if resume_id else "Displaying filtered job descriptions.",
        "data": paginated_results,
        "pagination": {
            "current_page": page,
            #"total_count": total_count,
            "total_pages": total_pages,
            "next_page": page + 1 if page < total_pages else None,
            "prev_page": page - 1 if page > 1 else None
        }
    }

def apply_resume_filter(query: Query, filter_key: str, filter_values: List[str]) -> Query:
    
    filter_mapping = {
        "skills":Resume.skills,
        "experience":Resume.experience,
        "education":Resume.education,
        "location":Resume.location,
        "languages":Resume.languages
    }

    if filter_key not in filter_mapping:
        raise ValueError(f"Invalid filter_key: {filter_key}")
    
    normalized_values = [value.strip().lower().replace(" ","") for value in filter_values if value]
    #normalized_values = [value for value in filter_values if value]
    
    print(normalized_values)
    if not normalized_values:
        return query

    column_attr = filter_mapping[filter_key]

    if isinstance(column_attr.type, JSONB):
        if filter_key == "education":
            query = query.filter(
                or_(
                    func.jsonb_path_exists(
                        column_attr,
                        f'$.degree_name[*] ? (@ like_regex ".*{value}.*" flag "i")'
                    )
                    for value in normalized_values
                )
            )
            
        elif filter_key == "skills":
            print("workingskills")
            query = query.filter(
                or_(
                    func.exists(
                        select(1)
                        .select_from(func.jsonb_array_elements_text(column_attr[skill_type]).alias("skill"))
                        .where(
                            func.lower(
                                func.replace(func.trim(text("skill")), ' ', '')
                            ).ilike(f"%{value}%")
                        )
                    )
                    for value in normalized_values
                    for skill_type in ["technical_skills", "soft_skills"]
                )
            )
                        
        elif filter_key == "location":
            query = query.filter(
                or_(
                    func.jsonb_path_exists(
                        column_attr,
                        f'$[*].state ? (@ like_regex ".*{value}.*" flag "i")'
                    ) |
                    func.jsonb_path_exists(
                        column_attr,
                        f'$[*].district ? (@ like_regex ".*{value}.*" flag "i")'
                    ) |
                    func.jsonb_path_exists(
                        column_attr,
                        f'$[*].country ? (@ like_regex ".*{value}.*" flag "i")'
                    )
                    for value in normalized_values
                )
            )

        elif filter_key == "languages":
            query=query.filter(
                or_(
                    func.jsonb_path_exists(
                        column_attr,
                        f'$[*].language ?(@ like_regex ".*{value}.*" flag "i")'
                    )
                    for value in normalized_values
                )
            )

        elif filter_key == "experience":
            print("ok")
            query = query.filter(
                or_(
                    func.jsonb_path_exists(
                        column_attr,
                        f'$.total_duration ?(@ >={float(value)})'
                    )
                    
                    for value in normalized_values
                )
            )
            
    return query
"""
def apply_resume_filter(query: Query, filter_key: str, filter_values: List[str]) -> Query:
    filter_mapping = {
        "skills": Resume.skills,
        "experience": Resume.experience,
        "education": Resume.education,
        "location": Resume.location,
        "languages": Resume.languages
    }
    
    if filter_key not in filter_mapping:
        raise ValueError(f"Invalid filter_key: {filter_key}")
    
    # Normalize values by removing extra spaces and converting to lowercase
    normalized_values = [
        value
        #value.strip().lower().replace(" ", "") 
        for value in filter_values 
        if value
    ]
    
    if not normalized_values:
        return query
    
    column_attr = filter_mapping[filter_key]
    
    print(type(column_attr))
    
    if isinstance(column_attr.type, JSONB):
        if filter_key == "education":
            print("education")
            query=query.filter(
                or_(
                    func.jsonb_path_exists(
                        column_attr,
                        f'$.degree_name[*] ? (@ like_regex ".*{value}.*" flag "i")'
                    )
                    for value in normalized_values
                )
            )
            
        elif filter_key == "skills":
            print("yes")
            print(normalized_values)
            query = query.filter(
                or_(
                    func.jsonb_path_exists(column_attr, 
                        f'$.technical_skills[*] ? (@  like_regex "{value}" flag "i")') |
                    func.jsonb_path_exists(column_attr,
                        f'$.soft_skills[*] ? (@ like_regex "{value}" flag "i")')
                    for value in normalized_values
                )
            )
            
        elif filter_key == "location":
            query = query.filter(
                or_(
                    func.regexp_replace(cast(column_attr["state"],String),' ', '', 'g').ilike(f"%{value}%") |
                    func.regexp_replace(cast(column_attr["district"],String), ' ', '', 'g').ilike(f"%{value}%") |
                    func.regexp_replace(cast(column_attr["country"],String), ' ', '', 'g').ilike(f"%{value}%")
                    for value in normalized_values 
                )
            )
            
            
          
        elif filter_key == "languages":
            query = query.filter(
                or_(
                    func.jsonb_path_exists(column_attr,
                        f'$[*].language ? (@ like_regex "{value}" flag "i")')
                    for value in normalized_values
                )
            )
            
        elif filter_key == "experience":
            query = query.filter(
                or_(
                    func.jsonb_path_exists(column_attr,
                        f'$.total_duration ? (@ >= {float(value)})')
                    for value in normalized_values
                )
            )
    
    return query

"""
def apply_jd_filter(query: Query, filter_key: str, filter_values: List[str]) -> Query:
    filter_mapping = {
    "job_titles": JobDescription.job_titles,
    "company": JobDescription.company,
    "location_required": JobDescription.locations,
    "employment_type": JobDescription.employment_type,
    "skills_required": JobDescription.skills_required,
    "experience_required": JobDescription.experience_required,
    "education_required":JobDescription.education_required,
}

    if filter_key not in filter_mapping:
        raise ValueError(f"Invalid filter_key: {filter_key}")
    
    normalized_values = [value.strip().lower().replace(" ","") for value in filter_values if value]
    #normalized_values=[value for value in filter_values if value]
    print(normalized_values)
    if not normalized_values:
        return query

    column_attr = filter_mapping[filter_key]

    if isinstance(column_attr.type, ARRAY):
        if filter_key == "job_titles":
            
            print("job_titles")
            query = query.filter(
                or_(
                    func.replace(func.array_to_string(column_attr, ','), ' ', '').ilike(f"%{value}%")
                    for value in normalized_values
                )
            )

        if filter_key == "employment_type":
            query = query.filter(
                or_(
                    func.array_to_string(column_attr, ',').ilike(f"%{value}%")
                    for value in normalized_values
                )
            )

    elif isinstance(column_attr.type, JSONB):
        print("start")
        
        if filter_key == "education_required":
            
            print("yes")
            query = query.filter(
                or_(
                    func.exists(
                        select(1)
                        .select_from(
                            func.jsonb_array_elements(column_attr).alias('edu_obj')
                        )
                        .where(
                            func.exists(
                                select(1)
                                .select_from(
                                    func.jsonb_array_elements(
                                        cast(func.coalesce(text("edu_obj->'degree_name'"), '[]'), JSONB)
                                    ).alias('degree')
                                )
                                .where(
                                    func.lower(
                                        func.replace(
                                            func.trim(text('degree::text')),
                                            ' ',
                                            ''
                                        )
                                    ).ilike(f'%{value}%')
                                )
                            )
                        )
                    )
                    for value in normalized_values
                )
            )     
        
        elif filter_key == "skills_required":
            query = query.filter(
                or_(
                    func.exists(
                        select(1)
                        .select_from(func.jsonb_array_elements_text(column_attr["technical"]).alias("tech_skill"))
                        .where(
                            func.lower(
                                func.replace(func.trim(text("tech_skill")), ' ', '')
                            ).ilike(f"%{value}%")
                        )
                    )
                    for value in normalized_values
                )
            )
            
        elif filter_key == "location_required":
            print("yes")
            query = query.filter(
                or_(
                    func.exists(
                        select(1)
                        .select_from(
                            func.jsonb_array_elements(column_attr).alias('loc_obj')
                        )
                        .where(
                            func.lower(
                                func.replace(
                                    func.trim(text("loc_obj->>'district'")),
                                    ' ',
                                    ''
                                )
                            ).ilike(f'%{value}%')
                        )
                    )
                    for value in normalized_values
                )
            )
        elif filter_key == "experience_required":
            query = query.filter(
                or_(
                    *[
                        and_(
                            func.jsonb_path_exists(
                                column_attr,
                                f'$.min_experience ? (@ <= {value.split("-")[0]})'
                            ),
                            func.jsonb_path_exists(
                                column_attr,
                                f'$.max_experience ? (@ >= {value.split("-")[1]})'
                            )
                        )
                        for value in normalized_values
                    ]
                )
            )
        else:
        #if isinstance(column_attr,String):
            print("lets start")
            query = query.filter(
                or_(
                    func.replace(column_attr," ","").ilike(f"%{value}%")
                    for value in normalized_values
                )
            )
    return query 


"""
def apply_jd_filter(query: Query, filter_key: str, filter_values: List[str]) -> Query:
    filter_mapping = {
        "job_titles": func.array_to_string(JobDescription.job_titles, ',').astext,
        "company": JobDescription.company.astext,
        "location_required": JobDescription.locations.astext,
        "employment_type": func.array_to_string(JobDescription.employment_type, ',').astext,
        "skills_required": JobDescription.skills_required.astext,
        "experience_required": JobDescription.experience_required.astext,
        "education_required": JobDescription.education_required.astext,
    }

    if filter_key not in filter_mapping:
        raise ValueError(f"Invalid filter_key: {filter_key}")
    
    normalized_values = [value.strip().lower().replace(" ", "") for value in filter_values if value]
    if not normalized_values:
        return query

    column_attr = filter_mapping[filter_key]

    if isinstance(column_attr.type, JSONB):
        if filter_key == "education_required":
            print("educationsss")
            query = query.filter(
                or_(
                    func.jsonb_path_exists(
                        column_attr, f'$.degree_name[*] ? (@ like_regex ".*{value}.*" flag "i")'
                    )
                    for value in normalized_values
                )
            )
        elif filter_key == "skills_required":
            query = query.filter(
                or_(
                    func.jsonb_path_exists(
                        column_attr, f'$.technical[*] ? (@ like_regex ".*{value}.*" flag "i")'
                    ) |
                    func.jsonb_path_exists(
                        column_attr, f'$.soft[*] ? (@ like_regex ".*{value}.*" flag "i")'
                    )
                    for value in normalized_values
                )
            )
        elif filter_key == "location_required":
            query = query.filter(
                or_(
                    func.jsonb_path_exists(
                        column_attr, f'$[*].state ? (@ like_regex ".*{value}.*" flag "i")'
                    ) |
                    func.jsonb_path_exists(
                        column_attr, f'$[*].district ? (@ like_regex ".*{value}.*" flag "i")'
                    ) |
                    func.jsonb_path_exists(
                        column_attr, f'$[*].country ? (@ like_regex ".*{value}.*" flag "i")'
                    )
                    for value in normalized_values
                )
            )
        elif filter_key == "experience_required":
            query = query.filter(
                or_(*[
                    and_(
                        func.jsonb_path_exists(column_attr, f'$.min_experience ? (@ == {value.split("-")[0]})'),
                        func.jsonb_path_exists(column_attr, f'$.max_experience ? (@ == {value.split("-")[1]})')
                    )
                    for value in normalized_values
                ])
            )
    else:
        query = query.filter(
            or_(
                func.lower(func.replace(column_attr, " ", "")).ilike(f"%{value}%")
                for value in normalized_values
            )
        )
    return query
"""