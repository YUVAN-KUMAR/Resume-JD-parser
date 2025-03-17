from sqlalchemy import Column, Integer, String, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import ARRAY,JSONB

Base = declarative_base()

class Resume(Base):
    __tablename__ = 'resumess'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    email = Column(ARRAY(String), nullable=True) 
    phone = Column(ARRAY(String), nullable=True) 
    location = Column(JSONB, nullable=True)
    expected_salary = Column(JSON, nullable=True)
    summary = Column(Text, nullable=True) 
    skills = Column(JSONB, nullable=True)
    experience = Column(JSONB, nullable=True) 
    education = Column(JSONB, nullable=True) 
    languages = Column(JSONB, nullable=True)  
    projects = Column(JSON, nullable=True) 
    certificates = Column(JSON, nullable=True)  
    search_summary = Column(String, nullable=False) 
    processed_search_summary=Column(String,nullable=False)
    raw_data=Column(JSON,nullable=True)

class JobDescription(Base):
    __tablename__ = "jdssss"

    id = Column(Integer, primary_key=True, index=True)
    job_titles = Column(ARRAY(String), nullable=False)
    company = Column(String, nullable=True)
    locations = Column(JSONB, nullable=True)
    employment_type = Column(ARRAY(String), nullable=True)
    salary_details = Column(JSON, nullable=True)
    responsibilities = Column(ARRAY(String), nullable=True) 
    skills_required = Column(JSONB, nullable=True) 
    experience_required = Column(JSONB, nullable=True)
    languages_required = Column(JSON, nullable=True) 
    education_required = Column(JSONB, nullable=True)  
    post_date = Column(String, nullable=True) 
    end_date = Column(String, nullable=True) 
    contact_us = Column(JSON, nullable=True)  
    additional_information = Column(String, nullable=True)
    search_summary=Column(String,nullable=False)
    processed_search_summary=Column(String,nullable=False)
    raw_data=Column(JSON,nullable=False)
    
    
class Job(Base):
    __tablename__ = "jobs"
    
    job_id = Column(String, primary_key=True, index=True)
    filepath = Column(String, index=True)
    file_type = Column(String)
    jobtype = Column(String)  
    status = Column(String)
    err_message = Column(Text)

