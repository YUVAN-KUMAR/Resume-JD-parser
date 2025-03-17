from typing import List, Optional,Dict,Union,Any
from pydantic import BaseModel

class Location(BaseModel):
    street: Optional[str] = None
    district: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None

class ExpectedSalary(BaseModel):
    salary: Optional[float] = None
    currency: Optional[str] = None
    per: Optional[str] = None

class Experience(BaseModel):
    job_title: Optional[str] = None
    company: Optional[str] = None
    Technology:Optional[List[str]]=None
    start_year: Optional[str] = None 
    end_year: Optional[str] = None 
    duration: Optional[float] = None 
    responsibilities: Optional[List[str]] = None
    
class ExperienceSchema(BaseModel):
    job: List[Experience] 
    total_duration: Optional[float] = None 
    

class Education(BaseModel):
    degree: Optional[str] = None
    degree_name:Optional[str]=None
    department: Optional[str] = None
    institution: Optional[str] = None
    start_year: Optional[str] = None 
    end_year: Optional[str] = None  
    duration:Optional[int]=None
    cgpa: Optional[Union[float, str]] = None  

class Language(BaseModel):
    language: Optional[str] = None
    proficiency: Optional[str] = None

class Skills(BaseModel):
    technical_skills: List[str] = []
    soft_skills: List[str] = []

class Project(BaseModel):
    title: Optional[str] = None
    skills: Optional[List[str]] = None
    responsibilities: Optional[List[str]] = None
    
class Certificate(BaseModel):
    title: Optional[str] = None
    issuer: Optional[str] = None
    year: Optional[str] = None
    
class RawDataSchema(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    expected_salary: Optional[str] = None
    summary: Optional[str] = None
    experience: Optional[List[Dict[str, Any]]] = None
    education: Optional[List[Dict[str, Any]]] = None
    languages: Optional[List[str]] = None
    skills: Optional[Dict[str, List[str]]] = None    
    projects: Optional[List[Dict[str, Any]]] = None
    certificates: Optional[List[str]] = None
    search_summary: Optional[str] = None

class ResumeSchema(BaseModel):
    id: int 
    name: str
    email: Optional[List[str]] = None
    phone: Optional[List[str]] = None
    location: Optional[Location]=None
    expected_salary: Optional[ExpectedSalary] = None
    summary: Optional[str] = None
    experience: Optional[ExperienceSchema] = None 
    #experience: Optional[List[Experience]]=None
    education: Optional[List[Education]] = None
    languages: Optional[List[Language]]= None
    skills: Optional[Skills]=None
    projects: Optional[List[Project]]= None
    certificates: Optional[List[Certificate]]= None
    search_summary: Optional[str] = None
    raw_data: Optional[RawDataSchema] = None
    match_percentage: Optional[float] = None
    
class PaginationSchema(BaseModel):
    current_page: int
    next_page: Optional[int] = None
    total_pages: int
    prev_page: Optional[int] = None
    
class ResumeResponseSchema(BaseModel):
    message: str
    data: List[ResumeSchema]=[]
    pagination: Optional[PaginationSchema]=None
    
class ResumeDetailResponseSchema(BaseModel):
    message: str
    data: Optional[ResumeSchema]=None


class Locations(BaseModel):
    street: Optional[str] = None
    district: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None

class SalaryDetails(BaseModel):
    salary: Optional[str] = None
    currency: Optional[str] = None
    per: Optional[str] = None  
    position: Optional[str] = None 

class ExperienceRequired(BaseModel):
    job_titles:Optional[str]=None
    skills: Optional[List[str]] = None
    min_experience: Optional[int] = None
    max_experience: Optional[int] = None
    experience_range: Optional[str] = None

class EducationRequired(BaseModel):
    degree: Optional[str] = None
    degree_name: Optional[List[str]] = None
    department: Optional[List[str]] = None
    year: Optional[List[str]] = None
    min_year: Optional[str] = None
    max_year: Optional[str] = None
    cgpa: Optional[str] = None

class LanguageRequired(BaseModel):
    language: Optional[str] = None
    proficiency: Optional[str] = None

class SkillsRequired(BaseModel):
    technical: Optional[List[str]] = None
    soft: Optional[List[str]] = None

class ContactUs(BaseModel):
    type: Optional[str] = None
    value: Optional[List[str]] = None
    
class rawContactDetails(BaseModel):
    type: str
    value: List[str] = []

class RawDataSchemaJD(BaseModel):
    job_title: Optional[List[str]] = None
    company: Optional[str] = None
    location: Optional[List[str]] = None
    employment_types: Optional[List[str]] = None
    salary_detail: Optional[List[str]] = None
    responsibilitie: Optional[List[str]] = None
    skills: Optional[Dict[str, List[str]]] = None
    education: Optional[List[str]] = None
    experience: Optional[List[str]] = None
    language: Optional[List[str]] = None
    post_date: Optional[str] = None
    end_date: Optional[str] = None
    contact: Optional[List[rawContactDetails]] = None
    additional_information: Optional[str] = None
    jd_summary: Optional[str] = None

class JobDescriptionSchema(BaseModel):
    id: int
    job_titles :List[str]
    company: str
    location: Optional[List[Locations]] = None  
    employment_type: Optional[List[str]] = None  
    salary: Optional[List[SalaryDetails]] = None 
    responsibilities: Optional[List[str]] = None
    skills_required: Optional[SkillsRequired] = None
    experience_required: Optional[List[ExperienceRequired]] = None
    education_required: Optional[List[EducationRequired]] = None
    language_required: Optional[List[LanguageRequired]] = None
    post_date: Optional[str] = None
    end_date: Optional[str] = None
    contact: Optional[List[ContactUs]] = None
    additional_information: Optional[str] = None
    jd_summary: Optional[str]=None
    raw_data: Optional[RawDataSchemaJD] = None
    match_percentage: Optional[float] = None

class PaginationSchemaJD(BaseModel):
    current_page: int
    next_page: Optional[int] = None
    total_pages: int
    prev_page: Optional[int] = None

class JobDescriptionResponseSchema(BaseModel):
    message: str
    data: List[JobDescriptionSchema]=[]
    pagination: Optional[PaginationSchemaJD] = None

class JobDescriptionDetailResponseSchema(BaseModel):
    message: str
    data: Optional[JobDescriptionSchema] = None

