import pandas as pd
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.callbacks import get_openai_callback
import os

# Set environment variables
os.environ["LANGCHAIN_TRACING"] = "true"
os.environ["LANGCHAIN_SESSION"] = "pdf_query"
os.environ["OPENAI_API_KEY"] = "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"

# Load resume data
pdf_folder_path = "data/"
loader = [UnstructuredPDFLoader(os.path.join(pdf_folder_path, fn)) for fn in os.listdir(pdf_folder_path)]

# Create skill matrix
with get_openai_callback() as cb:
    index = VectorstoreIndexCreator().from_loaders(loader)
    skill_matrix = pd.DataFrame(columns=["Requirements and Skills", "Match"])
    
    # Add JD skills to the skill matrix
    jd_skills = [
        "Python and associated frameworks such as Django and Flask",
        "Python's multi-process architecture and threading restrictions",
        "Server-side templating languages such as Jinja 2 and Mako",
        "Integration of many data sources into a single system",
        "Collaboration on projects and ability to work independently",
        "Ability to test Python programs successfully",
        "Excellent communication and teamwork abilities",
        "Knowledge of front-end technology like JavaScript and HTML5",
        "Excellent problem-solving abilities",
        "Bachelor of Science in Computer Science, Engineering, or a related discipline",
        "Understanding of front-end technologies such as JavaScript, HTML5, and CSS3",
        "Knowledge of some ORM (Object Relational Mapper) libraries",
        "Combining multiple data sources and databases into a single system",
        "User authentication and authorization knowledge across systems, servers, and environments"
    ]
    skill_matrix["Requirements and Skills"] = jd_skills
    
    # Check resume skills against JD skills
    for doc in index.documents:
        resume_skills = doc.metadata["skills"]
        match_skills = []
        for skill in jd_skills:
            if any(keyword.lower() in skill.lower() for keyword in resume_skills):
                match_skills.append("Yes")
            else:
                match_skills.append("No")
        skill_matrix[doc.metadata["filename"]] = match_skills
    
    print(skill_matrix)
    print(cb)
