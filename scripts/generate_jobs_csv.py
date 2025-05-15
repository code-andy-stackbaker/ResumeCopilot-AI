# scripts/generate_jobs_csv.py
import csv
import random

NUM_RECORDS = 1500
OUTPUT_CSV_FILE = "generated_jobs_1500.csv" # You can change this if you changed it before, e.g., "generate_jobs_csv.csv"

# --- Data Pools for Variety ---
JOB_TITLES = [
    "Senior Software Engineer (Python)", "Lead Frontend Developer (React)", "Principal Data Scientist (NLP/CV)",
    "Cloud Solutions Architect (AWS)", "DevOps Team Lead (Kubernetes)", "Full Stack Engineer (Node.js/Vue.js)",
    "Machine Learning Engineer (Recommender Systems)", "Backend Developer (Java/Spring Microservices)",
    "Azure Cloud Engineer", "Software Engineer (Golang/Distributed Systems)", "iOS Application Developer (Swift/SwiftUI)",
    "Junior Full Stack Developer", "QA Automation Lead (Selenium/Cypress)", "Database Architect (PostgreSQL/NoSQL)",
    "Lead UX/UI Designer (Figma)", "Senior Technical Writer (API Documentation)", "Data Engineering Manager",
    "Cybersecurity Operations Lead", "Head of Product (B2B SaaS)", "Staff Software Engineer (Platform)",
    "AI Research Scientist (LLMs)", "Linux Systems Architect", "Technical Program Manager (AI/ML)",
    "Embedded Firmware Engineer (C/C++)", "Data Visualization Expert (Tableau/D3.js)", "MLOps Engineer (Kubeflow/SageMaker)",
    "Director of Information Security (CISO)", "Cloud Data Warehouse Specialist (Snowflake/BigQuery)",
    "Senior Vue.js Developer", "Automation Solutions Engineer (Python/RPA)", "Enterprise Java Architect",
    "Distinguished Software Engineer", "SQL Database Developer (MSSQL/Oracle)", "Lead Blockchain Engineer (Solidity)",
    "Customer Success Director (Enterprise Tech)", "Agile Coach / Scrum Master Lead", "Cloud Network Security Architect",
    "Deep Reinforcement Learning Researcher", "Principal Site Reliability Engineer (SRE)", "SAP Solution Architect (FICO/SD)",
    "Mobile Architect (iOS/Android)", "Head of Data Governance", "Senior Embedded Linux Developer (Yocto)",
    "Digital Creative Director", "Business Intelligence Manager", "FinOps Lead (Cloud Cost Optimization)",
    "Android Performance Engineer", "Real-Time Data Platform Lead (Kafka/Spark)", "Lead DFIR Analyst",
    "VP of Engineering - Mobile Platforms", "API Integration Architect", "Chief Data Warehouse Architect",
    "Executive Game Producer", "IT GRC Manager (Governance, Risk, Compliance)", "Lead Computer Vision Engineer (SLAM)",
    "Head of Product Design (UX/UI) - Enterprise", "Principal Machine Learning Scientist (Anomaly Detection)",
    "Senior Sales Engineer (Cloud & AI)", "Lead Distributed Systems Engineer (Databases)", "Chief Data Scientist"
]

COMPANY_PREFIXES = [
    "Innovate", "Cloud", "Data", "NextGen", "Code", "AI Core", "Enterprise", "Digital", "MicroLink", "AppCraft",
    "Web", "Quality", "SecureDB", "Pixel", "DocuTech", "Streamline", "MobileFirst", "Analytics", "SecureNet",
    "Future", "InfraServe", "Bridgepointe", "BigData", "MobileMinds", "ReliableWeb", "CRM", "PixelPlay",
    "ConnectSphere", "ClientFirst", "AgileFlow", "CoreEmbedded", "Insightful", "ModelDeploy", "SecureHoldings",
    "DataLake", "VueFront", "AutoTask", "ArchInnovate", "LeadSoft", "MSSQL", "Decentralized", "ClientValue",
    "AgileSprint", "NetGuard", "RLInnovate", "Uptime", "EnterpriseCore", "LegacyMobile", "DataTrust",
    "KernelWorks", "AdSpark", "DataDriven", "SpendWise", "SpeedyApp", "StreamScale", "CyberTrace",
    "AppSuite", "ConnectAPI", "Warehouse", "PlayForge", "Regulytix", "SpatialMap", "ProWorkflow",
    "PatternWatch", "CloudSell", "ScaleDB", "InsightPrime"
]

COMPANY_SUFFIXES = [
    "Tech", "Solutions", "Inc.", "Labs", "Systems", "Co.", "Group", "Platforms", "Dynamics", "Services",
    "Studio", "Experts", "Software", "Hosting", "Analytics", "Global", "Digital", "AI", "Data", "Corp",
    "Limited", "Enterprises"
]

LOCATIONS = [
    "San Francisco, CA", "New York, NY", "Austin, TX", "Seattle, WA", "Boston, MA", "Chicago, IL", "Dallas, TX",
    "Los Angeles, CA", "Portland, OR", "Raleigh, NC", "San Diego, CA", "Denver, CO", "Miami, FL",
    "Washington D.C.", "Atlanta, GA", "Cambridge, MA", "Phoenix, AZ", "Pittsburgh, PA", "Remote"
]

RESPONSIBILITY_PHRASES = [
    "Design, develop, and deploy high-quality software solutions.",
    "Collaborate with cross-functional teams including product managers, designers, and other engineers.",
    "Write clean, maintainable, and well-tested code following established best practices.",
    "Participate actively in code reviews, providing constructive feedback and adhering to quality standards.",
    "Troubleshoot, debug, and resolve complex technical issues in both production and development environments.",
    "Lead architectural design discussions and contribute to making key technical decisions for new features and systems.",
    "Mentor junior engineers, foster a collaborative learning environment, and help build team expertise in modern technologies.",
    "Drive innovation by researching, evaluating, and staying up-to-date with emerging technologies, frameworks, and industry trends.",
    "Develop, maintain, and improve CI/CD pipelines for automated building, testing, and deployment processes.",
    "Optimize applications and systems for maximum performance, scalability, fault tolerance, and overall reliability.",
    "Create and maintain comprehensive technical documentation, including design specifications, API guides, and operational procedures.",
    "Work closely with business stakeholders and product owners to gather and refine requirements, translating them into actionable technical specifications.",
    "Manage and optimize cloud infrastructure resources (e.g., on AWS, Azure, GCP) ensuring security, availability, and cost-effectiveness.",
    "Implement, configure, and maintain robust monitoring, logging, and alerting systems to ensure system health and proactive issue detection.",
    "Conduct in-depth research, build proof-of-concepts, and experiment with new AI/ML models, algorithms, and techniques."
]

QUALIFICATION_PHRASES = [
    "A Bachelor's or Master's degree in Computer Science, Software Engineering, Data Science, or a closely related technical field.",
    "Demonstrable {years}+ years of professional experience as a {title_base} or in a similar capacity, with a strong portfolio of successful projects.",
    "Expert-level proficiency in {skill1}, and strong working knowledge of {skill2}, and {skill3}.",
    "A solid and comprehensive understanding of core software development principles, design patterns, and agile best practices.",
    "Proven experience working effectively within Agile/Scrum development methodologies and a fast-paced environment.",
    "Exceptional problem-solving, critical thinking, and analytical skills with a keen attention to detail.",
    "Excellent verbal and written communication skills, with the ability to articulate complex technical ideas clearly to diverse audiences.",
    "Demonstrated ability to work independently with minimal supervision, as well as collaboratively as part of a larger, distributed team.",
    "Proficiency with modern version control systems like Git and collaborative platforms such as GitHub or GitLab.",
    "Significant hands-on experience with major cloud platforms such as AWS (EC2, S3, RDS, Lambda), Azure (VMs, Blob, SQL DB), or GCP (Compute Engine, GCS, Cloud SQL).",
    "Practical experience with containerization technologies (Docker) and orchestration tools (Kubernetes, Docker Swarm).",
    "A track record of successfully delivering complex, scalable software projects from ideation through to deployment and maintenance.",
    "Deep expertise in relational database design, schema management, query optimization (SQL), and experience with NoSQL databases (e.g., MongoDB, Cassandra, Redis).",
    "Extensive experience in designing, developing, and consuming RESTful APIs, with bonus points for GraphQL experience.",
    "Contributions to open-source projects or a strong public portfolio (e.g., GitHub) showcasing your skills and passion is highly regarded."
]

SKILL_SETS = [
    ["Python", "Django", "PostgreSQL"], ["JavaScript", "React", "Node.js"], ["Java", "Spring Boot", "Microservices"],
    ["C#", ".NET Core", "Azure DevOps"], ["Golang", "Kubernetes", "gRPC"], ["Swift", "iOS Development", "SwiftUI"],
    ["Kotlin", "Android Development", "Jetpack Compose"], ["TypeScript", "Angular", "NgRx"], ["Scala", "Apache Spark", "Apache Kafka"],
    ["Ruby on Rails", "GraphQL APIs", "Sidekiq"], ["PHP", "Laravel", "Vue.js"], ["Terraform", "Ansible", "AWS CloudFormation"],
    ["Prometheus", "Grafana", "Elasticsearch (ELK)"], ["TensorFlow", "PyTorch", "Scikit-learn"], ["Natural Language Processing (NLP)", "Large Language Models (LLMs)", "spaCy"],
    ["Computer Vision", "OpenCV", "PyTorch"], ["Data Engineering", "ETL Pipelines", "Airflow"], ["Cybersecurity", "SIEM", "Penetration Testing"]
]

COMPANY_BLURBS = [
    "We are a fast-growing technology leader in the {domain} space, consistently pushing the boundaries of innovation and committed to delivering excellence.",
    "Our core mission is to revolutionize the {domain} industry by developing cutting-edge solutions, fostering a customer-centric approach, and building a world-class team.",
    "Join a dynamic, diverse, and collaborative team that highly values continuous learning, professional growth, and making a tangible impact in the ever-evolving field of {domain}.",
    "We offer a highly competitive salary package, comprehensive benefits, stock options, and a flexible, supportive work environment that encourages tackling complex {domain} challenges.",
    "As an recognized industry pioneer and thought leader in {domain}, we are actively seeking passionate, driven individuals to help us define and shape the future of this exciting space."
]
DOMAIN_AREAS = ["cloud computing services", "artificial intelligence research", "fintech innovation", "digital healthcare technology", "global e-commerce platforms", "advanced cybersecurity solutions", "big data analytics", "next-generation mobile applications", "enterprise SaaS solutions", "immersive gaming experiences", "IoT and Edge Computing", "sustainable energy tech"]

def generate_job_description(title, company, location):
    """Generates a somewhat realistic and longer job description."""
    # Select a primary skill set and some additional preferred skills
    primary_skills = random.choice(SKILL_SETS)
    additional_skills_pool = [skill for skillset in SKILL_SETS if skillset != primary_skills for skill in skillset] + \
                             ["Agile methodologies", "CI/CD tools", "data visualization", "machine learning frameworks", "technical leadership", "project management"]
    preferred_skills_count = random.randint(3, 5)
    preferred_skills = random.sample(additional_skills_pool, k=min(preferred_skills_count, len(additional_skills_pool)))

    # Base title for qualification phrase
    title_base = title.split('(')[0].split('-')[0].strip()
    if "developer" in title_base.lower() or "engineer" in title_base.lower():
        years_experience = random.randint(3, 10) # More relevant for dev roles
    else:
        years_experience = random.randint(2, 8)


    description_parts = [
        f"Company Overview: {random.choice(COMPANY_BLURBS).format(domain=random.choice(DOMAIN_AREAS))}\n",
        f"Position: {title}\nLocation: {('Fully Remote, Global' if location == 'Remote' else location)}\nCompany: {company}\n",
        f"Job Summary:\nWe are currently seeking a highly skilled and motivated {title} to join our talented and forward-thinking team {('globally (as a remote position)' if location == 'Remote' else f'at our {location} office')}. This is an exceptional opportunity to work on challenging, impactful projects, contribute to a market-leading product, and grow your career in a supportive environment. The ideal candidate will be a proactive problem-solver with a passion for technology and a drive to deliver high-quality results.\n",
        "Key Responsibilities Include:",
        "- " + "\n- ".join(random.sample(RESPONSIBILITY_PHRASES, k=min(random.randint(4, 6), len(RESPONSIBILITY_PHRASES)))),
        "\nRequired Qualifications & Experience:",
        "- " + "\n- ".join(random.sample(QUALIFICATION_PHRASES, k=min(random.randint(4, 6), len(QUALIFICATION_PHRASES)))).format(
            years=years_experience, 
            title_base=title_base, 
            skill1=primary_skills[0], 
            skill2=primary_skills[1], 
            skill3=primary_skills[2] if len(primary_skills) > 2 else primary_skills[0]
        ),
        "\nPreferred Skills & Bonus Qualifications:",
        "- Experience with " + ", ".join(preferred_skills) + ".",
        "- Demonstrated ability to lead projects or mentor team members.",
        "- Strong understanding of {secondary_domain} or experience in a related industry.".format(secondary_domain=random.choice(DOMAIN_AREAS)),
        f"\nWhat We Offer:\n- A competitive compensation package including salary, bonus, and stock options.\n- Comprehensive health, dental, and vision benefits.\n- Generous paid time off and flexible working hours.\n- Opportunities for professional development, training, and conference attendance.\n- A vibrant and inclusive company culture with regular team events.\n\nIf you are a results-driven {title_base} with a passion for {primary_skills[0]}, {primary_skills[1]}, and tackling complex challenges, we strongly encourage you to apply. Join {company} and help us build the future!"
    ]
    return "\n\n".join(description_parts) # Use double newline for paragraph breaks

def generate_jobs_csv(filename=OUTPUT_CSV_FILE, num_records=NUM_RECORDS):
    """Generates a CSV file with the specified number of job records."""
    headers = ["external_job_id", "title", "company", "location", "job_description_text"]
    
    print(f"Starting CSV generation for {num_records} records into '{filename}'...")
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        # Use csv.QUOTE_ALL to ensure all fields are quoted, especially the description
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        writer.writerow(headers)
        
        for i in range(1, num_records + 1):
            external_id = f"job_{i:04d}" # e.g., job_0001, job_0002
            title = random.choice(JOB_TITLES)
            company = f"{random.choice(COMPANY_PREFIXES)} {random.choice(COMPANY_SUFFIXES)}"
            location = random.choice(LOCATIONS)
            
            description = generate_job_description(title, company, location)
            
            writer.writerow([external_id, title, company, location, description])
            
            if i % 100 == 0:
                print(f"Generated {i}/{num_records} records...")
                
    print(f"\nSuccessfully generated {num_records} job records into '{filename}'")

if __name__ == "__main__":
    generate_jobs_csv()