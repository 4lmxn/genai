import wikipediaapi
from pydantic import BaseModel
from typing import Optional, List

class InstitutionDetails(BaseModel):
    founder: Optional[str]
    founded: Optional[str]
    branches: Optional[List[str]]
    number_of_employees: Optional[int]
    summary: Optional[str]

def fetch_institution_details(institution_name: str) -> InstitutionDetails:
    wiki_wiki = wikipediaapi.Wikipedia(
        user_agent="MyNotebook/1.0 (contact: myemail@example.com)",
        language='en'
    )
    page = wiki_wiki.page(institution_name)

    if not page.exists():
        raise ValueError(f"Page for '{institution_name}' not found on Wikipedia.")

    founder = founded = None
    branches = []
    number_of_employees = None
    summary = page.summary[:500]

    for line in page.text.split('\n'):
        if 'Founder' in line:
            founder = line.split(':')[-1].strip()
        elif 'Founded' in line:
            founded = line.split(':')[-1].strip()
        elif 'Branches' in line:
            branches = [b.strip() for b in line.split(':')[-1].split(',')]
        elif 'Number of employees' in line:
            try:
                number_of_employees = int(line.split(':')[-1].strip().replace(',', ''))
            except ValueError:
                number_of_employees = None

    return InstitutionDetails(
        founder=founder,
        founded=founded,
        branches=branches if branches else None,
        number_of_employees=number_of_employees,
        summary=summary
    )

name = input("Institution Name: ")
result = fetch_institution_details(name)

print(f"\nInstitution: {name}")
print(f"Founder: {result.founder or 'N/A'}")
print(f"Founded: {result.founded or 'N/A'}")
print(f"Branches: {', '.join(result.branches) if result.branches else 'N/A'}")
print(f"Employees: {result.number_of_employees or 'N/A'}")
print(f"Summary: {result.summary or 'N/A'}")
