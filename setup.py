from setuptools import find_packages,setup
from typing import List

HYPEN_e_DOT="- e."

def get_requirements(file_path : str) ->List[str] :
    """
    This Function will return list of requiremnet
    """
    requirement=[]

    with open( file_path ) as file_obj :
        requirement = file_obj.readlines()
        requirement = [req.replace("\n"," ") for req in requirement]

    if HYPEN_e_DOT in requirement :
        requirement.remove( HYPEN_e_DOT )

    return requirement

setup(
    name = "mlproject",
    version = "0.0.1",
    author = "Jerom Kurian",
    author_email = "jeromk60@gmail.com",
    packages = find_packages() ,
    install_requires = get_requirements("requirements.txt")
)