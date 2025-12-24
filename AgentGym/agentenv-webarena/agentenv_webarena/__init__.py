import os
import sys

sys.path.append(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "webarena")
)

# webarena urls
os.environ["SHOPPING"] = "http://metis.lti.cs.cmu.edu:7770"
os.environ["SHOPPING_ADMIN"] = (
    "http://metis.lti.cs.cmu.edu:7780/admin"
)
os.environ["REDDIT"] = "http://metis.lti.cs.cmu.edu:9999"
os.environ["GITLAB"] = "http://metis.lti.cs.cmu.edu:8023"
os.environ["MAP"] = "http://metis.lti.cs.cmu.edu:3000"
os.environ["WIKIPEDIA"] = (
    "http://metis.lti.cs.cmu.edu:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
)
os.environ["HOMEPAGE"] = "http://metis.lti.cs.cmu.edu:4399"

os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_BASE_URL"] = ""

os.chdir("./webarena")

from .launch import launch
from .server import app
