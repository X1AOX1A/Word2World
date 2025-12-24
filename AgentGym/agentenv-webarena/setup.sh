#!/bin/bash
cd ./webarena
pip install -r requirements.txt
playwright install-deps
playwright install 
pip install -e .
pip3 install gunicorn

export SHOPPING="http://metis.lti.cs.cmu.edu:7770"
export SHOPPING_ADMIN="http://metis.lti.cs.cmu.edu:7780/admin"
export REDDIT="http://metis.lti.cs.cmu.edu:9999"
export GITLAB="http://metis.lti.cs.cmu.edu:8023"
export MAP="http://metis.lti.cs.cmu.edu:3000"
export WIKIPEDIA="http://metis.lti.cs.cmu.edu:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export HOMEPAGE="http://metis.lti.cs.cmu.edu:4399"
export OPENAI_API_KEY=""
export OPENAI_BASE_URL=""


python scripts/generate_test_data.py
mkdir -p ./.auth
python browser_env/auto_login.py
python agent/prompts/to_json.py

cd ..

pip install -e .
