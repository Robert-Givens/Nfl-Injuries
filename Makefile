venv:
	python3 -m venv venv
	. venv/bin/activate; pip install -r requirements.txt

run: venv
	. venv/bin/activate; python data_cleaning.py
	. venv/bin/activate; python data_analysis.py