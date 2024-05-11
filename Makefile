# Define the default target
.PHONY: all venv
all: run

# Setup virtual environment and install dependencies
venv: 
	python3 -m venv venv
	@source venv/bin/activate; \
	pip install -r requirements.txt

# Main program
run:
	@source venv/bin/activate;
	python ./code/data_cleaning.py
	python ./code/data_analysis.py
	quarto render ./poster/poster.qmd