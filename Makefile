.PHONY: install clean generate estimate test all

all: estimate test

test:

clean:
	rm -rf ./data

estimate: generate
	poetry run estimate

generate: install
	poetry run generate

install:
	pip install poetry
	poetry install
