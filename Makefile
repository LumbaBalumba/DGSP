.PHONY: generate, estimate, stats, all, prep_env

all: stats

prep_env:
	pip install poetry
	poetry install

generate: prep_env
	poetry run generate


estimate: generate
	poetry run estimate

stats: estimate
	poetry run stats
