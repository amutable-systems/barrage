PY_DIRS := "./barrage ./tests"

check:
	ruff format --check {{PY_DIRS}}
	ruff check  {{PY_DIRS}}
	codespell {{PY_DIRS}}
	mypy {{PY_DIRS}}
	ty check {{PY_DIRS}}

fmt:
	ruff format {{PY_DIRS}}
	ruff check --fix {{PY_DIRS}}

test *barrage_args:
	python3 -m barrage -t ./tests {{barrage_args}}
