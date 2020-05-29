all: main.exe

clean: py_clean
tests: py_tests
lint: py_lint

# Python lint, tests and clean
py_lint: py_pep8 py_pylint 

py_pep8:
	pep8 --ignore=E111 ./src
	
py_pylint:
	pylint --rcfile=tools/.pylintrc ./src

py_tests:
	nosetests source/ --with-xunit --xunit-file=build/nosetests.xml

py_clean:
	find . | grep -E "(__pycache__|\.pyc)" | xargs rm -rf
	if [ -f build/nosetests.xml ] ; \
	then \
		rm -rf build/nosetests.xml ; \
	fi;

pytree: 
	tree ./ --prune --dirsfirst -P *.py

# py_pep8:
# 	python .git/hooks/pre-commit.py
# 
# # Need to disable lib ignore
# py_pylint:
#	git-pylint-commit-hook
