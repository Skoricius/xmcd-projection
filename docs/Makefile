# Makefile for Sphinx documentation

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXGEN     ?= sphinx-apidoc
NBCONVERT     ?= jupyter
PYTHON        ?= python
PIP           ?= pip
SOURCEDIR     = .
BUILDDIR      = _build
TEMPLATESDIR  = templates
STATICDIR     = _static

example_notebooks := $(wildcard ../examples/*.ipynb)
example_names = $(foreach path, $(example_notebooks), $(basename $(notdir $(path))))
example_nbs = $(foreach name, $(example_names), examples/$(name).ipynb)

html: conf.py index.rst install.rst examples.rst README.rst xmcd_projection.rst $(example_nbs) .deps
	echo $(example_nbs)
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	touch "$(BUILDDIR)/html/.nojekyll"
	rm -rvf "$(BUILDDIR)/html/_static/examples/*"
	# mv "$(BUILDDIR)/html/_static/examples" "$(BUILDDIR)/html/examples" || true
	# mv "$(BUILDDIR)/html/_static/images" "$(BUILDDIR)/html/images" || true
	# cp "$(STATICDIR)/favicon.ico" "$(BUILDDIR)/html/favicon.ico" || true

.deps: requirements.txt
	$(PIP) install -r requirements.txt
	$(PIP) freeze > .deps

README.rst: ../README.md .deps
	pandoc --from=gfm --to=rst --output=README.rst ../README.md

examples/%.ipynb: ../examples/%.ipynb .deps
	python -c "import os; os.makedirs('./examples', exist_ok=True)"
	cp $< $(abspath $@) 

# examples.%.rst: $(STATICDIR)/examples/%.html examples.template
# 	$(PYTHON) -c "open('$@', 'w').write(open('examples.template').read().format(name='$(*F)', url='$<'))"

xmcd_projection.rst: .deps
	$(SPHINXGEN) -eTf -t "$(TEMPLATESDIR)" -o "$(SOURCEDIR)" ../xmcd_projection

clean:
	rm -rvf "$(BUILDDIR)" "examples" xmcd_projection*.rst .deps README.rst


# %: Makefile
# 	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	
# # # All other Sphinx builders
# %: conf.py index.rst install.rst examples.rst README.rst xmcd_projection.rst $(example_rsts) $(example_htmls)
# 	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
# #
# .PHONY: %