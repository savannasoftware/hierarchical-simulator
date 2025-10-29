# Makefile for Sphinx documentation

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD  ?= sphinx-build
SOURCEDIR    = .
BUILDDIR     = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Custom targets for common development tasks

# Clean all build artifacts
clean:
	@echo "Removing everything under '$(BUILDDIR)'..."
	@rm -rf $(BUILDDIR)
	@rm -rf generated/
	@echo "Clean completed."

# Build HTML documentation with live reload
livehtml:
	@echo "Starting live HTML build..."
	@sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O) \
		--host 0.0.0.0 \
		--port 8000 \
		--open-browser \
		--delay 1 \
		--watch ../src/ \
		--ignore "*.pyc" \
		--ignore "*/.git/*" \
		--ignore "*/__pycache__/*"

# Build HTML with warnings as errors (strict mode)
strict:
	@echo "Building HTML in strict mode (warnings as errors)..."
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O) -W

# Build all formats
all: html epub pdf
	@echo "All documentation formats built successfully."

# Build and check links
linkcheck:
	@echo "Checking external links..."
	@$(SPHINXBUILD) -M linkcheck "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Build API documentation
apidoc:
	@echo "Generating API documentation..."
	@sphinx-apidoc -o api/ ../src/hierarchical_simulator/ --force --module-first
	@echo "API documentation generated."

# Build documentation and run tests
test: apidoc strict linkcheck
	@echo "Documentation tests completed."

# Development build (fast, with warnings)
dev:
	@echo "Building development documentation..."
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O) -E -a

# Production build (clean, strict, all formats)
prod: clean apidoc strict epub pdf
	@echo "Production documentation build completed."

# Install documentation dependencies
install:
	@echo "Installing documentation dependencies..."
	@pip install -r requirements.txt
	@echo "Dependencies installed."

# Spell check (requires sphinxcontrib-spelling)
spelling:
	@echo "Running spell check..."
	@$(SPHINXBUILD) -M spelling "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Check documentation coverage
coverage:
	@echo "Checking documentation coverage..."
	@$(SPHINXBUILD) -M coverage "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "Coverage report: $(BUILDDIR)/coverage/python.txt"

# Build LaTeX and PDF
pdf:
	@echo "Building PDF documentation..."
	@$(SPHINXBUILD) -M latexpdf "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Build EPUB
epub:
	@echo "Building EPUB documentation..."
	@$(SPHINXBUILD) -M epub "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Build single HTML page
singlehtml:
	@echo "Building single HTML page..."
	@$(SPHINXBUILD) -M singlehtml "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Show build statistics
stats:
	@echo "Documentation statistics:"
	@find . -name "*.rst" -type f | wc -l | xargs echo "RST files:"
	@find . -name "*.md" -type f | wc -l | xargs echo "Markdown files:"
	@find . -name "*.ipynb" -type f | wc -l | xargs echo "Notebook files:"
	@find $(BUILDDIR)/html -name "*.html" -type f 2>/dev/null | wc -l | xargs echo "Generated HTML files:" || echo "Generated HTML files: 0 (run 'make html' first)"

# Open documentation in browser (macOS/Linux)
open:
	@echo "Opening documentation in browser..."
	@if [ -f "$(BUILDDIR)/html/index.html" ]; then \
		if command -v open >/dev/null 2>&1; then \
			open "$(BUILDDIR)/html/index.html"; \
		elif command -v xdg-open >/dev/null 2>&1; then \
			xdg-open "$(BUILDDIR)/html/index.html"; \
		else \
			echo "Please open $(BUILDDIR)/html/index.html in your browser"; \
		fi \
	else \
		echo "Documentation not built. Run 'make html' first."; \
	fi

# Deploy to GitHub Pages (requires ghp-import)
deploy:
	@echo "Deploying to GitHub Pages..."
	@if [ ! -d "$(BUILDDIR)/html" ]; then \
		echo "Building documentation first..."; \
		$(MAKE) html; \
	fi
	@ghp-import -n -p -f $(BUILDDIR)/html
	@echo "Documentation deployed to GitHub Pages."

# Watch for changes and rebuild (requires watchdog)
watch:
	@echo "Watching for changes..."
	@watchmedo shell-command \
		--patterns="*.rst;*.md;*.py" \
		--recursive \
		--command='make html' \
		.

# Generate requirements.txt from setup.py
requirements:
	@echo "Generating requirements.txt..."
	@pip-compile --extra=docs ../pyproject.toml
	@echo "Requirements generated."

# Help with custom targets
help-custom:
	@echo "Custom Makefile targets:"
	@echo "  clean       Remove all build artifacts"
	@echo "  livehtml    Build HTML with live reload (requires sphinx-autobuild)"
	@echo "  strict      Build HTML with warnings as errors"
	@echo "  all         Build HTML, EPUB, and PDF"
	@echo "  linkcheck   Check external links"
	@echo "  apidoc      Generate API documentation"
	@echo "  test        Run documentation tests"
	@echo "  dev         Development build (fast)"
	@echo "  prod        Production build (clean, strict, all formats)"
	@echo "  install     Install documentation dependencies"
	@echo "  spelling    Run spell check (requires sphinxcontrib-spelling)"
	@echo "  coverage    Check documentation coverage"
	@echo "  stats       Show documentation statistics"
	@echo "  open        Open documentation in browser"
	@echo "  deploy      Deploy to GitHub Pages (requires ghp-import)"
	@echo "  watch       Watch for changes and rebuild (requires watchdog)"
	@echo "  requirements Generate requirements.txt"

# Show current configuration
info:
	@echo "Sphinx build configuration:"
	@echo "  Source directory: $(SOURCEDIR)"
	@echo "  Build directory:  $(BUILDDIR)"
	@echo "  Sphinx options:   $(SPHINXOPTS)"
	@echo "  Sphinx command:   $(SPHINXBUILD)"
	@$(SPHINXBUILD) --version

.PHONY: clean livehtml strict all linkcheck apidoc test dev prod install spelling coverage pdf epub singlehtml stats open deploy watch requirements help-custom info
