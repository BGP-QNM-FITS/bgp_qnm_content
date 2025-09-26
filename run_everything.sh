#!/bin/bash

echo "Running all plotting scripts..."

# Run all .py plotting scripts (excluding automate_pages.py and scripts starting with 0_)
for script in 1_mode_content.py 1_mode_content_paper.py 2_amplitude_stability.py 2_amplitude_stability_paper.py 3_epsilon.py 4_fits.py; do
    if [ -f "$script" ]; then
        echo "Running $script"
        python "$script" || echo "Error running $script, moving to the next script..."
    fi
done

echo "Generating website pages..."
python automate_pages.py || echo "Error running automate_pages.py"

echo "Building Sphinx HTML documentation..."
cd docs
sphinx-build -b html docs docs || echo "Error building Sphinx HTML documentation"

echo "All done!"
