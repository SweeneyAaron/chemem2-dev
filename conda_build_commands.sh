#!/bin/bash
# Complete conda build process for ChemEM

set -e

echo "Building ChemEM conda package"
echo "============================="

# Step 1: Create missing files
echo "Preparing project files..."
if [ -f "create_missing_files.sh" ]; then
    chmod +x create_missing_files.sh
    ./create_missing_files.sh
else
    echo "WARNING: create_missing_files.sh not found, continuing without it..."
fi

# Step 2: Verify environment
echo ""
echo "Checking conda environment..."
if [ -z "$CONDA_PREFIX" ]; then
    echo "ERROR: No conda environment active. Please run:"
    echo "   conda activate your_env_name"
    exit 1
fi

echo "Using conda environment: $CONDA_PREFIX"

# Step 3: Check dependencies are available
echo ""
echo "Checking build dependencies..."
REQUIRED_PACKAGES=("cmake" "ninja" "pybind11" "rdkit" "boost-cpp" "eigen")

for package in "${REQUIRED_PACKAGES[@]}"; do
    if conda list "$package" &> /dev/null; then
        echo "  OK: $package"
    else
        echo "  MISSING: $package"
        echo "     Install with: conda install -c conda-forge $package"
        exit 1
    fi
done

# Step 4: Check conda-build is available
if ! command -v conda-build &> /dev/null; then
    echo "Installing conda-build..."
    conda install conda-build conda-verify -y
fi

# Step 5: Clean previous builds
echo ""
echo "Cleaning previous builds..."
rm -rf build/
rm -rf conda-dist/
rm -rf ChemEM/*.so ChemEM/*.dll ChemEM/*.dylib 2>/dev/null || true

# Step 6: Test basic cmake build first (optional but recommended)
echo ""
echo "Testing CMake build..."
if cmake -S . -B build -DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DCMAKE_BUILD_TYPE=Release; then
    echo "CMake configuration successful"
    if cmake --build build -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4); then
        echo "CMake build successful"
        # Check if modules were built
        if ls ChemEM/*.so ChemEM/*.dll ChemEM/*.dylib 2>/dev/null; then
            echo "Compiled modules found in ChemEM/"
            ls -la ChemEM/*.so ChemEM/*.dll ChemEM/*.dylib 2>/dev/null || true
        else
            echo "WARNING: No compiled modules found - check CMake build"
        fi
    else
        echo "ERROR: CMake build failed"
        exit 1
    fi
else
    echo "ERROR: CMake configuration failed"
    exit 1
fi

# Step 7: Build conda package
echo ""
echo "Building conda package..."
echo "This may take several minutes..."

# Create output directory
mkdir -p conda-dist

# Build the package
if conda build . --output-folder conda-dist --no-anaconda-upload; then
    echo ""
    echo "SUCCESS: Conda build completed!"
    
    # Find the built package
    PACKAGE=$(find conda-dist -name "*.tar.bz2" | head -1)
    
    if [ -f "$PACKAGE" ]; then
        echo "Built package: $PACKAGE"
        
        # Show package size
        SIZE=$(du -sh "$PACKAGE" | cut -f1)
        echo "Package size: $SIZE"
        
        # Show package contents
        echo ""
        echo "Package contents (first 15 files):"
        tar -tf "$PACKAGE" | head -15
        
        echo ""
        echo "Compiled modules in package:"
        tar -tf "$PACKAGE" | grep -E "\.(so|dll|dylib)$" || echo "No compiled modules found in package"
        
        echo ""
        echo "Testing package installation..."
        
        # Test in a clean environment
        TEST_ENV="test-chemem-$(date +%s)"
        echo "Creating test environment: $TEST_ENV"
        conda create -n "$TEST_ENV" python=3.11 -y -q
        
        if conda install -n "$TEST_ENV" "$PACKAGE" -y -q; then
            echo "Package installs successfully"
            
            # Test imports
            echo "Testing imports..."
            if conda run -n "$TEST_ENV" python -c "
try:
    import ChemEM
    print('SUCCESS: ChemEM imports successfully')
    
    import ChemEM.docking
    print('SUCCESS: ChemEM.docking imports successfully')
    
    import ChemEM.grid_maps
    print('SUCCESS: ChemEM.grid_maps imports successfully')
    
    import ChemEM.ligand_fitting
    print('SUCCESS: ChemEM.ligand_fitting imports successfully')
    
    print('SUCCESS: All modules import successfully!')
    print('SUCCESS: Package is ready for distribution!')
    
except ImportError as e:
    print(f'ERROR: Import failed: {e}')
    import sys
    sys.exit(1)
"; then
                echo "All tests passed!"
            else
                echo "ERROR: Import tests failed"
            fi
        else
            echo "ERROR: Package installation failed"
        fi
        
        # Cleanup test environment
        echo "Cleaning up test environment..."
        conda env remove -n "$TEST_ENV" -y -q
        
    else
        echo "ERROR: No package file found!"
        exit 1
    fi
    
else
    echo "ERROR: Conda build failed!"
    echo ""
    echo "Common solutions:"
    echo "1. Check meta.yaml for syntax errors"
    echo "2. Ensure all dependencies are available"
    echo "3. Check CMakeLists.txt for issues"
    echo "4. Verify all source files exist"
    echo "5. Check that ChemEM/__init__.py exists"
    exit 1
fi

echo ""
echo "Next steps:"
echo "1. Test your package locally: conda install $PACKAGE"
echo "2. Upload to anaconda.org: anaconda upload $PACKAGE"
echo "3. Or use in CI/CD for multi-platform builds"
echo ""
echo "Package location: $PACKAGE"