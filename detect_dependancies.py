#!/usr/bin/env python3
"""
Script to detect and analyze dependencies
"""
import os
import re
import sys
from pathlib import Path

def scan_cpp_includes(directory):
    """Scan C++ files for #include statements"""
    includes = set()
    cpp_extensions = ['.cpp', '.hpp', '.h', '.cc', '.cxx']
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in cpp_extensions):
                filepath = Path(root) / file
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        # Find #include statements
                        include_pattern = r'^\s*#include\s*[<"](.*?)[>"]'
                        matches = re.findall(include_pattern, content, re.MULTILINE)
                        includes.update(matches)
                except Exception as e:
                    print(f"Warning: Could not read {filepath}: {e}")
    
    return includes

def scan_python_imports(directory):
    """Scan Python files for import statements"""
    imports = set()
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                filepath = Path(root) / file
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Find import statements
                        import_patterns = [
                            r'^\s*import\s+([a-zA-Z_][a-zA-Z0-9_]*)',
                            r'^\s*from\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+import'
                        ]
                        for pattern in import_patterns:
                            matches = re.findall(pattern, content, re.MULTILINE)
                            imports.update(matches)
                except Exception as e:
                    print(f"Warning: Could not read {filepath}: {e}")
    
    return imports

def analyze_cpp_dependencies(includes):
    """Map C++ includes to conda packages"""
    cpp_to_conda = {
        # Boost libraries
        'boost/': 'boost-cpp',
        'boost/config.hpp': 'boost-cpp',
        'boost/algorithm/': 'boost-cpp',
        'boost/math/': 'boost-cpp',
        'boost/random/': 'boost-cpp',
        'boost/filesystem/': 'boost-cpp',
        
        # Eigen
        'Eigen/': 'eigen',
        'eigen3/': 'eigen',
        
        # RDKit
        'RDKit/': 'rdkit',
        'rdkit/': 'rdkit',
        'GraphMol/': 'rdkit',
        
        # OpenMP
        'omp.h': 'libomp (macOS), libgomp (Linux), openmp (Windows)',
        
        # Standard math
        'cmath': 'built-in',
        'math.h': 'built-in',
        'numeric': 'built-in',
        'algorithm': 'built-in',
        'vector': 'built-in',
        'string': 'built-in',
        'iostream': 'built-in',
        'fstream': 'built-in',
        
        # BLAS/LAPACK
        'cblas.h': 'openblas or mkl',
        'lapacke.h': 'openblas or mkl',
        
        # HDF5
        'hdf5.h': 'hdf5',
        'H5Cpp.h': 'hdf5',
    }
    
    dependencies = set()
    unknown_includes = []
    
    for include in includes:
        found = False
        for pattern, conda_pkg in cpp_to_conda.items():
            if pattern in include:
                if conda_pkg != 'built-in':
                    dependencies.add(conda_pkg)
                found = True
                break
        
        if not found and not include.startswith(('ChemEM/', './', '../')):
            unknown_includes.append(include)
    
    return dependencies, unknown_includes

def analyze_python_dependencies(imports):
    """Map Python imports to conda packages"""
    python_to_conda = {
        'numpy': 'numpy',
        'scipy': 'scipy', 
        'pandas': 'pandas',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'sklearn': 'scikit-learn',
        'rdkit': 'rdkit',
        'networkx': 'networkx',
        'h5py': 'h5py',
        'PIL': 'pillow',
        'cv2': 'opencv',
        'tensorflow': 'tensorflow',
        'torch': 'pytorch',
        'Bio': 'biopython',
        'MDAnalysis': 'mdanalysis',
        'numba': 'numba',
        'tqdm': 'tqdm',
        'pytest': 'pytest',
        'jupyter': 'jupyter',
    }
    
    dependencies = set()
    unknown_imports = []
    
    for imp in imports:
        # Skip built-in modules
        builtin_modules = {'os', 'sys', 'pathlib', 're', 'json', 'time', 'datetime', 
                          'collections', 'itertools', 'functools', 'typing', 'copy',
                          'math', 'random', 'string', 'io', 'gc', 'warnings'}
        
        if imp in builtin_modules or imp.startswith('ChemEM'):
            continue
            
        if imp in python_to_conda:
            dependencies.add(python_to_conda[imp])
        else:
            unknown_imports.append(imp)
    
    return dependencies, unknown_imports

def main():
    print("🔍 ChemEM Dependency Analysis")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not Path('ChemEM').exists():
        print("❌ ChemEM directory not found. Run this script from your project root.")
        return
    
    # Scan C++ dependencies
    print("\n📁 Scanning C++ files...")
    cpp_includes = scan_cpp_includes('ChemEM/cpp')
    cpp_deps, unknown_cpp = analyze_cpp_dependencies(cpp_includes)
    
    print(f"Found {len(cpp_includes)} C++ includes:")
    for inc in sorted(cpp_includes):  # Show first 10
        print(f"  - {inc}")
    #if len(cpp_includes) > 10:
    #    print(f"  ... and {len(cpp_includes) - 10} more")
    
    # Scan Python dependencies  
    print("\n🐍 Scanning Python files...")
    python_imports = scan_python_imports('ChemEM')
    python_deps, unknown_python = analyze_python_dependencies(python_imports)
    
    print(f"Found {len(python_imports)} Python imports:")
    for imp in sorted(python_imports):  # Show first 10
        print(f"  - {imp}")
    #if len(python_imports) > 10:
    #    print(f"  ... and {len(python_imports) - 10} more")
    
    # Summary
    print("\n📦 Required Conda Dependencies:")
    all_deps = cpp_deps.union(python_deps)
    for dep in sorted(all_deps):
        print(f"  - {dep}")
    
    if unknown_cpp:
        print(f"\n❓ Unknown C++ includes (may need additional packages):")
        for unk in sorted(unknown_cpp):
            print(f"  - {unk}")
        #if len(unknown_cpp) > 5:
        #    print(f"  ... and {len(unknown_cpp) - 5} more")
    
    if unknown_python:
        print(f"\n❓ Unknown Python imports (may need additional packages):")
        for unk in sorted(unknown_python):
            print(f"  - {unk}")
        #if len(unknown_python) > 5:
        #    print(f"  ... and {len(unknown_python) - 5} more")
    
    print(f"\n🎯 Next Steps:")
    print(f"1. Add the identified dependencies to your meta.yaml")
    print(f"2. Test with: conda env create -f environment.yml")
    print(f"3. Verify unknown dependencies manually")

if __name__ == "__main__":
    main()