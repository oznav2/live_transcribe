#!/usr/bin/env python3
"""
Validate Dockerfile.ivrit for common Python syntax issues
"""

import re
import sys

def check_dockerfile():
    """Check Dockerfile for common issues"""
    
    with open('Dockerfile.ivrit', 'r') as f:
        lines = f.readlines()
    
    issues = []
    
    for i, line in enumerate(lines, 1):
        # Check for Python one-liners with try/except
        if 'python3 -c' in line and 'try:' in line and 'except' in line:
            # Check if try and except are on same line (invalid)
            if not ('exec(' in line or 'importlib' in line):
                issues.append(f"Line {i}: Invalid try/except on single line in Python command")
        
        # Check for colon after import on same line
        if 'python3 -c' in line and 'import' in line and '; try:' in line:
            issues.append(f"Line {i}: Cannot have 'try:' after semicolon on same line")
        
        # Check for old model references
        if 'ivrit-large-v3-turbo' in line and 'ivrit-ai/whisper-large-v3-turbo-ct2' not in line:
            issues.append(f"Line {i}: Reference to old GGML model 'ivrit-large-v3-turbo'")
        
        # Check for whisper.cpp references
        if 'whisper-cli' in line and '# ' not in line[:line.index('whisper-cli') if 'whisper-cli' in line else 0]:
            issues.append(f"Line {i}: Reference to whisper.cpp CLI (removed)")
    
    return issues

def main():
    """Run validation"""
    print("Validating Dockerfile.ivrit...")
    print("=" * 60)
    
    issues = check_dockerfile()
    
    if issues:
        print("❌ Issues found:")
        for issue in issues:
            print(f"  • {issue}")
        return 1
    else:
        print("✅ No Python syntax issues found in Dockerfile")
        return 0

if __name__ == "__main__":
    sys.exit(main())