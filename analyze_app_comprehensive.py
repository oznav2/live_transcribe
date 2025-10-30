#!/usr/bin/env python3
"""
Comprehensive analysis of app.py to identify issues after GGML removal
"""

import re
import ast

def analyze_code():
    """Analyze app.py for potential issues"""
    
    with open('app.py', 'r') as f:
        content = f.read()
    
    issues = []
    warnings = []
    
    # Parse AST
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        print(f"Syntax error in app.py: {e}")
        return
    
    # 1. Check for undefined variables related to removed code
    print("=" * 60)
    print("1. CHECKING FOR UNDEFINED REFERENCES")
    print("=" * 60)
    
    # Variables that were removed
    removed_vars = ['WHISPER_CPP_PATH', 'WHISPER_CPP_AVAILABLE']
    for var in removed_vars:
        if var in content:
            # Check if it's not in a comment
            for i, line in enumerate(content.split('\n'), 1):
                if var in line and not line.strip().startswith('#'):
                    issues.append(f"Line {i}: Found reference to removed variable '{var}'")
    
    # 2. Check model type handling in functions
    print("\n2. CHECKING MODEL TYPE HANDLING")
    print("=" * 60)
    
    # Find all places where model_config["type"] is checked
    pattern = r'model_config\["type"\]\s*==\s*"([^"]+)"'
    matches = re.finditer(pattern, content)
    valid_types = ['openai', 'faster_whisper', 'deepgram']
    
    for match in matches:
        model_type = match.group(1)
        if model_type not in valid_types:
            line_num = content[:match.start()].count('\n') + 1
            issues.append(f"Line {line_num}: Invalid model type '{model_type}'")
    
    # Check for incomplete if-elif chains (missing else after removing ggml)
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'elif model_config["type"] == "faster_whisper":' in line:
            # Check if there's content after this elif
            next_lines = lines[i+1:i+20]  # Check next 20 lines
            has_else = False
            has_next_elif = False
            indent_level = len(line) - len(line.lstrip())
            
            for next_line in next_lines:
                if next_line.strip() == '':
                    continue
                next_indent = len(next_line) - len(next_line.lstrip())
                if next_indent <= indent_level:
                    if next_line.strip().startswith('else:'):
                        has_else = True
                    elif next_line.strip().startswith('elif'):
                        has_next_elif = True
                    break
            
            if not has_else and not has_next_elif:
                # Check if this is the last condition in a chain
                prev_lines = lines[max(0, i-10):i]
                has_if = any('if model_config["type"]' in l for l in prev_lines)
                if has_if:
                    warnings.append(f"Line {i+1}: elif chain might be incomplete after GGML removal")
    
    # 3. Check for model variable usage
    print("\n3. CHECKING MODEL VARIABLE USAGE")
    print("=" * 60)
    
    # Check for direct model usage vs wrapped model
    if 'model["model"]' in content:
        warnings.append("Found model['model'] access - verify this is for faster_whisper only")
    
    # 4. Check function parameters
    print("\n4. CHECKING FUNCTION PARAMETERS")
    print("=" * 60)
    
    # Check transcribe_chunk function
    transcribe_chunk_pattern = r'def transcribe_chunk\(([^)]+)\):'
    match = re.search(transcribe_chunk_pattern, content)
    if match:
        params = match.group(1)
        if 'model_config' in params and 'model' in params:
            print("✓ transcribe_chunk has correct parameters")
        else:
            issues.append("transcribe_chunk missing required parameters")
    
    # 5. Check for incomplete transcription handling
    print("\n5. CHECKING TRANSCRIPTION FUNCTIONS")
    print("=" * 60)
    
    transcription_functions = [
        'transcribe_chunk',
        'transcribe_audio_stream', 
        'transcribe_with_incremental_output'
    ]
    
    for func_name in transcription_functions:
        func_pattern = rf'def {func_name}\([^)]*\):.*?(?=\ndef |\nasync def |$)'
        match = re.search(func_pattern, content, re.DOTALL)
        if match:
            func_body = match.group()
            
            # Check if faster_whisper is handled
            if 'faster_whisper' not in func_body and func_name != 'transcribe_vod_with_deepgram':
                warnings.append(f"{func_name} might not handle faster_whisper models")
            
            # Check for empty elif blocks
            if re.search(r'elif.*?:\s*\n\s*#[^\n]*\n\s*(elif|else|$)', func_body):
                issues.append(f"{func_name} has empty elif block after GGML removal")
    
    # 6. Check MODEL_CONFIGS consistency
    print("\n6. CHECKING MODEL_CONFIGS")
    print("=" * 60)
    
    model_configs_pattern = r'MODEL_CONFIGS = \{([^}]+)\}'
    match = re.search(model_configs_pattern, content, re.DOTALL)
    if not match:
        # Check for dynamic MODEL_CONFIGS
        if 'MODEL_CONFIGS = {}' in content:
            print("✓ MODEL_CONFIGS is dynamically built")
            
            # Check if faster_whisper models are added
            if 'if FASTER_WHISPER_AVAILABLE:' in content:
                if '"ivrit-ct2"' in content:
                    print("✓ ivrit-ct2 is configured")
                else:
                    issues.append("ivrit-ct2 not found in MODEL_CONFIGS")
    
    # 7. Check environment variables
    print("\n7. CHECKING ENVIRONMENT VARIABLES")  
    print("=" * 60)
    
    env_vars = re.findall(r'os\.getenv\("([^"]+)"', content)
    deprecated_vars = ['WHISPER_CPP_PATH', 'WHISPER_CPP_THREADS', 'IVRIT_MODEL_PATH']
    
    for var in env_vars:
        if var in deprecated_vars:
            line_num = content.index(f'os.getenv("{var}"')
            line_num = content[:line_num].count('\n') + 1
            issues.append(f"Line ~{line_num}: Using deprecated env var '{var}'")
    
    # 8. Check for consistent model access patterns
    print("\n8. CHECKING MODEL ACCESS PATTERNS")
    print("=" * 60)
    
    # Check for inconsistent model unwrapping
    if 'if isinstance(model, dict) and model.get("type") == "faster_whisper":' in content:
        count = content.count('if isinstance(model, dict) and model.get("type") == "faster_whisper":')
        print(f"✓ Found {count} instances of proper model unwrapping")
    
    # Check if model["model"] is used without checking
    bare_model_access = re.findall(r'(?<!if isinstance\(model, dict\).*\n.*)model\["model"\]', content)
    if bare_model_access:
        warnings.append(f"Found {len(bare_model_access)} unchecked model['model'] accesses")
    
    # Report results
    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)
    
    if issues:
        print("\n❌ CRITICAL ISSUES FOUND:")
        for issue in issues:
            print(f"  • {issue}")
    else:
        print("\n✓ No critical issues found")
    
    if warnings:
        print("\n⚠️  WARNINGS:")
        for warning in warnings:
            print(f"  • {warning}")
    
    return issues, warnings

if __name__ == "__main__":
    issues, warnings = analyze_code()
    
    if issues:
        exit(1)
    exit(0)