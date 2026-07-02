import os
import shutil
import subprocess

def main():
    src_dir = r"C:\Users\darsh\Downloads\Hack_to_Skills"
    test_dir = os.path.join(src_dir, "test_sandbox")
    
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)
    
    # Copy essential files
    files_to_copy = [
        "precompute.py",
        "rank.py",
        "requirements.txt",
        "sample_100.jsonl",
    ]
    for f in files_to_copy:
        shutil.copy(os.path.join(src_dir, f), os.path.join(test_dir, f))
        
    shutil.copytree(os.path.join(src_dir, "src"), os.path.join(test_dir, "src"))
    
    print("Setting up virtual environment...")
    subprocess.run(["python", "-m", "venv", "venv"], cwd=test_dir, check=True)
    
    python_exe = os.path.join(test_dir, "venv", "Scripts", "python.exe")
    pip_exe = os.path.join(test_dir, "venv", "Scripts", "pip.exe")
    
    print("Installing dependencies...")
    subprocess.run([pip_exe, "install", "-r", "requirements.txt"], cwd=test_dir, check=True)
    
    print("Running precompute...")
    subprocess.run([python_exe, "precompute.py", "--candidates", "sample_100.jsonl"], cwd=test_dir, check=True)
    
    print("Running rank...")
    subprocess.run([python_exe, "rank.py", "--candidates", "sample_100.jsonl", "--out", "sandbox_submission.csv"], cwd=test_dir, check=True)
    
    print("Done! CSV generated.")
    if os.path.exists(os.path.join(test_dir, "sandbox_submission.csv")):
        print("Success: sandbox_submission.csv exists.")

if __name__ == "__main__":
    main()
