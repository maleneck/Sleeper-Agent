import sys
import os
import subprocess

scripts_to_run = [
    "first_data.py",
    "baseline_4_models.py",
    "zero_shot_experiment.py",
    "second_data.py",
    "second_experiment.py"
]

def run_pipeline(scripts):
    print("Starting the ML pipeline...")

    for script in scripts:
        # Check if the file actually exists before trying to run it
        if not os.path.exists(script):
            print(f"\n Error: Could not find '{script}' in the current directory.")
            sys.exit(1)

        print(f"\n{'========================================='}")
        print(f"Starting execution of: {script}")
        print(f"{'==========================================='}\n")

        try:
            # sys.executable ensures the script uses the exact same Python environment 
            # (and installed packages) that is running this runner script.
            subprocess.run([sys.executable, script], check=True)
            print(f"\n Finished {script}")

        except subprocess.CalledProcessError as e:
            # check=True causes this exception if the script fails (returns non-zero exit code)
            print(f"\n Error occurred while running {script}")
            print(f"Exit code: {e.returncode}")
            print("Halting pipeline to prevent downstream errors.")
            sys.exit(e.returncode)

        except KeyboardInterrupt:
            print(f"\n Pipeline was manually interrupted during {script}.")
            sys.exit(1)

        except Exception as e:
            print(f"\n An unexpected error occurred: {e}")
            sys.exit(1)


run_pipeline(scripts_to_run)
print("\n All scripts executed successfully! Pipeline complete.")