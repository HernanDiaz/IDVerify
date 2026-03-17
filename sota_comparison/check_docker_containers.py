"""Check all Docker containers (running and stopped) and show logs for trufor_inference."""
import subprocess

print("=== docker ps -a ===")
r = subprocess.run(["docker", "ps", "-a"], capture_output=True, text=True)
print(r.stdout)
if r.stderr:
    print("STDERR:", r.stderr)

print("\n=== docker logs trufor_inference (last 50 lines) ===")
r2 = subprocess.run(["docker", "logs", "--tail", "50", "trufor_inference"],
                    capture_output=True, text=True)
if r2.stdout:
    print(r2.stdout)
if r2.stderr:
    print(r2.stderr)
if r2.returncode != 0:
    print(f"(exit {r2.returncode} — container may not exist or already removed with --rm)")

print("\n=== docker images trufor_docverify ===")
r3 = subprocess.run(["docker", "images", "trufor_docverify"], capture_output=True, text=True)
print(r3.stdout)
