"""Get TruFor Docker entrypoint help text to see correct argument syntax."""
import subprocess

print("=== trufor_docverify --help ===")
r = subprocess.run(
    ["docker", "run", "--rm", "trufor_docverify", "--help"],
    capture_output=True, text=True, timeout=30
)
print("STDOUT:", r.stdout)
print("STDERR:", r.stderr)
print(f"Exit: {r.returncode}")

print("\n=== Dockerfile ENTRYPOINT / CMD inspection ===")
r2 = subprocess.run(
    ["docker", "inspect", "--format", "{{json .Config.Entrypoint}} {{json .Config.Cmd}}", "trufor_docverify"],
    capture_output=True, text=True, timeout=10
)
print(r2.stdout)

print("\n=== List files in /trufor workdir ===")
r3 = subprocess.run(
    ["docker", "run", "--rm", "--entrypoint", "ls", "trufor_docverify", "-la", "/trufor"],
    capture_output=True, text=True, timeout=15
)
print("STDOUT:", r3.stdout)
print("STDERR:", r3.stderr)
