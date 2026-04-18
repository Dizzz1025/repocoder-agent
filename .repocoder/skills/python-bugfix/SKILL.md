# Python Bugfix

Use this skill when the task is about fixing a Python bug with the smallest safe change.

Recommended workflow:
1. Identify the failing file and the smallest likely fix.
2. Prefer minimal `replace` patches over broader edits.
3. Run validation commands after each accepted change.
4. If a retry is needed, keep the patch scope narrow.

Watch-outs:
- Avoid editing unrelated files.
- Treat config, auth, and entrypoint files as higher risk.
- Prefer fixes that preserve existing test intent.
