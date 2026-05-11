---
name: Vercel Deploy Workflow
description: How to safely preview and deploy changes to the Vercel-hosted cell registration viewer
type: feedback
---

Always preview before pushing to production — use `vercel` CLI (no flags) to get a one-off preview URL, then push to `main` only after approval.

**Why:** Pushing to GitHub main triggers Vercel production deploy immediately, and user was burned by a broken deploy going live before it was tested.

**How to apply:**
- Use `vercel` (no `--prod`) from the repo directory to deploy to a throwaway preview URL
- Only `git push origin main` after user has verified the preview
- Before pushing HTML that iframes other files, run `git ls-files | grep .html` to confirm all referenced files are committed — missing files cause Vercel 404s
- Local HTTP server (`python3 -m http.server 8765`) can be used for quick local checks; kill with `pkill -f "python3 -m http.server 8765"` when done
- Vercel CLI install: `npm i -g vercel`
