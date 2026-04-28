# PathSight — Frontend

React + TypeScript single-page app built with Vite. Calls the backend `/api` endpoints to run cancer predictions and display results.

---

## Prerequisites — Node.js

You need Node.js 18 or later. If you don't have it, install it via [nvm](https://github.com/nvm-sh/nvm) (recommended on macOS):

```bash
# 1. Install nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash

# 2. Restart your terminal, then install and activate Node 20 LTS
nvm install 20
nvm use 20

# Verify
node --version   # should print v20.x.x
npm --version
```

Alternatively, install Node directly from [nodejs.org](https://nodejs.org) or via Homebrew:

```bash
brew install node
```

---

## Install dependencies

```bash
npm install
# or, from the repo root:
make frontend-install
```

---

## Development server

```bash
npm run dev
# or, from the repo root:
make run-frontend-dev
```

The Vite dev server starts at **http://localhost:5173** and proxies all `/api`
requests to `http://127.0.0.1:8000` (the local backend). Start the backend
first — see the root README for instructions.

---

## Other commands

```bash
# Type-check and build for production (output: frontend/dist/)
npm run build
# or: make frontend-build

# Preview the production build locally
npm run preview

# Lint
npm run lint
```

---

## Environment notes

- The dev server proxy is configured in `vite.config.ts`. If your backend runs
  on a different port, update the proxy target there.
- For Docker/production builds, the API base URL is set at build time via the
  `VITE_API_BASE=/api` environment variable, and nginx routes `/api` to the
  backend container.
