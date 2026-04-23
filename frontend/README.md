# PathSight frontend

```bash
npm install
npm run dev
```

Vite dev server proxies `/api` and `/static` to `http://127.0.0.1:8000`. Start the backend first.

Build:

```bash
npm run build
```

Docker / nginx uses `VITE_API_BASE=/api` at build time.
