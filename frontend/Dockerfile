# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Stage 1 — build the Vite bundle
#
# Vite inlines every VITE_* value at build time, so the Supabase settings have
# to be present here rather than at runtime.
#
# VITE_API_BASE_URL is pinned to "/api/v1" — the same prefix FastAPI mounts its
# routes under (settings.api_v1_str). Keeping the public path identical to the
# internal one means nothing in the chain has to rewrite it, and FastAPI's
# trailing-slash redirects stay valid: a Location of /api/v1/categories/ is
# correct from the browser's point of view too. The default "/api" in
# src/services/api.ts would force a rewrite at every hop instead.
# ---------------------------------------------------------------------------
FROM node:20-slim AS build

WORKDIR /app

COPY package*.json ./
RUN npm ci --legacy-peer-deps

COPY . .

ARG VITE_SUPABASE_URL
ARG VITE_SUPABASE_ANON_KEY
ARG VITE_API_BASE_URL=/api/v1
ENV VITE_SUPABASE_URL=$VITE_SUPABASE_URL \
    VITE_SUPABASE_ANON_KEY=$VITE_SUPABASE_ANON_KEY \
    VITE_API_BASE_URL=$VITE_API_BASE_URL

RUN npm run build


# ---------------------------------------------------------------------------
# Stage 2 — serve the static bundle
# ---------------------------------------------------------------------------
FROM nginx:1.27-alpine AS runtime

# gettext supplies envsubst, which the nginx entrypoint runs over
# /etc/nginx/templates/*.template before starting the server.
RUN apk add --no-cache gettext curl

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf.template /etc/nginx/templates/default.conf.template

# Defaults target the Compose topology. Azure Container Apps overrides both
# (BACKEND_SCHEME=https), and the Kubernetes manifests point BACKEND_FQDN at
# the backend Service DNS name.
ENV BACKEND_SCHEME=http \
    BACKEND_FQDN=app:8000

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -fsS http://localhost:8080/healthz || exit 1
