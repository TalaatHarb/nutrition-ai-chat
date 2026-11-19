FROM --platform=$BUILDPLATFORM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY . .
RUN npm run build

FROM nginxinc/nginx-unprivileged:stable-alpine
USER root
RUN apk add --no-cache jq
ENV API_URL=/api BACKEND_URL=http://nutrition-ai-chat-be:8000 NGINX_ENVSUBST_FILTER=^BACKEND_URL$
COPY nginx/default.conf.template /etc/nginx/templates/default.conf.template
COPY --chmod=755 nginx/40-runtime-env.sh /docker-entrypoint.d/40-runtime-env.sh
COPY --from=build --chown=101:101 /app/dist/ /usr/share/nginx/html/
USER 101:101
EXPOSE 8080
