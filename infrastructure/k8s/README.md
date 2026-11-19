# Deploy Nutrition Assistant on Kubernetes

The manifests follow the reference project's namespace, ConfigMap, Secret, Deployment, Service, and Ingress layout. No database or Ollama workload is included. Only the frontend is exposed through HTTPS; its nginx proxies `/api/*` to the backend ClusterIP Service and strips the `/api` prefix. Buffering is disabled at both nginx hops for streaming replies. The chat has no access control, so anyone who reaches the Ingress can use the configured Ollama instance.

## Cluster setup

1. In [`backend/configmap.yaml`](backend/configmap.yaml), set `OLLAMA_MODEL` to a tool-capable model installed on your Ollama server (default `qwen2.5:3b`) and `OLLAMA_URL` to the server's base URL, without `/api`. The default assumes an `ollama` Service in the same namespace. For a service in another namespace, use its fully qualified service name, for example `http://ollama.models.svc.cluster.local:11434`.
2. Point `nutrition-ai.talaatharb.net` at the cluster's nginx ingress controller. The [Ingress](ingress.yaml) uses the `nginx` IngressClass and the `letsencrypt-prod` cert-manager ClusterIssuer from the reference deployment.
3. Publish the images by pushing a tag such as `v1.2.3`. Make both GHCR packages public for anonymous pulls, or configure a registry pull secret if the packages are private. Set both `newTag` values in [`kustomization.yaml`](kustomization.yaml) to the published version (for example `1.2.3`, without `v`) instead of `latest` for repeatable rollouts.

From the repository root, for an Ollama endpoint that does not require a key:

```powershell
kubectl apply -f infrastructure/k8s/00-namespace.yaml
kubectl apply -f infrastructure/k8s/secrets-example.yaml
kubectl apply -k infrastructure/k8s
kubectl -n nutrition-ai-chat rollout status deployment/nutrition-ai-chat-be
kubectl -n nutrition-ai-chat rollout status deployment/nutrition-ai-chat-fe
```

If Ollama requires authentication, create the same Secret (`nutrition-ai-chat-be-secret`, key `OLLAMA_API_KEY`) with your cluster's secret-management tooling instead of applying the empty example. Do not commit the key or reapply the empty template over a configured Secret. The backend sends a bearer token upstream only when this key is nonempty.

Check `https://nutrition-ai.talaatharb.net/api/health`: `ollama: true` means the backend can reach Ollama's version endpoint. Send a chat message to confirm the configured model is available. Pod probes use `/healthz`, which does not depend on Ollama.

## Configuration and upgrades

The frontend ConfigMap sets `API_URL=/api` (the public API base read by React) and `BACKEND_URL=http://nutrition-ai-chat-be:8000` (the internal nginx upstream, without a trailing slash). Leave the defaults for same-origin routing. `API_URL` is public browser configuration; never put secrets in it. If you configure a separate public API origin, also set `CORS_ORIGINS` on the backend to the frontend origin.

For version upgrades or rollback, set both image tags to the desired published version and reapply `kubectl apply -k infrastructure/k8s`. Environment changes from a ConfigMap or Secret only reach new pods; after applying those changes, run `kubectl -n nutrition-ai-chat rollout restart deployment/<name>` for the affected deployment(s).

## Local Docker

From the repository root, with Ollama listening on the Docker Desktop host:

```powershell
docker build -t nutrition-ai-chat-be:local server
docker build -t nutrition-ai-chat-fe:local client
docker network create nutrition-ai-chat
docker run -d --name nutrition-ai-chat-be --network nutrition-ai-chat -e OLLAMA_URL=http://host.docker.internal:11434 nutrition-ai-chat-be:local
docker run -d --name nutrition-ai-chat-fe --network nutrition-ai-chat -p 127.0.0.1:8080:8080 nutrition-ai-chat-fe:local
```

Open `http://localhost:8080`. Use an Ollama URL reachable from Docker if `host.docker.internal` is unavailable or Ollama runs elsewhere.
