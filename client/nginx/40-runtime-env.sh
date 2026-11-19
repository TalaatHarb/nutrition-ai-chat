#!/bin/sh
set -eu

jq -cnr --arg api_url "$API_URL" \
  '"window._env_ = " + ({API_URL: $api_url} | tojson) + ";"' \
  > /usr/share/nginx/html/env-config.js
